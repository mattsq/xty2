"""Tier 0 — the gradient executor (`DESIGN.md` §7, `FIDELITY.md` §3, P4).

Four of the invariants `FIDELITY.md` §3 lists are properties of this loop, and
each of them is a way a run can be wrong while looking entirely healthy:

* **Trainable isolation** — a component outside `stage.trainable` receives no
  gradient at all, not merely no optimiser step. A frozen encoder that still
  fills in `.grad` reads as trained to the §6.2 probe and leaves a stale
  gradient for whatever runs next.
* **Determinism** — same seed, same loss trace. Without it every other
  comparison in the repo (a mutation test, a Tier 2 regression, a bisect) is
  comparing noise.
* **The dataset is not mutated** — the batches a run consumed are bit-identical
  afterwards, which is `DESIGN.md` §7.1's "no stage mutates the source
  dataset" in the only form that can fail a build.
* **Provenance is earned** — `trained_on_row_ids` is the rows the loop actually
  stepped on, so the §7.2 leakage check has something falsifiable to run
  against rather than a label somebody set.

The loop is also checked for the thing that is obvious right up until it is
wrong: that the optimiser is connected at all, and connected to the parameters
the stage named.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from xty2.core import (
    ComponentGraph,
    Constant,
    GradientClipping,
    Ramp,
    Recipe,
    TrainingError,
    WeightDecay,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.objectives import ObservedOutcomeNLL, ObservedTreatmentNLL
from xty2.training import (
    Checkpoint,
    GradientProbe,
    RunDirectory,
    StageResult,
    run_stage,
    trainable_only,
)

from tests.invariants.conftest import (
    HIDDEN,
    ToyEncoder,
    ToyOutcomeHead,
    ToyPropensity,
    make_batch,
    make_schema,
    optimiser,
    stage,
)

STEPS = 8


def _recipe(**stage_overrides: object) -> Recipe:
    """A two-head recipe whose objectives are the real supervised likelihoods.

    The toy objective returns a constant zero, which is fine for the compiler
    and useless here: a loop that never stepped would pass every assertion
    about a loss that has no gradient.
    """
    defaults: dict[str, object] = {
        "objectives": (
            Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="mean"),
            Weighted(ObservedTreatmentNLL(), weight=1.0, reduction="mean"),
        ),
        "trainable": ("encoder", "outcome_head", "propensity"),
        "steps": STEPS,
    }
    return Recipe(
        name="two_head",
        schema=make_schema(),
        system=ComponentGraph(
            [ToyEncoder(width=HIDDEN), ToyOutcomeHead(), ToyPropensity()]
        ),
        program=(stage(**(defaults | stage_overrides)),),  # type: ignore[arg-type]
        card="docs/recipes/two_head.md",
    )


def _batches(count: int = STEPS, *, seed: int | None = None) -> list[XTYBatch]:
    """A fixed list of batches, so a run is a function of the seed alone."""
    generator = torch.Generator().manual_seed(seed if seed is not None else 11)
    return [make_batch(x=torch.randn(7, 4, generator=generator)) for _ in range(count)]


def _run(recipe: Recipe | None = None) -> StageResult:
    resolved = compile(recipe if recipe is not None else _recipe())
    return run_stage(resolved, "fit", _batches(), seed=0)


# ---------------------------------------------------------------------------
# Trainable isolation (FIDELITY.md §3)
# ---------------------------------------------------------------------------


def test_parameters_outside_trainable_receive_no_gradient() -> None:
    recipe = _recipe(trainable=("outcome_head",))
    run = compile(recipe)
    run_stage(run, "fit", _batches(), seed=0)
    for name in ("encoder", "propensity"):
        for parameter_name, parameter in run.graph[name].named_parameters():
            assert parameter.grad is None or bool(
                torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
            ), f"{name}.{parameter_name} accumulated a gradient"


def test_parameters_outside_trainable_do_not_move() -> None:
    # The gradient assertion above and this one fail for different reasons: a
    # frozen component can be gradient-free and still be moved by an optimiser
    # that was handed it, or by weight decay applied to a parameter group it
    # should never have been in.
    recipe = _recipe(trainable=("outcome_head",))
    run = compile(recipe)
    before = {
        name: parameter.detach().clone()
        for name, parameter in run.graph.named_parameters()
    }
    run_stage(run, "fit", _batches(), seed=0)
    after = dict(run.graph.named_parameters())
    for name, original in before.items():
        moved = not torch.equal(original, after[name].detach())
        assert moved == ("outcome_head" in name), f"{name} moved: {moved}"


def test_freezing_is_restored_after_the_stage() -> None:
    # The stage borrows `requires_grad`; it does not own it. A later stage that
    # trains the encoder must not find it frozen by an earlier one (P8).
    run = compile(_recipe(trainable=("outcome_head",)))
    run_stage(run, "fit", _batches(), seed=0)
    assert all(parameter.requires_grad for parameter in run.graph.parameters())


def test_a_trainable_component_that_was_already_frozen_is_rejected() -> None:
    run = compile(_recipe())
    for parameter in run.graph["encoder"].parameters():
        parameter.requires_grad_(False)
    with pytest.raises(TrainingError, match="requires_grad=False"):
        run_stage(run, "fit", _batches(), seed=0)


def test_a_rejected_stage_leaves_the_graph_as_it_found_it() -> None:
    # The rejection must not corrupt the graph it protects. `propensity` is
    # last in topological order, so a scan that froze as it went would have
    # frozen the encoder and the outcome head before reaching the problem —
    # and nothing would put them back.
    run = compile(_recipe(trainable=("propensity",)))
    for parameter in run.graph["propensity"].parameters():
        parameter.requires_grad_(False)
    with pytest.raises(TrainingError, match="requires_grad=False"):
        run_stage(run, "fit", _batches(), seed=0)
    for name in ("encoder", "outcome_head"):
        assert all(p.requires_grad for p in run.graph[name].parameters())


def test_a_compiled_stage_from_another_run_is_rejected() -> None:
    # Two compatible graphs would otherwise let a foreign stage's objectives
    # and optimiser run here, while the checkpoint recorded *this* recipe's
    # plan digest — provenance that is wrong rather than missing.
    run = compile(_recipe())
    foreign = compile(_recipe(steps=2))
    with pytest.raises(TrainingError, match="not the one"):
        run_stage(run, foreign.stage("fit"), _batches(), seed=0)


def test_the_run_s_own_compiled_stage_is_accepted() -> None:
    run = compile(_recipe())
    result = run_stage(run, run.stage("fit"), _batches(), seed=0)
    assert result.stage == "fit"


def test_trainable_only_yields_exactly_the_named_components() -> None:
    graph = ComponentGraph(
        [ToyEncoder(width=HIDDEN), ToyOutcomeHead(), ToyPropensity()]
    )
    with trainable_only(graph, ("encoder",)) as parameters:
        assert [name for name, _ in parameters] == [
            "encoder.net.weight",
            "encoder.net.bias",
        ]
        assert not any(p.requires_grad for p in graph["propensity"].parameters())
    assert all(p.requires_grad for p in graph["propensity"].parameters())


# ---------------------------------------------------------------------------
# Determinism (FIDELITY.md §3)
# ---------------------------------------------------------------------------


def _sampled_run(seed: int) -> StageResult:
    """A run whose batches are *drawn*, so the seed reaches the loop.

    Determinism against a fixed list of batches and a dropout-free graph is
    determinism nothing could break: the loop would satisfy it while ignoring
    `seed` entirely. The sampler here draws from the global RNG, exactly as a
    real loader does, which is what makes the two assertions below load-
    bearing — one that the seed fixes the run, the other that it reaches it.
    """
    torch.manual_seed(4)  # the parameters start identical in every case
    run = compile(_recipe())
    return run_stage(run, "fit", _sampler(), seed=seed)


def _sampler() -> Iterator[XTYBatch]:
    while True:
        yield make_batch()


def test_the_same_seed_gives_the_same_loss_trace() -> None:
    assert _sampled_run(7).trace == _sampled_run(7).trace


def test_the_same_seed_gives_the_same_parameters() -> None:
    # A trace can agree while the parameters do not — the trace is measured
    # before each step, so the last update is not in it at all.
    first, second = _sampled_run(7), _sampled_run(7)
    for name, parameter in first.checkpoint.parameters.items():
        assert torch.equal(parameter, second.checkpoint.parameters[name])


def test_a_different_seed_is_visible_in_the_trace() -> None:
    # Determinism is only worth asserting if the seed reaches anything: a loop
    # that ignored it would satisfy the two tests above trivially.
    assert _sampled_run(7).trace != _sampled_run(8).trace


# ---------------------------------------------------------------------------
# No stage mutates the source dataset (DESIGN.md §7.1)
# ---------------------------------------------------------------------------


def test_the_batches_a_run_consumed_are_unchanged() -> None:
    batches = _batches()
    before = [batch.clone() for batch in batches]
    run_stage(compile(_recipe()), "fit", batches, seed=0)
    assert all(
        batch.equal_to(original)
        for batch, original in zip(batches, before, strict=True)
    )


# ---------------------------------------------------------------------------
# The loop does what it says
# ---------------------------------------------------------------------------


def test_the_optimiser_actually_steps() -> None:
    # Not evidence about any paper (`CLAUDE.md`): evidence that the optimiser
    # is connected to the parameters the stage named. Repeating one batch makes
    # the descent monotone, so this is a wiring assertion and not a claim about
    # convergence.
    batch = make_batch()
    run = compile(_recipe())
    result = run_stage(run, "fit", [batch] * STEPS, seed=0)
    assert result.trace[-1] < result.trace[0]
    assert len(result.records) == STEPS


def test_every_step_is_recorded_with_its_terms() -> None:
    result = _run()
    record = result.records[0]
    assert [term.name for term in record.terms] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
    ]
    assert record.rows == 7
    assert record.grad_norm > 0


def test_the_learning_rate_follows_its_schedule() -> None:
    # A warmup stated in a card and not applied by the loop is invisible in the
    # loss curve and fatal to a reproduction, so the rate each step ran at is
    # recorded rather than assumed.
    warmup = optimiser(lr=0.1, lr_schedule=Ramp(0.0, 1.0, steps=4))
    result = run_stage(compile(_recipe(optimiser=warmup)), "fit", _batches(), seed=0)
    assert [round(record.lr, 6) for record in result.records[:5]] == [
        0.0,
        0.025,
        0.05,
        0.075,
        0.1,
    ]
    assert result.records[-1].lr == pytest.approx(0.1)


def test_clipping_bounds_the_step() -> None:
    clipped = optimiser(
        lr=0.5, clipping=GradientClipping.by_norm(1e-4), weight_decay=WeightDecay.none()
    )
    run = compile(_recipe(optimiser=clipped))
    before = {
        name: parameter.detach().clone()
        for name, parameter in run.graph.named_parameters()
    }
    result = run_stage(run, "fit", _batches(), seed=0)
    for name, parameter in run.graph.named_parameters():
        moved = float((parameter.detach() - before[name]).norm())
        # lr 0.5 against a clipped norm of 1e-4 bounds the whole run's travel.
        assert moved < 0.5 * 1e-4 * STEPS + 1e-9
    # The recorded norm is the one *before* clipping — the diagnostic that
    # says the clip was active at all.
    assert result.records[0].grad_norm > 1e-4


def test_a_schedule_that_would_ascend_the_gradient_is_rejected() -> None:
    # A `Schedule` checks a weight is finite, not that it is positive — a loss
    # weight may be negative. Applied to the learning rate it would run
    # gradient ascent, and writing into the parameter groups is the one path
    # around torch's constructor-time check.
    ascending = optimiser(lr=0.1, lr_schedule=Ramp(-1.0, 1.0, steps=4))
    with pytest.raises(Exception, match="ascends the gradient"):
        run_stage(compile(_recipe(optimiser=ascending)), "fit", _batches(), seed=0)


def test_a_component_left_in_eval_mode_stays_there() -> None:
    # `graph.train()` is recursive, so restoring one saved root flag would put
    # a submodule the caller had placed in eval mode back into training — the
    # frozen-BatchNorm case, silently undone.
    run = compile(_recipe())
    run.graph["propensity"].eval()
    run_stage(run, "fit", _batches(), seed=0)
    assert not run.graph["propensity"].training
    assert run.graph["encoder"].training


def test_a_batch_source_that_runs_dry_is_an_error() -> None:
    with pytest.raises(TrainingError, match="ran dry at step 3"):
        run_stage(compile(_recipe()), "fit", _batches(3), seed=0)


def test_a_source_that_yields_something_else_is_an_error() -> None:
    def wrong() -> Iterator[object]:
        yield "not a batch"

    with pytest.raises(TrainingError, match="XTYBatch"):
        run_stage(compile(_recipe()), "fit", wrong(), seed=0)  # type: ignore[arg-type]


def test_weight_decay_reaches_only_the_group_it_declares() -> None:
    # `optimisation.weight_decay` carries whether biases are decayed
    # (`FIDELITY.md` §2). With no objective at all — an empty batch source is
    # not available here, so use a decay large enough to dominate — the bias of
    # a frozen-gradient parameter would still shrink if it were in the decayed
    # group.
    spec = optimiser(
        name="sgd", lr=0.1, weight_decay=WeightDecay(value=0.5, on_norm_and_bias=False)
    )
    run = compile(_recipe(optimiser=spec))
    groups = spec.build(list(run.graph.named_parameters())).param_groups
    assert [group["weight_decay"] for group in groups] == [0.5, 0.0]
    assert all(parameter.ndim >= 2 for parameter in groups[0]["params"])
    assert all(parameter.ndim < 2 for parameter in groups[1]["params"])


# ---------------------------------------------------------------------------
# The probe stays off unless a run asks for it (DESIGN.md §6.2)
# ---------------------------------------------------------------------------


def test_the_gradient_probe_is_off_by_default() -> None:
    result = _run()
    assert all(record.gradients is None for record in result.records)


def test_an_enabled_probe_reports_on_its_own_cadence() -> None:
    result = run_stage(
        compile(_recipe()),
        "fit",
        _batches(),
        seed=0,
        probe=GradientProbe.periodic(every=4),
    )
    measured = [r.step for r in result.records if r.gradients is not None]
    assert measured == [0, 4]
    report = result.records[0].gradients
    assert report is not None
    assert set(report.norms) == {"observed_outcome_nll", "observed_treatment_nll"}


# ---------------------------------------------------------------------------
# Provenance is earned, not declared (DESIGN.md §7.1)
# ---------------------------------------------------------------------------


def test_the_checkpoint_holds_exactly_the_rows_the_loop_stepped_on() -> None:
    batches = _batches()
    result = run_stage(compile(_recipe()), "fit", batches, seed=0)
    expected = torch.unique(torch.cat([batch.row_id for batch in batches]))
    assert torch.equal(result.checkpoint.trained_on_row_ids, expected)


def test_the_checkpoint_holds_only_the_trained_components() -> None:
    result = run_stage(
        compile(_recipe(trainable=("outcome_head",))), "fit", _batches(), seed=0
    )
    assert result.checkpoint.components == ("outcome_head",)
    assert all(name.startswith("outcome_head") for name in result.checkpoint.parameters)


def test_the_checkpoint_records_the_plan_it_ran_under() -> None:
    run = compile(_recipe())
    result = run_stage(run, "fit", _batches(), seed=0)
    assert result.checkpoint.plan_digest == run.plan.digest
    assert result.checkpoint.seed == 0
    assert result.checkpoint.steps == STEPS
    assert result.checkpoint.fold is None


# ---------------------------------------------------------------------------
# The run directory
# ---------------------------------------------------------------------------


def test_a_run_writes_its_plan_checkpoint_and_log(tmp_path: Path) -> None:
    run = compile(_recipe())
    directory = RunDirectory.create(tmp_path / "run")
    result = run_stage(run, "fit", _batches(), seed=0, run_dir=directory)

    assert (directory.root / "plan.txt").read_text() == run.plan.render()
    reloaded = directory.read_checkpoint("fit")
    assert torch.equal(
        reloaded.trained_on_row_ids, result.checkpoint.trained_on_row_ids
    )
    log = directory.read_log("fit")
    assert len(log) == STEPS
    assert log[0]["step"] == 0
    assert [term["name"] for term in log[0]["terms"]] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
    ]


def test_a_second_run_into_one_directory_is_refused(tmp_path: Path) -> None:
    run = compile(_recipe())
    directory = RunDirectory.create(tmp_path / "run")
    run_stage(run, "fit", _batches(), seed=0, run_dir=directory)
    with pytest.raises(Exception, match="already exists"):
        run_stage(run, "fit", _batches(), seed=1, run_dir=directory)


def test_the_stage_result_renders(tmp_path: Path) -> None:
    del tmp_path
    result = run_stage(compile(_recipe()), "fit", _batches(), seed=0)
    rendered = result.render(every=4)
    assert "stage fit (two_head, seed 0)" in rendered
    assert "step 0" in rendered
    assert "step 4" in rendered
    assert "checkpoint two_head/fit" in rendered


def test_a_checkpoint_cannot_be_constructed_by_hand() -> None:
    # The provenance fields are computed by the factory from the run, so a
    # direct call is the one way they could be asserted instead (§7.1).
    with pytest.raises(Exception, match="executor factory"):
        Checkpoint(
            recipe="two_head",
            stage="fit",
            fold=None,
            trained_on_row_ids=torch.zeros(3, dtype=torch.long),
            parameters={},
            components=(),
            steps=1,
            seed=0,
            plan_digest="0" * 64,
        )


def test_a_constant_schedule_leaves_the_rate_alone() -> None:
    spec = optimiser(lr=0.02, lr_schedule=Constant(1.0))
    result = run_stage(compile(_recipe(optimiser=spec)), "fit", _batches(), seed=0)
    assert {record.lr for record in result.records} == {0.02}
