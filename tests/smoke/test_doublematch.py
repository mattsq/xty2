"""Tier 1 — DoubleMatch on `fixmatch.md` §6.1's fixture, paired against `w_s = 0`.

The DGP, the label budget, the quota and the seeds are `fixmatch`'s, and they
are **imported** rather than restated: `doublematch.md` §6.1 says the fixture is
that one "in full and without modification", and a copy would be a second thing
to keep true. What is new here is what the run is measured on. Eq. (3) has no
gate, so mask rate and impurity say nothing about it; the questions are whether
it learns anything, and what the representation does while it learns — which
its own loss cannot answer, because a collapsed encoder attains the best
possible value. What eq. (3) then does to held-out treatment prediction came
out mixed on one seed and is **not** asserted anywhere here: the card's §6.2
records the numbers and §6's ten-seed Tier 2 target owns the claim. Tier 1 is
the wiring tier (`FIDELITY.md` §3), and a directional single-seed assertion is
a result wearing wiring's clothes.

The last test is the one that produced deviation 9. It is kept executable, and
differential, so that the encoder's output normalisation cannot quietly come
back — and so that the fact both encoders collapse, while only one recovers,
cannot be rounded off into "the declared one does not collapse".
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.components import MLPEncoder
from xty2.components._nn import CFRNET_INITIALISATION, TORCH_LINEAR_INITIALISATION
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    ComponentGraph,
    Constant,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    XTYBatch,
    compile,
)
from xty2.recipes import doublematch
from xty2.recipes.tarnet import ENCODER_WIDTHS
from xty2.training import StageResult, run_stage

from tests.smoke.test_fixmatch import (
    SEPARATED,
    STEPS,
    _dataset,
    _on_the_training_scale,
    _Population,
    _populations,
    _schema,
    _with_steps,
)

SELF_SUPERVISED = "cosine_feature_consistency"
PSEUDO_LABEL = "pseudo_label_treatment_nll"
COLLAPSE_STEPS = 400
"""Long enough for the declared encoder to have left the basin (card §6.2)."""
BATCH_ROWS = 512
"""`B + mu B`, the quota the imported fixture's recipe declares."""


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _with_weight(recipe: Recipe, weight: float) -> Recipe:
    """`w_s`, and nothing else: the paired ablation of eq. (4)."""
    stage = recipe.program[0]
    term = replace(stage.objectives[3], weight=Constant(weight))
    objectives = (*stage.objectives[:3], term, *stage.objectives[4:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _with_encoder(recipe: Recipe, initialisation: str) -> Recipe:
    """Swap the encoder's initialisation, and nothing else.

    Used for **both** arms of the collapse comparison, including the one whose
    initialisation it does not change. Replacing the encoder consumes
    construction RNG in a different order from building the recipe, so an arm
    built this way and an arm left alone would differ in their draws as well as
    in the field under test. Putting both through the same path costs one
    redundant construction and buys an A/B on one flag.
    """
    encoder = MLPEncoder(
        input_dim=recipe.schema.num_features,
        widths=ENCODER_WIDTHS,
        activation="elu",
        normalisation="row_l2",
        dropout=0.0,
        initialisation=initialisation,
    )
    return replace(
        recipe,
        system=ComponentGraph([encoder, *list(recipe.system.components)[1:]]),
    )


def _term(result: StageResult, step: int, name: str) -> object:
    return next(term for term in result.records[step].terms if term.name == name)


def _mean(result: StageResult, steps: range, name: str, field: str) -> float:
    terms = [_term(result, step, name) for step in steps]
    values: list[float]
    if field == "value":
        values = [float(term.value) for term in terms]  # type: ignore[attr-defined]
    else:
        values = [float(term.diagnostics[field]) for term in terms]  # type: ignore[attr-defined]
    return sum(values) / len(values)


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    treatment_nll: float
    student_treatment_nll: float
    outcome_nll: float
    frequency_nll: float


def _evaluate(
    run: CompiledRun, result: StageResult, test: _Population, train: XTYBatch
) -> _Metrics:
    """The EMA the paper reports from, and the network it averages.

    Both, deliberately. `fixmatch.md` §6.2 records that an EMA number can
    improve while the network under it gets worse, so a claim resting on the
    EMA alone is a claim about a reporting device. Eq. (3) trains the encoder
    both arms share, which is exactly where that divergence would hide.
    """
    schema = run.recipe.schema
    assert result.teacher is not None
    with torch.no_grad():
        values = result.teacher.graph.evaluate(
            test.batch, schema=schema, only=run.graph.names
        )
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(outcome, GaussianOutcome)
        student = run.graph.evaluate(test.batch, schema=schema, only=run.graph.names)
        student_propensity = student[Port.T_GIVEN_X]
        assert isinstance(student_propensity, CategoricalTreatment)

        frequencies = torch.bincount(train.t[train.t_observed], minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(test.batch.batch_size, -1)
        return _Metrics(
            run=run,
            result=result,
            treatment_nll=float(F.nll_loss(propensity.log_probs, test.batch.t)),
            student_treatment_nll=float(
                F.nll_loss(student_propensity.log_probs, test.batch.t)
            ),
            outcome_nll=float(-outcome.log_prob(test.batch.y, test.batch.t).mean()),
            frequency_nll=float(F.nll_loss(baseline, test.batch.t)),
        )


def _run(recipe: Recipe, train: XTYBatch, test: _Population, steps: int) -> _Metrics:
    run = compile(_with_steps(recipe, steps))
    result = run_stage(run, "joint_fit", _dataset(train), seed=90_010)
    return _evaluate(run, result, _on_the_training_scale(test, result), train)


@pytest.fixture(scope="module")
def paired_fit() -> tuple[_Metrics, _Metrics]:
    """DoubleMatch and its `w_s = 0` ablation, same seeds and same batches."""
    schema = _schema()
    train, test = _populations(SEPARATED, seed=90_001)

    torch.manual_seed(90_006)
    scheduled = doublematch(schema)
    torch.manual_seed(90_006)
    ablated = _with_weight(doublematch(schema), 0.0)
    for name, value in scheduled.system.state_dict().items():
        assert torch.equal(value, ablated.system.state_dict()[name])

    return (
        _run(scheduled, train, test, STEPS),
        _run(ablated, train, test, STEPS),
    )


def test_the_self_supervised_term_learns_the_invariance(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Eq. (3) is descended, and the ablation's copy of it is not.

    The ablated arm computes the same term at weight 0 and logs it without
    descending it. Note what that control is and is not: `h` is trained by
    eq. (3) and by nothing else, so at `w_s = 0` it keeps its initialisation
    and its output is dominated by its own random bias (card §7). The
    ablation's ≈0 is therefore close to definitional — a fixed random direction
    against `z` — and it is asserted here as a floor on the *logging path*,
    not as evidence that the other four terms would have failed to align the
    representation. The adversarial review corrected an earlier version of this
    docstring that claimed the latter.
    """
    scheduled, ablated = paired_fit
    late = range(STEPS - 100, STEPS)
    assert _mean(scheduled.result, range(1), SELF_SUPERVISED, "value") > -0.2
    assert _mean(scheduled.result, late, SELF_SUPERVISED, "value") < -0.5
    assert abs(_mean(ablated.result, late, SELF_SUPERVISED, "value")) < 0.2


def test_the_representation_does_not_collapse_at_any_point(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Over the whole trajectory, not over its last hundred steps.

    A collapsed encoder attains eq. (3)'s minimum, so "the loss went down" is
    compatible with the worst possible outcome, and the concentration pair is
    the only thing that can tell the two apart. The first version of this test
    averaged the last hundred steps of a 3,000-step run and called that "does
    not collapse"; an adversarial review pointed out that the recipe it was
    then guarding spent 135 steps at a concentration above 0.99 before
    recovering, which a terminal reading cannot see. Deviation 9 is the fix for
    that, and this is the assertion that would have caught it.
    """
    scheduled, ablated = paired_fit
    trajectory = [
        _term(scheduled.result, step, SELF_SUPERVISED).diagnostics[  # type: ignore[attr-defined]
            "target_concentration"
        ]
        for step in range(STEPS)
    ]
    assert max(trajectory) < 0.9
    assert trajectory[-1] < 0.5
    # Not asserted as "more concentrated than the ablation": measured on this
    # architecture the term leaves the representation *less* concentrated than
    # the `w_s = 0` arm does, where on the one deviation 9 replaced it left it
    # more so (card §6.2). Neither direction is a property of eq. (3) worth
    # holding a recipe to.
    assert (
        0.0
        < _mean(
            ablated.result,
            range(STEPS - 100, STEPS),
            SELF_SUPERVISED,
            "target_concentration",
        )
        < 0.9
    )


def test_the_term_trains_on_rows_the_gate_rejects(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The paper's title claim, as an assertion about one step's row counts.

    Eq. (2) keeps a fraction of the batch and eq. (3) keeps all of it, in the
    same step, from the same two forward passes. This is `BACKLOG.md` §2.6's
    "row eligibility for one loss must not determine eligibility for all
    losses", and it needs no framework mechanism beyond `Objective.rows`.
    """
    scheduled, _ = paired_fit
    late = range(STEPS - 100, STEPS)
    coverage = _mean(scheduled.result, late, PSEUDO_LABEL, "coverage")
    assert 0.0 < coverage < 1.0
    for step in (0, STEPS // 2, STEPS - 1):
        assert _term(scheduled.result, step, SELF_SUPERVISED).n == BATCH_ROWS  # type: ignore[attr-defined]
        assert _term(scheduled.result, step, PSEUDO_LABEL).n == BATCH_ROWS  # type: ignore[attr-defined]


def test_the_projection_head_is_actually_trained(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """`theta_h` is in eq. (6) and in the optimiser, so it must have moved."""
    scheduled, _ = paired_fit
    initial = scheduled.run.initial_parameters()
    trained = dict(scheduled.run.graph.named_parameters())
    moved = [
        name
        for name in initial
        if "projection_head" in name and not torch.equal(initial[name], trained[name])
    ]
    assert moved


def test_the_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, _ = paired_fit
    assert scheduled.treatment_nll < scheduled.frequency_nll


def test_both_arms_learn_a_propensity_worth_having(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The floor, asserted before anything is compared.

    A paired ratio between two models that have both learned nothing is a
    number with no content — and §6.2's `row_l2` table is exactly that failure,
    so this is not hypothetical.
    """
    scheduled, ablated = paired_fit
    assert scheduled.treatment_nll < 0.75 * scheduled.frequency_nll
    assert ablated.treatment_nll < 0.75 * ablated.frequency_nll


def test_the_outcome_stack_is_not_damaged(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Eq. (3) trains the encoder the outcome head reads, so this is not free."""
    scheduled, ablated = paired_fit
    assert scheduled.outcome_nll < 1.05 * ablated.outcome_nll


def test_only_the_declared_initialisation_stays_out_of_the_collapse_basin() -> None:
    """Deviation 9, kept executable — and **differential**, which it must be.

    An earlier version ran one arm alone and asserted concentration, loss,
    coverage and a baseline-level NLL at 200 steps; the adversarial review
    showed the architecture that recipe *shipped* satisfied all four of them
    too, so the test could not tell the condemned configuration from the
    declared one and guarded nothing.

    Both arms now go through the same construction path and differ in one
    field. Under CFRNet's initialisation the representation is a hundredth of
    the scale eq. (3) is written for, `row_l2` hands that factor to the
    encoder, and the run goes to one direction and stays: the gate never opens
    and the propensity never beats the training frequencies. Under torch's it
    does not happen at all.
    """
    train, test = _populations(SEPARATED, seed=90_001)
    torch.manual_seed(90_006)
    small = _run(
        _with_encoder(doublematch(_schema()), CFRNET_INITIALISATION),
        train,
        test,
        COLLAPSE_STEPS,
    )
    torch.manual_seed(90_006)
    declared = _run(
        _with_encoder(doublematch(_schema()), TORCH_LINEAR_INITIALISATION),
        train,
        test,
        COLLAPSE_STEPS,
    )
    late = range(COLLAPSE_STEPS - 50, COLLAPSE_STEPS)

    assert _mean(small.result, late, SELF_SUPERVISED, "target_concentration") > 0.99
    assert _mean(small.result, late, SELF_SUPERVISED, "value") < -0.99
    assert _mean(small.result, late, PSEUDO_LABEL, "coverage") == 0.0
    assert small.treatment_nll > 0.95 * small.frequency_nll

    trajectory = [
        _term(declared.result, step, SELF_SUPERVISED).diagnostics[  # type: ignore[attr-defined]
            "target_concentration"
        ]
        for step in range(COLLAPSE_STEPS)
    ]
    assert max(trajectory) < 0.9
    assert _mean(declared.result, late, PSEUDO_LABEL, "coverage") > 0.0
    assert declared.treatment_nll < 0.9 * declared.frequency_nll
