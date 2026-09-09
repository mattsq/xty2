"""The card's three-seed, four-arm Tier 1 study, without directional gates.

`docs/recipes/barlow_twins.md` §6.3 declares this packet: bases 419, 523 and
631, the §6.2 offsets, split sizes and four arms, at 200 pretraining and 300
downstream steps with the 1,000-step marginal ramp left alone. What it asserts
is wiring — finite losses and gradients, a normalised propensity, an encoder
that moved, and the stage transition the card describes — and what it reports
is §6.4's metrics. It promotes no direction to an assertion: §6.4's bounds are
ten-seed, full-budget statements, and a three-seed smoke fit at a fifth of the
pretraining budget is not evidence for or against them.

The four arms are §6.2's. `full` and `diagonal_only` differ by exactly one
number — `lambda`, set to zero, with the objective still declared — and the
test checks that they are otherwise the same fit: same initial tensors, same
sampled row IDs and same cached view draws. `no_pretrain` drops the stage and
the inheritance edge and shifts the execution seed by `STREAM_STRIDE` so that
its downstream row stream is stage index 1's. `vicreg` is the contextual arm
of §2, given the same encoder and head initial tensors because its expander
consumes construction RNG differently (`components/barlow_twins.py`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor
from xty2.core import (
    CategoricalTreatment,
    ComponentGraph,
    GaussianOutcome,
    Port,
    Program,
    Schema,
    TrainingPopulation,
    ViewSpec,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    SEPARATED,
    continuous_schema,
    on_the_training_scale,
    take,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.causal import (
    candidate_treatment_means,
    sqrt_pehe,
    treatment_contrast,
)
from xty2.evaluation.vicreg_views import OracleSymmetry
from xty2.objectives.barlow_twins import cross_correlation
from xty2.recipes import barlow_twins, vicreg
from xty2.recipes.barlow_twins import (
    BARLOW_TWINS_BATCH_SIZE,
    BARLOW_TWINS_OBSERVED_TREATMENTS,
    NORMALISATION_EPSILON,
    POPULATION_CORRECTION,
)
from xty2.training import STREAM_STRIDE, executors, run_program

PRETRAIN_STEPS = 200
JOINT_FIT_STEPS = 300
"""Card §6.3's explicit smoke overrides, not the §4 budget."""

TRAIN_ROWS = 1024
TEST_ROWS = 2048
EVAL_BATCHES = 16
BRANCHES = 2
"""Card §6.2: 16 disjoint held-out batches of 128 rows, two views each."""

ACTIVE_VARIANCE = 100.0 * NORMALISATION_EPSILON
"""Card §6.4's activity cutoff: a coordinate counts when its *raw* population
variance clears one hundred epsilons in **both** branches, which is what stops
`C` from being an artefact of the epsilon in its own denominator."""

ARMS = ("full", "diagonal_only", "no_pretrain", "vicreg")
PROJECTORS = {"vicreg": "vicreg_expander"}
"""Every arm but the contextual one embeds through `barlow_twins_projector`."""

HEADS = ("tarnet_head", "categorical_propensity")


@pytest.fixture(scope="module", autouse=True)
def _one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _snapshot(graph: ComponentGraph) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in graph.state_dict().items()}


def _transferred(name: str) -> bool:
    """The tensors §6.2 holds identical across all four arms."""
    return "mlp_encoder." in name or any(head + "." in name for head in HEADS)


def _parameter_norm(parameters: dict[str, Tensor], prefix: str) -> float:
    total = math.fsum(
        float(value.double().square().sum())
        for name, value in parameters.items()
        if name.startswith(prefix)
    )
    return math.sqrt(total)


def _held_out_views(
    schema: Schema,
    batches: Sequence[XTYBatch],
    population: TrainingPopulation,
    *,
    base: int,
) -> tuple[tuple[XTYBatch, XTYBatch], ...]:
    """Card §6.2's two independent draws per evaluation batch, shared by arms.

    Realised once from the shared training population, so the arms differ by
    the encoder they run on these rows and by nothing about the rows.
    """
    symmetry = OracleSymmetry()
    return tuple(
        (
            symmetry.apply(
                batch,
                schema,
                population=population,
                generator=torch.Generator().manual_seed(
                    base + 20000 + BRANCHES * index
                ),
            ),
            symmetry.apply(
                batch,
                schema,
                population=population,
                generator=torch.Generator().manual_seed(
                    base + 20001 + BRANCHES * index
                ),
            ),
        )
        for index, batch in enumerate(batches)
    )


def _embedding_metrics(
    graph: ComponentGraph,
    schema: Schema,
    views: Sequence[tuple[XTYBatch, XTYBatch]],
    *,
    projector: str,
) -> dict[str, float]:
    """Card §6.4's per-seed embedding statistics at the pretraining checkpoint.

    One `C` per held-out batch rather than one per branch: §6.4's `a` and `r`
    are reductions of the cross-view matrix, so the two branches enter together.
    `cross_correlation` is imported rather than rewritten, as §6.2 asks — a
    second transcription is how the diagnostic stops being about the loss.
    """
    alignment: list[float] = []
    redundancy: list[float] = []
    active: list[float] = []
    diagonal_error: list[float] = []
    variances: tuple[list[float], list[float]] = ([], [])
    concentration: list[float] = []
    with torch.no_grad():
        for pair in views:
            # §6.2's "independent branch draws", at the diagnostic rather than
            # at training: one seed for both branches would make `C` a view's
            # correlation with itself and `a` almost free.
            assert not torch.equal(pair[0].x, pair[1].x)
            embeddings: list[Tensor] = []
            live: list[Tensor] = []
            for branch, view in enumerate(pair):
                value = graph.evaluate(
                    view, schema=schema, only=("mlp_encoder", projector)
                )[Port.X_PROJ]
                assert isinstance(value, Tensor)
                embeddings.append(value)
                # The *raw* variance, before the loss's normalisation: §6.4's
                # activity cutoff is about the embedding, not about `C`.
                variance = value.var(dim=0, correction=POPULATION_CORRECTION)
                variances[branch].append(float(variance.mean()))
                live.append(variance > ACTIVE_VARIANCE)
            first, second = embeddings
            width = first.shape[1]
            correlation = cross_correlation(
                first,
                second,
                epsilon=NORMALISATION_EPSILON,
                correction=POPULATION_CORRECTION,
            )
            diagonal = correlation.diagonal()
            off = correlation.masked_select(
                ~torch.eye(width, dtype=torch.bool)
            ).square()
            alignment.append(float(diagonal.mean()))
            redundancy.append(float(off.sum()) / (width * (width - 1)))
            diagonal_error.append(float((1.0 - diagonal).square().sum()) / width)
            active.append(float((live[0] & live[1]).double().mean()))
            centred = first - first.mean(dim=0)
            covariance = centred.T @ centred / first.shape[0]
            eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp(min=0.0)
            total = float(eigenvalues.sum())
            # No arm here is a collapse control, so a null covariance is a
            # failure to report rather than a share to invent.
            assert total > 0.0
            concentration.append(float(eigenvalues.max()) / total)

    def mean(values: Sequence[float]) -> float:
        assert len(values) == EVAL_BATCHES
        return math.fsum(values) / EVAL_BATCHES

    return {
        "diagonal_alignment": mean(alignment),
        "redundancy": mean(redundancy),
        "active_fraction": mean(active),
        "diagonal_error": mean(diagonal_error),
        "raw_variance_first": mean(variances[0]),
        "raw_variance_second": mean(variances[1]),
        "covariance_top_eigenvalue_share": mean(concentration),
    }


@pytest.mark.parametrize("base", [419, 523, 631])
def test_paired_mechanism_study(base: int, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = continuous_schema(6)
    train = two_cluster_population(
        TRAIN_ROWS, seed=base + 1, row_offset=0, low=SEPARATED
    )
    test = two_cluster_population(
        TEST_ROWS, seed=base + 2, row_offset=10000, low=SEPARATED
    )
    data = training_dataset(schema, train.batch)
    paired_rows: dict[str, list[Tensor]] = {}
    paired_populations: dict[str, TrainingPopulation] = {}
    paired_views: list[Tensor] = []
    initial: dict[str, Tensor] = {}
    report: dict[str, dict[str, float]] = {}
    pretrained: dict[str, dict[str, Tensor]] = {}
    rows: dict[str, list[Tensor]] = {}
    views: list[Tensor] = []
    transition: dict[str, dict[str, Tensor]] = {}
    optimisers: list[torch.optim.Optimizer] = []
    # Mutated rather than rebound, so the traced closures below read the
    # arm being run rather than closing over a loop variable.
    embedded: dict[str, str] = {}
    current_stage = ""
    original_step = executors._step
    original_view = ViewSpec.apply
    original_evaluate = ComponentGraph.evaluate

    for arm in ARMS:
        projector = PROJECTORS.get(arm, "barlow_twins_projector")
        embedded["projector"] = projector
        torch.manual_seed(base + 6)
        build = vicreg if arm == "vicreg" else barlow_twins
        recipe = build(
            schema,
            first_transforms=(OracleSymmetry(),),
            second_transforms=(OracleSymmetry(),),
        )
        pretrain, fit = recipe.program
        pretrain = replace(pretrain, steps=PRETRAIN_STEPS)
        fit = replace(fit, steps=JOINT_FIT_STEPS)
        if arm == "diagonal_only":
            # One number, not one term: the objective stays declared. The name
            # is checked rather than assumed, because a mismatch here would
            # silently make this arm a second copy of `full`.
            assert [term.objective.name for term in pretrain.objectives].count(
                "cross_correlation_off_diagonal"
            ) == 1
            pretrain = replace(
                pretrain,
                objectives=tuple(
                    replace(term, weight=0.0)
                    if term.objective.name == "cross_correlation_off_diagonal"
                    else term
                    for term in pretrain.objectives
                ),
            )
        stages = (
            (replace(fit, initialise_from=None),)
            if arm == "no_pretrain"
            else (pretrain, fit)
        )
        if arm == "vicreg":
            # §6.2: the contextual arm inherits the encoder and the heads. Its
            # expander draws a different number of tensors, so its own
            # construction seed cannot deliver them. The copy happens before
            # `compile`, because compilation snapshots the graph and every
            # stage is restored to that snapshot (`training/executors.py`).
            system = recipe.system.state_dict()
            for name, value in initial.items():
                if _transferred(name):
                    assert name in system
                    system[name].copy_(value)
        run = compile(replace(recipe, program=Program(stages)))
        start = _snapshot(run.graph)
        if not initial:
            initial = start
        assert all(
            torch.equal(value, initial[name])
            for name, value in start.items()
            if arm != "vicreg" or _transferred(name)
        )
        assert any(_transferred(name) for name in start)
        rows.clear()
        rows.update({stage.name: [] for stage in stages})
        views.clear()
        transition.clear()
        optimisers.clear()

        def traced_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal current_stage
            active_run, compiled, batch, _, optimiser, _, step = args
            current_stage = compiled.name
            rows[current_stage].append(batch.row_id.clone())
            if step == 0:
                transition[current_stage] = _snapshot(active_run.graph)
                assert not optimiser.state
                assert all(optimiser is not previous for previous in optimisers)
                optimisers.append(optimiser)
            return original_step(*args, **kwargs)

        def traced_view(*args: Any, **kwargs: Any) -> Any:
            assert current_stage == "pretrain"
            population = kwargs["population"]
            assert population is not None
            assert torch.equal(population.rows.row_id, train.batch.row_id)
            result = original_view(*args, **kwargs)
            views.append(result.x.clone())
            return result

        def traced_evaluate(*args: Any, **kwargs: Any) -> Any:
            names = tuple(kwargs["only"])
            if current_stage == "pretrain":
                assert not any(head in names for head in HEADS)
            else:
                assert embedded["projector"] not in names
            return original_evaluate(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(executors, "_step", traced_step)
            patch.setattr(ViewSpec, "apply", traced_view)
            patch.setattr(ComponentGraph, "evaluate", traced_evaluate)
            result = run_program(
                run,
                {stage.name: data for stage in stages},
                seed=base + 10000 + (STREAM_STRIDE if arm == "no_pretrain" else 0),
            )

        for stage in result.stages:
            assert all(
                math.isfinite(r.total) and math.isfinite(r.grad_norm)
                for r in stage.records
            )
            assert all(
                math.isfinite(term.value) for r in stage.records for term in r.terms
            )
            assert stage.population is not None
            assert (
                int(stage.population.rows.t_observed.sum())
                == BARLOW_TWINS_OBSERVED_TREATMENTS
            )
            assert torch.equal(stage.population.rows.row_id, train.batch.row_id)
            torch.testing.assert_close(
                stage.population.statistics["x_location"], train.batch.x.mean(0)
            )
            for key, expected in {
                "x_scale": train.batch.x.std(0, correction=0),
                "y_location": train.batch.y.mean(0),
                "y_scale": train.batch.y.std(0, correction=0),
            }.items():
                torch.testing.assert_close(stage.population.statistics[key], expected)
            assert set(stage.checkpoint.trained_on_row_ids.tolist()) <= set(
                range(TRAIN_ROWS)
            )
            if arm == "full":
                paired_rows[stage.stage] = rows[stage.stage]
                paired_populations[stage.stage] = stage.population
            else:
                # Equal label counts do not establish a paired comparison.
                # Match the actual labels and fitted scale for each stage;
                # pretraining itself never consumes the treatment mask.
                paired = paired_populations[stage.stage]
                assert torch.equal(
                    stage.population.rows.t_observed, paired.rows.t_observed
                )
                for key, value in paired.statistics.items():
                    assert torch.equal(stage.population.statistics[key], value)
                assert len(rows[stage.stage]) == len(paired_rows[stage.stage])
                assert all(
                    torch.equal(a, b)
                    for a, b in zip(
                        rows[stage.stage], paired_rows[stage.stage], strict=True
                    )
                )

        expected_views = 0 if arm == "no_pretrain" else BRANCHES * PRETRAIN_STEPS
        assert len(views) == expected_views
        if arm == "full":
            paired_views = list(views)
        elif arm != "no_pretrain":
            assert all(
                torch.equal(a, b) for a, b in zip(views, paired_views, strict=True)
            )
        fit_start = transition["joint_fit"]
        for name, value in start.items():
            if any(head + "." in name for head in HEADS):
                assert torch.equal(value, fit_start[name])
        final = _snapshot(run.graph)
        assert any(
            not torch.equal(value, final[name])
            for name, value in fit_start.items()
            if "mlp_encoder." in name
        )
        for name, value in fit_start.items():
            if projector + "." in name:
                assert torch.equal(value, final[name])

        population = result.stage("joint_fit").population
        assert population is not None
        heldout = on_the_training_scale(test.batch, population)
        run.graph.eval()
        with torch.no_grad():
            predictions = run.graph.evaluate(
                heldout, schema=schema, only=("mlp_encoder", *HEADS)
            )
            propensity = predictions[Port.T_GIVEN_X]
            outcome = predictions[Port.Y_GIVEN_XT]
            assert isinstance(propensity, CategoricalTreatment)
            assert isinstance(outcome, GaussianOutcome)
            probabilities = propensity.log_probs.exp()
            assert bool(torch.isfinite(probabilities).all())
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(TEST_ROWS))
            metrics = {
                "treatment_nll": float(-propensity.log_prob(heldout.t).mean()),
                "outcome_nll": float(-outcome.log_prob(heldout.y, heldout.t).mean()),
            }
            means = candidate_treatment_means(
                outcome,
                batch_size=TEST_ROWS,
                num_treatments=schema.treatment_cardinality,
                device=heldout.t.device,
            )
            # Report conditional-mean effect error in original DGP units,
            # not against realised noisy outcome differences (§6.4).
            effect = treatment_contrast(means) * population.statistics["y_scale"]
            metrics["treatment_effect_rmse"] = float(
                sqrt_pehe(effect, test.true_effect)
            )
            if arm != "no_pretrain":
                checkpoint = result.stage("pretrain").checkpoint
                pretrained[arm] = {
                    name: value.detach().clone()
                    for name, value in checkpoint.parameters.items()
                    if name.startswith("mlp_encoder.")
                }
                for name, value in {
                    **checkpoint.parameters,
                    **checkpoint.buffers,
                }.items():
                    assert torch.equal(value, fit_start["_components." + name])
                assert any(
                    not torch.equal(value, start["_components." + name])
                    for name, value in checkpoint.parameters.items()
                    if name.startswith("mlp_encoder.")
                )
                # Card §6.2 evaluates the terminal pretraining checkpoint,
                # before fine-tuning, with the hidden BN in eval mode on its
                # frozen training buffers.
                saved = {**checkpoint.parameters, **checkpoint.buffers}
                state = run.graph.state_dict()
                for name, value in saved.items():
                    state["_components." + name].copy_(value)
                before = _snapshot(run.graph)
                batches = [
                    take(heldout, torch.arange(b * 128, (b + 1) * 128))
                    for b in range(EVAL_BATCHES)
                ]
                assert all(
                    batch.batch_size == BARLOW_TWINS_BATCH_SIZE for batch in batches
                )
                metrics.update(
                    _embedding_metrics(
                        run.graph,
                        schema,
                        _held_out_views(schema, batches, population, base=base),
                        projector=projector,
                    )
                )
                metrics["encoder_parameter_norm"] = _parameter_norm(
                    dict(checkpoint.parameters), "mlp_encoder."
                )
                metrics["projector_parameter_norm"] = _parameter_norm(
                    dict(checkpoint.parameters), projector + "."
                )
                assert all(
                    torch.equal(value, run.graph.state_dict()[name])
                    for name, value in before.items()
                )
            else:
                assert all(
                    torch.equal(value, fit_start[name]) for name, value in start.items()
                )
        assert all(math.isfinite(value) for value in metrics.values())
        report[arm] = metrics
    # `lambda` reaches the optimiser: the two arms share every stream and every
    # initial tensor, so a pretrained encoder that is bit-identical would mean
    # the zeroed weight changed nothing about the fit.
    assert any(
        not torch.equal(value, pretrained["diagonal_only"][name])
        for name, value in pretrained["full"].items()
    )
    print(json.dumps({"base": base, "arms": report}, sort_keys=True))
