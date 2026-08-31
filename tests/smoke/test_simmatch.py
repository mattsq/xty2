"""Tier 1 — SimMatch wiring on the card's two-cluster treatment DGP.

`docs/recipes/simmatch.md` §6.2 predeclares eight Tier 1 arms. This module runs
the first: the full recipe against the no-propagation arm the §6 pair is
defined by, on one seed of the §6.1 fixture, holding everything else — seeds,
initial parameters, batches, optimiser, schedule, views, bank, instance loss —
fixed. It also scores the two propagated targets against the fixture's hidden
treatments, which the card asks for "whether they improve or not".

The remaining arms (the two one-directional ablations, `lambda_in = 0`, the
zero-momentum bank, permuted memory labels, and the skewed-propensity
distribution-alignment diagnostic) are the predeclared study and are not run
here; the card's status says so.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    CosineDecay,
    Dataset,
    FeatureSpec,
    GaussianOutcome,
    OutcomeSpec,
    Port,
    Program,
    Recipe,
    Schema,
    XTYBatch,
    compile,
)
from xty2.core.rows import resolve_rows
from xty2.objectives import (
    LabeledMemoryInstanceConsistency,
    LabeledSimilarityMemory,
    SimilarityMatchingTreatmentNLL,
)
from xty2.recipes import simmatch
from xty2.recipes.simmatch import SIMILARITY_MATCHING
from xty2.training import StageResult, run_stage

FEATURES = 6
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048
STEPS = 600
CLUSTER_SIGNAL = 0.45
SEPARATED = 0.02
OBSERVED = 64
MEMORY_TERM = "similarity_matching_treatment_nll"
INSTANCE_TERM = "labeled_memory_instance_consistency"
WARMUP = SIMILARITY_MATCHING.warmup_steps


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _schema() -> Schema:
    return Schema(
        features=tuple(
            FeatureSpec(f"x{column}", "continuous") for column in range(FEATURES)
        ),
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


def _rows(rows: int, *, seed: int, row_offset: int) -> XTYBatch:
    """The card's §6.1 mechanism, fully observed; the recipe hides treatments."""
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < 0.5).float()
    sign = 2.0 * cluster - 1.0
    x = epsilon_x.clone()
    x[:, :4] = CLUSTER_SIGNAL * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = SEPARATED + (1.0 - 2.0 * SEPARATED) * cluster
    t = (u_t < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    y = baseline + t * (1.0 + 0.5 * torch.tanh(x[:, 2])) + 0.5 * epsilon_y
    return XTYBatch(
        x=x,
        t=t,
        y=y,
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(row_offset, row_offset + rows),
    )


def _dataset(train: XTYBatch) -> Dataset:
    return Dataset(
        schema=_schema(),
        rows=train,
        assignments={"train": torch.arange(train.batch_size)},
    )


def _with_steps(recipe: Recipe, steps: int) -> Recipe:
    """A shorter budget with the cosine schedule re-based on it (deviation 3)."""
    stage = recipe.program[0]
    schedule = stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    optimiser = replace(stage.optimiser, lr_schedule=replace(schedule, steps=steps))
    return replace(
        recipe,
        program=Program((replace(stage, steps=steps, optimiser=optimiser),)),
    )


def _without_propagation(recipe: Recipe) -> Recipe:
    """The §6 ablation: `alpha = 1` and no unfolding. Nothing else moves."""
    stage = recipe.program[0]
    semantic = stage.objectives[2].objective
    instance = stage.objectives[3].objective
    assert isinstance(semantic, SimilarityMatchingTreatmentNLL)
    assert isinstance(instance, LabeledMemoryInstanceConsistency)
    spec = replace(SIMILARITY_MATCHING, alpha=1.0, unfold=False)
    objectives = (
        *stage.objectives[:2],
        replace(stage.objectives[2], objective=replace(semantic, spec=spec)),
        replace(stage.objectives[3], objective=replace(instance, spec=spec)),
        *stage.objectives[4:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _term(result: StageResult, step: int, name: str) -> float:
    record = next(term for term in result.records[step].terms if term.name == name)
    return float(record.value)


def _diagnostic(result: StageResult, step: int, name: str, key: str) -> float:
    record = next(term for term in result.records[step].terms if term.name == name)
    return float(record.diagnostics[key])


def _mean(result: StageResult, name: str, steps: range) -> float:
    return sum(_term(result, step, name) for step in steps) / len(steps)


@dataclass(frozen=True)
class _Targets:
    """The two propagated targets, scored against the hidden treatments."""

    propagated_nll: float
    aligned_nll: float
    aggregated_nll: float
    coverage: float


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    treatment_nll: float
    ema_treatment_nll: float
    frequency_nll: float
    outcome_nll: float
    targets: _Targets


def _target_quality(run: CompiledRun, result: StageResult) -> _Targets:
    """Score `hat p`, `DA(p^w)` and class-aggregated `hat q` on hidden `t`.

    The run's own bank is stage-local state and is gone by the time a stage
    returns (deviation 7), so this rebuilds one from the same population and
    the *trained* student: fill every slot, then read the targets the next step
    would have used. That measures what the card claims — whether propagation
    improves the target given the representation the run learned — rather than
    replaying a bank the framework does not hand back.
    """
    population = result.population
    assert population is not None
    stage = run.recipe.program[0]
    objective = stage.objectives[2].objective
    assert isinstance(objective, SimilarityMatchingTreatmentNLL)
    memory = objective.initial_state(population)
    assert isinstance(memory, LabeledSimilarityMemory)
    batch = population.rows
    rows = resolve_rows(batch, "t_missing")
    with torch.no_grad():
        values = run.graph.evaluate(
            batch, schema=run.recipe.schema, only=run.graph.names
        )
        propensity = values[Port.T_GIVEN_X]
        embedding = values[Port.X_PROJ]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(embedding, torch.Tensor)
        # Fill every slot, then step past the warm-up so the read is the one a
        # training step would have made.
        for step in range(WARMUP + 1):
            targets = memory.prepare(
                step=step,
                raw_probabilities=propensity.probs,
                weak_embeddings=embedding,
                batch=batch,
                eligible_rows=rows,
                support_rows=resolve_rows(batch, "t_observed"),
            )
        assert targets.propagated and targets.instance is not None
        hidden = batch.t.index_select(0, rows)
        aggregated = torch.zeros_like(targets.aligned).index_add(
            1, memory.labels, targets.instance
        )
        return _Targets(
            propagated_nll=float(F.nll_loss(targets.semantic.log(), hidden)),
            aligned_nll=float(F.nll_loss(targets.aligned.log(), hidden)),
            aggregated_nll=float(F.nll_loss(aggregated.clamp_min(1e-12).log(), hidden)),
            coverage=targets.coverage,
        )


def _evaluate(run: CompiledRun, result: StageResult, test: XTYBatch) -> _Metrics:
    schema = run.recipe.schema
    population = result.population
    assert population is not None and result.teacher is not None
    scaled = test.replace(
        y=(test.y - population.statistics["y_location"])
        / population.statistics["y_scale"]
    )
    with torch.no_grad():
        student = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        teacher = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        propensity = student[Port.T_GIVEN_X]
        ema_propensity = teacher[Port.T_GIVEN_X]
        outcome = teacher[Port.Y_GIVEN_XT]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(ema_propensity, CategoricalTreatment)
        assert isinstance(outcome, GaussianOutcome)
        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(observed, minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        return _Metrics(
            run=run,
            result=result,
            treatment_nll=float(F.nll_loss(propensity.log_probs, scaled.t)),
            ema_treatment_nll=float(F.nll_loss(ema_propensity.log_probs, scaled.t)),
            frequency_nll=float(F.nll_loss(baseline, scaled.t)),
            outcome_nll=float(-outcome.log_prob(scaled.y, scaled.t).mean()),
            targets=_target_quality(run, result),
        )


def _run(recipe: Recipe, train: XTYBatch, test: XTYBatch) -> _Metrics:
    run = compile(_with_steps(recipe, STEPS))
    result = run_stage(run, "joint_fit", _dataset(train), seed=100_000)
    return _evaluate(run, result, test)


@pytest.fixture(scope="module")
def paired_fit() -> tuple[_Metrics, _Metrics]:
    """The §6 pair: full SimMatch and its no-propagation arm, one seed."""
    schema = _schema()
    train = _rows(TRAIN_ROWS, seed=90_001, row_offset=0)
    test = _rows(TEST_ROWS, seed=90_002, row_offset=10_000)

    torch.manual_seed(90_006)
    full = simmatch(schema)
    torch.manual_seed(90_006)
    ablated = _without_propagation(simmatch(schema))
    for name, value in full.system.state_dict().items():
        assert torch.equal(value, ablated.system.state_dict()[name])

    return _run(full, train, test), _run(ablated, train, test)


def test_the_label_budget_and_the_bank_are_the_declared_ones(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """`K = 64` is a fact of the population, so it is asserted here (card §4)."""
    full, _ = paired_fit
    population = full.result.population
    assert population is not None
    assert int(population.rows.t_observed.sum()) == OBSERVED
    observed = population.rows.t[population.rows.t_observed]
    assert int(torch.bincount(observed, minlength=2).min()) > 0
    assert full.targets.coverage == 1.0


def test_the_instance_loss_is_silent_until_the_bank_is_filled(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 invariant 6, in a real run rather than on a hand-built bank."""
    for metrics in paired_fit:
        result = metrics.result
        for step in range(WARMUP):
            assert _term(result, step, INSTANCE_TERM) == 0.0
            assert _diagnostic(result, step, INSTANCE_TERM, "propagated") == 0.0
        assert _diagnostic(result, WARMUP, MEMORY_TERM, "bank_coverage") == 1.0
        assert _diagnostic(result, WARMUP, INSTANCE_TERM, "propagated") == 1.0
        assert _term(result, WARMUP, INSTANCE_TERM) > 0.0


def test_the_instance_loss_and_the_supervised_term_fall_after_the_warm_up(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1, first clause.

    Eq. (5) is ungated, so its level is readable directly; it starts below the
    `log K` a uniform slot distribution would charge and comes down from there.
    """
    full, _ = paired_fit
    early = range(WARMUP, WARMUP + 100)
    late = range(STEPS - 100, STEPS)
    uniform = float(torch.tensor(float(OBSERVED)).log())
    for name in (INSTANCE_TERM, "observed_treatment_nll"):
        assert _mean(full.result, name, late) < _mean(full.result, name, early), name
    assert _mean(full.result, INSTANCE_TERM, late) < uniform
    # `hat q` sharpens as the geometry forms; a flat target would make the
    # falling loss a statement about scale rather than about matching.
    entropy_early = sum(
        _diagnostic(full.result, step, INSTANCE_TERM, "target_entropy")
        for step in early
    ) / len(early)
    entropy_late = sum(
        _diagnostic(full.result, step, INSTANCE_TERM, "target_entropy") for step in late
    ) / len(late)
    assert entropy_late < entropy_early < uniform


def test_the_gate_and_not_the_level_is_how_equation_two_is_read(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1, second clause.

    The gated term *rises* while its gate opens, which is the reading
    `fixmatch.md` §6 records for that paper's eq. (4). What must be true is that
    the rise is the curriculum: more rows accepted, and accepted at a
    confidence at or above the declared threshold.
    """
    full, _ = paired_fit
    early = range(WARMUP, WARMUP + 100)
    late = range(STEPS - 100, STEPS)
    rate_early = sum(
        _diagnostic(full.result, step, MEMORY_TERM, "coverage") for step in early
    ) / len(early)
    rate_late = sum(
        _diagnostic(full.result, step, MEMORY_TERM, "coverage") for step in late
    ) / len(late)
    assert rate_early < 0.5 < rate_late
    accepted = sum(
        _diagnostic(full.result, step, MEMORY_TERM, "accepted_confidence")
        for step in late
    ) / len(late)
    assert accepted >= SIMILARITY_MATCHING.threshold


def test_the_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1: the fit is worth comparing at all."""
    full, _ = paired_fit
    assert full.treatment_nll < full.frequency_nll
    assert full.ema_treatment_nll < full.frequency_nll


def test_propagation_moves_the_run_it_is_supposed_to_move(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The pair is controlled, so any difference is equations (8) and (10)."""
    full, ablated = paired_fit
    assert full.result.trace[:WARMUP] == ablated.result.trace[:WARMUP]
    assert full.result.trace != ablated.result.trace
    assert full.targets.propagated_nll != full.targets.aligned_nll
    # The ablation is the card's `hat p = p^w`, exactly.
    assert ablated.targets.propagated_nll == ablated.targets.aligned_nll


def test_the_propagated_targets_are_scored_whether_they_improve_or_not(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1's third clause. Tier 2 owns the direction.

    One seed at a fifth of the declared budget cannot support a claim about
    the mechanism, so this asserts only that both targets are live, finite and
    better than an uninformative one; the ten-seed §6 contract is where a
    direction is allowed to be read.
    """
    full, _ = paired_fit
    uninformative = float(torch.tensor(2.0).log())
    for measured in (
        full.targets.propagated_nll,
        full.targets.aligned_nll,
        full.targets.aggregated_nll,
    ):
        assert 0.0 < measured < uninformative


def test_the_outcome_head_is_not_damaged_by_the_two_extra_terms(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    full, ablated = paired_fit
    assert full.outcome_nll < 1.05 * ablated.outcome_nll
