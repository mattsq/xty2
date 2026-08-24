"""Tier 1 — Mean Teacher wiring on a clustered treatment DGP."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Constant,
    FeatureSpec,
    GaussianOutcome,
    OutcomeSpec,
    Port,
    Program,
    Recipe,
    Schema,
    SigmoidRamp,
    XTYBatch,
    compile,
)
from xty2.recipes import mean_teacher
from xty2.training import StageResult, run_stage

FEATURES = 6
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048
BATCH_SIZE = 256
STEPS = 3_000
OBSERVED_TREATMENTS = 205


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    """Small dense layers are faster and deterministic without thread fan-out."""
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


@dataclass(frozen=True)
class _Population:
    batch: XTYBatch
    true_effect: torch.Tensor


def _population(
    rows: int, *, seed: int, observed_treatments: int, row_offset: int
) -> _Population:
    """The card's redundant-cluster mechanism at Tier-1 scale."""
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < 0.5).float()
    sign = 2.0 * cluster - 1.0
    x = epsilon_x.clone()
    x[:, :4] = 0.8 * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = 0.15 + 0.70 * cluster
    t = (u_t < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    true_effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    y = baseline + t * true_effect + 0.5 * epsilon_y

    observed = torch.zeros(rows, dtype=torch.bool)
    if observed_treatments:
        missingness = torch.Generator().manual_seed(seed + 10_000)
        selected = torch.randperm(rows, generator=missingness)[:observed_treatments]
        observed[selected] = True
    return _Population(
        batch=XTYBatch(
            x=x,
            t=t,
            y=y,
            t_observed=observed,
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        true_effect=true_effect,
    )


def _take(batch: XTYBatch, rows: torch.Tensor) -> XTYBatch:
    return XTYBatch(
        x=batch.x.index_select(0, rows),
        t=batch.t.index_select(0, rows),
        y=batch.y.index_select(0, rows),
        t_observed=batch.t_observed.index_select(0, rows),
        y_observed=batch.y_observed.index_select(0, rows),
        row_id=batch.row_id.index_select(0, rows),
    )


@dataclass(frozen=True)
class _BatchStream:
    population: XTYBatch
    indices: torch.Tensor

    def __iter__(self) -> Iterator[XTYBatch]:
        for rows in self.indices:
            yield _take(self.population, rows)


def _batch_indices(rows: int, *, steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randperm(rows, generator=generator)[:BATCH_SIZE] for _ in range(steps)]
    )


def _zero_consistency(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    consistency = replace(stage.objectives[-1], weight=Constant(0.0))
    return replace(
        recipe,
        program=Program(
            (replace(stage, objectives=(*stage.objectives[:-1], consistency)),)
        ),
    )


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    treatment_nll: float
    frequency_nll: float
    ate: float
    sqrt_pehe: float
    disagreement: float


def _teacher_predictions(
    run: CompiledRun, result: StageResult, batch: XTYBatch
) -> tuple[CategoricalTreatment, GaussianOutcome]:
    assert result.teacher is not None
    with torch.no_grad():
        values = result.teacher.graph.evaluate(
            batch,
            schema=run.recipe.schema,
            only=run.graph.names,
        )
    propensity = values[Port.T_GIVEN_X]
    outcome = values[Port.Y_GIVEN_XT]
    assert isinstance(propensity, CategoricalTreatment)
    assert isinstance(outcome, GaussianOutcome)
    return propensity, outcome


def _disagreement(run: CompiledRun, result: StageResult, test: XTYBatch) -> float:
    assert result.teacher is not None
    values: list[torch.Tensor] = []
    only = ("mlp_encoder", "categorical_propensity")
    with torch.no_grad():
        for draw in range(16):
            key = 20_000 + draw
            student_batch = run.recipe.view("student_x").apply(
                test, run.recipe.schema, rng_key=key
            )
            teacher_batch = run.recipe.view("teacher_x").apply(
                test, run.recipe.schema, rng_key=key
            )
            student = run.graph.evaluate(
                student_batch, schema=run.recipe.schema, only=only
            )[Port.T_GIVEN_X]
            teacher = result.teacher.graph.evaluate(
                teacher_batch, schema=run.recipe.schema, only=only
            )[Port.T_GIVEN_X]
            assert isinstance(student, CategoricalTreatment)
            assert isinstance(teacher, CategoricalTreatment)
            values.append((student.probs - teacher.probs).square().mean())
    return float(torch.stack(values).mean())


def _evaluate(
    run: CompiledRun,
    result: StageResult,
    test: _Population,
    train: XTYBatch,
    outcome_scale: float,
) -> _Metrics:
    propensity, outcome = _teacher_predictions(run, result, test.batch)
    with torch.no_grad():
        treatment_nll = float(F.nll_loss(propensity.log_probs, test.batch.t))
        frequencies = torch.bincount(train.t[train.t_observed], minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(test.batch.batch_size, -1)
        frequency_nll = float(F.nll_loss(baseline, test.batch.t))
        candidates = torch.arange(2).expand(test.batch.batch_size, 2)
        means = outcome.mean(candidates)
        effect = (means[:, 1] - means[:, 0]) * outcome_scale
        ate = float(effect.mean())
        sqrt_pehe = float((effect - test.true_effect).square().mean().sqrt())
    return _Metrics(
        run=run,
        result=result,
        treatment_nll=treatment_nll,
        frequency_nll=frequency_nll,
        ate=ate,
        sqrt_pehe=sqrt_pehe,
        disagreement=_disagreement(run, result, test.batch),
    )


@pytest.fixture(scope="module")
def paired_fit() -> tuple[_Metrics, _Metrics]:
    schema = _schema()
    train = _population(
        TRAIN_ROWS,
        seed=90_001,
        observed_treatments=OBSERVED_TREATMENTS,
        row_offset=0,
    )
    test = _population(
        TEST_ROWS,
        seed=90_003,
        observed_treatments=TEST_ROWS,
        row_offset=10_000,
    )
    outcome_mean = train.batch.y.mean()
    outcome_scale = float(train.batch.y.std(unbiased=False))
    train_batch = train.batch.replace(y=(train.batch.y - outcome_mean) / outcome_scale)
    test_batch = test.batch.replace(y=(test.batch.y - outcome_mean) / outcome_scale)
    test = replace(test, batch=test_batch)
    indices = _batch_indices(TRAIN_ROWS, steps=STEPS, seed=90_005)
    batches = _BatchStream(train_batch, indices)

    torch.manual_seed(90_006)
    scheduled_recipe = mean_teacher(schema)
    torch.manual_seed(90_006)
    zero_recipe = _zero_consistency(mean_teacher(schema))
    for name, value in scheduled_recipe.system.state_dict().items():
        assert torch.equal(value, zero_recipe.system.state_dict()[name])

    scheduled_run = compile(scheduled_recipe)
    zero_run = compile(zero_recipe)
    scheduled_result = run_stage(scheduled_run, "joint_fit", batches, seed=90_010)
    zero_result = run_stage(zero_run, "joint_fit", batches, seed=90_010)
    return (
        _evaluate(scheduled_run, scheduled_result, test, train_batch, outcome_scale),
        _evaluate(zero_run, zero_result, test, train_batch, outcome_scale),
    )


def test_total_loss_decreases(paired_fit: tuple[_Metrics, _Metrics]) -> None:
    scheduled, _ = paired_fit
    # Before step 1,000 the marginal objective is still acquiring weight, so
    # its mixed total is not comparable with the final objective. Compare the
    # first fully-weighted window with the tail instead.
    early = sum(scheduled.result.trace[1_000:1_100]) / 100
    tail = sum(scheduled.result.trace[-100:]) / 100
    assert tail < 0.99 * early


def test_teacher_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, _ = paired_fit
    assert scheduled.treatment_nll < scheduled.frequency_nll


def test_teacher_treatment_contrasts_stay_in_the_wide_analytic_band(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, _ = paired_fit
    assert scheduled.ate == pytest.approx(1.0, abs=0.75)
    assert scheduled.sqrt_pehe < 1.0


def test_scheduled_consistency_reduces_held_out_perturbation_disagreement(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, zero = paired_fit
    assert scheduled.disagreement < zero.disagreement


def test_the_final_teacher_remains_gradient_free(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, _ = paired_fit
    teacher = scheduled.result.teacher
    assert teacher is not None
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def _short_trace(schedule: SigmoidRamp) -> tuple[CompiledRun, StageResult]:
    recipe = mean_teacher(_schema())
    stage = recipe.program[0]
    weighted = replace(stage.objectives[-1], weight=schedule)
    recipe = replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(*stage.objectives[:-1], weighted),
                    steps=41,
                ),
            )
        ),
    )
    population = _population(
        512,
        seed=91_001,
        observed_treatments=128,
        row_offset=30_000,
    ).batch
    batches = _BatchStream(population, _batch_indices(512, steps=41, seed=91_005))
    run = compile(recipe)
    return run, run_stage(run, "joint_fit", batches, seed=91_010)


def _assert_reviewed_ramp_trace(result: StageResult) -> None:
    expected = {
        0: 2.0 * math.exp(-5.0),
        20: 2.0 * math.exp(-1.25),
        40: 2.0,
    }
    for step, weight in expected.items():
        consistency = next(
            term for term in result.records[step].terms if term.name == "consistency"
        )
        assert consistency.weight == pytest.approx(weight)


def test_the_40_step_ramp_passes_and_a_400_step_mutant_fails() -> None:
    run, result = _short_trace(SigmoidRamp(end=2.0, steps=40))
    assert run.stage("joint_fit").objectives[-1].weight.describe() == (
        "sigmoid ramp to 2.0 over 40 steps: 2.0 * exp(-5 * (1 - min(step/40, 1))^2)"
    )
    _assert_reviewed_ramp_trace(result)

    _, mutant = _short_trace(SigmoidRamp(end=2.0, steps=400))
    with pytest.raises(AssertionError):
        _assert_reviewed_ramp_trace(mutant)
