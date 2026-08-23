"""Tier 1 — CNFlow end-to-end on an analytic two-treatment DGP."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.components import ConditionalFlowOutcome
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    FeatureSpec,
    OutcomeSpec,
    Port,
    Recipe,
    Schema,
    XTYBatch,
    compile,
)
from xty2.recipes import cnflow
from xty2.training import StageResult, run_stage

FEATURES = 6
TRAIN_ROWS = 128
TEST_ROWS = 512
BATCH_SIZE = 48
STEPS = 3_000


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    """Small dense layers are faster and more stable without thread fan-out."""
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


def _population(
    rows: int, *, seed: int, missing_fraction: float, row_offset: int
) -> tuple[XTYBatch, torch.Tensor]:
    """An easy propensity and shared baseline with analytic treatment effect 1."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, FEATURES, generator=generator)
    assignment_score = 3.0 * x[:, 0]
    t = (assignment_score + 0.15 * torch.randn(rows, generator=generator) > 0).long()
    baseline = 0.5 * x[:, 1] - 0.4 * x[:, 2] + 0.2 * x[:, 3] + 0.1 * torch.sin(x[:, 4])
    true_effect = torch.ones(rows)
    y = baseline + t * true_effect + 0.1 * torch.randn(rows, generator=generator)

    observed = torch.ones(rows, dtype=torch.bool)
    missing = round(rows * missing_fraction)
    if missing:
        scores = torch.rand(rows, generator=generator)
        observed[:] = False
        observed[scores.topk(rows - missing).indices] = True

    frequencies = torch.bincount(t[observed], minlength=2).float()
    frequencies /= frequencies.sum()
    weights = torch.ones(rows)
    observed_t = t[observed]
    weights[observed] = 1.0 / (2.0 * frequencies.index_select(0, observed_t))
    batch = XTYBatch(
        x=x,
        t=t,
        y=y,
        t_observed=observed,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(row_offset, row_offset + rows),
        weight=weights,
    )
    return batch, true_effect


def _take(batch: XTYBatch, rows: torch.Tensor) -> XTYBatch:
    assert batch.weight is not None
    return XTYBatch(
        x=batch.x.index_select(0, rows),
        t=batch.t.index_select(0, rows),
        y=batch.y.index_select(0, rows),
        t_observed=batch.t_observed.index_select(0, rows),
        y_observed=batch.y_observed.index_select(0, rows),
        row_id=batch.row_id.index_select(0, rows),
        weight=batch.weight.index_select(0, rows),
    )


def _batches(population: XTYBatch) -> tuple[XTYBatch, ...]:
    generator = torch.Generator().manual_seed(99)
    return tuple(
        _take(
            population,
            torch.randperm(population.batch_size, generator=generator)[:BATCH_SIZE],
        )
        for _ in range(STEPS)
    )


def _complete_case_ablation(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    assert stage.objectives[-1].name == "missing_treatment_marginal_nll"
    return replace(
        recipe,
        program=(replace(stage, objectives=stage.objectives[:-1]),),
    )


@dataclass(frozen=True)
class _Metrics:
    result: StageResult
    propensity_log_loss: float
    frequency_log_loss: float
    marginal_outcome_nll: float
    ate: float
    sqrt_pehe: float


def _evaluate(
    run: CompiledRun,
    result: StageResult,
    test: XTYBatch,
    true_effect: torch.Tensor,
    train: XTYBatch,
) -> _Metrics:
    with torch.no_grad():
        state = run.state("joint_fit", test)
        propensity = state.default[Port.T_GIVEN_X]
        outcome = state.default[Port.Y_GIVEN_XT]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(outcome, ConditionalFlowOutcome)
        propensity_log_loss = float(F.nll_loss(propensity.log_probs, test.t))
        frequencies = torch.bincount(train.t[train.t_observed], minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(test.batch_size, -1)
        frequency_log_loss = float(F.nll_loss(baseline, test.t))
        candidates = torch.arange(2).expand(test.batch_size, 2)
        marginal_outcome_nll = float(
            -torch.logsumexp(
                propensity.log_prob(candidates) + outcome.log_prob(test.y, candidates),
                dim=-1,
            ).mean()
        )
        means = outcome.mean(candidates)
        effect = means[:, 1] - means[:, 0]
        ate = float(effect.mean())
        sqrt_pehe = float((effect - true_effect).square().mean().sqrt())
    return _Metrics(
        result=result,
        propensity_log_loss=propensity_log_loss,
        frequency_log_loss=frequency_log_loss,
        marginal_outcome_nll=marginal_outcome_nll,
        ate=ate,
        sqrt_pehe=sqrt_pehe,
    )


@pytest.fixture(scope="module")
def paired_fit() -> tuple[_Metrics, _Metrics]:
    schema = _schema()
    train, _ = _population(TRAIN_ROWS, seed=1, missing_fraction=0.5, row_offset=0)
    test, true_effect = _population(
        TEST_ROWS, seed=2, missing_fraction=0.0, row_offset=2_000
    )
    assert int(train.t_observed.sum()) == TRAIN_ROWS // 2
    batches = _batches(train)

    torch.manual_seed(17)
    marginal_recipe = cnflow(schema)
    torch.manual_seed(17)
    complete_case_recipe = _complete_case_ablation(cnflow(schema))
    for name, value in marginal_recipe.system.state_dict().items():
        assert torch.equal(value, complete_case_recipe.system.state_dict()[name])
    assert (
        marginal_recipe.program[0].optimiser
        == complete_case_recipe.program[0].optimiser
    )
    assert marginal_recipe.program[0].steps == complete_case_recipe.program[0].steps

    marginal_run = compile(marginal_recipe)
    complete_case_run = compile(complete_case_recipe)
    marginal_result = run_stage(marginal_run, "joint_fit", batches, seed=23)
    complete_case_result = run_stage(complete_case_run, "joint_fit", batches, seed=23)
    return (
        _evaluate(marginal_run, marginal_result, test, true_effect, train),
        _evaluate(
            complete_case_run,
            complete_case_result,
            test,
            true_effect,
            train,
        ),
    )


def test_training_loss_decreases(paired_fit: tuple[_Metrics, _Metrics]) -> None:
    marginal, _ = paired_fit
    tail = sum(marginal.result.trace[-100:]) / 100
    assert tail < 0.75 * marginal.result.trace[0]


def test_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    marginal, _ = paired_fit
    assert marginal.propensity_log_loss < marginal.frequency_log_loss


def test_estimated_ate_is_in_the_declared_wide_band(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    marginal, _ = paired_fit
    assert marginal.ate == pytest.approx(1.0, abs=0.75)


def test_marginalisation_beats_the_paired_complete_case_ablation(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    marginal, complete_case = paired_fit
    assert marginal.marginal_outcome_nll < complete_case.marginal_outcome_nll - 0.05
