"""Tier 1 — TARNet end-to-end on an analytic two-treatment DGP."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
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
from xty2.recipes import tarnet
from xty2.training import StageResult, run_stage

FEATURES = 6
TRAIN_ROWS = 192
TEST_ROWS = 1_024
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
    """A confounded nonlinear outcome with analytic constant treatment effect 1."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, FEATURES, generator=generator)
    assignment_score = 1.5 * x[:, 0] - x[:, 1] + 0.6 * x[:, 2]
    t = (assignment_score + 0.3 * torch.randn(rows, generator=generator) > 0).long()
    baseline = (
        0.8 * x[:, 0]
        + 0.5 * x[:, 1].square()
        - 0.6 * x[:, 2] * x[:, 3]
        + 0.3 * torch.sin(2.0 * x[:, 4])
    )
    true_effect = torch.ones(rows)
    y = baseline + t * true_effect + 0.1 * torch.randn(rows, generator=generator)

    observed = torch.ones(rows, dtype=torch.bool)
    missing = round(rows * missing_fraction)
    if missing:
        # Ranking independent uniforms gives an exactly-50% MCAR mask, avoiding
        # a test whose stated missingness drifts with a binomial draw.
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


def _dataset(population: XTYBatch) -> Dataset:
    """The train rows, under the assignment name the recipe's policy declares.

    Tier 1 used to build the batch stream itself — one seeded permutation per
    step, first `tarnet.BATCH_SIZE` rows. That is now `UniformSampler`'s definition
    and the recipe owns it, so the fixture hands over rows and an assignment
    and stops deciding what a step sees.
    """
    return Dataset(
        schema=_schema(),
        rows=population,
        assignments={"fit": torch.arange(population.batch_size)},
    )


def _complete_case_ablation(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    assert stage.objectives[-1].name == "missing_treatment_marginal_nll"
    return replace(
        recipe,
        program=Program((replace(stage, objectives=stage.objectives[:-1]),)),
    )


@dataclass(frozen=True)
class _Metrics:
    result: StageResult
    propensity_log_loss: float
    frequency_log_loss: float
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
        assert isinstance(outcome, GaussianOutcome)
        propensity_log_loss = float(F.nll_loss(propensity.log_probs, test.t))
        frequencies = torch.bincount(train.t[train.t_observed], minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(test.batch_size, -1)
        frequency_log_loss = float(F.nll_loss(baseline, test.t))
        candidates = torch.arange(2).expand(test.batch_size, 2)
        means = outcome.mean(candidates)
        effect = means[:, 1] - means[:, 0]
        ate = float(effect.mean())
        sqrt_pehe = float((effect - true_effect).square().mean().sqrt())
    return _Metrics(
        result=result,
        propensity_log_loss=propensity_log_loss,
        frequency_log_loss=frequency_log_loss,
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
    batches = _dataset(train)

    torch.manual_seed(17)
    marginal_recipe = tarnet(schema)
    torch.manual_seed(17)
    complete_case_recipe = _complete_case_ablation(tarnet(schema))
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
    assert marginal_result.trace[0] == complete_case_result.trace[0]
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
    assert marginal.sqrt_pehe < 0.8 * complete_case.sqrt_pehe
