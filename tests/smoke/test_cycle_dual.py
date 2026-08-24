"""Tier 1 — cycle-dual staged posterior/outcome fit on its analytic DGP."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from dataclasses import dataclass

import pytest
import torch
from xty2.core import (
    FeatureSpec,
    GaussianOutcome,
    OutcomeSpec,
    Port,
    Schema,
    XTYBatch,
    compile,
)
from xty2.recipes import cycle_dual
from xty2.training import ProgramResult, PseudoLabels, run_program

FEATURES = 4
TRAIN_ROWS = 512
TEST_ROWS = 1_024
POSTERIOR_STEPS = 500
OUTCOME_STEPS = 1_000


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


def _population(
    rows: int,
    *,
    seed: int,
    observed_probability: float,
    row_offset: int,
) -> XTYBatch:
    """The card's posterior-imputation DGP at Tier-1 scale."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, FEATURES, generator=generator)
    treatment_uniform = torch.rand(rows, generator=generator)
    outcome_noise = torch.randn(rows, generator=generator)
    missingness_uniform = torch.rand(rows, generator=generator)
    propensity = torch.sigmoid(0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2])
    treatment = (treatment_uniform < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.25 * x[:, 1] + 0.25 * (x[:, 2].square() - 1.0)
    effect = 1.5 + 0.5 * torch.tanh(x[:, 0])
    outcome = baseline + treatment * effect + 0.5 * outcome_noise
    row_id = torch.arange(row_offset, row_offset + rows)
    return XTYBatch(
        x=x,
        t=treatment,
        y=outcome,
        t_observed=missingness_uniform < observed_probability,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=row_id,
        fold_id=row_id % 5,
    )


@dataclass(frozen=True)
class _Metrics:
    result: ProgramResult
    labels: PseudoLabels
    source_unchanged: bool
    observed_preserved: bool
    hidden_accuracy: float
    standardised_ate: float
    original_scale_ate: float
    outcome_scale: float


@pytest.fixture(scope="module")
def fit() -> _Metrics:
    train = _population(
        TRAIN_ROWS,
        seed=110_000,
        observed_probability=0.30,
        row_offset=0,
    )
    test = _population(
        TEST_ROWS,
        seed=110_002,
        observed_probability=1.0,
        row_offset=10_000,
    )
    x_mean = train.x.mean(dim=0)
    x_scale = train.x.std(dim=0, unbiased=False)
    outcome_mean = train.y.mean()
    outcome_scale = train.y.std(unbiased=False)
    train = train.replace(
        x=(train.x - x_mean) / x_scale,
        y=(train.y - outcome_mean) / outcome_scale,
    )
    test = test.replace(
        x=(test.x - x_mean) / x_scale,
        y=(test.y - outcome_mean) / outcome_scale,
    )
    before = train.clone()

    torch.manual_seed(110_010)
    run = compile(cycle_dual(_schema()))
    result = run_program(
        run,
        {
            "posterior_labels": (train,),
            "outcome_fit": itertools.repeat(train, OUTCOME_STEPS),
        },
        seed=110_010,
    )
    labels = result.stage("posterior_labels").pseudo_labels
    assert labels is not None
    hidden_truth = train.t[train.t_missing]
    hidden_accuracy = float((labels.labels == hidden_truth).float().mean())
    joined = labels.apply_to(train, treatment_cardinality=2)

    with torch.no_grad():
        distribution = run.state("outcome_fit", test).default[Port.Y_GIVEN_XT]
        assert isinstance(distribution, GaussianOutcome)
        candidates = torch.arange(2).expand(test.batch_size, 2)
        standardised_means = distribution.mean(candidates)
        original_means = outcome_mean + outcome_scale * standardised_means
        standardised_ate = float(
            (standardised_means[:, 1] - standardised_means[:, 0]).mean()
        )
        original_scale_ate = float((original_means[:, 1] - original_means[:, 0]).mean())
    return _Metrics(
        result=result,
        labels=labels,
        source_unchanged=train.equal_to(before),
        observed_preserved=bool(
            torch.equal(joined.t[train.t_observed], train.t[train.t_observed])
        ),
        hidden_accuracy=hidden_accuracy,
        standardised_ate=standardised_ate,
        original_scale_ate=original_scale_ate,
        outcome_scale=float(outcome_scale),
    )


def test_both_stage_losses_decrease(fit: _Metrics) -> None:
    posterior_trace = fit.result.stage("posterior_labels").trace
    for fold in range(5):
        start = posterior_trace[fold * POSTERIOR_STEPS]
        tail = (
            sum(
                posterior_trace[
                    (fold + 1) * POSTERIOR_STEPS - 50 : (fold + 1) * POSTERIOR_STEPS
                ]
            )
            / 50
        )
        assert tail < start
    outcome_trace = fit.result.stage("outcome_fit").trace
    assert sum(outcome_trace[-100:]) / 100 < 0.8 * outcome_trace[0]


def test_posterior_labels_are_accurate_out_of_fold_and_outcome_dependent(
    fit: _Metrics,
) -> None:
    assert fit.labels.prediction_mode == "out_of_fold"
    assert fit.labels.used_y is True
    assert fit.hidden_accuracy >= 0.75


def test_the_functional_join_preserves_the_source_and_gold_treatments(
    fit: _Metrics,
) -> None:
    assert fit.source_unchanged
    assert fit.observed_preserved


def test_ate_is_scored_only_after_inverse_outcome_scaling(fit: _Metrics) -> None:
    assert fit.original_scale_ate == pytest.approx(
        fit.standardised_ate * fit.outcome_scale,
        abs=1e-6,
    )
    assert fit.original_scale_ate == pytest.approx(1.5, abs=0.5)
