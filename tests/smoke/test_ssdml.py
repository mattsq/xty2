"""Tier 1 — SSDML cross-fitted labels and array ATE fit on its fixed DGP."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from xty2.core import (
    FeatureSpec,
    OutcomeSpec,
    Schema,
    XTYBatch,
    compile,
    resolve_rows,
)
from xty2.estimators import SSDMLATEAction
from xty2.recipes import ssdml
from xty2.training import ProgramResult, PseudoLabels, run_program

FEATURES = 6
ROWS = 800
PROPENSITY_STEPS = 500


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


def _population() -> XTYBatch:
    """The card's staged-imputation IRM DGP at Tier-1 scale."""
    generator = torch.Generator().manual_seed(120_000)
    x = torch.randn(ROWS, FEATURES, generator=generator)
    treatment_uniform = torch.rand(ROWS, generator=generator)
    outcome_noise = torch.randn(ROWS, generator=generator)
    missingness_uniform = torch.rand(ROWS, generator=generator)
    propensity = 0.05 + 0.90 * torch.sigmoid(
        1.25 * x[:, 0] - 0.75 * x[:, 1] + 0.5 * x[:, 2]
    )
    treatment = (treatment_uniform < propensity).long()
    baseline = x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2]
    effect = 1.0 + 0.25 * x[:, 3]
    outcome = baseline + treatment * effect + outcome_noise
    row_id = torch.arange(ROWS)
    return XTYBatch(
        x=x,
        t=treatment,
        y=outcome,
        t_observed=missingness_uniform < 0.50,
        y_observed=torch.ones(ROWS, dtype=torch.bool),
        row_id=row_id,
        fold_id=row_id % 5,
    )


@dataclass(frozen=True)
class _Metrics:
    result: ProgramResult
    labels: PseudoLabels
    source_unchanged: bool
    hidden_accuracy: float
    frequency_accuracy: float
    staged_state: Mapping[str, Tensor]
    oracle_state: Mapping[str, Tensor]
    repeated_state: Mapping[str, Tensor]


@pytest.fixture(scope="module")
def fit() -> _Metrics:
    batch = _population()
    before = batch.clone()
    torch.manual_seed(120_010)
    run = compile(ssdml(_schema()))
    result = run_program(
        run,
        {
            "propensity_labels": (batch,),
            "dml_ate": (batch,),
        },
        seed=120_010,
    )
    labels = result.stage("propensity_labels").pseudo_labels
    assert labels is not None
    hidden_truth = batch.t[batch.t_missing]
    hidden_accuracy = float((labels.labels == hidden_truth).float().mean())
    frequency = torch.bincount(batch.t[batch.t_observed], minlength=2).argmax()
    frequency_accuracy = float((hidden_truth == frequency).float().mean())
    staged_state = {
        name.removeprefix("ssdml_ate."): value
        for name, value in result.stage("dml_ate").checkpoint.parameters.items()
    }

    joined = labels.apply_to(batch, treatment_cardinality=2)
    action = run.stage("dml_ate").action
    assert isinstance(action, SSDMLATEAction)
    repeated_state = action.fit(joined, resolve_rows(joined, "all"), seed=999)
    oracle = batch.replace(t_observed=torch.ones(ROWS, dtype=torch.bool))
    oracle_state = action.fit(oracle, resolve_rows(oracle, "all"), seed=999)
    return _Metrics(
        result=result,
        labels=labels,
        source_unchanged=batch.equal_to(before),
        hidden_accuracy=hidden_accuracy,
        frequency_accuracy=frequency_accuracy,
        staged_state=staged_state,
        oracle_state=oracle_state,
        repeated_state=repeated_state,
    )


def test_propensity_fit_decreases_in_every_fold(fit: _Metrics) -> None:
    trace = fit.result.stage("propensity_labels").trace
    for fold in range(5):
        start = trace[fold * PROPENSITY_STEPS]
        tail = (
            sum(
                trace[
                    (fold + 1) * PROPENSITY_STEPS - 50 : (fold + 1) * PROPENSITY_STEPS
                ]
            )
            / 50
        )
        assert tail < start


def test_labels_are_out_of_fold_and_beat_the_frequency_classifier(
    fit: _Metrics,
) -> None:
    assert fit.labels.prediction_mode == "out_of_fold"
    assert fit.labels.used_y is False
    assert fit.hidden_accuracy > fit.frequency_accuracy + 0.05


def test_staged_and_oracle_ate_estimates_stay_in_coarse_tier1_bands(
    fit: _Metrics,
) -> None:
    assert float(fit.staged_state["ate"]) == pytest.approx(1.0, abs=0.4)
    assert float(fit.oracle_state["ate"]) == pytest.approx(1.0, abs=0.5)


def test_array_state_is_complete_finite_clipped_and_repeatable(fit: _Metrics) -> None:
    assert set(fit.staged_state) == {
        "ate",
        "diagnostic_standard_error",
        "influence_score",
        "g0_hat",
        "g1_hat",
        "m_hat",
        "row_id",
        "fold_id",
    }
    for name, value in fit.staged_state.items():
        assert torch.isfinite(value).all()
        assert torch.equal(value, fit.repeated_state[name])
    assert bool((fit.staged_state["m_hat"] >= 0.025).all())
    assert bool((fit.staged_state["m_hat"] <= 0.975).all())
    assert fit.source_unchanged
