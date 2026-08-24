"""Tier 0 — predictive, causal and calibration metric contracts."""

from __future__ import annotations

import math

import pytest
import torch
from xty2.core import CategoricalTreatment, GaussianOutcome
from xty2.evaluation import (
    absolute_ate_error,
    average_treatment_effect,
    candidate_treatment_means,
    classification_accuracy,
    conditional_outcome_nll,
    expected_calibration_error,
    multiclass_brier_score,
    root_mean_squared_error,
    sqrt_pehe,
    treatment_contrast,
    treatment_nll,
)


def test_predictive_metrics_use_the_distribution_contract() -> None:
    treatment = CategoricalTreatment(
        torch.log(torch.tensor([[0.8, 0.2], [0.25, 0.75], [0.6, 0.4]]))
    )
    observed = torch.tensor([0, 1, 1])
    assert treatment_nll(treatment, observed) == pytest.approx(
        -math.log(0.8 * 0.75 * 0.4) / 3
    )

    outcome = GaussianOutcome(
        loc=torch.tensor(
            [
                [[0.0], [1.0]],
                [[0.5], [1.5]],
                [[1.0], [2.0]],
            ]
        ),
        scale=torch.ones(3, 2, 1),
    )
    y = torch.tensor([[0.0], [1.5], [0.0]])
    direct = -outcome.log_prob(y, observed).mean() + math.log(2.0)
    assert conditional_outcome_nll(
        outcome,
        y,
        observed,
        log_abs_jacobian=math.log(2.0),
    ) == pytest.approx(float(direct))


def test_candidate_means_and_causal_metrics_keep_the_candidate_axis() -> None:
    loc = torch.tensor(
        [
            [[1.0], [2.0], [4.0]],
            [[2.0], [4.0], [8.0]],
        ]
    )
    outcome = GaussianOutcome(loc=loc, scale=torch.ones_like(loc))
    means = candidate_treatment_means(
        outcome,
        batch_size=2,
        num_treatments=3,
    )
    assert torch.equal(means, loc)
    effect = treatment_contrast(means, treated=2, control=0)
    assert torch.equal(effect, torch.tensor([[3.0], [6.0]]))
    assert average_treatment_effect(effect) == pytest.approx(4.5)
    truth = torch.tensor([[2.0], [7.0]])
    assert sqrt_pehe(effect, truth) == pytest.approx(1.0)
    assert absolute_ate_error(average_treatment_effect(effect), 4.0) == 0.5


def test_calibration_metrics_have_exact_small_examples() -> None:
    probs = torch.tensor([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]])
    target = torch.tensor([0, 1, 1, 1])
    assert classification_accuracy(probs, target) == pytest.approx(0.75)
    truth = torch.nn.functional.one_hot(target, num_classes=2).float()
    expected_brier = (probs - truth).square().sum(dim=-1).mean()
    assert multiclass_brier_score(probs, target) == pytest.approx(float(expected_brier))
    assert expected_calibration_error(probs, target, bins=4) == pytest.approx(0.15)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: root_mean_squared_error(torch.ones(2), torch.ones(3)),
            "same shape",
        ),
        (
            lambda: expected_calibration_error(
                torch.tensor([[0.8, 0.8]]), torch.tensor([0])
            ),
            "sum to one",
        ),
        (
            lambda: treatment_contrast(torch.ones(2, 2), treated=1, control=1),
            "different treatments",
        ),
    ],
)
def test_metric_contract_failures_are_actionable(call: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()  # type: ignore[operator]
