"""Predictive metrics over the distribution protocols (`DESIGN.md` §3.1)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from xty2.core.distributions import OutcomeDistribution, TreatmentDistribution


def conditional_outcome_nll(
    distribution: OutcomeDistribution,
    y: Tensor,
    t: Tensor,
    *,
    log_abs_jacobian: float = 0.0,
) -> Tensor:
    """Mean ``-log p(y|x,t)`` on observed/candidate rows.

    ``log_abs_jacobian`` converts a density fitted in transformed outcome
    coordinates back to the reported coordinates. For
    ``z=(y-location)/scale`` it is ``log(scale)`` per scalar event, so
    ``log p_Y = log p_Z - log_abs_jacobian``.
    """
    if not math.isfinite(log_abs_jacobian):
        raise ValueError(
            "conditional outcome NLL needs a finite log_abs_jacobian, got "
            f"{log_abs_jacobian!r}"
        )
    log_prob = distribution.log_prob(y, t)
    expected = tuple(t.shape)
    if tuple(log_prob.shape) != expected:
        raise ValueError(
            "outcome log_prob must match the treatment selection shape; "
            f"expected {expected}, got {tuple(log_prob.shape)}"
        )
    _require_finite("outcome log probabilities", log_prob)
    return -(log_prob - log_abs_jacobian).mean()


def treatment_nll(distribution: TreatmentDistribution, t: Tensor) -> Tensor:
    """Mean categorical treatment negative log-likelihood."""
    log_prob = distribution.log_prob(t)
    if tuple(log_prob.shape) != tuple(t.shape):
        raise ValueError(
            "treatment log_prob must match the treatment selection shape; "
            f"expected {tuple(t.shape)}, got {tuple(log_prob.shape)}"
        )
    _require_finite("treatment log probabilities", log_prob)
    return -log_prob.mean()


def root_mean_squared_error(prediction: Tensor, target: Tensor) -> Tensor:
    """Root mean squared error over identically shaped finite tensors."""
    _matching_finite_tensors(prediction, target)
    return (prediction - target).square().mean().sqrt()


def _matching_finite_tensors(left: Tensor, right: Tensor) -> None:
    if left.shape != right.shape:
        raise ValueError(
            f"metric inputs must have the same shape, got {tuple(left.shape)} "
            f"and {tuple(right.shape)}"
        )
    if left.numel() == 0:
        raise ValueError("a metric over zero values is undefined")
    _require_finite("prediction", left)
    _require_finite("target", right)


def _require_finite(label: str, value: Tensor) -> None:
    if value.numel() == 0:
        raise ValueError(f"{label} is empty")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} must be finite")


__all__ = [
    "conditional_outcome_nll",
    "root_mean_squared_error",
    "treatment_nll",
]
