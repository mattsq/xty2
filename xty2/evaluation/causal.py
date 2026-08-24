"""Causal point metrics computed from candidate-treatment outcome means."""

from __future__ import annotations

import torch
from torch import Tensor

from xty2.core.distributions import OutcomeDistribution
from xty2.evaluation.predictive import root_mean_squared_error


def candidate_treatment_means(
    distribution: OutcomeDistribution,
    *,
    batch_size: int,
    num_treatments: int,
) -> Tensor:
    """Return all candidate means with shape ``[B,K,*Dy]``."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if num_treatments < 2:
        raise ValueError(f"num_treatments must be at least two, got {num_treatments}")
    candidates = torch.arange(num_treatments, dtype=torch.long).expand(
        batch_size, num_treatments
    )
    means = distribution.mean(candidates)
    if means.shape[:2] != (batch_size, num_treatments):
        raise ValueError(
            "candidate means must begin [B,K]; expected "
            f"{(batch_size, num_treatments)}, got {tuple(means.shape)}"
        )
    if not bool(torch.isfinite(means).all()):
        raise ValueError("candidate means must be finite")
    return means


def treatment_contrast(
    means: Tensor,
    *,
    treated: int = 1,
    control: int = 0,
) -> Tensor:
    """Treatment-wise mean contrast from ``means: [B,K,*Dy]``."""
    if means.ndim < 2 or means.shape[0] < 1 or means.shape[1] < 2:
        raise ValueError(
            "candidate means must have shape [B,K,*Dy] with B > 0 and K >= 2, "
            f"got {tuple(means.shape)}"
        )
    treatments = int(means.shape[1])
    for label, value in (("treated", treated), ("control", control)):
        if type(value) is not int or not 0 <= value < treatments:
            raise ValueError(
                f"{label} must be an integer in [0,{treatments}), got {value!r}"
            )
    if treated == control:
        raise ValueError("treated and control must name different treatments")
    if not bool(torch.isfinite(means).all()):
        raise ValueError("candidate means must be finite")
    return means[:, treated] - means[:, control]


def average_treatment_effect(effect: Tensor) -> Tensor:
    """Mean estimated treatment contrast."""
    if effect.numel() == 0:
        raise ValueError("ATE over zero effects is undefined")
    if not bool(torch.isfinite(effect).all()):
        raise ValueError("estimated effects must be finite")
    return effect.mean()


def sqrt_pehe(estimated_effect: Tensor, true_effect: Tensor) -> Tensor:
    """Square-root PEHE for row-aligned treatment effects."""
    return root_mean_squared_error(estimated_effect, true_effect)


def absolute_ate_error(estimated_ate: Tensor | float, true_ate: float) -> float:
    """Absolute error of one ATE point estimate."""
    estimate = float(estimated_ate)
    value = torch.tensor([estimate, true_ate], dtype=torch.float64)
    if not bool(torch.isfinite(value).all()):
        raise ValueError(
            f"ATE estimate and truth must be finite, got {estimate!r}, {true_ate!r}"
        )
    return abs(estimate - true_ate)


__all__ = [
    "absolute_ate_error",
    "average_treatment_effect",
    "candidate_treatment_means",
    "sqrt_pehe",
    "treatment_contrast",
]
