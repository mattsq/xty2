"""Categorical treatment calibration and discrimination metrics."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def classification_accuracy(probs: Tensor, target: Tensor) -> Tensor:
    """Argmax accuracy for ``probs: [B,K]`` and ``target: [B]``."""
    _validate(probs, target)
    return (probs.argmax(dim=-1) == target).float().mean()


def multiclass_brier_score(probs: Tensor, target: Tensor) -> Tensor:
    """Mean row-wise multiclass Brier score."""
    _validate(probs, target)
    truth = F.one_hot(target, num_classes=probs.shape[1]).to(probs.dtype)
    return (probs - truth).square().sum(dim=-1).mean()


def expected_calibration_error(
    probs: Tensor,
    target: Tensor,
    *,
    bins: int = 10,
) -> Tensor:
    """Top-label ECE with equal-width confidence bins on ``[0,1]``."""
    _validate(probs, target)
    if type(bins) is not int or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins!r}")
    confidence, prediction = probs.max(dim=-1)
    correct = (prediction == target).to(probs.dtype)
    # floor(confidence*bins) assigns confidence==1 to `bins`; clamp it to the
    # final legal bin rather than silently dropping perfectly confident rows.
    assignment = torch.clamp((confidence * bins).long(), max=bins - 1)
    total = probs.new_zeros(())
    for index in range(bins):
        rows = assignment == index
        count = int(rows.sum())
        if count == 0:
            continue
        gap = (correct[rows].mean() - confidence[rows].mean()).abs()
        total = total + gap * (count / target.numel())
    return total


def _validate(probs: Tensor, target: Tensor) -> None:
    if probs.ndim != 2 or probs.shape[0] < 1 or probs.shape[1] < 2:
        raise ValueError(
            f"probabilities must have shape [B,K] with B > 0 and K >= 2, got "
            f"{tuple(probs.shape)}"
        )
    if target.dtype != torch.long or target.shape != (probs.shape[0],):
        raise ValueError(
            "targets must be a long tensor with shape [B]; expected "
            f"{(probs.shape[0],)}, got {target.dtype} {tuple(target.shape)}"
        )
    if not bool(torch.isfinite(probs).all()):
        raise ValueError("probabilities must be finite")
    if not bool(((probs >= 0.0) & (probs <= 1.0)).all()):
        raise ValueError("probabilities must lie in [0,1]")
    if not torch.allclose(
        probs.sum(dim=-1),
        torch.ones(probs.shape[0], dtype=probs.dtype, device=probs.device),
        atol=1e-6,
        rtol=1e-6,
    ):
        raise ValueError("probability rows must sum to one")
    if int(target.min()) < 0 or int(target.max()) >= probs.shape[1]:
        raise ValueError(
            f"targets must lie in [0,{probs.shape[1]}), got "
            f"{int(target.min())}..{int(target.max())}"
        )


__all__ = [
    "classification_accuracy",
    "expected_calibration_error",
    "multiclass_brier_score",
]
