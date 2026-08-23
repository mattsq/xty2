"""Small neural-network mechanics shared by the first recipe components."""

from __future__ import annotations

import math
from collections.abc import Iterable

from torch import nn

from xty2.core.errors import GraphError

CFRNET_INITIALISATION = "normal std=0.1/sqrt(fan_in), bias=0"
TORCH_LINEAR_INITIALISATION = "torch Linear default Kaiming-uniform"


def validate_widths(widths: object, *, owner: str) -> tuple[int, ...]:
    """Return a non-empty tuple of positive hidden widths."""
    if not isinstance(widths, tuple | list):
        raise GraphError(f"{owner}.widths must be a tuple of integers, got {widths!r}")
    resolved = tuple(widths)
    if not resolved or any(type(width) is not int or width < 1 for width in resolved):
        raise GraphError(
            f"{owner}.widths must be a non-empty tuple of positive integers, "
            f"got {resolved!r}"
        )
    return resolved


def validate_dimension(value: object, *, field: str, owner: str) -> int:
    """Return one positive tensor dimension."""
    if type(value) is not int or value < 1:
        raise GraphError(f"{owner}.{field} must be a positive integer, got {value!r}")
    return value


def validate_dropout(value: object, *, owner: str) -> float:
    """Return a finite dropout probability in `[0, 1)`."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise GraphError(f"{owner}.dropout must be a number, got {type(value)}")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise GraphError(
            f"{owner}.dropout must be finite and in [0, 1), got {probability!r}"
        )
    return probability


def elu_stack(
    input_dim: int, widths: Iterable[int], *, dropout: float
) -> tuple[nn.Sequential, int]:
    """An ELU MLP body and its output width."""
    layers: list[nn.Module] = []
    current = input_dim
    for width in widths:
        layers.extend((nn.Linear(current, width), nn.ELU()))
        if dropout:
            layers.append(nn.Dropout(dropout))
        current = width
    return nn.Sequential(*layers), current


def relu_stack(
    input_dim: int, widths: Iterable[int], *, dropout: float
) -> tuple[nn.Sequential, int]:
    """A ReLU MLP body using Torch's affine-layer defaults."""
    layers: list[nn.Module] = []
    current = input_dim
    for width in widths:
        layers.extend((nn.Linear(current, width), nn.ReLU()))
        if dropout:
            layers.append(nn.Dropout(dropout))
        current = width
    return nn.Sequential(*layers), current


def initialise_cfrnet(module: nn.Module) -> None:
    """Apply the pinned CFRNet affine-layer initialisation in place."""
    for child in module.modules():
        if isinstance(child, nn.Linear):
            nn.init.normal_(
                child.weight, mean=0.0, std=0.1 / math.sqrt(child.in_features)
            )
            if child.bias is not None:
                nn.init.zeros_(child.bias)


__all__ = ["CFRNET_INITIALISATION", "TORCH_LINEAR_INITIALISATION"]
