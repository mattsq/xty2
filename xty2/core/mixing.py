"""Declarative cross-realisation MixUp plans.

``ViewSpec`` transforms one batch under one random draw.  MixUp is different:
its partner is sampled from a pool assembled from several views and row
populations.  These small declarations keep that pool in the compiled recipe;
the executor materialises the matching features and records the exact
permutation and coefficient in ``MixingPlan`` for the target-side objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from xty2.core.errors import CompileError, LossError, Xty2Error, require_str
from xty2.core.graph import Realisation
from xty2.core.rows import Rows, validate_population


@dataclass(frozen=True)
class MixMember:
    """One ordered member of a mixing pool."""

    realisation: Realisation
    rows: Rows

    def __post_init__(self) -> None:
        if not isinstance(self.realisation, Realisation):
            raise CompileError("MixMember.realisation must be a Realisation")
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise CompileError(f"MixMember has invalid rows {self.rows!r}") from error


@dataclass(frozen=True)
class MixSpec:
    """A pooled MixUp declaration whose output draw corresponds to a member."""

    name: str
    members: tuple[MixMember, ...]
    alpha: float
    rule: str

    def __post_init__(self) -> None:
        name = require_str("MixSpec.name", self.name, error=CompileError)
        if not name.isidentifier():
            raise CompileError(
                f"MixSpec.name must be a Python identifier, got {name!r}"
            )
        object.__setattr__(self, "members", tuple(self.members))
        if not self.members or any(
            not isinstance(member, MixMember) for member in self.members
        ):
            raise CompileError(
                "MixSpec.members must be a non-empty tuple of MixMember values"
            )
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, int | float):
            raise CompileError("MixSpec.alpha must be a finite positive number")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0:
            raise CompileError("MixSpec.alpha must be a finite positive number")
        object.__setattr__(self, "alpha", float(self.alpha))
        if self.rule not in ("max", "identity"):
            raise CompileError("MixSpec.rule must be 'max' or 'identity'")

    def output(self, member: int) -> Realisation:
        if type(member) is not int or not 0 <= member < len(self.members):
            raise CompileError(f"MixSpec {self.name!r} has no member {member!r}")
        return Realisation(view=self.name, draw=member)

    def describe(self) -> tuple[str, ...]:
        members = ", ".join(
            f"{member.realisation} rows={member.rows}" for member in self.members
        )
        return (
            f"{self.name}: alpha={self.alpha:g}, rule={self.rule}",
            f"members (output draw order): {members}",
        )


@dataclass(frozen=True)
class MixingPlan:
    """The sampled partner and coefficient for one output member."""

    first_rows: Tensor
    partner_rows: Tensor
    partner_is_observed: Tensor
    coefficient: Tensor

    def __post_init__(self) -> None:
        size = self.first_rows.numel()
        if self.first_rows.ndim != 1 or self.first_rows.dtype != torch.long:
            raise LossError(
                "MixingPlan.first_rows must be a one-dimensional long tensor"
            )
        if self.partner_rows.dtype != torch.long:
            raise LossError("MixingPlan.partner_rows must be a long tensor")
        if self.partner_is_observed.dtype != torch.bool:
            raise LossError("MixingPlan.partner_is_observed must be a bool tensor")
        if not self.coefficient.is_floating_point():
            raise LossError("MixingPlan.coefficient must be a floating-point tensor")
        devices = {
            self.first_rows.device,
            self.partner_rows.device,
            self.partner_is_observed.device,
            self.coefficient.device,
        }
        if len(devices) != 1:
            raise LossError("MixingPlan fields must be on one device")
        if any(
            value.ndim != 1 or value.numel() != size
            for value in (self.partner_rows, self.partner_is_observed, self.coefficient)
        ):
            raise LossError("MixingPlan fields must be aligned one-dimensional tensors")
        if size and not bool(
            ((self.coefficient >= 0.5) & (self.coefficient <= 1.0)).all()
        ):
            raise LossError("MixingPlan coefficients must lie in [0.5, 1.0]")
        if not bool(torch.isfinite(self.coefficient).all()):
            raise LossError("MixingPlan coefficients must be finite")


__all__ = ["MixMember", "MixSpec", "MixingPlan"]
