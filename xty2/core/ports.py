"""Ports: named statistical quantities, and their shape contracts.

Components exchange semantic quantities, not architectures (`DESIGN.md` §2).
Raw inputs are ports too (§2.2): the batch is never handed to a component, so
`requires` is the single dependency declaration and `Y_RAW` reachability — the
fact the leakage rule reasons about (§7.2) — is a property of the graph rather
than a flag somebody remembered to set.

There is deliberately no `T_RAW`. No v1 component consumes the observed
treatment: outcome heads receive candidate `t` through `log_prob(y, t)`, the
energy head emits one value per treatment, and propensity and posterior heads
take representations and `Y_RAW`. It would also be the one port that cannot be
typed honestly as a tensor, since `t` is only valid where `t_observed`.

Every port has a `PortSpec` fixing its type and shape contract, and
`PortSpec.check` is what makes the contract executable rather than tabular.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import torch
from torch import Tensor

from xty2.core.distributions import OutcomeDistribution, TreatmentDistribution
from xty2.core.errors import PortContractError
from xty2.core.schema import Schema


class Port(StrEnum):
    """The framework's vocabulary. Adding one is a design decision (§11)."""

    X_RAW = "x"
    Y_RAW = "y"
    X_REPR = "x_repr"
    Y_GIVEN_XT = "p(y|x,t)"
    T_GIVEN_X = "p(t|x)"
    T_GIVEN_XY = "q(t|x,y)"
    JOINT_ENERGY = "energy(x,t,y)"
    RECONSTRUCTION = "reconstruction"


class Axis(StrEnum):
    """A symbolic axis in a port's shape contract."""

    BATCH = "B"
    """Resolved from the batch."""
    FEATURES = "D"
    """Resolved from `Schema.num_features`."""
    TREATMENTS = "K"
    """Resolved from `Schema.treatment_cardinality`."""
    FREE = "H"
    """Any positive size — a representation width the schema does not fix."""
    OUTCOME = "*Dy"
    """Variadic; resolved from `OutcomeSpec.shape`. Only ever the last axis."""


PortKind = Literal["tensor", "treatment_distribution", "outcome_distribution"]

PortDtype = Literal["float", "outcome"]
"""`outcome` defers to `OutcomeSpec.kind`: float for continuous, long for a class."""


@dataclass(frozen=True)
class PortSpec:
    """The type and shape contract for one port.

    For distribution ports, `shape` describes the tensor the contract fixes —
    `probs` for a treatment distribution — and is `None` for
    `Y_GIVEN_XT`, whose contract is a set of call signatures rather than a
    shape and is checked by `xty2.core.conformance` instead.
    """

    port: Port
    kind: PortKind
    shape: tuple[Axis, ...] | None
    description: str
    dtype: PortDtype = "float"

    def check(self, value: object, *, batch_size: int, schema: Schema) -> None:
        """Raise `PortContractError` unless `value` satisfies this port.

        This is the cheap, per-value check the compiler and the executors can
        afford to run. It is not a substitute for the conformance functions:
        no shape check can establish that an outcome head broadcasts candidate
        treatments correctly.
        """
        if self.kind == "tensor":
            if not isinstance(value, Tensor):
                raise self._error(f"expected a Tensor, got {type(value)}")
            self._check_dtype(value, schema=schema)
            self._check_shape(value, batch_size=batch_size, schema=schema)
            return

        if self.kind == "treatment_distribution":
            if not isinstance(value, TreatmentDistribution):
                raise self._error(
                    f"expected a TreatmentDistribution (probs, log_prob), got "
                    f"{type(value)}"
                )
            probs = value.probs
            if not probs.is_floating_point():
                raise self._error(f"probs must be a float tensor, got {probs.dtype}")
            self._check_shape(probs, batch_size=batch_size, schema=schema, name="probs")
            totals = probs.sum(dim=-1)
            if not torch.allclose(totals, torch.ones_like(totals), atol=1e-5):
                raise self._error("probs rows must sum to 1")
            return

        if not isinstance(value, OutcomeDistribution):
            raise self._error(
                f"expected an OutcomeDistribution (log_prob, mean, sample), got "
                f"{type(value)}"
            )

    # -- internals ---------------------------------------------------------

    def _check_dtype(self, value: Tensor, *, schema: Schema) -> None:
        if self.dtype == "float" or schema.outcome.is_continuous:
            if not value.is_floating_point():
                raise self._error(f"expected a float tensor, got {value.dtype}")
        elif value.dtype != torch.long:
            raise self._error(
                f"expected a long tensor for a categorical outcome, got {value.dtype}"
            )

    def _check_shape(
        self, value: Tensor, *, batch_size: int, schema: Schema, name: str = ""
    ) -> None:
        if self.shape is None:  # pragma: no cover - only Y_GIVEN_XT has no shape
            raise self._error(
                "has no shape contract; check it with xty2.core.conformance"
            )
        expected = resolve_shape(self.shape, batch_size=batch_size, schema=schema)
        actual = tuple(value.shape)
        subject = f"{name} " if name else ""
        if len(actual) != len(expected):
            raise self._error(
                f"{subject}must have {len(expected)} axes {self._render()}, got "
                f"shape {actual}"
            )
        for axis, (want, got) in enumerate(zip(expected, actual, strict=True)):
            if want is None:  # a free axis: any positive size
                if got < 1:
                    raise self._error(
                        f"{subject}axis {axis} is free but must be positive, got {got}"
                    )
                continue
            if want != got:
                raise self._error(
                    f"{subject}axis {axis} must be {want} {self._render()}, got "
                    f"shape {actual}"
                )

    def _render(self) -> str:
        return "[" + ", ".join(str(axis) for axis in self.shape or ()) + "]"

    def _error(self, message: str) -> PortContractError:
        return PortContractError(f"port {self.port}: {message}")


def resolve_shape(
    shape: tuple[Axis, ...], *, batch_size: int, schema: Schema
) -> tuple[int | None, ...]:
    """Resolve a symbolic shape against a batch size and a schema.

    A `None` entry is a free axis, matching any positive size.
    """
    resolved: list[int | None] = []
    for axis in shape:
        if axis is Axis.BATCH:
            resolved.append(batch_size)
        elif axis is Axis.FEATURES:
            resolved.append(schema.num_features)
        elif axis is Axis.TREATMENTS:
            resolved.append(schema.treatment_cardinality)
        elif axis is Axis.FREE:
            resolved.append(None)
        else:
            resolved.extend(schema.outcome.shape)
    return tuple(resolved)


PORT_SPECS: dict[Port, PortSpec] = {
    Port.X_RAW: PortSpec(
        Port.X_RAW, "tensor", (Axis.BATCH, Axis.FEATURES), "raw features, as batch.x"
    ),
    Port.Y_RAW: PortSpec(
        Port.Y_RAW,
        "tensor",
        (Axis.BATCH, Axis.OUTCOME),
        "raw outcome, as batch.y",
        dtype="outcome",
    ),
    Port.X_REPR: PortSpec(
        Port.X_REPR, "tensor", (Axis.BATCH, Axis.FREE), "a learned representation of x"
    ),
    Port.T_GIVEN_X: PortSpec(
        Port.T_GIVEN_X,
        "treatment_distribution",
        (Axis.BATCH, Axis.TREATMENTS),
        "propensity / treatment assignment",
    ),
    Port.T_GIVEN_XY: PortSpec(
        Port.T_GIVEN_XY,
        "treatment_distribution",
        (Axis.BATCH, Axis.TREATMENTS),
        "posterior over the treatment, used to infer missing t",
    ),
    Port.Y_GIVEN_XT: PortSpec(
        Port.Y_GIVEN_XT,
        "outcome_distribution",
        None,
        "outcome model; satisfies the candidate-treatment contract (§3.1)",
    ),
    Port.JOINT_ENERGY: PortSpec(
        Port.JOINT_ENERGY,
        "tensor",
        (Axis.BATCH, Axis.TREATMENTS),
        "one energy per treatment value",
    ),
    Port.RECONSTRUCTION: PortSpec(
        Port.RECONSTRUCTION,
        "tensor",
        (Axis.BATCH, Axis.FEATURES),
        "reconstruction of x, matching X_RAW",
    ),
}


def port_spec(port: Port) -> PortSpec:
    """The `PortSpec` for `port`. Every port has one, by test."""
    try:
        return PORT_SPECS[port]
    except KeyError:  # pragma: no cover - unreachable while the test below holds
        raise PortContractError(f"port {port} has no PortSpec") from None
