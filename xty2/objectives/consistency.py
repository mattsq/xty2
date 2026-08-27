"""Agreement between one statistical port under two realisations (§4.2, §5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

import torch

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import LossError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows, treatment_distribution
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population

ConsistencyDivergence = Literal["kl", "mse"]
StopGrad = Literal["left", "right", "none"]

CONSISTENCY_DIVERGENCES: tuple[ConsistencyDivergence, ...] = get_args(
    ConsistencyDivergence
)
STOP_GRADIENTS: tuple[StopGrad, ...] = get_args(StopGrad)


@dataclass(frozen=True)
class ConsistencyLoss:
    """Match a treatment distribution across two views or parameter sets.

    ``kl`` is directed ``KL(left || right)``. ``mse`` is the mean squared
    difference between the two probability vectors. The detached side is a
    required, card-governed choice and ``detaches`` is derived from it, so the
    compiler and the arithmetic cannot disagree about the gradient path.
    """

    port: Port
    left: Realisation
    right: Realisation
    divergence: ConsistencyDivergence = REQUIRED
    stop_grad: StopGrad = REQUIRED
    rows: Rows = "all"
    name: str = "consistency"

    CARD_KEYS: ClassVar[dict[str, str]] = {"stop_grad": "gradients.detached_targets"}

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("consistency objective name", self.name, error=LossError):
            raise LossError("ConsistencyLoss.name must be non-empty")
        if not isinstance(self.port, Port):
            raise LossError(
                f"ConsistencyLoss.port must be a Port, got {type(self.port)}"
            )
        if port_spec(self.port).kind != "treatment_distribution":
            raise LossError(
                f"ConsistencyLoss currently compares treatment distributions, "
                f"but port {self.port!s} carries {port_spec(self.port).kind}. "
                "Adding outcome/tensor semantics waits for a recipe that needs "
                "them (DESIGN.md §11)."
            )
        left: object = self.left
        right: object = self.right
        if not isinstance(left, Realisation) or not isinstance(right, Realisation):
            raise LossError("ConsistencyLoss.left and right must be Realisations")
        if self.left == self.right:
            raise LossError(
                f"ConsistencyLoss compares {self.left} with itself; the term "
                "would be identically zero and every extra forward pass dead"
            )
        if self.divergence not in CONSISTENCY_DIVERGENCES:
            raise LossError(
                f"ConsistencyLoss.divergence must be one of "
                f"{list(CONSISTENCY_DIVERGENCES)!r}, got {self.divergence!r}"
            )
        if self.stop_grad not in STOP_GRADIENTS:
            raise LossError(
                f"ConsistencyLoss.stop_grad must be one of "
                f"{list(STOP_GRADIENTS)!r}, got {self.stop_grad!r}"
            )
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(f"ConsistencyLoss {self.name!r}: {error}") from error

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.left), (self.port, self.right)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        if self.stop_grad == "left":
            return frozenset({(self.port, self.left)})
        if self.stop_grad == "right":
            return frozenset({(self.port, self.right)})
        return frozenset()

    def plan_details(self) -> tuple[str, ...]:
        """Arithmetic choices that must change the plan and its digest."""
        return (f"divergence = {self.divergence!r}",)

    @property
    def batch_coupled(self) -> bool:
        """No: the divergence pairs one row's two realisations with each other."""
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        left = treatment_distribution(
            state, self.port, self.left, objective=self.name
        ).probs
        right = treatment_distribution(
            state, self.port, self.right, objective=self.name
        ).probs
        if self.stop_grad == "left":
            left = left.detach()
        elif self.stop_grad == "right":
            right = right.detach()

        if self.divergence == "mse":
            per_row = (left - right).square().mean(dim=-1)
        else:
            tiny = torch.finfo(left.dtype).tiny
            per_row = (
                left * (left.clamp_min(tiny).log() - right.clamp_min(tiny).log())
            ).sum(dim=-1)
        if per_row.shape[0] != batch.batch_size:
            raise LossError(
                f"ConsistencyLoss {self.name!r} got {per_row.shape[0]} rows "
                f"from its realisations for a batch of {batch.batch_size}"
            )
        return reduce_rows(per_row, rows)


__all__ = [
    "CONSISTENCY_DIVERGENCES",
    "STOP_GRADIENTS",
    "ConsistencyDivergence",
    "ConsistencyLoss",
    "StopGrad",
]
