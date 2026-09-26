"""VIME's two pretext losses: mask estimation and feature reconstruction.

`docs/recipes/vime.md` §3.1 transcribes them. Both read the clean row and its
corrupted copy as two realisations of `X_RAW`, so the clean row reaches the
loss as a target and never as an encoder input. Neither detaches anything:
`X_RAW` has no parameters, and Eq. 4 minimises over the encoder and both
estimators jointly.

Three details are easy to lose and are stated here as well as in the card:

* **The mask label is the set of cells that changed**, `1[x != x~]`, as the
  reference `pretext_generator` returns it (`m_new = 1 * (x != x_tilde)`). The
  view's drawn mask is never exported. On a continuous column the two differ
  only when a donor happens to hold the row's own value (`vime.md` §7).
* **Eq. 6 averages over all `d` features**, masked or not. Unmasked cells are
  visible to the encoder, so part of the loss is an identity copy. That is the
  published objective. The masked-cells-only variant (`cells="masked"`) is the
  adaptive-task proposal's, and waits for its reviewed card.
* **Categorical and ordinal features are refused**, rather than scored with a
  squared error on class codes (`vime.md` deviation 5). The paper switches
  Eq. 6 to cross-entropy for them; that branch is not built yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import torch.nn.functional as F
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.errors import LossError, PortContractError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows
from xty2.core.ports import Port
from xty2.core.rows import RowIndex, Rows, validate_population
from xty2.core.schema import Schema

ReconstructionCells = Literal["all", "masked"]
"""Which cells enter the per-row mean of `FeatureReconstruction`."""

BUILT_CELLS: tuple[ReconstructionCells, ...] = ("all",)
"""Only VIME's Eq. 6 is built. `masked` arrives with its reviewed consumer."""


@dataclass(frozen=True)
class MaskEstimationBCE:
    """Eq. 5: `-(1/d) sum_j [m_j log s_m_j + (1 - m_j) log(1 - s_m_j)]`.

    The logits are read from `FEATURE_MASK_LOGITS` under `corrupted`, and the
    label `m` is `1[x != x~]` from `X_RAW` under `clean` and `corrupted`. The
    cross-entropy is computed from logits (`vime.md` §7).

    Attributes:
        clean: The realisation of the uncorrupted row `x`.
        corrupted: The realisation of `x~`. The encoder reads this one.
        rows: The population this term is entitled to.
        name: Keys the per-objective log.
    """

    clean: Realisation
    corrupted: Realisation
    rows: Rows = "all"
    name: str = "mask_estimation_bce"

    def __post_init__(self) -> None:
        _validate_pair(self, self.name, self.clean, self.corrupted, self.rows)

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.FEATURE_MASK_LOGITS, self.corrupted),
                (Port.X_RAW, self.corrupted),
                (Port.X_RAW, self.clean),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: the label is a function of `X_RAW`, which has no parameters."""
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        return False

    def plan_details(self) -> tuple[str, ...]:
        """The roles of the two `X_RAW` realisations, which `requires` cannot show."""
        return (
            f"logits = {Port.FEATURE_MASK_LOGITS} @ {self.corrupted}",
            f"label m = 1[x @ {self.clean} != x @ {self.corrupted}], per cell",
            "per row = binary cross-entropy from logits, mean over all d features",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        _require_continuous(ctx.schema, self.name)
        logits = _tensor(state, Port.FEATURE_MASK_LOGITS, self.corrupted, self.name)
        clean = _tensor(state, Port.X_RAW, self.clean, self.name)
        corrupted = _tensor(state, Port.X_RAW, self.corrupted, self.name)
        _require_rows(self.name, batch, logits, clean, corrupted)
        label = (clean != corrupted).to(logits.dtype)
        per_row = F.binary_cross_entropy_with_logits(
            logits, label, reduction="none"
        ).mean(dim=1)
        if rows.numel() == 0:
            return reduce_rows(per_row, rows)
        eligible = label.index_select(0, rows)
        return reduce_rows(
            per_row, rows, diagnostics={"mask_rate": float(eligible.mean())}
        )


@dataclass(frozen=True)
class FeatureReconstruction:
    """Eq. 6: `(1/d) sum_j (x_j - s_r_j(e(x~)))^2`, over the chosen cells.

    Attributes:
        clean: The realisation of the target row `x`.
        corrupted: The realisation `RECONSTRUCTION` is read under, and the one
            the encoder saw. It is also what decides which cells were masked.
        cells: `"all"` is VIME's Eq. 6. `"masked"` is reserved for the
            adaptive-task proposal's conditional feature prediction and is
            rejected until a reviewed card needs it (`vime.md` §5.1).
        rows: The population this term is entitled to.
        name: Keys the per-objective log.
    """

    clean: Realisation
    corrupted: Realisation
    cells: ReconstructionCells = "all"
    rows: Rows = "all"
    name: str = "feature_reconstruction"

    def __post_init__(self) -> None:
        _validate_pair(self, self.name, self.clean, self.corrupted, self.rows)
        if self.cells not in get_args(ReconstructionCells):
            raise LossError(
                f"FeatureReconstruction.cells must be one of "
                f"{list(get_args(ReconstructionCells))!r}, got {self.cells!r}"
            )
        if self.cells not in BUILT_CELLS:
            raise LossError(
                f"FeatureReconstruction(cells={self.cells!r}) is the "
                "adaptive-task proposal's loss, not VIME's, and is not built "
                "until a reviewed card needs it (docs/recipes/vime.md §5.1)."
            )

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.RECONSTRUCTION, self.corrupted),
                (Port.X_RAW, self.corrupted),
                (Port.X_RAW, self.clean),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: the target is `X_RAW`, which has no parameters."""
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        return False

    def plan_details(self) -> tuple[str, ...]:
        """The target's realisation and the cells, which `requires` cannot show."""
        return (
            f"prediction = {Port.RECONSTRUCTION} @ {self.corrupted}",
            f"target = x @ {self.clean}",
            f"cells = {self.cells}",
            "per row = squared error, mean over all d features",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        _require_continuous(ctx.schema, self.name)
        prediction = _tensor(state, Port.RECONSTRUCTION, self.corrupted, self.name)
        clean = _tensor(state, Port.X_RAW, self.clean, self.name)
        corrupted = _tensor(state, Port.X_RAW, self.corrupted, self.name)
        _require_rows(self.name, batch, prediction, clean, corrupted)
        squared = (clean - prediction).square()
        per_row = squared.mean(dim=1)
        if rows.numel() == 0:
            return reduce_rows(per_row, rows)
        # Which cells changed, for the log only: the loss is over every cell.
        changed = (clean != corrupted).index_select(0, rows)
        masked = squared.detach().index_select(0, rows)[changed]
        diagnostics = (
            {"masked_cell_mse": float(masked.mean())} if masked.numel() else {}
        )
        return reduce_rows(per_row, rows, diagnostics=diagnostics)


def _validate_pair(
    owner: object,
    name: str,
    clean: Realisation,
    corrupted: Realisation,
    rows: Rows,
) -> None:
    title = type(owner).__name__
    if not require_str(f"{title} name", name, error=LossError):
        raise LossError(f"{title}.name must be non-empty")
    left: object = clean
    right: object = corrupted
    if not isinstance(left, Realisation) or not isinstance(right, Realisation):
        raise LossError(f"{title}.clean and corrupted must be Realisations")
    if clean == corrupted:
        raise LossError(
            f"{title} reads {clean} as both the clean and the corrupted row. "
            "No cell could be masked, and the pretext task would be the "
            "identity (VIME Eq. 3)."
        )
    try:
        validate_population(rows)
    except Xty2Error as error:
        raise LossError(f"{title} {name!r}: {error}") from error


def _require_continuous(schema: Schema, name: str) -> None:
    discrete = [spec.name for spec in schema.features if spec.kind != "continuous"]
    if discrete:
        raise LossError(
            f"objective {name!r} scores every feature as continuous, and the "
            f"schema declares {discrete!r} categorical or ordinal. VIME switches "
            "Eq. 6 to cross-entropy for those, and that branch is not built "
            "(docs/recipes/vime.md deviation 5)."
        )


def _tensor(state: State, port: Port, realisation: Realisation, name: str) -> Tensor:
    value = state[realisation][port]
    if not isinstance(value, Tensor):
        raise PortContractError(
            f"objective {name!r} read port {str(port)!r} under {realisation} as a "
            f"tensor, but it carries {type(value)}. Its PortSpec is the contract "
            "(DESIGN.md §2)."
        )
    return value


def _require_rows(name: str, batch: XTYBatch, *values: Tensor) -> None:
    for value in values:
        if value.shape[0] != batch.batch_size:
            raise LossError(
                f"objective {name!r} got {value.shape[0]} rows for a batch of "
                f"{batch.batch_size}"
            )
        if value.ndim != 2 or value.shape != values[0].shape:
            raise LossError(
                f"objective {name!r} compares tensors of shapes "
                f"{[tuple(item.shape) for item in values]!r}; each must be [B, D]"
            )


__all__ = [
    "BUILT_CELLS",
    "FeatureReconstruction",
    "MaskEstimationBCE",
    "ReconstructionCells",
]
