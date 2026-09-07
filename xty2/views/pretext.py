"""Self-supervised transforms whose labels are deterministic from row order."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from xty2.core.batch import XTYBatch
from xty2.core.data import TrainingPopulation
from xty2.core.errors import ViewError
from xty2.core.schema import Schema


@dataclass(frozen=True)
class ColumnRoll:
    """Roll eligible rows' feature coordinates by four row-block labels."""

    shifts: tuple[int, ...] = (0, 1, 2, 3)
    rows: str = "t_missing"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shifts", tuple(self.shifts))
        if self.shifts != (0, 1, 2, 3):
            raise ViewError("ColumnRoll currently implements the reviewed four shifts")
        if self.rows != "t_missing":
            raise ViewError("ColumnRoll currently applies to t_missing rows")

    def validate(self, schema: Schema) -> None:
        if schema.num_features < 4:
            raise ViewError("ColumnRoll needs at least four feature columns")
        signatures = {(feature.kind, feature.bounds) for feature in schema.features}
        if len(signatures) != 1:
            raise ViewError(
                "ColumnRoll requires exchangeable feature kinds and bounds; rolling "
                "values across incompatible schema columns is invalid"
            )

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        self.validate(schema)
        return frozenset(schema.feature_names)

    @staticmethod
    def labels(count: int, *, device: torch.device) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.arange(count, device=device) * 4 // count

    def apply(
        self,
        batch: XTYBatch,
        schema: Schema,
        *,
        generator: torch.Generator,
        population: TrainingPopulation | None = None,
    ) -> XTYBatch:
        del generator, population
        self.validate(schema)
        rows = torch.nonzero(batch.t_missing, as_tuple=False).flatten()
        labels = self.labels(rows.numel(), device=batch.device)
        selected = batch.x.index_select(0, rows)
        rolled = (
            torch.stack(
                [
                    torch.roll(row, shifts=int(label.item()), dims=0)
                    for row, label in zip(selected, labels, strict=True)
                ]
            )
            if rows.numel()
            else selected.clone()
        )
        return batch.replace(x=batch.x.index_copy(0, rows, rolled))

    def describe(self) -> str:
        return "ColumnRoll(shifts=[0, 1, 2, 3], rows=t_missing, blocks=quarters)"


__all__ = ["ColumnRoll"]
