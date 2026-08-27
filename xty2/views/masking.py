"""Feature masking for tabular views (`DESIGN.md` §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from xty2.core.batch import XTYBatch
from xty2.core.data import TrainingPopulation
from xty2.core.errors import ViewError
from xty2.core.schema import FeatureSpec, Schema


@dataclass(frozen=True)
class FeatureMask:
    """Independently replace feature cells with a schema-valid mask value.

    ``columns=None`` means every mutable feature.  Immutable features are
    omitted even when explicitly listed; ``FeatureSpec.mutable=False`` is an
    absolute promise, not an instruction a transform may override.
    """

    p: float
    columns: tuple[str, ...] | None = None
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.columns is not None:
            object.__setattr__(self, "columns", tuple(self.columns))
        if isinstance(self.p, bool) or not isinstance(self.p, int | float):
            raise ViewError(f"FeatureMask.p must be a number, got {type(self.p)}")
        if not 0.0 <= float(self.p) <= 1.0:
            raise ViewError(f"FeatureMask.p must be in [0, 1], got {self.p!r}")
        if isinstance(self.value, bool) or not isinstance(self.value, int | float):
            raise ViewError(
                f"FeatureMask.value must be a finite number, got {type(self.value)}"
            )
        if not math.isfinite(float(self.value)):
            raise ViewError(f"FeatureMask.value must be finite, got {self.value!r}")
        if self.columns is not None:
            if not self.columns:
                raise ViewError(
                    "FeatureMask.columns cannot be empty; use p=0 for no mask"
                )
            if any(not _is_name(name) for name in self.columns):
                raise ViewError("FeatureMask.columns must contain non-empty names")
            if len(set(self.columns)) != len(self.columns):
                raise ViewError("FeatureMask.columns cannot contain duplicates")

    def validate(self, schema: Schema) -> None:
        _selected(schema, self.columns)

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        if float(self.p) == 0.0:
            return frozenset()
        return frozenset(
            spec.name for spec in _selected(schema, self.columns) if spec.mutable
        )

    def apply(
        self,
        batch: XTYBatch,
        schema: Schema,
        *,
        generator: torch.Generator,
        population: TrainingPopulation | None = None,
    ) -> XTYBatch:
        # Batch-local by construction: a mask fills with a constant and a
        # jitter is relative to the row's own value, so neither has anything
        # to ask the training population.
        del population
        self.validate(schema)
        names = self.affected_columns(schema)
        specs = tuple(spec for spec in schema.features if spec.name in names)
        if not specs:
            return batch.replace(x=batch.x.clone())
        indices = torch.tensor(
            [schema.index_of(spec.name) for spec in specs],
            dtype=torch.long,
            device=batch.device,
        )
        selected = batch.x.index_select(1, indices)
        mask = torch.rand(
            selected.shape,
            dtype=selected.dtype,
            device=selected.device,
            generator=generator,
        ) < float(self.p)
        fills = torch.tensor(
            [_fill_value(spec, float(self.value)) for spec in specs],
            dtype=selected.dtype,
            device=selected.device,
        )
        replacement = torch.where(mask, fills[None, :], selected)
        return batch.replace(x=batch.x.index_copy(1, indices, replacement))

    def describe(self) -> str:
        columns = "all" if self.columns is None else "[" + ", ".join(self.columns) + "]"
        return (
            f"FeatureMask(p={float(self.p)!r}, columns={columns}, "
            f"value={float(self.value)!r})"
        )


def _selected(
    schema: Schema, columns: tuple[str, ...] | None
) -> tuple[FeatureSpec, ...]:
    if columns is None:
        return schema.features
    unknown = sorted(set(columns) - set(schema.feature_names))
    if unknown:
        raise ViewError(
            f"FeatureMask names unknown column(s) {unknown!r}; have "
            f"{schema.feature_names!r}"
        )
    wanted = set(columns)
    return tuple(spec for spec in schema.features if spec.name in wanted)


def _fill_value(spec: FeatureSpec, value: float) -> float:
    resolved = round(value) if spec.kind in ("categorical", "ordinal") else value
    if spec.bounds is not None:
        low, high = spec.bounds
        resolved = min(max(resolved, low), high)
    return float(resolved)


def _is_name(value: object) -> bool:
    return isinstance(value, str) and bool(value)


__all__ = ["FeatureMask"]
