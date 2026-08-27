"""Schema-aware bounded perturbations (`DESIGN.md` §1.2, §5)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from xty2.core.batch import XTYBatch
from xty2.core.data import TrainingPopulation
from xty2.core.errors import ViewError
from xty2.core.schema import FeatureSpec, Schema


@dataclass(frozen=True)
class BoundedJitter:
    """Add independent Gaussian noise in each feature's natural units.

    The standard deviation for a selected column is its
    ``FeatureSpec.perturbation_scale`` times ``scale``.  Continuous values are
    clipped to inclusive bounds; ordinal values are rounded and then clipped.
    Categorical features cannot be jittered.  Explicitly selected immutable
    features are left untouched.
    """

    columns: tuple[str, ...]
    scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        if not self.columns:
            raise ViewError("BoundedJitter needs at least one column")
        if any(not _is_name(name) for name in self.columns):
            raise ViewError("BoundedJitter.columns must contain non-empty names")
        if len(set(self.columns)) != len(self.columns):
            raise ViewError("BoundedJitter.columns cannot contain duplicates")
        if isinstance(self.scale, bool) or not isinstance(self.scale, int | float):
            raise ViewError(
                f"BoundedJitter.scale must be a number, got {type(self.scale)}"
            )
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise ViewError(
                f"BoundedJitter.scale must be finite and positive, got {self.scale!r}"
            )

    def validate(self, schema: Schema) -> None:
        for spec in _selected(schema, self.columns):
            if not spec.mutable:
                continue
            if spec.kind == "categorical":
                raise ViewError(
                    f"BoundedJitter cannot perturb categorical column "
                    f"{spec.name!r}; use a categorical transform"
                )
            if spec.perturbation_scale is None:
                raise ViewError(
                    f"BoundedJitter column {spec.name!r} has no "
                    "FeatureSpec.perturbation_scale in natural units"
                )

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        self.validate(schema)
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
        specs = tuple(
            spec
            for spec in _selected(schema, self.columns)
            if spec.name in self.affected_columns(schema)
        )
        if not specs:
            return batch.replace(x=batch.x.clone())
        indices = torch.tensor(
            [schema.index_of(spec.name) for spec in specs],
            dtype=torch.long,
            device=batch.device,
        )
        selected = batch.x.index_select(1, indices)
        noise = torch.randn(
            selected.shape,
            dtype=selected.dtype,
            device=selected.device,
            generator=generator,
        )
        scales = torch.tensor(
            [_scale_for(spec) * float(self.scale) for spec in specs],
            dtype=selected.dtype,
            device=selected.device,
        )
        replacement = selected + noise * scales[None, :]
        for column, spec in enumerate(specs):
            values = replacement[:, column]
            if spec.kind == "ordinal":
                values = values.round()
            if spec.bounds is not None:
                low, high = spec.bounds
                values = values.clamp(min=low, max=high)
            replacement = replacement.index_copy(
                1,
                torch.tensor([column], dtype=torch.long, device=batch.device),
                values[:, None],
            )
        return batch.replace(x=batch.x.index_copy(1, indices, replacement))

    def describe(self) -> str:
        columns = "[" + ", ".join(self.columns) + "]"
        return f"BoundedJitter(columns={columns}, scale={float(self.scale)!r})"


def _selected(schema: Schema, columns: tuple[str, ...]) -> tuple[FeatureSpec, ...]:
    unknown = sorted(set(columns) - set(schema.feature_names))
    if unknown:
        raise ViewError(
            f"BoundedJitter names unknown column(s) {unknown!r}; have "
            f"{schema.feature_names!r}"
        )
    wanted = set(columns)
    return tuple(spec for spec in schema.features if spec.name in wanted)


def _scale_for(spec: FeatureSpec) -> float:
    scale = spec.perturbation_scale
    if scale is None:  # validate() reports this with the view and column named.
        raise ViewError(f"BoundedJitter column {spec.name!r} has no perturbation scale")
    return float(scale)


def _is_name(value: object) -> bool:
    return isinstance(value, str) and bool(value)


__all__ = ["BoundedJitter"]
