"""SCARF's random feature corruption (`DESIGN.md` §5).

Every other transform in this package writes a value the *transform* chose — a
constant fill, a jittered number clipped back into bounds. This one writes a
value the *column* already held somewhere else in the batch, which is what
makes it defensible on tabular physical data: bounds, kinds and any implicit
support constraint hold by construction rather than by a clamp, and there is no
setting of its one hyperparameter that can produce a row the schema declares
impossible.

Two details of the paper's procedure are easy to lose and are therefore stated
in the code as well as in `docs/recipes/scarf.md` §3.1:

* **`q` is a count, not a per-cell rate.** Exactly `floor(rate * M)` columns are
  corrupted in every row, drawn without replacement. A per-cell Bernoulli mask
  — what `FeatureMask` does — has the same mean and a different variance, and
  is a different augmentation.
* **The donor is drawn per `(row, column)`.** Using one donor row for all of a
  row's corrupted cells would preserve that row's cross-feature dependence,
  which is exactly what the corruption exists to destroy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from xty2.core.batch import XTYBatch
from xty2.core.data import TrainingPopulation
from xty2.core.errors import ViewError
from xty2.core.schema import FeatureSpec, Schema


@dataclass(frozen=True)
class FeatureCorruption:
    """Replace `floor(rate * M)` of each row's features with marginal draws.

    ``columns=None`` means every mutable feature.  Immutable features are
    omitted even when explicitly listed, and are not counted in ``M``:
    ``FeatureSpec.mutable=False`` is an absolute promise, not an instruction a
    transform may override.

    The marginal is the **batch's** empirical column distribution rather than
    the training set's, which is a deviation the consuming card records and the
    `view-population-statistics` ledger entry costs (`DESIGN.md` §11.4). A view
    is a pure function of `(batch, rng_key)`, so the batch is the whole
    population it can see.
    """

    rate: float
    columns: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.columns is not None:
            object.__setattr__(self, "columns", tuple(self.columns))
        if isinstance(self.rate, bool) or not isinstance(self.rate, int | float):
            raise ViewError(
                f"FeatureCorruption.rate must be a number, got {type(self.rate)}"
            )
        if not math.isfinite(float(self.rate)) or not 0.0 <= float(self.rate) <= 1.0:
            raise ViewError(
                f"FeatureCorruption.rate is a fraction of the features and must "
                f"be in [0, 1], got {self.rate!r}"
            )
        if self.columns is not None:
            if not self.columns:
                raise ViewError(
                    "FeatureCorruption.columns cannot be empty; use rate=0 for "
                    "no corruption"
                )
            if any(not _is_name(name) for name in self.columns):
                raise ViewError("FeatureCorruption.columns must hold non-empty names")
            if len(set(self.columns)) != len(self.columns):
                raise ViewError("FeatureCorruption.columns cannot contain duplicates")

    def validate(self, schema: Schema) -> None:
        _selected(schema, self.columns)

    def corrupted_per_row(self, schema: Schema) -> int:
        """`q = floor(rate * M)`, over the mutable columns this may touch."""
        return math.floor(float(self.rate) * len(self._mutable(schema)))

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        if self.corrupted_per_row(schema) == 0:
            return frozenset()
        return frozenset(spec.name for spec in self._mutable(schema))

    def apply(
        self,
        batch: XTYBatch,
        schema: Schema,
        *,
        generator: torch.Generator,
        population: TrainingPopulation | None = None,
    ) -> XTYBatch:
        """`x~`: `floor(cM)` cells per row, each redrawn from its column.

        The draw is from the column's empirical marginal **over the training
        population** — "the uniform distribution over the values that feature
        takes on across the training dataset" — not over the batch in hand. The
        two agree in expectation when batches are drawn uniformly, and differ
        in the tail: a value held by fewer than one row in `B` cannot be drawn
        into a batch it is not in.

        `population` is therefore required rather than optional here. A
        transform that quietly fell back to the batch would be the deviation
        `scarf.md` §5.2 used to carry, restored as a default nobody reads.
        """
        if population is None:
            raise ViewError(
                "FeatureCorruption draws each replacement from the training "
                "population's empirical marginal, and this stage supplied none. "
                "A stage declaring ExternalBatches has no training population; "
                "declare a sampler so the loader builds one."
            )
        self.validate(schema)
        names = self.affected_columns(schema)
        specs = tuple(spec for spec in self._mutable(schema) if spec.name in names)
        rows, width = batch.batch_size, len(specs)
        if not width or not rows:
            return batch.replace(x=batch.x.clone())
        indices = torch.tensor(
            [schema.index_of(spec.name) for spec in specs],
            dtype=torch.long,
            device=batch.device,
        )
        selected = batch.x.index_select(1, indices)
        # The donor pool is the training population's columns, which is what
        # makes this the paper's marginal rather than the batch's.
        donor_pool = population.rows.x.index_select(1, indices)

        # `q` columns per row, drawn without replacement: a random permutation
        # per row, keep the first `q`. Taking the smallest `q` of a uniform
        # draw is the same distribution as sampling the subset directly, and is
        # one kernel rather than a Python loop over rows.
        keys = torch.rand(
            (rows, width),
            dtype=selected.dtype,
            device=batch.device,
            generator=generator,
        )
        ranks = keys.argsort(dim=1).argsort(dim=1)
        corrupt = ranks < self.corrupted_per_row(schema)

        donors = torch.randint(
            donor_pool.shape[0],
            (rows, width),
            dtype=torch.long,
            device=batch.device,
            generator=generator,
        )
        replacement = torch.where(corrupt, donor_pool.gather(0, donors), selected)
        return batch.replace(x=batch.x.index_copy(1, indices, replacement))

    def describe(self) -> str:
        columns = "all" if self.columns is None else "[" + ", ".join(self.columns) + "]"
        return f"FeatureCorruption(rate={float(self.rate)!r}, columns={columns})"

    def _mutable(self, schema: Schema) -> tuple[FeatureSpec, ...]:
        return tuple(spec for spec in _selected(schema, self.columns) if spec.mutable)


def _selected(
    schema: Schema, columns: tuple[str, ...] | None
) -> tuple[FeatureSpec, ...]:
    if columns is None:
        return schema.features
    unknown = sorted(set(columns) - set(schema.feature_names))
    if unknown:
        raise ViewError(
            f"FeatureCorruption names unknown column(s) {unknown!r}; have "
            f"{schema.feature_names!r}"
        )
    wanted = set(columns)
    return tuple(spec for spec in schema.features if spec.name in wanted)


def _is_name(value: object) -> bool:
    return isinstance(value, str) and bool(value)


__all__ = ["FeatureCorruption"]
