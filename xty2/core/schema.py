"""Feature and outcome metadata, resolved once (`DESIGN.md` §1.2).

The schema is what lets augmentation be validated statically instead of
producing impossible rows at step 4,000: it carries per-column kind, bounds,
perturbation scale, mutability and derivation. It also validates *itself* —
`derived_from` must be complete (every named column exists) and acyclic — so a
view that recomputes a derived column has a graph to walk (P6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from xty2.core.errors import SchemaError

if TYPE_CHECKING:  # pragma: no cover - import cycle only exists for typing
    from xty2.core.batch import XTYBatch

FeatureKind = Literal["continuous", "categorical", "ordinal"]
OutcomeKind = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class FeatureSpec:
    """Metadata for one column of `XTYBatch.x`.

    Attributes:
        name: Column name; unique within a schema.
        kind: How the column behaves under perturbation.
        bounds: Inclusive `(low, high)` limits an augmentation must respect.
        perturbation_scale: Jitter scale in *natural units*, not sigmas.
        mutable: Whether an augmentation may touch this column at all.
        derived_from: Columns this one is computed from. Perturbing any of
            them stales this column, which is what the derived-column rule
            (`DESIGN.md` §1.2) exists to reject.
    """

    name: str
    kind: FeatureKind
    bounds: tuple[float, float] | None = None
    perturbation_scale: float | None = None
    mutable: bool = True
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise SchemaError("FeatureSpec.name must be a non-empty string")
        object.__setattr__(self, "derived_from", tuple(self.derived_from))
        if self.name in self.derived_from:
            raise SchemaError(f"feature {self.name!r} is derived from itself")
        if self.bounds is not None:
            low, high = self.bounds
            if not low < high:
                raise SchemaError(
                    f"feature {self.name!r} has bounds {self.bounds!r}; "
                    "expected (low, high) with low < high"
                )
        if self.perturbation_scale is not None:
            if self.perturbation_scale <= 0.0:
                raise SchemaError(
                    f"feature {self.name!r} has perturbation_scale "
                    f"{self.perturbation_scale!r}; expected a positive value"
                )
            if self.kind == "categorical":
                raise SchemaError(
                    f"feature {self.name!r} is categorical and cannot carry a "
                    "perturbation_scale — jitter on a category is not a value"
                )


@dataclass(frozen=True)
class OutcomeSpec:
    """Metadata for `XTYBatch.y`.

    `shape` is the trailing shape `Dy`: `()` means `y` is `[B]`, `(2,)` means
    `[B, 2]`. It is the same `Dy` the candidate-treatment contract quotes
    (`DESIGN.md` §3.1), so outcome heads and the port checks agree on one
    definition.
    """

    name: str = "y"
    kind: OutcomeKind = "continuous"
    shape: tuple[int, ...] = ()
    num_classes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        if any(dim < 1 for dim in self.shape):
            raise SchemaError(f"outcome shape {self.shape!r} has a non-positive axis")
        if self.kind == "categorical":
            if self.shape != ():
                raise SchemaError(
                    "a categorical outcome is a single class index per row; "
                    f"shape must be (), got {self.shape!r}"
                )
            if self.num_classes is None or self.num_classes < 2:
                raise SchemaError(
                    "a categorical outcome needs num_classes >= 2, got "
                    f"{self.num_classes!r}"
                )
        elif self.num_classes is not None:
            raise SchemaError("num_classes is meaningless for a continuous outcome")

    @property
    def is_continuous(self) -> bool:
        return self.kind == "continuous"


@dataclass(frozen=True)
class Schema:
    """The feature list, the treatment cardinality `K`, and the outcome spec."""

    features: tuple[FeatureSpec, ...]
    treatment_cardinality: int
    outcome: OutcomeSpec = field(default_factory=OutcomeSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", tuple(self.features))
        if not self.features:
            raise SchemaError("a schema needs at least one feature")
        if self.treatment_cardinality < 2:
            raise SchemaError(
                "treatment_cardinality is the number of treatment values K and "
                f"must be at least 2, got {self.treatment_cardinality}"
            )
        self._validate_feature_graph()

    # -- the feature graph -------------------------------------------------

    def _validate_feature_graph(self) -> None:
        """Reject duplicate names, dangling `derived_from`, and cycles."""
        seen: set[str] = set()
        for spec in self.features:
            if spec.name in seen:
                raise SchemaError(f"duplicate feature name {spec.name!r}")
            seen.add(spec.name)

        for spec in self.features:
            unknown = [parent for parent in spec.derived_from if parent not in seen]
            if unknown:
                raise SchemaError(
                    f"feature {spec.name!r} is derived from unknown column(s) "
                    f"{unknown!r}; derived_from must name features of this schema"
                )

        # Iterative DFS with an explicit colour map: a cycle is reported with
        # the path that closes it, because "cycle detected" is not actionable.
        parents = {spec.name: spec.derived_from for spec in self.features}
        state: dict[str, int] = dict.fromkeys(parents, 0)  # 0 unseen, 1 open, 2 done
        for root in parents:
            if state[root] != 0:
                continue
            path: list[str] = []
            stack: list[tuple[str, bool]] = [(root, False)]
            while stack:
                node, closing = stack.pop()
                if closing:
                    state[node] = 2
                    path.pop()
                    continue
                if state[node] == 1:
                    cycle = [*path[path.index(node) :], node]
                    raise SchemaError(
                        "derived_from must be acyclic; found cycle "
                        + " -> ".join(cycle)
                    )
                if state[node] == 2:
                    continue
                state[node] = 1
                path.append(node)
                stack.append((node, True))
                for parent in parents[node]:
                    stack.append((parent, False))

    # -- lookups -----------------------------------------------------------

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.features)

    @property
    def num_features(self) -> int:
        """`D`, the width of `XTYBatch.x`."""
        return len(self.features)

    def feature(self, name: str) -> FeatureSpec:
        for spec in self.features:
            if spec.name == name:
                return spec
        raise SchemaError(f"unknown feature {name!r}; have {self.feature_names!r}")

    def index_of(self, name: str) -> int:
        """Column index of `name` in `XTYBatch.x`."""
        for index, spec in enumerate(self.features):
            if spec.name == name:
                return index
        raise SchemaError(f"unknown feature {name!r}; have {self.feature_names!r}")

    # -- data agreement ----------------------------------------------------

    def validate_batch(self, batch: XTYBatch) -> None:
        """Check a batch against this schema.

        `XTYBatch` validates what it can know on its own (`DESIGN.md` §1.1);
        everything that needs `D`, `K` or `Dy` is checked here.
        """
        if batch.x.shape[1] != self.num_features:
            raise SchemaError(
                f"batch.x has {batch.x.shape[1]} columns but the schema declares "
                f"{self.num_features}"
            )
        if batch.batch_size and int(batch.t.max()) >= self.treatment_cardinality:
            raise SchemaError(
                f"batch.t holds value {int(batch.t.max())} but the schema declares "
                f"K = {self.treatment_cardinality}; t must be in [0, K) on *every* "
                "row, observed or not — missingness is carried by t_observed"
            )
        if tuple(batch.y.shape[1:]) != self.outcome.shape:
            raise SchemaError(
                f"batch.y has trailing shape {tuple(batch.y.shape[1:])!r} but the "
                f"schema declares Dy = {self.outcome.shape!r}"
            )
        if self.outcome.is_continuous:
            if not batch.y.is_floating_point():
                raise SchemaError(
                    f"a continuous outcome needs a float y, got {batch.y.dtype}"
                )
        else:
            if batch.y.dtype != torch.long:
                raise SchemaError(
                    f"a categorical outcome needs a long y, got {batch.y.dtype}"
                )
            num_classes = self.outcome.num_classes
            assert num_classes is not None  # guaranteed by OutcomeSpec validation
            if batch.batch_size and (
                int(batch.y.min()) < 0 or int(batch.y.max()) >= num_classes
            ):
                raise SchemaError(
                    f"batch.y holds a class index outside [0, {num_classes})"
                )
