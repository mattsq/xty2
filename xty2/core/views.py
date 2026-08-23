"""Declarative data views and their execution contract (`DESIGN.md` §5).

``ViewSpec`` belongs in ``core`` for the same reason ``Stage`` does: it is
compiler input.  Concrete augmentations live in :mod:`xty2.views`; the core
knows only the small ``ViewTransform`` protocol they satisfy.

A view is a pure function of ``(batch, rng_key)``.  The explicit key matters:
using the ambient torch RNG would make the result depend on which other views
happened to run first.  ``ViewSpec.apply`` folds the view name into that key,
runs every transform against one private generator, checks functional
behaviour, and enforces the declared preserved fields.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.errors import ViewError, require_str
from xty2.core.graph import IDENTITY_VIEW
from xty2.core.schema import Schema

PreservedField = Literal[
    "x",
    "t",
    "y",
    "t_observed",
    "y_observed",
    "row_id",
    "fold_id",
    "weight",
]

PRESERVED_FIELDS = frozenset(
    {
        "x",
        "t",
        "y",
        "t_observed",
        "y_observed",
        "row_id",
        "fold_id",
        "weight",
    }
)

FeatureValues = Mapping[str, Tensor]
RecomputeFunction = Callable[[FeatureValues], Tensor]


@runtime_checkable
class ViewTransform(Protocol):
    """A schema-aware, functional transform used by a ``ViewSpec``."""

    def validate(self, schema: Schema) -> None:
        """Reject settings that are not meaningful for ``schema``."""

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        """Columns this transform may change for some RNG draw."""

    def apply(
        self, batch: XTYBatch, schema: Schema, *, generator: torch.Generator
    ) -> XTYBatch:
        """Return a transformed batch without writing into ``batch``."""

    def describe(self) -> str:
        """A stable one-line description for the execution plan."""


@dataclass(frozen=True)
class RecomputeRule:
    """Rebuild one derived feature after its dependencies were perturbed.

    The function receives a read-only mapping from every feature name to its
    current ``[B]`` column and returns the replacement ``[B]`` column.  Rules
    are run in feature-graph order, so a rule may consume a derived parent that
    an earlier rule has already rebuilt.
    """

    feature: str
    function: RecomputeFunction
    name: str = ""

    def __post_init__(self) -> None:
        feature = require_str("recompute feature", self.feature, error=ViewError)
        if not feature:
            raise ViewError("a RecomputeRule needs a non-empty feature name")
        if not callable(self.function):
            raise ViewError(
                f"RecomputeRule for {feature!r} needs a callable, got "
                f"{type(self.function)}"
            )
        name = self.name or getattr(
            self.function, "__name__", type(self.function).__name__
        )
        if not require_str("recompute rule name", name, error=ViewError):
            raise ViewError(f"RecomputeRule for {feature!r} needs a stable name")
        object.__setattr__(self, "name", name)

    def apply(self, batch: XTYBatch, schema: Schema) -> XTYBatch:
        """Return ``batch`` with this rule's derived column replaced."""
        columns = MappingProxyType(
            {
                spec.name: batch.x[:, index].clone()
                for index, spec in enumerate(schema.features)
            }
        )
        value = self.function(columns)
        if not isinstance(value, Tensor):
            raise ViewError(
                f"recompute rule {self.name!r} for {self.feature!r} returned "
                f"{type(value)}, expected a Tensor"
            )
        expected = (batch.batch_size,)
        if tuple(value.shape) != expected:
            raise ViewError(
                f"recompute rule {self.name!r} for {self.feature!r} returned "
                f"shape {tuple(value.shape)}, expected {expected}"
            )
        if value.device != batch.device:
            raise ViewError(
                f"recompute rule {self.name!r} for {self.feature!r} returned "
                f"values on {value.device}, expected {batch.device}"
            )
        if value.dtype != batch.x.dtype:
            raise ViewError(
                f"recompute rule {self.name!r} for {self.feature!r} returned "
                f"{value.dtype}, expected {batch.x.dtype}"
            )
        spec = schema.feature(self.feature)
        if spec.bounds is not None and value.numel():
            low, high = spec.bounds
            if not bool(((value >= low) & (value <= high)).all()):
                raise ViewError(
                    f"recompute rule {self.name!r} produced {self.feature!r} "
                    f"outside its inclusive bounds {spec.bounds!r}"
                )
        index = torch.tensor(
            [schema.index_of(self.feature)], dtype=torch.long, device=batch.device
        )
        x = batch.x.index_copy(1, index, value[:, None])
        return batch.replace(x=x)

    def describe(self) -> str:
        return f"{self.feature} <- {self.name}"


@dataclass(frozen=True)
class ViewSpec:
    """A named sequence of transforms plus its preservation contract."""

    name: str
    transforms: tuple[ViewTransform, ...]
    preserves: frozenset[PreservedField]
    recompute_rules: tuple[RecomputeRule, ...] = ()

    def __post_init__(self) -> None:
        name = require_str("view name", self.name, error=ViewError)
        if not name.isidentifier():
            raise ViewError(
                f"view name {name!r} must be a Python identifier: it keys "
                "realisations and the execution plan"
            )
        if name == IDENTITY_VIEW:
            raise ViewError(
                f"{IDENTITY_VIEW!r} is the built-in untransformed view and "
                "cannot also name a ViewSpec"
            )
        transforms = tuple(self.transforms)
        if not transforms:
            raise ViewError(
                f"view {name!r} has no transforms; use the built-in "
                f"{IDENTITY_VIEW!r} view for an unchanged batch"
            )
        for transform in transforms:
            if not isinstance(transform, ViewTransform):
                raise ViewError(
                    f"view {name!r} holds {type(transform)}; transforms satisfy "
                    "ViewTransform (validate, affected_columns, apply, describe)"
                )
        preserves = frozenset(self.preserves)
        unknown = sorted(set(preserves) - PRESERVED_FIELDS)
        if unknown:
            raise ViewError(
                f"view {name!r} preserves unknown batch field(s) {unknown!r}; "
                f"expected fields from {sorted(PRESERVED_FIELDS)!r}"
            )
        rules = tuple(self.recompute_rules)
        if any(not isinstance(rule, RecomputeRule) for rule in rules):
            raise ViewError(f"view {name!r} holds a non-RecomputeRule entry")
        duplicates = _duplicates(tuple(rule.feature for rule in rules))
        if duplicates:
            raise ViewError(
                f"view {name!r} has more than one recompute rule for {duplicates!r}"
            )
        object.__setattr__(self, "transforms", transforms)
        object.__setattr__(self, "preserves", preserves)
        object.__setattr__(self, "recompute_rules", rules)

    def validate(self, schema: Schema) -> None:
        """Validate transforms and the derived-column rule against ``schema``."""
        affected: set[str] = set()
        known = set(schema.feature_names)
        for transform in self.transforms:
            transform.validate(schema)
            columns = set(transform.affected_columns(schema))
            unknown = sorted(columns - known)
            if unknown:
                raise ViewError(
                    f"transform {transform.describe()!r} in view {self.name!r} "
                    f"claims unknown column(s) {unknown!r}"
                )
            immutable = sorted(
                name for name in columns if not schema.feature(name).mutable
            )
            if immutable:
                raise ViewError(
                    f"transform {transform.describe()!r} in view {self.name!r} "
                    f"would touch immutable column(s) {immutable!r}; "
                    "FeatureSpec.mutable=False is absolute (DESIGN.md §5)"
                )
            affected.update(columns)

        rule_names = {rule.feature for rule in self.recompute_rules}
        unknown_rules = sorted(rule_names - known)
        if unknown_rules:
            raise ViewError(
                f"view {self.name!r} registers recompute rule(s) for unknown "
                f"column(s) {unknown_rules!r}; have {schema.feature_names!r}"
            )
        for rule in self.recompute_rules:
            spec = schema.feature(rule.feature)
            if not spec.derived_from:
                raise ViewError(
                    f"view {self.name!r} registers recompute rule "
                    f"{rule.name!r} for {rule.feature!r}, which is not derived"
                )
            if not spec.mutable:
                raise ViewError(
                    f"view {self.name!r} would recompute immutable derived "
                    f"column {rule.feature!r}"
                )

        stale = _stale_derived(schema, affected)
        missing = sorted(stale - rule_names)
        if missing:
            causes = sorted(affected)
            raise ViewError(
                f"view {self.name!r} may perturb {causes!r}, which makes "
                f"derived column(s) {missing!r} stale. Register a recompute "
                "rule for every affected derived column, or do not perturb "
                "its dependency (DESIGN.md §1.2)."
            )
        redundant = sorted(rule_names - stale)
        if redundant:
            raise ViewError(
                f"view {self.name!r} registers recompute rule(s) for "
                f"{redundant!r}, but none of their dependencies are perturbed"
            )

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        """Every column the final view may replace, including recomputes."""
        affected = {
            name
            for transform in self.transforms
            for name in transform.affected_columns(schema)
        }
        affected.update(rule.feature for rule in self.recompute_rules)
        return frozenset(affected)

    def apply(self, batch: XTYBatch, schema: Schema, *, rng_key: int) -> XTYBatch:
        """Compute the view once, deterministically, without mutating ``batch``."""
        if type(rng_key) is not int:
            raise ViewError(f"rng_key must be an int, got {type(rng_key)}")
        schema.validate_batch(batch)
        self.validate(schema)
        original = batch.clone()
        generator = torch.Generator(device=batch.device)
        generator.manual_seed(_view_seed(rng_key, self.name))

        # A transform receives a private copy. The post-call equality check
        # still rejects an in-place implementation, while the user's source
        # batch remains intact even on that failing path.
        current = batch.clone()
        for transform in self.transforms:
            before = current.clone()
            result = transform.apply(current, schema, generator=generator)
            if not current.equal_to(before):
                raise ViewError(
                    f"transform {transform.describe()!r} in view {self.name!r} "
                    "wrote into its input batch. Transforms are functional "
                    "(DESIGN.md §1.1, §5)."
                )
            if not isinstance(result, XTYBatch):
                raise ViewError(
                    f"transform {transform.describe()!r} in view {self.name!r} "
                    f"returned {type(result)}, expected XTYBatch"
                )
            schema.validate_batch(result)
            current = result

        for rule in _ordered_rules(schema, self.recompute_rules):
            current = rule.apply(current, schema)
            schema.validate_batch(current)

        changed = [
            field
            for field in sorted(self.preserves)
            if not _field_equal(getattr(original, field), getattr(current, field))
        ]
        if changed:
            raise ViewError(
                f"view {self.name!r} declares preserves={sorted(self.preserves)!r} "
                f"but changed {changed!r}"
            )
        if not batch.equal_to(original):
            raise ViewError(
                f"view {self.name!r} mutated its source batch; views are pure "
                "functions of (batch, rng_key)"
            )
        return current

    def transform_descriptions(self) -> tuple[str, ...]:
        return tuple(transform.describe() for transform in self.transforms)

    def recompute_descriptions(self) -> tuple[str, ...]:
        return tuple(rule.describe() for rule in self.recompute_rules)


def _view_seed(rng_key: int, name: str) -> int:
    """Fold a view name into a run/step key without Python's salted hash."""
    digest = hashlib.blake2b(f"{rng_key}:{name}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _stale_derived(schema: Schema, affected: set[str]) -> set[str]:
    """Derived columns invalidated directly or through a changed ancestor."""
    changed = set(affected)
    stale: set[str] = set()
    progressed = True
    while progressed:
        progressed = False
        for spec in schema.features:
            if not spec.derived_from or spec.name in stale:
                continue
            if spec.name in affected or any(
                parent in changed for parent in spec.derived_from
            ):
                stale.add(spec.name)
                changed.add(spec.name)
                progressed = True
    return stale


def _ordered_rules(
    schema: Schema, rules: tuple[RecomputeRule, ...]
) -> tuple[RecomputeRule, ...]:
    """Rules in dependency order, independent of declaration order."""
    by_name = {rule.feature: rule for rule in rules}
    ordered: list[RecomputeRule] = []
    placed: set[str] = set()
    while len(ordered) < len(rules):
        ready = [
            rule
            for rule in rules
            if rule.feature not in placed
            and all(
                parent not in by_name or parent in placed
                for parent in schema.feature(rule.feature).derived_from
            )
        ]
        if not ready:  # Schema already guarantees an acyclic feature graph.
            raise ViewError("recompute rules could not be ordered")  # pragma: no cover
        ready.sort(key=lambda rule: schema.index_of(rule.feature))
        ordered.append(ready[0])
        placed.add(ready[0].feature)
    return tuple(ordered)


def _field_equal(left: object, right: object) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, Tensor) and isinstance(right, Tensor):
        return bool(torch.equal(left, right))
    return left == right


def _duplicates(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for name in names:
        if name in seen:
            repeated.add(name)
        seen.add(name)
    return sorted(repeated)


__all__ = [
    "PRESERVED_FIELDS",
    "FeatureValues",
    "PreservedField",
    "RecomputeFunction",
    "RecomputeRule",
    "ViewSpec",
    "ViewTransform",
]
