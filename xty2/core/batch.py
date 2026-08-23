"""`XTYBatch` — the one batch type (`DESIGN.md` §1.1).

Two rules from the design are enforced here rather than documented:

* **No sentinel values.** Missingness is carried by `t_observed` / `y_observed`
  and by nothing else. `t` is validated as a valid class index on *every* row,
  so `t == -1` is not representable and cannot leak into `one_hot`, into an
  embedding, or into a loss reduction. Where `t_observed` is false, `t` holds
  an arbitrary valid index and reading it is a bug.
* **Functional transforms.** `frozen=True` stops field rebinding; it does not
  stop an in-place write to a tensor leaf. `replace()` and `clone()` are
  functional, and `assert_unchanged_by` gives the Tier 0 test its teeth: it
  clones a batch, runs a transform, and asserts the original is bit-identical
  afterwards.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import Any, cast

import torch
from torch import Tensor

from xty2.core.errors import BatchContractError


@dataclass(frozen=True, eq=False)
class XTYBatch:
    """A batch of rows with possibly-missing treatments.

    Every tensor shares one batch axis of length `B`, in one row order. That
    is what makes a `RowIndex` (`xty2.core.rows`) index the batch and the
    computed state interchangeably.
    """

    x: Tensor
    """`[B, D]` float features; mixed types are encoded upstream."""
    t: Tensor
    """`[B]` long, in `[0, K)` on every row — see the no-sentinel rule."""
    y: Tensor
    """`[B, *Dy]` outcome."""
    t_observed: Tensor
    """`[B]` bool; false rows carry an arbitrary valid `t`."""
    y_observed: Tensor
    """`[B]` bool; all-true in v1, still carried explicitly."""
    row_id: Tensor
    """`[B]` long, unique, a stable index into the source dataset."""
    fold_id: Tensor | None = None
    """`[B]` long fold assignment, for cross-fitting and leakage checks."""
    weight: Tensor | None = None
    """`[B]` float sample weights; finite and non-negative."""

    def __post_init__(self) -> None:
        self._check_field("x", self.x, ndim=2, dtype="float")
        batch_size = self.x.shape[0]
        self._check_field("t", self.t, ndim=1, dtype="long", size=batch_size)
        self._check_field("y", self.y, ndim=None, dtype=None, size=batch_size)
        self._check_field(
            "t_observed", self.t_observed, ndim=1, dtype="bool", size=batch_size
        )
        self._check_field(
            "y_observed", self.y_observed, ndim=1, dtype="bool", size=batch_size
        )
        self._check_field("row_id", self.row_id, ndim=1, dtype="long", size=batch_size)
        if self.fold_id is not None:
            self._check_field(
                "fold_id", self.fold_id, ndim=1, dtype="long", size=batch_size
            )
        if self.weight is not None:
            self._check_field(
                "weight", self.weight, ndim=1, dtype="float", size=batch_size
            )

        if batch_size:
            if int(self.t.min()) < 0:
                raise BatchContractError(
                    "batch.t holds a negative value. Missingness is carried by "
                    "t_observed, never by a sentinel (DESIGN.md §1.1): on rows "
                    "where t is unobserved, store an arbitrary valid class index."
                )
            if len(set(self.row_id.tolist())) != batch_size:
                raise BatchContractError(
                    "batch.row_id must be unique — artifacts and provenance "
                    "checks are keyed by it (DESIGN.md §7.1)"
                )
            if self.fold_id is not None and int(self.fold_id.min()) < 0:
                raise BatchContractError("batch.fold_id must be non-negative")
            # `weight.min() < 0` is false for NaN, so a NaN weight would pass
            # the non-negative contract and reach a weighted loss instead.
            if self.weight is not None and not bool(
                (torch.isfinite(self.weight) & (self.weight >= 0)).all()
            ):
                raise BatchContractError("batch.weight must be finite and non-negative")

        devices = {tensor.device for tensor in self._tensors().values()}
        if len(devices) > 1:
            raise BatchContractError(
                "every batch tensor must be on one device, got "
                f"{sorted(map(str, devices))}"
            )

    @staticmethod
    def _check_field(
        name: str, value: object, *, ndim: int | None, dtype: str | None, size: int = -1
    ) -> None:
        if not isinstance(value, Tensor):
            raise BatchContractError(
                f"batch.{name} must be a Tensor, got {type(value)}"
            )
        if ndim is not None and value.ndim != ndim:
            raise BatchContractError(
                f"batch.{name} must be {ndim}-D, got shape {tuple(value.shape)}"
            )
        if value.ndim < 1:
            raise BatchContractError(
                f"batch.{name} must have a leading batch axis, got a scalar tensor"
            )
        if dtype == "float" and not value.is_floating_point():
            raise BatchContractError(
                f"batch.{name} must be a float tensor, got {value.dtype}"
            )
        if dtype == "long" and value.dtype != torch.long:
            raise BatchContractError(
                f"batch.{name} must be a long tensor, got {value.dtype}"
            )
        if dtype == "bool" and value.dtype != torch.bool:
            raise BatchContractError(
                f"batch.{name} must be a bool tensor, got {value.dtype}"
            )
        if size >= 0 and value.shape[0] != size:
            raise BatchContractError(
                f"batch.{name} has batch axis {value.shape[0]}, expected {size} — "
                "every batch tensor shares one batch axis in one row order"
            )

    def _tensors(self) -> dict[str, Tensor]:
        present: dict[str, Tensor] = {}
        for spec in fields(self):
            value = getattr(self, spec.name)
            if isinstance(value, Tensor):
                present[spec.name] = value
        return present

    # -- basic properties --------------------------------------------------

    @property
    def batch_size(self) -> int:
        """`B`."""
        return int(self.x.shape[0])

    def __len__(self) -> int:
        return self.batch_size

    @property
    def device(self) -> torch.device:
        return self.x.device

    @property
    def t_missing(self) -> Tensor:
        """`[B]` bool, the complement of `t_observed`."""
        return ~self.t_observed

    # -- functional transforms --------------------------------------------

    def replace(self, **changes: Tensor | None) -> XTYBatch:
        """Return a new, revalidated batch with `changes` applied."""
        unknown = set(changes) - {spec.name for spec in fields(self)}
        if unknown:
            raise BatchContractError(f"unknown batch field(s) {sorted(unknown)!r}")
        return replace(self, **cast(dict[str, Any], changes))

    def clone(self) -> XTYBatch:
        """Return a batch whose tensors share no storage with this one."""
        return replace(
            self, **{name: tensor.clone() for name, tensor in self._tensors().items()}
        )

    def equal_to(self, other: XTYBatch) -> bool:
        """Bit-identical comparison, field by field.

        `eq=False` on the dataclass is deliberate: a generated `__eq__` would
        compare tensors and return a tensor, and `assert a == b` on a tensor is
        either a crash or, worse, a silently truthy scalar.
        """
        mine, theirs = self._tensors(), other._tensors()
        if mine.keys() != theirs.keys():
            return False
        return all(
            tensor.dtype == theirs[name].dtype
            and tensor.shape == theirs[name].shape
            and bool(torch.equal(tensor, theirs[name]))
            for name, tensor in mine.items()
        )


def assert_unchanged_by(
    transform: Callable[[XTYBatch], XTYBatch], batch: XTYBatch
) -> XTYBatch:
    """Run `transform` and assert it did not write into `batch`.

    The batch-immutability invariant (`FIDELITY.md` §3, Tier 0) in executable
    form: every registered transform is run through this. It returns the
    transform's output so a caller can go on to check what the transform
    *did*, and it raises `BatchContractError` — not `AssertionError` — so the
    failure survives `python -O`.
    """
    before = batch.clone()
    result = transform(batch)
    if not batch.equal_to(before):
        raise BatchContractError(
            f"{getattr(transform, '__name__', transform)!r} wrote into its input "
            "batch. Transforms are functional: return a new XTYBatch and perform "
            "no in-place write on any tensor reachable from the input "
            "(DESIGN.md §1.1)."
        )
    if not isinstance(result, XTYBatch):
        raise BatchContractError(
            f"{getattr(transform, '__name__', transform)!r} returned "
            f"{type(result)}, expected an XTYBatch"
        )
    return result
