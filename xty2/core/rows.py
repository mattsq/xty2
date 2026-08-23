"""Row populations and their resolution (`DESIGN.md` §1.3, §7.0).

Every objective declares which rows it is entitled to, and every stage declares
which rows it may touch at all. The engine resolves those declarations against
a batch into an explicit `RowIndex`; it does **not** hand over a pre-sliced
batch, because state holds full-batch quantities — some of them distribution
objects that are not generally sliceable — and pre-slicing only one side is how
predictions get silently misaligned against rows.

Two kinds of emptiness are distinguished here, and keeping them apart is the
point of the module:

* empty **in this batch** — a runtime `n = 0`, logged as coverage;
* empty **by construction** — `t_observed ∩ t_missing`, which is a compile-time
  error (P2) rather than an objective that returns zero forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.errors import Xty2Error

Rows = Literal["all", "t_observed", "t_missing", "y_observed"]
"""The declared row populations (`DESIGN.md` §1.3)."""

RowIndex = Tensor
"""`[n]` long, indices into the shared batch axis."""


@dataclass(frozen=True)
class _Population:
    """A row population as the facts it asserts and the facts it denies.

    Disjointness is then decidable without a batch: two populations are empty
    by construction iff one requires a fact the other forbids.
    """

    name: str
    requires: frozenset[str] = frozenset()
    forbids: frozenset[str] = frozenset()


ROW_POPULATIONS: dict[Rows, _Population] = {
    "all": _Population("all"),
    "t_observed": _Population("t_observed", requires=frozenset({"t"})),
    "t_missing": _Population("t_missing", forbids=frozenset({"t"})),
    "y_observed": _Population("y_observed", requires=frozenset({"y"})),
}


def validate_population(population: Rows) -> None:
    """Raise unless `population` is one of the declared row populations."""
    if population not in ROW_POPULATIONS:
        raise Xty2Error(
            f"unknown row population {population!r}; "
            f"expected one of {list(get_args(Rows))!r}"
        )


def row_mask(population: Rows, batch: XTYBatch) -> Tensor:
    """`[B]` bool mask of the rows in `population`."""
    validate_population(population)
    if population == "all":
        return torch.ones(batch.batch_size, dtype=torch.bool, device=batch.device)
    if population == "t_observed":
        return batch.t_observed
    if population == "t_missing":
        return batch.t_missing
    return batch.y_observed


def resolve_rows(batch: XTYBatch, *populations: Rows) -> RowIndex:
    """Resolve the intersection of `populations` to a `RowIndex`.

    `Stage.rows` and `Objective.rows` compose by intersection (`DESIGN.md`
    §7.0), which is what passing more than one population means here. With no
    population named the result is every row.

    Returns an empty `[0]` long tensor when nothing is eligible — never `None`,
    never a `NaN`-producing empty reduction (`DESIGN.md` §1.3).
    """
    mask = torch.ones(batch.batch_size, dtype=torch.bool, device=batch.device)
    for population in populations:
        mask = mask & row_mask(population, batch)
    return torch.nonzero(mask, as_tuple=False).flatten()


def populations_are_disjoint(left: Rows, right: Rows) -> bool:
    """Is `left ∩ right` empty for *every* batch, not just this one?

    This is the predicate the compiler applies to `Stage.rows` against
    `Objective.rows` (`DESIGN.md` §7.0); it is here, with the populations
    themselves, because it is a statement about their meaning rather than
    about any program.
    """
    validate_population(left)
    validate_population(right)
    a, b = ROW_POPULATIONS[left], ROW_POPULATIONS[right]
    return bool(a.requires & b.forbids) or bool(b.requires & a.forbids)
