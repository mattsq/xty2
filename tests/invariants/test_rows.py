"""Tier 0 — row populations (`DESIGN.md` §1.3, §7.0).

The engine resolves declarations into an index rather than a pre-sliced batch,
so the same index addresses `batch` and `state`. These tests pin that, and pin
the distinction between the two kinds of emptiness: empty in this batch (a
runtime `n = 0`, logged) and empty by construction (a compile error in P2).
"""

import itertools

import pytest
import torch
from xty2.core import (
    ROW_POPULATIONS,
    Rows,
    Xty2Error,
    XTYBatch,
    populations_are_disjoint,
    resolve_rows,
    row_mask,
    validate_population,
)

from tests.invariants.conftest import BATCH_SIZE, make_batch

POPULATIONS: tuple[Rows, ...] = ("all", "t_observed", "t_missing", "y_observed")


def test_the_declared_populations_are_exactly_these_four() -> None:
    assert tuple(ROW_POPULATIONS) == POPULATIONS


@pytest.mark.parametrize("population", POPULATIONS)
def test_a_mask_is_boolean_and_batch_shaped(population: Rows, batch: XTYBatch) -> None:
    mask = row_mask(population, batch)
    assert mask.dtype == torch.bool
    assert mask.shape == (BATCH_SIZE,)


def test_each_population_selects_the_rows_it_names(batch: XTYBatch) -> None:
    assert resolve_rows(batch, "all").tolist() == list(range(BATCH_SIZE))
    assert resolve_rows(batch, "t_observed").tolist() == [0, 1, 2, 3]
    assert resolve_rows(batch, "t_missing").tolist() == [4, 5, 6]
    assert resolve_rows(batch, "y_observed").tolist() == list(range(BATCH_SIZE))


def test_no_populations_means_every_row(batch: XTYBatch) -> None:
    assert resolve_rows(batch).tolist() == list(range(BATCH_SIZE))


def test_populations_compose_by_intersection(batch: XTYBatch) -> None:
    """`Stage.rows ∩ Objective.rows` is what passing two populations means."""
    partial_y = torch.zeros(BATCH_SIZE, dtype=torch.bool)
    partial_y[2:6] = True
    scoped = make_batch(y_observed=partial_y)
    assert resolve_rows(scoped, "t_observed", "y_observed").tolist() == [2, 3]
    assert resolve_rows(scoped, "t_missing", "y_observed").tolist() == [4, 5]
    assert resolve_rows(batch, "all", "t_missing").tolist() == [4, 5, 6]


def test_a_row_index_addresses_batch_and_state_alike(batch: XTYBatch) -> None:
    """One batch axis, one index — the reason rows are handed over as indices."""
    rows = resolve_rows(batch, "t_observed")
    state_like = torch.arange(BATCH_SIZE, dtype=torch.float32) * 10.0
    assert torch.equal(batch.t[rows], batch.t[:4])
    assert torch.equal(state_like[rows], state_like[:4])
    assert batch.row_id[rows].tolist() == [100, 101, 102, 103]


def test_an_empty_population_is_an_empty_index_not_a_failure() -> None:
    """Zero eligible rows in *this* batch is normal and must stay indexable."""
    all_observed = make_batch(t_observed=torch.ones(BATCH_SIZE, dtype=torch.bool))
    rows = resolve_rows(all_observed, "t_missing")
    assert rows.dtype == torch.long
    assert rows.shape == (0,)
    assert rows.numel() == 0
    assert all_observed.y[rows].shape == (0,)


def test_an_empty_batch_resolves_to_an_empty_index() -> None:
    empty = XTYBatch(
        x=torch.zeros(0, 4),
        t=torch.zeros(0, dtype=torch.long),
        y=torch.zeros(0),
        t_observed=torch.zeros(0, dtype=torch.bool),
        y_observed=torch.zeros(0, dtype=torch.bool),
        row_id=torch.zeros(0, dtype=torch.long),
    )
    assert resolve_rows(empty, "all").shape == (0,)


def test_only_complementary_populations_are_empty_by_construction() -> None:
    """The compile-time half of the emptiness rule (`DESIGN.md` §7.0)."""
    assert populations_are_disjoint("t_observed", "t_missing")
    assert populations_are_disjoint("t_missing", "t_observed")
    disjoint = {
        pair
        for pair in itertools.product(POPULATIONS, repeat=2)
        if populations_are_disjoint(*pair)
    }
    assert disjoint == {("t_observed", "t_missing"), ("t_missing", "t_observed")}


def test_a_population_that_is_merely_empty_today_is_not_disjoint() -> None:
    """`y_observed ∩ t_missing` can be empty in a batch; it is still legal."""
    assert not populations_are_disjoint("y_observed", "t_missing")
    none_observed = make_batch(y_observed=torch.zeros(BATCH_SIZE, dtype=torch.bool))
    assert resolve_rows(none_observed, "y_observed", "t_missing").numel() == 0


@pytest.mark.parametrize("bad", ["", "T_OBSERVED", "t_unobserved", "y_missing"])
def test_an_unknown_population_is_rejected(bad: str, batch: XTYBatch) -> None:
    with pytest.raises(Xty2Error, match="unknown row population"):
        validate_population(bad)  # type: ignore[arg-type]
    with pytest.raises(Xty2Error, match="unknown row population"):
        row_mask(bad, batch)  # type: ignore[arg-type]
    with pytest.raises(Xty2Error, match="unknown row population"):
        populations_are_disjoint(bad, "all")  # type: ignore[arg-type]
