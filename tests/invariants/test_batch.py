"""Tier 0 — `XTYBatch`: mask semantics, shape contracts, immutability.

`DESIGN.md` §1.1. Two rules carry the weight: missingness is never a sentinel,
and transforms never write into their input.
"""

import dataclasses

import pytest
import torch
from xty2.core import BatchContractError, XTYBatch, assert_unchanged_by

from tests.invariants.conftest import BATCH_SIZE, NUM_FEATURES, make_batch

# -- the no-sentinel rule ---------------------------------------------------


def test_a_negative_treatment_is_not_representable() -> None:
    """`t = -1` is the sentinel that leaks into one_hot and into embeddings."""
    t = torch.zeros(BATCH_SIZE, dtype=torch.long)
    t[2] = -1
    with pytest.raises(
        BatchContractError, match="Missingness is carried by t_observed"
    ):
        make_batch(t=t)


def test_missingness_needs_the_mask_not_the_value(batch: XTYBatch) -> None:
    """Unobserved rows still hold a valid class index, and the mask is the truth."""
    assert bool(batch.t_observed[:4].all())
    assert not bool(batch.t_observed[4:].any())
    assert bool((batch.t >= 0).all())
    assert bool(torch.equal(batch.t_missing, ~batch.t_observed))


def test_masks_are_required_and_boolean() -> None:
    with pytest.raises(BatchContractError, match="t_observed must be a bool tensor"):
        make_batch(t_observed=torch.ones(BATCH_SIZE, dtype=torch.long))
    with pytest.raises(TypeError):
        XTYBatch(  # type: ignore[call-arg]
            x=torch.randn(BATCH_SIZE, NUM_FEATURES),
            t=torch.zeros(BATCH_SIZE, dtype=torch.long),
            y=torch.randn(BATCH_SIZE),
            row_id=torch.arange(BATCH_SIZE),
        )


# -- shape, dtype and row-id contracts --------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("x", torch.randn(BATCH_SIZE), "must be 2-D"),
        ("x", torch.zeros(BATCH_SIZE, NUM_FEATURES, dtype=torch.long), "float tensor"),
        ("t", torch.zeros(BATCH_SIZE), "long tensor"),
        ("t", torch.zeros(BATCH_SIZE, 1, dtype=torch.long), "must be 1-D"),
        ("y", torch.tensor(1.0), "leading batch axis"),
        ("row_id", torch.zeros(BATCH_SIZE), "long tensor"),
        ("fold_id", torch.zeros(BATCH_SIZE), "long tensor"),
        ("weight", torch.zeros(BATCH_SIZE, dtype=torch.long), "float tensor"),
        ("y_observed", torch.zeros(BATCH_SIZE), "bool tensor"),
    ],
)
def test_field_contracts_are_enforced(
    field: str, value: torch.Tensor, message: str
) -> None:
    with pytest.raises(BatchContractError, match=message):
        make_batch(**{field: value})


@pytest.mark.parametrize("field", ["t", "y", "t_observed", "y_observed", "row_id"])
def test_every_tensor_shares_one_batch_axis(field: str) -> None:
    short = {
        "t": torch.zeros(BATCH_SIZE - 1, dtype=torch.long),
        "y": torch.randn(BATCH_SIZE - 1),
        "t_observed": torch.ones(BATCH_SIZE - 1, dtype=torch.bool),
        "y_observed": torch.ones(BATCH_SIZE - 1, dtype=torch.bool),
        "row_id": torch.arange(BATCH_SIZE - 1),
    }[field]
    with pytest.raises(BatchContractError, match="batch axis"):
        make_batch(**{field: short})


def test_row_ids_must_be_unique() -> None:
    duplicated = torch.zeros(BATCH_SIZE, dtype=torch.long)
    with pytest.raises(BatchContractError, match="row_id must be unique"):
        make_batch(row_id=duplicated)


def test_optional_fields_are_validated_when_present() -> None:
    with pytest.raises(BatchContractError, match="fold_id must be non-negative"):
        make_batch(fold_id=torch.full((BATCH_SIZE,), -1, dtype=torch.long))
    with pytest.raises(BatchContractError, match="weight must be finite"):
        make_batch(weight=-torch.ones(BATCH_SIZE))
    ok = make_batch(
        fold_id=torch.zeros(BATCH_SIZE, dtype=torch.long),
        weight=torch.ones(BATCH_SIZE),
    )
    assert ok.fold_id is not None and ok.weight is not None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_weight_is_rejected(bad: float) -> None:
    """`weight.min() < 0` is false for NaN, so the contract cannot rest on it.

    A NaN that survives construction reaches a weighted loss and propagates
    into the experiment instead of failing at the batch boundary.
    """
    weight = torch.ones(BATCH_SIZE)
    weight[3] = bad
    with pytest.raises(BatchContractError, match="finite and non-negative"):
        make_batch(weight=weight)


def test_an_empty_batch_is_legal(batch: XTYBatch) -> None:
    """B = 0 is degenerate, not invalid; the row-population rules rely on it."""
    empty = XTYBatch(
        x=torch.zeros(0, NUM_FEATURES),
        t=torch.zeros(0, dtype=torch.long),
        y=torch.zeros(0),
        t_observed=torch.zeros(0, dtype=torch.bool),
        y_observed=torch.zeros(0, dtype=torch.bool),
        row_id=torch.zeros(0, dtype=torch.long),
    )
    assert len(empty) == 0
    assert batch.batch_size == BATCH_SIZE


def test_multidimensional_outcomes_are_allowed() -> None:
    batch = make_batch(y=torch.randn(BATCH_SIZE, 2))
    assert batch.y.shape == (BATCH_SIZE, 2)


# -- immutability -----------------------------------------------------------


def test_fields_cannot_be_rebound(batch: XTYBatch) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.x = torch.zeros_like(batch.x)  # type: ignore[misc]


def test_replace_is_functional_and_revalidates(batch: XTYBatch) -> None:
    before = batch.clone()
    replaced = batch.replace(x=torch.zeros_like(batch.x))
    assert batch.equal_to(before)
    assert not replaced.equal_to(batch)
    with pytest.raises(BatchContractError, match="Missingness is carried"):
        batch.replace(t=torch.full_like(batch.t, -1))
    with pytest.raises(BatchContractError, match="unknown batch field"):
        batch.replace(nonsense=None)


def test_clone_shares_no_storage(batch: XTYBatch) -> None:
    cloned = batch.clone()
    assert cloned.equal_to(batch)
    cloned.x.add_(1.0)
    assert not cloned.equal_to(batch)


def test_equal_to_compares_dtype_shape_and_values(batch: XTYBatch) -> None:
    assert batch.equal_to(batch.clone())
    assert not batch.equal_to(batch.replace(y=batch.y + 1))
    assert not batch.equal_to(batch.replace(weight=torch.ones(BATCH_SIZE)))


def test_assert_unchanged_by_accepts_a_functional_transform(batch: XTYBatch) -> None:
    def scale_x(source: XTYBatch) -> XTYBatch:
        return source.replace(x=source.x * 2.0)

    result = assert_unchanged_by(scale_x, batch)
    assert torch.allclose(result.x, batch.x * 2.0)


def test_assert_unchanged_by_catches_an_in_place_write(batch: XTYBatch) -> None:
    """The mutation test for the harness itself.

    `frozen=True` does not stop this write, which is the whole reason the check
    exists (`DESIGN.md` §1.1). A harness that has never been seen to fail is
    not known to work.
    """

    def jitter_in_place(source: XTYBatch) -> XTYBatch:
        source.x.add_(1.0)
        return source

    with pytest.raises(BatchContractError, match="wrote into its input batch"):
        assert_unchanged_by(jitter_in_place, batch)


def test_assert_unchanged_by_rejects_a_non_batch_return(batch: XTYBatch) -> None:
    def wrong_return(source: XTYBatch) -> XTYBatch:
        return source.x  # type: ignore[return-value]

    with pytest.raises(BatchContractError, match="expected an XTYBatch"):
        assert_unchanged_by(wrong_return, batch)
