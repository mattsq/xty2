"""Tier 0 — the objective contract and the four things every objective needs.

`DESIGN.md` §4 and §6.1, and the two rules they lean on:

* **the zero-eligible-row rule** (§1.3) — an objective with no eligible rows
  returns `value=0.0, n=0`, never a `NaN` from a mean over an empty tensor. It
  is the single most common cause of a silently dead objective, and it is
  obeyed here once, in `reduce_rows`, rather than in each loss;
* **the no-sentinel rule** (§1.1) — `batch.t` is only meaningful where
  `t_observed`, and objectives evaluate full-batch quantities before gathering,
  so `treatment_at` makes the ineligible positions unreadable rather than
  merely forbidden.
"""

from typing import cast

import pytest
import torch
from xty2.core import (
    DEFAULT,
    REQUIRED,
    CategoricalTreatment,
    CompileError,
    GaussianOutcome,
    LossError,
    LossTerm,
    Port,
    PortContractError,
    PortValue,
    Ramp,
    State,
    Weighted,
    XTYBatch,
    apply_reduction,
    candidate_treatments,
    outcome_distribution,
    reduce_rows,
    treatment_at,
    treatment_distribution,
)
from xty2.objectives import ObservedOutcomeNLL

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    ToyObjective,
    backward,
    make_batch,
    make_schema,
)

# ---------------------------------------------------------------------------
# LossTerm
# ---------------------------------------------------------------------------


def test_a_loss_term_carries_a_scalar_and_its_row_count() -> None:
    term = LossTerm(value=torch.tensor(1.5), n=4)
    assert float(term.value) == 1.5
    assert term.n == 4
    assert not term.is_empty


def test_a_non_scalar_value_is_rejected() -> None:
    with pytest.raises(LossError, match="must be a scalar"):
        LossTerm(value=torch.zeros(3), n=3)


def test_a_negative_row_count_is_rejected() -> None:
    with pytest.raises(LossError, match="non-negative"):
        LossTerm(value=torch.tensor(0.0), n=-1)


def test_a_bool_is_not_a_row_count() -> None:
    # `True` is an int by inheritance, and `n = True` would report one row on
    # every batch while every reduction quietly used it as a multiplier.
    with pytest.raises(LossError, match="must be an int"):
        LossTerm(value=torch.tensor(0.0), n=True)


def test_an_empty_term_must_be_exactly_zero() -> None:
    with pytest.raises(LossError, match="n = 0 but value"):
        LossTerm(value=torch.tensor(float("nan")), n=0)


def test_the_empty_term_is_zero_and_detached() -> None:
    term = LossTerm.empty(like=torch.zeros(3, requires_grad=True))
    assert term.is_empty
    assert float(term.value) == 0.0
    assert not term.value.requires_grad


def test_diagnostics_are_finite_floats() -> None:
    # A scalar tensor is accepted and coerced: an objective computing a
    # diagnostic alongside its value has one in hand, not a float.
    reported = cast("dict[str, float]", {"coverage": torch.tensor(0.25)})
    term = LossTerm(value=torch.tensor(0.5), n=2, diagnostics=reported)
    assert term.diagnostics == {"coverage": 0.25}
    with pytest.raises(LossError, match="must be finite"):
        LossTerm(value=torch.tensor(0.5), n=2, diagnostics={"bad": float("inf")})
    with pytest.raises(LossError, match="must be a float"):
        LossTerm(value=torch.tensor(0.5), n=2, diagnostics={"bad": "0.5"})  # type: ignore[dict-item]


def test_diagnostics_cannot_be_written_through() -> None:
    # A logging surface, not a second return channel: the mixer copies these
    # into the step log and nothing downstream may edit them.
    term = LossTerm(value=torch.tensor(0.5), n=2, diagnostics={"coverage": 0.25})
    with pytest.raises(TypeError):
        term.diagnostics["coverage"] = 1.0  # type: ignore[index]


# ---------------------------------------------------------------------------
# Reduction (DESIGN.md §6.1)
# ---------------------------------------------------------------------------


def test_the_three_reductions_are_what_the_table_says() -> None:
    term = LossTerm(value=torch.tensor(2.0), n=4)
    assert float(apply_reduction(term, "mean", batch_size=10)) == 2.0
    assert float(apply_reduction(term, "sum", batch_size=10)) == 8.0
    population = float(apply_reduction(term, "population", batch_size=10))
    assert population == pytest.approx(0.8)


def test_sum_and_mean_differ_by_a_factor_that_varies_per_batch() -> None:
    # The reason `losses.reduction` is a required card field: with a row
    # population that varies batch to batch — every `t_missing` term — the two
    # modes weight the same objective differently on every step.
    first = LossTerm(value=torch.tensor(2.0), n=2)
    second = LossTerm(value=torch.tensor(2.0), n=6)
    assert float(apply_reduction(first, "mean", batch_size=8)) == float(
        apply_reduction(second, "mean", batch_size=8)
    )
    assert float(apply_reduction(first, "sum", batch_size=8)) != float(
        apply_reduction(second, "sum", batch_size=8)
    )


def test_an_unknown_reduction_is_rejected() -> None:
    term = LossTerm(value=torch.tensor(1.0), n=1)
    with pytest.raises(LossError, match="unknown reduction"):
        apply_reduction(term, "median", batch_size=4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# reduce_rows — the zero-eligible-row rule, in one place
# ---------------------------------------------------------------------------


def test_reduce_rows_averages_over_the_eligible_set() -> None:
    per_row = torch.tensor([1.0, 2.0, 3.0, 4.0])
    term = reduce_rows(per_row, torch.tensor([1, 3]))
    assert float(term.value) == 3.0
    assert term.n == 2


def test_no_eligible_rows_gives_zero_and_no_nan() -> None:
    term = reduce_rows(torch.tensor([1.0, 2.0]), torch.zeros(0, dtype=torch.long))
    assert term.n == 0
    assert float(term.value) == 0.0
    assert not torch.isnan(term.value)


def test_an_ineligible_nan_never_reaches_the_value() -> None:
    # Ineligible rows may hold anything at all — that is what makes them
    # ineligible — and gathering before reducing is what keeps them out.
    per_row = torch.tensor([1.0, float("nan"), 3.0])
    term = reduce_rows(per_row, torch.tensor([0, 2]))
    assert float(term.value) == 2.0


def test_reduce_rows_keeps_the_gradient_path() -> None:
    per_row = torch.arange(4, dtype=torch.float32).requires_grad_()
    backward(reduce_rows(per_row, torch.tensor([0, 1])).value)
    assert per_row.grad is not None
    assert torch.equal(per_row.grad, torch.tensor([0.5, 0.5, 0.0, 0.0]))


def test_per_row_losses_must_share_the_batch_axis() -> None:
    with pytest.raises(LossError, match=r"must be \[B\]"):
        reduce_rows(torch.zeros(3, 2), torch.tensor([0]))


def test_rows_must_be_a_long_index() -> None:
    with pytest.raises(LossError, match="long RowIndex"):
        reduce_rows(torch.zeros(3), torch.tensor([True, False, True]))


# ---------------------------------------------------------------------------
# treatment_at — the no-sentinel rule
# ---------------------------------------------------------------------------


def test_treatment_at_carries_nothing_from_an_ineligible_row() -> None:
    batch = make_batch()
    rows = torch.tensor([0, 2])
    masked = treatment_at(batch, rows)
    assert torch.equal(masked.index_select(0, rows), batch.t.index_select(0, rows))
    ineligible = [i for i in range(BATCH_SIZE) if i not in (0, 2)]
    assert torch.equal(
        masked[ineligible], torch.zeros(len(ineligible), dtype=torch.long)
    )


def test_treatment_at_does_not_write_into_the_batch() -> None:
    batch = make_batch()
    before = batch.t.clone()
    treatment_at(batch, torch.tensor([1, 4]))
    assert torch.equal(batch.t, before)


def test_treatment_at_with_no_eligible_rows_reads_nothing() -> None:
    batch = make_batch()
    masked = treatment_at(batch, torch.zeros(0, dtype=torch.long))
    assert torch.equal(masked, torch.zeros(BATCH_SIZE, dtype=torch.long))


# ---------------------------------------------------------------------------
# candidate_treatments — the matrix §3.1 is stated against
# ---------------------------------------------------------------------------


def test_the_candidate_matrix_is_arange_k_in_every_row() -> None:
    candidates = candidate_treatments(make_batch(), make_schema())
    assert candidates.shape == (BATCH_SIZE, NUM_TREATMENTS)
    assert torch.equal(
        candidates, torch.arange(NUM_TREATMENTS).expand(BATCH_SIZE, NUM_TREATMENTS)
    )


def test_the_candidate_matrix_is_independent_of_the_observed_treatment() -> None:
    # Built as `t[:, None].expand(B, K)` it would repeat each row's own
    # treatment across every column, turning exact marginalisation into K
    # copies of the complete-case term (`FIDELITY.md` §3).
    schema = make_schema()
    shuffled = make_batch(t=torch.zeros(BATCH_SIZE, dtype=torch.long))
    assert torch.equal(
        candidate_treatments(make_batch(), schema),
        candidate_treatments(shuffled, schema),
    )


# ---------------------------------------------------------------------------
# Narrowing a port at the read
# ---------------------------------------------------------------------------


def _state(**ports: PortValue) -> State:
    return State({DEFAULT: {Port(name): value for name, value in ports.items()}})


def test_a_port_carrying_the_wrong_kind_is_reported_at_the_read() -> None:
    state = _state(**{Port.Y_GIVEN_XT: CategoricalTreatment(torch.zeros(2, 3))})
    with pytest.raises(PortContractError, match="as an outcome distribution"):
        outcome_distribution(state, Port.Y_GIVEN_XT, DEFAULT, objective="nll")


def test_a_treatment_port_carrying_an_outcome_head_is_rejected() -> None:
    head = GaussianOutcome(loc=torch.zeros(2, 3), scale=torch.ones(2, 3))
    state = _state(**{Port.T_GIVEN_X: head})
    with pytest.raises(PortContractError, match="as a treatment distribution"):
        treatment_distribution(state, Port.T_GIVEN_X, DEFAULT, objective="nll")


# ---------------------------------------------------------------------------
# Weighted — no default for a paper-governed field (DESIGN.md §9.1)
# ---------------------------------------------------------------------------


def test_a_weighted_term_carries_a_schedule_and_a_reduction() -> None:
    term = Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="mean")
    assert term.name == "observed_outcome_nll"
    assert term.rows == "t_observed"
    assert term.requires == frozenset({(Port.Y_GIVEN_XT, DEFAULT)})
    assert term.weight_at(0) == 1.0
    assert term.schedule.describe() == "constant 1.0"


def test_a_schedule_may_be_given_instead_of_a_number() -> None:
    term = Weighted(
        ObservedOutcomeNLL(), weight=Ramp(0.0, 0.5, steps=10), reduction="sum"
    )
    assert term.weight_at(0) == 0.0
    assert term.weight_at(10) == 0.5


def test_a_term_with_no_weight_is_rejected() -> None:
    with pytest.raises(CompileError, match="no usable default"):
        Weighted(ObservedOutcomeNLL(), reduction="mean")


def test_a_term_with_no_reduction_is_rejected() -> None:
    # `DESIGN.md` §6.1 writes `mean` as the default; it is REQUIRED here
    # because that same section explains why the wrong choice is invisible.
    with pytest.raises(CompileError, match="no usable default"):
        Weighted(ObservedOutcomeNLL(), weight=1.0)


def test_a_term_with_an_unknown_reduction_is_rejected() -> None:
    with pytest.raises(LossError, match="unknown reduction"):
        Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="median")  # type: ignore[arg-type]


def test_something_that_is_not_an_objective_is_rejected() -> None:
    with pytest.raises(CompileError, match="Weighted holds an Objective"):
        Weighted(object(), weight=1.0, reduction="mean")  # type: ignore[arg-type]


def test_the_required_sentinel_has_no_truth_value() -> None:
    # It stands in for a number the recipe has not set, so a field that
    # escaped the construction-time check fails loudly at first use.
    with pytest.raises(Exception, match="REQUIRED is a sentinel"):
        bool(REQUIRED)


# ---------------------------------------------------------------------------
# The protocol itself
# ---------------------------------------------------------------------------


def test_an_objective_is_recognised_structurally(batch: XTYBatch) -> None:
    toy = ToyObjective(name="toy", requires=frozenset({(Port.X_REPR, DEFAULT)}))
    assert Weighted(toy, weight=1.0, reduction="mean").objective is toy
    del batch
