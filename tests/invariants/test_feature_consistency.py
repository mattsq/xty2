"""Tier 0 — the cosine feature-consistency objective (`DESIGN.md` §4.2).

The arithmetic under test is DoubleMatch eq. (3). Four properties are the ones
a wrong implementation gets wrong quietly: the two sides are *different ports*
and swapping them computes a different term; the target carries no gradient;
nothing is gated, so the denominator is the eligible set and only that; and the
value alone cannot tell a learned invariance from a collapsed representation,
which is what the diagnostics are for.
"""

import math

import pytest
import torch
from xty2.core import (
    CardKeyError,
    CategoricalTreatment,
    LossError,
    Port,
    Realisation,
    State,
    TrainContext,
)
from xty2.objectives import CosineFeatureConsistency

from tests.invariants.conftest import BATCH_SIZE, make_batch, make_schema

PREDICTION = Realisation(view="strong_x")
TARGET = Realisation(view="weak_x")
WIDTH = 5


def _state(prediction: torch.Tensor, target: torch.Tensor) -> State:
    return State(
        {
            PREDICTION: {Port.X_PROJ: prediction},
            TARGET: {Port.X_REPR: target},
        }
    )


def _objective(**overrides: object) -> CosineFeatureConsistency:
    defaults: dict[str, object] = {
        "prediction_port": Port.X_PROJ,
        "target_port": Port.X_REPR,
        "prediction": PREDICTION,
        "target": TARGET,
        "stop_grad": "target",
        "rows": "all",
    }
    return CosineFeatureConsistency(**(defaults | overrides))  # type: ignore[arg-type]


def _context() -> TrainContext:
    return TrainContext(global_step=0, schema=make_schema())


def _reference(prediction: torch.Tensor, target: torch.Tensor) -> float:
    """Eq. (3), transcribed as a loop over `i` rather than as a matrix."""
    rows = prediction.shape[0]
    total = 0.0
    for i in range(rows):
        cosine = float(
            torch.dot(prediction[i], target[i])
            / (prediction[i].norm() * target[i].norm())
        )
        total += -cosine
    return total / rows


def test_the_loss_is_the_papers_expression() -> None:
    prediction = torch.randn(BATCH_SIZE, WIDTH)
    target = torch.randn(BATCH_SIZE, WIDTH)
    term = _objective().compute(
        _state(prediction, target), make_batch(), torch.arange(BATCH_SIZE), _context()
    )
    assert term.n == BATCH_SIZE
    assert float(term.value) == pytest.approx(_reference(prediction, target), abs=1e-6)


def test_the_published_expression_carries_no_offset() -> None:
    """Eq. (3) is `-cos`; the reference implementation computes `1 - cos`.

    The two differ by a constant that contributes no gradient, so nothing about
    training distinguishes them and a logged curve compared against the
    reference's sits exactly 1.0 lower. That is a fact about the number this
    objective reports, which makes it this test's business rather than a
    comment's (card §7).
    """
    identical = torch.randn(BATCH_SIZE, WIDTH)
    term = _objective().compute(
        _state(identical, identical.clone()),
        make_batch(),
        torch.arange(BATCH_SIZE),
        _context(),
    )
    assert float(term.value) == pytest.approx(-1.0, abs=1e-6)


def test_the_cosine_ignores_the_scale_of_either_side() -> None:
    """Both sides are normalised inside `compute`, as eq. (3) writes them."""
    prediction = torch.randn(BATCH_SIZE, WIDTH)
    target = torch.randn(BATCH_SIZE, WIDTH)
    rows = torch.arange(BATCH_SIZE)
    plain = _objective().compute(
        _state(prediction, target), make_batch(), rows, _context()
    )
    scaled = _objective().compute(
        _state(prediction * 17.0, target * 0.03), make_batch(), rows, _context()
    )
    assert float(scaled.value) == pytest.approx(float(plain.value), abs=1e-6)


def test_only_the_prediction_side_takes_a_gradient() -> None:
    """Eq. (3): "we consider `z_i` as constant when evaluating the gradient"."""
    prediction = torch.randn(BATCH_SIZE, WIDTH, requires_grad=True)
    target = torch.randn(BATCH_SIZE, WIDTH, requires_grad=True)
    term = _objective().compute(
        _state(prediction, target), make_batch(), torch.arange(BATCH_SIZE), _context()
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    assert prediction.grad is not None
    assert torch.any(prediction.grad != 0.0)
    assert target.grad is None


def test_the_declared_detach_is_the_one_that_runs() -> None:
    objective = _objective()
    assert objective.requires == frozenset(
        {(Port.X_PROJ, PREDICTION), (Port.X_REPR, TARGET)}
    )
    assert objective.detaches == frozenset({(Port.X_REPR, TARGET)})
    assert not objective.batch_coupled


def test_the_denominator_counts_every_eligible_row_and_nothing_else() -> None:
    """Nothing is gated: `n` is the row set, and a subset averages over itself."""
    prediction = torch.randn(BATCH_SIZE, WIDTH)
    target = torch.randn(BATCH_SIZE, WIDTH)
    rows = torch.tensor([1, 4, 5])
    term = _objective().compute(
        _state(prediction, target), make_batch(), rows, _context()
    )
    assert term.n == 3
    assert float(term.value) == pytest.approx(
        _reference(prediction[rows], target[rows]), abs=1e-6
    )


def test_no_eligible_rows_is_the_zero_term_with_no_diagnostics() -> None:
    term = _objective().compute(
        _state(torch.randn(BATCH_SIZE, WIDTH), torch.randn(BATCH_SIZE, WIDTH)),
        make_batch(),
        torch.zeros(0, dtype=torch.long),
        _context(),
    )
    assert term.is_empty
    assert float(term.value) == 0.0
    assert dict(term.diagnostics) == {}


def test_a_collapsed_representation_scores_the_best_possible_loss() -> None:
    """Why the diagnostics exist, stated as the failure they detect.

    Eq. (3) has no negative pairs, so an encoder that maps every row to one
    direction attains `-1` — the *minimum* — while having stopped telling rows
    apart. The loss cannot distinguish that from a learned invariance; the
    concentration pair can, and this is the case that would be silent without
    it.
    """
    direction = torch.randn(1, WIDTH).expand(BATCH_SIZE, WIDTH).contiguous()
    term = _objective().compute(
        _state(direction, direction.clone()),
        make_batch(),
        torch.arange(BATCH_SIZE),
        _context(),
    )
    assert float(term.value) == pytest.approx(-1.0, abs=1e-6)
    assert term.diagnostics["prediction_concentration"] == pytest.approx(1.0, abs=1e-6)
    assert term.diagnostics["target_concentration"] == pytest.approx(1.0, abs=1e-6)


def test_spread_out_embeddings_concentrate_near_one_over_root_n() -> None:
    """The other end of the scale the card's §6.1 tolerance is stated on."""
    rows = 512
    torch.manual_seed(11)
    prediction = torch.randn(rows, 64)
    target = torch.randn(rows, 64)
    term = _objective().compute(
        _state(prediction, target),
        make_batch(
            x=torch.randn(rows, 4),
            t=torch.zeros(rows, dtype=torch.long),
            y=torch.randn(rows),
            t_observed=torch.ones(rows, dtype=torch.bool),
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(rows),
        ),
        torch.arange(rows),
        _context(),
    )
    isotropic = 1.0 / math.sqrt(rows)
    assert term.diagnostics["prediction_concentration"] < 5.0 * isotropic
    assert term.diagnostics["target_concentration"] < 5.0 * isotropic


def test_the_concentrations_are_taken_over_the_eligible_rows_only() -> None:
    """A statistic about the rows the term saw, not about the batch."""
    collapsed = torch.randn(1, WIDTH).expand(4, WIDTH)
    spread = torch.eye(3, WIDTH)
    prediction = torch.cat([collapsed, spread])
    term = _objective().compute(
        _state(prediction, prediction.clone()),
        make_batch(),
        torch.arange(4),
        _context(),
    )
    assert term.diagnostics["prediction_concentration"] == pytest.approx(1.0, abs=1e-6)


def test_the_roles_reach_the_plan_because_the_term_is_not_symmetric() -> None:
    """Swapping the sides trains the encoder and freezes the head instead.

    `requires` is a set, so nothing else in the plan distinguishes the two
    assignments; `plan_details` is what puts the difference in the digest
    (`DESIGN.md` §4).
    """
    details = _objective().plan_details()
    assert details[0] == "prediction (trained) = x_proj @ view=strong_x params=student"
    assert details[1] == "target (detached) = x_repr @ view=weak_x params=student"
    swapped = _objective(
        prediction_port=Port.X_REPR,
        target_port=Port.X_PROJ,
        prediction=TARGET,
        target=PREDICTION,
    )
    assert swapped.requires == _objective().requires
    assert swapped.plan_details() != details
    assert swapped.detaches != _objective().detaches


def test_the_two_sides_may_share_a_port_but_not_a_realisation_too() -> None:
    """One port under two views is a legitimate declaration; itself is not."""
    shared = _objective(prediction_port=Port.X_REPR)
    assert shared.requires == frozenset(
        {(Port.X_REPR, PREDICTION), (Port.X_REPR, TARGET)}
    )
    with pytest.raises(LossError, match="with itself"):
        _objective(prediction_port=Port.X_REPR, prediction=TARGET)


def test_a_distribution_port_is_refused_with_the_objective_that_takes_one() -> None:
    with pytest.raises(LossError, match="ConsistencyLoss"):
        _objective(target_port=Port.T_GIVEN_X)


def test_the_stop_gradient_has_one_reviewed_value() -> None:
    with pytest.raises(LossError, match="trivial optimum"):
        _objective(stop_grad="none")


def test_the_stop_gradient_is_a_card_key_with_no_default() -> None:
    with pytest.raises(CardKeyError, match=r"gradients\.detached_targets"):
        CosineFeatureConsistency(
            prediction_port=Port.X_PROJ,
            target_port=Port.X_REPR,
            prediction=PREDICTION,
            target=TARGET,
        )


def test_mismatched_widths_are_a_named_failure_rather_than_a_broadcast() -> None:
    with pytest.raises(LossError, match="dimension-preserving"):
        _objective().compute(
            _state(torch.randn(BATCH_SIZE, WIDTH), torch.randn(BATCH_SIZE, WIDTH + 1)),
            make_batch(),
            torch.arange(BATCH_SIZE),
            _context(),
        )


def test_a_port_carrying_a_distribution_at_runtime_is_a_contract_failure() -> None:
    """The `PortSpec` is the contract; the read is where it is enforced."""
    state = State(
        {
            PREDICTION: {Port.X_PROJ: CategoricalTreatment(torch.randn(BATCH_SIZE, 3))},
            TARGET: {Port.X_REPR: torch.randn(BATCH_SIZE, WIDTH)},
        }
    )
    with pytest.raises(Exception, match="embedding tensor"):
        _objective().compute(state, make_batch(), torch.arange(BATCH_SIZE), _context())


def test_a_realisation_with_the_wrong_row_count_is_named() -> None:
    with pytest.raises(LossError, match="for a batch of"):
        _objective().compute(
            _state(torch.randn(BATCH_SIZE + 1, WIDTH), torch.randn(BATCH_SIZE, WIDTH)),
            make_batch(),
            torch.arange(BATCH_SIZE),
            _context(),
        )


def test_an_unknown_row_population_is_rejected_by_name() -> None:
    with pytest.raises(LossError, match="cosine_feature_consistency"):
        _objective(rows="t_confident")
