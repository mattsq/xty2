"""Tier 0 — the view/parameter-set consistency objective (`DESIGN.md` §4.2)."""

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
from xty2.objectives import ConsistencyLoss

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)

LEFT = Realisation(view="weak_x")
RIGHT = Realisation(view="strong_x")


def _state(left_logits: torch.Tensor, right_logits: torch.Tensor) -> State:
    return State(
        {
            LEFT: {Port.T_GIVEN_X: CategoricalTreatment(left_logits)},
            RIGHT: {Port.T_GIVEN_X: CategoricalTreatment(right_logits)},
        }
    )


def _objective(**overrides: object) -> ConsistencyLoss:
    defaults: dict[str, object] = {
        "port": Port.T_GIVEN_X,
        "left": LEFT,
        "right": RIGHT,
        "divergence": "kl",
        "stop_grad": "none",
        "rows": "all",
    }
    return ConsistencyLoss(**(defaults | overrides))  # type: ignore[arg-type]


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    left = torch.linspace(-1.0, 1.0, BATCH_SIZE * NUM_TREATMENTS).reshape(
        BATCH_SIZE, NUM_TREATMENTS
    )
    right = torch.flip(left, dims=(1,)) + 0.2
    return left, right


def test_kl_is_the_mean_directed_divergence_over_eligible_rows() -> None:
    left_logits, right_logits = _inputs()
    rows = torch.tensor([0, 2, 5])
    term = _objective().compute(
        _state(left_logits, right_logits),
        make_batch(),
        rows,
        TrainContext(global_step=0, schema=make_schema()),
    )
    left = left_logits.softmax(dim=-1)
    right = right_logits.softmax(dim=-1)
    expected = (left * (left.log() - right.log())).sum(dim=-1)[rows].mean()
    assert torch.allclose(term.value, expected)
    assert term.n == 3


def test_mse_is_the_mean_over_classes_then_rows() -> None:
    left_logits, right_logits = _inputs()
    rows = torch.arange(BATCH_SIZE)
    term = _objective(divergence="mse").compute(
        _state(left_logits, right_logits),
        make_batch(),
        rows,
        TrainContext(global_step=0, schema=make_schema()),
    )
    expected = (
        (left_logits.softmax(-1) - right_logits.softmax(-1)).square().mean(-1).mean()
    )
    assert torch.allclose(term.value, expected)


@pytest.mark.parametrize("detached", ["left", "right"])
def test_stop_grad_detaches_exactly_the_declared_side(detached: str) -> None:
    left_logits, right_logits = _inputs()
    left_logits.requires_grad_()
    right_logits.requires_grad_()
    objective = _objective(stop_grad=detached)
    term = objective.compute(
        _state(left_logits, right_logits),
        make_batch(),
        torch.arange(BATCH_SIZE),
        TrainContext(global_step=0, schema=make_schema()),
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    if detached == "left":
        assert left_logits.grad is None
        assert right_logits.grad is not None
        assert objective.detaches == frozenset({(Port.T_GIVEN_X, LEFT)})
    else:
        assert right_logits.grad is None
        assert left_logits.grad is not None
        assert objective.detaches == frozenset({(Port.T_GIVEN_X, RIGHT)})


def test_no_rows_returns_the_framework_zero_term() -> None:
    left_logits, right_logits = _inputs()
    term = _objective().compute(
        _state(left_logits, right_logits),
        make_batch(),
        torch.zeros(0, dtype=torch.long),
        TrainContext(global_step=0, schema=make_schema()),
    )
    assert term.n == 0
    assert float(term.value) == 0.0


def test_stop_grad_is_explicit_and_card_bound() -> None:
    with pytest.raises(CardKeyError, match="detached_targets"):
        ConsistencyLoss(
            port=Port.T_GIVEN_X,
            left=LEFT,
            right=RIGHT,
            divergence="kl",
        )


def test_an_identical_pair_is_rejected_as_an_identically_zero_term() -> None:
    with pytest.raises(LossError, match=r"compares .* with itself"):
        _objective(right=LEFT)


def test_a_non_distribution_port_is_rejected_until_a_recipe_needs_semantics() -> None:
    with pytest.raises(LossError, match="currently compares treatment distributions"):
        _objective(port=Port.X_REPR)


def test_unknown_divergence_and_stop_gradient_are_rejected() -> None:
    with pytest.raises(LossError, match="divergence"):
        _objective(divergence="js")
    with pytest.raises(LossError, match="stop_grad"):
        _objective(stop_grad="target")
