"""Tier 0 — the confidence-gated pseudo-label objective (`DESIGN.md` §4.2).

The arithmetic under test is FixMatch eq. (4). Three properties are the ones a
wrong implementation gets wrong quietly: the label comes from the *target*
realisation and the loss from the *prediction* one; a rejected row contributes
zero to a mean whose denominator still counts it; and the gate is on the target
probability rather than on anything the prediction side knows.
"""

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
from xty2.objectives import PseudoLabelTreatmentNLL

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)

TARGET = Realisation(view="weak_x")
PREDICTION = Realisation(view="strong_x")


def _state(target_logits: torch.Tensor, prediction_logits: torch.Tensor) -> State:
    return State(
        {
            TARGET: {Port.T_GIVEN_X: CategoricalTreatment(target_logits)},
            PREDICTION: {Port.T_GIVEN_X: CategoricalTreatment(prediction_logits)},
        }
    )


def _objective(**overrides: object) -> PseudoLabelTreatmentNLL:
    defaults: dict[str, object] = {
        "port": Port.T_GIVEN_X,
        "target": TARGET,
        "prediction": PREDICTION,
        "threshold": 0.95,
        "sharpening": "hard",
        "stop_grad": "target",
        "rows": "all",
    }
    return PseudoLabelTreatmentNLL(**(defaults | overrides))  # type: ignore[arg-type]


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Sharp targets in the first rows, flat ones in the last.

    Row `i` gets logit scale `i`, so `max(q)` increases down the batch and a
    threshold in between splits it — which is what makes the gate observable
    rather than all-or-nothing.
    """
    directions = torch.tensor([1.0, -1.0, 0.5]).expand(BATCH_SIZE, NUM_TREATMENTS)
    scales = torch.arange(BATCH_SIZE, dtype=torch.float32).flip(0)[:, None]
    target = directions * scales
    prediction = torch.linspace(-1.0, 1.0, BATCH_SIZE * NUM_TREATMENTS).reshape(
        BATCH_SIZE, NUM_TREATMENTS
    )
    return target, prediction


def _expected(
    target_logits: torch.Tensor, prediction_logits: torch.Tensor, threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = target_logits.softmax(dim=-1)
    confidence, labels = probs.max(dim=-1)
    accepted = confidence >= threshold
    log_probs = prediction_logits.log_softmax(dim=-1)
    per_row = -log_probs.gather(1, labels[:, None]).squeeze(1) * accepted.float()
    return per_row, accepted


def _context() -> TrainContext:
    return TrainContext(global_step=0, schema=make_schema())


def test_the_value_is_the_masked_cross_entropy_over_every_eligible_row() -> None:
    target, prediction = _inputs()
    rows = torch.arange(BATCH_SIZE)
    term = _objective().compute(
        _state(target, prediction), make_batch(), rows, _context()
    )
    per_row, _ = _expected(target, prediction, 0.95)
    assert torch.allclose(term.value, per_row.mean())
    assert term.n == BATCH_SIZE


def test_the_denominator_counts_the_rows_the_gate_rejected() -> None:
    """Eq. (4) divides by `mu*B`, not by the retained rows.

    The distinction is invisible when everything clears the gate, so this
    compares against the mean over retained rows and requires them to differ —
    a version that averaged only the accepted rows would pass every other
    assertion in this file.
    """
    target, prediction = _inputs()
    rows = torch.arange(BATCH_SIZE)
    per_row, accepted = _expected(target, prediction, 0.95)
    assert 0 < int(accepted.sum()) < BATCH_SIZE, "fixture must split the batch"
    term = _objective().compute(
        _state(target, prediction), make_batch(), rows, _context()
    )
    assert torch.allclose(term.value, per_row.mean())
    assert not torch.allclose(term.value, per_row[accepted].mean())


def test_coverage_is_the_paper_mask_rate_over_the_eligible_rows() -> None:
    target, prediction = _inputs()
    rows = torch.tensor([0, 1, BATCH_SIZE - 2, BATCH_SIZE - 1])
    term = _objective().compute(
        _state(target, prediction), make_batch(), rows, _context()
    )
    _, accepted = _expected(target, prediction, 0.95)
    expected = float(accepted[rows].float().mean())
    assert term.diagnostics["coverage"] == pytest.approx(expected)
    probs = target.softmax(dim=-1).max(dim=-1).values
    retained = probs[rows][accepted[rows]]
    assert term.diagnostics["accepted_confidence"] == pytest.approx(
        float(retained.mean())
    )


def test_a_closed_gate_contributes_exactly_zero_but_still_sees_its_rows() -> None:
    target, prediction = _inputs()
    term = _objective(threshold=1.0).compute(
        _state(target, prediction), make_batch(), torch.arange(BATCH_SIZE), _context()
    )
    assert float(term.value) == 0.0
    assert term.n == BATCH_SIZE
    assert term.diagnostics["coverage"] == 0.0
    assert term.diagnostics["accepted_confidence"] == 0.0


def test_an_open_gate_retains_every_row() -> None:
    target, prediction = _inputs()
    term = _objective(threshold=0.0).compute(
        _state(target, prediction), make_batch(), torch.arange(BATCH_SIZE), _context()
    )
    per_row, _ = _expected(target, prediction, 0.0)
    assert torch.allclose(term.value, per_row.mean())
    assert term.diagnostics["coverage"] == 1.0


def test_the_label_comes_from_the_target_and_the_loss_from_the_prediction() -> None:
    """Swapping the two realisations must change the number.

    With the same port on both sides, an implementation that read the label off
    the prediction — or charged the loss against the target — is a transposition
    no shape check would catch.
    """
    target, prediction = _inputs()
    rows = torch.arange(BATCH_SIZE)
    straight = _objective().compute(
        _state(target, prediction), make_batch(), rows, _context()
    )
    swapped = _objective(target=PREDICTION, prediction=TARGET).compute(
        _state(target, prediction), make_batch(), rows, _context()
    )
    assert not torch.allclose(straight.value, swapped.value)


def test_no_gradient_reaches_the_target_side() -> None:
    target, prediction = _inputs()
    target.requires_grad_()
    prediction.requires_grad_()
    objective = _objective()
    term = objective.compute(
        _state(target, prediction), make_batch(), torch.arange(BATCH_SIZE), _context()
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    assert target.grad is None
    assert prediction.grad is not None
    assert objective.detaches == frozenset({(Port.T_GIVEN_X, TARGET)})
    assert objective.requires == frozenset(
        {(Port.T_GIVEN_X, TARGET), (Port.T_GIVEN_X, PREDICTION)}
    )


def test_no_rows_returns_the_framework_zero_term_without_diagnostics() -> None:
    target, prediction = _inputs()
    term = _objective().compute(
        _state(target, prediction),
        make_batch(),
        torch.zeros(0, dtype=torch.long),
        _context(),
    )
    assert term.n == 0
    assert float(term.value) == 0.0
    assert dict(term.diagnostics) == {}


def test_the_three_card_governed_fields_have_no_default() -> None:
    for missing in ("threshold", "sharpening", "stop_grad"):
        arguments: dict[str, object] = {
            "port": Port.T_GIVEN_X,
            "target": TARGET,
            "prediction": PREDICTION,
            "threshold": 0.95,
            "sharpening": "hard",
            "stop_grad": "target",
        }
        del arguments[missing]
        with pytest.raises(CardKeyError, match="no usable default"):
            PseudoLabelTreatmentNLL(**arguments)  # type: ignore[arg-type]


def test_the_plan_details_state_the_gate_and_its_denominator() -> None:
    assert _objective().plan_details() == (
        "label = arg max of the target realisation",
        "gate = max prob >= threshold",
        "denominator = every eligible row; rejected rows contribute 0",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target": PREDICTION}, "with itself"),
        ({"port": Port.Y_GIVEN_XT}, "labels a treatment"),
        ({"threshold": 1.5}, "must be in"),
        ({"threshold": -0.1}, "must be in"),
        ({"sharpening": "temperature"}, "must be 'hard'"),
        ({"stop_grad": "none"}, "must be 'target'"),
        ({"rows": "confident"}, "unknown row population"),
    ],
)
def test_rejected_configurations(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(LossError, match=message):
        _objective(**overrides)


def test_a_row_exactly_at_the_threshold_is_retained() -> None:
    """`>=`, not `>` — card §7's first row, pinned.

    Eq. (4) and eq. (6) gate on `max(q) >= tau` while algorithm 1 writes `>`,
    and the card records `>=` as the reading the reference implementation
    settles. Neither boundary fixture pins it: at `tau=0` both comparisons
    accept every row, and at `tau=1` no float32 softmax reaches exactly one.
    So the threshold here is *read off* the fixture — the row's own confidence
    — which makes equality exact and the two comparisons disagree by one row.
    """
    target, prediction = _inputs()
    probs = target.softmax(dim=-1)
    confidence = probs.max(dim=-1).values
    row = int(confidence.argmin())
    threshold = float(confidence[row])

    term = _objective(threshold=threshold).compute(
        _state(target, prediction),
        make_batch(),
        torch.arange(BATCH_SIZE),
        _context(),
    )
    accepted = confidence >= threshold
    strictly = confidence > threshold
    assert int(accepted.sum()) == int(strictly.sum()) + 1, (
        "fixture must isolate one row"
    )
    assert term.diagnostics["coverage"] == pytest.approx(float(accepted.float().mean()))
    per_row, _ = _expected(target, prediction, threshold)
    assert torch.allclose(term.value, per_row.mean())
