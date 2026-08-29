"""Tier 0 — FreeMatch's self-adaptive threshold and fairness (`freematch.md` §3).

Four properties of eqs. (5)-(11) are the ones a wrong implementation gets wrong
quietly, and each is asserted against the arithmetic rather than against a
trajectory:

* the three EMAs are the paper's exactly, initialised at `1/C` and folding in no
  batch at `t = 0` (eqs. 5, 6, 10);
* `tau_t(c)` is `MaxNorm(p~_t) * tau_t`, so the most-predicted class sits at
  `tau_t` exactly and every other strictly below it (eq. 7);
* with the state pinned at a uniform `p~` and a chosen `tau_t`, eq. (8) *is*
  FixMatch's eq. (4) at that threshold — the strongest available statement that
  SAT adds a threshold rule and changes nothing else;
* the shared update is idempotent within a step, so the two objectives read the
  same statistics whichever order the mixer reaches them in. That is the
  property `freematch.md` §5.1 checked the sibling-state shape against, and
  without it the loss would depend on the order two lines appear in a recipe.

The fairness term gets one assertion the others do not: its **sign**. Card
deviation 7 implements eq. (11) without its leading minus, on the argument that
the minus makes the term drive the batch marginal away from the model's own.
`test_the_fairness_term_is_minimised_when_the_two_marginals_agree` is that
argument as a property rather than as prose.
"""

from __future__ import annotations

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
from xty2.objectives import (
    PseudoLabelTreatmentNLL,
    SelfAdaptiveFairness,
    SelfAdaptiveThreshold,
    SelfAdaptiveThresholds,
    SelfAdaptiveThresholdTreatmentNLL,
)

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)

TARGET = Realisation(view="weak_x")
PREDICTION = Realisation(view="strong_x")
DECAY = 0.9
"""Fast enough that six steps move the EMAs visibly; the paper's is 0.999."""

SAT_NAME = "self_adaptive_threshold_treatment_nll"
FAIRNESS_NAME = "self_adaptive_fairness"


def _policy(decay: float = DECAY) -> SelfAdaptiveThreshold:
    return SelfAdaptiveThreshold(decay=decay)


def _gate_objective(**overrides: object) -> SelfAdaptiveThresholdTreatmentNLL:
    defaults: dict[str, object] = {
        "port": Port.T_GIVEN_X,
        "target": TARGET,
        "prediction": PREDICTION,
        "num_treatments": NUM_TREATMENTS,
        "threshold": _policy(),
        "sharpening": "hard",
        "stop_grad": "target",
        "rows": "all",
    }
    return SelfAdaptiveThresholdTreatmentNLL(**(defaults | overrides))  # type: ignore[arg-type]


def _fairness_objective(**overrides: object) -> SelfAdaptiveFairness:
    defaults: dict[str, object] = {
        "port": Port.T_GIVEN_X,
        "target": TARGET,
        "prediction": PREDICTION,
        "statistics": SAT_NAME,
        "rows": "all",
    }
    return SelfAdaptiveFairness(**(defaults | overrides))  # type: ignore[arg-type]


def _state(target_logits: torch.Tensor, prediction_logits: torch.Tensor) -> State:
    return State(
        {
            TARGET: {Port.T_GIVEN_X: CategoricalTreatment(target_logits)},
            PREDICTION: {Port.T_GIVEN_X: CategoricalTreatment(prediction_logits)},
        }
    )


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """`max(q)` falls down the batch, so any threshold in between splits it."""
    directions = torch.tensor([1.0, -1.0, 0.5]).expand(BATCH_SIZE, NUM_TREATMENTS)
    scales = torch.arange(BATCH_SIZE, dtype=torch.float32).flip(0)[:, None]
    prediction = torch.linspace(-1.0, 1.0, BATCH_SIZE * NUM_TREATMENTS).reshape(
        BATCH_SIZE, NUM_TREATMENTS
    )
    return directions * scales, prediction


def _context(states: dict[str, object], step: int = 1) -> TrainContext:
    return TrainContext(
        global_step=step,
        schema=make_schema(),
        stage="joint_fit",
        objective_states=states,
    )


def _rows(batch_size: int = BATCH_SIZE) -> torch.Tensor:
    return torch.arange(batch_size, dtype=torch.long)


def _fresh(policy: SelfAdaptiveThreshold | None = None) -> SelfAdaptiveThresholds:
    return SelfAdaptiveThresholds(NUM_TREATMENTS, policy or _policy())


# ---------------------------------------------------------------------------
# The three EMAs (eqs. 5, 6, 10)
# ---------------------------------------------------------------------------


def test_every_statistic_starts_uniform_and_the_gate_starts_at_one_over_k() -> None:
    """Eqs. (5) and (6) at `t = 0`, and card §7's choice for eq. (10).

    `freematch.md` §2's first limitation is a consequence of exactly this: with
    `K = 2` a softmax has `max(q) >= 0.5` on every row, so a gate at `1/K`
    admits the whole batch and FreeMatch opens ungated where FixMatch opens
    closed.
    """
    state = _fresh()
    uniform = 1.0 / NUM_TREATMENTS
    assert state.tau == pytest.approx(uniform)
    flat = torch.full((NUM_TREATMENTS,), uniform).double()
    assert torch.allclose(state.marginal, flat)
    assert torch.allclose(state.histogram, flat)
    assert torch.allclose(state.thresholds(), flat)
    assert state.last_observed_step is None


def test_step_zero_records_itself_and_folds_in_no_batch() -> None:
    """Eq. (5)'s `t = 0` case is a value, not the first term of the recursion."""
    state = _fresh()
    probs = torch.softmax(_inputs()[0], dim=-1)
    state.observe(0, probs)
    assert state.tau == pytest.approx(1.0 / NUM_TREATMENTS)
    assert state.last_observed_step == 0


def test_the_emas_are_the_papers_arithmetic() -> None:
    """Eqs. (5), (6) and (10) against the same numbers computed by hand."""
    state = _fresh()
    probs = torch.softmax(_inputs()[0], dim=-1).double()
    state.observe(1, probs)

    uniform = 1.0 / NUM_TREATMENTS
    confidence = float(probs.max(dim=-1).values.mean())
    expected_tau = DECAY * uniform + (1 - DECAY) * confidence
    expected_marginal = DECAY * uniform + (1 - DECAY) * probs.mean(dim=0)
    share = torch.bincount(probs.argmax(dim=-1), minlength=NUM_TREATMENTS).double()
    share /= float(probs.shape[0])
    expected_histogram = DECAY * uniform + (1 - DECAY) * share

    assert state.tau == pytest.approx(expected_tau)
    assert torch.allclose(state.marginal, expected_marginal)
    assert torch.allclose(state.histogram, expected_histogram)


def test_the_update_is_idempotent_within_one_step() -> None:
    """`freematch.md` §5.1's shape obligation, as arithmetic.

    Two objectives share this state and both call `observe`. If the second call
    folded the batch in again, the statistics — and therefore the gate — would
    depend on how many terms happened to read them, which is a dependency on
    the order and number of lines in a recipe.
    """
    state = _fresh()
    probs = torch.softmax(_inputs()[0], dim=-1)
    state.observe(1, probs)
    once = (state.tau, state.marginal.clone(), state.histogram.clone())
    state.observe(1, probs)
    state.observe(1, torch.softmax(_inputs()[1], dim=-1))
    assert state.tau == once[0]
    assert torch.equal(state.marginal, once[1])
    assert torch.equal(state.histogram, once[2])


def test_two_row_counts_at_one_step_are_refused_rather_than_ignored() -> None:
    """The other half of the idempotence rule, and the reason it needs one.

    Idempotence makes declaration order inert only if the two objectives are
    entitled to the same rows: otherwise whichever the mixer reached first would
    decide which set eqs. (5), (6) and (10) averaged, and no reader of either
    declaration could see which. A repeat at one step with a different row count
    is therefore an error naming both counts.
    """
    state = _fresh()
    probs = torch.softmax(_inputs()[0], dim=-1)
    state.observe(1, probs)
    with pytest.raises(LossError, match="different row counts"):
        state.observe(1, probs[:2])
    # The same count is still the ordinary idempotent case.
    state.observe(1, probs)


def test_max_norm_puts_the_leading_class_at_tau_and_the_rest_below_it() -> None:
    """Eq. (7), and the difference from FlexMatch `freematch.md` §2 records.

    FlexMatch's `beta` normalises a count by its maximum, and at `K = 2` on a
    balanced fixture both classes reach 1 together, collapsing the per-class
    threshold to `tau`. `MaxNorm(p~)` collapses only when `p~` is *exactly*
    uniform, which it is not after any step.
    """
    state = _fresh()
    skewed = torch.zeros(16, NUM_TREATMENTS)
    skewed[:12, 0] = 4.0
    skewed[12:, 1] = 4.0
    state.observe(1, torch.softmax(skewed, dim=-1))
    thresholds = state.thresholds()
    assert float(thresholds.max()) == pytest.approx(state.tau)
    assert float(thresholds.min()) < state.tau
    leading = int(state.marginal.argmax())
    assert int(thresholds.argmax()) == leading


def test_an_empty_batch_leaves_the_state_free_to_be_written_later() -> None:
    """A term with no eligible rows must not consume the step's one update."""
    state = _fresh()
    state.observe(1, torch.zeros(0, NUM_TREATMENTS))
    # Read into locals: the property is `int | None`, and a type checker that
    # narrowed it on the first assertion would treat the second as dead.
    after_empty = state.last_observed_step
    state.observe(1, torch.softmax(_inputs()[0], dim=-1))
    after_rows = state.last_observed_step
    assert after_empty is None
    assert after_rows == 1
    assert state.tau != pytest.approx(1.0 / NUM_TREATMENTS)


# ---------------------------------------------------------------------------
# Eq. (8)
# ---------------------------------------------------------------------------


def test_the_gate_is_fixmatchs_when_the_marginal_is_uniform() -> None:
    """Eq. (8) at a uniform `p~` is eq. (4) at `tau_t`.

    `MaxNorm` of a uniform vector is all ones, so `tau_t(c) = tau_t` for every
    `c` and eq. (8) reduces to FixMatch's constant gate. Asserting the two
    objectives' values are equal is the strongest available statement that SAT
    is a threshold rule and changes nothing else about the term.

    The state is put there by observing a *symmetric* batch — one confident row
    per class — which leaves eq. (6) at `1/C` exactly while eq. (5) moves. That
    is a real state a run could reach, not a hand-set one.
    """
    target_logits, prediction_logits = _inputs()
    state = _state(target_logits, prediction_logits)
    batch = make_batch()
    rows = _rows()

    pinned = _fresh(_policy(decay=0.5))
    # One confident row per class, cycled, plus one uniform row to make the
    # count `BATCH_SIZE`: the column means are exactly `1/C`, so eq. (6) does
    # not move. The row count matches what `compute` will fold in below, which
    # `observe` requires of a repeat at one step.
    symmetric = torch.zeros(BATCH_SIZE, NUM_TREATMENTS)
    cycled = torch.arange(BATCH_SIZE - 1)
    symmetric[cycled, cycled % NUM_TREATMENTS] = 3.0
    pinned.observe(1, torch.softmax(symmetric, dim=-1))
    thresholds = pinned.thresholds()
    assert bool((thresholds == thresholds[0]).all())
    assert float(thresholds[0]) == pytest.approx(pinned.tau)

    # `compute` calls `observe` for this step and finds it already laid down,
    # so the gate below is the threshold asserted above.
    ours = _gate_objective(threshold=_policy(decay=0.5)).compute(
        state, batch, rows, _context({SAT_NAME: pinned}, step=1)
    )
    theirs = PseudoLabelTreatmentNLL(
        port=Port.T_GIVEN_X,
        target=TARGET,
        prediction=PREDICTION,
        threshold=float(thresholds[0]),
        sharpening="hard",
        stop_grad="target",
        rows="all",
    ).compute(state, batch, rows, _context({}, step=1))
    assert float(ours.value) == pytest.approx(float(theirs.value))
    assert ours.diagnostics["coverage"] == theirs.diagnostics["coverage"]
    assert 0.0 < ours.diagnostics["coverage"] < 1.0, (
        "the fixture must straddle the threshold or the equality is vacuous"
    )


def test_the_statistics_fold_in_this_batch_before_this_batch_is_gated() -> None:
    """Algorithm 1 lines 3-5 before line 9, which is why the term is coupled.

    The threshold the gate uses is the *updated* one, so a run of `compute` on
    the same state moves the number a second run would gate on. FlexMatch's
    objective is the counter-example: its thresholds come from marks laid down
    by earlier steps and this step's marks are written afterwards.
    """
    target_logits, prediction_logits = _inputs()
    state = _state(target_logits, prediction_logits)
    shared = _fresh()
    objective = _gate_objective()
    before = shared.tau
    objective.compute(state, make_batch(), _rows(), _context({SAT_NAME: shared}, 1))
    assert shared.last_observed_step == 1
    assert shared.tau != before
    assert objective.batch_coupled is True


def test_the_gate_reports_the_trajectory_numbers_the_card_reads() -> None:
    """`tau_global` and the threshold pair are the mechanism guardrails of §6."""
    target_logits, prediction_logits = _inputs()
    term = _gate_objective().compute(
        _state(target_logits, prediction_logits),
        make_batch(),
        _rows(),
        _context({SAT_NAME: _fresh()}, 1),
    )
    assert set(term.diagnostics) == {
        "coverage",
        "accepted_confidence",
        "tau_global",
        "threshold_min",
        "threshold_max",
    }
    assert term.n == BATCH_SIZE


def test_the_state_needs_no_training_population() -> None:
    """`flexmatch.md` §5.1's prediction about the shape, checked directly.

    That card took `initial_state(population: TrainingPopulation | None)` rather
    than a required population *because* FreeMatch would not need one. This is
    the assertion that says the reason was real.
    """
    built = _gate_objective().initial_state(None)
    assert isinstance(built, SelfAdaptiveThresholds)
    assert built.classes == NUM_TREATMENTS


# ---------------------------------------------------------------------------
# Eq. (11)
# ---------------------------------------------------------------------------


def _fairness_value(
    strong: torch.Tensor, state: SelfAdaptiveThresholds, *, retained: torch.Tensor
) -> float:
    """Eq. (11) computed here, from the definitions, at deviation 7's sign."""
    counts = torch.bincount(
        strong.argmax(dim=-1)[retained], minlength=state.classes
    ).double()
    support = counts > 0
    probability = (strong.double() * retained.double()[:, None]).sum(dim=0)
    reference = state.marginal / state.histogram
    a = reference[support] / reference[support].sum()
    ratio = probability[support] / counts[support]
    b = ratio / ratio.sum()
    return float(-(a * b.log()).sum())


def test_the_fairness_term_is_the_papers_arithmetic_without_its_minus() -> None:
    """Eqs. (9) and (11), against the same numbers computed from the definitions."""
    target_logits, prediction_logits = _inputs()
    shared = _fresh()
    shared.observe(1, torch.softmax(target_logits, dim=-1))

    term = _fairness_objective().compute(
        _state(target_logits, prediction_logits),
        make_batch(),
        _rows(),
        _context({SAT_NAME: shared}, 2),
    )
    target = torch.softmax(target_logits, dim=-1)
    confidence, labels = target.max(dim=-1)
    thresholds = shared.thresholds().float()
    retained = confidence > thresholds.index_select(0, labels)
    expected = _fairness_value(
        torch.softmax(prediction_logits, dim=-1), shared, retained=retained
    )
    assert float(term.value) == pytest.approx(expected, rel=1e-5)
    assert term.n == BATCH_SIZE


def test_the_fairness_term_is_minimised_when_the_two_marginals_agree() -> None:
    """Card deviation 7, as a property rather than as prose.

    `H(A, B)` has its minimum over the simplex at `B = A`; `-H(A, B)` — eq. (11)
    exactly as printed — has a *maximum* there, so minimising it drives `B` to a
    corner. The paper says in four places that the term is for diverse
    predictions, so this is the sign that serves the stated purpose, and §6.1's
    `literal` arm is where the other one gets measured rather than argued.
    """
    a = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)

    def cross_entropy(b: torch.Tensor) -> float:
        return float(-(a * b.log()).sum())

    at_a = cross_entropy(a)
    for perturbation in (0.05, 0.1, 0.2):
        skewed = a + torch.tensor([perturbation, -perturbation, 0.0])
        assert cross_entropy(skewed) > at_a
        corner = torch.tensor([1.0 - 2 * perturbation, perturbation, perturbation])
        assert cross_entropy(corner) > at_a
    assert at_a == pytest.approx(float(-(a * a.log()).sum()))


def test_a_class_with_an_empty_histogram_bin_leaves_both_sum_norms() -> None:
    """Card §7's convention, and the reason it is not "read the ratio as zero".

    `\\bar h(c) = 0` makes eq. (9)'s ratio undefined. Excluding the class keeps
    the term finite and differentiable; treating `1/\\bar h(c)` as zero would
    put `-A(c) log 0` into the sum, an arbitrarily large penalty carrying no
    gradient at all.
    """
    shared = _fresh()
    shared.observe(1, torch.softmax(_inputs()[0], dim=-1))
    # Every strong-view arg max on class 0, so classes 1 and 2 have empty bins.
    prediction_logits = torch.zeros(BATCH_SIZE, NUM_TREATMENTS)
    prediction_logits[:, 0] = 6.0
    target_logits = torch.full((BATCH_SIZE, NUM_TREATMENTS), 0.0)
    target_logits[:, 0] = 6.0

    term = _fairness_objective().compute(
        _state(target_logits, prediction_logits),
        make_batch(),
        _rows(),
        _context({SAT_NAME: shared}, 2),
    )
    assert term.diagnostics["fairness_support"] == 1.0
    assert float(term.value) == 0.0
    assert math.isfinite(float(term.value))
    assert term.n == BATCH_SIZE


def test_the_two_terms_read_one_state_in_either_order() -> None:
    """The sibling read, and the property that makes declaration order inert."""
    target_logits, prediction_logits = _inputs()
    state = _state(target_logits, prediction_logits)
    batch, rows = make_batch(), _rows()
    gate, fairness = _gate_objective(), _fairness_objective()

    first = _fresh()
    gate_first = gate.compute(state, batch, rows, _context({SAT_NAME: first}, 1))
    fairness_second = fairness.compute(
        state, batch, rows, _context({SAT_NAME: first}, 1)
    )

    second = _fresh()
    fairness_first = fairness.compute(
        state, batch, rows, _context({SAT_NAME: second}, 1)
    )
    gate_second = gate.compute(state, batch, rows, _context({SAT_NAME: second}, 1))

    assert float(gate_first.value) == float(gate_second.value)
    assert float(fairness_first.value) == float(fairness_second.value)
    assert first.tau == second.tau


def test_the_fairness_term_names_the_objective_it_reads() -> None:
    """A stage without that objective is an error naming both, not a KeyError."""
    target_logits, prediction_logits = _inputs()
    with pytest.raises(LossError, match="reads the self-adaptive threshold state"):
        _fairness_objective(statistics="somebody_else").compute(
            _state(target_logits, prediction_logits),
            make_batch(),
            _rows(),
            _context({SAT_NAME: _fresh()}, 1),
        )


def test_both_terms_declare_the_weak_view_detached_and_the_batch_coupled() -> None:
    """Neither side of eq. (11) trains through `q_b`; both read the batch."""
    for objective in (_gate_objective(), _fairness_objective()):
        assert objective.detaches == frozenset({(Port.T_GIVEN_X, TARGET)})
        assert objective.requires == frozenset(
            {(Port.T_GIVEN_X, TARGET), (Port.T_GIVEN_X, PREDICTION)}
        )
        assert objective.batch_coupled is True


def _spread_predictions() -> torch.Tensor:
    """Strong-view logits whose arg max covers every class.

    `_inputs()`'s prediction rows increase along the class axis, so every row
    picks the last class and eq. (9)'s histogram has one non-empty bin — which
    card §7's convention correctly makes inert. A term under test for its
    gradient needs a support of at least two.
    """
    logits = torch.zeros(BATCH_SIZE, NUM_TREATMENTS)
    logits[torch.arange(BATCH_SIZE), torch.arange(BATCH_SIZE) % NUM_TREATMENTS] = 3.0
    return logits


def test_the_fairness_term_trains_through_the_strong_view() -> None:
    """`p_bar` carries the gradient, so the term is not a constant."""
    target_logits = _inputs()[0]
    prediction_logits = _spread_predictions().requires_grad_(True)
    shared = _fresh()
    shared.observe(1, torch.softmax(target_logits, dim=-1))
    term = _fairness_objective().compute(
        _state(target_logits, prediction_logits),
        make_batch(),
        _rows(),
        _context({SAT_NAME: shared}, 2),
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    assert prediction_logits.grad is not None
    assert float(prediction_logits.grad.abs().sum()) > 0.0


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def test_the_policy_prints_the_form_the_card_writes() -> None:
    assert repr(SelfAdaptiveThreshold(decay=0.999)) == "self_adaptive(decay=0.999)"


@pytest.mark.parametrize("decay", [0.0, 1.0, -0.5, 1.5])
def test_a_decay_outside_the_open_unit_interval_is_rejected(decay: float) -> None:
    with pytest.raises(LossError, match="lambda in "):
        SelfAdaptiveThreshold(decay=decay)


def test_the_three_card_keys_are_the_ones_section_four_states() -> None:
    assert sorted(SelfAdaptiveThresholdTreatmentNLL.CARD_KEYS.values()) == [
        "gradients.detached_targets",
        "losses.confidence_threshold",
        "losses.sharpening",
    ]


def test_the_gate_rule_has_no_default() -> None:
    """`DESIGN.md` §9.1: a paper-governed field carries the sentinel."""
    with pytest.raises(CardKeyError, match=r"losses\.confidence_threshold"):
        SelfAdaptiveThresholdTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=TARGET,
            prediction=PREDICTION,
            num_treatments=NUM_TREATMENTS,
            sharpening="hard",
            stop_grad="target",
        )


def test_a_bare_float_threshold_is_refused_with_the_objective_that_takes_one() -> None:
    with pytest.raises(LossError, match="PseudoLabelTreatmentNLL"):
        _gate_objective(threshold=0.95)


def test_one_realisation_on_both_sides_is_refused() -> None:
    with pytest.raises(LossError, match="on both sides"):
        _gate_objective(prediction=TARGET)
    with pytest.raises(LossError, match="on both sides"):
        _fairness_objective(prediction=TARGET)


def test_a_non_treatment_port_is_refused_on_both_terms() -> None:
    for build in (_gate_objective, _fairness_objective):
        with pytest.raises(LossError, match="per treatment"):
            build(port=Port.X_REPR)


def test_the_fairness_term_will_not_be_built_without_the_objective_it_reads() -> None:
    """A wiring reference with no default, rather than a `REQUIRED` sentinel.

    `DESIGN.md` §9.1's sentinel is for a paper-governed hyperparameter, and
    which sibling holds the statistics is not one. A field with no default at
    all is the stronger guard: the objective cannot be constructed without it.
    """
    with pytest.raises(TypeError, match="statistics"):
        SelfAdaptiveFairness(  # type: ignore[call-arg]
            port=Port.T_GIVEN_X, target=TARGET, prediction=PREDICTION
        )


def test_a_declared_cardinality_that_disagrees_with_the_schema_is_refused() -> None:
    """The state is `[C]` wide and is built before the first batch (§3.1)."""
    target_logits, prediction_logits = _inputs()
    objective = _gate_objective(num_treatments=NUM_TREATMENTS)
    context = TrainContext(
        global_step=1,
        schema=make_schema(treatment_cardinality=NUM_TREATMENTS),
        stage="joint_fit",
        objective_states={SAT_NAME: _fresh()},
    )
    # Sanity: the matched case computes.
    objective.compute(
        _state(target_logits, prediction_logits), make_batch(), _rows(), context
    )
    mismatched = _gate_objective(num_treatments=NUM_TREATMENTS + 1)
    with pytest.raises(LossError, match="num_treatments"):
        mismatched.compute(
            _state(target_logits, prediction_logits), make_batch(), _rows(), context
        )
