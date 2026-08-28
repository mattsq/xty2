"""Tier 0 — Curriculum Pseudo Labeling's state and gate (`flexmatch.md` §3).

Three properties of eq. (8) are the ones a wrong implementation gets wrong
quietly, and each is asserted against the arithmetic rather than against a
trajectory:

* `T(c)` is read from marks laid down by **earlier** steps, and this step's
  marks are written afterwards (Alg. 1 lines 4-12, then 13-17, then 18);
* the mark is set at the fixed `tau` and the loss is gated at `T(c)`, which are
  two different numbers on the same batch;
* with every class fully learned the term is FixMatch's eq. (4) exactly, which
  is the strongest available statement that CPL adds a threshold and changes
  nothing else.
"""

import math

import pytest
import torch
from xty2.core import (
    CardKeyError,
    CategoricalTreatment,
    CompileError,
    LossError,
    LossTerm,
    Port,
    PseudoLabelAction,
    Realisation,
    State,
    TrainContext,
    Weighted,
)
from xty2.objectives import (
    UNUSED,
    CurriculumPseudoLabelTreatmentNLL,
    CurriculumStatus,
    CurriculumThreshold,
    PseudoLabelTreatmentNLL,
)

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
    stage,
)

TARGET = Realisation(view="weak_x")
PREDICTION = Realisation(view="strong_x")
TAU = 0.5
"""Low enough that the fixture below splits the batch at both gates."""

POPULATION = torch.arange(100, 100 + 4 * BATCH_SIZE, dtype=torch.long)
"""`N = 4B` rows, of which `make_batch`'s ids are the first `B`."""


def _policy(**overrides: object) -> CurriculumThreshold:
    defaults: dict[str, object] = {
        "tau": TAU,
        "warm_up": True,
        "mapping": "convex",
    }
    return CurriculumThreshold(**(defaults | overrides))  # type: ignore[arg-type]


def _objective(**overrides: object) -> CurriculumPseudoLabelTreatmentNLL:
    defaults: dict[str, object] = {
        "port": Port.T_GIVEN_X,
        "target": TARGET,
        "prediction": PREDICTION,
        "threshold": _policy(),
        "sharpening": "hard",
        "stop_grad": "target",
        "rows": "all",
    }
    return CurriculumPseudoLabelTreatmentNLL(**(defaults | overrides))  # type: ignore[arg-type]


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


def _context(status: CurriculumStatus, name: str) -> TrainContext:
    return TrainContext(
        global_step=0, schema=make_schema(), objective_states={name: status}
    )


def _fresh(policy: CurriculumThreshold | None = None) -> CurriculumStatus:
    return CurriculumStatus(POPULATION, policy or _policy())


def _fully_learned(policy: CurriculumThreshold) -> CurriculumStatus:
    """Every row marked, and every class equally, so `beta(c) = 1` for all `c`.

    `sigma` is then flat, `max_c sigma` is its own value and the unused count is
    zero, so eq. (11) and eq. (6) agree and `M(1) = 1` puts every threshold at
    `tau`.
    """
    status = CurriculumStatus(POPULATION, policy)
    labels = torch.arange(POPULATION.numel()) % NUM_TREATMENTS
    status.mark(POPULATION, labels, torch.ones(POPULATION.numel()))
    return status


# ---------------------------------------------------------------------------
# The threshold rule
# ---------------------------------------------------------------------------


def test_every_threshold_is_zero_before_any_row_is_marked() -> None:
    """Algorithm 1 at `t = 0`, and `flexmatch.md` §2's first limitation.

    `sigma = 0` makes eq. (11)'s denominator the unused count, `beta = 0` and
    `M(0) = 0`. The gate is therefore *open* at initialisation rather than
    closed, which is the opposite of FixMatch and the single most surprising
    consequence of the published procedure.
    """
    status = _fresh()
    assert status.size == POPULATION.numel()
    assert status.unused() == POPULATION.numel()
    assert torch.equal(
        status.learning_effect(NUM_TREATMENTS), torch.zeros(NUM_TREATMENTS).long()
    )
    assert torch.equal(
        status.thresholds(NUM_TREATMENTS),
        torch.zeros(NUM_TREATMENTS, dtype=torch.float64),
    )


def test_the_warm_up_denominator_is_the_unused_count_while_it_dominates() -> None:
    """Eq. (11), and Algorithm 1 lines 6-9 as one expression.

    The branch is redundant — eq. (11)'s denominator *is* `max_c sigma` once
    that dominates — so this asserts the identity at the crossing rather than
    trusting it: one mark against `N - 1` unused divides by `N - 1`.
    """
    status = _fresh()
    status.mark(POPULATION[:1], torch.tensor([1]), torch.ones(1))
    beta = 1.0 / (POPULATION.numel() - 1)
    expected = beta / (2.0 - beta) * TAU
    thresholds = status.thresholds(NUM_TREATMENTS)
    assert thresholds[1] == pytest.approx(expected)
    assert float(thresholds[0]) == 0.0
    assert float(thresholds[2]) == 0.0


def test_without_the_warm_up_one_mark_puts_its_class_at_tau() -> None:
    """§3.2's ablation, and why the paper corrected eq. (6).

    Divide by `max_c sigma` alone and a single accepted row makes its own class
    fully learned, which is exactly the failure the warm-up exists to prevent.
    """
    status = _fresh(_policy(warm_up=False))
    status.mark(POPULATION[:1], torch.tensor([1]), torch.ones(1))
    thresholds = status.thresholds(NUM_TREATMENTS)
    assert float(thresholds[1]) == pytest.approx(TAU)
    assert float(thresholds[0]) == 0.0


def test_the_convex_mapping_is_below_the_identity_everywhere_between_0_and_1() -> None:
    """§3.3's whole argument for `x / (2 - x)`, as an inequality.

    "A monotone increasing convex function lets the thresholds grow slowly when
    `beta_t(c)` is small, and become more sensitive as `beta_t(c)` gets larger."
    Eq. (7) is eq. (12) at `M = id`, so the two agree at the ends and the convex
    one is strictly lower in between.
    """
    convex = _policy(mapping="convex")
    identity = _policy(mapping="identity")
    beta = torch.linspace(0.0, 1.0, 11, dtype=torch.float64)
    mapped, plain = convex.map(beta), identity.map(beta)
    assert float(mapped[0]) == float(plain[0]) == 0.0
    assert float(mapped[-1]) == float(plain[-1]) == 1.0
    assert bool((mapped[1:-1] < plain[1:-1]).all())
    # Monotone, and never above 1 — §3.3's two stated requirements on `M`.
    assert bool((mapped.diff() > 0).all())
    assert float(mapped.max()) <= 1.0


def test_a_fully_learned_class_reaches_tau_and_never_exceeds_it() -> None:
    """§3.3: `M` has "a range from 0 to 1 so that the flexible thresholds range
    from 0 to `tau`"."""
    thresholds = _fully_learned(_policy()).thresholds(NUM_TREATMENTS)
    assert float(thresholds.max()) <= TAU
    assert float(thresholds.max()) == pytest.approx(TAU)


# ---------------------------------------------------------------------------
# The marks
# ---------------------------------------------------------------------------


def test_marks_start_unused_and_are_keyed_by_row_id() -> None:
    """Algorithm 1 line 2, and the reason the key is `row_id` (§7.1).

    A `QuotaSampler` draw is a fresh subset each step, so a mark stored against
    a batch position would be attributed to a different row next step.
    """
    status = _fresh()
    assert torch.equal(status.marks, torch.full_like(POPULATION, UNUSED))
    status.mark(POPULATION[[3, 5]], torch.tensor([2, 0]), torch.tensor([1.0, 1.0]))
    expected = torch.full_like(POPULATION, UNUSED)
    expected[3] = 2
    expected[5] = 0
    assert torch.equal(status.marks, expected)


def test_a_mark_is_set_at_tau_and_not_at_the_per_class_threshold() -> None:
    """Algorithm 1 line 14 against eq. (8) — the two gates of the module note.

    Marking at `T(c)` instead would make `sigma` self-reinforcing: a class whose
    threshold had fallen would mark more rows, which would raise its own `beta`.
    """
    status = _fresh()
    below, above = TAU - 0.01, TAU + 0.01
    assert float(status.thresholds(NUM_TREATMENTS).max()) == 0.0
    status.mark(POPULATION[:2], torch.tensor([0, 1]), torch.tensor([below, above]))
    # Both rows clear the *threshold* (which is 0); only one clears `tau`.
    assert status.unused() == POPULATION.numel() - 1
    assert int(status.marks[1]) == 1
    assert int(status.marks[0]) == UNUSED


def test_a_mark_is_overwritten_but_never_cleared() -> None:
    """§7: line 15 is the only write and nothing restores `-1`.

    Three claims, and the third needs a *different* row: a `mark` that cleared
    the table before writing would still leave the row it just wrote set, so
    re-marking one row cannot see it. An adversarial mutation sweep found that
    gap — the step that matters is that marking row B leaves row A alone, which
    is what makes `sigma` a count over the whole run rather than over one batch.
    """
    status = _fresh()
    status.mark(POPULATION[:1], torch.tensor([0]), torch.ones(1))
    status.mark(POPULATION[:1], torch.tensor([2]), torch.ones(1))
    assert int(status.marks[0]) == 2, "a later class overwrites an earlier one"
    status.mark(POPULATION[:1], torch.tensor([1]), torch.zeros(1))
    assert int(status.marks[0]) == 2, "a row below tau does not clear its mark"

    status.mark(POPULATION[3:4], torch.tensor([1]), torch.ones(1))
    assert int(status.marks[3]) == 1
    assert int(status.marks[0]) == 2, "marking another row leaves this one marked"
    assert status.unused() == POPULATION.numel() - 2
    assert torch.equal(status.learning_effect(NUM_TREATMENTS), torch.tensor([0, 1, 1]))


def test_a_row_outside_the_population_is_an_error_rather_than_a_silent_write() -> None:
    status = _fresh()
    with pytest.raises(LossError, match="not in the population"):
        status.mark(
            torch.tensor([POPULATION[-1].item() + 1]), torch.tensor([0]), torch.ones(1)
        )


def test_an_empty_or_repeated_population_is_refused() -> None:
    with pytest.raises(LossError, match="empty population"):
        CurriculumStatus(torch.zeros(0, dtype=torch.long), _policy())
    with pytest.raises(LossError, match="repeated row_ids"):
        CurriculumStatus(torch.tensor([1, 1, 2]), _policy())


# ---------------------------------------------------------------------------
# The loss
# ---------------------------------------------------------------------------


def test_a_fully_learned_curriculum_is_fixmatchs_gate_exactly() -> None:
    """The strongest available statement that CPL adds a threshold and no more.

    At `beta(c) = 1` for every class, eq. (12) puts every `T(c)` at `tau` and
    eq. (8) becomes FixMatch's eq. (4). The two objectives are compared on one
    batch, one state and one row set — value, `n` and coverage.
    """
    target, prediction = _inputs()
    rows = torch.arange(BATCH_SIZE)
    batch = make_batch()
    objective = _objective()
    status = _fully_learned(objective.threshold)
    ours = objective.compute(
        _state(target, prediction), batch, rows, _context(status, objective.name)
    )
    theirs = PseudoLabelTreatmentNLL(
        port=Port.T_GIVEN_X,
        target=TARGET,
        prediction=PREDICTION,
        threshold=TAU,
        sharpening="hard",
        stop_grad="target",
        rows="all",
    ).compute(
        _state(target, prediction),
        batch,
        rows,
        TrainContext(global_step=0, schema=make_schema()),
    )
    # The fixture must actually split the batch, or this compares two terms
    # that both kept everything and would agree whatever the gate did.
    assert 0.0 < ours.diagnostics["coverage"] < 1.0
    assert torch.allclose(ours.value, theirs.value)
    assert ours.n == theirs.n
    assert ours.diagnostics["coverage"] == theirs.diagnostics["coverage"]


def test_a_zero_threshold_keeps_every_row_and_fixmatchs_gate_keeps_fewer() -> None:
    """The first limitation of `flexmatch.md` §2, as arithmetic on one batch."""
    target, prediction = _inputs()
    rows = torch.arange(BATCH_SIZE)
    objective = _objective()
    term = objective.compute(
        _state(target, prediction),
        make_batch(),
        rows,
        _context(_fresh(objective.threshold), objective.name),
    )
    assert term.diagnostics["coverage"] == 1.0
    assert term.diagnostics["threshold_max"] == 0.0
    log_probs = prediction.log_softmax(dim=-1)
    labels = target.softmax(dim=-1).argmax(dim=-1)
    expected = -log_probs.gather(1, labels[:, None]).squeeze(1)
    assert torch.allclose(term.value, expected.mean())


def test_the_threshold_is_read_before_this_steps_marks_are_written() -> None:
    """Algorithm 1's ordering — lines 4-12, then 13-17, then the loss at 18.

    The first step must gate on the state it was *handed*, not on the state its
    own marks create. So the first call runs against a fresh status, where every
    `T(c)` is zero and therefore every row is kept, and the second call runs
    against the status the first one left behind, where some `T(c)` has risen
    and fewer rows are kept. A `compute` that marked before it gated would make
    the first call behave like the second.

    An earlier version of this test handed *both* calls a fresh status and
    asserted the two values were equal, which compares two identical
    computations from identical state — it tested determinism, and an
    adversarial review showed a mark-before-gate mutant passed it.
    """
    target, prediction = _inputs()
    rows = torch.arange(BATCH_SIZE)
    batch = make_batch()
    objective = _objective()
    # The population is the batch's own rows, so one step marks enough of it to
    # lift a threshold materially. Over `POPULATION`'s 4B rows the unused count
    # still dominates eq. (11) after one step and every `T(c)` stays near zero,
    # which would make the second call indistinguishable from the first.
    status = CurriculumStatus(POPULATION[:BATCH_SIZE], objective.threshold)

    first = objective.compute(
        _state(target, prediction), batch, rows, _context(status, objective.name)
    )
    assert first.diagnostics["threshold_max"] == 0.0
    assert first.diagnostics["coverage"] == 1.0, (
        "at a zero threshold the first step keeps every row; if it kept fewer, "
        "it gated on marks it had already written"
    )

    assert status.unused() < status.size, "the step must have marked something"
    assert float(status.thresholds(NUM_TREATMENTS).max()) > 0.0
    second = objective.compute(
        _state(target, prediction), batch, rows, _context(status, objective.name)
    )
    assert second.diagnostics["threshold_max"] > 0.0
    assert second.diagnostics["coverage"] < 1.0
    assert not torch.equal(first.value, second.value)


def test_the_denominator_counts_the_rows_the_gate_rejected() -> None:
    """Eq. (8) divides by `mu B`, as FixMatch's eq. (3) does."""
    target, prediction = _inputs()
    rows = torch.arange(BATCH_SIZE)
    objective = _objective()
    status = _fully_learned(objective.threshold)
    term = objective.compute(
        _state(target, prediction), make_batch(), rows, _context(status, objective.name)
    )
    probs = target.softmax(dim=-1)
    confidence, labels = probs.max(dim=-1)
    accepted = confidence > TAU
    assert 0 < int(accepted.sum()) < BATCH_SIZE, "fixture must split the batch"
    per_row = (
        -prediction.log_softmax(dim=-1).gather(1, labels[:, None]).squeeze(1)
        * accepted.float()
    )
    assert torch.allclose(term.value, per_row.mean())
    assert not torch.allclose(term.value, per_row[accepted].mean())


def _one_row(logits: torch.Tensor, status: CurriculumStatus) -> LossTerm:
    """One row through `compute`, at the given target logits."""
    objective = CurriculumPseudoLabelTreatmentNLL(
        port=Port.T_GIVEN_X,
        target=TARGET,
        prediction=PREDICTION,
        threshold=status_policy(status),
        sharpening="hard",
        stop_grad="target",
        rows="all",
    )
    batch = make_batch(
        x=torch.randn(1, 4),
        t=torch.zeros(1, dtype=torch.long),
        y=torch.zeros(1),
        t_observed=torch.ones(1, dtype=torch.bool),
        y_observed=torch.ones(1, dtype=torch.bool),
        row_id=torch.tensor([100]),
    )
    return objective.compute(
        State(
            {
                TARGET: {Port.T_GIVEN_X: CategoricalTreatment(logits)},
                PREDICTION: {Port.T_GIVEN_X: CategoricalTreatment(torch.zeros(1, 2))},
            }
        ),
        batch,
        torch.tensor([0]),
        _context(status, objective.name),
    )


def status_policy(status: CurriculumStatus) -> CurriculumThreshold:
    """The policy a status was built with. Kept explicit for `_one_row`."""
    return status._policy


def _fully_learned_single(policy: CurriculumThreshold) -> CurriculumStatus:
    """A one-row population, marked, so every `T(c)` sits at `tau`."""
    status = CurriculumStatus(torch.tensor([100]), policy)
    status.mark(torch.tensor([100]), torch.tensor([0]), torch.ones(1))
    return status


def test_the_gate_is_strict_where_fixmatchs_is_not() -> None:
    """Eqs. (5), (8) and Alg. 1 line 14 all write `>` (card §7).

    A set of measure zero in a real run, and asserted anyway: it is the one
    place the two objectives' source differs for a reason a reader might
    otherwise take for a transcription slip.

    Both sides of the boundary, because one side alone cannot tell `>` from
    `>=`. An earlier version passed `torch.tensor([[0.0, 0.0]]).log()` as
    *logits* — negative infinity, whose softmax is `NaN` — so its
    `coverage == 0.0` held because `NaN > 0.5` is false, and an adversarial
    review showed a `>=` mutant passed the whole suite.
    """
    policy = _policy()
    # Logits, not probabilities: equal logits give p = 0.5 = tau exactly.
    at_tau = torch.zeros(1, 2)
    # A hair above: softmax([e, 0])[0] > 0.5 for any e > 0.
    above_tau = torch.tensor([[0.01, 0.0]])

    exact = _one_row(at_tau, _fully_learned_single(policy))
    assert exact.diagnostics["threshold_max"] == pytest.approx(TAU)
    assert exact.diagnostics["coverage"] == 0.0, "`> tau` rejects the exact tie"

    over = _one_row(above_tau, _fully_learned_single(policy))
    assert over.diagnostics["coverage"] == 1.0, "and keeps anything above it"

    # The *mark* gate is `>` too (Alg. 1 line 14), and it is a different
    # comparison in a different method, so it needs its own tie.
    status = _fresh()
    status.mark(POPULATION[:1], torch.tensor([0]), torch.tensor([TAU]))
    assert status.unused() == POPULATION.numel(), "`> tau` does not mark the tie"
    status.mark(POPULATION[:1], torch.tensor([0]), torch.tensor([TAU + 1e-6]))
    assert status.unused() == POPULATION.numel() - 1, "and marks just above it"


def test_the_mark_gate_and_the_loss_gate_are_not_collapsed_in_compute() -> None:
    """Card §3.2's "two gates", at the level where a wiring error would live.

    `test_a_mark_is_set_at_tau_and_not_at_the_per_class_threshold` exercises
    `CurriculumStatus.mark`, which hard-codes `tau` and therefore cannot see a
    `compute` that passed it `T(c)` instead. An adversarial review made exactly
    that mutation and the whole Tier 0 suite passed.

    The fixture is one row whose confidence sits *between* the two gates: above
    `T(c)`, which a single mark has lifted only slightly off zero, and below
    `tau`. Eq. (8) must keep it; algorithm 1 line 14 must not mark it. A
    `compute` that marked at `T(c)` would mark it, and one that gated at `tau`
    would drop it.
    """
    policy = _policy()
    status = CurriculumStatus(POPULATION, policy)
    status.mark(POPULATION[:1], torch.tensor([0]), torch.ones(1))
    thresholds = status.thresholds(NUM_TREATMENTS)
    lowest = float(thresholds[0])
    assert 0.0 < lowest < TAU, "the fixture needs the two gates far apart"

    # p(class 0) strictly between T(0) and tau.
    target = float((lowest + TAU) / 2.0)
    logit = math.log(target / (1.0 - target))
    marked_before = status.unused()
    term = _one_row(torch.tensor([[logit, 0.0]]), status)

    assert term.diagnostics["coverage"] == 1.0, (
        "eq. (8) gates at T(c), which this row clears"
    )
    assert status.unused() == marked_before, (
        "algorithm 1 line 14 marks at tau, which this row does not clear"
    )


def test_the_target_side_is_detached_and_the_prediction_side_is_not() -> None:
    target, prediction = _inputs()
    target = target.clone().requires_grad_(True)
    prediction = prediction.clone().requires_grad_(True)
    objective = _objective()
    term = objective.compute(
        _state(target, prediction),
        make_batch(),
        torch.arange(BATCH_SIZE),
        _context(_fully_learned(objective.threshold), objective.name),
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    assert target.grad is None
    assert prediction.grad is not None
    assert objective.detaches == frozenset({(Port.T_GIVEN_X, TARGET)})


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def test_the_state_is_built_over_the_rows_the_objective_is_entitled_to() -> None:
    """`N` is the objective's own population, not the whole table.

    `rows="all"` is this recipe's value and FixMatch's footnote 2 is why; a term
    declared over `t_missing` would count only those, which is what eq. (11)
    would then be normalising against.
    """
    from xty2.core import Dataset, MissingnessSpec, PreprocessSpec, SplitSpec
    from xty2.core.data import DataSpec
    from xty2.training.loading import build_population

    batch = make_batch()
    dataset = Dataset(
        schema=make_schema(),
        rows=batch,
        assignments={"train": torch.arange(BATCH_SIZE)},
    )
    population = build_population(
        dataset,
        DataSpec(
            split=SplitSpec(protocol="tier 0 fixture", train="train"),
            preprocess=PreprocessSpec(features="none", outcome="none"),
            missingness=MissingnessSpec(mechanism="observed"),
        ),
        seed=0,
    )
    everything = _objective().initial_state(population)
    missing = _objective(rows="t_missing").initial_state(population)
    assert isinstance(everything, CurriculumStatus)
    assert isinstance(missing, CurriculumStatus)
    assert everything.size == BATCH_SIZE
    assert missing.size == int((~batch.t_observed).sum())


def test_a_stage_with_no_population_is_refused_rather_than_guessed_at() -> None:
    with pytest.raises(LossError, match="needs the stage's training population"):
        _objective().initial_state(None)


def test_a_stateful_objective_in_a_cross_fit_stage_is_a_compile_error() -> None:
    """The executor gap an adversarial review found, closed at declaration time.

    `_run_cross_fit` slices a fresh training batch per fold and calls
    `_run_stage` without a `TrainingPopulation`, so this objective would have
    raised at the first step of the first fold with an error about `N` rather
    than about the pairing. `Stage` declares its executor, so the compiler can
    see the conflict; `DESIGN.md` §8 wants this class of failure there.
    """
    with pytest.raises(CompileError, match="uses cross_fit and holds the stateful"):
        stage(
            name="folded",
            executor="cross_fit",
            objectives=(Weighted(_objective(), weight=1.0, reduction="mean"),),
            action=PseudoLabelAction(port=Port.T_GIVEN_X),
        )


def test_a_stateless_objective_in_a_cross_fit_stage_still_compiles() -> None:
    """The rejection is about state, not about cross-fitting — `ssdml` uses it."""
    folded = stage(
        name="folded",
        executor="cross_fit",
        objectives=(
            Weighted(
                PseudoLabelTreatmentNLL(
                    port=Port.T_GIVEN_X,
                    target=TARGET,
                    prediction=PREDICTION,
                    threshold=TAU,
                    sharpening="hard",
                    stop_grad="target",
                    rows="all",
                ),
                weight=1.0,
                reduction="mean",
            ),
        ),
        action=PseudoLabelAction(port=Port.T_GIVEN_X),
    )
    assert folded.executor == "cross_fit"


def test_the_objective_is_not_batch_coupled() -> None:
    """It carries state across steps; it does not couple rows within one."""
    assert _objective().batch_coupled is False


def test_the_gate_rule_is_one_card_key_and_has_no_default() -> None:
    with pytest.raises(CardKeyError, match="no usable default"):
        CurriculumPseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=TARGET,
            prediction=PREDICTION,
            sharpening="hard",
            stop_grad="target",
        )
    assert CurriculumPseudoLabelTreatmentNLL.CARD_KEYS["threshold"] == (
        "losses.confidence_threshold"
    )


def test_a_bare_float_threshold_names_the_objective_that_takes_one() -> None:
    with pytest.raises(LossError, match="PseudoLabelTreatmentNLL"):
        _objective(threshold=0.95)


def test_the_policy_renders_the_way_card_section_four_writes_it() -> None:
    """The plan renders with `!r` and the card cross-check compares `str`."""
    policy = _policy(tau=0.95)
    assert repr(policy) == "curriculum(tau=0.95, warm_up=true, mapping=convex)"
    assert str(policy) == repr(policy)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("tau", 1.5, "must be in"),
        ("tau", "high", "must be a number"),
        ("warm_up", "yes", "must be a bool"),
        ("mapping", "concave", "'convex' or 'identity'"),
    ],
)
def test_the_policy_rejects_a_rule_it_cannot_state(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(LossError, match=message):
        _policy(**{field: value})


def test_one_realisation_on_both_sides_is_refused() -> None:
    with pytest.raises(LossError, match="entropy minimisation"):
        _objective(prediction=TARGET)


def test_a_non_treatment_port_is_refused_with_the_reason() -> None:
    with pytest.raises(LossError, match="per-class over"):
        _objective(port=Port.X_REPR)
