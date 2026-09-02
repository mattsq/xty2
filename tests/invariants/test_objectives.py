"""Tier 0 — the three shipped objectives (`DESIGN.md` §4, §4.1).

The load-bearing test in this file is `test_exact_marginalisation_equals_brute
_force`. Exact marginalisation over `t` is the objective the whole design is
arranged around, it is written in log space for stability, and *nothing about
reading it tells you whether it is right*. So it is checked against an explicit
Python loop over `k` in probability space, at double precision, on a batch with
`B != K`.

That loop is also the candidate-treatment contract (`DESIGN.md` §3.1) exercised
through an objective rather than through a distribution: each column of the
`[B, K]` term has to agree with a separate `[B]` call at the same treatment,
with `y` passed unexpanded both times. A head that broadcasts ambiguously
passes every shape assertion and fails here.
"""

import math

import pytest
import torch
from torch import Tensor
from xty2.core import (
    DEFAULT,
    CardKeyError,
    CategoricalTreatment,
    ComponentGraph,
    GaussianOutcome,
    LossError,
    Port,
    PortValue,
    Realisation,
    Recipe,
    Schema,
    State,
    TrainContext,
    Weighted,
    XTYBatch,
    compile,
    resolve_rows,
    treatment_at,
)
from xty2.objectives import (
    MissingTreatmentMarginalNLL,
    ObservedOutcomeMSE,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)

from tests.invariants.conftest import (
    BATCH_SIZE,
    HIDDEN,
    NUM_TREATMENTS,
    ToyEncoder,
    ToyOutcomeHead,
    ToyPropensity,
    backward,
    make_batch,
    make_schema,
    stage,
)

# `B != K` throughout, and asserted here rather than assumed: a test with
# B == K passes under accidental broadcasting and proves nothing.
assert BATCH_SIZE != NUM_TREATMENTS


def _double_batch(**overrides: object) -> XTYBatch:
    """A batch whose outcome is float64, so brute force can be exact."""
    defaults: dict[str, object] = {
        "x": torch.randn(BATCH_SIZE, 4, dtype=torch.float64),
        "y": torch.randn(BATCH_SIZE, dtype=torch.float64),
    }
    return make_batch(**(defaults | overrides))


def _heads(batch: XTYBatch) -> tuple[GaussianOutcome, CategoricalTreatment]:
    generator = torch.Generator().manual_seed(11)
    shape = (batch.batch_size, NUM_TREATMENTS)
    head = GaussianOutcome(
        loc=torch.randn(shape, generator=generator, dtype=torch.float64),
        scale=torch.rand(shape, generator=generator, dtype=torch.float64) + 0.5,
    )
    propensity = CategoricalTreatment(
        torch.randn(shape, generator=generator, dtype=torch.float64)
    )
    return head, propensity


def _state(head: GaussianOutcome, propensity: CategoricalTreatment) -> State:
    return State({DEFAULT: {Port.Y_GIVEN_XT: head, Port.T_GIVEN_X: propensity}})


def _ctx(schema: Schema | None = None, *, step: int = 0) -> TrainContext:
    return TrainContext(global_step=step, schema=schema or make_schema(), stage="fit")


# ---------------------------------------------------------------------------
# MissingTreatmentMarginalNLL — the objective the design is arranged around
# ---------------------------------------------------------------------------


def _brute_force_marginal(
    head: GaussianOutcome, propensity: CategoricalTreatment, batch: XTYBatch
) -> Tensor:
    """`-log Σ_k p(t=k|x) p(y|x,t=k)`, in probability space, one k at a time."""
    probs = propensity.probs
    total = torch.zeros(batch.batch_size, dtype=torch.float64)
    for k in range(NUM_TREATMENTS):
        at_k = torch.full((batch.batch_size,), k, dtype=torch.long)
        # `y` unexpanded, `t` of rank 1: the observed-treatment mode of §3.1.
        total = total + probs[:, k] * head.log_prob(batch.y, at_k).exp()
    return -total.log()


def test_exact_marginalisation_equals_brute_force() -> None:
    batch = _double_batch()
    head, propensity = _heads(batch)
    rows = resolve_rows(batch, "t_missing")
    objective = MissingTreatmentMarginalNLL(grad_path="both")

    term = objective.compute(_state(head, propensity), batch, rows, _ctx())
    expected = _brute_force_marginal(head, propensity, batch).index_select(0, rows)

    assert term.n == int(rows.numel())
    assert float(term.value) == pytest.approx(float(expected.mean()), abs=1e-12)


def test_the_candidate_columns_agree_with_one_call_per_treatment() -> None:
    # The candidate-treatment contract, as the marginalisation term uses it:
    # column k of the [B, K] result is the [B] result at t = k, with y passed
    # unexpanded in both. Expanding y at the call site instead — `y: [B]`
    # against `t: [B, K]` — aligns B against K and raises for B != K.
    batch = _double_batch()
    head, _ = _heads(batch)
    candidates = torch.arange(NUM_TREATMENTS).expand(BATCH_SIZE, NUM_TREATMENTS)
    columns = head.log_prob(batch.y, candidates)
    assert columns.shape == (BATCH_SIZE, NUM_TREATMENTS)
    for k in range(NUM_TREATMENTS):
        at_k = torch.full((BATCH_SIZE,), k, dtype=torch.long)
        assert torch.allclose(columns[:, k], head.log_prob(batch.y, at_k))


def test_marginalisation_uses_every_treatment_not_the_observed_one() -> None:
    # If the candidate matrix were built from the observed `t`, the term would
    # move when `t` moved. It is a function of x and y alone on rows where t is
    # missing, and this is what says so.
    batch = _double_batch()
    head, propensity = _heads(batch)
    rows = resolve_rows(batch, "t_missing")
    objective = MissingTreatmentMarginalNLL(grad_path="both")

    first = objective.compute(_state(head, propensity), batch, rows, _ctx())
    relabelled = batch.replace(t=(batch.t + 1) % NUM_TREATMENTS)
    second = objective.compute(_state(head, propensity), relabelled, rows, _ctx())
    assert float(first.value) == float(second.value)


def test_marginalisation_over_one_row_is_a_logsumexp_by_hand() -> None:
    # The smallest fully-written-out case, so the formula itself is checked
    # against arithmetic and not only against another implementation of it.
    batch = _double_batch(
        t_observed=torch.zeros(BATCH_SIZE, dtype=torch.bool),
        y=torch.zeros(BATCH_SIZE, dtype=torch.float64),
    )
    head = GaussianOutcome(
        loc=torch.zeros(BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64),
        scale=torch.ones(BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64),
    )
    propensity = CategoricalTreatment(
        torch.zeros(BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64)
    )
    rows = resolve_rows(batch, "t_missing")
    term = MissingTreatmentMarginalNLL(grad_path="both").compute(
        _state(head, propensity), batch, rows, _ctx()
    )
    # Every candidate has probability 1/K and density N(0; 0, 1), so the sum is
    # exactly that density and the term is its negative log.
    assert float(term.value) == pytest.approx(0.5 * math.log(2.0 * math.pi), abs=1e-12)


def test_marginalisation_with_no_missing_treatments_is_the_zero_term() -> None:
    batch = _double_batch(t_observed=torch.ones(BATCH_SIZE, dtype=torch.bool))
    head, propensity = _heads(batch)
    rows = resolve_rows(batch, "t_missing")
    term = MissingTreatmentMarginalNLL(grad_path="both").compute(
        _state(head, propensity), batch, rows, _ctx()
    )
    assert term.n == 0
    assert float(term.value) == 0.0
    assert not torch.isnan(term.value)


def test_the_objective_declares_the_two_ports_it_needs() -> None:
    objective = MissingTreatmentMarginalNLL(grad_path="both")
    assert objective.rows == "t_missing"
    assert objective.requires == frozenset(
        {(Port.T_GIVEN_X, DEFAULT), (Port.Y_GIVEN_XT, DEFAULT)}
    )


# ---------------------------------------------------------------------------
# The gradient path — a required card field (FIDELITY.md §2)
# ---------------------------------------------------------------------------


def _grad_norms(grad_path: str) -> tuple[float, float]:
    """`(propensity grad, outcome grad)` norms under one gradient path."""
    batch = _double_batch()
    logits = torch.randn(
        BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64, requires_grad=True
    )
    loc = torch.randn(
        BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64, requires_grad=True
    )
    head = GaussianOutcome(loc=loc, scale=torch.ones_like(loc))
    propensity = CategoricalTreatment(logits)
    rows = resolve_rows(batch, "t_missing")
    objective = MissingTreatmentMarginalNLL(grad_path=grad_path)  # type: ignore[arg-type]
    backward(objective.compute(_state(head, propensity), batch, rows, _ctx()).value)
    return (
        0.0 if logits.grad is None else float(logits.grad.norm()),
        0.0 if loc.grad is None else float(loc.grad.norm()),
    )


def test_grad_path_both_trains_both_heads() -> None:
    propensity, outcome = _grad_norms("both")
    assert propensity > 0.0
    assert outcome > 0.0


def test_grad_path_propensity_trains_only_the_treatment_head() -> None:
    # Several papers stop-gradient one side of this term. Which one is a
    # required card field precisely because the run looks identical either way.
    propensity, outcome = _grad_norms("propensity")
    assert propensity > 0.0
    assert outcome == 0.0


def test_grad_path_outcome_trains_only_the_outcome_head() -> None:
    propensity, outcome = _grad_norms("outcome")
    assert propensity == 0.0
    assert outcome > 0.0


def test_the_gradient_path_has_no_default() -> None:
    with pytest.raises(CardKeyError, match="marginal_nll_grad_path"):
        MissingTreatmentMarginalNLL()


def test_an_unknown_gradient_path_is_rejected() -> None:
    with pytest.raises(CardKeyError, match="expected one of"):
        MissingTreatmentMarginalNLL(grad_path="stopgrad")  # type: ignore[arg-type]


def test_the_gradient_path_reaches_the_plan_as_a_card_key() -> None:
    # Presence, not correctness (`FIDELITY.md` §1.2): this proves the recipe
    # states which head the term trains, never that it states the right one.
    recipe = Recipe(
        name="marginal",
        schema=make_schema(),
        system=ComponentGraph(
            [ToyEncoder(width=HIDDEN), ToyOutcomeHead(), ToyPropensity()]
        ),
        program=(
            stage(
                objectives=(
                    Weighted(
                        MissingTreatmentMarginalNLL(grad_path="propensity"),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                # `grad_path="propensity"` detaches the outcome side, so the
                # outcome head is deliberately *not* trainable here — the
                # dead-trainable check rejects a stage that names it.
                trainable=("encoder", "propensity"),
            ),
        ),
        card="docs/recipes/marginal.md",
    )
    hyperparameters = compile(recipe).plan.hyperparameters
    assert hyperparameters["gradients.marginal_nll_grad_path"] == "propensity"
    assert hyperparameters["architecture.widths_depths"] == {"encoder": HIDDEN}


# ---------------------------------------------------------------------------
# The two complete-case terms
# ---------------------------------------------------------------------------


def test_the_observed_outcome_nll_is_the_gaussian_log_density() -> None:
    batch = _double_batch()
    head, _ = _heads(batch)
    rows = resolve_rows(batch, "t_observed")
    state = State({DEFAULT: {Port.Y_GIVEN_XT: head}})

    term = ObservedOutcomeNLL().compute(state, batch, rows, _ctx())

    expected = torch.stack(
        [
            -head.log_prob(batch.y, torch.full((BATCH_SIZE,), int(batch.t[row])))[row]
            for row in rows.tolist()
        ]
    )
    assert term.n == int(rows.numel())
    assert float(term.value) == pytest.approx(float(expected.mean()), abs=1e-12)


def test_the_observed_outcome_nll_applies_batch_weights_before_reduction() -> None:
    weights = torch.linspace(0.25, 2.0, BATCH_SIZE, dtype=torch.float64)
    batch = _double_batch(weight=weights)
    head, _ = _heads(batch)
    rows = resolve_rows(batch, "t_observed")
    state = State({DEFAULT: {Port.Y_GIVEN_XT: head}})

    term = ObservedOutcomeNLL().compute(state, batch, rows, _ctx())

    per_row = -head.log_prob(batch.y, treatment_at(batch, rows)) * weights
    expected = per_row.index_select(0, rows).mean()
    assert term.n == int(rows.numel())
    assert float(term.value) == pytest.approx(float(expected), abs=1e-12)


def test_the_observed_outcome_mse_is_exact_weighted_squared_error() -> None:
    weights = torch.linspace(0.25, 2.0, BATCH_SIZE, dtype=torch.float64)
    batch = _double_batch(weight=weights)
    head, _ = _heads(batch)
    rows = resolve_rows(batch, "t_observed")
    state = State({DEFAULT: {Port.Y_GIVEN_XT: head}})

    term = ObservedOutcomeMSE().compute(state, batch, rows, _ctx())

    error = head.mean(treatment_at(batch, rows)) - batch.y
    expected = (weights * error.square()).index_select(0, rows).mean()
    assert term.n == int(rows.numel())
    assert float(term.value) == pytest.approx(float(expected), abs=1e-12)


def test_the_observed_treatment_nll_is_the_categorical_cross_entropy() -> None:
    batch = _double_batch()
    _, propensity = _heads(batch)
    rows = resolve_rows(batch, "t_observed")
    state = State({DEFAULT: {Port.T_GIVEN_X: propensity}})

    term = ObservedTreatmentNLL().compute(state, batch, rows, _ctx())

    expected = torch.nn.functional.cross_entropy(
        propensity.logits.index_select(0, rows),
        batch.t.index_select(0, rows),
        reduction="mean",
    )
    assert float(term.value) == pytest.approx(float(expected), abs=1e-12)


def test_the_observed_treatment_nll_can_consume_a_named_realisation() -> None:
    batch = _double_batch()
    _, propensity = _heads(batch)
    rows = resolve_rows(batch, "t_observed")
    noisy_student = Realisation(view="student_x")
    state = State({noisy_student: {Port.T_GIVEN_X: propensity}})
    objective = ObservedTreatmentNLL(realisation=noisy_student)

    term = objective.compute(state, batch, rows, _ctx())

    expected = torch.nn.functional.cross_entropy(
        propensity.logits.index_select(0, rows),
        batch.t.index_select(0, rows),
        reduction="mean",
    )
    assert float(term.value) == pytest.approx(float(expected), abs=1e-12)
    assert objective.requires == frozenset({(Port.T_GIVEN_X, noisy_student)})


def test_the_observed_treatment_nll_rejects_a_non_realisation() -> None:
    with pytest.raises(LossError, match="must be a Realisation"):
        ObservedTreatmentNLL(realisation="student_x")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "objective",
    [ObservedOutcomeMSE(), ObservedOutcomeNLL(), ObservedTreatmentNLL()],
    ids=["outcome_mse", "outcome_nll", "treatment"],
)
def test_an_unobserved_treatment_is_never_read(objective: object) -> None:
    # The no-sentinel rule with teeth: `t` on unobserved rows holds an arbitrary
    # valid class index, so a term that read it would move when that arbitrary
    # value moved. Both terms are computed full-batch and gathered afterwards,
    # which is exactly where such a read would happen.
    batch = _double_batch()
    head, propensity = _heads(batch)
    state = State({DEFAULT: {Port.Y_GIVEN_XT: head, Port.T_GIVEN_X: propensity}})
    rows = resolve_rows(batch, "t_observed")
    scrambled = batch.replace(
        t=torch.where(batch.t_observed, batch.t, (batch.t + 1) % NUM_TREATMENTS)
    )

    first = objective.compute(state, batch, rows, _ctx())  # type: ignore[attr-defined]
    second = objective.compute(state, scrambled, rows, _ctx())  # type: ignore[attr-defined]
    assert float(first.value) == float(second.value)


@pytest.mark.parametrize(
    "objective, port",
    [
        (ObservedOutcomeNLL(), Port.Y_GIVEN_XT),
        (ObservedOutcomeMSE(), Port.Y_GIVEN_XT),
        (ObservedTreatmentNLL(), Port.T_GIVEN_X),
    ],
    ids=["outcome_nll", "outcome_mse", "treatment"],
)
def test_a_complete_case_term_with_no_observed_rows_is_zero(
    objective: object, port: Port
) -> None:
    batch = _double_batch(t_observed=torch.zeros(BATCH_SIZE, dtype=torch.bool))
    head, propensity = _heads(batch)
    values: dict[Port, PortValue] = {
        Port.Y_GIVEN_XT: head,
        Port.T_GIVEN_X: propensity,
    }
    state = State({DEFAULT: {port: values[port]}})
    rows = resolve_rows(batch, "t_observed")

    term = objective.compute(state, batch, rows, _ctx())  # type: ignore[attr-defined]
    assert term.n == 0
    assert float(term.value) == 0.0


def test_the_complete_case_terms_declare_their_ports_and_rows() -> None:
    assert ObservedOutcomeMSE().rows == "t_observed"
    assert ObservedOutcomeMSE().requires == frozenset({(Port.Y_GIVEN_XT, DEFAULT)})
    assert ObservedOutcomeNLL().rows == "t_observed"
    assert ObservedOutcomeNLL().requires == frozenset({(Port.Y_GIVEN_XT, DEFAULT)})
    assert ObservedTreatmentNLL().rows == "t_observed"
    assert ObservedTreatmentNLL().requires == frozenset({(Port.T_GIVEN_X, DEFAULT)})
