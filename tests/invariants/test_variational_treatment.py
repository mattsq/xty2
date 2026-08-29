"""Tier 0 — the discrete-treatment ELBO and reviewed recipe contract."""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor
from xty2.components import CategoricalPosterior
from xty2.core import (
    DEFAULT,
    CategoricalTreatment,
    ExternalBatches,
    GaussianOutcome,
    Port,
    Program,
    State,
    TrainContext,
    TreatmentDistribution,
    XTYBatch,
    compile,
)
from xty2.core.rows import resolve_rows
from xty2.objectives import MissingTreatmentMarginalNLL, VariationalTreatmentELBO
from xty2.recipes import variational_treatment
from xty2.recipes.variational_treatment import (
    DATA_POLICY,
    MISSING_PER_BATCH,
    OBSERVED_PER_BATCH,
    POSTERIOR_SUPERVISION_WEIGHT,
    POSTERIOR_WIDTHS,
    VARIATIONAL_TREATMENT_STEPS,
)

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)

ROOT = Path(__file__).resolve().parents[2]


class _FixedTreatment:
    """A protocol-conforming categorical distribution that permits exact zeros."""

    def __init__(self, probs: Tensor) -> None:
        self._probs = probs

    @property
    def probs(self) -> Tensor:
        return self._probs

    def log_prob(self, t: Tensor) -> Tensor:
        negative_infinity = torch.full_like(self._probs, -torch.inf)
        log_probs = torch.where(self._probs > 0, self._probs.log(), negative_infinity)
        index = t[:, None] if t.ndim == 1 else t
        gathered = torch.gather(log_probs, 1, index)
        return gathered.squeeze(1) if t.ndim == 1 else gathered


def _state(
    *,
    propensity_logits: Tensor | None = None,
    outcome_loc: Tensor | None = None,
    posterior_logits: Tensor | None = None,
    posterior: TreatmentDistribution | None = None,
) -> State:
    propensity_logits = (
        torch.randn(BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64)
        if propensity_logits is None
        else propensity_logits
    )
    outcome_loc = (
        torch.randn(BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64)
        if outcome_loc is None
        else outcome_loc
    )
    posterior_logits = (
        torch.randn(BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64)
        if posterior_logits is None
        else posterior_logits
    )
    posterior_value: TreatmentDistribution = (
        CategoricalTreatment(posterior_logits) if posterior is None else posterior
    )
    return State(
        {
            DEFAULT: {
                Port.T_GIVEN_X: CategoricalTreatment(propensity_logits),
                Port.Y_GIVEN_XT: GaussianOutcome(
                    loc=outcome_loc,
                    scale=torch.ones_like(outcome_loc),
                ),
                Port.T_GIVEN_XY: posterior_value,
            }
        }
    )


def _terms(state: State, batch: XTYBatch) -> tuple[Tensor, Tensor]:
    rows = resolve_rows(batch, "t_missing")
    context = TrainContext(0, make_schema())
    variational = VariationalTreatmentELBO().compute(
        state, batch, rows, context
    ).value
    exact = MissingTreatmentMarginalNLL(grad_path="both").compute(
        state, batch, rows, context
    ).value
    return variational, exact


def test_the_bound_equals_exact_marginal_plus_kl_on_the_same_state() -> None:
    torch.manual_seed(12)
    batch = make_batch(y=torch.randn(BATCH_SIZE, dtype=torch.float64))
    state = _state()
    variational, exact = _terms(state, batch)

    candidates = torch.arange(NUM_TREATMENTS).expand(BATCH_SIZE, NUM_TREATMENTS)
    propensity = state.default[Port.T_GIVEN_X]
    outcome = state.default[Port.Y_GIVEN_XT]
    posterior = state.default[Port.T_GIVEN_XY]
    assert isinstance(propensity, CategoricalTreatment)
    assert isinstance(outcome, GaussianOutcome)
    assert isinstance(posterior, CategoricalTreatment)
    log_joint = propensity.log_prob(candidates) + outcome.log_prob(batch.y, candidates)
    log_true_posterior = log_joint - torch.logsumexp(log_joint, dim=-1, keepdim=True)
    log_q = posterior.log_prob(candidates)
    kl = (posterior.probs * (log_q - log_true_posterior)).sum(dim=-1)
    rows = resolve_rows(batch, "t_missing")
    expected_gap = kl.index_select(0, rows).mean()

    torch.testing.assert_close(variational - exact, expected_gap)
    assert float(variational - exact) >= -1e-12


def test_the_bound_is_tight_when_q_is_the_exact_posterior() -> None:
    torch.manual_seed(13)
    batch = make_batch(y=torch.randn(BATCH_SIZE, dtype=torch.float64))
    base = _state()
    candidates = torch.arange(NUM_TREATMENTS).expand(BATCH_SIZE, NUM_TREATMENTS)
    propensity = base.default[Port.T_GIVEN_X]
    outcome = base.default[Port.Y_GIVEN_XT]
    assert isinstance(propensity, CategoricalTreatment)
    assert isinstance(outcome, GaussianOutcome)
    log_joint = propensity.log_prob(candidates) + outcome.log_prob(batch.y, candidates)
    state = State(
        {
            DEFAULT: {
                Port.T_GIVEN_X: propensity,
                Port.Y_GIVEN_XT: outcome,
                Port.T_GIVEN_XY: CategoricalTreatment(log_joint.detach()),
            }
        }
    )
    variational, exact = _terms(state, batch)
    torch.testing.assert_close(variational, exact, rtol=1e-10, atol=1e-10)


def test_uniform_q_contributes_exactly_negative_log_k_entropy() -> None:
    torch.manual_seed(14)
    batch = make_batch(y=torch.randn(BATCH_SIZE, dtype=torch.float64))
    state = _state(
        posterior_logits=torch.zeros(
            BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64
        )
    )
    rows = resolve_rows(batch, "t_missing")
    value = VariationalTreatmentELBO().compute(
        state, batch, rows, TrainContext(0, make_schema())
    ).value
    candidates = torch.arange(NUM_TREATMENTS).expand(BATCH_SIZE, NUM_TREATMENTS)
    propensity = state.default[Port.T_GIVEN_X]
    outcome = state.default[Port.Y_GIVEN_XT]
    assert isinstance(propensity, CategoricalTreatment)
    assert isinstance(outcome, GaussianOutcome)
    energy = (
        -propensity.log_prob(candidates) - outcome.log_prob(batch.y, candidates)
    ).mean(dim=-1)
    expected = energy.index_select(0, rows).mean() - math.log(NUM_TREATMENTS)
    torch.testing.assert_close(value, expected)


def test_one_hot_q_reduces_to_the_selected_complete_case_term() -> None:
    torch.manual_seed(15)
    batch = make_batch(y=torch.randn(BATCH_SIZE, dtype=torch.float64))
    selected = 1
    probs = torch.zeros(BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64)
    probs[:, selected] = 1.0
    state = _state(posterior=_FixedTreatment(probs))
    rows = resolve_rows(batch, "t_missing")
    value = VariationalTreatmentELBO().compute(
        state, batch, rows, TrainContext(0, make_schema())
    ).value
    candidates = torch.arange(NUM_TREATMENTS).expand(BATCH_SIZE, NUM_TREATMENTS)
    propensity = state.default[Port.T_GIVEN_X]
    outcome = state.default[Port.Y_GIVEN_XT]
    assert isinstance(propensity, CategoricalTreatment)
    assert isinstance(outcome, GaussianOutcome)
    selected_loss = (
        -propensity.log_prob(candidates) - outcome.log_prob(batch.y, candidates)
    )[:, selected]
    torch.testing.assert_close(value, selected_loss.index_select(0, rows).mean())
    assert bool(torch.isfinite(value))


def test_gradients_reach_propensity_outcome_and_posterior() -> None:
    torch.manual_seed(16)
    batch = make_batch(y=torch.randn(BATCH_SIZE, dtype=torch.float64))
    propensity_logits = torch.randn(
        BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64, requires_grad=True
    )
    outcome_loc = torch.randn(
        BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64, requires_grad=True
    )
    posterior_logits = torch.randn(
        BATCH_SIZE, NUM_TREATMENTS, dtype=torch.float64, requires_grad=True
    )
    objective = VariationalTreatmentELBO()
    rows = resolve_rows(batch, "t_missing")
    term = objective.compute(
        _state(
            propensity_logits=propensity_logits,
            outcome_loc=outcome_loc,
            posterior_logits=posterior_logits,
        ),
        batch,
        rows,
        TrainContext(0, make_schema()),
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    assert objective.detaches == frozenset()
    for tensor in (propensity_logits, outcome_loc, posterior_logits):
        assert tensor.grad is not None
        assert float(tensor.grad.abs().sum()) > 0.0


def test_candidates_come_from_schema_and_never_from_batch_t() -> None:
    torch.manual_seed(17)
    y = torch.randn(BATCH_SIZE, dtype=torch.float64)
    first = make_batch(y=y)
    second = make_batch(y=y, t=torch.zeros(BATCH_SIZE, dtype=torch.long))
    state = _state()
    rows = resolve_rows(first, "t_missing")
    context = TrainContext(0, make_schema())
    objective = VariationalTreatmentELBO()
    a = objective.compute(state, first, rows, context).value
    b = objective.compute(state, second, rows, context).value
    torch.testing.assert_close(a, b)
    assert BATCH_SIZE != NUM_TREATMENTS


def test_empty_missing_population_is_zero_and_objective_is_not_batch_coupled() -> None:
    batch = make_batch(
        y=torch.randn(BATCH_SIZE, dtype=torch.float64),
        t_observed=torch.ones(BATCH_SIZE, dtype=torch.bool),
    )
    objective = VariationalTreatmentELBO()
    term = objective.compute(
        _state(),
        batch,
        resolve_rows(batch, "t_missing"),
        TrainContext(0, make_schema()),
    )
    assert term.n == 0
    assert float(term.value) == 0.0
    assert not objective.batch_coupled


def test_external_batches_are_accepted_for_the_uncoupled_objective() -> None:
    recipe = variational_treatment(make_schema())
    stage = replace(recipe.program[0], sampler=ExternalBatches())
    changed = replace(recipe, program=Program((stage,)), data=None)
    compile(changed)


def test_recipe_has_one_nonleaking_stage_and_no_pseudo_label_action() -> None:
    recipe = variational_treatment(make_schema())
    run = compile(recipe)
    assert recipe.purpose == "causal"
    assert len(run.stages) == 1
    stage = recipe.program[0]
    assert stage.name == "elbo_fit"
    assert not stage.allow_leakage
    assert stage.action is None
    assert stage.steps == VARIATIONAL_TREATMENT_STEPS
    assert [term.name for term in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        "variational_treatment_elbo",
        "posterior_treatment_nll",
    ]
    assert [term.reduction for term in stage.objectives] == [
        "population",
        "population",
        "population",
        "mean",
    ]
    assert stage.objectives[-1].weight_at(0) == POSTERIOR_SUPERVISION_WEIGHT


def test_plan_matches_the_reviewed_card_and_shared_data_bindings() -> None:
    recipe = variational_treatment(make_schema())
    plan = compile(recipe).plan
    hyperparameters = plan.hyperparameters
    assert hyperparameters["optimisation.batch_size"] == 128
    assert hyperparameters["optimisation.labelled_unlabelled_ratio"] == 15.0
    assert hyperparameters["optimisation.total_steps_or_epochs"] == 3000
    assert hyperparameters["architecture.widths_depths"]["categorical_posterior"] == (
        "concat(X_RAW, Y_RAW) -> [300] -> K"
    )
    assert POSTERIOR_WIDTHS == (300,)
    assert OBSERVED_PER_BATCH + MISSING_PER_BATCH == 128

    posterior = recipe.system["categorical_posterior"]
    assert isinstance(posterior, CategoricalPosterior)
    assert posterior.standardisation == DATA_POLICY.standardisation
    assert posterior.outcome_scaling == DATA_POLICY.outcome_scaling
    assert hyperparameters["data.standardisation"] == DATA_POLICY.standardisation
    assert hyperparameters["data.outcome_scaling"] == DATA_POLICY.outcome_scaling


def test_recipe_contains_declarations_and_no_control_flow() -> None:
    tree = ast.parse((ROOT / "xty2/recipes/variational_treatment.py").read_text())
    forbidden = (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.IfExp)
    assert not [node for node in ast.walk(tree) if isinstance(node, forbidden)]
