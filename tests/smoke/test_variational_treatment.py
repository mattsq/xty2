"""Tier 1 — variational treatment fitting on the card's overlap DGP."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    ComponentGraph,
    Constant,
    Dataset,
    FeatureSpec,
    GaussianOutcome,
    OutcomeSpec,
    Port,
    Program,
    Recipe,
    Schema,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.objectives import MissingTreatmentMarginalNLL
from xty2.recipes import variational_treatment
from xty2.training import StageResult, run_stage

FEATURES = 6
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048
STEPS = 3_000
CLUSTER_SIGNAL = 0.45
LOW_PROPENSITY = 0.25
OBSERVED_TREATMENTS = 64


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _schema() -> Schema:
    return Schema(
        features=tuple(
            FeatureSpec(f"x{column}", "continuous") for column in range(FEATURES)
        ),
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


@dataclass(frozen=True)
class _Population:
    batch: XTYBatch
    baseline: torch.Tensor
    effect: torch.Tensor


def _population(rows: int, *, seed: int, row_offset: int) -> _Population:
    """FixMatch §6.1 with the reviewed 0.25/0.75 assignment overlap."""
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < 0.5).float()
    sign = 2.0 * cluster - 1.0
    x = epsilon_x.clone()
    x[:, :4] = CLUSTER_SIGNAL * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = LOW_PROPENSITY + (1.0 - 2.0 * LOW_PROPENSITY) * cluster
    t = (u_t < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    y = baseline + t * effect + 0.5 * epsilon_y

    return _Population(
        batch=XTYBatch(
            x=x,
            t=t,
            y=y,
            t_observed=torch.ones(rows, dtype=torch.bool),
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        baseline=baseline,
        effect=effect,
    )


def _dataset(train: XTYBatch) -> Dataset:
    return Dataset(
        schema=_schema(),
        rows=train,
        assignments={"train": torch.arange(train.batch_size)},
    )


def _exact_arm(recipe: Recipe) -> Recipe:
    """Same serving graph and fit, replacing q's two terms by exact marginalisation."""
    graph = ComponentGraph(
        [component for component in recipe.system.components if component.name != "categorical_posterior"]
    )
    stage = recipe.program[0]
    exact = Weighted(
        MissingTreatmentMarginalNLL(grad_path="both"),
        weight=1.0,
        reduction="population",
    )
    changed_stage = replace(
        stage,
        objectives=(*stage.objectives[:2], exact),
        trainable=("mlp_encoder", "tarnet_head", "categorical_propensity"),
    )
    return replace(recipe, system=graph, program=Program((changed_stage,)))


def _alpha_zero(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    posterior = replace(stage.objectives[-1], weight=Constant(0.0))
    return replace(
        recipe,
        program=Program((replace(stage, objectives=(*stage.objectives[:-1], posterior)),)),
    )


def _scaled_test(test: XTYBatch, result: StageResult) -> XTYBatch:
    population = result.population
    assert population is not None
    location = population.statistics["y_location"]
    scale = population.statistics["y_scale"]
    return test.replace(y=(test.y - location) / scale)


def _term(result: StageResult, step: int, name: str) -> object:
    return next(term for term in result.records[step].terms if term.name == name)


def _gap(result: StageResult, step: int) -> float:
    term = _term(result, step, "variational_treatment_elbo")
    return float(term.diagnostics["amortisation_gap"])  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    marginal_nll: float
    treatment_nll: float
    outcome_nll: float
    posterior_nll: float | None
    amortisation_gap: float | None


def _evaluate(run: CompiledRun, result: StageResult, test: XTYBatch) -> _Metrics:
    scaled = _scaled_test(test, result)
    with torch.no_grad():
        values = run.graph.evaluate(scaled, schema=run.recipe.schema, only=run.graph.names)
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(outcome, GaussianOutcome)
        candidates = torch.arange(2).expand(scaled.batch_size, 2)
        log_joint = propensity.log_prob(candidates) + outcome.log_prob(
            scaled.y, candidates
        )
        marginal_nll = float(-torch.logsumexp(log_joint, dim=-1).mean())
        treatment_nll = float(F.nll_loss(propensity.log_probs, scaled.t))
        outcome_nll = float(-outcome.log_prob(scaled.y, scaled.t).mean())

        posterior_value = values.get(Port.T_GIVEN_XY)
        if isinstance(posterior_value, CategoricalTreatment):
            posterior_nll = float(F.nll_loss(posterior_value.log_probs, scaled.t))
            log_model_posterior = log_joint - torch.logsumexp(
                log_joint, dim=-1, keepdim=True
            )
            gap = (
                posterior_value.probs
                * (posterior_value.log_probs - log_model_posterior)
            ).sum(dim=-1)
            amortisation_gap = float(gap.mean())
        else:
            posterior_nll = None
            amortisation_gap = None

    return _Metrics(
        run=run,
        result=result,
        marginal_nll=marginal_nll,
        treatment_nll=treatment_nll,
        outcome_nll=outcome_nll,
        posterior_nll=posterior_nll,
        amortisation_gap=amortisation_gap,
    )


def _fit(recipe: Recipe, train: XTYBatch, test: XTYBatch) -> _Metrics:
    run = compile(recipe)
    result = run_stage(run, "elbo_fit", _dataset(train), seed=94_010)
    return _evaluate(run, result, test)


@dataclass(frozen=True)
class _Fits:
    variational: _Metrics
    exact: _Metrics
    alpha_zero: _Metrics
    test_raw: _Population


@pytest.fixture(scope="module")
def fits() -> _Fits:
    schema = _schema()
    train = _population(TRAIN_ROWS, seed=94_001, row_offset=0)
    test = _population(TEST_ROWS, seed=94_003, row_offset=10_000)

    torch.manual_seed(94_006)
    variational_recipe = variational_treatment(schema)
    torch.manual_seed(94_006)
    exact_recipe = _exact_arm(variational_treatment(schema))
    torch.manual_seed(94_006)
    alpha_zero_recipe = _alpha_zero(variational_treatment(schema))

    serving = {"mlp_encoder", "tarnet_head", "categorical_propensity"}
    variational_state = variational_recipe.system.state_dict()
    exact_state = exact_recipe.system.state_dict()
    for name, value in variational_state.items():
        if name.split(".", 1)[0] in serving:
            assert torch.equal(value, exact_state[name])

    return _Fits(
        variational=_fit(variational_recipe, train.batch, test.batch),
        exact=_fit(exact_recipe, train.batch, test.batch),
        alpha_zero=_fit(alpha_zero_recipe, train.batch, test.batch),
        test_raw=test,
    )


def test_both_paired_arms_reduce_their_mixed_training_loss(fits: _Fits) -> None:
    for metrics in (fits.variational, fits.exact):
        early = sum(metrics.result.trace[:100]) / 100
        late = sum(metrics.result.trace[-100:]) / 100
        assert late < early


def test_the_bound_holds_on_every_logged_training_step(fits: _Fits) -> None:
    for step in range(STEPS):
        term = _term(fits.variational.result, step, "variational_treatment_elbo")
        exact = float(term.diagnostics["exact_marginal_nll"])  # type: ignore[attr-defined]
        gap = float(term.diagnostics["amortisation_gap"])  # type: ignore[attr-defined]
        assert float(term.value) + 1e-6 >= exact  # type: ignore[attr-defined]
        assert gap >= -1e-6


def test_the_amortisation_gap_is_finite_and_learns_over_the_run(fits: _Fits) -> None:
    first = sum(_gap(fits.variational.result, step) for step in range(50)) / 50
    last = sum(_gap(fits.variational.result, step) for step in range(STEPS - 50, STEPS)) / 50
    heldout = fits.variational.amortisation_gap
    assert heldout is not None
    assert math.isfinite(first) and math.isfinite(last) and math.isfinite(heldout)
    assert last < first


def test_posterior_supervision_ablation_runs_and_changes_q(fits: _Fits) -> None:
    supervised = fits.variational.posterior_nll
    unsupervised = fits.alpha_zero.posterior_nll
    assert supervised is not None and unsupervised is not None
    assert math.isfinite(supervised) and math.isfinite(unsupervised)
    assert supervised != pytest.approx(unsupervised, abs=1e-5)


def test_overlap_fixture_has_a_real_bayes_posterior_advantage(fits: _Fits) -> None:
    batch = fits.test_raw.batch
    cluster_probability = torch.sigmoid(2.5 * batch.x[:, :4].sum(dim=-1))
    p1 = LOW_PROPENSITY + (1.0 - 2.0 * LOW_PROPENSITY) * cluster_probability
    propensity = torch.stack((1.0 - p1, p1), dim=-1)

    means = torch.stack(
        (fits.test_raw.baseline, fits.test_raw.baseline + fits.test_raw.effect),
        dim=-1,
    )
    residual = (batch.y[:, None] - means) / 0.5
    log_py = -0.5 * residual.square() - math.log(0.5) - 0.5 * math.log(2.0 * math.pi)
    log_joint = propensity.log() + log_py
    posterior = torch.softmax(log_joint, dim=-1)

    propensity_nll = float(F.nll_loss(propensity.log(), batch.t))
    posterior_nll = float(F.nll_loss(posterior.log(), batch.t))
    assert posterior_nll < propensity_nll
    assert posterior_nll < propensity_nll - 0.05


def test_variational_and_exact_serving_fits_remain_in_the_same_regime(fits: _Fits) -> None:
    ratio = fits.variational.marginal_nll / fits.exact.marginal_nll
    assert math.isfinite(ratio)
    assert 0.5 < ratio < 1.5
    assert fits.variational.posterior_nll is not None
    assert fits.variational.posterior_nll < 1.0
