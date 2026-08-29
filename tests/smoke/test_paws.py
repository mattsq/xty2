"""Tier 1 — PAWS pretraining and the card's declared mechanism diagnostics.

The primary fixture runs the exact 1,000-step pretraining stage. Secondary
fits keep that budget and change one declared item at a time: me-max weight,
support size, or the skewed treatment prior. These are wiring measurements,
not reproduction claims; the ten-seed paired downstream target remains Tier 2.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from xty2.core import (
    CompiledRun,
    Dataset,
    FeatureSpec,
    MissingnessSpec,
    OutcomeSpec,
    Port,
    Program,
    Quota,
    QuotaSampler,
    Recipe,
    Schema,
    TrainingPopulation,
    XTYBatch,
    compile,
)
from xty2.recipes import paws
from xty2.recipes.paws import PRETRAIN_STEPS
from xty2.training import ObjectiveLog, StageResult, run_stage
from xty2.training.loading import build_population

FEATURES = 6
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048
PRIMARY_LABELS = 64
SKEW_LABELS = 160
PRIMARY_SEED = 90_001
TEST_SEED = 90_003
INITIALISATION_SEED = 90_006
RUN_SEED = 90_010
CONSISTENCY = "support_set_pseudo_label_consistency"
ME_MAX = "mean_entropy_maximisation"


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
    bayes_treatment: Tensor


def _primary_population(rows: int, *, seed: int, row_offset: int) -> _Population:
    """Card §6.1's inherited two-cluster treatment DGP."""
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < 0.5).long()
    sign = 2.0 * cluster.float() - 1.0
    x = epsilon_x.clone()
    x[:, :4] = 0.45 * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = 0.02 + 0.96 * cluster.float()
    treatment = (u_t < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    outcome = baseline + treatment * effect + 0.5 * epsilon_y
    return _Population(
        batch=XTYBatch(
            x=x,
            t=treatment,
            y=outcome,
            t_observed=torch.ones(rows, dtype=torch.bool),
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        bayes_treatment=cluster,
    )


def _skewed_population(rows: int, *, seed: int, row_offset: int) -> _Population:
    """A learnable binary treatment with a 15% marginal."""
    generator = torch.Generator().manual_seed(seed)
    latent = torch.randn(rows, generator=generator)
    noise = torch.randn(rows, FEATURES, generator=generator)
    # The standard-normal 85th percentile gives P(t=1) approximately 0.15.
    treatment = (latent > 1.036433389).long()
    x = noise
    x[:, :4] = latent[:, None] + 0.35 * noise[:, :4]
    outcome = 0.4 * x[:, 0] - 0.2 * x[:, 4] + treatment.float()
    return _Population(
        batch=XTYBatch(
            x=x,
            t=treatment,
            y=outcome,
            t_observed=torch.ones(rows, dtype=torch.bool),
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        bayes_treatment=treatment,
    )


def _dataset(population: _Population) -> Dataset:
    return Dataset(
        schema=_schema(),
        rows=population.batch,
        assignments={"train": torch.arange(population.batch.batch_size)},
    )


def _without_me_max(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    objectives = (
        stage.objectives[0],
        replace(stage.objectives[1], weight=0.0),
    )
    return replace(
        recipe,
        program=Program((replace(stage, objectives=objectives), recipe.program[1])),
    )


def _with_support_size(recipe: Recipe, per_treatment: int) -> Recipe:
    sampler = QuotaSampler(
        quotas=(
            Quota(rows="t_observed", size=per_treatment, stratify="t"),
            Quota(rows="t_missing", size=128),
        )
    )
    stage = replace(recipe.program[0], sampler=sampler)
    return replace(recipe, program=Program((stage, recipe.program[1])))


def _with_label_budget(recipe: Recipe, labels: int) -> Recipe:
    assert recipe.data is not None
    data = replace(
        recipe.data,
        missingness=MissingnessSpec(mechanism="mcar", observed=labels),
    )
    return replace(recipe, data=data)


def _term(result: StageResult, step: int, name: str) -> ObjectiveLog:
    return next(term for term in result.records[step].terms if term.name == name)


def _mean_term(result: StageResult, steps: range, name: str) -> float:
    return sum(_term(result, step, name).value for step in steps) / len(steps)


def _entropy(result: StageResult) -> float:
    term = _term(result, -1, ME_MAX)
    return float(term.diagnostics["marginal_entropy"])


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    before_nll: float
    after_nll: float
    before_accuracy: float
    after_accuracy: float
    predicted_marginal: tuple[float, float]
    large_flip_rate: float
    small_flip_rate: float


def _paws_nn(
    run: CompiledRun, population: TrainingPopulation, test: XTYBatch
) -> tuple[float, float, tuple[float, float]]:
    with torch.no_grad():
        support_values = run.graph.evaluate(
            population.rows,
            schema=run.recipe.schema,
            only=("mlp_encoder", "projection_head"),
        )
        query_values = run.graph.evaluate(
            test,
            schema=run.recipe.schema,
            only=("mlp_encoder", "projection_head"),
        )
        support = support_values[Port.X_PROJ]
        query = query_values[Port.X_PROJ]
        assert isinstance(support, Tensor) and isinstance(query, Tensor)
        rows = population.rows.t_observed.nonzero(as_tuple=False).flatten()
        labels = population.rows.t.index_select(0, rows)
        smoothed = 0.9 * F.one_hot(labels, 2).to(query.dtype) + 0.05
        probabilities = (
            torch.softmax(
                F.normalize(query, dim=-1)
                @ F.normalize(support.index_select(0, rows), dim=-1).T
                / 0.1,
                dim=-1,
            )
            @ smoothed
        )
        return (
            float(F.nll_loss(probabilities.log(), test.t)),
            float((probabilities.argmax(dim=-1) == test.t).float().mean()),
            tuple(float(value) for value in probabilities.mean(dim=0)),
        )  # type: ignore[return-value]


def _flip_rates(
    run: CompiledRun,
    population: TrainingPopulation,
    bayes_treatment: Tensor,
) -> tuple[float, float]:
    # On the inherited cluster DGP, the sign of the first four columns is the
    # Bayes class. This checks what the two declared corruption strengths do to
    # that class before any parameter is trained.
    rates: list[float] = []
    for key, name in enumerate(("paws_large_x", "paws_small_x"), start=7_000):
        viewed = run.recipe.view(name).apply(
            population.rows,
            run.recipe.schema,
            rng_key=key,
            population=population,
        )
        predicted = (viewed.x[:, :4].mean(dim=1) > 0.0).long()
        rates.append(float((predicted != bayes_treatment).float().mean()))
    return rates[0], rates[1]


def _run(recipe: Recipe, train: _Population, test: _Population) -> _Metrics:
    data = _dataset(train)
    torch.manual_seed(INITIALISATION_SEED)
    run = compile(recipe)
    assert recipe.data is not None
    population = build_population(data, recipe.data, seed=RUN_SEED)
    counts = torch.bincount(population.rows.t[population.rows.t_observed], minlength=2)
    assert int(counts.min()) >= 16
    before_nll, before_accuracy, _ = _paws_nn(run, population, test.batch)
    flip_rates = _flip_rates(run, population, train.bayes_treatment)
    result = run_stage(run, "pretrain", data, seed=RUN_SEED)
    assert result.population is not None
    after_nll, after_accuracy, marginal = _paws_nn(run, result.population, test.batch)
    return _Metrics(
        run=run,
        result=result,
        before_nll=before_nll,
        after_nll=after_nll,
        before_accuracy=before_accuracy,
        after_accuracy=after_accuracy,
        predicted_marginal=marginal,
        large_flip_rate=flip_rates[0],
        small_flip_rate=flip_rates[1],
    )


def _fresh(transform: Callable[[Recipe], Recipe] | None = None) -> Recipe:
    """One shared initialisation for every diagnostic arm."""
    torch.manual_seed(INITIALISATION_SEED)
    recipe = paws(_schema())
    if transform is None:
        return recipe
    return transform(recipe)


@dataclass(frozen=True)
class _Evidence:
    primary: _Metrics
    no_me_max: _Metrics
    support_8: _Metrics
    skewed: _Metrics
    skewed_no_me_max: _Metrics


@pytest.fixture(scope="module")
def evidence() -> _Evidence:
    primary_train = _primary_population(TRAIN_ROWS, seed=PRIMARY_SEED, row_offset=0)
    primary_test = _primary_population(TEST_ROWS, seed=TEST_SEED, row_offset=10_000)
    skew_train = _skewed_population(
        TRAIN_ROWS, seed=PRIMARY_SEED + 100, row_offset=20_000
    )
    skew_test = _skewed_population(TEST_ROWS, seed=TEST_SEED + 100, row_offset=30_000)

    base = _fresh()
    no_me_max = _fresh(_without_me_max)
    support_8 = _fresh(lambda recipe: _with_support_size(recipe, 8))
    skew_base = _fresh(lambda recipe: _with_label_budget(recipe, SKEW_LABELS))
    skew_no_me_max = _fresh(
        lambda recipe: _without_me_max(_with_label_budget(recipe, SKEW_LABELS))
    )
    return _Evidence(
        primary=_run(base, primary_train, primary_test),
        no_me_max=_run(no_me_max, primary_train, primary_test),
        support_8=_run(support_8, primary_train, primary_test),
        skewed=_run(skew_base, skew_train, skew_test),
        skewed_no_me_max=_run(skew_no_me_max, skew_train, skew_test),
    )


def test_the_primary_consistency_value_falls(evidence: _Evidence) -> None:
    early = _mean_term(evidence.primary.result, range(50), CONSISTENCY)
    late = _mean_term(
        evidence.primary.result,
        range(PRETRAIN_STEPS - 50, PRETRAIN_STEPS),
        CONSISTENCY,
    )
    assert late < early


def test_diagnostic_arms_start_from_the_same_parameters(evidence: _Evidence) -> None:
    assert evidence.primary.before_nll == evidence.no_me_max.before_nll
    assert evidence.primary.before_nll == evidence.support_8.before_nll
    assert evidence.skewed.before_nll == evidence.skewed_no_me_max.before_nll


def test_the_support_classifier_improves_on_held_out_rows(evidence: _Evidence) -> None:
    assert evidence.primary.after_nll < evidence.primary.before_nll
    assert evidence.primary.after_accuracy > evidence.primary.before_accuracy


def test_the_two_view_strengths_have_the_declared_order(evidence: _Evidence) -> None:
    assert evidence.primary.large_flip_rate < evidence.primary.small_flip_rate


def test_me_max_keeps_the_primary_marginal_non_collapsed(evidence: _Evidence) -> None:
    assert _entropy(evidence.primary.result) >= 0.95 * math.log(2)
    # The ablation is deliberately measured even if its direction differs from
    # the image result; its held-out and entropy values are recorded in §6.2.
    assert math.isfinite(_entropy(evidence.no_me_max.result))
    assert math.isfinite(evidence.no_me_max.after_nll)


def test_the_skewed_falsification_pair_runs_with_a_fillable_support(
    evidence: _Evidence,
) -> None:
    assert sum(evidence.skewed.predicted_marginal) == pytest.approx(1.0)
    assert sum(evidence.skewed_no_me_max.predicted_marginal) == pytest.approx(1.0)
    assert math.isfinite(evidence.skewed.after_nll)
    assert math.isfinite(evidence.skewed_no_me_max.after_nll)


def test_support_size_sensitivity_is_measured_at_8_and_16(evidence: _Evidence) -> None:
    primary_batch = evidence.primary.run.plan.hyperparameters[
        "optimisation.batch_size"
    ]["pretrain"]
    smaller_batch = evidence.support_8.run.plan.hyperparameters[
        "optimisation.batch_size"
    ]["pretrain"]
    assert (primary_batch, smaller_batch) == (160, 144)
    assert math.isfinite(evidence.support_8.after_nll)
