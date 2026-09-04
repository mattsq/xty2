"""SoftMatch's paired continuous-weight-versus-constant-gate benchmark."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    Weighted,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    ClusterPopulation,
    column,
    configure_worker,
    continuous_schema,
    on_the_training_scale,
    parallel_replicates,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec
from xty2.objectives import (
    ConfidenceGaussian,
    PseudoLabelTreatmentNLL,
    SoftWeightedTreatmentNLL,
)
from xty2.recipes import softmatch
from xty2.recipes.fixmatch import STRONG_X, WEAK_X
from xty2.recipes.softmatch import SOFTMATCH_TERM
from xty2.training import StageResult, run_stage

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 90_000
_CONSTANT_TAU = 0.95


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten paired fits predeclared by ``softmatch.md`` section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "fixmatch.md §6.1's project-local seed-locked two-cluster XTY "
                "DGP (6 features, K=2), unmodified"
            ),
            "variant": (
                "paired fit against a constant-gate arm — this recipe with eq. "
                "(2)'s lambda replaced by FixMatch's indicator at tau = 0.95, "
                "so both arms share deviation 2's views — same seeds and same "
                "batches"
            ),
            "split": (
                "1024 train rows with 64 observed treatments, 2048 held-out "
                "rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio on the EMA parameters, SoftMatch "
                "over the constant-gate arm; the paper's own quantity f(p) "
                "(eq. 3) and quality g(p) (eq. 4) on the same batches as the "
                "trade-off guardrails"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "ratio < 1.0 in mean on both the EMA and the trained parameters, "
                "by at least one standard error; terminal quantity f(p) above "
                "the paired arm's terminal mask rate by at least one standard "
                "error; terminal lambda-weighted impurity 1 - g(p) no worse than "
                "1.25x the paired arm's retained-label impurity and below 0.15 "
                "absolutely; held-out outcome NLL within 1.05x of the paired arm; "
                "terminal mu_t above 1/K by at least one standard error and "
                "terminal sigma_t^2 below its 1.0 initialisation"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"softmatch card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "ema_treatment_NLL_ratio", column(rows, "ema_ratio"), 1.0
            ),
            MetricResult.upper_bound(
                "trained_treatment_NLL_ratio", column(rows, "trained_ratio"), 1.0
            ),
            MetricResult.lower_bound(
                "terminal_quantity_advantage",
                column(rows, "quantity_advantage"),
                0.0,
            ),
            MetricResult.upper_bound(
                "weighted_impurity_vs_1.25x_gate",
                column(rows, "impurity_excess"),
                0.0,
            ),
            MetricResult.upper_bound(
                "weighted_pseudo_label_impurity",
                column(rows, "soft_impurity"),
                0.15,
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio", column(rows, "outcome_ratio"), 1.05
            ),
            MetricResult.lower_bound("terminal_mu_hat", column(rows, "mu_hat"), 0.5),
            MetricResult.upper_bound(
                "terminal_sigma_squared", column(rows, "sigma_squared"), 1.0
            ),
            MetricResult.information(
                "terminal_soft_quantity", column(rows, "soft_quantity")
            ),
            MetricResult.information(
                "terminal_constant_mask_rate", column(rows, "constant_quantity")
            ),
            MetricResult.information(
                "constant_retained_label_impurity",
                column(rows, "constant_impurity"),
            ),
        ),
        interpretation=(
            "This is the predeclared project-local SoftMatch mechanism target: "
            "whether continuous Gaussian weights improve missing-treatment "
            "classification while raising weighted quantity without losing the "
            "declared pseudo-label quality. It is not an image benchmark."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = two_cluster_population(_TRAIN_ROWS, seed=base + 1, row_offset=0)
    test = two_cluster_population(_TEST_ROWS, seed=base + 2, row_offset=10_000)
    data = training_dataset(schema, train.batch)

    torch.manual_seed(base + 6)
    soft_recipe = softmatch(schema)
    torch.manual_seed(base + 6)
    constant_recipe = _constant_gate(softmatch(schema))
    for name, value in soft_recipe.system.state_dict().items():
        if not torch.equal(value, constant_recipe.system.state_dict()[name]):
            raise RuntimeError(f"softmatch paired initial state differs at {name!r}")

    soft_run = compile(soft_recipe)
    constant_run = compile(constant_recipe)
    stage_seed = base + 10_000
    soft_result = run_stage(soft_run, "joint_fit", data, seed=stage_seed)
    constant_result = run_stage(constant_run, "joint_fit", data, seed=stage_seed)
    if not torch.equal(
        soft_result.checkpoint.trained_on_row_ids,
        constant_result.checkpoint.trained_on_row_ids,
    ):
        raise RuntimeError("softmatch paired arms saw different training rows")

    soft = _evaluate(soft_run, soft_result, test, soft=True)
    constant = _evaluate(constant_run, constant_result, test, soft=False)
    for name in ("ema_treatment_nll", "trained_treatment_nll", "outcome_nll"):
        if constant[name] <= 0.0:
            raise RuntimeError(f"constant arm produced non-positive {name}")
    return {
        "ema_ratio": soft["ema_treatment_nll"] / constant["ema_treatment_nll"],
        "trained_ratio": (
            soft["trained_treatment_nll"] / constant["trained_treatment_nll"]
        ),
        "outcome_ratio": soft["outcome_nll"] / constant["outcome_nll"],
        "quantity_advantage": soft["quantity"] - constant["quantity"],
        "impurity_excess": soft["impurity"] - 1.25 * constant["impurity"],
        "soft_quantity": soft["quantity"],
        "constant_quantity": constant["quantity"],
        "soft_impurity": soft["impurity"],
        "constant_impurity": constant["impurity"],
        "mu_hat": soft["mu_hat"],
        "sigma_squared": soft["sigma_squared"],
    }


def _constant_gate(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    objective = stage.objectives[2].objective
    if not isinstance(objective, SoftWeightedTreatmentNLL):
        raise RuntimeError("softmatch's third term is no longer its weighted loss")
    gated = Weighted(
        PseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=WEAK_X,
            prediction=STRONG_X,
            threshold=_CONSTANT_TAU,
            sharpening="hard",
            stop_grad="target",
            rows="all",
        ),
        weight=stage.objectives[2].weight,
        reduction=stage.objectives[2].reduction,
    )
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(*stage.objectives[:2], gated, *stage.objectives[3:]),
                ),
            )
        ),
    )


def _evaluate(
    run: CompiledRun,
    result: StageResult,
    test: ClusterPopulation,
    *,
    soft: bool,
) -> dict[str, float]:
    population = result.population
    if population is None or result.teacher is None:
        raise RuntimeError("softmatch benchmark expected population and EMA teacher")
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        teacher_values = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        teacher_propensity = teacher_values[Port.T_GIVEN_X]
        teacher_outcome = teacher_values[Port.Y_GIVEN_XT]
        trained_values = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        trained_propensity = trained_values[Port.T_GIVEN_X]
        train_values = run.graph.evaluate(
            population.rows, schema=schema, only=run.graph.names
        )
        train_propensity = train_values[Port.T_GIVEN_X]
        if not all(
            isinstance(value, CategoricalTreatment)
            for value in (teacher_propensity, trained_propensity, train_propensity)
        ) or not isinstance(teacher_outcome, GaussianOutcome):
            raise TypeError("softmatch benchmark expected its reviewed P5 heads")
        assert isinstance(teacher_propensity, CategoricalTreatment)
        assert isinstance(trained_propensity, CategoricalTreatment)
        assert isinstance(train_propensity, CategoricalTreatment)
        labels = train_propensity.probs.argmax(dim=-1)
        if soft:
            state = result.objective_states.get(SOFTMATCH_TERM)
            if not isinstance(state, ConfidenceGaussian):
                raise RuntimeError("softmatch result did not expose Gaussian state")
            weights = state.weights(train_propensity.probs)
            mu_hat, sigma_squared = state.mean, state.variance
        else:
            weights = (
                train_propensity.probs.max(dim=-1).values >= _CONSTANT_TAU
            ).float()
            mu_hat = sigma_squared = 0.0
        wrong = (labels != population.rows.t).float()
        impurity = (
            float((weights * wrong).sum() / weights.sum())
            if float(weights.sum()) > 0.0
            else 0.0
        )
        return {
            "ema_treatment_nll": float(
                F.nll_loss(teacher_propensity.log_probs, scaled.t)
            ),
            "trained_treatment_nll": float(
                F.nll_loss(trained_propensity.log_probs, scaled.t)
            ),
            "outcome_nll": float(-teacher_outcome.log_prob(scaled.y, scaled.t).mean()),
            "quantity": float(weights.mean()),
            "impurity": impurity,
            "mu_hat": mu_hat,
            "sigma_squared": sigma_squared,
        }


__all__ = ["run"]
