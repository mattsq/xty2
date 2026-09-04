"""SoftMatch's weighting and Uniform-Alignment fidelity benchmark.

The original balanced binary target established that the truncated Gaussian
beats a constant gate, but made Uniform Alignment almost the identity. This
runner uses the repository's reviewed K=4 skewed-prior fixture and a balanced
held-out population so the paper's second named contribution is load-bearing.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor
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
    cluster_population,
    column,
    configure_worker,
    continuous_schema,
    on_the_training_scale,
    parallel_replicates,
    training_dataset,
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
_CLASSES = 4
_TRAIN_PRIOR = (0.55, 0.25, 0.13, 0.07)
_EFFECTS = (0.0, 1.0, 0.4, 1.6)
_CONSTANT_TAU = 0.95


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten three-way paired fits predeclared by card section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked K=4 cluster XTY DGP (6 features), "
                "training prior (0.55, 0.25, 0.13, 0.07), balanced held-out "
                "prior, regular-simplex centres at separation 1.8"
            ),
            "variant": (
                "three paired fits — SoftMatch with Uniform Alignment, the "
                "paper's all-class without-UA ablation (eq. 5), and the "
                "constant gate at tau = 0.95 — same initial states, seeds, "
                "views and batches"
            ),
            "split": (
                "1024 long-tailed train rows with 64 MCAR-observed treatments, "
                "2048 balanced held-out rows with every treatment observed"
            ),
            "metric": (
                "balanced held-out macro p(t|x) NLL ratios on EMA parameters, "
                "UA over no-UA and SoftMatch over the constant gate; reduction "
                "in class-wise mean-weight dispersion from applying UA to the "
                "same terminal predictions; quantity f(p) and quality g(p) "
                "guardrails"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "UA/no-UA EMA macro NLL ratio < 1.0 in mean by at least one "
                "standard error; SoftMatch/constant EMA macro NLL ratio < 1.0 "
                "by the same rule; class-wise mean-weight coefficient-of-"
                "variation after UA below the unaligned value on the same "
                "predictions by at least one standard error; terminal quantity "
                "f(p) above the constant gate's mask rate by at least one "
                "standard error; Gaussian-weighted pseudo-label impurity below "
                "the same model's unweighted pseudo-label impurity by at least "
                "one standard error; held-out outcome NLL within 1.05x of the "
                "constant arm; terminal mu_t above 1/K by at least one standard "
                "error and terminal sigma_t^2 below its 1.0 initialisation"
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
                "ua_vs_no_ua_ema_macro_NLL_ratio",
                column(rows, "ua_no_ua_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "softmatch_vs_constant_ema_macro_NLL_ratio",
                column(rows, "soft_constant_ratio"),
                1.0,
            ),
            MetricResult.lower_bound(
                "class_weight_CV_reduction_from_UA",
                column(rows, "class_weight_cv_reduction"),
                0.0,
            ),
            MetricResult.lower_bound(
                "terminal_quantity_advantage",
                column(rows, "quantity_advantage"),
                0.0,
            ),
            MetricResult.lower_bound(
                "weighted_impurity_advantage",
                column(rows, "weighted_impurity_advantage"),
                0.0,
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio", column(rows, "outcome_ratio"), 1.05
            ),
            MetricResult.lower_bound("terminal_mu_hat", column(rows, "mu_hat"), 0.25),
            MetricResult.upper_bound(
                "terminal_sigma_squared", column(rows, "sigma_squared"), 1.0
            ),
            MetricResult.information(
                "aligned_class_weight_CV", column(rows, "aligned_class_weight_cv")
            ),
            MetricResult.information(
                "unaligned_class_weight_CV",
                column(rows, "unaligned_class_weight_cv"),
            ),
            MetricResult.information(
                "terminal_soft_quantity", column(rows, "soft_quantity")
            ),
            MetricResult.information(
                "terminal_constant_mask_rate", column(rows, "constant_quantity")
            ),
            MetricResult.information(
                "soft_weighted_pseudo_label_impurity",
                column(rows, "soft_weighted_impurity"),
            ),
            MetricResult.information(
                "soft_unweighted_pseudo_label_impurity",
                column(rows, "soft_unweighted_impurity"),
            ),
            MetricResult.information(
                "constant_retained_label_impurity",
                column(rows, "constant_impurity"),
            ),
        ),
        interpretation=(
            "This revised project-local target exercises both named SoftMatch "
            "contributions. The skewed K=4 training marginal gives Uniform "
            "Alignment work to do; balanced held-out macro NLL asks whether "
            "that correction generalises rather than merely fitting the skew. "
            "The class-weight diagnostic is appendix A.7's before/after-UA "
            "measurement on the same predictions."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6, treatments=_CLASSES)
    train = cluster_population(
        _TRAIN_ROWS,
        seed=base + 1,
        row_offset=0,
        classes=_CLASSES,
        prior=_TRAIN_PRIOR,
        effects=_EFFECTS,
    )
    test = cluster_population(
        _TEST_ROWS,
        seed=base + 2,
        row_offset=10_000,
        classes=_CLASSES,
        effects=_EFFECTS,
    )
    data = training_dataset(schema, train.batch)

    recipes: dict[str, Recipe] = {}
    for name in ("soft", "no_ua", "constant"):
        torch.manual_seed(base + 6)
        candidate = softmatch(schema)
        if name == "no_ua":
            candidate = _without_uniform_alignment(candidate)
        elif name == "constant":
            candidate = _constant_gate(candidate)
        recipes[name] = candidate
    reference = recipes["soft"].system.state_dict()
    for arm, candidate in recipes.items():
        for name, value in reference.items():
            if not torch.equal(value, candidate.system.state_dict()[name]):
                raise RuntimeError(
                    f"softmatch paired initial state differs in {arm!r} at {name!r}"
                )

    runs = {name: compile(recipe) for name, recipe in recipes.items()}
    stage_seed = base + 10_000
    results = {
        name: run_stage(run, "joint_fit", data, seed=stage_seed)
        for name, run in runs.items()
    }
    trained_rows = results["soft"].checkpoint.trained_on_row_ids
    for arm, result in results.items():
        if not torch.equal(trained_rows, result.checkpoint.trained_on_row_ids):
            raise RuntimeError(f"softmatch arm {arm!r} saw different training rows")

    soft = _evaluate(runs["soft"], results["soft"], test, mode="soft")
    no_ua = _evaluate(runs["no_ua"], results["no_ua"], test, mode="no_ua")
    constant = _evaluate(runs["constant"], results["constant"], test, mode="constant")
    for name in ("ema_macro_nll", "outcome_nll"):
        if no_ua[name] <= 0.0 or constant[name] <= 0.0:
            raise RuntimeError(f"comparison arm produced non-positive {name}")
    return {
        "ua_no_ua_ratio": soft["ema_macro_nll"] / no_ua["ema_macro_nll"],
        "soft_constant_ratio": soft["ema_macro_nll"] / constant["ema_macro_nll"],
        "outcome_ratio": soft["outcome_nll"] / constant["outcome_nll"],
        "class_weight_cv_reduction": (
            soft["unaligned_class_weight_cv"] - soft["aligned_class_weight_cv"]
        ),
        "quantity_advantage": soft["quantity"] - constant["quantity"],
        "weighted_impurity_advantage": (
            soft["unweighted_impurity"] - soft["weighted_impurity"]
        ),
        "aligned_class_weight_cv": soft["aligned_class_weight_cv"],
        "unaligned_class_weight_cv": soft["unaligned_class_weight_cv"],
        "soft_quantity": soft["quantity"],
        "constant_quantity": constant["quantity"],
        "soft_weighted_impurity": soft["weighted_impurity"],
        "soft_unweighted_impurity": soft["unweighted_impurity"],
        "constant_impurity": constant["weighted_impurity"],
        "mu_hat": soft["mu_hat"],
        "sigma_squared": soft["sigma_squared"],
    }


def _without_uniform_alignment(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    weighted = stage.objectives[2]
    objective = weighted.objective
    if not isinstance(objective, SoftWeightedTreatmentNLL):
        raise RuntimeError("softmatch's third term is no longer its weighted loss")
    changed = replace(
        weighted,
        objective=replace(
            objective,
            weighting=replace(objective.weighting, alignment="none"),
        ),
    )
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(
                        *stage.objectives[:2],
                        changed,
                        *stage.objectives[3:],
                    ),
                ),
            )
        ),
    )


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


def _macro_mean(values: Tensor, labels: Tensor, classes: int) -> float:
    means = []
    for level in range(classes):
        selected = values[labels == level]
        if not selected.numel():
            raise RuntimeError(f"balanced held-out draw has no rows for class {level}")
        means.append(selected.mean())
    return float(torch.stack(means).mean())


def _class_weight_cv(weights: Tensor, labels: Tensor, classes: int) -> float:
    """Coefficient of variation of mean weights grouped by pseudo-label."""
    totals = torch.zeros(classes, device=weights.device, dtype=weights.dtype)
    totals.scatter_add_(0, labels, weights)
    counts = torch.bincount(labels, minlength=classes).to(weights)
    means = torch.where(counts > 0, totals / counts.clamp_min(1.0), 0.0)
    mean = means.mean()
    if float(mean) <= 0.0:
        raise RuntimeError("class-wise mean weights are all zero")
    return float(means.std(unbiased=False) / mean)


def _evaluate(
    run: CompiledRun,
    result: StageResult,
    test: ClusterPopulation,
    *,
    mode: str,
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
        train_values = run.graph.evaluate(
            population.rows, schema=schema, only=run.graph.names
        )
        train_propensity = train_values[Port.T_GIVEN_X]
        if (
            not isinstance(teacher_propensity, CategoricalTreatment)
            or not isinstance(train_propensity, CategoricalTreatment)
            or not isinstance(teacher_outcome, GaussianOutcome)
        ):
            raise TypeError("softmatch benchmark expected its reviewed P5 heads")

        labels = train_propensity.probs.argmax(dim=-1)
        wrong = (labels != population.rows.t).float()
        raw_weights = torch.ones_like(wrong)
        aligned_weights = raw_weights
        mu_hat = sigma_squared = 0.0
        if mode in ("soft", "no_ua"):
            state = result.objective_states.get(SOFTMATCH_TERM)
            if not isinstance(state, ConfidenceGaussian):
                raise RuntimeError("softmatch result did not expose Gaussian state")
            aligned_weights = state.weights(train_propensity.probs)
            raw_weights = state.weights(train_propensity.probs, apply_alignment=False)
            weights = aligned_weights
            mu_hat, sigma_squared = state.mean, state.variance
        elif mode == "constant":
            weights = (
                train_propensity.probs.max(dim=-1).values >= _CONSTANT_TAU
            ).float()
        else:
            raise ValueError(f"unknown softmatch benchmark mode {mode!r}")

        weight_sum = weights.sum()
        weighted_impurity = (
            float((weights * wrong).sum() / weight_sum)
            if float(weight_sum) > 0.0
            else 0.0
        )
        per_row_nll = F.nll_loss(
            teacher_propensity.log_probs, scaled.t, reduction="none"
        )
        return {
            "ema_macro_nll": _macro_mean(per_row_nll, scaled.t, _CLASSES),
            "outcome_nll": float(-teacher_outcome.log_prob(scaled.y, scaled.t).mean()),
            "quantity": float(weights.mean()),
            "weighted_impurity": weighted_impurity,
            "unweighted_impurity": float(wrong.mean()),
            "aligned_class_weight_cv": _class_weight_cv(
                aligned_weights, labels, _CLASSES
            ),
            "unaligned_class_weight_cv": _class_weight_cv(
                raw_weights, labels, _CLASSES
            ),
            "mu_hat": mu_hat,
            "sigma_squared": sigma_squared,
        }


__all__ = ["run"]
