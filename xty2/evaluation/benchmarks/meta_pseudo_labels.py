"""Meta Pseudo Labels' paired minority-trap Tier 2 target (card section 6.5).

Each replicate runs three arms from the same independently seeded teacher and
student parameters and with the same batch, view, and categorical-sampling
streams: the forced-zero-feedback control, MPL as card section 4 declares it
(meta-score weight 1.0), and a declared dose arm at meta-score weight 20. The
streams are exogenous and remain paired; their realised labels may diverge
after feedback changes the teacher probabilities, as card section 6 allows.

Only the weight-1.0 arm determines the status. The dose arm is built here, the
same way the zero-feedback arm is, so the recipe and its plan keep weight 1.0;
its metrics are reported as information.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import torch
from torch.nn import functional as F

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Port,
    Program,
    Recipe,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    bool_float,
    cluster_population,
    cluster_posterior,
    column,
    configure_worker,
    continuous_schema,
    parallel_replicates,
    training_dataset,
    treatment_posterior,
)
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec
from xty2.objectives import MetaFeedbackCoefficient
from xty2.recipes import INNER_ROLE, OUTER_ROLE, meta_pseudo_labels
from xty2.training import ObjectiveLog, StageResult, StepRecord, run_meta_gradient

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 94_000
_META = "teacher_meta_score"
_UDA = "teacher_uda_consistency"
_TSA = "teacher_tsa_nll"


class _Clusters(TypedDict):
    classes: int
    groups: tuple[int, ...]
    within: float
    noise: float
    prior: tuple[float, ...]


# Card section 6.1: three clusters, the 5% minority (cluster 1) beside the
# large cluster 0 and labelled like the distant cluster 2.
_CLUSTERS: _Clusters = {
    "classes": 3,
    "groups": (2, 1),
    "within": 0.6,
    "noise": 0.4,
    "prior": (0.475, 0.05, 0.475),
}
_LABELS = (0, 1, 1)
_LOW = 0.02
_MINORITY = 1
_DOSE_WEIGHT = 20.0
_ARMS = ("mpl", "zero", "dose")


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the 100 paired replicates predeclared by the card's section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local minority-trap XTY DGP (6 features, K=2): "
                "cluster_population(classes=3, groups=(2,1), within=0.6, "
                "noise=0.4, prior=(0.475,0.05,0.475)) relabelled by (0,1,1), "
                "specified in §6.1"
            ),
            "variant": (
                "paired MPL (meta-score weight 1.0) versus the "
                "forced-zero-feedback arm, plus a declared weight-20 dose arm "
                "reported as information"
            ),
            "metric": (
                "paired gain in inner-student accuracy on the held-out "
                "minority region (Bayes cluster posterior picks the minority "
                "cluster), MPL weight 1.0 minus h=0; h=0 minority accuracy as "
                "the fixture-validity guardrail; finiteness, outer-teacher NLL "
                "ratio and class-mass concentration as before; the dose arm's "
                "gains, both NLL ratios and the h diagnostics as information"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "minority-accuracy gain >= 0.10 in mean by at least one "
                "stderr; h=0 minority accuracy <= 0.60 (the trap binds); "
                "outer-teacher NLL <= 1.10x the zero-feedback arm; finite "
                "losses and gradients in every replicate; terminal student "
                "class-mass concentration < 0.95"
            ),
            "seeds": "100",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 100:
        raise ValueError(
            f"meta_pseudo_labels card reviewed 100 replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.lower_bound(
                "student_minority_accuracy_gain",
                column(rows, "student_minority_gain"),
                0.10,
            ),
            MetricResult.upper_bound(
                "zero_feedback_student_minority_accuracy",
                column(rows, "zero_student_minority"),
                0.60,
            ),
            MetricResult.upper_bound(
                "outer_teacher_treatment_NLL_ratio",
                column(rows, "teacher_ratio"),
                1.10,
            ),
            MetricResult.lower_bound(
                "all_losses_gradients_and_checkpoints_finite",
                column(rows, "finite"),
                1.0,
            ),
            MetricResult.upper_bound(
                "terminal_student_class_mass_concentration",
                column(rows, "max_concentration"),
                0.95,
            ),
            MetricResult.information(
                "student_treatment_NLL_ratio", column(rows, "student_ratio")
            ),
            MetricResult.information(
                "teacher_minority_accuracy_gain",
                column(rows, "teacher_minority_gain"),
            ),
            MetricResult.information(
                "dose_student_minority_accuracy_gain",
                column(rows, "dose_student_minority_gain"),
            ),
            MetricResult.information(
                "dose_teacher_minority_accuracy_gain",
                column(rows, "dose_teacher_minority_gain"),
            ),
            MetricResult.information(
                "dose_student_treatment_NLL_ratio",
                column(rows, "dose_student_ratio"),
            ),
            MetricResult.information(
                "dose_outer_teacher_treatment_NLL_ratio",
                column(rows, "dose_teacher_ratio"),
            ),
            MetricResult.information(
                "dose_all_losses_gradients_and_checkpoints_finite",
                column(rows, "dose_finite"),
            ),
            MetricResult.information(
                "mpl_student_minority_accuracy", column(rows, "mpl_student_minority")
            ),
            MetricResult.information(
                "dose_student_minority_accuracy",
                column(rows, "dose_student_minority"),
            ),
            MetricResult.information(
                "bayes_minority_accuracy", column(rows, "bayes_minority")
            ),
            MetricResult.information(
                "minority_region_fraction", column(rows, "minority_fraction")
            ),
            MetricResult.information(
                "mpl_student_treatment_NLL",
                column(rows, "mpl_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "zero_feedback_student_treatment_NLL",
                column(rows, "zero_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "dose_student_treatment_NLL",
                column(rows, "dose_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "mpl_outer_teacher_treatment_NLL",
                column(rows, "mpl_teacher_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "zero_feedback_outer_teacher_treatment_NLL",
                column(rows, "zero_teacher_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "bayes_treatment_NLL", column(rows, "bayes_nll"), unit="nat/row"
            ),
            MetricResult.information(
                "marginal_frequency_NLL", column(rows, "frequency_nll"), unit="nat/row"
            ),
            MetricResult.information(
                "mpl_student_class_mass_concentration",
                column(rows, "mpl_concentration"),
            ),
            MetricResult.information(
                "zero_feedback_student_class_mass_concentration",
                column(rows, "zero_concentration"),
            ),
            MetricResult.information("feedback_h_mean", column(rows, "h_mean")),
            MetricResult.information(
                "feedback_h_population_std", column(rows, "h_std")
            ),
            MetricResult.information(
                "feedback_h_positive_fraction", column(rows, "h_positive")
            ),
            MetricResult.information(
                "feedback_h_nonzero_fraction", column(rows, "h_nonzero")
            ),
            MetricResult.information(
                "terminal_feedback_baseline", column(rows, "baseline")
            ),
            MetricResult.information(
                "terminal_sampled_label_accuracy",
                column(rows, "sampled_label_accuracy"),
            ),
            MetricResult.information(
                "terminal_sampled_label_entropy",
                column(rows, "sampled_label_entropy"),
            ),
            MetricResult.information(
                "terminal_UDA_gate_coverage", column(rows, "uda_coverage")
            ),
            MetricResult.information(
                "terminal_TSA_retained_fraction", column(rows, "tsa_fraction")
            ),
            MetricResult.information(
                "weak_view_label_flip_rate", column(rows, "weak_flip_rate")
            ),
            MetricResult.information(
                "strong_view_label_flip_rate", column(rows, "strong_flip_rate")
            ),
        ),
        interpretation=(
            "This is the predeclared project-local Meta Pseudo Labels mechanism "
            "target of card section 6.5: whether appendix A's centred hard-label "
            "score-function feedback, at the section 4 meta-score weight, "
            "repairs a teacher that UDA's consistency leads to absorb a 5% "
            "minority cluster into its neighbour's class, measured as the "
            "student's paired accuracy gain on that region over the otherwise "
            "identical h=0 execution. A declared weight-20 dose arm is reported "
            "as information and does not determine the status. It is not a "
            "reproduction of a published image-accuracy number; the tabular "
            "DGP, views, architecture, and 3,000-step budget are the reviewed "
            "deviations listed in the card."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = cluster_population(
        _TRAIN_ROWS, seed=base + 1, row_offset=0, labels=_LABELS, **_CLUSTERS
    )
    test = cluster_population(
        _TEST_ROWS, seed=base + 2, row_offset=10_000, labels=_LABELS, **_CLUSTERS
    )
    data = training_dataset(schema, train.batch)
    full_recipe = meta_pseudo_labels(schema)
    recipes = {
        "mpl": full_recipe,
        "zero": _zero_feedback(full_recipe),
        "dose": _dosed(full_recipe, _DOSE_WEIGHT),
    }
    runs = {name: compile(recipe) for name, recipe in recipes.items()}
    flips = _flip_rates(full_recipe, train.batch, base + 10_000)
    results = {
        name: run_meta_gradient(
            run,
            "meta_train",
            data,
            seed=base + 10_000,
            role_seeds={OUTER_ROLE: base + 6, INNER_ROLE: base + 7},
            hard_label_seed=base + 20_000,
        )
        for name, run in runs.items()
    }
    reference_rows = results["mpl"].role_checkpoints[INNER_ROLE].trained_on_row_ids
    for name, result in results.items():
        for role in (INNER_ROLE, OUTER_ROLE):
            if not torch.equal(
                result.role_checkpoints[role].trained_on_row_ids, reference_rows
            ):
                raise RuntimeError(
                    f"meta_pseudo_labels arm {name!r} role {role!r} saw "
                    "different training rows"
                )

    region = cluster_posterior(test.batch.x, **_CLUSTERS).argmax(dim=1) == _MINORITY
    if not bool(region.any()):
        raise RuntimeError("the held-out draw has no minority-region rows")
    bayes = treatment_posterior(test.batch.x, low=_LOW, labels=_LABELS, **_CLUSTERS)
    metrics = {
        name: _evaluate(runs[name], result, test.batch, region)
        for name, result in results.items()
    }
    zero_student = metrics["zero"]["student_nll"]
    zero_teacher = metrics["zero"]["teacher_nll"]
    if zero_student <= 0.0 or zero_teacher <= 0.0:
        raise RuntimeError("the zero-feedback arm produced a non-positive NLL")
    diagnostics = _feedback_diagnostics(results["mpl"])
    zero_minority = metrics["zero"]["student_minority"]
    zero_teacher_minority = metrics["zero"]["teacher_minority"]
    return {
        "student_minority_gain": metrics["mpl"]["student_minority"] - zero_minority,
        "zero_student_minority": zero_minority,
        "teacher_ratio": metrics["mpl"]["teacher_nll"] / zero_teacher,
        "finite": bool_float(_finite(results["mpl"]) and _finite(results["zero"])),
        "max_concentration": max(
            metrics["mpl"]["concentration"], metrics["zero"]["concentration"]
        ),
        "student_ratio": metrics["mpl"]["student_nll"] / zero_student,
        "teacher_minority_gain": (
            metrics["mpl"]["teacher_minority"] - zero_teacher_minority
        ),
        "dose_student_minority_gain": (
            metrics["dose"]["student_minority"] - zero_minority
        ),
        "dose_teacher_minority_gain": (
            metrics["dose"]["teacher_minority"] - zero_teacher_minority
        ),
        "dose_student_ratio": metrics["dose"]["student_nll"] / zero_student,
        "dose_teacher_ratio": metrics["dose"]["teacher_nll"] / zero_teacher,
        "dose_finite": bool_float(_finite(results["dose"])),
        "mpl_student_minority": metrics["mpl"]["student_minority"],
        "dose_student_minority": metrics["dose"]["student_minority"],
        "bayes_minority": float(
            (bayes.argmax(dim=1)[region] == test.batch.t[region]).float().mean()
        ),
        "minority_fraction": float(region.float().mean()),
        "mpl_student_nll": metrics["mpl"]["student_nll"],
        "zero_student_nll": zero_student,
        "dose_student_nll": metrics["dose"]["student_nll"],
        "mpl_teacher_nll": metrics["mpl"]["teacher_nll"],
        "zero_teacher_nll": zero_teacher,
        "bayes_nll": float(F.nll_loss(bayes.log(), test.batch.t)),
        "frequency_nll": metrics["mpl"]["frequency_nll"],
        "mpl_concentration": metrics["mpl"]["concentration"],
        "zero_concentration": metrics["zero"]["concentration"],
        "weak_flip_rate": flips[0],
        "strong_flip_rate": flips[1],
        **diagnostics,
    }


def _zero_feedback(recipe: Recipe) -> Recipe:
    """The reviewed control: force centred h to zero and change nothing else."""
    stage = recipe.program[0]
    meta = stage.meta_gradient
    if meta is None or not isinstance(meta.feedback, MetaFeedbackCoefficient):
        raise RuntimeError("meta_pseudo_labels lost its reviewed feedback contract")
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    meta_gradient=replace(
                        meta, feedback=replace(meta.feedback, force_zero=True)
                    ),
                ),
            )
        ),
    )


def _dosed(recipe: Recipe, weight: float) -> Recipe:
    """The declared dose arm: the meta-score weight changed and nothing else."""
    stage = recipe.program[0]
    objectives = tuple(
        replace(weighted, weight=weight)
        if weighted.objective.name == _META
        else weighted
        for weighted in stage.objectives
    )
    if objectives == stage.objectives:
        raise RuntimeError(f"meta_pseudo_labels has no {_META!r} objective to dose")
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _evaluate(
    run: CompiledRun, result: StageResult, test: XTYBatch, region: torch.Tensor
) -> dict[str, float]:
    with torch.no_grad():
        student = result.role_graphs[INNER_ROLE].evaluate(
            test, schema=run.recipe.schema, only=run.graph.names
        )[Port.T_GIVEN_X]
        teacher = result.role_graphs[OUTER_ROLE].evaluate(
            test, schema=run.recipe.schema, only=run.graph.names
        )[Port.T_GIVEN_X]
    if not isinstance(student, CategoricalTreatment) or not isinstance(
        teacher, CategoricalTreatment
    ):
        raise TypeError(
            "meta_pseudo_labels benchmark expected categorical propensities"
        )
    population = result.population
    if population is None:
        raise RuntimeError(
            "meta_pseudo_labels benchmark expected a training population"
        )
    observed = population.rows.t[population.rows.t_observed]
    frequency = torch.bincount(observed, minlength=2).float()
    frequency /= frequency.sum()
    baseline = frequency.log().expand(test.batch_size, -1)
    truth = test.t[region]
    return {
        "student_nll": float(F.nll_loss(student.log_probs, test.t)),
        "teacher_nll": float(F.nll_loss(teacher.log_probs, test.t)),
        "frequency_nll": float(F.nll_loss(baseline, test.t)),
        "concentration": float(student.probs.mean(dim=0).max()),
        "student_minority": float(
            (student.probs.argmax(dim=1)[region] == truth).float().mean()
        ),
        "teacher_minority": float(
            (teacher.probs.argmax(dim=1)[region] == truth).float().mean()
        ),
    }


def _feedback_diagnostics(result: StageResult) -> dict[str, float]:
    meta = [_term(record, _META) for record in result.records]
    h = [float(term.diagnostics["h"]) for term in meta]
    terminal = meta[-1].diagnostics
    return {
        "h_mean": statistics.fmean(h),
        "h_std": statistics.pstdev(h),
        "h_positive": statistics.fmean(value > 0.0 for value in h),
        "h_nonzero": statistics.fmean(value != 0.0 for value in h),
        "baseline": float(terminal["baseline"]),
        "sampled_label_accuracy": float(terminal["sampled_label_accuracy"]),
        "sampled_label_entropy": float(terminal["sampled_label_entropy"]),
        "uda_coverage": float(_term(result.records[-1], _UDA).diagnostics["coverage"]),
        "tsa_fraction": float(
            _term(result.records[-1], _TSA).diagnostics["retained_fraction"]
        ),
    }


def _term(record: StepRecord, name: str) -> ObjectiveLog:
    for term in record.terms:
        if term.name == name:
            return term
    raise RuntimeError(f"meta_pseudo_labels step has no term named {name!r}")


def _finite(result: StageResult) -> bool:
    for record in result.records:
        values = [
            record.lr,
            record.total,
            record.grad_norm,
            *record.role_lrs.values(),
            *record.role_grad_norms.values(),
        ]
        for term in record.terms:
            values.extend((term.value, term.weight, term.weighted))
            values.extend(term.diagnostics.values())
        if not all(math.isfinite(float(value)) for value in values):
            return False
    return all(
        bool(torch.isfinite(value).all())
        for checkpoint in result.role_checkpoints.values()
        for value in checkpoint.parameters.values()
    )


def _flip_rates(
    recipe: Recipe, train: XTYBatch, stage_seed: int
) -> tuple[float, float]:
    """How often each view changes the Bayes label of a training row.

    The label is the fixture's exact Bayes rule, so the rate is about the view
    and not about any fitted model.
    """
    clean = treatment_posterior(train.x, low=_LOW, labels=_LABELS, **_CLUSTERS)
    label = clean.argmax(dim=1)
    rates = []
    for view in ("weak_x", "strong_x"):
        realised = recipe.view(view).apply(train, recipe.schema, rng_key=stage_seed)
        viewed = treatment_posterior(
            realised.x, low=_LOW, labels=_LABELS, **_CLUSTERS
        ).argmax(dim=1)
        rates.append(float((viewed != label).float().mean()))
    return rates[0], rates[1]


__all__ = ["run"]
