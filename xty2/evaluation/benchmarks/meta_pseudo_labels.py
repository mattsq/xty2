"""Meta Pseudo Labels' paired feedback/no-feedback Tier 2 target.

Each replicate runs the two arms from the same independently seeded teacher
and student parameters and with the same batch, view, and categorical-sampling
streams.  The streams are exogenous and remain paired; their realised labels
may diverge after feedback changes the teacher probabilities, as card section
6 explicitly allows.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace
from pathlib import Path

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
    column,
    configure_worker,
    continuous_schema,
    parallel_replicates,
    training_dataset,
    two_cluster_population,
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
_SIGNAL_COLUMNS = 4


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten paired replicates predeclared by the card's section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in §6.1 and inherited from uda.md"
            ),
            "variant": (
                "paired MPL feedback versus the forced-zero-feedback arm "
                "defined below; same initial teacher/student parameters, "
                "batches, views, hard-label RNG and UDA objectives"
            ),
            "metric": (
                "held-out inner-student treatment NLL ratio, MPL over the "
                "zero-feedback arm; held-out outer-teacher NLL ratio, "
                "finiteness and terminal student class-mass concentration as "
                "status-determining guardrails; sampled-label accuracy, h and "
                "baseline trajectories, UDA gate coverage, TSA retained "
                "fraction and view label-flip rates as reported diagnostics"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "student NLL ratio < 1.0 in mean by at least one stderr; "
                "outer-teacher NLL <= 1.10x the zero-feedback arm; finite "
                "losses and gradients in every replicate; terminal student "
                "class-mass concentration < 0.95"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"meta_pseudo_labels card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "student_treatment_NLL_ratio", column(rows, "student_ratio"), 1.0
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
            "target: whether appendix A's centred hard-label score-function "
            "feedback improves the inner student's treatment NLL over the "
            "otherwise identical h=0 execution. It is not a reproduction of "
            "a published image-accuracy number; the tabular DGP, views, "
            "architecture, and 3,000-step budget are the reviewed deviations "
            "listed in the card."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = two_cluster_population(_TRAIN_ROWS, seed=base + 1, row_offset=0)
    test = two_cluster_population(_TEST_ROWS, seed=base + 2, row_offset=10_000)
    data = training_dataset(schema, train.batch)
    full_recipe = meta_pseudo_labels(schema)
    recipes = {"mpl": full_recipe, "zero": _zero_feedback(full_recipe)}
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

    metrics = {
        name: _evaluate(runs[name], result, test.batch)
        for name, result in results.items()
    }
    zero_student = metrics["zero"]["student_nll"]
    zero_teacher = metrics["zero"]["teacher_nll"]
    if zero_student <= 0.0 or zero_teacher <= 0.0:
        raise RuntimeError("the zero-feedback arm produced a non-positive NLL")
    diagnostics = _feedback_diagnostics(results["mpl"])
    return {
        "student_ratio": metrics["mpl"]["student_nll"] / zero_student,
        "teacher_ratio": metrics["mpl"]["teacher_nll"] / zero_teacher,
        "finite": bool_float(all(_finite(result) for result in results.values())),
        "max_concentration": max(
            metrics["mpl"]["concentration"], metrics["zero"]["concentration"]
        ),
        "mpl_student_nll": metrics["mpl"]["student_nll"],
        "zero_student_nll": zero_student,
        "mpl_teacher_nll": metrics["mpl"]["teacher_nll"],
        "zero_teacher_nll": zero_teacher,
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


def _evaluate(
    run: CompiledRun, result: StageResult, test: XTYBatch
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
    return {
        "student_nll": float(F.nll_loss(student.log_probs, test.t)),
        "teacher_nll": float(F.nll_loss(teacher.log_probs, test.t)),
        "frequency_nll": float(F.nll_loss(baseline, test.t)),
        "concentration": float(student.probs.mean(dim=0).max()),
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
    label = train.x[:, :_SIGNAL_COLUMNS].sum(dim=-1) > 0
    rates = []
    for view in ("weak_x", "strong_x"):
        realised = recipe.view(view).apply(train, recipe.schema, rng_key=stage_seed)
        crossed = realised.x[:, :_SIGNAL_COLUMNS].sum(dim=-1) > 0
        rates.append(float((crossed != label).float().mean()))
    return rates[0], rates[1]


__all__ = ["run"]
