"""Meta Pseudo Labels' paired minority-trap Tier 2 target (card sections 6.5, 6.6).

Each replicate runs three arms from the same independently seeded teacher and
student parameters and with the same batch, view, and categorical-sampling
streams: the forced-zero-feedback control, MPL as card section 4 declares it
(meta-score weight 1.0), and a declared dose arm at meta-score weight 20. The
streams are exogenous and remain paired; their realised labels may diverge
after feedback changes the teacher probabilities, as card section 6 allows.
Every arm then fine-tunes its student on the labelled rows, as paper section
3.2 does, and the status reads the fine-tuned students.

Only the weight-1.0 arm determines the status. The dose arm is built here, the
same way the zero-feedback arm is, so the recipe and its plan keep weight 1.0;
its metrics are reported as information. So is a third stage that fine-tunes
the same student at the declared learning rate 1e-3 (card section 6.6).
"""

from __future__ import annotations

import copy
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
    ComponentGraph,
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
from xty2.training import (
    Checkpoint,
    MetaGradientSeeds,
    ObjectiveLog,
    ProgramResult,
    StageResult,
    StepRecord,
    run_program,
)

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
_META_STAGE = "meta_train"
_FINETUNE = "finetune"
_INFORMATION = "finetune_information"
# Card section 6.6: the declared information-only fine-tuning rate.
_INFORMATION_LR = 1e-3


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
                "forced-zero-feedback arm, each followed by the paper's "
                "labelled fine-tuning of the student, plus a declared "
                "weight-20 dose arm and a declared fine-tuning learning rate "
                "of 1e-3, both reported as information"
            ),
            "metric": (
                "paired gain in fine-tuned student accuracy on the held-out "
                "minority region (Bayes cluster posterior picks the minority "
                "cluster), MPL weight 1.0 minus h=0; fine-tuned h=0 minority "
                "accuracy as the fixture-validity guardrail; finiteness over "
                "both stages, outer-teacher NLL ratio and fine-tuned student "
                "class-mass concentration; the gains before fine-tuning, the "
                "dose arm, the 1e-3 fine-tuning stage, the NLL ratios and the "
                "h diagnostics as information"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "fine-tuned minority-accuracy gain >= 0.10 in mean by at "
                "least one stderr; fine-tuned h=0 minority accuracy <= 0.60 "
                "(the trap binds); outer-teacher NLL <= 1.10x the "
                "zero-feedback arm; finite losses and gradients in both "
                "stages of every replicate; fine-tuned student class-mass "
                "concentration < 0.95"
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
    information = [
        ("pre_finetune_student_minority_accuracy_gain", "pre_student_minority_gain"),
        ("pre_finetune_student_treatment_NLL_ratio", "pre_student_ratio"),
        ("pre_finetune_zero_feedback_student_minority_accuracy", "pre_zero_minority"),
        ("student_treatment_NLL_ratio", "student_ratio"),
        ("teacher_minority_accuracy_gain", "teacher_minority_gain"),
        ("lr_1e-3_student_minority_accuracy_gain", "info_student_minority_gain"),
        ("lr_1e-3_student_treatment_NLL_ratio", "info_student_ratio"),
        ("lr_1e-3_zero_feedback_student_minority_accuracy", "info_zero_minority"),
        ("dose_student_minority_accuracy_gain", "dose_student_minority_gain"),
        ("dose_pre_finetune_student_minority_accuracy_gain", "dose_pre_minority_gain"),
        ("dose_teacher_minority_accuracy_gain", "dose_teacher_minority_gain"),
        ("dose_student_treatment_NLL_ratio", "dose_student_ratio"),
        ("dose_outer_teacher_treatment_NLL_ratio", "dose_teacher_ratio"),
        ("dose_all_losses_gradients_and_checkpoints_finite", "dose_finite"),
        ("mpl_student_minority_accuracy", "mpl_student_minority"),
        ("dose_student_minority_accuracy", "dose_student_minority"),
        ("bayes_minority_accuracy", "bayes_minority"),
        ("minority_region_fraction", "minority_fraction"),
        ("mpl_finetune_parameter_shift", "mpl_shift"),
        ("zero_feedback_finetune_parameter_shift", "zero_shift"),
        ("mpl_lr_1e-3_parameter_shift", "mpl_info_shift"),
        ("zero_feedback_lr_1e-3_parameter_shift", "zero_info_shift"),
        ("mpl_student_class_mass_concentration", "mpl_concentration"),
        ("zero_feedback_student_class_mass_concentration", "zero_concentration"),
        ("feedback_h_mean", "h_mean"),
        ("feedback_h_population_std", "h_std"),
        ("feedback_h_positive_fraction", "h_positive"),
        ("feedback_h_nonzero_fraction", "h_nonzero"),
        ("terminal_feedback_baseline", "baseline"),
        ("terminal_sampled_label_accuracy", "sampled_label_accuracy"),
        ("terminal_sampled_label_entropy", "sampled_label_entropy"),
        ("terminal_UDA_gate_coverage", "uda_coverage"),
        ("terminal_TSA_retained_fraction", "tsa_fraction"),
        ("weak_view_label_flip_rate", "weak_flip_rate"),
        ("strong_view_label_flip_rate", "strong_flip_rate"),
    ]
    nats = [
        ("mpl_student_treatment_NLL", "mpl_student_nll"),
        ("zero_feedback_student_treatment_NLL", "zero_student_nll"),
        ("dose_student_treatment_NLL", "dose_student_nll"),
        ("mpl_pre_finetune_student_treatment_NLL", "mpl_pre_student_nll"),
        ("zero_feedback_pre_finetune_student_treatment_NLL", "zero_pre_student_nll"),
        ("mpl_outer_teacher_treatment_NLL", "mpl_teacher_nll"),
        ("zero_feedback_outer_teacher_treatment_NLL", "zero_teacher_nll"),
        ("bayes_treatment_NLL", "bayes_nll"),
        ("marginal_frequency_NLL", "frequency_nll"),
    ]
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
            *(
                MetricResult.information(name, column(rows, key))
                for name, key in information
            ),
            *(
                MetricResult.information(name, column(rows, key), unit="nat/row")
                for name, key in nats
            ),
        ),
        interpretation=(
            "This is the predeclared project-local Meta Pseudo Labels mechanism "
            "target of card sections 6.5 and 6.6: whether appendix A's centred "
            "hard-label score-function feedback, at the section 4 meta-score "
            "weight, repairs a teacher that UDA's consistency leads to absorb a "
            "5% minority cluster into its neighbour's class, measured as the "
            "paired accuracy gain on that region of the student after the "
            "paper's labelled fine-tuning (section 3.2), over the otherwise "
            "identical h=0 execution. A declared weight-20 dose arm and a "
            "declared fine-tuning rate of 1e-3 are reported as information and "
            "do not determine the status. It is not a reproduction of a "
            "published image-accuracy number; the tabular DGP, views, "
            "architecture, 3,000-step budget and the fine-tuning batch are the "
            "reviewed deviations listed in the card."
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
    runs = {
        name: compile(_with_information_stage(recipe))
        for name, recipe in recipes.items()
    }
    flips = _flip_rates(full_recipe, train.batch, base + 10_000)
    seeds = MetaGradientSeeds(
        roles={OUTER_ROLE: base + 6, INNER_ROLE: base + 7}, hard_label=base + 20_000
    )
    results = {
        name: run_program(
            run,
            {stage.name: data for stage in run.stages},
            seed=base + 10_000,
            meta_seeds={_META_STAGE: seeds},
        )
        for name, run in runs.items()
    }
    _check_paired_rows(results)

    region = cluster_posterior(test.batch.x, **_CLUSTERS).argmax(dim=1) == _MINORITY
    if not bool(region.any()):
        raise RuntimeError("the held-out draw has no minority-region rows")
    bayes = treatment_posterior(test.batch.x, low=_LOW, labels=_LABELS, **_CLUSTERS)
    metrics = {
        name: _evaluate(runs[name], result, test.batch, region)
        for name, result in results.items()
    }
    zero = metrics["zero"]
    for key in ("student_nll", "pre_student_nll", "info_student_nll", "teacher_nll"):
        if zero[key] <= 0.0:
            raise RuntimeError(f"the zero-feedback arm produced a non-positive {key}")
    mpl = metrics["mpl"]
    dose = metrics["dose"]
    diagnostics = _feedback_diagnostics(results["mpl"].stage(_META_STAGE))
    return {
        "student_minority_gain": mpl["student_minority"] - zero["student_minority"],
        "zero_student_minority": zero["student_minority"],
        "teacher_ratio": mpl["teacher_nll"] / zero["teacher_nll"],
        "finite": bool_float(_finite(results["mpl"]) and _finite(results["zero"])),
        "max_concentration": max(mpl["concentration"], zero["concentration"]),
        "pre_student_minority_gain": (
            mpl["pre_student_minority"] - zero["pre_student_minority"]
        ),
        "pre_student_ratio": mpl["pre_student_nll"] / zero["pre_student_nll"],
        "pre_zero_minority": zero["pre_student_minority"],
        "student_ratio": mpl["student_nll"] / zero["student_nll"],
        "teacher_minority_gain": mpl["teacher_minority"] - zero["teacher_minority"],
        "info_student_minority_gain": (
            mpl["info_student_minority"] - zero["info_student_minority"]
        ),
        "info_student_ratio": mpl["info_student_nll"] / zero["info_student_nll"],
        "info_zero_minority": zero["info_student_minority"],
        "dose_student_minority_gain": (
            dose["student_minority"] - zero["student_minority"]
        ),
        "dose_pre_minority_gain": (
            dose["pre_student_minority"] - zero["pre_student_minority"]
        ),
        "dose_teacher_minority_gain": (
            dose["teacher_minority"] - zero["teacher_minority"]
        ),
        "dose_student_ratio": dose["student_nll"] / zero["student_nll"],
        "dose_teacher_ratio": dose["teacher_nll"] / zero["teacher_nll"],
        "dose_finite": bool_float(_finite(results["dose"])),
        "mpl_student_minority": mpl["student_minority"],
        "dose_student_minority": dose["student_minority"],
        "bayes_minority": float(
            (bayes.argmax(dim=1)[region] == test.batch.t[region]).float().mean()
        ),
        "minority_fraction": float(region.float().mean()),
        "mpl_shift": mpl["shift"],
        "zero_shift": zero["shift"],
        "mpl_info_shift": mpl["info_shift"],
        "zero_info_shift": zero["info_shift"],
        "mpl_student_nll": mpl["student_nll"],
        "zero_student_nll": zero["student_nll"],
        "dose_student_nll": dose["student_nll"],
        "mpl_pre_student_nll": mpl["pre_student_nll"],
        "zero_pre_student_nll": zero["pre_student_nll"],
        "mpl_teacher_nll": mpl["teacher_nll"],
        "zero_teacher_nll": zero["teacher_nll"],
        "bayes_nll": float(F.nll_loss(bayes.log(), test.batch.t)),
        "frequency_nll": mpl["frequency_nll"],
        "mpl_concentration": mpl["concentration"],
        "zero_concentration": zero["concentration"],
        "weak_flip_rate": flips[0],
        "strong_flip_rate": flips[1],
        **diagnostics,
    }


def _check_paired_rows(results: dict[str, ProgramResult]) -> None:
    """Every arm and stage must have stepped on the same training rows."""
    reference = results["mpl"].stage(_META_STAGE)
    meta_rows = reference.role_checkpoints[INNER_ROLE].trained_on_row_ids
    tuned_rows = results["mpl"].stage(_FINETUNE).checkpoint.trained_on_row_ids
    for name, program in results.items():
        meta = program.stage(_META_STAGE)
        for role in (INNER_ROLE, OUTER_ROLE):
            if not torch.equal(
                meta.role_checkpoints[role].trained_on_row_ids, meta_rows
            ):
                raise RuntimeError(
                    f"meta_pseudo_labels arm {name!r} role {role!r} saw "
                    "different training rows"
                )
        for stage in (_FINETUNE, _INFORMATION):
            rows = program.stage(stage).checkpoint.trained_on_row_ids
            if not torch.equal(rows, tuned_rows):
                raise RuntimeError(
                    f"meta_pseudo_labels arm {name!r} stage {stage!r} fine-tuned "
                    "on different rows"
                )


def _zero_feedback(recipe: Recipe) -> Recipe:
    """The reviewed control: force centred h to zero and change nothing else."""
    stage, *later = recipe.program
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
                *later,
            )
        ),
    )


def _dosed(recipe: Recipe, weight: float) -> Recipe:
    """The declared dose arm: the meta-score weight changed and nothing else."""
    stage, *later = recipe.program
    objectives = tuple(
        replace(weighted, weight=weight)
        if weighted.objective.name == _META
        else weighted
        for weighted in stage.objectives
    )
    if objectives == stage.objectives:
        raise RuntimeError(f"meta_pseudo_labels has no {_META!r} objective to dose")
    return replace(
        recipe,
        program=Program((replace(stage, objectives=objectives), *later)),
    )


def _with_information_stage(recipe: Recipe) -> Recipe:
    """Append the declared 1e-3 fine-tuning of the same student (card §6.6).

    It initialises from the same role checkpoint as `finetune` and changes only
    the learning rate, so the recipe and its plan keep the paper's 1e-5.
    """
    finetune = recipe.program.stage(_FINETUNE)
    if finetune.initialise_role != INNER_ROLE:
        raise RuntimeError("meta_pseudo_labels fine-tuning lost its student source")
    information = replace(
        finetune,
        name=_INFORMATION,
        optimiser=replace(finetune.optimiser, lr=_INFORMATION_LR),
    )
    return replace(recipe, program=Program((*recipe.program, information)))


def _graph_at(run: CompiledRun, checkpoint: Checkpoint) -> ComponentGraph:
    """A copy of the compiled graph holding one checkpoint's tensors."""
    graph = copy.deepcopy(run.graph)
    state = graph.state_dict()
    for name, value in {**checkpoint.parameters, **checkpoint.buffers}.items():
        state["_components." + name].copy_(value)
    return graph


def _shift(after: Checkpoint, before: Checkpoint) -> float:
    """L2 norm of the parameter change one fine-tuning stage made."""
    return math.sqrt(
        sum(
            float((after.parameters[name] - value).square().sum())
            for name, value in before.parameters.items()
        )
    )


def _evaluate(
    run: CompiledRun, program: ProgramResult, test: XTYBatch, region: torch.Tensor
) -> dict[str, float]:
    meta = program.stage(_META_STAGE)
    tuned = program.stage(_FINETUNE).checkpoint
    information = program.stage(_INFORMATION).checkpoint
    student_before = meta.role_checkpoints[INNER_ROLE]
    graphs = {
        "pre_student": meta.role_graphs[INNER_ROLE],
        "student": _graph_at(run, tuned),
        "info_student": _graph_at(run, information),
        "teacher": meta.role_graphs[OUTER_ROLE],
    }
    distributions: dict[str, CategoricalTreatment] = {}
    with torch.no_grad():
        for name, graph in graphs.items():
            value = graph.evaluate(
                test, schema=run.recipe.schema, only=run.graph.names
            )[Port.T_GIVEN_X]
            if not isinstance(value, CategoricalTreatment):
                raise TypeError(
                    "meta_pseudo_labels benchmark expected categorical propensities"
                )
            distributions[name] = value
    population = meta.population
    if population is None:
        raise RuntimeError(
            "meta_pseudo_labels benchmark expected a training population"
        )
    observed = population.rows.t[population.rows.t_observed]
    frequency = torch.bincount(observed, minlength=2).float()
    frequency /= frequency.sum()
    baseline = frequency.log().expand(test.batch_size, -1)
    truth = test.t[region]
    metrics = {
        "frequency_nll": float(F.nll_loss(baseline, test.t)),
        "concentration": float(distributions["student"].probs.mean(dim=0).max()),
        "shift": _shift(tuned, student_before),
        "info_shift": _shift(information, student_before),
    }
    for name, distribution in distributions.items():
        metrics[f"{name}_nll"] = float(F.nll_loss(distribution.log_probs, test.t))
        metrics[f"{name}_minority"] = float(
            (distribution.probs.argmax(dim=1)[region] == truth).float().mean()
        )
    return metrics


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


def _finite(program: ProgramResult) -> bool:
    """Finite logs and checkpoints in every stage of one arm's program."""
    for result in program.stages:
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
        checkpoints = list(result.role_checkpoints.values())
        if result.has_checkpoint:
            checkpoints.append(result.checkpoint)
        if not checkpoints:
            return False
        if not all(
            bool(torch.isfinite(value).all())
            for checkpoint in checkpoints
            for value in checkpoint.parameters.values()
        ):
            return False
    return True


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
