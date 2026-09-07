"""ReMixMatch's paired mechanism benchmark from card section 6.

Four arms share the card's skewed K=4 fixture, initial parameters, batch
stream, optimiser settings for active components, and augmentation transforms.
The comparisons test the ReMixMatch bundle, distribution alignment, and MixUp.
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
    Realisation,
    Recipe,
    XTYBatch,
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
from xty2.objectives import AnchoredLabelGuess, ObservedTreatmentNLL
from xty2.recipes import remixmatch
from xty2.recipes.remixmatch import GUESS_OWNER
from xty2.training import StageResult, run_stage

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 90_000
_CLASSES = 4
_TRAIN_PRIOR = (0.55, 0.25, 0.13, 0.07)
_EFFECTS = (0.0, 1.0, 0.4, 1.6)
_TERMINAL_STEPS = 100
_MIXED_LABELLED = "mixed_labelled_treatment_nll"
_MIXED_UNLABELLED = "mixed_unlabelled_treatment_nll"
_PRETEXT = "pretext_transform_nll"
_MARGINAL_DIAGNOSTICS = (
    "labelled_vs_true_training_marginal_L1",
    "labelled_vs_true_unlabelled_marginal_L1",
    "true_unlabelled_vs_training_marginal_L1",
    "aligned_vs_labelled_marginal_L1",
    "unaligned_vs_labelled_marginal_L1",
    "aligned_vs_true_unlabelled_marginal_L1",
    "unaligned_vs_true_unlabelled_marginal_L1",
    "alignment_true_unlabelled_marginal_L1_advantage",
    *(
        f"{name}_class_{level}"
        for name in (
            "estimated_labelled",
            "true_training",
            "true_unlabelled",
            "aligned_window",
            "unaligned_window",
        )
        for level in range(_CLASSES)
    ),
)


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the four ten-seed arms predeclared by card section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked cluster XTY DGP (6 features, K=4, "
                "long-tailed train prior, balanced held-out prior), specified in 6.1"
            ),
            "variant": (
                "four paired fits - full ReMixMatch; no ReMixMatch "
                "(observed-treatment NLL on the first strong view, no mixing "
                "or pseudo-targets, shared causal terms retained); no distribution "
                "alignment (use_dm = false); no MixUp (alpha -> the identity "
                "pool); all other mechanics paired"
            ),
            "split": (
                "1024 train rows with exactly 64 observed treatments; 2048 "
                "held-out rows with every treatment observed and a balanced prior"
            ),
            "metric": (
                "held-out balanced macro treatment NLL for student and evaluation "
                "EMA; held-out outcome NLL guardrail; terminal L1 distance between "
                "the model's predicted marginal and p(y), the estimated labelled "
                "marginal alignment targets, with the distances to the true "
                "training and true unlabelled marginals reported beside it; "
                "pretext accuracy; per-copy target agreement; mixed-entry lambda' "
                "distribution"
            ),
            "published": "none - no published number applies to this adaptation",
            # The card uses a YAML folded block.  ReproductionSpec's deliberately
            # small scalar parser binds its marker; the complete block still
            # participates in spec.digest, while the MetricResults below encode
            # each executable threshold.
            "tolerance": ">",
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"remixmatch card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "full_vs_no_remixmatch_student_macro_NLL_ratio",
                column(rows, "full_no_remixmatch_student_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "full_vs_no_remixmatch_ema_macro_NLL_ratio",
                column(rows, "full_no_remixmatch_ema_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "full_vs_no_alignment_student_macro_NLL_ratio",
                column(rows, "full_no_alignment_student_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "full_vs_no_alignment_ema_macro_NLL_ratio",
                column(rows, "full_no_alignment_ema_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "full_vs_no_remixmatch_outcome_NLL_ratio",
                column(rows, "outcome_ratio"),
                1.05,
            ),
            MetricResult.lower_bound(
                "alignment_labelled_marginal_L1_advantage",
                column(rows, "alignment_labelled_marginal_L1_advantage"),
                0.0,
            ),
            MetricResult.lower_bound(
                "terminal_pretext_accuracy", column(rows, "pretext_accuracy"), 0.25
            ),
            MetricResult.lower_bound(
                "terminal_mixed_lambda_min", column(rows, "lambda_min"), 0.5
            ),
            MetricResult.upper_bound(
                "terminal_mixed_lambda_max", column(rows, "lambda_max"), 1.0
            ),
            MetricResult.information(
                "full_vs_no_mixup_student_macro_NLL_ratio",
                column(rows, "full_no_mixup_student_ratio"),
            ),
            MetricResult.information(
                "full_vs_no_mixup_ema_macro_NLL_ratio",
                column(rows, "full_no_mixup_ema_ratio"),
            ),
            MetricResult.information(
                "terminal_target_copy_agreement",
                column(rows, "target_copy_agreement"),
            ),
            MetricResult.information(
                "terminal_anchor_entropy", column(rows, "anchor_entropy")
            ),
            MetricResult.information(
                "terminal_mixed_lambda_mean", column(rows, "lambda_mean")
            ),
            MetricResult.information(
                "alignment_marginal_L1_advantage",
                column(rows, "alignment_advantage"),
            ),
            MetricResult.information(
                "aligned_model_marginal_L1", column(rows, "aligned_marginal_l1")
            ),
            MetricResult.information(
                "unaligned_model_marginal_L1",
                column(rows, "unaligned_marginal_l1"),
            ),
            *(
                MetricResult.information(name, column(rows, name))
                for name in _MARGINAL_DIAGNOSTICS
            ),
        ),
        interpretation=(
            "This is the predeclared project-local ReMixMatch mechanism target, "
            "not a reproduction of the paper's image benchmarks. It asks whether "
            "the ReMixMatch bundle and distribution alignment improve balanced "
            "treatment classification on the card's deliberately skewed fixture. "
            "The no_remixmatch arm retains the shared causal marginal term but "
            "has no pooled MixUp or pseudo-targets. The alignment guardrail is "
            "the distance to p(y), the marginal alignment targets; the "
            "true-training and true-unlabelled distances are reported beside it "
            "and gate nothing (card deviation 9)."
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
    for arm in ("full", "no_remixmatch", "no_alignment", "no_mixup"):
        torch.manual_seed(base + 6)
        if arm == "no_remixmatch":
            candidate = _no_remixmatch(remixmatch(schema))
        elif arm == "no_alignment":
            candidate = remixmatch(schema, use_alignment=False)
        elif arm == "no_mixup":
            candidate = remixmatch(schema, mix_rule="identity")
        else:
            candidate = remixmatch(schema)
        recipes[arm] = candidate

    reference = recipes["full"].system.state_dict()
    for arm, candidate in recipes.items():
        for name, value in reference.items():
            if not torch.equal(value, candidate.system.state_dict()[name]):
                raise RuntimeError(
                    f"remixmatch paired initial state differs in {arm!r} at {name!r}"
                )

    runs = {name: compile(recipe) for name, recipe in recipes.items()}
    stage_seed = base + 10_000
    results = {
        name: run_stage(run, "joint_fit", data, seed=stage_seed)
        for name, run in runs.items()
    }
    trained_rows = results["full"].checkpoint.trained_on_row_ids
    full_population = results["full"].population
    if full_population is None:
        raise RuntimeError("remixmatch full arm has no training population")
    for arm, result in results.items():
        if not torch.equal(trained_rows, result.checkpoint.trained_on_row_ids):
            raise RuntimeError(f"remixmatch arm {arm!r} saw different training rows")
        population = result.population
        if population is None or not (
            torch.equal(full_population.rows.row_id, population.rows.row_id)
            and torch.equal(full_population.rows.t_observed, population.rows.t_observed)
        ):
            raise RuntimeError(f"remixmatch arm {arm!r} has a different label split")

    evaluated = {
        name: _evaluate(runs[name], result, test) for name, result in results.items()
    }
    for arm in ("no_remixmatch", "no_alignment", "no_mixup"):
        for metric in ("student_macro_nll", "ema_macro_nll"):
            if evaluated[arm][metric] <= 0.0:
                raise RuntimeError(f"{arm} produced a non-positive {metric}")
    if evaluated["no_remixmatch"]["outcome_nll"] <= 0.0:
        raise RuntimeError("no_remixmatch arm produced a non-positive outcome NLL")

    full = evaluated["full"]
    baseline = evaluated["no_remixmatch"]
    no_alignment = evaluated["no_alignment"]
    no_mixup = evaluated["no_mixup"]
    aligned_l1 = _marginal_l1(results["full"], train.batch.t)
    unaligned_l1 = _marginal_l1(results["no_alignment"], train.batch.t)
    mechanism = _terminal_mechanism(
        runs["full"], results["full"], rng_key=base + 20_000
    )
    return {
        "full_no_remixmatch_student_ratio": full["student_macro_nll"]
        / baseline["student_macro_nll"],
        "full_no_remixmatch_ema_ratio": full["ema_macro_nll"]
        / baseline["ema_macro_nll"],
        "full_no_alignment_student_ratio": full["student_macro_nll"]
        / no_alignment["student_macro_nll"],
        "full_no_alignment_ema_ratio": full["ema_macro_nll"]
        / no_alignment["ema_macro_nll"],
        "full_no_mixup_student_ratio": full["student_macro_nll"]
        / no_mixup["student_macro_nll"],
        "full_no_mixup_ema_ratio": full["ema_macro_nll"] / no_mixup["ema_macro_nll"],
        "outcome_ratio": full["outcome_nll"] / baseline["outcome_nll"],
        "alignment_advantage": unaligned_l1 - aligned_l1,
        "aligned_marginal_l1": aligned_l1,
        "unaligned_marginal_l1": unaligned_l1,
        **_marginal_diagnostics(results["full"], results["no_alignment"], train.batch),
        **mechanism,
    }


def _no_remixmatch(recipe: Recipe) -> Recipe:
    """Remove every ReMixMatch target path, retaining the shared causal stack.

    Zeroing the three auxiliary weights is insufficient: labelled pooled
    MixUp still reads unlabelled pseudo-targets. Supervise the same strong
    labelled source directly and remove the mixing pool and anchor owner.
    """
    stage = recipe.program[0]
    objectives = tuple(
        replace(
            item,
            objective=ObservedTreatmentNLL(realisation=Realisation(view="strong_x")),
        )
        if item.name == _MIXED_LABELLED
        else item
        for item in stage.objectives
        if item.name
        in ("observed_outcome_nll", _MIXED_LABELLED, "missing_treatment_marginal_nll")
    )
    # Keep the component graph/initialisation paired, but the compiler rightly
    # rejects an unused head in the trainable or weight-decay declarations.
    trainable = tuple(name for name in stage.trainable if name != "pretext_head")
    decay = stage.optimiser.weight_decay
    if decay.components is None:
        raise RuntimeError("remixmatch requires an explicit weight-decay scope")
    optimiser = replace(
        stage.optimiser,
        weight_decay=replace(
            decay,
            components=tuple(name for name in decay.components if name in trainable),
        ),
    )
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=objectives,
                    trainable=trainable,
                    optimiser=optimiser,
                ),
            )
        ),
        views=(replace(recipe.view("strong_x"), draws=1),),
        mixes=(),
    )


def _macro_mean(values: Tensor, labels: Tensor) -> float:
    means = []
    for level in range(_CLASSES):
        selected = values[labels == level]
        if not selected.numel():
            raise RuntimeError(f"balanced held-out draw has no rows for class {level}")
        means.append(selected.mean())
    return float(torch.stack(means).mean())


def _evaluate(
    run: CompiledRun, result: StageResult, test: ClusterPopulation
) -> dict[str, float]:
    population = result.population
    if population is None or result.teacher is None:
        raise RuntimeError("remixmatch benchmark expected population and EMA teacher")
    scaled = on_the_training_scale(test.batch, population)
    with torch.no_grad():
        student = run.graph.evaluate(
            scaled, schema=run.recipe.schema, only=run.graph.names
        )
        teacher = result.teacher.graph.evaluate(
            scaled, schema=run.recipe.schema, only=run.graph.names
        )
        student_propensity = student[Port.T_GIVEN_X]
        teacher_propensity = teacher[Port.T_GIVEN_X]
        teacher_outcome = teacher[Port.Y_GIVEN_XT]
        if (
            not isinstance(student_propensity, CategoricalTreatment)
            or not isinstance(teacher_propensity, CategoricalTreatment)
            or not isinstance(teacher_outcome, GaussianOutcome)
        ):
            raise TypeError("remixmatch benchmark expected its reviewed P5 heads")
        student_nll = F.nll_loss(
            student_propensity.log_probs, scaled.t, reduction="none"
        )
        teacher_nll = F.nll_loss(
            teacher_propensity.log_probs, scaled.t, reduction="none"
        )
        return {
            "student_macro_nll": _macro_mean(student_nll, scaled.t),
            "ema_macro_nll": _macro_mean(teacher_nll, scaled.t),
            "outcome_nll": float(-teacher_outcome.log_prob(scaled.y, scaled.t).mean()),
        }


def _guess(result: StageResult) -> AnchoredLabelGuess:
    state = result.objective_states.get(GUESS_OWNER)
    if not isinstance(state, AnchoredLabelGuess):
        raise RuntimeError("remixmatch result did not expose its anchor state")
    return state


def _marginal_l1(result: StageResult, true_treatment: Tensor) -> float:
    """Original gate: unlabelled weak-anchor window versus all-training truth.

    ``_TRAIN_PRIOR`` generates clusters.  Treatment is subsequently sampled
    from the 0.98-on-cluster assignment distribution, so that tuple is neither
    the population treatment parameter nor, after the finite draw, this
    replicate's true training marginal.  The benchmark may read all hidden
    labels; the recipe cannot.
    """
    marginal = _guess(result).prediction_marginal
    truth = torch.bincount(true_treatment, minlength=_CLASSES).to(
        dtype=marginal.dtype, device=marginal.device
    )
    truth /= truth.sum()
    return float((marginal - truth).abs().sum())


def _marginal_diagnostics(
    full: StageResult, no_alignment: StageResult, truth: XTYBatch
) -> dict[str, float]:
    """Read finite-label target error after fitting; never change training state."""
    population = full.population
    if population is None:
        raise RuntimeError("marginal diagnostics need the training population")
    missing_ids = population.rows.row_id[population.rows.t_missing]
    missing = torch.isin(truth.row_id, missing_ids)
    if not missing.any() or int(missing.sum()) != missing_ids.numel():
        raise RuntimeError("marginal diagnostics could not match unlabelled row IDs")

    def histogram(labels: Tensor) -> Tensor:
        counts = torch.bincount(labels, minlength=_CLASSES).to(torch.float64).cpu()
        return counts / counts.sum()

    labelled = _guess(full).labelled_marginal
    # Both arms estimate `p(y)` from the same 64 rows in the same order, so the
    # promoted guardrail compares two windows against one shared vector rather
    # than each against its own.  Only the full arm reads it; if they ever
    # diverged the comparison would be between two different targets.
    if not torch.equal(labelled, _guess(no_alignment).labelled_marginal):
        raise RuntimeError("remixmatch paired arms estimated different p(y)")
    aligned = _guess(full).prediction_marginal
    unaligned = _guess(no_alignment).prediction_marginal
    true_training = histogram(truth.t)
    true_unlabelled = histogram(truth.t[missing])

    def l1(left: Tensor, right: Tensor) -> float:
        return float((left - right).abs().sum())

    values = {
        "labelled_vs_true_training_marginal_L1": l1(labelled, true_training),
        "labelled_vs_true_unlabelled_marginal_L1": l1(labelled, true_unlabelled),
        "true_unlabelled_vs_training_marginal_L1": l1(true_unlabelled, true_training),
        "aligned_vs_labelled_marginal_L1": l1(aligned, labelled),
        "unaligned_vs_labelled_marginal_L1": l1(unaligned, labelled),
        "alignment_labelled_marginal_L1_advantage": (
            l1(unaligned, labelled) - l1(aligned, labelled)
        ),
        "aligned_vs_true_unlabelled_marginal_L1": l1(aligned, true_unlabelled),
        "unaligned_vs_true_unlabelled_marginal_L1": l1(unaligned, true_unlabelled),
        "alignment_true_unlabelled_marginal_L1_advantage": (
            l1(unaligned, true_unlabelled) - l1(aligned, true_unlabelled)
        ),
    }
    for name, marginal in (
        ("estimated_labelled", labelled),
        ("true_training", true_training),
        ("true_unlabelled", true_unlabelled),
        ("aligned_window", aligned),
        ("unaligned_window", unaligned),
    ):
        values.update(
            {
                f"{name}_class_{level}": float(marginal[level])
                for level in range(_CLASSES)
            }
        )
    return values


@torch.no_grad()
def _terminal_mechanism(
    run: CompiledRun, result: StageResult, *, rng_key: int
) -> dict[str, float]:
    """Read terminal target/copy agreement and the realised MixUp distribution."""
    population = result.population
    if population is None:
        raise RuntimeError("remixmatch terminal reading needs the population")
    state = run.state(
        run.stage("joint_fit"),
        population.rows,
        rng_key=rng_key,
        population=population,
    )
    mix = run.recipe.mixes[0]
    coefficients = torch.cat(
        [
            state.mixing_plan(mix.output(index)).coefficient
            for index in range(len(mix.members))
        ]
    )
    rows = torch.nonzero(population.rows.t_missing, as_tuple=False).flatten()
    guess = _guess(result)
    anchor = state[Realisation(view="weak_x")][Port.T_GIVEN_X]
    if not isinstance(anchor, CategoricalTreatment):
        raise TypeError("remixmatch terminal anchor has no treatment distribution")
    raw = anchor.probs.index_select(0, rows).to(torch.float64)
    target = raw * (
        (guess.labelled_marginal.to(raw.device) + 1e-6)
        / (guess.prediction_marginal.to(raw.device) + 1e-6)
    )
    target = target / target.sum(dim=-1, keepdim=True)
    target = target.square()
    target = target / target.sum(dim=-1, keepdim=True)
    agreements = []
    with torch.no_grad():
        for draw in range(8):
            viewed = run.recipe.view("strong_x").apply(
                population.rows,
                run.recipe.schema,
                rng_key=rng_key,
                draw=draw,
                population=population,
            )
            values = run.graph.evaluate(
                viewed, schema=run.recipe.schema, only=run.graph.names
            )
            prediction = values[Port.T_GIVEN_X]
            if not isinstance(prediction, CategoricalTreatment):
                raise TypeError("remixmatch strong copy has no treatment distribution")
            copy = prediction.probs.index_select(0, rows).to(torch.float64)
            agreements.append((target * copy).sum(dim=-1).mean())
    terminal = range(len(result.records) - _TERMINAL_STEPS, len(result.records))
    pretext_accuracy = sum(
        _diagnostic(result, step, _PRETEXT, "accuracy") for step in terminal
    ) / len(terminal)
    lambda_mean = sum(
        (
            _diagnostic(result, step, _MIXED_LABELLED, "lambda_mean")
            + _diagnostic(result, step, _MIXED_UNLABELLED, "lambda_mean")
        )
        / 2.0
        for step in terminal
    ) / len(terminal)
    entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1).mean()
    return {
        "pretext_accuracy": pretext_accuracy,
        "target_copy_agreement": float(torch.stack(agreements).mean()),
        "anchor_entropy": float(entropy),
        "lambda_min": float(coefficients.min()),
        "lambda_max": float(coefficients.max()),
        "lambda_mean": lambda_mean,
    }


def _diagnostic(result: StageResult, step: int, name: str, field: str) -> float:
    for term in result.records[step].terms:
        if term.name == name:
            try:
                return float(term.diagnostics[field])
            except KeyError:
                raise RuntimeError(
                    f"term {name!r} logged no {field!r} at step {step}"
                ) from None
    raise RuntimeError(f"step {step} has no term named {name!r}")


__all__ = ["run"]
