"""UDA's four paired mechanism arms from card section 6.

The fixture is `fixmatch.md` §6.1's, which `uda.md` §6.1 restates verbatim, so
the two cards' numbers are about the recipes rather than about two worlds.

Four arms run from one initialisation and one batch/view stream: full UDA, the
`lambda_uda = 0` arm that carries the primary attribution, a `tau = 1` arm and
an arm whose training-signal-annealed supervised term is replaced by the
ordinary observed-treatment NLL. Only the first pair is a target. Card §6 says
so twice — "report sharpening and TSA effects without retroactively choosing
their sign" — so the other two arms enter the result as informational ratios
and nothing here reads their direction.

Two of the card's tolerance clauses are about the objective's arithmetic rather
than about a fit, and both are read at **step 0**, the one step at which the
arms share weak logits: `tau` must lower the target's entropy and must leave the
gate's membership alone. They are recorded as all-replicate boolean guardrails
(`bool_float`), which is what makes a `tau` that leaked into the gate a Tier 2
failure rather than a footnote.

§6.2's view label-flip rule is measured on all ten fixtures too, but reports
rather than decides: §6.2 names the primary NLL target and the outcome
guardrail as the only things that set this card's status, and calls a broken
flip ceiling "a data-policy failure, not evidence against UDA".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Constant,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    ClusterPopulation,
    bool_float,
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
    ConfidenceMaskedConsistencyLoss,
    ObservedTreatmentNLL,
    TrainingSignalAnnealedTreatmentNLL,
)
from xty2.recipes import uda
from xty2.recipes.uda import WEAK_X
from xty2.training import StageResult, run_stage

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 94_000
_CONSISTENCY = "uda_consistency"
_TSA = "tsa_observed_treatment_nll"
_SIGNAL_COLUMNS = 4
"""Columns 0-3 carry the cluster signal; the Bayes boundary is their sum."""


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten four-arm replicates predeclared by ``uda.md`` section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in §6.1"
            ),
            "variant": (
                "four paired arms — full UDA; no-consistency (lambda_uda=0); "
                "no-sharpening (tau=1); no-TSA (ordinary observed-treatment "
                "NLL); all other mechanics paired"
            ),
            "split": (
                "1024 train rows with exactly 64 observed treatments; 2048 "
                "held-out rows with every treatment observed"
            ),
            "metric": (
                "held-out treatment NLL for student and evaluation EMA; "
                "held-out outcome NLL guardrail; UDA gate coverage/confidence "
                "and target entropy; TSA retained fraction and ceiling"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "full/no-consistency held-out treatment-NLL ratio < 1.0 in mean "
                "by at least one stderr for student and EMA; outcome NLL <= "
                "1.05x no-consistency; tau=0.4 target entropy < tau=1 on matched "
                "weak logits; tau must not change gate membership on fixed "
                "logits; report the learned TSA retained fraction without "
                "imposing a direction; report sharpening and TSA effects "
                "without retroactively choosing their sign"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(f"uda card reviewed ten replicates, got {spec.seed_count}")
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            # "< 1.0" in the card against "<= 1.0" here, as `fixmatch`'s and
            # `doublematch`'s modules record: one point of a continuous
            # statistic, and a strict relation added to the reporting
            # vocabulary for one card would be `DESIGN.md` §11.2's convenience
            # quadrant with nothing riding on it.
            MetricResult.upper_bound(
                "student_treatment_NLL_ratio", column(rows, "student_ratio"), 1.0
            ),
            MetricResult.upper_bound(
                "ema_treatment_NLL_ratio", column(rows, "ema_ratio"), 1.0
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio", column(rows, "outcome_ratio"), 1.05
            ),
            MetricResult.lower_bound(
                "sharpening_lowers_target_entropy",
                column(rows, "entropy_guardrail"),
                1.0,
            ),
            MetricResult.lower_bound(
                "sharpening_leaves_the_gate_unchanged",
                column(rows, "gate_guardrail"),
                1.0,
            ),
            # §6.2's Tier 1 rule 8 — weak < strong <= 5% of Bayes labels
            # flipped — measured on all ten fixtures rather than on Tier 1's
            # one, and **informational**. Card §6.2 says what may decide this
            # card's status: "Only the primary full-versus-no-consistency NLL
            # target plus outcome guardrail determines `reproduced` versus
            # `deviating`", and the same card calls a broken flip ceiling "a
            # data-policy failure, not evidence against UDA". A required
            # metric here would let the fixture's view strength answer a
            # question about the method. It is reported per fixture, as a
            # boolean and as the worst rate, so that a breach is visible in
            # the result rather than absent from it — see §6.2, where the one
            # fixture over the ceiling is recorded.
            MetricResult.information(
                "views_respect_the_label_flip_guardrail",
                column(rows, "flip_guardrail"),
            ),
            MetricResult.information(
                "uda_ema_treatment_NLL", column(rows, "uda_ema_nll"), unit="nat/row"
            ),
            MetricResult.information(
                "no_consistency_ema_treatment_NLL",
                column(rows, "no_consistency_ema_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "uda_student_treatment_NLL",
                column(rows, "uda_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "no_consistency_student_treatment_NLL",
                column(rows, "no_consistency_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "marginal_frequency_NLL",
                column(rows, "frequency_nll"),
                unit="nat/row",
            ),
            # Mechanism effects, reported without a predeclared sign.
            MetricResult.information(
                "no_sharpening_ema_treatment_NLL_ratio",
                column(rows, "no_sharpening_ema_ratio"),
            ),
            MetricResult.information(
                "no_sharpening_student_treatment_NLL_ratio",
                column(rows, "no_sharpening_student_ratio"),
            ),
            MetricResult.information(
                "no_tsa_ema_treatment_NLL_ratio", column(rows, "no_tsa_ema_ratio")
            ),
            MetricResult.information(
                "no_tsa_student_treatment_NLL_ratio",
                column(rows, "no_tsa_student_ratio"),
            ),
            MetricResult.information(
                "initial_TSA_retained_fraction", column(rows, "initial_tsa_fraction")
            ),
            MetricResult.information(
                "terminal_TSA_retained_fraction",
                column(rows, "terminal_tsa_fraction"),
            ),
            MetricResult.information(
                "terminal_TSA_ceiling", column(rows, "terminal_tsa_ceiling")
            ),
            MetricResult.information(
                "initial_gate_coverage", column(rows, "initial_coverage")
            ),
            MetricResult.information(
                "terminal_gate_coverage", column(rows, "terminal_coverage")
            ),
            MetricResult.information(
                "terminal_accepted_confidence",
                column(rows, "terminal_accepted_confidence"),
            ),
            MetricResult.information(
                "initial_target_entropy", column(rows, "initial_target_entropy")
            ),
            MetricResult.information(
                "terminal_target_entropy", column(rows, "terminal_target_entropy")
            ),
            MetricResult.information(
                "untrained_weak_confidence", column(rows, "untrained_confidence")
            ),
            MetricResult.information(
                "weak_view_label_flip_rate", column(rows, "weak_flip_rate")
            ),
            MetricResult.information(
                "strong_view_label_flip_rate", column(rows, "strong_flip_rate")
            ),
            MetricResult.information(
                "observed_treatment_prevalence", column(rows, "observed_prevalence")
            ),
            MetricResult.information(
                "missing_treatment_prevalence", column(rows, "missing_prevalence")
            ),
            MetricResult.information(
                "held_out_treatment_prevalence", column(rows, "held_out_prevalence")
            ),
        ),
        interpretation=(
            "This is the predeclared project-local UDA mechanism target: "
            "whether eq. (1)'s gated, temperature-sharpened consistency term "
            "improves a treatment propensity over the otherwise identical fit "
            "with that term's weight set to zero, TSA retained in both. It is "
            "not a reproduction of any published UDA number: the inputs, the "
            "augmentations, the architecture and the 3,000-step budget all "
            "differ (deviations 1-4), and appendix B.2's CIFAR-10 error has no "
            "counterpart here. The sharpening and TSA arms are mechanism "
            "measurements, not targets."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = two_cluster_population(_TRAIN_ROWS, seed=base + 1, row_offset=0)
    test = two_cluster_population(_TEST_ROWS, seed=base + 2, row_offset=10_000)
    data = training_dataset(schema, train.batch)
    stage_seed = base + 10_000

    arms: tuple[tuple[str, Callable[[Recipe], Recipe]], ...] = (
        ("full", lambda recipe: recipe),
        ("no_consistency", _without_consistency),
        ("no_sharpening", _without_sharpening),
        ("no_tsa", _without_tsa),
    )
    recipes: dict[str, Recipe] = {}
    for name, build in arms:
        torch.manual_seed(base + 6)
        recipes[name] = build(uda(schema))
    reference = recipes["full"].system.state_dict()
    for name, recipe in recipes.items():
        for parameter, value in reference.items():
            if not torch.equal(value, recipe.system.state_dict()[parameter]):
                raise RuntimeError(
                    f"uda arm {name!r} starts from different parameters at "
                    f"{parameter!r}"
                )

    runs = {name: compile(recipe) for name, recipe in recipes.items()}
    # Card §6.1's pre-training reports, read from the shared initialisation
    # before any arm has taken a gradient.
    flips = _flip_rates(recipes["full"], train.batch, stage_seed)
    untrained = _untrained_weak_confidence(
        runs["full"], recipes["full"], train.batch, stage_seed
    )

    results = {
        name: run_stage(run, "joint_fit", data, seed=stage_seed)
        for name, run in runs.items()
    }
    reference_rows = results["full"].checkpoint.trained_on_row_ids
    for name, result in results.items():
        if not torch.equal(result.checkpoint.trained_on_row_ids, reference_rows):
            raise RuntimeError(f"uda arm {name!r} saw different training rows")

    metrics = {
        name: _evaluate(runs[name], result, test) for name, result in results.items()
    }
    for name in ("ema_treatment_nll", "student_treatment_nll", "ema_outcome_nll"):
        if metrics["no_consistency"][name] <= 0.0:
            raise RuntimeError(
                f"the no-consistency arm produced a non-positive {name}, so the "
                "paired ratio the card declares is undefined"
            )

    full = results["full"]
    ordinary = results["no_sharpening"]
    return {
        "student_ratio": (
            metrics["full"]["student_treatment_nll"]
            / metrics["no_consistency"]["student_treatment_nll"]
        ),
        "ema_ratio": (
            metrics["full"]["ema_treatment_nll"]
            / metrics["no_consistency"]["ema_treatment_nll"]
        ),
        "outcome_ratio": (
            metrics["full"]["ema_outcome_nll"]
            / metrics["no_consistency"]["ema_outcome_nll"]
        ),
        "no_sharpening_ema_ratio": (
            metrics["no_sharpening"]["ema_treatment_nll"]
            / metrics["no_consistency"]["ema_treatment_nll"]
        ),
        "no_sharpening_student_ratio": (
            metrics["no_sharpening"]["student_treatment_nll"]
            / metrics["no_consistency"]["student_treatment_nll"]
        ),
        "no_tsa_ema_ratio": (
            metrics["no_tsa"]["ema_treatment_nll"]
            / metrics["no_consistency"]["ema_treatment_nll"]
        ),
        "no_tsa_student_ratio": (
            metrics["no_tsa"]["student_treatment_nll"]
            / metrics["no_consistency"]["student_treatment_nll"]
        ),
        "entropy_guardrail": bool_float(
            _diagnostic(full, 0, _CONSISTENCY, "target_entropy")
            < _diagnostic(ordinary, 0, _CONSISTENCY, "target_entropy")
        ),
        # Identical coverage *and* identical mean accepted confidence at the
        # one step whose weak logits the two arms share. The gate reads
        # untempered probabilities, so a `tau` that reached it would move both.
        "gate_guardrail": bool_float(
            _diagnostic(full, 0, _CONSISTENCY, "coverage")
            == _diagnostic(ordinary, 0, _CONSISTENCY, "coverage")
            and _diagnostic(full, 0, _CONSISTENCY, "accepted_confidence")
            == _diagnostic(ordinary, 0, _CONSISTENCY, "accepted_confidence")
        ),
        "flip_guardrail": bool_float(0.0 <= flips[0] < flips[1] <= 0.05),
        "initial_tsa_fraction": _diagnostic(full, 0, _TSA, "retained_fraction"),
        "terminal_tsa_fraction": _diagnostic(full, -1, _TSA, "retained_fraction"),
        "terminal_tsa_ceiling": _diagnostic(full, -1, _TSA, "tsa_ceiling"),
        "initial_coverage": _diagnostic(full, 0, _CONSISTENCY, "coverage"),
        "terminal_coverage": _diagnostic(full, -1, _CONSISTENCY, "coverage"),
        "terminal_accepted_confidence": _diagnostic(
            full, -1, _CONSISTENCY, "accepted_confidence"
        ),
        "initial_target_entropy": _diagnostic(full, 0, _CONSISTENCY, "target_entropy"),
        "terminal_target_entropy": _diagnostic(
            full, -1, _CONSISTENCY, "target_entropy"
        ),
        "untrained_confidence": untrained,
        "weak_flip_rate": flips[0],
        "strong_flip_rate": flips[1],
        "uda_ema_nll": metrics["full"]["ema_treatment_nll"],
        "no_consistency_ema_nll": metrics["no_consistency"]["ema_treatment_nll"],
        "uda_student_nll": metrics["full"]["student_treatment_nll"],
        "no_consistency_student_nll": (
            metrics["no_consistency"]["student_treatment_nll"]
        ),
        "frequency_nll": metrics["full"]["frequency_nll"],
        **_prevalence(full, test),
    }


def _without_consistency(recipe: Recipe) -> Recipe:
    """The primary attribution arm: `lambda_uda = 0` and nothing else.

    The term stays in the program at weight zero rather than being removed, so
    both arms plan the same forward passes over the same two views and the
    difference between them is the scheduled gradient.
    """
    stage = recipe.program[0]
    weighted = stage.objectives[2]
    if not isinstance(weighted.objective, ConfidenceMaskedConsistencyLoss):
        raise RuntimeError("uda's third term is no longer the consistency loss")
    objectives = (
        *stage.objectives[:2],
        replace(weighted, weight=Constant(0.0)),
        *stage.objectives[3:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _without_sharpening(recipe: Recipe) -> Recipe:
    """`tau = 1`: the ordinary weak softmax as the consistency target."""
    stage = recipe.program[0]
    weighted = stage.objectives[2]
    if not isinstance(weighted.objective, ConfidenceMaskedConsistencyLoss):
        raise RuntimeError("uda's third term is no longer the consistency loss")
    objectives = (
        *stage.objectives[:2],
        replace(
            weighted, objective=replace(weighted.objective, target_temperature=1.0)
        ),
        *stage.objectives[3:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _without_tsa(recipe: Recipe) -> Recipe:
    """Appendix A.1's ceiling removed, the same weak view and weight kept."""
    stage = recipe.program[0]
    weighted = stage.objectives[1]
    if not isinstance(weighted.objective, TrainingSignalAnnealedTreatmentNLL):
        raise RuntimeError("uda's second term is no longer the TSA loss")
    replacement = Weighted(
        ObservedTreatmentNLL(realisation=WEAK_X),
        weight=weighted.weight,
        reduction=weighted.reduction,
    )
    objectives = (stage.objectives[0], replacement, *stage.objectives[2:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _flip_rates(
    recipe: Recipe, train: XTYBatch, stage_seed: int
) -> tuple[float, float]:
    """Bayes-label flip rate of each view, measured before training.

    The latent cluster is not retained by `XTYBatch`, but for this symmetric
    mixture the Bayes boundary is `sum(x[0:4]) = 0`, which is enough to say
    whether a masked view has crossed it. Tier 1 measures the same two numbers
    on seed 94000; this is the same computation on all ten fixtures.
    """
    label = train.x[:, :_SIGNAL_COLUMNS].sum(dim=-1) > 0
    rates = []
    for view in ("weak_x", "strong_x"):
        realised = recipe.view(view).apply(train, recipe.schema, rng_key=stage_seed)
        crossed = realised.x[:, :_SIGNAL_COLUMNS].sum(dim=-1) > 0
        rates.append(float((crossed != label).float().mean()))
    return rates[0], rates[1]


def _untrained_weak_confidence(
    run: CompiledRun, recipe: Recipe, train: XTYBatch, stage_seed: int
) -> float:
    """Card §6.1's untrained weak-confidence report, as its mean.

    Read before `run_stage`, which trains `run.graph` in place.
    """
    weak = recipe.view("weak_x").apply(train, recipe.schema, rng_key=stage_seed)
    with torch.no_grad():
        values = run.graph.evaluate(weak, schema=recipe.schema, only=run.graph.names)
    propensity = values[Port.T_GIVEN_X]
    if not isinstance(propensity, CategoricalTreatment):
        raise TypeError("uda benchmark expected a categorical propensity")
    return float(propensity.probs.max(dim=-1).values.mean())


def _prevalence(result: StageResult, test: ClusterPopulation) -> dict[str, float]:
    """Treatment prevalence by split and by missingness, per card §6.1."""
    population = result.population
    if population is None:
        raise RuntimeError("uda benchmark expected a training population")
    rows = population.rows
    observed = rows.t_observed
    if not bool(observed.any()) or bool(observed.all()):
        raise RuntimeError(
            "the recipe's data policy did not leave both an observed and a "
            "missing treatment population"
        )
    return {
        "observed_prevalence": float(rows.t[observed].float().mean()),
        "missing_prevalence": float(rows.t[~observed].float().mean()),
        "held_out_prevalence": float(test.batch.t.float().mean()),
    }


def _evaluate(
    run: CompiledRun, result: StageResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out NLLs off both parameter sets the card's tolerance names."""
    population = result.population
    if population is None:
        raise RuntimeError("uda benchmark expected a training population")
    if result.teacher is None:
        raise RuntimeError("uda declares an evaluation EMA and reported none")
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        teacher = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        teacher_propensity = teacher[Port.T_GIVEN_X]
        teacher_outcome = teacher[Port.Y_GIVEN_XT]
        if not isinstance(teacher_propensity, CategoricalTreatment) or not isinstance(
            teacher_outcome, GaussianOutcome
        ):
            raise TypeError("uda benchmark expected its reviewed P5 heads")

        student = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        student_propensity = student[Port.T_GIVEN_X]
        if not isinstance(student_propensity, CategoricalTreatment):
            raise TypeError("uda benchmark expected a categorical propensity")

        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(
            observed, minlength=schema.treatment_cardinality
        ).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        return {
            "ema_treatment_nll": float(
                F.nll_loss(teacher_propensity.log_probs, scaled.t)
            ),
            "ema_outcome_nll": float(
                -teacher_outcome.log_prob(scaled.y, scaled.t).mean()
            ),
            "student_treatment_nll": float(
                F.nll_loss(student_propensity.log_probs, scaled.t)
            ),
            "frequency_nll": float(F.nll_loss(baseline, scaled.t)),
        }


def _diagnostic(result: StageResult, step: int, name: str, field: str) -> float:
    """One logged field of one term at one step, or a named failure.

    A missing diagnostic would otherwise surface as a `KeyError` inside a spawn
    worker with no indication of which term or which step lost it.
    """
    for term in result.records[step].terms:
        if term.name != name:
            continue
        try:
            return float(term.diagnostics[field])
        except KeyError:
            raise RuntimeError(
                f"term {name!r} logged no {field!r} at step {step}; it logged "
                f"{sorted(term.diagnostics)!r}"
            ) from None
    raise RuntimeError(f"step {step} has no term named {name!r}")


__all__ = ["run"]
