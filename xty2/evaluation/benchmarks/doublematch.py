"""DoubleMatch's paired `w_s = 0` benchmark from card section 6.

The fixture is `fixmatch.md` §6.1's, imported rather than restated, because
`doublematch.md` §6.1 adopts it "in full and without modification" and the
`w_s = 0` arm of this pair is — by the paper's own §III — FixMatch. Two
transcriptions of one DGP is how the two cards' numbers would stop being about
the recipes.

What is new here is what gets measured. Eq. (3) has no gate, so the mask rate
and impurity `fixmatch`'s module reports say nothing about the term under test;
and the term's own value cannot stand in for them, because a collapsed encoder
attains its minimum. So the guardrails are the pair the objective logs for
exactly that reason — feature alignment and target concentration — and the
concentration one is read over the **whole trajectory**, not its last steps.
Card §6.2 is why: the architecture deviation 10 withdrew passes a terminal
reading of that guardrail and fails a trajectory one, having spent 135 steps
above 0.99 before climbing back out.

Predictive metrics come from both parameter sets. `fixmatch.md` §6.2 records
that an EMA number can improve while the network under it gets worse, and
eq. (3) trains the encoder both arms share, which is where that divergence
would hide.
"""

from __future__ import annotations

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
from xty2.evaluation.benchmarks.fixmatch import (
    _BASE_SEED,
    _PSEUDO_LABEL,
    _TEST_ROWS,
    _TRAIN_ROWS,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.objectives import CosineFeatureConsistency
from xty2.recipes import doublematch
from xty2.training import StageResult, run_stage

_SELF_SUPERVISED = "cosine_feature_consistency"
_TERMINAL_STEPS = 100
"""What "terminal" counts, and it is a window rather than the last step.

Card §6's tolerance names a terminal alignment without fixing the window. One
step of a stochastic quota is a draw, not a terminus; a hundred is the window
Tier 1 already reads its late means over (`tests/smoke/test_doublematch.py`),
so the two tiers report the same statistic under the same name.
"""


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired DoubleMatch / `w_s = 0` fits."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in fixmatch.md 6.1 and reused unchanged"
            ),
            "variant": (
                "paired fit against an otherwise identical w_s = 0 ablation, "
                "same seeds and same batches"
            ),
            "split": (
                "1024 train rows with 64 observed treatments, 2048 held-out "
                "rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio on both the EMA and the trained "
                "parameters, DoubleMatch over the w_s = 0 ablation"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "ratio < 1.0 in mean on both the EMA and the trained "
                "parameters (fixmatch.md 6.2's rule, and 6.2 below is why this "
                "card may not weaken it); held-out outcome NLL within 1.05x of "
                "the ablation; terminal alignment (mean cos(h(v), z)) above 0.5 "
                "while target concentration stays below 0.9 at *every* step, "
                "not merely at the end - the architecture deviation 10 withdrew "
                "passes a terminal reading of that guardrail and fails a "
                "trajectory one (6.2)"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"doublematch card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            # "< 1.0" in the card against "<= 1.0" here, as `fixmatch`'s module
            # records: one point of a continuous statistic, and a strict
            # relation added to the reporting vocabulary for one card would be
            # `DESIGN.md` §11.2's convenience quadrant with nothing riding on
            # it.
            MetricResult.upper_bound(
                "ema_treatment_NLL_ratio", column(rows, "ema_ratio"), 1.0
            ),
            MetricResult.upper_bound(
                "trained_treatment_NLL_ratio", column(rows, "trained_ratio"), 1.0
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio", column(rows, "outcome_ratio"), 1.05
            ),
            MetricResult.lower_bound(
                "terminal_feature_alignment", column(rows, "alignment"), 0.5
            ),
            MetricResult.upper_bound(
                "max_target_concentration", column(rows, "max_concentration"), 0.9
            ),
            MetricResult.information(
                "doublematch_ema_treatment_NLL",
                column(rows, "doublematch_ema_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "ablation_ema_treatment_NLL",
                column(rows, "ablation_ema_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "marginal_frequency_NLL",
                column(rows, "frequency_nll"),
                unit="nat/row",
            ),
            # Eq. (3) is meant to train the encoder the gate reads, so the two
            # arms' terminal coverage is the number that says whether it moved
            # the thing eq. (2) consumes. Informational: no direction for it is
            # predeclared, and card §6.2 measured none.
            MetricResult.information(
                "doublematch_terminal_coverage", column(rows, "coverage")
            ),
            MetricResult.information(
                "ablation_terminal_coverage", column(rows, "ablation_coverage")
            ),
        ),
        interpretation=(
            "This is the predeclared project-local DoubleMatch mechanism "
            "target: does eq. (3)'s detached weak-feature/projected "
            "strong-feature cosine consistency, trained on the rows eq. (2)'s "
            "gate rejects, improve a *treatment* propensity over the otherwise "
            "identical w_s = 0 fit — which by the paper's §III is FixMatch. It "
            "is not a reproduction of Wallin et al., whose inputs, labels, "
            "architecture, metric and 352,000-step budget all differ and whose "
            "headline claim is about training speed, which a fixed shared "
            "budget cannot measure (deviation 3)."
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
    scheduled_recipe = doublematch(schema)
    torch.manual_seed(base + 6)
    ablated_recipe = _zero_weight(doublematch(schema))
    for name, value in scheduled_recipe.system.state_dict().items():
        if not torch.equal(value, ablated_recipe.system.state_dict()[name]):
            raise RuntimeError(f"doublematch paired initial state differs at {name!r}")

    scheduled_run = compile(scheduled_recipe)
    ablated_run = compile(ablated_recipe)
    # Identical seeds: both stages are index 0 of a one-stage program, so the
    # sampler stream, the view keys and the parameter initialisation all match
    # and `w_s` is the only difference between the arms.
    scheduled = run_stage(scheduled_run, "joint_fit", data, seed=base + 10_000)
    ablated = run_stage(ablated_run, "joint_fit", data, seed=base + 10_000)

    doublematch_metrics = _evaluate(scheduled_run, scheduled, test)
    ablation_metrics = _evaluate(ablated_run, ablated, test)
    for name, value in ablation_metrics.items():
        if name.endswith("nll") and value <= 0.0:
            raise RuntimeError(
                f"the ablation produced a non-positive {name}, so the paired "
                "ratio the card declares is undefined"
            )
    return {
        "ema_ratio": (
            doublematch_metrics["ema_treatment_nll"]
            / ablation_metrics["ema_treatment_nll"]
        ),
        "trained_ratio": (
            doublematch_metrics["trained_treatment_nll"]
            / ablation_metrics["trained_treatment_nll"]
        ),
        "outcome_ratio": (
            doublematch_metrics["ema_outcome_nll"] / ablation_metrics["ema_outcome_nll"]
        ),
        "alignment": _alignment(scheduled),
        "max_concentration": _max_target_concentration(scheduled),
        "coverage": _terminal_coverage(scheduled),
        "ablation_coverage": _terminal_coverage(ablated),
        "doublematch_ema_nll": doublematch_metrics["ema_treatment_nll"],
        "ablation_ema_nll": ablation_metrics["ema_treatment_nll"],
        "frequency_nll": doublematch_metrics["frequency_nll"],
    }


def _zero_weight(recipe: Recipe) -> Recipe:
    """The `w_s = 0` ablation: the same fit without eq. (3)'s gradient.

    The term stays in the program at weight zero rather than being removed, so
    the two arms plan the same four forward passes and the same graph — the
    projection head included, still constructed and still initialised from the
    same draws — and the difference between them is the scheduled gradient.
    """
    stage = recipe.program[0]
    consistency = stage.objectives[3].objective
    if not isinstance(consistency, CosineFeatureConsistency):
        raise RuntimeError(
            "doublematch's fourth term is no longer the feature-consistency loss"
        )
    ablated = replace(stage.objectives[3], weight=Constant(0.0))
    objectives = (*stage.objectives[:3], ablated, *stage.objectives[4:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _evaluate(
    run: CompiledRun, result: StageResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out NLLs off the EMA the paper reports, and off the network under it."""
    population = result.population
    if population is None:
        raise RuntimeError(
            "the fitting stage reported no training population; the recipe "
            "declares a sampler, so one is expected"
        )
    if result.teacher is None:
        raise RuntimeError("doublematch declares an evaluation EMA and reported none")
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        values = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        if not isinstance(propensity, CategoricalTreatment) or not isinstance(
            outcome, GaussianOutcome
        ):
            raise TypeError("doublematch benchmark expected its reviewed P5 heads")
        ema_treatment_nll = float(F.nll_loss(propensity.log_probs, scaled.t))
        ema_outcome_nll = float(-outcome.log_prob(scaled.y, scaled.t).mean())

        trained = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        trained_propensity = trained[Port.T_GIVEN_X]
        if not isinstance(trained_propensity, CategoricalTreatment):
            raise TypeError("doublematch benchmark expected a categorical propensity")
        trained_treatment_nll = float(
            F.nll_loss(trained_propensity.log_probs, scaled.t)
        )

        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(observed, minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        frequency_nll = float(F.nll_loss(baseline, scaled.t))
    return {
        "ema_treatment_nll": ema_treatment_nll,
        "trained_treatment_nll": trained_treatment_nll,
        "ema_outcome_nll": ema_outcome_nll,
        "frequency_nll": frequency_nll,
    }


def _diagnostic(result: StageResult, step: int, name: str, field: str) -> float:
    """One logged field of one term at one step, or a named failure.

    A missing diagnostic would otherwise surface as a `KeyError` inside a spawn
    worker with no indication of which term or which step lost it.
    """
    record = result.records[step]
    for term in record.terms:
        if term.name != name:
            continue
        if field == "value":
            return float(term.value)
        try:
            return float(term.diagnostics[field])
        except KeyError:
            raise RuntimeError(
                f"term {name!r} logged no {field!r} at step {step}; it logged "
                f"{sorted(term.diagnostics)!r}"
            ) from None
    raise RuntimeError(f"step {step} has no term named {name!r}")


def _alignment(result: StageResult) -> float:
    """`mean cos(h(v), z)` over the terminal window — the term's value, negated.

    Eq. (3) is `-cos`, so the alignment the card's tolerance names is the
    negative of what the objective logs. Taken from the run rather than
    recomputed on held-out rows: the invariance eq. (3) trains is a property of
    the two *views*, which the held-out population is not drawn under.
    """
    steps = range(len(result.records) - _TERMINAL_STEPS, len(result.records))
    values = [-_diagnostic(result, step, _SELF_SUPERVISED, "value") for step in steps]
    return sum(values) / len(values)


def _max_target_concentration(result: StageResult) -> float:
    """The worst point of the collapse guardrail, over every step of the run."""
    return max(
        _diagnostic(result, step, _SELF_SUPERVISED, "target_concentration")
        for step in range(len(result.records))
    )


def _terminal_coverage(result: StageResult) -> float:
    """What fraction of the batch eq. (2)'s gate retained, over the same window."""
    steps = range(len(result.records) - _TERMINAL_STEPS, len(result.records))
    values = [_diagnostic(result, step, _PSEUDO_LABEL, "coverage") for step in steps]
    return sum(values) / len(values)
