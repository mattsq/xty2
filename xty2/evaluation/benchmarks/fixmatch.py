"""FixMatch's paired `lambda_u = 0` benchmark from card section 6.

Two parameter sets, and the split between them is not cosmetic. Predictive
metrics come from the **EMA** copy, because that is the model section 2.4
reports. The paper's mask rate (eq. 6) and impurity (eq. 5) come from the
**trained** network, because they describe the labels the run actually trained
on — eq. (4) reads the current parameters, so an EMA mask rate would be a
statistic of a model that never gated anything.

The batch composition is the recipe's now: `QuotaSampler` mixes eq. (3)'s
`B = 64` labelled rows with eq. (4)'s `mu B = 448` unlabelled ones every step,
and the 64-label budget and the outcome scaling are `DataSpec` declarations.
This module supplies rows and evaluates; it does not decide any of the three.
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
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.objectives import PseudoLabelTreatmentNLL
from xty2.recipes import fixmatch
from xty2.training import StageResult, run_stage

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 90_000
_PSEUDO_LABEL = "pseudo_label_treatment_nll"


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired FixMatch / `lambda_u = 0` fits."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in 6.1"
            ),
            "variant": (
                "paired fit against an otherwise identical lambda_u = 0 "
                "ablation, same seeds and same batches"
            ),
            "split": (
                "1024 train rows with 64 observed treatments, 2048 held-out "
                "rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio on the EMA parameters, FixMatch over "
                "the lambda_u = 0 ablation; paper mask rate (eq. 6) and "
                "impurity (eq. 5), measured on the trained network, as "
                "guardrails"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "ratio < 1.0 in mean on both the EMA and the trained parameters "
                "(see 6.2); terminal mask rate above 0.2; impurity of retained "
                "labels < 0.15; held-out outcome NLL within 1.05x of the "
                "ablation"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"fixmatch card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            # "< 1.0" in the card against "<= 1.0" here, in both ratios: one
            # point of a continuous statistic, and a strict relation added to
            # the reporting vocabulary for one card would be the convenience
            # quadrant of `DESIGN.md` §11.2 with nothing riding on it.
            MetricResult.upper_bound(
                "ema_treatment_NLL_ratio", column(rows, "ema_ratio"), 1.0
            ),
            MetricResult.upper_bound(
                "trained_treatment_NLL_ratio", column(rows, "trained_ratio"), 1.0
            ),
            MetricResult.lower_bound(
                "terminal_mask_rate", column(rows, "mask_rate"), 0.2
            ),
            MetricResult.upper_bound(
                "retained_label_impurity", column(rows, "impurity"), 0.15
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio", column(rows, "outcome_ratio"), 1.05
            ),
            MetricResult.information(
                "fixmatch_ema_treatment_NLL",
                column(rows, "fixmatch_ema_nll"),
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
        ),
        interpretation=(
            "This is the predeclared project-local FixMatch mechanism target: "
            "can a missing *treatment* label be recovered by confidence-gated "
            "weak/strong pseudo-labelling, composed with the reviewed causal "
            "stack. It is not a reproduction of Sohn et al., whose inputs, "
            "labels, architecture and metric all differ and whose estimand is "
            "an image class."
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
    scheduled_recipe = fixmatch(schema)
    torch.manual_seed(base + 6)
    ablated_recipe = _zero_weight(fixmatch(schema))
    for name, value in scheduled_recipe.system.state_dict().items():
        if not torch.equal(value, ablated_recipe.system.state_dict()[name]):
            raise RuntimeError(f"fixmatch paired initial state differs at {name!r}")

    scheduled_run = compile(scheduled_recipe)
    ablated_run = compile(ablated_recipe)
    # Identical seeds: both stages are index 0 of a one-stage program, so the
    # sampler stream, the view keys and the parameter initialisation all match
    # and `lambda_u` is the only difference between the arms.
    scheduled = run_stage(scheduled_run, "joint_fit", data, seed=base + 10_000)
    ablated = run_stage(ablated_run, "joint_fit", data, seed=base + 10_000)

    fixmatch_metrics = _evaluate(scheduled_run, scheduled, test)
    ablation_metrics = _evaluate(ablated_run, ablated, test)
    for name, value in ablation_metrics.items():
        if name.endswith("nll") and value <= 0.0:
            raise RuntimeError(
                f"the ablation produced a non-positive {name}, so the paired "
                "ratio the card declares is undefined"
            )
    return {
        "ema_ratio": (
            fixmatch_metrics["ema_treatment_nll"]
            / ablation_metrics["ema_treatment_nll"]
        ),
        "trained_ratio": (
            fixmatch_metrics["trained_treatment_nll"]
            / ablation_metrics["trained_treatment_nll"]
        ),
        "outcome_ratio": (
            fixmatch_metrics["ema_outcome_nll"] / ablation_metrics["ema_outcome_nll"]
        ),
        "mask_rate": fixmatch_metrics["mask_rate"],
        "impurity": fixmatch_metrics["impurity"],
        "fixmatch_ema_nll": fixmatch_metrics["ema_treatment_nll"],
        "ablation_ema_nll": ablation_metrics["ema_treatment_nll"],
        "frequency_nll": fixmatch_metrics["frequency_nll"],
    }


def _zero_weight(recipe: Recipe) -> Recipe:
    """The `lambda_u = 0` ablation: the same fit without eq. (4).

    The term stays in the program at weight zero rather than being removed, so
    the two arms plan the same forward passes and the difference is the
    scheduled gradient rather than the graph.
    """
    stage = recipe.program[0]
    gated = stage.objectives[2].objective
    if not isinstance(gated, PseudoLabelTreatmentNLL):
        raise RuntimeError("fixmatch's third term is no longer the pseudo-label loss")
    ablated = replace(stage.objectives[2], weight=Constant(0.0))
    objectives = (*stage.objectives[:2], ablated, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _evaluate(
    run: CompiledRun, result: StageResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out NLLs off the EMA; the paper's gate statistics off the trained net."""
    population = result.population
    if population is None:
        raise RuntimeError(
            "the fitting stage reported no training population; the recipe "
            "declares a sampler, so one is expected"
        )
    if result.teacher is None:
        raise RuntimeError("fixmatch declares an evaluation EMA and reported none")
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    threshold = _threshold(run)
    with torch.no_grad():
        values = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        if not isinstance(propensity, CategoricalTreatment) or not isinstance(
            outcome, GaussianOutcome
        ):
            raise TypeError("fixmatch benchmark expected its reviewed P5 heads")
        ema_treatment_nll = float(F.nll_loss(propensity.log_probs, scaled.t))
        ema_outcome_nll = float(-outcome.log_prob(scaled.y, scaled.t).mean())

        trained = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        trained_propensity = trained[Port.T_GIVEN_X]
        if not isinstance(trained_propensity, CategoricalTreatment):
            raise TypeError("fixmatch benchmark expected a categorical propensity")
        trained_treatment_nll = float(
            F.nll_loss(trained_propensity.log_probs, scaled.t)
        )

        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(observed, minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        frequency_nll = float(F.nll_loss(baseline, scaled.t))

        # Eq. (6)'s mask rate and eq. (5)'s impurity, over *every* training row
        # — the term's population is `all`, because footnote 2 puts the
        # labelled rows into U as well. Measuring over the missing rows alone
        # would report a different statistic under the same name. Impurity
        # needs the true `t`, which is why it is measured here rather than by
        # the objective.
        train_values = run.graph.evaluate(
            population.rows, schema=schema, only=run.graph.names
        )
        train_propensity = train_values[Port.T_GIVEN_X]
        if not isinstance(train_propensity, CategoricalTreatment):
            raise TypeError("fixmatch benchmark expected a categorical propensity")
        confidence, labels = train_propensity.probs.max(dim=-1)
        retained = confidence >= threshold
        mask_rate = float(retained.float().mean())
        impurity = (
            float((labels[retained] != population.rows.t[retained]).float().mean())
            if int(retained.sum())
            else 0.0
        )
    return {
        "ema_treatment_nll": ema_treatment_nll,
        "trained_treatment_nll": trained_treatment_nll,
        "ema_outcome_nll": ema_outcome_nll,
        "frequency_nll": frequency_nll,
        "mask_rate": mask_rate,
        "impurity": impurity,
    }


def _threshold(run: CompiledRun) -> float:
    """`tau`, read from the compiled recipe rather than restated here.

    A benchmark carrying its own copy of a card-governed number is a second
    place for it to drift from the recipe, which is the whole failure the card
    cross-check exists to stop.
    """
    for weighted in run.recipe.program[0].objectives:
        if weighted.name == _PSEUDO_LABEL:
            gated = weighted.objective
            if not isinstance(gated, PseudoLabelTreatmentNLL):
                break
            return float(gated.threshold)
    raise RuntimeError("fixmatch's gated pseudo-label term is no longer in the program")
