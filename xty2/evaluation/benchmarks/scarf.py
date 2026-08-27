"""SCARF's paired pretrained/unpretrained benchmark from card section 6.

The comparison is the whole design: the same `joint_fit` stage, the same seeds,
the same batch stream and the same initial parameters, run once from the
SCARF-pretrained encoder and once from the recipe's initialisation. Nothing
here compares against Bahri et al., and the module says so in its
interpretation rather than leaving a reader to infer it.

Three things this module deliberately does **not** do, because the recipe now
declares them: mask treatments to the 40-label budget, standardise the outcome,
and choose the batch size. All three are `DataSpec` and `UniformSampler`
declarations, applied by the loader and printed in the plan. A benchmark that
did any of them here would be applying a policy twice and reporting a protocol
the plan does not describe.
"""

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
    XTYBatch,
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
from xty2.recipes import scarf
from xty2.recipes.scarf import PRETRAIN_STEPS
from xty2.training import STREAM_STRIDE, ProgramResult, run_program

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 90_000
_CONTRASTIVE = "info_nce_contrastive"


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired pretrained/unpretrained fits."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in 6.1"
            ),
            "variant": (
                "paired fit against the identical joint_fit stage with no "
                "pretraining, same seeds and same batches"
            ),
            "split": (
                "1024 train rows with 40 observed treatments, 2048 held-out "
                "rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio, pretrained over unpretrained; "
                "positive-pair alignment of the pretrained encoder as a "
                "mechanism guardrail"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "NLL ratio < 1.0 in mean; held-out outcome NLL within 1.05x of "
                "the unpretrained arm; mean cosine similarity of a row to its "
                "own corrupted view at least 0.2 above its mean similarity to "
                "the other rows of the batch"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(f"scarf card reviewed ten replicates, got {spec.seed_count}")
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            # The card writes "< 1.0" and this is "<= 1.0". The difference is a
            # single point of a continuous statistic, and the alternative — a
            # strict relation added to the reporting vocabulary for one card —
            # is the convenience quadrant of `DESIGN.md` §11.2 with nothing
            # riding on it. Recorded here rather than left for a reader to
            # notice.
            MetricResult.upper_bound(
                "held_out_treatment_NLL_ratio",
                column(rows, "treatment_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio",
                column(rows, "outcome_ratio"),
                1.05,
            ),
            MetricResult.lower_bound(
                "terminal_alignment_minus_uniformity",
                column(rows, "alignment_gap"),
                0.2,
            ),
            MetricResult.information(
                "pretrained_treatment_NLL",
                column(rows, "pretrained_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "unpretrained_treatment_NLL",
                column(rows, "unpretrained_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "marginal_frequency_NLL",
                column(rows, "frequency_nll"),
                unit="nat/row",
            ),
            MetricResult.information("terminal_alignment", column(rows, "alignment")),
            MetricResult.information("terminal_uniformity", column(rows, "uniformity")),
        ),
        interpretation=(
            "This is the predeclared project-local SCARF mechanism target: does "
            "an encoder trained only on the covariance of x help a scarce-label "
            "treatment fit. It is not a reproduction of Bahri et al., whose "
            "evidence is 69 OpenML-CC18 datasets under three label regimes and "
            "whose downstream task carries no treatment."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = two_cluster_population(_TRAIN_ROWS, seed=base + 1, row_offset=0)
    test = two_cluster_population(_TEST_ROWS, seed=base + 2, row_offset=10_000)
    data = training_dataset(schema, train.batch)

    # Both arms start from bit-identical parameters: the pairing is the whole
    # measurement, so a difference in initialisation would be a second variable.
    torch.manual_seed(base + 6)
    pretrained_recipe = scarf(schema)
    torch.manual_seed(base + 6)
    ablated_recipe = _unpretrained(scarf(schema))
    for name, value in pretrained_recipe.system.state_dict().items():
        if not torch.equal(value, ablated_recipe.system.state_dict()[name]):
            raise RuntimeError(f"scarf paired initial state differs at {name!r}")

    pretrained_run = compile(pretrained_recipe)
    ablated_run = compile(ablated_recipe)
    full = run_program(
        pretrained_run,
        {"pretrain": data, "joint_fit": data},
        seed=base + 10_000,
    )
    # The ablation's single stage is index 0, so its seed is offset by one
    # stride to give its fit the same stochastic stream as the paired arm's.
    bare = run_program(
        ablated_run, {"joint_fit": data}, seed=base + 10_000 + STREAM_STRIDE
    )

    pretrained = _evaluate(pretrained_run, full, test, train.batch)
    unpretrained = _evaluate(ablated_run, bare, test, train.batch)
    if unpretrained["treatment_nll"] <= 0.0 or unpretrained["outcome_nll"] <= 0.0:
        raise RuntimeError(
            "the unpretrained arm produced a non-positive NLL, so the paired "
            "ratio the card declares is undefined"
        )
    alignment, uniformity = _terminal_alignment(full)
    return {
        "treatment_ratio": (
            pretrained["treatment_nll"] / unpretrained["treatment_nll"]
        ),
        "outcome_ratio": pretrained["outcome_nll"] / unpretrained["outcome_nll"],
        "alignment_gap": alignment - uniformity,
        "alignment": alignment,
        "uniformity": uniformity,
        "pretrained_nll": pretrained["treatment_nll"],
        "unpretrained_nll": unpretrained["treatment_nll"],
        "frequency_nll": pretrained["frequency_nll"],
    }


def _unpretrained(recipe: Recipe) -> Recipe:
    """The ablation: the same fitting stage, from the recipe's initialisation.

    `initialise_from` is what the pretraining delivers, so dropping the stage
    without dropping the edge would leave a program that cannot run. Both go
    and nothing else is touched — same graph, same objectives, same optimiser,
    same sampler, same budget.
    """
    fit = recipe.program[1]
    return replace(recipe, program=Program((replace(fit, initialise_from=None),)))


def _evaluate(
    run: CompiledRun,
    result: ProgramResult,
    test: ClusterPopulation,
    train: XTYBatch,
) -> dict[str, float]:
    """Held-out NLLs, on the outcome scale the run itself fitted."""
    population = result.stage("joint_fit").population
    if population is None:
        raise RuntimeError(
            "the fitting stage reported no training population; the recipe "
            "declares a sampler, so one is expected"
        )
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        values = run.graph.evaluate(
            scaled,
            schema=schema,
            only=("mlp_encoder", "tarnet_head", "categorical_propensity"),
        )
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        if not isinstance(propensity, CategoricalTreatment) or not isinstance(
            outcome, GaussianOutcome
        ):
            raise TypeError("scarf benchmark expected its reviewed P5 heads")
        treatment_nll = float(F.nll_loss(propensity.log_probs, scaled.t))
        outcome_nll = float(-outcome.log_prob(scaled.y, scaled.t).mean())
        # The baseline is the *labelled* training rows' marginal, which is what
        # a model with no covariate information would predict. It reads the
        # policy's mask rather than the source data's, because that is the
        # supervision the fit actually had.
        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(observed, minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        frequency_nll = float(F.nll_loss(baseline, scaled.t))
    del train
    return {
        "treatment_nll": treatment_nll,
        "outcome_nll": outcome_nll,
        "frequency_nll": frequency_nll,
    }


def _terminal_alignment(result: ProgramResult) -> tuple[float, float]:
    """`L_cont`'s two diagnostics at the last pretraining step.

    The guardrail distinguishes learning from collapse, which no single loss
    number does: a collapsed encoder drives `L_cont` to exactly 0 — *higher*
    than an untrained network's — with alignment and uniformity equal.
    """
    record = result.stage("pretrain").records[PRETRAIN_STEPS - 1]
    term = next(entry for entry in record.terms if entry.name == _CONTRASTIVE)
    return (
        float(term.diagnostics["alignment"]),
        float(term.diagnostics["uniformity"]),
    )
