"""Cycle-dual's out-of-fold posterior-imputation benchmark."""

from __future__ import annotations

import itertools
from dataclasses import replace
from pathlib import Path

import torch

from xty2.core import (
    CompiledRun,
    CompileError,
    GaussianOutcome,
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
)
from xty2.evaluation.causal import (
    absolute_ate_error,
    average_treatment_effect,
    candidate_treatment_means,
    treatment_contrast,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.recipes import cycle_dual
from xty2.training import run_program

_TRAIN_ROWS = 2_048
_VALIDATION_ROWS = 1_024
_TEST_ROWS = 2_048
_OUTCOME_STEPS = 1_000


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten deterministic posterior/outcome programs."""
    del cache_root
    spec.bind(
        {
            "dataset": "fixed project-local posterior-imputation DGP",
            "variant": "binary treatment; 70% treatment MCAR; five folds",
            "split": (
                "independent 2048 train / 1024 validation / 2048 test per replicate"
            ),
            "metric": "absolute_ATE_error",
            "published": "n/a",
            "tolerance": "0.35 from analytic ATE 1.5",
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"cycle_dual card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "absolute_ATE_error",
                column(rows, "ate_error"),
                0.35,
                unit="outcome units",
            ),
            MetricResult.lower_bound(
                "hidden_treatment_accuracy",
                column(rows, "hidden_accuracy"),
                0.75,
            ),
            MetricResult.lower_bound(
                "out_of_fold_and_used_y",
                column(rows, "provenance"),
                1.0,
            ),
            MetricResult.lower_bound(
                "observed_treatments_preserved",
                column(rows, "observed_preserved"),
                1.0,
            ),
            MetricResult.lower_bound(
                "source_batch_unchanged",
                column(rows, "source_unchanged"),
                1.0,
            ),
            MetricResult.lower_bound(
                "unsafe_recipe_rejected",
                column(rows, "unsafe_rejected"),
                1.0,
            ),
            MetricResult.information(
                "validation_absolute_ATE_error",
                column(rows, "validation_ate_error"),
                unit="outcome units",
            ),
        ),
        interpretation=(
            "This matches the card's project-local staged posterior-imputation "
            "target and its executable leakage/provenance guardrails. It is not "
            "a reproduction of CycleGAN or DualGAN."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = 110_000 + 100 * index
    train = _population(
        _TRAIN_ROWS,
        seed=base,
        observed_probability=0.30,
        row_offset=0,
    )
    validation = _population(
        _VALIDATION_ROWS,
        seed=base + 1,
        observed_probability=1.0,
        row_offset=10_000,
    )
    test = _population(
        _TEST_ROWS,
        seed=base + 2,
        observed_probability=1.0,
        row_offset=20_000,
    )
    x_mean = train.x.mean(dim=0)
    x_scale = train.x.std(dim=0, unbiased=False)
    outcome_mean = train.y.mean()
    outcome_scale = train.y.std(unbiased=False)
    if not bool((x_scale > 0).all()) or float(outcome_scale) <= 0.0:
        raise RuntimeError("cycle_dual training standardisation is degenerate")
    train = train.replace(
        x=(train.x - x_mean) / x_scale,
        y=(train.y - outcome_mean) / outcome_scale,
    )
    validation = validation.replace(
        x=(validation.x - x_mean) / x_scale,
        y=(validation.y - outcome_mean) / outcome_scale,
    )
    test = test.replace(
        x=(test.x - x_mean) / x_scale,
        y=(test.y - outcome_mean) / outcome_scale,
    )
    before = train.clone()
    torch.manual_seed(base + 10)
    recipe = cycle_dual(continuous_schema(4))
    unsafe_rejected = _unsafe_rejected(recipe)
    run = compile(recipe)
    result = run_program(
        run,
        {
            "posterior_labels": (train,),
            "outcome_fit": itertools.repeat(train, _OUTCOME_STEPS),
        },
        seed=base + 10,
    )
    labels = result.stage("posterior_labels").pseudo_labels
    if labels is None:
        raise RuntimeError("cycle_dual emitted no posterior labels")
    joined = labels.apply_to(train, treatment_cardinality=2)
    hidden_accuracy = float(
        (joined.t[train.t_missing] == train.t[train.t_missing]).float().mean()
    )
    observed_preserved = torch.equal(
        joined.t[train.t_observed], train.t[train.t_observed]
    )
    provenance = labels.prediction_mode == "out_of_fold" and labels.used_y is True
    estimate = _ate(run, test, outcome_scale=float(outcome_scale))
    validation_estimate = _ate(
        run,
        validation,
        outcome_scale=float(outcome_scale),
    )
    return {
        "ate_error": absolute_ate_error(estimate, 1.5),
        "validation_ate_error": absolute_ate_error(validation_estimate, 1.5),
        "hidden_accuracy": hidden_accuracy,
        "provenance": bool_float(provenance),
        "observed_preserved": bool_float(observed_preserved),
        "source_unchanged": bool_float(train.equal_to(before)),
        "unsafe_rejected": bool_float(unsafe_rejected),
    }


def _population(
    rows: int,
    *,
    seed: int,
    observed_probability: float,
    row_offset: int,
) -> XTYBatch:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 4, generator=generator)
    treatment_uniform = torch.rand(rows, generator=generator)
    outcome_noise = torch.randn(rows, generator=generator)
    missingness_uniform = torch.rand(rows, generator=generator)
    propensity = torch.sigmoid(0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.25 * x[:, 2])
    treatment = (treatment_uniform < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.25 * x[:, 1] + 0.25 * (x[:, 2].square() - 1.0)
    effect = 1.5 + 0.5 * torch.tanh(x[:, 0])
    outcome = baseline + treatment * effect + 0.5 * outcome_noise
    row_id = torch.arange(row_offset, row_offset + rows)
    return XTYBatch(
        x=x,
        t=treatment,
        y=outcome,
        t_observed=missingness_uniform < observed_probability,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=row_id,
        fold_id=row_id % 5,
    )


def _ate(run: CompiledRun, batch: XTYBatch, *, outcome_scale: float) -> float:
    with torch.no_grad():
        distribution = run.state("outcome_fit", batch).default[Port.Y_GIVEN_XT]
        if not isinstance(distribution, GaussianOutcome):
            raise TypeError("cycle_dual benchmark expected Gaussian outcome")
        means = candidate_treatment_means(
            distribution,
            batch_size=batch.batch_size,
            num_treatments=2,
            device=batch.t.device,
        )
        effect = treatment_contrast(means) * outcome_scale
        return float(average_treatment_effect(effect))


def _unsafe_rejected(recipe: Recipe) -> bool:
    first, second = recipe.program
    unsafe = replace(
        recipe,
        program=Program((replace(first, executor="gradient"), second)),
    )
    try:
        compile(unsafe)
    except CompileError:
        return True
    return False


__all__ = ["run"]
