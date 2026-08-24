"""Mean Teacher's paired clustered-treatment benchmark from card section 6."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Constant,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    BatchStream,
    batch_indices,
    column,
    configure_worker,
    continuous_schema,
    parallel_replicates,
    standardise_outcome,
)
from xty2.evaluation.causal import (
    candidate_treatment_means,
    sqrt_pehe,
    treatment_contrast,
)
from xty2.evaluation.predictive import treatment_nll
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.recipes import mean_teacher
from xty2.training import StageResult, run_stage

_TRAIN_ROWS = 4_096
_VALIDATION_ROWS = 2_048
_TEST_ROWS = 4_096
_OBSERVED_TREATMENTS = 205
_STEPS = 3_000
_BATCH_SIZE = 256


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired scheduled/zero-consistency fits."""
    del cache_root
    spec.require("dataset", "xty2 analytic redundant-cluster treatment DGP")
    spec.require(
        "metric",
        "held-out masked-view student/teacher probability-MSE ratio; "
        "treatment NLL and sqrt_PEHE guardrails",
    )
    spec.require(
        "tolerance",
        "mean ratio <= 0.90; mean d_NLL <= 0.02 nat/row; mean d_sqrt_PEHE <= 0.10",
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"Mean Teacher card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "consistency_MSE_ratio",
                column(rows, "ratio"),
                0.90,
            ),
            MetricResult.upper_bound(
                "paired_d_treatment_NLL",
                column(rows, "d_nll"),
                0.02,
                unit="nat/row",
            ),
            MetricResult.upper_bound(
                "paired_d_sqrt_PEHE",
                column(rows, "d_pehe"),
                0.10,
                unit="outcome units",
            ),
            MetricResult.information(
                "mean_teacher_treatment_NLL",
                column(rows, "scheduled_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "zero_consistency_treatment_NLL",
                column(rows, "zero_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "mean_teacher_sqrt_PEHE",
                column(rows, "scheduled_pehe"),
                unit="outcome units",
            ),
            MetricResult.information(
                "zero_consistency_sqrt_PEHE",
                column(rows, "zero_pehe"),
                unit="outcome units",
            ),
        ),
        interpretation=(
            "This matches the predeclared project-local Mean Teacher mechanism "
            "target. It does not reproduce the source paper's image benchmarks."
        ),
    )


@dataclass(frozen=True)
class _Population:
    batch: XTYBatch
    true_effect: Tensor


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = 90_000 + 100 * index
    train = _population(_TRAIN_ROWS, seed=base + 1, row_offset=0)
    validation = _population(_VALIDATION_ROWS, seed=base + 2, row_offset=10_000)
    test = _population(_TEST_ROWS, seed=base + 3, row_offset=20_000)
    observed = torch.zeros(_TRAIN_ROWS, dtype=torch.bool)
    missingness = torch.Generator().manual_seed(base + 4)
    selected = torch.randperm(_TRAIN_ROWS, generator=missingness)[:_OBSERVED_TREATMENTS]
    observed[selected] = True
    train = replace(train, batch=train.batch.replace(t_observed=observed))
    _, outcome_scale, transformed = standardise_outcome(
        train.batch, validation.batch, test.batch
    )
    train = replace(train, batch=transformed[0])
    validation = replace(validation, batch=transformed[1])
    test = replace(test, batch=transformed[2])
    indices = batch_indices(
        _TRAIN_ROWS,
        steps=_STEPS,
        batch_size=_BATCH_SIZE,
        seed=base + 5,
    )
    schema = continuous_schema(6)
    torch.manual_seed(base + 6)
    scheduled_recipe = mean_teacher(schema)
    torch.manual_seed(base + 6)
    zero_recipe = _zero_consistency(mean_teacher(schema))
    for name, value in scheduled_recipe.system.state_dict().items():
        if not torch.equal(value, zero_recipe.system.state_dict()[name]):
            raise RuntimeError(f"Mean Teacher paired initial state differs at {name!r}")
    scheduled_run = compile(scheduled_recipe)
    zero_run = compile(zero_recipe)
    scheduled_result = run_stage(
        scheduled_run,
        "joint_fit",
        BatchStream(train.batch, indices),
        seed=base + 10_000,
    )
    zero_result = run_stage(
        zero_run,
        "joint_fit",
        BatchStream(train.batch, indices),
        seed=base + 10_000,
    )
    if scheduled_result.steps != _STEPS or zero_result.steps != _STEPS:
        raise RuntimeError("Mean Teacher paired fit did not run exactly 3,000 steps")
    # Validation remains a finite diagnostic and never selects a checkpoint.
    _ = _teacher_metrics(
        scheduled_run,
        scheduled_result,
        validation,
        outcome_scale=outcome_scale,
    )
    scheduled = _teacher_metrics(
        scheduled_run,
        scheduled_result,
        test,
        outcome_scale=outcome_scale,
    )
    zero = _teacher_metrics(
        zero_run,
        zero_result,
        test,
        outcome_scale=outcome_scale,
    )
    scheduled_disagreement = _disagreement(
        scheduled_run, scheduled_result, test.batch, base=base
    )
    zero_disagreement = _disagreement(zero_run, zero_result, test.batch, base=base)
    if zero_disagreement == 0.0:
        raise RuntimeError(
            "zero-consistency disagreement was exactly zero; the card declares "
            "the ratio invalid in this case"
        )
    return {
        "ratio": scheduled_disagreement / zero_disagreement,
        "d_nll": scheduled["nll"] - zero["nll"],
        "d_pehe": scheduled["pehe"] - zero["pehe"],
        "scheduled_nll": scheduled["nll"],
        "zero_nll": zero["nll"],
        "scheduled_pehe": scheduled["pehe"],
        "zero_pehe": zero["pehe"],
        "scheduled_disagreement": scheduled_disagreement,
        "zero_disagreement": zero_disagreement,
    }


def _population(rows: int, *, seed: int, row_offset: int) -> _Population:
    generator = torch.Generator().manual_seed(seed)
    cluster_uniform = torch.rand(rows, generator=generator)
    feature_noise = torch.randn(rows, 6, generator=generator)
    treatment_uniform = torch.rand(rows, generator=generator)
    outcome_noise = torch.randn(rows, generator=generator)
    cluster = (cluster_uniform < 0.5).float()
    sign = 2.0 * cluster - 1.0
    x = feature_noise.clone()
    x[:, :4] = 0.8 * sign[:, None] + 0.6 * feature_noise[:, :4]
    propensity = 0.15 + 0.70 * cluster
    treatment = (treatment_uniform < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    true_effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    outcome = baseline + treatment * true_effect + 0.5 * outcome_noise
    return _Population(
        batch=XTYBatch(
            x=x,
            t=treatment,
            y=outcome,
            t_observed=torch.ones(rows, dtype=torch.bool),
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        true_effect=true_effect,
    )


def _zero_consistency(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    consistency = replace(stage.objectives[-1], weight=Constant(0.0))
    return replace(
        recipe,
        program=Program(
            (replace(stage, objectives=(*stage.objectives[:-1], consistency)),)
        ),
    )


def _teacher_metrics(
    run: CompiledRun,
    result: StageResult,
    population: _Population,
    *,
    outcome_scale: float,
) -> dict[str, float]:
    if result.teacher is None:
        raise RuntimeError("Mean Teacher benchmark has no final teacher")
    with torch.no_grad():
        state = result.teacher.graph.evaluate(
            population.batch,
            schema=run.recipe.schema,
            only=run.graph.names,
        )
        propensity = state[Port.T_GIVEN_X]
        outcome = state[Port.Y_GIVEN_XT]
        if not isinstance(propensity, CategoricalTreatment):
            raise TypeError("Mean Teacher benchmark expected categorical propensity")
        if not isinstance(outcome, GaussianOutcome):
            raise TypeError("Mean Teacher benchmark expected Gaussian outcome")
        nll = treatment_nll(propensity, population.batch.t)
        means = candidate_treatment_means(
            outcome,
            batch_size=population.batch.batch_size,
            num_treatments=2,
        )
        effect = treatment_contrast(means) * outcome_scale
        pehe = sqrt_pehe(effect, population.true_effect)
    return {"nll": float(nll), "pehe": float(pehe)}


def _disagreement(
    run: CompiledRun,
    result: StageResult,
    batch: XTYBatch,
    *,
    base: int,
) -> float:
    if result.teacher is None:
        raise RuntimeError("Mean Teacher benchmark has no final teacher")
    values: list[Tensor] = []
    only = ("mlp_encoder", "categorical_propensity")
    with torch.no_grad():
        for draw in range(16):
            key = base + 20_000 + draw
            student_batch = run.recipe.view("student_x").apply(
                batch, run.recipe.schema, rng_key=key
            )
            teacher_batch = run.recipe.view("teacher_x").apply(
                batch, run.recipe.schema, rng_key=key
            )
            student = run.graph.evaluate(
                student_batch, schema=run.recipe.schema, only=only
            )[Port.T_GIVEN_X]
            teacher = result.teacher.graph.evaluate(
                teacher_batch, schema=run.recipe.schema, only=only
            )[Port.T_GIVEN_X]
            if not isinstance(student, CategoricalTreatment) or not isinstance(
                teacher, CategoricalTreatment
            ):
                raise TypeError("Mean Teacher disagreement expected probabilities")
            values.append((student.probs - teacher.probs).square().mean())
    return float(torch.stack(values).mean())


__all__ = ["run"]
