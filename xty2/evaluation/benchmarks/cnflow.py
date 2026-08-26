"""CNFlow's paired non-Gaussian outcome benchmark from card section 6."""

from __future__ import annotations

import copy
import math
from functools import partial
from pathlib import Path

import torch
from torch import Tensor

from xty2.components import (
    CategoricalPropensity,
    ConditionalFlow,
    ConditionalFlowOutcome,
    MLPEncoder,
    TARNetHead,
)
from xty2.components._nn import (
    CFRNET_INITIALISATION,
    TORCH_LINEAR_INITIALISATION,
)
from xty2.components.density import (
    NFLOWS_INITIALISATION,
    RANDOM_PERMUTATION,
    STANDARD_NORMAL,
)
from xty2.core import (
    CompiledRun,
    Component,
    ComponentGraph,
    Constant,
    GaussianOutcome,
    GradientClipping,
    OptimiserSpec,
    Port,
    Recipe,
    Schema,
    Stage,
    WeightDecay,
    Weighted,
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
    take,
)
from xty2.evaluation.causal import (
    candidate_treatment_means,
    treatment_contrast,
)
from xty2.evaluation.predictive import conditional_outcome_nll
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.objectives import (
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)
from xty2.training import run_stage

_TRAIN_ROWS = 4_096
_VALIDATION_ROWS = 2_048
_TEST_ROWS = 4_096
_STEPS = 3_000
_BATCH_SIZE = 256
_EVALUATION_BATCH_SIZE = 256


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the card's ten paired flow/Gaussian fits."""
    del cache_root
    spec.bind(
        {
            "dataset": "xty2 analytic non-Gaussian outcome DGP",
            "variant": (
                "section 6.1 scalar-Y equations; six Gaussian X; binary "
                "confounded T; centred heteroskedastic two-component outcome "
                "mixture"
            ),
            "samples": "{train: 4096, validation: 2048, test: 4096}",
            "missingness": (
                "exactly 2048 training treatments MCAR; all outcomes and all "
                "validation/test treatments observed"
            ),
            "preprocessing": (
                "raw X; population-standardise Y from all training outcomes; "
                "evaluate in original Y units"
            ),
            "pairing": (
                "identical populations, missingness mask, ordered batches and "
                "bit-identical initial shared parameters"
            ),
            "training": (
                "batch_size=256; 3000 final-checkpoint Adam steps; validation "
                "is diagnostic only"
            ),
            "primary_metric": (
                "test conditional outcome NLL p(Y|X,T), explicitly not joint "
                "or missing-treatment marginal NLL"
            ),
            "guardrail": "test sqrt_PEHE against analytic tau(X)",
            "published": "n/a",
            "tolerance": (
                "mean paired d_NLL <= -0.10 nat/row; mean paired d_PEHE <= 0.10"
            ),
            "seeds": (
                "r=0..9 with base 70000+100*r and fixed stream offsets from "
                "sections 6.1-6.2"
            ),
            "report": (
                "per-model means plus paired-difference means and sample "
                "stderrs over 10 replicates"
            ),
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(f"CNFlow card reviewed ten replicates, got {spec.seed_count}")
    rows = parallel_replicates(partial(_replicate), spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "paired_d_conditional_NLL",
                column(rows, "d_nll"),
                -0.10,
                unit="nat/row",
            ),
            MetricResult.upper_bound(
                "paired_d_sqrt_PEHE",
                column(rows, "d_pehe"),
                0.10,
                unit="outcome units",
            ),
            MetricResult.information(
                "flow_conditional_NLL",
                column(rows, "flow_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "gaussian_conditional_NLL",
                column(rows, "gaussian_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "flow_sqrt_PEHE",
                column(rows, "flow_pehe"),
                unit="outcome units",
            ),
            MetricResult.information(
                "gaussian_sqrt_PEHE",
                column(rows, "gaussian_pehe"),
                unit="outcome units",
            ),
        ),
        interpretation=(
            "This is the predeclared project-local conditional-density "
            "validation target. Matching it validates the CNFlow recipe's "
            "limited claim and is not a reproduction of Durkan et al."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = 70_000 + 100 * index
    train, _ = _population(
        _TRAIN_ROWS,
        seed=base + 1,
        row_offset=0,
        treatment_observed=True,
    )
    validation, _ = _population(
        _VALIDATION_ROWS,
        seed=base + 2,
        row_offset=10_000,
        treatment_observed=True,
    )
    test, true_effect = _population(
        _TEST_ROWS,
        seed=base + 3,
        row_offset=20_000,
        treatment_observed=True,
    )
    missing_generator = torch.Generator().manual_seed(base + 4)
    missing = torch.randperm(_TRAIN_ROWS, generator=missing_generator)[:2_048]
    observed = torch.ones(_TRAIN_ROWS, dtype=torch.bool)
    observed[missing] = False
    train = train.replace(t_observed=observed)
    _, outcome_scale, transformed = standardise_outcome(train, validation, test)
    train, validation, test = transformed
    indices = batch_indices(
        _TRAIN_ROWS,
        steps=_STEPS,
        batch_size=_BATCH_SIZE,
        seed=base + 5,
    )
    schema = continuous_schema(6)
    flow_recipe, gaussian_recipe = _paired_recipes(schema, base=base)
    flow_run = compile(flow_recipe)
    gaussian_run = compile(gaussian_recipe)
    flow_result = run_stage(
        flow_run,
        "joint_fit",
        BatchStream(train, indices),
        seed=base + 9,
    )
    gaussian_result = run_stage(
        gaussian_run,
        "joint_fit",
        BatchStream(train, indices),
        seed=base + 9,
    )
    # Validation is deliberately diagnostic-only. Evaluating it here makes a
    # broken transform visible in the result trace without selecting a model.
    flow_validation_nll = _model_nll(
        flow_run,
        validation,
        outcome_scale=outcome_scale,
    )
    gaussian_validation_nll = _model_nll(
        gaussian_run,
        validation,
        outcome_scale=outcome_scale,
    )
    if flow_result.steps != _STEPS or gaussian_result.steps != _STEPS:
        raise RuntimeError("CNFlow paired fit did not execute 3,000 final steps")
    flow = _model_metrics(
        flow_run,
        test,
        true_effect,
        outcome_scale=outcome_scale,
    )
    gaussian = _model_metrics(
        gaussian_run,
        test,
        true_effect,
        outcome_scale=outcome_scale,
    )
    return {
        "flow_nll": flow["nll"],
        "gaussian_nll": gaussian["nll"],
        "d_nll": flow["nll"] - gaussian["nll"],
        "flow_pehe": flow["pehe"],
        "gaussian_pehe": gaussian["pehe"],
        "d_pehe": flow["pehe"] - gaussian["pehe"],
        "flow_validation_nll": flow_validation_nll,
        "gaussian_validation_nll": gaussian_validation_nll,
    }


def _population(
    rows: int,
    *,
    seed: int,
    row_offset: int,
    treatment_observed: bool,
) -> tuple[XTYBatch, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, 6, generator=generator)
    treatment_uniform = torch.rand(rows, generator=generator)
    mixture_uniform = torch.rand(rows, generator=generator)
    noise = torch.randn(rows, generator=generator)
    propensity = 0.1 + 0.8 * torch.sigmoid(1.25 * x[:, 0] - x[:, 1] + 0.5 * x[:, 2])
    treatment = (treatment_uniform < propensity).long()
    baseline = (
        0.8 * x[:, 0]
        + 0.5 * (x[:, 1].square() - 1.0)
        - 0.6 * x[:, 2] * x[:, 3]
        + 0.3 * torch.sin(2.0 * x[:, 4])
    )
    true_effect = 1.0 + 0.5 * torch.tanh(x[:, 0])
    mixture_probability = 0.2 + 0.6 * torch.sigmoid(
        0.75 * x[:, 4] - 0.5 * x[:, 5] + 0.5 * treatment
    )
    outcome_scale = 0.15 + 0.10 * torch.sigmoid(x[:, 5]) + 0.05 * treatment
    mixture = (mixture_uniform < mixture_probability).float()
    outcome = (
        baseline
        + treatment * true_effect
        + 1.5 * (mixture - mixture_probability)
        + outcome_scale * noise
    )
    return (
        XTYBatch(
            x=x,
            t=treatment,
            y=outcome,
            t_observed=torch.full((rows,), treatment_observed, dtype=torch.bool),
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        true_effect,
    )


def _paired_recipes(schema: Schema, *, base: int) -> tuple[Recipe, Recipe]:
    torch.manual_seed(base + 6)
    shared_encoder = MLPEncoder(
        input_dim=6,
        widths=(128, 128),
        activation="relu",
        normalisation="none",
        dropout=0.0,
        initialisation=TORCH_LINEAR_INITIALISATION,
    )
    shared_propensity = CategoricalPropensity(
        representation_dim=128,
        num_treatments=2,
        activation="linear logits",
        normalisation="none",
        dropout=0.0,
        initialisation=CFRNET_INITIALISATION,
        output_parameterisation="K softmax logits",
    )
    torch.manual_seed(base + 7)
    flow = ConditionalFlow(
        representation_dim=128,
        num_treatments=2,
        outcome=schema.outcome,
        num_transforms=5,
        hidden_features=128,
        num_blocks=2,
        use_residual_blocks=True,
        num_bins=8,
        tails="linear",
        tail_bound=3.0,
        permutation=RANDOM_PERMUTATION,
        activation="relu",
        normalisation="none",
        dropout=0.0,
        initialisation=NFLOWS_INITIALISATION,
        base_distribution=STANDARD_NORMAL,
        mean_samples=100,
    )
    torch.manual_seed(base + 8)
    gaussian = TARNetHead(
        representation_dim=128,
        num_treatments=2,
        outcome=schema.outcome,
        widths=(100, 100, 100),
        activation="elu",
        normalisation="none",
        dropout=0.0,
        initialisation=CFRNET_INITIALISATION,
        output_parameterisation="K means; fixed Gaussian scale=1.0",
    )
    return (
        _recipe(
            schema,
            copy.deepcopy(shared_encoder),
            flow,
            copy.deepcopy(shared_propensity),
            name="cnflow",
        ),
        _recipe(
            schema,
            copy.deepcopy(shared_encoder),
            gaussian,
            copy.deepcopy(shared_propensity),
            name="cnflow_gaussian_comparator",
        ),
    )


def _recipe(
    schema: Schema,
    encoder: MLPEncoder,
    outcome: Component,
    propensity: CategoricalPropensity,
    *,
    name: str,
) -> Recipe:
    return Recipe(
        name=name,
        schema=schema,
        system=ComponentGraph((encoder, outcome, propensity)),
        program=(
            Stage(
                name="joint_fit",
                objectives=(
                    Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="population"),
                    Weighted(
                        ObservedTreatmentNLL(), weight=1.0, reduction="population"
                    ),
                    Weighted(
                        MissingTreatmentMarginalNLL(grad_path="both"),
                        weight=1.0,
                        reduction="population",
                    ),
                ),
                trainable=(encoder.name, outcome.name, propensity.name),
                rows="all",
                optimiser=OptimiserSpec(
                    name="adam",
                    lr=1e-3,
                    weight_decay=WeightDecay.none(),
                    lr_schedule=Constant(1.0),
                    clipping=GradientClipping.none(),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                ),
                steps=_STEPS,
            ),
        ),
        card="docs/recipes/cnflow.md",
        purpose="causal",
    )


def _outcome_distribution(
    run: CompiledRun, batch: XTYBatch
) -> ConditionalFlowOutcome | GaussianOutcome:
    state = run.state("joint_fit", batch)
    distribution = state.default[Port.Y_GIVEN_XT]
    if not isinstance(distribution, ConditionalFlowOutcome | GaussianOutcome):
        raise TypeError("CNFlow benchmark expected a supported outcome distribution")
    return distribution


def _model_nll(
    run: CompiledRun,
    batch: XTYBatch,
    *,
    outcome_scale: float,
) -> float:
    total = 0.0
    with torch.no_grad():
        for start in range(0, batch.batch_size, _EVALUATION_BATCH_SIZE):
            rows = torch.arange(
                start,
                min(start + _EVALUATION_BATCH_SIZE, batch.batch_size),
            )
            chunk = take(batch, rows)
            distribution = _outcome_distribution(run, chunk)
            value = conditional_outcome_nll(
                distribution,
                chunk.y,
                chunk.t,
                log_abs_jacobian=math.log(outcome_scale),
            )
            total += float(value) * chunk.batch_size
    return total / batch.batch_size


def _model_metrics(
    run: CompiledRun,
    batch: XTYBatch,
    true_effect: Tensor,
    *,
    outcome_scale: float,
) -> dict[str, float]:
    # Chunking changes only memory use. NLL is accumulated by row and PEHE by
    # squared-error sum before the one final square root, so this is exactly
    # the full-population reduction declared in the card.
    nll_total = 0.0
    squared_error_total = 0.0
    with torch.no_grad():
        for start in range(0, batch.batch_size, _EVALUATION_BATCH_SIZE):
            stop = min(start + _EVALUATION_BATCH_SIZE, batch.batch_size)
            rows = torch.arange(start, stop)
            chunk = take(batch, rows)
            distribution = _outcome_distribution(run, chunk)
            nll = conditional_outcome_nll(
                distribution,
                chunk.y,
                chunk.t,
                log_abs_jacobian=math.log(outcome_scale),
            )
            means = candidate_treatment_means(
                distribution,
                batch_size=chunk.batch_size,
                num_treatments=2,
                device=chunk.t.device,
            )
            effect = treatment_contrast(means) * outcome_scale
            truth = true_effect.index_select(0, rows)
            nll_total += float(nll) * chunk.batch_size
            squared_error_total += float((effect - truth).square().sum())
    return {
        "nll": nll_total / batch.batch_size,
        "pehe": math.sqrt(squared_error_total / batch.batch_size),
    }


__all__ = ["run"]
