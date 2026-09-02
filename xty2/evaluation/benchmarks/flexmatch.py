"""FlexMatch's paired curriculum-versus-constant benchmark from card section 6."""

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
    Weighted,
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
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec
from xty2.objectives import (
    CurriculumPseudoLabelTreatmentNLL,
    CurriculumStatus,
    PseudoLabelTreatmentNLL,
)
from xty2.recipes import flexmatch
from xty2.recipes.fixmatch import STRONG_X, WEAK_X
from xty2.recipes.flexmatch import TAU
from xty2.training import StageResult, run_stage

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 90_000
_CURRICULUM = "curriculum_pseudo_label_treatment_nll"


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten paired fits predeclared by ``flexmatch.md`` section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "fixmatch.md §6.1's project-local seed-locked two-cluster XTY "
                "DGP (6 features, K=2), unmodified"
            ),
            "variant": (
                "paired fit against a constant-gate arm — this recipe with eq. "
                "(8) replaced by eq. (3) at tau = 0.95, so both arms share "
                "deviation 2's views — same seeds and same batches"
            ),
            "split": (
                "1024 train rows with 64 observed treatments, 2048 held-out "
                "rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio on the EMA parameters, FlexMatch over "
                "the constant-gate arm; the paper's mask rate and impurity, and "
                "the per-class threshold trajectory, as guardrails"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "ratio < 1.0 in mean on both the EMA and the trained parameters, "
                "by at least one standard error; terminal mask rate above 0.2; "
                "impurity of retained labels < 0.15; held-out outcome NLL within "
                "1.05x of the constant-gate arm; max_c T(c) >= 0.9 tau by the end "
                "of the run and min_c T(c) < 0.5 tau at step 0"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"flexmatch card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
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
            MetricResult.lower_bound(
                "terminal_threshold_max", column(rows, "threshold_max"), 0.9 * TAU
            ),
            MetricResult.upper_bound(
                "initial_threshold_min",
                column(rows, "initial_threshold_min"),
                0.5 * TAU,
            ),
            MetricResult.information(
                "terminal_marked_fraction", column(rows, "marked_fraction")
            ),
            MetricResult.information(
                "flexmatch_ema_treatment_NLL",
                column(rows, "flexmatch_ema_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "constant_ema_treatment_NLL",
                column(rows, "constant_ema_nll"),
                unit="nat/row",
            ),
        ),
        interpretation=(
            "This is the predeclared project-local FlexMatch mechanism target: "
            "whether per-class curriculum thresholds improve missing-treatment "
            "classification over an otherwise identical constant gate. It is "
            "not a reproduction of the paper's image benchmarks."
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
    curriculum_recipe = flexmatch(schema)
    torch.manual_seed(base + 6)
    constant_recipe = _constant_gate(flexmatch(schema))
    for name, value in curriculum_recipe.system.state_dict().items():
        if not torch.equal(value, constant_recipe.system.state_dict()[name]):
            raise RuntimeError(f"flexmatch paired initial state differs at {name!r}")

    curriculum_run = compile(curriculum_recipe)
    constant_run = compile(constant_recipe)
    stage_seed = base + 10_000
    curriculum_result = run_stage(curriculum_run, "joint_fit", data, seed=stage_seed)
    constant_result = run_stage(constant_run, "joint_fit", data, seed=stage_seed)
    if not torch.equal(
        curriculum_result.checkpoint.trained_on_row_ids,
        constant_result.checkpoint.trained_on_row_ids,
    ):
        raise RuntimeError("flexmatch paired arms saw different training rows")

    curriculum_metrics = _evaluate(
        curriculum_run, curriculum_result, test, with_curriculum=True
    )
    constant_metrics = _evaluate(
        constant_run, constant_result, test, with_curriculum=False
    )
    for name in ("ema_treatment_nll", "trained_treatment_nll", "ema_outcome_nll"):
        if constant_metrics[name] <= 0.0:
            raise RuntimeError(
                f"the constant arm produced a non-positive {name}, so the "
                "paired ratio is undefined"
            )
    return {
        "ema_ratio": (
            curriculum_metrics["ema_treatment_nll"]
            / constant_metrics["ema_treatment_nll"]
        ),
        "trained_ratio": (
            curriculum_metrics["trained_treatment_nll"]
            / constant_metrics["trained_treatment_nll"]
        ),
        "outcome_ratio": (
            curriculum_metrics["ema_outcome_nll"] / constant_metrics["ema_outcome_nll"]
        ),
        "mask_rate": curriculum_metrics["mask_rate"],
        "impurity": curriculum_metrics["impurity"],
        "threshold_max": curriculum_metrics["threshold_max"],
        "initial_threshold_min": curriculum_metrics["initial_threshold_min"],
        "marked_fraction": curriculum_metrics["marked_fraction"],
        "flexmatch_ema_nll": curriculum_metrics["ema_treatment_nll"],
        "constant_ema_nll": constant_metrics["ema_treatment_nll"],
    }


def _constant_gate(recipe: Recipe) -> Recipe:
    """Replace eq. (8) with eq. (3), preserving every other declaration."""
    stage = recipe.program[0]
    curriculum = stage.objectives[2].objective
    if not isinstance(curriculum, CurriculumPseudoLabelTreatmentNLL):
        raise RuntimeError("flexmatch's third term is no longer the curriculum loss")
    gated = Weighted(
        PseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=WEAK_X,
            prediction=STRONG_X,
            threshold=TAU,
            sharpening="hard",
            stop_grad="target",
            rows="all",
        ),
        weight=stage.objectives[2].weight,
        reduction=stage.objectives[2].reduction,
    )
    objectives = (*stage.objectives[:2], gated, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _evaluate(
    run: CompiledRun,
    result: StageResult,
    test: ClusterPopulation,
    *,
    with_curriculum: bool,
) -> dict[str, float]:
    """Evaluate both parameter sets and, for FlexMatch, its final live gate."""
    population = result.population
    if population is None:
        raise RuntimeError("flexmatch benchmark expected a training population")
    if result.teacher is None:
        raise RuntimeError("flexmatch benchmark expected its evaluation EMA")
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        teacher_values = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        teacher_propensity = teacher_values[Port.T_GIVEN_X]
        teacher_outcome = teacher_values[Port.Y_GIVEN_XT]
        if not isinstance(teacher_propensity, CategoricalTreatment) or not isinstance(
            teacher_outcome, GaussianOutcome
        ):
            raise TypeError("flexmatch benchmark expected its reviewed P5 heads")
        metrics = {
            "ema_treatment_nll": float(
                F.nll_loss(teacher_propensity.log_probs, scaled.t)
            ),
            "ema_outcome_nll": float(
                -teacher_outcome.log_prob(scaled.y, scaled.t).mean()
            ),
        }

        trained_values = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        trained_propensity = trained_values[Port.T_GIVEN_X]
        if not isinstance(trained_propensity, CategoricalTreatment):
            raise TypeError("flexmatch benchmark expected a categorical propensity")
        metrics["trained_treatment_nll"] = float(
            F.nll_loss(trained_propensity.log_probs, scaled.t)
        )

        if not with_curriculum:
            return metrics
        status = result.objective_states.get(_CURRICULUM)
        if not isinstance(status, CurriculumStatus):
            raise RuntimeError("flexmatch result did not expose its curriculum state")
        thresholds = status.thresholds(schema.treatment_cardinality).to(
            population.rows.x.dtype
        )
        train_values = run.graph.evaluate(
            population.rows, schema=schema, only=run.graph.names
        )
        train_propensity = train_values[Port.T_GIVEN_X]
        if not isinstance(train_propensity, CategoricalTreatment):
            raise TypeError("flexmatch benchmark expected a categorical propensity")
        confidence, labels = train_propensity.probs.max(dim=-1)
        retained = confidence > thresholds.index_select(0, labels)
        metrics.update(
            {
                "mask_rate": float(retained.float().mean()),
                "impurity": (
                    float(
                        (labels[retained] != population.rows.t[retained]).float().mean()
                    )
                    if int(retained.sum())
                    else 0.0
                ),
                "threshold_max": float(thresholds.max()),
                "initial_threshold_min": _diagnostic(result, 0, "threshold_min"),
                "marked_fraction": 1.0 - status.unused() / status.size,
            }
        )
        return metrics


def _diagnostic(result: StageResult, step: int, field: str) -> float:
    for term in result.records[step].terms:
        if term.name == _CURRICULUM:
            return float(term.diagnostics[field])
    raise RuntimeError(f"flexmatch step {step} has no curriculum term")


__all__ = ["run"]
