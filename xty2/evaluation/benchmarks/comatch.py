"""CoMatch's paired FixMatch benchmark from card section 6.

The two arms use one component graph on the same fixture and stage seed; the
baseline replaces CoMatch's two gradients with FixMatch's hard weak/strong
pseudo-label objective. Their shared quota sampler therefore draws the same
observed and missing row populations at every step. CoMatch's mechanism
guardrails are read from the diagnostics its two losses emitted while training;
the retained-label impurity is read from a single terminal preparation of the
run's own memory bank.
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
    Recipe,
    ViewSpec,
    Weighted,
    compile,
)
from xty2.core.rows import resolve_rows
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
from xty2.evaluation.benchmarks.fixmatch import _BASE_SEED, _TEST_ROWS, _TRAIN_ROWS
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec
from xty2.objectives import (
    InfoNCEContrastive,
    MemorySmoothedLabels,
    PseudoLabelTreatmentNLL,
)
from xty2.recipes import comatch
from xty2.recipes.comatch import (
    COMATCH_STEPS,
    PSEUDO_LABEL_TERM,
    STRONG_X,
    WEAK_X,
)
from xty2.training import StageResult, run_stage

_GRAPH_TERM = "pseudo_label_graph_contrastive"
_TERMINAL_STEPS = 100
_VIEW_KEY_OFFSET = 4


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten paired CoMatch / FixMatch fits declared by the card."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in 6.1"
            ),
            "variant": (
                "paired fit against a matched FixMatch-objective arm, same "
                "initial parameters, declared views, seeds, fixture, batches, "
                "optimiser on shared parameters and schedule"
            ),
            "split": (
                "1024 train rows with 64 observed treatments, 2048 held-out "
                "rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio, comatch over fixmatch, reported for "
                "the student and for the evaluation EMA; terminal mask rate, "
                "retained-label impurity, mean edges per row and embedding "
                "alignment adjusted by the exact different-treatment pair "
                "fraction as mechanism guardrails"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "treatment-NLL ratio < 1.0 in mean by at least one standard "
                "error, on the EMA and the student alike; held-out outcome NLL "
                "within 1.05x of the fixmatch arm; terminal mask rate at least "
                "0.5; mean edges per row strictly between 1.0 (self-loops only) "
                "and 0.5 * mu * B; mean cosine similarity of a row to its own "
                "second strong view at least 0.2 above its mean similarity to "
                "the other unlabelled rows of the batch, after dividing the "
                "raw margin by the exact fraction of ordered distinct missing-"
                "row pairs with different treatments"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(f"comatch card reviewed ten replicates, got {spec.seed_count}")
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
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
                "terminal_mask_rate", column(rows, "mask_rate"), 0.5
            ),
            MetricResult.interval(
                "terminal_edges_per_row",
                column(rows, "edges_per_row"),
                1.000001,
                223.999999,
            ),
            MetricResult.lower_bound(
                "cross_class_adjusted_alignment_margin",
                column(rows, "adjusted_alignment_margin"),
                0.2,
            ),
            MetricResult.information(
                "retained_label_impurity", column(rows, "impurity")
            ),
            MetricResult.information(
                "terminal_same_row_cosine", column(rows, "same_row_cosine")
            ),
            MetricResult.information(
                "terminal_cross_row_cosine", column(rows, "cross_row_cosine")
            ),
            MetricResult.information(
                "raw_cross_view_alignment_margin", column(rows, "alignment_margin")
            ),
            MetricResult.information(
                "different_treatment_pair_fraction",
                column(rows, "cross_class_fraction"),
            ),
            MetricResult.information(
                "terminal_repeated_bank_rows", column(rows, "repeated_bank_rows")
            ),
            MetricResult.information(
                "comatch_student_treatment_NLL",
                column(rows, "comatch_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "fixmatch_student_treatment_NLL",
                column(rows, "fixmatch_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "comatch_ema_treatment_NLL",
                column(rows, "comatch_ema_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "fixmatch_ema_treatment_NLL",
                column(rows, "fixmatch_ema_nll"),
                unit="nat/row",
            ),
        ),
        interpretation=(
            "This is the predeclared project-local CoMatch mechanism target: "
            "whether memory-smoothed labels and their pseudo-label graph improve "
            "missing-treatment classification over the paper's FixMatch baseline. "
            "It is not a reproduction of the paper's image benchmarks."
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
    comatch_recipe = comatch(schema)
    torch.manual_seed(base + 6)
    fixmatch_recipe = _fixmatch_objective(comatch(schema))
    for name, value in comatch_recipe.system.state_dict().items():
        if not torch.equal(value, fixmatch_recipe.system.state_dict()[name]):
            raise RuntimeError(f"comatch paired initial state differs at {name!r}")
    comatch_run = compile(comatch_recipe)
    fixmatch_run = compile(fixmatch_recipe)
    stage_seed = base + 10_000
    comatch_result = run_stage(comatch_run, "joint_fit", data, seed=stage_seed)
    fixmatch_result = run_stage(fixmatch_run, "joint_fit", data, seed=stage_seed)
    if not torch.equal(
        comatch_result.checkpoint.trained_on_row_ids,
        fixmatch_result.checkpoint.trained_on_row_ids,
    ):
        raise RuntimeError("comatch paired arms saw different training rows")

    full = _evaluate(comatch_run, comatch_result, test)
    baseline = _evaluate(fixmatch_run, fixmatch_result, test)
    terminal = _terminal_targets(
        comatch_run, comatch_result, rng_key=base + _VIEW_KEY_OFFSET
    )
    for name in ("student_treatment_nll", "ema_treatment_nll", "ema_outcome_nll"):
        if baseline[name] <= 0.0:
            raise RuntimeError(f"fixmatch produced a non-positive {name}")
    same = _terminal(comatch_result, _GRAPH_TERM, "alignment")
    cross = _terminal(comatch_result, _GRAPH_TERM, "uniformity")
    cross_class_fraction = _cross_class_fraction(comatch_result)
    return {
        "student_ratio": full["student_treatment_nll"]
        / baseline["student_treatment_nll"],
        "ema_ratio": full["ema_treatment_nll"] / baseline["ema_treatment_nll"],
        "outcome_ratio": full["ema_outcome_nll"] / baseline["ema_outcome_nll"],
        "mask_rate": _terminal(comatch_result, PSEUDO_LABEL_TERM, "coverage"),
        "edges_per_row": _terminal(comatch_result, _GRAPH_TERM, "edges_per_row"),
        "alignment_margin": same - cross,
        "adjusted_alignment_margin": (same - cross) / cross_class_fraction,
        "cross_class_fraction": cross_class_fraction,
        "same_row_cosine": same,
        "cross_row_cosine": cross,
        "repeated_bank_rows": _terminal(
            comatch_result, _GRAPH_TERM, "repeated_bank_rows"
        ),
        "impurity": terminal["impurity"],
        "comatch_student_nll": full["student_treatment_nll"],
        "fixmatch_student_nll": baseline["student_treatment_nll"],
        "comatch_ema_nll": full["ema_treatment_nll"],
        "fixmatch_ema_nll": baseline["ema_treatment_nll"],
    }


def _cross_class_fraction(result: StageResult) -> float:
    """Fraction of ordered distinct missing-row pairs with different labels."""
    population = result.population
    if population is None:
        raise RuntimeError("comatch alignment adjustment needs the population")
    batch = population.rows
    hidden = batch.t.index_select(0, resolve_rows(batch, "t_missing"))
    counts = torch.bincount(hidden, minlength=2).to(torch.float64)
    total = int(hidden.numel())
    if total < 2:
        raise RuntimeError("comatch alignment adjustment needs two missing rows")
    different = total * total - float(counts.square().sum())
    fraction = different / (total * (total - 1))
    if not 0.0 < fraction <= 1.0:
        raise RuntimeError(f"invalid different-treatment pair fraction {fraction}")
    return fraction


def _fixmatch_objective(recipe: Recipe) -> Recipe:
    """Replace CoMatch's gradients with hard weak/strong pseudo-labelling.

    The inserted term is FixMatch equation (4), scoped to CoMatch's disjoint
    unlabelled population so no row-policy difference enters the pair. A
    zero-weight identity-target contrastive term realizes the second strong
    view and projection without contributing a gradient, keeping the forward
    surface and complete component graph identical.
    """
    stage = recipe.program[0]
    baseline = Weighted(
        PseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=WEAK_X,
            prediction=STRONG_X[0],
            threshold=0.95,
            sharpening="hard",
            stop_grad="target",
            rows="t_missing",
            name="fixmatch_pseudo_label_treatment_nll",
        ),
        weight=1.0,
        reduction="mean",
    )
    realised_projection = Weighted(
        InfoNCEContrastive(
            port=Port.X_PROJ,
            anchor=STRONG_X[0],
            contrast=STRONG_X[1],
            temperature=0.2,
            rows="t_missing",
            name="zero_weight_projection_realisation",
        ),
        weight=0.0,
        reduction="mean",
    )
    objectives = (
        *stage.objectives[:2],
        baseline,
        realised_projection,
        *stage.objectives[4:],
    )
    return replace(
        recipe,
        program=Program((replace(stage, objectives=objectives),)),
    )


def _evaluate(
    run: CompiledRun, result: StageResult, test: ClusterPopulation
) -> dict[str, float]:
    population = result.population
    if population is None:
        raise RuntimeError("comatch benchmark expected a fitted training population")
    if result.teacher is None:
        raise RuntimeError(f"{run.recipe.name} declares an evaluation EMA")
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        teacher = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        student = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        ema_propensity = teacher[Port.T_GIVEN_X]
        propensity = student[Port.T_GIVEN_X]
        outcome = teacher[Port.Y_GIVEN_XT]
        if (
            not isinstance(ema_propensity, CategoricalTreatment)
            or not isinstance(propensity, CategoricalTreatment)
            or not isinstance(outcome, GaussianOutcome)
        ):
            raise TypeError("comatch benchmark expected its reviewed P5 heads")
        return {
            "student_treatment_nll": float(F.nll_loss(propensity.log_probs, scaled.t)),
            "ema_treatment_nll": float(F.nll_loss(ema_propensity.log_probs, scaled.t)),
            "ema_outcome_nll": float(-outcome.log_prob(scaled.y, scaled.t).mean()),
        }


def _terminal_targets(
    run: CompiledRun, result: StageResult, *, rng_key: int
) -> dict[str, float]:
    population = result.population
    if population is None:
        raise RuntimeError("comatch terminal target reading needs the population")
    memory = result.objective_states.get(PSEUDO_LABEL_TERM)
    if not isinstance(memory, MemorySmoothedLabels):
        raise RuntimeError("comatch terminal target reading found no label memory")
    batch = population.rows
    rows = resolve_rows(batch, "t_missing")
    support = resolve_rows(batch, "t_observed")
    weak = _view(run, "weak_x").apply(
        batch, run.recipe.schema, rng_key=rng_key, population=population
    )
    with torch.no_grad():
        values = run.graph.evaluate(
            weak, schema=run.recipe.schema, only=run.graph.names
        )
        propensity = values[Port.T_GIVEN_X]
        embedding = values[Port.X_PROJ]
        if not isinstance(propensity, CategoricalTreatment) or not isinstance(
            embedding, Tensor
        ):
            raise TypeError("comatch benchmark expected a propensity and projection")
        pseudo = memory.prepare(
            step=COMATCH_STEPS,
            raw_probabilities=propensity.probs,
            weak_embeddings=embedding,
            batch=batch,
            eligible_rows=rows,
            support_rows=support,
        )
        confidence, labels = pseudo.max(dim=-1)
        retained = confidence >= memory.graph.thresholds.pseudo_label
        hidden = batch.t.index_select(0, rows)
        impurity = (
            float((labels[retained] != hidden[retained]).float().mean())
            if bool(retained.any())
            else 0.0
        )
    return {"impurity": impurity}


def _view(run: CompiledRun, name: str) -> ViewSpec:
    for view in run.recipe.views:
        if view.name == name:
            return view
    raise RuntimeError(f"comatch no longer declares a view named {name!r}")


def _diagnostic(result: StageResult, step: int, name: str, field: str) -> float:
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


def _terminal(result: StageResult, name: str, field: str) -> float:
    steps = range(len(result.records) - _TERMINAL_STEPS, len(result.records))
    return sum(_diagnostic(result, step, name, field) for step in steps) / len(steps)


__all__ = ["run"]
