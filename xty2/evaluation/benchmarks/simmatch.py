"""SimMatch's paired no-propagation benchmark from card section 6.

The fixture is `fixmatch.md` §6.1's, imported rather than restated, because
`simmatch.md` §6.1 says it is "the shipped FixMatch fixture, repeated here so
this card's benchmark contract does not depend on a deleted or amended section
of another card". Two transcriptions of one DGP is how the two cards' numbers
would stop being about the recipes.

The pair is the one §6 defines: full SimMatch against `hat p = p^w` and
`hat q = q^w`, which is `alpha = 1` with equations (7)-(8)'s unfolding
disabled. Everything else — the soft semantic loss, distribution alignment, the
projection head, the labelled bank, the instance loss, the temporal update, the
gate, the quotas, the views, the optimiser, the schedule, the initial
parameters, the seeds and the batch stream — is held identical, so the
difference between the arms is the two propagation arrows and nothing else.

Four of the eight required numbers are *target* quality rather than model
quality, and they are read from the run's own final bank
(`StageResult.objective_states`) rather than from a bank rebuilt beside it: the
question §6 asks is whether propagation improved the targets the run actually
trained on, and a rebuilt bank answers a neighbouring question about a bank
nothing wrote. Card §6's tolerance reads them terminally, which is what one
post-training preparation on the whole training population measures.

The gate rate and the alignment margin come from the run's own step records
instead, for `doublematch`'s reason: both describe the *views* eq. (2) and
eq. (5) charged, and the held-out population is not drawn under either.
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
    compile,
)
from xty2.core.rows import resolve_rows
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
from xty2.evaluation.benchmarks.fixmatch import (
    _BASE_SEED,
    _TEST_ROWS,
    _TRAIN_ROWS,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.objectives import (
    LabeledMemoryInstanceConsistency,
    LabeledSimilarityMemory,
    SimilarityMatchingTreatmentNLL,
)
from xty2.recipes import simmatch
from xty2.recipes.simmatch import MEMORY_TERM, SIMILARITY_MATCHING, SIMMATCH_STEPS
from xty2.training import StageResult, run_stage

_INSTANCE_TERM = "labeled_memory_instance_consistency"
_TERMINAL_STEPS = 100
"""What "terminal" counts for the gate rate, and it is a window not a step.

One step of a stochastic quota is a draw, not a terminus. A hundred is the
window `tests/smoke/test_simmatch.py` already reads its late means over, so the
two tiers report the same statistic under the same name.
"""

_VIEW_KEY_OFFSET = 4
"""The view RNG key of the terminal target reading, offset from the base seed.

Equations (7)-(10) read `p^w` and `z^w`, so scoring their targets on an
untransformed batch would score a third realisation no equation names. One key
per replicate makes the draw deterministic and different in each of the ten, so
the reported mean averages over view noise instead of over one lucky mask.

The alignment margin needs no key: it is the run's own per-step statistic, read
over the terminal window from the diagnostics eq. (5) logs while it trains, on
the realisations the loss actually charged.
"""


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired SimMatch / no-propagation fits."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in 6.1"
            ),
            "variant": (
                "paired full SimMatch against no semantic-instance "
                "propagation; identical seeds, initialisation, batches, "
                "optimiser, schedule, views and all non-propagation mechanics"
            ),
            "split": (
                "1024 train rows with exactly 64 observed treatments; 2048 "
                "held-out rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio full over no-propagation, for "
                "student and evaluation EMA; online hidden-label NLL ratios "
                "for hat p over p^w and aggregate(hat q) over aggregate(q^w); "
                "outcome NLL, gate rate, bank coverage and representation "
                "alignment as guardrails"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "held-out treatment-NLL ratio < 1.0 in mean by at least one "
                "standard error for both student and EMA; terminal hat-p "
                "target-NLL ratio < 1.0; terminal aggregate-hat-q target-NLL "
                "ratio < 1.0; held-out outcome NLL within 1.05x of the "
                "ablation; terminal gate rate >= 0.5; bank coverage = 1.0 "
                "before the first propagated target; mean same-row weak/strong "
                "cosine at least 0.2 above mean cross-row cosine"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"simmatch card reviewed ten replicates, got {spec.seed_count}"
        )
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
                "terminal_hat_p_target_NLL_ratio", column(rows, "hat_p_ratio"), 1.0
            ),
            MetricResult.upper_bound(
                "terminal_aggregate_hat_q_target_NLL_ratio",
                column(rows, "hat_q_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio", column(rows, "outcome_ratio"), 1.05
            ),
            MetricResult.lower_bound(
                "terminal_gate_rate", column(rows, "gate_rate"), 0.5
            ),
            MetricResult.lower_bound(
                "bank_coverage_before_first_propagation",
                column(rows, "bank_coverage"),
                1.0,
            ),
            MetricResult.lower_bound(
                "cross_view_alignment_margin", column(rows, "alignment_margin"), 0.2
            ),
            MetricResult.information(
                "simmatch_student_treatment_NLL",
                column(rows, "simmatch_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "ablation_student_treatment_NLL",
                column(rows, "ablation_student_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "simmatch_ema_treatment_NLL",
                column(rows, "simmatch_ema_nll"),
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
            # The two halves of the alignment margin, so a margin that fails
            # says which of "the views agree" and "every row agrees with every
            # other row" it failed by.
            MetricResult.information(
                "terminal_same_row_cosine", column(rows, "same_row_cosine")
            ),
            MetricResult.information(
                "terminal_cross_row_cosine", column(rows, "cross_row_cosine")
            ),
            MetricResult.information(
                "ablation_terminal_gate_rate", column(rows, "ablation_gate_rate")
            ),
            MetricResult.information(
                "ablation_cross_view_alignment_margin",
                column(rows, "ablation_alignment_margin"),
            ),
            # The four target NLLs behind the two target ratios. A ratio alone
            # cannot distinguish "propagation made the target worse" from "both
            # targets are good and one is sharper", and those are different
            # findings about equations (8) and (10).
            MetricResult.information(
                "terminal_hat_p_target_NLL",
                column(rows, "hat_p_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "terminal_p_weak_target_NLL",
                column(rows, "p_weak_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "terminal_aggregate_hat_q_target_NLL",
                column(rows, "aggregate_hat_q_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "terminal_aggregate_q_weak_target_NLL",
                column(rows, "aggregate_q_weak_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "terminal_nearest_slot_agreement", column(rows, "slot_agreement")
            ),
        ),
        interpretation=(
            "This is the predeclared project-local SimMatch mechanism target: "
            "do equations (8) and (10) — the instance distribution over "
            "labelled slots calibrated by the semantic prediction, and the "
            "semantic target smoothed by that distribution's class aggregate — "
            "improve a *treatment* propensity and the targets it trains on, "
            "against an otherwise identical fit whose two propagation arrows "
            "are switched off. It is not a reproduction of Zheng et al., whose "
            "inputs, labels, architecture, augmentation vocabulary, metric and "
            "CIFAR-scale budget all differ (deviations 2, 3, 4 and 10)."
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
    full_recipe = simmatch(schema)
    torch.manual_seed(base + 6)
    ablated_recipe = _without_propagation(simmatch(schema))
    for name, value in full_recipe.system.state_dict().items():
        if not torch.equal(value, ablated_recipe.system.state_dict()[name]):
            raise RuntimeError(f"simmatch paired initial state differs at {name!r}")

    full_run = compile(full_recipe)
    ablated_run = compile(ablated_recipe)
    # Identical seeds: both stages are index 0 of a one-stage program, so the
    # sampler stream, the view keys and the parameter initialisation all match
    # and the two propagation arrows are the only difference between the arms.
    full = run_stage(full_run, "joint_fit", data, seed=base + 10_000)
    ablated = run_stage(ablated_run, "joint_fit", data, seed=base + 10_000)

    full_metrics = _evaluate(full_run, full, test)
    ablated_metrics = _evaluate(ablated_run, ablated, test)
    for name, value in ablated_metrics.items():
        if value <= 0.0:
            raise RuntimeError(
                f"the ablation produced a non-positive {name}, so the paired "
                "ratio the card declares is undefined"
            )
    targets = _terminal_targets(full_run, full, rng_key=base + _VIEW_KEY_OFFSET)
    same_row = _terminal(full, _INSTANCE_TERM, "same_row_cosine")
    cross_row = _terminal(full, _INSTANCE_TERM, "cross_row_cosine")
    return {
        **targets,
        "student_ratio": (
            full_metrics["student_treatment_nll"]
            / ablated_metrics["student_treatment_nll"]
        ),
        "ema_ratio": (
            full_metrics["ema_treatment_nll"] / ablated_metrics["ema_treatment_nll"]
        ),
        "outcome_ratio": (
            full_metrics["ema_outcome_nll"] / ablated_metrics["ema_outcome_nll"]
        ),
        "hat_p_ratio": targets["hat_p_nll"] / targets["p_weak_nll"],
        "hat_q_ratio": targets["aggregate_hat_q_nll"] / targets["aggregate_q_weak_nll"],
        "gate_rate": _terminal(full, MEMORY_TERM, "coverage"),
        "ablation_gate_rate": _terminal(ablated, MEMORY_TERM, "coverage"),
        "bank_coverage": _coverage_before_first_propagation(full),
        "alignment_margin": same_row - cross_row,
        "same_row_cosine": same_row,
        "cross_row_cosine": cross_row,
        "slot_agreement": _terminal(full, _INSTANCE_TERM, "nearest_slot_agreement"),
        "ablation_alignment_margin": (
            _terminal(ablated, _INSTANCE_TERM, "same_row_cosine")
            - _terminal(ablated, _INSTANCE_TERM, "cross_row_cosine")
        ),
        "simmatch_student_nll": full_metrics["student_treatment_nll"],
        "ablation_student_nll": ablated_metrics["student_treatment_nll"],
        "simmatch_ema_nll": full_metrics["ema_treatment_nll"],
        "ablation_ema_nll": ablated_metrics["ema_treatment_nll"],
        "frequency_nll": full_metrics["frequency_nll"],
    }


def _without_propagation(recipe: Recipe) -> Recipe:
    """The §6 ablation: `hat p = p^w` and `hat q = q^w`. Nothing else moves.

    `alpha = 1` makes equation (10) return the aligned weak prediction exactly,
    and `unfold = False` makes equation (8) return equation (7)'s `q^w`
    unchanged. Both objectives keep the same shared spec object, so the bank,
    its momentum, the alignment window, the warm-up, the gate and both
    temperatures are the full arm's — and the memory still refuses a spec its
    owner does not carry.
    """
    stage = recipe.program[0]
    semantic = stage.objectives[2].objective
    instance = stage.objectives[3].objective
    if not isinstance(semantic, SimilarityMatchingTreatmentNLL) or not isinstance(
        instance, LabeledMemoryInstanceConsistency
    ):
        raise RuntimeError(
            "simmatch's third and fourth terms are no longer equations (2) and (5)"
        )
    spec = replace(SIMILARITY_MATCHING, alpha=1.0, unfold=False)
    objectives = (
        *stage.objectives[:2],
        replace(stage.objectives[2], objective=replace(semantic, spec=spec)),
        replace(stage.objectives[3], objective=replace(instance, spec=spec)),
        *stage.objectives[4:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _evaluate(
    run: CompiledRun, result: StageResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out NLLs from both parameter sets (deviation 9 reports both)."""
    population = result.population
    if population is None:
        raise RuntimeError(
            "the fitting stage reported no training population; the recipe "
            "declares a sampler, so one is expected"
        )
    if result.teacher is None:
        raise RuntimeError("simmatch declares an evaluation EMA and reported none")
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        teacher = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        student = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        ema_propensity = teacher[Port.T_GIVEN_X]
        outcome = teacher[Port.Y_GIVEN_XT]
        propensity = student[Port.T_GIVEN_X]
        if (
            not isinstance(ema_propensity, CategoricalTreatment)
            or not isinstance(propensity, CategoricalTreatment)
            or not isinstance(outcome, GaussianOutcome)
        ):
            raise TypeError("simmatch benchmark expected its reviewed P5 heads")
        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(observed, minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        return {
            "student_treatment_nll": float(F.nll_loss(propensity.log_probs, scaled.t)),
            "ema_treatment_nll": float(F.nll_loss(ema_propensity.log_probs, scaled.t)),
            "ema_outcome_nll": float(-outcome.log_prob(scaled.y, scaled.t).mean()),
            "frequency_nll": float(F.nll_loss(baseline, scaled.t)),
        }


def _terminal_targets(
    run: CompiledRun, result: StageResult, *, rng_key: int
) -> dict[str, float]:
    """Score both propagated targets, and both unpropagated ones, on hidden `t`.

    One preparation of the run's own final bank against the whole training
    population under the weak view. Four distributions come out of that single
    call, and none of them is recomputed here:

    * `hat p` is `targets.semantic` and `p^w` is `targets.aligned`.
    * `aggregate(hat q)` is equation (9) applied to `targets.instance`.
    * `aggregate(q^w)` is recovered from equation (10) itself. That equation is
      `hat p = alpha p^w + (1 - alpha) aggregate(q^w)`, so the aggregate is
      `(hat p - alpha p^w) / (1 - alpha)` — the arm's own arithmetic rearranged,
      rather than a second transcription of equations (7) and (9) that could
      drift from the shipped one. It is why this reads the *full* arm: at the
      ablation's `alpha = 1` the rearrangement divides by zero, which is another
      way of saying the ablation has no aggregate to score.
    """
    population = result.population
    if population is None:
        raise RuntimeError("simmatch's terminal target reading needs the population")
    memory = result.objective_states.get(MEMORY_TERM)
    if not isinstance(memory, LabeledSimilarityMemory):
        raise RuntimeError(
            f"stage state {MEMORY_TERM!r} is {type(memory).__name__}, not the "
            "labelled similarity memory the terminal targets are read from"
        )
    spec = memory.spec
    if spec.alpha >= 1.0:
        raise RuntimeError(
            "the terminal target ratios are defined on the full arm, whose "
            f"alpha is below 1; this state carries alpha={spec.alpha!r}"
        )
    batch = population.rows
    rows = resolve_rows(batch, "t_missing")
    schema = run.recipe.schema
    weak = _view(run, "weak_x").apply(
        batch, schema, rng_key=rng_key, population=population
    )
    with torch.no_grad():
        values = run.graph.evaluate(weak, schema=schema, only=run.graph.names)
        propensity = values[Port.T_GIVEN_X]
        embedding = values[Port.X_PROJ]
        if not isinstance(propensity, CategoricalTreatment) or not isinstance(
            embedding, Tensor
        ):
            raise TypeError("simmatch benchmark expected a propensity and a projection")
        # A step past the end of the run: the bank is the one the last step
        # wrote, and `prepare` refuses to move backwards.
        targets = memory.prepare(
            step=SIMMATCH_STEPS,
            raw_probabilities=propensity.probs,
            weak_embeddings=embedding,
            batch=batch,
            eligible_rows=rows,
            support_rows=resolve_rows(batch, "t_observed"),
        )
        if not targets.propagated or targets.instance is None:
            raise RuntimeError(
                "the terminal preparation did not propagate; the bank of a "
                f"finished {SIMMATCH_STEPS}-step run is expected to be covered"
            )
        labels = memory.labels.to(device=targets.aligned.device)
        aggregate_hat_q = torch.zeros_like(targets.aligned).index_add(
            1, labels, targets.instance
        )
        aggregate_q_weak = (targets.semantic - spec.alpha * targets.aligned) / (
            1.0 - spec.alpha
        )
        hidden = batch.t.index_select(0, rows)
        return {
            "hat_p_nll": _target_nll(targets.semantic, hidden),
            "p_weak_nll": _target_nll(targets.aligned, hidden),
            "aggregate_hat_q_nll": _target_nll(aggregate_hat_q, hidden),
            "aggregate_q_weak_nll": _target_nll(aggregate_q_weak, hidden),
        }


def _view(run: CompiledRun, name: str) -> ViewSpec:
    for view in run.recipe.views:
        if view.name == name:
            return view
    raise RuntimeError(f"simmatch no longer declares a view named {name!r}")


def _target_nll(distribution: Tensor, hidden: Tensor) -> float:
    """Cross-entropy of one soft target against the fixture's hidden `t`."""
    return float(F.nll_loss(distribution.clamp_min(1e-12).log(), hidden))


def _diagnostic(result: StageResult, step: int, name: str, field: str) -> float:
    """One logged field of one term at one step, or a named failure."""
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
    """The mean of one diagnostic over the terminal window."""
    steps = range(len(result.records) - _TERMINAL_STEPS, len(result.records))
    return sum(_diagnostic(result, step, name, field) for step in steps) / len(steps)


def _coverage_before_first_propagation(result: StageResult) -> float:
    """Was every slot filled at the step equations (8) and (10) first ran?

    Reported as an executable guardrail rather than as a fraction, because the
    card's tolerance is an equality: a bank that propagated while a slot was
    still empty would be reading a vector nothing wrote, and there is no
    partial credit for that.
    """
    for step in range(len(result.records)):
        if _diagnostic(result, step, MEMORY_TERM, "propagated") == 1.0:
            return bool_float(
                _diagnostic(result, step, MEMORY_TERM, "bank_coverage") == 1.0
            )
    raise RuntimeError("no step of the run propagated, so the pair measures nothing")
