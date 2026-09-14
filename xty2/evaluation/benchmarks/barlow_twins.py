"""Barlow Twins' four-arm paired mechanism study from card section 6.

The measurement is the pairing. `full` and `diagonal_only` are the same recipe
with equation (1)'s `lambda` set to exactly zero, and everything else —
components, initial tensors, data policy, row stream, cached view draws, step
counts and downstream objectives — is held fixed, so a difference between them
is attributable to the off-diagonal penalty and to nothing else. `no_pretrain`
drops the pretraining stage to price the transfer, and `vicreg` is card §2's
contextual arm, which carries no acceptance bound because its objective *and*
its projector's hidden biases both differ.

Card §6.4 turns that into four one-standard-error targets: the full arm's
cross-view diagonal alignment, the fraction of its embedding coordinates that
are active rather than epsilon-dominated, the redundancy the off-diagonal term
removes relative to its own ablation, and a budget on what transferring the
encoder costs the factual outcome fit.

**What this module measures and what it does not.** Nothing here is a claim
about Zbontar et al.'s ImageNet numbers. Card §5 records five judgement
departures — the parser's `lambda` and the output BatchNorm's reductions, a
tabular encoder and a 512-wide projector, oracle DGP symmetries in place of
image augmentations, Adam on a fixed 1,000/3,000-step budget in place of LARS
on an epoch schedule, and a local XTY fitting stack — and §6's protocol tests
the *mechanism* under those departures on this fixture. Card §6.4 also states
what the arithmetic forbids: at `B = 128` and `d = 512` a centred cross-product
has rank at most 127, so `C = I` is unreachable and zero loss is not the target.

**How the pairing is checked.** Card §6.2 asks for equality of the actual
sampled rows and view draws rather than of the seeds that produced them, and it
is checked four ways, from the strongest evidence down:

* The hashed row ids and corrupted view tensors the executor itself passed
  through `ViewSpec.apply`, over the whole of pretraining, must agree across
  all three pretraining arms. That is observed from inside the run rather than
  re-derived beside it, and it covers the contextual arm, whose objectives have
  different names and so cannot be compared term by term.
* Every logged objective value and diagnostic at pretraining step 0 must agree
  bit for bit between `full` and `diagonal_only`. Those two hold identical
  parameters at that step, so an identical unweighted `cross_correlation_*`
  value is a statement about the rows and *both* view draws the executor fed
  the loss.
* The stage seeds the executor reports must agree, which is what says the
  no-pretraining arm's single stage really did land on the stream the paired
  arms' second stage walks, `STREAM_STRIDE` and all.
* The row-id stream, both corrupted view tensors and the treatment mask are
  then materialised through the loader entry points the executor calls —
  `build_population` and `iterate` for the rows, `ViewSpec.apply` for the
  views — and compared element by element. This catches a difference in what
  each arm *declares* (a sampler, a view spec, a missingness budget) that equal
  seeds and equal hashes over a shorter stage would hide.

Two properties need a hook inside the step loop and are pinned by Tier 1 on
every declared seed rather than re-proved here: that each stage builds a fresh
optimiser with empty state, and that the heads are untouched at the transition.
Tier 2 observes their consequences instead — a pretraining checkpoint holding
no head parameter, and a projector that fine-tuning leaves bit-identical.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Dataset,
    DataSpec,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    Schema,
    TrainingPopulation,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    SEPARATED,
    ClusterPopulation,
    column,
    configure_worker,
    continuous_schema,
    on_the_training_scale,
    parallel_replicates,
    take,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.causal import (
    absolute_ate_error,
    average_treatment_effect,
    candidate_treatment_means,
    sqrt_pehe,
    treatment_contrast,
)
from xty2.evaluation.predictive import treatment_nll
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec
from xty2.evaluation.vicreg_views import (
    PRESERVATION_TOLERANCE,
    OracleSymmetry,
    fit_with_trace,
    targets,
)
from xty2.objectives.barlow_twins import cross_correlation
from xty2.recipes import barlow_twins, vicreg
from xty2.recipes.barlow_twins import (
    BARLOW_TWINS_BATCH_SIZE,
    BARLOW_TWINS_OBSERVED_TREATMENTS,
    BARLOW_TWINS_PRETRAIN_STEPS,
    NORMALISATION_EPSILON,
    OFF_DIAGONAL_WEIGHT,
    POPULATION_CORRECTION,
    PRESERVED_FIELDS,
)
from xty2.training import STREAM_STRIDE, ProgramResult
from xty2.training.loading import build_population, iterate

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_ROW_OFFSET = 10_000
_BASE_SEED = 390_000
"""Card §6.2: replicate `i` runs at `base = 390000 + 100 * i`, with the train
rows at row offset 0 and the held-out rows at 10000. Tier 1's bases 419/523/631
are a different stream and are never reused here."""

_EVAL_BATCHES = 16
"""Card §6.2: 16 disjoint held-out batches of `BARLOW_TWINS_BATCH_SIZE` rows."""

_BRANCHES = 2
_VIEW_SEED_BASE = 20_000
"""Card §6.2: the two view generators for batch `b` are seeded
`base + 20000 + 2 * b` and `base + 20001 + 2 * b`."""

_ARMS = ("full", "diagonal_only", "no_pretrain", "vicreg")
_PRETRAINED = ("full", "diagonal_only", "vicreg")
_PAIRED = ("full", "diagonal_only")
"""The two arms that differ by one number. They share the recipe, so they share
the objective *names* the step-0 comparison indexes by; the contextual arm does
not, and is paired through the executor's own view/row hashes instead."""

_ACTIVE_VARIANCE = 100.0 * NORMALISATION_EPSILON
"""Card §6.4's activity cutoff: a coordinate counts when its *raw* population
variance clears one hundred epsilons in **both** branches, which is what keeps
`C` from being an artefact of the epsilon in its own denominator."""

_OBJECTIVES = ("cross_correlation_diagonal", "cross_correlation_off_diagonal")
_OFF_DIAGONAL = "cross_correlation_off_diagonal"

_ENCODER = "mlp_encoder."
_HEADS = ("tarnet_head.", "categorical_propensity.")
_PROJECTORS = {"vicreg": "vicreg_expander."}
"""Every arm but the contextual one embeds through `barlow_twins_projector`."""

_DEFAULT_PROJECTOR = "barlow_twins_projector."

_ARM_DIAGNOSTICS = (
    "outcome_NLL",
    "treatment_NLL",
    "ATE_error",
    "treatment_effect_RMSE",
)
_EMBEDDING_DIAGNOSTICS = (
    "diagonal_alignment",
    "redundancy",
    "active_fraction",
    "diagonal_error",
    "raw_variance_first",
    "raw_variance_second",
    "covariance_top_eigenvalue_share",
    "encoder_parameter_norm",
    "projector_parameter_norm",
)
"""Card §6.4's audit set: "report raw variances/norms, diagonal error D/d,
redundancy, covariance eigenvalue concentration, and both branches' variances
to distinguish scale from rank"."""

_PAIRED_DIAGNOSTICS = (
    "pretraining_treatment_NLL_cost",
    "contextual_vicreg_redundancy_gap",
    "contextual_vicreg_alignment_gap",
    "contextual_vicreg_outcome_NLL_cost",
)
_VIEW_DIAGNOSTICS = (
    "view_max_target_error",
    "view_distinct_row_fraction",
    "view_changed_coordinates",
)


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired four-arm replicates and score card §6.4's four bounds."""
    del cache_root
    spec.bind(
        {
            "dataset": "fixed two-cluster XTY DGP, low=SEPARATED",
            "variant": (
                "full Barlow Twins; diagonal-only; no pretraining; contextual VICReg"
            ),
            "split": (
                "1024 train, 40 observed treatments; 2048 fully observed held-out rows"
            ),
            "metric": (
                "diagonal alignment; active dimensions; paired redundancy gap; "
                "factual outcome NLL cost"
            ),
            "published": "none - project-local tabular mechanism adaptation",
            "tolerance": "all required one-standard-error bounds in section 6.4",
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"barlow_twins card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    # Card §6.4's four required bounds, in its order and in its own spelling —
    # `mean - SE >= t` and `mean + SE <= t`, which is exactly what
    # `MetricResult.passed` requires: the relation must hold by at least one
    # standard error of the ten replicate observations.
    required = (
        MetricResult.lower_bound(
            "full_arm_diagonal_alignment",
            column(rows, "full_diagonal_alignment"),
            0.5,
        ),
        MetricResult.lower_bound(
            "full_arm_active_fraction",
            column(rows, "full_active_fraction"),
            0.9,
        ),
        MetricResult.lower_bound(
            "off_diagonal_ablation_redundancy_gap",
            column(rows, "off_diagonal_ablation_redundancy_gap"),
            0.01,
        ),
        MetricResult.upper_bound(
            "pretraining_outcome_NLL_cost",
            column(rows, "pretraining_outcome_NLL_cost"),
            0.05,
            unit="nat/row",
        ),
    )
    # Everything else §6.4 asks to be reported. `MetricResult` keeps the whole
    # replicate vector, so the artifact carries the per-seed numbers rather
    # than only their mean, and a metric that is already a target keeps that
    # name rather than being reported twice under it.
    scored = {metric.name for metric in required}
    reported = (
        *(
            (f"{arm}_arm_{name}", f"{arm}_{name}", name.endswith("NLL"))
            for arm in _ARMS
            for name in _ARM_DIAGNOSTICS
        ),
        *(
            (f"{arm}_arm_{name}", f"{arm}_{name}", False)
            for arm in _PRETRAINED
            for name in _EMBEDDING_DIAGNOSTICS
        ),
        *(
            (name, name, name.endswith("cost"))
            for name in (*_PAIRED_DIAGNOSTICS, *_VIEW_DIAGNOSTICS)
        ),
    )
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            *required,
            *(
                MetricResult.information(
                    name, column(rows, key), unit="nat/row" if nats else ""
                )
                for name, key, nats in reported
                if name not in scored
            ),
        ),
        interpretation=(
            "This is card §6.2's paired four-arm mechanism study on the "
            "approved oracle-symmetry fixture at fresh bases 390000+100*i. It "
            "is not a reproduction of Zbontar et al.: card §2 excludes "
            "ImageNet, superiority to VICReg, general tabular augmentation "
            "validity and causal identification from the claim. The decisive "
            "comparison is Barlow Twins against its own zero-lambda arm, which "
            "differs by that one number and by nothing about the components, "
            "the initial tensors, the rows or the view draws, so the "
            "redundancy gap attributes an effect to the off-diagonal penalty "
            "on this fixture at this budget. The alignment and activity bounds "
            "are what stop that gap being bought with a collapsed embedding, "
            "and the outcome bound is a guardrail on transfer rather than "
            "evidence about treatment effects. The contextual VICReg arm, the "
            "treatment NLL and the conditional-mean effect error carry no "
            "target and are reported for the audit §6.4 requires before any "
            "downstream number is read as a statement about the objective."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = two_cluster_population(
        _TRAIN_ROWS, seed=base + 1, row_offset=0, low=SEPARATED
    )
    test = two_cluster_population(
        _TEST_ROWS, seed=base + 2, row_offset=_ROW_OFFSET, low=SEPARATED
    )
    data = training_dataset(schema, train.batch)

    runs: dict[str, CompiledRun] = {}
    results: dict[str, ProgramResult] = {}
    traces: dict[str, dict[str, str]] = {}
    initial: dict[str, Tensor] = {}
    for arm in _ARMS:
        torch.manual_seed(base + 6)
        build = vicreg if arm == "vicreg" else barlow_twins
        recipe = build(
            schema,
            first_transforms=(OracleSymmetry(),),
            second_transforms=(OracleSymmetry(),),
        )
        if arm == "vicreg":
            # Card §6.2: the contextual arm is given the encoder and head
            # tensors the other three start from, because its expander draws a
            # different number of tensors from the construction RNG and so its
            # own seed cannot deliver them. The copy is before `compile`, which
            # snapshots the graph every stage is restored to.
            _adopt(recipe, initial)
        recipe = _arm(recipe, arm)
        state = {
            name: value.clone() for name, value in recipe.system.state_dict().items()
        }
        if not initial:
            initial = state
        else:
            _require_same_start(state, initial, arm=arm)
        runs[arm] = compile(recipe)
        results[arm], traces[arm] = fit_with_trace(
            runs[arm],
            {stage.name: data for stage in recipe.program},
            # Card §6.2: the no-pretraining arm's fit is stage 0, so its
            # execution seed carries one stride to put it on the stream the
            # other arms' stage 1 walks. `_require_one_stream` checks it landed.
            seed=base + 10_000 + (STREAM_STRIDE if arm == "no_pretrain" else 0),
        )
    _require_one_pretraining_stream(traces)
    _require_one_stream(runs, results, data, base=base)
    _require_pretraining_touched_only_what_it_declares(runs, results, initial)

    metrics: dict[str, float] = {}
    for arm in _ARMS:
        metrics.update(
            (f"{arm}_{name}", value)
            for name, value in _downstream(runs[arm], results[arm], test).items()
        )
    population = _fit_population(results["full"])
    batches = _held_out_batches(test, population)
    views = _held_out_views(schema, batches, population, base=base)
    metrics.update(_require_view_contract(train, test, population, batches, views))
    for arm in _PRETRAINED:
        metrics.update(
            (f"{arm}_{name}", value)
            for name, value in _embeddings(
                runs[arm], results[arm], schema, views, arm=arm
            ).items()
        )
    metrics["off_diagonal_ablation_redundancy_gap"] = (
        metrics["diagonal_only_redundancy"] - metrics["full_redundancy"]
    )
    metrics["pretraining_outcome_NLL_cost"] = (
        metrics["full_outcome_NLL"] - metrics["no_pretrain_outcome_NLL"]
    )
    metrics["pretraining_treatment_NLL_cost"] = (
        metrics["full_treatment_NLL"] - metrics["no_pretrain_treatment_NLL"]
    )
    metrics["contextual_vicreg_redundancy_gap"] = (
        metrics["vicreg_redundancy"] - metrics["full_redundancy"]
    )
    metrics["contextual_vicreg_alignment_gap"] = (
        metrics["full_diagonal_alignment"] - metrics["vicreg_diagonal_alignment"]
    )
    metrics["contextual_vicreg_outcome_NLL_cost"] = (
        metrics["full_outcome_NLL"] - metrics["vicreg_outcome_NLL"]
    )
    return metrics


def _transferred(name: str) -> bool:
    """The tensors card §6.2 holds identical across all four arms."""
    return _ENCODER in name or any(head in name for head in _HEADS)


def _adopt(recipe: Recipe, initial: Mapping[str, Tensor]) -> None:
    """Copy the shared encoder and head tensors into the contextual arm."""
    system = recipe.system.state_dict()
    adopted = 0
    for name, value in initial.items():
        if not _transferred(name):
            continue
        if name not in system:
            raise RuntimeError(
                f"the contextual vicreg arm has no tensor {name!r} to adopt; "
                "card §6.2 gives it the other arms' encoder and heads"
            )
        system[name].copy_(value)
        adopted += 1
    if adopted == 0:
        raise RuntimeError("the contextual vicreg arm adopted no shared tensor")


def _require_same_start(
    state: Mapping[str, Tensor], initial: Mapping[str, Tensor], *, arm: str
) -> None:
    """Card §6.2: match initial tensors, not merely construction seeds."""
    compared = [name for name in state if arm != "vicreg" or _transferred(name)]
    if not any(_transferred(name) for name in compared):
        raise RuntimeError(f"barlow_twins arm {arm!r} shares no tensor with 'full'")
    differing = sorted(
        name for name in compared if not torch.equal(state[name], initial[name])
    )
    if differing:
        raise RuntimeError(
            f"barlow_twins arm {arm!r} does not start where 'full' does: "
            f"{differing[:4]!r}"
        )


def _arm(recipe: Recipe, arm: str) -> Recipe:
    """The card's four arms, each one edit away from a reviewed recipe.

    `diagonal_only` sets `lambda` to exactly zero and changes nothing else, so
    the objective is still declared, still draws its views and still reports
    its diagnostics — which is what lets pretraining step 0 be compared across
    the pair. `no_pretrain` drops the stage *and* the inheritance edge, because
    `initialise_from` without its source is a program that cannot run. The
    contextual arm is `vicreg` unedited, at its own reviewed (25, 25, 1).
    """
    if arm == "vicreg":
        return recipe
    pretrain, fit = recipe.program
    if arm == "no_pretrain":
        return replace(recipe, program=Program((replace(fit, initialise_from=None),)))
    if arm == "diagonal_only":
        declared = [term.objective.name for term in pretrain.objectives]
        if declared.count(_OFF_DIAGONAL) != 1:
            raise RuntimeError(
                f"barlow_twins pretraining declares {declared!r}; the ablation "
                f"needs exactly one {_OFF_DIAGONAL!r} term to zero"
            )
        # Card §4 schedules this term at "constant 0.0051", so both endpoints
        # of the stage are checked: zeroing a weight that was never the
        # reviewed one ablates something other than eq. (1)'s `lambda`.
        steps = (0, pretrain.steps - 1)
        if any(
            term.weight_at(step) != OFF_DIAGONAL_WEIGHT
            for term in pretrain.objectives
            if term.objective.name == _OFF_DIAGONAL
            for step in steps
        ):
            raise RuntimeError(
                f"the full arm's {_OFF_DIAGONAL!r} weight is not card §4's "
                f"constant {OFF_DIAGONAL_WEIGHT!r}, so zeroing it ablates "
                "something else"
            )
        pretrain = replace(
            pretrain,
            objectives=tuple(
                replace(term, weight=0.0)
                if term.objective.name == _OFF_DIAGONAL
                else term
                for term in pretrain.objectives
            ),
        )
        if any(
            term.weight_at(step) != 0.0
            for term in pretrain.objectives
            if term.objective.name == _OFF_DIAGONAL
            for step in steps
        ):  # pragma: no cover - `Weighted` coerces a number to `Constant`
            raise RuntimeError("the ablated arm still charges an off-diagonal weight")
    return replace(recipe, program=Program((pretrain, fit)))


# ---------------------------------------------------------------------------
# Card §6.2's pairing contract
# ---------------------------------------------------------------------------


def _require_one_pretraining_stream(traces: Mapping[str, dict[str, str]]) -> None:
    """The executor's own row ids and view tensors, hashed over pretraining.

    The strongest of the four checks and the only one that covers the
    contextual arm, whose objectives are named differently and so cannot be
    compared term by term at step 0.
    """
    expected = str(_BRANCHES * BARLOW_TWINS_PRETRAIN_STEPS)
    for arm in _PRETRAINED:
        if traces[arm] != traces["full"]:
            raise RuntimeError(
                f"barlow_twins arm {arm!r} and 'full' fed different rows or "
                "view draws to pretraining"
            )
    if traces["full"]["calls"] != expected:
        raise RuntimeError(
            f"pretraining drew {traces['full']['calls']} views where card §4's "
            f"{BARLOW_TWINS_PRETRAIN_STEPS} steps of two branches need {expected}"
        )
    if traces["no_pretrain"]["calls"] != "0":
        raise RuntimeError(
            "the no-pretraining arm drew a view, so it still has a pretraining "
            "stage and prices nothing"
        )


def _require_one_stream(
    runs: Mapping[str, CompiledRun],
    results: Mapping[str, ProgramResult],
    data: Dataset,
    *,
    base: int,
) -> None:
    """Card §6.2: the arms must share rows and view draws, not merely seeds."""
    reference = results["full"]
    for stage in ("pretrain", "joint_fit"):
        arms = _PRETRAINED if stage == "pretrain" else _ARMS
        expected = reference.stage(stage).seed
        for arm in arms:
            ran_at = results[arm].stage(stage).seed
            if ran_at != expected:
                raise RuntimeError(
                    f"barlow_twins arm {arm!r} ran {stage!r} at seed {ran_at} "
                    f"where 'full' ran it at {expected}; the paired arms would "
                    "be walking different stochastic streams"
                )

    # At pretraining step 0 the paired arms hold identical parameters, so equal
    # unweighted objective values and equal diagnostics are a statement about
    # the rows and both view draws the executor fed the loss.
    first = {
        term.name: (term.value, dict(term.diagnostics))
        for term in reference.stage("pretrain").records[0].terms
    }
    if sorted(first) != sorted(_OBJECTIVES):
        raise RuntimeError(
            f"barlow_twins pretraining logged {sorted(first)!r}; the card "
            f"declares {sorted(_OBJECTIVES)!r}"
        )
    for arm in _PAIRED:
        for term in results[arm].stage("pretrain").records[0].terms:
            value, diagnostics = first[term.name]
            if term.value != value or dict(term.diagnostics) != diagnostics:
                raise RuntimeError(
                    f"barlow_twins arm {arm!r} and 'full' disagree at "
                    f"pretraining step 0 on {term.name!r}: the two arms did not "
                    "see the same rows under the same two view draws"
                )

    for stage in ("pretrain", "joint_fit"):
        arms = _PRETRAINED if stage == "pretrain" else _ARMS
        streams = {
            arm: _declared_stream(
                runs[arm],
                data,
                stage=stage,
                seed=results[arm].stage(stage).seed,
                draw_views=stage == "pretrain",
            )
            for arm in arms
        }
        rows, views, mask = streams["full"]
        if int(mask.sum()) != BARLOW_TWINS_OBSERVED_TREATMENTS:
            raise RuntimeError(
                f"card §6.2 observes {BARLOW_TWINS_OBSERVED_TREATMENTS} "
                f"treatments, the {stage!r} population observes {int(mask.sum())}"
            )
        for arm in arms:
            other_rows, other_views, other_mask = streams[arm]
            if not _equal_sequences(rows, other_rows):
                raise RuntimeError(
                    f"barlow_twins arm {arm!r} draws different {stage!r} row "
                    "ids than 'full' does"
                )
            if not _equal_sequences(views, other_views):
                raise RuntimeError(
                    f"barlow_twins arm {arm!r} draws different {stage!r} views "
                    "than 'full' does"
                )
            if not torch.equal(mask, other_mask):
                raise RuntimeError(
                    f"barlow_twins arm {arm!r} fits {stage!r} under a different "
                    "treatment mask than 'full' does"
                )


def _declared_stream(
    run: CompiledRun,
    data: Dataset,
    *,
    stage: str,
    seed: int,
    draw_views: bool,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    """Materialise what this arm's own declarations produce at `seed`.

    Through the loader entry points the executor itself calls, so this is the
    row stream and the view draws the arm receives rather than a second
    implementation of them. `rng_key = seed + step` is the executor's own view
    key (`training/executors.py:_run_stage`).
    """
    compiled = next(item for item in run.stages if item.name == stage)
    spec = run.recipe.data
    if not isinstance(spec, DataSpec):  # pragma: no cover - compile() rejects this
        raise RuntimeError(f"barlow_twins stage {stage!r} has no data policy")
    population = build_population(data, spec, seed=seed)
    rows: list[Tensor] = []
    views: list[Tensor] = []
    for step, batch in enumerate(
        iterate(population, compiled.stage.sampler, steps=compiled.steps, seed=seed)
    ):
        rows.append(batch.row_id.clone())
        if not draw_views:
            continue
        for view in run.recipe.views:
            views.append(
                view.apply(
                    batch,
                    run.recipe.schema,
                    rng_key=seed + step,
                    population=population,
                ).x.clone()
            )
    return rows, views, population.rows.t_observed.clone()


def _require_pretraining_touched_only_what_it_declares(
    runs: Mapping[str, CompiledRun],
    results: Mapping[str, ProgramResult],
    initial: Mapping[str, Tensor],
) -> None:
    """Card §3.2: heads stay initial, the projector never fine-tunes.

    `run_program` restores the recipe's initial state before each stage and
    then overlays the named checkpoint, so a checkpoint holding no head tensor
    is what makes "the heads are identical before fine-tuning" true. The
    converse — the projector is in no fitting-stage forward pass and in no
    trainable set — is read off the live graph after the program: whatever
    pretraining left there is still there.
    """
    for arm in _PRETRAINED:
        projector = _PROJECTORS.get(arm, _DEFAULT_PROJECTOR)
        checkpoint = results[arm].stage("pretrain").checkpoint
        saved = {**checkpoint.parameters, **checkpoint.buffers}
        stray = sorted(name for name in saved if any(head in name for head in _HEADS))
        if stray:
            raise RuntimeError(
                f"barlow_twins arm {arm!r} pretrained the downstream heads "
                f"{stray!r}; card §3.2 trains only the encoder and the projector"
            )
        if not any(_ENCODER in name for name in checkpoint.parameters):
            raise RuntimeError(
                f"barlow_twins arm {arm!r} pretrained no encoder parameter, so "
                "there is nothing for the fitting stage to inherit"
            )
        if not any(projector in name for name in checkpoint.parameters):
            raise RuntimeError(
                f"barlow_twins arm {arm!r} pretrained no {projector!r} "
                "parameter, so the embedding it is measured on has no head"
            )
        if all(
            torch.equal(value, initial["_components." + name])
            for name, value in checkpoint.parameters.items()
            if _ENCODER in name
        ):
            raise RuntimeError(
                f"barlow_twins arm {arm!r} left the encoder at its "
                "initialisation; the transfer under study would be the identity"
            )
        final = runs[arm].graph.state_dict()
        for name, value in saved.items():
            if projector in name and not torch.equal(
                value, final["_components." + name]
            ):
                raise RuntimeError(
                    f"barlow_twins arm {arm!r} changed projector tensor "
                    f"{name!r} during fine-tuning; card §3.2 never executes the "
                    "projector downstream"
                )
    # `lambda` has to reach the optimiser. The pair shares every stream and
    # every initial tensor, so a bit-identical pretrained encoder would mean
    # the zeroed weight changed nothing about the fit.
    pretrained = {
        arm: {
            name: value
            for name, value in results[arm]
            .stage("pretrain")
            .checkpoint.parameters.items()
            if _ENCODER in name
        }
        for arm in _PAIRED
    }
    if all(
        torch.equal(value, pretrained["diagonal_only"][name])
        for name, value in pretrained["full"].items()
    ):
        raise RuntimeError(
            "the full and diagonal-only arms pretrained bit-identical encoders, "
            "so card §6.2's ablation changed nothing the optimiser saw"
        )


def _equal_sequences(left: Sequence[Tensor], right: Sequence[Tensor]) -> bool:
    return len(left) == len(right) and all(
        torch.equal(one, other) for one, other in zip(left, right, strict=True)
    )


# ---------------------------------------------------------------------------
# Card §6.2's held-out evaluation batches and their view contract
# ---------------------------------------------------------------------------


def _fit_population(result: ProgramResult) -> TrainingPopulation:
    population = result.stage("joint_fit").population
    if population is None:
        raise RuntimeError(
            "the barlow_twins fitting stage reported no training population; "
            "the recipe declares a sampler, so one is expected"
        )
    return population


def _held_out_batches(
    test: ClusterPopulation, population: TrainingPopulation
) -> tuple[XTYBatch, ...]:
    """Card §6.2's 16 disjoint held-out batches, on the fitted feature scale."""
    if _EVAL_BATCHES * BARLOW_TWINS_BATCH_SIZE != _TEST_ROWS:
        raise RuntimeError(
            f"card §6.2 evaluates {_EVAL_BATCHES} disjoint batches of "
            f"{BARLOW_TWINS_BATCH_SIZE} rows, which does not partition "
            f"{_TEST_ROWS}"
        )
    scaled = on_the_training_scale(test.batch, population)
    return tuple(
        take(
            scaled,
            torch.arange(
                index * BARLOW_TWINS_BATCH_SIZE, (index + 1) * BARLOW_TWINS_BATCH_SIZE
            ),
        )
        for index in range(_EVAL_BATCHES)
    )


def _held_out_views(
    schema: Schema,
    batches: Sequence[XTYBatch],
    population: TrainingPopulation,
    *,
    base: int,
) -> tuple[tuple[XTYBatch, XTYBatch], ...]:
    """The two view draws per evaluation batch, realised once and shared.

    Card §6.2 reuses these across arms, so they are computed from the shared
    training population — one donor pool and one fitted scaling for every arm,
    which the pairing check has already established — and handed to each arm's
    encoder unchanged. The generators are the card's, seeded per batch and per
    branch so the two branches of one batch are independent draws rather than
    one draw scored against itself.
    """
    symmetry = OracleSymmetry()
    pairs: list[tuple[XTYBatch, XTYBatch]] = []
    for index, batch in enumerate(batches):
        drawn = tuple(
            symmetry.apply(
                batch,
                schema,
                population=population,
                generator=torch.Generator().manual_seed(
                    base + _VIEW_SEED_BASE + _BRANCHES * index + branch
                ),
            )
            for branch in range(_BRANCHES)
        )
        pairs.append((drawn[0], drawn[1]))
    return tuple(pairs)


def _require_view_contract(
    train: ClusterPopulation,
    test: ClusterPopulation,
    population: TrainingPopulation,
    batches: Sequence[XTYBatch],
    views: Sequence[tuple[XTYBatch, XTYBatch]],
) -> dict[str, float]:
    """Card §6.2's five checks on the held-out views, before any metric.

    Maximum target error, nonidentity, independent branch draws, unchanged row
    metadata, and training-only donor and scaler provenance. Each is a stop
    rather than a number, because a view that moves a DGP target or that leaves
    the rows alone makes §6.4's alignment a statement about something else.
    """
    donor = population.rows.row_id
    if not torch.equal(donor, train.batch.row_id):
        raise RuntimeError(
            "the held-out views were drawn from a population that is not the "
            "fitted training rows; card §6.2 requires a training-only donor"
        )
    held_out = set(test.batch.row_id.tolist())
    if held_out & set(donor.tolist()):
        raise RuntimeError("a held-out row entered the view donor pool")
    for key, expected in {
        "x_location": train.batch.x.mean(0),
        "x_scale": train.batch.x.std(0, correction=0),
        "y_location": train.batch.y.mean(0),
        "y_scale": train.batch.y.std(0, correction=0),
    }.items():
        if not torch.allclose(population.statistics[key], expected):
            raise RuntimeError(
                f"the fitted {key!r} is not the training-only statistic card "
                "§6.2 requires"
            )

    location = population.statistics["x_location"]
    scale = population.statistics["x_scale"]
    errors: list[float] = []
    changed: list[float] = []
    distinct: list[float] = []
    for batch, pair in zip(batches, views, strict=True):
        clean = targets(batch.x * scale + location)
        for view in pair:
            error = float((targets(view.x * scale + location) - clean).abs().max())
            if error > PRESERVATION_TOLERANCE:
                raise RuntimeError(
                    f"a held-out view moved a DGP target by {error!r}, above "
                    f"card §6.2's {PRESERVATION_TOLERANCE!r}"
                )
            errors.append(error)
            changed.append(float((view.x != batch.x).double().sum(-1).mean()))
            for field in sorted(PRESERVED_FIELDS):
                kept, seen = getattr(batch, field), getattr(view, field)
                if kept is None or seen is None:
                    if kept is not seen:
                        raise RuntimeError(f"a held-out view dropped {field!r}")
                    continue
                if not torch.equal(kept, seen):
                    raise RuntimeError(
                        f"a held-out view changed the preserved field {field!r}"
                    )
        moved = float((pair[0].x != batch.x).any(-1).double().mean())
        if moved < 1.0:
            raise RuntimeError(
                f"only {moved!r} of one held-out batch's rows were transformed; "
                "an identity view scores alignment for free"
            )
        if torch.equal(pair[0].x, pair[1].x):
            raise RuntimeError(
                "a held-out batch's two branches are the same tensor, so `C` "
                "would be a view's correlation with itself"
            )
        distinct.append(float((pair[0].x != pair[1].x).any(-1).double().mean()))
    # Averaged over the 16 batches rather than taken at the worst one: two
    # independent draws agree on a row whenever they draw the same donor under
    # the same two symmetry bits, which happens for about one row in four
    # thousand, and a bound on a single batch would be a coin flip on that tail
    # rather than a statement about independence.
    shared = math.fsum(distinct) / len(distinct)
    if shared < 0.99:
        raise RuntimeError(
            f"the two branches of the held-out batches agreed on {1 - shared!r} "
            "of their rows; card §6.2 requires independent branch draws"
        )
    return {
        "view_max_target_error": max(errors),
        "view_distinct_row_fraction": shared,
        "view_changed_coordinates": math.fsum(changed) / len(changed),
    }


# ---------------------------------------------------------------------------
# Card §6.4's metrics
# ---------------------------------------------------------------------------


def _downstream(
    run: CompiledRun, result: ProgramResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out fit on the clean rows, on the scale the run itself fitted.

    Card §6.4 asks for the factual outcome NLL on the shared
    training-standardised outcome scale, so no Jacobian is applied: the number
    is `-log p(z | x, t)` in the standardised coordinate, and the four arms
    share one scale because they share one training population. The treatment
    NLL, the ATE error and the conditional-mean effect RMSE are informational.
    """
    population = _fit_population(result)
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    run.graph.eval()
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
            raise TypeError("barlow_twins benchmark expected its declared heads")
        probabilities = propensity.log_probs.exp()
        if not bool(torch.isfinite(probabilities).all()) or not bool(
            torch.allclose(
                probabilities.sum(-1), torch.ones(scaled.batch_size), atol=1e-5
            )
        ):
            raise RuntimeError(
                "barlow_twins propensity did not return a class distribution"
            )
        means = candidate_treatment_means(
            outcome,
            batch_size=scaled.batch_size,
            num_treatments=schema.treatment_cardinality,
            device=scaled.t.device,
        )
        # The heads predict the standardised outcome, so the effect returns to
        # the DGP's own units through the scale the run fitted on `train`, and
        # the error is against the conditional-mean truth rather than against a
        # realised noisy outcome difference (card §6.4).
        effect = treatment_contrast(means) * population.statistics["y_scale"]
        return {
            "outcome_NLL": float(-outcome.log_prob(scaled.y, scaled.t).mean()),
            "treatment_NLL": float(treatment_nll(propensity, scaled.t)),
            "ATE_error": absolute_ate_error(
                average_treatment_effect(effect), float(test.true_effect.mean())
            ),
            "treatment_effect_RMSE": float(sqrt_pehe(effect, test.true_effect)),
        }


def _embeddings(
    run: CompiledRun,
    result: ProgramResult,
    schema: Schema,
    views: Sequence[tuple[XTYBatch, XTYBatch]],
    *,
    arm: str,
) -> dict[str, float]:
    """Card §6.4's per-seed embedding statistics at the terminal checkpoint.

    The checkpoint is restored into the live graph *after* the downstream
    numbers are taken, so the fitting stage's encoder cannot leak into a
    measurement the card defines "before fine-tuning". `eval()` plus the
    checkpoint's own buffers is what "hidden BN in eval mode on frozen training
    buffers" means: the projector's normalisation uses the running statistics
    pretraining left, not statistics of the evaluation batch.

    One `C` per held-out batch rather than one per branch: §6.4's `a`, `r` and
    `f` are reductions of the cross-view matrix, so the two branches enter
    together. `cross_correlation` is imported rather than rewritten, as §6.2
    asks — a second transcription is how the diagnostic stops being about the
    loss it is meant to describe.
    """
    projector = _PROJECTORS.get(arm, _DEFAULT_PROJECTOR)
    checkpoint = result.stage("pretrain").checkpoint
    saved = {**checkpoint.parameters, **checkpoint.buffers}
    state = run.graph.state_dict()
    for name, value in saved.items():
        state["_components." + name].copy_(value)
    # The restore has to have happened: fine-tuning moved the encoder, so a
    # graph that still matches the fitting stage's tensors would be measuring
    # the transferred representation the card evaluates "before fine-tuning".
    restored = run.graph.state_dict()
    if any(
        not torch.equal(restored["_components." + name], value)
        for name, value in saved.items()
    ):
        raise RuntimeError(
            f"barlow_twins arm {arm!r} did not restore its terminal "
            "pretraining checkpoint before card §6.4's diagnostics"
        )
    run.graph.eval()
    alignment: list[float] = []
    redundancy: list[float] = []
    active: list[float] = []
    diagonal_error: list[float] = []
    variances: tuple[list[float], list[float]] = ([], [])
    concentration: list[float] = []
    components = ("mlp_encoder", projector.rstrip("."))
    with torch.no_grad():
        for pair in views:
            embeddings: list[Tensor] = []
            live: list[Tensor] = []
            for branch, view in enumerate(pair):
                embedded = run.graph.evaluate(view, schema=schema, only=components)[
                    Port.X_PROJ
                ]
                if not isinstance(embedded, Tensor):
                    raise TypeError(
                        f"barlow_twins arm {arm!r} did not return an embedding"
                    )
                embeddings.append(embedded)
                # The *raw* variance, before the loss's normalisation: §6.4's
                # activity cutoff is about the embedding, not about `C`.
                variance = embedded.var(dim=0, correction=POPULATION_CORRECTION)
                variances[branch].append(float(variance.mean()))
                live.append(variance > _ACTIVE_VARIANCE)
            first, second = embeddings
            width = first.shape[1]
            correlation = cross_correlation(
                first,
                second,
                epsilon=NORMALISATION_EPSILON,
                correction=POPULATION_CORRECTION,
            )
            diagonal = correlation.diagonal()
            off = correlation.masked_select(
                ~torch.eye(width, dtype=torch.bool)
            ).square()
            # Card §6.4 averages `C_ij^2` over the `d * (d - 1)` **ordered**
            # off-diagonal pairs, so the mask has to select exactly that many
            # entries: a mask that keeps the diagonal, or one triangle of it,
            # divides a different sum by the same denominator.
            if off.numel() != width * (width - 1) or diagonal.numel() != width:
                raise RuntimeError(
                    f"the off-diagonal mask selected {off.numel()} of "
                    f"{width * (width - 1)} ordered pairs"
                )
            alignment.append(float(diagonal.mean()))
            redundancy.append(float(off.sum()) / (width * (width - 1)))
            diagonal_error.append(float((1.0 - diagonal).square().sum()) / width)
            active.append(float((live[0] & live[1]).double().mean()))
            centred = first - first.mean(dim=0)
            covariance = centred.T @ centred / first.shape[0]
            eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp(min=0.0)
            total = float(eigenvalues.sum())
            # No arm here is a collapse control, so a null covariance is a
            # failure to report rather than a share to invent.
            if not total > 0.0:
                raise RuntimeError(
                    f"barlow_twins arm {arm!r} produced a null embedding "
                    "covariance, so its eigenvalue share has no denominator"
                )
            concentration.append(float(eigenvalues.max()) / total)
    return {
        "diagonal_alignment": _mean(alignment),
        "redundancy": _mean(redundancy),
        "active_fraction": _mean(active),
        "diagonal_error": _mean(diagonal_error),
        "raw_variance_first": _mean(variances[0]),
        "raw_variance_second": _mean(variances[1]),
        "covariance_top_eigenvalue_share": _mean(concentration),
        "encoder_parameter_norm": _parameter_norm(checkpoint.parameters, _ENCODER),
        "projector_parameter_norm": _parameter_norm(checkpoint.parameters, projector),
    }


def _parameter_norm(parameters: Mapping[str, Tensor], prefix: str) -> float:
    """The Euclidean norm of one component's saved parameters."""
    total = math.fsum(
        float(value.double().square().sum())
        for name, value in parameters.items()
        if prefix in name
    )
    if not total > 0.0:
        raise RuntimeError(f"no pretrained parameter matched {prefix!r}")
    return math.sqrt(total)


def _mean(values: Sequence[float]) -> float:
    """Card §6.4's "average each over 16 batches to obtain one value per seed"."""
    if len(values) != _EVAL_BATCHES:
        raise RuntimeError(
            f"expected {_EVAL_BATCHES} held-out observations, got {len(values)}"
        )
    return math.fsum(values) / len(values)
