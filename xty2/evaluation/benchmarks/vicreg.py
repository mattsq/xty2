"""VICReg's four-arm paired mechanism study from card section 6.

The measurement is the pairing. Three of the four arms are the same recipe with
one of equation (6)'s coefficients set to exactly zero, and the fourth removes
pretraining altogether; everything else — components, initial tensors, data
policy, row stream, view draws, step counts and downstream objectives — is held
fixed, so a difference between two arms is attributable to the coefficient that
differs. Card section 6.4 turns that into four one-standard-error targets: the
full arm's embedding spread, the spread the variance term is responsible for,
the redundant covariance energy the covariance term removes, and a budget on
what transferring the encoder costs the factual outcome fit.

**What this module measures and what it does not.** Nothing here is a claim
about Bardes et al.'s ImageNet numbers. Card section 5 records five judgement
departures — the author-code reductions, a tabular encoder and a 512-wide
expander, feature corruption in place of image transformations, Adam on a fixed
1,000/3,000-step budget in place of LARS on an epoch schedule, and a local XTY
fitting stack — and the section 6 protocol exists to test the *mechanism* under
those departures on this fixture.

**How the pairing is checked.** Card section 6.2 asks for equality of the
actual sampled row ids and view draws rather than of the seeds that produced
them, and it is checked three ways, from the strongest evidence down:

* Every logged objective value and diagnostic at pretraining step 0 must agree
  bit for bit across the three pretraining arms. The three arms hold identical
  parameters at that step, so an identical `embedding_invariance` value is a
  statement about the rows and *both* view draws the executor actually fed the
  loss — observed from inside the run, not re-derived beside it.
* The stage seeds the executor reports must agree, which is what says the
  no-pretraining arm's single stage really did land on the stream the paired
  arms' second stage used, `STREAM_STRIDE` and all.
* The row-id stream and both corrupted view tensors are then materialised
  through the loader entry points the executor itself calls — `build_population`
  and `iterate` for the rows, `ViewSpec.apply` for the views — and compared
  element by element across arms. This catches a difference in *what each arm
  declares* (a sampler, a corruption rate, a missingness mask) that equal seeds
  would hide.

Two properties need a hook inside the step loop and are pinned by Tier 1 on
every declared seed rather than re-proved here: that the fitting stage builds a
fresh optimiser with empty state, and that the heads are untouched at the
transition. Tier 2 observes their consequences instead — a pretraining
checkpoint holding no head parameter, and an expander that fine-tuning leaves
bit-identical.
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
    CLUSTER_SIGNAL,
    SEPARATED,
    SIGNAL_COLUMNS,
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
    treatment_contrast,
)
from xty2.evaluation.predictive import treatment_nll
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec
from xty2.recipes import vicreg
from xty2.recipes.vicreg import (
    SAMPLE_CORRECTION,
    VARIANCE_EPSILON,
    VICREG_BATCH_SIZE,
    VICREG_CORRUPTION_RATE,
)
from xty2.training import STREAM_STRIDE, ProgramResult, run_program
from xty2.training.loading import build_population, iterate
from xty2.views import FeatureCorruption

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 190_000
"""Card §6.2: replicate `i` runs at `base = 190000 + 100 * i`."""

_EVAL_BATCHES = 16
"""Card §6.2: 16 disjoint held-out batches of `VICREG_BATCH_SIZE` rows."""

_BRANCHES = 2
_VIEW_SEED_BASE = 20_000
"""Card §6.2: the two corruption generators for batch `b` are seeded
`base + 20000 + 2 * b` and `base + 20001 + 2 * b`."""

_ARMS = ("full", "no_variance", "no_covariance", "no_pretrain")
_PRETRAINED = ("full", "no_variance", "no_covariance")
_COLLAPSED_STD = 0.1
"""Card §6.4's "fraction of dimensions with standard deviation below 0.1"."""

_NOISE_SIGMA = 0.6
"""`fixmatch.md` §6.1's within-cluster feature noise, as `common.py` writes it
in `cluster_population` (`0.6 * epsilon_x`)."""

_LOG_ODDS_SLOPE = 2.5
"""Card §6.4's `p(c=1 | x) = sigmoid(2.5 * sum_{i<4} x_i)`, bound by value.

The card derives it as `2 * 0.45 / 0.6^2` from the fixture's centres and noise;
binding the card's number and checking it against the fixture's own constants is
what makes a change to either one an error here rather than a silent drift.
"""

_TREATED_FLOOR = SEPARATED
_TREATED_RANGE = 1.0 - 2.0 * SEPARATED
"""Card §6.4's `p(t=1 | x) = 0.02 + 0.96 * p(c=1 | x)`, from §6.1's
`p(t=1 | c) = 0.02 + 0.96c` at the fixture's `low = SEPARATED`."""

_OBJECTIVES = ("embedding_invariance", "embedding_variance", "embedding_covariance")
_HEADS = ("tarnet_head.", "categorical_propensity.")
_ENCODER = "mlp_encoder."
_EXPANDER = "vicreg_expander."

_ARM_DIAGNOSTICS = (
    "outcome_NLL",
    "treatment_NLL",
    "ATE_error",
)
_EMBEDDING_DIAGNOSTICS = (
    "embedding_spread",
    "embedding_redundancy",
    "collapsed_dimension_fraction",
    "covariance_top_eigenvalue_share",
    "encoder_parameter_norm",
    "expander_parameter_norm",
)


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired four-arm replicates and score card §6.4's four targets."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "shared two_cluster_population DGP, 6 features and K=2; section 6.2"
            ),
            "variant": (
                "full VICReg versus paired zero-variance, zero-covariance and "
                "no-pretraining arms"
            ),
            "split": (
                "1024 train with 40 observed treatments; 2048 fully observed "
                "held-out rows"
            ),
            "metric": (
                "terminal embedding spread, paired variance and redundancy "
                "effects, factual outcome NLL difference; treatment NLL "
                "informational"
            ),
            "published": "none - project-local tabular adaptation",
            "tolerance": "all required one-standard-error bounds in section 6.4",
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(f"vicreg card reviewed ten replicates, got {spec.seed_count}")
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    # The four required targets of card §6.4, in its order. Each is written as
    # the card writes it — `mean - SE >= t` and `mean + SE <= t` — which is
    # exactly `MetricResult.passed`: the mean must satisfy the relation by at
    # least one standard error.
    required = (
        MetricResult.lower_bound(
            "full_arm_embedding_spread",
            column(rows, "full_embedding_spread"),
            0.5,
        ),
        MetricResult.lower_bound(
            "variance_ablation_spread_gap",
            column(rows, "variance_ablation_spread_gap"),
            0.1,
        ),
        MetricResult.lower_bound(
            "covariance_ablation_redundancy_gap",
            column(rows, "covariance_ablation_redundancy_gap"),
            0.01,
        ),
        MetricResult.upper_bound(
            "pretraining_outcome_NLL_cost",
            column(rows, "pretraining_outcome_NLL_cost"),
            0.05,
            unit="nat/row",
        ),
    )
    # Everything else §6.4 asks to be reported. "Report all per-seed values,
    # both arms' absolute values and their paired differences": `MetricResult`
    # keeps the whole replicate vector, so the artifact carries the per-seed
    # numbers rather than only their mean. `full_arm_embedding_spread` is
    # already a target above, and a metric reported twice is a name collision
    # rather than a second reading, so the targets win the name.
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
            for name in (
                "pretraining_treatment_NLL_cost",
                "view_induced_bayes_propensity_shift",
                "signal_free_corruption_bayes_propensity_shift",
            )
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
            "This is a project-local mechanism study, not a reproduction of "
            "Bardes et al. The three pretraining arms differ by exactly one of "
            "equation (6)'s coefficients and the fourth removes pretraining, so "
            "the spread and redundancy gaps attribute an effect to the variance "
            "and covariance terms on this fixture at this budget. The outcome "
            "target is a guardrail on transfer, not evidence of causal "
            "identification: card §2 excludes ImageNet reproduction, "
            "superiority to SCARF and treatment-effect recovery from the claim, "
            "and the treatment NLL, ATE error and view-damage diagnostics are "
            "reported for the audit card §6.4 requires before a downstream "
            "number is read as a statement about the objective."
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
        _TEST_ROWS, seed=base + 2, row_offset=10_000, low=SEPARATED
    )
    data = training_dataset(schema, train.batch)

    runs: dict[str, CompiledRun] = {}
    results: dict[str, ProgramResult] = {}
    initial: dict[str, Tensor] | None = None
    for arm in _ARMS:
        torch.manual_seed(base + 6)
        recipe = _arm(vicreg(schema), arm)
        state = {
            name: value.clone() for name, value in recipe.system.state_dict().items()
        }
        if initial is None:
            initial = state
        elif any(
            not torch.equal(value, initial[name]) for name, value in state.items()
        ):
            raise RuntimeError(f"vicreg arm {arm!r} does not start where 'full' does")
        runs[arm] = compile(recipe)
        results[arm] = run_program(
            runs[arm],
            {stage.name: data for stage in recipe.program},
            # Card §6.2: the ablation's fit is stage 0, so its execution seed
            # carries one stride to put it on the stream the paired arms' stage
            # 1 walks. The assertions below check that it landed there.
            seed=base + 10_000 + (STREAM_STRIDE if arm == "no_pretrain" else 0),
        )
    assert initial is not None
    _require_one_stream(runs, results, data, base=base)
    _require_pretraining_touched_only_what_it_declares(runs, results, initial)

    metrics: dict[str, float] = {}
    for arm in _ARMS:
        metrics.update(
            (f"{arm}_{name}", value)
            for name, value in _downstream(runs[arm], results[arm], test).items()
        )
    population = _fit_population(results["full"])
    views = _held_out_views(schema, test, population, base=base)
    for arm in _PRETRAINED:
        metrics.update(
            (f"{arm}_{name}", value)
            for name, value in _embeddings(
                runs[arm], results[arm], schema, views, arm=arm
            ).items()
        )
    metrics.update(_view_damage(schema, test, population, views, base=base))
    metrics["variance_ablation_spread_gap"] = (
        metrics["full_embedding_spread"] - metrics["no_variance_embedding_spread"]
    )
    metrics["covariance_ablation_redundancy_gap"] = (
        metrics["no_covariance_embedding_redundancy"]
        - metrics["full_embedding_redundancy"]
    )
    metrics["pretraining_outcome_NLL_cost"] = (
        metrics["full_outcome_NLL"] - metrics["no_pretrain_outcome_NLL"]
    )
    metrics["pretraining_treatment_NLL_cost"] = (
        metrics["full_treatment_NLL"] - metrics["no_pretrain_treatment_NLL"]
    )
    return metrics


def _arm(recipe: Recipe, arm: str) -> Recipe:
    """The card's four arms, each one edit away from the reviewed recipe.

    The two ablations set one `Weighted.weight` to exactly zero and change
    nothing else, so the objective still runs, still draws its views and still
    reports its diagnostics — which is what lets pretraining step 0 be compared
    across arms. The fourth drops the pretraining stage *and* the inheritance
    edge, because `initialise_from` without its source is a program that cannot
    run.
    """
    pretrain, fit = recipe.program
    if arm == "no_pretrain":
        return replace(recipe, program=Program((replace(fit, initialise_from=None),)))
    if arm != "full":
        target = "embedding_" + arm.removeprefix("no_")
        pretrain = replace(
            pretrain,
            objectives=tuple(
                replace(term, weight=0.0) if term.objective.name == target else term
                for term in pretrain.objectives
            ),
        )
    return replace(recipe, program=Program((pretrain, fit)))


# ---------------------------------------------------------------------------
# Card §6.2's pairing contract
# ---------------------------------------------------------------------------


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
                    f"vicreg arm {arm!r} ran {stage!r} at seed {ran_at} where "
                    f"'full' ran it at {expected}; the paired arms would be "
                    "walking different stochastic streams"
                )

    # The strongest evidence, and the only one taken from inside the loop: at
    # pretraining step 0 the three arms hold identical parameters, so equal
    # unweighted objective values and equal diagnostics are a statement about
    # the rows and both view draws the executor fed the loss.
    first = {
        term.name: (term.value, dict(term.diagnostics))
        for term in reference.stage("pretrain").records[0].terms
    }
    if sorted(first) != sorted(_OBJECTIVES):
        raise RuntimeError(
            f"vicreg pretraining logged {sorted(first)!r}; the card declares "
            f"{sorted(_OBJECTIVES)!r}"
        )
    for arm in _PRETRAINED:
        for term in results[arm].stage("pretrain").records[0].terms:
            value, diagnostics = first[term.name]
            if term.value != value or dict(term.diagnostics) != diagnostics:
                raise RuntimeError(
                    f"vicreg arm {arm!r} and 'full' disagree at pretraining step "
                    f"0 on {term.name!r}: the two arms did not see the same rows "
                    "under the same two view draws"
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
                base=base,
            )
            for arm in arms
        }
        rows, views, mask = streams["full"]
        for arm in arms:
            other_rows, other_views, other_mask = streams[arm]
            if not _equal_sequences(rows, other_rows):
                raise RuntimeError(
                    f"vicreg arm {arm!r} draws different {stage!r} row ids than "
                    "'full' does"
                )
            if not _equal_sequences(views, other_views):
                raise RuntimeError(
                    f"vicreg arm {arm!r} draws different {stage!r} views than "
                    "'full' does"
                )
            if not torch.equal(mask, other_mask):
                raise RuntimeError(
                    f"vicreg arm {arm!r} fits {stage!r} under a different "
                    "treatment mask than 'full' does"
                )


def _declared_stream(
    run: CompiledRun,
    data: Dataset,
    *,
    stage: str,
    seed: int,
    draw_views: bool,
    base: int,
) -> tuple[list[Tensor], list[Tensor], Tensor]:
    """Materialise what this arm's own declarations produce at `seed`.

    Through the loader entry points `_feed` itself calls, so this is the row
    stream and the view draws the arm receives rather than a second
    implementation of them. `rng_key = seed + step` is the executor's own view
    key (`executors._run_stage`).
    """
    del base
    compiled = next(item for item in run.stages if item.name == stage)
    spec = run.recipe.data
    if not isinstance(spec, DataSpec):  # pragma: no cover - compile() rejects this
        raise RuntimeError(f"vicreg stage {stage!r} has no data policy to apply")
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
    """Card §6.2 and §3.2: heads stay initial, the expander never fine-tunes.

    `run_program` restores the recipe's initial state before each stage and
    then overlays the named checkpoint, so a checkpoint holding no head tensor
    is what makes "the heads are identical before fine-tuning" true. The
    converse — the expander is absent from the fitting stage's forward pass and
    trainable set — is read off the live graph after the program: whatever
    pretraining left there is still there.
    """
    for arm in _PRETRAINED:
        checkpoint = results[arm].stage("pretrain").checkpoint
        saved = {**checkpoint.parameters, **checkpoint.buffers}
        stray = sorted(name for name in saved if name.startswith(_HEADS))
        if stray:
            raise RuntimeError(
                f"vicreg arm {arm!r} pretrained the downstream heads {stray!r}; "
                "card §3.2 trains only the encoder and the expander"
            )
        if not any(name.startswith(_ENCODER) for name in checkpoint.parameters):
            raise RuntimeError(
                f"vicreg arm {arm!r} pretrained no encoder parameter, so there "
                "is nothing for the fitting stage to inherit"
            )
        if all(
            torch.equal(value, initial["_components." + name])
            for name, value in checkpoint.parameters.items()
            if name.startswith(_ENCODER)
        ):
            raise RuntimeError(
                f"vicreg arm {arm!r} left the encoder at its initialisation; "
                "the transfer under study would be the identity"
            )
        final = runs[arm].graph.state_dict()
        for name, value in saved.items():
            if name.startswith(_EXPANDER) and not torch.equal(
                value, final["_components." + name]
            ):
                raise RuntimeError(
                    f"vicreg arm {arm!r} changed expander tensor {name!r} during "
                    "fine-tuning; the paper discards the expander (§3)"
                )


def _equal_sequences(left: Sequence[Tensor], right: Sequence[Tensor]) -> bool:
    return len(left) == len(right) and all(
        torch.equal(one, other) for one, other in zip(left, right, strict=True)
    )


# ---------------------------------------------------------------------------
# Card §6.4's metrics
# ---------------------------------------------------------------------------


def _fit_population(result: ProgramResult) -> TrainingPopulation:
    population = result.stage("joint_fit").population
    if population is None:
        raise RuntimeError(
            "the vicreg fitting stage reported no training population; the "
            "recipe declares a sampler, so one is expected"
        )
    return population


def _downstream(
    run: CompiledRun, result: ProgramResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out fit on the clean rows, on the scale the run itself fitted.

    Card §6.4 asks for the factual outcome NLL "on the same
    training-standardised scale", so no Jacobian is applied: the number is
    `-log p(z | x, t)` in the standardised outcome coordinate, and the four
    arms share one scale because they share one training population. The
    treatment NLL and the ATE error are the card's informational diagnostics
    and carry no target.
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
            raise TypeError("vicreg benchmark expected its declared causal heads")
        probabilities = propensity.log_probs.exp()
        if not bool(torch.isfinite(probabilities).all()) or not bool(
            torch.allclose(
                probabilities.sum(-1), torch.ones(scaled.batch_size), atol=1e-5
            )
        ):
            raise RuntimeError("vicreg propensity did not return a class distribution")
        means = candidate_treatment_means(
            outcome,
            batch_size=scaled.batch_size,
            num_treatments=schema.treatment_cardinality,
            device=scaled.t.device,
        )
        # The heads predict the standardised outcome, so the effect returns to
        # the DGP's own units through the scale the run fitted on `train`.
        effect = treatment_contrast(means) * float(population.statistics["y_scale"])
        return {
            "outcome_NLL": float(-outcome.log_prob(scaled.y, scaled.t).mean()),
            "treatment_NLL": float(treatment_nll(propensity, scaled.t)),
            "ATE_error": absolute_ate_error(
                average_treatment_effect(effect), float(test.true_effect.mean())
            ),
        }


def _held_out_batches(
    test: ClusterPopulation, population: TrainingPopulation
) -> tuple[XTYBatch, ...]:
    """Card §6.2's 16 disjoint held-out batches, on the fitted feature scale."""
    if _EVAL_BATCHES * VICREG_BATCH_SIZE != _TEST_ROWS:
        raise RuntimeError(
            f"card §6.2 evaluates {_EVAL_BATCHES} disjoint batches of "
            f"{VICREG_BATCH_SIZE} rows, which does not partition {_TEST_ROWS}"
        )
    scaled = on_the_training_scale(test.batch, population)
    return tuple(
        take(
            scaled,
            torch.arange(index * VICREG_BATCH_SIZE, (index + 1) * VICREG_BATCH_SIZE),
        )
        for index in range(_EVAL_BATCHES)
    )


def _held_out_views(
    schema: Schema,
    test: ClusterPopulation,
    population: TrainingPopulation,
    *,
    base: int,
) -> tuple[XTYBatch, ...]:
    """The two corruption draws per evaluation batch, realised once.

    Card §6.2 says to reuse these realised views across arms, so they are
    computed here from the shared training population — the donor pool and the
    fitted scaling are the same tensors for every arm, which the pairing check
    has already established — and handed to each arm's encoder unchanged. The
    generators are the card's, seeded per batch and per branch so that the two
    branches of one batch are independent draws rather than one draw twice.
    """
    corruption = FeatureCorruption(rate=VICREG_CORRUPTION_RATE, columns=None)
    views: list[XTYBatch] = []
    for index, batch in enumerate(_held_out_batches(test, population)):
        for branch in range(_BRANCHES):
            generator = torch.Generator().manual_seed(
                base + _VIEW_SEED_BASE + _BRANCHES * index + branch
            )
            views.append(
                corruption.apply(
                    batch, schema, population=population, generator=generator
                )
            )
    return tuple(views)


def _embeddings(
    run: CompiledRun,
    result: ProgramResult,
    schema: Schema,
    views: Sequence[XTYBatch],
    *,
    arm: str,
) -> dict[str, float]:
    """Card §6.4's embedding statistics at the terminal pretraining checkpoint.

    The checkpoint is restored into the live graph *after* the downstream
    numbers are taken, so the fitting stage's encoder cannot leak into a
    measurement the card defines "before fine-tuning". `eval()` plus the
    checkpoint's own buffers is what "frozen training BN buffers" means: the
    expander's normalisation uses the running statistics pretraining left, not
    statistics of the evaluation batch.
    """
    checkpoint = result.stage("pretrain").checkpoint
    saved = {**checkpoint.parameters, **checkpoint.buffers}
    state = run.graph.state_dict()
    for name, value in saved.items():
        state["_components." + name].copy_(value)
    run.graph.eval()
    spread: list[float] = []
    redundancy: list[float] = []
    collapsed: list[float] = []
    concentration: list[float] = []
    with torch.no_grad():
        for view in views:
            embedding = run.graph.evaluate(
                view, schema=schema, only=("mlp_encoder", "vicreg_expander")
            )[Port.X_PROJ]
            if not isinstance(embedding, Tensor):
                raise TypeError("vicreg expander did not return an embedding tensor")
            deviation = (
                embedding.var(dim=0, correction=SAMPLE_CORRECTION) + VARIANCE_EPSILON
            ).sqrt()
            centred = embedding - embedding.mean(dim=0)
            covariance = centred.T @ centred / (embedding.shape[0] - SAMPLE_CORRECTION)
            diagonal = float(covariance.diagonal().square().sum())
            if not diagonal > 0.0:
                # Card §6.4: "reject a zero denominator ... rather than
                # awarding a collapsed embedding perfect decorrelation". It is
                # required of the full and no-covariance arms and applied to
                # all three, because a ratio with no denominator is not a
                # number the no-variance arm's row should carry either.
                raise RuntimeError(
                    f"vicreg arm {arm!r} produced an embedding with zero "
                    "diagonal covariance energy, so §6.4's redundancy ratio has "
                    "no denominator"
                )
            off_diagonal = float(
                covariance.square()
                .masked_select(~torch.eye(embedding.shape[1], dtype=torch.bool))
                .sum()
            )
            eigenvalues = torch.linalg.eigvalsh(covariance.double()).clamp(min=0.0)
            total = float(eigenvalues.sum())
            if not total > 0.0:  # pragma: no cover - implied by the diagonal check
                raise RuntimeError(f"vicreg arm {arm!r} produced a null covariance")
            spread.append(float(deviation.mean()))
            redundancy.append(off_diagonal / diagonal)
            collapsed.append(float((deviation < _COLLAPSED_STD).double().mean()))
            concentration.append(float(eigenvalues.max()) / total)
    return {
        "embedding_spread": _mean(spread),
        "embedding_redundancy": _mean(redundancy),
        "collapsed_dimension_fraction": _mean(collapsed),
        "covariance_top_eigenvalue_share": _mean(concentration),
        "encoder_parameter_norm": _parameter_norm(checkpoint.parameters, _ENCODER),
        "expander_parameter_norm": _parameter_norm(checkpoint.parameters, _EXPANDER),
    }


def _parameter_norm(parameters: Mapping[str, Tensor], prefix: str) -> float:
    """The Euclidean norm of one component's saved parameters."""
    total = math.fsum(
        float(value.double().square().sum())
        for name, value in parameters.items()
        if name.startswith(prefix)
    )
    return math.sqrt(total)


def _mean(values: Sequence[float]) -> float:
    """The card's "average ... over the two views and then the 16 batches"."""
    if len(values) != _EVAL_BATCHES * _BRANCHES:
        raise RuntimeError(
            f"expected {_EVAL_BATCHES * _BRANCHES} view observations, got {len(values)}"
        )
    return math.fsum(values) / len(values)


# ---------------------------------------------------------------------------
# Card §6.4's view-damage diagnostic
# ---------------------------------------------------------------------------


def _bayes_propensity(x: Tensor) -> Tensor:
    """`p(t=1 | x)` under card §6.4's closed-form mixture posterior.

    On the *original* feature scale and never from the fitted classifier or a
    hard cluster assignment: a feature-wise corrupted row draws each cell from
    an independent training-population donor, so it has no single latent `c`
    and only the mixture posterior is defined for it.
    """
    logits = _LOG_ODDS_SLOPE * x[:, :SIGNAL_COLUMNS].sum(dim=-1)
    return _TREATED_FLOOR + _TREATED_RANGE * torch.sigmoid(logits)


def _original_scale(x: Tensor, population: TrainingPopulation) -> Tensor:
    """Undo the run's fitted z-scoring, which the views were drawn under."""
    return x * population.statistics["x_scale"] + population.statistics["x_location"]


def _view_damage(
    schema: Schema,
    test: ClusterPopulation,
    population: TrainingPopulation,
    views: Sequence[XTYBatch],
    *,
    base: int,
) -> dict[str, float]:
    """How much of the cluster signal the study's corruption destroys.

    Reported before any downstream number is interpreted, as card §6.4 asks,
    together with its own control: columns 4 and 5 carry no cluster signal and
    drop out of the log-odds, so a corruption confined to them must move this
    diagnostic by *exactly* zero. A nonzero control would mean the number is
    measuring the corruption's noise rather than its damage, and the run stops
    instead of reporting it.
    """
    derived = 2.0 * CLUSTER_SIGNAL / _NOISE_SIGMA**2
    if not math.isclose(_LOG_ODDS_SLOPE, derived, rel_tol=1e-12):
        raise RuntimeError(
            f"card §6.4 writes the log-odds slope as {_LOG_ODDS_SLOPE}, but the "
            f"fixture's centres and noise give {derived}"
        )
    signal_free = tuple(
        spec.name for spec in schema.features[SIGNAL_COLUMNS:] if spec.mutable
    )
    if not signal_free:
        raise RuntimeError("the fixture has no signal-free column to control with")
    control = FeatureCorruption(rate=1.0, columns=signal_free)
    shift: list[float] = []
    control_shift: list[float] = []
    for index, batch in enumerate(_held_out_batches(test, population)):
        clean = _bayes_propensity(_original_scale(batch.x, population))
        for branch in range(_BRANCHES):
            view = views[_BRANCHES * index + branch]
            corrupted = _bayes_propensity(_original_scale(view.x, population))
            shift.append(float((corrupted - clean).abs().mean()))
            generator = torch.Generator().manual_seed(
                base + _VIEW_SEED_BASE + _BRANCHES * index + branch
            )
            unaffected = _bayes_propensity(
                _original_scale(
                    control.apply(
                        batch, schema, population=population, generator=generator
                    ).x,
                    population,
                )
            )
            control_shift.append(float((unaffected - clean).abs().max()))
    if max(control_shift) != 0.0:
        raise RuntimeError(
            "corrupting only the signal-free columns moved card §6.4's Bayes "
            f"propensity by {max(control_shift)!r}; the diagnostic is not "
            "measuring cluster damage"
        )
    return {
        "view_induced_bayes_propensity_shift": _mean(shift),
        "signal_free_corruption_bayes_propensity_shift": _mean(control_shift),
    }
