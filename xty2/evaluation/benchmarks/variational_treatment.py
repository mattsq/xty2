"""The paired eq. (7) + eq. (9) versus exact-marginalisation benchmark.

Card `docs/recipes/variational_treatment.md` §6 declares a *substitution*, not a
leaderboard: arm **A** replaces `MissingTreatmentMarginalNLL`'s exact sum over
candidate treatments with M2's variational bound plus its labelled posterior
term, and arm **B** is the same graph, the same rows, the same batches and the
same budget with the exact sum still in place. Everything that could differ
between the two other than that substitution is held equal here — the fixture
draws, the parameter initialisation of every shared component, the sampler
stream and the step count — so a ratio between them is about the objective.

Both arms are *scored* by exact marginalisation. Arm A is therefore measured by
what it bounds rather than by its own bound, which is the only way the two
numbers are comparable at all; §6's preamble says so, and this module never
reads `variational_treatment_elbo`'s value as a held-out metric.

The fixture is `fixmatch.md` §6.1's `cluster_population` with the one declared
change of §6.1: the assignment overlap moves from 0.02/0.98 to 0.25/0.75, which
is `low=0.25` in that generator's own vocabulary. It is imported rather than
restated for the reason `doublematch`'s module gives — two transcriptions of one
DGP is how two cards' numbers stop being about the recipes.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn import functional as F

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    ComponentGraph,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    Weighted,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    ClusterPopulation,
    cluster_population,
    column,
    configure_worker,
    continuous_schema,
    on_the_training_scale,
    parallel_replicates,
    training_dataset,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.objectives import MissingTreatmentMarginalNLL
from xty2.recipes import variational_treatment
from xty2.recipes.variational_treatment import VARIATIONAL_TREATMENT_STEPS
from xty2.training import StageResult, run_stage

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 90_000
"""`fixmatch.md` §6.1's stream, which §6.1 adopts along with its generator.

Not this card's Tier 1 seeds, and the difference is the whole point of the
sentence: §6.1 changes exactly one thing about `fixmatch.md` §6.1 — the
assignment overlap — and "seed streams" is on the list of things it takes
unchanged. `s_r = 90000 + 100r` with the population draws at `s_r+1` and
`s_r+2`, the initialisation at `s_r+6` and the stage at `s_r+10000` is that
stream, offsets included. A benchmark that quietly picked its own would stamp
this block's digest on numbers the block did not describe.

`uda` and `meta_pseudo_labels` sit at 94_000 because their own §6.1 says so.
This card's does not.
"""

_OVERLAP = 0.25
"""§6.1's one declared change to `fixmatch.md` §6.1.

`cluster_population`'s `low` is the mass the assignment puts on the *wrong*
cluster, so `low=0.25` is exactly `p(t=1|c) = 0.25 + 0.5c` at `K = 2`. The card
states the assignment and the generator states the parameter; this line is the
only place the two spellings meet.
"""

_ELBO = "variational_treatment_elbo"
_POSTERIOR = "categorical_posterior"
_WINDOW = 50
"""How many steps "terminal" and "step-50" each average over.

§6's tolerance asks for the gap's "terminal value below its step-50 value in
mean" without fixing a window, exactly as `doublematch.md` §6 asked for a
terminal alignment without one. Fifty is not a new choice: §6.2's Tier 1 item 3
already reports "mean KL(q ‖ posterior) over the first 50 and last 50 steps",
so the two tiers report the same statistic under the same name.

The *bounded* gap is the held-out one — §6's preamble puts every metric on the
held-out population — and the trajectory pair is read from the training log,
where a step-50 value exists at all. Both windows ship as informational metrics
so the alternative reading, which bounds the terminal training window instead,
can be checked against the same artifact rather than re-run.
"""


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired variational / exact-marginalisation fits."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2) with overlapping treatment assignment, specified in 6.1"
            ),
            "variant": (
                "paired substitution of eq. (7) + eq. (9) for exact "
                "marginalisation; same fixture, seeds, batches, backbone, "
                "optimiser and step budget in both arms"
            ),
            "split": (
                "1024 train rows with 64 observed treatments, 2048 held-out "
                "rows with every treatment and outcome observed"
            ),
            "metric": (
                "held-out exact marginal NLL ratio, variational arm over exact "
                "arm; amortisation gap, posterior advantage, the share of its "
                "own model posterior's y-information the amortised head "
                "recovers, and two likelihood guardrails as declared in 6.2"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "held_out_marginal_NLL_ratio mean plus one stderr at most 1.02; "
                "amortisation_gap mean plus one stderr at most 0.10 nats; "
                "posterior_information_captured mean minus one stderr at least "
                "0.8, with model_posterior_information mean minus one stderr at "
                "least 0.02 nats so that a fixture whose y carries no treatment "
                "information voids this pair rather than passing it (6.4); "
                "posterior_advantage mean plus one stderr strictly below 0.0 "
                "nats; held_out_outcome_NLL_ratio and "
                "held_out_treatment_NLL_ratio each at most 1.05"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"variational_treatment card reviewed ten replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "held_out_marginal_NLL_ratio", column(rows, "marginal_ratio"), 1.02
            ),
            MetricResult.upper_bound(
                "amortisation_gap",
                column(rows, "held_out_gap"),
                0.10,
                unit="nat/row",
            ),
            # §6.4's replacement for the withdrawn gap-trajectory clause: the
            # share of its own model posterior's y-information that the
            # amortised head recovers, and the denominator that voids the pair
            # on a fixture where there is no such information to recover.
            MetricResult.lower_bound(
                "posterior_information_captured", column(rows, "captured"), 0.8
            ),
            MetricResult.lower_bound(
                "model_posterior_information",
                column(rows, "model_information"),
                0.02,
                unit="nat/row",
            ),
            # "Strictly below 0.0" in the card against "<= 0.0" here, as
            # `fixmatch`'s and `doublematch`'s modules record for their own
            # strict tolerances: one point of a continuous statistic, and a
            # strict relation added to the reporting vocabulary for one card
            # would be `DESIGN.md` §11.2's convenience quadrant with nothing
            # riding on it. The one-stderr rule already excludes the point.
            MetricResult.upper_bound(
                "posterior_advantage",
                column(rows, "posterior_advantage"),
                0.0,
                unit="nat/row",
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio", column(rows, "outcome_ratio"), 1.05
            ),
            MetricResult.upper_bound(
                "held_out_treatment_NLL_ratio", column(rows, "treatment_ratio"), 1.05
            ),
            MetricResult.information(
                "variational_marginal_NLL",
                column(rows, "variational_marginal_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "exact_marginal_NLL",
                column(rows, "exact_marginal_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "held_out_posterior_NLL",
                column(rows, "posterior_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "held_out_propensity_NLL",
                column(rows, "treatment_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "held_out_model_posterior_NLL",
                column(rows, "model_posterior_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "posterior_travel_from_uniform",
                column(rows, "posterior_travel"),
                unit="nat/row",
            ),
            MetricResult.information(
                "training_amortisation_gap_first_50",
                column(rows, "gap_first"),
                unit="nat/row",
            ),
            MetricResult.information(
                "training_amortisation_gap_last_50",
                column(rows, "gap_last"),
                unit="nat/row",
            ),
        ),
        interpretation=(
            "This is the predeclared project-local substitution target: does "
            "replacing xty2's exact sum over candidate treatments with M2's "
            "eq. (7) bound plus eq. (9)'s labelled posterior term cost the "
            "serving path anything measurable, while the amortised q(t|x,y) it "
            "adds stays close to the model posterior and beats p(t|x) on rows "
            "where y is observed. It is not a reproduction of Kingma et al., "
            "whose latent, decoder, datasets, metric and 300,000-update budget "
            "all differ (deviations 1, 3 and 5); no published number applies. "
            "Both arms are scored by exact marginalisation, so the variational "
            "arm is measured by what it bounds rather than by its own bound."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = cluster_population(_TRAIN_ROWS, seed=base + 1, row_offset=0, low=_OVERLAP)
    test = cluster_population(
        _TEST_ROWS, seed=base + 2, row_offset=10_000, low=_OVERLAP
    )
    data = training_dataset(schema, train.batch)

    torch.manual_seed(base + 6)
    variational_recipe = variational_treatment(schema)
    torch.manual_seed(base + 6)
    exact_recipe = _exact_arm(variational_treatment(schema))
    _require_shared_initialisation(variational_recipe, exact_recipe)

    variational_run = compile(variational_recipe)
    exact_run = compile(exact_recipe)
    # Identical seeds: both stages are index 0 of a one-stage program with the
    # same quotas, so the sampler stream matches and the objective is the only
    # difference between the arms.
    variational = run_stage(variational_run, "elbo_fit", data, seed=base + 10_000)
    exact = run_stage(exact_run, "elbo_fit", data, seed=base + 10_000)

    variational_metrics = _evaluate(variational_run, variational, test)
    exact_metrics = _evaluate(exact_run, exact, test)
    for name, value in exact_metrics.items():
        if name.endswith("nll") and value <= 0.0:
            raise RuntimeError(
                f"the exact arm produced a non-positive {name}, so the paired "
                "ratio the card declares is undefined"
            )
    if "posterior_nll" in exact_metrics:
        raise RuntimeError(
            "the exact arm exposed a T_GIVEN_XY head; section 6 defines it as "
            "the graph without categorical_posterior"
        )
    first, last = _gap_windows(variational)
    return {
        "marginal_ratio": (
            variational_metrics["marginal_nll"] / exact_metrics["marginal_nll"]
        ),
        "outcome_ratio": (
            variational_metrics["outcome_nll"] / exact_metrics["outcome_nll"]
        ),
        "treatment_ratio": (
            variational_metrics["treatment_nll"] / exact_metrics["treatment_nll"]
        ),
        "held_out_gap": variational_metrics["amortisation_gap"],
        "captured": _captured(variational_metrics),
        "model_information": _model_information(variational_metrics),
        "model_posterior_nll": variational_metrics["model_posterior_nll"],
        "posterior_travel": variational_metrics["posterior_travel"],
        "gap_first": first,
        "gap_last": last,
        "posterior_advantage": (
            variational_metrics["posterior_nll"] - variational_metrics["treatment_nll"]
        ),
        "variational_marginal_nll": variational_metrics["marginal_nll"],
        "exact_marginal_nll": exact_metrics["marginal_nll"],
        "posterior_nll": variational_metrics["posterior_nll"],
        "treatment_nll": variational_metrics["treatment_nll"],
    }


def _exact_arm(recipe: Recipe) -> Recipe:
    """Arm B of §6: the same serving graph, marginalised exactly.

    Both of `q`'s terms go and one `MissingTreatmentMarginalNLL` takes their
    place at the same constant weight and the same `population` reduction, and
    `categorical_posterior` leaves the graph with them — unlike the zero-weight
    ablations `fixmatch` and `doublematch` build, because §6 declares a
    substitution rather than a switched-off term, and a posterior head with no
    objective would still consume optimiser state and draws.
    """
    graph = ComponentGraph(
        [
            component
            for component in recipe.system.components
            if component.name != _POSTERIOR
        ]
    )
    stage = recipe.program[0]
    if stage.objectives[2].name != _ELBO:
        raise RuntimeError(
            "variational_treatment's third term is no longer the ELBO; "
            f"it is {stage.objectives[2].name!r}"
        )
    exact = Weighted(
        MissingTreatmentMarginalNLL(grad_path="both"),
        weight=1.0,
        reduction="population",
    )
    changed_stage = replace(
        stage,
        objectives=(*stage.objectives[:2], exact),
        trainable=("mlp_encoder", "tarnet_head", "categorical_propensity"),
    )
    return replace(recipe, system=graph, program=Program((changed_stage,)))


def _require_shared_initialisation(variational: Recipe, exact: Recipe) -> None:
    """Every parameter both arms have must start at the same value.

    §6's claim that "nothing in A's serving path is larger than B's" is about
    shape; this is about the draw. The two graphs are built from the same
    manual seed and `categorical_posterior` is constructed last, so the shared
    components consume the same numbers — a reordering of the graph would
    silently break the pairing and nothing else would notice.
    """
    variational_state = variational.system.state_dict()
    exact_state = exact.system.state_dict()
    for name, value in exact_state.items():
        if name not in variational_state:
            raise RuntimeError(f"the exact arm has an unpaired parameter {name!r}")
        if not torch.equal(value, variational_state[name]):
            raise RuntimeError(
                f"variational_treatment paired initial state differs at {name!r}"
            )


def _evaluate(
    run: CompiledRun, result: StageResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out likelihoods, all scored by the exact marginal (§6 preamble)."""
    population = result.population
    if population is None:
        raise RuntimeError(
            "the fitting stage reported no training population; the recipe "
            "declares a sampler, so one is expected"
        )
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    treatments = schema.treatment_cardinality
    metrics: dict[str, float] = {}
    with torch.no_grad():
        values = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        if not isinstance(propensity, CategoricalTreatment) or not isinstance(
            outcome, GaussianOutcome
        ):
            raise TypeError(
                "variational_treatment benchmark expected its reviewed P5 heads"
            )
        candidates = torch.arange(treatments).expand(scaled.batch_size, treatments)
        log_joint = propensity.log_prob(candidates) + outcome.log_prob(
            scaled.y, candidates
        )
        metrics["marginal_nll"] = float(-torch.logsumexp(log_joint, dim=-1).mean())
        metrics["treatment_nll"] = float(F.nll_loss(propensity.log_probs, scaled.t))
        metrics["outcome_nll"] = float(-outcome.log_prob(scaled.y, scaled.t).mean())

        posterior = values.get(Port.T_GIVEN_XY)
        if posterior is not None:
            if not isinstance(posterior, CategoricalTreatment):
                raise TypeError(
                    "variational_treatment benchmark expected a categorical "
                    "amortised posterior"
                )
            # The arm's *own* model posterior, which deviation 1 leaves
            # computable by normalising p(t|x) p(y|x,t) over the same K
            # candidates. `KL(q ‖ ·)` against it is eq. (7)'s whole slack
            # (§3.2); its own held-out NLL is §6.4's ceiling for `q`.
            log_model_posterior = log_joint - torch.logsumexp(
                log_joint, dim=-1, keepdim=True
            )
            metrics["posterior_nll"] = float(F.nll_loss(posterior.log_probs, scaled.t))
            metrics["model_posterior_nll"] = float(
                F.nll_loss(log_model_posterior, scaled.t)
            )
            gap = (posterior.probs * (posterior.log_probs - log_model_posterior)).sum(
                dim=-1
            )
            metrics["amortisation_gap"] = float(gap.mean())
            # How far that posterior has travelled from the uniform it starts
            # at: `log K - H(posterior)`. Informational, and §6.4's evidence
            # that the fixture gives `q` a moving target rather than a
            # stationary one it matches for free.
            travel = (
                log_model_posterior.exp() * (log_model_posterior + math.log(treatments))
            ).sum(dim=-1)
            metrics["posterior_travel"] = float(travel.mean())
    return metrics


def _model_information(metrics: dict[str, float]) -> float:
    """`NLL[p(t|x)] - NLL[p(t|x,y)]`: what observing `y` is worth to this arm.

    §6.4's denominator and its own guard. On a fixture where `y` says nothing
    about `t` this collapses toward zero, the captured share stops being
    defined in any useful sense, and the tolerance voids the pair rather than
    letting a small gap read as success.
    """
    return metrics["treatment_nll"] - metrics["model_posterior_nll"]


def _captured(metrics: dict[str, float]) -> float:
    """The share of that information the amortised head recovers (§6.4).

    A `q` that ignores `y` scores 0 because it can do no better than the
    propensity; a noisy one scores low or negative. Above 1 is possible and is
    reported rather than clipped: eq. (9) supervises `q` against observed
    treatments, which the model posterior never sees, so `q` can be better
    calibrated against the truth than the product it approximates.
    """
    information = _model_information(metrics)
    if information <= 0.0:
        raise RuntimeError(
            "this arm's model posterior did not beat its own propensity on the "
            f"held-out rows (advantage {information:.6g} nat/row), so the "
            "captured share section 6.4 declares has no denominator"
        )
    return (metrics["treatment_nll"] - metrics["posterior_nll"]) / information


def _gap_windows(result: StageResult) -> tuple[float, float]:
    """Mean logged `KL(q ‖ posterior)` over the first and last `_WINDOW` steps."""
    steps = len(result.records)
    if steps != VARIATIONAL_TREATMENT_STEPS:
        raise RuntimeError(
            f"the card fixes {VARIATIONAL_TREATMENT_STEPS} steps and the stage "
            f"logged {steps}"
        )
    first = [_diagnostic(result, step) for step in range(_WINDOW)]
    last = [_diagnostic(result, step) for step in range(steps - _WINDOW, steps)]
    return sum(first) / _WINDOW, sum(last) / _WINDOW


def _diagnostic(result: StageResult, step: int) -> float:
    """The ELBO term's logged amortisation gap at one step, or a named failure."""
    for term in result.records[step].terms:
        if term.name != _ELBO:
            continue
        try:
            return float(term.diagnostics["amortisation_gap"])
        except KeyError:
            raise RuntimeError(
                f"term {_ELBO!r} logged no 'amortisation_gap' at step {step}; "
                f"it logged {sorted(term.diagnostics)!r}"
            ) from None
    raise RuntimeError(f"step {step} has no term named {_ELBO!r}")
