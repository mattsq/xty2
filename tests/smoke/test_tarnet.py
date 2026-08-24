"""Tier 1 — `tarnet` fits the synthetic DGP (`FIDELITY.md` §3, `PLAN.md` P5).

**This is a wiring test, not a fidelity claim.** It answers "is this recipe
connected to the data at all", which Tier 0 cannot and Tier 2 is too slow to.
Nothing here says anything about TARNet's published numbers; that is §6 of the
card and P12's job.

The four assertions are the ones `FIDELITY.md` §3 lists, and the last of them
is the load-bearing one:

> with 50% of `t` missing at random, the recipe using exact marginalisation
> beats the complete-case baseline of the same recipe.

It fails if the marginalisation term is scheduled to zero, is masked to an
empty row set, or is detached — because in every one of those cases the two
fits below are the *same fit*, run from the same seed on the same batches, and
a strict inequality between identical numbers is false. That is the class of
silent death the whole mixer logging surface exists to expose.

It is run over three seeds and compared on the means. A single-seed comparison
on a noisy metric is not evidence, and `FIDELITY.md` §3 says so about Tier 2;
it is no more true here.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pytest
import torch
from torch import Tensor
from xty2.components import MLPArchitecture
from xty2.core import (
    ComponentGraph,
    Constant,
    GradientClipping,
    OptimiserSpec,
    OutcomeDistribution,
    Port,
    TreatmentDistribution,
    WeightDecay,
    XTYBatch,
    compile,
)
from xty2.recipes import tarnet
from xty2.training import StageResult, StepRecord, run_stage

from tests.smoke.synthetic import NUM_TREATMENTS, SyntheticXTY, minibatches

# The card architecture at Tier 1 sizes. Tier 1 is about a minute per recipe,
# so the widths are smaller than card §4's — an override, and therefore a
# departure from the card that is visible in the plan this compiles to.
ARCHITECTURE = MLPArchitecture(
    representation=(64, 64),
    head=(32,),
    activation="elu",
    normalisation="none",
    dropout=0.0,
    initialisation="xavier_normal",
)
OPTIMISER = OptimiserSpec(
    name="adam",
    lr=3e-3,
    weight_decay=WeightDecay.none(),
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
)

STEPS = 600
BATCH_SIZE = 256
TRAIN_ROWS = 1024
TEST_ROWS = 2048
MISSING_RATE = 0.5
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Evaluation:
    """What a fitted recipe scores on a held-out split with `t` observed."""

    factual_nll: float
    """Mean `-log p(y | x, t)` at the true treatment."""
    treatment_nll: float
    """Mean `-log p(t | x)`, against which the frequency baseline is compared."""
    ate: Tensor
    """`[K]` estimated ATE of each arm against arm 0, from treatment-wise means."""


def fit(
    dgp: SyntheticXTY, train: XTYBatch, *, marginal_weight: float, seed: int
) -> tuple[StageResult, Evaluation]:
    """One fit of the recipe, and its held-out evaluation.

    `marginal_weight` is a plain number rather than the card's ramp, so that
    the complete-case baseline is the *same recipe* with one weight at zero
    and nothing else changed. Both fits see the same batches in the same
    order, from the same seed.
    """
    torch.manual_seed(seed)
    run = compile(
        tarnet(
            dgp.schema,
            architecture=ARCHITECTURE,
            optimiser=OPTIMISER,
            steps=STEPS,
            marginal_weight=marginal_weight,
        )
    )
    result = run_stage(run, "fit", minibatches(train, BATCH_SIZE, seed=7), seed=seed)
    return result, evaluate(run.graph, dgp)


def evaluate(graph: ComponentGraph, dgp: SyntheticXTY) -> Evaluation:
    """Score a fitted graph on a fresh split whose treatments are all observed."""
    test = dgp.draw(TEST_ROWS, seed=2, missing_rate=0.0)
    was_training = graph.training
    graph.eval()
    with torch.no_grad():
        ports = graph.evaluate(test, schema=dgp.schema)
    graph.train(was_training)

    head = ports[Port.Y_GIVEN_XT]
    propensity = ports[Port.T_GIVEN_X]
    assert isinstance(head, OutcomeDistribution)
    assert isinstance(propensity, TreatmentDistribution)

    candidates = torch.arange(NUM_TREATMENTS).expand(TEST_ROWS, NUM_TREATMENTS)
    means = head.mean(candidates)
    return Evaluation(
        factual_nll=float(-head.log_prob(test.y, test.t).mean()),
        treatment_nll=float(-propensity.log_prob(test.t).mean()),
        ate=(means - means[:, :1]).mean(dim=0),
    )


@dataclass(frozen=True)
class Fits:
    """Both arms of the comparison, over every seed. Computed once."""

    dgp: SyntheticXTY
    marginalised: tuple[tuple[StageResult, Evaluation], ...]
    complete_case: tuple[tuple[StageResult, Evaluation], ...]
    baseline_treatment_nll: float


@pytest.fixture(scope="module")
def fits() -> Fits:
    """Six fits: marginalised and complete-case, three seeds each."""
    dgp = SyntheticXTY.build()
    marginalised = []
    complete_case = []
    for seed in SEEDS:
        train = dgp.draw(TRAIN_ROWS, seed=100 + seed, missing_rate=MISSING_RATE)
        marginalised.append(fit(dgp, train, marginal_weight=1.0, seed=seed))
        complete_case.append(fit(dgp, train, marginal_weight=0.0, seed=seed))
    train = dgp.draw(TRAIN_ROWS, seed=100, missing_rate=MISSING_RATE)
    return Fits(
        dgp=dgp,
        marginalised=tuple(marginalised),
        complete_case=tuple(complete_case),
        baseline_treatment_nll=_frequency_baseline(dgp, train),
    )


def _frequency_baseline(dgp: SyntheticXTY, train: XTYBatch) -> float:
    """Held-out log-loss of predicting the training marginal frequency of `t`.

    The baseline a propensity head has to beat to be doing anything at all.
    Fitted on the rows where `t` was observed, because those are the rows a
    complete-case estimate of the marginal would have.
    """
    counts = torch.bincount(train.t[train.t_observed], minlength=NUM_TREATMENTS).float()
    frequencies = counts / counts.sum()
    test = dgp.draw(TEST_ROWS, seed=2, missing_rate=0.0)
    return float(-frequencies.log()[test.t].mean())


def test_the_tier_marker_is_applied_by_directory(
    request: pytest.FixtureRequest,
) -> None:
    assert request.node.get_closest_marker("tier1") is not None


# ---------------------------------------------------------------------------
# The four assertions of FIDELITY.md §3
# ---------------------------------------------------------------------------


def test_the_training_loss_decreases(fits: Fits) -> None:
    result, _ = fits.marginalised[0]
    window = STEPS // 10
    first = [record.total for record in result.records[:window]]
    last = [record.total for record in result.records[-window:]]
    assert sum(last) / window < sum(first) / window

    # And the factual term specifically, not only the total: with three terms
    # mixed, a falling total is consistent with one of them going nowhere.
    def factual(records: Sequence[StepRecord]) -> float:
        return _mean(record.terms[0].value for record in records)

    assert factual(result.records[-window:]) < factual(result.records[:window])


def test_the_propensity_head_beats_the_frequency_baseline(fits: Fits) -> None:
    for _, evaluation in fits.marginalised:
        assert evaluation.treatment_nll < fits.baseline_treatment_nll


def test_the_estimated_ate_lands_near_the_analytic_one(fits: Fits) -> None:
    # A wide band on purpose. Tier 1 asks whether the treatment-wise means are
    # wired to the data, not how accurate they are; a tight band here would
    # fail on seed noise and teach everyone to widen it.
    truth = fits.dgp.true_ate()
    for _, evaluation in fits.marginalised:
        assert torch.allclose(evaluation.ate, truth, atol=0.5)


def test_marginalisation_beats_complete_case_at_half_missing(fits: Fits) -> None:
    # The load-bearing assertion. See the module docstring for what it fails
    # on and why it is a mean over seeds rather than a single comparison.
    marginalised = _mean(evaluation.factual_nll for _, evaluation in fits.marginalised)
    complete_case = _mean(
        evaluation.factual_nll for _, evaluation in fits.complete_case
    )
    assert marginalised < complete_case, (
        f"exact marginalisation scored {marginalised:.6f} against the "
        f"complete-case baseline's {complete_case:.6f} on held-out factual "
        "NLL. Equality means the two fits were the same fit: the marginal "
        "term is scheduled to zero, sees no eligible rows, or carries no "
        "gradient (DESIGN.md §1.3, §4)."
    )


def test_the_marginal_term_actually_saw_rows(fits: Fits) -> None:
    # The diagnostic behind the assertion above, asserted directly so that a
    # failure there can be attributed. Half the rows are missing a treatment,
    # so the term's coverage should sit near half the batch on every step.
    result, _ = fits.marginalised[0]
    for record in result.records:
        term = record.terms[2]
        assert term.name == "missing_treatment_marginal_nll"
        assert term.n > 0
        assert 0.3 < term.coverage < 0.7
    assert result.records[-1].terms[2].weighted != 0.0


def test_the_complete_case_baseline_leaves_the_term_out_of_the_total(
    fits: Fits,
) -> None:
    # The other half of the comparison being honest: the baseline still
    # *computes* the term and logs it, and contributes zero of it to the
    # total. A baseline that had stopped computing it would be a different
    # recipe rather than the same one with one weight at zero.
    result, _ = fits.complete_case[0]
    for record in result.records:
        term = record.terms[2]
        assert term.n > 0
        assert term.weight == 0.0
        assert term.weighted == 0.0


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return float(sum(collected) / len(collected))
