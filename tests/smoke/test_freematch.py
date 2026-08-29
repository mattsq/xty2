"""Tier 1 — FreeMatch on `fixmatch.md` §6.1's fixture, against a constant gate.

The DGP, the label budget, the quota and the seeds are `fixmatch`'s, and they
are **imported** rather than restated: `freematch.md` §6.1 says the fixture is
that one "in full and without modification", and a copy would be a second thing
to keep true.

Two arms, one initialisation, one batch stream:

* **`freematch` as declared** — eq. (8) and eq. (11) at `w_u = 1`, `w_f = 0.05`;
* **the constant-gate arm** — the same recipe with eq. (8) replaced by
  FixMatch's eq. (4) at `tau = 0.95` and the fairness term dropped, which is
  card §6.1's `constant` arm and, under deviation 2's shared views, the same arm
  `flexmatch`'s Tier 1 pairs against.

What is asserted is the *mechanism*: that `tau_t` starts at `1/K`, that the gate
is therefore open on the whole batch where the constant arm's is shut, that
`tau_t` rises and the per-class thresholds separate, and that both arms learn a
propensity worth having. What is **not** asserted is a direction on held-out NLL
between them — `FIDELITY.md` §3 makes that a Tier 2 claim and §6 owns it, and
`flexmatch.md` §6.2 records what treating a single-seed trajectory as a property
of the method cost that card.

One test here exists because of a lesson from a neighbouring card rather than
from this one. `test_the_gate_leaves_its_warm_up` is the guardrail for
deviation 2: FreeMatch's threshold starts at `1/K` and, at `K = 2`, admits every
row, so a strong view that is not label-preserving has an ungated eq. (8) to pin
the propensity with — which is exactly what `flexmatch.md` §6.2 measured on
three initialisation seeds of five at `fixmatch`'s 0.5.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
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
    XTYBatch,
    compile,
)
from xty2.objectives import PseudoLabelTreatmentNLL
from xty2.recipes import freematch
from xty2.recipes.fixmatch import STRONG_X, WEAK_X
from xty2.recipes.freematch import SAT_TERM
from xty2.training import StageResult, run_stage

from tests.smoke.test_fixmatch import (
    SEPARATED,
    STEPS,
    _dataset,
    _on_the_training_scale,
    _Population,
    _populations,
    _schema,
    _with_steps,
)

FAIRNESS = "self_adaptive_fairness"
CONSTANT = "pseudo_label_treatment_nll"
CONSTANT_TAU = 0.95
"""FixMatch's `tau`, and card §6.1's `constant` arm — `flexmatch`'s comparison."""

BATCH_ROWS = 512
"""`B + mu B`, the quota the imported fixture's recipe declares."""

CLASSES = 2
"""`K` of the §6.1 DGP. `tau_0 = 1/K = 0.5` follows from it, and so does the
fact that a softmax over two classes clears that on every row."""


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _constant_gate(recipe: Recipe) -> Recipe:
    """Card §6.1's `constant` arm: eq. (4) at `tau`, and no fairness term.

    Taken *within* this recipe rather than against `fixmatch`, because
    deviation 2 gives the two recipes different strong views. Everything else —
    components, passes, rows, reductions, weights, schedules, optimiser, quota —
    is shared by construction.
    """
    stage = recipe.program[0]
    gated = Weighted(
        PseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=WEAK_X,
            prediction=STRONG_X,
            threshold=CONSTANT_TAU,
            sharpening="hard",
            stop_grad="target",
            rows="all",
        ),
        weight=1.0,
        reduction="mean",
    )
    objectives = (*stage.objectives[:2], gated, *stage.objectives[4:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _term(result: StageResult, step: int, name: str) -> object:
    return next(term for term in result.records[step].terms if term.name == name)


def _diagnostic(result: StageResult, name: str, field: str) -> list[float]:
    """One diagnostic over the whole trajectory.

    Whole-trajectory rather than terminal, for the reason `doublematch.md` §6.2
    records: a state that spent the run in one place and moved at the end reads
    the same terminally as one that never moved at all.
    """
    return [
        float(_term(result, step, name).diagnostics[field])  # type: ignore[attr-defined]
        for step in range(len(result.records))
    ]


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    treatment_nll: float
    student_treatment_nll: float
    outcome_nll: float
    frequency_nll: float


def _evaluate(
    run: CompiledRun, result: StageResult, test: _Population, train: XTYBatch
) -> _Metrics:
    """The EMA §5.1 evaluates from, and the network it averages.

    Both, for the reason `fixmatch.md` §6.2 gives: an EMA number can improve
    while the network under it gets worse, so a claim resting on the EMA alone
    is a claim about a reporting device.
    """
    schema = run.recipe.schema
    assert result.teacher is not None
    with torch.no_grad():
        values = result.teacher.graph.evaluate(
            test.batch, schema=schema, only=run.graph.names
        )
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(outcome, GaussianOutcome)
        student = run.graph.evaluate(test.batch, schema=schema, only=run.graph.names)
        student_propensity = student[Port.T_GIVEN_X]
        assert isinstance(student_propensity, CategoricalTreatment)

        frequencies = torch.bincount(train.t[train.t_observed], minlength=CLASSES)
        shares = frequencies.float()
        shares /= shares.sum()
        baseline = shares.log().expand(test.batch.batch_size, -1)
        return _Metrics(
            run=run,
            result=result,
            treatment_nll=float(F.nll_loss(propensity.log_probs, test.batch.t)),
            student_treatment_nll=float(
                F.nll_loss(student_propensity.log_probs, test.batch.t)
            ),
            outcome_nll=float(-outcome.log_prob(test.batch.y, test.batch.t).mean()),
            frequency_nll=float(F.nll_loss(baseline, test.batch.t)),
        )


def _run(recipe: Recipe, train: XTYBatch, test: _Population, steps: int) -> _Metrics:
    run = compile(_with_steps(recipe, steps))
    result = run_stage(run, "joint_fit", _dataset(train), seed=90_010)
    return _evaluate(run, result, _on_the_training_scale(test, result), train)


@dataclass(frozen=True)
class _Arms:
    adaptive: _Metrics
    constant: _Metrics


@pytest.fixture(scope="module")
def arms() -> _Arms:
    """The two arms, from one initialisation and one batch stream."""
    schema = _schema()
    train, test = _populations(SEPARATED, seed=90_001)

    torch.manual_seed(90_006)
    adaptive = freematch(schema)
    torch.manual_seed(90_006)
    constant = _constant_gate(freematch(schema))
    for name, value in adaptive.system.state_dict().items():
        assert torch.equal(value, constant.system.state_dict()[name])

    return _Arms(
        adaptive=_run(adaptive, train, test, STEPS),
        constant=_run(constant, train, test, STEPS),
    )


# ---------------------------------------------------------------------------
# Self-adaptive thresholding
# ---------------------------------------------------------------------------


def test_the_gate_starts_at_one_over_k_and_therefore_starts_open(
    arms: _Arms,
) -> None:
    """Card §2's first limitation, and the opposite of what FixMatch does.

    Eqs. (5) and (6) put `tau_0 = p~_0(c) = 1/C`, so `MaxNorm` is all ones and
    every threshold is `0.5`. A two-class softmax clears that on every row, so
    eq. (8) trains on the hard arg max of an untrained network over the whole
    batch — where eq. (4)'s constant `tau` admits nothing at all until the model
    has sharpened, which the constant arm shows on the same batch.
    """
    first = _term(arms.adaptive.result, 0, SAT_TERM)
    assert first.diagnostics["tau_global"] == pytest.approx(1.0 / CLASSES)  # type: ignore[attr-defined]
    assert first.diagnostics["threshold_max"] == pytest.approx(1.0 / CLASSES)  # type: ignore[attr-defined]
    assert first.diagnostics["threshold_min"] == pytest.approx(1.0 / CLASSES)  # type: ignore[attr-defined]
    assert first.diagnostics["coverage"] == 1.0  # type: ignore[attr-defined]
    assert _term(arms.constant.result, 0, CONSTANT).diagnostics["coverage"] == 0.0  # type: ignore[attr-defined]


def test_the_gate_leaves_its_warm_up(arms: _Arms) -> None:
    """Deviation 2's guardrail, on the mechanism that makes it load-bearing.

    `tau_t` is the EMA of the model's own confidence, so it only rises if the
    model becomes confident — and until it does, eq. (8) is charging a one-hot
    strong-view target on every row. Under a strong view that is not
    label-preserving that pins the weak-view confidence and `tau_t` never leaves
    `1/K`, which is the fixed point `flexmatch.md` §6.2 measured on three
    initialisation seeds of five at `fixmatch`'s 0.5 strong view.

    A failure here means the strong view has stopped being label-preserving, and
    `flexmatch.md` §5.2's table is where to look before this assertion is
    touched. **The seed is not arbitrary**: 90_006 is that card's replicate
    `r = 0`, one of the three that locked under the 0.5 view.
    """
    tau = _diagnostic(arms.adaptive.result, SAT_TERM, "tau_global")
    highest = _diagnostic(arms.adaptive.result, SAT_TERM, "threshold_max")
    lowest = _diagnostic(arms.adaptive.result, SAT_TERM, "threshold_min")

    assert tau[0] == pytest.approx(1.0 / CLASSES)
    assert tau[-1] > 1.0 / CLASSES
    assert max(tau) == tau[-1], "eq. (5) is an EMA of a confidence that rises"
    # Eq. (7)'s local half, which unlike FlexMatch's `beta` does not collapse at
    # `K = 2`: `MaxNorm(p~)` separates unless `p~` is exactly uniform.
    assert max(high - low for high, low in zip(highest, lowest, strict=True)) > 0.0
    assert all(high == pytest.approx(t) for high, t in zip(highest, tau, strict=True))


def test_both_freematch_terms_are_charged_on_every_row(arms: _Arms) -> None:
    """Eqs. (8) and (11) both take `all` (FixMatch's footnote 2, inherited).

    The wiring floor: a term that had quietly stopped seeing rows would satisfy
    the trajectory assertions above by seeing nothing.
    """
    for step in (0, STEPS // 2, STEPS - 1):
        for name in (SAT_TERM, FAIRNESS):
            assert _term(arms.adaptive.result, step, name).n == BATCH_ROWS  # type: ignore[attr-defined]


def test_the_fairness_term_has_a_support_to_work_over(arms: _Arms) -> None:
    """Card §7's exclusion convention, measured rather than assumed.

    A class no retained row predicts leaves both `SumNorm`s, and below two
    surviving classes the term is zero. That is correct and it is also inert, so
    a run in which it were *always* inert would be one where eq. (11) never
    fired at all — which the card would want to know about rather than infer.
    """
    support = _diagnostic(arms.adaptive.result, FAIRNESS, "fairness_support")
    entropy = _diagnostic(arms.adaptive.result, FAIRNESS, "marginal_entropy")
    live = [value for value in support if value >= 2.0]
    assert len(live) > 0.5 * len(support), (
        "eq. (11) was inert on most steps; card §7's convention is biting"
    )
    assert max(entropy) > 0.0
    assert max(entropy) <= math.log(CLASSES) + 1e-6, (
        "the marginal entropy of a K-class distribution cannot exceed log K"
    )


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_the_supervised_term_decreases_in_both_arms(arms: _Arms) -> None:
    """Tier 1's actual question — is the fit connected to the data at all.

    Eq. (8)'s contribution moves with the gate, so the mixed total says nothing
    here; eq. (3)'s cross-entropy on the labelled rows is the term that must
    come down, and it is also the term that stayed flat above `log 2` in the
    runs `flexmatch.md` deviation 2 was written about.
    """
    for arm in (arms.adaptive, arms.constant):
        values = [
            float(_term(arm.result, step, "observed_treatment_nll").value)  # type: ignore[attr-defined]
            for step in range(len(arm.result.records))
        ]
        early = sum(values[:100]) / 100.0
        late = sum(values[-100:]) / 100.0
        assert late < 0.5 * early
        assert late < 0.693, "log 2 — the labelled cross-entropy of a coin flip"


def test_both_arms_learn_a_propensity_worth_having(arms: _Arms) -> None:
    """The floor, asserted before anything is compared.

    A paired ratio between two models that have both learned nothing is a number
    with no content, and `flexmatch.md` §6.2's first draft was exactly that
    failure.
    """
    for arm in (arms.adaptive, arms.constant):
        assert arm.treatment_nll < 0.75 * arm.frequency_nll
        assert arm.student_treatment_nll < 0.75 * arm.frequency_nll


def test_the_outcome_stack_is_not_damaged(arms: _Arms) -> None:
    """Eqs. (8) and (11) train the encoder the outcome head reads.

    Card §6's tolerance is non-inferiority within 5% against this exact arm.
    Directional treatment-NLL claims stay in §6 — this one is a guardrail, and a
    guardrail is allowed to be one-sided.
    """
    assert arms.adaptive.outcome_nll < 1.05 * arms.constant.outcome_nll


def test_the_two_arms_saw_the_same_rows(arms: _Arms) -> None:
    """The pair is a pair: same seeds, same quota, same batches.

    `stateful-sampler` (`DESIGN.md` §11.4) is the ledger row that protects this,
    and SAT is exactly the kind of mechanism that would break it if the
    threshold had been put in the sampler instead of in the objective — it reads
    the model to decide which rows *train*, never which rows are *drawn*.
    """
    assert torch.equal(
        arms.adaptive.result.checkpoint.trained_on_row_ids,
        arms.constant.result.checkpoint.trained_on_row_ids,
    )
