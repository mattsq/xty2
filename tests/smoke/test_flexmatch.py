"""Tier 1 — FlexMatch on `fixmatch.md` §6.1's fixture, against a constant gate.

The DGP, the label budget, the quota and the seeds are `fixmatch`'s, and they
are **imported** rather than restated: `flexmatch.md` §6.1 says the fixture is
that one "in full and without modification", and a copy would be a second thing
to keep true.

Two arms, one initialisation, one batch stream:

* **`flexmatch` as declared**;
* **the constant-gate arm** — the same recipe with eq. (8) replaced by
  FixMatch's eq. (3) at `tau`, which under deviation 2's shared views is the §6
  pair exactly. Tier 0 proves the two plans differ in the gate and in nothing
  else.

What is asserted is the *mechanism*: that the thresholds start at zero, rise
with the marks and reach `tau`, that a per-class spread opens up, and that both
arms learn a propensity worth having. What is **not** asserted is a direction on
held-out NLL between them — `FIDELITY.md` §3 makes that a Tier 2 claim and §6
owns it, and an earlier version of this file got into trouble by treating a
single-seed trajectory as a property of the method.

That trouble is worth stating, because one test here exists because of it. The
first draft of this recipe inherited `fixmatch`'s strong view, which is not
label-preserving (`flexmatch.md` §5.2), and on three initialisation seeds of
five the curriculum then never left its warm-up: nothing cleared `tau`, no mark
was laid, and `T(c)` stayed at zero for the whole run. The draft asserted that
outcome from one seed, which would have frozen an artefact into CI.
`test_the_curriculum_leaves_its_warm_up` is the assertion that would have failed
on those seeds, and it is the one guarding deviation 2 now.
"""

from __future__ import annotations

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
from xty2.recipes import flexmatch
from xty2.recipes.fixmatch import STRONG_X, WEAK_X
from xty2.recipes.flexmatch import TAU
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

CURRICULUM = "curriculum_pseudo_label_treatment_nll"
CONSTANT = "pseudo_label_treatment_nll"
BATCH_ROWS = 512
"""`B + mu B`, the quota the imported fixture's recipe declares."""


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _constant_gate(recipe: Recipe) -> Recipe:
    """Eq. (8) replaced by eq. (3): the same term at a fixed `tau`.

    The §6 pair, and it is taken *within* this recipe rather than against
    `fixmatch` because deviation 2 gives the two recipes different strong views.
    Everything else — components, passes, rows, reductions, weights, schedules,
    optimiser, quota — is shared by construction.
    """
    stage = recipe.program[0]
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
        weight=1.0,
        reduction="mean",
    )
    objectives = (*stage.objectives[:2], gated, *stage.objectives[3:])
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
    """The EMA §4 reports from, and the network it averages.

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

        frequencies = torch.bincount(train.t[train.t_observed], minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(test.batch.batch_size, -1)
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
    curriculum: _Metrics
    constant: _Metrics


@pytest.fixture(scope="module")
def arms() -> _Arms:
    """The two arms, from one initialisation and one batch stream."""
    schema = _schema()
    train, test = _populations(SEPARATED, seed=90_001)

    torch.manual_seed(90_006)
    curriculum = flexmatch(schema)
    torch.manual_seed(90_006)
    constant = _constant_gate(flexmatch(schema))
    for name, value in curriculum.system.state_dict().items():
        assert torch.equal(value, constant.system.state_dict()[name])

    return _Arms(
        curriculum=_run(curriculum, train, test, STEPS),
        constant=_run(constant, train, test, STEPS),
    )


# ---------------------------------------------------------------------------
# The curriculum
# ---------------------------------------------------------------------------


def test_every_threshold_starts_at_zero_and_the_gate_starts_open(
    arms: _Arms,
) -> None:
    """Algorithm 1 at `t = 0`, and the opposite of what FixMatch does.

    With no mark laid down, `beta = 0`, `M(0) = 0` and every row clears a
    threshold of zero — where eq. (3)'s constant `tau` admits nothing at all
    until the model has sharpened, which the constant-gate arm shows on the
    same batch.
    """
    first = _term(arms.curriculum.result, 0, CURRICULUM)
    assert first.diagnostics["threshold_max"] == 0.0  # type: ignore[attr-defined]
    assert first.diagnostics["marked_fraction"] == 0.0  # type: ignore[attr-defined]
    assert first.diagnostics["coverage"] == 1.0  # type: ignore[attr-defined]
    assert _term(arms.constant.result, 0, CONSTANT).diagnostics["coverage"] == 0.0  # type: ignore[attr-defined]


def test_the_curriculum_leaves_its_warm_up(arms: _Arms) -> None:
    """Deviation 2's guardrail, and the assertion an earlier draft lacked.

    CPL can only start once a row clears the fixed `tau` (algorithm 1 line 14),
    and until then eq. (8) is ungated. Under a strong view that is not
    label-preserving that is a fixed point: the term pins the propensity below
    `tau`, no mark is laid, and `T(c)` never leaves zero — which is what
    `fixmatch`'s 0.5 strong view did on three initialisation seeds of five
    (`flexmatch.md` §5.2, §6.2). At the 0.2 deviation 2 derives from the
    Bayes-optimal label-flip rate it does not happen on any of them.

    A failure here means the strong view has stopped being label-preserving, and
    §5.2's table is where to look before this assertion is touched.

    **The seed matters and is not arbitrary.** This module's initialisation seed
    is 90_006, which is §6.2's replicate `r = 0` — one of the three seeds of five
    that *did* lock under the 0.5 view. So this guardrail demonstrably catches
    the regression it is written for, on the one seed it can afford to run.
    Changing the seed without checking it against §6.2's per-seed table would
    silently drop that: `r = 3` and `r = 4` did not lock, and the same assertion
    on either of them would pass through the bug.
    """
    marked = _diagnostic(arms.curriculum.result, CURRICULUM, "marked_fraction")
    highest = _diagnostic(arms.curriculum.result, CURRICULUM, "threshold_max")
    lowest = _diagnostic(arms.curriculum.result, CURRICULUM, "threshold_min")
    coverage = _diagnostic(arms.curriculum.result, CURRICULUM, "coverage")

    assert marked[0] == 0.0
    assert marked[-1] > 0.5
    assert max(highest) == pytest.approx(TAU)
    # A per-class spread at some point in the run, which is the whole of what
    # CPL adds over a constant: one class at the ceiling while another is still
    # being let through more cheaply. Over the trajectory rather than at the
    # end, because with K = 2 and a near-balanced fixture both classes reach
    # `beta = 1` eventually and the spread closes (card §2, third limitation).
    # `min(...) >= 0` was here too and an adversarial review removed it: `max`
    # and `min` of one dict cannot cross, so it could not fail.
    assert max(high - low for high, low in zip(highest, lowest, strict=True)) > 0.1
    assert coverage[0] == 1.0
    assert min(coverage) < 0.95


def test_the_term_is_charged_on_every_row(arms: _Arms) -> None:
    """Eq. (8)'s population is `all` (FixMatch's footnote 2, inherited).

    The wiring floor: a term that had quietly stopped seeing rows would satisfy
    the trajectory assertions above by seeing nothing.
    """
    for step in (0, STEPS // 2, STEPS - 1):
        assert _term(arms.curriculum.result, step, CURRICULUM).n == BATCH_ROWS  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_the_supervised_term_decreases_in_both_arms(arms: _Arms) -> None:
    """Tier 1's actual question — is the fit connected to the data at all.

    Eq. (8)'s contribution moves with the gate, so the mixed total says nothing
    here; eq. (10)'s cross-entropy on the labelled rows is the term that must
    come down, and it is also the term that stayed flat above `log 2` in the
    runs deviation 2 was written about.
    """
    for arm in (arms.curriculum, arms.constant):
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
    with no content, and §6.2's first draft was exactly that failure.
    """
    for arm in (arms.curriculum, arms.constant):
        assert arm.treatment_nll < 0.75 * arm.frequency_nll
        assert arm.student_treatment_nll < 0.75 * arm.frequency_nll


def test_the_outcome_stack_is_not_damaged(arms: _Arms) -> None:
    """Eq. (8) trains the encoder the outcome head reads, so this is not free.

    Card §6's tolerance is non-inferiority within 5% against this exact arm.
    Directional treatment-NLL claims stay in §6 — this one is a guardrail, and
    a guardrail is allowed to be one-sided.
    """
    assert arms.curriculum.outcome_nll < 1.05 * arms.constant.outcome_nll


def test_the_two_arms_saw_the_same_rows(arms: _Arms) -> None:
    """The pair is a pair: same seeds, same quota, same batches.

    `stateful-sampler` (`DESIGN.md` §11.4) is the ledger row that protects this,
    and CPL is exactly the kind of mechanism that would break it if the
    curriculum had been put in the sampler instead of in the objective — it
    reads the model to decide which rows *train*, never which rows are *drawn*.
    """
    assert torch.equal(
        arms.curriculum.result.checkpoint.trained_on_row_ids,
        arms.constant.result.checkpoint.trained_on_row_ids,
    )
