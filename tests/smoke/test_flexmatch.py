"""Tier 1 — FlexMatch on `fixmatch.md` §6.1's fixture, against its own ablation.

The DGP, the label budget, the quota and the seeds are `fixmatch`'s, and they
are **imported** rather than restated: `flexmatch.md` §6.1 says the fixture is
that one "in full and without modification", and a copy would be a second thing
to keep true.

What is new here is that the mechanism under test has a state, and the state has
a fixed point. Curriculum Pseudo Labeling starts every threshold at zero
(algorithm 1 lines 2, 5-7), so eq. (8) opens on the whole batch; it raises a
class's threshold only once rows clear the *fixed* `tau`; and on this fixture no
row ever does, because the term it is gating trains the propensity on its own
arg max from step 0 and pins it below `tau` for the rest of the run. That is
`flexmatch.md` §2's first limitation, and §6.2 records it.

Two arms are therefore run, and the second is what makes the reading a
measurement rather than a story:

* **`flexmatch` as declared**, where none of the curriculum happens;
* **`flexmatch` at `lambda = 0`** — the same objective, the same state and the
  same diagnostics, *computed and logged on every step and descended on none*.
  Its marks accumulate, its thresholds climb to `tau` and its per-class spread
  opens up, which is the evidence that the wiring is right and the lock is a
  property of the mechanism rather than of this port.

The §6 comparison against `fixmatch` is deliberately **not** a third arm here.
It is a Tier 2 claim (`FIDELITY.md` §3), running it would duplicate a fit
`tests/smoke/test_fixmatch.py` already performs on this exact fixture and these
exact seeds on every PR, and eighty seconds of CI is a real price for a column
§6.2 records from a one-off run. What is asserted here is the mechanism, in both
the state where it works and the state where it does not.
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
    Constant,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    XTYBatch,
    compile,
)
from xty2.recipes import flexmatch
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
BATCH_ROWS = 512
"""`B + mu B`, the quota the imported fixture's recipe declares."""


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _zero_weight(recipe: Recipe) -> Recipe:
    """`lambda = 0`: eq. (8) computed and logged, and not descended."""
    stage = recipe.program[0]
    ablated = replace(stage.objectives[2], weight=Constant(0.0))
    objectives = (*stage.objectives[:2], ablated, *stage.objectives[3:])
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
    declared: _Metrics
    ablated: _Metrics


@pytest.fixture(scope="module")
def arms() -> _Arms:
    """The two arms, from one initialisation and one batch stream."""
    schema = _schema()
    train, test = _populations(SEPARATED, seed=90_001)

    torch.manual_seed(90_006)
    declared_recipe = flexmatch(schema)
    torch.manual_seed(90_006)
    ablated_recipe = _zero_weight(flexmatch(schema))
    for name, value in declared_recipe.system.state_dict().items():
        assert torch.equal(value, ablated_recipe.system.state_dict()[name])

    return _Arms(
        declared=_run(declared_recipe, train, test, STEPS),
        ablated=_run(ablated_recipe, train, test, STEPS),
    )


# ---------------------------------------------------------------------------
# The mechanism, where it works
# ---------------------------------------------------------------------------


def test_every_threshold_starts_at_zero_and_the_gate_starts_open(
    arms: _Arms,
) -> None:
    """Algorithm 1 at `t = 0`, and the opposite of what FixMatch does.

    With no mark laid down, `beta = 0`, `M(0) = 0` and every row clears a
    threshold of zero — where eq. (3)'s constant `tau` admits nothing at all
    until the model has sharpened, which
    `test_fixmatch.py::test_the_gate_opens_as_the_model_sharpens` asserts of the
    other half of the contrast on the same fixture.
    """
    for arm in (arms.declared, arms.ablated):
        first = _term(arm.result, 0, CURRICULUM)
        assert first.diagnostics["threshold_max"] == 0.0  # type: ignore[attr-defined]
        assert first.diagnostics["marked_fraction"] == 0.0  # type: ignore[attr-defined]
        assert first.diagnostics["coverage"] == 1.0  # type: ignore[attr-defined]


def test_the_curriculum_climbs_when_the_model_is_allowed_to_sharpen(
    arms: _Arms,
) -> None:
    """The wiring evidence, and the control the next test is read against.

    At `lambda = 0` the term is computed and logged on every step and descended
    on none, so the model sharpens under eqs. (10) and the marginal term alone.
    Every part of CPL then does what §3 says: rows clear `tau`, `sigma` rises,
    the best-learned class reaches `T(c) = tau`, the least-learned one stays
    below it, and the gate closes on the difference.
    """
    marked = _diagnostic(arms.ablated.result, CURRICULUM, "marked_fraction")
    highest = _diagnostic(arms.ablated.result, CURRICULUM, "threshold_max")
    lowest = _diagnostic(arms.ablated.result, CURRICULUM, "threshold_min")
    coverage = _diagnostic(arms.ablated.result, CURRICULUM, "coverage")

    assert marked[0] == 0.0
    assert marked[-1] > 0.5
    assert marked == sorted(marked), "marks are sticky, so sigma cannot fall"
    assert max(highest) == pytest.approx(TAU)
    # A per-class spread, which is the whole of what CPL adds: one class at the
    # ceiling while another is still being let through more cheaply.
    assert lowest[-1] < highest[-1]
    assert coverage[0] == 1.0
    assert min(coverage) < 0.95


# ---------------------------------------------------------------------------
# The mechanism, where it does not
# ---------------------------------------------------------------------------


def test_the_declared_arm_never_leaves_the_warm_up(arms: _Arms) -> None:
    """`flexmatch.md` §2's first limitation, and §6.2's finding.

    CPL needs a row to clear `tau` before any threshold can rise, and the term
    it is gating — unfiltered, because the threshold is zero — trains the
    propensity on its own arg max from step 0 and holds it below `tau` for the
    whole run. The zero threshold is an absorbing state of the mechanism on this
    fixture, and the arm above is what says it is the *descended* term that puts
    it there rather than anything about the wiring.

    A failure here is not a broken test: it means the finding has changed, and
    the card's §2 and §6.2 have to be rewritten before this assertion is.
    """
    marked = _diagnostic(arms.declared.result, CURRICULUM, "marked_fraction")
    highest = _diagnostic(arms.declared.result, CURRICULUM, "threshold_max")
    coverage = _diagnostic(arms.declared.result, CURRICULUM, "coverage")
    assert max(marked) == 0.0
    assert max(highest) == 0.0
    assert min(coverage) == 1.0
    # And the propensity pays for it: it does not beat the frequencies of its
    # own labelled rows, which the `lambda = 0` arm comfortably does.
    assert arms.declared.treatment_nll > 0.9 * arms.declared.frequency_nll
    assert arms.ablated.treatment_nll < 0.75 * arms.ablated.frequency_nll


def test_the_term_is_charged_on_every_row_in_both_arms(arms: _Arms) -> None:
    """Eq. (8)'s population is `all` (FixMatch's footnote 2, inherited).

    The wiring floor: a term that had quietly stopped seeing rows would satisfy
    the trajectory assertions above by seeing nothing.
    """
    for arm in (arms.declared, arms.ablated):
        for step in (0, STEPS // 2, STEPS - 1):
            assert _term(arm.result, step, CURRICULUM).n == BATCH_ROWS  # type: ignore[attr-defined]


def _labelled(arm: _Metrics) -> tuple[float, float]:
    """Eq. (10)'s cross-entropy, averaged over the first and last 100 steps."""
    values = [
        float(_term(arm.result, step, "observed_treatment_nll").value)  # type: ignore[attr-defined]
        for step in range(len(arm.result.records))
    ]
    return sum(values[:100]) / 100.0, sum(values[-100:]) / 100.0


def test_the_supervised_term_decreases_when_the_gate_is_not_descended(
    arms: _Arms,
) -> None:
    """Tier 1's actual question — is the fit connected to the data at all.

    Eq. (8)'s contribution moves with the gate, so the mixed total says nothing
    here; eq. (10)'s cross-entropy on the labelled rows is the term that must
    come down. It does in the arm whose unlabelled term is not applied, which is
    what makes the next test a statement about that term.
    """
    early, late = _labelled(arms.ablated)
    assert late < 0.5 * early


def test_the_locked_arm_never_fits_its_labelled_rows_either(arms: _Arms) -> None:
    """The finding is not confined to the unlabelled term, and §6.2 says so.

    Eq. (8) at a zero threshold averages over 512 rows against eq. (10)'s 64, at
    the same weight, and its targets are the model's own arg max. On this
    fixture it does not merely fail to help: eq. (10) stays flat for 3,000 steps
    and above `log K`, so the propensity is confidently wrong on the very rows
    whose treatment it was given.

    Asserted rather than described, and asserted against the arm that does fit,
    so that "FlexMatch trains badly here" cannot be confused with "this fixture
    is unfittable". A failure means the finding has changed and the card's §2
    and §6.2 have to be rewritten before this assertion is.
    """
    early, late = _labelled(arms.declared)
    assert late > 0.9 * early
    assert late > 0.693, "log 2 — the labelled cross-entropy of a coin flip"
    assert _labelled(arms.ablated)[1] < 0.5


def test_the_outcome_stack_survives_the_locked_curriculum(arms: _Arms) -> None:
    """Eq. (8) trains the encoder the outcome head reads, so this is not free.

    Recorded as a guardrail rather than as a claim: card §6's tolerance is
    non-inferiority within 5%, and the point of asserting it on a locked run is
    that a mechanism which has stopped helping must still not be actively
    damaging the half of the model it shares. Taken against the `lambda = 0`
    arm, which is a tighter control than `fixmatch` would be — same recipe, same
    optimiser, same batches, one weight apart.
    """
    assert arms.declared.outcome_nll < 1.05 * arms.ablated.outcome_nll


def test_the_two_arms_saw_the_same_rows(arms: _Arms) -> None:
    """The pair is a pair: same seeds, same quota, same batches.

    `stateful-sampler` (`DESIGN.md` §11.4) is the ledger row that protects this,
    and CPL is exactly the kind of mechanism that would break it if the
    curriculum had been put in the sampler instead of in the objective — it
    reads the model to decide which rows *train*, never which rows are *drawn*.
    """
    assert torch.equal(
        arms.declared.result.checkpoint.trained_on_row_ids,
        arms.ablated.result.checkpoint.trained_on_row_ids,
    )
