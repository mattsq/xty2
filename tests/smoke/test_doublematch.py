"""Tier 1 — DoubleMatch on `fixmatch.md` §6.1's fixture, paired against `w_s = 0`.

The DGP, the label budget, the quota and the seeds are `fixmatch`'s, and they
are **imported** rather than restated: `doublematch.md` §6.1 says the fixture is
that one "in full and without modification", and a copy would be a second thing
to keep true. What is new here is what the run is measured on. Eq. (3) has no
gate, so mask rate and impurity say nothing about it; the questions are whether
it learns anything, whether it does so without collapsing the representation —
which its own loss cannot answer, because a collapsed encoder attains the best
possible value — and what it then does to the propensity and the outcome stack.
The last of those came out mixed on one seed, and the test that reports it says
so rather than asserting the half that looks better (card §6.2).

The last test is the one that produced deviation 9. It is kept executable so
that the encoder's output normalisation cannot quietly come back.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.components import MLPEncoder
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    ComponentGraph,
    Constant,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    XTYBatch,
    compile,
)
from xty2.recipes import doublematch
from xty2.recipes.tarnet import ENCODER_WIDTHS
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

SELF_SUPERVISED = "cosine_feature_consistency"
PSEUDO_LABEL = "pseudo_label_treatment_nll"
COLLAPSE_STEPS = 200
BATCH_ROWS = 512
"""`B + mu B`, the quota the imported fixture's recipe declares."""


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _with_weight(recipe: Recipe, weight: float) -> Recipe:
    """`w_s`, and nothing else: the paired ablation of eq. (4)."""
    stage = recipe.program[0]
    term = replace(stage.objectives[3], weight=Constant(weight))
    objectives = (*stage.objectives[:3], term, *stage.objectives[4:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _with_row_l2_encoder(recipe: Recipe) -> Recipe:
    """The shared P5 backbone this recipe deviates from (deviation 9)."""
    encoder = MLPEncoder(
        input_dim=recipe.schema.num_features,
        widths=ENCODER_WIDTHS,
        activation="elu",
        normalisation="row_l2",
        dropout=0.0,
        initialisation=CFRNET_INITIALISATION,
    )
    return replace(
        recipe,
        system=ComponentGraph([encoder, *list(recipe.system.components)[1:]]),
    )


def _term(result: StageResult, step: int, name: str) -> object:
    return next(term for term in result.records[step].terms if term.name == name)


def _mean(result: StageResult, steps: range, name: str, field: str) -> float:
    terms = [_term(result, step, name) for step in steps]
    values: list[float]
    if field == "value":
        values = [float(term.value) for term in terms]  # type: ignore[attr-defined]
    else:
        values = [float(term.diagnostics[field]) for term in terms]  # type: ignore[attr-defined]
    return sum(values) / len(values)


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
    """The EMA the paper reports from, and the network it averages.

    Both, deliberately. `fixmatch.md` §6.2 records that an EMA number can
    improve while the network under it gets worse, so a claim resting on the
    EMA alone is a claim about a reporting device. Eq. (3) trains the encoder
    both arms share, which is exactly where that divergence would hide.
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


@pytest.fixture(scope="module")
def paired_fit() -> tuple[_Metrics, _Metrics]:
    """DoubleMatch and its `w_s = 0` ablation, same seeds and same batches."""
    schema = _schema()
    train, test = _populations(SEPARATED, seed=90_001)

    torch.manual_seed(90_006)
    scheduled = doublematch(schema)
    torch.manual_seed(90_006)
    ablated = _with_weight(doublematch(schema), 0.0)
    for name, value in scheduled.system.state_dict().items():
        assert torch.equal(value, ablated.system.state_dict()[name])

    return (
        _run(scheduled, train, test, STEPS),
        _run(ablated, train, test, STEPS),
    )


def test_the_self_supervised_term_learns_the_invariance(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Eq. (3) is descended, and the ablation's copy of it is not.

    The ablated arm computes the same term at weight 0, so its value is a
    read-out of what the other objectives do to the representation on their
    own. That it stays near zero — orthogonal, on average — is what makes the
    scheduled arm's alignment attributable to the term rather than to training.
    """
    scheduled, ablated = paired_fit
    late = range(STEPS - 100, STEPS)
    assert _mean(scheduled.result, range(1), SELF_SUPERVISED, "value") > -0.2
    assert _mean(scheduled.result, late, SELF_SUPERVISED, "value") < -0.5
    assert abs(_mean(ablated.result, late, SELF_SUPERVISED, "value")) < 0.2


def test_the_representation_does_not_collapse_under_the_term(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The card's §6 guardrail, and the reason the diagnostics exist.

    A collapsed encoder attains eq. (3)'s minimum, so "the loss went down" is
    compatible with the worst possible outcome. The term does concentrate the
    representation relative to the ablation — that is what an invariance is —
    and the claim is that it stops well short of one direction.
    """
    scheduled, ablated = paired_fit
    late = range(STEPS - 100, STEPS)
    concentration = _mean(
        scheduled.result, late, SELF_SUPERVISED, "target_concentration"
    )
    assert concentration < 0.8
    assert concentration > _mean(
        ablated.result, late, SELF_SUPERVISED, "target_concentration"
    )


def test_the_term_trains_on_rows_the_gate_rejects(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The paper's title claim, as an assertion about one step's row counts.

    Eq. (2) keeps a fraction of the batch and eq. (3) keeps all of it, in the
    same step, from the same two forward passes. This is `BACKLOG.md` §2.6's
    "row eligibility for one loss must not determine eligibility for all
    losses", and it needs no framework mechanism beyond `Objective.rows`.
    """
    scheduled, _ = paired_fit
    late = range(STEPS - 100, STEPS)
    coverage = _mean(scheduled.result, late, PSEUDO_LABEL, "coverage")
    assert 0.0 < coverage < 1.0
    for step in (0, STEPS // 2, STEPS - 1):
        assert _term(scheduled.result, step, SELF_SUPERVISED).n == BATCH_ROWS  # type: ignore[attr-defined]
        assert _term(scheduled.result, step, PSEUDO_LABEL).n == BATCH_ROWS  # type: ignore[attr-defined]


def test_the_projection_head_is_actually_trained(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """`theta_h` is in eq. (6) and in the optimiser, so it must have moved."""
    scheduled, _ = paired_fit
    initial = scheduled.run.initial_parameters()
    trained = dict(scheduled.run.graph.named_parameters())
    moved = [
        name
        for name in initial
        if "projection_head" in name and not torch.equal(initial[name], trained[name])
    ]
    assert moved


def test_the_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, _ = paired_fit
    assert scheduled.treatment_nll < scheduled.frequency_nll


def test_both_arms_learn_a_propensity_worth_having(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The floor, asserted before anything is compared.

    A paired ratio between two models that have both learned nothing is a
    number with no content — and §6.2's `row_l2` table is exactly that failure,
    so this is not hypothetical.
    """
    scheduled, ablated = paired_fit
    assert scheduled.treatment_nll < 0.75 * scheduled.frequency_nll
    assert ablated.treatment_nll < 0.75 * ablated.frequency_nll


def test_eq_three_moves_the_held_out_propensity_and_the_two_readings_disagree(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The card's §6 metric on one seed, and the reason it is not a result.

    The EMA is the model §IV of the paper reports from and the one §6 declares,
    and on it eq. (3) wins here. The network the EMA averages goes the other
    way by more. Both directions are asserted, because both were measured and
    the disagreement is the finding: a single-seed claim about eq. (3) on this
    fixture is not supportable in either direction, which is why §6's target is
    a ten-seed mean and why this card claims the mechanism rather than the
    improvement (§6.2).

    A second initialisation of the same architecture agrees on the first line
    and flips the second, so neither margin should be read as a property of the
    method. What this test pins is that the recipe still produces the numbers
    the card records.
    """
    scheduled, ablated = paired_fit
    assert scheduled.treatment_nll < ablated.treatment_nll
    assert scheduled.student_treatment_nll > ablated.student_treatment_nll


def test_the_outcome_stack_is_not_damaged(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Eq. (3) trains the encoder the outcome head reads, so this is not free."""
    scheduled, ablated = paired_fit
    assert scheduled.outcome_nll < 1.05 * ablated.outcome_nll


def test_a_unit_sphere_representation_collapses_under_eq_three() -> None:
    """Deviation 9, kept executable.

    With CFRNet's `row_l2` on the encoder's output — the backbone every other
    xty2 recipe shares — eq. (3) drives the whole representation to one
    direction within a few dozen steps, and it never comes back: the cosine is
    the entire geometry of a unit-sphere representation, so agreement and
    collapse are the same move. The supervised cross-entropy sits at `log 2`
    and the confidence gate never opens.

    Two hundred steps is enough because the failure is immediate; the card's
    §6.2 records the same run at the full budget and at four values of `w_s`.
    """
    train, test = _populations(SEPARATED, seed=90_001)
    torch.manual_seed(90_006)
    metrics = _run(
        _with_row_l2_encoder(doublematch(_schema())), train, test, COLLAPSE_STEPS
    )
    late = range(COLLAPSE_STEPS - 50, COLLAPSE_STEPS)
    assert _mean(metrics.result, late, SELF_SUPERVISED, "target_concentration") > 0.99
    assert _mean(metrics.result, late, SELF_SUPERVISED, "value") < -0.99
    assert _mean(metrics.result, late, PSEUDO_LABEL, "coverage") == 0.0
    assert metrics.treatment_nll > 0.95 * metrics.frequency_nll
