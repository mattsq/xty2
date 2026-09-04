"""Tier 1 — SoftMatch against the card's paired constant-gate arm."""

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
from xty2.objectives import (
    ConfidenceGaussian,
    PseudoLabelTreatmentNLL,
    SoftWeightedTreatmentNLL,
    TruncatedGaussianWeighting,
)
from xty2.recipes import softmatch
from xty2.recipes.fixmatch import STRONG_X, WEAK_X
from xty2.recipes.softmatch import SOFTMATCH_TERM
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

CONSTANT_TAU = 0.95


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _constant_gate(recipe: Recipe, threshold: float = CONSTANT_TAU) -> Recipe:
    stage = recipe.program[0]
    gated = Weighted(
        PseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=WEAK_X,
            prediction=STRONG_X,
            threshold=threshold,
            sharpening="hard",
            stop_grad="target",
            rows="all",
        ),
        weight=stage.objectives[2].weight,
        reduction=stage.objectives[2].reduction,
    )
    objectives = (*stage.objectives[:2], gated, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _no_ua(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    objective = stage.objectives[2].objective
    assert isinstance(objective, SoftWeightedTreatmentNLL)
    weighting = replace(objective.weighting, alignment="none")
    changed = replace(
        stage.objectives[2], objective=replace(objective, weighting=weighting)
    )
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(*stage.objectives[:2], changed, *stage.objectives[3:]),
                ),
            )
        ),
    )


def _term(result: StageResult, step: int, name: str) -> object:
    return next(term for term in result.records[step].terms if term.name == name)


@dataclass(frozen=True)
class _Metrics:
    result: StageResult
    treatment_nll: float
    student_treatment_nll: float
    outcome_nll: float
    frequency_nll: float


def _evaluate(
    run: CompiledRun, result: StageResult, test: _Population, train: XTYBatch
) -> _Metrics:
    graph = run.graph
    schema = run.recipe.schema
    assert result.teacher is not None
    scaled = _on_the_training_scale(test, result)
    with torch.no_grad():
        teacher = result.teacher.graph.evaluate(
            scaled.batch, schema=schema, only=graph.names
        )
        propensity = teacher[Port.T_GIVEN_X]
        outcome = teacher[Port.Y_GIVEN_XT]
        student = graph.evaluate(scaled.batch, schema=schema, only=graph.names)[
            Port.T_GIVEN_X
        ]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(outcome, GaussianOutcome)
        assert isinstance(student, CategoricalTreatment)
        observed = train.t[train.t_observed]
        frequencies = torch.bincount(observed, minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch.batch_size, -1)
        return _Metrics(
            result=result,
            treatment_nll=float(F.nll_loss(propensity.log_probs, scaled.batch.t)),
            student_treatment_nll=float(F.nll_loss(student.log_probs, scaled.batch.t)),
            outcome_nll=float(-outcome.log_prob(scaled.batch.y, scaled.batch.t).mean()),
            frequency_nll=float(F.nll_loss(baseline, scaled.batch.t)),
        )


@dataclass(frozen=True)
class _Arms:
    soft: _Metrics
    constant: _Metrics


@pytest.fixture(scope="module")
def arms() -> _Arms:
    schema = _schema()
    train, test = _populations(SEPARATED, seed=90_001)
    torch.manual_seed(90_006)
    adaptive = softmatch(schema)
    torch.manual_seed(90_006)
    constant = _constant_gate(softmatch(schema))
    for name, value in adaptive.system.state_dict().items():
        assert torch.equal(value, constant.system.state_dict()[name])

    adaptive_run = compile(_with_steps(adaptive, STEPS))
    constant_run = compile(_with_steps(constant, STEPS))
    adaptive_result = run_stage(adaptive_run, "joint_fit", _dataset(train), seed=90_010)
    constant_result = run_stage(constant_run, "joint_fit", _dataset(train), seed=90_010)
    assert torch.equal(
        adaptive_result.checkpoint.trained_on_row_ids,
        constant_result.checkpoint.trained_on_row_ids,
    )
    return _Arms(
        soft=_evaluate(adaptive_run, adaptive_result, test, train),
        constant=_evaluate(constant_run, constant_result, test, train),
    )


def test_the_weighting_state_moves_and_the_loss_is_non_degenerate(arms: _Arms) -> None:
    first = _term(arms.soft.result, 0, SOFTMATCH_TERM)
    final = _term(arms.soft.result, -1, SOFTMATCH_TERM)
    assert 0.5 < final.diagnostics["quantity"] < 1.0  # type: ignore[attr-defined]
    assert final.diagnostics["mu_hat"] > 0.5  # type: ignore[attr-defined]
    assert final.diagnostics["sigma_squared"] < 1.0  # type: ignore[attr-defined]
    assert final.diagnostics["sigma_squared"] < first.diagnostics["sigma_squared"]  # type: ignore[attr-defined]
    state = arms.soft.result.objective_states[SOFTMATCH_TERM]
    assert isinstance(state, ConfidenceGaussian)
    assert state.last_observed_step == STEPS - 1


def test_softmatch_beats_the_constant_gate_on_the_smoke_seed(arms: _Arms) -> None:
    assert arms.soft.treatment_nll < arms.constant.treatment_nll
    assert arms.soft.student_treatment_nll < arms.constant.student_treatment_nll


def test_both_arms_learn_and_the_outcome_stack_is_not_damaged(arms: _Arms) -> None:
    for arm in (arms.soft, arms.constant):
        assert arm.treatment_nll < 0.75 * arm.frequency_nll
        assert arm.student_treatment_nll < 0.75 * arm.frequency_nll
    assert arms.soft.outcome_nll < 1.05 * arms.constant.outcome_nll


def test_the_no_ua_and_matched_gate_arms_are_expressible(arms: _Arms) -> None:
    no_ua = _no_ua(softmatch(_schema())).program[0].objectives[2].objective
    assert isinstance(no_ua, SoftWeightedTreatmentNLL)
    assert no_ua.weighting == TruncatedGaussianWeighting(
        decay=0.999, n_sigma=2, alignment="none"
    )
    state = arms.soft.result.objective_states[SOFTMATCH_TERM]
    assert isinstance(state, ConfidenceGaussian)
    matched = _constant_gate(softmatch(_schema()), state.mean)
    gate = matched.program[0].objectives[2].objective
    assert isinstance(gate, PseudoLabelTreatmentNLL)
    assert gate.threshold == pytest.approx(state.mean)
