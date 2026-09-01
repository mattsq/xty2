"""Tier 1 — UDA's four paired mechanism arms on the card's fixed DGP.

The full, no-consistency, no-sharpening and no-TSA arms share initial
parameters, data policy, sampler and view RNG.  This tier asserts wiring and a
useful propensity, not the ten-seed full-versus-no-consistency direction owned
by Tier 2.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
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
    Realisation,
    Recipe,
    Weighted,
    compile,
)
from xty2.objectives import (
    ConfidenceMaskedConsistencyLoss,
    ObservedTreatmentNLL,
    TrainingSignalAnnealedTreatmentNLL,
    UDAConfidenceThresholds,
)
from xty2.recipes import UDA_THRESHOLDS, uda
from xty2.training import ObjectiveLog, StageResult, run_stage

from tests.smoke.test_fixmatch import (
    SEPARATED,
    TEST_ROWS,
    TRAIN_ROWS,
    _dataset,
    _on_the_training_scale,
    _Population,
    _population,
    _schema,
)

BASE_SEED = 94_000
TRAIN_SEED = BASE_SEED + 1
TEST_SEED = BASE_SEED + 2
INITIALISATION_SEED = BASE_SEED + 6
STAGE_SEED = BASE_SEED + 10_000
STEPS = 3_000
CONSISTENCY = "uda_consistency"
TSA = "tsa_observed_treatment_nll"
WEAK_X = Realisation(view="weak_x")
WEAK_X_SECOND = Realisation(view="weak_x", draw=1)


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _populations() -> tuple[torch.Tensor, _Population, _Population]:
    train = _population(
        TRAIN_ROWS,
        seed=TRAIN_SEED,
        observed_treatments=TRAIN_ROWS,
        row_offset=0,
        low=SEPARATED,
    )
    test = _population(
        TEST_ROWS,
        seed=TEST_SEED,
        observed_treatments=TEST_ROWS,
        row_offset=10_000,
        low=SEPARATED,
    )
    # The latent cluster is not retained by XTYBatch.  For this symmetric
    # mixture its Bayes class boundary is sum(x[0:4]) = 0, which is enough to
    # report whether a view crosses the decision boundary.
    bayes_label = train.batch.x[:, :4].sum(dim=-1) > 0
    return bayes_label, train, test


def _replace_consistency(
    recipe: Recipe,
    *,
    weight: float | None = None,
    temperature: float | None = None,
    thresholds: UDAConfidenceThresholds | None = None,
    prediction: Realisation | None = None,
) -> Recipe:
    stage = recipe.program[0]
    weighted = stage.objectives[2]
    objective = weighted.objective
    assert isinstance(objective, ConfidenceMaskedConsistencyLoss)
    replacement_objective = objective
    if temperature is not None:
        replacement_objective = replace(
            replacement_objective, target_temperature=temperature
        )
    if thresholds is not None:
        replacement_objective = replace(replacement_objective, thresholds=thresholds)
    if prediction is not None:
        replacement_objective = replace(replacement_objective, prediction=prediction)
    replacement = replace(weighted, objective=replacement_objective)
    if weight is not None:
        replacement = replace(replacement, weight=Constant(weight))
    objectives = (*stage.objectives[:2], replacement, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _without_tsa(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    source = stage.objectives[1]
    replacement = Weighted(
        ObservedTreatmentNLL(realisation=WEAK_X),
        weight=source.weight,
        reduction=source.reduction,
    )
    objectives = (stage.objectives[0], replacement, *stage.objectives[2:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _replace_thresholds(recipe: Recipe, thresholds: UDAConfidenceThresholds) -> Recipe:
    stage = recipe.program[0]
    tsa = stage.objectives[1]
    consistency = stage.objectives[2]
    assert isinstance(tsa.objective, TrainingSignalAnnealedTreatmentNLL)
    assert isinstance(consistency.objective, ConfidenceMaskedConsistencyLoss)
    objectives = (
        stage.objectives[0],
        replace(tsa, objective=replace(tsa.objective, thresholds=thresholds)),
        replace(
            consistency,
            objective=replace(consistency.objective, thresholds=thresholds),
        ),
        *stage.objectives[3:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _one_step(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    return replace(recipe, program=Program((replace(stage, steps=1),)))


def _weak_to_weak(recipe: Recipe) -> Recipe:
    diagnostic = _replace_consistency(recipe, prediction=WEAK_X_SECOND)
    weak = diagnostic.view("weak_x")
    views = (replace(weak, draws=2),)
    return replace(diagnostic, views=views)


def _term(result: StageResult, step: int, name: str) -> ObjectiveLog:
    return next(term for term in result.records[step].terms if term.name == name)


def _all_finite(result: StageResult) -> bool:
    for record in result.records:
        values = [record.lr, record.total, record.grad_norm]
        for term in record.terms:
            values.extend((term.value, term.weight, term.weighted))
            values.extend(term.diagnostics.values())
        if not all(math.isfinite(float(value)) for value in values):
            return False
    return True


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    student_treatment_nll: float
    ema_treatment_nll: float
    ema_outcome_nll: float
    frequency_nll: float
    initial_coverage: float
    initial_accepted_confidence: float
    initial_target_entropy: float
    initial_consistency_loss: float
    terminal_coverage: float
    terminal_accepted_confidence: float
    terminal_target_entropy: float
    terminal_consistency_loss: float
    initial_tsa_fraction: float | None
    terminal_tsa_fraction: float | None
    terminal_tsa_ceiling: float | None
    weak_flip_rate: float
    strong_flip_rate: float


def _view_flip_rates(
    recipe: Recipe, train: _Population, label: torch.Tensor
) -> tuple[float, float]:
    weak = recipe.view("weak_x").apply(train.batch, recipe.schema, rng_key=STAGE_SEED)
    strong = recipe.view("strong_x").apply(
        train.batch, recipe.schema, rng_key=STAGE_SEED
    )
    weak_label = weak.x[:, :4].sum(dim=-1) > 0
    strong_label = strong.x[:, :4].sum(dim=-1) > 0
    return (
        float((weak_label != label).float().mean()),
        float((strong_label != label).float().mean()),
    )


def _evaluate(
    run: CompiledRun,
    result: StageResult,
    test: _Population,
    flip_rates: tuple[float, float],
) -> _Metrics:
    scaled_test = _on_the_training_scale(test, result)
    assert result.teacher is not None
    assert result.population is not None
    with torch.no_grad():
        student = run.graph.evaluate(
            scaled_test.batch, schema=run.recipe.schema, only=run.graph.names
        )
        ema = result.teacher.graph.evaluate(
            scaled_test.batch, schema=run.recipe.schema, only=run.graph.names
        )
        student_propensity = student[Port.T_GIVEN_X]
        ema_propensity = ema[Port.T_GIVEN_X]
        ema_outcome = ema[Port.Y_GIVEN_XT]
        assert isinstance(student_propensity, CategoricalTreatment)
        assert isinstance(ema_propensity, CategoricalTreatment)
        assert isinstance(ema_outcome, GaussianOutcome)

        training = result.population.rows
        frequencies = torch.bincount(
            training.t[training.t_observed], minlength=2
        ).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled_test.batch.batch_size, -1)
        consistency = _term(result, 0, CONSISTENCY)
        terminal_consistency = _term(result, result.steps - 1, CONSISTENCY)
        tsa = next((term for term in result.records[0].terms if term.name == TSA), None)
        terminal_tsa = next(
            (term for term in result.records[-1].terms if term.name == TSA), None
        )
        return _Metrics(
            run=run,
            result=result,
            student_treatment_nll=float(
                F.nll_loss(student_propensity.log_probs, scaled_test.batch.t)
            ),
            ema_treatment_nll=float(
                F.nll_loss(ema_propensity.log_probs, scaled_test.batch.t)
            ),
            ema_outcome_nll=float(
                -ema_outcome.log_prob(scaled_test.batch.y, scaled_test.batch.t).mean()
            ),
            frequency_nll=float(F.nll_loss(baseline, scaled_test.batch.t)),
            initial_coverage=float(consistency.diagnostics["coverage"]),
            initial_accepted_confidence=float(
                consistency.diagnostics["accepted_confidence"]
            ),
            initial_target_entropy=float(consistency.diagnostics["target_entropy"]),
            initial_consistency_loss=consistency.value,
            terminal_coverage=float(terminal_consistency.diagnostics["coverage"]),
            terminal_accepted_confidence=float(
                terminal_consistency.diagnostics["accepted_confidence"]
            ),
            terminal_target_entropy=float(
                terminal_consistency.diagnostics["target_entropy"]
            ),
            terminal_consistency_loss=terminal_consistency.value,
            initial_tsa_fraction=(
                float(tsa.diagnostics["retained_fraction"]) if tsa is not None else None
            ),
            terminal_tsa_fraction=(
                float(terminal_tsa.diagnostics["retained_fraction"])
                if terminal_tsa is not None
                else None
            ),
            terminal_tsa_ceiling=(
                float(terminal_tsa.diagnostics["tsa_ceiling"])
                if terminal_tsa is not None
                else None
            ),
            weak_flip_rate=flip_rates[0],
            strong_flip_rate=flip_rates[1],
        )


def _run(
    recipe: Recipe,
    train: _Population,
    test: _Population,
    flip_rates: tuple[float, float],
) -> _Metrics:
    run = compile(recipe)
    result = run_stage(run, "joint_fit", _dataset(train.batch), seed=STAGE_SEED)
    return _evaluate(run, result, test, flip_rates)


@pytest.fixture(scope="module")
def four_arms() -> Mapping[str, _Metrics]:
    label, train, test = _populations()
    torch.manual_seed(INITIALISATION_SEED)
    full = uda(_schema())
    flip_rates = _view_flip_rates(full, train, label)

    recipes: dict[str, Recipe] = {"full": full}
    torch.manual_seed(INITIALISATION_SEED)
    recipes["no_consistency"] = _replace_consistency(uda(_schema()), weight=0.0)
    torch.manual_seed(INITIALISATION_SEED)
    recipes["no_sharpening"] = _replace_consistency(uda(_schema()), temperature=1.0)
    torch.manual_seed(INITIALISATION_SEED)
    recipes["no_tsa"] = _without_tsa(uda(_schema()))

    reference = recipes["full"].system.state_dict()
    for recipe in recipes.values():
        for name, value in reference.items():
            assert torch.equal(value, recipe.system.state_dict()[name])
    return {
        name: _run(recipe, train, test, flip_rates) for name, recipe in recipes.items()
    }


def test_all_four_predeclared_arms_run_finite(
    four_arms: Mapping[str, _Metrics],
) -> None:
    assert set(four_arms) == {
        "full",
        "no_consistency",
        "no_sharpening",
        "no_tsa",
    }
    assert all(metrics.result.steps == STEPS for metrics in four_arms.values())
    assert all(_all_finite(metrics.result) for metrics in four_arms.values())


def test_full_uda_beats_the_observed_frequency_baseline(
    four_arms: Mapping[str, _Metrics],
) -> None:
    full = four_arms["full"]
    assert full.student_treatment_nll < full.frequency_nll
    assert full.ema_treatment_nll < full.frequency_nll


def test_sharpening_changes_entropy_and_not_step_zero_membership(
    four_arms: Mapping[str, _Metrics],
) -> None:
    full = four_arms["full"]
    ordinary = four_arms["no_sharpening"]
    assert full.initial_coverage == ordinary.initial_coverage
    assert full.initial_target_entropy < ordinary.initial_target_entropy


def test_tsa_fraction_and_view_label_flips_are_reported_without_a_direction(
    four_arms: Mapping[str, _Metrics],
) -> None:
    full = four_arms["full"]
    assert full.initial_tsa_fraction is not None
    assert full.terminal_tsa_fraction is not None
    assert full.terminal_tsa_ceiling is not None
    assert 0.0 <= full.initial_tsa_fraction <= 1.0
    assert 0.0 <= full.terminal_tsa_fraction <= 1.0
    assert 0.0 <= full.weak_flip_rate < full.strong_flip_rate <= 0.05


def test_the_evaluation_ema_is_not_a_training_target(
    four_arms: Mapping[str, _Metrics],
) -> None:
    for metrics in four_arms.values():
        teacher = metrics.result.teacher
        assert teacher is not None
        assert teacher.spec.role == "evaluation"
        assert all(not parameter.requires_grad for parameter in teacher.parameters())
        assert all(parameter.grad is None for parameter in teacher.parameters())


def test_the_two_diagnostic_arms_execute_without_performance_claims() -> None:
    _, train, _ = _populations()
    always = UDAConfidenceThresholds(
        unsupervised=0.0,
        tsa_schedule=UDA_THRESHOLDS.tsa_schedule,
        scale=UDA_THRESHOLDS.scale,
        total_steps=UDA_THRESHOLDS.total_steps,
    )
    torch.manual_seed(INITIALISATION_SEED)
    open_gate = _one_step(_replace_thresholds(uda(_schema()), always))
    open_result = run_stage(
        compile(open_gate), "joint_fit", _dataset(train.batch), seed=STAGE_SEED
    )
    assert _term(open_result, 0, CONSISTENCY).diagnostics["coverage"] == 1.0

    torch.manual_seed(INITIALISATION_SEED)
    weak_to_weak = _one_step(_weak_to_weak(uda(_schema())))
    weak_result = run_stage(
        compile(weak_to_weak), "joint_fit", _dataset(train.batch), seed=STAGE_SEED
    )
    term = _term(weak_result, 0, CONSISTENCY)
    assert math.isfinite(float(term.value))
    assert term.n == 448


def test_the_no_consistency_arm_changes_only_the_weight() -> None:
    torch.manual_seed(INITIALISATION_SEED)
    full = uda(_schema())
    ablated = _replace_consistency(full, weight=0.0)
    full_stage = full.program[0]
    ablated_stage = ablated.program[0]
    assert full_stage.objectives[2].objective == ablated_stage.objectives[2].objective
    full_weight = full_stage.objectives[2].weight
    ablated_weight = ablated_stage.objectives[2].weight
    assert isinstance(full_weight, Constant)
    assert isinstance(ablated_weight, Constant)
    assert full_weight.nominal == 1.0
    assert ablated_weight.nominal == 0.0
    assert full_stage.objectives[:2] == ablated_stage.objectives[:2]
    assert full_stage.objectives[3:] == ablated_stage.objectives[3:]


def test_the_no_tsa_arm_replaces_only_the_supervised_treatment_objective() -> None:
    torch.manual_seed(INITIALISATION_SEED)
    full = uda(_schema())
    ablated = _without_tsa(full)
    assert isinstance(
        full.program[0].objectives[1].objective,
        TrainingSignalAnnealedTreatmentNLL,
    )
    assert isinstance(ablated.program[0].objectives[1].objective, ObservedTreatmentNLL)
    assert full.program[0].objectives[0] == ablated.program[0].objectives[0]
    assert full.program[0].objectives[2:] == ablated.program[0].objectives[2:]
