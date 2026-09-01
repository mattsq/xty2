"""Tier 0 — UDA's two objectives and its card-plan boundary."""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

import pytest
import torch
from xty2.core import (
    DEFAULT,
    REQUIRED,
    CardKeyError,
    CategoricalTreatment,
    CosineDecay,
    FeatureSpec,
    LossError,
    Port,
    Realisation,
    Schema,
    State,
    StatefulObjective,
    TrainContext,
    compile,
)
from xty2.objectives import (
    ConfidenceMaskedConsistencyLoss,
    TrainingSignalAnnealedTreatmentNLL,
    UDAConfidenceThresholds,
)
from xty2.recipes import (
    TARGET_TEMPERATURE,
    UDA_STEPS,
    UDA_STRONG_MASK_RATE,
    UDA_THRESHOLDS,
    uda,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "uda.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "uda.py"
TARGET = Realisation(view="weak_x")
PREDICTION = Realisation(view="strong_x")


def _thresholds(
    *, unsupervised: float = 0.8, total_steps: int = 3_000
) -> UDAConfidenceThresholds:
    return UDAConfidenceThresholds(
        unsupervised=unsupervised,
        tsa_schedule="exp_schedule",
        scale=5.0,
        total_steps=total_steps,
    )


def _state(target_logits: torch.Tensor, prediction_logits: torch.Tensor) -> State:
    return State(
        {
            TARGET: {Port.T_GIVEN_X: CategoricalTreatment(target_logits)},
            PREDICTION: {Port.T_GIVEN_X: CategoricalTreatment(prediction_logits)},
        }
    )


def _consistency(**overrides: Any) -> ConfidenceMaskedConsistencyLoss:
    defaults: dict[str, Any] = {
        "port": Port.T_GIVEN_X,
        "target": TARGET,
        "prediction": PREDICTION,
        "thresholds": _thresholds(),
        "target_temperature": 0.4,
        "sharpening": "softmax_temperature",
        "stop_grad": "target",
        "divergence": "kl",
        "rows": "t_missing",
    }
    return ConfidenceMaskedConsistencyLoss(**(defaults | overrides))


def _tsa(**overrides: Any) -> TrainingSignalAnnealedTreatmentNLL:
    defaults: dict[str, Any] = {
        "port": Port.T_GIVEN_X,
        "realisation": TARGET,
        "thresholds": _thresholds(),
    }
    return TrainingSignalAnnealedTreatmentNLL(**(defaults | overrides))


def _context(step: int = 0) -> TrainContext:
    return TrainContext(global_step=step, schema=make_schema())


def _recipe_schema() -> Schema:
    return make_schema(
        features=tuple(FeatureSpec(f"x{column}", "continuous") for column in range(6)),
        treatment_cardinality=2,
    )


def _consistency_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    direction = torch.tensor([2.0, -1.0, 0.5]).expand(BATCH_SIZE, NUM_TREATMENTS)
    scale = torch.linspace(3.0, 0.0, BATCH_SIZE)[:, None]
    target = direction * scale
    prediction = (
        torch.flip(target, dims=(1,)) + torch.linspace(-0.2, 0.2, BATCH_SIZE)[:, None]
    )
    return target, prediction


def test_uda_consistency_matches_the_pinned_masked_direct_kl() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    rows = torch.tensor([4, 5, 6])
    term = _consistency().compute(
        _state(target_logits, prediction_logits), make_batch(), rows, _context()
    )
    ordinary = target_logits.softmax(dim=-1)
    sharpened = (target_logits / 0.4).softmax(dim=-1)
    prediction = prediction_logits.softmax(dim=-1)
    accepted = ordinary.max(dim=-1).values > 0.8
    expected = (sharpened * (sharpened.log() - prediction.log())).sum(
        dim=-1
    ) * accepted.float()
    assert torch.allclose(term.value, expected[rows].mean(), atol=1e-6)
    assert term.n == len(rows)


def test_temperature_changes_the_target_but_not_gate_membership() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    rows = torch.arange(BATCH_SIZE)
    state = _state(target_logits, prediction_logits)
    sharp = _consistency(target_temperature=0.4).compute(
        state, make_batch(), rows, _context()
    )
    ordinary = _consistency(target_temperature=1.0).compute(
        state, make_batch(), rows, _context()
    )
    assert sharp.diagnostics["coverage"] == ordinary.diagnostics["coverage"]
    assert sharp.diagnostics["target_entropy"] < ordinary.diagnostics["target_entropy"]


def test_tau_one_is_exactly_the_ordinary_weak_distribution() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    rows = torch.arange(BATCH_SIZE)
    term = _consistency(
        target_temperature=1.0, thresholds=_thresholds(unsupervised=0.0)
    ).compute(_state(target_logits, prediction_logits), make_batch(), rows, _context())
    target = target_logits.softmax(dim=-1)
    prediction = prediction_logits.softmax(dim=-1)
    expected = (target * (target.log() - prediction.log())).sum(dim=-1).mean()
    assert torch.allclose(term.value, expected)


def test_rejected_consistency_rows_remain_in_the_denominator() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    rows = torch.arange(BATCH_SIZE)
    objective = _consistency()
    term = objective.compute(
        _state(target_logits, prediction_logits), make_batch(), rows, _context()
    )
    confidence = target_logits.softmax(dim=-1).max(dim=-1).values
    accepted = confidence > 0.8
    assert 0 < int(accepted.sum()) < BATCH_SIZE
    target = (target_logits / 0.4).softmax(dim=-1)
    prediction = prediction_logits.softmax(dim=-1)
    per_row = (target * (target.log() - prediction.log())).sum(dim=-1)
    assert torch.allclose(term.value, (per_row * accepted.float()).mean())
    assert not torch.allclose(term.value, per_row[accepted].mean())


def test_an_all_rejected_consistency_batch_is_zero_with_nonzero_n() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    rows = torch.arange(BATCH_SIZE)
    term = _consistency(thresholds=_thresholds(unsupervised=1.0)).compute(
        _state(target_logits, prediction_logits), make_batch(), rows, _context()
    )
    assert float(term.value) == 0.0
    assert term.n == BATCH_SIZE
    assert term.diagnostics["coverage"] == 0.0


def test_the_uda_gate_uses_strict_greater_than() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    confidence = target_logits.softmax(dim=-1).max(dim=-1).values
    row = int(confidence.argmin())
    threshold = float(confidence[row])
    term = _consistency(thresholds=_thresholds(unsupervised=threshold)).compute(
        _state(target_logits, prediction_logits),
        make_batch(),
        torch.tensor([row]),
        _context(),
    )
    assert term.diagnostics["coverage"] == 0.0
    assert float(term.value) == 0.0


def test_consistency_gradients_reach_only_the_prediction() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    target_logits.requires_grad_()
    prediction_logits.requires_grad_()
    objective = _consistency(thresholds=_thresholds(unsupervised=0.0))
    term = objective.compute(
        _state(target_logits, prediction_logits),
        make_batch(),
        torch.arange(BATCH_SIZE),
        _context(),
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    assert target_logits.grad is None
    assert prediction_logits.grad is not None
    assert objective.detaches == frozenset({(Port.T_GIVEN_X, TARGET)})


def test_consistency_with_no_rows_uses_the_framework_zero_term() -> None:
    target_logits, prediction_logits = _consistency_inputs()
    term = _consistency().compute(
        _state(target_logits, prediction_logits),
        make_batch(),
        torch.zeros(0, dtype=torch.long),
        _context(),
    )
    assert term.n == 0
    assert float(term.value) == 0.0
    assert dict(term.diagnostics) == {}


def _tsa_logits() -> torch.Tensor:
    batch = make_batch()
    logits = torch.zeros(BATCH_SIZE, NUM_TREATMENTS)
    strengths = torch.tensor([3.0, -2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    logits[torch.arange(BATCH_SIZE), batch.t] = strengths
    return logits


def test_tsa_gates_on_true_class_and_averages_over_retained_rows() -> None:
    logits = _tsa_logits()
    batch = make_batch()
    rows = torch.tensor([0, 1, 2, 3])
    term = _tsa().compute(
        _state(logits, torch.zeros_like(logits)), batch, rows, _context()
    )
    probs = logits.softmax(dim=-1)
    correct = probs[torch.arange(BATCH_SIZE), batch.t]
    ceiling = _thresholds().tsa_ceiling(0, NUM_TREATMENTS)
    retained = correct <= ceiling
    per_row = -logits.log_softmax(dim=-1)[torch.arange(BATCH_SIZE), batch.t]
    expected = per_row[rows][retained[rows]].mean()
    assert 0 < int(retained[rows].sum()) < len(rows)
    assert torch.allclose(term.value, expected)
    assert term.n == len(rows)
    assert term.diagnostics["retained_fraction"] == pytest.approx(
        float(retained[rows].float().mean())
    )
    assert term.diagnostics["tsa_ceiling"] == pytest.approx(ceiling)


def test_tsa_uses_the_true_class_probability_not_the_argmax_confidence() -> None:
    batch = make_batch()
    logits = torch.zeros(BATCH_SIZE, NUM_TREATMENTS)
    # Row 0 predicts class 1 confidently while its true class is 0. TSA must
    # retain it because p(true class) is low, despite max(p) being high.
    logits[0] = torch.tensor([-3.0, 3.0, 0.0])
    term = _tsa().compute(
        _state(logits, torch.zeros_like(logits)),
        batch,
        torch.tensor([0]),
        _context(),
    )
    assert term.diagnostics["retained_fraction"] == 1.0
    assert float(term.value) > 0.0


def test_tsa_all_dropped_is_zero_and_keeps_the_declared_row_count() -> None:
    batch = make_batch()
    logits = torch.full((BATCH_SIZE, NUM_TREATMENTS), -10.0)
    logits[torch.arange(BATCH_SIZE), batch.t] = 10.0
    rows = torch.tensor([0, 1, 2, 3])
    term = _tsa().compute(
        _state(logits, torch.zeros_like(logits)), batch, rows, _context()
    )
    assert float(term.value) == 0.0
    assert term.n == len(rows)
    assert term.diagnostics["retained_fraction"] == 0.0


def test_the_exponential_tsa_ceiling_matches_the_source_formula() -> None:
    policy = _thresholds()
    observed = [policy.tsa_ceiling(step, NUM_TREATMENTS) for step in (0, 1_500, 3_000)]
    expected = [
        math.exp(5.0 * (step / 3_000 - 1.0)) * (1.0 - 1 / NUM_TREATMENTS)
        + 1 / NUM_TREATMENTS
        for step in (0, 1_500, 3_000)
    ]
    assert observed == pytest.approx(expected)
    assert observed[0] > 1 / NUM_TREATMENTS
    assert observed[0] < observed[1] < observed[2] == pytest.approx(1.0)


def test_tsa_descends_the_supervised_branch_and_is_batch_coupled() -> None:
    logits = _tsa_logits().requires_grad_()
    objective = _tsa()
    term = objective.compute(
        _state(logits, torch.zeros_like(logits)),
        make_batch(),
        torch.tensor([0, 1, 2, 3]),
        _context(),
    )
    term.value.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert objective.detaches == frozenset()
    assert objective.batch_coupled


def test_tsa_with_no_rows_uses_the_framework_zero_term() -> None:
    logits = _tsa_logits()
    term = _tsa().compute(
        _state(logits, torch.zeros_like(logits)),
        make_batch(),
        torch.zeros(0, dtype=torch.long),
        _context(),
    )
    assert term.n == 0
    assert float(term.value) == 0.0
    assert dict(term.diagnostics) == {}


def test_paper_governed_uda_fields_have_no_usable_defaults() -> None:
    arguments: dict[str, object] = {
        "port": Port.T_GIVEN_X,
        "target": TARGET,
        "prediction": PREDICTION,
        "thresholds": _thresholds(),
        "target_temperature": 0.4,
        "sharpening": "softmax_temperature",
        "stop_grad": "target",
    }
    for missing in ("thresholds", "target_temperature", "sharpening", "stop_grad"):
        mutant = dict(arguments)
        del mutant[missing]
        with pytest.raises(CardKeyError, match="no usable default"):
            ConfidenceMaskedConsistencyLoss(**mutant)  # type: ignore[arg-type]
    with pytest.raises(CardKeyError, match="no usable default"):
        TrainingSignalAnnealedTreatmentNLL(
            port=Port.T_GIVEN_X,
            realisation=TARGET,
            thresholds=REQUIRED,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"prediction": TARGET}, "with itself"),
        ({"port": Port.X_REPR}, "treatment distributions"),
        ({"target_temperature": 0.0}, "positive"),
        ({"sharpening": "hard"}, "softmax_temperature"),
        ({"stop_grad": "none"}, "must be 'target'"),
        ({"divergence": "mse"}, "must be 'kl'"),
        ({"rows": "confident"}, "unknown row population"),
    ],
)
def test_invalid_consistency_configurations_are_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(LossError, match=message):
        _consistency(**overrides)


def test_invalid_threshold_policies_are_rejected() -> None:
    with pytest.raises(LossError, match=r"in \[0, 1\]"):
        _thresholds(unsupervised=1.1)
    with pytest.raises(LossError, match="exp_schedule"):
        UDAConfidenceThresholds(0.8, "linear_schedule", 5.0, 3_000)  # type: ignore[arg-type]
    with pytest.raises(LossError, match="positive"):
        UDAConfidenceThresholds(0.8, "exp_schedule", 0.0, 3_000)
    with pytest.raises(LossError, match="at least 1"):
        UDAConfidenceThresholds(0.8, "exp_schedule", 5.0, 0)


def test_the_recipe_compiles_the_reviewed_stage_and_realisations() -> None:
    run = compile(uda(_recipe_schema()))
    stage = run.stage("joint_fit")
    assert run.graph.names == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )
    assert stage.steps == UDA_STEPS
    assert sorted(str(item.realisation) for item in stage.passes) == sorted(
        str(realisation) for realisation in (DEFAULT, TARGET, PREDICTION)
    )
    assert stage.teacher is not None
    assert stage.teacher.role == "evaluation"
    assert not any(item.realisation.params == "teacher" for item in stage.passes)
    assert [item.name for item in stage.objectives] == [
        "observed_outcome_nll",
        "tsa_observed_treatment_nll",
        "uda_consistency",
        "missing_treatment_marginal_nll",
    ]
    assert [item.rows for item in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("t_missing",),
        ("t_missing",),
    ]
    assert [item.reduction for item in stage.objectives] == [
        "population",
        "mean",
        "mean",
        "population",
    ]
    assert [item.weight.nominal for item in stage.objectives] == [
        1.0,
        1.0,
        1.0,
        0.5,
    ]
    assert not any(
        isinstance(item.objective, StatefulObjective) for item in stage.objectives
    )


def test_the_recipe_is_declarations_only() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(tree)
    )


def test_the_recipe_carries_the_reviewed_views_and_shared_threshold_policy() -> None:
    recipe = uda(_recipe_schema())
    assert [view.name for view in recipe.views] == ["weak_x", "strong_x"]
    assert recipe.views[0].transforms == (FeatureMask(p=0.1, columns=None, value=0.0),)
    assert recipe.views[1].transforms == (
        FeatureMask(p=0.1, columns=None, value=0.0),
        FeatureMask(p=UDA_STRONG_MASK_RATE, columns=None, value=0.0),
    )
    tsa = recipe.program[0].objectives[1].objective
    consistency = recipe.program[0].objectives[2].objective
    assert isinstance(tsa, TrainingSignalAnnealedTreatmentNLL)
    assert isinstance(consistency, ConfidenceMaskedConsistencyLoss)
    assert tsa.thresholds is UDA_THRESHOLDS
    assert consistency.thresholds is UDA_THRESHOLDS
    assert consistency.target_temperature == TARGET_TEMPERATURE


def test_the_reviewed_optimisation_and_teacher_reach_the_plan() -> None:
    run = compile(uda(_recipe_schema()))
    hyperparameters = run.plan.hyperparameters
    assert hyperparameters["optimisation.optimiser"] == (
        "sgd(momentum=0.9, nesterov=True)"
    )
    assert hyperparameters["optimisation.lr"] == 0.03
    assert hyperparameters["optimisation.weight_decay"] == (
        "0.0005 (all trainable components; all parameters)"
    )
    assert hyperparameters["optimisation.batch_size"] == 512
    assert hyperparameters["optimisation.labelled_unlabelled_ratio"] == 7.0
    assert hyperparameters["teacher.ema_decay"] == 0.9999
    assert hyperparameters["teacher.ema_applies_to_buffers"] is True
    schedule = run.stage("joint_fit").stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    assert schedule.steps == UDA_STEPS
    assert schedule.phase == 7 / 16


def _card_section_four() -> dict[str, str | dict[str, str]]:
    text = CARD.read_text(encoding="utf-8")
    section = text.split("## 4. Mechanics checklist", 1)[1].split(
        "## 5. Deviations from the paper", 1
    )[0]
    match = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert match is not None
    answered: dict[str, str | dict[str, str]] = {}
    current = ""
    key = ""
    for line in match.group(1).splitlines():
        statement = line.split("#", 1)[0].rstrip()
        if not statement:
            continue
        indent = len(statement) - len(statement.lstrip())
        name, _, value = statement.strip().partition(":")
        if indent == 0:
            current = name
        elif indent == 2:
            key = f"{current}.{name}"
            if value.strip() == "n/a":
                key = ""
                continue
            answered[key] = value.strip()
        elif indent == 4 and key:
            nested = answered.get(key)
            if not isinstance(nested, dict):
                nested = {}
                answered[key] = nested
            nested[name] = value.strip()
    return answered


def test_every_answered_card_key_reaches_the_plan() -> None:
    hyperparameters = compile(uda(_recipe_schema())).plan.hyperparameters
    answered = _card_section_four()
    assert not sorted(set(answered) - set(hyperparameters))
    assert repr(hyperparameters["losses.confidence_threshold"]) == (
        "uda(unsupervised=0.8, tsa=exp_schedule(scale=5, steps=3000))"
    )
    assert hyperparameters["losses.temperature"] == 0.4
    assert hyperparameters["losses.sharpening"] == "softmax_temperature"
    assert hyperparameters["gradients.detached_targets"] == "target"
    assert hyperparameters["losses.schedules"] == {
        "joint_fit.observed_outcome_nll": "constant 1.0",
        "joint_fit.tsa_observed_treatment_nll": "constant 1.0",
        "joint_fit.uda_consistency": "constant 1.0",
        "joint_fit.missing_treatment_marginal_nll": ("ramp 0.0 -> 0.5 over 1000 steps"),
    }


def test_plan_details_expose_both_gates_and_denominators() -> None:
    plan = compile(uda(_recipe_schema())).plan
    rendered = plan.render()
    assert "max untempered target probability > 0.8" in rendered
    assert "target = softmax(log(p_target) / 0.4)" in rendered
    assert "rejected rows contribute 0" in rendered
    assert "retain true-class probability <= TSA ceiling" in rendered
    assert "retained eligible rows, clamped at 1" in rendered
    mutant = _consistency(target_temperature=1.0)
    assert mutant.plan_details() != _consistency().plan_details()


def test_the_architecture_is_the_reviewed_project_local_stack() -> None:
    hyperparameters = compile(uda(_recipe_schema())).plan.hyperparameters
    assert hyperparameters["architecture.widths_depths"] == {
        "mlp_encoder": ENCODER_WIDTHS,
        "tarnet_head": (f"2 independent heads, each {list(OUTCOME_WIDTHS)}"),
        "categorical_propensity": f"linear {ENCODER_WIDTHS[-1]} -> 2",
    }


def test_the_shared_threshold_policy_is_stable_and_diffable() -> None:
    assert repr(UDA_THRESHOLDS) == (
        "uda(unsupervised=0.8, tsa=exp_schedule(scale=5, steps=3000))"
    )
    assert _thresholds() == UDA_THRESHOLDS
    assert _thresholds(unsupervised=0.7) != UDA_THRESHOLDS


def test_the_card_records_the_review_and_both_review_corrections() -> None:
    text = CARD.read_text(encoding="utf-8")
    flattened = " ".join(text.split())
    assert "without imposing a direction" in flattened
    assert "cannot be attributed to the marginal term" in flattened
    assert "ChatGPT" in text
