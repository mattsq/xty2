"""Tier 0 — the reviewed Mean Teacher recipe and card-plan boundary."""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xty2.core import (
    DEFAULT,
    CategoricalTreatment,
    CompileError,
    FeatureSpec,
    GaussianOutcome,
    LossError,
    OutcomeSpec,
    Port,
    Program,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    SigmoidRamp,
    TeacherSpec,
    XTYBatch,
    compile,
)
from xty2.objectives import ConsistencyLoss, ObservedTreatmentNLL
from xty2.recipes import (
    ENCODER_WIDTHS,
    OUTCOME_WIDTHS,
    cnflow,
    mean_teacher,
    tarnet_extension,
)
from xty2.training import EMATeacher
from xty2.views import FeatureMask

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "mean_teacher.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "mean_teacher.py"
PRESERVED = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)
STUDENT_X = Realisation(view="student_x")
TEACHER_X = Realisation(view="teacher_x", params="teacher")


def _schema(*, derived: bool = False) -> Schema:
    return Schema(
        features=(
            FeatureSpec("mass", "continuous"),
            FeatureSpec("speed", "continuous"),
            FeatureSpec(
                "momentum",
                "continuous",
                derived_from=("mass", "speed") if derived else (),
            ),
            FeatureSpec("site", "categorical", mutable=False),
        ),
        treatment_cardinality=3,
        outcome=OutcomeSpec(),
    )


def _batch(rows: int = 512) -> XTYBatch:
    x = torch.ones(rows, 4)
    x[:, 3] = 1.0
    observed = torch.arange(rows) % 3 == 0
    return XTYBatch(
        x=x,
        t=torch.arange(rows) % 3,
        y=torch.linspace(-1.0, 1.0, rows),
        t_observed=observed,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
        fold_id=torch.arange(rows) % 5,
        weight=torch.linspace(0.5, 1.5, rows),
    )


def _consistency(recipe: Recipe) -> ConsistencyLoss:
    objective = recipe.program[0].objectives[-1].objective
    assert isinstance(objective, ConsistencyLoss)
    return objective


def test_the_recipe_plans_exactly_the_three_reviewed_realisations() -> None:
    run = compile(mean_teacher(_schema()))
    assert run.graph.names == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )
    assert len(run.stages) == 1
    stage = run.stage("joint_fit")
    assert stage.steps == 3_000
    assert stage.stage.rows == "all"
    assert stage.trainable == run.graph.names
    assert [forward.realisation for forward in stage.passes] == [
        DEFAULT,
        STUDENT_X,
        TEACHER_X,
    ]
    assert [forward.components for forward in stage.passes] == [
        run.graph.names,
        ("mlp_encoder", "categorical_propensity"),
        ("mlp_encoder", "categorical_propensity"),
    ]


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_the_stage_has_exactly_the_four_reviewed_objectives() -> None:
    stage = compile(mean_teacher(_schema())).stage("joint_fit")
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        "missing_treatment_marginal_nll",
        "consistency",
    ]
    assert [objective.rows for objective in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("t_missing",),
        ("all",),
    ]
    assert [objective.reduction for objective in stage.objectives] == [
        "population",
        "population",
        "population",
        "population",
    ]
    assert stage.objectives[0].objective.requires == frozenset(
        {(Port.Y_GIVEN_XT, DEFAULT)}
    )
    assert stage.objectives[1].objective.requires == frozenset(
        {(Port.T_GIVEN_X, STUDENT_X)}
    )
    assert stage.objectives[2].objective.requires == frozenset(
        {(Port.Y_GIVEN_XT, DEFAULT), (Port.T_GIVEN_X, DEFAULT)}
    )
    assert _consistency(mean_teacher(_schema())).detaches == frozenset(
        {(Port.T_GIVEN_X, TEACHER_X)}
    )


def test_the_two_views_are_distinct_reproducible_feature_masks() -> None:
    recipe = mean_teacher(_schema())
    assert [view.name for view in recipe.views] == ["student_x", "teacher_x"]
    assert recipe.views[0] is not recipe.views[1]
    for view in recipe.views:
        assert view.preserves == PRESERVED
        assert view.transforms == (FeatureMask(p=0.1, columns=None, value=0.0),)
        assert "x" not in view.preserves

    batch = _batch()
    student = recipe.view("student_x").apply(batch, recipe.schema, rng_key=91)
    repeated = recipe.view("student_x").apply(batch, recipe.schema, rng_key=91)
    teacher = recipe.view("teacher_x").apply(batch, recipe.schema, rng_key=91)
    assert student.equal_to(repeated)
    assert not torch.equal(student.x, teacher.x)
    assert torch.equal(student.x[:, 3], batch.x[:, 3])
    assert torch.equal(teacher.x[:, 3], batch.x[:, 3])


def _momentum(columns: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return columns["mass"] * columns["speed"]


def test_derived_features_need_and_accept_explicit_recompute_rules() -> None:
    with pytest.raises(CompileError, match="makes derived column"):
        compile(mean_teacher(_schema(derived=True)))

    rule = RecomputeRule("momentum", _momentum, name="mass_times_speed")
    run = compile(mean_teacher(_schema(derived=True), recompute_rules=(rule,)))
    assert [view.recomputes for view in run.plan.views] == [
        ("momentum <- mass_times_speed",),
        ("momentum <- mass_times_speed",),
    ]


def test_the_teacher_and_both_ramps_match_the_card() -> None:
    stage = compile(mean_teacher(_schema())).stage("joint_fit")
    assert stage.teacher == TeacherSpec(
        decay=0.99,
        applies_to_buffers=False,
        train_mode=True,
        requires_grad=False,
        role="consistency_target",
    )
    marginal = stage.objectives[2].weight
    consistency = stage.objectives[3].weight
    assert marginal.describe() == "ramp 0.0 -> 0.5 over 1000 steps"
    assert isinstance(consistency, SigmoidRamp)
    assert consistency.end == 3.0
    assert consistency.steps == 40
    assert consistency.describe() == (
        "sigmoid ramp to 3.0 over 40 steps: 3.0 * exp(-5 * (1 - min(step/40, 1))^2)"
    )


def test_the_teacher_starts_as_an_exact_isolated_copy_and_supports_evaluation() -> None:
    recipe = mean_teacher(_schema())
    spec = recipe.program[0].teacher
    assert spec is not None
    teacher = EMATeacher(recipe.system, spec)
    assert teacher.graph is not recipe.system
    for name, value in recipe.system.state_dict().items():
        assert torch.equal(value, teacher.graph.state_dict()[name])
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())

    batch = _batch(32)
    with torch.no_grad():
        student_values = recipe.system.evaluate(
            batch, schema=recipe.schema, only=recipe.system.names
        )
        teacher_values = teacher.graph.evaluate(
            batch, schema=recipe.schema, only=teacher.graph.names
        )
    student_propensity = student_values[Port.T_GIVEN_X]
    teacher_propensity = teacher_values[Port.T_GIVEN_X]
    student_outcome = student_values[Port.Y_GIVEN_XT]
    teacher_outcome = teacher_values[Port.Y_GIVEN_XT]
    assert isinstance(student_propensity, CategoricalTreatment)
    assert isinstance(teacher_propensity, CategoricalTreatment)
    assert isinstance(student_outcome, GaussianOutcome)
    assert isinstance(teacher_outcome, GaussianOutcome)
    assert torch.equal(student_propensity.probs, teacher_propensity.probs)
    candidates = torch.arange(3).expand(batch.batch_size, 3)
    assert torch.equal(
        student_outcome.mean(candidates), teacher_outcome.mean(candidates)
    )


def test_the_architecture_and_optimiser_are_the_reviewed_p5_stack() -> None:
    recipe = mean_teacher(_schema())
    run = compile(recipe)
    hyperparameters = run.plan.hyperparameters
    assert hyperparameters["architecture.widths_depths"] == {
        "mlp_encoder": ENCODER_WIDTHS,
        "tarnet_head": f"3 independent heads, each {list(OUTCOME_WIDTHS)}",
        "categorical_propensity": "linear 200 -> 3",
    }
    assert hyperparameters["optimisation.lr"] == 1e-3
    assert hyperparameters["optimisation.lr_schedule"] == (
        "staircase 1.0 * 0.97^floor(step/100)"
    )
    assert hyperparameters["optimisation.weight_decay"] == (
        "0.0001 (components tarnet_head only; norm and bias exempt)"
    )

    torch.manual_seed(19)
    mean_graph = mean_teacher(_schema()).system
    torch.manual_seed(19)
    tarnet_graph = tarnet_extension(_schema()).system
    assert mean_graph.names == tarnet_graph.names
    for name, value in mean_graph.state_dict().items():
        assert torch.equal(value, tarnet_graph.state_dict()[name])


def test_default_supervision_keeps_the_p5_and_p7_plans_on_identity() -> None:
    objective = ObservedTreatmentNLL()
    assert objective.realisation == DEFAULT
    assert objective.requires == frozenset({(Port.T_GIVEN_X, DEFAULT)})
    for recipe in (tarnet_extension(_schema()), cnflow(_schema())):
        rendered = compile(recipe).plan.render()
        assert "requires  p(t|x) @ view=identity params=student" in rendered
        assert "student_x" not in rendered


def _answered_card_keys() -> set[str]:
    text = CARD.read_text(encoding="utf-8")
    section = text.split("## 4. Mechanics checklist", 1)[1].split(
        "## 5. Deviations from the paper", 1
    )[0]
    match = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert match is not None
    keys: set[str] = set()
    current = ""
    for line in match.group(1).splitlines():
        statement = line.split("#", 1)[0].rstrip()
        if not statement:
            continue
        indent = len(statement) - len(statement.lstrip())
        name, _, value = statement.strip().partition(":")
        if indent == 0:
            current = name
        elif indent == 2 and value.strip() != "n/a":
            keys.add(f"{current}.{name}")
    return keys


def test_every_answered_card_key_and_view_reaches_the_plan() -> None:
    plan = compile(mean_teacher(_schema())).plan
    missing = sorted(_answered_card_keys() - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    assert [view.name for view in plan.views] == ["student_x", "teacher_x"]
    assert all(
        view.transforms == ("FeatureMask(p=0.1, columns=all, value=0.0)",)
        for view in plan.views
    )
    assert "forward passes (3)" in plan.render()


def test_removing_either_view_is_a_named_compile_failure() -> None:
    recipe = mean_teacher(_schema())
    for views, missing in (
        (recipe.views[1:], "student_x"),
        (recipe.views[:1], "teacher_x"),
    ):
        with pytest.raises(CompileError, match=missing):
            compile(replace(recipe, views=views))


def test_detaching_the_student_instead_of_the_teacher_is_rejected() -> None:
    recipe = mean_teacher(_schema())
    stage = recipe.program[0]
    objective = _consistency(recipe)
    detached_student = replace(objective, stop_grad="left")
    changed = replace(
        stage.objectives[-1],
        objective=detached_student,
    )
    with pytest.raises(CompileError, match="teacher target"):
        compile(
            replace(
                recipe,
                program=Program(
                    (replace(stage, objectives=(*stage.objectives[:-1], changed)),)
                ),
            )
        )


def test_a_buffer_ema_mutation_fails_the_recipe_contract() -> None:
    recipe = mean_teacher(_schema())
    stage = recipe.program[0]
    assert stage.teacher is not None
    mutant = replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    teacher=replace(stage.teacher, applies_to_buffers=True),
                ),
            )
        ),
    )
    compiled = compile(mutant).stage("joint_fit")
    with pytest.raises(AssertionError):
        assert compiled.teacher == TeacherSpec(
            decay=0.99,
            applies_to_buffers=False,
            train_mode=True,
            requires_grad=False,
            role="consistency_target",
        )


def test_identity_on_both_consistency_sides_is_rejected() -> None:
    with pytest.raises(LossError, match="with itself"):
        ConsistencyLoss(
            port=Port.T_GIVEN_X,
            left=DEFAULT,
            right=DEFAULT,
            divergence="mse",
            stop_grad="right",
            rows="all",
        )
