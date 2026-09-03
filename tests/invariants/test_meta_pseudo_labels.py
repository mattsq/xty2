"""Tier 0 — Meta Pseudo Labels' roles and atomic update contract."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xty2.core import (
    META_GRADIENT_ORDER,
    CategoricalTreatment,
    CompiledRun,
    CompileError,
    Constant,
    Dataset,
    FeatureSpec,
    GradientClipping,
    GraphError,
    OptimiserSpec,
    Port,
    Program,
    Realisation,
    Recipe,
    Schema,
    State,
    TeacherSpec,
    WeightDecay,
    XTYBatch,
    compile,
)
from xty2.objectives import MetaFeedbackCoefficient, MetaPseudoLabelScore
from xty2.recipes import INNER_ROLE, OUTER_ROLE, meta_pseudo_labels
from xty2.training import RunDirectory, StageResult, run_meta_gradient, run_stage

CARD = Path("docs/recipes/meta_pseudo_labels.md")
RECIPE_SOURCE = Path("xty2/recipes/meta_pseudo_labels.py")


def _schema() -> Schema:
    return Schema(
        tuple(FeatureSpec(f"x{index}", "continuous") for index in range(6)),
        treatment_cardinality=2,
    )


def _dataset() -> Dataset:
    generator = torch.Generator().manual_seed(91)
    rows = 1_024
    cluster = (torch.rand(rows, generator=generator) < 0.5).float()
    x = torch.randn(rows, 6, generator=generator) * 0.6
    x[:, :4] += (2.0 * cluster - 1.0)[:, None] * 0.45
    t = (torch.rand(rows, generator=generator) < (0.02 + 0.96 * cluster)).long()
    batch = XTYBatch(
        x=x,
        t=t,
        y=torch.zeros(rows),
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
    )
    return Dataset(
        schema=_schema(), rows=batch, assignments={"train": torch.arange(rows)}
    )


def _short_recipe(*, steps: int = 2, force_zero: bool = False) -> Recipe:
    recipe = meta_pseudo_labels(_schema())
    stage = recipe.program[0]
    meta = stage.meta_gradient
    assert meta is not None
    feedback = meta.feedback
    assert isinstance(feedback, MetaFeedbackCoefficient)
    meta = replace(meta, feedback=replace(feedback, force_zero=force_zero))
    return replace(
        recipe,
        program=Program((replace(stage, steps=steps, meta_gradient=meta),)),
    )


def _run(
    *, steps: int = 2, force_zero: bool = False
) -> tuple[CompiledRun, StageResult]:
    run = compile(_short_recipe(steps=steps, force_zero=force_zero))
    result = run_meta_gradient(
        run,
        "meta_train",
        _dataset(),
        seed=104_000,
        role_seeds={OUTER_ROLE: 94_006, INNER_ROLE: 94_007},
        hard_label_seed=114_000,
    )
    return run, result


def _card_section_four() -> dict[str, str | dict[str, str]]:
    text = CARD.read_text(encoding="utf-8")
    match = re.search(r"## 4\..*?```yaml\n(.*?)\n```", text, re.DOTALL)
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


def _rendered(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def test_recipe_is_declarative_and_card_matches_every_planned_value() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(tree)
    )
    planned = compile(meta_pseudo_labels(_schema())).plan.hyperparameters
    mismatched: list[str] = []
    checked = 0
    for key, stated in _card_section_four().items():
        value = planned.get(key)
        if value is None:
            mismatched.append(f"{key}: absent from plan")
            continue
        if isinstance(stated, str):
            if not isinstance(value, Mapping) and _rendered(value) != stated:
                mismatched.append(f"{key}: card {stated!r} vs plan {value!r}")
            checked += 1
            continue
        assert isinstance(value, Mapping), f"{key} is scoped in the card only"
        for scope, expected in stated.items():
            if scope not in value:
                mismatched.append(f"{key}[{scope}]: absent from plan")
                continue
            if key == "architecture.widths_depths":
                expected = expected.replace("K", "2")
            if _rendered(value[scope]) != expected:
                mismatched.append(
                    f"{key}[{scope}]: card {expected!r} vs plan {value[scope]!r}"
                )
            checked += 1
    assert not mismatched, "card and plan disagree: " + "; ".join(mismatched)
    assert checked >= 45


def test_realisation_role_and_within_step_state_are_validated_and_rendered() -> None:
    realised = Realisation(view="weak_x", role=INNER_ROLE, state="post_update")
    assert str(realised).endswith("role=inner_student state=post_update")
    with pytest.raises(GraphError, match="role must be a Python identifier"):
        Realisation(role="not a role")
    with pytest.raises(GraphError, match=r"pre_update.*post_update"):
        Realisation(state="later")  # type: ignore[arg-type]


def test_plan_prints_roles_optimisers_order_states_and_card_keys() -> None:
    run = compile(meta_pseudo_labels(_schema()))
    plan = run.plan.render()
    assert "executor: meta_gradient" in plan
    assert " -> ".join(META_GRADIENT_ORDER) in plan
    assert "role outer_teacher" in plan
    assert "role inner_student" in plan
    assert "state=post_update" in plan
    assert "explicit hard_label_seed stream" in plan
    hyperparameters = run.plan.hyperparameters
    assert hyperparameters["optimisation.optimiser"] == {
        INNER_ROLE: "sgd(momentum=0.9, nesterov=True)",
        OUTER_ROLE: "sgd(momentum=0.9, nesterov=True)",
    }
    assert hyperparameters["optimisation.lr"] == {
        INNER_ROLE: 0.03,
        OUTER_ROLE: 0.03,
    }
    assert (
        hyperparameters["losses.weights"]["meta_train.student_labelled_feedback_nll"]
        == 0.0
    )
    assert hyperparameters["losses.temperature"] == {
        "student_pseudo_label_nll": 1.0,
        "teacher_uda_consistency": 0.4,
    }
    assert all(
        not run.graph.port_depends_on_raw_outcome(port)
        for port in (Port.X_REPR, Port.T_GIVEN_X)
    )


def test_meta_contract_rejects_unbounded_or_ambiguous_variants() -> None:
    recipe = meta_pseudo_labels(_schema())
    stage = recipe.program[0]
    meta = stage.meta_gradient
    assert meta is not None
    with pytest.raises(CompileError, match="exactly one inner"):
        replace(meta, inner_steps=2)
    with pytest.raises(CompileError, match="six-step update order"):
        replace(meta, update_order=tuple(reversed(META_GRADIENT_ORDER)))
    with pytest.raises(CompileError, match="cannot also maintain an EMA"):
        replace(
            stage,
            teacher=TeacherSpec(
                decay=0.9,
                applies_to_buffers=True,
                train_mode=False,
                requires_grad=False,
                role="evaluation",
            ),
        )


def test_ordinary_gradient_executor_cannot_run_role_tagged_objectives() -> None:
    recipe = meta_pseudo_labels(_schema())
    stage = recipe.program[0]
    optimiser = OptimiserSpec(
        name="sgd",
        lr=0.1,
        weight_decay=WeightDecay.none(),
        lr_schedule=Constant(1.0),
        clipping=GradientClipping.none(),
    )
    ordinary = replace(
        stage,
        executor="gradient",
        roles=(),
        meta_gradient=None,
        trainable=("mlp_encoder", "categorical_propensity"),
        optimiser=optimiser,
    )
    with pytest.raises(CompileError, match="cannot produce"):
        compile(replace(recipe, program=Program((ordinary,))))
    with pytest.raises(Exception, match="run_meta_gradient"):
        run_stage(compile(recipe), "meta_train", _dataset(), seed=0)


def test_feedback_is_cosine_similarity_with_updated_baseline_and_detached_h() -> None:
    coefficient = MetaFeedbackCoefficient(
        kind="cosine_similarity",
        baseline_decay=0.99,
        baseline_initial=0.0,
        baseline_order="update_then_subtract",
    )
    state = coefficient.new_state()
    pseudo = (torch.tensor([3.0, 0.0], requires_grad=True),)
    labelled = (torch.tensor([2.0, 0.0], requires_grad=True),)
    raw, centred, baseline = coefficient.compute(pseudo, labelled, state)
    assert float(raw) == pytest.approx(1.0)
    assert baseline == pytest.approx(0.01)
    assert float(centred) == pytest.approx(0.99)
    assert not centred.requires_grad
    zero, neutral, _ = coefficient.compute(
        (torch.zeros(2),), (torch.ones(2),), coefficient.new_state()
    )
    assert float(zero) == 0.0
    assert float(neutral) == 0.0


def test_positive_feedback_score_step_raises_sampled_class_probability() -> None:
    teacher = Realisation(view="strong_x", role=OUTER_ROLE)
    logits = torch.nn.Parameter(torch.tensor([[0.0, 0.0]]))
    before = torch.softmax(logits.detach(), dim=-1)[0, 1]
    state = State({teacher: {Port.T_GIVEN_X: CategoricalTreatment(logits)}})
    score = MetaPseudoLabelScore(
        port=Port.T_GIVEN_X,
        teacher=teacher,
        detached_target="hard categorical sample and centred feedback coefficient",
    )
    term = score.score_loss(state, torch.tensor([0]), torch.tensor([1]))
    (0.5 * term.value).backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    with torch.no_grad():
        logits.add_(logits.grad, alpha=-0.1)
    after = torch.softmax(logits.detach(), dim=-1)[0, 1]
    assert after > before


def test_score_function_gradient_matches_a_direct_hard_sample_calculation() -> None:
    teacher = Realisation(view="strong_x", role=OUTER_ROLE)
    sampled = torch.tensor([1, 0])
    feedback = torch.tensor(0.37)
    logits = torch.tensor([[0.2, -0.4], [-0.1, 0.7]], requires_grad=True)
    state = State({teacher: {Port.T_GIVEN_X: CategoricalTreatment(logits)}})
    score = MetaPseudoLabelScore(
        port=Port.T_GIVEN_X,
        teacher=teacher,
        detached_target="hard categorical sample and centred feedback coefficient",
    )
    implemented = (
        feedback * score.score_loss(state, torch.tensor([0, 1]), sampled).value
    )
    (implemented_gradient,) = torch.autograd.grad(implemented, logits)

    direct_logits = logits.detach().clone().requires_grad_()
    direct = (
        -feedback
        * torch.log_softmax(direct_logits, dim=-1)[torch.arange(2), sampled].mean()
    )
    (direct_gradient,) = torch.autograd.grad(direct, direct_logits)
    assert torch.allclose(implemented_gradient, direct_gradient)


def test_role_graphs_and_optimiser_effects_are_disjoint_and_deterministic() -> None:
    _, first = _run()
    _, second = _run()
    assert first.trace == second.trace
    assert set(first.role_graphs) == {INNER_ROLE, OUTER_ROLE}
    inner_ids = {
        id(parameter) for parameter in first.role_graphs[INNER_ROLE].parameters()
    }
    outer_ids = {
        id(parameter) for parameter in first.role_graphs[OUTER_ROLE].parameters()
    }
    assert inner_ids.isdisjoint(outer_ids)
    for role in (INNER_ROLE, OUTER_ROLE):
        left = first.role_checkpoints[role]
        right = second.role_checkpoints[role]
        assert left.seed == 104_000
        assert all(
            torch.equal(value, right.parameters[name])
            for name, value in left.parameters.items()
        )
        assert all(torch.isfinite(value).all() for value in left.parameters.values())
    assert any(
        not torch.equal(
            first.role_checkpoints[INNER_ROLE].parameters[name],
            first.role_checkpoints[OUTER_ROLE].parameters[name],
        )
        for name in first.role_checkpoints[INNER_ROLE].parameters
    )
    assert all(
        math.isfinite(value)
        for record in first.records
        for value in (
            record.total,
            *record.role_grad_norms.values(),
            *(
                diagnostic
                for term in record.terms
                for diagnostic in term.diagnostics.values()
            ),
        )
    )


def test_zero_feedback_is_identical_before_the_first_feedback_update() -> None:
    full_run, full = _run(steps=1)
    zero_run, zero = _run(steps=1, force_zero=True)
    full_terms = {term.name: term for term in full.records[0].terms}
    zero_terms = {term.name: term for term in zero.records[0].terms}
    for name in (
        "student_pseudo_label_nll",
        "student_labelled_feedback_nll",
        "teacher_tsa_nll",
        "teacher_uda_consistency",
    ):
        assert full_terms[name].value == zero_terms[name].value
    assert (
        full_terms["teacher_meta_score"].diagnostics["baseline"]
        == zero_terms["teacher_meta_score"].diagnostics["baseline"]
    )
    assert zero_terms["teacher_meta_score"].diagnostics["h"] == 0.0
    assert torch.equal(
        full.role_checkpoints[INNER_ROLE].parameters[
            "categorical_propensity.logits.weight"
        ],
        zero.role_checkpoints[INNER_ROLE].parameters[
            "categorical_propensity.logits.weight"
        ],
    )
    assert full_run.plan.digest != zero_run.plan.digest


def test_role_checkpoints_round_trip_under_explicit_role_paths(
    tmp_path: Path,
) -> None:
    run = compile(_short_recipe(steps=1))
    directory = RunDirectory.create(tmp_path / "run")
    result = run_meta_gradient(
        run,
        "meta_train",
        _dataset(),
        seed=104_000,
        role_seeds={OUTER_ROLE: 94_006, INNER_ROLE: 94_007},
        hard_label_seed=114_000,
        run_dir=directory,
    )
    for role, checkpoint in result.role_checkpoints.items():
        path = tmp_path / "run" / "stages" / "meta_train" / "roles" / role
        assert (path / "checkpoint.pt").is_file()
        loaded = directory.read_checkpoint("meta_train", role=role)
        assert loaded.plan_digest == checkpoint.plan_digest
        assert loaded.components == checkpoint.components
