"""Tier 0 — the reviewed cycle-dual posterior and staged recipe."""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xty2.components import CategoricalPosterior
from xty2.components._nn import TORCH_LINEAR_INITIALISATION
from xty2.core import (
    CategoricalTreatment,
    CompileError,
    FeatureSpec,
    GaussianOutcome,
    GraphError,
    LossError,
    OutcomeSpec,
    Port,
    Program,
    PseudoLabelAction,
    Recipe,
    Schema,
    XTYBatch,
    check_outcome_distribution_contract,
    check_treatment_distribution_contract,
    compile,
)
from xty2.objectives import ObservedTreatmentNLL
from xty2.recipes import (
    CYCLE_DUAL_ENCODER_WIDTHS,
    CYCLE_DUAL_OUTCOME_WIDTHS,
    CYCLE_DUAL_POSTERIOR_WIDTHS,
    cycle_dual,
)

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "cycle_dual.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "cycle_dual.py"


def _schema(*, outcome: OutcomeSpec | None = None) -> Schema:
    return Schema(
        features=tuple(FeatureSpec(f"x{column}", "continuous") for column in range(4)),
        treatment_cardinality=2,
        outcome=outcome if outcome is not None else OutcomeSpec(),
    )


def _batch() -> XTYBatch:
    rows = 15
    return XTYBatch(
        x=torch.randn(rows, 4),
        t=torch.arange(rows) % 2,
        y=torch.randn(rows),
        t_observed=torch.arange(rows) % 3 == 0,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(100, 100 + rows),
        fold_id=torch.arange(rows) % 5,
    )


def test_the_recipe_plans_the_reviewed_two_stage_transition() -> None:
    run = compile(cycle_dual(_schema()))
    assert run.graph.names == (
        "categorical_posterior",
        "mlp_encoder",
        "tarnet_head",
    )
    assert [stage.name for stage in run.stages] == [
        "posterior_labels",
        "outcome_fit",
    ]
    posterior = run.stage("posterior_labels")
    outcome = run.stage("outcome_fit")
    assert posterior.executor == "cross_fit"
    assert posterior.steps == 500
    assert posterior.trainable == ("categorical_posterior",)
    assert posterior.passes[0].components == ("categorical_posterior",)
    assert isinstance(posterior.action, PseudoLabelAction)
    assert posterior.action.port == Port.T_GIVEN_XY
    assert posterior.action.rows == "t_missing"
    assert posterior.action_uses_y is True
    assert [objective.name for objective in posterior.objectives] == [
        "observed_posterior_nll"
    ]
    assert posterior.objectives[0].objective.requires == frozenset(
        {(Port.T_GIVEN_XY, posterior.action.realisation)}
    )
    assert outcome.executor == "gradient"
    assert outcome.steps == 1_000
    assert outcome.inputs == ("posterior_labels",)
    assert outcome.trainable == ("mlp_encoder", "tarnet_head")
    assert outcome.passes[0].components == ("mlp_encoder", "tarnet_head")
    assert "action uses raw y: true" in run.plan.render()


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(tree)
    )


def test_the_posterior_and_outcome_heads_satisfy_their_contracts() -> None:
    schema = _schema()
    batch = _batch()
    run = compile(cycle_dual(schema))
    posterior_state = run.state("posterior_labels", batch)
    posterior = posterior_state.default[Port.T_GIVEN_XY]
    outcome_state = run.state("outcome_fit", batch)
    outcome = outcome_state.default[Port.Y_GIVEN_XT]
    assert isinstance(posterior, CategoricalTreatment)
    assert isinstance(outcome, GaussianOutcome)
    check_treatment_distribution_contract(posterior, num_treatments=2)
    check_outcome_distribution_contract(outcome, y=batch.y, num_treatments=2)


def test_observed_treatment_supervision_names_the_posterior_port() -> None:
    objective = ObservedTreatmentNLL(
        name="observed_posterior_nll",
        port=Port.T_GIVEN_XY,
    )
    assert objective.requires == frozenset({(Port.T_GIVEN_XY, objective.realisation)})
    with pytest.raises(LossError, match="T_GIVEN_X or T_GIVEN_XY"):
        ObservedTreatmentNLL(port=Port.X_RAW)
    with pytest.raises(GraphError, match="supports only continuous outcomes"):
        CategoricalPosterior(
            input_dim=4,
            num_treatments=2,
            outcome=OutcomeSpec(kind="categorical", num_classes=2),
            widths=(8,),
            activation="relu",
            normalisation="none",
            dropout=0.0,
            initialisation=TORCH_LINEAR_INITIALISATION,
            output_parameterisation="K softmax logits",
            standardisation="none",
            outcome_scaling="none",
            treatment_encoding="binary",
        )


def test_the_real_recipe_mutation_triggers_the_circular_fit_guard() -> None:
    recipe = cycle_dual(_schema())
    unsafe_source = replace(recipe.program[0], executor="gradient")
    mutant = replace(
        recipe,
        program=Program((unsafe_source, recipe.program[1])),
    )
    with pytest.raises(CompileError, match=r"circular q\(t\|x,y\)"):
        compile(mutant)


def test_the_plan_names_every_reviewed_architecture_choice() -> None:
    hyperparameters = compile(cycle_dual(_schema())).plan.hyperparameters
    assert hyperparameters["architecture.widths_depths"] == {
        "categorical_posterior": (
            f"concat(X_RAW, Y_RAW) -> {list(CYCLE_DUAL_POSTERIOR_WIDTHS)!r} -> K"
        ),
        "mlp_encoder": CYCLE_DUAL_ENCODER_WIDTHS,
        "tarnet_head": (
            f"2 independent heads, each {list(CYCLE_DUAL_OUTCOME_WIDTHS)!r}"
        ),
    }
    assert hyperparameters["architecture.activation"] == {
        "categorical_posterior": "relu",
        "mlp_encoder": "relu",
        "tarnet_head": "relu",
    }
    assert hyperparameters["architecture.initialisation"] == {
        "categorical_posterior": TORCH_LINEAR_INITIALISATION,
        "mlp_encoder": TORCH_LINEAR_INITIALISATION,
        "tarnet_head": TORCH_LINEAR_INITIALISATION,
    }
    assert hyperparameters["optimisation.total_steps_or_epochs"] == {
        "posterior_labels": 500,
        "outcome_fit": 1_000,
    }


def _answered_card_keys() -> set[str]:
    text = CARD.read_text(encoding="utf-8")
    section = text.split("## 4. Mechanics checklist", 1)[1].split(
        "## 5. Deviations from the papers", 1
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


def test_every_answered_card_key_reaches_the_plan() -> None:
    recipe: Recipe = cycle_dual(_schema())
    missing = sorted(_answered_card_keys() - set(compile(recipe).plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
