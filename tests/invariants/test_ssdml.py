"""Tier 0 — the reviewed SSDML recipe and deterministic array action."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
import torch
from xty2.core import (
    CompileError,
    FeatureSpec,
    OutcomeSpec,
    Port,
    PseudoLabelAction,
    Recipe,
    Schema,
    TrainingError,
    XTYBatch,
    compile,
    resolve_rows,
)
from xty2.estimators import SSDMLATEAction
from xty2.recipes import SSDML_ENCODER_WIDTHS, ssdml

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "ssdml.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "ssdml.py"


def _schema(
    *, treatment_cardinality: int = 2, outcome: OutcomeSpec | None = None
) -> Schema:
    return Schema(
        features=tuple(FeatureSpec(f"x{column}", "continuous") for column in range(3)),
        treatment_cardinality=treatment_cardinality,
        outcome=outcome if outcome is not None else OutcomeSpec(),
    )


def _action(**overrides: object) -> SSDMLATEAction:
    values: dict[str, object] = {
        "num_treatments": 2,
        "outcome": OutcomeSpec(),
        "ridge_penalty": 0.001,
        "propensity_clip": (0.025, 0.975),
        "folds": 5,
        "max_irls_iterations": 100,
        "irls_relative_tolerance": 1e-8,
    }
    return SSDMLATEAction(**(values | overrides))  # type: ignore[arg-type]


def _oracle_batch(rows: int = 250) -> XTYBatch:
    generator = torch.Generator().manual_seed(11_001)
    x = torch.randn(rows, 3, generator=generator)
    treatment = (
        torch.rand(rows, generator=generator)
        < torch.sigmoid(0.7 * x[:, 0] - 0.4 * x[:, 1])
    ).long()
    y = 0.8 * x[:, 0] - 0.3 * x[:, 1] + treatment.float()
    return XTYBatch(
        x=x,
        t=treatment,
        y=y,
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(500, 500 + rows),
        fold_id=torch.arange(rows) % 5,
    )


def test_the_recipe_plans_cross_fit_then_array_fit() -> None:
    run = compile(ssdml(_schema()))
    assert run.graph.names == ("mlp_encoder", "categorical_propensity")
    assert [stage.name for stage in run.stages] == [
        "propensity_labels",
        "dml_ate",
    ]
    labels = run.stage("propensity_labels")
    dml = run.stage("dml_ate")
    assert labels.executor == "cross_fit"
    assert labels.steps == 500
    assert labels.trainable == run.graph.names
    assert isinstance(labels.action, PseudoLabelAction)
    assert labels.action.port == Port.T_GIVEN_X
    assert labels.action_uses_y is False
    assert dml.executor == "array_fit"
    assert dml.inputs == ("propensity_labels",)
    assert isinstance(dml.action, SSDMLATEAction)
    assert dml.passes == ()
    assert dml.objectives == ()
    assert dml.trainable == ()
    rendered = run.plan.render()
    assert "action uses raw y: false" in rendered
    assert "action: array fit ssdml_ate" in rendered


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(tree)
    )


def test_the_action_returns_complete_deterministic_held_out_state() -> None:
    batch = _oracle_batch()
    before = batch.clone()
    rows = resolve_rows(batch, "all")
    first = _action().fit(batch, rows, seed=17)
    second = _action().fit(batch, rows, seed=99)
    assert batch.equal_to(before)
    assert set(first) == {
        "ate",
        "diagnostic_standard_error",
        "influence_score",
        "g0_hat",
        "g1_hat",
        "m_hat",
        "row_id",
        "fold_id",
    }
    for name in first:
        assert torch.equal(first[name], second[name])
    assert float(first["ate"]) == pytest.approx(1.0, abs=0.02)
    assert float(first["influence_score"].mean()) == pytest.approx(0.0, abs=1e-6)
    assert bool((first["m_hat"] >= 0.025).all())
    assert bool((first["m_hat"] <= 0.975).all())
    assert torch.equal(first["row_id"], batch.row_id)
    assert batch.fold_id is not None
    assert torch.equal(first["fold_id"], batch.fold_id)
    assert torch.isfinite(first["diagnostic_standard_error"])


def test_the_action_rejects_unsupported_schema_and_runtime_inputs() -> None:
    with pytest.raises(CompileError, match="binary treatment only"):
        _action(num_treatments=3)
    with pytest.raises(CompileError, match="scalar continuous outcome"):
        _action(outcome=OutcomeSpec(shape=(1,)))

    batch = _oracle_batch()
    four_folds = batch.replace(fold_id=torch.arange(batch.batch_size) % 4)
    with pytest.raises(TrainingError, match="exactly five"):
        _action().fit(four_folds, resolve_rows(four_folds, "all"), seed=1)
    unresolved = batch.replace(
        t_observed=torch.zeros(batch.batch_size, dtype=torch.bool)
    )
    with pytest.raises(TrainingError, match="unresolved missing treatments"):
        _action().fit(unresolved, resolve_rows(unresolved, "all"), seed=1)


def test_logistic_nonconvergence_fails_loudly() -> None:
    batch = _oracle_batch()
    with pytest.raises(TrainingError, match="did not converge"):
        _action(max_irls_iterations=1).fit(
            batch,
            resolve_rows(batch, "all"),
            seed=1,
        )


def test_the_plan_names_the_reviewed_array_mechanics() -> None:
    hyperparameters = compile(ssdml(_schema())).plan.hyperparameters
    assert hyperparameters["architecture.widths_depths"] == {
        "mlp_encoder": SSDML_ENCODER_WIDTHS,
        "categorical_propensity": "linear 64 -> 2",
        "dml_ate": (
            "two linear ridge outcome nuisances and one logistic ridge "
            "propensity per fold"
        ),
    }
    assert hyperparameters["optimisation.weight_decay"] == {
        "propensity_labels": "none",
        "dml_ate": "ridge penalty 0.001 on nuisance slopes; intercept exempt",
    }
    assert hyperparameters["data.split_protocol"] == {
        "dml_ate": (
            "supplied fold_id with exactly five non-empty folds; every nuisance "
            "prediction held out"
        )
    }


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


def test_every_answered_card_key_reaches_the_plan() -> None:
    recipe: Recipe = ssdml(_schema())
    missing = sorted(_answered_card_keys() - set(compile(recipe).plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
