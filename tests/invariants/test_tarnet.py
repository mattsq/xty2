"""Tier 0 — the reviewed TARNet components, recipe and card-plan boundary."""

import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from xty2.core import (
    CategoricalTreatment,
    CompileError,
    GaussianOutcome,
    GraphError,
    OutcomeSpec,
    Port,
    Program,
    Recipe,
    WeightDecay,
    check_outcome_distribution_contract,
    check_treatment_distribution_contract,
    compile,
)
from xty2.recipes import ENCODER_WIDTHS, tarnet

from tests.invariants.conftest import make_batch, make_schema

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "tarnet.md"


def test_the_recipe_is_exactly_one_graph_and_one_stage() -> None:
    run = compile(tarnet(make_schema()))
    assert run.graph.names == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )
    assert len(run.stages) == 1
    stage = run.stage("joint_fit")
    assert stage.stage.rows == "all"
    assert stage.steps == 3_000
    assert stage.trainable == run.graph.names
    assert stage.passes[0].components == run.graph.names
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        "missing_treatment_marginal_nll",
    ]
    assert [objective.rows for objective in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("t_missing",),
    ]


def test_the_gaussian_tarnet_recipe_rejects_a_categorical_outcome() -> None:
    schema = make_schema(outcome=OutcomeSpec(kind="categorical", num_classes=2))
    with pytest.raises(GraphError, match="supports only continuous outcomes"):
        tarnet(schema)


def test_the_real_heads_satisfy_both_distribution_contracts() -> None:
    schema = make_schema()
    batch = make_batch()
    state = compile(tarnet(schema)).state("joint_fit", batch)
    outcome = state.default[Port.Y_GIVEN_XT]
    propensity = state.default[Port.T_GIVEN_X]
    assert isinstance(outcome, GaussianOutcome)
    assert isinstance(propensity, CategoricalTreatment)
    check_outcome_distribution_contract(
        outcome, y=batch.y, num_treatments=schema.treatment_cardinality
    )
    check_treatment_distribution_contract(
        propensity, num_treatments=schema.treatment_cardinality
    )


def test_the_encoder_row_normalises_its_representation() -> None:
    state = compile(tarnet(make_schema())).state("joint_fit", make_batch())
    representation = state.default[Port.X_REPR]
    assert isinstance(representation, torch.Tensor)
    assert torch.allclose(
        representation.norm(dim=-1),
        torch.ones(representation.shape[0]),
        rtol=1e-5,
        atol=1e-6,
    )


def test_the_reference_initialisation_reaches_every_affine_layer() -> None:
    graph = tarnet(make_schema()).system
    for component in graph.components:
        for module in component.modules():
            if not isinstance(module, nn.Linear):
                continue
            assert module.bias is not None
            assert torch.equal(module.bias, torch.zeros_like(module.bias))
            expected = 0.1 / module.in_features**0.5
            # An empirical check over the actual tensor: wide enough not to
            # turn RNG fluctuation into a build failure, narrow enough to catch
            # torch's default Kaiming initialiser or an unscaled 0.1 normal.
            assert float(module.weight.detach().std()) == pytest.approx(
                expected, rel=0.3
            )


def test_the_plan_names_each_component_s_architecture() -> None:
    hyperparameters = compile(tarnet(make_schema())).plan.hyperparameters
    assert hyperparameters["architecture.widths_depths"] == {
        "mlp_encoder": ENCODER_WIDTHS,
        "tarnet_head": "3 independent heads, each [100, 100, 100]",
        "categorical_propensity": "linear 200 -> 3",
    }
    assert hyperparameters["architecture.output_parameterisation"] == {
        "tarnet_head": "K means; fixed Gaussian scale=1.0",
        "categorical_propensity": "K softmax logits",
    }
    assert hyperparameters["optimisation.lr_schedule"] == (
        "staircase 1.0 * 0.97^floor(step/100)"
    )
    assert hyperparameters["optimisation.weight_decay"] == (
        "0.0001 (components tarnet_head only; norm and bias exempt)"
    )


def test_only_tarnet_head_matrices_are_in_the_decayed_group() -> None:
    recipe = tarnet(make_schema())
    parameters = [
        (f"{component.name}.{name}", parameter)
        for component in recipe.system.components
        for name, parameter in component.named_parameters()
    ]
    optimiser = recipe.program[0].optimiser.build(parameters)
    decayed = {id(parameter) for parameter in optimiser.param_groups[0]["params"]}
    expected = {
        id(parameter)
        for name, parameter in parameters
        if name.startswith("tarnet_head.") and parameter.ndim >= 2
    }
    assert decayed == expected


def test_a_decay_scope_outside_the_stage_is_a_compile_error() -> None:
    recipe = tarnet(make_schema())
    stage = recipe.program[0]
    decay = WeightDecay(
        value=1e-4, on_norm_and_bias=False, components=("not_the_head",)
    )
    changed = replace(
        recipe,
        program=Program(
            (replace(stage, optimiser=replace(stage.optimiser, weight_decay=decay)),)
        ),
    )
    with pytest.raises(CompileError, match="scopes weight decay"):
        compile(changed)


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


def _assert_card_is_covered(recipe: Recipe) -> None:
    plan = compile(recipe).plan
    missing = sorted(_answered_card_keys() - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)


def test_every_answered_card_key_reaches_the_plan() -> None:
    _assert_card_is_covered(tarnet(make_schema()))


def test_the_card_cross_check_fails_when_a_real_binding_is_removed() -> None:
    recipe = tarnet(make_schema())
    stage = recipe.program[0]
    complete_case = replace(
        recipe,
        program=Program((replace(stage, objectives=stage.objectives[:2]),)),
    )
    with pytest.raises(AssertionError, match=r"gradients\.marginal_nll_grad_path"):
        _assert_card_is_covered(complete_case)
