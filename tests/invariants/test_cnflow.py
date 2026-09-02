"""Tier 0 — the reviewed CNFlow component, recipe and Gate 1 boundary."""

import ast
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from xty2.components import ConditionalFlow, ConditionalFlowOutcome
from xty2.components._nn import TORCH_LINEAR_INITIALISATION
from xty2.components.density import (
    NFLOWS_INITIALISATION,
    RANDOM_PERMUTATION,
    STANDARD_NORMAL,
    _fixed_antithetic_standard_normal,
)
from xty2.core import (
    CategoricalTreatment,
    CompileError,
    ComponentGraph,
    ExternalBatches,
    GraphError,
    OutcomeSpec,
    Port,
    Recipe,
    TrainContext,
    check_outcome_distribution_contract,
    check_treatment_distribution_contract,
    compile,
    resolve_rows,
)
from xty2.evaluation.benchmarks.cnflow import _paired_recipes
from xty2.objectives import MissingTreatmentMarginalNLL
from xty2.recipes import CNFLOW_ENCODER_WIDTHS, cnflow

from tests.invariants.conftest import make_batch, make_schema

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "cnflow.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "cnflow.py"

FLOW_ARCHITECTURE = (
    "5 RQ-NSF autoregressive transforms, each hidden=128 with 2 residual blocks, "
    '8 bins, tails="linear", tail_bound=3; random permutation after each transform'
)
FLOW_OUTPUT = (
    "StandardNormal base -> 5 conditional RQ-NSF(AR) transforms with explicit "
    "linear tails outside [-3, 3] over flattened continuous Y; categorical t is "
    "one-hot context; 100 fixed-antithetic draws approximate mean"
)


def _flow(**overrides: Any) -> ConditionalFlow:
    defaults: dict[str, Any] = {
        "representation_dim": CNFLOW_ENCODER_WIDTHS[-1],
        "num_treatments": 3,
        "outcome": OutcomeSpec(),
        "num_transforms": 5,
        "hidden_features": 128,
        "num_blocks": 2,
        "use_residual_blocks": True,
        "num_bins": 8,
        "tails": "linear",
        "tail_bound": 3.0,
        "permutation": RANDOM_PERMUTATION,
        "activation": "relu",
        "normalisation": "none",
        "dropout": 0.0,
        "initialisation": NFLOWS_INITIALISATION,
        "base_distribution": STANDARD_NORMAL,
        "mean_samples": 100,
    }
    return ConditionalFlow(**(defaults | overrides))


def test_the_recipe_is_exactly_one_graph_and_one_stage() -> None:
    run = compile(cnflow(make_schema()))
    assert run.graph.names == (
        "mlp_encoder",
        "conditional_flow",
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
    assert all(objective.reduction == "population" for objective in stage.objectives)
    assert all(
        objective.weight.describe() == "constant 1.0" for objective in stage.objectives
    )


def test_the_tier2_pair_declares_its_external_batch_stream() -> None:
    with torch.random.fork_rng():
        flow, gaussian = _paired_recipes(make_schema(), base=70_000)
    for recipe in (flow, gaussian):
        stage = compile(recipe).stage("joint_fit").stage
        assert isinstance(stage.sampler, ExternalBatches)


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_the_flow_recipe_rejects_a_categorical_outcome() -> None:
    schema = make_schema(outcome=OutcomeSpec(kind="categorical", num_classes=2))
    with pytest.raises(GraphError, match="supports only continuous outcomes"):
        cnflow(schema)


@pytest.mark.parametrize("shape", [(), (2,), (2, 2)])
def test_the_real_flow_satisfies_the_candidate_treatment_contract(
    shape: tuple[int, ...],
) -> None:
    schema = make_schema(outcome=OutcomeSpec(shape=shape))
    y = torch.randn(make_batch().batch_size, *shape) if shape else torch.randn(7)
    batch = make_batch(y=y)
    state = compile(cnflow(schema)).state("joint_fit", batch)
    outcome = state.default[Port.Y_GIVEN_XT]
    propensity = state.default[Port.T_GIVEN_X]
    assert isinstance(outcome, ConditionalFlowOutcome)
    assert isinstance(propensity, CategoricalTreatment)
    check_outcome_distribution_contract(
        outcome, y=batch.y, num_treatments=schema.treatment_cardinality
    )
    check_treatment_distribution_contract(
        propensity, num_treatments=schema.treatment_cardinality
    )


def test_linear_tails_keep_ordinary_out_of_bound_outcomes_in_support() -> None:
    batch = make_batch(y=torch.linspace(-8.0, 8.0, 7))
    outcome = (
        compile(cnflow(make_schema()))
        .state("joint_fit", batch)
        .default[Port.Y_GIVEN_XT]
    )
    assert isinstance(outcome, ConditionalFlowOutcome)
    candidates = torch.arange(3).expand(batch.batch_size, 3)
    assert torch.isfinite(outcome.log_prob(batch.y, candidates)).all()


def test_treatment_changes_only_the_conditioner_context() -> None:
    batch = make_batch()
    run = compile(cnflow(make_schema()))
    component = run.graph["conditional_flow"]
    outcome = run.state("joint_fit", batch).default[Port.Y_GIVEN_XT]
    assert isinstance(component, ConditionalFlow)
    assert isinstance(outcome, ConditionalFlowOutcome)

    candidates = torch.arange(3).expand(batch.batch_size, 3)
    permuted = candidates.roll(1, dims=1)
    _, context = outcome._context(candidates)
    _, changed = outcome._context(permuted)
    width = outcome.representation_dim

    assert outcome.event_dim == 1
    assert component.event_dim == 1
    assert context.shape == (batch.batch_size, 3, width + 3)
    assert torch.equal(context[..., :width], changed[..., :width])
    assert not torch.equal(context[..., width:], changed[..., width:])


def test_linear_tails_and_bound_are_required_constructor_invariants() -> None:
    with pytest.raises(GraphError, match="tails must be 'linear'"):
        _flow(tails=None)
    with pytest.raises(GraphError, match="tail_bound supports only 3"):
        _flow(tail_bound=1.0)


def test_fixed_antithetic_mean_points_do_not_advance_the_global_rng() -> None:
    torch.manual_seed(113)
    before = torch.random.get_rng_state()
    points = _fixed_antithetic_standard_normal(100, 4)
    after = torch.random.get_rng_state()
    assert torch.equal(before, after)
    assert torch.equal(points[:50], -points[50:])
    assert torch.allclose(points.mean(dim=0), torch.zeros(4), atol=1e-7)


def test_the_real_flow_uses_the_existing_exact_marginal_objective() -> None:
    schema = make_schema()
    batch = make_batch()
    state = compile(cnflow(schema)).state("joint_fit", batch)
    outcome = state.default[Port.Y_GIVEN_XT]
    propensity = state.default[Port.T_GIVEN_X]
    assert isinstance(outcome, ConditionalFlowOutcome)
    assert isinstance(propensity, CategoricalTreatment)

    rows = resolve_rows(batch, "t_missing")
    objective = MissingTreatmentMarginalNLL(grad_path="both")
    term = objective.compute(
        state,
        batch,
        rows,
        TrainContext(global_step=0, schema=schema, stage="joint_fit"),
    )
    total = torch.zeros(batch.batch_size)
    for treatment in range(schema.treatment_cardinality):
        at_treatment = torch.full((batch.batch_size,), treatment, dtype=torch.long)
        total = (
            total
            + propensity.log_prob(at_treatment).exp()
            * outcome.log_prob(batch.y, at_treatment).exp()
        )
    expected = -total.log().index_select(0, rows).mean()

    assert term.n == int(rows.numel())
    assert torch.allclose(term.value, expected, rtol=1e-5, atol=1e-6)


def test_the_real_marginal_term_trains_both_flow_and_propensity() -> None:
    schema = make_schema()
    batch = make_batch()
    recipe = cnflow(schema)
    state = compile(recipe).state("joint_fit", batch)
    rows = resolve_rows(batch, "t_missing")
    term = MissingTreatmentMarginalNLL(grad_path="both").compute(
        state,
        batch,
        rows,
        TrainContext(global_step=0, schema=schema, stage="joint_fit"),
    )
    term.value.backward()  # type: ignore[no-untyped-call]

    for name in ("conditional_flow", "categorical_propensity"):
        gradients = [
            parameter.grad
            for parameter in recipe.system[name].parameters()
            if parameter.grad is not None
        ]
        assert gradients
        assert sum(float(gradient.norm()) for gradient in gradients) > 0.0


def test_the_plan_names_the_full_reviewed_architecture() -> None:
    hyperparameters = compile(cnflow(make_schema())).plan.hyperparameters
    assert hyperparameters["architecture.widths_depths"] == {
        "mlp_encoder": CNFLOW_ENCODER_WIDTHS,
        "conditional_flow": FLOW_ARCHITECTURE,
        "categorical_propensity": "linear 128 -> 3",
    }
    assert hyperparameters["architecture.initialisation"] == {
        "mlp_encoder": TORCH_LINEAR_INITIALISATION,
        "conditional_flow": NFLOWS_INITIALISATION,
        "categorical_propensity": "normal std=0.1/sqrt(fan_in), bias=0",
    }
    assert hyperparameters["architecture.output_parameterisation"] == {
        "conditional_flow": FLOW_OUTPUT,
        "categorical_propensity": "K softmax logits",
    }
    assert hyperparameters["data.treatment_encoding"] == (
        "one-hot K-vector appended to X_REPR as flow context; never part of the "
        "flow event"
    )
    assert hyperparameters["optimisation.lr_schedule"] == "constant 1.0"
    assert hyperparameters["optimisation.weight_decay"] == "none"


def test_removing_the_flow_head_is_a_named_compile_failure() -> None:
    recipe = cnflow(make_schema())
    without_flow = ComponentGraph(
        [
            recipe.system["mlp_encoder"],
            recipe.system["categorical_propensity"],
        ]
    )
    with pytest.raises(CompileError, match=r"requires port 'p\(y\|x,t\)'"):
        compile(replace(recipe, system=without_flow))


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
    _assert_card_is_covered(cnflow(make_schema()))


def test_the_card_cross_check_fails_when_treatment_context_is_unbound() -> None:
    recipe = cnflow(make_schema())
    flow = recipe.system["conditional_flow"]
    assert isinstance(flow, ConditionalFlow)
    original = ConditionalFlow.CARD_KEYS
    ConditionalFlow.CARD_KEYS = {
        field: key
        for field, key in original.items()
        if key != "data.treatment_encoding"
    }
    try:
        with pytest.raises(AssertionError, match=r"data\.treatment_encoding"):
            _assert_card_is_covered(recipe)
    finally:
        ConditionalFlow.CARD_KEYS = original


def test_the_centered_mixture_formula_in_the_card_has_the_declared_mean() -> None:
    q = torch.linspace(0.2, 0.8, 17)
    positive = q * 1.5 * (1.0 - q)
    negative = (1.0 - q) * -1.5 * q
    assert torch.allclose(positive + negative, torch.zeros_like(q), rtol=0.0, atol=1e-7)
