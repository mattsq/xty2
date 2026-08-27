"""Tier 0 — the DoubleMatch recipe and its card-plan boundary."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from xty2.core import (
    DEFAULT,
    CompileError,
    ComponentGraph,
    CosineDecay,
    FeatureSpec,
    OutcomeSpec,
    Port,
    Program,
    Realisation,
    Recipe,
    Schema,
    XTYBatch,
    compile,
)
from xty2.objectives import CosineFeatureConsistency
from xty2.recipes import doublematch, fixmatch
from xty2.recipes.doublematch import SELF_SUPERVISED_WEIGHT
from xty2.recipes.fixmatch import WEAK_X_LABELLED

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "doublematch.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "doublematch.py"
WEAK_X = Realisation(view="weak_x")
STRONG_X = Realisation(view="strong_x")
SELF_SUPERVISED = "cosine_feature_consistency"
STEPS = 3_000
BATCH_ROWS = 8


def _schema() -> Schema:
    return Schema(
        features=(
            FeatureSpec("mass", "continuous"),
            FeatureSpec("speed", "continuous"),
            FeatureSpec("momentum", "continuous"),
            FeatureSpec("site", "categorical", mutable=False),
        ),
        treatment_cardinality=3,
        outcome=OutcomeSpec(),
    )


def _batch() -> XTYBatch:
    return XTYBatch(
        x=torch.randn(BATCH_ROWS, 4),
        t=torch.arange(BATCH_ROWS) % 3,
        y=torch.randn(BATCH_ROWS),
        t_observed=torch.arange(BATCH_ROWS) % 2 == 0,
        y_observed=torch.ones(BATCH_ROWS, dtype=torch.bool),
        row_id=torch.arange(BATCH_ROWS),
    )


def _self_supervised(recipe: Recipe) -> CosineFeatureConsistency:
    objective = recipe.program[0].objectives[3].objective
    assert isinstance(objective, CosineFeatureConsistency)
    return objective


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_the_stage_has_exactly_the_five_reviewed_objectives() -> None:
    stage = compile(doublematch(_schema())).stage("joint_fit")
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        "pseudo_label_treatment_nll",
        SELF_SUPERVISED,
        "missing_treatment_marginal_nll",
    ]
    # Eq. (3) is the row-population claim `BACKLOG.md` §2.6 asks for: the term
    # that follows a gated one is entitled to every row the gated one rejected.
    assert [objective.rows for objective in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("all",),
        ("all",),
        ("t_missing",),
    ]
    # Eqs. (1), (2) and (3) each average over their own population, so their
    # ratios are the paper's weights whatever a batch happens to hold.
    assert [objective.reduction for objective in stage.objectives] == [
        "population",
        "mean",
        "mean",
        "mean",
        "population",
    ]
    assert [objective.weight.nominal for objective in stage.objectives] == [
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
    ]


def test_the_self_supervised_term_is_the_papers_own() -> None:
    """Eq. (3)'s two sides, its stop-gradient and its population."""
    objective = _self_supervised(doublematch(_schema()))
    assert objective.prediction_port is Port.X_PROJ
    assert objective.target_port is Port.X_REPR
    assert objective.prediction == STRONG_X
    assert objective.target == WEAK_X
    assert objective.stop_grad == "target"
    assert objective.rows == "all"
    assert objective.detaches == frozenset({(Port.X_REPR, WEAK_X)})


def test_the_self_supervised_weight_is_constant_and_the_reviewed_one() -> None:
    """Eq. (4) states `w_s` as a scalar; the reference exposes it as a flag."""
    stage = compile(doublematch(_schema())).stage("joint_fit")
    weight = stage.objectives[3].weight
    assert SELF_SUPERVISED_WEIGHT == 0.5
    assert weight.describe() == "constant 0.5"
    assert [weight(step) for step in (0, 1, 1_000, STEPS)] == [0.5] * 4


def test_the_targets_weak_draw_is_the_one_the_pseudo_label_reads() -> None:
    """Algorithm 1 computes `z_i` once and derives both `w_i` and eq. (3).

    Pointing eq. (3) at the other draw of the same view would compile, cost a
    forward pass, and quietly be a different method — the target would no
    longer be the features the label came from.
    """
    stage = compile(doublematch(_schema())).stage("joint_fit")
    pseudo_label, self_supervised = stage.objectives[2], stage.objectives[3]
    assert (Port.T_GIVEN_X, WEAK_X) in pseudo_label.objective.requires
    assert (Port.X_REPR, WEAK_X) in self_supervised.objective.requires
    assert WEAK_X.draw == 0 and WEAK_X_LABELLED.draw == 1


def test_eq_three_costs_one_linear_layer_and_no_forward_pass() -> None:
    """The paper's "minimal computational overhead", as a property of the plan.

    Four passes, exactly `fixmatch`'s four: eq. (3) rides on the strong pass
    that eq. (2) already needs, and the projection head is the only component
    that pass gains.
    """
    stage = compile(doublematch(_schema())).stage("joint_fit")
    passes = {str(forward.realisation): forward.components for forward in stage.passes}
    assert len(passes) == 4
    assert set(passes) == {
        str(realisation) for realisation in (DEFAULT, WEAK_X, WEAK_X_LABELLED, STRONG_X)
    }
    assert passes[str(STRONG_X)] == (
        "mlp_encoder",
        "categorical_propensity",
        "projection_head",
    )
    for weak in (WEAK_X, WEAK_X_LABELLED):
        assert "projection_head" not in passes[str(weak)]


def test_the_projection_head_is_one_dimension_preserving_affine_layer() -> None:
    """§III: "a single dimension-preserving linear layer".

    The card's §7 says `activation` is inert at one layer and that §4's
    declaration cannot become load-bearing without failing here. This is that
    test: the built module is a single `Linear` and nothing else, so no
    activation and no output normalisation is applied whatever the field says.
    """
    recipe = doublematch(_schema())
    head = recipe.system["projection_head"]
    network = head.network
    assert isinstance(network, nn.Sequential)
    assert len(network) == 1
    linear = network[0]
    assert isinstance(linear, nn.Linear)
    assert linear.in_features == linear.out_features == 200
    assert linear.bias is not None  # ref impl: `tf.layers.dense` defaults to a bias
    assert head.normalisation == "none"

    # Affine and unnormalised, which is what eq. (3) needs: it takes the cosine
    # itself, and a head that emitted unit vectors would hide the scale the
    # bias sees (card §5 deviation 4).
    with torch.no_grad():
        values = recipe.system.evaluate(
            _batch(), schema=recipe.schema, only=("mlp_encoder", "projection_head")
        )
    projected = values[Port.X_PROJ]
    assert isinstance(projected, torch.Tensor)
    assert projected.shape == (BATCH_ROWS, 200)
    assert not torch.allclose(projected.norm(dim=-1), torch.ones(BATCH_ROWS), atol=1e-3)


def test_the_projection_head_is_trained_and_decayed_like_f_and_g() -> None:
    """Eq. (6) sums `||theta_f||^2 + ||theta_g||^2 + ||theta_h||^2`."""
    stage = compile(doublematch(_schema())).stage("joint_fit")
    assert stage.trainable == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
        "projection_head",
    )
    decay = stage.stage.optimiser.weight_decay
    assert decay.value == 5e-4
    assert decay.components is None  # every trainable component, `h` included
    assert decay.on_norm_and_bias is False


def test_a_projection_head_no_objective_reads_is_a_compile_error() -> None:
    """The dead-trainable rule is what keeps `h` honest if eq. (3) is dropped."""
    recipe = doublematch(_schema())
    stage = recipe.program[0]
    objectives = tuple(
        weighted for weighted in stage.objectives if weighted.name != SELF_SUPERVISED
    )
    mutant = replace(recipe, program=Program((replace(stage, objectives=objectives),)))
    with pytest.raises(CompileError, match="no active objective depends on"):
        compile(mutant)


def test_the_rate_schedule_is_equation_five_at_the_papers_gamma() -> None:
    run = compile(doublematch(_schema()))
    schedule = run.stage("joint_fit").stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    assert schedule.steps == run.stage("joint_fit").steps == STEPS
    for step in (0, 750, 1_500, STEPS):
        # eq. (5): `eta_0 cos(gamma pi k / 2K)` at `gamma = 7/8`.
        assert schedule(step) == pytest.approx(
            math.cos((7.0 / 8.0) * math.pi * step / (2.0 * STEPS))
        )


# ---------------------------------------------------------------------------
# "Identical to FixMatch when w_s = 0" (§III)
# ---------------------------------------------------------------------------


def _fixmatch_shaped(recipe: Recipe) -> Recipe:
    """`doublematch` with eq. (3) and the projection head taken back out."""
    stage = recipe.program[0]
    objectives = tuple(
        weighted for weighted in stage.objectives if weighted.name != SELF_SUPERVISED
    )
    trainable = tuple(name for name in stage.trainable if name != "projection_head")
    components = [
        component
        for component in recipe.system.components
        if component.name != "projection_head"
    ]
    return replace(
        recipe,
        system=ComponentGraph(components),
        program=Program((replace(stage, objectives=objectives, trainable=trainable),)),
    )


def test_removing_eq_three_leaves_exactly_the_fixmatch_plan() -> None:
    """The paper's own sentence, and exactly how far it goes here.

    §III: "our loss function is identical to that used in FixMatch when
    `w_s = 0`". Dropping eq. (3) and the projection head leaves a plan that
    differs from `fixmatch`'s in five lines, and the point of the test is that
    they are enumerable: three are the recipe's identity (its name, its card,
    and one sentence of prose about the fixture, which the plan prints twice),
    and the fifth is **deviation 9** — the encoder's output normalisation, the
    one place this recipe departs from the shared P5 backbone.

    Everything else — every component, view, forward pass, row population,
    reduction, weight, schedule, optimiser setting and sampler quota — is the
    same value. So the card's §6 pair is within this recipe rather than across
    the two, and this test is what says why that is necessary rather than
    tidy.
    """
    schema = _schema()
    torch.manual_seed(4)
    ours = compile(_fixmatch_shaped(doublematch(schema))).plan.render().splitlines()
    torch.manual_seed(4)
    theirs = compile(fixmatch(schema)).plan.render().splitlines()

    assert len(ours) == len(theirs)
    differing = [
        (mine, yours) for mine, yours in zip(ours, theirs, strict=True) if mine != yours
    ]
    assert len(differing) == 5
    assert differing[0][0].startswith("recipe:")
    assert differing[1][0].startswith("card:")
    assert differing[2][0].strip().startswith("split ")
    assert differing[4][0].strip().startswith("data.split_protocol")
    # Deviation 9, and the only computational difference of the five.
    assert differing[3] == (
        "    mlp_encoder            = 'none'",
        "    mlp_encoder            = 'row_l2'",
    )
    for mine, yours in differing[:3] + differing[4:]:
        assert mine.replace("doublematch", "fixmatch").replace("/STL", "") == yours


def test_the_two_recipes_build_byte_identical_shared_parameters() -> None:
    """The projection head is new weight, and everything else is not.

    `h` is constructed last, so the three components `fixmatch` also has
    consume the same construction-time RNG and initialise identically. That is
    what lets §6's pair share a seed and differ in eq. (3) alone.
    """
    schema = _schema()
    torch.manual_seed(19)
    ours = doublematch(schema).system.state_dict()
    torch.manual_seed(19)
    theirs = fixmatch(schema).system.state_dict()
    assert set(theirs) < set(ours)
    for name, value in theirs.items():
        assert torch.equal(value, ours[name])
    assert any("projection_head" in name for name in set(ours) - set(theirs))


# ---------------------------------------------------------------------------
# The card boundary
# ---------------------------------------------------------------------------


def _card_section_four() -> dict[str, str | dict[str, str]]:
    """Card §4 as data: `{canonical_key: value}` or `{key: {scope: value}}`."""
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
    plan = compile(doublematch(_schema())).plan
    answered = _card_section_four()
    missing = sorted(set(answered) - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)


def test_the_per_objective_card_values_are_the_ones_that_ran() -> None:
    """The half of §4 a recipe drifts from, compared value by value.

    Every `losses.*` and `gradients.stop_gradients` entry is keyed by
    `<stage>.<objective>` and rendered by the compiler from what the term
    actually holds, so a card that said `population` where the recipe says
    `mean` — or named the wrong detached side — fails here.
    """
    plan = compile(doublematch(_schema())).plan
    answered = _card_section_four()
    for key in (
        "losses.reduction",
        "losses.eligible_rows",
        "losses.schedules",
        "gradients.stop_gradients",
    ):
        assert answered[key] == plan.hyperparameters[key], key
    weights = answered["losses.weights"]
    assert isinstance(weights, dict)
    assert {name: float(value) for name, value in weights.items()} == (
        plan.hyperparameters["losses.weights"]
    )


def test_the_scalar_card_values_are_the_ones_that_ran() -> None:
    plan = compile(doublematch(_schema())).plan
    hyperparameters = plan.hyperparameters
    assert hyperparameters["gradients.detached_targets"] == "target"
    assert hyperparameters["losses.confidence_threshold"] == 0.95
    assert hyperparameters["losses.sharpening"] == "hard"
    assert hyperparameters["optimisation.lr"] == 0.03
    assert (
        hyperparameters["optimisation.optimiser"] == "sgd(momentum=0.9, nesterov=True)"
    )
    assert hyperparameters["optimisation.weight_decay"] == (
        "0.0005 (all trainable components; norm and bias exempt)"
    )
    assert hyperparameters["optimisation.batch_size"] == 512
    assert hyperparameters["optimisation.labelled_unlabelled_ratio"] == 7.0
    assert hyperparameters["optimisation.total_steps_or_epochs"] == STEPS
    assert hyperparameters["teacher.ema_decay"] == 0.999
    assert hyperparameters["architecture.widths_depths"]["projection_head"] == (200,)
    assert hyperparameters["architecture.normalisation"]["projection_head"] == "none"


def test_the_terms_arithmetic_is_in_the_plan() -> None:
    """Which side is trained is stated by nothing else in the plan.

    `requires` renders as a sorted set, so an eq. (3) that trained the encoder
    through the weak view and froze the projection head would print the same
    two ports. `plan_details` is what puts the roles, the sign and the
    denominator into the rendered plan and therefore into its digest
    (`DESIGN.md` §4).
    """
    render = compile(doublematch(_schema())).plan.render()
    for detail in (
        "prediction (trained) = x_proj @ view=strong_x params=student",
        "target (detached) = x_repr @ view=weak_x params=student",
        "value = -cosine(prediction, target), per row",
        "denominator = every eligible row; nothing is gated",
    ):
        assert f"setting   {detail}" in render


def test_swapping_the_roles_of_eq_three_never_reaches_a_digest() -> None:
    """In *this* recipe the compiler catches the swap before a plan exists.

    Training `z_i` and detaching `h(v_i)` leaves the projection head reachable
    only through a stop-gradient, which is the second half of the
    dead-trainable rule (`DESIGN.md` §8.4). Worth asserting rather than
    assuming: it means the role lines above are the digest's defence for
    objectives in general, and the graph is this recipe's own.
    """
    recipe = doublematch(_schema())
    stage = recipe.program[0]
    swapped = replace(
        stage.objectives[3],
        objective=replace(
            _self_supervised(recipe),
            prediction_port=Port.X_REPR,
            target_port=Port.X_PROJ,
            prediction=WEAK_X,
            target=STRONG_X,
        ),
    )
    objectives = (*stage.objectives[:3], swapped, *stage.objectives[4:])
    mutant = replace(recipe, program=Program((replace(stage, objectives=objectives),)))
    with pytest.raises(CompileError, match="through a stop-gradient"):
        compile(mutant)
