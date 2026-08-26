"""Tier 0 — the FixMatch recipe and its card-plan boundary."""

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
    DEFAULT,
    CompileError,
    CosineDecay,
    FeatureSpec,
    OutcomeSpec,
    Port,
    Program,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    XTYBatch,
    compile,
)
from xty2.objectives import PseudoLabelTreatmentNLL
from xty2.recipes import ENCODER_WIDTHS, OUTCOME_WIDTHS, fixmatch, mean_teacher, tarnet
from xty2.views import FeatureMask

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "fixmatch.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "fixmatch.py"
PRESERVED = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)
WEAK_X = Realisation(view="weak_x")
STRONG_X = Realisation(view="strong_x")


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


def _pseudo_label(recipe: Recipe) -> PseudoLabelTreatmentNLL:
    objective = recipe.program[0].objectives[2].objective
    assert isinstance(objective, PseudoLabelTreatmentNLL)
    return objective


def test_the_recipe_plans_exactly_the_three_reviewed_realisations() -> None:
    run = compile(fixmatch(_schema()))
    assert run.graph.names == ("mlp_encoder", "tarnet_head", "categorical_propensity")
    assert len(run.stages) == 1
    stage = run.stage("joint_fit")
    assert stage.steps == 3_000
    assert stage.stage.rows == "all"
    assert stage.trainable == run.graph.names
    assert stage.teacher is None
    assert sorted(str(forward.realisation) for forward in stage.passes) == sorted(
        str(realisation) for realisation in (DEFAULT, WEAK_X, STRONG_X)
    )
    for forward in stage.passes:
        expected = (
            run.graph.names
            if forward.realisation == DEFAULT
            else ("mlp_encoder", "categorical_propensity")
        )
        assert forward.components == expected


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_the_stage_has_exactly_the_four_reviewed_objectives() -> None:
    stage = compile(fixmatch(_schema())).stage("joint_fit")
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        "pseudo_label_treatment_nll",
        "missing_treatment_marginal_nll",
    ]
    assert [objective.rows for objective in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("all",),
        ("t_missing",),
    ]
    # Eq. (3) divides by the labelled batch and eq. (4) by the unlabelled one:
    # each FixMatch term averages over its own rows, so their ratio is lambda_u.
    # The two retained P5 likelihood terms keep TARNet's whole-batch average.
    assert [objective.reduction for objective in stage.objectives] == [
        "population",
        "mean",
        "mean",
        "population",
    ]
    assert [objective.weight.nominal for objective in stage.objectives] == [
        1.0,
        1.0,
        1.0,
        0.5,
    ]
    assert stage.objectives[1].objective.requires == frozenset(
        {(Port.T_GIVEN_X, WEAK_X)}
    )
    assert stage.objectives[2].objective.requires == frozenset(
        {(Port.T_GIVEN_X, WEAK_X), (Port.T_GIVEN_X, STRONG_X)}
    )


def test_the_pseudo_label_term_carries_the_reviewed_gate() -> None:
    objective = _pseudo_label(fixmatch(_schema()))
    assert objective.threshold == 0.95
    assert objective.sharpening == "hard"
    assert objective.stop_grad == "target"
    assert objective.rows == "all"
    assert objective.target == WEAK_X
    assert objective.prediction == STRONG_X
    assert objective.detaches == frozenset({(Port.T_GIVEN_X, WEAK_X)})


def test_the_unlabelled_weight_is_constant_because_the_paper_rejects_a_ramp() -> None:
    stage = compile(fixmatch(_schema())).stage("joint_fit")
    pseudo_label = stage.objectives[2]
    assert pseudo_label.weight.describe() == "constant 1.0"
    assert [pseudo_label.weight(step) for step in (0, 1, 1_000, 3_000)] == [1.0] * 4
    # The marginal term's P5 ramp is still there, and is the only one.
    assert stage.objectives[3].weight.describe() == "ramp 0.0 -> 0.5 over 1000 steps"


def test_the_strong_view_is_the_weak_one_with_more_corruption_layered_on() -> None:
    """The reference does not swap the weak transform out on the strong branch.

    It samples the ordinary augmentation a second time, independently, and then
    layers CTAugment and Cutout onto *that*. The strong view therefore lists
    both transforms rather than one stronger one, which is what stops the two
    branches reading as alternatives.
    """
    recipe = fixmatch(_schema())
    assert [view.name for view in recipe.views] == ["weak_x", "strong_x"]
    assert recipe.views[0].transforms == (FeatureMask(p=0.1, columns=None, value=0.0),)
    assert recipe.views[1].transforms == (
        FeatureMask(p=0.1, columns=None, value=0.0),
        FeatureMask(p=0.5, columns=None, value=0.0),
    )
    for view in recipe.views:
        assert view.preserves == PRESERVED
        assert "x" not in view.preserves

    batch = _batch()
    weak = recipe.view("weak_x").apply(batch, recipe.schema, rng_key=91)
    repeated = recipe.view("weak_x").apply(batch, recipe.schema, rng_key=91)
    strong = recipe.view("strong_x").apply(batch, recipe.schema, rng_key=91)
    assert weak.equal_to(repeated)
    assert not torch.equal(weak.x, strong.x)
    # The strong view is the stronger one, which is the whole weak/strong
    # relation FixMatch section 2.3 is about; asserting it on the realised
    # batches keeps the two rates from being swapped in the recipe.
    masked_weak = (weak.x[:, :3] == 0.0).float().mean()
    masked_strong = (strong.x[:, :3] == 0.0).float().mean()
    assert masked_weak < masked_strong
    assert torch.equal(weak.x[:, 3], batch.x[:, 3])
    assert torch.equal(strong.x[:, 3], batch.x[:, 3])


def _momentum(columns: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return columns["mass"] * columns["speed"]


def test_derived_features_need_and_accept_explicit_recompute_rules() -> None:
    with pytest.raises(CompileError, match="makes derived column"):
        compile(fixmatch(_schema(derived=True)))

    rule = RecomputeRule("momentum", _momentum, name="mass_times_speed")
    run = compile(fixmatch(_schema(derived=True), recompute_rules=(rule,)))
    assert [view.recomputes for view in run.plan.views] == [
        ("momentum <- mass_times_speed",),
        ("momentum <- mass_times_speed",),
    ]


def test_the_optimiser_is_the_papers_own_and_not_the_p5_stack() -> None:
    run = compile(fixmatch(_schema()))
    hyperparameters = run.plan.hyperparameters
    assert (
        hyperparameters["optimisation.optimiser"] == "sgd(momentum=0.9, nesterov=True)"
    )
    assert hyperparameters["optimisation.lr"] == 0.03
    # The reference decays the variables whose name carries `kernel`, so the
    # exemption is not a framework default sneaking in — it is the line of the
    # implementation that the paper's "L2 penalty of all weights" glosses over.
    assert hyperparameters["optimisation.weight_decay"] == (
        "0.0005 (all trainable components; norm and bias exempt)"
    )
    schedule = run.stage("joint_fit").stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    assert schedule.steps == run.stage("joint_fit").steps
    for step in (0, 1_500, 3_000):
        assert schedule(step) == pytest.approx(
            math.cos(7.0 * math.pi * step / (16.0 * 3_000))
        )


def test_the_architecture_is_the_shared_p5_stack() -> None:
    hyperparameters = compile(fixmatch(_schema())).plan.hyperparameters
    assert hyperparameters["architecture.widths_depths"] == {
        "mlp_encoder": ENCODER_WIDTHS,
        "tarnet_head": f"3 independent heads, each {list(OUTCOME_WIDTHS)}",
        "categorical_propensity": "linear 200 -> 3",
    }
    torch.manual_seed(19)
    fixmatch_graph = fixmatch(_schema()).system
    torch.manual_seed(19)
    tarnet_graph = tarnet(_schema()).system
    assert fixmatch_graph.names == tarnet_graph.names
    for name, value in fixmatch_graph.state_dict().items():
        assert torch.equal(value, tarnet_graph.state_dict()[name])


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
    plan = compile(fixmatch(_schema())).plan
    answered = _answered_card_keys()
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    # The three keys this recipe is the first to answer, spelled out so that a
    # card that quietly dropped one would fail here rather than pass a subset.
    assert {
        "losses.confidence_threshold",
        "losses.sharpening",
        "gradients.detached_targets",
    } <= answered
    assert plan.hyperparameters["losses.confidence_threshold"] == 0.95
    assert plan.hyperparameters["losses.sharpening"] == "hard"
    assert [view.name for view in plan.views] == ["weak_x", "strong_x"]
    assert [view.transforms for view in plan.views] == [
        ("FeatureMask(p=0.1, columns=all, value=0.0)",),
        (
            "FeatureMask(p=0.1, columns=all, value=0.0)",
            "FeatureMask(p=0.5, columns=all, value=0.0)",
        ),
    ]
    assert "forward passes (3)" in plan.render()


def test_the_card_answers_no_teacher_key_and_the_recipe_declares_none() -> None:
    # Deviation 5: the paper's EMA is an evaluation device, and a TeacherSpec
    # no objective reads is a compile error rather than a silent no-op.
    assert not {key for key in _answered_card_keys() if key.startswith("teacher.")}
    assert compile(fixmatch(_schema())).stage("joint_fit").teacher is None


def test_the_gate_arithmetic_is_in_the_plan_digest() -> None:
    """A different denominator must not share this plan's identity.

    `plan_details` is the only place the mask-then-average choice appears, so
    the digest is what stops two recipes that compute different losses from
    looking like the same run (`DESIGN.md` §4).
    """
    plan = compile(fixmatch(_schema())).plan
    assert (
        "denominator = every eligible row; rejected rows contribute 0" in plan.render()
    )
    mutant = _with_pseudo_label(fixmatch(_schema()), threshold=0.5)
    assert compile(mutant).plan.digest != plan.digest


def _with_pseudo_label(recipe: Recipe, **overrides: object) -> Recipe:
    stage = recipe.program[0]
    objective = replace(_pseudo_label(recipe), **overrides)  # type: ignore[arg-type]
    weighted = replace(stage.objectives[2], objective=objective)
    objectives = (*stage.objectives[:2], weighted, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def test_removing_either_view_is_a_named_compile_failure() -> None:
    recipe = fixmatch(_schema())
    for views, missing in (
        (recipe.views[1:], "weak_x"),
        (recipe.views[:1], "strong_x"),
    ):
        with pytest.raises(CompileError, match=missing):
            compile(replace(recipe, views=views))


def test_the_recipe_shares_the_mean_teacher_weak_view_strength() -> None:
    # Card §7: reusing the reviewed 0.1 mask rate for the weak view is what
    # makes the two recipes' weak augmentation the same declaration rather
    # than two independently chosen numbers.
    weak = fixmatch(_schema()).views[0].transforms
    assert weak == mean_teacher(_schema()).views[0].transforms


def test_nothing_in_the_graph_couples_rows_within_a_forward_pass() -> None:
    """Why three separate forward passes are equivalent to one fused pass.

    The reference concatenates labelled, weak and strong examples into a single
    call and interleaves them first, so that every device's BatchNorm
    population sees a mixture of the three streams rather than one of them.
    xty2 plans one pass per realisation instead, which is arithmetically the
    same thing only while no component holds batch-coupled state: `row_l2`
    normalises each row independently and nothing here carries a running
    statistic. Assert that, so a component that later grows one fails here
    rather than silently computing three sets of statistics.
    """
    graph = fixmatch(_schema()).system
    assert list(graph.buffers()) == []
    hyperparameters = compile(fixmatch(_schema())).plan.hyperparameters
    assert hyperparameters["architecture.normalisation"] == {
        "mlp_encoder": "row_l2",
        "tarnet_head": "none",
        "categorical_propensity": "none",
    }
