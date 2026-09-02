"""Tier 0 — the FlexMatch recipe, its card-plan boundary and its state lifecycle.

Two things are asserted here that no other recipe's Tier 0 file has had to
assert. The first is that swapping eq. (8)'s gate back for FixMatch's leaves a
plan identical to `fixmatch`'s in every line but the recipe's own identity — so
the §6 pair really does differ in the gate and in nothing else. The second is
the lifecycle of `docs/recipes/flexmatch.md` §5.1's framework addition: the
curriculum's marks are built per stage *execution*, so two runs of one compiled
recipe are identical and a paired ablation sharing an objective instance cannot
leak one arm's history into the other.
"""

from __future__ import annotations

import ast
import difflib
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xty2 import objectives
from xty2.core import (
    CosineDecay,
    Dataset,
    FeatureSpec,
    OutcomeSpec,
    Port,
    Program,
    Recipe,
    Schema,
    StatefulObjective,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.objectives import (
    CurriculumPseudoLabelTreatmentNLL,
    CurriculumStatus,
    CurriculumThreshold,
    PseudoLabelTreatmentNLL,
)
from xty2.recipes import doublematch, fixmatch, flexmatch
from xty2.recipes.fixmatch import (
    STRONG_X,
    WEAK_MASK_RATE,
    WEAK_X,
    WEAK_X_LABELLED,
)
from xty2.recipes.flexmatch import (
    CURRICULUM,
    FLEXMATCH_STEPS,
    STRONG_MASK_RATE,
    TAU,
)
from xty2.training import run_stage
from xty2.training.executors import _objective_states

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "flexmatch.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "flexmatch.py"
CURRICULUM_TERM = "curriculum_pseudo_label_treatment_nll"
TRAIN_ROWS = 64
STEPS = 6


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


def _gate(recipe: Recipe) -> CurriculumPseudoLabelTreatmentNLL:
    objective = recipe.program[0].objectives[2].objective
    assert isinstance(objective, CurriculumPseudoLabelTreatmentNLL)
    return objective


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_the_stage_has_exactly_the_four_reviewed_objectives() -> None:
    stage = compile(flexmatch(_schema())).stage("joint_fit")
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        CURRICULUM_TERM,
        "missing_treatment_marginal_nll",
    ]
    # `all` for eq. (8) is FixMatch's footnote 2, inherited — and it is also the
    # population `N` is counted over, so it is two claims in one value.
    assert [objective.rows for objective in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("all",),
        ("t_missing",),
    ]
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


def test_the_gate_is_the_papers_own() -> None:
    """Eq. (8)'s two sides, its stop-gradient, its population and its rule."""
    objective = _gate(flexmatch(_schema()))
    assert objective.port is Port.T_GIVEN_X
    assert objective.target == WEAK_X
    assert objective.prediction == STRONG_X
    assert objective.sharpening == "hard"
    assert objective.stop_grad == "target"
    assert objective.rows == "all"
    assert objective.detaches == frozenset({(Port.T_GIVEN_X, WEAK_X)})
    assert objective.threshold == CURRICULUM
    assert (objective.threshold.tau, objective.threshold.warm_up) == (TAU, True)
    assert objective.threshold.mapping == "convex"


def test_the_curriculum_weight_is_constant_and_the_papers() -> None:
    """Eq. (9) states `lambda` as a fixed scalar; the curriculum is the gate."""
    stage = compile(flexmatch(_schema())).stage("joint_fit")
    weight = stage.objectives[2].weight
    assert weight.describe() == "constant 1.0"
    assert [weight(step) for step in (0, 1, 1_000, FLEXMATCH_STEPS)] == [1.0] * 4


def test_the_labelled_term_reads_the_second_weak_draw() -> None:
    """Eq. (10) on its own draw of `omega`, per FixMatch's footnote 2."""
    stage = compile(flexmatch(_schema())).stage("joint_fit")
    assert (Port.T_GIVEN_X, WEAK_X_LABELLED) in stage.objectives[1].objective.requires
    assert WEAK_X.draw == 0 and WEAK_X_LABELLED.draw == 1


def test_the_rate_schedule_is_fixmatchs_at_our_budget() -> None:
    run = compile(flexmatch(_schema()))
    schedule = run.stage("joint_fit").stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    assert schedule.steps == run.stage("joint_fit").steps == FLEXMATCH_STEPS
    for step in (0, 750, 1_500, FLEXMATCH_STEPS):
        assert schedule(step) == pytest.approx(
            math.cos((7.0 / 16.0) * math.pi * step / FLEXMATCH_STEPS)
        )


# ---------------------------------------------------------------------------
# "CPL is a threshold and nothing else"
# ---------------------------------------------------------------------------


def _with_fixmatchs_gate(recipe: Recipe) -> Recipe:
    """Eq. (8) replaced by eq. (3): the same term at a constant `tau`."""
    stage = recipe.program[0]
    swapped = Weighted(
        PseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=WEAK_X,
            prediction=STRONG_X,
            threshold=TAU,
            sharpening="hard",
            stop_grad="target",
            rows="all",
        ),
        weight=1.0,
        reduction="mean",
    )
    objectives = (*stage.objectives[:2], swapped, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def test_swapping_the_gate_back_leaves_the_fixmatch_plan_bar_deviation_two() -> None:
    """The §6 pair, as a property of the plan rather than of the prose.

    Every component, forward pass, row population, reduction, weight, schedule,
    optimiser setting and sampler quota is the same value as `fixmatch`'s, so a
    difference between the two runs is attributable to two enumerable things and
    no others. Three of the five differing lines are the recipe's identity — its
    name, its card, and one word of the split protocol, which the plan prints
    twice because FlexMatch reports on STL-10 and FixMatch's card does not name
    it. The other two are **deviation 2**: the strong view's extra corruption is
    0.2 here and 0.5 there, which card §5.2 measures rather than inherits.

    That the list is exhaustive is the assertion. `flexmatch`'s §6 pair is
    therefore taken *within* this recipe — the same views, the gate swapped —
    and this test is what says why that is necessary rather than tidy.
    """
    schema = _schema()
    torch.manual_seed(4)
    ours = compile(_with_fixmatchs_gate(flexmatch(schema))).plan.render().splitlines()
    torch.manual_seed(4)
    theirs = compile(fixmatch(schema)).plan.render().splitlines()

    assert len(ours) == len(theirs)
    differing = [
        (mine, yours) for mine, yours in zip(ours, theirs, strict=True) if mine != yours
    ]
    assert len(differing) == 5, "\n".join(
        difflib.unified_diff(theirs, ours, "fixmatch", "flexmatch", lineterm="", n=0)
    )
    assert differing[0][0].startswith("recipe:")
    assert differing[1][0].startswith("card:")
    assert differing[2][0].strip().startswith("split ")
    assert differing[4][0].strip().startswith("data.split_protocol")
    # Deviation 2, and the only computational difference of the five.
    assert differing[3] == (
        "    transform  FeatureMask(p=0.2, columns=all, value=0.0)",
        "    transform  FeatureMask(p=0.5, columns=all, value=0.0)",
    )
    for mine, yours in differing[:3] + differing[4:]:
        assert mine.replace("flexmatch", "fixmatch").replace("/STL", "") == yours


def test_the_cards_prose_about_the_views_matches_the_plan() -> None:
    """§4's YAML has no key for a view, so nothing else checks these two lines.

    Both went stale once: the §3.2 mapping row and the §4 paragraph kept saying
    `FeatureMask(p=0.5)` after deviation 2 moved the recipe to 0.2, and the
    card-key cross-check could not see it because it parses §4's YAML block
    only. `CLAUDE.md` rule 4 makes §3.2 the artifact the reviewer diffs the plan
    against, so a stale row there is the review failing silently.

    Deliberately a search for the rendered transform strings rather than a
    parse of the table: what has to be true is that the numbers the card prints
    are the numbers the plan prints, and any prose that names the other one is
    the failure.
    """
    text = CARD.read_text(encoding="utf-8")
    mapping = text.split("### 3.2 Mapping to xty2", 1)[1].split("## 4.", 1)[0]
    checklist = text.split("## 4. Mechanics checklist", 1)[1].split("## 5.", 1)[0]
    plan = compile(flexmatch(_schema())).plan
    weak, strong = plan.views
    assert strong.transforms == (
        f"FeatureMask(p={WEAK_MASK_RATE}, columns=all, value=0.0)",
        f"FeatureMask(p={STRONG_MASK_RATE}, columns=all, value=0.0)",
    )
    for section, where in ((mapping, "§3.2"), (checklist, "§4")):
        assert f"FeatureMask(p={STRONG_MASK_RATE})" in section, (
            f"{where} does not name the strong view the plan runs "
            f"(p={STRONG_MASK_RATE})"
        )
        assert f"FeatureMask(p={WEAK_MASK_RATE})" in section
    # The one number that must NOT appear unqualified: `fixmatch`'s 0.5. It may
    # be named as the thing this recipe departs *from*, which both sections do,
    # so the check is that every mention sits on a line that says so.
    for section, where in ((mapping, "§3.2"), (checklist, "§4")):
        for line in section.splitlines():
            if "FeatureMask(p=0.5)" in line:
                assert "fixmatch" in line or "deviation 2" in line, (
                    f"{where} names FeatureMask(p=0.5) without saying it is "
                    f"`fixmatch`'s: {line!r}"
                )
    assert weak.draws == 2


def test_the_curriculum_term_is_the_only_line_the_gate_adds_to_the_plan() -> None:
    """What the reviewer diffs: eight `setting` lines, all about the gate."""
    stage = compile(flexmatch(_schema())).stage("joint_fit")
    details = stage.objectives[2].plan_details
    assert details == (
        "label = arg max of the target realisation",
        "gate = max prob > T(label), the per-class threshold (eq. 8)",
        "beta(c) = sigma(c) / max(max_c sigma, N - sum_c sigma) (eq. 11)",
        "T(c) = M(beta(c)) * tau, M(x) = x / (2 - x) (eq. 12)",
        "sigma(c) = rows ever marked class c (Alg. 1 line 5)",
        "marks are set at the fixed tau, not at T(c) (Alg. 1 line 14)",
        "marks are per-stage state keyed by row_id, and are never cleared",
        "denominator = every eligible row; rejected rows contribute 0",
    )


def test_the_weak_view_is_fixmatchs_and_the_strong_one_is_measured() -> None:
    """Deviation 2, as the one declared difference between the two recipes.

    The weak view is `fixmatch`'s to the byte — same transform, same rate, same
    two draws, same preserved fields — because eq. (5), eq. (8) and algorithm 1
    line 14 all read it and the §6 pair must not differ there. The strong one
    keeps its *shape* (the weak transform with more corruption layered on, as
    the reference does) at a rate card §5.2 measured: 0.2 rather than 0.5.
    """
    schema = _schema()
    ours = compile(flexmatch(schema)).plan.views
    theirs = compile(fixmatch(schema)).plan.views
    assert [view.name for view in ours] == ["weak_x", "strong_x"]
    weak, strong = ours
    their_weak, their_strong = theirs
    assert (weak.name, weak.transforms, weak.preserves, weak.draws) == (
        their_weak.name,
        their_weak.transforms,
        their_weak.preserves,
        their_weak.draws,
    )
    assert strong.preserves == their_strong.preserves
    assert strong.transforms == (
        f"FeatureMask(p={WEAK_MASK_RATE}, columns=all, value=0.0)",
        f"FeatureMask(p={STRONG_MASK_RATE}, columns=all, value=0.0)",
    )
    assert STRONG_MASK_RATE == 0.2
    assert their_strong.transforms[1] == "FeatureMask(p=0.5, columns=all, value=0.0)"


def test_the_strong_view_keeps_the_bayes_optimal_label() -> None:
    """Card §5.2's criterion, recomputed rather than quoted.

    FixMatch §2.3 asks a strong augmentation to be severe *and* label-preserving.
    On the §6.1 DGP the Bayes-optimal rule is closed form — a two-component
    Gaussian mixture in the four signal columns — so "label-preserving" is a
    number this test can check without training anything, which is what makes
    §5.2 a measurement rather than a preference. The declared 0.2 keeps the
    label on more than 90% of rows at more than twice the weak view's
    corruption; `fixmatch`'s 0.5 does not clear the first bar, which is the
    whole of deviation 2.
    """
    flips = {
        rate: _bayes_label_flip_rate(1.0 - (1.0 - WEAK_MASK_RATE) * (1.0 - rate))
        for rate in (STRONG_MASK_RATE, 0.5)
    }
    assert flips[STRONG_MASK_RATE] < 0.10
    assert flips[0.5] > 0.15
    effective = 1.0 - (1.0 - WEAK_MASK_RATE) * (1.0 - STRONG_MASK_RATE)
    assert effective >= 2.0 * WEAK_MASK_RATE


def _bayes_label_flip_rate(mask_rate: float, rows: int = 40_000) -> float:
    """`P(arg max p(t | strong view) != arg max p(t | x))` on the §6.1 DGP.

    Closed form up to the Monte Carlo draw: `x_j | c ~ N(0.45 (2c - 1), 0.6^2)`
    for the four signal columns, `c ~ Bern(0.5)`, and `FeatureMask` replaces a
    masked column with a constant that carries no information about `c`, so the
    Bayes rule conditions on the visible signal columns alone.
    """
    signal, sd, columns = 0.45, 0.6, 4
    generator = torch.Generator().manual_seed(90_001)
    cluster = (torch.rand(rows, generator=generator) < 0.5).float()
    x = signal * (2 * cluster - 1)[:, None] + sd * torch.randn(
        rows, columns, generator=generator
    )
    visible = torch.rand(rows, columns, generator=generator) >= mask_rate

    def posterior(mask: torch.Tensor) -> torch.Tensor:
        def loglik(mu: float) -> torch.Tensor:
            return (-((x - mu) ** 2) / (2 * sd**2) * mask).sum(dim=1)

        return torch.sigmoid(loglik(signal) - loglik(-signal))

    clean = posterior(torch.ones_like(visible)) >= 0.5
    masked = posterior(visible) >= 0.5
    return float((clean != masked).float().mean())


# ---------------------------------------------------------------------------
# The state lifecycle (`flexmatch.md` §5.1)
# ---------------------------------------------------------------------------


def _batch(rows: int = TRAIN_ROWS) -> XTYBatch:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(rows, 4, generator=generator)
    x[:, 3] = (torch.arange(rows) % 2).float()
    return XTYBatch(
        x=x,
        t=torch.arange(rows) % 3,
        y=torch.randn(rows, generator=generator),
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
    )


def _dataset() -> Dataset:
    rows = _batch()
    return Dataset(
        schema=_schema(),
        rows=rows,
        assignments={"train": torch.arange(rows.batch_size)},
    )


def _short(recipe: Recipe) -> Recipe:
    """A handful of steps, with a quota this fixture can actually fill."""
    from xty2.core import Quota, QuotaSampler

    data = recipe.data
    assert data is not None
    stage = recipe.program[0]
    schedule = stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    steps=STEPS,
                    optimiser=replace(
                        stage.optimiser, lr_schedule=replace(schedule, steps=STEPS)
                    ),
                    sampler=QuotaSampler(
                        quotas=(
                            Quota(rows="t_observed", size=4),
                            Quota(rows="t_missing", size=28),
                        )
                    ),
                ),
            )
        ),
        data=replace(
            data,
            missingness=replace(data.missingness, observed=16),
        ),
    )


def test_state_stays_opt_in_across_the_objective_package() -> None:
    """Opt-in, checked over the whole objective package (`DESIGN.md` §4).

    Card §5.1 claims that no objective that existed before this one declares
    `initial_state`, so the executor builds an empty mapping for every stage
    that predates FlexMatch and no plan, digest or recorded result can move. An
    earlier version of this test checked `fixmatch` alone while the card claimed
    the property of every recipe; an adversarial review pointed out the gap.

    Enumerating `xty2.objectives.__all__` rather than compiling every recipe is
    what makes it exhaustive: the claim is about objectives, and a recipe can
    only carry an objective this list names. It is also schema-free, where
    compiling each recipe would need a schema each one accepts.

    The list has grown since, and that is the mechanism working rather than the
    claim weakening: `SelfAdaptiveThresholdTreatmentNLL` is the second consumer
    this card's §5.1 named in advance (`freematch.md` §5.1), and the CoMatch and
    SimMatch memories are the third and fourth, each reviewed on its own card. The
    property being asserted is that the set is *this* set — opting in is
    deliberate, and an objective that acquired state by accident would show up
    here.
    """
    stateful = sorted(
        name
        for name in objectives.__all__
        if isinstance(getattr(objectives, name), type)
        and isinstance(getattr(objectives, name), StatefulObjective)
    )
    assert stateful == [
        "CurriculumPseudoLabelTreatmentNLL",
        "MemorySmoothedPseudoLabelTreatmentNLL",
        "SelfAdaptiveThresholdTreatmentNLL",
        "SimilarityMatchingTreatmentNLL",
    ], "exactly the four declared objectives carry per-stage state"
    instantiated = {
        "ObservedOutcomeNLL": objectives.ObservedOutcomeNLL(),
        "ObservedTreatmentNLL": objectives.ObservedTreatmentNLL(),
        "MissingTreatmentMarginalNLL": objectives.MissingTreatmentMarginalNLL(
            grad_path="both"
        ),
        "PseudoLabelTreatmentNLL": PseudoLabelTreatmentNLL(
            port=Port.T_GIVEN_X,
            target=WEAK_X,
            prediction=STRONG_X,
            threshold=TAU,
            sharpening="hard",
            stop_grad="target",
            rows="all",
        ),
    }
    for name, objective in instantiated.items():
        assert not isinstance(objective, StatefulObjective), name


def test_no_existing_stage_builds_objective_state() -> None:
    """The executor's own path, on the recipes that share this schema.

    `_objective_states(stage, None)` is what `_run_stage` calls. An empty result
    on a `None` population is the proof that nothing is constructed and nothing
    can raise for a stage whose objectives are all stateless.
    """
    schema = _schema()
    for name, builder in (("fixmatch", fixmatch), ("doublematch", doublematch)):
        torch.manual_seed(3)
        for compiled in compile(builder(schema)).stages:
            assert _objective_states(compiled, None) == {}, f"{name}.{compiled.name}"
    flex = compile(flexmatch(schema)).stage("joint_fit")
    assert [
        objective.name
        for objective in flex.objectives
        if isinstance(objective.objective, StatefulObjective)
    ] == [CURRICULUM_TERM]


def _recipe() -> Recipe:
    """A fresh recipe under a fixed seed — fresh parameters, fresh everything.

    Rebuilt rather than shared, because `run_stage` trains `recipe.system` in
    place: two runs of one recipe *object* start from different parameters and
    would prove nothing about the state.

    `tau` is dropped to 0.05 so that a six-step run actually lays marks down.
    The lifecycle is what is under test here and it is only observable once the
    state has been written to; at the recipe's own 0.95 an untrained network
    marks nothing and every assertion below would hold of a state nobody
    touched. What the curriculum does at the *declared* `tau` is `flexmatch.md`
    §6.2's subject, and Tier 1 is where it is measured.
    """
    torch.manual_seed(11)
    recipe = _short(flexmatch(_schema()))
    stage = recipe.program[0]
    gate = stage.objectives[2].objective
    assert isinstance(gate, CurriculumPseudoLabelTreatmentNLL)
    eager = replace(
        stage.objectives[2],
        objective=replace(gate, threshold=replace(CURRICULUM, tau=0.05)),
    )
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(
                        *stage.objectives[:2],
                        eager,
                        *stage.objectives[3:],
                    ),
                ),
            )
        ),
    )


def _marked(result: object) -> list[float]:
    return [
        term.diagnostics["marked_fraction"]
        for record in result.records  # type: ignore[attr-defined]
        for term in record.terms
        if term.name == CURRICULUM_TERM
    ]


def test_the_state_is_built_per_stage_execution_and_not_per_recipe() -> None:
    """The property the whole shape exists for.

    A recipe is an immutable declaration, so two runs from equal declarations
    must give equal traces. If the marks lived on the objective instance the
    second run would start from the first run's curriculum and diverge at the
    first step the thresholds mattered.
    """
    first = run_stage(compile(_recipe()), "joint_fit", _dataset(), seed=5)
    second = run_stage(compile(_recipe()), "joint_fit", _dataset(), seed=5)
    assert first.trace == second.trace
    assert _marked(first) == _marked(second)
    # The curriculum has to have *moved* for that equality to mean anything: a
    # state that was never written would be equal between two runs trivially.
    assert max(_marked(first)) > 0.0
    final = first.objective_states[CURRICULUM_TERM]
    assert isinstance(final, CurriculumStatus)
    assert final.unused() < final.size


def test_two_arms_sharing_one_objective_instance_do_not_share_a_curriculum() -> None:
    """The paired-ablation footgun the lifecycle closes.

    `dataclasses.replace` on a `Weighted` keeps the *same* objective instance,
    which is how every paired ablation in this repository is built. The second
    arm below is a fresh recipe whose gate is literally the first arm's object,
    so state held on the objective would carry the first arm's marks into it.
    """
    alone = run_stage(compile(_recipe()), "joint_fit", _dataset(), seed=5)

    first = _recipe()
    shared = first.program[0].objectives[2].objective
    second = _recipe()
    stage = second.program[0]
    second = replace(
        second,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(
                        *stage.objectives[:2],
                        replace(stage.objectives[2], objective=shared),
                        *stage.objectives[3:],
                    ),
                ),
            )
        ),
    )
    assert second.program[0].objectives[2].objective is shared
    run_stage(compile(first), "joint_fit", _dataset(), seed=5)
    after = run_stage(compile(second), "joint_fit", _dataset(), seed=5)
    assert alone.trace == after.trace
    assert _marked(alone) == _marked(after)


def test_the_marks_start_unused_and_cover_the_whole_training_population() -> None:
    """Algorithm 1 line 2, over the rows the run will actually draw from."""
    recipe = _short(flexmatch(_schema()))
    compiled = compile(recipe).stage("joint_fit")
    from xty2.training.loading import build_population

    policy = recipe.data
    assert policy is not None
    population = build_population(_dataset(), policy, seed=5)
    states = _objective_states(compiled, population)
    status = states[CURRICULUM_TERM]
    assert isinstance(status, CurriculumStatus)
    assert status.size == TRAIN_ROWS
    assert status.unused() == TRAIN_ROWS
    assert float(status.thresholds(3).max()) == 0.0


# ---------------------------------------------------------------------------
# Card §4 against the plan (`FIDELITY.md` §1.2)
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


def _rendered(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def test_every_answered_card_key_reaches_the_plan() -> None:
    plan = compile(flexmatch(_schema())).plan
    answered = set(_card_section_four())
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    assert "losses.confidence_threshold" in answered
    assert plan.hyperparameters["losses.confidence_threshold"] == CURRICULUM


def test_the_gate_rule_is_one_card_key_holding_the_whole_policy() -> None:
    """§4's reading of the closed vocabulary, asserted rather than asserted at.

    `card_keys.py` refuses two fields bound to one key and tells the author to
    bind one holding a tuple. This is that: `tau`, the warm-up and the mapping
    reach the card through a single `losses.confidence_threshold`, and the
    vocabulary in `FIDELITY.md` §2 is unchanged.
    """
    keys: Mapping[str, str] = CurriculumPseudoLabelTreatmentNLL.CARD_KEYS
    assert sorted(keys.values()) == [
        "gradients.detached_targets",
        "losses.confidence_threshold",
        "losses.sharpening",
    ]
    plan = compile(flexmatch(_schema())).plan
    assert repr(plan.hyperparameters["losses.confidence_threshold"]) == (
        "curriculum(tau=0.95, warm_up=true, mapping=convex)"
    )


def test_the_card_and_the_plan_agree_on_every_value_section_four_states() -> None:
    """Key presence is not the cross-check; the values are."""
    hyperparameters = compile(flexmatch(_schema())).plan.hyperparameters
    mismatched: list[str] = []
    symbolic = {"architecture.widths_depths": {"K": "3", "X_REPR": "200"}}
    checked = 0
    for key, stated in _card_section_four().items():
        planned = hyperparameters.get(key)
        if planned is None:
            mismatched.append(f"{key}: absent from the plan")
            continue
        if isinstance(stated, str):
            if not isinstance(planned, dict) and _rendered(planned) != stated:
                mismatched.append(f"{key}: card {stated!r} vs plan {planned!r}")
            checked += 1
            continue
        assert isinstance(planned, dict), f"{key} is scoped in the card only"
        for scope, value in stated.items():
            if scope not in planned:
                mismatched.append(f"{key}[{scope}]: absent from the plan")
                continue
            resolved = value
            for symbol, concrete in symbolic.get(key, {}).items():
                resolved = resolved.replace(symbol, concrete)
            if _rendered(planned[scope]) != resolved:
                mismatched.append(
                    f"{key}[{scope}]: card {resolved!r} vs plan {planned[scope]!r}"
                )
            checked += 1
    assert not mismatched, "card and plan disagree: " + "; ".join(mismatched)
    assert checked >= 25


def test_the_card_declares_the_gate_rule_the_recipe_runs() -> None:
    stated = _card_section_four()["losses.confidence_threshold"]
    assert stated == repr(CURRICULUM)
    assert CurriculumThreshold(tau=0.95, warm_up=True, mapping="convex") == CURRICULUM


def test_the_card_records_the_tier_two_result() -> None:
    """`FIDELITY.md` §1.1: only the completed Tier 2 run sets reproduced."""
    header = CARD.read_text(encoding="utf-8").split("\n", 4)
    assert any("**Status:**" in line for line in header)
    status = next(line for line in header if "**Status:**" in line)
    assert "`reproduced`" in status
