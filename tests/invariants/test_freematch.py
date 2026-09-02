"""Tier 0 — the FreeMatch recipe, its card-plan boundary and its shared state.

Three things are asserted here that no earlier recipe's Tier 0 file has had to
assert.

**Two objectives, one state.** Eq. (12) gives eq. (8) and eq. (11) separate
weights, so they are two `Weighted` terms; eqs. (5), (6) and (10) are one set of
statistics both read. `freematch.md` §5.1 takes one sentence of `DESIGN.md` §4
for that, on the condition that the shared update is idempotent within a step —
so `test_the_two_freematch_terms_may_be_declared_in_either_order` runs the stage
with the two lines swapped and compares the trace to the last bit.

**The gate is open at step 0.** `tau_0(c) = 1/K`, and a `K = 2` softmax has
`max(q) >= 0.5` on every row, so eq. (8) is ungated on the whole batch at
initialisation. That is card §2's first limitation and it is what makes
deviation 2 — `flexmatch`'s label-preserving strong view rather than
`fixmatch`'s — load-bearing here rather than cosmetic.

**The state needs no population.** `flexmatch.md` §5.1 chose
`initial_state(population: TrainingPopulation | None)` over a required
population *because* this recipe would not need one. That prediction is checked
against the executor's own path rather than against the objective alone.
"""

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
    CosineDecay,
    Dataset,
    ExternalBatches,
    FeatureSpec,
    OutcomeSpec,
    Port,
    Program,
    Quota,
    QuotaSampler,
    Recipe,
    Schema,
    StatefulObjective,
    XTYBatch,
    compile,
)
from xty2.core.errors import CompileError
from xty2.objectives import (
    SelfAdaptiveFairness,
    SelfAdaptiveThreshold,
    SelfAdaptiveThresholds,
    SelfAdaptiveThresholdTreatmentNLL,
)
from xty2.recipes import freematch
from xty2.recipes.fixmatch import (
    STRONG_X,
    WEAK_MASK_RATE,
    WEAK_X,
    WEAK_X_LABELLED,
)
from xty2.recipes.flexmatch import STRONG_MASK_RATE
from xty2.recipes.freematch import (
    EMA_DECAY,
    FAIRNESS_WEIGHT,
    FREEMATCH_STEPS,
    SAT,
    SAT_TERM,
    UNSUPERVISED_WEIGHT,
)
from xty2.training import run_stage
from xty2.training.executors import _objective_states

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "freematch.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "freematch.py"
FAIRNESS_TERM = "self_adaptive_fairness"
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
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


def _gate(recipe: Recipe) -> SelfAdaptiveThresholdTreatmentNLL:
    objective = recipe.program[0].objectives[2].objective
    assert isinstance(objective, SelfAdaptiveThresholdTreatmentNLL)
    return objective


def _fairness(recipe: Recipe) -> SelfAdaptiveFairness:
    objective = recipe.program[0].objectives[3].objective
    assert isinstance(objective, SelfAdaptiveFairness)
    return objective


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_the_stage_has_exactly_the_five_reviewed_objectives() -> None:
    stage = compile(freematch(_schema())).stage("joint_fit")
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        SAT_TERM,
        FAIRNESS_TERM,
        "missing_treatment_marginal_nll",
    ]
    # `all` for eqs. (8) and (11) is FixMatch's footnote 2, inherited.
    assert [objective.rows for objective in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("all",),
        ("all",),
        ("t_missing",),
    ]
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
        UNSUPERVISED_WEIGHT,
        FAIRNESS_WEIGHT,
        0.5,
    ]


def test_the_two_freematch_weights_are_the_papers_and_constant() -> None:
    """Eq. (12) states `w_u` and `w_f` as fixed scalars; adaptation is in `tau`."""
    stage = compile(freematch(_schema())).stage("joint_fit")
    assert UNSUPERVISED_WEIGHT == 1.0
    assert FAIRNESS_WEIGHT == 0.05
    for index, nominal in ((2, 1.0), (3, 0.05)):
        weight = stage.objectives[index].weight
        assert weight.describe() == f"constant {nominal}"
        steps = (0, 1, 1_000, FREEMATCH_STEPS)
        assert [weight(step) for step in steps] == [nominal] * 4


def test_the_gate_is_the_papers_own() -> None:
    """Eq. (8)'s two sides, its stop-gradient, its population and its rule."""
    objective = _gate(freematch(_schema()))
    assert objective.port is Port.T_GIVEN_X
    assert objective.target == WEAK_X
    assert objective.prediction == STRONG_X
    assert objective.sharpening == "hard"
    assert objective.stop_grad == "target"
    assert objective.rows == "all"
    assert objective.num_treatments == 2
    assert objective.detaches == frozenset({(Port.T_GIVEN_X, WEAK_X)})
    assert objective.threshold == SAT
    assert objective.threshold.decay == EMA_DECAY == 0.999


def test_the_fairness_term_names_the_gate_it_shares_statistics_with() -> None:
    """The sibling read, as a property of the assembly rather than of prose."""
    recipe = freematch(_schema())
    fairness = _fairness(recipe)
    assert fairness.statistics == SAT_TERM == _gate(recipe).name
    assert fairness.rows == "all"
    assert fairness.detaches == frozenset({(Port.T_GIVEN_X, WEAK_X)})
    assert fairness.requires == frozenset(
        {(Port.T_GIVEN_X, WEAK_X), (Port.T_GIVEN_X, STRONG_X)}
    )


def test_the_two_freematch_terms_share_one_row_population() -> None:
    """Eqs. (5), (6) and (10) are averages over one unlabelled batch.

    Both terms fold that batch into the shared state, so if they were entitled
    to different rows the statistics would depend on which one the mixer reached
    first. `SelfAdaptiveThresholds.observe` refuses a repeat at one step with a
    different row count; this is the declaration that keeps it from firing.
    """
    recipe = freematch(_schema())
    assert _gate(recipe).rows == _fairness(recipe).rows == "all"
    stage = compile(recipe).stage("joint_fit")
    gate, fairness = stage.objectives[2], stage.objectives[3]
    assert gate.rows == fairness.rows == ("all",)


def test_both_freematch_terms_declare_themselves_batch_coupled() -> None:
    """Algorithm 1 lines 3-5 before line 9, and what the compiler does with it.

    `flexmatch.md` §5.1 predicted the opposite ordering and therefore the
    opposite answer here; `freematch.md` §5.1 records the correction. The
    consequence is enforced rather than documented: a stage holding either term
    may not hand batch construction back to a caller.
    """
    stage = compile(freematch(_schema())).stage("joint_fit")
    coupled = [
        objective.name
        for objective in stage.objectives
        if objective.objective.batch_coupled
    ]
    assert coupled == [SAT_TERM, FAIRNESS_TERM]


def test_a_stage_holding_these_terms_cannot_declare_external_batches() -> None:
    """`DESIGN.md` §7's bar, on the recipe that now has two coupled terms."""
    recipe = freematch(_schema())
    stage = recipe.program[0]
    external = replace(
        recipe,
        program=Program((replace(stage, sampler=ExternalBatches()),)),
        # A `data` policy with nothing sampling it is its own compile error
        # (`DESIGN.md` §7.1), and it fires first; dropped so the failure under
        # test is the batch-coupling bar and not that one.
        data=None,
    )
    with pytest.raises(CompileError, match="batch"):
        compile(external)


def test_the_labelled_term_reads_the_second_weak_draw() -> None:
    """Eq. (3) on its own draw of `omega`, per FixMatch's footnote 2."""
    stage = compile(freematch(_schema())).stage("joint_fit")
    assert (Port.T_GIVEN_X, WEAK_X_LABELLED) in stage.objectives[1].objective.requires
    assert WEAK_X.draw == 0 and WEAK_X_LABELLED.draw == 1


def test_the_rate_schedule_is_the_papers_at_our_budget() -> None:
    run = compile(freematch(_schema()))
    schedule = run.stage("joint_fit").stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    assert schedule.steps == run.stage("joint_fit").steps == FREEMATCH_STEPS
    for step in (0, 750, 1_500, FREEMATCH_STEPS):
        assert schedule(step) == pytest.approx(
            math.cos((7.0 / 16.0) * math.pi * step / FREEMATCH_STEPS)
        )


# ---------------------------------------------------------------------------
# The views (deviation 2)
# ---------------------------------------------------------------------------


def test_the_strong_view_is_the_one_flexmatch_measured() -> None:
    """Deviation 2: `flexmatch`'s 0.2, not `fixmatch`'s 0.5.

    Card §2's first limitation is why this recipe cannot inherit the 0.5 the way
    `fixmatch` could: `tau_0(c) = 1/K` leaves eq. (8) ungated on the whole batch
    at `K = 2`, which is the configuration `flexmatch.md` §5.2 measured and
    §6.2 saw lock on three initialisation seeds of five.
    """
    plan = compile(freematch(_schema())).plan
    assert [view.name for view in plan.views] == ["weak_x", "strong_x"]
    weak, strong = plan.views
    assert weak.transforms == (
        f"FeatureMask(p={WEAK_MASK_RATE}, columns=all, value=0.0)",
    )
    assert weak.draws == 2
    assert strong.transforms == (
        f"FeatureMask(p={WEAK_MASK_RATE}, columns=all, value=0.0)",
        f"FeatureMask(p={STRONG_MASK_RATE}, columns=all, value=0.0)",
    )
    assert STRONG_MASK_RATE == 0.2


def test_the_cards_prose_about_the_views_matches_the_plan() -> None:
    """§4's YAML has no key for a view, so nothing else checks these two lines.

    The same check `flexmatch`'s Tier 0 carries, and for the same reason: both
    of its view lines went stale after deviation 2 moved the rate, and the
    card-key cross-check could not see it because it parses §4's YAML only.
    """
    text = CARD.read_text(encoding="utf-8")
    mapping = text.split("### 3.2 Mapping to xty2", 1)[1].split("## 4.", 1)[0]
    checklist = text.split("## 4. Mechanics checklist", 1)[1].split("## 5.", 1)[0]
    for section, where in ((mapping, "§3.2"), (checklist, "§4")):
        assert f"FeatureMask(p={STRONG_MASK_RATE})" in section, (
            f"{where} does not name the strong view the plan runs "
            f"(p={STRONG_MASK_RATE})"
        )
        assert f"FeatureMask(p={WEAK_MASK_RATE})" in section
        for line in section.splitlines():
            if "FeatureMask(p=0.5)" in line:
                assert "fixmatch" in line or "deviation 2" in line, (
                    f"{where} names FeatureMask(p=0.5) without saying it is "
                    f"`fixmatch`'s: {line!r}"
                )


def test_the_two_terms_are_the_only_lines_they_add_to_the_plan() -> None:
    """What the reviewer diffs, as a snapshot of both `plan_details` blocks."""
    stage = compile(freematch(_schema())).stage("joint_fit")
    assert stage.objectives[2].plan_details == (
        "label = arg max of the target realisation",
        "gate = max prob > T(label), the per-class threshold (eq. 8)",
        "tau_t = 0.999 tau_(t-1) + 0.001 mean_b max(q_b) (eq. 5)",
        "p~_t = 0.999 p~_(t-1) + 0.001 mean_b q_b (eq. 6)",
        "T(c) = MaxNorm(p~_t)(c) * tau_t (eq. 7)",
        "tau_0 = p~_0(c) = h~_0(c) = 1/K, and step 0 folds in no batch",
        "the three EMAs fold in this batch before this batch is gated",
        "p~_t and h~_t are [2], checked against the schema",
        "denominator = every eligible row; rejected rows contribute 0",
    )
    assert stage.objectives[3].plan_details == (
        f"reads tau_t, p~_t and h~_t from objective {SAT_TERM!r}",
        "p_bar = mean of the strong-view probabilities over retained rows",
        "h_bar = histogram of the strong-view arg max over retained rows",
        "A = SumNorm(p~_t / h~_t), B = SumNorm(p_bar / h_bar) (eq. 11)",
        "classes with an empty h_bar bin leave both SumNorms (card §7)",
        "loss = H(A, B), eq. (11) without its minus (card deviation 7)",
        "fewer than two surviving classes contributes 0",
    )


# ---------------------------------------------------------------------------
# The shared state (`freematch.md` §5.1)
# ---------------------------------------------------------------------------


def _batch(rows: int = TRAIN_ROWS) -> XTYBatch:
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(rows, 4, generator=generator)
    x[:, 3] = (torch.arange(rows) % 2).float()
    return XTYBatch(
        x=x,
        t=torch.arange(rows) % 2,
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
    stage = recipe.program[0]
    schedule = stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    data = recipe.data
    assert data is not None
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
        data=replace(data, missingness=replace(data.missingness, observed=16)),
    )


def _swapped(recipe: Recipe) -> Recipe:
    """The same stage with eq. (11) declared before eq. (8)."""
    stage = recipe.program[0]
    objectives = (
        *stage.objectives[:2],
        stage.objectives[3],
        stage.objectives[2],
        *stage.objectives[4:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _recipe() -> Recipe:
    """A fresh recipe under a fixed seed — fresh parameters, fresh everything.

    Rebuilt rather than shared, because `run_stage` trains `recipe.system` in
    place: two runs of one recipe *object* start from different parameters and
    would prove nothing about the state.

    The decay is dropped to 0.9 so that six steps move `tau_t` visibly. What
    the mechanism does at the declared 0.999 is card §6's subject, and Tier 1 is
    where it is measured.
    """
    torch.manual_seed(11)
    recipe = _short(freematch(_schema()))
    stage = recipe.program[0]
    eager = replace(
        stage.objectives[2],
        objective=replace(_gate(recipe), threshold=SelfAdaptiveThreshold(decay=0.9)),
    )
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(*stage.objectives[:2], eager, *stage.objectives[3:]),
                ),
            )
        ),
    )


def _tau(result: object) -> list[float]:
    return [
        term.diagnostics["tau_global"]
        for record in result.records  # type: ignore[attr-defined]
        for term in record.terms
        if term.name == SAT_TERM
    ]


def test_the_state_is_built_per_stage_execution_and_not_per_recipe() -> None:
    """The property the whole shape exists for (`DESIGN.md` §4).

    A recipe is an immutable declaration, so two runs from equal declarations
    must give equal traces. If the EMAs lived on the objective instance the
    second run would start from the first run's `tau_t` and diverge at the first
    step the gate mattered.
    """
    first = run_stage(compile(_recipe()), "joint_fit", _dataset(), seed=5)
    second = run_stage(compile(_recipe()), "joint_fit", _dataset(), seed=5)
    assert first.trace == second.trace
    assert _tau(first) == _tau(second)
    # The state has to have *moved* for that equality to mean anything.
    assert max(_tau(first)) > min(_tau(first))


def test_the_two_freematch_terms_may_be_declared_in_either_order() -> None:
    """§5.1's shape obligation, at the level the recipe is written at.

    The mixer computes objectives in declaration order and both terms fold the
    batch into one state, so without the idempotent update the loss would depend
    on which of two lines came first. Bit-identical traces are the assertion.
    """
    declared = run_stage(compile(_recipe()), "joint_fit", _dataset(), seed=5)
    swapped = run_stage(compile(_swapped(_recipe())), "joint_fit", _dataset(), seed=5)
    assert _tau(declared) == _tau(swapped)
    for ours, theirs in zip(declared.records, swapped.records, strict=True):
        assert {term.name: term.value for term in ours.terms} == {
            term.name: term.value for term in theirs.terms
        }


def test_the_state_is_built_without_a_training_population() -> None:
    """`flexmatch.md` §5.1's prediction, through the executor's own path.

    That card chose `TrainingPopulation | None` because FreeMatch's statistics
    are batch averages with no `N` to count. `_objective_states(stage, None)` is
    what `_run_stage` calls, so a `None` population producing a usable state is
    the prediction holding on the path that matters.
    """
    compiled = compile(_short(freematch(_schema()))).stage("joint_fit")
    states = _objective_states(compiled, None)
    assert list(states) == [SAT_TERM]
    state = states[SAT_TERM]
    assert isinstance(state, SelfAdaptiveThresholds)
    assert state.classes == 2
    assert state.tau == pytest.approx(0.5)
    assert state.last_observed_step is None


def test_only_the_gate_declares_state_and_the_fairness_term_reads_it() -> None:
    """Opt-in: the reader declares nothing, so the executor builds nothing for it."""
    stage = compile(freematch(_schema())).stage("joint_fit")
    stateful = [
        objective.name
        for objective in stage.objectives
        if isinstance(objective.objective, StatefulObjective)
    ]
    assert stateful == [SAT_TERM]
    assert not isinstance(_fairness(freematch(_schema())), StatefulObjective)


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
    plan = compile(freematch(_schema())).plan
    answered = set(_card_section_four())
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    assert "losses.confidence_threshold" in answered
    assert plan.hyperparameters["losses.confidence_threshold"] == SAT


def test_the_gate_rule_is_one_card_key_holding_the_whole_policy() -> None:
    """§4's reading of the closed vocabulary, and the second card to take it.

    FreeMatch's gate contains no threshold at all — `tau_t(c)` is a function of
    the training history and `lambda` is the only number a recipe sets — so a
    `float` bound to `losses.confidence_threshold` would have nothing to hold.
    """
    keys: Mapping[str, str] = SelfAdaptiveThresholdTreatmentNLL.CARD_KEYS
    assert sorted(keys.values()) == [
        "gradients.detached_targets",
        "losses.confidence_threshold",
        "losses.sharpening",
    ]
    plan = compile(freematch(_schema())).plan
    assert repr(plan.hyperparameters["losses.confidence_threshold"]) == (
        "self_adaptive(decay=0.999)"
    )
    # The reader binds nothing: `lambda` is one number of one rule, and two
    # objectives binding one canonical key is what `card_keys.py` refuses.
    assert not hasattr(SelfAdaptiveFairness, "CARD_KEYS")


def test_the_card_and_the_plan_agree_on_every_value_section_four_states() -> None:
    """Key presence is not the cross-check; the values are."""
    hyperparameters = compile(freematch(_schema())).plan.hyperparameters
    mismatched: list[str] = []
    symbolic = {"architecture.widths_depths": {"K": "2", "X_REPR": "200"}}
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
    assert checked >= 28


def test_the_card_declares_the_gate_rule_the_recipe_runs() -> None:
    stated = _card_section_four()["losses.confidence_threshold"]
    assert stated == repr(SAT)
    assert SelfAdaptiveThreshold(decay=0.999) == SAT


def test_the_card_status_matches_the_evidence() -> None:
    """The declared ten-seed Tier 2 target passes every required metric."""
    header = CARD.read_text(encoding="utf-8").split("\n", 4)
    status = next(line for line in header if "**Status:**" in line)
    assert status == "**Status:** `reproduced`"


def test_the_literal_reading_of_equation_eleven_is_expressible() -> None:
    """Deviation 7's claim that its alternative needs no code.

    Card §6.1 declares a `literal` arm — eq. (11) exactly as printed — and says
    it costs nothing to run because negating `w_f` negates the term. If that
    stopped being true the deviation would become an argument nobody can check,
    which is the thing §5 exists to prevent.
    """
    recipe = freematch(_schema())
    stage = recipe.program[0]
    literal = replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    objectives=(
                        *stage.objectives[:3],
                        replace(stage.objectives[3], weight=-FAIRNESS_WEIGHT),
                        *stage.objectives[4:],
                    ),
                ),
            )
        ),
    )
    plan = compile(literal).plan
    assert plan.hyperparameters["losses.weights"][f"joint_fit.{FAIRNESS_TERM}"] == (
        -FAIRNESS_WEIGHT
    )
