"""Tier 0 — ReMixMatch alignment, synthetic rows, and assembly.

The numbered assertions below are `docs/recipes/remixmatch.md` §6.2's Tier 0
list, in its order. Two of them exist because the reference's warm-up is easy
to get wrong in a way a converged run hides: `PMovingAverage` is a `[128, K]`
variable *initialised to* `1/K` rather than a buffer that fills up, and
`PData` starts a `0.999` EMA that the source's million-step budget washes out
and this card's three thousand steps do not.
"""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
import torch
from xty2.core import (
    FeatureSpec,
    MixingPlan,
    OutcomeSpec,
    Port,
    Program,
    PseudoLabelAction,
    Realisation,
    Schema,
    TeacherSpec,
    ViewSpec,
    XTYBatch,
    compile,
)
from xty2.core.errors import CompileError
from xty2.core.loss import TrainContext, treatment_distribution
from xty2.core.rows import resolve_rows
from xty2.objectives import (
    AnchoredLabelGuess,
    AnchoredTargetTreatmentNLL,
    MixedTargetTreatmentNLL,
)
from xty2.recipes import remixmatch
from xty2.recipes.remixmatch import GUESS_OWNER
from xty2.views import ColumnRoll

CLASSES = 4
CAPACITY = 128
UNIFORM = torch.full((CLASSES,), 1.0 / CLASSES, dtype=torch.float64)
# 32 zeros, 16 ones, 8 twos, 8 threes over the 64 observed rows.
SUPPORT_MARGINAL = torch.tensor([0.5, 0.25, 0.125, 0.125], dtype=torch.float64)


def _schema() -> Schema:
    return Schema(
        features=tuple(FeatureSpec(f"x{i}", "continuous") for i in range(6)),
        treatment_cardinality=CLASSES,
        outcome=OutcomeSpec(),
    )


def _batch() -> XTYBatch:
    observed = torch.arange(128) < 64
    treatments = torch.cat(
        [
            torch.zeros(32, dtype=torch.long),
            torch.ones(16, dtype=torch.long),
            torch.full((8,), 2, dtype=torch.long),
            torch.full((8,), 3, dtype=torch.long),
            torch.arange(64) % CLASSES,
        ]
    )
    return XTYBatch(
        x=torch.randn(128, 6),
        t=treatments,
        y=torch.randn(128),
        t_observed=observed,
        y_observed=torch.ones(128, dtype=torch.bool),
        row_id=torch.arange(128),
    )


def _guess(**overrides: object) -> AnchoredLabelGuess:
    settings: dict[str, object] = {
        "capacity": CAPACITY,
        "labelled_decay": 0.999,
        "epsilon": 1e-6,
        "temperature": 0.5,
        "use_alignment": True,
    }
    settings.update(overrides)
    return AnchoredLabelGuess(CLASSES, **settings)  # type: ignore[arg-type]


def _prepare(
    guess: AnchoredLabelGuess, probabilities: torch.Tensor, *, step: int
) -> torch.Tensor:
    return guess.prepare(
        step=step,
        probabilities=probabilities,
        batch=_batch(),
        rows=torch.arange(64, 128),
        support_rows=torch.arange(64),
    )


def _window_mean(entries: list[torch.Tensor]) -> torch.Tensor:
    """`PMovingAverage.__call__` on a buffer pre-filled with `1/K`."""
    filled = [UNIFORM] * (CAPACITY - len(entries)) + entries
    mean = torch.stack(filled[-CAPACITY:]).mean(dim=0)
    return mean / mean.sum()


# 1, 4 -----------------------------------------------------------------------


def test_alignment_precedes_probability_power_sharpening() -> None:
    rows = torch.arange(64, 128)
    anchor = torch.tensor([0.5, 0.2, 0.2, 0.1], dtype=torch.float64)
    probabilities = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    probabilities[rows] = anchor
    guess = _guess()

    first = _prepare(guess, probabilities, step=0).index_select(0, rows)
    # Step zero reads the initial statistics; its own observations are visible
    # only to the next step, like the source's post_ops updates.
    expected = anchor * (UNIFORM + 1e-6) / (_window_mean([]) + 1e-6)
    expected = expected / expected.sum()
    expected = expected.square()
    expected = expected / expected.sum()
    assert torch.allclose(first[0], expected)

    flat = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    second = _prepare(guess, flat, step=1).index_select(0, rows)
    # p(y) after one bias-corrected update is exactly the observed marginal.
    aligned = 0.25 * (SUPPORT_MARGINAL + 1e-6) / (_window_mean([anchor]) + 1e-6)
    aligned = aligned / aligned.sum()
    aligned = aligned.square()
    assert torch.allclose(second[0], aligned / aligned.sum())


def test_alignment_is_the_identity_when_the_two_marginals_agree() -> None:
    rows = torch.arange(64, 128)
    probabilities = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    probabilities[rows] = SUPPORT_MARGINAL
    guess = _guess(temperature=1.0)
    for step in range(CAPACITY + 1):
        target = _prepare(guess, probabilities, step=step).index_select(0, rows)
    assert torch.allclose(guess.prediction_marginal, SUPPORT_MARGINAL)
    assert torch.allclose(guess.labelled_marginal, SUPPORT_MARGINAL)
    assert torch.allclose(target[0], SUPPORT_MARGINAL, atol=1e-9)


def test_sharpening_is_a_power_on_probabilities_not_a_scaled_softmax() -> None:
    rows = torch.arange(64, 128)
    logits = torch.tensor([1.5, 0.2, -0.4, 0.1], dtype=torch.float64)
    anchor = torch.softmax(logits, dim=-1)
    probabilities = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    probabilities[rows] = anchor
    guess = _guess()
    for step in range(CAPACITY):
        _prepare(guess, probabilities, step=step)
    target = _prepare(guess, probabilities, step=CAPACITY).index_select(0, rows)[0]

    # Alignment has moved q, so the two sharpening paths cannot be confused.
    assert not torch.allclose(target, torch.softmax(logits / 0.5, dim=-1), atol=1e-4)
    assert not torch.allclose(guess.labelled_marginal, guess.prediction_marginal)

    # Sharpening cannot increase entropy.
    def entropy(p: torch.Tensor) -> float:
        return float(-(p * p.clamp_min(1e-12).log()).sum())

    assert entropy(target) < entropy(anchor)


def test_sharpening_at_temperature_one_is_the_identity() -> None:
    rows = torch.arange(64, 128)
    anchor = torch.tensor([0.5, 0.2, 0.2, 0.1], dtype=torch.float64)
    probabilities = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    probabilities[rows] = anchor
    guess = _guess(temperature=1.0, use_alignment=False)
    target = _prepare(guess, probabilities, step=0).index_select(0, rows)
    assert torch.allclose(target[0], anchor)


# 2 --------------------------------------------------------------------------


def test_the_prediction_window_is_prefilled_uniform_like_pmovingaverage() -> None:
    """`PMovingAverage` averages 128 slots from the first step, not `n` of them.

    A window that grew from empty would report the current batch's own mean at
    step one, which is a much harder alignment than the source's — and at this
    card's 3,000-step budget the difference covers the whole `lambda_U` ramp.
    """
    rows = torch.arange(64, 128)
    anchor = torch.tensor([0.55, 0.25, 0.13, 0.07], dtype=torch.float64)
    probabilities = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    probabilities[rows] = anchor
    guess = _guess(use_alignment=False)

    assert torch.allclose(guess.prediction_marginal, UNIFORM)
    for step in range(33):
        _prepare(guess, probabilities, step=step)
        assert torch.allclose(
            guess.prediction_marginal, _window_mean([anchor] * (step + 1))
        )
    # Still far from the raw entry: 33 of 128 slots have been written.
    assert not torch.allclose(guess.prediction_marginal, anchor, atol=1e-3)


def test_prediction_history_is_a_128_entry_fifo() -> None:
    guess = _guess(temperature=1.0, use_alignment=False)
    for step in range(CAPACITY + 1):
        value = (
            torch.tensor([0.7, 0.1, 0.1, 0.1], dtype=torch.float64)
            if step == 0
            else torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        )
        _prepare(guess, value.expand(128, -1), step=step)
    # After 129 writes the first stream has been shifted out entirely.
    assert torch.allclose(
        guess.prediction_marginal,
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
    )


# 3 --------------------------------------------------------------------------


def test_the_window_takes_the_unaligned_anchor_and_p_of_y_the_observed_mean() -> None:
    rows = torch.arange(64, 128)
    anchor = torch.tensor([0.5, 0.2, 0.2, 0.1], dtype=torch.float64)
    probabilities = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    probabilities[rows] = anchor
    guess = _guess()
    target = _prepare(guess, probabilities, step=0).index_select(0, rows)[0]

    # The raw prediction is what enters the window — never the aligned target.
    assert torch.allclose(guess.prediction_marginal, _window_mean([anchor]))
    assert not torch.allclose(guess.prediction_marginal, _window_mean([target]))
    assert torch.allclose(guess.labelled_marginal, SUPPORT_MARGINAL)


def test_the_labelled_marginal_is_unbiased_from_its_first_update() -> None:
    """Deviation 10: the reference's 0.999 EMA starts uniform and is washed out
    by a 1,048,576-step budget. Over 3,000 steps it would still carry 5% of its
    initial value, and 37% at step 1,000, so the estimate is bias-corrected."""
    rows = torch.arange(64, 128)
    probabilities = torch.full((128, CLASSES), 0.25, dtype=torch.float64)
    probabilities[rows] = SUPPORT_MARGINAL
    guess = _guess()
    assert torch.allclose(guess.labelled_marginal, UNIFORM)
    for step in range(5):
        _prepare(guess, probabilities, step=step)
        assert torch.allclose(guess.labelled_marginal, SUPPORT_MARGINAL)
    # Uncorrected, the same decay would still be 0.5% of the way there.
    naive = (1.0 - 0.999**5) * SUPPORT_MARGINAL + 0.999**5 * torch.zeros(CLASSES)
    assert not torch.allclose(naive, SUPPORT_MARGINAL)


# 5, 8 -----------------------------------------------------------------------


def test_compiled_pool_materialises_every_member_with_shared_provenance() -> None:
    run = compile(remixmatch(_schema()))
    stage = run.stage("joint_fit")
    state = run.state(stage, _batch(), rng_key=17)
    mix = run.recipe.mixes[0]
    assert len(mix.members) == 10  # |X_hat| + |U_hat| at K = 8
    entries = 0
    for index in range(10):
        plan = state.mixing_plan(mix.output(index))
        assert isinstance(plan, MixingPlan)
        assert bool(((plan.coefficient >= 0.5) & (plan.coefficient <= 1.0)).all())
        assert plan.first_rows.numel() == 64
        entries += plan.first_rows.numel()
    assert entries == 64 + 64 * 9
    assert run.recipe.view("pretext_x").source == mix.members[0].realisation


def test_a_mixed_entry_keeps_its_first_source_row_ids() -> None:
    run = compile(remixmatch(_schema()))
    batch = _batch()
    state = run.state(run.stage("joint_fit"), batch, rng_key=5)
    mix = run.recipe.mixes[0]
    observed = torch.nonzero(batch.t_observed, as_tuple=False).flatten()
    missing = torch.nonzero(batch.t_missing, as_tuple=False).flatten()
    assert torch.equal(state.mixing_plan(mix.output(0)).first_rows, observed)
    for index in range(1, 10):
        assert torch.equal(state.mixing_plan(mix.output(index)).first_rows, missing)


# 6 --------------------------------------------------------------------------


def test_mixed_targets_never_read_a_hidden_treatment() -> None:
    """`treatment_at` zeroes every row outside the eligible set, so a target
    built from a hidden `t` would change when that `t` does. It must not."""
    schema = _schema()
    run = compile(remixmatch(schema))
    stage = run.stage("joint_fit")
    batch = _batch()
    poisoned = batch.replace(
        t=torch.where(batch.t_observed, batch.t, (batch.t + 1) % CLASSES)
    )
    ctx = TrainContext(
        global_step=0,
        schema=schema,
        stage="joint_fit",
        objective_states={GUESS_OWNER: _guess()},
    )
    weighted = next(
        item for item in stage.objectives if item.name == "mixed_labelled_treatment_nll"
    )
    values = []
    for candidate in (batch, poisoned):
        state = run.state(stage, candidate, rng_key=3)
        rows = torch.nonzero(candidate.t_observed, as_tuple=False).flatten()
        ctx = replace(ctx, objective_states={GUESS_OWNER: _guess()})
        term = weighted.objective.compute(state, candidate, rows, ctx)
        values.append(float(term.value))
    assert values[0] == pytest.approx(values[1])


# 7 --------------------------------------------------------------------------


def test_pseudo_label_actions_cannot_consume_synthetic_rows() -> None:
    recipe = remixmatch(_schema())
    stage = recipe.program[0]
    changed = replace(
        stage,
        action=PseudoLabelAction(
            port=Port.T_GIVEN_X,
            realisation=recipe.mixes[0].output(0),
        ),
    )
    with pytest.raises(CompileError, match="may not read mixed realisation"):
        compile(replace(recipe, program=Program((changed,))))


def test_teacher_passes_cannot_consume_synthetic_rows() -> None:
    recipe = remixmatch(_schema())
    stage = recipe.program[0]
    teacher_mixed = replace(recipe.mixes[0].output(0), params="teacher")
    objective = cast(
        MixedTargetTreatmentNLL,
        next(
            item.objective
            for item in stage.objectives
            if item.name == "mixed_labelled_treatment_nll"
        ),
    )
    changed = replace(objective, predictions=(teacher_mixed,))
    program = Program(
        (
            replace(
                stage,
                objectives=tuple(
                    replace(item, objective=changed)
                    if item.name == "mixed_labelled_treatment_nll"
                    else item
                    for item in stage.objectives
                ),
                teacher=TeacherSpec(
                    decay=0.999,
                    applies_to_buffers=False,
                    train_mode=False,
                    requires_grad=False,
                    role="evaluation",
                ),
            ),
        )
    )
    with pytest.raises(CompileError, match="view=mixed params=teacher"):
        compile(replace(recipe, program=program))


def test_mixing_plans_have_no_artifact_or_evaluation_surface() -> None:
    run = compile(remixmatch(_schema()))
    state = run.state(run.stage("joint_fit"), _batch(), rng_key=1)
    with pytest.raises(Exception, match="holds no mixing plan"):
        state.mixing_plan(Realisation(view="weak_x"))


# 9 --------------------------------------------------------------------------


def test_the_anchor_target_carries_no_gradient_and_pretext_stays_local() -> None:
    schema = _schema()
    run = compile(remixmatch(schema))
    stage = run.stage("joint_fit")
    batch = _batch()
    state = run.state(stage, batch, rng_key=11)
    guess = _guess()
    ctx = TrainContext(
        global_step=0,
        schema=schema,
        stage="joint_fit",
        objective_states={GUESS_OWNER: guess},
    )
    rows = torch.nonzero(batch.t_missing, as_tuple=False).flatten()
    weighted = next(item for item in stage.objectives if item.name == GUESS_OWNER)
    target = guess.prepare(
        step=0,
        probabilities=treatment_distribution(
            state, Port.T_GIVEN_X, Realisation(view="weak_x"), objective=GUESS_OWNER
        ).probs,
        batch=batch,
        rows=rows,
        support_rows=torch.nonzero(batch.t_observed, as_tuple=False).flatten(),
    )
    assert not target.requires_grad
    loss = weighted.objective.compute(state, batch, rows, ctx).value
    assert loss.requires_grad  # the prediction side still trains

    # The pretext realisation reaches no head but the pretext one: in the
    # source the rotated images feed `classifier_rot` and nothing else.
    pretext = set(state[Realisation(view="pretext_x")])
    assert Port.PRETEXT_GIVEN_X in pretext
    assert not pretext & {Port.T_GIVEN_X, Port.Y_GIVEN_XT}
    passes = {forward.realisation.view: forward.components for forward in stage.passes}
    assert passes["pretext_x"] == ("mlp_encoder", "pretext_head")


# 10 -------------------------------------------------------------------------


def test_the_guess_is_idempotent_and_order_independent_within_a_step() -> None:
    schema = _schema()
    run = compile(remixmatch(schema))
    stage = run.stage("joint_fit")
    batch = _batch()
    state = run.state(stage, batch, rng_key=13)
    names = [
        "mixed_labelled_treatment_nll",
        "mixed_unlabelled_treatment_nll",
        GUESS_OWNER,
    ]
    by_name = {item.name: item for item in stage.objectives}

    def losses(order: list[str]) -> list[float]:
        ctx = TrainContext(
            global_step=0,
            schema=schema,
            stage="joint_fit",
            objective_states={GUESS_OWNER: _guess()},
        )
        out = {}
        for name in order:
            weighted = by_name[name]
            rows = resolve_rows(batch, weighted.objective.rows)
            term = weighted.objective.compute(state, batch, rows, ctx)
            out[name] = float(term.value)
        return [out[name] for name in names]

    forward = losses(names)
    backward = losses(list(reversed(names)))
    assert forward == pytest.approx(backward)


# 11 -------------------------------------------------------------------------


def test_no_objective_in_this_recipe_applies_a_confidence_gate() -> None:
    stage = remixmatch(_schema()).program[0]
    for weighted in stage.objectives:
        assert not hasattr(weighted.objective, "confidence_threshold") or (
            weighted.objective.confidence_threshold == "n/a"
        )
    # Arbitrarily low-confidence anchors still train every eligible row.
    rows = torch.arange(64, 128)
    flat = torch.full((128, CLASSES), 1.0 / CLASSES, dtype=torch.float64)
    guess = _guess()
    target = _prepare(guess, flat, step=0).index_select(0, rows)
    assert int((target.sum(dim=-1) > 0).sum()) == rows.numel()


# 12 -------------------------------------------------------------------------


def test_mean_redux_compiles_all_k_plus_one_target_copies() -> None:
    recipe = remixmatch(_schema(), strong_draws=2, redux="mean")
    run = compile(recipe)
    plan = run.plan.render()
    assert "redux='mean'" in plan
    assert "target copies=2" in plan


def test_the_plan_prints_every_source_governed_value_without_a_card_key() -> None:
    plan = compile(remixmatch(_schema())).plan.render()
    for fragment in (
        "K=8",
        "redux='1st'",
        "alignment window=128",
        "labelled EMA=0.999",
        "epsilon=1e-06",
        "alpha=0.75",
        "rule=max",
    ):
        assert fragment in plan, fragment


# 13 -------------------------------------------------------------------------


def test_state_is_fresh_per_stage_execution() -> None:
    recipe = remixmatch(_schema())
    objective = cast(
        AnchoredTargetTreatmentNLL,
        next(
            item.objective
            for item in recipe.program[0].objectives
            if item.name == GUESS_OWNER
        ),
    )
    first = cast(AnchoredLabelGuess, objective.initial_state(None))
    second = cast(AnchoredLabelGuess, objective.initial_state(None))
    assert first is not second
    assert first.last_prepared_step is None and second.last_prepared_step is None
    assert torch.allclose(first.prediction_marginal, UNIFORM)
    assert torch.allclose(first.labelled_marginal, UNIFORM)


# Framework guards this card added -------------------------------------------


def test_a_derived_view_must_name_an_ordinary_student_realisation() -> None:
    recipe = remixmatch(_schema())
    views = tuple(
        replace(view, source=replace(view.source, params="teacher"))
        if view.name == "pretext_x" and view.source is not None
        else view
        for view in recipe.views
    )
    with pytest.raises(CompileError, match="ordinary student realisation"):
        compile(replace(recipe, views=views))


def test_a_derived_view_cannot_name_a_missing_draw() -> None:
    recipe = remixmatch(_schema(), strong_draws=2)
    views = tuple(
        replace(view, source=Realisation(view="strong_x", draw=7))
        if view.name == "pretext_x"
        else view
        for view in recipe.views
    )
    with pytest.raises(CompileError, match="unavailable"):
        compile(replace(recipe, views=views))


def test_the_pretext_term_is_batch_coupled() -> None:
    """Its labels are four constant quarters of whatever arrives, so a row's
    target depends on the other eligible rows of its own batch."""
    stage = remixmatch(_schema()).program[0]
    weighted = next(
        item for item in stage.objectives if item.name == "pretext_transform_nll"
    )
    assert weighted.objective.batch_coupled is True
    assert not torch.equal(
        ColumnRoll.labels(64, device=torch.device("cpu")),
        ColumnRoll.labels(32, device=torch.device("cpu")).repeat(2),
    )


def test_column_roll_refuses_a_schema_whose_columns_are_not_exchangeable() -> None:
    mixed = Schema(
        features=(
            FeatureSpec("a", "continuous"),
            FeatureSpec("b", "continuous"),
            FeatureSpec("c", "continuous"),
            FeatureSpec("d", "categorical"),
        ),
        treatment_cardinality=CLASSES,
        outcome=OutcomeSpec(),
    )
    view = ViewSpec(
        name="pretext_x",
        transforms=(ColumnRoll(),),
        preserves=frozenset({"t", "t_observed", "y", "y_observed", "row_id"}),
    )
    with pytest.raises(Exception, match="exchangeable"):
        view.validate(mixed)
