"""Tier 0 — ReMixMatch alignment, synthetic rows, and assembly."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from xty2.core import (
    FeatureSpec,
    MixingPlan,
    OutcomeSpec,
    Port,
    Program,
    PseudoLabelAction,
    Schema,
    XTYBatch,
    compile,
)
from xty2.core.errors import CompileError
from xty2.objectives import AnchoredLabelGuess
from xty2.recipes import remixmatch


def _schema() -> Schema:
    return Schema(
        features=tuple(FeatureSpec(f"x{i}", "continuous") for i in range(6)),
        treatment_cardinality=4,
        outcome=OutcomeSpec(),
    )


def _batch() -> XTYBatch:
    observed = torch.arange(128) < 64
    return XTYBatch(
        x=torch.randn(128, 6),
        t=torch.arange(128) % 4,
        y=torch.randn(128),
        t_observed=observed,
        y_observed=torch.ones(128, dtype=torch.bool),
        row_id=torch.arange(128),
    )


def test_alignment_precedes_probability_power_sharpening() -> None:
    batch = _batch()
    batch = batch.replace(t=batch.t.index_fill(0, torch.arange(64), 0))
    rows = torch.arange(64, 128)
    support = torch.arange(64)
    probabilities = torch.full((128, 4), 0.25)
    probabilities[rows] = torch.tensor([0.5, 0.2, 0.2, 0.1])
    guess = AnchoredLabelGuess(
        4,
        capacity=128,
        labelled_decay=0.5,
        epsilon=1e-6,
        temperature=0.5,
        use_alignment=True,
    )
    actual = guess.prepare(
        step=0,
        probabilities=probabilities,
        batch=batch,
        rows=rows,
        support_rows=support,
    ).index_select(0, rows)
    # Step zero reads the initial uniform statistics; its skewed observations
    # are visible only to the next step, like the source's post_ops updates.
    labelled = torch.full((4,), 0.25)
    old_prediction = torch.full((4,), 0.25)
    aligned = probabilities[rows] * (labelled + 1e-6) / (old_prediction + 1e-6)
    aligned /= aligned.sum(-1, keepdim=True)
    expected = aligned.square()
    expected /= expected.sum(-1, keepdim=True)
    assert torch.allclose(actual, expected)

    second_raw = torch.tensor([0.25, 0.25, 0.25, 0.25]).expand(128, -1)
    second = guess.prepare(
        step=1,
        probabilities=second_raw,
        batch=batch,
        rows=rows,
        support_rows=support,
    ).index_select(0, rows)
    old_labelled = torch.tensor([0.625, 0.125, 0.125, 0.125])
    old_prediction = torch.tensor([0.5, 0.2, 0.2, 0.1])
    aligned_second = second_raw[rows] * (old_labelled + 1e-6) / (old_prediction + 1e-6)
    aligned_second /= aligned_second.sum(-1, keepdim=True)
    expected_second = aligned_second.square()
    expected_second /= expected_second.sum(-1, keepdim=True)
    assert torch.allclose(second, expected_second)


def test_prediction_history_is_a_128_entry_fifo() -> None:
    batch = _batch()
    rows = torch.arange(64, 128)
    support = torch.arange(64)
    guess = AnchoredLabelGuess(
        4,
        capacity=128,
        labelled_decay=0.999,
        epsilon=1e-6,
        temperature=1.0,
        use_alignment=False,
    )
    for step in range(129):
        value = (
            torch.tensor([0.7, 0.1, 0.1, 0.1])
            if step == 0
            else torch.tensor([0.1, 0.2, 0.3, 0.4])
        )
        probabilities = value.expand(128, -1)
        guess.prepare(
            step=step,
            probabilities=probabilities,
            batch=batch,
            rows=rows,
            support_rows=support,
        )
    assert torch.allclose(
        guess.prediction_marginal,
        torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
    )


def test_compiled_pool_materialises_every_member_with_shared_provenance() -> None:
    run = compile(remixmatch(_schema()))
    stage = run.stage("joint_fit")
    state = run.state(stage, _batch(), rng_key=17)
    outputs = run.recipe.mixes[0]
    assert len(outputs.members) == 10
    for index in range(10):
        plan = state.mixing_plan(outputs.output(index))
        assert isinstance(plan, MixingPlan)
        assert bool(((plan.coefficient >= 0.5) & (plan.coefficient <= 1.0)).all())
        assert plan.first_rows.numel() == 64
    assert (
        run.recipe.view("pretext_x").source
        == run.recipe.mixes[0].members[0].realisation
    )


def test_mean_redux_compiles_all_k_plus_one_target_copies() -> None:
    recipe = remixmatch(_schema(), strong_draws=2, redux="mean")
    run = compile(recipe)
    plan = run.plan.render()
    assert "redux='mean'" in plan
    assert "target copies=2" in plan


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
