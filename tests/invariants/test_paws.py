"""Tier 0 — PAWS support classification, multi-view loss, and recipe plan."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from xty2.core import (
    ExternalBatches,
    FeatureSpec,
    OutcomeSpec,
    Port,
    Program,
    Realisation,
    Schema,
    State,
    TrainContext,
    TrainingPopulation,
    XTYBatch,
    compile,
)
from xty2.core.errors import CompileError
from xty2.core.rows import resolve_rows
from xty2.objectives import (
    MeanEntropyMaximisation,
    SupportSetPseudoLabelConsistency,
)
from xty2.recipes import paws
from xty2.recipes.paws import (
    LARGE_X,
    PAWS_SAMPLER,
    SHARPENING,
    SMALL_X,
    SUPPORT_CLASSIFIER,
)
from xty2.training.loading import iterate

ROOT = Path(__file__).resolve().parents[2]
FEATURES = 6
EMBEDDING = 5
CLASSES = 2


def _schema() -> Schema:
    return Schema(
        features=tuple(
            FeatureSpec(f"x{column}", "continuous") for column in range(FEATURES)
        ),
        treatment_cardinality=CLASSES,
        outcome=OutcomeSpec(),
    )


def _batch(missing: int = 3) -> XTYBatch:
    support = 4
    rows = support + missing
    return XTYBatch(
        x=torch.randn(rows, FEATURES),
        t=torch.tensor([0, 0, 1, 1, *([0] * missing)]),
        y=torch.randn(rows),
        t_observed=torch.tensor([True] * support + [False] * missing),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
    )


def _tensors(batch: XTYBatch, *, seed: int = 7) -> dict[Realisation, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        realisation: torch.randn(
            batch.batch_size,
            EMBEDDING,
            generator=generator,
            dtype=torch.float64,
            requires_grad=True,
        )
        for realisation in (*LARGE_X, *SMALL_X)
    }


def _state(tensors: dict[Realisation, Tensor]) -> State:
    return State(
        {
            realisation: {Port.X_PROJ: embedding}
            for realisation, embedding in tensors.items()
        }
    )


def _classifier(
    tensors: dict[Realisation, Tensor],
    batch: XTYBatch,
    query: Realisation,
    rows: Tensor,
    *,
    detach: bool,
) -> Tensor:
    support_rows = resolve_rows(batch, "t_observed")
    queries = tensors[query].index_select(0, rows)
    supports = torch.cat(
        [
            tensor.index_select(0, support_rows)
            for tensor in (tensors[r] for r in LARGE_X)
        ]
    )
    labels = batch.t.index_select(0, support_rows).repeat(len(LARGE_X))
    smoothed = (1.0 - 0.1) * F.one_hot(labels, CLASSES).to(
        torch.float64
    ) + 0.1 / CLASSES
    if detach:
        queries = queries.detach()
        supports = supports.detach()
    weights = torch.softmax(
        F.normalize(queries, dim=-1) @ F.normalize(supports, dim=-1).T / 0.1,
        dim=-1,
    )
    return weights @ smoothed


def _sharpen(probabilities: Tensor) -> Tensor:
    powered = probabilities.pow(1.0 / SHARPENING)
    return powered / powered.sum(dim=-1, keepdim=True)


def _eq4_reference(
    tensors: dict[Realisation, Tensor],
    batch: XTYBatch,
    rows: Tensor,
    *,
    detach_targets: bool,
) -> Tensor:
    predictions = [
        _classifier(tensors, batch, view, rows, detach=False)
        for view in (*LARGE_X, *SMALL_X)
    ]
    large_targets = [
        _sharpen(_classifier(tensors, batch, view, rows, detach=detach_targets))
        for view in LARGE_X
    ]
    mean_target = (large_targets[0] + large_targets[1]) / 2.0
    targets = [large_targets[1], large_targets[0], *([mean_target] * len(SMALL_X))]
    targets = [target.masked_fill(target < 1e-4, 0.0) for target in targets]
    return torch.stack(
        [
            -(torch.xlogy(target, prediction)).sum(dim=-1)
            for target, prediction in zip(targets, predictions, strict=True)
        ]
    ).mean()


def test_support_classifier_is_the_hand_computed_soft_nearest_neighbour() -> None:
    batch = _batch()
    tensors = _tensors(batch)
    rows = resolve_rows(batch, "t_missing")
    actual = SUPPORT_CLASSIFIER.probabilities(
        _state(tensors), batch, LARGE_X[0], LARGE_X, rows, classes=CLASSES
    )
    expected = _classifier(tensors, batch, LARGE_X[0], rows, detach=False)
    torch.testing.assert_close(actual, expected)


def test_label_smoothing_sets_the_exact_floor_and_cap() -> None:
    labels = torch.tensor([0, 1])
    matrix = SUPPORT_CLASSIFIER.smoothed_labels(labels, classes=CLASSES)
    assert float(matrix.min()) == pytest.approx(0.1 / CLASSES)
    assert float(matrix.max()) == pytest.approx(1.0 - 0.1 + 0.1 / CLASSES)


def test_eq4_uses_swapped_large_targets_and_their_mean_for_small_views() -> None:
    batch = _batch()
    tensors = _tensors(batch)
    rows = resolve_rows(batch, "t_missing")
    objective = SupportSetPseudoLabelConsistency(
        classifier=SUPPORT_CLASSIFIER,
        large=LARGE_X,
        small=SMALL_X,
        sharpening=SHARPENING,
        stop_grad="target",
        target_floor=1e-4,
    )
    actual = objective.compute(
        _state(tensors), batch, rows, TrainContext(0, _schema())
    ).value

    expected = _eq4_reference(tensors, batch, rows, detach_targets=True)
    torch.testing.assert_close(actual, expected)


def test_target_roles_are_detached_but_both_large_predictions_and_supports_train() -> (
    None
):
    batch = _batch()
    tensors = _tensors(batch)
    rows = resolve_rows(batch, "t_missing")
    objective = SupportSetPseudoLabelConsistency(
        classifier=SUPPORT_CLASSIFIER,
        large=LARGE_X,
        small=SMALL_X,
        sharpening=SHARPENING,
        stop_grad="target",
        target_floor=1e-4,
    )
    term = objective.compute(_state(tensors), batch, rows, TrainContext(0, _schema()))
    term.value.backward()  # type: ignore[no-untyped-call]

    reference_tensors = {
        view: tensor.detach().clone().requires_grad_()
        for view, tensor in tensors.items()
    }
    expected = _eq4_reference(reference_tensors, batch, rows, detach_targets=True)
    expected.backward()  # type: ignore[no-untyped-call]
    live_target_tensors = {
        view: tensor.detach().clone().requires_grad_()
        for view, tensor in tensors.items()
    }
    live_target = _eq4_reference(live_target_tensors, batch, rows, detach_targets=False)
    live_target.backward()  # type: ignore[no-untyped-call]

    support_rows = resolve_rows(batch, "t_observed")
    difference_from_live_target = 0.0
    for view in LARGE_X:
        gradient = tensors[view].grad
        reference_gradient = reference_tensors[view].grad
        live_gradient = live_target_tensors[view].grad
        assert gradient is not None
        assert reference_gradient is not None
        assert live_gradient is not None
        torch.testing.assert_close(gradient, reference_gradient)
        difference_from_live_target += float((gradient - live_gradient).abs().sum())
        assert float(gradient.index_select(0, rows).abs().sum()) > 0.0
        assert float(gradient.index_select(0, support_rows).abs().sum()) > 0.0
    for view in SMALL_X:
        gradient = tensors[view].grad
        reference_gradient = reference_tensors[view].grad
        assert gradient is not None
        assert reference_gradient is not None
        torch.testing.assert_close(gradient, reference_gradient)
        assert float(gradient.index_select(0, rows).abs().sum()) > 0.0
    assert difference_from_live_target > 1e-6


def test_me_max_is_negative_entropy_of_all_eight_sharpened_views() -> None:
    batch = _batch()
    tensors = _tensors(batch)
    rows = resolve_rows(batch, "t_missing")
    objective = MeanEntropyMaximisation(
        classifier=SUPPORT_CLASSIFIER,
        views=(*LARGE_X, *SMALL_X),
        support_views=LARGE_X,
        sharpening=SHARPENING,
    )
    actual = objective.compute(_state(tensors), batch, rows, TrainContext(0, _schema()))
    probabilities = torch.stack(
        [
            _sharpen(_classifier(tensors, batch, view, rows, detach=False))
            for view in (*LARGE_X, *SMALL_X)
        ]
    )
    marginal = probabilities.mean(dim=(0, 1))
    expected = torch.xlogy(marginal, marginal).sum()
    torch.testing.assert_close(actual.value, expected)
    actual.value.backward()  # type: ignore[no-untyped-call]
    total_gradient = 0.0
    for tensor in tensors.values():
        gradient = tensor.grad
        assert gradient is not None
        total_gradient += float(gradient.abs().sum())
    assert total_gradient > 0.0


def test_me_max_is_unchanged_when_identical_anchor_rows_are_repeated() -> None:
    batch = _batch(missing=3)
    tensors = _tensors(batch)
    objective = MeanEntropyMaximisation(
        classifier=SUPPORT_CLASSIFIER,
        views=(*LARGE_X, *SMALL_X),
        support_views=LARGE_X,
        sharpening=SHARPENING,
    )
    rows = resolve_rows(batch, "t_missing")
    first = objective.compute(_state(tensors), batch, rows, TrainContext(0, _schema()))

    repeated_batch = _batch(missing=6)
    repeated_tensors = {
        view: torch.cat(
            (tensor[:4].detach(), tensor[4:].detach().repeat_interleave(2, dim=0))
        ).requires_grad_()
        for view, tensor in tensors.items()
    }
    repeated_rows = resolve_rows(repeated_batch, "t_missing")
    second = objective.compute(
        _state(repeated_tensors),
        repeated_batch,
        repeated_rows,
        TrainContext(0, _schema()),
    )
    assert (first.n, second.n) == (3, 6)
    torch.testing.assert_close(first.value, second.value)


def test_recipe_shares_one_classifier_and_declares_both_terms_batch_coupled() -> None:
    recipe = paws(_schema())
    first, second = (weighted.objective for weighted in recipe.program[0].objectives)
    assert isinstance(first, SupportSetPseudoLabelConsistency)
    assert isinstance(second, MeanEntropyMaximisation)
    assert first.classifier is second.classifier is SUPPORT_CLASSIFIER
    assert first.batch_coupled and second.batch_coupled


def test_external_batches_cannot_hide_the_paws_batch_composition() -> None:
    recipe = paws(_schema())
    pretrain = replace(recipe.program[0], sampler=ExternalBatches())
    changed = replace(recipe, program=Program((pretrain, recipe.program[1])))
    with pytest.raises(CompileError, match="read the rest of the batch"):
        compile(changed)


def test_support_population_is_checked_against_the_stage_scope() -> None:
    recipe = paws(_schema())
    pretrain = replace(recipe.program[0], rows="t_missing", sampler=ExternalBatches())
    changed = replace(recipe, program=Program((pretrain, recipe.program[1])))
    with pytest.raises(CompileError, match=r"support rows.*empty by construction"):
        compile(changed)


def test_stratified_quota_draws_16_unique_supports_per_level_and_128_anchors() -> None:
    rows = 240
    observed = torch.zeros(rows, dtype=torch.bool)
    observed[:40] = True
    treatments = torch.zeros(rows, dtype=torch.long)
    treatments[20:40] = 1
    treatments[120:] = 1
    batch = XTYBatch(
        x=torch.randn(rows, FEATURES),
        t=treatments,
        y=torch.randn(rows),
        t_observed=observed,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
    )
    population = TrainingPopulation._issue(
        rows=batch,
        assignment="train",
        statistics={},
        fitted_on_row_ids=batch.row_id,
        spec_digest="test",
    )
    drawn = next(iterate(population, PAWS_SAMPLER, steps=1, seed=42))
    supports = drawn.t[drawn.t_observed]
    assert drawn.batch_size == 160
    assert torch.bincount(supports, minlength=2).tolist() == [16, 16]
    assert int(drawn.t_missing.sum()) == 128
    assert torch.unique(drawn.row_id).numel() == 160


def test_plan_matches_the_reviewed_paws_mechanics() -> None:
    plan = compile(paws(_schema())).plan
    pretrain = plan.stages[0]
    joint = plan.stages[1]
    assert len(pretrain.passes) == 8
    assert pretrain.trainable == ("mlp_encoder", "projection_head")
    assert joint.initialise_from == "pretrain"
    assert "projection_head" not in joint.trainable
    hyperparameters = plan.hyperparameters
    assert hyperparameters["optimisation.batch_size"] == {
        "pretrain": 160,
        "joint_fit": 160,
    }
    assert hyperparameters["optimisation.labelled_unlabelled_ratio"] == {
        "pretrain": 4.0,
        "joint_fit": 4.0,
    }
    assert hyperparameters["losses.temperature"] == {
        "pretrain.support_set_pseudo_label_consistency": 0.1,
        "pretrain.mean_entropy_maximisation": 0.1,
    }
    assert hyperparameters["losses.sharpening"] == {
        "pretrain.support_set_pseudo_label_consistency": 0.25,
        "pretrain.mean_entropy_maximisation": 0.25,
    }
    assert hyperparameters["gradients.detached_targets"] == {
        "pretrain.support_set_pseudo_label_consistency": "target"
    }
    assert hyperparameters["optimisation.lr_schedule"]["pretrain"] == (
        "warmup cosine 0.25 -> 1.0 over 17 steps, then cosine -> 0.01 at 1000 steps"
    )


def test_paws_recipe_contains_declarations_and_no_control_flow() -> None:
    tree = ast.parse((ROOT / "xty2/recipes/paws.py").read_text())
    forbidden = (ast.If, ast.For, ast.While, ast.Try, ast.Match, ast.IfExp)
    assert not [node for node in ast.walk(tree) if isinstance(node, forbidden)]


def test_card_records_review_and_passing_tier_one_evidence() -> None:
    card = (ROOT / "docs/recipes/paws.md").read_text()
    assert "Card reviewed (status → `reviewed`) | Codex | 2026-08-29" in card
    assert "Plan diffed against §3.2 and §4 | Codex | 2026-08-29" in card


def test_the_card_records_the_tier_two_result() -> None:
    """`FIDELITY.md` §1.1: only the completed Tier 2 run sets reproduced."""
    card = (ROOT / "docs/recipes/paws.md").read_text()
    assert "**Status:** `reproduced`" in card
    ledger = card.split("### 6.3 Result ledger", 1)[1].split("## 7.", 1)[0]
    assert "held_out_treatment_NLL_ratio" in ledger
    # The five criteria §6's tolerance names, each carried as its own required
    # metric: a ledger row naming fewer of them would be a narrower run than
    # the card declares.
    for metric in (
        "held_out_outcome_NLL_ratio",
        "paws_nn_over_marginal_prior_NLL",
        "terminal_marginal_entropy",
        "positive_view_alignment_gap",
    ):
        assert metric in ledger, metric
