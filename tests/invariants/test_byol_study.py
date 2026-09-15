"""Independent smoke-arm, diagnostic and review regression oracles."""

from unittest.mock import patch

import pytest
import torch
from xty2.core import LARS, CompileError, ComponentGraph, Port, compile
from xty2.core.schedules import as_schedule
from xty2.evaluation.benchmarks.common import continuous_schema
from xty2.evaluation.byol_study import (
    ARMS,
    arm_recipe,
    encoder_effective_rank,
    require_equal,
    training_batch,
)
from xty2.evaluation.simsiam_study import embedding_metrics, snapshot
from xty2.evaluation.vicreg_views import OracleSymmetry
from xty2.objectives import NormalizedSquaredFeatureConsistency
from xty2.recipes import byol
from xty2.training.teacher import EMATeacher


def test_training_payload_hides_oracle_treatments_functionally() -> None:
    from xty2.evaluation.benchmarks.common import two_cluster_population

    batch = two_cluster_population(8, seed=3, row_offset=0, low=0.5).batch
    mask = torch.tensor([True, False, True, False, True, False, True, False])
    batch = batch.replace(t=torch.ones(8, dtype=torch.long), t_observed=mask)
    before = batch.clone()
    hidden = training_batch(batch)
    assert batch.equal_to(before)
    assert hidden.equal_to(batch.replace(t=torch.tensor([1, 0, 1, 0, 1, 0, 1, 0])))


def test_smoke_arms_bind_budgets_schedules_and_actual_initial_tensors() -> None:
    initial = None
    for arm in ARMS:
        torch.manual_seed(48)
        recipe = arm_recipe(
            byol(
                continuous_schema(6),
                first_transforms=(OracleSymmetry(),),
                second_transforms=(OracleSymmetry(),),
            ),
            arm,
        )
        run = compile(recipe)
        if initial is None:
            initial = snapshot(run.graph)
        require_equal(snapshot(run.graph), initial)
        fit = recipe.program[-1]
        assert fit.steps == 200
        assert fit.objectives[-1].schedule(100) == 0.25
        assert fit.objectives[-1].schedule(200) == 0.5
        assert fit.teacher is None
        if arm == "no_pretrain":
            assert len(recipe.program) == 1 and fit.initialise_from is None
            continue
        pretrain = recipe.program[0]
        assert pretrain.steps == 100 and fit.initialise_from == "pretrain"
        assert pretrain.optimiser.lr_at(0) == 0.0
        assert pretrain.optimiser.lr_at(1) == 0.1
        assert pretrain.optimiser.lr_at(100) == 0.0
        assert pretrain.teacher is not None
        assert pretrain.teacher.applies_to_buffers is False
        assert pretrain.teacher.train_mode is True
        assert as_schedule(pretrain.teacher.decay)(50) == pytest.approx(
            0 if arm == "zero_decay" else 0.998
        )
        passes = run.stages[0].passes
        assert all(
            "byol_predictor" not in p.components
            for p in passes
            if p.realisation.params == "teacher"
        )
        for term in pretrain.objectives:
            objective = term.objective
            assert isinstance(objective, NormalizedSquaredFeatureConsistency)
            assert objective.prediction_port == (
                Port.X_PROJ if arm == "no_predictor" else Port.X_PRED
            )
            assert objective.target_port == Port.X_PROJ
        assert ("byol_predictor" in pretrain.trainable) == (arm != "no_predictor")


def test_rank_uses_centred_energy_and_the_byol_floor() -> None:
    assert encoder_effective_rank(torch.ones(8, 3)) == 0
    rank_one = torch.tensor([[-2.0, -4.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 4.0]])
    assert encoder_effective_rank(rank_one) == pytest.approx(1)
    isotropic = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    assert encoder_effective_rank(isotropic + 7) == pytest.approx(2)
    assert encoder_effective_rank(isotropic * 1e-7) == 0
    assert encoder_effective_rank(isotropic * 1e-6) == pytest.approx(2)
    # Unequal singular values distinguish energy probabilities from s/sum(s).
    anisotropic = isotropic * torch.tensor([1.0, 2.0])
    assert encoder_effective_rank(anisotropic) == pytest.approx(1.6493848884661177)


@pytest.mark.parametrize(
    "bad",
    [
        torch.full((4, 2), float("nan")),
        torch.full((4, 2), float("inf")),
        torch.empty(4, 0),
        torch.ones(4),
        torch.ones(1, 2),
    ],
)
def test_rank_rejects_invalid_inputs(bad: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        encoder_effective_rank(bad)


def test_pairing_oracle_rejects_missing_changed_and_empty_states() -> None:
    for left, right in (
        ({"x": torch.ones(2)}, {}),
        ({"x": torch.ones(2)}, {"x": torch.zeros(2)}),
        ({}, {}),
    ):
        with pytest.raises(RuntimeError):
            require_equal(left, right)


@pytest.mark.parametrize("knob", ["lr", "momentum", "eta"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_direct_lars_rejects_nonfinite_hyperparameters(knob: str, value: float) -> None:
    options = {"lr": 0.1, "momentum": 0.9, "eta": 0.001, knob: value}
    with pytest.raises(CompileError, match="finite"):
        LARS([torch.nn.Parameter(torch.ones(2, 2))], **options)


def test_new_oracles_kill_rank_pairing_and_validation_mutants() -> None:
    from xty2.evaluation import byol_study

    # No numerical floor: the existing SimSiam metric differs from BYOL here.
    with (
        patch.object(
            byol_study,
            "encoder_effective_rank",
            side_effect=lambda z: embedding_metrics(z)["effective_rank"],
        ),
        patch(f"{__name__}.encoder_effective_rank", byol_study.encoder_effective_rank),
        pytest.raises(AssertionError),
    ):
        test_rank_uses_centred_energy_and_the_byol_floor()
    with (
        patch(f"{__name__}.require_equal", return_value=None),
        pytest.raises(pytest.fail.Exception),
    ):
        test_pairing_oracle_rejects_missing_changed_and_empty_states()
    with (
        patch("xty2.core.optimisation._require_finite", return_value=None),
        pytest.raises(pytest.fail.Exception),
    ):
        test_direct_lars_rejects_nonfinite_hyperparameters("eta", float("nan"))
    with (
        patch(f"{__name__}.training_batch", side_effect=lambda batch: batch),
        pytest.raises(AssertionError),
    ):
        test_training_payload_hides_oracle_treatments_functionally()


@pytest.mark.parametrize("mutant", ["frozen_target", "double_update", "buffer_ema"])
def test_smoke_update_oracle_kills_teacher_mutants(mutant: str) -> None:
    from dataclasses import replace

    from xty2.evaluation.byol_study import study

    original = EMATeacher.update

    def changed(teacher: EMATeacher, student: ComponentGraph, step: int) -> None:
        if mutant == "frozen_target":
            return
        if mutant == "buffer_ema":
            teacher.spec = replace(teacher.spec, applies_to_buffers=True)
        original(teacher, student, step)
        if mutant == "double_update":
            original(teacher, student, step)

    with (
        patch.object(EMATeacher, "update", changed),
        pytest.raises((AssertionError, RuntimeError)),
    ):
        study(42)


def test_smoke_update_oracle_kills_wrong_lars_momentum() -> None:
    from xty2.evaluation.byol_study import study

    original = LARS.step

    def changed(optimiser: LARS) -> None:
        for group in optimiser.param_groups:
            group["momentum"] = 0.0
        original(optimiser)

    with patch.object(LARS, "step", changed), pytest.raises(AssertionError):
        study(42)
