"""Independent smoke-arm, diagnostic and review regression oracles."""

import math
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
from xty2.objectives.feature_consistency import squared_norm_floor_normalize
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
        # Literal per-arm decays at the midpoint, from
        # `1 - (1 - base) * 0.5 * (1 + cos(pi * 50/100))` = `1 - (1 - base)/2`.
        # `source_ema` keeps the inherited 0.996 so deviation 7 has a control;
        # every other pretraining arm carries the re-derived 0.68722.
        assert as_schedule(pretrain.teacher.decay)(50) == pytest.approx(
            {"zero_decay": 0.0, "source_ema": 0.998}.get(arm, 0.84361)
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


def scalar_view_alignment(
    p: torch.Tensor, t: torch.Tensor, epsilon: float = 1e-12
) -> float:
    """Card §6.4's `A` for one directional pair, one coordinate at a time.

    A Python loop for the reason `test_byol.py`'s loss oracle is one: it shares
    no call path with the diagnostic, so a torch expression that happened to be
    the same one would prove nothing. The floor is on the **squared** norm.
    """
    total = 0.0
    for row in range(p.shape[0]):
        p_squared = math.fsum(float(p[row, j]) ** 2 for j in range(p.shape[1]))
        t_squared = math.fsum(float(t[row, j]) ** 2 for j in range(t.shape[1]))
        p_scale = 1.0 / math.sqrt(max(p_squared, epsilon))
        t_scale = 1.0 / math.sqrt(max(t_squared, epsilon))
        total += math.fsum(
            float(p[row, j]) * p_scale * float(t[row, j]) * t_scale
            for j in range(p.shape[1])
        )
    return total / p.shape[0]


@pytest.mark.parametrize("scale", [1.0, 1e-7])
def test_view_alignment_uses_the_source_floor_and_not_the_cosine(scale: float) -> None:
    """`A` is the cosine above the floor and is not the cosine below it.

    The same distinction §3.1 makes for the distance, made again for §6.4's
    statistic, because the statistic is written from the same normalisation and
    a reader is most likely to assume `cosine_similarity` would have done.
    """
    generator = torch.Generator().manual_seed(7)
    p = torch.randn(9, 4, generator=generator, dtype=torch.float64) * scale
    t = torch.randn(9, 4, generator=generator, dtype=torch.float64) * scale
    unit_p = squared_norm_floor_normalize(p, 1e-12)
    unit_t = squared_norm_floor_normalize(t, 1e-12)
    measured = float((unit_p * unit_t).sum(-1).mean())
    assert measured == pytest.approx(scalar_view_alignment(p, t), abs=1e-12)

    cosine = float(
        torch.nn.functional.cosine_similarity(p, t, dim=-1, eps=1e-12).mean()
    )
    above_the_floor = scale == 1.0
    assert (measured == pytest.approx(cosine, abs=1e-9)) is above_the_floor

    # And the tie to §3.1: `A = 1 - residual/4` holds only above the floor,
    # where both sides are unit vectors. The residual is the sum of the two
    # directional squared distances, so one direction contributes 2 - 2cos.
    distance = float((unit_p - unit_t).square().sum(-1).mean())
    tied = measured == pytest.approx(1.0 - distance / 2.0, abs=1e-9)
    assert tied is above_the_floor


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
    # Card §6.4's `A` under torch's floor on the norm rather than the source's
    # floor on the square: identical above the floor, wrong below it.
    with (
        patch(
            f"{__name__}.squared_norm_floor_normalize",
            lambda value, epsilon: torch.nn.functional.normalize(
                value, dim=-1, eps=epsilon
            ),
        ),
        pytest.raises(AssertionError),
    ):
        test_view_alignment_uses_the_source_floor_and_not_the_cosine(1e-7)


def test_smoke_transfer_oracle_kills_a_fine_tuned_encoder() -> None:
    """Deviation 4, as amended: `joint_fit` may not move the transferred encoder.

    The mutant is this card's own 2026-09-16 protocol — declare the encoder
    trainable downstream — and it is the one §6.4's audit note shows costs a
    factor of forty in the endpoint's dynamic range. It must not be reachable
    again without the declaration changing.
    """
    from dataclasses import replace

    from xty2.core import Program
    from xty2.evaluation import byol_study

    original = byol_study.arm_recipe

    def thawed(recipe: object, arm: str, **options: object) -> object:
        result = original(recipe, arm, **options)  # type: ignore[arg-type]
        stages = list(result.program)
        stages[-1] = replace(
            stages[-1], trainable=("mlp_encoder", *stages[-1].trainable)
        )
        return replace(result, program=Program(tuple(stages)))

    with (
        patch.object(byol_study, "arm_recipe", thawed),
        pytest.raises(RuntimeError, match="pairing/transfer mismatch"),
    ):
        byol_study.study(42, pretrain_steps=3, fit_steps=3, ramp_steps=3)


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
