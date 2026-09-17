"""Tier 0 — independent oracles for BYOL's loss, schedules, LARS and target.

`docs/recipes/byol.md` §6.3 predeclared this suite. Every oracle here is written
from the pinned source rather than from the implementation it checks:
`utils/helpers.regression_loss` and `l2_normalize` for the loss,
`utils/schedules.target_ema` and `learning_schedule` for the two curves,
`utils/optimizers.lars` for the update rule, and `byol_experiment._update_fn`
for the order the target moves in. The mutant tests at the bottom are the
evidence that the oracles bite: each one breaks a single source mechanic and
asserts the corresponding oracle fails.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from torch import Tensor, nn
from xty2.components import BYOLPredictor, BYOLProjector
from xty2.core import (
    LARS,
    CompiledRun,
    CompileError,
    Constant,
    CosineEMADecay,
    GradientClipping,
    GraphError,
    LossError,
    OptimiserSpec,
    Port,
    Program,
    Schedule,
    State,
    TrainContext,
    WarmupCosine,
    WeightDecay,
    Xty2Error,
    adapts_under_lars,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    SEPARATED,
    continuous_schema,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.vicreg_views import OracleSymmetry
from xty2.objectives import NormalizedSquaredFeatureConsistency
from xty2.recipes import byol
from xty2.recipes.byol import (
    BASE_TARGET_EMA,
    BATCH_SIZE,
    DATA_POLICY,
    ONLINE_A,
    ONLINE_B,
    PRETRAIN_LR,
    PRETRAIN_STEPS,
    TARGET_A,
    TARGET_B,
    WARMUP_STEPS,
)
from xty2.training import executors
from xty2.training.executors import trainable_only
from xty2.training.loading import build_population
from xty2.training.loss_mixer import LossMixer
from xty2.training.teacher import EMATeacher

from tests.invariants import test_doublematch as card_parser

CARD = Path(__file__).resolve().parents[2] / "docs" / "recipes" / "byol.md"
ROWS = 7
WIDTH = 3


def recipe_run() -> CompiledRun:
    return compile(
        byol(
            continuous_schema(6),
            first_transforms=(OracleSymmetry(),),
            second_transforms=(OracleSymmetry(),),
        )
    )


def objective(
    reverse: bool = False, stop: Literal["target", "none"] = "target"
) -> NormalizedSquaredFeatureConsistency:
    return NormalizedSquaredFeatureConsistency(
        prediction_port=Port.X_PRED,
        target_port=Port.X_PROJ,
        prediction=ONLINE_B if reverse else ONLINE_A,
        target=TARGET_A if reverse else TARGET_B,
        stop_grad=stop,
        name="byol_b_to_a" if reverse else "byol_a_to_b",
    )


def tensors(scale: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    base = torch.arange(1, ROWS * WIDTH + 1, dtype=torch.float64).reshape(ROWS, WIDTH)
    return tuple(
        (torch.sin(base * c) * scale).requires_grad_() for c in (0.4, 0.7, 1.1, 1.7)
    )  # type: ignore[return-value]


def fixture_batch(rows: int = ROWS) -> Any:
    return two_cluster_population(rows, seed=1, row_offset=0, low=SEPARATED).batch


def context() -> TrainContext:
    return TrainContext(global_step=0, schema=continuous_schema(6))


# ---------------------------------------------------------------------------
# `helpers.regression_loss`, written out one coordinate at a time
# ---------------------------------------------------------------------------


def scalar_regression_loss(x: Tensor, y: Tensor, epsilon: float = 1e-12) -> Tensor:
    """`sum((l2_normalize(x) - l2_normalize(y))**2, axis=-1)`, meaned over rows.

    Deliberately a Python loop over rows and coordinates: the point of this
    oracle is that it shares no call path with the objective, so a torch
    expression that happened to be the same one would prove nothing. The floor
    is applied to the **squared** sum exactly as `l2_normalize` applies it.
    """
    terms = []
    for row in range(x.shape[0]):
        x_squared, y_squared = (x.new_zeros(()) for _ in range(2))
        for j in range(x.shape[1]):
            x_squared = x_squared + x[row, j] ** 2
            y_squared = y_squared + y[row, j] ** 2
        x_scale = 1.0 / torch.sqrt(torch.maximum(x_squared, x.new_tensor(epsilon)))
        y_scale = 1.0 / torch.sqrt(torch.maximum(y_squared, y.new_tensor(epsilon)))
        distance = x.new_zeros(())
        for j in range(x.shape[1]):
            distance = distance + (x[row, j] * x_scale - y[row, j] * y_scale) ** 2
        terms.append(distance)
    return torch.stack(terms).mean()


@pytest.mark.parametrize("scale", [1.0, 1e-7])
@pytest.mark.parametrize("stop", ["target", "none"])
def test_directional_values_and_gradients(
    scale: float, stop: Literal["target", "none"]
) -> None:
    """Both directions at weight 1, against the scalar oracle and its autograd.

    `scale=1e-7` puts every row's squared norm below `1e-12`, which is the only
    regime where the source's floor is observable: `test_the_floor_is_on_the
    _squared_norm` establishes that a cosine cannot reproduce these numbers.
    """
    pa, za, pb, zb = tensors(scale)
    state = State(
        {
            ONLINE_A: {Port.X_PRED: pa},
            ONLINE_B: {Port.X_PRED: pb},
            TARGET_A: {Port.X_PROJ: za},
            TARGET_B: {Port.X_PROJ: zb},
        }
    )
    batch, ctx, rows = fixture_batch(), context(), torch.arange(ROWS)
    loss = objective(stop=stop).compute(state, batch, rows, ctx).value
    loss = loss + objective(True, stop).compute(state, batch, rows, ctx).value
    expected = scalar_regression_loss(
        pa, zb.detach() if stop == "target" else zb
    ) + scalar_regression_loss(pb, za.detach() if stop == "target" else za)
    torch.testing.assert_close(loss, expected)
    actual = torch.autograd.grad(loss, (pa, za, pb, zb), allow_unused=True)
    oracle = torch.autograd.grad(expected, (pa, za, pb, zb), allow_unused=True)
    for index, (got, want) in enumerate(zip(actual, oracle, strict=True)):
        if stop == "target" and index in (1, 3):
            assert got is want is None
        else:
            assert got is not None
            assert want is not None
            assert bool(got.ne(0).any())
            torch.testing.assert_close(got, want)
    assert bool(objective(stop=stop).detaches) == (stop == "target")


def test_the_floor_is_on_the_squared_norm_not_the_norm() -> None:
    """A cosine shortcut, and torch's own `normalize`, both miss this.

    `l2_normalize` divides by `sqrt(max(sum(v**2), 1e-12))`. Above the floor
    that is an ordinary unit vector and `2 - 2cos` agrees; below it the vectors
    come out shorter than unit length and the squared distance is smaller than
    any cosine expression can report. The card's §3.1 note — that the second
    equality of eq. (2) need not hold there — is exactly this.
    """
    rows = torch.arange(ROWS)
    for scale, agrees in ((1.0, True), (1e-7, False)):
        pa, _, _, zb = tensors(scale)
        state = State({ONLINE_A: {Port.X_PRED: pa}, TARGET_B: {Port.X_PROJ: zb}})
        value = objective().compute(state, fixture_batch(), rows, context()).value
        cosine = torch.nn.functional.cosine_similarity(pa, zb, dim=-1, eps=1e-12)
        torch_normalised = (
            (
                torch.nn.functional.normalize(pa, dim=-1, eps=1e-12)
                - torch.nn.functional.normalize(zb, dim=-1, eps=1e-12)
            )
            .pow(2)
            .sum(-1)
            .mean()
        )
        assert torch.allclose(value, 2 - 2 * cosine.mean()) is agrees
        assert torch.allclose(value, torch_normalised) is agrees
        torch.testing.assert_close(value, scalar_regression_loss(pa, zb))
    # And the direction of the disagreement: sub-unit vectors are closer.
    pa, _, _, zb = tensors(1e-7)
    state = State({ONLINE_A: {Port.X_PRED: pa}, TARGET_B: {Port.X_PROJ: zb}})
    below = objective().compute(state, fixture_batch(), rows, context()).value.detach()
    cosine = torch.nn.functional.cosine_similarity(pa.detach(), zb.detach(), dim=-1)
    assert float(below) < float(2 - 2 * cosine.mean())


def test_exactly_zero_vectors_stay_finite() -> None:
    """The case `l2_normalize`'s floor exists for.

    `v / ||v||` is 0/0 here and every downstream number becomes NaN; the source
    divides by `sqrt(max(0, 1e-12))` instead, so the normalised vector is zero,
    the distance is zero and the gradient is zero rather than undefined. A
    floor applied to the norm rather than the square would rescue the value and
    not the gradient, which is why both are asserted.
    """
    zeros = torch.zeros(ROWS, WIDTH, dtype=torch.float64, requires_grad=True)
    other = torch.zeros(ROWS, WIDTH, dtype=torch.float64, requires_grad=True)
    state = State({ONLINE_A: {Port.X_PRED: zeros}, TARGET_B: {Port.X_PROJ: other}})
    value = (
        objective(stop="none")
        .compute(state, fixture_batch(), torch.arange(ROWS), context())
        .value
    )
    torch.testing.assert_close(value, scalar_regression_loss(zeros, other))
    assert float(value.detach()) == 0.0
    gradients = torch.autograd.grad(value, (zeros, other))
    for gradient in gradients:
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(ROWS),
        torch.ones(ROWS, 0),
        torch.full((ROWS, WIDTH), float("nan")),
        torch.full((ROWS, WIDTH), float("inf")),
        torch.ones(ROWS + 1, WIDTH),
    ],
)
def test_bad_embeddings_are_rejected(bad: Tensor) -> None:
    state = State(
        {
            ONLINE_A: {Port.X_PRED: bad},
            TARGET_B: {Port.X_PROJ: torch.ones(ROWS, WIDTH)},
        }
    )
    with pytest.raises(LossError):
        objective().compute(state, fixture_batch(), torch.arange(ROWS), context())


def test_a_width_mismatch_is_not_broadcast() -> None:
    state = State(
        {
            ONLINE_A: {Port.X_PRED: torch.ones(ROWS, WIDTH)},
            TARGET_B: {Port.X_PROJ: torch.ones(ROWS, WIDTH + 1)},
        }
    )
    with pytest.raises(LossError, match="mis-declared head"):
        objective().compute(state, fixture_batch(), torch.arange(ROWS), context())


def test_the_declaration_itself_is_checked() -> None:
    with pytest.raises(LossError, match="with itself"):
        replace(objective(), target_port=Port.X_PRED, target=ONLINE_A)
    with pytest.raises(LossError, match="must be 'target' or 'none'"):
        replace(objective(), stop_grad="both")  # type: ignore[arg-type]
    with pytest.raises(Xty2Error, match="no usable default"):
        NormalizedSquaredFeatureConsistency(
            prediction_port=Port.X_PRED,
            target_port=Port.X_PROJ,
            prediction=ONLINE_A,
            target=TARGET_B,
        )
    with pytest.raises(LossError, match="squared distance between two embeddings"):
        replace(objective(), target_port=Port.T_GIVEN_X)


def test_an_empty_eligible_scope_is_a_zero_term_without_diagnostics() -> None:
    pa, _, _, zb = tensors(1.0)
    state = State({ONLINE_A: {Port.X_PRED: pa}, TARGET_B: {Port.X_PROJ: zb}})
    term = objective().compute(
        state, fixture_batch(), torch.zeros(0, dtype=torch.long), context()
    )
    assert float(term.value) == 0.0
    assert term.diagnostics == {}


# ---------------------------------------------------------------------------
# `networks.MLP`
# ---------------------------------------------------------------------------


def test_source_topology_and_bias_policy() -> None:
    run = recipe_run()
    projector, predictor = run.graph["byol_projector"], run.graph["byol_predictor"]
    assert isinstance(projector, BYOLProjector)
    assert isinstance(predictor, BYOLPredictor)
    for head, sizes in ((projector, (256, 4096, 256)), (predictor, (256, 4096, 256))):
        layers = head.network
        assert [type(module) for module in layers] == [
            nn.Linear,
            nn.BatchNorm1d,
            nn.ReLU,
            nn.Linear,
        ]
        hidden, norm, output = layers[0], layers[1], layers[3]
        assert isinstance(hidden, nn.Linear) and isinstance(output, nn.Linear)
        assert isinstance(norm, nn.BatchNorm1d)
        # `hk.Linear(with_bias=True)` then `hk.Linear(with_bias=False)`.
        assert hidden._parameters["bias"] is not None
        # `hk.Linear(with_bias=False)` registers no bias at all, so this is a
        # missing parameter rather than a zeroed one.
        assert output._parameters["bias"] is None
        assert (hidden.in_features, hidden.out_features) == sizes[:2]
        assert (output.in_features, output.out_features) == sizes[1:]
        # bn_config: decay_rate 0.9 is torch momentum 0.1, eps 1e-5, affine.
        assert norm.eps == 1e-5 and norm.momentum == 0.1
        assert norm.affine and norm.track_running_stats
        torch.testing.assert_close(norm.weight, torch.ones(4096))
        torch.testing.assert_close(norm.bias, torch.zeros(4096))
    assert (projector.INPUT, projector.OUTPUT) == (Port.X_REPR, Port.X_PROJ)
    assert (predictor.INPUT, predictor.OUTPUT) == (Port.X_PROJ, Port.X_PRED)


def test_singleton_training_batchnorm_is_rejected() -> None:
    run = recipe_run()
    schema = run.recipe.schema
    rows = two_cluster_population(8, seed=5, row_offset=0, low=SEPARATED).batch
    run.graph.train()
    single = rows.replace(
        x=rows.x[:1],
        t=rows.t[:1],
        y=rows.y[:1],
        t_observed=rows.t_observed[:1],
        y_observed=rows.y_observed[:1],
        row_id=rows.row_id[:1],
    )
    with pytest.raises(GraphError, match="at least two rows"):
        run.graph.evaluate(
            single, schema=schema, only=("mlp_encoder", "byol_projector")
        )


# ---------------------------------------------------------------------------
# `schedules.target_ema` and `schedules.learning_schedule`
# ---------------------------------------------------------------------------


def reference_cosine_decay(step: int, max_steps: int, initial: float) -> float:
    """`schedules._cosine_decay`, including its `jnp.minimum` clamp."""
    clamped = min(step, max_steps)
    return initial * 0.5 * (1.0 + math.cos(math.pi * clamped / max_steps))


def reference_target_ema(step: int, base: float, max_steps: int) -> float:
    return 1.0 - (1.0 - base) * reference_cosine_decay(step, max_steps, 1.0)


def reference_learning_schedule(step: int, total: int, warmup: int) -> float:
    """`schedules.learning_schedule` as a multiplier on the scaled rate."""
    if step < warmup:
        return step / warmup
    return reference_cosine_decay(step - warmup, total - warmup, 1.0)


def test_the_target_decay_is_the_reference_curve_over_the_applied_domain() -> None:
    schedule = CosineEMADecay(base=BASE_TARGET_EMA, steps=PRETRAIN_STEPS)
    for step in (0, WARMUP_STEPS, PRETRAIN_STEPS // 2, PRETRAIN_STEPS - 1):
        assert schedule(step) == pytest.approx(
            reference_target_ema(step, BASE_TARGET_EMA, PRETRAIN_STEPS), abs=1e-15
        )
    assert schedule(0) == BASE_TARGET_EMA
    for step in range(PRETRAIN_STEPS):
        assert BASE_TARGET_EMA <= schedule(step) < 1.0
        if step:
            assert schedule(step) > schedule(step - 1)
    # The endpoint the executor never reaches, and the one `nominal` reports
    # instead of it (card §4, the indexing contract).
    assert schedule(PRETRAIN_STEPS) == 1.0
    assert schedule.nominal == schedule(PRETRAIN_STEPS - 1)
    assert schedule.nominal < 1.0


def test_the_teacher_accepts_the_curve_and_its_horizon_is_the_stage_budget() -> None:
    """The two facts the schedule has to satisfy to be runnable at all.

    `TeacherSpec` checks step 0 and `nominal`, and `EMATeacher._decay_at`
    re-checks every step it reads, so a schedule reporting the k=1000 endpoint
    would be rejected at compile time for a value no update applies. The horizon
    check is the other half: a curve whose `steps` outlived the stage would run
    a prefix of itself, which no plan line would show.
    """
    pretrain = recipe_run().recipe.program[0]
    assert pretrain.teacher is not None
    decay = pretrain.teacher.decay
    assert isinstance(decay, CosineEMADecay)
    assert decay.steps == pretrain.steps == PRETRAIN_STEPS
    assert decay.base == BASE_TARGET_EMA
    teacher = EMATeacher(recipe_run().graph, pretrain.teacher)
    for step in (0, PRETRAIN_STEPS // 2, PRETRAIN_STEPS - 1):
        assert teacher._decay_at(step) == pytest.approx(
            reference_target_ema(step, BASE_TARGET_EMA, PRETRAIN_STEPS), abs=1e-15
        )
    with pytest.raises(Xty2Error, match=r"\[0, 1\)"):
        teacher._decay_at(PRETRAIN_STEPS)
    # The bound that makes the choice of `nominal` load-bearing: a decay that
    # settles at 1 is rejected before the stage runs at all.
    with pytest.raises(CompileError, match=r"\[0, 1\)"):
        replace(pretrain.teacher, decay=Constant(1.0))


@pytest.mark.parametrize("base", [-0.1, 1.0, 1.5, float("nan")])
def test_an_inadmissible_base_is_rejected(base: float) -> None:
    with pytest.raises(Xty2Error):
        CosineEMADecay(base=base, steps=10)


def test_the_learning_rate_is_the_reference_schedule_at_every_step() -> None:
    spec = recipe_run().recipe.program[0].optimiser
    schedule = spec.lr_schedule
    assert isinstance(schedule, WarmupCosine)
    assert (schedule.start, schedule.final) == (0.0, 0.0)
    assert (schedule.warmup, schedule.steps) == (WARMUP_STEPS, PRETRAIN_STEPS)
    assert spec.lr == pytest.approx(0.2 * BATCH_SIZE / 256)
    for step in range(PRETRAIN_STEPS):
        assert spec.lr_at(step) == pytest.approx(
            PRETRAIN_LR
            * reference_learning_schedule(step, PRETRAIN_STEPS, WARMUP_STEPS),
            abs=1e-15,
        )
    assert spec.lr_at(0) == 0.0
    assert spec.lr_at(WARMUP_STEPS) == pytest.approx(PRETRAIN_LR)
    assert spec.lr_at(PRETRAIN_STEPS) == 0.0


# ---------------------------------------------------------------------------
# `optimizers.lars`
# ---------------------------------------------------------------------------


def lars_oracle(
    parameters: Sequence[Tensor],
    gradients: Sequence[Tensor],
    buffers: Sequence[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    eta: float,
) -> tuple[list[Tensor], list[Tensor]]:
    """`optax.chain(add_weight_decay, scale_by_lars, scale(-lr))`, by hand.

    Written from the reference's four lines rather than from `LARS.step`: the
    decay is added first and the trust ratio is computed from the *decayed*
    update, the multiplier is exactly 1 when either norm is zero, momentum is a
    plain accumulation, and an excluded parameter still takes the momentum step.
    """
    moved, updated = [], []
    for parameter, gradient, buffer in zip(parameters, gradients, buffers, strict=True):
        excluded = parameter.ndim < 2
        update = gradient if excluded else gradient + weight_decay * parameter
        if not excluded:
            parameter_norm = parameter.pow(2).sum().sqrt()
            update_norm = update.pow(2).sum().sqrt()
            if float(parameter_norm) > 0.0 and float(update_norm) > 0.0:
                update = update * (eta * parameter_norm / update_norm)
        accumulated = momentum * buffer + update
        moved.append(parameter - lr * accumulated)
        updated.append(accumulated)
    return moved, updated


def lars_spec(weight_decay: float = 1.5e-6, **overrides: Any) -> OptimiserSpec:
    fields: dict[str, Any] = {
        "name": "lars",
        "lr": 0.1,
        "momentum": 0.9,
        "eta": 1e-3,
        "weight_decay": WeightDecay(
            value=weight_decay, on_norm_and_bias=False, components=None
        )
        if weight_decay
        else WeightDecay.none(),
        "lr_schedule": 1.0,
        "clipping": GradientClipping.none(),
    }
    return OptimiserSpec(**{**fields, **overrides})


def test_two_lars_steps_match_the_hand_computed_update() -> None:
    """Two steps, because one cannot distinguish the momentum accumulation.

    The module carries a weight matrix, a bias and a BatchNorm scale and shift,
    so the same call exercises the adapted and the excluded path together —
    `exclude_bias_and_norm` in torch's vocabulary is rank below two.
    """
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(4, 3), nn.BatchNorm1d(3)).double()
    named = list(module.named_parameters())
    spec = lars_spec()
    optimiser = spec.build(named)
    parameters = [parameter for _, parameter in named]
    expected = [parameter.detach().clone() for parameter in parameters]
    buffers = [torch.zeros_like(parameter) for parameter in parameters]
    for step in (1, 2):
        gradients = [
            torch.full_like(parameter, 0.1 * step)
            + torch.arange(parameter.numel(), dtype=parameter.dtype).reshape(
                parameter.shape
            )
            for parameter in parameters
        ]
        for parameter, gradient in zip(parameters, gradients, strict=True):
            parameter.grad = gradient.clone()
        optimiser.step()
        expected, buffers = lars_oracle(
            expected,
            gradients,
            buffers,
            lr=spec.lr,
            weight_decay=1.5e-6,
            momentum=0.9,
            eta=1e-3,
        )
        for parameter, wanted in zip(parameters, expected, strict=True):
            torch.testing.assert_close(parameter.detach(), wanted)
    assert [adapts_under_lars(parameter) for parameter in parameters] == [
        True,
        False,
        False,
        False,
    ]


def test_a_zero_norm_leaves_the_update_unscaled_rather_than_undefined() -> None:
    """`jnp.where(param_norm > 0, jnp.where(update_norm > 0, ratio, 1.), 1.)`.

    Both guards matter and they fail differently: dividing by a zero update
    norm gives a non-finite parameter, while scaling by a zero parameter norm
    silently freezes a parameter that the reference still moves.
    """
    zero_parameter = torch.zeros(3, 2, requires_grad=True)
    zero_gradient = torch.ones(3, 2, requires_grad=True)
    optimiser = LARS(
        [{"params": [zero_parameter, zero_gradient], "weight_decay": 0.0}],
        lr=0.5,
        momentum=0.0,
        eta=1e-3,
    )
    zero_parameter.grad = torch.ones(3, 2)
    zero_gradient.grad = torch.zeros(3, 2)
    optimiser.step()
    assert bool(torch.isfinite(zero_parameter).all())
    torch.testing.assert_close(zero_parameter.detach(), torch.full((3, 2), -0.5))
    torch.testing.assert_close(zero_gradient.detach(), torch.ones(3, 2))


def test_the_optimiser_identity_carries_its_own_knobs() -> None:
    assert lars_spec().description == (
        "lars(momentum=0.9, eta=0.001, adaptation on parameters of rank two or more)"
    )
    with pytest.raises(CompileError, match="no eta"):
        lars_spec(eta=None)
    with pytest.raises(CompileError, match="eta must be positive"):
        lars_spec(eta=0.0)
    with pytest.raises(CompileError, match="no Nesterov"):
        lars_spec(nesterov=True)
    with pytest.raises(CompileError, match="takes no eta"):
        lars_spec(name="sgd")
    with pytest.raises(CompileError, match="takes no betas"):
        lars_spec(betas=(0.5, 0.5))
    assert isinstance(lars_spec().build(list(nn.Linear(2, 2).named_parameters())), LARS)


def test_the_recipe_exempts_bias_and_norm_from_decay_and_adaptation() -> None:
    spec = recipe_run().recipe.program[0].optimiser
    assert spec.name == "lars"
    assert spec.weight_decay.value == 1.5e-6
    assert spec.weight_decay.on_norm_and_bias is False
    assert spec.weight_decay.components is None
    run = recipe_run()
    with trainable_only(run.graph, run.stages[0].trainable) as named:
        for name, parameter in named:
            assert spec.weight_decay.decays(name, parameter) is adapts_under_lars(
                parameter
            ), name


# ---------------------------------------------------------------------------
# `byol_experiment._update_fn` and `_make_initial_state`
# ---------------------------------------------------------------------------


def run_pretrain(
    run: CompiledRun,
    *,
    steps: int = 1,
    start: int = 0,
    seed: int = 310006,
    rows: int = 64,
) -> tuple[Any, EMATeacher, Any]:
    """The executor's own order per step: forward, backward, step, EMA.

    `_step` rather than a re-implementation of it, because the order is the
    thing under test: predict with the previous target, update the online
    parameters, then move the target once. `before` is the target as it stood
    when the last step began, which is what eq. (1)'s oracle needs.
    """
    torch.manual_seed(seed)
    schema = run.recipe.schema
    population = build_population(
        training_dataset(
            schema,
            two_cluster_population(
                rows, seed=seed + 1, row_offset=0, low=SEPARATED
            ).batch,
        ),
        DATA_POLICY,
        seed=seed + 2,
    )
    compiled = run.stages[0]
    assert compiled.teacher is not None
    run.graph.train()
    with trainable_only(run.graph, compiled.trainable) as named:
        teacher = EMATeacher(run.graph, compiled.teacher)
        optimiser = compiled.optimiser.build(named)
        mixer = LossMixer.for_stage(compiled)
        before: dict[str, Tensor] = {}
        stepped = None
        for offset in range(steps):
            before = {
                name: parameter.detach().clone()
                for name, parameter in teacher.graph.named_parameters()
            }
            stepped = executors._step(
                run,
                compiled,
                population.rows,
                mixer,
                optimiser,
                [parameter for _, parameter in named],
                start + offset,
                rng_key=seed + 3 + offset,
                teacher=teacher,
                population=population,
            )
    return stepped, teacher, (before, population)


def test_the_target_starts_as_an_exact_copy_and_moves_once_per_step() -> None:
    """Appendix A's initialisation (deviation 5) and eq. (1) after eq. (3)."""
    run = recipe_run()
    compiled = run.stages[0]
    assert compiled.teacher is not None
    teacher = EMATeacher(run.graph, compiled.teacher)
    student = dict(run.graph.named_parameters())
    for name, parameter in teacher.graph.named_parameters():
        torch.testing.assert_close(parameter, student[name])
        assert parameter is not student[name]
        assert parameter.requires_grad is False

    run = recipe_run()
    stepped, teacher, (before, _) = run_pretrain(run, start=7)
    assert math.isfinite(float(stepped.total.detach()))
    decay = reference_target_ema(7, BASE_TARGET_EMA, PRETRAIN_STEPS)
    online = dict(run.graph.named_parameters())
    moved = 0
    for name, parameter in teacher.graph.named_parameters():
        # `x + (1 - tau) * (y - x)` with x the target and y the *updated*
        # online parameters — the EMA reads the post-optimiser values.
        wanted = before[name] + (1.0 - decay) * (online[name].detach() - before[name])
        torch.testing.assert_close(parameter, wanted)
        assert parameter.grad is None
        moved += int(bool(parameter.ne(before[name]).any()))
    assert moved > 0


def test_both_directions_read_one_target_parameter_set() -> None:
    """The target must not move between the two directional losses.

    `loss_fn` applies the target network once over both views and `_update_fn`
    moves it afterwards, so the two directions see identical target parameters.
    The check is that the state's two teacher realisations were produced by the
    same parameters: re-running them against the teacher graph reproduces the
    tensors the loss consumed.
    """
    run = recipe_run()
    compiled = run.stages[0]
    assert compiled.teacher is not None
    schema = run.recipe.schema
    population = build_population(
        training_dataset(
            schema,
            two_cluster_population(64, seed=11, row_offset=0, low=SEPARATED).batch,
        ),
        DATA_POLICY,
        seed=12,
    )
    teacher = EMATeacher(run.graph, compiled.teacher)
    run.graph.train()
    state = run.state(
        compiled,
        population.rows,
        rng_key=8,
        teacher_graph=teacher.graph,
        population=population,
    )
    for realisation in (TARGET_A, TARGET_B):
        value = state[realisation][Port.X_PROJ]
        assert isinstance(value, Tensor)
        assert value.requires_grad is False
        assert value.grad_fn is None
    # The teacher's parameters are unchanged by having been read.
    for name, parameter in teacher.graph.named_parameters():
        torch.testing.assert_close(
            parameter, dict(run.graph.named_parameters())[name].detach()
        )


def test_the_target_owns_its_batch_norm_statistics() -> None:
    """`ema_applies_to_buffers=false` plus `train_mode=true`.

    `_update_fn` EMAs parameters only and returns the target's own forwarded
    state, so every target statistic comes from a target forward. Two facts
    follow, and the copied predictor is what makes them separable: the target's
    projector BN advances twice per step under its own parameters and diverges
    from the online one as soon as the two parameter sets do, while the target's
    predictor BN — a module `EMATeacher`'s graph copy carries and no target pass
    runs (card §3.2) — stays exactly at initialisation. Turning buffer EMA on
    moves that untouched module, which is the mutant below.
    """
    run = recipe_run()
    compiled = run.stages[0]
    assert compiled.teacher is not None
    assert compiled.teacher.applies_to_buffers is False
    assert compiled.teacher.train_mode is True
    _, teacher, _ = run_pretrain(run, steps=2)
    student = dict(run.graph.named_buffers())
    target = dict(teacher.graph.named_buffers())

    def buffer(name: str, side: dict[str, Tensor]) -> Tensor:
        return side[f"_components.{name}"]

    for head in ("byol_projector", "byol_predictor"):
        stat = f"{head}.network.1.num_batches_tracked"
        # Two views per step, in a-then-b order, on the online side.
        assert int(buffer(stat, student)) == 4, head
    assert int(buffer("byol_projector.network.1.num_batches_tracked", target)) == 4
    assert int(buffer("byol_predictor.network.1.num_batches_tracked", target)) == 0
    for statistic, initial in (("running_mean", 0.0), ("running_var", 1.0)):
        name = f"byol_predictor.network.1.{statistic}"
        untouched = torch.full_like(buffer(name, target), initial)
        torch.testing.assert_close(buffer(name, target), untouched)
        assert bool(buffer(name, student).ne(untouched).any())
        projector = f"byol_projector.network.1.{statistic}"
        assert bool(buffer(projector, target).ne(buffer(projector, student)).any())


def test_the_target_supplies_projections_and_never_predictions() -> None:
    """The teacher passes run the encoder and projector and stop there."""
    compiled = recipe_run().stages[0]
    passes = {forward.realisation: forward.components for forward in compiled.passes}
    assert (
        passes[ONLINE_A]
        == passes[ONLINE_B]
        == (
            "mlp_encoder",
            "byol_projector",
            "byol_predictor",
        )
    )
    assert passes[TARGET_A] == passes[TARGET_B] == ("mlp_encoder", "byol_projector")
    assert len(passes) == 4
    for term in compiled.objectives:
        objective_ = term.objective
        assert isinstance(objective_, NormalizedSquaredFeatureConsistency)
        assert objective_.target_port is Port.X_PROJ
        assert objective_.target.params == "teacher"
        assert objective_.prediction_port is Port.X_PRED
        assert objective_.prediction.params == "student"
        assert objective_.target.view != objective_.prediction.view


def test_only_the_online_parameters_take_gradients() -> None:
    run = recipe_run()
    _, teacher, _ = run_pretrain(run)
    for component in ("mlp_encoder", "byol_projector", "byol_predictor"):
        assert any(
            parameter.grad is not None and bool(parameter.grad.ne(0).any())
            for parameter in run.graph[component].parameters()
        ), component
    for parameter in teacher.graph.parameters():
        assert parameter.requires_grad is False
        assert parameter.grad is None
    for component in ("tarnet_head", "categorical_propensity"):
        assert all(
            parameter.grad is None for parameter in run.graph[component].parameters()
        ), component


# ---------------------------------------------------------------------------
# Program shape: what pretraining may read, and what transfers
# ---------------------------------------------------------------------------


def test_pretraining_reads_no_outcome_and_transfers_only_the_encoder() -> None:
    run = recipe_run()
    pretrain, joint_fit = run.recipe.program
    assert not run.graph.port_depends_on_raw_outcome(Port.X_PRED)
    assert not run.graph.port_depends_on_raw_outcome(Port.X_PROJ)
    assert pretrain.trainable == ("mlp_encoder", "byol_projector", "byol_predictor")
    for term in pretrain.objectives:
        for port, _ in term.objective.requires:
            assert port in (Port.X_PRED, Port.X_PROJ)
    assert joint_fit.initialise_from == "pretrain"
    assert joint_fit.teacher is None
    # Deviation 4, as amended: the encoder transfers and is *not* declared
    # trainable. Its absence is asserted separately from the tuple compare,
    # which a rewrite alongside the recipe would satisfy without noticing.
    assert joint_fit.trainable == ("tarnet_head", "categorical_propensity")
    assert "mlp_encoder" not in joint_fit.trainable
    assert "mlp_encoder" in {c for f in run.stages[-1].passes for c in f.components}
    # A fresh optimiser, not the pretraining one: no LARS state travels.
    assert joint_fit.optimiser.name == "adam"
    assert joint_fit.optimiser is not pretrain.optimiser
    downstream = {
        component
        for forward in run.stages[1].passes
        for component in forward.components
    }
    assert "byol_projector" not in downstream
    assert "byol_predictor" not in downstream


def test_the_target_base_is_the_preset_row_the_local_horizon_reaches() -> None:
    """Card §5 row 7, written from `configs/byol.py` rather than from the recipe.

    Two assertions, not one. The literal is the oracle: it shares no call path
    with `recipes/byol.py` and fails the moment the base drifts. The expression
    beside it restates the derivation so the literal is checkable by eye rather
    than being a number to keep in step by hand.
    """
    # `_EMA_PRESETS`, and `max_steps = num_epochs * train_images_per_epoch //
    # batch_size` with the pinned ImageNet count and the 4096 preset batch.
    presets = {40: 0.97, 100: 0.99, 300: 0.99, 1000: 0.996}
    steps = {epochs: epochs * 1281167 // 4096 for epochs in presets}
    assert steps == {40: 12511, 100: 31278, 300: 93835, 1000: 312784}

    assert pytest.approx(0.68722) == BASE_TARGET_EMA

    # 1000 steps of batch 128 over the §6 fixture's 1024 training rows.
    assert PRETRAIN_STEPS * BATCH_SIZE / 1024 == 125.0
    reached = max(epochs for epochs in presets if epochs <= 125)
    assert reached == 100
    translated = {
        epochs: 1.0 - (1.0 - base) * steps[epochs] / PRETRAIN_STEPS
        for epochs, base in presets.items()
    }
    assert pytest.approx(translated[reached]) == BASE_TARGET_EMA

    # The row §5 row 3 originally inherited is unreachable at this horizon: no
    # base in [0, 1) reproduces it, which is why row 7 reselects rather than
    # rescales. `TeacherSpec` and `EMATeacher` both reject a decay outside that
    # interval, so this is a fact about the curve and not about our taste.
    assert translated[1000] < 0.0
    with pytest.raises(Xty2Error):
        CosineEMADecay(base=translated[1000], steps=PRETRAIN_STEPS)


def test_the_inherited_base_is_observably_a_different_regime() -> None:
    """§5 row 7's measured claim: integrated tracking, `sum(1 - tau)`.

    The mutant this kills is the card's original choice — inherit 0.996 — and
    the number it fails on is the one row 7 quotes.
    """

    def tracking(base: float) -> float:
        curve = CosineEMADecay(base=base, steps=PRETRAIN_STEPS)
        return math.fsum(1.0 - curve.value(k) for k in range(PRETRAIN_STEPS))

    # The 100-epoch row this card now takes, at its own horizon.
    assert tracking(BASE_TARGET_EMA) == pytest.approx(156.5, abs=0.1)
    assert math.fsum(
        (1.0 - 0.99) * 0.5 * (1.0 + math.cos(math.pi * k / 31278)) for k in range(31278)
    ) == pytest.approx(156.4, abs=0.1)
    # The inherited row at this horizon is two, not one hundred and fifty-six.
    assert tracking(0.996) == pytest.approx(2.0, abs=0.01)


# ---------------------------------------------------------------------------
# The card boundary
# ---------------------------------------------------------------------------


def test_every_answered_card_value_matches_the_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Card §4 against `plan.hyperparameters`, key by key and value by value.

    One key is compared numerically rather than as text: `teacher.ema_decay` is
    `1 - (1 - 0.996) * 0.5 * (1 + cos(pi * 999/1000))`, whose last bit depends on
    the platform's `cos`. The card states the number to full precision and this
    accepts it to within floating-point tolerance, because the final ulp of a
    libm call is not a fact about the method.
    """
    monkeypatch.setattr(card_parser, "CARD", CARD)
    answers = card_parser._card_section_four()
    plan = recipe_run().plan.hyperparameters
    checked = 0
    for key, answer in answers.items():
        assert key in plan, key
        actual = plan[key]
        scopes = answer if isinstance(answer, dict) else {"": answer}
        for scope, stated in scopes.items():
            if key == "architecture.widths_depths":
                stated = stated.replace("X_REPR", "256").replace("K", "2")
            stated = stated.strip('"')
            values = (
                ([actual[scope]] if scope else list(actual.values()))
                if isinstance(actual, dict)
                else [actual]
            )
            for value in values:
                rendered = card_parser._rendered(value)
                if isinstance(value, float):
                    assert float(stated) == pytest.approx(value), (key, scope)
                else:
                    assert rendered == stated, (key, scope, value, stated)
                checked += 1
    assert checked >= 70


def test_the_card_and_the_recipe_index_agree() -> None:
    from xty2.evaluation.reporting import card_status

    card = CARD.read_text(encoding="utf-8")
    assert card_status(card) in ("reproduced", "deviating")
    index = (CARD.parents[1] / "RECIPES.md").read_text(encoding="utf-8")
    row = next(line for line in index.splitlines() if "byol.md" in line)
    assert "`byol`" in row
    # Tier 2 now exists; its independent rescore is in test_byol_benchmark.py.
    ledger = card.split("### 6.1 Result ledger", 1)[1].split("###", 1)[0]
    assert "| | | | | |" not in ledger
    assert "ema_outcome_nll_gain" in ledger


# ---------------------------------------------------------------------------
# The mutants (card §6.3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutant",
    [
        "halve_the_loss",
        "detach_the_prediction",
        "use_target_predictions",
        "remove_target_detach",
        "float_the_norm_instead_of_the_square",
    ],
)
def test_loss_oracles_kill_mutants(
    mutant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = NormalizedSquaredFeatureConsistency.compute

    def altered(
        self: NormalizedSquaredFeatureConsistency, state: State, *args: Any
    ) -> Any:
        if mutant == "halve_the_loss":
            term = original(self, state, *args)
            return replace(term, value=term.value * 0.5)
        if mutant == "remove_target_detach":
            return original(replace(self, stop_grad="none"), state, *args)
        if mutant == "use_target_predictions":
            return original(replace(self, target_port=Port.X_PRED), state, *args)
        if mutant == "float_the_norm_instead_of_the_square":
            return original(replace(self, epsilon=1e-24), state, *args)
        values = {
            realisation: {
                port: value.detach()
                if port == Port.X_PRED and isinstance(value, Tensor)
                else value
                for port, value in state[realisation].items()
            }
            for realisation in state.realisations
        }
        return original(self, State(values), *args)

    monkeypatch.setattr(NormalizedSquaredFeatureConsistency, "compute", altered)
    with pytest.raises((AssertionError, RuntimeError, KeyError, Xty2Error)):
        if mutant == "float_the_norm_instead_of_the_square":
            test_the_floor_is_on_the_squared_norm_not_the_norm()
        else:
            test_directional_values_and_gradients(1.0, "target")


@pytest.mark.parametrize(
    "mutant", ["ema_the_buffers", "move_the_target_twice", "shift_the_schedule"]
)
def test_teacher_oracles_kill_mutants(
    mutant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recipe_run()
    pretrain, fit = run.recipe.program
    teacher = pretrain.teacher
    assert teacher is not None
    if mutant == "ema_the_buffers":
        teacher = replace(teacher, applies_to_buffers=True)
    if mutant == "shift_the_schedule":
        teacher = replace(
            teacher,
            decay=CosineEMADecay(base=BASE_TARGET_EMA, steps=PRETRAIN_STEPS + 1),
        )
    altered = compile(
        replace(
            run.recipe,
            program=Program((replace(pretrain, teacher=teacher), fit)),
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "recipe_run", lambda: altered)
    if mutant == "move_the_target_twice":
        original_update = EMATeacher.update
        monkeypatch.setattr(
            EMATeacher,
            "update",
            lambda self, student, step: (
                original_update(self, student, step),
                original_update(self, student, step),
            )[0],
        )
    with pytest.raises(AssertionError):
        if mutant == "ema_the_buffers":
            test_the_target_owns_its_batch_norm_statistics()
        elif mutant == "shift_the_schedule":
            test_the_teacher_accepts_the_curve_and_its_horizon_is_the_stage_budget()
        else:
            test_the_target_starts_as_an_exact_copy_and_moves_once_per_step()


@pytest.mark.parametrize("mutant", ["decay_after_adaptation", "adapt_everything"])
def test_lars_oracle_kills_mutants(
    mutant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    @torch.no_grad()
    def altered(self: LARS, closure: Any = None) -> Any:
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                adapt = True if mutant == "adapt_everything" else parameter.ndim >= 2
                if mutant == "decay_after_adaptation":
                    if adapt:
                        update = _adapt(parameter, update, group["eta"])
                    if group["weight_decay"]:
                        update = update.add(parameter, alpha=group["weight_decay"])
                else:
                    if group["weight_decay"]:
                        update = update.add(parameter, alpha=group["weight_decay"])
                    if adapt:
                        update = _adapt(parameter, update, group["eta"])
                state = self.state[parameter]
                buffer = state.setdefault(
                    "momentum_buffer", torch.zeros_like(parameter)
                )
                buffer.mul_(group["momentum"]).add_(update)
                parameter.add_(buffer, alpha=-group["lr"])
        return None

    monkeypatch.setattr(LARS, "step", altered)
    with pytest.raises(AssertionError):
        test_two_lars_steps_match_the_hand_computed_update()


def _adapt(parameter: Tensor, update: Tensor, eta: float) -> Tensor:
    parameter_norm = torch.linalg.vector_norm(parameter)
    update_norm = torch.linalg.vector_norm(update)
    if float(parameter_norm) > 0.0 and float(update_norm) > 0.0:
        return update.mul(parameter_norm.mul(eta).div(update_norm))
    return update


def test_the_schedule_contract_holds_for_the_new_curve() -> None:
    schedule: Schedule = CosineEMADecay(base=0.99, steps=40)
    forwards = [schedule(step) for step in range(60)]
    assert [schedule(step) for step in reversed(range(60))] == list(reversed(forwards))
    assert str(schedule) == schedule.describe()
    with pytest.raises(Xty2Error, match="non-negative"):
        schedule(-1)
    with pytest.raises(Xty2Error, match="at least 1"):
        CosineEMADecay(base=0.99, steps=0)
