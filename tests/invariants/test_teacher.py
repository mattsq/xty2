"""Tier 0 — EMA teacher isolation, updates, buffers and plan surface (P8)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import ClassVar, cast

import pytest
import torch
from xty2.core import (
    DEFAULT,
    CardKeyError,
    CompiledRun,
    CompileError,
    Component,
    ComponentGraph,
    Port,
    PortValue,
    PortView,
    Program,
    Realisation,
    Recipe,
    TeacherSpec,
    TrainingError,
    Weighted,
    compile,
    treatment_distribution,
)
from xty2.objectives import ConsistencyLoss, ObservedTreatmentNLL, StopGrad
from xty2.training import Checkpoint, StageResult, run_program, run_stage

from tests.invariants.conftest import (
    HIDDEN,
    NUM_FEATURES,
    ToyPropensity,
    make_batch,
    make_schema,
    stage,
)

TEACHER = Realisation(params="teacher")


class BufferedEncoder(Component):
    """A trainable encoder with one float and one integral state buffer."""

    CARD_KEYS: ClassVar[Mapping[str, str]] = {}
    running: torch.Tensor
    batches: torch.Tensor

    def __init__(self, name: str = "encoder") -> None:
        super().__init__(name, requires={Port.X_RAW}, provides={Port.X_REPR})
        self.net = torch.nn.Linear(NUM_FEATURES, HIDDEN)
        self.register_buffer("running", torch.zeros(()))
        self.register_buffer("batches", torch.zeros((), dtype=torch.long))

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        if self.training:
            with torch.no_grad():
                self.running[...] = self.running + 1.0
                self.batches[...] = self.batches + 1
        return {Port.X_REPR: self.net(ports.tensor(Port.X_RAW))}


def _teacher_spec(
    *, applies_to_buffers: bool = True, train_mode: bool = False, decay: float = 0.5
) -> TeacherSpec:
    return TeacherSpec(
        decay=decay,
        applies_to_buffers=applies_to_buffers,
        train_mode=train_mode,
        requires_grad=False,
    )


def _recipe(
    *,
    teacher: TeacherSpec | None = None,
    stop_grad: StopGrad = "right",
    trainable: tuple[str, ...] = ("encoder", "propensity"),
) -> Recipe:
    objectives = [Weighted(ObservedTreatmentNLL(), weight=1.0, reduction="mean")]
    if teacher is not None:
        objectives.append(
            Weighted(
                ConsistencyLoss(
                    port=Port.T_GIVEN_X,
                    left=DEFAULT,
                    right=TEACHER,
                    divergence="mse",
                    stop_grad=stop_grad,
                ),
                weight=1.0,
                reduction="mean",
            )
        )
    return Recipe(
        name="teacher_test",
        schema=make_schema(),
        system=ComponentGraph([BufferedEncoder(), ToyPropensity()]),
        program=(
            stage(
                objectives=tuple(objectives),
                trainable=trainable,
                teacher=teacher,
                steps=1,
            ),
        ),
        card="docs/recipes/teacher_test.md",
    )


def _run(
    spec: TeacherSpec,
) -> tuple[CompiledRun, StageResult, dict[str, torch.Tensor]]:
    torch.manual_seed(31)
    run = compile(_recipe(teacher=spec))
    initial = {
        name: value.detach().clone() for name, value in run.graph.named_parameters()
    }
    result = run_stage(run, "fit", [make_batch()], seed=5)
    return run, result, initial


def test_every_teacher_choice_is_required_and_card_bound() -> None:
    with pytest.raises(CardKeyError, match="no usable default"):
        TeacherSpec()
    spec = _teacher_spec()
    plan = compile(_recipe(teacher=spec)).plan
    assert plan.hyperparameters["teacher.ema_decay"] == 0.5
    assert plan.hyperparameters["teacher.ema_applies_to_buffers"] is True
    assert plan.hyperparameters["teacher.teacher_in_train_mode"] is False
    assert plan.hyperparameters["teacher.teacher_requires_grad"] is False
    assert (
        "teacher: ema(decay=0.5, buffers=ema, mode=eval, "
        "requires_grad=False)" in plan.render()
    )


def test_multistage_teacher_choices_are_keyed_by_stage() -> None:
    recipe = _recipe(teacher=_teacher_spec(decay=0.5))
    first = recipe.program[0]
    second = replace(first, name="refine", teacher=_teacher_spec(decay=0.9))
    program = Program((first, second))

    hyperparameters = compile(replace(recipe, program=program)).plan.hyperparameters
    assert hyperparameters["teacher.ema_decay"] == {"fit": 0.5, "refine": 0.9}
    assert hyperparameters["teacher.teacher_requires_grad"] == {
        "fit": False,
        "refine": False,
    }


@pytest.mark.parametrize("decay", [-0.1, 1.0, float("nan"), True])
def test_teacher_decay_is_a_finite_fraction(decay: object) -> None:
    with pytest.raises(CompileError, match=r"\[0, 1\)"):
        TeacherSpec(
            decay=decay,  # type: ignore[arg-type]
            applies_to_buffers=True,
            train_mode=False,
            requires_grad=False,
        )


def test_a_teacher_can_never_require_gradients() -> None:
    with pytest.raises(CompileError, match="must be false"):
        TeacherSpec(
            decay=0.9,
            applies_to_buffers=True,
            train_mode=False,
            requires_grad=True,  # type: ignore[arg-type]
        )


def test_a_teacher_realisation_needs_a_stage_teacher() -> None:
    recipe = _recipe()
    consistency = Weighted(
        ConsistencyLoss(
            port=Port.T_GIVEN_X,
            left=DEFAULT,
            right=TEACHER,
            divergence="mse",
            stop_grad="right",
        ),
        weight=1.0,
        reduction="mean",
    )
    configured = recipe.program[0]
    with pytest.raises(CompileError, match="teacher parameter set"):
        compile(
            Recipe(
                name=recipe.name,
                schema=recipe.schema,
                system=recipe.system,
                program=(
                    stage(
                        objectives=(*configured.objectives, consistency),
                        trainable=configured.trainable,
                        steps=1,
                    ),
                ),
                card=recipe.card,
            )
        )


def test_a_configured_but_unused_teacher_is_rejected() -> None:
    spec = _teacher_spec()
    recipe = _recipe()
    plain = recipe.program[0]
    with pytest.raises(CompileError, match="no active objective"):
        compile(
            Recipe(
                name=recipe.name,
                schema=recipe.schema,
                system=recipe.system,
                program=(
                    stage(
                        objectives=plain.objectives,
                        trainable=plain.trainable,
                        teacher=spec,
                        steps=1,
                    ),
                ),
                card=recipe.card,
            )
        )


def test_teacher_targets_must_declare_their_stop_gradient() -> None:
    with pytest.raises(CompileError, match="without declaring them"):
        compile(_recipe(teacher=_teacher_spec(), stop_grad="none"))


def test_teacher_parameters_are_isolated_from_autograd() -> None:
    run, result, _ = _run(_teacher_spec())
    teacher = result.teacher
    assert teacher is not None
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())

    state = run.state("fit", make_batch(), teacher_graph=teacher.graph)
    distribution = treatment_distribution(
        state, Port.T_GIVEN_X, TEACHER, objective="teacher_isolation_test"
    )
    assert not distribution.probs.requires_grad


def test_missing_teacher_graph_is_rejected_before_the_student_forward() -> None:
    run = compile(_recipe(teacher=_teacher_spec()))
    encoder = cast(BufferedEncoder, run.graph["encoder"])
    running = encoder.running.clone()
    batches = encoder.batches.clone()

    with pytest.raises(TrainingError, match="no teacher parameter graph"):
        run.state("fit", make_batch())

    assert torch.equal(encoder.running, running)
    assert torch.equal(encoder.batches, batches)


def test_invalid_teacher_graphs_are_runtime_training_errors() -> None:
    teacher_run = compile(_recipe(teacher=_teacher_spec()))
    batch = make_batch()
    with pytest.raises(TrainingError, match="student graph"):
        teacher_run.state("fit", batch, teacher_graph=teacher_run.graph)

    plain_run = compile(_recipe())
    with pytest.raises(TrainingError, match="declares no teacher"):
        plain_run.state("fit", batch, teacher_graph=teacher_run.graph)


def test_teacher_parameters_follow_the_post_step_student_by_ema() -> None:
    run, result, initial = _run(_teacher_spec(decay=0.5))
    teacher = result.teacher
    assert teacher is not None
    student_parameters = dict(run.graph.named_parameters())
    teacher_parameters = dict(teacher.graph.named_parameters())
    assert any(
        not torch.equal(initial[name], student_parameters[name]) for name in initial
    )
    for name, parameter in teacher_parameters.items():
        expected = initial[name].mul(0.5).add(student_parameters[name], alpha=0.5)
        assert torch.allclose(parameter, expected)


def test_float_buffers_use_ema_and_integral_buffers_copy() -> None:
    _, result, _ = _run(_teacher_spec(applies_to_buffers=True, train_mode=False))
    teacher = result.teacher
    assert teacher is not None
    encoder = cast(BufferedEncoder, teacher.graph["encoder"])
    assert torch.equal(encoder.running, torch.tensor(0.5))
    assert torch.equal(encoder.batches, torch.tensor(1))


def test_buffers_remain_teacher_owned_when_buffer_ema_is_off() -> None:
    _, result, _ = _run(_teacher_spec(applies_to_buffers=False, train_mode=False))
    teacher = result.teacher
    assert teacher is not None
    encoder = cast(BufferedEncoder, teacher.graph["encoder"])
    assert torch.equal(encoder.running, torch.tensor(0.0))
    assert torch.equal(encoder.batches, torch.tensor(0))


def test_teacher_train_mode_updates_its_own_buffers() -> None:
    _, result, _ = _run(_teacher_spec(applies_to_buffers=False, train_mode=True))
    teacher = result.teacher
    assert teacher is not None
    assert teacher.graph.training
    encoder = cast(BufferedEncoder, teacher.graph["encoder"])
    assert torch.equal(encoder.running, torch.tensor(1.0))
    assert torch.equal(encoder.batches, torch.tensor(1))


def test_freezing_a_component_also_freezes_its_buffers() -> None:
    run = compile(_recipe(trainable=("propensity",)))
    encoder = cast(BufferedEncoder, run.graph["encoder"])
    before = encoder.running.clone()
    result = run_stage(run, "fit", [make_batch()], seed=0)
    assert torch.equal(encoder.running, before)
    assert not result.checkpoint.buffers


def test_stage_checkpoint_carries_immutable_trained_component_buffers() -> None:
    run = compile(_recipe())
    result = run_stage(run, "fit", [make_batch()], seed=0)
    checkpoint = result.checkpoint
    assert checkpoint.buffer("encoder.running").item() == 1.0
    borrowed = checkpoint.buffers["encoder.running"]
    borrowed.zero_()
    assert checkpoint.buffer("encoder.running").item() == 1.0


def test_stage_initialisation_restores_checkpoint_buffers() -> None:
    objective = (Weighted(ObservedTreatmentNLL(), weight=1.0, reduction="mean"),)
    recipe = Recipe(
        name="buffer_transition",
        schema=make_schema(),
        system=ComponentGraph([BufferedEncoder(), ToyPropensity()]),
        program=Program(
            (
                stage(
                    name="first",
                    objectives=objective,
                    trainable=("encoder", "propensity"),
                    steps=1,
                ),
                stage(
                    name="second",
                    objectives=objective,
                    trainable=("encoder", "propensity"),
                    initialise_from="first",
                    steps=1,
                ),
            )
        ),
        card="docs/recipes/buffer_transition.md",
    )
    result = run_program(
        compile(recipe), {"first": [make_batch()], "second": [make_batch()]}, seed=0
    )
    assert result.stage("first").checkpoint.buffer("encoder.running").item() == 1.0
    assert result.stage("second").checkpoint.buffer("encoder.running").item() == 2.0


def test_checkpoint_buffers_round_trip_immutably(tmp_path: Path) -> None:
    checkpoint = run_stage(compile(_recipe()), "fit", [make_batch()], seed=0).checkpoint
    reloaded = Checkpoint.load(checkpoint.save(tmp_path / "checkpoint.pt"))
    assert reloaded.buffer("encoder.running").item() == 1.0
    borrowed = reloaded.buffers["encoder.running"]
    borrowed.fill_(99.0)
    assert reloaded.buffer("encoder.running").item() == 1.0
