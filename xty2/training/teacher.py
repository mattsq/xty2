"""The stage-local EMA parameter set (`DESIGN.md` §2.1, `PLAN.md` P8).

The teacher is a complete, distinct copy of the component graph. Its
parameters are made gradient-free at construction and every teacher forward is
also run under ``torch.no_grad()`` by ``CompiledRun.state``. Both protections
are intentional: the first is the invariant a caller can inspect, while the
second prevents an output graph from being built through buffers or future
parameter-like state.

Buffer policy and module mode are separate card-driven choices. With buffer
EMA enabled, floating/complex buffers receive the same exponential update as
parameters and integral buffers (for example ``num_batches_tracked``) are
copied exactly. With it disabled, the teacher owns its buffers: they evolve
only if the card places the teacher in training mode.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping

import torch
from torch import Tensor, nn

from xty2.core.errors import TrainingError
from xty2.core.graph import ComponentGraph
from xty2.core.recipe import TeacherSpec
from xty2.core.schedules import as_schedule


class EMATeacher(nn.Module):
    """A gradient-free EMA copy of one student component graph."""

    def __init__(self, student: ComponentGraph, spec: TeacherSpec) -> None:
        super().__init__()
        if not isinstance(spec, TeacherSpec):
            raise TrainingError(f"EMATeacher needs a TeacherSpec, got {type(spec)}")
        self.spec = spec
        self._graph = copy.deepcopy(student)
        for parameter in self._graph.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._graph.train(spec.train_mode)
        self._check_structure(student)

    @property
    def graph(self) -> ComponentGraph:
        """The distinct graph evaluated for ``params='teacher'`` passes."""
        return self._graph

    @torch.no_grad()
    def update(self, student: ComponentGraph, step: int) -> None:
        """Move this teacher towards the current student once.

        `step` is the global optimiser step the update belongs to, and it is
        required rather than defaulted because `TeacherSpec.decay` is a
        schedule: a default would silently pin a scheduled decay to one point
        on its curve, which is the class of silent difference `FIDELITY.md` §2
        exists to catch. A constant decay ignores it.
        """
        self._check_structure(student)
        teacher_parameters = dict(self._graph.named_parameters())
        student_parameters = dict(student.named_parameters())
        decay = self._decay_at(step)
        for name, teacher_parameter in teacher_parameters.items():
            source_parameter = student_parameters[name].detach()
            teacher_parameter.mul_(decay).add_(source_parameter, alpha=1.0 - decay)

        if self.spec.applies_to_buffers:
            teacher_buffers = dict(self._graph.named_buffers())
            student_buffers = dict(student.named_buffers())
            for name, teacher_buffer in teacher_buffers.items():
                source_buffer = student_buffers[name].detach()
                if teacher_buffer.is_floating_point() or teacher_buffer.is_complex():
                    teacher_buffer.mul_(decay).add_(source_buffer, alpha=1.0 - decay)
                else:
                    teacher_buffer.copy_(source_buffer)

        # A defensive assertion at the boundary where an update happened. A
        # future refactor must not replace a Parameter object and accidentally
        # restore the default requires_grad=True.
        leaked = [
            name
            for name, parameter in self._graph.named_parameters()
            if parameter.requires_grad or parameter.grad is not None
        ]
        if leaked:
            raise TrainingError(
                f"EMA teacher parameters {leaked!r} acquired gradients. Teacher "
                "parameters must remain requires_grad=False with grad=None "
                "(FIDELITY.md Tier 0)."
            )

    def _decay_at(self, step: int) -> float:
        """The card's decay at `step`, re-checked at the step that uses it.

        `TeacherSpec` checks step 0 and the nominal value, which is every value
        a `Constant` ever takes and only two of the values a schedule takes.
        The rest are checked here, where a decay outside [0, 1) would otherwise
        turn one EMA update into an extrapolation away from the student and
        leave no trace but a diverging teacher.
        """
        decay = float(as_schedule(self.spec.decay)(step))
        if not 0.0 <= decay < 1.0:
            raise TrainingError(
                f"teacher EMA decay must be in [0, 1); "
                f"{as_schedule(self.spec.decay).describe()} is {decay!r} at "
                f"step {step}"
            )
        return decay

    def _check_structure(self, student: ComponentGraph) -> None:
        teacher_parameters = _tensor_shapes(self._graph.named_parameters())
        student_parameters = _tensor_shapes(student.named_parameters())
        if teacher_parameters != student_parameters:
            raise TrainingError(
                "the EMA teacher and student have different parameter "
                "structures; a teacher realisation is another parameter set "
                "of the same component graph"
            )
        teacher_buffers = _tensor_shapes(self._graph.named_buffers())
        student_buffers = _tensor_shapes(student.named_buffers())
        if teacher_buffers != student_buffers:
            raise TrainingError(
                "the EMA teacher and student have different buffer structures; "
                "buffer policy can choose how values update, not change which "
                "buffers exist"
            )


def _tensor_shapes(
    named: Iterable[tuple[str, Tensor]],
) -> Mapping[str, tuple[torch.Size, torch.dtype]]:
    return {name: (tensor.shape, tensor.dtype) for name, tensor in named}


__all__ = ["EMATeacher"]
