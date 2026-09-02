"""Validation-based selection of an immutable checkpoint from a gradient run."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from xty2.core.compile import CompiledRun
from xty2.core.errors import TrainingError

ValidationScore = Callable[[CompiledRun], float]


@dataclass(frozen=True)
class SelectionResult:
    """The completed optimiser step and score chosen by validation."""

    step: int
    score: float


@dataclass
class MinimumValidationSelection:
    """Retain the graph state with the smallest periodic validation score."""

    every: int
    score: ValidationScore
    _best: SelectionResult | None = field(default=None, init=False, repr=False)
    _state: Mapping[str, Tensor] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.every) is not int or self.every < 1:
            raise TrainingError(
                "validation selection interval must be a positive int, got "
                f"{self.every!r}"
            )
        if not callable(self.score):
            raise TrainingError("validation selection score must be callable")

    @property
    def result(self) -> SelectionResult | None:
        return self._best

    def consider(self, run: CompiledRun, completed_steps: int, *, final: bool) -> None:
        """Score an interval boundary (and always the final checkpoint)."""
        if not final and completed_steps % self.every:
            return
        modes = {module: module.training for module in run.graph.modules()}
        run.graph.eval()
        try:
            with torch.no_grad():
                value = float(self.score(run))
        finally:
            for module, was in modes.items():
                module.training = was
        if not math.isfinite(value):
            raise TrainingError(
                f"validation selection score at step {completed_steps} is {value}"
            )
        if self._best is None or value < self._best.score:
            self._best = SelectionResult(step=completed_steps, score=value)
            self._state = {
                name: tensor.detach().clone()
                for name, tensor in run.graph.state_dict().items()
            }

    def restore(self, run: CompiledRun) -> SelectionResult:
        """Restore the selected parameters and buffers after the search run."""
        if self._best is None or self._state is None:
            raise TrainingError("validation selection observed no checkpoints")
        run.graph.load_state_dict(self._state, strict=True)
        return self._best


__all__ = ["MinimumValidationSelection", "SelectionResult", "ValidationScore"]
