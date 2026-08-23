"""Weight schedules: `Constant`, `Ramp`, `Step` (`DESIGN.md` §6).

A schedule is a function of `ctx.global_step` that a `Weighted` term consults
for its weight. Three of them exist in v1 and a fourth needs a second real
consumer first (§11).

They live in `core/` rather than beside the mixer for the reason `DESIGN.md`
§10 gives for `Stage` and `Recipe`: they are the compiler's *input*. A recipe
declares them, `compile()` prints them into the execution plan, and the
reviewer diffs that against card §4's `losses.schedules` line. `training/`
imports `core/` and never the reverse, so a declaration the compiler reads
belongs in the leaf layer.

Two properties are load-bearing and are enforced here rather than trusted:

* **A schedule is a pure function of the step.** No internal cursor, nothing
  that advances when it is called. A schedule that counted its own calls would
  give a different trace under a gradient probe, which calls nothing extra but
  would still be blamed for it.
* **`describe()` is stable.** It is what the plan prints, and a description
  that varied run to run would make the plan diff noise.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from bisect import bisect_right
from dataclasses import dataclass

from xty2.core.errors import Xty2Error


class Schedule(ABC):
    """A weight as a function of the global step.

    Subclasses implement `value`, `nominal` and `describe`. `__call__` is the
    caller's entry point and validates both sides of the contract, so a
    schedule cannot be asked for a negative step and cannot answer with a
    non-finite weight.
    """

    @abstractmethod
    def value(self, step: int) -> float:
        """The weight at `step`. Called only with `step >= 0`."""

    @property
    @abstractmethod
    def nominal(self) -> float:
        """The weight this schedule settles at.

        Papers state a term's weight and its ramp separately — "λ = 100, ramped
        up over the first 80 epochs" — so the plan prints this as the weight and
        `describe()` as the schedule. For `Constant` the two agree.
        """

    @abstractmethod
    def describe(self) -> str:
        """One stable line for the execution plan."""

    def __call__(self, step: int) -> float:
        # `TrainContext` has already established that the step is a
        # non-negative int; this catches a schedule called directly.
        if step < 0:
            raise Xty2Error(f"global_step must be non-negative, got {step}")
        weight = float(self.value(step))
        if not math.isfinite(weight):
            raise Xty2Error(
                f"{self.describe()} produced a non-finite weight {weight!r} at "
                f"step {step}. A weight that is not a number silently poisons "
                "the total and every gradient downstream of it."
            )
        return weight

    def __str__(self) -> str:
        return self.describe()


@dataclass(frozen=True)
class Constant(Schedule):
    """The same weight at every step."""

    weight: float

    def __post_init__(self) -> None:
        _require_finite("Constant.weight", self.weight)

    def value(self, step: int) -> float:
        del step
        return float(self.weight)

    @property
    def nominal(self) -> float:
        return float(self.weight)

    def describe(self) -> str:
        return f"constant {float(self.weight)!r}"


@dataclass(frozen=True)
class Ramp(Schedule):
    """A linear ramp from `start` to `end` over `steps`, flat thereafter.

    Attributes:
        start: The weight at step 0.
        end: The weight from `steps` onwards.
        steps: The ramp length **in optimiser steps**. Papers state ramps in
            epochs about as often as in steps, and which one a card means is a
            required field (`FIDELITY.md` §2, `losses.schedules`); the
            conversion is the recipe's to do, so that the plan and the card
            speak the same unit.
    """

    start: float
    end: float
    steps: int

    def __post_init__(self) -> None:
        _require_finite("Ramp.start", self.start)
        _require_finite("Ramp.end", self.end)
        if self.steps < 1:
            raise Xty2Error(
                f"Ramp.steps must be at least 1, got {self.steps}. A zero-length "
                "ramp is a Constant, and writing it as one keeps the plan honest "
                "about what the recipe does."
            )

    def value(self, step: int) -> float:
        if step >= self.steps:
            return float(self.end)
        fraction = step / self.steps
        return float(self.start) + fraction * (float(self.end) - float(self.start))

    @property
    def nominal(self) -> float:
        return float(self.end)

    def describe(self) -> str:
        return (
            f"ramp {float(self.start)!r} -> {float(self.end)!r} over {self.steps} steps"
        )


@dataclass(frozen=True)
class Step(Schedule):
    """A piecewise-constant weight that jumps at `boundaries`.

    Attributes:
        weights: `len(boundaries) + 1` values. `weights[0]` holds from step 0
            until the first boundary.
        boundaries: Strictly increasing, positive step numbers at which the
            weight changes to the next entry of `weights`.
    """

    weights: tuple[float, ...]
    boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", tuple(self.weights))
        object.__setattr__(self, "boundaries", tuple(self.boundaries))
        if len(self.weights) != len(self.boundaries) + 1:
            raise Xty2Error(
                f"Step has {len(self.weights)} weights and "
                f"{len(self.boundaries)} boundaries; it needs exactly one more "
                "weight than boundary — the first holds from step 0."
            )
        for weight in self.weights:
            _require_finite("Step.weights", weight)
        if any(boundary < 1 for boundary in self.boundaries):
            raise Xty2Error(
                f"Step.boundaries must be positive step numbers, got "
                f"{self.boundaries!r}; the weight before the first boundary is "
                "weights[0]"
            )
        pairs = zip(self.boundaries, self.boundaries[1:], strict=False)
        if any(later <= earlier for earlier, later in pairs):
            raise Xty2Error(
                f"Step.boundaries must be strictly increasing, got {self.boundaries!r}"
            )

    def value(self, step: int) -> float:
        return float(self.weights[bisect_right(self.boundaries, step)])

    @property
    def nominal(self) -> float:
        return float(self.weights[-1])

    def describe(self) -> str:
        segments = [f"{float(self.weights[0])!r} from 0"]
        segments += [
            f"{float(weight)!r} from {boundary}"
            for weight, boundary in zip(self.weights[1:], self.boundaries, strict=True)
        ]
        return "step " + ", ".join(segments)


def as_schedule(weight: float | Schedule) -> Schedule:
    """Coerce a bare number to `Constant`, so recipes may write `weight=1.0`.

    A number is an unambiguous schedule, unlike a missing one — which is why
    this coerces a float and rejects `None`. `Weighted` has no default weight
    at all (`DESIGN.md` §9.1).
    """
    if isinstance(weight, Schedule):
        return weight
    if isinstance(weight, bool) or not isinstance(weight, int | float):
        raise Xty2Error(
            f"a weight is a number or a Schedule, got {type(weight)}. The v1 "
            f"schedules are Constant, Ramp and Step (DESIGN.md §6)."
        )
    return Constant(float(weight))


def _require_finite(label: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise Xty2Error(f"{label} must be a number, got {type(value)}")
    if not math.isfinite(float(value)):
        raise Xty2Error(f"{label} must be finite, got {float(value)!r}")


__all__ = ["Constant", "Ramp", "Schedule", "Step", "as_schedule"]
