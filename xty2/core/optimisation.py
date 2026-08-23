"""How a gradient stage descends: optimiser, learning rate, decay, clipping.

These are declarations, not machinery. They live in `core/` for the reason
`DESIGN.md` §10 gives for `Stage` and `schedules`: a recipe declares them,
`compile()` reads them into the execution plan, and `training/` imports
`core/` and never the reverse. What *runs* them is the gradient executor.

Every field here binds a card key from the `optimisation` block of
`FIDELITY.md` §2 and therefore carries the `REQUIRED` sentinel (§9.1). That is
not uniform pedantry — each of the four is a number a paper states and a
framework would otherwise supply silently:

* **`lr` and the optimiser identity** are the obvious ones, and the identity
  includes momentum or betas, because "SGD" and "SGD with Nesterov momentum
  0.9" are different optimisers stated in one line of a paper. They bind one
  key between them, as §9.1 requires, by binding a rendered `description`.
* **`weight_decay` carries whether it reaches biases and norm parameters**,
  which `FIDELITY.md` §2 calls out by name: the executor has to put every
  parameter in some group, so *something* decides this, and a framework
  default deciding it is exactly the invisible difference the checklist
  exists to surface. `WeightDecay` is one field holding both halves rather
  than two fields sharing a key, which the card-key vocabulary forbids.
* **`lr_schedule` is a real schedule, not a description.** A string field
  saying "cosine with warmup" would let a card claim a schedule the run does
  not have. It reuses `core.schedules` as a *multiplier* on `lr`, so warmup is
  `Ramp(0.0, 1.0, steps=n)` and no schedule is `Constant(1.0)`.
* **`clipping` binds `gradients.gradient_clipping`**, and lives here rather
  than on the stage because the mode and the threshold are one decision with
  the optimiser step they precede.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, get_args

import torch
from torch import Tensor

from xty2.core.card_keys import REQUIRED, is_required
from xty2.core.errors import CompileError
from xty2.core.schedules import Schedule, as_schedule

OptimiserName = Literal["adam", "adamw", "sgd"]
"""The optimisers v1 builds. A fourth is a new branch of `build`, not a new
mechanism — and it arrives with the recipe that needs it (`DESIGN.md` §11)."""

OPTIMISER_NAMES: Final[tuple[OptimiserName, ...]] = get_args(OptimiserName)

ClipMode = Literal["none", "norm", "value"]
"""The clipping modes `FIDELITY.md` §2 asks a card to choose between."""


# ---------------------------------------------------------------------------
# The two value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightDecay:
    """A decay coefficient *and* who it applies to (`FIDELITY.md` §2).

    One field holding both halves, because a canonical key names one value
    (`DESIGN.md` §9.1) and `optimisation.weight_decay` is annotated "and
    whether it applies to biases / norm params". Splitting it into two fields
    would either need a second key — the vocabulary is closed — or leave the
    second half unbound and therefore un-cross-checkable.

    Attributes:
        value: The coefficient passed to the optimiser.
        on_norm_and_bias: Whether one-dimensional parameters — biases, and the
            scale/shift of every norm layer — are decayed too. Papers that say
            "weight decay 1e-4" almost always mean the matrices only, and the
            difference is invisible in a diff and in a loss curve alike.
    """

    value: float = REQUIRED
    on_norm_and_bias: bool = REQUIRED

    def __post_init__(self) -> None:
        _require_set("WeightDecay", "value", self.value, "optimisation.weight_decay")
        _require_set(
            "WeightDecay",
            "on_norm_and_bias",
            self.on_norm_and_bias,
            "optimisation.weight_decay",
        )
        _require_finite("WeightDecay.value", self.value)
        if self.value < 0:
            raise CompileError(
                f"WeightDecay.value must be non-negative, got {self.value!r}"
            )
        if not isinstance(self.on_norm_and_bias, bool):
            raise CompileError(
                "WeightDecay.on_norm_and_bias must be a bool, got "
                f"{type(self.on_norm_and_bias)}"
            )

    @classmethod
    def none(cls) -> WeightDecay:
        """No decay at all, written explicitly so the plan says so."""
        return cls(value=0.0, on_norm_and_bias=False)

    @property
    def applies(self) -> bool:
        """Is any parameter actually decayed?"""
        return float(self.value) > 0.0

    def describe(self) -> str:
        """One stable line for the execution plan."""
        if not self.applies:
            return "none"
        reach = "all parameters" if self.on_norm_and_bias else "norm and bias exempt"
        return f"{float(self.value)!r} ({reach})"

    def decays(self, parameter: Tensor) -> bool:
        """Does this parameter belong in the decayed group?

        The rule is the usual one and it is stated once, here, rather than in
        the executor: a parameter with fewer than two dimensions is a bias or a
        norm scale, and everything else is a weight matrix.
        """
        return self.applies and (self.on_norm_and_bias or parameter.ndim >= 2)


@dataclass(frozen=True)
class GradientClipping:
    """Where the gradient is cut before the step (`FIDELITY.md` §2).

    Attributes:
        mode: `none`, `norm` (global norm, the common one) or `value`
            (element-wise).
        threshold: The bound. `None` exactly when `mode == "none"` — a
            threshold on a disabled clip is a number that reads as active in
            the plan and does nothing in the run.
    """

    mode: ClipMode = REQUIRED
    threshold: float | None = None

    def __post_init__(self) -> None:
        _require_set(
            "GradientClipping", "mode", self.mode, "gradients.gradient_clipping"
        )
        if self.mode not in get_args(ClipMode):
            raise CompileError(
                f"GradientClipping.mode must be one of {list(get_args(ClipMode))!r}, "
                f"got {self.mode!r} (FIDELITY.md §2, gradients.gradient_clipping)"
            )
        if self.mode == "none":
            if self.threshold is not None:
                raise CompileError(
                    f"GradientClipping(mode='none') carries threshold "
                    f"{self.threshold!r}. A threshold on a disabled clip prints "
                    "into the plan as though clipping were active and does "
                    "nothing in the run."
                )
            return
        if self.threshold is None:
            raise CompileError(
                f"GradientClipping(mode={self.mode!r}) needs a threshold; the "
                "mode and the number are one card field "
                "(FIDELITY.md §2, gradients.gradient_clipping)."
            )
        _require_finite("GradientClipping.threshold", self.threshold)
        if self.threshold <= 0:
            raise CompileError(
                f"GradientClipping.threshold must be positive, got {self.threshold!r}"
            )

    @classmethod
    def none(cls) -> GradientClipping:
        """No clipping, written explicitly — "none" must be explicit (§2)."""
        return cls(mode="none")

    @classmethod
    def by_norm(cls, threshold: float) -> GradientClipping:
        """Clip the global gradient norm to `threshold`."""
        return cls(mode="norm", threshold=threshold)

    @classmethod
    def by_value(cls, threshold: float) -> GradientClipping:
        """Clip every gradient element to `[-threshold, threshold]`."""
        return cls(mode="value", threshold=threshold)

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def describe(self) -> str:
        """One stable line for the execution plan."""
        if not self.enabled:
            return "none"
        return f"{self.mode} at {float(self.threshold or 0.0)!r}"

    def apply(self, parameters: Sequence[Tensor]) -> float:
        """Clip `parameters` in place and return the norm **before** clipping.

        The pre-clip norm is returned rather than the post-clip one because it
        is the diagnostic: a run whose clip is permanently active is a run
        whose stated threshold is doing something the paper may not have
        intended, and the post-clip norm — pinned at the threshold — is exactly
        the number that would hide it.
        """
        if self.mode == "norm":
            threshold = float(self.threshold or 0.0)
            return float(torch.nn.utils.clip_grad_norm_(parameters, threshold))
        total = gradient_norm(parameters)
        if self.mode == "value":
            torch.nn.utils.clip_grad_value_(parameters, float(self.threshold or 0.0))
        return total


def gradient_norm(parameters: Iterable[Tensor]) -> float:
    """The global L2 norm of `parameters`' gradients; 0.0 when there are none."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))


# ---------------------------------------------------------------------------
# The optimiser spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimiserSpec:
    """What a gradient stage optimises with (`DESIGN.md` §7, `FIDELITY.md` §2).

    Attributes:
        name: `adam`, `adamw` or `sgd`.
        lr: The base learning rate. `lr_schedule` multiplies it.
        weight_decay: The coefficient and its reach.
        lr_schedule: A **multiplier** on `lr`, as a function of the global
            step. `Constant(1.0)` is "no schedule", written explicitly;
            `Ramp(0.0, 1.0, steps=n)` is linear warmup.
        clipping: Where the gradient is cut before the step.
        momentum: SGD only. Rejected on the Adam family, where it is not a
            knob and would silently do nothing.
        nesterov: SGD only, and only with momentum.
        betas: The Adam family only.
        eps: The Adam family only.
    """

    name: OptimiserName = REQUIRED
    lr: float = REQUIRED
    weight_decay: WeightDecay = REQUIRED
    lr_schedule: Schedule = REQUIRED
    clipping: GradientClipping = REQUIRED
    momentum: float = 0.0
    nesterov: bool = False
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "description": "optimisation.optimiser",
        "lr": "optimisation.lr",
        "schedule_description": "optimisation.lr_schedule",
        "decay_description": "optimisation.weight_decay",
        "clipping_description": "gradients.gradient_clipping",
    }
    """Three of the five bind a *rendered* field rather than the object.

    `plan.hyperparameters` is what a reviewer diffs against a card's §4 YAML,
    where these are written as prose — "constant, no warmup", "1e-4, not on
    biases". A dataclass repr in that column is diffable by a machine and not
    by eye, and eye is the half that establishes fidelity (`FIDELITY.md`
    §1.2). The structured values stay on the spec, which is what the executor
    reads.
    """

    def __post_init__(self) -> None:
        for field, key in (
            ("name", "optimisation.optimiser"),
            ("lr", "optimisation.lr"),
            ("weight_decay", "optimisation.weight_decay"),
            ("lr_schedule", "optimisation.lr_schedule"),
            ("clipping", "gradients.gradient_clipping"),
        ):
            _require_set("OptimiserSpec", field, getattr(self, field), key)
        if self.name not in OPTIMISER_NAMES:
            raise CompileError(
                f"unknown optimiser {self.name!r}; v1 builds "
                f"{list(OPTIMISER_NAMES)!r}. A fifth is a new branch of "
                "OptimiserSpec.build and arrives with the recipe that needs it."
            )
        _require_finite("OptimiserSpec.lr", self.lr)
        if self.lr <= 0:
            raise CompileError(f"OptimiserSpec.lr must be positive, got {self.lr!r}")
        if not isinstance(self.weight_decay, WeightDecay):
            raise CompileError(
                "OptimiserSpec.weight_decay is a WeightDecay — the coefficient "
                "and whether it reaches biases and norm parameters are one card "
                f"field (FIDELITY.md §2) — got {type(self.weight_decay)}"
            )
        if not isinstance(self.clipping, GradientClipping):
            raise CompileError(
                "OptimiserSpec.clipping is a GradientClipping, got "
                f"{type(self.clipping)}"
            )
        object.__setattr__(self, "lr_schedule", as_schedule(self.lr_schedule))
        self._reject_inapplicable_knobs()

    def _reject_inapplicable_knobs(self) -> None:
        """A knob the chosen optimiser ignores is a paper detail that vanished.

        `SGD(momentum=0.9)` written against Adam is not a harmless extra
        argument: the recipe says the paper's momentum and the run has none,
        and nothing downstream — plan, log or loss curve — would say so.
        """
        adam = self.name in ("adam", "adamw")
        if adam and (self.momentum != 0.0 or self.nesterov):
            raise CompileError(
                f"optimiser {self.name!r} takes no momentum, but this spec sets "
                f"momentum={self.momentum!r}, nesterov={self.nesterov!r}. The "
                "run would silently have none. Use 'sgd', or drop the knob."
            )
        if not adam and self.betas != (0.9, 0.999):
            raise CompileError(
                f"optimiser {self.name!r} takes no betas, but this spec sets "
                f"betas={self.betas!r}. The run would silently ignore them."
            )
        if not adam:
            _require_finite("OptimiserSpec.momentum", self.momentum)
            if self.momentum < 0:
                raise CompileError(
                    f"OptimiserSpec.momentum must be non-negative, got "
                    f"{self.momentum!r}"
                )
            if self.nesterov and self.momentum <= 0:
                raise CompileError(
                    "Nesterov momentum needs a positive momentum; this spec sets "
                    f"momentum={self.momentum!r}, which torch rejects and a "
                    "paper never means."
                )

    @property
    def description(self) -> str:
        """The optimiser's full identity — the value bound to the card key.

        Momentum and betas are in here rather than in keys of their own,
        because the vocabulary is closed and because a paper states them in the
        same breath as the optimiser: "SGD with Nesterov momentum 0.9".
        """
        if self.name in ("adam", "adamw"):
            return f"{self.name}(betas={self.betas!r}, eps={self.eps!r})"
        return (
            f"{self.name}(momentum={float(self.momentum)!r}, "
            f"nesterov={self.nesterov!r})"
        )

    @property
    def schedule_description(self) -> str:
        """The learning-rate schedule as one line — the bound card value."""
        return self.lr_schedule.describe()

    @property
    def decay_description(self) -> str:
        """The weight decay and its reach as one line."""
        return self.weight_decay.describe()

    @property
    def clipping_description(self) -> str:
        """The clipping mode and threshold as one line."""
        return self.clipping.describe()

    def lr_at(self, step: int) -> float:
        """The learning rate at `step`: `lr` times the schedule's multiplier.

        A negative product is rejected here rather than by torch. A `Schedule`
        checks that a weight is finite and not that it is positive, because a
        *loss* weight may legitimately be negative — and the executor sets the
        rate by writing into the optimiser's parameter groups, which is the one
        path around torch's constructor-time check. A schedule that dipped
        below zero would run gradient *ascent* for as long as it stayed there,
        on a run whose plan and card both read as correct.
        """
        rate = float(self.lr) * self.lr_schedule(step)
        if rate < 0:
            raise CompileError(
                f"{self.lr_schedule.describe()} puts the learning rate at "
                f"{rate!r} on step {step}. A negative rate ascends the "
                "gradient; the schedule multiplies `lr` and must stay "
                "non-negative (DESIGN.md §6, §7)."
            )
        return rate

    def describe_lines(self) -> tuple[str, ...]:
        """The spec as the plan prints it, one card field per line."""
        return (
            f"optimiser     {self.description}",
            f"lr            {float(self.lr)!r}",
            f"lr schedule   {self.schedule_description}",
            f"weight decay  {self.decay_description}",
            f"clipping      {self.clipping_description}",
        )

    def build(self, parameters: Sequence[tuple[str, Tensor]]) -> torch.optim.Optimizer:
        """The torch optimiser over `parameters`, in decayed / undecayed groups.

        Args:
            parameters: `(name, parameter)` pairs — the stage's trainable
                parameters, named so a failure can say which one.

        Raises:
            CompileError: if the stage has no parameters to train, which torch
                would otherwise report as "empty parameter list" with nothing
                about which stage or component is at fault.
        """
        if not parameters:
            raise CompileError(
                "the optimiser was given no parameters. A stage that trains "
                "nothing is a compile error (DESIGN.md §8.4); reaching an "
                "optimiser with an empty list means the trainable components "
                "hold no parameters at all."
            )
        decayed = [p for _, p in parameters if self.weight_decay.decays(p)]
        rest = [p for _, p in parameters if not self.weight_decay.decays(p)]
        groups: list[dict[str, Any]] = []
        if decayed:
            groups.append(
                {"params": decayed, "weight_decay": float(self.weight_decay.value)}
            )
        if rest:
            groups.append({"params": rest, "weight_decay": 0.0})
        if self.name == "sgd":
            return torch.optim.SGD(
                groups,
                lr=float(self.lr),
                momentum=float(self.momentum),
                nesterov=self.nesterov,
            )
        family = torch.optim.AdamW if self.name == "adamw" else torch.optim.Adam
        return family(groups, lr=float(self.lr), betas=self.betas, eps=float(self.eps))


def _require_set(owner: str, field: str, value: object, key: str) -> None:
    if is_required(value):
        raise CompileError(
            f"{owner} was given no {field!r}. It binds card key {key!r} and is "
            "governed by the paper, so it has no usable default — the recipe "
            "sets it explicitly (DESIGN.md §9.1, CLAUDE.md standing rules)."
        )


def _require_finite(label: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CompileError(f"{label} must be a number, got {type(value)}")
    if not math.isfinite(float(value)):
        raise CompileError(f"{label} must be finite, got {float(value)!r}")


__all__ = [
    "OPTIMISER_NAMES",
    "ClipMode",
    "GradientClipping",
    "OptimiserName",
    "OptimiserSpec",
    "WeightDecay",
    "gradient_norm",
]
