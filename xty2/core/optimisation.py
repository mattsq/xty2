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
* **`weight_decay` carries which components it reaches and whether it reaches
  biases and norm parameters**,
  which `FIDELITY.md` §2 calls out by name: the executor has to put every
  parameter in some group, so *something* decides this, and a framework
  default deciding it is exactly the invisible difference the checklist
  exists to surface. `WeightDecay` is one field holding coefficient, component
  scope and parameter reach rather than several fields sharing a key, which
  the card-key vocabulary forbids.
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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Final, Literal, get_args, overload

import torch
from torch import Tensor

from xty2.core.card_keys import REQUIRED, is_required
from xty2.core.errors import CompileError
from xty2.core.schedules import Schedule, as_schedule

OptimiserName = Literal["adam", "adamw", "lars", "sgd"]
"""The optimisers v1 builds. A fifth is a new branch of `build`, not a new
mechanism — and it arrives with the recipe that needs it (`DESIGN.md` §11).
`lars` arrives with BYOL, whose pinned `utils/optimizers.lars` is the
optimiser its learning rate, weight decay and trust coefficient are all
stated against; running it under SGD would be a different update rule with
the card's numbers on it."""

OPTIMISER_NAMES: Final[tuple[OptimiserName, ...]] = get_args(OptimiserName)

ClipMode = Literal["none", "norm", "value"]
"""The clipping modes `FIDELITY.md` §2 asks a card to choose between."""


# ---------------------------------------------------------------------------
# The two value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightDecay:
    """A decay coefficient *and* who it applies to (`FIDELITY.md` §2).

    One field holds coefficient, component scope and parameter reach because a
    canonical key names one value (`DESIGN.md` §9.1). Splitting them would
    either need new keys — the vocabulary is closed — or leave part of the
    policy unbound and therefore un-cross-checkable.

    Attributes:
        value: The coefficient passed to the optimiser.
        on_norm_and_bias: Whether one-dimensional parameters — biases, and the
            scale/shift of every norm layer — are decayed too. Papers that say
            "weight decay 1e-4" almost always mean the matrices only, and the
            difference is invisible in a diff and in a loss curve alike.
        components: Component names the decay reaches, or `None` for every
            trainable component. This is part of the same card decision: the
            TARNet reference regularises its outcome heads but not its shared
            representation.
    """

    value: float = REQUIRED
    on_norm_and_bias: bool = REQUIRED
    components: tuple[str, ...] | None = REQUIRED

    def __post_init__(self) -> None:
        _require_set("WeightDecay", "value", self.value, "optimisation.weight_decay")
        _require_set(
            "WeightDecay",
            "on_norm_and_bias",
            self.on_norm_and_bias,
            "optimisation.weight_decay",
        )
        _require_set(
            "WeightDecay",
            "components",
            self.components,
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
        if self.components is not None:
            object.__setattr__(self, "components", tuple(self.components))
            if not self.components:
                raise CompileError(
                    "WeightDecay.components is either None (all trainable "
                    "components) or a non-empty tuple of component names"
                )
            invalid = [name for name in self.components if not name.isidentifier()]
            if invalid:
                raise CompileError(
                    "WeightDecay.components must contain Python-identifier "
                    f"component names, got {invalid!r}"
                )
            if len(set(self.components)) != len(self.components):
                raise CompileError(
                    f"WeightDecay.components contains duplicates: {self.components!r}"
                )
            if not self.applies:
                raise CompileError(
                    "WeightDecay with value 0.0 cannot carry a component scope: "
                    "the scope would print as active but decay nothing"
                )

    @classmethod
    def none(cls) -> WeightDecay:
        """No decay at all, written explicitly so the plan says so."""
        return cls(value=0.0, on_norm_and_bias=False, components=None)

    @property
    def applies(self) -> bool:
        """Is any parameter actually decayed?"""
        return float(self.value) > 0.0

    def describe(self) -> str:
        """One stable line for the execution plan."""
        if not self.applies:
            return "none"
        scope = (
            "all trainable components"
            if self.components is None
            else "components " + ", ".join(self.components) + " only"
        )
        reach = "all parameters" if self.on_norm_and_bias else "norm and bias exempt"
        return f"{float(self.value)!r} ({scope}; {reach})"

    def decays(self, name: str, parameter: Tensor) -> bool:
        """Does this parameter belong in the decayed group?

        The rule is the usual one and it is stated once, here, rather than in
        the executor: a parameter with fewer than two dimensions is a bias or a
        norm scale, and everything else is a weight matrix. Component scope is
        resolved from the qualified name yielded by the stage freezer.
        """
        component = _parameter_component(name)
        selected = self.components is None or component in self.components
        return (
            self.applies and selected and (self.on_norm_and_bias or parameter.ndim >= 2)
        )


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
        eta: LARS only, and required there: the trust coefficient that scales
            every adapted update by `eta * ||parameter|| / ||update||`. It has
            no default because it is a number BYOL's config states
            (`optimizer_config.eta`), and a framework default for it would be
            exactly the silent difference §9.1 exists to prevent.
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
    eta: float | None = None

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
                "OptimiserSpec.weight_decay is a WeightDecay — coefficient, "
                "component scope, and whether it reaches biases and norm "
                "parameters are one card field (FIDELITY.md §2) — got "
                f"{type(self.weight_decay)}"
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
        if self.name == "lars":
            if self.eta is None:
                raise CompileError(
                    "OptimiserSpec(name='lars') was given no eta. The trust "
                    "coefficient is part of the optimiser's identity and binds "
                    "'optimisation.optimiser' with the rest of it, so it has no "
                    "default (DESIGN.md §9.1)."
                )
            _require_finite("OptimiserSpec.eta", self.eta)
            if self.eta <= 0:
                raise CompileError(
                    f"OptimiserSpec.eta must be positive, got {self.eta!r}. A "
                    "non-positive trust coefficient scales every adapted update "
                    "to zero or flips its sign."
                )
            if self.nesterov:
                raise CompileError(
                    "LARS takes no Nesterov momentum. The pinned "
                    "`utils/optimizers.scale_by_lars` accumulates "
                    "`mu = momentum * mu + update` and steps `-lr * mu`; a "
                    "look-ahead there is a different optimiser."
                )
        elif self.eta is not None:
            raise CompileError(
                f"optimiser {self.name!r} takes no eta, but this spec sets "
                f"eta={self.eta!r}. The trust coefficient is LARS's, and the run "
                "would silently ignore it."
            )
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
        if self.name == "lars":
            # The adaptation exclusion is part of the identity rather than of
            # `weight_decay`: BYOL's `exclude_bias_and_norm` filters the trust
            # ratio as well as the decay, and the two filters are separately
            # stated in the source even where they coincide.
            return (
                f"lars(momentum={float(self.momentum)!r}, "
                f"eta={float(self.eta or 0.0)!r}, "
                "adaptation on parameters of rank two or more)"
            )
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
        if self.weight_decay.components is not None:
            present = {_parameter_component(name) for name, _ in parameters}
            missing = sorted(set(self.weight_decay.components) - present)
            if missing:
                raise CompileError(
                    f"weight decay is scoped to components {missing!r}, but the "
                    f"optimiser parameters name {sorted(present)!r}. A scoped "
                    "card field must reach the component it names."
                )
        assignments = [
            (parameter, self.weight_decay.decays(name, parameter))
            for name, parameter in parameters
        ]
        decayed = [parameter for parameter, applies in assignments if applies]
        rest = [parameter for parameter, applies in assignments if not applies]
        groups: list[dict[str, Any]] = []
        if decayed:
            groups.append(
                {"params": decayed, "weight_decay": float(self.weight_decay.value)}
            )
        if rest:
            groups.append({"params": rest, "weight_decay": 0.0})
        if self.name == "lars":
            return LARS(
                groups,
                lr=float(self.lr),
                momentum=float(self.momentum),
                eta=float(self.eta or 0.0),
            )
        if self.name == "sgd":
            return torch.optim.SGD(
                groups,
                lr=float(self.lr),
                momentum=float(self.momentum),
                nesterov=self.nesterov,
            )
        family = torch.optim.AdamW if self.name == "adamw" else torch.optim.Adam
        return family(groups, lr=float(self.lr), betas=self.betas, eps=float(self.eps))


# ---------------------------------------------------------------------------
# The one optimiser torch does not ship
# ---------------------------------------------------------------------------


class LARS(torch.optim.Optimizer):
    """BYOL's pinned `utils/optimizers.lars`, in the order that reference applies it.

    `optax.chain(add_weight_decay(...), scale_by_lars(...), scale(-lr))` is
    three transformations over one gradient, and the order is the method:

    1. `update = grad + weight_decay * parameter`, for the parameters the decay
       filter admits;
    2. `update *= eta * ||parameter|| / ||update||`, for the parameters the
       adaptation filter admits and only where both norms are positive —
       otherwise the multiplier is exactly 1, which is the source's `jnp.where`
       and not a clamp or an epsilon;
    3. `mu = momentum * mu + update`, a plain accumulation rather than the
       convex `momentum * mu + (1 - momentum) * update` some LARS write-ups use;
    4. `parameter -= lr * mu`.

    Two of those are the ones a reimplementation usually gets wrong, so they are
    stated here rather than assumed: the trust ratio is computed from the update
    **after** the decay has been added, and a parameter the adaptation filter
    excludes still receives the momentum step — it is left unscaled, not left
    alone.

    The filters follow `optimizers.exclude_bias_and_norm`, which rejects a
    parameter named `b` or living under a module whose name contains `norm`. In
    torch that set is exactly the parameters of rank below two: every bias and
    every norm scale and shift is one-dimensional, and every weight matrix is
    not. The **decay** filter reaches this class as the per-group
    `weight_decay` that `OptimiserSpec.build` has already resolved from the
    card's `WeightDecay`, so a card that decays biases gets what it declared;
    the **adaptation** filter is LARS's own and is applied here, because it is
    part of the update rule rather than of the decay policy.

    Args:
        params: Parameters or parameter groups, as torch takes them.
        lr: The base rate. The executor writes the scheduled rate into the
            groups before each step, exactly as it does for the torch
            optimisers.
        momentum: The accumulation coefficient.
        eta: The trust coefficient.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float,
        momentum: float,
        eta: float,
    ) -> None:
        if lr < 0.0:
            raise CompileError(f"LARS lr must be non-negative, got {lr!r}")
        if momentum < 0.0:
            raise CompileError(f"LARS momentum must be non-negative, got {momentum!r}")
        if eta <= 0.0:
            raise CompileError(f"LARS eta must be positive, got {eta!r}")
        defaults: dict[str, Any] = {
            "lr": float(lr),
            "momentum": float(momentum),
            "eta": float(eta),
            "weight_decay": 0.0,
        }
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """One LARS update over every group, in the four steps above."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            eta = float(group["eta"])
            weight_decay = float(group["weight_decay"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if weight_decay != 0.0:
                    update = update.add(parameter, alpha=weight_decay)
                if adapts_under_lars(parameter):
                    parameter_norm = torch.linalg.vector_norm(parameter)
                    update_norm = torch.linalg.vector_norm(update)
                    if float(parameter_norm) > 0.0 and float(update_norm) > 0.0:
                        update = update * (eta * parameter_norm / update_norm)
                state = self.state[parameter]
                buffer = state.get("momentum_buffer")
                if buffer is None:
                    buffer = torch.zeros_like(parameter)
                    state["momentum_buffer"] = buffer
                buffer.mul_(momentum).add_(update)
                parameter.add_(buffer, alpha=-lr)
        return loss


def adapts_under_lars(parameter: Tensor) -> bool:
    """Does the trust ratio reach this parameter?

    `optimizers.exclude_bias_and_norm` in torch's vocabulary: biases and norm
    scales are one-dimensional and weight matrices are not. Exported so that a
    card stating one exclusion for both filters — as BYOL's does, because the
    source passes the same filter twice — can assert that its `WeightDecay`
    reach and this one actually coincide, rather than assuming it.
    """
    return parameter.ndim >= 2


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


def _parameter_component(name: str) -> str:
    """The component prefix of a qualified parameter name.

    `trainable_only` yields `component.submodule.parameter`. Accept the
    `ComponentGraph.named_parameters()` spelling too because diagnostics and
    tests naturally inspect it as `_components.component...`.
    """
    parts = name.split(".")
    if len(parts) > 1 and parts[0] == "_components":
        return parts[1]
    return parts[0]


__all__ = [
    "LARS",
    "OPTIMISER_NAMES",
    "ClipMode",
    "GradientClipping",
    "OptimiserName",
    "OptimiserSpec",
    "WeightDecay",
    "adapts_under_lars",
    "gradient_norm",
]
