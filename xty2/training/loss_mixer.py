"""The loss mixer and the logging surface it owes (`DESIGN.md` §6, §6.2).

The mixer is the single place that knows about more than one objective, which
makes it the single place three things belong.

**Weighting and reduction.** Objectives return an unweighted mean over their
own eligible rows (§4); the mixer applies the schedule and the reduction mode a
paper prescribes (§6.1). Keeping them here is what leaves the logged raw value
comparable across runs and across modes.

**The zero-eligible-row rule.** A term with `n == 0` is excluded from the total
and logged with its coverage, rather than contributing a `NaN` from a mean over
an empty tensor (§1.3). It is still logged: an objective that has quietly
stopped seeing rows should be visible in the trace, not absent from it.

**Gradient attribution.** Once several objectives are mixed, validation
performance alone cannot tell you that a term has become numerically
irrelevant, or that two terms are fighting. Both happened in the previous
codebase and both were invisible. `GradientProbe` answers the first with a
per-objective gradient norm on the shared parameters and the second with the
pairwise cosine between those gradients.

The probe is **off unless a run asks for it**. `DESIGN.md` §6.2 gives 200 steps
as the cadence when it is on, and `PLAN.md` P3 requires it off by default and
off in CI; `GradientProbe()` is therefore disabled and `GradientProbe.periodic()`
is the 200-step setting a run opts into. It costs one extra backward pass per
objective, which is not a price a test suite should pay on every PR.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import ClassVar, Final

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.compile import CompiledObjective, CompiledStage
from xty2.core.errors import LossError
from xty2.core.graph import State
from xty2.core.loss import LossTerm, Reduction, TrainContext, apply_reduction
from xty2.core.rows import Rows, resolve_rows

PERIODIC_STEPS: Final = 200
"""The cadence `DESIGN.md` §6.2 gives for the probe *when it is enabled*."""


# ---------------------------------------------------------------------------
# What a step logs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectiveLog:
    """One objective's line in the per-step log (`DESIGN.md` §6.2).

    Attributes:
        name: The objective's name; the log is keyed by it, which is why the
            compiler rejects two objectives in one stage sharing one.
        rows: The eligible populations, as the compiler intersected them (§7.0).
        reduction: The mode this term's value entered the total under (§6.1).
        value: The **raw, unweighted** mean over eligible rows.
        weight: The schedule's value at this step.
        weighted: What actually entered the total: reduction applied, then
            weight. Zero for a term with no eligible rows.
        n: Eligible rows in this batch.
        coverage: `n / B` — the fraction of the batch this term saw.
        diagnostics: Whatever the objective reported alongside its value.
    """

    name: str
    rows: tuple[Rows, ...]
    reduction: Reduction
    value: float
    weight: float
    weighted: float
    n: int
    coverage: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Did this term see no eligible rows, and so stay out of the total?"""
        return self.n == 0


@dataclass(frozen=True)
class GradientReport:
    """Per-objective gradient norms and pairwise cosines (`DESIGN.md` §6.2).

    Both are computed on the **shared** parameters — those at least two
    objectives reach — because that is the only set on which "these two terms
    are fighting" is a meaningful statement. A parameter only one objective
    touches cannot be in conflict with anything.

    Norms are of the *weighted* contribution, since that is what the optimiser
    actually sees: a term whose weight has been scheduled to zero should read
    as numerically irrelevant, which is the diagnostic this exists for. Cosines
    are scale-free and so unaffected by the weight, except in sign.

    Attributes:
        step: The global step this was measured at.
        shared_parameters: How many parameter tensors two or more objectives
            reach. Zero means the cosines below are all zero by construction
            and say nothing.
        norms: `{objective: ‖g‖}` over the shared parameters.
        cosines: `{(a, b): cos(g_a, g_b)}` for `a < b` by name. Zero when
            either gradient is zero — there is no direction to compare.
    """

    step: int
    shared_parameters: int
    norms: Mapping[str, float]
    cosines: Mapping[tuple[str, str], float]

    def render(self) -> str:
        """The report as text, in a stable order."""
        lines = [
            f"  gradients @ step {self.step} (shared tensors: {self.shared_parameters})"
        ]
        width = max((len(name) for name in self.norms), default=0)
        for name in sorted(self.norms):
            lines.append(f"    |grad| {name:<{width}} = {self.norms[name]:.6g}")
        for left, right in sorted(self.cosines):
            lines.append(
                f"    cos({left}, {right}) = {self.cosines[(left, right)]:+.4f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MixedLoss:
    """The total, and everything the step logged to produce it.

    Attributes:
        total: The scalar the executor descends. Terms with `n == 0` are not
            in it, so a step in which *every* term was empty carries a
            detached zero — deliberately, since there is nothing to descend.
        step: `ctx.global_step`.
        stage: The stage that produced it.
        batch_size: `B`, which `population` reduction and coverage need.
        terms: One `ObjectiveLog` per objective, in the stage's order.
        gradients: The probe's report, or `None` when it did not run.
    """

    total: Tensor
    step: int
    stage: str
    batch_size: int
    terms: tuple[ObjectiveLog, ...]
    gradients: GradientReport | None = None

    @property
    def active(self) -> tuple[ObjectiveLog, ...]:
        """The terms that had eligible rows and so entered the total."""
        return tuple(term for term in self.terms if not term.is_empty)

    def term(self, name: str) -> ObjectiveLog:
        """The log line for objective `name`."""
        for term in self.terms:
            if term.name == name:
                return term
        raise LossError(
            f"no objective {name!r} in this step; it mixed "
            f"{[term.name for term in self.terms]!r}"
        )

    def render(self) -> str:
        """The step as a text table, for a run log or a PR body."""
        header = (
            f"step {self.step} (stage {self.stage}, B = {self.batch_size}, "
            f"total = {float(self.total.detach()):.6g})"
        )
        width = max((len(term.name) for term in self.terms), default=0)
        rows = max((len(_rows(term.rows)) for term in self.terms), default=0)
        lines = [header]
        for term in self.terms:
            lines.append(
                f"  {term.name:<{width}}  {_rows(term.rows):<{rows}}  "
                f"{term.reduction:<10}  n {term.n:<5}  value {term.value:>12.6g}  "
                f"weight {term.weight:>10.6g}  weighted {term.weighted:>12.6g}"
            )
            for key in sorted(term.diagnostics):
                lines.append(f"      {key} = {term.diagnostics[key]:.6g}")
        if self.gradients is not None:
            lines.append(self.gradients.render())
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


# ---------------------------------------------------------------------------
# The gradient probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GradientProbe:
    """When to measure per-objective gradients (`DESIGN.md` §6.2).

    Attributes:
        every: Measure on steps divisible by this. `None` — the default — is
            off, which is what CI runs under: the probe costs one backward
            pass per objective and proves nothing about correctness.
    """

    every: int | None = None

    PERIODIC: ClassVar[int] = PERIODIC_STEPS

    def __post_init__(self) -> None:
        if self.every is not None and self.every < 1:
            raise LossError(
                f"GradientProbe.every must be a positive step count or None "
                f"(off), got {self.every!r}"
            )

    @classmethod
    def periodic(cls, every: int = PERIODIC_STEPS) -> GradientProbe:
        """The §6.2 cadence, as an explicit opt-in."""
        return cls(every=every)

    @property
    def enabled(self) -> bool:
        return self.every is not None

    def should_run(self, step: int) -> bool:
        """Is `step` one of the steps this probe measures?"""
        return self.every is not None and step % self.every == 0


def gradient_report(
    contributions: Mapping[str, Tensor],
    parameters: Sequence[Tensor],
    *,
    step: int,
) -> GradientReport:
    """Per-objective gradient norms and pairwise cosines on shared parameters.

    Kept as a function rather than a method so that it is testable against
    constructed objectives whose gradients are analytically identical or
    analytically opposed — which is the only way a cosine trace is known to
    mean what it says (`PLAN.md` P3).

    Args:
        contributions: `{objective: weighted scalar}`. A contribution with no
            gradient path — detached, or scheduled to zero — is reported as an
            all-zero gradient rather than skipped, because "this term is no
            longer training anything" is the finding.
        parameters: The parameters to measure on. Those that do not require a
            gradient are dropped.
        step: The global step, recorded in the report.

    Returns:
        A `GradientReport` over the parameters at least two objectives reach.
    """
    measurable = [p for p in parameters if p.requires_grad]
    if not measurable:
        raise LossError(
            "the gradient probe was given no parameters that require a "
            "gradient. It measures per-objective gradients on the *shared* "
            "parameters (DESIGN.md §6.2), so it needs the stage's trainable "
            "parameters to measure on."
        )
    names = sorted(contributions)
    grads: dict[str, list[Tensor | None]] = {
        name: _gradients_of(contributions[name], measurable) for name in names
    }
    touched = [
        sum(1 for name in names if grads[name][index] is not None)
        for index in range(len(measurable))
    ]
    shared = [index for index, count in enumerate(touched) if count >= 2]
    flat = {name: _flatten(grads[name], measurable, shared) for name in names}
    norms = {name: float(vector.norm()) for name, vector in flat.items()}
    cosines: dict[tuple[str, str], float] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            scale = norms[left] * norms[right]
            cosines[(left, right)] = (
                float(torch.dot(flat[left], flat[right])) / scale if scale > 0 else 0.0
            )
    return GradientReport(
        step=step,
        shared_parameters=len(shared),
        norms=norms,
        cosines=cosines,
    )


def _gradients_of(
    contribution: Tensor, parameters: Sequence[Tensor]
) -> list[Tensor | None]:
    if not contribution.requires_grad:
        return [None] * len(parameters)
    return list(
        torch.autograd.grad(
            contribution,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    )


def _flatten(
    grads: Sequence[Tensor | None], parameters: Sequence[Tensor], shared: Sequence[int]
) -> Tensor:
    if not shared:
        return torch.zeros(1, dtype=parameters[0].dtype, device=parameters[0].device)
    pieces: list[Tensor] = []
    for index in shared:
        gradient = grads[index]
        # A parameter this objective does not reach contributes zeros rather
        # than being dropped: the vectors have to stay aligned across
        # objectives or the cosine compares different coordinates.
        pieces.append(
            torch.zeros_like(parameters[index]).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
        )
    return torch.cat(pieces)


# ---------------------------------------------------------------------------
# The mixer
# ---------------------------------------------------------------------------


class LossMixer:
    """Weights, reduces, sums and logs a stage's objectives (`DESIGN.md` §6).

    It is built from *compiled* objectives, because the eligible row set is the
    compiler's answer (§7.0) and the mixer must not re-derive it: a mixer that
    resolved rows itself would be a second implementation of the rule whose
    empty-by-construction case the compiler already rejected.

    This is also the seam that later supports alternating updates, per-objective
    update frequency, loss normalisation, GradNorm and PCGrad. None of those are
    implemented in v1 (§6, §11); the seam is all that exists.
    """

    __slots__ = ("_objectives", "_probe")

    def __init__(
        self,
        objectives: Sequence[CompiledObjective],
        *,
        probe: GradientProbe | None = None,
    ) -> None:
        if not objectives:
            raise LossError(
                "a LossMixer needs at least one objective; a stage with none is "
                "already a compile error (DESIGN.md §8)"
            )
        self._objectives = tuple(objectives)
        self._probe = probe or GradientProbe()

    @classmethod
    def for_stage(
        cls, stage: CompiledStage, *, probe: GradientProbe | None = None
    ) -> LossMixer:
        """The mixer for a compiled stage's objectives."""
        return cls(stage.objectives, probe=probe)

    @property
    def names(self) -> tuple[str, ...]:
        """The objective names, in the stage's order."""
        return tuple(compiled.name for compiled in self._objectives)

    @property
    def probe(self) -> GradientProbe:
        return self._probe

    def mix(
        self,
        state: State,
        batch: XTYBatch,
        ctx: TrainContext,
        *,
        parameters: Iterable[Tensor] = (),
    ) -> MixedLoss:
        """Compute every objective, weight it, and sum what is eligible.

        Args:
            state: The stage's planned forward passes, already run.
            batch: The full batch. `rows` indexes it and `state` alike.
            ctx: Supplies `global_step` — what the schedules are functions of.
            parameters: The stage's trainable parameters, for the gradient
                probe. Ignored when the probe is off, which is the default.

        Returns:
            A `MixedLoss` carrying the total and the full §6.2 step log.
        """
        total = torch.zeros((), dtype=batch.x.dtype, device=batch.device)
        logs: list[ObjectiveLog] = []
        contributions: dict[str, Tensor] = {}

        for compiled in self._objectives:
            rows = resolve_rows(batch, *compiled.rows)
            term = self._compute(compiled, state, batch, rows, ctx)
            weight = compiled.weight(ctx.global_step)
            if term.is_empty:
                # Excluded from the total, still logged: an objective that has
                # stopped seeing rows must be visible in the trace (§1.3).
                weighted = torch.zeros((), dtype=total.dtype, device=total.device)
            else:
                weighted = (
                    apply_reduction(
                        term, compiled.reduction, batch_size=batch.batch_size
                    )
                    * weight
                )
                total = total + weighted
                contributions[compiled.name] = weighted
            logs.append(
                ObjectiveLog(
                    name=compiled.name,
                    rows=compiled.rows,
                    reduction=compiled.reduction,
                    value=float(term.value.detach()),
                    weight=weight,
                    weighted=float(weighted.detach()),
                    n=term.n,
                    coverage=term.n / batch.batch_size,
                    diagnostics=term.diagnostics,
                )
            )

        gradients = None
        if self._probe.should_run(ctx.global_step) and contributions:
            gradients = gradient_report(
                contributions, tuple(parameters), step=ctx.global_step
            )
        return MixedLoss(
            total=total,
            step=ctx.global_step,
            stage=ctx.stage,
            batch_size=batch.batch_size,
            terms=tuple(logs),
            gradients=gradients,
        )

    def _compute(
        self,
        compiled: CompiledObjective,
        state: State,
        batch: XTYBatch,
        rows: Tensor,
        ctx: TrainContext,
    ) -> LossTerm:
        term = compiled.objective.compute(state, batch, rows, ctx)
        if not isinstance(term, LossTerm):
            raise LossError(
                f"objective {compiled.name!r} returned {type(term)}; compute "
                "returns a LossTerm (DESIGN.md §4)"
            )
        if term.n != int(rows.numel()):
            raise LossError(
                f"objective {compiled.name!r} reported n = {term.n} for an "
                f"eligible set of {int(rows.numel())} rows. `n` is the size of "
                "the set the objective was given, and the coverage log and every "
                "`sum` and `population` reduction are computed from it "
                "(DESIGN.md §4, §6.1)."
            )
        return term


def _rows(populations: tuple[Rows, ...]) -> str:
    return " & ".join(populations)


__all__ = [
    "PERIODIC_STEPS",
    "GradientProbe",
    "GradientReport",
    "LossMixer",
    "MixedLoss",
    "ObjectiveLog",
    "gradient_report",
]
