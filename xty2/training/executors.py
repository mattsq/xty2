"""The `gradient` executor: one stage, one loop (`DESIGN.md` §7, `PLAN.md` P4).

Everything the loop does was decided before it started. `compile()` chose the
forward passes, the eligible row sets and the trainable components; the mixer
weights, reduces and logs; the `OptimiserSpec` says what descends and how fast.
This module is what remains: get a batch, run the planned passes, mix, step,
record.

Three properties are structural rather than remembered, and each is a Tier 0
invariant (`FIDELITY.md` §3):

**Trainable isolation.** `trainable_only` sets `requires_grad=False` on every
parameter outside the stage's `trainable` list for the duration of the run, so
a component upstream of a trained head is *executed* without accumulating a
gradient. Building the optimiser over the right parameters is not enough on its
own: a frozen encoder that still fills in `.grad` looks trained to any probe
that reads gradients, and would be trained by any later stage that reused the
optimiser's parameter list.

**Determinism.** The loop seeds torch once at the top, so a batch source that
draws randomly, dropout and every other stochastic element are all downstream
of one number. Same seed, same trace.

**The dataset is never mutated.** Batches are read and nothing else: no
transform, no in-place write, no writing back a prediction. `DESIGN.md` §7.1
requires it of every stage, and here it is simply what the loop does not do.

The executor runs **one** stage, because `Program` does not exist yet (P8).
`array_fit` and `cross_fit` are P10. What ships here is the executor the first
recipe needs, and no seam for the ones it does not.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.compile import CompiledRun, CompiledStage
from xty2.core.errors import TrainingError
from xty2.core.graph import ComponentGraph
from xty2.core.loss import TrainContext
from xty2.training.artifacts import Checkpoint, RunDirectory
from xty2.training.loss_mixer import (
    GradientProbe,
    GradientReport,
    LossMixer,
    MixedLoss,
    ObjectiveLog,
)

BatchSource = Iterable[XTYBatch]
"""What feeds a stage.

Deliberately the plainest thing that works: the executor pulls `stage.steps`
batches and does not decide what a batch contains. Sampling, the
labelled/unlabelled mix and the epoch boundary are the caller's, because no
loader exists yet and a half-built one here would own card keys it could not
check (`FIDELITY.md` §2, `optimisation.batch_size`).
"""


# ---------------------------------------------------------------------------
# What a run records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StepRecord:
    """One optimiser step, as the log records it (`DESIGN.md` §6.2).

    Floats and nothing else. A record holding the step's `total` as a live
    tensor would keep that step's autograd graph alive for the whole run,
    which turns "keep the trace" into a memory leak.

    Attributes:
        step: The global step.
        lr: The learning rate this step ran at — `lr` times the schedule's
            multiplier. Recorded rather than assumed, so a mis-stated warmup
            shows up in the trace instead of in a result.
        total: The mixed total that was descended.
        grad_norm: The global gradient norm **before** clipping.
        rows: How many rows the batch held.
        terms: The per-objective §6.2 lines.
        gradients: The probe's report on the steps it ran, else `None`.
    """

    step: int
    lr: float
    total: float
    grad_norm: float
    rows: int
    terms: tuple[ObjectiveLog, ...]
    gradients: GradientReport | None = None

    def as_json(self) -> dict[str, Any]:
        """The record as plain JSON-able data, for the run directory's log."""
        record: dict[str, Any] = {
            "step": self.step,
            "lr": self.lr,
            "total": self.total,
            "grad_norm": self.grad_norm,
            "rows": self.rows,
            "terms": [
                {
                    "name": term.name,
                    "rows": list(term.rows),
                    "reduction": term.reduction,
                    "value": term.value,
                    "weight": term.weight,
                    "weighted": term.weighted,
                    "n": term.n,
                    "coverage": term.coverage,
                    "diagnostics": dict(term.diagnostics),
                }
                for term in self.terms
            ],
        }
        if self.gradients is not None:
            record["gradients"] = {
                "shared_parameters": self.gradients.shared_parameters,
                "norms": dict(self.gradients.norms),
                "cosines": {
                    f"{left}|{right}": value
                    for (left, right), value in self.gradients.cosines.items()
                },
            }
        return record


@dataclass(frozen=True)
class StageResult:
    """What running one stage produced.

    Attributes:
        stage: The stage's name.
        recipe: The recipe it belongs to.
        seed: The seed the loop ran under.
        records: One `StepRecord` per optimiser step, in order.
        checkpoint: The stage's parameters and provenance (`DESIGN.md` §7.1).
    """

    stage: str
    recipe: str
    seed: int
    records: tuple[StepRecord, ...]
    checkpoint: Checkpoint
    paths: Mapping[str, str] = field(default_factory=dict)
    """Where the artifacts were written, when a run directory was given."""

    @property
    def trace(self) -> tuple[float, ...]:
        """The total at each step — what the determinism invariant compares."""
        return tuple(record.total for record in self.records)

    @property
    def steps(self) -> int:
        return len(self.records)

    def render(self, *, every: int = 1) -> str:
        """The trace as text, for a run log or a PR body."""
        lines = [f"stage {self.stage} ({self.recipe}, seed {self.seed})"]
        for record in self.records:
            if record.step % every:
                continue
            terms = "  ".join(
                f"{term.name}={term.value:.6g}(n={term.n})" for term in record.terms
            )
            lines.append(
                f"  step {record.step:<6} lr {record.lr:.6g}  "
                f"total {record.total:.6g}  |grad| {record.grad_norm:.6g}  {terms}"
            )
        lines.append(f"  {self.checkpoint.describe()}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trainable isolation
# ---------------------------------------------------------------------------


@contextmanager
def trainable_only(
    graph: ComponentGraph, trainable: Sequence[str]
) -> Iterator[tuple[tuple[str, Tensor], ...]]:
    """Freeze everything outside `trainable`, and restore it afterwards.

    Yields the `(qualified name, parameter)` pairs the stage may update, in
    the order the components were named.

    Freezing rather than merely excluding from the optimiser is the point.
    An excluded-but-unfrozen component still accumulates `.grad` on every
    backward, which (a) makes the trainable-isolation invariant untestable by
    reading gradients, and (b) leaves stale gradients for whatever runs next.

    Nothing is frozen until every component has been checked. A scan that
    froze as it went would leave the components it had already reached frozen
    when a later one turned out to be ineligible — the rejection would then
    corrupt the graph it exists to protect, and the caller would have no way to
    put it back.

    Raises:
        TrainingError: if a component the stage says it trains was already
            frozen. Freezing is by component name (`DESIGN.md` §8), so a
            parameter frozen underneath one is a silent no-op step.
    """
    to_freeze: list[Tensor] = []
    updatable: list[tuple[str, Tensor]] = []
    names = set(trainable)
    for name in graph.names:
        component = graph[name]
        for parameter_name, parameter in component.named_parameters():
            qualified = f"{name}.{parameter_name}"
            if name not in names:
                to_freeze.append(parameter)
                continue
            if not parameter.requires_grad:
                raise TrainingError(
                    f"stage trains {name!r}, but its parameter "
                    f"{qualified!r} has requires_grad=False. The optimiser "
                    "would hold a parameter no gradient reaches and every "
                    "step for it would be a no-op; freezing is by "
                    "component name (DESIGN.md §8)."
                )
            updatable.append((qualified, parameter))
    frozen = [(candidate, candidate.requires_grad) for candidate in to_freeze]
    for candidate in to_freeze:
        candidate.requires_grad_(False)
    try:
        yield tuple(updatable)
    finally:
        for restored, was in frozen:
            restored.requires_grad_(was)


# ---------------------------------------------------------------------------
# The factory that constructs the artifact (DESIGN.md §7.1)
# ---------------------------------------------------------------------------


def _emit_checkpoint(
    run: CompiledRun,
    stage: CompiledStage,
    parameters: Sequence[tuple[str, Tensor]],
    row_ids: Sequence[Tensor],
    *,
    steps: int,
    seed: int,
    fold: int | None = None,
) -> Checkpoint:
    """Build the stage's `Checkpoint` from what the run actually did.

    **Module-private, and that is the guard.** `Checkpoint.__init__` refuses a
    direct call, but a factory anyone can reach is a factory anyone can hand
    invented row ids to — the guard would then stop only the honest caller.
    `run_stage` is the one thing that calls this, and it passes the row ids of
    the batches it stepped on, so a `Checkpoint` in existence came from a run.
    Nothing here can certify that: Python has no way to prove a caller's
    provenance, and the claim is that the *only path in the package* produces
    it from the loop, not that the object is unforgeable.

    Every field that can be derived is derived rather than passed:
    `trained_on_row_ids` is the sorted, deduplicated union of the recorded
    batches, `plan_digest` is the digest of the plan those steps ran under, and
    `components` is the stage's trainable list.
    """
    seen = (
        torch.cat([rows.reshape(-1) for rows in row_ids])
        if row_ids
        else torch.zeros(0, dtype=torch.long)
    )
    return Checkpoint._issue(
        recipe=run.recipe.name,
        stage=stage.name,
        fold=fold,
        trained_on_row_ids=torch.unique(seen.cpu()),
        parameters=dict(parameters),
        components=tuple(stage.trainable),
        steps=steps,
        seed=seed,
        plan_digest=run.plan.digest,
    )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_stage(
    run: CompiledRun,
    stage: str | CompiledStage,
    batches: BatchSource,
    *,
    seed: int,
    run_dir: RunDirectory | None = None,
    probe: GradientProbe | None = None,
) -> StageResult:
    """Train one stage for `stage.steps` optimiser steps.

    Args:
        run: The compiled recipe. It decided the forward passes, the eligible
            rows and the trainable set; nothing here chooses any of them.
        stage: The stage to run, by name or compiled.
        batches: Where the rows come from. Must supply at least `stage.steps`
            batches; running dry is an error rather than a short run, because
            a stage that silently trained for fewer steps than its card says
            is the kind of difference nothing downstream would show.
        seed: Seeds torch before anything stochastic happens, which is what
            makes the loss trace reproducible.
        run_dir: Where to write the plan, the checkpoint and the log. `None`
            runs without writing anything.
        probe: The §6.2 gradient probe. Off by default, and so off in CI.

    Returns:
        The trace, the per-step log and the stage's checkpoint.
    """
    compiled = _resolve(run, stage)
    spec = compiled.optimiser
    graph = run.graph
    torch.manual_seed(seed)

    plan_path = run_dir.write_plan(run.plan) if run_dir is not None else None

    records: list[StepRecord] = []
    seen: list[Tensor] = []
    with trainable_only(graph, compiled.trainable) as parameters:
        optimiser = spec.build(parameters)
        tensors = [parameter for _, parameter in parameters]
        mixer = LossMixer.for_stage(compiled, probe=probe)
        source = iter(batches)
        modes = {module: module.training for module in graph.modules()}
        graph.train()
        try:
            for step in range(compiled.steps):
                batch = _next_batch(source, step, compiled)
                lr = _set_learning_rate(optimiser, spec.lr_at(step))
                mixed = _step(run, compiled, batch, mixer, optimiser, tensors, step)
                records.append(
                    StepRecord(
                        step=step,
                        lr=lr,
                        total=float(mixed.total.detach()),
                        grad_norm=mixed.grad_norm,
                        rows=batch.batch_size,
                        terms=mixed.loss.terms,
                        gradients=mixed.loss.gradients,
                    )
                )
                # Cloned, not viewed: a source that reuses one buffer across
                # batches would otherwise rewrite the provenance of every step
                # already recorded (DESIGN.md §7.1).
                seen.append(batch.row_id.detach().clone())
        finally:
            # Restored module by module. `graph.train(flag)` is recursive, so
            # one saved root flag would silently put a submodule a caller had
            # placed in eval mode back into training.
            for module, was in modes.items():
                module.training = was
        checkpoint = _emit_checkpoint(
            run, compiled, parameters, seen, steps=len(records), seed=seed
        )

    paths: dict[str, str] = {}
    if run_dir is not None and plan_path is not None:
        paths["plan"] = str(plan_path)
        paths["checkpoint"] = str(run_dir.write_checkpoint(checkpoint))
        paths["log"] = str(
            run_dir.write_log(compiled.name, [r.as_json() for r in records])
        )
    return StageResult(
        stage=compiled.name,
        recipe=run.recipe.name,
        seed=seed,
        records=tuple(records),
        checkpoint=checkpoint,
        paths=paths,
    )


@dataclass(frozen=True)
class _Stepped:
    """One step's mixed loss and the gradient norm it was clipped from."""

    loss: MixedLoss
    grad_norm: float

    @property
    def total(self) -> Tensor:
        return self.loss.total


def _step(
    run: CompiledRun,
    compiled: CompiledStage,
    batch: XTYBatch,
    mixer: LossMixer,
    optimiser: torch.optim.Optimizer,
    tensors: Sequence[Tensor],
    step: int,
) -> _Stepped:
    """Forward, mix, backward, clip, step — in that order and no other."""
    state = run.state(compiled, batch)
    ctx = TrainContext(global_step=step, schema=run.recipe.schema, stage=compiled.name)
    mixed = mixer.mix(state, batch, ctx, parameters=tensors)
    optimiser.zero_grad(set_to_none=True)
    grad_norm = 0.0
    # A step in which every term was empty carries a detached zero (§1.3):
    # there is nothing to descend, and calling backward on it would raise.
    # The step is still recorded, so a stage that has stopped seeing rows is
    # visible in the trace rather than absent from it.
    if mixed.total.requires_grad:
        mixed.total.backward()  # type: ignore[no-untyped-call]
        grad_norm = compiled.optimiser.clipping.apply(tensors)
        optimiser.step()
    return _Stepped(loss=mixed, grad_norm=grad_norm)


def _resolve(run: CompiledRun, stage: str | CompiledStage) -> CompiledStage:
    """The stage `run` compiled under that name — never one from another run.

    A `CompiledStage` carries its own objectives, optimiser and step count, so
    one compiled from a different recipe would be executed happily against
    *this* run's graph whenever the two graphs happen to be compatible. The
    checkpoint would then record this run's recipe and plan digest over a fit
    the plan does not describe — a provenance field that is wrong rather than
    missing, which is the one failure `DESIGN.md` §7.1 is written to prevent.
    """
    resolved = run.stage(stage if isinstance(stage, str) else stage.name)
    if not isinstance(stage, str) and resolved is not stage:
        raise TrainingError(
            f"the compiled stage {stage.name!r} passed here is not the one "
            f"recipe {run.recipe.name!r} compiled under that name. A stage is "
            "executed against the run that planned it, or the checkpoint's "
            "plan digest describes a fit that never happened (DESIGN.md §7.1). "
            "Pass the stage's name, or the CompiledStage from this run."
        )
    return resolved


def _next_batch(
    source: Iterator[XTYBatch], step: int, compiled: CompiledStage
) -> XTYBatch:
    try:
        batch = next(source)
    except StopIteration:
        raise TrainingError(
            f"the batch source ran dry at step {step} of stage "
            f"{compiled.name!r}, which is declared to run for {compiled.steps} "
            "steps. A stage that silently trained for fewer steps than its "
            "card states is not visible anywhere downstream — supply a source "
            "that yields at least that many batches."
        ) from None
    if not isinstance(batch, XTYBatch):
        raise TrainingError(
            f"the batch source yielded {type(batch)} at step {step}; a stage "
            "consumes XTYBatch (DESIGN.md §1.1)"
        )
    return batch


def _set_learning_rate(optimiser: torch.optim.Optimizer, lr: float) -> float:
    """Apply this step's learning rate to every parameter group."""
    for group in optimiser.param_groups:
        group["lr"] = lr
    return lr


__all__ = [
    "BatchSource",
    "StageResult",
    "StepRecord",
    "run_stage",
    "trainable_only",
]
