"""The gradient executor and ordered program runner (`DESIGN.md` §7).

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

P8 adds the deliberately small outer loop: a `Program` is executed in order,
each stage starts from the recipe's initial state plus exactly the checkpoint
named by `initialise_from`. P10 makes execution explicit rather than inferred:
``gradient`` keeps this loop, ``array_fit`` calls a functional array action
once over resolved rows, and ``cross_fit`` repeats the gradient fit per actual
``fold_id`` then emits held-out pseudo labels whose provenance can be checked.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor, nn

from xty2.core.batch import XTYBatch
from xty2.core.compile import CompiledObjective, CompiledRun, CompiledStage
from xty2.core.data import Dataset, ExternalBatches, TrainingPopulation
from xty2.core.errors import ArtifactError, TrainingError
from xty2.core.graph import ComponentGraph, State
from xty2.core.loss import (
    LossTerm,
    StatefulObjective,
    TrainContext,
    apply_reduction,
    treatment_distribution,
)
from xty2.core.recipe import ArrayFitAction, PseudoLabelAction
from xty2.core.rows import resolve_rows
from xty2.objectives.meta_pseudo_labels import (
    MetaFeedbackCoefficient,
    MetaPseudoLabelScore,
    SampledTeacherTreatmentNLL,
)
from xty2.training.artifacts import Checkpoint, PseudoLabels, RunDirectory
from xty2.training.loading import build_population, check_fitted_on, iterate
from xty2.training.loss_mixer import (
    GradientProbe,
    GradientReport,
    LossMixer,
    MixedLoss,
    ObjectiveLog,
)
from xty2.training.selection import MinimumValidationSelection, SelectionResult
from xty2.training.teacher import EMATeacher

STREAM_STRIDE = 1_000_000
"""How far apart `run_program` and `run_cross_fit` space their child seeds.

A stage's view keys are `seed`, `seed + 1`, ... one per optimiser step, because
the Mean Teacher card pins the per-step key as `s_r + 10000 + j` against the
seed its caller supplies (`docs/recipes/mean_teacher.md` §6.2). That schedule
is only collision-free while sibling stages and folds start further apart than
either of them can walk, so their seeds are spaced by this stride rather than
by one.
"""

MAX_STAGE_STEPS = STREAM_STRIDE // 2
"""The longest stage whose keys are guaranteed to stay inside its own stride.

One stage or fold spends one key per optimiser step and then one more per
held-out prediction batch, and the prediction batches are the captured step
batches, so half the stride is the budget `_run_stage` enforces.
"""

BatchSource = Iterable[XTYBatch]
"""What feeds a stage.

Deliberately the plainest thing that works: the executor pulls `stage.steps`
batches and does not decide what a batch contains. Sampling, the
labelled/unlabelled mix and the epoch boundary are the caller's, because no
loader exists yet and a half-built one here would own card keys it could not
check (`FIDELITY.md` §2, `optimisation.batch_size`).
"""

StageData = "BatchSource | Dataset"
"""What a caller hands a stage: a `Dataset` where the stage declares a sampler,
an iterable of batches where it declares `ExternalBatches`.

Which one is not a runtime preference. The stage's declaration decides, and
handing over the other kind is an error rather than a silent reinterpretation:
a `Dataset` fed to an `ExternalBatches` stage would be a policy nothing
applied, and an iterable fed to a sampling stage would be the batch composition
the card pins down, quietly replaced by whatever arrived.
"""

BatchSources = Mapping[str, "BatchSource | Dataset"]
"""One explicitly named source per stage in an ordered program."""


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
    role_lrs: Mapping[str, float] = field(default_factory=dict)
    role_grad_norms: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "role_lrs", MappingProxyType(dict(self.role_lrs)))
        object.__setattr__(
            self,
            "role_grad_norms",
            MappingProxyType(dict(self.role_grad_norms)),
        )

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
        if self.role_lrs:
            record["role_lrs"] = dict(self.role_lrs)
        if self.role_grad_norms:
            record["role_grad_norms"] = dict(self.role_grad_norms)
        return record


@dataclass(frozen=True)
class StageResult:
    """What running one stage produced.

    Attributes:
        stage: The stage's name.
        recipe: The recipe it belongs to.
        seed: The seed the loop ran under.
        records: One `StepRecord` per optimiser step, in order.
        checkpoint: Gradient and array-fit stages expose one primary immutable
            checkpoint. A cross-fit/action-only stage instead exposes fold
            checkpoints and/or pseudo labels; accessing ``checkpoint`` then is
            a loud error rather than an arbitrary fold choice.
        teacher: The stage-local EMA parameter set, when one was declared.
            It is a live model for inspection or inference, not the immutable
            student checkpoint written as the stage artifact.
        population: The training population the declared policy produced, and
            `None` for a stage that took the caller's batches. It carries the
            statistics the standardisation fitted and the rows they were fitted
            on, which is how a caller applies the *same* transform to held-out
            rows: refitting them on the evaluation split is the leakage
            `FIDELITY.md` §2 names first, and a run that could not hand back
            what it fitted would make refitting the path of least resistance.
        objective_states: The final executor-owned state of each stateful
            objective. Stateless stages expose an empty mapping. This is
            diagnostic state, not model state: checkpoints remain parameters
            and buffers only.
    """

    stage: str
    recipe: str
    seed: int
    records: tuple[StepRecord, ...]
    _checkpoint: Checkpoint | None = field(repr=False)
    fold_checkpoints: Mapping[int, Checkpoint] = field(default_factory=dict)
    role_checkpoints: Mapping[str, Checkpoint] = field(default_factory=dict)
    role_graphs: Mapping[str, ComponentGraph] = field(default_factory=dict, repr=False)
    pseudo_labels: PseudoLabels | None = None
    teacher: EMATeacher | None = None
    population: TrainingPopulation | None = None
    objective_states: Mapping[str, object] = field(default_factory=dict)
    selection: SelectionResult | None = None
    paths: Mapping[str, str] = field(default_factory=dict)
    """Where the artifacts were written, when a run directory was given."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(
            self,
            "fold_checkpoints",
            MappingProxyType(dict(sorted(self.fold_checkpoints.items()))),
        )
        object.__setattr__(
            self,
            "role_checkpoints",
            MappingProxyType(dict(sorted(self.role_checkpoints.items()))),
        )
        object.__setattr__(
            self,
            "role_graphs",
            MappingProxyType(dict(sorted(self.role_graphs.items()))),
        )
        object.__setattr__(
            self, "objective_states", MappingProxyType(dict(self.objective_states))
        )
        object.__setattr__(self, "paths", MappingProxyType(dict(self.paths)))

    @property
    def trace(self) -> tuple[float, ...]:
        """The total at each step — what the determinism invariant compares."""
        return tuple(record.total for record in self.records)

    @property
    def steps(self) -> int:
        return len(self.records)

    @property
    def has_checkpoint(self) -> bool:
        return self._checkpoint is not None

    @property
    def checkpoint(self) -> Checkpoint:
        """The stage's single primary checkpoint, when it has one."""
        if self._checkpoint is None:
            raise TrainingError(
                f"stage {self.stage!r} has no single checkpoint. Cross-fit "
                "stages expose fold_checkpoints, meta-gradient stages expose "
                "role_checkpoints, and action-only stages expose their "
                "pseudo-label artifact."
            )
        return self._checkpoint

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
        if self._checkpoint is not None:
            lines.append(f"  {self._checkpoint.describe()}")
        for fold, checkpoint in sorted(self.fold_checkpoints.items()):
            lines.append(f"  fold {fold}: {checkpoint.describe()}")
        for role, checkpoint in sorted(self.role_checkpoints.items()):
            lines.append(f"  role {role}: {checkpoint.describe()}")
        if self.pseudo_labels is not None:
            lines.append(f"  {self.pseudo_labels.describe()}")
        return "\n".join(lines)


@dataclass(frozen=True)
class ProgramResult:
    """The ordered results and checkpoints produced by one program run."""

    recipe: str
    seed: int
    stages: tuple[StageResult, ...]

    def stage(self, name: str) -> StageResult:
        """The result for stage `name`."""
        for result in self.stages:
            if result.stage == name:
                return result
        raise TrainingError(
            f"program result for {self.recipe!r} has no stage {name!r}; it has "
            f"{[result.stage for result in self.stages]!r}"
        )

    @property
    def checkpoints(self) -> Mapping[str, Checkpoint]:
        """Every primary stage checkpoint, keyed by stage name and read-only."""
        return MappingProxyType(
            {
                result.stage: result.checkpoint
                for result in self.stages
                if result.has_checkpoint
            }
        )

    @property
    def pseudo_labels(self) -> Mapping[str, PseudoLabels]:
        """Every emitted pseudo-label table, keyed by producing stage."""
        return MappingProxyType(
            {
                result.stage: result.pseudo_labels
                for result in self.stages
                if result.pseudo_labels is not None
            }
        )


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
        buffers=dict(_component_buffers(run.graph, stage.trainable)),
        components=tuple(stage.trainable),
        steps=steps,
        seed=seed,
        plan_digest=run.plan.digest,
    )


def _emit_array_checkpoint(
    run: CompiledRun,
    stage: CompiledStage,
    action: ArrayFitAction,
    state: object,
    row_ids: Tensor,
    *,
    seed: int,
) -> Checkpoint:
    """Issue tensor state returned by one functional array fit."""
    if not isinstance(state, Mapping) or not state:
        raise TrainingError(
            f"array-fit action {action.name!r} returned no tensor state. A "
            "checkpoint must contain the complete fitted state, not merely a "
            "success flag."
        )
    qualified: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not name or "\n" in name:
            raise TrainingError(
                f"array-fit action {action.name!r} returned invalid state name {name!r}"
            )
        if not isinstance(value, Tensor):
            raise TrainingError(
                f"array-fit action {action.name!r} state {name!r} is "
                f"{type(value)}, expected Tensor"
            )
        qualified[f"{action.name}.{name}"] = value
    return Checkpoint._issue(
        recipe=run.recipe.name,
        stage=stage.name,
        fold=None,
        trained_on_row_ids=torch.unique(row_ids.detach().cpu()),
        parameters=qualified,
        buffers={},
        components=(action.name,),
        steps=1,
        seed=seed,
        plan_digest=run.plan.digest,
    )


def _emit_pseudo_labels(
    run: CompiledRun,
    stage: CompiledStage,
    checkpoints: Mapping[int, Checkpoint],
    row_ids: Sequence[Tensor],
    predicted_by_fold: Sequence[Tensor],
    labels: Sequence[Tensor],
) -> PseudoLabels:
    """Issue one deduplicated side table from predictions the executor made."""
    action = stage.action
    if not isinstance(action, PseudoLabelAction):
        raise TrainingError(
            f"stage {stage.name!r} does not declare a PseudoLabelAction"
        )
    rows = torch.cat([value.reshape(-1).cpu() for value in row_ids])
    folds = torch.cat([value.reshape(-1).cpu() for value in predicted_by_fold])
    values = torch.cat([value.reshape(-1).cpu() for value in labels])
    if not (rows.numel() == folds.numel() == values.numel()):
        raise TrainingError("pseudo-label prediction fields are not row-aligned")
    by_row: dict[int, tuple[int, int]] = {}
    for row, fold, label in zip(
        rows.tolist(), folds.tolist(), values.tolist(), strict=True
    ):
        candidate = (int(fold), int(label))
        existing = by_row.get(int(row))
        if existing is not None and existing != candidate:
            raise TrainingError(
                f"row {row} was pseudo-labelled more than once with conflicting "
                f"(fold, label) values {existing!r} and {candidate!r}"
            )
        by_row[int(row)] = candidate
    ordered = sorted(by_row)
    return PseudoLabels._issue(
        source_stage=stage.name,
        source_checkpoints=checkpoints,
        predicted_by_fold=torch.tensor(
            [by_row[row][0] for row in ordered], dtype=torch.long
        ),
        row_id=torch.tensor(ordered, dtype=torch.long),
        labels=torch.tensor([by_row[row][1] for row in ordered], dtype=torch.long),
        used_y=run.graph.port_depends_on_raw_outcome(action.port),
    )


# ---------------------------------------------------------------------------
# What feeds a stage
# ---------------------------------------------------------------------------


def _feed(
    run: CompiledRun,
    compiled: CompiledStage,
    data: BatchSource | Dataset,
    *,
    seed: int,
) -> tuple[BatchSource, TrainingPopulation | None]:
    """Resolve what the caller supplied against what the stage declared.

    Where the stage samples, this is the whole of the loader: the declared
    policy is applied to the supplied rows, the fitted statistics are checked
    against the training assignment they claim to come from, and the stream is
    drawn from the result. The plan therefore *causes* the batches rather than
    describing them, which is the difference between a compiled policy and a
    caller-supplied loader that reports on itself (`DESIGN.md` §7.1).
    """
    sampler = compiled.stage.sampler
    if isinstance(sampler, ExternalBatches):
        if isinstance(data, Dataset):
            raise TrainingError(
                f"stage {compiled.name!r} declares ExternalBatches — the caller "
                "supplies its batches — but was handed a Dataset. Nothing would "
                "apply a policy to it. Declare a sampler on the stage, or pass "
                "the batches."
            )
        return data, None
    if not isinstance(data, Dataset):
        raise TrainingError(
            f"stage {compiled.name!r} declares {type(sampler).__name__} and "
            f"draws its own batches, but was handed {type(data).__name__}. Pass "
            "a Dataset: accepting an iterable here would silently replace the "
            "batch composition the card pins down."
        )
    spec = run.recipe.data
    if spec is None:  # pragma: no cover - compile() rejects this pairing
        raise TrainingError(
            f"recipe {run.recipe.name!r} has a sampling stage and no data policy"
        )
    population = build_population(data, spec, seed=seed)
    check_fitted_on(population, data, spec)
    return iterate(population, sampler, steps=compiled.steps, seed=seed), population


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def run_stage(
    run: CompiledRun,
    stage: str | CompiledStage,
    batches: BatchSource | Dataset,
    *,
    seed: int,
    run_dir: RunDirectory | None = None,
    probe: GradientProbe | None = None,
    selection: MinimumValidationSelection | None = None,
) -> StageResult:
    """Train one stage for `stage.steps` optimiser steps.

    The graph is restored to the state captured by ``compile()`` before the
    stage starts, so separate calls cannot form an undeclared transition.

    Args:
        run: The compiled recipe. It decided the forward passes, the eligible
            rows and the trainable set; nothing here chooses any of them.
        stage: The stage to run, by name or compiled.
        batches: Where the rows come from — a `Dataset` where the stage
            declares a sampler, an iterable where it declares
            `ExternalBatches`. An iterable must supply at least `stage.steps`
            batches; running dry is an error rather than a short run, because
            a stage that silently trained for fewer steps than its card says
            is the kind of difference nothing downstream would show.
        seed: Seeds torch before anything stochastic happens, which is what
            makes the loss trace reproducible.
        run_dir: Where to write the plan, the checkpoint and the log. `None`
            runs without writing anything.
        probe: The §6.2 gradient probe. Off by default, and so off in CI.
        selection: Optional periodic validation selection. Training still runs
            for the stage's full declared budget; the returned checkpoint and
            live graph are restored to the lowest-scoring observed state.

    Returns:
        The trace, the per-step log and the stage's checkpoint.
    """
    compiled = _resolve(run, stage)
    if compiled.executor != "gradient":
        if compiled.executor == "meta_gradient":
            raise TrainingError(
                f"stage {compiled.name!r} declares executor='meta_gradient'; "
                "call run_meta_gradient and supply its role and hard-label seeds"
            )
        raise TrainingError(
            f"stage {compiled.name!r} declares executor={compiled.executor!r}; "
            "call run_array_fit or run_cross_fit so the declared executor is "
            "not silently replaced by the gradient loop."
        )
    if compiled.initialise_from is not None:
        raise TrainingError(
            f"stage {compiled.name!r} initialises from "
            f"{compiled.initialise_from!r}. Run it through run_program, which "
            "resolves that earlier immutable checkpoint before executing the "
            "stage; run_stage cannot silently ignore the declared transition."
        )
    if not compiled.objectives:
        raise TrainingError(
            f"stage {compiled.name!r} is action-only. Run it through "
            "run_program so its source checkpoint and artifact edge are "
            "resolved before prediction."
        )
    source, population = _feed(run, compiled, batches, seed=seed)
    _restore_initial_state(
        run,
        parameters=run.initial_parameters(),
        buffers=run.initial_buffers(),
    )
    result = _run_gradient_or_action(
        run,
        compiled,
        source,
        seed=seed,
        run_dir=run_dir,
        probe=probe,
        source_checkpoint=None,
        population=population,
        selection=selection,
    )
    return replace(result, population=population)


def run_meta_gradient(
    run: CompiledRun,
    stage: str | CompiledStage,
    batches: BatchSource | Dataset,
    *,
    seed: int,
    role_seeds: Mapping[str, int],
    hard_label_seed: int,
    run_dir: RunDirectory | None = None,
) -> StageResult:
    """Run one reviewed, atomic one-inner-step/one-outer-step stage.

    Role initialisation and categorical sampling have explicit seeds because
    the MPL card pairs those streams independently of batch/view RNG.  The
    executor owns both graph copies and both optimiser states; neither can be
    aliased by a caller.
    """
    compiled = _resolve(run, stage)
    if compiled.executor != "meta_gradient":
        raise TrainingError(
            f"stage {compiled.name!r} declares executor={compiled.executor!r}, "
            "not 'meta_gradient'"
        )
    meta = compiled.stage.meta_gradient
    if meta is None:  # pragma: no cover - Stage rejects this
        raise TrainingError(f"meta-gradient stage {compiled.name!r} has no contract")
    supplied = dict(role_seeds)
    expected = {role.name for role in compiled.roles}
    if set(supplied) != expected:
        raise TrainingError(
            f"meta-gradient stage {compiled.name!r} needs role seeds for "
            f"{sorted(expected)!r}, got {sorted(supplied)!r}"
        )
    if any(type(value) is not int for value in supplied.values()):
        raise TrainingError("meta-gradient role seeds must be integers")
    if type(hard_label_seed) is not int:
        raise TrainingError("hard_label_seed must be an integer")
    source, population = _feed(run, compiled, batches, seed=seed)
    role_graphs = {
        role.name: _initialised_role_graph(run.graph, supplied[role.name])
        for role in compiled.roles
    }
    parameter_ids = [
        {id(parameter) for parameter in graph.parameters()}
        for graph in role_graphs.values()
    ]
    if len(parameter_ids) != 2 or parameter_ids[0] & parameter_ids[1]:
        raise TrainingError("meta-gradient role graphs share Parameter objects")
    result = _run_meta_gradient_stage(
        run,
        compiled,
        source,
        role_graphs=role_graphs,
        seed=seed,
        hard_label_seed=hard_label_seed,
        population=population,
        run_dir=run_dir,
    )
    return replace(result, population=population)


def _initialised_role_graph(source: ComponentGraph, seed: int) -> ComponentGraph:
    """Clone one declaration and independently replay its declared init."""
    graph = copy.deepcopy(source)
    torch.manual_seed(seed)
    for component in graph.components:
        initialisation = getattr(component, "initialisation", None)
        linear_layers = [
            module for module in component.modules() if isinstance(module, nn.Linear)
        ]
        if not linear_layers:
            continue
        if initialisation == "normal std=0.1/sqrt(fan_in), bias=0":
            for layer in linear_layers:
                nn.init.normal_(
                    layer.weight,
                    mean=0.0,
                    std=0.1 / (layer.in_features**0.5),
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            continue
        if initialisation == "torch Linear default Kaiming-uniform":
            for layer in linear_layers:
                layer.reset_parameters()
            continue
        raise TrainingError(
            f"component {component.name!r} cannot initialise an independent "
            f"parameter role from declared policy {initialisation!r}"
        )
    graph.zero_grad(set_to_none=True)
    return graph


def _run_meta_gradient_stage(
    run: CompiledRun,
    compiled: CompiledStage,
    batches: BatchSource,
    *,
    role_graphs: Mapping[str, ComponentGraph],
    seed: int,
    hard_label_seed: int,
    population: TrainingPopulation | None,
    run_dir: RunDirectory | None,
) -> StageResult:
    if compiled.steps > MAX_STAGE_STEPS:
        raise TrainingError(
            f"stage {compiled.name!r} runs {compiled.steps} steps, beyond the "
            f"{MAX_STAGE_STEPS} collision-free view-key budget"
        )
    meta = compiled.stage.meta_gradient
    assert meta is not None
    feedback = meta.feedback
    if not isinstance(feedback, MetaFeedbackCoefficient):
        raise TrainingError(
            "the shipped meta_gradient executor currently implements the "
            "reviewed MetaFeedbackCoefficient contract only"
        )
    by_role = {role.name: role for role in compiled.roles}
    inner_role = by_role[meta.inner_role]
    outer_role = by_role[meta.outer_role]
    inner_graph = role_graphs[meta.inner_role]
    outer_graph = role_graphs[meta.outer_role]
    inner_objective = _compiled_objective(compiled, meta.inner_objective)
    feedback_objective = _compiled_objective(compiled, meta.feedback_objective)
    score_objective = _compiled_objective(compiled, meta.meta_objective)
    if not isinstance(inner_objective.objective, SampledTeacherTreatmentNLL):
        raise TrainingError("meta-gradient inner objective has the wrong contract")
    if not isinstance(score_objective.objective, MetaPseudoLabelScore):
        raise TrainingError("meta-gradient score objective has the wrong contract")

    records: list[StepRecord] = []
    seen: list[Tensor] = []
    baseline = feedback.new_state()
    sample_generator = torch.Generator(device=next(inner_graph.parameters()).device)
    sample_generator.manual_seed(hard_label_seed)
    source = iter(batches)
    plan_path = run_dir.write_plan(run.plan) if run_dir is not None else None

    from contextlib import ExitStack

    with ExitStack() as stack:
        inner_named = stack.enter_context(
            trainable_only(inner_graph, inner_role.trainable)
        )
        outer_named = stack.enter_context(
            trainable_only(outer_graph, outer_role.trainable)
        )
        inner_parameters = tuple(parameter for _, parameter in inner_named)
        outer_parameters = tuple(parameter for _, parameter in outer_named)
        inner_optimiser = inner_role.optimiser.build(inner_named)
        outer_optimiser = outer_role.optimiser.build(outer_named)
        inner_graph.train()
        outer_graph.train()
        torch.manual_seed(seed)
        for step in range(compiled.steps):
            batch = _next_batch(source, step, compiled)
            run.recipe.schema.validate_batch(batch)
            views = _meta_views(
                run, compiled, batch, rng_key=seed + step, population=population
            )
            pre = _meta_state(
                run, compiled, views, role_graphs, within_step="pre_update"
            )
            ctx = TrainContext(
                global_step=step, schema=run.recipe.schema, stage=compiled.name
            )
            inner_rows = resolve_rows(batch, *inner_objective.rows)
            sampled = inner_objective.objective.sample(
                pre, inner_rows, generator=sample_generator
            )
            inner_term = inner_objective.objective.sampled_loss(
                pre, inner_rows, sampled
            )
            inner_contribution, inner_log = _meta_log(
                inner_objective, inner_term, batch, step
            )
            inner_optimiser.zero_grad(set_to_none=True)
            if inner_contribution.requires_grad:
                inner_contribution.backward()  # type: ignore[no-untyped-call]
            pseudo_gradients = tuple(
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in inner_parameters
            )
            inner_grad_norm = inner_role.optimiser.clipping.apply(inner_parameters)
            inner_lr = _set_learning_rate(
                inner_optimiser, inner_role.optimiser.lr_at(step)
            )
            if inner_contribution.requires_grad:
                inner_optimiser.step()

            post = _meta_state(
                run, compiled, views, role_graphs, within_step="post_update"
            )
            feedback_rows = resolve_rows(batch, *feedback_objective.rows)
            feedback_term = feedback_objective.objective.compute(
                post, batch, feedback_rows, ctx
            )
            labelled_gradients = torch.autograd.grad(
                feedback_term.value,
                inner_parameters,
                allow_unused=True,
            )
            h_raw, h, baseline_value = feedback.compute(
                pseudo_gradients, tuple(labelled_gradients), baseline
            )
            inner_optimiser.zero_grad(set_to_none=True)
            _, feedback_log = _meta_log(feedback_objective, feedback_term, batch, step)

            score_rows = resolve_rows(batch, *score_objective.rows)
            score_term = score_objective.objective.score_loss(pre, score_rows, sampled)
            score_base = apply_reduction(
                score_term, score_objective.reduction, batch_size=batch.batch_size
            ) * score_objective.weight(step)
            score_contribution = h * score_base
            teacher_total = score_contribution
            score_diagnostics = {
                "h_raw": float(h_raw),
                "h": float(h),
                "baseline": baseline_value,
                "sampled_label_accuracy": float(
                    (sampled == batch.t.index_select(0, score_rows)).float().mean()
                ),
                "sampled_label_entropy": _label_entropy(
                    sampled, run.recipe.schema.treatment_cardinality
                ),
            }
            score_log = _objective_log(
                score_objective,
                score_term,
                batch,
                step,
                weighted=score_contribution,
                diagnostics=score_diagnostics,
            )
            outer_logs: list[ObjectiveLog] = []
            for name in meta.outer_objectives:
                objective = _compiled_objective(compiled, name)
                rows = resolve_rows(batch, *objective.rows)
                term = objective.objective.compute(pre, batch, rows, ctx)
                contribution, log = _meta_log(objective, term, batch, step)
                teacher_total = teacher_total + contribution
                outer_logs.append(log)

            outer_optimiser.zero_grad(set_to_none=True)
            if teacher_total.requires_grad:
                teacher_total.backward()  # type: ignore[no-untyped-call]
            outer_grad_norm = outer_role.optimiser.clipping.apply(outer_parameters)
            outer_lr = _set_learning_rate(
                outer_optimiser, outer_role.optimiser.lr_at(step)
            )
            if teacher_total.requires_grad:
                outer_optimiser.step()
            records.append(
                StepRecord(
                    step=step,
                    lr=outer_lr,
                    total=float(teacher_total.detach()),
                    grad_norm=outer_grad_norm,
                    rows=batch.batch_size,
                    terms=(inner_log, feedback_log, score_log, *outer_logs),
                    role_lrs={meta.inner_role: inner_lr, meta.outer_role: outer_lr},
                    role_grad_norms={
                        meta.inner_role: inner_grad_norm,
                        meta.outer_role: outer_grad_norm,
                    },
                )
            )
            seen.append(batch.row_id.detach().clone())

        role_checkpoints = {
            role.name: _emit_role_checkpoint(
                run,
                compiled,
                role_graphs[role.name],
                role.trainable,
                seen,
                seed=seed,
            )
            for role in compiled.roles
        }
    paths: dict[str, str] = {}
    if run_dir is not None and plan_path is not None:
        paths["plan"] = str(plan_path)
        for role, checkpoint in role_checkpoints.items():
            paths[f"checkpoint.{role}"] = str(
                run_dir.write_checkpoint(checkpoint, role=role)
            )
        paths["log"] = str(
            run_dir.write_log(compiled.name, [record.as_json() for record in records])
        )
    return StageResult(
        stage=compiled.name,
        recipe=run.recipe.name,
        seed=seed,
        records=tuple(records),
        _checkpoint=None,
        role_checkpoints=role_checkpoints,
        role_graphs=role_graphs,
        objective_states={feedback.name: baseline},
        paths=paths,
    )


def _compiled_objective(compiled: CompiledStage, name: str) -> CompiledObjective:
    for objective in compiled.objectives:
        if objective.name == name:
            return objective
    raise TrainingError(f"meta-gradient stage {compiled.name!r} has no {name!r}")


def _label_entropy(labels: Tensor, classes: int) -> float:
    counts = torch.bincount(labels, minlength=classes).to(dtype=torch.float32)
    probabilities = counts / counts.sum()
    positive = probabilities > 0
    return float(-(probabilities[positive] * probabilities[positive].log()).sum())


def _meta_views(
    run: CompiledRun,
    compiled: CompiledStage,
    batch: XTYBatch,
    *,
    rng_key: int,
    population: TrainingPopulation | None,
) -> Mapping[tuple[str, int], XTYBatch]:
    views: dict[tuple[str, int], XTYBatch] = {("identity", 0): batch}
    for forward in compiled.passes:
        key = (forward.realisation.view, forward.realisation.draw)
        if key in views:
            continue
        views[key] = run.recipe.view(key[0]).apply(
            batch,
            run.recipe.schema,
            rng_key=rng_key,
            draw=key[1],
            population=population,
        )
    return views


def _meta_state(
    run: CompiledRun,
    compiled: CompiledStage,
    views: Mapping[tuple[str, int], XTYBatch],
    role_graphs: Mapping[str, ComponentGraph],
    *,
    within_step: str,
) -> State:
    values = {}
    for forward in compiled.passes:
        realisation = forward.realisation
        if realisation.state != within_step:
            continue
        graph = role_graphs[realisation.role]
        viewed = views[(realisation.view, realisation.draw)]
        values[realisation] = graph.evaluate(
            viewed, schema=run.recipe.schema, only=forward.components
        )
    return State(values)


def _meta_log(
    objective: CompiledObjective,
    term: LossTerm,
    batch: XTYBatch,
    step: int,
) -> tuple[Tensor, ObjectiveLog]:
    contribution = apply_reduction(
        term, objective.reduction, batch_size=batch.batch_size
    ) * objective.weight(step)
    return contribution, _objective_log(
        objective, term, batch, step, weighted=contribution
    )


def _objective_log(
    objective: CompiledObjective,
    term: LossTerm,
    batch: XTYBatch,
    step: int,
    *,
    weighted: Tensor,
    diagnostics: Mapping[str, float] | None = None,
) -> ObjectiveLog:
    merged = dict(term.diagnostics)
    merged.update(diagnostics or {})
    return ObjectiveLog(
        name=objective.name,
        rows=objective.rows,
        reduction=objective.reduction,
        value=float(term.value.detach()),
        weight=objective.weight(step),
        weighted=float(weighted.detach()),
        n=term.n,
        coverage=term.n / batch.batch_size,
        diagnostics=merged,
    )


def _emit_role_checkpoint(
    run: CompiledRun,
    stage: CompiledStage,
    graph: ComponentGraph,
    components: Sequence[str],
    row_ids: Sequence[Tensor],
    *,
    seed: int,
) -> Checkpoint:
    seen = torch.cat([rows.reshape(-1) for rows in row_ids])
    return Checkpoint._issue(
        recipe=run.recipe.name,
        stage=stage.name,
        fold=None,
        trained_on_row_ids=torch.unique(seen.cpu()),
        parameters=dict(_component_parameters(graph, components)),
        buffers=dict(_component_buffers(graph, components)),
        components=tuple(components),
        steps=stage.steps,
        seed=seed,
        plan_digest=run.plan.digest,
    )


def run_array_fit(
    run: CompiledRun,
    stage: str | CompiledStage,
    batches: BatchSource | Dataset,
    *,
    seed: int,
    run_dir: RunDirectory | None = None,
) -> StageResult:
    """Execute one explicit functional ``array_fit`` stage."""
    compiled = _resolve(run, stage)
    if compiled.executor != "array_fit":
        raise TrainingError(
            f"stage {compiled.name!r} declares executor={compiled.executor!r}, "
            "not 'array_fit'"
        )
    _restore_initial_state(
        run,
        parameters=run.initial_parameters(),
        buffers=run.initial_buffers(),
    )
    source, population = _feed(run, compiled, batches, seed=seed)
    return replace(
        _run_array_fit(
            run,
            compiled,
            _materialise_batches(source, compiled),
            seed=seed,
            run_dir=run_dir,
        ),
        population=population,
    )


def run_cross_fit(
    run: CompiledRun,
    stage: str | CompiledStage,
    batches: BatchSource | Dataset,
    *,
    seed: int,
    run_dir: RunDirectory | None = None,
    probe: GradientProbe | None = None,
) -> StageResult:
    """Fit every actual fold and emit held-out treatment pseudo labels."""
    compiled = _resolve(run, stage)
    if compiled.executor != "cross_fit":
        raise TrainingError(
            f"stage {compiled.name!r} declares executor={compiled.executor!r}, "
            "not 'cross_fit'"
        )
    if compiled.initialise_from is not None:
        raise TrainingError(
            f"stage {compiled.name!r} initialises from "
            f"{compiled.initialise_from!r}. Run it through run_program so the "
            "named checkpoint can be restored independently for every fold."
        )
    source, population = _feed(run, compiled, batches, seed=seed)
    return replace(
        _run_cross_fit(
            run,
            compiled,
            _materialise_batches(source, compiled),
            seed=seed,
            run_dir=run_dir,
            probe=probe,
            initialise_from=None,
        ),
        population=population,
    )


def run_program(
    run: CompiledRun,
    batches: BatchSources,
    *,
    seed: int,
    run_dir: RunDirectory | None = None,
    probes: Mapping[str, GradientProbe] | None = None,
) -> ProgramResult:
    """Execute every compiled stage in order.

    Each stage starts from a fresh copy of the recipe's initial graph state.
    When ``initialise_from`` is set, the named earlier checkpoint is overlaid
    on that state. This makes stage transitions data rather than an implicit
    consequence of whichever stage happened to mutate the shared graph last.

    The stage seed is ``seed + stage_index * STREAM_STRIDE``. It is recorded in
    each checkpoint and gives distinct, deterministic stochastic streams
    without adding another paper-governed recipe field. The stride is what
    makes them distinct: a stage walks one view key per optimiser step from
    its own seed upwards, so consecutively numbered stage seeds would hand
    stage ``i`` step 1 the key stage ``i + 1`` already used at step 0.
    """
    sources = dict(batches)
    expected = tuple(stage.name for stage in run.stages)
    missing = [name for name in expected if name not in sources]
    extra = sorted(set(sources) - set(expected))
    if missing or extra:
        raise TrainingError(
            f"program {run.recipe.name!r} needs one batch source per stage; "
            f"missing {missing!r}, unexpected {extra!r}, stages {list(expected)!r}"
        )
    probe_map = dict(probes or {})
    unknown_probes = sorted(set(probe_map) - set(expected))
    if unknown_probes:
        raise TrainingError(
            f"gradient probes name unknown stages {unknown_probes!r}; this "
            f"program has {list(expected)!r}"
        )

    # The baseline belongs to compilation, not to this function call. A
    # CompiledRun may be executed more than once; snapshotting its already-
    # trained live graph here would make a second execution a different
    # program despite an identical plan and seed.
    initial_parameters = run.initial_parameters()
    initial_buffers = run.initial_buffers()
    results: list[StageResult] = []
    by_name: dict[str, StageResult] = {}
    if run_dir is not None:
        run_dir.write_plan(run.plan)

    for index, compiled in enumerate(run.stages):
        _restore_initial_state(
            run,
            parameters=initial_parameters,
            buffers=initial_buffers,
        )
        source_checkpoint: Checkpoint | None = None
        if compiled.initialise_from is not None:
            source_checkpoint = by_name[compiled.initialise_from].checkpoint
            _restore_checkpoint(run, source_checkpoint)

        inputs: list[PseudoLabels] = []
        for input_name in compiled.inputs:
            labels = by_name[input_name].pseudo_labels
            if labels is None:
                raise TrainingError(
                    f"stage {compiled.name!r} consumes {input_name!r}, but that "
                    "stage emitted no PseudoLabels"
                )
            try:
                labels.validate_for(run)
            except ArtifactError as error:
                raise TrainingError(
                    f"stage {compiled.name!r} could not load pseudo-label input "
                    f"{input_name!r}: {error}"
                ) from error
            inputs.append(labels)

        stage_seed = seed + index * STREAM_STRIDE
        drawn, population = _feed(
            run, compiled, sources[compiled.name], seed=stage_seed
        )
        # Artifact joins wrap the resolved stream rather than the caller's
        # source: a pseudo-label side table replaces the treatment placeholder
        # of rows the policy has already made missing, which is the order
        # §7.1 describes and the only one under which the join is meaningful.
        stage_batches: BatchSource = _apply_inputs(
            drawn,
            inputs,
            treatment_cardinality=run.recipe.schema.treatment_cardinality,
        )
        if compiled.executor == "gradient":
            result = _run_gradient_or_action(
                run,
                compiled,
                stage_batches,
                seed=stage_seed,
                run_dir=run_dir,
                probe=probe_map.get(compiled.name),
                source_checkpoint=source_checkpoint,
                population=population,
            )
        elif compiled.executor == "array_fit":
            result = _run_array_fit(
                run,
                compiled,
                _materialise_batches(stage_batches, compiled),
                seed=stage_seed,
                run_dir=run_dir,
            )
        elif compiled.executor == "cross_fit":
            result = _run_cross_fit(
                run,
                compiled,
                _materialise_batches(stage_batches, compiled),
                seed=stage_seed,
                run_dir=run_dir,
                probe=probe_map.get(compiled.name),
                initialise_from=source_checkpoint,
            )
        else:
            raise TrainingError(
                f"program stage {compiled.name!r} uses meta_gradient; execute "
                "it with run_meta_gradient so its role and hard-label seeds "
                "are supplied explicitly"
            )
        result = replace(result, population=population)
        results.append(result)
        by_name[compiled.name] = result

    return ProgramResult(recipe=run.recipe.name, seed=seed, stages=tuple(results))


def _run_gradient_or_action(
    run: CompiledRun,
    compiled: CompiledStage,
    batches: BatchSource,
    *,
    seed: int,
    run_dir: RunDirectory | None,
    probe: GradientProbe | None,
    source_checkpoint: Checkpoint | None,
    population: TrainingPopulation | None = None,
    selection: MinimumValidationSelection | None = None,
) -> StageResult:
    """Execute a normal fit, optionally followed by a pseudo-label action."""
    materialised: list[XTYBatch] | None = None
    fit_batches: BatchSource = batches
    if isinstance(compiled.action, PseudoLabelAction):
        if compiled.objectives:
            materialised = []
            fit_batches = _capture_step_batches(batches, compiled, materialised)
        else:
            materialised = _materialise_batches(batches, compiled)
    if compiled.objectives:
        result = _run_stage(
            run,
            compiled,
            fit_batches,
            seed=seed,
            run_dir=run_dir,
            probe=probe,
            population=population,
            selection=selection,
        )
        if materialised is None:
            return result
        labels = _predict_pseudo_labels(
            run,
            compiled,
            materialised,
            checkpoints={0: result.checkpoint},
            fold=0,
            seed=seed,
            teacher=result.teacher,
        )
        paths = dict(result.paths)
        if run_dir is not None:
            paths["pseudo_labels"] = str(run_dir.write_pseudo_labels(labels))
        return replace(result, pseudo_labels=labels, paths=paths)

    if not isinstance(compiled.action, PseudoLabelAction):
        raise TrainingError(
            f"gradient stage {compiled.name!r} has neither objectives nor a "
            "PseudoLabelAction"
        )
    if source_checkpoint is None:
        raise TrainingError(
            f"action-only stage {compiled.name!r} needs initialise_from so its "
            "predictions name the checkpoint that produced them"
        )
    assert materialised is not None
    teacher = (
        EMATeacher(run.graph, compiled.teacher)
        if compiled.teacher is not None
        else None
    )
    labels = _predict_pseudo_labels(
        run,
        compiled,
        materialised,
        checkpoints={0: source_checkpoint},
        fold=0,
        seed=seed,
        teacher=teacher,
    )
    action_paths: dict[str, str] = {}
    if run_dir is not None:
        action_paths["plan"] = str(run_dir.write_plan(run.plan))
        action_paths["pseudo_labels"] = str(run_dir.write_pseudo_labels(labels))
    return StageResult(
        stage=compiled.name,
        recipe=run.recipe.name,
        seed=seed,
        records=(),
        _checkpoint=None,
        pseudo_labels=labels,
        teacher=teacher,
        paths=action_paths,
    )


def _run_array_fit(
    run: CompiledRun,
    compiled: CompiledStage,
    batches: Sequence[XTYBatch],
    *,
    seed: int,
    run_dir: RunDirectory | None,
) -> StageResult:
    action = compiled.action
    if not isinstance(action, ArrayFitAction) or isinstance(action, PseudoLabelAction):
        raise TrainingError(f"array-fit stage {compiled.name!r} has no ArrayFitAction")
    for candidate in batches:
        run.recipe.schema.validate_batch(candidate)
    batch = _combine_batches(batches)
    rows = resolve_rows(batch, *(compiled.action_rows or ("all",)))
    if rows.numel() == 0:
        raise TrainingError(
            f"array-fit stage {compiled.name!r} has zero eligible rows; an "
            "estimator fit on an empty array is not a valid stage"
        )
    before = batch.clone()
    torch.manual_seed(seed)
    state = action.fit(batch, rows, seed=seed)
    if not batch.equal_to(before):
        raise TrainingError(
            f"array-fit action {action.name!r} mutated its input batch. Stage "
            "actions return state and never write into the source dataset "
            "(DESIGN.md §7.1)."
        )
    checkpoint = _emit_array_checkpoint(
        run,
        compiled,
        action,
        state,
        batch.row_id[rows],
        seed=seed,
    )
    paths: dict[str, str] = {}
    if run_dir is not None:
        paths["plan"] = str(run_dir.write_plan(run.plan))
        paths["checkpoint"] = str(run_dir.write_checkpoint(checkpoint))
    return StageResult(
        stage=compiled.name,
        recipe=run.recipe.name,
        seed=seed,
        records=(),
        _checkpoint=checkpoint,
        paths=paths,
    )


def _run_cross_fit(
    run: CompiledRun,
    compiled: CompiledStage,
    batches: Sequence[XTYBatch],
    *,
    seed: int,
    run_dir: RunDirectory | None,
    probe: GradientProbe | None,
    initialise_from: Checkpoint | None,
) -> StageResult:
    action = compiled.action
    if not isinstance(action, PseudoLabelAction):
        raise TrainingError(
            f"cross-fit stage {compiled.name!r} has no PseudoLabelAction"
        )
    for candidate in batches:
        run.recipe.schema.validate_batch(candidate)
    data = _combine_batches(batches)
    if data.fold_id is None:
        raise TrainingError(
            f"cross-fit stage {compiled.name!r} needs batch.fold_id on every row"
        )
    folds = sorted(set(data.fold_id.tolist()))
    if len(folds) < 2:
        raise TrainingError(
            f"cross-fit stage {compiled.name!r} needs at least two folds, got {folds!r}"
        )

    checkpoints: dict[int, Checkpoint] = {}
    row_ids: list[Tensor] = []
    predicted_by_fold: list[Tensor] = []
    predictions: list[Tensor] = []
    records: list[StepRecord] = []
    paths: dict[str, str] = {}
    for offset, fold in enumerate(folds):
        assert data.fold_id is not None
        train_rows = torch.nonzero(data.fold_id != fold, as_tuple=False).flatten()
        predict_rows = torch.nonzero(data.fold_id == fold, as_tuple=False).flatten()
        if train_rows.numel() == 0 or predict_rows.numel() == 0:
            raise TrainingError(
                f"fold {fold} of stage {compiled.name!r} has "
                f"{train_rows.numel()} train rows and {predict_rows.numel()} "
                "prediction rows; both must be non-empty"
            )
        train_batch = data.index_select(train_rows)
        predict_batch = data.index_select(predict_rows)
        _restore_initial_state(
            run,
            parameters=run.initial_parameters(),
            buffers=run.initial_buffers(),
        )
        if initialise_from is not None:
            _restore_checkpoint(run, initialise_from)
        fold_seed = seed + offset * STREAM_STRIDE
        fold_result = _run_stage(
            run,
            compiled,
            [train_batch] * compiled.steps,
            seed=fold_seed,
            run_dir=run_dir,
            probe=probe,
            fold=fold,
        )
        checkpoint = fold_result.checkpoint
        checkpoints[fold] = checkpoint
        records.extend(fold_result.records)
        for key, value in fold_result.paths.items():
            paths[f"fold.{fold}.{key}"] = value
        batch_rows, batch_folds, batch_predictions = _pseudo_predictions(
            run,
            compiled,
            [predict_batch],
            fold=fold,
            seed=fold_seed,
            teacher=fold_result.teacher,
        )
        row_ids.extend(batch_rows)
        predicted_by_fold.extend(batch_folds)
        predictions.extend(batch_predictions)

    labels = _emit_pseudo_labels(
        run,
        compiled,
        checkpoints,
        row_ids,
        predicted_by_fold,
        predictions,
    )
    try:
        labels.validate_for(run)
    except ArtifactError as error:
        raise TrainingError(
            f"cross-fit stage {compiled.name!r} produced invalid provenance: {error}"
        ) from error
    if run_dir is not None:
        paths["plan"] = str(run_dir.write_plan(run.plan))
        paths["pseudo_labels"] = str(run_dir.write_pseudo_labels(labels))
    return StageResult(
        stage=compiled.name,
        recipe=run.recipe.name,
        seed=seed,
        records=tuple(records),
        _checkpoint=None,
        fold_checkpoints=MappingProxyType(dict(sorted(checkpoints.items()))),
        pseudo_labels=labels,
        paths=paths,
    )


def _predict_pseudo_labels(
    run: CompiledRun,
    compiled: CompiledStage,
    batches: Sequence[XTYBatch],
    *,
    checkpoints: Mapping[int, Checkpoint],
    fold: int,
    seed: int,
    teacher: EMATeacher | None,
) -> PseudoLabels:
    row_ids, folds, labels = _pseudo_predictions(
        run,
        compiled,
        batches,
        fold=fold,
        seed=seed,
        teacher=teacher,
    )
    artifact = _emit_pseudo_labels(
        run,
        compiled,
        checkpoints,
        row_ids,
        folds,
        labels,
    )
    try:
        artifact.validate_for(run)
    except ArtifactError as error:
        raise TrainingError(
            f"stage {compiled.name!r} produced invalid pseudo-label provenance: {error}"
        ) from error
    return artifact


def _pseudo_predictions(
    run: CompiledRun,
    compiled: CompiledStage,
    batches: Sequence[XTYBatch],
    *,
    fold: int,
    seed: int,
    teacher: EMATeacher | None,
) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
    action = compiled.action
    if not isinstance(action, PseudoLabelAction):
        raise TrainingError(
            f"stage {compiled.name!r} has no PseudoLabelAction to evaluate"
        )
    row_ids: list[Tensor] = []
    folds: list[Tensor] = []
    labels: list[Tensor] = []
    torch.manual_seed(seed)
    with torch.no_grad():
        for batch_index, batch in enumerate(batches):
            state = run.state(
                compiled,
                batch,
                # The held-out prediction passes continue this stage or fold's
                # own key run rather than starting a second scheme: training
                # consumed `seed .. seed + steps - 1`, so prediction takes the
                # keys directly above. Multiplying the seed instead put these
                # keys back among the training keys of a low-seeded run.
                rng_key=seed + compiled.steps + batch_index,
                teacher_graph=teacher.graph if teacher is not None else None,
            )
            rows = resolve_rows(batch, *(compiled.action_rows or ("all",)))
            distribution = treatment_distribution(
                state,
                action.port,
                action.realisation,
                objective=f"pseudo-label action {compiled.name}",
            )
            predicted = distribution.probs.argmax(dim=1)
            row_ids.append(batch.row_id[rows].detach().cpu())
            folds.append(torch.full((rows.numel(),), fold, dtype=torch.long))
            labels.append(predicted[rows].detach().cpu())
    return row_ids, folds, labels


def _apply_inputs(
    batches: BatchSource,
    inputs: Sequence[PseudoLabels],
    *,
    treatment_cardinality: int,
) -> Iterator[XTYBatch]:
    """Functionally join every declared side table as batches are loaded."""
    for batch in batches:
        if not isinstance(batch, XTYBatch):
            raise TrainingError(
                f"a stage batch source yielded {type(batch)}, expected XTYBatch"
            )
        loaded = batch
        for labels in inputs:
            try:
                loaded = labels.apply_to(
                    loaded,
                    treatment_cardinality=treatment_cardinality,
                )
            except ArtifactError as error:
                raise TrainingError(
                    f"could not join pseudo labels from {labels.source_stage!r}: "
                    f"{error}"
                ) from error
        yield loaded


def _materialise_batches(
    batches: BatchSource, compiled: CompiledStage
) -> list[XTYBatch]:
    """Collect a source whose executor contract explicitly requires finiteness.

    Array-fit, cross-fit and action-only prediction operate on one finite
    population. Gradient fitting is different: its general ``BatchSource``
    may be cycling or unbounded, so a fitting gradient action captures batches
    as the seeded optimiser loop consumes them instead.
    """
    materialised = list(batches)
    if not materialised:
        raise TrainingError(f"stage {compiled.name!r} received no batches")
    for index, batch in enumerate(materialised):
        if not isinstance(batch, XTYBatch):
            raise TrainingError(
                f"stage {compiled.name!r} batch {index} is {type(batch)}, "
                "expected XTYBatch"
            )
    return materialised


def _capture_step_batches(
    batches: BatchSource,
    compiled: CompiledStage,
    captured: list[XTYBatch],
) -> Iterator[XTYBatch]:
    """Yield exactly ``steps`` batches and retain immutable prediction copies.

    This wrapper is consumed inside ``_run_stage`` after its torch seed is set,
    preserving the ordinary gradient executor's stochastic batch-source
    semantics. It never asks an unbounded source for a batch beyond the fit.
    """
    source = iter(batches)
    for step in range(compiled.steps):
        batch = _next_batch(source, step, compiled)
        captured.append(batch.clone())
        yield batch


def _combine_batches(batches: Sequence[XTYBatch]) -> XTYBatch:
    """Concatenate one finite dataset through the batch structural contract."""
    if not batches:
        raise TrainingError("cannot combine an empty batch sequence")
    return XTYBatch.cat(batches)


def _objective_states(
    compiled: CompiledStage, population: TrainingPopulation | None
) -> Mapping[str, object]:
    """One fresh state per stateful objective of this stage (`core/loss.py`).

    Empty for every stage whose objectives are all stateless, which is every
    stage but `flexmatch`'s today — so no existing run gains a code path, and
    no plan, digest or recorded result moves.

    `initial_state` returning `None` is refused here rather than at the read.
    An objective that declares the protocol and then hands back nothing would
    otherwise fail inside `compute`, at the first step, with an error about the
    lookup instead of about the declaration.
    """
    states: dict[str, object] = {}
    for objective in compiled.objectives:
        loss = objective.objective
        if not isinstance(loss, StatefulObjective):
            continue
        state = loss.initial_state(population)
        if state is None:
            raise TrainingError(
                f"objective {objective.name!r} declares StatefulObjective and "
                "its initial_state returned None. A stateful objective returns "
                "the state it will read back through TrainContext "
                "(DESIGN.md §4)."
            )
        states[objective.name] = state
    return states


def _run_stage(
    run: CompiledRun,
    compiled: CompiledStage,
    batches: BatchSource,
    *,
    seed: int,
    run_dir: RunDirectory | None,
    probe: GradientProbe | None,
    fold: int | None = None,
    population: TrainingPopulation | None = None,
    selection: MinimumValidationSelection | None = None,
) -> StageResult:
    """Execute one already-resolved stage for `run_stage` or `run_program`."""
    if compiled.steps > MAX_STAGE_STEPS:
        raise TrainingError(
            f"stage {compiled.name!r} runs {compiled.steps} steps, more than "
            f"the {MAX_STAGE_STEPS} its share of the {STREAM_STRIDE}-wide gap "
            "between sibling stage and fold seeds allows; its view keys would "
            "collide with the next stage's. Widen STREAM_STRIDE before "
            "running a stage this long."
        )
    spec = compiled.optimiser
    graph = run.graph
    if selection is not None and selection.result is not None:
        raise TrainingError(
            "a MinimumValidationSelection instance belongs to one stage run; "
            "construct a fresh selector for each execution"
        )
    if selection is not None and compiled.teacher is not None:
        raise TrainingError(
            "validation selection of an EMA-teacher stage is ambiguous; the "
            "selector currently restores only the student graph"
        )
    torch.manual_seed(seed)
    graph.zero_grad(set_to_none=True)

    plan_path = run_dir.write_plan(run.plan) if run_dir is not None else None

    records: list[StepRecord] = []
    seen: list[Tensor] = []
    with trainable_only(graph, compiled.trainable) as parameters:
        teacher = (
            EMATeacher(graph, compiled.teacher)
            if compiled.teacher is not None
            else None
        )
        optimiser = spec.build(parameters)
        tensors = [parameter for _, parameter in parameters]
        mixer = LossMixer.for_stage(compiled, probe=probe)
        # Built here and nowhere else, which is what makes it per *execution*
        # rather than per recipe: a stage run twice gets two fresh states, and
        # a paired ablation whose arms share an objective instance cannot leak
        # one arm's history into the other (`core/loss.py`, StatefulObjective).
        objective_states = _objective_states(compiled, population)
        source = iter(batches)
        modes = {module: module.training for module in graph.modules()}
        graph.train()
        # Freezing by component name includes stateful buffers: a frozen
        # BatchNorm encoder must not change its running statistics while a
        # downstream head trains. All modes are restored below.
        for name in graph.names:
            if name not in compiled.trainable:
                graph[name].eval()
        try:
            for step in range(compiled.steps):
                batch = _next_batch(source, step, compiled)
                lr = _set_learning_rate(optimiser, spec.lr_at(step))
                mixed = _step(
                    run,
                    compiled,
                    batch,
                    mixer,
                    optimiser,
                    tensors,
                    step,
                    # The caller owns the benchmark's random streams. A stage
                    # seed is therefore the first view key and each optimiser
                    # step advances it once. Mean Teacher card §6 pins exactly
                    # `s_r + 10000 + step`; multiplying the supplied seed here
                    # silently ran a different reviewed protocol.
                    rng_key=seed + step,
                    teacher=teacher,
                    population=population,
                    objective_states=objective_states,
                )
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
                if selection is not None:
                    selection.consider(
                        run,
                        step + 1,
                        final=step + 1 == compiled.steps,
                    )
        finally:
            # Restored module by module. `graph.train(flag)` is recursive, so
            # one saved root flag would silently put a submodule a caller had
            # placed in eval mode back into training.
            for module, was in modes.items():
                module.training = was
        selected = selection.restore(run) if selection is not None else None
        checkpoint_steps = selected.step if selected is not None else len(records)
        checkpoint = _emit_checkpoint(
            run,
            compiled,
            parameters,
            seen[:checkpoint_steps],
            steps=checkpoint_steps,
            seed=seed,
            fold=fold,
        )

    paths: dict[str, str] = {}
    if run_dir is not None and plan_path is not None:
        paths["plan"] = str(plan_path)
        paths["checkpoint"] = str(run_dir.write_checkpoint(checkpoint))
        paths["log"] = str(
            run_dir.write_log(
                compiled.name,
                [r.as_json() for r in records],
                fold=fold,
            )
        )
    return StageResult(
        stage=compiled.name,
        recipe=run.recipe.name,
        seed=seed,
        records=tuple(records),
        _checkpoint=checkpoint,
        teacher=teacher,
        objective_states=objective_states,
        selection=selected,
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
    *,
    rng_key: int,
    teacher: EMATeacher | None,
    population: TrainingPopulation | None = None,
    objective_states: Mapping[str, object] | None = None,
) -> _Stepped:
    """Forward, mix, backward, clip, step — in that order and no other."""
    state = run.state(
        compiled,
        batch,
        rng_key=rng_key,
        teacher_graph=teacher.graph if teacher is not None else None,
        population=population,
    )
    ctx = TrainContext(
        global_step=step,
        schema=run.recipe.schema,
        stage=compiled.name,
        objective_states=objective_states or {},
    )
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
    if teacher is not None:
        # Standard Mean Teacher order: predict with the previous teacher,
        # update the student, then move the teacher towards the new student.
        teacher.update(run.graph, step)
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


def _restore_checkpoint(run: CompiledRun, checkpoint: Checkpoint) -> None:
    """Overlay one verified earlier-stage checkpoint on the reset graph."""
    if checkpoint.recipe != run.recipe.name:
        raise TrainingError(
            f"checkpoint {checkpoint.stage!r} belongs to recipe "
            f"{checkpoint.recipe!r}, not {run.recipe.name!r}"
        )
    if checkpoint.plan_digest != run.plan.digest:
        raise TrainingError(
            f"checkpoint {checkpoint.stage!r} was produced under plan "
            f"{checkpoint.plan_digest[:12]}, not this run's "
            f"{run.plan.digest[:12]}. Stage initialisation may only consume an "
            "artifact from the exact compiled program."
        )
    source = run.stage(checkpoint.stage)
    if checkpoint.components != source.trainable:
        raise TrainingError(
            f"checkpoint {checkpoint.stage!r} carries components "
            f"{checkpoint.components!r}, but that compiled stage trains "
            f"{source.trainable!r}"
        )
    if checkpoint.steps != source.steps:
        raise TrainingError(
            f"checkpoint {checkpoint.stage!r} records {checkpoint.steps} "
            f"steps, but its compiled stage declares {source.steps}"
        )
    _copy_named(
        _component_parameters(run.graph, checkpoint.components),
        checkpoint.parameters,
        what=f"checkpoint {checkpoint.stage!r} parameters",
    )
    _copy_named(
        _component_buffers(run.graph, checkpoint.components),
        checkpoint.buffers,
        what=f"checkpoint {checkpoint.stage!r} buffers",
    )


def _restore_initial_state(
    run: CompiledRun,
    *,
    parameters: Mapping[str, Tensor],
    buffers: Mapping[str, Tensor],
) -> None:
    """Restore the graph snapshot captured by ``compile()``."""
    _copy_named(
        run.graph.named_parameters(),
        parameters,
        what="the recipe's initial parameters",
    )
    _copy_named(
        run.graph.named_buffers(),
        buffers,
        what="the recipe's initial buffers",
    )
    run.graph.zero_grad(set_to_none=True)


def _component_parameters(
    graph: ComponentGraph, components: Sequence[str]
) -> tuple[tuple[str, Tensor], ...]:
    return tuple(
        (f"{component_name}.{name}", parameter)
        for component_name in components
        for name, parameter in graph[component_name].named_parameters()
    )


def _component_buffers(
    graph: ComponentGraph, components: Sequence[str]
) -> tuple[tuple[str, Tensor], ...]:
    return tuple(
        (f"{component_name}.{name}", buffer)
        for component_name in components
        for name, buffer in graph[component_name].named_buffers()
    )


@torch.no_grad()
def _copy_named(
    named: Iterable[tuple[str, Tensor]],
    saved: Mapping[str, Tensor],
    *,
    what: str,
) -> None:
    """Copy a complete named tensor mapping, rejecting partial restoration."""
    targets = dict(named)
    missing = sorted(set(targets) - set(saved))
    extra = sorted(set(saved) - set(targets))
    if missing or extra:
        raise TrainingError(
            f"{what} do not match the component graph; missing {missing!r}, "
            f"unexpected {extra!r}"
        )
    for name, target in targets.items():
        source = saved[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise TrainingError(
                f"{what} tensor {name!r} has shape {tuple(source.shape)} and "
                f"dtype {source.dtype}; the live graph expects "
                f"{tuple(target.shape)} and {target.dtype}"
            )
        target.copy_(source.to(device=target.device))


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
    "MAX_STAGE_STEPS",
    "STREAM_STRIDE",
    "BatchSource",
    "BatchSources",
    "ProgramResult",
    "StageResult",
    "StepRecord",
    "run_array_fit",
    "run_cross_fit",
    "run_meta_gradient",
    "run_program",
    "run_stage",
    "trainable_only",
]
