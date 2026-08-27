"""The declarative surface `compile()` consumes (`DESIGN.md` §7, §9).

A recipe is an assembly of registered components, objectives and views plus
explicit hyperparameters, and **it contains no logic**. That rule is only
enforceable if the thing a recipe assembles is data, so the three types here
are deliberately inert: they validate their own shape and hold no behaviour.

`Objective` is a structural protocol rather than a base class, so a loss is an
ordinary object that happens to satisfy four members and the compiler can check
one it has not been asked to run. `Weighted` is what a stage actually holds
(§6, §7): an objective together with the two things the *paper* governs about
its use — the weight schedule and the reduction — neither of which has a
default. `Stage` carries the fields the compiler checks, together with the two
the gradient executor needs — the optimiser and the step count, both of them
card-bound and so both `REQUIRED`. `Program` is the ordered, immutable stage
sequence; `initialise_from` may point only backwards through it. `TeacherSpec`
makes every paper-governed EMA choice explicit before a teacher realisation can
be planned. P10 adds explicit executors, pseudo-label actions, artifact inputs
and the narrow functional array-fit contract. Those declarations make the
causal leakage rule program data rather than runtime convention.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    get_args,
    overload,
    runtime_checkable,
)

from torch import Tensor

from xty2.core.card_keys import REQUIRED, card_hyperparameters, is_required
from xty2.core.data import (
    SAMPLERS,
    DataSpec,
    ExternalBatches,
    SamplerSpec,
    draws_from_population,
)
from xty2.core.errors import CompileError, Xty2Error, require_str
from xty2.core.graph import ComponentGraph, Realisation, State
from xty2.core.loss import (
    LossTerm,
    Reduction,
    TrainContext,
    validate_reduction,
)
from xty2.core.optimisation import OptimiserSpec
from xty2.core.ports import Port
from xty2.core.rows import (
    RowIndex,
    Rows,
    populations_are_disjoint,
    validate_population,
)
from xty2.core.schedules import Constant, Schedule, as_schedule
from xty2.core.schema import Schema
from xty2.core.views import ViewSpec

if TYPE_CHECKING:  # pragma: no cover - the batch is only named in a signature
    from xty2.core.batch import XTYBatch

Purpose = Literal["causal", "predictive"]
"""What the recipe is for. `predictive` is what may opt out of the leakage
rule (`DESIGN.md` §7.2); `causal` may not."""

Executor = Literal["gradient", "array_fit", "cross_fit"]
"""The three deliberately explicit stage executors (`DESIGN.md` §7)."""


@runtime_checkable
class ArrayFitAction(Protocol):
    """A functional array-based fit executed outside autograd (P10).

    The action receives one immutable ``XTYBatch`` plus the stage's resolved
    rows and returns tensor state for an immutable checkpoint. It does not
    mutate the recipe, graph or batch. Returning tensors keeps checkpoint state
    portable instead of pickling an opaque estimator object.

    P11 supplies the first real implementation (SSDML). P10 defines and tests
    this executor seam with a minimal double; the runtime never infers it from
    the accidental presence of a ``fit`` method.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in the plan and checkpoint state names."""

    def fit(
        self,
        batch: XTYBatch,
        rows: RowIndex,
        *,
        seed: int,
    ) -> Mapping[str, Tensor]:
        """Fit on ``rows`` and return a complete tensor state mapping."""


@dataclass(frozen=True)
class PseudoLabelAction:
    """Emit hard treatment labels from one distribution port.

    The source is a port, never a caller-set ``used_y`` flag. Outcome
    dependence is therefore derived from the producing subgraph by the
    compiler and executor (`DESIGN.md` §2.2, §7.2).

    This emits argmax labels only. A confidence gate exists on the *objective*
    path — `PseudoLabelTreatmentNLL` masks per row — and deliberately did not
    follow onto this one: `cycle_dual`, the only staged consumer, states no gate
    in either of its papers and marks `losses.confidence_threshold` as `n/a`
    (`cycle_dual.md` §5.6). So a gate here would drop no paper mechanic, which
    is what `DESIGN.md` §11.2 Q1 asks, and §11.4 carries it as `staged-gate`
    with nothing paying for it. A reviewed card whose §4 names a threshold on a
    staged writeback is what changes that.
    """

    port: Port
    rows: Rows = "t_missing"
    realisation: Realisation = field(default_factory=Realisation)

    def __post_init__(self) -> None:
        if self.port not in (Port.T_GIVEN_X, Port.T_GIVEN_XY):
            raise CompileError(
                "PseudoLabelAction emits treatment labels from T_GIVEN_X or "
                f"T_GIVEN_XY, got {self.port!r}. The source port is what makes "
                "outcome dependence derivable (DESIGN.md §7.2)."
            )
        validate_rows(self.rows, "PseudoLabelAction")
        if not isinstance(self.realisation, Realisation):
            raise CompileError(
                "PseudoLabelAction.realisation must be a Realisation, got "
                f"{type(self.realisation)}"
            )

    @property
    def name(self) -> str:
        return "pseudo_labels"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.realisation)})

    def describe(self) -> str:
        """One stable execution-plan line."""
        return f"hard argmax of {self.port} @ {self.realisation}, rows {self.rows}"


@runtime_checkable
class Objective(Protocol):
    """What the compiler reads from a loss (`DESIGN.md` §4).

    The three declarations are properties rather than plain attributes for a
    reason: an objective is something the compiler *inspects*, so a frozen
    dataclass has to be able to satisfy it and nothing in the framework may
    write them back. `compute` is the one member that does work.
    """

    @property
    def name(self) -> str:
        """Unique within a stage; it keys the per-objective logging (§6.2)."""

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        """`(port, realisation)` pairs.

        Naming the realisation is what lets the compiler plan exactly the
        forward passes the objectives demand, and no more (§2.1).
        """

    @property
    def rows(self) -> Rows:
        """The population this objective is entitled to.

        The stage's own scope is intersected in by the compiler (§7.0); this
        is the objective's half of that.
        """

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """The subset of `requires` this objective reads but does not train.

        A stop-gradient is invisible to the graph: `requires` says a term
        *reads* `p(t|x)`, and the compiler orders the forward pass from that,
        but a `.detach()` inside `compute` means no gradient ever reaches the
        component that produced it. Without this declaration the dead-trainable
        check (§8.4) accepts a stage whose sole trainable is the detached side
        — which compiles, trains, and makes every optimiser step a no-op.

        It is a required member rather than an optional attribute for the
        reason §7.1 gives about provenance: a declaration the compiler falls
        back to a default for is a declaration that can be forgotten, and
        forgetting this one restores exactly the hole it exists to close.

        Return the empty set when the term backpropagates through everything it
        reads, which is the ordinary case. Where a card field governs the
        stop-gradient — `gradients.marginal_nll_grad_path`,
        `gradients.stop_gradients` — derive this from that field rather than
        stating it twice.
        """

    @property
    def batch_coupled(self) -> bool:
        """Does this term's per-row value depend on the *other* rows of the batch?

        True for `InfoNCEContrastive`, whose negatives are the other `N - 1`
        rows, and false for every likelihood term, whose per-row value would be
        unchanged if the batch were split in two. The distinction is what makes
        `optimisation.batch_size` a paper-governed number rather than a
        deployment detail, and `compile()` uses it for exactly one rule: a
        stage that hands batch construction back to the caller
        (`ExternalBatches`) may not hold a term whose arithmetic depends on how
        many rows arrive.

        A required member rather than an attribute with a `False` fallback, on
        the same argument `detaches` is stated with: a declaration the compiler
        supplies a default for is a declaration that can be forgotten, and
        forgetting this one restores the hole `scarf.md` §5.6 was written
        about. Barlow Twins and VICReg — whose losses are computed from a
        cross-correlation or covariance over the batch axis — are the next
        terms that will answer true.
        """

    def compute(
        self,
        state: State,
        batch: XTYBatch,
        rows: RowIndex,
        ctx: TrainContext,
    ) -> LossTerm:
        """The **unweighted** loss over `rows` (`DESIGN.md` §4).

        `state` and `batch` are both full-batch and share one batch axis, so
        the same `rows` indexes both and alignment is automatic. The objective
        gathers by it; it is not handed a pre-sliced batch, because state holds
        distribution objects that are not generally sliceable.

        An objective never weights its own value, never calls `.backward()`,
        never mutates `state`, the batch or parameters, and never touches
        parameters directly.

        The one thing it *may* mutate is state of its own: an objective that
        implements `StatefulObjective.initial_state` is handed the result back
        through `ctx.objective_states` and may write to it (`DESIGN.md` §4).
        That state is built once per stage execution by the executor, so it is
        a property of the run rather than of the recipe, and it is not an
        artifact.
        """


@dataclass(frozen=True)
class Weighted:
    """An objective as a stage uses it (`DESIGN.md` §6, §6.1).

    Neither field has a default, and that is the point. Both are paper-governed
    (`FIDELITY.md` §2 names `losses.weights` and `losses.reduction`), and the
    standing rule is that a paper-governed field carries the `REQUIRED`
    sentinel so it cannot fall through to a framework default (§9.1).

    `DESIGN.md` §6.1 writes `reduction` with `mean` as its default. It is
    `REQUIRED` here instead, because that section also explains why: `sum` and
    `mean` differ by a factor that varies *per batch* whenever the row
    population does, so the wrong choice yields a model that trains, looks
    reasonable, and weights its semi-supervised term differently from the
    paper. That is exactly the failure the sentinel exists to prevent, and a
    default is what would let it through unread.

    Attributes:
        objective: The loss. Its `name` keys the per-objective log (§6.2).
        weight: A `Schedule`, or a number coerced to `Constant`.
        reduction: How the term's mean-over-rows value enters the total.
    """

    objective: Objective
    weight: Schedule | float = REQUIRED
    reduction: Reduction = REQUIRED

    def __post_init__(self) -> None:
        candidate: object = self.objective
        if not isinstance(candidate, Objective):
            missing = [
                member
                for member in (
                    "name",
                    "requires",
                    "rows",
                    "detaches",
                    "batch_coupled",
                    "compute",
                )
                if not hasattr(candidate, member)
            ]
            raise CompileError(
                f"Weighted holds an Objective — name, requires, rows, detaches, "
                f"batch_coupled, compute — "
                f"got {type(candidate)}, which is missing {missing!r} "
                "(DESIGN.md §4)."
            )
        if is_required(self.weight):
            raise CompileError(_no_default("weight", self.objective, "losses.weights"))
        if is_required(self.reduction):
            raise CompileError(
                _no_default("reduction", self.objective, "losses.reduction")
            )
        object.__setattr__(self, "weight", as_schedule(self.weight))
        validate_reduction(self.reduction, where=f"objective {self.objective.name!r}")

    @property
    def name(self) -> str:
        """The wrapped objective's name."""
        return self.objective.name

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        """The wrapped objective's `(port, realisation)` requirements."""
        return self.objective.requires

    @property
    def rows(self) -> Rows:
        """The wrapped objective's row population."""
        return self.objective.rows

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """The wrapped objective's stop-gradients."""
        return self.objective.detaches

    @property
    def batch_coupled(self) -> bool:
        """Whether the wrapped objective reads the rest of the batch."""
        return self.objective.batch_coupled

    @property
    def schedule(self) -> Schedule:
        """The weight as a `Schedule`. A number given for it is a `Constant`."""
        return as_schedule(self.weight)

    def weight_at(self, step: int) -> float:
        """The weight this term carries at `step`."""
        return self.schedule(step)


def _no_default(field: str, objective: object, key: str) -> str:
    name = getattr(objective, "name", type(objective).__name__)
    return (
        f"objective {name!r} was given no {field}. It binds card key {key!r} and "
        "is governed by the paper, so it has no usable default — the recipe sets "
        "it explicitly (DESIGN.md §9.1, CLAUDE.md standing rules)."
    )


TeacherRole = Literal["consistency_target", "evaluation"]
"""What a stage's EMA copy is *for* (`DESIGN.md` §2.1, §11).

Two methods keep an EMA and mean opposite things by it. Mean Teacher's is a
target: an objective reads `params="teacher"` and the EMA is part of the
training signal. FixMatch's is a reporting device — its pseudo-label comes from
the current network, and the EMA exists only to be evaluated with, so no
objective reads it at all.

The compiler cannot tell those apart from the graph, and the difference decides
whether "no objective requires a teacher realisation" is a silent no-op or the
whole point. So the stage says which, and the compiler checks the declaration
against the passes it planned rather than guessing from them.
"""


def _is_decay(value: object) -> bool:
    """A finite number in [0, 1) — the range an EMA decay must lie in."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and 0.0 <= float(value) < 1.0


@dataclass(frozen=True)
class TeacherSpec:
    """The card-driven EMA parameter set a stage maintains (`PLAN.md` P8).

    Four of the five fields are required because all four appear in the
    mechanics checklist. In particular, buffer handling and module mode are
    independent: a teacher in training mode may update its own BatchNorm
    statistics even when student buffers are not included in the EMA.

    `role` is the fifth and is required for a different reason — it binds no
    card key, because it is not a number a paper states but a fact about this
    program that only the recipe knows.

    `decay` is a `Schedule`, or a number coerced to `Constant` — the same
    surface `Weighted.weight` has, for the same reason. Mean Teacher raises
    decay after ramp-up in the evaluation it reports from, so a single constant
    could not state that method as published; `mean_teacher.md` §5 carried the
    gap as a framework limitation until this existed, and `DESIGN.md` §11.4
    records discharging the `ema-decay-schedule` ledger row against it. A
    number behaves exactly as it did before it was
    a schedule: `describe()` prints the plain decay for a `Constant`, so every
    plan, digest and recorded result predating this field stands unchanged, and
    `tests/invariants/test_teacher.py` asserts that rather than assuming it.
    """

    decay: Schedule | float = REQUIRED
    applies_to_buffers: bool = REQUIRED
    train_mode: bool = REQUIRED
    requires_grad: Literal[False] = REQUIRED
    role: TeacherRole = REQUIRED

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "nominal_decay": "teacher.ema_decay",
        "applies_to_buffers": "teacher.ema_applies_to_buffers",
        "train_mode": "teacher.teacher_in_train_mode",
        "requires_grad": "teacher.teacher_requires_grad",
    }

    def __post_init__(self) -> None:
        # Resolve the bindings first: an omitted paper-governed value should
        # fail as an omission, not later as a type error involving REQUIRED.
        card_hyperparameters(self)
        if isinstance(self.decay, Schedule):
            decay: Schedule = self.decay
        else:
            # Validated before coercion: `Constant` rejects a non-finite weight
            # with its own error, and a decay out of range should read as a
            # decay problem rather than as a schedule problem.
            if not _is_decay(self.decay):
                raise CompileError(
                    f"teacher EMA decay must be a finite number in [0, 1), or a "
                    f"Schedule of them, got {self.decay!r}"
                )
            decay = Constant(float(self.decay))
        # A schedule cannot be checked at every step it will ever be asked
        # about, so the two reachable now are checked now and the rest at the
        # update that reads them (`training/teacher.py`). Both ends matter:
        # step 0 is where it starts and `nominal` is where it settles, and a
        # decay outside [0, 1) at either end is a sign error, not a tuning
        # choice.
        for label, at in (("at step 0", decay(0)), ("nominal", decay.nominal)):
            if not 0.0 <= at < 1.0:
                raise CompileError(
                    f"teacher EMA decay must be a finite number in [0, 1); "
                    f"{decay.describe()} is {at!r} {label}"
                )
        object.__setattr__(self, "decay", decay)
        for field_name in ("applies_to_buffers", "train_mode", "requires_grad"):
            value = getattr(self, field_name)
            if type(value) is not bool:
                raise CompileError(
                    f"TeacherSpec.{field_name} must be bool, got {value!r} "
                    f"({type(value).__name__})"
                )
        if self.requires_grad:
            raise CompileError(
                "TeacherSpec.requires_grad must be false. A teacher is an EMA "
                "target parameter set, never an optimiser target; allowing a "
                "gradient would violate the teacher-isolation invariant "
                "(FIDELITY.md Tier 0)."
            )
        if is_required(self.role):
            raise CompileError(
                "TeacherSpec was constructed without a role. An EMA copy is "
                "either a 'consistency_target' an objective reads or an "
                "'evaluation' set nothing trains against, and the compiler "
                "checks the declaration against the passes it planned rather "
                "than inferring one from the other (DESIGN.md §2.1)."
            )
        if self.role not in get_args(TeacherRole):
            raise CompileError(
                f"TeacherSpec.role must be one of {list(get_args(TeacherRole))!r}, "
                f"got {self.role!r}"
            )

    @property
    def nominal_decay(self) -> float:
        """The decay the schedule settles at — the number a card states.

        `teacher.ema_decay` binds here rather than to `decay` itself so that a
        recipe that schedules its decay still reports one number to the §4
        cross-check, and so that the value a recipe reported before `decay`
        could be a schedule is the value it reports now. `losses.weights` and
        `losses.schedules` split the same way for the same reason: a paper
        states a rate and its schedule as two separate facts.

        Deliberately tolerant of an unvalidated `decay`, because
        `card_hyperparameters` reads it during `__post_init__` — before the
        range check below, so that an omitted decay fails as an omission
        rather than as a range error about the sentinel.
        """
        decay = self.decay
        return decay.nominal if isinstance(decay, Schedule) else decay

    def describe(self) -> str:
        """One stable plan line.

        A constant decay prints exactly as it did before `decay` could be a
        schedule, so this line — and every digest over it — is unchanged for
        every recipe that does not schedule its decay.
        """
        buffers = "ema" if self.applies_to_buffers else "independent"
        mode = "train" if self.train_mode else "eval"
        schedule = as_schedule(self.decay)
        scheduled = (
            "" if isinstance(schedule, Constant) else f", decay_schedule={schedule}"
        )
        return (
            f"ema(decay={self.nominal_decay!r}{scheduled}, buffers={buffers}, "
            f"mode={mode}, requires_grad=False, role={self.role})"
        )


@dataclass(frozen=True)
class Stage:
    """One step of the program (`DESIGN.md` §7).

    Attributes:
        name: Unique within a program; names the stage's artifacts and its
            section of the printed plan.
        objectives: The `Weighted` terms active in this stage. Bare objectives
            are rejected: the weight and the reduction are paper-governed and
            have no default (`DESIGN.md` §6.1, §9.1).
        trainable: Component names this stage updates. A name that is not a
            component, or a component no active objective depends on, is a
            compile error — the second catches a dead-weight stage.
        rows: The stage's row scope. The eligible set for each objective is
            this intersected with the objective's own population (§7.0).
        initialise_from: An earlier stage whose immutable checkpoint supplies
            this stage's initial values. Components absent from that checkpoint
            retain the recipe's initial values; no immediately preceding stage
            is inherited implicitly.
        teacher: The EMA teacher this stage maintains, or `None`. A teacher
            realisation is compilable only when this is present.
        action: An optional pseudo-label emission or functional array fit.
            Actions are explicit program data, never conditionals in a recipe.
        inputs: Earlier pseudo-label stages whose immutable side tables are
            joined functionally when this stage loads its batches.
        executor: ``gradient``, ``array_fit`` or ``cross_fit``. The recipe says
            which one; the runtime never guesses from an object's methods.
        allow_leakage: A per-consuming-stage opt-out from §7.2, valid only for
            a recipe whose purpose is ``predictive``.
        optimiser: How the stage descends. `REQUIRED` for a stage that has
            objectives: it binds four card keys, none of which may fall
            through to a framework default (§9.1).
        steps: Optimiser steps, and **steps rather than epochs** —
            `FIDELITY.md` §2 records that "epochs on a semi-supervised loader
            are ambiguous", so the framework counts the unambiguous unit and a
            card stating epochs converts, writing the conversion into its §7.
            Also `REQUIRED` where there are objectives; it binds
            `optimisation.total_steps_or_epochs`.
        sampler: What feeds the stage — `UniformSampler`, `QuotaSampler` or
            `ExternalBatches`. `REQUIRED`, because it owns the other two
            `optimisation` keys and there is no honest default: a sampler
            inherited from the framework would be a batch composition the card
            never stated, and a silent `ExternalBatches` would be the old hole
            with a new name. `ExternalBatches` says the caller supplies the
            rows and is rejected for a stage holding a batch-coupled objective
            (`docs/proposals/loader.md` §10).
    """

    name: str
    objectives: tuple[Weighted, ...] = ()
    trainable: tuple[str, ...] = ()
    rows: Rows = "all"
    initialise_from: str | None = None
    teacher: TeacherSpec | None = None
    action: PseudoLabelAction | ArrayFitAction | None = None
    inputs: tuple[str, ...] = ()
    executor: Executor = "gradient"
    allow_leakage: bool = False
    optimiser: OptimiserSpec = REQUIRED
    steps: int = REQUIRED
    sampler: SamplerSpec = REQUIRED

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "steps": "optimisation.total_steps_or_epochs"
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "trainable", tuple(self.trainable))
        object.__setattr__(self, "inputs", tuple(self.inputs))
        if not require_str("stage name", self.name, error=CompileError).isidentifier():
            raise CompileError(
                f"stage name {self.name!r} must be a Python identifier: it names "
                "the stage's artifacts and its section of the execution plan"
            )
        for entry in self.objectives:
            if not isinstance(entry, Weighted):
                raise CompileError(
                    f"stage {self.name!r} holds {type(entry)} in `objectives`; a "
                    "stage holds Weighted terms (DESIGN.md §6). Wrap the "
                    "objective in Weighted(objective, weight=..., "
                    "reduction=...) — both are paper-governed and neither has a "
                    "default."
                )
        validate_rows(self.rows, f"stage {self.name!r}")
        if self.initialise_from is not None:
            source = require_str(
                "Stage.initialise_from", self.initialise_from, error=CompileError
            )
            if not source.isidentifier():
                raise CompileError(
                    f"stage {self.name!r} initialises from {source!r}, which "
                    "must be a Python identifier naming an earlier stage"
                )
        if self.teacher is not None and not isinstance(self.teacher, TeacherSpec):
            raise CompileError(
                f"stage {self.name!r} holds {type(self.teacher)} as its teacher; "
                "expected TeacherSpec or None"
            )
        if self.executor not in ("gradient", "array_fit", "cross_fit"):
            raise CompileError(
                f"stage {self.name!r} has executor {self.executor!r}; expected "
                "'gradient', 'array_fit' or 'cross_fit' (DESIGN.md §7)."
            )
        if type(self.allow_leakage) is not bool:
            raise CompileError(
                f"stage {self.name!r} has allow_leakage={self.allow_leakage!r}; "
                "it must be bool"
            )
        if self.allow_leakage and not self.inputs:
            raise CompileError(
                f"stage {self.name!r} sets allow_leakage=True but consumes no "
                "artifact inputs. The §7.2 opt-out belongs on the consuming "
                "stage whose outcome fit intentionally uses unsafe labels."
            )
        self._check_action()
        self._check_gradient_fields()
        self._check_sampler()
        duplicates = _duplicates(self.trainable)
        if duplicates:
            raise CompileError(
                f"stage {self.name!r} lists {duplicates!r} in `trainable` more "
                "than once"
            )
        duplicate_inputs = _duplicates(self.inputs)
        if duplicate_inputs:
            raise CompileError(
                f"stage {self.name!r} lists artifact inputs {duplicate_inputs!r} "
                "more than once"
            )
        for source in self.inputs:
            source_name = require_str("Stage.inputs entry", source, error=CompileError)
            if not source_name.isidentifier():
                raise CompileError(
                    f"stage {self.name!r} input {source_name!r} must be a Python "
                    "identifier naming an earlier pseudo-label stage"
                )

    def _check_action(self) -> None:
        """Keep each executor's public contract narrow and unambiguous."""
        action = self.action
        is_pseudo = isinstance(action, PseudoLabelAction)
        is_array = action is not None and isinstance(action, ArrayFitAction)
        if action is not None and not is_pseudo and not is_array:
            raise CompileError(
                f"stage {self.name!r} holds {type(action)} as its action; "
                "expected PseudoLabelAction or an ArrayFitAction"
            )
        if action is not None:
            action_name = require_str(
                "stage action name", action.name, error=CompileError
            )
            if not action_name.isidentifier():
                raise CompileError(
                    f"stage {self.name!r} action name {action_name!r} must be a "
                    "Python identifier"
                )

        if self.executor == "array_fit":
            if not is_array or is_pseudo:
                raise CompileError(
                    f"stage {self.name!r} uses array_fit, so its action must "
                    "implement ArrayFitAction.fit(batch, rows, seed=...)."
                )
            if self.objectives or self.trainable or self.teacher is not None:
                raise CompileError(
                    f"stage {self.name!r} uses array_fit outside autograd; it "
                    "cannot also declare objectives, graph trainables or an EMA "
                    "teacher. Put gradient work in a separate stage."
                )
            if self.initialise_from is not None:
                raise CompileError(
                    f"stage {self.name!r} uses array_fit and cannot initialise "
                    "the component graph from a checkpoint it does not fit. Use "
                    "artifact inputs for data dependencies."
                )
            return

        if is_array:
            raise CompileError(
                f"stage {self.name!r} holds an ArrayFitAction but declares "
                f"executor={self.executor!r}; array actions run only under "
                "executor='array_fit'."
            )
        if self.executor == "cross_fit":
            if not is_pseudo:
                raise CompileError(
                    f"stage {self.name!r} uses cross_fit but declares no "
                    "PseudoLabelAction. P10 cross-fitting fits the declared "
                    "gradient objectives per fold and emits their held-out "
                    "treatment predictions."
                )
            if not self.objectives:
                raise CompileError(
                    f"stage {self.name!r} uses cross_fit but has no objectives "
                    "to fit in each training fold"
                )
        if (
            is_pseudo
            and self.executor == "gradient"
            and not self.objectives
            and self.initialise_from is None
        ):
            raise CompileError(
                f"action-only pseudo-label stage {self.name!r} needs "
                "initialise_from naming the checkpoint whose parameters make "
                "its predictions. Without it the artifact has no producing "
                "checkpoint provenance (DESIGN.md §7.1)."
            )
        if not self.objectives and self.trainable:
            raise CompileError(
                f"stage {self.name!r} has no objectives but declares trainable "
                f"components {list(self.trainable)!r}; an action-only stage "
                "does not descend a gradient"
            )

    def _check_sampler(self) -> None:
        """Every stage says what feeds it, and no stage inherits an answer.

        `REQUIRED` rather than a default for the reason `optimiser` and `steps`
        are: the field owns `optimisation.batch_size` and
        `optimisation.labelled_unlabelled_ratio`, and a default is precisely
        the silent inheritance §9.1 exists to make impossible. Unlike those two
        it is required of *every* stage, including action-only and array-fit
        ones — those read rows even when they descend no gradient, and where
        the rows come from is exactly what was previously unsaid.
        """
        if is_required(self.sampler):
            raise CompileError(
                f"stage {self.name!r} declares no sampler, so nothing says what "
                "feeds it. Declare UniformSampler(batch_size=...), a "
                "QuotaSampler, or ExternalBatches() if the caller supplies the "
                "rows — the last is a statement the plan prints, not a default "
                "(DESIGN.md §9.1, docs/proposals/loader.md §3.4)."
            )
        if not isinstance(self.sampler, SAMPLERS):
            raise CompileError(
                f"stage {self.name!r} holds {type(self.sampler)} as its sampler; "
                f"expected one of {[cls.__name__ for cls in SAMPLERS]!r}"
            )
        if isinstance(self.sampler, ExternalBatches):
            return
        if self.executor != "gradient":
            raise CompileError(
                f"stage {self.name!r} declares executor={self.executor!r} and a "
                "sampler. An array or cross fit consumes one finite row-keyed "
                "table rather than a stream of drawn batches, and no card asks "
                "for a sampled one, so building that path would be a mechanism "
                "with no consumer (DESIGN.md §11.2). Declare ExternalBatches() "
                "and let the caller supply the table."
            )
        for population in self.sampler.rows:
            if populations_are_disjoint(self.rows, population):
                raise CompileError(
                    f"stage {self.name!r} scopes rows to {self.rows!r} but its "
                    f"sampler draws a quota from {population!r}, which is empty "
                    "by construction. A quota that can never be filled is a "
                    "wiring error, not a runtime shortfall (DESIGN.md §7.0)."
                )

    def _check_gradient_fields(self) -> None:
        """A stage that descends a gradient says how, and for how long.

        Checked here rather than in `compile()` for the reason `Weighted`
        checks its own weight and reduction: both fields bind card keys, so
        the `REQUIRED` sentinel is what stops them inheriting a framework
        default, and a sentinel that survives construction is one that reaches
        a plan and prints as a number.

        A stage with no objectives is left alone because P10 action-only and
        array-fit stages intentionally do not descend a gradient.
        """
        if not self.objectives:
            return
        for field_name, key in (
            ("optimiser", "the optimisation.* keys"),
            ("steps", "optimisation.total_steps_or_epochs"),
        ):
            if is_required(getattr(self, field_name)):
                raise CompileError(
                    f"stage {self.name!r} has objectives but no {field_name!r}, so "
                    "nothing says how it descends. It binds card key(s) "
                    f"{key} and is governed by the paper, so it has no usable "
                    "default — the recipe sets it explicitly (DESIGN.md §9.1)."
                )
        if not isinstance(self.optimiser, OptimiserSpec):
            raise CompileError(
                f"stage {self.name!r} holds {type(self.optimiser)} as its "
                "optimiser; it holds an OptimiserSpec (DESIGN.md §7)."
            )
        if type(self.steps) is not int or self.steps < 1:
            raise CompileError(
                f"stage {self.name!r} runs for {self.steps!r} steps; it needs a "
                "positive integer number of optimiser steps. Steps, not epochs "
                "— epochs on a semi-supervised loader are ambiguous "
                "(FIDELITY.md §2)."
            )


@dataclass(frozen=True, init=False)
class Program(Sequence[Stage]):
    """An immutable ordered list of stages (`DESIGN.md` §7).

    A program is deliberately not a DAG. Stage names are unique and an
    `initialise_from` edge points only to an earlier element, so execution is a
    single readable loop and no scheduler exists to make ordering decisions.
    """

    stages: tuple[Stage, ...]

    def __init__(self, stages: Sequence[Stage]) -> None:
        object.__setattr__(self, "stages", tuple(stages))
        self.__post_init__()

    def __post_init__(self) -> None:
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, Stage):
                raise CompileError(
                    f"program entry {index} is {type(stage)}, expected Stage"
                )
        duplicates = _duplicates(tuple(stage.name for stage in self.stages))
        if duplicates:
            raise CompileError(
                f"program has more than one stage called {duplicates!r}; stages "
                "are referenced by name"
            )
        positions = {stage.name: index for index, stage in enumerate(self.stages)}
        for index, stage in enumerate(self.stages):
            source = stage.initialise_from
            if source is not None:
                source_index = positions.get(source)
                if source_index is None:
                    raise CompileError(
                        f"stage {stage.name!r} initialises from unknown stage "
                        f"{source!r}; this program has {list(positions)!r}"
                    )
                if source_index >= index:
                    raise CompileError(
                        f"stage {stage.name!r} initialises from {source!r}, which "
                        "is not an earlier stage. Program is an ordered list, "
                        "not a DAG; initialise_from may only point backwards "
                        "(DESIGN.md §7)."
                    )
                producer = self.stages[source_index]
                if producer.executor != "gradient" or not producer.objectives:
                    raise CompileError(
                        f"stage {stage.name!r} initialises from {source!r}, but "
                        f"that stage uses executor={producer.executor!r} and "
                        "does not emit one restorable component-graph "
                        "checkpoint. Use inputs for pseudo-label artifacts; "
                        "cross-fit fold checkpoints are prediction provenance, "
                        "not implicit initialisation state."
                    )

            for input_name in stage.inputs:
                input_index = positions.get(input_name)
                if input_index is None:
                    raise CompileError(
                        f"stage {stage.name!r} consumes unknown artifact stage "
                        f"{input_name!r}; this program has {list(positions)!r}"
                    )
                if input_index >= index:
                    raise CompileError(
                        f"stage {stage.name!r} consumes artifact {input_name!r}, "
                        "which is not from an earlier stage. Program is an "
                        "ordered list, not a DAG (DESIGN.md §7)."
                    )
                input_stage = self.stages[input_index]
                if not isinstance(input_stage.action, PseudoLabelAction):
                    raise CompileError(
                        f"stage {stage.name!r} names {input_name!r} in inputs, "
                        "but that stage does not emit PseudoLabels. P10 inputs "
                        "name immutable pseudo-label side tables; checkpoint "
                        "state is named by initialise_from."
                    )

    def __iter__(self) -> Iterator[Stage]:
        return iter(self.stages)

    def __len__(self) -> int:
        return len(self.stages)

    @overload
    def __getitem__(self, index: int) -> Stage: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Stage, ...]: ...

    def __getitem__(self, index: int | slice) -> Stage | tuple[Stage, ...]:
        return self.stages[index]

    def stage(self, name: str) -> Stage:
        """The stage called `name`."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise CompileError(
            f"program has no stage {name!r}; it has "
            f"{[stage.name for stage in self.stages]!r}"
        )


@dataclass(frozen=True, init=False)
class Recipe:
    """A named method: a graph, a program, a schema and a card.

    Attributes:
        name: The registry name (`tarnet`, `cnflow`, ...).
        schema: Resolved once, so views, ports and objectives are validated
            statically rather than failing at step 4,000 (§1.2).
        system: The component graph.
        program: The stages, **in order**. Not a DAG (§7).
        card: Path to the recipe's spec card. A recipe without one cannot be
            reviewed, so it is a required field rather than an optional
            annotation (`FIDELITY.md` §1).
        views: Named data views available to objective realisations. The
            untransformed ``identity`` view is built in and is not listed.
        purpose: `causal` or `predictive` (§7.2).
        data: The split, standardisation and missingness policy, and the owner
            of the four `data.*` card keys. Required exactly when some stage
            declares a sampler — the same rule `optimiser` and `steps` follow,
            and for the same reason: a policy is required when there is
            something to apply it to, and a policy over batches the caller
            builds would be a card key nothing could check, which is the
            failure this capability exists to end. It holds a declaration, not
            rows: a `Recipe` is a method, and the data arrives at run time as a
            `Dataset`.
    """

    name: str
    schema: Schema
    system: ComponentGraph
    program: Program
    card: str
    purpose: Purpose = "causal"
    views: tuple[ViewSpec, ...] = ()
    data: DataSpec | None = None

    def __init__(
        self,
        name: str,
        schema: Schema,
        system: ComponentGraph,
        program: Program | Sequence[Stage],
        card: str,
        purpose: Purpose = "causal",
        views: Sequence[ViewSpec] = (),
        data: DataSpec | None = None,
    ) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "system", system)
        resolved_program = program if isinstance(program, Program) else Program(program)
        object.__setattr__(
            self,
            "program",
            resolved_program,
        )
        object.__setattr__(self, "card", card)
        object.__setattr__(self, "purpose", purpose)
        object.__setattr__(self, "views", tuple(views))
        object.__setattr__(self, "data", data)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not require_str("recipe name", self.name, error=CompileError).isidentifier():
            raise CompileError(f"recipe name {self.name!r} must be a Python identifier")
        if self.purpose not in ("causal", "predictive"):
            raise CompileError(
                f"recipe {self.name!r} has purpose {self.purpose!r}; expected "
                "'causal' or 'predictive' (DESIGN.md §7.2)"
            )
        if not require_str("recipe card", self.card, error=CompileError):
            raise CompileError(
                f"recipe {self.name!r} names no card. Every recipe has a card at "
                "docs/recipes/<name>.md, written and reviewed before the code "
                "(FIDELITY.md §1)."
            )
        if not self.program:
            raise CompileError(f"recipe {self.name!r} has an empty program")
        for view in self.views:
            if not isinstance(view, ViewSpec):
                raise CompileError(
                    f"recipe {self.name!r} holds {type(view)} in `views`; "
                    "expected ViewSpec declarations (DESIGN.md §5)"
                )
        duplicate_views = _duplicates(tuple(view.name for view in self.views))
        if duplicate_views:
            raise CompileError(
                f"recipe {self.name!r} has more than one view called "
                f"{duplicate_views!r}; realisations resolve views by name"
            )
        self._check_data()

    def _check_data(self) -> None:
        """A policy exactly where there is something for it to govern.

        Declaring one over `ExternalBatches` stages would print four `data.*`
        values in the plan that no code path applies — a card key asserting
        rather than binding, which `DESIGN.md` §7.1 rejects for provenance and
        §9.1 for hyperparameters. Omitting one where a stage samples leaves the
        same keys `n/a` on a run the framework *is* loading, which is the debt
        `tarnet.md` §5.5 records.
        """
        samples = [
            stage.name
            for stage in self.program
            if not is_required(stage.sampler) and draws_from_population(stage.sampler)
        ]
        if self.data is None:
            if samples:
                raise CompileError(
                    f"recipe {self.name!r} declares no `data` policy, but "
                    f"stage(s) {samples!r} sample from a population. The split, "
                    "standardisation and missingness a sampled run depends on "
                    "bind the four `data.*` card keys and have no default "
                    "(DESIGN.md §9.1)."
                )
            return
        if not isinstance(self.data, DataSpec):
            raise CompileError(
                f"recipe {self.name!r} holds {type(self.data)} as its data "
                "policy; expected DataSpec"
            )
        if not samples:
            raise CompileError(
                f"recipe {self.name!r} declares a `data` policy, but every "
                "stage takes ExternalBatches, so nothing applies it. Four "
                "`data.*` keys would print in the plan as though they governed "
                "the run (DESIGN.md §7.1)."
            )

    def stage(self, name: str) -> Stage:
        """The stage called `name`."""
        try:
            return self.program.stage(name)
        except CompileError as error:
            raise CompileError(
                f"recipe {self.name!r} has no stage {name!r}; it has "
                f"{[stage.name for stage in self.program]!r}"
            ) from error

    def view(self, name: str) -> ViewSpec:
        """The declared view called ``name`` (identity is handled by the run)."""
        for view in self.views:
            if view.name == name:
                return view
        raise CompileError(
            f"recipe {self.name!r} has no view {name!r}; it has "
            f"{[view.name for view in self.views]!r} plus the built-in "
            "'identity' view"
        )


def _duplicates(names: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for name in names:
        if name in seen:
            repeated.add(name)
        seen.add(name)
    return sorted(repeated)


def validate_rows(rows: Rows, where: str) -> None:
    """Re-raise an unknown row population as the compile rejection it is.

    `validate_population` knows the vocabulary but not who used it wrongly, and
    "unknown row population 't_known'" without a stage or objective name is not
    an actionable message in a program with a dozen of both.
    """
    try:
        validate_population(rows)
    except Xty2Error as error:
        raise CompileError(f"{where}: {error}") from error
