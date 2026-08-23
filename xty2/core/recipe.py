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
be planned. The remaining §7 fields (`inputs`, `executor`, `allow_leakage`)
arrive with the second executor and leakage rule that need them (P10).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Literal,
    Protocol,
    overload,
    runtime_checkable,
)

from xty2.core.card_keys import REQUIRED, card_hyperparameters, is_required
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
from xty2.core.rows import RowIndex, Rows, validate_population
from xty2.core.schedules import Schedule, as_schedule
from xty2.core.schema import Schema
from xty2.core.views import ViewSpec

if TYPE_CHECKING:  # pragma: no cover - the batch is only named in a signature
    from xty2.core.batch import XTYBatch

Purpose = Literal["causal", "predictive"]
"""What the recipe is for. `predictive` is what may opt out of the leakage
rule (`DESIGN.md` §7.2); `causal` may not."""


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
        never mutates state and never touches parameters directly.
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
                for member in ("name", "requires", "rows", "detaches", "compute")
                if not hasattr(candidate, member)
            ]
            raise CompileError(
                f"Weighted holds an Objective — name, requires, rows, detaches, "
                f"compute — "
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


@dataclass(frozen=True)
class TeacherSpec:
    """The card-driven EMA parameter set a stage maintains (`PLAN.md` P8).

    All four fields are required because all four appear in the mechanics
    checklist. In particular, buffer handling and module mode are independent:
    a teacher in training mode may update its own BatchNorm statistics even
    when student buffers are not included in the EMA.
    """

    decay: float = REQUIRED
    applies_to_buffers: bool = REQUIRED
    train_mode: bool = REQUIRED
    requires_grad: Literal[False] = REQUIRED

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "decay": "teacher.ema_decay",
        "applies_to_buffers": "teacher.ema_applies_to_buffers",
        "train_mode": "teacher.teacher_in_train_mode",
        "requires_grad": "teacher.teacher_requires_grad",
    }

    def __post_init__(self) -> None:
        # Resolve the bindings first: an omitted paper-governed value should
        # fail as an omission, not later as a type error involving REQUIRED.
        card_hyperparameters(self)
        if isinstance(self.decay, bool) or not isinstance(self.decay, (int, float)):
            raise CompileError(
                f"teacher EMA decay must be a finite number in [0, 1), got "
                f"{self.decay!r}"
            )
        decay = float(self.decay)
        if not math.isfinite(decay) or not 0.0 <= decay < 1.0:
            raise CompileError(
                f"teacher EMA decay must be a finite number in [0, 1), got "
                f"{self.decay!r}"
            )
        object.__setattr__(self, "decay", decay)
        for field in ("applies_to_buffers", "train_mode", "requires_grad"):
            value = getattr(self, field)
            if type(value) is not bool:
                raise CompileError(
                    f"TeacherSpec.{field} must be bool, got {value!r} "
                    f"({type(value).__name__})"
                )
        if self.requires_grad:
            raise CompileError(
                "TeacherSpec.requires_grad must be false. A teacher is an EMA "
                "target parameter set, never an optimiser target; allowing a "
                "gradient would violate the teacher-isolation invariant "
                "(FIDELITY.md Tier 0)."
            )

    def describe(self) -> str:
        """One stable plan line."""
        buffers = "ema" if self.applies_to_buffers else "independent"
        mode = "train" if self.train_mode else "eval"
        return (
            f"ema(decay={self.decay!r}, buffers={buffers}, mode={mode}, "
            "requires_grad=False)"
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
        optimiser: How the stage descends. `REQUIRED` for a stage that has
            objectives: it binds four card keys, none of which may fall
            through to a framework default (§9.1).
        steps: Optimiser steps, and **steps rather than epochs** —
            `FIDELITY.md` §2 records that "epochs on a semi-supervised loader
            are ambiguous", so the framework counts the unambiguous unit and a
            card stating epochs converts, writing the conversion into its §7.
            Also `REQUIRED` where there are objectives; it binds
            `optimisation.total_steps_or_epochs`.

    `batch_size` and `labelled_unlabelled_ratio` are the other two
    `optimisation` keys and are deliberately absent: both are properties of
    what feeds the stage, and no loader exists yet. A field the executor
    could not check would be a card key nothing verifies, which is the
    failure mode `DESIGN.md` §7.1 rejects for provenance.
    """

    name: str
    objectives: tuple[Weighted, ...] = ()
    trainable: tuple[str, ...] = ()
    rows: Rows = "all"
    initialise_from: str | None = None
    teacher: TeacherSpec | None = None
    optimiser: OptimiserSpec = REQUIRED
    steps: int = REQUIRED

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "steps": "optimisation.total_steps_or_epochs"
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "trainable", tuple(self.trainable))
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
        self._check_gradient_fields()
        duplicates = _duplicates(self.trainable)
        if duplicates:
            raise CompileError(
                f"stage {self.name!r} lists {duplicates!r} in `trainable` more "
                "than once"
            )

    def _check_gradient_fields(self) -> None:
        """A stage that descends a gradient says how, and for how long.

        Checked here rather than in `compile()` for the reason `Weighted`
        checks its own weight and reduction: both fields bind card keys, so
        the `REQUIRED` sentinel is what stops them inheriting a framework
        default, and a sentinel that survives construction is one that reaches
        a plan and prints as a number.

        A stage with no objectives is left alone: it is already a compile
        error (`DESIGN.md` §8), and rejecting it here for the wrong reason
        would replace that message with a less useful one.
        """
        if not self.objectives:
            return
        for field, key in (
            ("optimiser", "the optimisation.* keys"),
            ("steps", "optimisation.total_steps_or_epochs"),
        ):
            if is_required(getattr(self, field)):
                raise CompileError(
                    f"stage {self.name!r} has objectives but no {field!r}, so "
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
            if source is None:
                continue
            source_index = positions.get(source)
            if source_index is None:
                raise CompileError(
                    f"stage {stage.name!r} initialises from unknown stage "
                    f"{source!r}; this program has {list(positions)!r}"
                )
            if source_index >= index:
                raise CompileError(
                    f"stage {stage.name!r} initialises from {source!r}, which is "
                    "not an earlier stage. Program is an ordered list, not a "
                    "DAG; initialise_from may only point backwards "
                    "(DESIGN.md §7)."
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
    """

    name: str
    schema: Schema
    system: ComponentGraph
    program: Program
    card: str
    purpose: Purpose = "causal"
    views: tuple[ViewSpec, ...] = ()

    def __init__(
        self,
        name: str,
        schema: Schema,
        system: ComponentGraph,
        program: Program | Sequence[Stage],
        card: str,
        purpose: Purpose = "causal",
        views: Sequence[ViewSpec] = (),
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
