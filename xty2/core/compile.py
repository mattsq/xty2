"""The compiler and the printable execution plan (`DESIGN.md` §8).

`compile(recipe)` is where the framework earns its keep. Everything it does is
a check that would otherwise be a comment: that every objective's
`(port, realisation)` requirement is actually produced, that a stage's
`trainable` list names components that exist and that something depends on,
that a stage scope and an objective's row population are not empty by
construction. Each rejection names what is wrong and what to do about it,
because an agent acting on a compile error is the reader these messages have.

The plan it prints is the review artifact. A reviewer diffs it against the
card's §3 mapping table and §4 checklist, and that diff is the review
(`FIDELITY.md` §1.2) — so it lists, per stage, the forward passes with the
components each one runs, the active objectives with their eligible rows and
declared ports, and the trainable parameter groups. It is deterministic: the
same recipe prints the same bytes, or a diff means nothing.

P10 adds the static half of the leakage guard here. A pseudo-label action names
the treatment-distribution port it reads, so ``used_y`` is the graph fact
``port_depends_on_raw_outcome`` rather than a producer-set flag. The runtime
half remains at artifact load, where actual fold row ids exist. Views are
validated here against the resolved schema before any stage is planned (P6,
§8.5). A teacher realisation is stage-local: it exists exactly when that stage
declares an EMA teacher, and its required ports are always detached because the
teacher parameter set is structurally gradient-free (P8).
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast, runtime_checkable

import torch

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import card_hyperparameters
from xty2.core.errors import CompileError, SchemaError, TrainingError, ViewError
from xty2.core.graph import (
    IDENTITY_VIEW,
    SOURCE_PORTS,
    ComponentGraph,
    Realisation,
    State,
)
from xty2.core.loss import Reduction
from xty2.core.optimisation import OptimiserSpec
from xty2.core.ports import Port
from xty2.core.recipe import (
    ArrayFitAction,
    Executor,
    Objective,
    PseudoLabelAction,
    Recipe,
    Stage,
    TeacherSpec,
    Weighted,
    validate_rows,
)
from xty2.core.rows import Rows, populations_are_disjoint
from xty2.core.schedules import Schedule
from xty2.core.views import ViewSpec


def plan_digest_of(rendered: str) -> str:
    """`sha256` of a rendered execution plan.

    A function rather than a method, because the two things that need it are a
    plan in memory and a `plan.txt` a run directory wrote earlier. One
    implementation is what lets the second be compared against the first.
    """
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# What compilation produces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardPass:
    """One evaluation of the graph, under one realisation (`DESIGN.md` §2.1).

    `components` is the closed subgraph the active objectives actually need,
    in topological order — a component nothing depends on is not run, which is
    what "plans the minimum number of forward passes" (§8.3) amounts to once
    the realisations are known.
    """

    realisation: Realisation
    components: tuple[str, ...]


@dataclass(frozen=True)
class CompiledObjective:
    """A weighted term with its eligible-row declaration resolved (§6, §7.0).

    `rows` is the stage scope and the objective's own population together, in
    the form `resolve_rows` takes. The intersection is not always itself a
    named population — `t_observed ∩ y_observed` is not — so it is carried as
    the populations to intersect rather than collapsed to one name.

    This is what the mixer is built from, and it is the only thing it is built
    from: the eligible set is the compiler's answer, so a mixer that re-derived
    it would be a second implementation of a rule the compiler has already
    rejected the empty-by-construction case of.
    """

    weighted: Weighted
    rows: tuple[Rows, ...]
    plan_details: tuple[str, ...]

    @property
    def objective(self) -> Objective:
        return self.weighted.objective

    @property
    def name(self) -> str:
        return self.weighted.name

    @property
    def weight(self) -> Schedule:
        return self.weighted.schedule

    @property
    def reduction(self) -> Reduction:
        return self.weighted.reduction


@runtime_checkable
class _ObjectiveWithPlanDetails(Protocol):
    """Optional stable mechanics an objective needs the plan to fingerprint."""

    def plan_details(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class CompiledStage:
    """A stage with its forward passes planned and its objectives resolved."""

    stage: Stage
    passes: tuple[ForwardPass, ...]
    objectives: tuple[CompiledObjective, ...]
    action_rows: tuple[Rows, ...] | None = None
    action_uses_y: bool | None = None

    @property
    def name(self) -> str:
        return self.stage.name

    @property
    def trainable(self) -> tuple[str, ...]:
        return self.stage.trainable

    @property
    def initialise_from(self) -> str | None:
        """The earlier checkpoint this stage starts from, if any."""
        return self.stage.initialise_from

    @property
    def teacher(self) -> TeacherSpec | None:
        """The stage-local EMA teacher declaration, if any."""
        return self.stage.teacher

    @property
    def optimiser(self) -> OptimiserSpec:
        """How this stage descends. Set — a stage with objectives cannot omit it."""
        return self.stage.optimiser

    @property
    def steps(self) -> int:
        """Optimiser steps, not epochs (`FIDELITY.md` §2)."""
        return self.stage.steps

    @property
    def executor(self) -> Executor:
        """The explicit executor kind; execution never infers one."""
        return self.stage.executor

    @property
    def action(self) -> PseudoLabelAction | ArrayFitAction | None:
        return self.stage.action

    @property
    def inputs(self) -> tuple[str, ...]:
        return self.stage.inputs


@dataclass(frozen=True)
class PlannedComponent:
    """One component's wiring, as the plan reports it."""

    name: str
    requires: tuple[Port, ...]
    provides: tuple[Port, ...]
    sources: Mapping[Port, str]
    """Which node supplies each required port — a component, or the source node."""
    reads_raw_outcome: bool
    """Does this component read `Y_RAW` itself?"""
    outcome_dependent: bool
    """Is `Y_RAW` in the transitive closure of its inputs (`DESIGN.md` §7.2)?"""


@dataclass(frozen=True)
class PlannedView:
    """One validated view, reduced to stable plan data."""

    name: str
    transforms: tuple[str, ...]
    preserves: tuple[str, ...]
    recomputes: tuple[str, ...]
    affected_columns: tuple[str, ...]
    draws: int = 1


@dataclass(frozen=True)
class ExecutionPlan:
    """The human-readable artifact a reviewer diffs against the card (§8).

    Attributes:
        hyperparameters: The flat `{canonical_key: value}` dict of §9.1 — what
            the card cross-check compares against, and what makes the printed
            plan diffable against card §4 by eye.
    """

    recipe: str
    purpose: str
    card: str
    features: int
    treatments: int
    views: tuple[PlannedView, ...]
    components: tuple[PlannedComponent, ...]
    stages: tuple[CompiledStage, ...]
    hyperparameters: Mapping[str, Any]

    def render(self) -> str:
        """The plan as text. Deterministic — the same recipe prints the same bytes."""
        lines: list[str] = [
            f"recipe: {self.recipe}",
            f"purpose: {self.purpose}",
            f"card: {self.card}",
            f"schema: D = {self.features}, K = {self.treatments}",
        ]
        if self.views:
            lines += ["", *self._view_lines()]
        lines += ["", *self._component_lines(), "", *self._lineage_lines()]
        for stage in self.stages:
            lines += ["", *self._stage_lines(stage)]
        lines += ["", *self._hyperparameter_lines()]
        return "\n".join(lines) + "\n"

    def __str__(self) -> str:
        return self.render()

    @property
    def digest(self) -> str:
        """`sha256` of the rendered plan — what an artifact records it came from.

        The plan is deterministic (that is what makes it diffable), so its
        digest identifies a compiled recipe exactly: components, wiring,
        objectives, schedules, row scopes and every resolved hyperparameter.
        A checkpoint carrying it can be told apart from one produced by a
        recipe that has since been edited, which is the question "is this
        artifact still the thing the card describes?" in a form something can
        check.
        """
        return plan_digest_of(self.render())

    # -- sections ----------------------------------------------------------

    def _component_lines(self) -> list[str]:
        width = _width(component.name for component in self.components)
        return ["components (topological order)"] + [
            f"  {component.name:<{width}}  {_ports(component.requires)} -> "
            f"{_ports(component.provides)}"
            for component in self.components
        ]

    def _view_lines(self) -> list[str]:
        lines = ["views"]
        for view in self.views:
            # Silent at one draw, so every plan written before the axis
            # existed renders and hashes exactly as it did.
            draws = f" ({view.draws} independent draws)" if view.draws > 1 else ""
            lines.append(f"  {view.name}{draws}")
            lines.append("    preserves  " + (", ".join(view.preserves) or "nothing"))
            lines.append(
                "    affects    "
                + (", ".join(view.affected_columns) or "no feature columns")
            )
            for transform in view.transforms:
                lines.append(f"    transform  {transform}")
            for recompute in view.recomputes:
                lines.append(f"    recompute  {recompute}")
        return lines

    def _lineage_lines(self) -> list[str]:
        width = _width(component.name for component in self.components)
        lines = ["data lineage"]
        for component in self.components:
            wiring = (
                ", ".join(
                    f"{port} <- {component.sources[port]}"
                    for port in component.requires
                )
                or "nothing"
            )
            lines.append(f"  {component.name:<{width}}  {wiring}")
        direct = [c.name for c in self.components if c.reads_raw_outcome]
        downstream = [c.name for c in self.components if c.outcome_dependent]
        lines.append(f"  reads raw y directly:  {_names(direct)}")
        lines.append(f"  depends on raw y:      {_names(downstream)}")
        return lines

    def _stage_lines(self, compiled: CompiledStage) -> list[str]:
        lines = [
            f"stage {compiled.name}",
            f"  rows: {compiled.stage.rows}",
            f"  executor: {compiled.executor}",
        ]
        if compiled.initialise_from is not None:
            lines.append(f"  initialise from: {compiled.initialise_from}")
        if compiled.inputs:
            lines.append(f"  inputs: {', '.join(compiled.inputs)}")
        if compiled.stage.allow_leakage:
            lines.append("  allow leakage: true (predictive only)")
        if compiled.teacher is not None:
            lines.append(f"  teacher: {compiled.teacher.describe()}")
        if isinstance(compiled.action, PseudoLabelAction):
            lines.append(f"  action: {compiled.action.describe()}")
            lines.append(f"  action rows: {_rows(compiled.action_rows or ('all',))}")
            lines.append(
                "  action uses raw y: "
                + ("true" if compiled.action_uses_y else "false")
            )
        elif compiled.action is not None:
            lines.append(f"  action: array fit {compiled.action.name}")
            lines.append(f"  action rows: {_rows(compiled.action_rows or ('all',))}")
        if compiled.objectives:
            lines += [
                f"  steps: {compiled.steps}",
                "  optimisation",
                *(f"    {line}" for line in compiled.optimiser.describe_lines()),
            ]
        lines.append(f"  forward passes ({len(compiled.passes)})")
        for forward in compiled.passes:
            components = " -> ".join(forward.components) or "nothing"
            lines.append(f"    {forward.realisation}: {components}")
        lines.append("  objectives")
        width = _width(objective.name for objective in compiled.objectives)
        rows = _width(_rows(objective.rows) for objective in compiled.objectives)
        for objective in compiled.objectives:
            requires = ", ".join(
                f"{port} @ {realisation}"
                for port, realisation in _sorted_requirements(objective.objective)
            )
            lines.append(
                f"    {objective.name:<{width}}  rows {_rows(objective.rows):<{rows}}"
                f"  reduction {objective.reduction}"
            )
            lines.append(f"      weight    {objective.weight.describe()}")
            for detail in objective.plan_details:
                lines.append(f"      setting   {detail}")
            lines.append(f"      requires  {requires or 'nothing'}")
            # A stop-gradient is invisible in `requires`, and which side a
            # paper detaches is exactly the one-line detail card §4 exists to
            # pin down — so the plan says it rather than implying it.
            detaches = ", ".join(
                f"{port} @ {realisation}"
                for port, realisation in sorted(
                    objective.objective.detaches,
                    key=lambda pair: (str(pair[0]), pair[1]),
                )
            )
            if detaches:
                lines.append(f"      detaches  {detaches}")
        if not compiled.objectives:
            lines.append("    none")
        lines.append("  trainable")
        lines.append(f"    {_names(compiled.trainable)}")
        return lines

    def _hyperparameter_lines(self) -> list[str]:
        if not self.hyperparameters:
            return ["hyperparameters", "  none"]
        scalars = {
            key: value
            for key, value in self.hyperparameters.items()
            if not isinstance(value, Mapping)
        }
        width = _width(scalars)
        lines = ["hyperparameters"]
        for key in sorted(self.hyperparameters):
            value = self.hyperparameters[key]
            if not isinstance(value, Mapping):
                lines.append(f"  {key:<{width}} = {value!r}")
                continue
            # Per-objective and per-component values render as blocks rather
            # than wide dicts: the card they are diffed against is a YAML
            # block, and a line nobody can read across is a line nobody checks.
            lines.append(f"  {key}")
            entries = _width(value)
            for label in sorted(value):
                lines.append(f"    {label:<{entries}} = {value[label]!r}")
        return lines


@dataclass(frozen=True)
class CompiledRun:
    """A recipe that has passed every check `compile()` can make."""

    recipe: Recipe
    stages: tuple[CompiledStage, ...]
    plan: ExecutionPlan
    _initial_parameters: tuple[tuple[str, torch.Tensor], ...] = field(
        repr=False, compare=False
    )
    _initial_buffers: tuple[tuple[str, torch.Tensor], ...] = field(
        repr=False, compare=False
    )

    @property
    def graph(self) -> ComponentGraph:
        return self.recipe.system

    def stage(self, name: str) -> CompiledStage:
        """The compiled stage called `name`."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise CompileError(
            f"recipe {self.recipe.name!r} has no stage {name!r}; it has "
            f"{[stage.name for stage in self.stages]!r}"
        )

    def initial_parameters(self) -> dict[str, torch.Tensor]:
        """Fresh clones of the parameter values captured by ``compile()``."""
        return {
            name: value.detach().clone() for name, value in self._initial_parameters
        }

    def initial_buffers(self) -> dict[str, torch.Tensor]:
        """Fresh clones of the buffer values captured by ``compile()``."""
        return {name: value.detach().clone() for name, value in self._initial_buffers}

    def state(
        self,
        stage: str | CompiledStage,
        batch: XTYBatch,
        *,
        rng_key: int = 0,
        teacher_graph: ComponentGraph | None = None,
    ) -> State:
        """Run one stage's planned forward passes over `batch`.

        This is the whole of what an executor needs from the graph: the
        compiler decided which realisations exist and which components each
        one runs, so nothing here chooses anything. Teacher passes use the
        stage-local EMA graph supplied by the executor and run under
        ``torch.no_grad()``; silently falling back to the student would make a
        teacher objective self-consistency under one parameter set.
        """
        compiled = self.stage(stage) if isinstance(stage, str) else stage
        # The requirement tracks the *passes*, not the declaration. A stage
        # whose teacher is an evaluation EMA (`role="evaluation"`) plans no
        # teacher pass, so demanding its parameter set here would force every
        # caller to build a copy this call would never read.
        needs_teacher = any(
            forward.realisation.params == "teacher" for forward in compiled.passes
        )
        if compiled.teacher is None:
            if teacher_graph is not None:
                raise TrainingError(
                    f"stage {compiled.name!r} declares no teacher, but a teacher "
                    "parameter graph was supplied"
                )
        elif teacher_graph is None and needs_teacher:
            raise TrainingError(
                f"stage {compiled.name!r} declares a teacher, but no teacher "
                "parameter graph was supplied. Execute the stage through "
                "run_stage/run_program so its TeacherSpec can build the EMA "
                "copy (PLAN.md P8)."
            )
        elif teacher_graph is self.graph:
            raise TrainingError(
                f"stage {compiled.name!r} was given the student graph as its "
                "teacher. A teacher realisation is a distinct EMA parameter "
                "set, never an alias of the student."
            )

        values = {}
        # Keyed by (view, draw): two draws of one view are two samples of the
        # same distribution and must not share a cache entry, while a student
        # and a teacher pass over the same draw still must.
        batches_by_view: dict[tuple[str, int], XTYBatch] = {(IDENTITY_VIEW, 0): batch}
        for forward in compiled.passes:
            graph = self.graph
            if forward.realisation.params == "teacher":
                graph = cast(ComponentGraph, teacher_graph)
            view_name = forward.realisation.view
            draw = forward.realisation.draw
            viewed = batches_by_view.get((view_name, draw))
            if viewed is None:
                viewed = self.recipe.view(view_name).apply(
                    batch, self.recipe.schema, rng_key=rng_key, draw=draw
                )
                batches_by_view[view_name, draw] = viewed
            if forward.realisation.params == "teacher":
                with torch.no_grad():
                    values[forward.realisation] = graph.evaluate(
                        viewed, schema=self.recipe.schema, only=forward.components
                    )
            else:
                values[forward.realisation] = graph.evaluate(
                    viewed, schema=self.recipe.schema, only=forward.components
                )
        return State(values)


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


def compile(recipe: Recipe) -> CompiledRun:
    """Check a recipe and plan its execution, or reject it with a reason.

    Named for the builtin it shadows inside this module, because `DESIGN.md` §8
    names it: `compile(recipe)` is the operation the whole document is written
    around, and a different name here would be a different vocabulary from the
    one every card and review uses.

    The structural checks on the graph itself — a port with two producers, an
    unsatisfiable `requires`, a cycle — already ran when the `ComponentGraph`
    was built. What is left is everything that depends on the *program*.

    Raises:
        CompileError: for every rejection in `DESIGN.md` §8 that this packet
            covers: an objective requiring a port nothing provides, a
            realisation nothing can produce, an unknown or dead `trainable`
            name, and a stage/objective row pairing that is empty by
            construction.
    """
    graph = recipe.system
    views = _validated_views(recipe)
    stages = tuple(
        _compile_stage(recipe, stage, _realisable(views, stage), views)
        for stage in recipe.program
    )
    _check_static_leakage(recipe, stages)
    _check_declared_draws(views, stages)
    plan = ExecutionPlan(
        recipe=recipe.name,
        purpose=recipe.purpose,
        card=recipe.card,
        features=recipe.schema.num_features,
        treatments=recipe.schema.treatment_cardinality,
        views=tuple(_plan_view(view, recipe) for view in views),
        components=tuple(_plan_component(graph, name) for name in graph.names),
        stages=stages,
        hyperparameters=_hyperparameters(recipe, stages),
    )
    return CompiledRun(
        recipe=recipe,
        stages=stages,
        plan=plan,
        _initial_parameters=_snapshot(graph.named_parameters()),
        _initial_buffers=_snapshot(graph.named_buffers()),
    )


def _snapshot(
    tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[tuple[str, torch.Tensor], ...]:
    """Detach and clone named model state for repeatable program initialisation."""
    return tuple((name, value.detach().clone()) for name, value in tensors)


def _validated_views(recipe: Recipe) -> tuple[ViewSpec, ...]:
    """Validate every declared view against the recipe's resolved schema."""
    for view in recipe.views:
        try:
            view.validate(recipe.schema)
        except (ViewError, SchemaError) as error:
            raise CompileError(
                f"view {view.name!r} of recipe {recipe.name!r} is invalid for "
                f"its schema: {error}"
            ) from error
    return tuple(recipe.views)


def _realisable(views: tuple[ViewSpec, ...], stage: Stage) -> frozenset[Realisation]:
    """The realisations this recipe can actually produce.

    Every declared view is available under student parameters. The same views
    are available under teacher parameters exactly when this stage declares a
    teacher; teacher availability does not leak from an earlier stage.

    Draws are deliberately not enumerated here. The set would grow with every
    view's declared count for no gain — a realisation is legal on this axis iff
    its draw is one the named view offers, which `_check_draw` decides against
    the `ViewSpec` and reports with the view named.
    """
    view_names = (IDENTITY_VIEW, *(view.name for view in views))
    realised = {Realisation(view=view) for view in view_names}
    if stage.teacher is not None:
        realised.update(Realisation(view=view, params="teacher") for view in view_names)
    return frozenset(realised)


def _compile_stage(
    recipe: Recipe,
    stage: Stage,
    realisable: frozenset[Realisation],
    views: tuple[ViewSpec, ...],
) -> CompiledStage:
    graph = recipe.system
    where = f"stage {stage.name!r} of recipe {recipe.name!r}"

    if not stage.objectives and stage.action is None:
        raise CompileError(
            f"{where} has neither objectives nor an action, so it would do "
            "nothing. A stage either descends declared objectives or executes "
            "an explicit P10 action (DESIGN.md §7)."
        )
    if stage.objectives:
        _reject_duplicate_objectives(stage, where)

    demanded: dict[Realisation, set[Port]] = {}
    trained: set[Port] = set()
    objectives: list[CompiledObjective] = []
    for weighted in stage.objectives:
        objective = weighted.objective
        rows = _effective_rows(stage, objective, where)
        for port, realisation in _sorted_requirements(objective):
            _check_port(graph, port, objective, where)
            _check_realisation(realisation, realisable, objective, where)
            _check_draw(realisation, views, f"objective {objective.name!r}", where)
            demanded.setdefault(realisation, set()).add(port)
        trained |= {port for port, _ in _check_detaches(objective, where)}
        objectives.append(
            CompiledObjective(
                weighted=weighted,
                rows=rows,
                plan_details=_objective_plan_details(objective, where),
            )
        )

    action_rows: tuple[Rows, ...] | None = None
    action_uses_y: bool | None = None
    if isinstance(stage.action, PseudoLabelAction):
        action = stage.action
        action_rows = _intersect_rows(
            stage.rows,
            action.rows,
            item=f"pseudo-label action {action.name!r}",
            where=where,
        )
        _check_action_port(graph, action.port, where)
        _check_action_realisation(action.realisation, realisable, where)
        _check_draw(action.realisation, views, "pseudo-label action", where)
        demanded.setdefault(action.realisation, set()).add(action.port)
        action_uses_y = graph.port_depends_on_raw_outcome(action.port)
    elif stage.action is not None:
        action_rows = (stage.rows,)

    passes = tuple(
        ForwardPass(realisation=realisation, components=graph.subgraph_for(ports))
        for realisation, ports in sorted(demanded.items())
    )
    if stage.objectives:
        _check_trainable(graph, stage, passes, trained, where)
    _check_teacher_use(stage, passes, where)
    if stage.objectives:
        _check_weight_decay_scope(stage, where)
    return CompiledStage(
        stage=stage,
        passes=passes,
        objectives=tuple(objectives),
        action_rows=action_rows,
        action_uses_y=action_uses_y,
    )


def _check_static_leakage(recipe: Recipe, stages: tuple[CompiledStage, ...]) -> None:
    """Reject outcome-dependent in-sample treatment labels before execution.

    The compiler knows the producing port's ``Y_RAW`` lineage, executor kind,
    ordered artifact edge and downstream trainable set. It deliberately does
    not inspect fold row ids here; those exist only on the artifact and are
    checked when it is loaded (`DESIGN.md` §7.2).
    """
    for compiled in stages:
        if compiled.stage.allow_leakage and recipe.purpose != "predictive":
            raise CompileError(
                f"stage {compiled.name!r} of causal recipe {recipe.name!r} sets "
                "allow_leakage=True. The opt-out is available only to a recipe "
                "whose purpose is explicitly 'predictive' (DESIGN.md §7.2)."
            )

    graph = recipe.system
    if Port.Y_GIVEN_XT not in graph.provides:
        return
    outcome_component = graph.producer(Port.Y_GIVEN_XT)
    by_name = {stage.name: stage for stage in stages}
    for consumer in stages:
        if outcome_component not in consumer.trainable:
            continue
        # Hard labels are joined as available treatments. An objective whose
        # effective scope still requires t_missing cannot touch those joined
        # rows, even if another objective in the same stage can.
        if not any(
            "t_missing" not in objective.rows
            and _objective_trains_component(graph, objective, outcome_component)
            for objective in consumer.objectives
        ):
            continue
        for source_name in consumer.inputs:
            source = by_name[source_name]
            if not isinstance(source.action, PseudoLabelAction):
                continue  # Program validation already prevents this.
            if not source.action_uses_y:
                continue
            if source.action_rows is not None and any(
                populations_are_disjoint(rows, "t_missing")
                for rows in source.action_rows
            ):
                # Labels emitted only for already-observed treatments never
                # replace the arbitrary placeholder of an originally missing
                # row, so the consumer cannot form the circular staged fit.
                continue
            if source.executor == "cross_fit":
                continue
            if recipe.purpose == "predictive" and consumer.stage.allow_leakage:
                continue
            opt_out = (
                " Set purpose='predictive' on the Recipe and "
                "allow_leakage=True on the consuming stage only if this is "
                "intentionally a predictive experiment."
            )
            raise CompileError(
                f"stage {consumer.name!r} consumes in-sample treatment labels "
                f"from stage {source.name!r}, whose source port "
                f"{source.action.port} transitively depends on Y_RAW, and "
                f"trains outcome component {outcome_component!r} on those "
                "rows. That is the circular q(t|x,y) -> p(y|x,t) fit rejected "
                f"by DESIGN.md §7.2.{opt_out} Otherwise use executor='cross_fit' "
                "so actual fold disjointness can be verified at artifact load."
            )


def _objective_trains_component(
    graph: ComponentGraph,
    compiled: CompiledObjective,
    component: str,
) -> bool:
    """Whether this term has a non-detached path through ``component``."""
    detached = compiled.objective.detaches
    return any(
        (port, realisation) not in detached and component in graph.subgraph_for({port})
        for port, realisation in compiled.objective.requires
    )


def _reject_duplicate_objectives(stage: Stage, where: str) -> None:
    seen: set[str] = set()
    for weighted in stage.objectives:
        name = weighted.name
        if name in seen:
            raise CompileError(
                f"{where} has more than one objective called {name!r}. Per-"
                "objective logging is keyed by name (DESIGN.md §6.2), so two "
                "terms sharing one would be indistinguishable in the trace that "
                "exists to attribute a bad number."
            )
        seen.add(name)


def _effective_rows(stage: Stage, objective: Objective, where: str) -> tuple[Rows, ...]:
    """`Stage.rows ∩ Objective.rows`, rejecting a pairing empty by construction."""
    validate_rows(objective.rows, f"objective {objective.name!r} in {where}")
    return _intersect_rows(
        stage.rows,
        objective.rows,
        item=f"objective {objective.name!r}",
        where=where,
    )


def _intersect_rows(
    stage_rows: Rows,
    item_rows: Rows,
    *,
    item: str,
    where: str,
) -> tuple[Rows, ...]:
    """Resolve two declared scopes without consulting a runtime batch."""
    if populations_are_disjoint(stage_rows, item_rows):
        raise CompileError(
            f"{where} scopes rows to {stage_rows!r} but {item} is entitled to "
            f"{item_rows!r}, and the two "
            "are empty by construction. That is a compile error rather than a "
            "term returning n = 0 on every batch forever, which is precisely the "
            "silently-dead objective the zero-eligible-row rule exists to make "
            "visible (DESIGN.md §7.0)."
        )
    populations: list[Rows] = []
    for rows in (stage_rows, item_rows):
        # "all" constrains nothing, and a population named on both sides is one
        # constraint, not two: `t_observed & t_observed` in a plan is noise.
        if rows != "all" and rows not in populations:
            populations.append(rows)
    return tuple(populations) or ("all",)


def _objective_plan_details(objective: Objective, where: str) -> tuple[str, ...]:
    """Snapshot stable, non-card mechanics that affect objective arithmetic."""
    if not isinstance(objective, _ObjectiveWithPlanDetails):
        return ()
    raw_details: object = objective.plan_details()
    if not isinstance(raw_details, tuple):
        raise CompileError(
            f"objective {objective.name!r} in {where} returned "
            f"{type(raw_details)} from plan_details(), expected a tuple of strings"
        )
    details: list[str] = []
    for detail in raw_details:
        if not isinstance(detail, str) or not detail or "\n" in detail:
            raise CompileError(
                f"objective {objective.name!r} in {where} returned invalid "
                f"stable plan detail {detail!r}; each must be a non-empty "
                "single line"
            )
        details.append(detail)
    return tuple(details)


def _check_detaches(
    objective: Objective, where: str
) -> frozenset[tuple[Port, Realisation]]:
    """The `(port, realisation)` pairs this objective actually trains through.

    `requires` says what a term *reads*, and the forward pass is planned from
    it; `detaches` says which of those reads carry no gradient back. The
    difference is what the dead-trainable rule has to reason about, because a
    stop-gradient is invisible in the graph — nothing about `p(t|x)` appearing
    in `requires` says whether a gradient ever reaches the head that produced
    it (`DESIGN.md` §4, §8.4).
    """
    detaches = frozenset(objective.detaches)
    requires = frozenset(objective.requires)
    stray = sorted(
        f"{port} @ {realisation}" for port, realisation in detaches - requires
    )
    if stray:
        raise CompileError(
            f"objective {objective.name!r} in {where} declares it detaches "
            f"{stray!r}, which it does not require. `detaches` names the subset "
            "of `requires` a term reads without training through (DESIGN.md §4)."
        )
    teacher_reads = frozenset(
        requirement for requirement in requires if requirement[1].params == "teacher"
    )
    undetached_teacher = sorted(
        f"{port} @ {realisation}" for port, realisation in teacher_reads - detaches
    )
    if undetached_teacher:
        raise CompileError(
            f"objective {objective.name!r} in {where} reads teacher target(s) "
            f"{undetached_teacher!r} without declaring them in `detaches`. "
            "Teacher parameters are structurally requires_grad=False, so the "
            "objective must make that stop-gradient visible to the compiler "
            "and execution plan (FIDELITY.md Tier 0)."
        )
    trained = requires - detaches
    if requires and not trained:
        raise CompileError(
            f"objective {objective.name!r} in {where} detaches every port it "
            "requires, so it contributes a constant to the total and trains "
            "nothing. A term with no gradient path is a diagnostic, not an "
            "objective (DESIGN.md §4)."
        )
    return trained


def _check_teacher_use(
    stage: Stage, passes: tuple[ForwardPass, ...], where: str
) -> None:
    """Check the teacher's declared role against the passes actually planned.

    Both directions are errors, and they are different errors. A
    `consistency_target` nothing reads is the silent no-op this check has
    always caught. An `evaluation` teacher that *is* read is the opposite
    mistake: the stage says the EMA is a reporting device while an objective
    trains against it, so the plan and the card would describe different
    methods (`DESIGN.md` §2.1).
    """
    if stage.teacher is None:
        return
    read = any(forward.realisation.params == "teacher" for forward in passes)
    if stage.teacher.role == "evaluation":
        if not read:
            return
        raise CompileError(
            f"{where} declares its teacher role='evaluation', but an objective "
            "or action requires a teacher realisation. An evaluation EMA is "
            "not part of the training signal — declare "
            "role='consistency_target' if the method reads it."
        )
    if read:
        return
    raise CompileError(
        f"{where} configures a TeacherSpec but no active objective or action requires "
        "a teacher realisation. Maintaining an unused EMA copy would be a "
        "silent no-op; remove the teacher, require params='teacher' "
        "explicitly, or declare role='evaluation' if it exists only to be "
        "reported with."
    )


def _check_port(
    graph: ComponentGraph, port: Port, objective: Objective, where: str
) -> None:
    if port in SOURCE_PORTS or port in graph.provides:
        return
    raise CompileError(
        f"objective {objective.name!r} in {where} requires port {str(port)!r}, "
        f"which no component provides. This graph provides "
        f"{sorted(str(p) for p in graph.provides)!r}. Add the component that "
        "produces it — an objective cannot compute it itself (DESIGN.md §4)."
    )


def _check_realisation(
    realisation: Realisation,
    realisable: frozenset[Realisation],
    objective: Objective,
    where: str,
) -> None:
    if replace(realisation, draw=0) in realisable:
        return
    raise CompileError(
        f"objective {objective.name!r} in {where} requires a port under "
        f"{realisation}, which this recipe cannot produce; it realises "
        f"{[str(r) for r in sorted(realisable)]}. A view is declared by a "
        "ViewSpec (DESIGN.md §5) and a teacher parameter set by a stage (§7)."
    )


def _check_declared_draws(
    views: tuple[ViewSpec, ...], stages: tuple[CompiledStage, ...]
) -> None:
    """Reject a view that declares more draws than the program realises.

    The mirror of `_check_teacher_use`, and for the same reason: `draws=3` on a
    view two realisations read is a claim the plan prints — `weak_x (3
    independent draws)` — with nothing behind it, and the third stream costs
    nothing only because it never runs. A reviewer diffing the plan against a
    card would be reading a number no forward pass supports.

    Program-wide rather than per-stage, because a later stage may be the only
    one that reads the second draw and that is a legal program.

    Deliberately narrower than it could be: a `ViewSpec` no realisation reads
    *at all* is still accepted, as it was before draws existed. That is the
    same smell one level up, and closing it would change a compiler rule for
    every recipe rather than for the axis this check came in with — a decision
    for whoever wants it, not a side effect of this one.
    """
    realised: dict[str, int] = {}
    for stage in stages:
        for forward in stage.passes:
            name = forward.realisation.view
            realised[name] = max(realised.get(name, 0), forward.realisation.draw + 1)
    for view in views:
        if view.draws == 1:
            continue
        used = realised.get(view.name, 0)
        if view.draws <= used:
            continue
        raise CompileError(
            f"view {view.name!r} of recipe declares draws={view.draws}, but no "
            f"objective or action realises more than {used} of them. An "
            "unrealised draw is a silent no-op the plan advertises: reduce "
            "`draws`, or name the draw that is missing (DESIGN.md §2.1)."
        )


def _check_draw(
    realisation: Realisation, views: tuple[ViewSpec, ...], subject: str, where: str
) -> None:
    """Reject a draw the named view does not offer (`DESIGN.md` §2.1).

    The draw axis is unbounded in the type and bounded by the `ViewSpec`, so
    this is where `draw=2` on a two-draw view stops. Without it the compiler
    would plan a third forward pass on an RNG stream no card named, which is
    the silent-extra-pass failure the realisation machinery exists to prevent.
    """
    if realisation.draw == 0:
        return
    declared = {view.name: view.draws for view in views}
    draws = declared.get(realisation.view, 1)
    if realisation.draw < draws:
        return
    raise CompileError(
        f"{subject} in {where} requires {realisation}, but view "
        f"{realisation.view!r} declares draws={draws}, so its draws are "
        f"0..{draws - 1}. A view offering more than one independent sample "
        "says so on its ViewSpec (DESIGN.md §2.1, §5)."
    )


def _check_action_port(graph: ComponentGraph, port: Port, where: str) -> None:
    if port in SOURCE_PORTS or port in graph.provides:
        return
    raise CompileError(
        f"pseudo-label action in {where} requires port {str(port)!r}, which no "
        f"component provides. This graph provides "
        f"{sorted(str(candidate) for candidate in graph.provides)!r}."
    )


def _check_action_realisation(
    realisation: Realisation,
    realisable: frozenset[Realisation],
    where: str,
) -> None:
    if replace(realisation, draw=0) in realisable:
        return
    raise CompileError(
        f"pseudo-label action in {where} requires {realisation}, which this "
        f"stage cannot produce; it realises "
        f"{[str(candidate) for candidate in sorted(realisable)]}."
    )


def _check_trainable(
    graph: ComponentGraph,
    stage: Stage,
    passes: tuple[ForwardPass, ...],
    trained: set[Port],
    where: str,
) -> None:
    # The other extreme of the dead-trainable rule below: both reject a stage
    # that cannot do what it appears to, one because a named component gets no
    # gradient and this one because nothing is named at all.
    if not stage.trainable:
        raise CompileError(
            f"{where} has objectives but an empty `trainable`, so it would "
            "descend a gradient into nothing: the optimiser gets no parameter "
            "group and every step is a no-op. Name the components this stage "
            "updates. A stage that deliberately emits an artifact without "
            "training removes the objectives and declares an explicit action "
            "and executor (DESIGN.md §7)."
        )
    unknown = [name for name in stage.trainable if name not in graph]
    if unknown:
        raise CompileError(
            f"{where} declares trainable {unknown!r}, which is not a component "
            f"of this graph; it holds {list(graph.names)!r}."
        )
    executed = {name for forward in passes for name in forward.components}
    dead = [name for name in stage.trainable if name not in executed]
    if dead:
        raise CompileError(
            f"{where} trains {dead!r}, which no active objective depends on. "
            "Its parameters would receive no gradient, so the stage carries "
            "dead weight and the recipe does not do what it appears to "
            "(DESIGN.md §8.4)."
        )
    # Reachable from the ports the objectives *train through*, not merely from
    # the ports they read: a component upstream of a detached port is executed
    # in the forward pass and still receives no gradient, so `executed` alone
    # accepts a stage whose every step is a no-op.
    reached = set(graph.subgraph_for(trained))
    detached = [name for name in stage.trainable if name not in reached]
    if detached:
        raise CompileError(
            f"{where} trains {detached!r}, which every active objective reads "
            "through a stop-gradient. The components run in the forward pass "
            "and receive no gradient, so the optimiser step is a no-op — the "
            "same dead-weight stage as above, hidden behind a detach rather "
            "than behind the wiring. Change the objective's gradient path, or "
            "train the components it does backpropagate into (DESIGN.md §4, "
            "§8.4)."
        )


def _check_weight_decay_scope(stage: Stage, where: str) -> None:
    """A component-scoped decay may name only components this stage trains."""
    decay = stage.optimiser.weight_decay
    if not decay.applies or decay.components is None:
        return
    unknown = sorted(set(decay.components) - set(stage.trainable))
    if unknown:
        raise CompileError(
            f"{where} scopes weight decay to {unknown!r}, but its trainable "
            f"components are {list(stage.trainable)!r}. A scoped optimiser "
            "policy must reach a component the stage actually updates."
        )


def _plan_component(graph: ComponentGraph, name: str) -> PlannedComponent:
    component = graph[name]
    requires = tuple(sorted(component.requires))
    return PlannedComponent(
        name=name,
        requires=requires,
        provides=tuple(sorted(component.provides)),
        sources={port: graph.producer(port) for port in requires},
        reads_raw_outcome=Port.Y_RAW in component.requires,
        outcome_dependent=graph.depends_on_raw_outcome(name),
    )


def _plan_view(view: ViewSpec, recipe: Recipe) -> PlannedView:
    return PlannedView(
        name=view.name,
        transforms=view.transform_descriptions(),
        preserves=tuple(sorted(view.preserves)),
        recomputes=view.recompute_descriptions(),
        affected_columns=tuple(sorted(view.affected_columns(recipe.schema))),
        draws=view.draws,
    )


def _hyperparameters(
    recipe: Recipe, stages: tuple[CompiledStage, ...]
) -> dict[str, Any]:
    """The flat `{canonical_key: value}` dict of `DESIGN.md` §9.1.

    Everything that can carry card keys contributes: components, stages and
    objectives. Architecture keys are component-valued by construction and
    therefore aggregate as `{component_name: value}`. In a multi-stage program,
    stage and optimiser bindings aggregate by stage, and objective bindings by
    ``<stage>.<objective>``. A program is allowed to change learning rate,
    duration or teacher policy between stages; collapsing those values would
    either reject a valid program or hide the transition from the plan. The
    single-stage scalar surface remains unchanged for recipe-card compatibility.

    The four `losses.*` keys are *derived* from the program rather than bound
    by a `CARD_KEYS` declaration, for the reason `FIDELITY.md` §2 annotates
    them with: each is "per objective", and a canonical key names one value.
    Aggregating them over the whole program is what makes them one value —
    and the recipe does supply them, so a cross-check that reported them
    absent would be reporting on the mechanism rather than on the recipe.
    """
    resolved: dict[str, Any] = {}
    owners: dict[str, str] = {}
    for component in recipe.system.components:
        _merge_component(resolved, owners, component)
    scoped = len(recipe.program) > 1
    for stage in recipe.program:
        stage_label = f"stage {stage.name!r}"
        if stage.objectives:
            _merge_owner(
                resolved,
                owners,
                stage,
                stage_label,
                scope=stage.name if scoped else None,
            )
            _merge_owner(
                resolved,
                owners,
                stage.optimiser,
                f"the optimiser of {stage_label}",
                scope=stage.name if scoped else None,
            )
        if stage.teacher is not None:
            _merge_owner(
                resolved,
                owners,
                stage.teacher,
                f"the teacher of {stage_label}",
                scope=stage.name if scoped else None,
            )
        if stage.action is not None:
            _merge_owner(
                resolved,
                owners,
                stage.action,
                f"the action of {stage_label}",
                scope=stage.name if scoped else None,
            )
        for weighted in stage.objectives:
            _merge_owner(
                resolved,
                owners,
                weighted.objective,
                f"objective {weighted.name!r} in stage {stage.name!r}",
                scope=f"{stage.name}.{weighted.name}" if scoped else None,
            )
    for key, value in _loss_hyperparameters(stages).items():
        _merge_value(resolved, owners, key, value, "the program's weighted terms")
    for key, value in _gradient_hyperparameters(stages).items():
        _merge_value(resolved, owners, key, value, "the program's gradient paths")
    return resolved


def _merge_component(
    resolved: dict[str, Any], owners: dict[str, str], component: object
) -> None:
    """Merge one component, namespacing every `architecture.*` binding."""
    name = getattr(component, "name", type(component).__name__)
    label = f"component {name!r}"
    for key, value in card_hyperparameters(component).items():
        if not key.startswith("architecture."):
            _merge_value(resolved, owners, key, value, label)
            continue
        existing = resolved.get(key)
        if existing is None:
            values: dict[str, Any] = {}
            resolved[key] = values
        elif isinstance(existing, dict):
            values = existing
        else:
            raise CompileError(
                f"card key {key!r} is component-valued, but {owners[key]} "
                f"already set the scalar {existing!r}. Architecture bindings "
                "are keyed by component so the plan cannot collapse modules."
            )
        if name in values:
            raise CompileError(
                f"card key {key!r} has two architecture bindings for component {name!r}"
            )
        values[str(name)] = value
        owners.setdefault(key, label)


def _loss_hyperparameters(
    stages: tuple[CompiledStage, ...],
) -> dict[str, dict[str, Any]]:
    """The four per-objective `losses.*` keys of `FIDELITY.md` §2.

    Each maps `"<stage>.<objective>"` — unique across a program, since stage
    names are unique and objective names are unique within a stage — to the
    value that term actually runs with:

    * `losses.weights` is the schedule's **nominal** weight, which is the λ a
      paper states;
    * `losses.schedules` is how the term gets there, which is the other half
      of the same sentence;
    * `losses.reduction` is the §6.1 mode;
    * `losses.eligible_rows` is the *effective* set — the stage scope and the
      objective's population intersected (§7.0) — because that is the set the
      term is actually given, and a card describing the objective's half alone
      would not describe what runs.
    """
    weights: dict[str, Any] = {}
    schedules: dict[str, Any] = {}
    reductions: dict[str, Any] = {}
    eligible: dict[str, Any] = {}
    for stage in stages:
        for objective in stage.objectives:
            label = f"{stage.name}.{objective.name}"
            weights[label] = objective.weight.nominal
            schedules[label] = objective.weight.describe()
            reductions[label] = objective.reduction
            eligible[label] = _rows(objective.rows)
    if not weights:
        return {}
    return {
        "losses.eligible_rows": eligible,
        "losses.reduction": reductions,
        "losses.schedules": schedules,
        "losses.weights": weights,
    }


def _gradient_hyperparameters(
    stages: tuple[CompiledStage, ...],
) -> dict[str, dict[str, str]]:
    """Render every objective's declared stop-gradient into the card surface."""
    paths: dict[str, str] = {}
    for stage in stages:
        for objective in stage.objectives:
            label = f"{stage.name}.{objective.name}"
            detached = sorted(
                f"{port} @ {realisation}"
                for port, realisation in objective.objective.detaches
            )
            paths[label] = ", ".join(detached) or "none"
    if not paths:
        return {}
    return {"gradients.stop_gradients": paths}


def _merge_owner(
    resolved: dict[str, Any],
    owners: dict[str, str],
    owner: object,
    label: str,
    *,
    scope: str | None,
) -> None:
    for key, value in card_hyperparameters(owner).items():
        if scope is None:
            _merge_value(resolved, owners, key, value, label)
        else:
            _merge_scoped_value(resolved, owners, key, scope, value, label)


def _merge_scoped_value(
    resolved: dict[str, Any],
    owners: dict[str, str],
    key: str,
    scope: str,
    value: object,
    label: str,
) -> None:
    """Merge a stage/objective-valued binding without collapsing its scope."""
    existing = resolved.get(key)
    if existing is None:
        values: dict[str, Any] = {}
        resolved[key] = values
    elif isinstance(existing, dict):
        values = existing
    else:
        raise CompileError(
            f"card key {key!r} is program-scoped, but {owners[key]} already "
            f"set the scalar {existing!r}. Multi-stage bindings are keyed by "
            "stage so the plan cannot hide a transition."
        )
    if scope in values and values[scope] != value:
        raise CompileError(
            f"card key {key!r} is bound twice with different values in "
            f"scope {scope!r}: {values[scope]!r} and {value!r} from {label}."
        )
    values[scope] = value
    owners.setdefault(key, label)


def _merge_value(
    resolved: dict[str, Any],
    owners: dict[str, str],
    key: str,
    value: object,
    label: str,
) -> None:
    if key in resolved and resolved[key] != value:
        raise CompileError(
            f"card key {key!r} is bound twice with different values: "
            f"{owners[key]} sets {resolved[key]!r} and {label} sets "
            f"{value!r}. A canonical key names one number (DESIGN.md §9.1)."
        )
    resolved[key] = value
    owners.setdefault(key, label)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _sorted_requirements(
    objective: Objective,
) -> tuple[tuple[Port, Realisation], ...]:
    """`(port, realisation)` pairs in a stable order, for plans and messages."""
    return tuple(sorted(objective.requires, key=lambda pair: (str(pair[0]), pair[1])))


def _width(names: Iterable[str]) -> int:
    return max((len(name) for name in names), default=0)


def _ports(ports: Iterable[Port]) -> str:
    return "[" + ", ".join(str(port) for port in ports) + "]"


def _names(names: Iterable[str]) -> str:
    return ", ".join(names) or "none"


def _rows(populations: tuple[Rows, ...]) -> str:
    return " & ".join(populations)


__all__ = [
    "CompiledObjective",
    "CompiledRun",
    "CompiledStage",
    "ExecutionPlan",
    "ForwardPass",
    "PlannedComponent",
    "PlannedView",
    "compile",
    "plan_digest_of",
]
