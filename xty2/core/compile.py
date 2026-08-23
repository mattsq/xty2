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

Two checks of §8 are not here yet, and are absent rather than stubbed: views
are validated against the schema when views exist (§8.5, P6), and the static
half of the leakage rule needs artifacts and executors to reason about (§8.7,
P10). `ComponentGraph` already derives what the second one will ask for —
which components transitively read `Y_RAW` — and the plan prints it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import card_hyperparameters
from xty2.core.errors import CompileError
from xty2.core.graph import (
    DEFAULT,
    SOURCE_PORTS,
    ComponentGraph,
    Realisation,
    State,
)
from xty2.core.loss import Reduction
from xty2.core.optimisation import OptimiserSpec
from xty2.core.ports import Port
from xty2.core.recipe import Objective, Recipe, Stage, Weighted, validate_rows
from xty2.core.rows import Rows, populations_are_disjoint
from xty2.core.schedules import Schedule


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


@dataclass(frozen=True)
class CompiledStage:
    """A stage with its forward passes planned and its objectives resolved."""

    stage: Stage
    passes: tuple[ForwardPass, ...]
    objectives: tuple[CompiledObjective, ...]

    @property
    def name(self) -> str:
        return self.stage.name

    @property
    def trainable(self) -> tuple[str, ...]:
        return self.stage.trainable

    @property
    def optimiser(self) -> OptimiserSpec:
        """How this stage descends. Set — a stage with objectives cannot omit it."""
        return self.stage.optimiser

    @property
    def steps(self) -> int:
        """Optimiser steps, not epochs (`FIDELITY.md` §2)."""
        return self.stage.steps


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
            "",
            *self._component_lines(),
            "",
            *self._lineage_lines(),
        ]
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
            f"  steps: {compiled.steps}",
            "  optimisation",
            *(f"    {line}" for line in compiled.optimiser.describe_lines()),
            f"  forward passes ({len(compiled.passes)})",
        ]
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

    def state(self, stage: str | CompiledStage, batch: XTYBatch) -> State:
        """Run one stage's planned forward passes over `batch`.

        This is the whole of what an executor (P4) needs from the graph: the
        compiler decided which realisations exist and which components each
        one runs, so nothing here chooses anything.
        """
        compiled = self.stage(stage) if isinstance(stage, str) else stage
        values = {}
        for forward in compiled.passes:
            if forward.realisation != DEFAULT:
                raise CompileError(
                    f"stage {compiled.name!r} plans {forward.realisation}, which "
                    "needs a view (DESIGN.md §5) or a teacher parameter set "
                    "(§7); neither exists yet"
                )
            values[forward.realisation] = self.graph.evaluate(
                batch, schema=self.recipe.schema, only=forward.components
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
    realisable = _realisable(recipe)
    stages = tuple(
        _compile_stage(recipe, stage, realisable) for stage in recipe.program
    )
    plan = ExecutionPlan(
        recipe=recipe.name,
        purpose=recipe.purpose,
        card=recipe.card,
        features=recipe.schema.num_features,
        treatments=recipe.schema.treatment_cardinality,
        components=tuple(_plan_component(graph, name) for name in graph.names),
        stages=stages,
        hyperparameters=_hyperparameters(recipe, stages),
    )
    return CompiledRun(recipe=recipe, stages=stages, plan=plan)


def _realisable(recipe: Recipe) -> frozenset[Realisation]:
    """The realisations this recipe can actually produce.

    A non-default realisation comes from one of exactly two places, and a
    recipe can declare neither yet: a `view` names a `ViewSpec` (`DESIGN.md`
    §5) and `params="teacher"` names a stage-built EMA copy (§7). Until it
    can, an objective asking for one is unsatisfiable rather than merely
    unimplemented, and saying so at compile time is the point — the
    alternative is a consistency loss that silently reads the student.
    """
    del recipe  # widened by the packets that add views and teacher parameters
    return frozenset({DEFAULT})


def _compile_stage(
    recipe: Recipe, stage: Stage, realisable: frozenset[Realisation]
) -> CompiledStage:
    graph = recipe.system
    where = f"stage {stage.name!r} of recipe {recipe.name!r}"

    if not stage.objectives:
        raise CompileError(
            f"{where} has no objectives, so it would train nothing. A stage that "
            "does something other than descend a gradient is a StageAction "
            "(DESIGN.md §7), which does not exist yet."
        )
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
            demanded.setdefault(realisation, set()).add(port)
        trained |= {port for port, _ in _check_detaches(objective, where)}
        objectives.append(CompiledObjective(weighted=weighted, rows=rows))

    passes = tuple(
        ForwardPass(realisation=realisation, components=graph.subgraph_for(ports))
        for realisation, ports in sorted(demanded.items())
    )
    _check_trainable(graph, stage, passes, trained, where)
    _check_weight_decay_scope(stage, where)
    return CompiledStage(stage=stage, passes=passes, objectives=tuple(objectives))


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
    if populations_are_disjoint(stage.rows, objective.rows):
        raise CompileError(
            f"{where} scopes rows to {stage.rows!r} but objective "
            f"{objective.name!r} is entitled to {objective.rows!r}, and the two "
            "are empty by construction. That is a compile error rather than a "
            "term returning n = 0 on every batch forever, which is precisely the "
            "silently-dead objective the zero-eligible-row rule exists to make "
            "visible (DESIGN.md §7.0)."
        )
    populations: list[Rows] = []
    for rows in (stage.rows, objective.rows):
        # "all" constrains nothing, and a population named on both sides is one
        # constraint, not two: `t_observed & t_observed` in a plan is noise.
        if rows != "all" and rows not in populations:
            populations.append(rows)
    return tuple(populations) or ("all",)


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
    trained = requires - detaches
    if requires and not trained:
        raise CompileError(
            f"objective {objective.name!r} in {where} detaches every port it "
            "requires, so it contributes a constant to the total and trains "
            "nothing. A term with no gradient path is a diagnostic, not an "
            "objective (DESIGN.md §4)."
        )
    return trained


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
    if realisation in realisable:
        return
    raise CompileError(
        f"objective {objective.name!r} in {where} requires a port under "
        f"{realisation}, which this recipe cannot produce; it realises "
        f"{[str(r) for r in sorted(realisable)]}. A view is declared by a "
        "ViewSpec (DESIGN.md §5) and a teacher parameter set by a stage (§7)."
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
            "updates. A stage that deliberately runs without training is a "
            "StageAction (DESIGN.md §7), which does not exist yet."
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


def _hyperparameters(
    recipe: Recipe, stages: tuple[CompiledStage, ...]
) -> dict[str, Any]:
    """The flat `{canonical_key: value}` dict of `DESIGN.md` §9.1.

    Everything that can carry card keys contributes: components, stages and
    objectives. Architecture keys are component-valued by construction and
    therefore aggregate as `{component_name: value}`; other owners binding one
    key to different values are rejected. The component namespace is what lets
    a reviewer see three different width/depth declarations without a
    last-write-wins collapse.

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
    for stage in recipe.program:
        _merge(resolved, owners, stage, f"stage {stage.name!r}")
        _merge(
            resolved,
            owners,
            stage.optimiser,
            f"the optimiser of stage {stage.name!r}",
        )
        for weighted in stage.objectives:
            _merge(
                resolved,
                owners,
                weighted.objective,
                f"objective {weighted.name!r} in stage {stage.name!r}",
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
    return {"gradients.stop_gradients": paths}


def _merge(
    resolved: dict[str, Any], owners: dict[str, str], owner: object, label: str
) -> None:
    for key, value in card_hyperparameters(owner).items():
        _merge_value(resolved, owners, key, value, label)


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
    "compile",
    "plan_digest_of",
]
