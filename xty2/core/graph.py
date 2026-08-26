"""Components, realisations, state and the component graph (`DESIGN.md` §2.1, §3).

Three rules from the design are enforced here rather than documented.

**A component is handed nothing but its ports.** There is no `XTYBatch`
argument on `forward`, so an undeclared dependency has no channel to arrive
through; raw `x` and raw `y` come from the virtual source node (§2.2) and are
declared like any other dependency. `requires` is therefore the single
dependency declaration, which is what makes `Y_RAW` reachability — the fact the
leakage rule reasons about (§7.2) — a property of the graph instead of a flag
somebody remembered to set.

**A component reads only what it declares.** `PortView` raises on an
undeclared read, so the rule is checked on every forward pass rather than at
review time.

**A component never computes a loss.** No `loss()` method exists on the base
class. Losses are objectives (§4), and keeping them apart is the single largest
departure from XTYLearner.
"""

from __future__ import annotations

import inspect
from abc import ABCMeta, abstractmethod
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Final, Literal, cast

from torch import Tensor, nn

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import card_hyperparameters
from xty2.core.errors import GraphError, require_str
from xty2.core.ports import Port, PortValue, port_spec
from xty2.core.schema import Schema

SOURCE_NAME: Final = "source"
"""The virtual source node's name in the plan. Not a component."""

GRAPH_COMPONENTS_NAME: Final = "_components"
"""The internal module-registry prefix in `ComponentGraph.named_parameters()`."""

SOURCE_PORTS: Final = frozenset({Port.X_RAW, Port.Y_RAW})
"""What the virtual source node supplies from the batch (`DESIGN.md` §2.2)."""

IDENTITY_VIEW: Final = "identity"
"""The view that is the batch itself. Real views arrive with `ViewSpec` (§5)."""

Params = Literal["student", "teacher"]


# ---------------------------------------------------------------------------
# Realisations and state
# ---------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Realisation:
    """The conditions one evaluation of the graph runs under (`DESIGN.md` §2.1).

    Consistency losses need a port under two augmentations; mean-teacher losses
    need it under two parameter sets. Both are the same idea — the graph is
    evaluated more than once — so both are one small key rather than two
    mechanisms.

    `order=True` is not decoration: the compiler sorts realisations so that a
    plan printed twice is byte-identical and therefore diffable against a card.
    """

    view: str = IDENTITY_VIEW
    params: Params = "student"
    draw: int = 0
    """Which independent sample of `view` this is (`DESIGN.md` §2.1).

    A view is a *distribution* over batches, and a method may need two samples
    from the same one — FixMatch's labelled rows enter its supervised term and
    its pseudo-label target under independently drawn weak augmentations. The
    number of draws a view offers is declared on the `ViewSpec`; this says
    which of them a realisation wants, and the compiler plans one forward pass
    per draw as it does for any other realisation.

    Draw `0` is the view's own stream. That is not a convention with nothing
    behind it: the seed for draw 0 is byte-identical to the seed a view had
    before this axis existed, so adding it changed no existing recipe's
    numbers, and `str` omits it so no existing plan or digest moved either.
    """

    def __post_init__(self) -> None:
        if not require_str("Realisation.view", self.view, error=GraphError):
            raise GraphError(
                f"Realisation.view must be a non-empty ViewSpec name, got {self.view!r}"
            )
        if self.params not in ("student", "teacher"):
            raise GraphError(
                f"Realisation.params must be 'student' or 'teacher', got "
                f"{self.params!r}. A third parameter set is a new realisation "
                "axis and needs a second consumer first (DESIGN.md §11)."
            )
        if type(self.draw) is not int or self.draw < 0:
            raise GraphError(
                f"Realisation.draw must be a non-negative int, got {self.draw!r}"
            )
        if self.draw and self.view == IDENTITY_VIEW:
            raise GraphError(
                f"Realisation names draw {self.draw} of the identity view. The "
                "identity view is the batch itself — there is nothing to draw "
                "a second sample of (DESIGN.md §2.1)."
            )

    def __str__(self) -> str:
        # Draw 0 is omitted so that every plan written before this axis existed
        # renders and hashes exactly as it did.
        draw = f" draw={self.draw}" if self.draw else ""
        return f"view={self.view}{draw} params={self.params}"


DEFAULT: Final = Realisation()
"""The batch itself, under the student parameters."""


class State:
    """Computed ports, keyed by realisation (`DESIGN.md` §2.1).

    A small wrapper rather than a bare dict, because the default-realisation
    lookup is used constantly and because a missing realisation has to be a
    loud, actionable failure: the compiler plans exactly the realisations the
    objectives declare, so reading one that is absent means an objective's
    `requires` is incomplete, not that the executor should compute more.
    """

    __slots__ = ("_by_realisation",)

    def __init__(self, values: Mapping[Realisation, Mapping[Port, PortValue]]) -> None:
        self._by_realisation: dict[Realisation, Mapping[Port, PortValue]] = {
            realisation: MappingProxyType(dict(ports))
            for realisation, ports in values.items()
        }

    def __getitem__(self, realisation: Realisation) -> Mapping[Port, PortValue]:
        try:
            return self._by_realisation[realisation]
        except KeyError:
            raise GraphError(
                f"state holds no {realisation}; this stage computes "
                f"{[str(r) for r in self.realisations]}. The compiler plans "
                "exactly the realisations the objectives declare (DESIGN.md "
                "§2.1), so declare it in the objective's `requires` rather than "
                "re-invoking the graph here."
            ) from None

    def __contains__(self, realisation: object) -> bool:
        return realisation in self._by_realisation

    def __len__(self) -> int:
        return len(self._by_realisation)

    @property
    def default(self) -> Mapping[Port, PortValue]:
        """`self[DEFAULT]` — the batch under the student parameters."""
        return self[DEFAULT]

    @property
    def realisations(self) -> tuple[Realisation, ...]:
        """Every realisation this state holds, in a stable order."""
        return tuple(sorted(self._by_realisation))

    def __repr__(self) -> str:
        return f"State({[str(r) for r in self.realisations]})"


class PortView(Mapping[Port, PortValue]):
    """The only argument a component's `forward` takes (`DESIGN.md` §3).

    Reading a port the component did not declare raises. Iteration is over the
    declared ports alone, so the undeclared ones are not merely awkward to
    reach — they are not visible.
    """

    __slots__ = ("_component", "_declared", "_values")

    def __init__(
        self,
        values: Mapping[Port, PortValue],
        *,
        declared: frozenset[Port],
        component: str,
    ) -> None:
        self._values = values
        self._declared = declared
        self._component = component

    def __getitem__(self, port: Port) -> PortValue:
        if port not in self._declared:
            raise GraphError(
                f"component {self._component!r} read port {str(port)!r} without "
                f"declaring it; requires = {render_ports(self._declared)}. A "
                "component reads only the ports in `requires`, raw inputs "
                "included (DESIGN.md §2.2, §3) — declare the dependency so the "
                "compiler can order it and the plan can show it."
            )
        try:
            return self._values[port]
        except KeyError:
            raise GraphError(
                f"component {self._component!r} requires port {str(port)!r}, "
                "which nothing has produced at this point in the graph"
            ) from None

    def tensor(self, port: Port) -> Tensor:
        """Read `port` and narrow it to a `Tensor`.

        A port carries a tensor or a distribution (`PortValue`), so every
        component that wants the tensor would otherwise narrow it itself. Doing
        it here means the failure — a distribution arriving where a tensor was
        expected — is reported at the read, against a named port, instead of as
        an attribute error somewhere inside a `forward`.
        """
        value = self[port]
        if not isinstance(value, Tensor):
            raise GraphError(
                f"component {self._component!r} read port {str(port)!r} as a "
                f"tensor, but it carries {type(value)}. Its PortSpec is the "
                "contract (DESIGN.md §2)."
            )
        return value

    def __iter__(self) -> Iterator[Port]:
        return iter(sorted(self._declared & self._values.keys()))

    def __contains__(self, port: object) -> bool:
        """Membership never raises — but an undeclared port is never a member.

        `Mapping` would otherwise answer this through `__getitem__`, turning a
        containment test into the undeclared-read error. Answering it directly
        keeps `in` usable without making it a way around the rule: a component
        can see that it does not have a port, never read one it did not
        declare.
        """
        return port in self._declared and port in self._values

    def __len__(self) -> int:
        return len(self._declared & self._values.keys())

    def __repr__(self) -> str:
        return f"PortView({self._component!r}, {render_ports(self._declared)})"


def render_ports(ports: Iterable[Port]) -> str:
    """`[x, x_repr]` — ports in a stable order, for messages and plans."""
    return "[" + ", ".join(sorted(str(port) for port in ports)) + "]"


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


class _ComponentMeta(ABCMeta):
    """Validate a component once, after the outermost `__init__` has returned.

    The checks have to run *after* construction — a subclass sets the fields a
    card key binds to — and exactly once, which rules out both a base-class
    `__init__` and a wrapper on every `__init__` in the hierarchy.
    """

    def __call__(cls, *args: object, **kwargs: object) -> Component:
        instance = cast("Component", super().__call__(*args, **kwargs))
        instance._validate_construction()
        return instance


class Component(nn.Module, metaclass=_ComponentMeta):
    """A parameterisation, and nothing else (`DESIGN.md` §3).

    Being an `nn.Module` is a requirement rather than an implementation
    detail: stages build optimiser param groups, freeze and unfreeze by
    component name and maintain EMA copies, and none of that is expressible
    against a structural protocol with no parameter interface.

    Attributes:
        name: Unique within a graph, and a Python identifier — it keys the
            module dict, the stage's `trainable` list and the printed plan.
        requires: Every port this component reads, raw inputs included.
        provides: Every port it writes. `forward` must return exactly these.
        CARD_KEYS: `{field: canonical_key}` for the paper-governed fields this
            component carries (`DESIGN.md` §9.1). Fields the paper does not
            govern keep ordinary defaults and no binding.
    """

    CARD_KEYS: ClassVar[Mapping[str, str]] = {}

    def __init__(
        self,
        name: str,
        *,
        requires: Iterable[Port],
        provides: Iterable[Port],
    ) -> None:
        super().__init__()
        self.name = name
        self.requires = frozenset(requires)
        self.provides = frozenset(provides)

    @abstractmethod
    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        """Compute exactly `provides` from exactly `requires`.

        There is no `XTYBatch` parameter, and adding one is a construction-time
        error: the batch is what an undeclared dependency would arrive through
        (`DESIGN.md` §2.2).
        """

    def hyperparameters(self) -> dict[str, object]:
        """This component's `{canonical_key: value}` card bindings."""
        return dict(card_hyperparameters(self))

    def extra_repr(self) -> str:
        return (
            f"{self.name!r}, {render_ports(self.requires)} -> "
            f"{render_ports(self.provides)}"
        )

    # -- construction-time contract ---------------------------------------

    def _validate_construction(self) -> None:
        """Run by the metaclass once the outermost `__init__` has returned."""
        if not require_str(
            "component name", self.name, error=GraphError
        ).isidentifier():
            raise GraphError(
                f"component name {self.name!r} must be a Python identifier: it "
                "keys the module dict, a stage's `trainable` list and the "
                "printed plan"
            )
        if self.name == SOURCE_NAME:
            raise GraphError(
                f"{SOURCE_NAME!r} is the virtual source node (DESIGN.md §2.2) "
                "and cannot also be a component name"
            )
        if self.name == GRAPH_COMPONENTS_NAME:
            raise GraphError(
                f"component name {GRAPH_COMPONENTS_NAME!r} is reserved for "
                "ComponentGraph's internal module registry. Reserving the name "
                "keeps qualified parameter ownership unambiguous."
            )
        for label, ports in (("requires", self.requires), ("provides", self.provides)):
            if any(not isinstance(port, Port) for port in ports):
                raise GraphError(
                    f"component {self.name!r} has {label} = {ports!r}; expected a "
                    "frozenset of Port members"
                )
        if not self.provides:
            raise GraphError(
                f"component {self.name!r} provides no port. A component that "
                "computes nothing the graph can name is either dead or is "
                "computing a loss, and losses are objectives (DESIGN.md §3, §4)."
            )
        overlap = self.requires & self.provides
        if overlap:
            raise GraphError(
                f"component {self.name!r} both requires and provides "
                f"{render_ports(overlap)}, which makes it its own dependency"
            )
        supplied = self.provides & SOURCE_PORTS
        if supplied:
            raise GraphError(
                f"component {self.name!r} provides {render_ports(supplied)}, "
                "which the virtual source node supplies from the batch "
                "(DESIGN.md §2.2). Two producers of a raw input is exactly the "
                "ambiguity the source node exists to remove."
            )
        self._validate_forward_signature()
        card_hyperparameters(self)

    def _validate_forward_signature(self) -> None:
        parameters = list(inspect.signature(type(self).forward).parameters.values())
        arguments = parameters[1:]  # drop `self`
        offending = [
            parameter.name
            for parameter in arguments
            if parameter.kind
            in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        if len(arguments) != 1 or offending:
            rendered = ", ".join(parameter.name for parameter in arguments) or "nothing"
            raise GraphError(
                f"component {self.name!r} declares forward(self, {rendered}); the "
                "contract is forward(self, ports: PortView) and nothing else. "
                "There is deliberately no XTYBatch argument — it is the channel "
                "an undeclared dependency would arrive through, which is why "
                "raw x and raw y are ports (DESIGN.md §2.2, §3)."
            )


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def source_ports(batch: XTYBatch) -> dict[Port, PortValue]:
    """The virtual source node: raw inputs as ports (`DESIGN.md` §2.2)."""
    return {Port.X_RAW: batch.x, Port.Y_RAW: batch.y}


class ComponentGraph(nn.Module):
    """A set of components, wired by the ports they declare.

    Everything structural is settled here, at construction: a port two
    components claim, a requirement nothing satisfies, a cycle. What is left
    for `compile()` is everything that depends on the *program* — which
    objectives are active, which rows a stage touches, what is trainable.
    """

    def __init__(self, components: Sequence[Component]) -> None:
        super().__init__()
        components = tuple(components)
        _reject_duplicate_names(components)
        producers = _resolve_producers(components)
        order = _topological_order(components, producers)
        self._components = nn.ModuleDict(
            {name: by_name(components, name) for name in order}
        )
        self._order: tuple[str, ...] = order
        self._producers: dict[Port, str] = producers

    # -- structure ---------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        """Component names in topological order."""
        return self._order

    @property
    def components(self) -> tuple[Component, ...]:
        """Components in topological order."""
        return tuple(self[name] for name in self._order)

    @property
    def provides(self) -> frozenset[Port]:
        """Every port some component produces. Source ports are not included."""
        return frozenset(self._producers)

    def __getitem__(self, name: str) -> Component:
        try:
            module = self._components[name]
        except KeyError:
            raise GraphError(
                f"unknown component {name!r}; this graph holds {list(self._order)!r}"
            ) from None
        return cast(Component, module)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._order

    def __len__(self) -> int:
        return len(self._order)

    def producer(self, port: Port) -> str:
        """Which node produces `port` — a component name, or the source node."""
        if port in SOURCE_PORTS:
            return SOURCE_NAME
        try:
            return self._producers[port]
        except KeyError:
            raise GraphError(
                f"no component provides port {str(port)!r}; this graph provides "
                f"{render_ports(self.provides)}"
            ) from None

    def subgraph_for(self, ports: Iterable[Port]) -> tuple[str, ...]:
        """The components needed to produce `ports`, in topological order.

        This is what "plans the minimum number of forward passes" means for a
        single pass (`DESIGN.md` §8.3): a component no active objective depends
        on is not executed at all.
        """
        wanted = {self.producer(port) for port in ports} - {SOURCE_NAME}
        needed: set[str] = set()
        frontier = list(wanted)
        while frontier:
            name = frontier.pop()
            if name in needed:
                continue
            needed.add(name)
            component = self[name]
            frontier.extend(
                self.producer(port)
                for port in component.requires
                if port not in SOURCE_PORTS
            )
        return tuple(name for name in self._order if name in needed)

    def ancestors(self, name: str) -> frozenset[str]:
        """Every component `name` transitively depends on, excluding itself."""
        return frozenset(self.subgraph_for(self[name].requires))

    def reads_raw_outcome(self) -> tuple[str, ...]:
        """Components that read `Y_RAW` directly, in topological order."""
        return tuple(name for name in self._order if Port.Y_RAW in self[name].requires)

    def depends_on_raw_outcome(self, name: str) -> bool:
        """Is `Y_RAW` in the transitive closure of `name`'s inputs?

        This is the predicate `used_y` is derived from (`DESIGN.md` §2.2, §7.2):
        a property of the graph, not a boolean a producer sets.
        """
        component = self[name]
        if Port.Y_RAW in component.requires:
            return True
        return any(
            Port.Y_RAW in self[ancestor].requires for ancestor in self.ancestors(name)
        )

    def outcome_dependent(self) -> tuple[str, ...]:
        """Every component whose transitive inputs include `Y_RAW`."""
        return tuple(name for name in self._order if self.depends_on_raw_outcome(name))

    def port_depends_on_raw_outcome(self, port: Port) -> bool:
        """Whether producing ``port`` transitively reads ``Y_RAW``.

        This is the single derivation used by the P10 compiler guard and the
        pseudo-label executor. Keeping it on the graph prevents those two
        enforcement points from growing subtly different definitions of
        ``used_y`` (`DESIGN.md` §2.2, §7.2).
        """
        if port == Port.Y_RAW:
            return True
        if port in SOURCE_PORTS:
            return False
        return self.depends_on_raw_outcome(self.producer(port))

    # -- execution ---------------------------------------------------------

    def evaluate(
        self,
        batch: XTYBatch,
        *,
        schema: Schema,
        only: Iterable[str] | None = None,
    ) -> dict[Port, PortValue]:
        """Run the graph over one batch and return every computed port.

        The returned mapping includes the source ports, so it is the full data
        lineage of the pass rather than only its endpoints.

        Args:
            batch: The rows to evaluate. Validated against `schema` first.
            schema: Resolves `D`, `K` and `Dy` for the port shape checks.
            only: Component names to run, as planned by `compile()`. `None`
                runs the whole graph. A selection that omits a dependency of
                something it includes is rejected rather than half-run.
        """
        schema.validate_batch(batch)
        selection = self._select(only)
        values: dict[Port, PortValue] = source_ports(batch)
        for name in selection:
            component = self[name]
            view = PortView(values, declared=component.requires, component=name)
            produced = component(view)
            _check_produced(produced, component, batch=batch, schema=schema)
            values.update(produced)
        return values

    def _select(self, only: Iterable[str] | None) -> tuple[str, ...]:
        if only is None:
            return self._order
        wanted = set(only)
        unknown = sorted(wanted - set(self._order))
        if unknown:
            raise GraphError(
                f"unknown component(s) {unknown!r}; this graph holds "
                f"{list(self._order)!r}"
            )
        for name in wanted:
            missing = sorted(
                self.producer(port)
                for port in self[name].requires
                if port not in SOURCE_PORTS and self.producer(port) not in wanted
            )
            if missing:
                raise GraphError(
                    f"the selection runs {name!r} without its dependencies "
                    f"{missing!r}. A forward pass is planned as a closed "
                    "subgraph (DESIGN.md §8.3); running part of one produces a "
                    "port that was never computed."
                )
        return tuple(name for name in self._order if name in wanted)

    def extra_repr(self) -> str:
        return " -> ".join(self._order)


def by_name(components: Sequence[Component], name: str) -> Component:
    """The component called `name`. Names are unique by the time this runs."""
    for component in components:
        if component.name == name:
            return component
    raise GraphError(f"unknown component {name!r}")  # pragma: no cover


def _reject_duplicate_names(components: Sequence[Component]) -> None:
    if not components:
        raise GraphError("a ComponentGraph needs at least one component")
    seen: set[str] = set()
    for component in components:
        if not isinstance(component, Component):
            raise GraphError(
                f"a ComponentGraph holds Components, got {type(component)}. A "
                "bare nn.Module has no ports, so the compiler cannot order it "
                "or check what it reads (DESIGN.md §3)."
            )
        if component.name in seen:
            raise GraphError(
                f"duplicate component name {component.name!r}; names key the "
                "module dict, `trainable` and the plan, so they must be unique"
            )
        seen.add(component.name)


def _resolve_producers(components: Sequence[Component]) -> dict[Port, str]:
    producers: dict[Port, str] = {}
    for component in components:
        for port in component.provides:
            if port in producers:
                raise GraphError(
                    f"components {producers[port]!r} and {component.name!r} both "
                    f"provide port {str(port)!r}. A port has one producer, "
                    "otherwise nothing decides which value an objective reads."
                )
            producers[port] = component.name
    for component in components:
        unsatisfied = sorted(
            str(port)
            for port in component.requires
            if port not in SOURCE_PORTS and port not in producers
        )
        if unsatisfied:
            available = render_ports(set(producers) | SOURCE_PORTS)
            raise GraphError(
                f"component {component.name!r} requires {unsatisfied!r}, which no "
                f"component in this graph provides. Available: {available}. Add "
                "a component that provides it, or drop the requirement — a "
                "component cannot reach around its ports for it (DESIGN.md §3)."
            )
    return producers


def _topological_order(
    components: Sequence[Component], producers: Mapping[Port, str]
) -> tuple[str, ...]:
    """Kahn's algorithm, resolving ties by declaration order.

    Ties are broken deterministically so that a plan printed from the same
    recipe twice is byte-identical.
    """
    dependencies = {
        component.name: {
            producers[port] for port in component.requires if port not in SOURCE_PORTS
        }
        for component in components
    }
    order: list[str] = []
    placed: set[str] = set()
    while len(order) < len(components):
        ready = [
            component.name
            for component in components
            if component.name not in placed and dependencies[component.name] <= placed
        ]
        if not ready:
            stuck = sorted(set(dependencies) - placed)
            raise GraphError(
                f"the component graph has a cycle among {stuck!r}. Ports flow "
                "one way: a component cannot depend, even transitively, on a "
                "port it helps produce."
            )
        order.append(ready[0])
        placed.add(ready[0])
    return tuple(order)


def _check_produced(
    produced: object, component: Component, *, batch: XTYBatch, schema: Schema
) -> None:
    """A component writes exactly `provides`, and each value fits its `PortSpec`."""
    if not isinstance(produced, dict):
        raise GraphError(
            f"component {component.name!r} returned {type(produced)}; forward "
            "returns dict[Port, PortValue] holding exactly its `provides`"
        )
    keys = set(produced)
    undeclared = keys - component.provides
    if undeclared:
        raise GraphError(
            f"component {component.name!r} wrote {render_ports(undeclared)}, "
            f"which it does not declare; provides = "
            f"{render_ports(component.provides)}. A component writes only the "
            "ports in `provides` (DESIGN.md §3)."
        )
    unwritten = component.provides - keys
    if unwritten:
        raise GraphError(
            f"component {component.name!r} declares {render_ports(unwritten)} "
            "but did not produce it. A declared port that is sometimes absent "
            "is a port the compiler cannot order."
        )
    for port, value in produced.items():
        port_spec(port).check(value, batch_size=batch.batch_size, schema=schema)
