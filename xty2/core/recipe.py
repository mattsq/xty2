"""The declarative surface `compile()` consumes (`DESIGN.md` §7, §9).

A recipe is an assembly of registered components, objectives and views plus
explicit hyperparameters, and **it contains no logic**. That rule is only
enforceable if the thing a recipe assembles is data, so the three types here
are deliberately inert: they validate their own shape and hold no behaviour.

`Objective` is the *compiler's* view of a loss — the three attributes the
compiler reads. P3 adds `compute()`, `LossTerm` and the mixer around it; a
structural protocol is what lets the compiler check objectives it cannot yet
run. `Stage` likewise carries the fields the compiler checks; the sequencing
fields of §7 (`initialise_from`, `inputs`, `executor`, artifacts) arrive with
the executor and the program that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from xty2.core.errors import CompileError, Xty2Error, require_str
from xty2.core.graph import ComponentGraph, Realisation
from xty2.core.ports import Port
from xty2.core.rows import Rows, validate_population
from xty2.core.schema import Schema

Purpose = Literal["causal", "predictive"]
"""What the recipe is for. `predictive` is what may opt out of the leakage
rule (`DESIGN.md` §7.2); `causal` may not."""


@runtime_checkable
class Objective(Protocol):
    """What the compiler reads from a loss (`DESIGN.md` §4).

    Declared read-only, and the three members are properties for that reason
    rather than for ceremony: an objective is a *declaration* the compiler
    inspects, so a frozen dataclass has to be able to satisfy it, and nothing
    in the framework may write these back.
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


@dataclass(frozen=True)
class Stage:
    """One step of the program (`DESIGN.md` §7).

    Attributes:
        name: Unique within a program; names the stage's artifacts and its
            section of the printed plan.
        objectives: The losses active in this stage.
        trainable: Component names this stage updates. A name that is not a
            component, or a component no active objective depends on, is a
            compile error — the second catches a dead-weight stage.
        rows: The stage's row scope. The eligible set for each objective is
            this intersected with the objective's own population (§7.0).
    """

    name: str
    objectives: tuple[Objective, ...] = ()
    trainable: tuple[str, ...] = ()
    rows: Rows = "all"

    def __post_init__(self) -> None:
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "trainable", tuple(self.trainable))
        if not require_str("stage name", self.name, error=CompileError).isidentifier():
            raise CompileError(
                f"stage name {self.name!r} must be a Python identifier: it names "
                "the stage's artifacts and its section of the execution plan"
            )
        validate_rows(self.rows, f"stage {self.name!r}")
        duplicates = _duplicates(self.trainable)
        if duplicates:
            raise CompileError(
                f"stage {self.name!r} lists {duplicates!r} in `trainable` more "
                "than once"
            )


@dataclass(frozen=True)
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
        purpose: `causal` or `predictive` (§7.2).
    """

    name: str
    schema: Schema
    system: ComponentGraph
    program: tuple[Stage, ...]
    card: str
    purpose: Purpose = "causal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "program", tuple(self.program))
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
        duplicates = _duplicates(tuple(stage.name for stage in self.program))
        if duplicates:
            raise CompileError(
                f"recipe {self.name!r} has more than one stage called "
                f"{duplicates!r}; stages are referenced by name"
            )

    def stage(self, name: str) -> Stage:
        """The stage called `name`."""
        for stage in self.program:
            if stage.name == name:
                return stage
        raise CompileError(
            f"recipe {self.name!r} has no stage {name!r}; it has "
            f"{[stage.name for stage in self.program]!r}"
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
