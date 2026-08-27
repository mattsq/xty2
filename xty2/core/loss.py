"""What an objective returns, and the four things every objective needs.

`LossTerm` and `TrainContext` are the objective contract of `DESIGN.md` §4;
`Reduction` and `apply_reduction` are §6.1. They live in `core/` because the
`Objective` protocol in `core/recipe.py` is stated in terms of them and because
`training/` imports `core/` and never the reverse.

The four helpers below exist so that three framework rules are obeyed
mechanically rather than remembered:

* `reduce_rows` is the **zero-eligible-row rule** (§1.3) in one place. An
  objective with no eligible rows returns `value=0.0, n=0` and never a `NaN`
  from a mean over an empty tensor, because no objective computes that mean
  itself.
* `treatment_at` is the **no-sentinel rule** (§1.1). Objectives evaluate
  full-batch quantities and gather afterwards — state holds distributions that
  are not sliceable, so pre-slicing one side is how predictions get misaligned
  (§4) — which means `batch.t` is *read* at rows the objective is not entitled
  to. This makes those positions unreadable rather than merely forbidden.
* `outcome_distribution` / `treatment_distribution` narrow a `PortValue` at the
  read, against a named port and a named objective, instead of failing as an
  attribute error inside a loss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, TypeVar, cast, get_args, runtime_checkable

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.data import TrainingPopulation
from xty2.core.distributions import OutcomeDistribution, TreatmentDistribution
from xty2.core.errors import LossError, PortContractError
from xty2.core.graph import Realisation, State
from xty2.core.ports import Port
from xty2.core.rows import RowIndex
from xty2.core.schema import Schema

Reduction = Literal["mean", "sum", "population"]
"""How a term's mean-over-eligible-rows value enters the total (`DESIGN.md` §6.1).

| Mode | Contribution | Use when |
|---|---|---|
| `mean` | `value` | the paper averages over the term's own rows |
| `sum` | `value * n` | the paper sums over rows |
| `population` | `value * n / B` | the paper averages over the *whole* batch |

`sum` and `mean` differ by a factor that varies *per batch* whenever the row
population does — which is every `t_missing` term — so choosing the wrong one
produces a model that trains, looks reasonable, and weights its semi-supervised
term differently from the paper.
"""

REDUCTIONS: tuple[Reduction, ...] = get_args(Reduction)


StateT = TypeVar("StateT")


@runtime_checkable
class StatefulObjective(Protocol):
    """An objective whose value depends on the steps that came before it.

    `DESIGN.md` §4 says an objective never mutates state, and that sentence is
    about `State` — the planned forward passes — parameters and the batch, all
    of which stay off limits. What it did not have a way to say is that a
    handful of published mechanics are *functions of the training history*:
    FlexMatch's per-class threshold is computed from marks accumulated over
    every step so far (`docs/recipes/flexmatch.md` §3.1, Alg. 1 lines 2, 5 and
    15), and no arrangement of ports, realisations or row populations computes
    it from one batch.

    So an objective may declare one piece of state of its own. Three properties
    make that safe rather than a hole in the contract:

    * **The framework owns the lifecycle.** The executor calls `initial_state`
      once per stage *execution* and hands the result back through
      `TrainContext.objective_states`. State never lives on the objective
      instance, so a recipe stays an immutable declaration, two runs of one
      compiled recipe are identical, and a paired ablation whose two arms share
      an objective instance cannot leak one arm's history into the other.
    * **It is not an artifact.** Nothing checkpoints it, restores it or keys
      provenance by it (`BACKLOG.md` §11.3 asks for explicit objective state
      first, and a first-class stateful artifact only once two recipes need the
      same lifecycle).
    * **It is opt-in and invisible to everything else.** An objective that does
      not declare `initial_state` is unaffected, and a stage holding none gets
      an empty mapping.

    A stateful objective is still not *batch*-coupled unless its per-row value
    depends on the other rows of its own batch — see `Objective.batch_coupled`,
    which asks a different question and gets a different answer here.
    """

    def initial_state(self, population: TrainingPopulation | None) -> object:
        """Fresh state for a stage about to run.

        `population` is `None` when the caller supplies batches directly
        (`ExternalBatches`), and it is the objective rather than the framework
        that decides whether it can run without one: FlexMatch needs `N` and
        the row identities and raises, where an objective carrying only a
        running average over batches needs neither.
        """


@dataclass(frozen=True)
class TrainContext:
    """What an objective may know about the run it is part of.

    Deliberately small. `global_step` is what schedules are functions of (§6),
    and `schema` is where `K` comes from — an objective that instead read `K`
    off the head's own output would agree with a head that had the wrong `K`,
    whereas passing the schema's `K` into `log_prob` makes the disagreement an
    error (`DESIGN.md` §3.1).

    Attributes:
        global_step: Optimiser steps completed **in the executing stage**. The
            name is a legacy of the single-stage era and this line used to say
            the opposite — "across the whole program, not the stage" — which
            the executor has never done (`executors.py` counts
            `range(compiled.steps)` per stage). Per-stage is also the only
            reading that matches how a schedule is declared: `losses.schedules`
            is keyed by `<stage>.<objective>` (§9.1), so a ramp "over 1,000
            steps" in a program's second stage means its own first thousand,
            not a window a previous stage already consumed. Corrected when
            `scarf` — the first recipe with both a schedule and a preceding
            stage — made the distinction load-bearing.
        schema: The resolved schema. `K`, `D` and `Dy` come from here.
        stage: The stage being executed; it labels the log, nothing else.
        objective_states: `{objective name: its state}` for the stateful
            objectives of this stage, built once per stage execution by the
            executor (`StatefulObjective`). Empty for every stage whose
            objectives are all stateless, which is every stage but
            `flexmatch`'s today. Read it through `objective_state`, never by
            indexing: the accessor is what turns "this objective declared no
            state" and "the executor handed me somebody else's" into errors
            that name the objective rather than a `KeyError` inside a loss.
    """

    global_step: int
    schema: Schema
    stage: str = ""
    objective_states: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.global_step < 0:
            raise LossError(
                f"TrainContext.global_step must be non-negative, got {self.global_step}"
            )
        object.__setattr__(
            self, "objective_states", MappingProxyType(dict(self.objective_states))
        )

    def objective_state(self, name: str, kind: type[StateT]) -> StateT:
        """This objective's own state, narrowed to the type it created.

        `kind` is checked rather than cast. An objective asks for the state it
        built in `initial_state`, so a mismatch means the executor and the
        objective disagree about which stage is running — which is worth an
        error naming both, not an `AttributeError` three lines later.
        """
        state = self.objective_states.get(name)
        if state is None:
            raise LossError(
                f"objective {name!r} asked for per-stage state and the stage "
                f"{self.stage or '<unnamed>'!r} has none for it. State is built "
                "by the executor from `initial_state` (DESIGN.md §4), so an "
                "objective computed outside a stage run — or one whose "
                "`initial_state` returned None — reaches here."
            )
        if not isinstance(state, kind):
            raise LossError(
                f"objective {name!r} expected {kind.__name__} state and the "
                f"stage holds {type(state).__name__}"
            )
        return state


@dataclass(frozen=True, eq=False)
class LossTerm:
    """One objective's contribution, **unweighted** (`DESIGN.md` §4).

    Attributes:
        value: Scalar, the mean over the eligible rows. Unweighted: weighting
            is the mixer's job, and an objective that applied its own weight
            would make the logged raw value incomparable across runs.
        n: Eligible rows. Equal to `rows.numel()`, which the mixer checks.
        diagnostics: Extra per-step numbers for the log. Floats only — this is
            a logging surface, not a second return channel.
    """

    value: Tensor
    n: int
    diagnostics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor):
            raise LossError(
                f"LossTerm.value must be a scalar Tensor, got {type(self.value)}"
            )
        if self.value.ndim != 0:
            raise LossError(
                f"LossTerm.value must be a scalar, got shape "
                f"{tuple(self.value.shape)} — an objective reduces over its "
                "eligible rows (DESIGN.md §4)"
            )
        # `type(...) is not int` rather than `isinstance`, so that `True` — an
        # int by inheritance — is rejected as the miscount it would be.
        if type(self.n) is not int:
            raise LossError(f"LossTerm.n must be an int, got {type(self.n)}")
        if self.n < 0:
            raise LossError(f"LossTerm.n must be non-negative, got {self.n}")
        if self.n == 0 and float(self.value.detach()) != 0.0:
            raise LossError(
                f"LossTerm has n = 0 but value {float(self.value.detach())!r}. An "
                "objective with no eligible rows returns exactly 0.0 — never a "
                "NaN from a mean over an empty tensor, which is the single most "
                "common cause of a silently dead objective (DESIGN.md §1.3)."
            )
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(_floats(cast("Mapping[str, object]", self.diagnostics))),
        )

    @property
    def is_empty(self) -> bool:
        """Did this term see no eligible rows in this batch?"""
        return self.n == 0

    @classmethod
    def empty(cls, *, like: Tensor) -> LossTerm:
        """The zero-eligible-row result, on `like`'s device and dtype.

        Deliberately detached: there is nothing to differentiate, and a zero
        that carried a graph would let an empty term look active to the
        gradient probe.
        """
        return cls(value=torch.zeros((), dtype=like.dtype, device=like.device), n=0)


def apply_reduction(term: LossTerm, reduction: Reduction, *, batch_size: int) -> Tensor:
    """`term.value` as it enters the total, before weighting (`DESIGN.md` §6.1).

    The mixer is the only caller: the reduction a paper prescribes is applied
    here rather than inside the objective, so the raw per-objective value stays
    comparable across runs and across modes.
    """
    validate_reduction(reduction, where="apply_reduction")
    if batch_size < 1:
        raise LossError(f"batch_size must be positive, got {batch_size}")
    if reduction == "mean":
        return term.value
    if reduction == "sum":
        return term.value * term.n
    return term.value * term.n / batch_size


def validate_reduction(reduction: Reduction, *, where: str) -> None:
    """Raise unless `reduction` is one of the three modes of `DESIGN.md` §6.1."""
    if reduction not in REDUCTIONS:
        raise LossError(
            f"{where}: unknown reduction {reduction!r}; expected one of "
            f"{list(REDUCTIONS)!r} (DESIGN.md §6.1)"
        )


# ---------------------------------------------------------------------------
# What every objective needs
# ---------------------------------------------------------------------------


def reduce_rows(
    per_row: Tensor,
    rows: RowIndex,
    *,
    diagnostics: Mapping[str, float] | None = None,
) -> LossTerm:
    """Gather `per_row` by `rows` and average — the whole reduction convention.

    Every objective ends here, which is what makes the zero-eligible-row rule a
    property of the framework rather than of each loss.

    Args:
        per_row: `[B]` per-row losses over the **full** batch. Full-batch
            because `state` and `batch` share one batch axis and one index
            (`DESIGN.md` §4).
        rows: The eligible set, already intersected by the compiler (§7.0).
        diagnostics: Extra numbers for the per-step log.

    Returns:
        `LossTerm(value=mean over rows, n=rows.numel())`, or the zero term when
        no row is eligible.
    """
    if per_row.ndim != 1:
        raise LossError(
            f"per-row losses must be [B], got shape {tuple(per_row.shape)}. The "
            "batch axis is shared with the batch and the state, and it is what "
            "`rows` indexes (DESIGN.md §4)."
        )
    if rows.ndim != 1 or rows.dtype != torch.long:
        raise LossError(
            f"rows must be a [n] long RowIndex, got shape {tuple(rows.shape)} of "
            f"{rows.dtype}"
        )
    if rows.numel() == 0:
        return LossTerm.empty(like=per_row)
    eligible = per_row.index_select(0, rows)
    return LossTerm(
        value=eligible.mean(),
        n=int(rows.numel()),
        diagnostics=dict(diagnostics or {}),
    )


def treatment_at(batch: XTYBatch, rows: RowIndex) -> Tensor:
    """`[B]` long: `batch.t` on `rows`, and a valid placeholder everywhere else.

    Objectives evaluate the outcome and treatment heads over the whole batch
    and gather afterwards, so `batch.t` would otherwise be *read* at rows the
    objective is not entitled to — arbitrary indices on unobserved rows
    (`DESIGN.md` §1.1), and observed-but-ineligible ones elsewhere. Both are
    bugs of the same kind, and both become unreachable here: the returned
    tensor carries no information from any row outside `rows`.

    Class 0 is the placeholder because it is the one index every schema has;
    the values at those positions are discarded by `reduce_rows`.
    """
    if rows.numel() == 0:
        return torch.zeros_like(batch.t)
    return torch.zeros_like(batch.t).index_copy(0, rows, batch.t.index_select(0, rows))


def candidate_treatments(batch: XTYBatch, schema: Schema) -> Tensor:
    """`[B, K]` holding `0..K-1` in every row — the candidate matrix of §3.1.

    Built from the schema's `K` and never from the observed `t`: a matrix built
    as `t[:, None].expand(B, K)` repeats each row's own treatment across every
    column, which turns exact marginalisation into `K` copies of the
    complete-case term and every candidate-contract assertion into a demand for
    treatment-insensitivity (`FIDELITY.md` §3).
    """
    return torch.arange(
        schema.treatment_cardinality, dtype=torch.long, device=batch.t.device
    ).expand(batch.batch_size, schema.treatment_cardinality)


def outcome_distribution(
    state: State, port: Port, realisation: Realisation, *, objective: str
) -> OutcomeDistribution:
    """Read `port` under `realisation` and narrow it to an `OutcomeDistribution`."""
    value = state[realisation][port]
    if not isinstance(value, OutcomeDistribution):
        raise PortContractError(
            f"objective {objective!r} read port {str(port)!r} under {realisation} "
            f"as an outcome distribution (log_prob, mean, sample), but it "
            f"carries {type(value)}. Its PortSpec is the contract (DESIGN.md §2)."
        )
    return value


def treatment_distribution(
    state: State, port: Port, realisation: Realisation, *, objective: str
) -> TreatmentDistribution:
    """Read `port` under `realisation` and narrow it to a `TreatmentDistribution`."""
    value = state[realisation][port]
    if not isinstance(value, TreatmentDistribution):
        raise PortContractError(
            f"objective {objective!r} read port {str(port)!r} under {realisation} "
            f"as a treatment distribution (probs, log_prob), but it carries "
            f"{type(value)}. Its PortSpec is the contract (DESIGN.md §2)."
        )
    return value


def _floats(diagnostics: Mapping[str, object]) -> dict[str, float]:
    if not isinstance(diagnostics, Mapping):
        raise LossError(
            f"LossTerm.diagnostics must be a mapping of name to float, got "
            f"{type(diagnostics)}"
        )
    resolved: dict[str, float] = {}
    for name, reported in diagnostics.items():
        value = reported
        if isinstance(value, Tensor):
            if value.ndim != 0:
                raise LossError(
                    f"diagnostic {name!r} must be scalar, got shape "
                    f"{tuple(value.shape)}"
                )
            value = float(value.detach())
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise LossError(
                f"diagnostic {name!r} must be a float, got {type(value)}. "
                "Diagnostics are a logging surface, not a second return channel."
            )
        if not math.isfinite(float(value)):
            raise LossError(f"diagnostic {name!r} must be finite, got {value!r}")
        resolved[name] = float(value)
    return resolved


__all__ = [
    "REDUCTIONS",
    "LossTerm",
    "Reduction",
    "StatefulObjective",
    "TrainContext",
    "apply_reduction",
    "candidate_treatments",
    "outcome_distribution",
    "reduce_rows",
    "treatment_at",
    "treatment_distribution",
    "validate_reduction",
]
