"""The two complete-case likelihood terms (`DESIGN.md` §4).

Both are the ordinary supervised losses of a semi-supervised recipe: fit
`p(y | x, t)` and `p(t | x)` on the rows where the treatment was actually
observed. They are separate objects rather than one configurable loss because
they train different heads on the same rows, and a bad number has to be
attributable to one of them (§0).

Neither is parameterised by port or realisation. The two-consumer rule (§11)
is the reason: a treatment NLL against `q(t|x,y)` and a likelihood under a
teacher's parameters are both real, and both arrive with the recipe that needs
them rather than as options nothing exercises.
"""

from __future__ import annotations

from dataclasses import dataclass

from xty2.core.batch import XTYBatch
from xty2.core.graph import DEFAULT, Realisation, State
from xty2.core.loss import (
    LossTerm,
    TrainContext,
    outcome_distribution,
    reduce_rows,
    treatment_at,
    treatment_distribution,
)
from xty2.core.ports import Port
from xty2.core.rows import RowIndex, Rows


@dataclass(frozen=True)
class ObservedOutcomeNLL:
    """`-log p(y | x, t)` at the **observed** treatment.

    Rows: `t_observed`. The outcome is present on every v1 row, so what makes a
    row eligible here is the treatment the likelihood conditions on — this is
    the complete-case term that `MissingTreatmentMarginalNLL` is measured
    against in Tier 1 (`FIDELITY.md` §3).
    """

    name: str = "observed_outcome_nll"

    @property
    def rows(self) -> Rows:
        return "t_observed"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.Y_GIVEN_XT, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: the term trains the head whose density it evaluates."""
        return frozenset()

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        head = outcome_distribution(
            state, Port.Y_GIVEN_XT, DEFAULT, objective=self.name
        )
        # `y` is passed unexpanded and `t` has rank 1, which is what selects
        # observed-treatment evaluation (DESIGN.md §3.1).
        per_row = -head.log_prob(batch.y, treatment_at(batch, rows))
        return reduce_rows(per_row, rows)


@dataclass(frozen=True)
class ObservedTreatmentNLL:
    """`-log p(t | x)` at the observed treatment: the propensity head's fit.

    Rows: `t_observed`. Tier 1 asks this head to beat the marginal-frequency
    baseline on held-out log-loss, which is the assertion that catches a
    propensity that has been wired to the wrong representation.
    """

    name: str = "observed_treatment_nll"

    @property
    def rows(self) -> Rows:
        return "t_observed"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.T_GIVEN_X, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: the term trains the propensity head it evaluates."""
        return frozenset()

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        propensity = treatment_distribution(
            state, Port.T_GIVEN_X, DEFAULT, objective=self.name
        )
        per_row = -propensity.log_prob(treatment_at(batch, rows))
        return reduce_rows(per_row, rows)


__all__ = ["ObservedOutcomeNLL", "ObservedTreatmentNLL"]
