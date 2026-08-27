"""The two complete-case likelihood terms (`DESIGN.md` §4).

Both are the ordinary supervised losses of a semi-supervised recipe: fit
`p(y | x, t)` and a declared treatment distribution on the rows where the
treatment was actually observed. They are separate objects because they train
different heads on the same rows, and a bad number has to be attributable to
one of them (§0).

`ObservedOutcomeNLL` stays on the default realisation.
`ObservedTreatmentNLL` names both its treatment-distribution port and
realisation: the defaults retain the ordinary `p(t | x)` fit, Mean Teacher can
name its student view, and P11's cycle-dual posterior can supervise
`q(t | x, y)` without introducing a duplicate objective.
"""

from __future__ import annotations

from dataclasses import dataclass

from xty2.core.batch import XTYBatch
from xty2.core.errors import LossError
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

    @property
    def batch_coupled(self) -> bool:
        """No: the density of one row's outcome does not read another's."""
        return False

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
        if batch.weight is not None:
            # Sample weights are row mechanics, distinct from the objective's
            # mixer weight. Multiplying before `reduce_rows` means a
            # `population` reduction is exactly `(1/B) Σ_i w_i nll_i`, which
            # is TARNet Eq. (3). Ineligible rows are discarded afterwards.
            per_row = per_row * batch.weight
        return reduce_rows(per_row, rows)


@dataclass(frozen=True)
class ObservedTreatmentNLL:
    """Observed-treatment NLL for `p(t | x)` or `q(t | x, y)`.

    Rows: `t_observed`. Tier 1 asks this head to beat the marginal-frequency
    baseline on held-out log-loss, which is the assertion that catches a
    propensity that has been wired to the wrong representation.
    """

    name: str = "observed_treatment_nll"
    realisation: Realisation = DEFAULT
    port: Port = Port.T_GIVEN_X

    def __post_init__(self) -> None:
        if not isinstance(self.realisation, Realisation):
            raise LossError(
                "ObservedTreatmentNLL.realisation must be a Realisation, got "
                f"{type(self.realisation)}"
            )
        if self.port not in (Port.T_GIVEN_X, Port.T_GIVEN_XY):
            raise LossError(
                "ObservedTreatmentNLL.port must be T_GIVEN_X or T_GIVEN_XY, "
                f"got {self.port!r}"
            )

    @property
    def rows(self) -> Rows:
        return "t_observed"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.realisation)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: the term trains the propensity head it evaluates."""
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        """No: `-log p(t_i | x_i)` is a function of row `i` alone."""
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        propensity = treatment_distribution(
            state, self.port, self.realisation, objective=self.name
        )
        per_row = -propensity.log_prob(treatment_at(batch, rows))
        return reduce_rows(per_row, rows)


__all__ = ["ObservedOutcomeNLL", "ObservedTreatmentNLL"]
