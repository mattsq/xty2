"""Discrete-treatment variational bound from Kingma et al. M2 eq. (7)."""

from __future__ import annotations

from dataclasses import dataclass

from xty2.core.batch import XTYBatch
from xty2.core.graph import DEFAULT, Realisation, State
from xty2.core.loss import (
    LossTerm,
    TrainContext,
    candidate_treatments,
    outcome_distribution,
    reduce_rows,
    treatment_distribution,
)
from xty2.core.ports import Port
from xty2.core.rows import RowIndex, Rows


@dataclass(frozen=True)
class VariationalTreatmentELBO:
    """Negative ELBO for a missing discrete treatment.

    For every missing-treatment row this computes

    ``sum_k q_k * (-log p(t=k|x) - log p(y|x,t=k) + log q_k)``.

    The candidate treatments come from the schema, never from ``batch.t``.
    Nothing is detached: the term trains the propensity, outcome head and
    outcome-aware posterior together.
    """

    name: str = "variational_treatment_elbo"

    @property
    def rows(self) -> Rows:
        return "t_missing"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, DEFAULT),
                (Port.Y_GIVEN_XT, DEFAULT),
                (Port.T_GIVEN_XY, DEFAULT),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        propensity = treatment_distribution(
            state, Port.T_GIVEN_X, DEFAULT, objective=self.name
        )
        outcome = outcome_distribution(
            state, Port.Y_GIVEN_XT, DEFAULT, objective=self.name
        )
        posterior = treatment_distribution(
            state, Port.T_GIVEN_XY, DEFAULT, objective=self.name
        )
        candidates = candidate_treatments(batch, ctx.schema)

        log_pt = propensity.log_prob(candidates)
        log_py = outcome.log_prob(batch.y, candidates)
        log_q = posterior.log_prob(candidates)
        q = posterior.probs
        per_row = (q * (-log_pt - log_py + log_q)).sum(dim=-1)
        return reduce_rows(per_row, rows)


__all__ = ["VariationalTreatmentELBO"]
