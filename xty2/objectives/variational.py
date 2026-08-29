"""Discrete-treatment variational bound from Kingma et al. M2 eq. (7)."""

from __future__ import annotations

from dataclasses import dataclass

import torch

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

    ``sum_k q_k * (-log p(t=k|x) - log p(y|x,t=k)) + sum_k q_k log q_k``.

    The candidate treatments come from the schema, never from ``batch.t``.
    Nothing is detached: the term trains the propensity, outcome head and
    outcome-aware posterior together. Exact zero mass is masked before the
    entropy multiplication, giving the limiting value ``0 * log(0) = 0``.
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
        safe_log_q = torch.where(q > 0, log_q, torch.zeros_like(log_q))
        per_row = (q * (-log_pt - log_py + safe_log_q)).sum(dim=-1)
        exact_per_row = -torch.logsumexp(log_pt + log_py, dim=-1)
        exact = reduce_rows(exact_per_row, rows)
        gap = reduce_rows(per_row - exact_per_row, rows)
        return reduce_rows(
            per_row,
            rows,
            diagnostics={
                "exact_marginal_nll": float(exact.value.detach()),
                "amortisation_gap": float(gap.value.detach()),
            },
        )


__all__ = ["VariationalTreatmentELBO"]
