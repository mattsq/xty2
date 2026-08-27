"""Exact marginalisation over the missing treatment (`DESIGN.md` §4.1).

This is the objective the whole design is arranged around. With `K` small, a
row whose treatment is missing still carries information about both heads:

$$-\\log \\sum_{k=1}^{K} p_\\theta(t=k \\mid x)\\, p_\\phi(y \\mid x, t=k)$$

computed in log space as `-logsumexp_k(log_pt[:, k] + log_py[:, k])`.

It requires exactly two ports and no architecture, which is what lets it work
unchanged across a TARNet head, a Gaussian density head, a conditional flow and
an energy model — the claim P7 tests by asserting this file is untouched by that
packet's diff.

**The gradient path is a required card field.** The term backpropagates into
both heads unless something stops it, and several papers stop-gradient one side.
That is precisely the kind of one-line paper detail that gets lost, so
`grad_path` binds `gradients.marginal_nll_grad_path` and carries no default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal, get_args

import torch

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import CardKeyError
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

GradPath = Literal["both", "propensity", "outcome"]
"""Which head the marginalisation term trains (`FIDELITY.md` §2).

`both` is the plain likelihood gradient. `propensity` detaches `log p(y|x,t)`,
so the term trains only the treatment head; `outcome` detaches `log p(t|x)`.
The name says what is *trained*, not what is detached, because that is how
papers state it.
"""

GRAD_PATHS: tuple[GradPath, ...] = get_args(GradPath)


@dataclass(frozen=True)
class MissingTreatmentMarginalNLL:
    """`-log Σ_k p(t=k|x) p(y|x,t=k)` on rows with a missing treatment.

    Rows: `t_missing`. The candidate matrix is built from the schema's `K` and
    never from the observed `t` — a matrix built as `t[:, None].expand(B, K)`
    repeats each row's own treatment across every column and silently turns
    this term into `K` copies of the complete-case one (`FIDELITY.md` §3).

    Attributes:
        grad_path: Which head this term trains. No default: it binds
            `gradients.marginal_nll_grad_path` (`DESIGN.md` §9.1).
        name: Keys the per-objective log (§6.2).
    """

    grad_path: GradPath = REQUIRED
    name: str = "missing_treatment_marginal_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "grad_path": "gradients.marginal_nll_grad_path"
    }

    def __post_init__(self) -> None:
        # Same construction-time contract components get from their metaclass:
        # a paper-governed field left unset fails here, not at the compile step
        # that happens to read `plan.hyperparameters`.
        card_hyperparameters(self)
        if self.grad_path not in GRAD_PATHS:
            raise CardKeyError(
                f"objective {self.name!r} has grad_path {self.grad_path!r}; "
                f"expected one of {list(GRAD_PATHS)!r}. The name says which head "
                "the term trains (DESIGN.md §4.1)."
            )

    @property
    def rows(self) -> Rows:
        return "t_missing"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.T_GIVEN_X, DEFAULT), (Port.Y_GIVEN_XT, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Whichever side `grad_path` stops the gradient on.

        Derived from the card-governed field rather than stated a second time,
        so the compiler's view of this term and the arithmetic in `compute`
        cannot disagree. This is what lets the dead-trainable check (§8.4) see
        that a stage training only the detached head is a no-op.
        """
        if self.grad_path == "propensity":
            return frozenset({(Port.Y_GIVEN_XT, DEFAULT)})
        if self.grad_path == "outcome":
            return frozenset({(Port.T_GIVEN_X, DEFAULT)})
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        """No: the sum is over treatments `k`, never over other rows."""
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        propensity = treatment_distribution(
            state, Port.T_GIVEN_X, DEFAULT, objective=self.name
        )
        head = outcome_distribution(
            state, Port.Y_GIVEN_XT, DEFAULT, objective=self.name
        )
        candidates = candidate_treatments(batch, ctx.schema)
        log_pt = propensity.log_prob(candidates)
        log_py = head.log_prob(batch.y, candidates)
        if self.grad_path == "propensity":
            log_py = log_py.detach()
        elif self.grad_path == "outcome":
            log_pt = log_pt.detach()
        per_row = -torch.logsumexp(log_pt + log_py, dim=-1)
        return reduce_rows(per_row, rows)


__all__ = ["GRAD_PATHS", "GradPath", "MissingTreatmentMarginalNLL"]
