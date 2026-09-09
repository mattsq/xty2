"""Barlow Twins' two terms, kept separate so the off-diagonal one can be ablated.

`docs/recipes/barlow_twins.md` §3.1 prints eq. (1)-(2) and then pins the *author
code* rather than the printed equations, because `BarlowTwins.forward` and
eq. (2) do not normalise the same way:

```python
c = self.bn(z1).T @ self.bn(z2)          # non-affine BatchNorm1d, train mode
c.div_(self.args.batch_size)             # divide by B, not by two column norms
on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
off_diag = off_diagonal(c).pow_(2).sum()
loss = on_diag + self.args.lambd * off_diag
```

Eq. (2) divides each entry by `sqrt(sum_b (z^A_bi)^2) * sqrt(sum_b (z^B_bj)^2)`,
with no centring and no epsilon. `nn.BatchNorm1d` centres, divides by a
*population* standard deviation, and adds `eps = 1e-5` **inside** the square
root. On a batch whose columns are already centred the two agree; on one whose
columns are not, they are different matrices, and the entry the diagonal term
drives to `1` is a different statistic. The card records the choice as
deviation 1, together with the parser's `lambd = 0.0051` against §2.2's printed
`0.005`.

`cross_correlation` below is that pinned arithmetic, written out statelessly:

```text
U^v = (Z^v - mean_b Z^v) / sqrt(var_b Z^v + epsilon)      # correction = 0
C   = (U^A)^T U^B / B
D   = sum_i (1 - C_ii)^2
O   = sum_{i != j} C_ij^2
```

Two objectives rather than one for the reason `DESIGN.md` §4 gives: eq. (1)
carries the one coefficient `lambda`, the card's §6 study sets it to exactly
zero, and a single term with an internal weight would make that study a change
to the loss rather than a change to the stage's declaration.

Three properties are shared by both and are decisions rather than details.

* **Nothing is detached.** `BarlowTwins.forward` descends both branches, and it
  descends them *through* the normalisation, so each branch's column means and
  variances carry gradient too. There is no target side, no predictor, no
  teacher and no memory bank, so `detaches` is empty on both and the
  dead-trainable check reaches the encoder through either one.
* **The two branches are symmetric.** Exchanging them transposes `C`, which
  leaves the diagonal pointwise unchanged and permutes the off-diagonal entries
  among themselves. `first` and `second` are therefore labels for the plan
  rather than roles, unlike `InfoNCEContrastive`'s anchor and contrast, and
  Tier 0 asserts the symmetry rather than assuming it.
* **Both are batch-coupled.** `C` is a second moment over the rows of the
  batch: its value is a different number if the batch is split, and its rank is
  bounded by `B - 1`, so `C = I` is unreachable at the card's `B = 128`,
  `d = 512` (§6.4). That declaration is what stops a stage holding these terms
  from handing batch construction back to the caller (`core/compile.py`), which
  is what makes `optimisation.batch_size` a governed number here.

Neither term divides by `d` or by `d * (d - 1)`: eq. (1) sums over coordinates
and the card keeps that convention, the way VICReg's covariance term keeps its
own internal coordinate reduction. The §6.4 *diagnostics* average, and they are
reported beside the losses rather than folded into them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, is_required
from xty2.core.errors import LossError, PortContractError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population

MINIMUM_ROWS = 2
"""One row has zero population variance in every coordinate, so `U` is the zero
matrix, `C` is the zero matrix, and the pair reports `D = d` with `O = 0` — a
perfectly non-redundant embedding inferred from a single row. Both terms refuse
it. That is a different failure from an empty batch, which `DESIGN.md` §1.3
already answers with the zero term."""


def cross_correlation(
    first: Tensor, second: Tensor, *, epsilon: float, correction: int
) -> Tensor:
    """`C` of the module note: the `[d, d]` cross-view correlation matrix.

    Pure arithmetic, shared by both terms so that the diagonal and off-diagonal
    penalties are two reductions of **one** matrix rather than two
    transcriptions of one formula. Card §6.2 asks the held-out diagnostics to
    recompute the same statistic; they import this rather than write it again.

    Args:
        first: `[B, d]` embeddings of one branch.
        second: `[B, d]` embeddings of the other, row-aligned with `first`.
        epsilon: The BatchNorm epsilon, inside each branch's square root.
        correction: The variance denominator's correction. `0` is the
            training-mode `BatchNorm1d` convention the author code runs under.

    Returns:
        `(U^A)^T U^B / B`, differentiable in both arguments and through the
        means and variances taken from them.
    """
    rows = first.shape[0]
    left, right = (
        (branch - branch.mean(dim=0, keepdim=True))
        / torch.sqrt(branch.var(dim=0, correction=correction) + epsilon)
        for branch in (first, second)
    )
    return left.transpose(0, 1) @ right / rows


@dataclass(frozen=True)
class CrossCorrelationDiagonal:
    """`D = sum_i (1 - C_ii)^2`: the cross-view alignment term of eq. (1).

    The invariance half of the objective, and the one a collapse defeats: at
    exactly constant embeddings the normalisation returns zeros, `C` is zero,
    and this term sits at its finite maximum `d` with no gradient to leave by.
    The card says so in §3.1 rather than promising escape from exact collapse.

    Attributes:
        port: The tensor port both branches are read from — the projector's
            output, not the encoder's.
        first: One branch's realisation.
        second: The other's. Exchanging the two transposes `C`, which leaves
            this term's value unchanged.
        epsilon: The `1e-5` inside each branch's square root. No default: it is
            a pinned constant of the author's `BatchNorm1d` (deviation 1), not
            a smoothing this repository chose.
        correction: The variance denominator's correction, `0` in the author
            code because training-mode `BatchNorm1d` normalises by the biased
            estimate. No default, same reason.
        rows: The population this term is entitled to, and the rows the
            correlation is taken over. Barlow Twins' is `all`; pretraining
            reads no label of any kind.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    first: Realisation
    second: Realisation
    epsilon: float = REQUIRED
    correction: int = REQUIRED
    rows: Rows = "all"
    name: str = "cross_correlation_diagonal"

    def __post_init__(self) -> None:
        _validate(
            type(self).__name__,
            name=self.name,
            port=self.port,
            first=self.first,
            second=self.second,
            rows=self.rows,
        )
        object.__setattr__(self, "epsilon", _epsilon(self.epsilon, self.name))
        object.__setattr__(self, "correction", _correction(self.correction, self.name))

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.first), (self.port, self.second)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: `BarlowTwins.forward` descends both branches (module note)."""
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        """Yes: `C` is a second moment over the eligible rows."""
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"branches = {self.port!s} @ {self.first} and {self.port!s} @ "
            f"{self.second}",
            "value = sum over i of (1 - C_ii)^2, summed over coordinates",
            f"C = (U^A)^T U^B / B, U = (Z - mean) / sqrt(var + {self.epsilon})",
            f"var over the eligible rows, correction = {self.correction}",
            "epsilon is inside the square root, per training-mode BatchNorm1d",
            "no division by d: eq. (1) sums the diagonal",
            f"refuses fewer than {MINIMUM_ROWS} eligible rows",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        first = _embedding(state, self.port, self.first, batch, self.name)
        second = _embedding(state, self.port, self.second, batch, self.name)
        if rows.numel() == 0:
            return LossTerm.empty(like=first)
        _require_rows(rows, self.name, "a cross-correlation")
        left = first.index_select(0, rows)
        right = second.index_select(0, rows)
        correlation = cross_correlation(
            left, right, epsilon=self.epsilon, correction=self.correction
        )
        diagonal = correlation.diagonal()
        value = (1.0 - diagonal).pow(2).sum()
        width = correlation.shape[0]
        return _batch_statistic(
            value,
            batch,
            rows,
            diagnostics={
                "mean_diagonal": float(diagonal.detach().mean()),
                "diagonal_error": float(value.detach()) / width,
                **_raw_variances(left, right, self.correction),
            },
        )


@dataclass(frozen=True)
class CrossCorrelationOffDiagonal:
    """`O = sum_{i != j} C_ij^2`: the redundancy-reduction term of eq. (1).

    Every **ordered** off-diagonal entry is included, as the author's
    `off_diagonal` helper does by flattening all but the diagonal: `C` is a
    cross-view matrix and is not symmetric, so `C_ij` and `C_ji` are two
    different correlations and summing one triangle would charge half the
    redundancy the paper charges (card §3.1).

    This is the term the card's §6 study sets to a weight of exactly zero. It is
    declared even in that arm, so the two arms differ by one number in the
    stage's declaration and not by the shape of the loss.

    Attributes:
        port: The tensor port both branches are read from.
        first: One branch's realisation.
        second: The other's. Exchanging the two transposes `C`, which permutes
            the off-diagonal entries among themselves.
        epsilon: The `1e-5` inside each branch's square root. No default.
        correction: The variance denominator's correction, `0`. No default.
        rows: The population this term is entitled to, and the rows the
            correlation is taken over.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    first: Realisation
    second: Realisation
    epsilon: float = REQUIRED
    correction: int = REQUIRED
    rows: Rows = "all"
    name: str = "cross_correlation_off_diagonal"

    def __post_init__(self) -> None:
        _validate(
            type(self).__name__,
            name=self.name,
            port=self.port,
            first=self.first,
            second=self.second,
            rows=self.rows,
        )
        object.__setattr__(self, "epsilon", _epsilon(self.epsilon, self.name))
        object.__setattr__(self, "correction", _correction(self.correction, self.name))

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.first), (self.port, self.second)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing (module note)."""
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        """Yes: `C` is a second moment over the eligible rows."""
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"branches = {self.port!s} @ {self.first} and {self.port!s} @ "
            f"{self.second}",
            "value = sum over every ordered i != j of C_ij^2",
            f"C = (U^A)^T U^B / B, U = (Z - mean) / sqrt(var + {self.epsilon})",
            f"var over the eligible rows, correction = {self.correction}",
            "epsilon is inside the square root, per training-mode BatchNorm1d",
            "both triangles are charged: C is not symmetric",
            "no division by d or d*(d-1): eq. (1) sums the off-diagonal",
            f"refuses fewer than {MINIMUM_ROWS} eligible rows",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        first = _embedding(state, self.port, self.first, batch, self.name)
        second = _embedding(state, self.port, self.second, batch, self.name)
        if rows.numel() == 0:
            return LossTerm.empty(like=first)
        _require_rows(rows, self.name, "a cross-correlation")
        left = first.index_select(0, rows)
        right = second.index_select(0, rows)
        correlation = cross_correlation(
            left, right, epsilon=self.epsilon, correction=self.correction
        )
        width = correlation.shape[0]
        # Select before squaring, as the author's `off_diagonal` does.
        # Subtracting the diagonal energy from the total can erase a small but
        # nonzero off-diagonal penalty in fp32.
        off_diagonal = ~torch.eye(width, dtype=torch.bool, device=correlation.device)
        value = correlation.masked_select(off_diagonal).pow(2).sum()
        pairs = max(width * (width - 1), 1)
        return _batch_statistic(
            value,
            batch,
            rows,
            diagnostics={
                "redundancy": float(value.detach()) / pairs,
                **_raw_variances(left, right, self.correction),
            },
        )


# ---------------------------------------------------------------------------
# Shared reporting and validation
# ---------------------------------------------------------------------------


def _raw_variances(first: Tensor, second: Tensor, correction: int) -> dict[str, float]:
    """Mean raw coordinate variance per branch, before normalisation.

    Logged by both terms because neither loss alone tells a decorrelated
    embedding from a collapsed one. `C` is scale-free up to `epsilon`, so `O`
    falls toward zero either way, and the diagonal term reports the same `D = d`
    for a collapsed embedding as for a live but cross-view-uncorrelated one.
    Only the raw variance says which happened, and card §6.4's activity guard
    is defined against exactly this number.
    """
    return {
        "raw_variance_first": float(
            first.detach().var(dim=0, correction=correction).mean()
        ),
        "raw_variance_second": float(
            second.detach().var(dim=0, correction=correction).mean()
        ),
    }


def _batch_statistic(
    value: Tensor,
    batch: XTYBatch,
    rows: RowIndex,
    *,
    diagnostics: dict[str, float],
) -> LossTerm:
    """One scalar batch statistic, returned through the per-row convention.

    `reduce_rows` is the whole reduction convention (`core/loss.py`), and a
    batch-coupled term has no per-row decomposition to hand it. Broadcasting the
    scalar over the eligible rows and letting `reduce_rows` average it back is
    the identity on the value and on its gradient, and it keeps `n` counting the
    same rows every other objective counts — so the mixer's row bookkeeping and
    the zero-eligible-row rule apply here unchanged rather than by exception.
    """
    per_row = torch.zeros(
        batch.batch_size, dtype=value.dtype, device=value.device
    ).index_copy(0, rows, value.expand(int(rows.numel())))
    return reduce_rows(per_row, rows, diagnostics=diagnostics)


def _embedding(
    state: State, port: Port, realisation: Realisation, batch: XTYBatch, objective: str
) -> Tensor:
    """Read `port` under `realisation` as a `[B, d]` embedding."""
    value = state[realisation][port]
    if not isinstance(value, Tensor):
        raise PortContractError(
            f"objective {objective!r} read port {str(port)!r} under "
            f"{realisation} as an embedding tensor, but it carries "
            f"{type(value)}. Its PortSpec is the contract (DESIGN.md §2)."
        )
    if value.shape[0] != batch.batch_size:
        raise LossError(
            f"objective {objective!r} got {value.shape[0]} rows from "
            f"{realisation} for a batch of {batch.batch_size}"
        )
    if value.ndim != 2:
        raise LossError(
            f"objective {objective!r} needs a [B, d] embedding from "
            f"{realisation}, got shape {tuple(value.shape)}. `C` is a matrix of "
            "the embedding coordinates against each other (eq. 2)."
        )
    return value


def _require_rows(rows: RowIndex, objective: str, statistic: str) -> None:
    """Refuse a batch too small for the statistic to mean anything."""
    if int(rows.numel()) < MINIMUM_ROWS:
        raise LossError(
            f"objective {objective!r} computes {statistic} over "
            f"{int(rows.numel())} eligible row(s). Fewer than {MINIMUM_ROWS} "
            "makes every coordinate's variance zero, so the normalised batch "
            "is the zero matrix and the pair reports perfect alignment failure "
            "with perfect non-redundancy from a single row."
        )


def _validate(
    owner: str,
    *,
    name: str,
    port: Port,
    first: Realisation,
    second: Realisation,
    rows: Rows,
) -> None:
    """The port, realisation and row checks both terms share."""
    if not require_str(f"{owner} name", name, error=LossError):
        raise LossError(f"{owner}.name must be non-empty")
    if not isinstance(port, Port):
        raise LossError(f"{owner}.port must be a Port, got {type(port)}")
    if port_spec(port).kind != "tensor":
        raise LossError(
            f"{owner} reads an embedding, but port {port!s} carries "
            f"{port_spec(port).kind}. `C` is a statistic of a tensor's "
            "coordinates, not of a distribution (DESIGN.md §11)."
        )
    branches: tuple[object, object] = (first, second)
    if not all(isinstance(branch, Realisation) for branch in branches):
        raise LossError(f"{owner}.first and second must be Realisations")
    if first == second:
        raise LossError(
            f"{owner} reads {first} twice. `C` would be one branch's own "
            "correlation matrix, whose diagonal is identically 1 — the "
            "alignment term would be exactly zero and the redundancy term "
            "would charge a within-view covariance the paper does not "
            "penalise. `Y^A` and `Y^B` are two distorted versions of one batch "
            "(eq. 1 and §2.1)."
        )
    try:
        validate_population(rows)
    except Xty2Error as error:
        raise LossError(f"{owner} {name!r}: {error}") from error


def _epsilon(value: object, name: str) -> float:
    """The finite positive constant inside each branch's square root."""
    if is_required(value):
        raise LossError(
            f"objective {name!r} was constructed without `epsilon`. It is the "
            "author's BatchNorm epsilon, 1e-5, and it decides what a "
            "near-constant coordinate normalises to; the recipe states it "
            "(docs/recipes/barlow_twins.md §3.1, DESIGN.md §9.1)."
        )
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LossError(f"objective {name!r}: epsilon must be a number, got {value!r}")
    number = float(value)
    if not number > 0.0 or number != number or number == float("inf"):
        raise LossError(
            f"objective {name!r}: epsilon must be finite and positive, got {value!r}"
        )
    return number


def _correction(value: object, name: str) -> int:
    """`0` or `1`: the author code runs `BatchNorm1d`, which takes `0`."""
    if is_required(value):
        raise LossError(
            f"objective {name!r} was constructed without `correction`. Paper "
            "and code disagree — Algorithm 1 leaves `std` unspecified and the "
            "executable path is training-mode BatchNorm1d, correction = 0 — so "
            "the recipe states it rather than inheriting torch's default "
            "(docs/recipes/barlow_twins.md §7, DESIGN.md §9.1)."
        )
    if type(value) is not int or value not in (0, 1):
        raise LossError(
            f"objective {name!r}: correction must be 0 or 1, got {value!r}. The "
            "author code normalises with `nn.BatchNorm1d`, whose training-mode "
            "statistic is the population variance, correction = 0 "
            "(docs/recipes/barlow_twins.md §7)."
        )
    return value


__all__ = [
    "MINIMUM_ROWS",
    "CrossCorrelationDiagonal",
    "CrossCorrelationOffDiagonal",
    "cross_correlation",
]
