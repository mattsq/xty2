"""VICReg's three terms, kept separate so each one can be weighted and ablated.

`docs/recipes/vicreg.md` §3.1 transcribes eq. (1)-(6) and then pins the *author
code* rather than the printed equations, because `VICReg.forward` and eq. (6)
are not the same function of the embeddings:

```text
L_code = 25 * mean_{i,j}(Z_ij - Z'_ij)^2      # F.mse_loss, elementwise
       + 25 * (v(Z) + v(Z')) / 2              # the mean of the two, not the sum
       + 1  * (c(Z) + c(Z'))                  # the sum of the two
```

Equation (6) writes `lambda*s + mu*(v+v') + nu*(c+c')` with `s` the *summed*
squared distance divided by `n` alone. Reading the printed coefficients
`(25, 25, 1)` onto the code's reductions therefore scales the invariance term
by `d` and the variance term by two, which at `d = 512` is a three-orders-of-
magnitude error in the term that holds the two branches together. The card
records the choice as deviation 1 and states the equivalent printed
coefficients, `(25/d, 12.5, 1)`, so that neither convention can be silently
half-adopted.

Three objectives rather than one for the reason `DESIGN.md` §4 gives: eq. (6)
carries three coefficients, the card's §6 study ablates two of them to exactly
zero, and a single term with three internal weights would make that study a
change to the loss rather than a change to the stage's declaration.

Two properties are shared by all three and are decisions rather than details.

* **Nothing is detached.** `VICReg.forward` descends both branches; there is no
  target side, no predictor and no teacher, so `detaches` is empty on all three
  and the dead-trainable check reaches the encoder through every one of them.
* **The two branches are symmetric.** Each term is invariant under exchanging
  its realisations — the invariance term squares the difference, the other two
  average or sum a per-branch statistic. `first` and `second` are therefore
  labels for the plan rather than roles, unlike `InfoNCEContrastive`'s anchor
  and contrast, and Tier 0 asserts the symmetry rather than assuming it.

The variance and covariance terms are **batch-coupled**: `Var` and `C` are
statistics over the rows of the batch, so their value is a different number if
the batch is split. That declaration is what stops a stage holding them from
handing batch construction back to the caller (`core/compile.py`), which is the
guarantee that makes `optimisation.batch_size` a governed number here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.errors import LossError, PortContractError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population

MINIMUM_ROWS = 2
"""`Var` with `correction=1` over one row is `0/0`, and a covariance over one
row is the zero matrix — a batch that would report perfect decorrelation and a
satisfied variance hinge from a single embedding. Both terms refuse it."""


@dataclass(frozen=True)
class EmbeddingInvariance:
    """`s/d`: the elementwise mean squared difference of the two branches.

    Eq. (5) with the author code's reduction — `F.mse_loss(x, y)`, which
    averages over rows *and* embedding dimensions, where eq. (5) averages over
    rows only. This is the one term of the three whose per-row value is
    independent of the other rows, so it is not batch-coupled.

    Attributes:
        port: The tensor port both branches are read from — the expander's
            output, not the encoder's.
        first: One branch's realisation.
        second: The other's. Exchanging the two is the same number.
        rows: The population this term is entitled to. VICReg's is `all`; it
            reads no label of any kind.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    first: Realisation
    second: Realisation
    rows: Rows = "all"
    name: str = "embedding_invariance"

    def __post_init__(self) -> None:
        _validate_branches(
            type(self).__name__,
            name=self.name,
            port=self.port,
            first=self.first,
            second=self.second,
            rows=self.rows,
        )

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.first), (self.port, self.second)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Nothing: `VICReg.forward` descends both branches (module note)."""
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        """No: `mean_j (z_ij - z'_ij)^2` reads row `i` and nothing else."""
        return False

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"branches = {self.port!s} @ {self.first} and {self.port!s} @ "
            f"{self.second}",
            "value = mean over eligible rows and over d of (z - z')^2",
            "reduction = author F.mse_loss, elementwise; eq. (5) divided by d",
            "nothing is detached; both branches carry gradient",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        first = _embedding(state, self.port, self.first, batch, self.name)
        second = _embedding(state, self.port, self.second, batch, self.name)
        per_row = (first - second).pow(2).mean(dim=-1)
        return reduce_rows(per_row, rows)


@dataclass(frozen=True)
class EmbeddingVariance:
    """`(v(Z) + v(Z')) / 2`, the hinged per-dimension standard deviation.

    Eq. (1)-(2): `v(Z) = (1/d) sum_j max(0, gamma - sqrt(Var(z^j) + eps))`,
    averaged over the two branches as `VICReg.forward` does rather than summed
    as eq. (6) writes. The epsilon is **inside** the square root, where it is a
    smoothing of the gradient at zero variance rather than a floor on the
    hinge: `sqrt(0 + 1e-4) = 0.01`, so a fully collapsed dimension is charged
    `gamma - 0.01` and not `gamma`.

    Attributes:
        port: The tensor port both branches are read from.
        first: One branch's realisation.
        second: The other's.
        gamma: The target standard deviation, `1` in the author code. No
            default: it is the constant that decides what "not collapsed"
            means, and the card names it (§3.1).
        epsilon: The `1e-4` inside the square root. No default, same reason.
        correction: The `Var` denominator's correction, `1` in the author code
            (`torch.Tensor.var`'s own default, made explicit here). No default.
        rows: The population this term is entitled to, and the rows the
            variance is taken over.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    first: Realisation
    second: Realisation
    gamma: float
    epsilon: float
    correction: int
    rows: Rows = "all"
    name: str = "embedding_variance"

    def __post_init__(self) -> None:
        _validate_branches(
            type(self).__name__,
            name=self.name,
            port=self.port,
            first=self.first,
            second=self.second,
            rows=self.rows,
        )
        object.__setattr__(self, "gamma", _positive(self.gamma, "gamma", self.name))
        object.__setattr__(
            self, "epsilon", _positive(self.epsilon, "epsilon", self.name)
        )
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
        """Yes: `Var(z^j)` is a statistic of the eligible rows themselves."""
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"branches = {self.port!s} @ {self.first} and {self.port!s} @ "
            f"{self.second}",
            f"value = (v(Z) + v(Z')) / 2, v(Z) = mean_j max(0, {self.gamma} - "
            f"sqrt(Var(z^j) + {self.epsilon}))",
            f"Var over the eligible rows, correction = {self.correction}",
            "epsilon is inside the square root, not a floor on the hinge",
            "the author averages the two branches; eq. (6) sums them",
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
        _require_rows(rows, self.name, "a variance")
        left = self._hinge(first.index_select(0, rows))
        right = self._hinge(second.index_select(0, rows))
        value = (left + right) / 2.0
        return _batch_statistic(
            value,
            batch,
            rows,
            diagnostics={
                "spread_first": _spread(first, rows, self.epsilon, self.correction),
                "spread_second": _spread(second, rows, self.epsilon, self.correction),
            },
        )

    def _hinge(self, embedding: Tensor) -> Tensor:
        """`v(Z)`: the mean over dimensions of the hinged deviation."""
        deviation = torch.sqrt(
            embedding.var(dim=0, correction=self.correction) + self.epsilon
        )
        return torch.relu(self.gamma - deviation).mean()


@dataclass(frozen=True)
class EmbeddingCovariance:
    """`c(Z) + c(Z')`: the off-diagonal energy of each branch's covariance.

    Eq. (3)-(4): `C(Z) = (1/(n-1)) sum_i (z_i - zbar)(z_i - zbar)^T` and
    `c(Z) = (1/d) sum_{i != j} C(Z)_{ij}^2`. The diagonal is excluded — it is
    the variance the previous term is *encouraging*, so including it would make
    the two objectives pull against each other — and the two branches are
    summed, which is what both eq. (6) and `VICReg.forward` do here.

    Attributes:
        port: The tensor port both branches are read from.
        first: One branch's realisation.
        second: The other's.
        correction: The covariance denominator's correction, `1` in the author
            code (`(x.T @ x) / (n - 1)`). No default.
        rows: The population this term is entitled to, and the rows the
            covariance is taken over.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    first: Realisation
    second: Realisation
    correction: int
    rows: Rows = "all"
    name: str = "embedding_covariance"

    def __post_init__(self) -> None:
        _validate_branches(
            type(self).__name__,
            name=self.name,
            port=self.port,
            first=self.first,
            second=self.second,
            rows=self.rows,
        )
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
        """Yes: `C(Z)` is a second moment over the eligible rows."""
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"branches = {self.port!s} @ {self.first} and {self.port!s} @ "
            f"{self.second}",
            "value = c(Z) + c(Z'), c(Z) = (1/d) * sum over off-diagonal C(Z)_ij^2",
            f"C over the eligible rows, denominator n - {self.correction}",
            "the diagonal is excluded: it is the variance term's target",
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
        _require_rows(rows, self.name, "a covariance")
        left = self._off_diagonal(first.index_select(0, rows))
        right = self._off_diagonal(second.index_select(0, rows))
        value = left + right
        return _batch_statistic(
            value,
            batch,
            rows,
            diagnostics={
                "off_diagonal_first": float(left.detach()),
                "off_diagonal_second": float(right.detach()),
                "diagonal_first": _diagonal_energy(first, rows, self.correction),
                "diagonal_second": _diagonal_energy(second, rows, self.correction),
            },
        )

    def _off_diagonal(self, embedding: Tensor) -> Tensor:
        """`c(Z)`: squared off-diagonal covariance, averaged over `d`."""
        covariance = _covariance(embedding, self.correction)
        squared = covariance.pow(2)
        return (squared.sum() - squared.diagonal().sum()) / embedding.shape[-1]


# ---------------------------------------------------------------------------
# Shared arithmetic and validation
# ---------------------------------------------------------------------------


def _covariance(embedding: Tensor, correction: int) -> Tensor:
    """`C(Z)` of eq. (3), over the rows it is handed."""
    centred = embedding - embedding.mean(dim=0, keepdim=True)
    return centred.transpose(0, 1) @ centred / (embedding.shape[0] - correction)


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


def _spread(
    embedding: Tensor, rows: RowIndex, epsilon: float, correction: int
) -> float:
    """`a = mean_j sqrt(Var_j + eps)`, the card's §6.4 spread statistic."""
    selected = embedding.detach().index_select(0, rows)
    return float(
        torch.sqrt(selected.var(dim=0, correction=correction) + epsilon).mean()
    )


def _diagonal_energy(embedding: Tensor, rows: RowIndex, correction: int) -> float:
    """`sum_j C_jj^2`, the denominator of §6.4's redundancy ratio.

    Logged beside the off-diagonal energy because the loss alone cannot tell a
    decorrelated embedding from a collapsed one: both drive `c(Z)` toward zero,
    and only the diagonal says which happened.
    """
    covariance = _covariance(embedding.detach().index_select(0, rows), correction)
    return float(covariance.diagonal().pow(2).sum())


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
            f"{realisation}, got shape {tuple(value.shape)}. `v` and `c` are "
            "statistics of the embedding dimensions (VICReg eq. 1-4)."
        )
    return value


def _require_rows(rows: RowIndex, objective: str, statistic: str) -> None:
    """Refuse a batch too small for the statistic to be defined."""
    if int(rows.numel()) < MINIMUM_ROWS:
        raise LossError(
            f"objective {objective!r} computes {statistic} over "
            f"{int(rows.numel())} eligible row(s). Fewer than {MINIMUM_ROWS} "
            "makes the correction-1 denominator zero or the statistic "
            "identically zero, which reads as a satisfied variance hinge and a "
            "perfectly decorrelated embedding from a single row."
        )


def _validate_branches(
    owner: str,
    *,
    name: str,
    port: Port,
    first: Realisation,
    second: Realisation,
    rows: Rows,
) -> None:
    """The port, realisation and row checks all three terms share."""
    if not require_str(f"{owner} name", name, error=LossError):
        raise LossError(f"{owner}.name must be non-empty")
    if not isinstance(port, Port):
        raise LossError(f"{owner}.port must be a Port, got {type(port)}")
    if port_spec(port).kind != "tensor":
        raise LossError(
            f"{owner} reads an embedding, but port {port!s} carries "
            f"{port_spec(port).kind}. VICReg's `v`, `c` and `s` are statistics "
            "of a tensor, not of a distribution (DESIGN.md §11)."
        )
    branches: tuple[object, object] = (first, second)
    if not all(isinstance(branch, Realisation) for branch in branches):
        raise LossError(f"{owner}.first and second must be Realisations")
    if first == second:
        raise LossError(
            f"{owner} reads {first} twice. The invariance term would be "
            "identically zero and the other two would charge one branch's "
            "statistic twice; VICReg's `Z` and `Z'` are two transformations of "
            "one batch (eq. 7)."
        )
    try:
        validate_population(rows)
    except Xty2Error as error:
        raise LossError(f"{owner} {name!r}: {error}") from error


def _positive(value: object, field: str, name: str) -> float:
    """A finite positive constant, as `gamma` and `epsilon` both must be."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LossError(f"objective {name!r}: {field} must be a number, got {value!r}")
    number = float(value)
    if not number > 0.0 or number != number or number == float("inf"):
        raise LossError(
            f"objective {name!r}: {field} must be finite and positive, got {value!r}"
        )
    return number


def _correction(value: object, name: str) -> int:
    """`0` or `1`: the author code takes `1` for both `Var` and `C`."""
    if type(value) is not int or value not in (0, 1):
        raise LossError(
            f"objective {name!r}: correction must be 0 or 1, got {value!r}. The "
            "author code uses the sample convention, correction = 1, for both "
            "`Var` and the covariance denominator (docs/recipes/vicreg.md §7)."
        )
    return value


__all__ = [
    "MINIMUM_ROWS",
    "EmbeddingCovariance",
    "EmbeddingInvariance",
    "EmbeddingVariance",
]
