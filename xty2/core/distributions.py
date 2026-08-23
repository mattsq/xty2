"""Distribution protocols and the reference implementations (`DESIGN.md` §3.1).

The candidate-treatment contract is the load-bearing detail of the whole
design, and three separate attempts to state it in prose were wrong. It stops
being prose here: `GaussianOutcome` is an executable reference implementation
of it, and `xty2.core.conformance` is the test every outcome head is later run
against.

The contract, restated as the shape table it is:

| call | `t` | returns |
|---|---|---|
| `log_prob(y, t)` | `[B]` | `[B]` |
| `log_prob(y, t)` | `[B, C]` | `[B, C]` |
| `mean(t)` | `[B]` | `[B, *Dy]` |
| `mean(t)` | `[B, C]` | `[B, C, *Dy]` |
| `sample(t, n)` | `[B]` | `[n, B, *Dy]` |
| `sample(t, n)` | `[B, C]` | `[n, B, C, *Dy]` |

The candidate axis is inserted immediately after the batch axis; the sample
axis leads. Two rules make it implementable:

1. **`y` is passed unexpanded, always.** The head inserts the candidate axis
   itself. Ambient broadcasting does not do this job: `y: [B]` against
   `t: [B, K]` aligns `B` against `K` and fails for every batch with `B != K`.
2. **The rank of `t` selects the mode, and nothing else does.** `t.ndim == 1`
   is observed-treatment evaluation, `t.ndim == 2` is candidate evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from xty2.core.errors import ContractError

TreatmentMode = Literal["observed", "candidate"]


@runtime_checkable
class OutcomeDistribution(Protocol):
    """`p(y | x, t)` for one batch, evaluable at candidate treatments."""

    def log_prob(self, y: Tensor, t: Tensor) -> Tensor: ...

    def mean(self, t: Tensor) -> Tensor: ...

    def sample(self, t: Tensor, n: int) -> Tensor: ...


@runtime_checkable
class TreatmentDistribution(Protocol):
    """`p(t | x)` or `q(t | x, y)` for one batch."""

    @property
    def probs(self) -> Tensor:
        """`[B, K]`, rows summing to 1."""

    def log_prob(self, t: Tensor) -> Tensor:
        """Same rank rule as the outcome contract: `[B] -> [B]`, `[B, C] -> [B, C]`."""


def treatment_mode(t: Tensor, num_treatments: int, *, batch_size: int) -> TreatmentMode:
    """Validate `t` against the contract and report which mode it selects.

    `batch_size` is required rather than optional because a `t` whose leading
    axis is *not* `B` is the one error this contract cannot survive: with a
    singleton leading axis, `gather` and then ambient broadcasting produce a
    correctly-shaped `[B]` or `[B, C]` result in which every row was scored
    under row 0's parameters. Nothing downstream can detect that, so the check
    belongs where no head can forget it.

    Args:
        t: `[B]` observed treatments or `[B, C]` candidate treatments.
        num_treatments: `K`. Every entry of `t` must lie in `[0, K)`.
        batch_size: `B`, the number of rows this distribution represents.

    Raises:
        ContractError: if `t` is not a long tensor of rank 1 or 2, if its
            leading axis is not `B`, or if it holds an index outside `[0, K)`
            — including the `-1` a caller might reach for to mean "missing"
            (`DESIGN.md` §1.1).
    """
    if t.dtype != torch.long:
        raise ContractError(f"t must be a long tensor of class indices, got {t.dtype}")
    if t.ndim not in (1, 2):
        raise ContractError(
            f"t must be [B] (observed) or [B, C] (candidate), got shape "
            f"{tuple(t.shape)} — the rank of t selects the mode (DESIGN.md §3.1)"
        )
    if t.shape[0] != batch_size:
        raise ContractError(
            f"t has batch axis {t.shape[0]} but this distribution was built for "
            f"{batch_size} rows. A singleton axis here broadcasts silently and "
            "scores every row under row 0 (DESIGN.md §3.1)."
        )
    if t.numel() and (int(t.min()) < 0 or int(t.max()) >= num_treatments):
        raise ContractError(
            f"t holds index {int(t.min())}..{int(t.max())}, outside [0, "
            f"{num_treatments}). Missing treatments are carried by a mask, never "
            "by a sentinel index (DESIGN.md §1.1)."
        )
    return "observed" if t.ndim == 1 else "candidate"


def _select(table: Tensor, t: Tensor) -> Tensor:
    """Gather `table: [B, K, *Dy]` at `t`, inserting the candidate axis.

    Returns `[B, *Dy]` for `t: [B]` and `[B, C, *Dy]` for `t: [B, C]`.
    """
    index = t[:, None] if t.ndim == 1 else t
    while index.ndim < table.ndim:
        index = index.unsqueeze(-1)
    index = index.expand(index.shape[0], index.shape[1], *table.shape[2:])
    gathered = torch.gather(table, 1, index)
    return gathered.squeeze(1) if t.ndim == 1 else gathered


@dataclass(frozen=True, eq=False)
class GaussianOutcome:
    """The reference `OutcomeDistribution`: one Gaussian per treatment value.

    This is deliberately the most trivial head that can satisfy the contract —
    it holds a table of means and scales rather than computing anything — so
    that it documents the *contract* and not a parameterisation. Real heads
    (P5's `tarnet_head`, P7's `conditional_flow`) are checked against the same
    conformance function.

    Attributes:
        loc: `[B, K, *Dy]` mean per row and treatment.
        scale: `[B, K, *Dy]` positive standard deviation per row and treatment.
    """

    loc: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        if self.loc.shape != self.scale.shape:
            raise ContractError(
                f"loc {tuple(self.loc.shape)} and scale {tuple(self.scale.shape)} "
                "must have the same shape"
            )
        if self.loc.ndim < 2:
            raise ContractError(
                f"loc must be [B, K, *Dy], got shape {tuple(self.loc.shape)}"
            )
        # `scale.min() <= 0` is false for NaN, which would let a NaN scale
        # through and turn every log-density downstream into NaN.
        if not bool((torch.isfinite(self.scale) & (self.scale > 0)).all()):
            raise ContractError("scale must be finite and positive")

    @property
    def batch_size(self) -> int:
        """`B`, the number of rows this distribution represents."""
        return int(self.loc.shape[0])

    @property
    def num_treatments(self) -> int:
        """`K`."""
        return int(self.loc.shape[1])

    @property
    def event_shape(self) -> tuple[int, ...]:
        """`Dy`, the trailing shape of one outcome."""
        return tuple(self.loc.shape[2:])

    def log_prob(self, y: Tensor, t: Tensor) -> Tensor:
        """`log p(y | x, t)`, summed over `Dy`. `y` arrives unexpanded."""
        mode = treatment_mode(t, self.num_treatments, batch_size=self.batch_size)
        if y.shape[0] != self.batch_size:
            raise ContractError(
                f"y has batch axis {y.shape[0]} but this distribution was built "
                f"for {self.batch_size} rows"
            )
        if tuple(y.shape[1:]) != self.event_shape:
            raise ContractError(
                f"y has trailing shape {tuple(y.shape[1:])}, expected "
                f"{self.event_shape} — pass y unexpanded (DESIGN.md §3.1)"
            )
        loc = _select(self.loc, t)
        scale = _select(self.scale, t)
        # The head inserts the candidate axis; the caller never reshapes y.
        target = y if mode == "observed" else y[:, None]
        z = (target - loc) / scale
        pointwise = -0.5 * z * z - scale.log() - 0.5 * math.log(2.0 * math.pi)
        event_dims = len(self.event_shape)
        if event_dims:
            return pointwise.sum(dim=tuple(range(-event_dims, 0)))
        return pointwise

    def mean(self, t: Tensor) -> Tensor:
        """`E[y | x, t]`, with the candidate axis immediately after the batch axis."""
        treatment_mode(t, self.num_treatments, batch_size=self.batch_size)
        return _select(self.loc, t)

    def sample(self, t: Tensor, n: int) -> Tensor:
        """`n` samples, with the sample axis leading."""
        treatment_mode(t, self.num_treatments, batch_size=self.batch_size)
        if n < 1:
            raise ContractError(f"n must be at least 1, got {n}")
        loc = _select(self.loc, t)
        scale = _select(self.scale, t)
        noise = torch.randn((n, *loc.shape), dtype=loc.dtype, device=loc.device)
        return loc + scale * noise


@dataclass(frozen=True, eq=False)
class CategoricalTreatment:
    """The reference `TreatmentDistribution`, parameterised by logits `[B, K]`.

    `log_prob` follows the same rank rule as the outcome contract, so
    `log_prob(all_k)` gives the `[B, K]` term the marginalisation objective
    needs (`DESIGN.md` §4.1) in log space, rather than via `probs.log()`.
    """

    logits: Tensor

    def __post_init__(self) -> None:
        if self.logits.ndim != 2:
            raise ContractError(
                f"logits must be [B, K], got shape {tuple(self.logits.shape)}"
            )

    @property
    def batch_size(self) -> int:
        """`B`, the number of rows this distribution represents."""
        return int(self.logits.shape[0])

    @property
    def num_treatments(self) -> int:
        """`K`."""
        return int(self.logits.shape[1])

    @property
    def probs(self) -> Tensor:
        """`[B, K]`, rows summing to 1."""
        return torch.softmax(self.logits, dim=-1)

    @property
    def log_probs(self) -> Tensor:
        """`[B, K]` in log space, without a round trip through `probs`."""
        return torch.log_softmax(self.logits, dim=-1)

    def log_prob(self, t: Tensor) -> Tensor:
        """`log p(t | ·)`: `[B] -> [B]`, `[B, C] -> [B, C]`."""
        mode = treatment_mode(t, self.num_treatments, batch_size=self.batch_size)
        index = t[:, None] if mode == "observed" else t
        gathered = torch.gather(self.log_probs, 1, index)
        return gathered.squeeze(1) if mode == "observed" else gathered
