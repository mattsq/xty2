"""The conformance tests every distribution implementation is run against.

`DESIGN.md` §3.1 specifies the candidate-treatment contract as a shape table
and requires an executable check of it; `FIDELITY.md` §3 lists that check and
the normalisation check as Tier 0 invariants. They live in the package rather
than in `tests/` for one reason: **every** outcome head added later — P5's
`tarnet_head`, P7's conditional flow, any energy head after that — has to be
run against exactly this function, and a check that each packet is free to
re-write is not a contract.

Both functions raise `ContractError` on failure and return `None` on success.
They raise rather than assert so they keep working under `python -O`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from torch import Tensor

from xty2.core.distributions import OutcomeDistribution, TreatmentDistribution
from xty2.core.errors import ContractError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


@contextmanager
def _fixed_seed(seed: int) -> Iterator[None]:
    """Seed the global RNG and restore the caller's state afterwards."""
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        yield
    finally:
        torch.random.set_rng_state(state)


def _candidate_matrix(
    batch_size: int, num_treatments: int, device: torch.device
) -> Tensor:
    """`M[i, k] = k`, the candidate matrix the contract is stated against.

    Building it from the *observed* `t` instead (`t[:, None].expand(B, K)`)
    repeats `t_i` across every column, which turns the column-agreement
    assertion into a demand for treatment-insensitivity and fails every correct
    head. That mistake is in `FIDELITY.md` §3 because it was made.
    """
    row = torch.arange(num_treatments, dtype=torch.long, device=device)
    return row[None, :].expand(batch_size, num_treatments)


def _reject_square_batch(batch_size: int, num_treatments: int) -> None:
    if batch_size == num_treatments:
        raise ValueError(
            f"run this check with B != K (got B = K = {batch_size}). A batch where "
            "B == K passes under accidental broadcasting and proves nothing "
            "(FIDELITY.md §3)."
        )


def check_outcome_distribution_contract(
    distribution: OutcomeDistribution,
    *,
    y: Tensor,
    num_treatments: int,
    num_samples: int = 3,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Check an `OutcomeDistribution` against `DESIGN.md` §3.1.

    Args:
        distribution: The head's output for one batch.
        y: `[B, *Dy]` outcomes, passed to the distribution **unexpanded**.
        num_treatments: `K`.
        num_samples: `n` for the `sample` shape and determinism checks.
        rtol: Relative tolerance for column agreement.
        atol: Absolute tolerance for column agreement.

    Raises:
        ContractError: on any contract violation, naming the call that failed.
        ValueError: if called with `B == K`, which cannot discriminate.
    """
    _require(
        isinstance(distribution, OutcomeDistribution),
        f"{type(distribution)} does not implement log_prob / mean / sample",
    )
    batch_size = int(y.shape[0])
    _reject_square_batch(batch_size, num_treatments)
    candidates = _candidate_matrix(batch_size, num_treatments, y.device)

    # -- mean: fixes Dy, and is checked first because CATE is computed from it.
    observed_mean = distribution.mean(_column(candidates, 0))
    event_shape = tuple(observed_mean.shape[1:])
    _require(
        tuple(y.shape[1:]) == event_shape,
        f"mean(t: [B]) returned trailing shape {event_shape} but y has "
        f"{tuple(y.shape[1:])}; they are the same Dy",
    )
    _require(
        tuple(observed_mean.shape) == (batch_size, *event_shape),
        f"mean(t: [B]) must return [B, *Dy] = {(batch_size, *event_shape)}, got "
        f"{tuple(observed_mean.shape)}",
    )
    candidate_mean = distribution.mean(candidates)
    _require(
        tuple(candidate_mean.shape) == (batch_size, num_treatments, *event_shape),
        "mean(t: [B, K]) must return [B, K, *Dy] = "
        f"{(batch_size, num_treatments, *event_shape)} — the candidate axis goes "
        f"immediately after the batch axis — got {tuple(candidate_mean.shape)}",
    )
    for k in range(num_treatments):
        column = distribution.mean(_column(candidates, k))
        _require(
            torch.allclose(candidate_mean[:, k], column, rtol=rtol, atol=atol),
            f"mean(t: [B, K])[:, {k}] disagrees with mean(t = {k}). A head that "
            "broadcasts the candidate axis ambiguously corrupts every CATE "
            "estimate downstream (DESIGN.md §3.1).",
        )

    # -- log_prob: y is passed unexpanded in both calls.
    candidate_log_prob = distribution.log_prob(y, candidates)
    _require(
        tuple(candidate_log_prob.shape) == (batch_size, num_treatments),
        "log_prob(y, t: [B, K]) must return [B, K] = "
        f"{(batch_size, num_treatments)}, got {tuple(candidate_log_prob.shape)}",
    )
    for k in range(num_treatments):
        column = distribution.log_prob(y, _column(candidates, k))
        _require(
            tuple(column.shape) == (batch_size,),
            f"log_prob(y, t: [B]) must return [B] = {(batch_size,)}, got "
            f"{tuple(column.shape)}",
        )
        _require(
            torch.allclose(candidate_log_prob[:, k], column, rtol=rtol, atol=atol),
            f"log_prob(y, t: [B, K])[:, {k}] disagrees with log_prob(y, t = {k})",
        )

    # -- sample: shape, sample axis leading, and determinism under a seed.
    with _fixed_seed(0):
        candidate_sample = distribution.sample(candidates, num_samples)
    _require(
        tuple(candidate_sample.shape)
        == (num_samples, batch_size, num_treatments, *event_shape),
        "sample(t: [B, K], n) must return [n, B, K, *Dy] = "
        f"{(num_samples, batch_size, num_treatments, *event_shape)}, got "
        f"{tuple(candidate_sample.shape)}",
    )
    with _fixed_seed(0):
        repeated = distribution.sample(candidates, num_samples)
    _require(
        bool(torch.equal(candidate_sample, repeated)),
        "sample is not deterministic given a seed; it must draw from the global "
        "RNG so a run is reproducible (FIDELITY.md §3)",
    )
    with _fixed_seed(0):
        observed_sample = distribution.sample(_column(candidates, 0), num_samples)
    _require(
        tuple(observed_sample.shape) == (num_samples, batch_size, *event_shape),
        "sample(t: [B], n) must return [n, B, *Dy] = "
        f"{(num_samples, batch_size, *event_shape)}, got "
        f"{tuple(observed_sample.shape)}",
    )


def check_treatment_distribution_contract(
    distribution: TreatmentDistribution,
    *,
    num_treatments: int,
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> None:
    """Check a `TreatmentDistribution`: normalisation, and the same rank rule.

    Raises:
        ContractError: on any contract violation.
        ValueError: if called with `B == K`.
    """
    _require(
        isinstance(distribution, TreatmentDistribution),
        f"{type(distribution)} does not implement probs / log_prob",
    )
    probs = distribution.probs
    _require(
        probs.ndim == 2 and int(probs.shape[1]) == num_treatments,
        f"probs must be [B, K] with K = {num_treatments}, got {tuple(probs.shape)}",
    )
    batch_size = int(probs.shape[0])
    _reject_square_batch(batch_size, num_treatments)
    _require(bool((probs >= 0).all()), "probs must be non-negative")
    totals = probs.sum(dim=-1)
    _require(
        torch.allclose(totals, torch.ones_like(totals), rtol=rtol, atol=atol),
        f"probs rows must sum to 1, got {totals.tolist()}",
    )

    candidates = _candidate_matrix(batch_size, num_treatments, probs.device)
    candidate_log_prob = distribution.log_prob(candidates)
    _require(
        tuple(candidate_log_prob.shape) == (batch_size, num_treatments),
        "log_prob(t: [B, K]) must return [B, K] = "
        f"{(batch_size, num_treatments)}, got {tuple(candidate_log_prob.shape)}",
    )
    _require(
        torch.allclose(candidate_log_prob, probs.log(), rtol=rtol, atol=atol),
        "log_prob disagrees with log(probs)",
    )
    for k in range(num_treatments):
        column = distribution.log_prob(_column(candidates, k))
        _require(
            tuple(column.shape) == (batch_size,),
            f"log_prob(t: [B]) must return [B] = {(batch_size,)}, got "
            f"{tuple(column.shape)}",
        )
        _require(
            torch.allclose(candidate_log_prob[:, k], column, rtol=rtol, atol=atol),
            f"log_prob(t: [B, K])[:, {k}] disagrees with log_prob(t = {k})",
        )


def _column(candidates: Tensor, k: int) -> Tensor:
    """`[B]` filled with `k`, taken from the candidate matrix itself."""
    return candidates[:, k].contiguous()
