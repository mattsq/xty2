"""Tier 0 — the candidate-treatment contract (`DESIGN.md` §3.1).

This is the invariant the whole design leans on: it is what lets
`MissingTreatmentMarginalNLL` (P3) work unchanged across a TARNet head, a
conditional flow and an energy model. Three attempts to state it in prose were
wrong, so it is stated here as an executable reference implementation
(`GaussianOutcome`) plus the conformance function every later head is run
against.

The second half of the file is the mutation test for the conformance function
itself: five heads that are wrong in the five ways heads are actually wrong,
each of which must be *caught*. A checker that has never been seen to fail is
not known to work.
"""

import math
import random

import pytest
import torch
from torch import Tensor
from xty2.core import (
    CategoricalTreatment,
    ContractError,
    GaussianOutcome,
    OutcomeDistribution,
    TreatmentDistribution,
    check_outcome_distribution_contract,
    check_treatment_distribution_contract,
    treatment_mode,
)

from tests.invariants.conftest import BATCH_SIZE, NUM_TREATMENTS

EVENT_SHAPES: tuple[tuple[int, ...], ...] = ((), (2,))


def make_gaussian(event_shape: tuple[int, ...] = ()) -> GaussianOutcome:
    shape = (BATCH_SIZE, NUM_TREATMENTS, *event_shape)
    return GaussianOutcome(loc=torch.randn(shape), scale=torch.rand(shape) + 0.5)


def make_y(event_shape: tuple[int, ...] = ()) -> Tensor:
    return torch.randn(BATCH_SIZE, *event_shape)


# -- the reference implementation satisfies the contract --------------------


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=["scalar-y", "vector-y"])
def test_the_reference_head_conforms(event_shape: tuple[int, ...]) -> None:
    assert BATCH_SIZE != NUM_TREATMENTS, "a B == K test proves nothing"
    check_outcome_distribution_contract(
        make_gaussian(event_shape), y=make_y(event_shape), num_treatments=NUM_TREATMENTS
    )


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=["scalar-y", "vector-y"])
def test_the_shape_table_holds_exactly(event_shape: tuple[int, ...]) -> None:
    head, y = make_gaussian(event_shape), make_y(event_shape)
    observed = torch.randint(0, NUM_TREATMENTS, (BATCH_SIZE,))
    candidates = torch.arange(NUM_TREATMENTS)[None, :].expand(
        BATCH_SIZE, NUM_TREATMENTS
    )

    assert head.log_prob(y, observed).shape == (BATCH_SIZE,)
    assert head.log_prob(y, candidates).shape == (BATCH_SIZE, NUM_TREATMENTS)
    assert head.mean(observed).shape == (BATCH_SIZE, *event_shape)
    assert head.mean(candidates).shape == (BATCH_SIZE, NUM_TREATMENTS, *event_shape)
    assert head.sample(observed, 4).shape == (4, BATCH_SIZE, *event_shape)
    assert head.sample(candidates, 4).shape == (
        4,
        BATCH_SIZE,
        NUM_TREATMENTS,
        *event_shape,
    )


def test_log_prob_is_a_gaussian_density() -> None:
    """The reference is checked against the closed form, not only against itself."""
    head, y = make_gaussian(), make_y()
    t = torch.randint(0, NUM_TREATMENTS, (BATCH_SIZE,))
    loc = head.loc[torch.arange(BATCH_SIZE), t]
    scale = head.scale[torch.arange(BATCH_SIZE), t]
    expected = (
        -0.5 * ((y - loc) / scale) ** 2 - scale.log() - 0.5 * math.log(2.0 * math.pi)
    )
    assert torch.allclose(head.log_prob(y, t), expected)


def test_y_is_passed_unexpanded_and_the_head_inserts_the_candidate_axis() -> None:
    """Expanding `y` at the call site is the mistake the contract forbids."""
    head, y = make_gaussian(), make_y()
    candidates = torch.arange(NUM_TREATMENTS)[None, :].expand(
        BATCH_SIZE, NUM_TREATMENTS
    )
    head.log_prob(y, candidates)
    with pytest.raises(ContractError, match="pass y unexpanded"):
        head.log_prob(y[:, None].expand(BATCH_SIZE, NUM_TREATMENTS), candidates)


def test_the_rank_of_t_selects_the_mode() -> None:
    observed = torch.zeros(BATCH_SIZE, dtype=torch.long)
    assert treatment_mode(observed, NUM_TREATMENTS, batch_size=BATCH_SIZE) == "observed"
    assert (
        treatment_mode(observed[:, None], NUM_TREATMENTS, batch_size=BATCH_SIZE)
        == "candidate"
    )
    with pytest.raises(ContractError, match="rank of t selects the mode"):
        treatment_mode(observed[None, :, None], NUM_TREATMENTS, batch_size=1)


@pytest.mark.parametrize(
    ("t", "message"),
    [
        (torch.zeros(BATCH_SIZE), "long tensor"),
        (torch.full((BATCH_SIZE,), -1, dtype=torch.long), "never"),
        (torch.full((BATCH_SIZE,), NUM_TREATMENTS, dtype=torch.long), "outside"),
    ],
)
def test_a_treatment_index_is_validated(t: Tensor, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        treatment_mode(t, NUM_TREATMENTS, batch_size=BATCH_SIZE)


def test_a_candidate_column_may_be_any_width() -> None:
    """`[B, C]` is not restricted to `C == K`; the marginal term uses `C == K`."""
    head, y = make_gaussian(), make_y()
    two = torch.zeros(BATCH_SIZE, 2, dtype=torch.long)
    assert head.log_prob(y, two).shape == (BATCH_SIZE, 2)
    assert head.mean(two).shape == (BATCH_SIZE, 2)


def test_the_reference_head_validates_its_own_parameters() -> None:
    with pytest.raises(ContractError, match="same shape"):
        GaussianOutcome(loc=torch.zeros(BATCH_SIZE, 3), scale=torch.ones(BATCH_SIZE, 2))
    with pytest.raises(ContractError, match=r"\[B, K, \*Dy\]"):
        GaussianOutcome(loc=torch.zeros(BATCH_SIZE), scale=torch.ones(BATCH_SIZE))
    with pytest.raises(ContractError, match="scale must be finite and positive"):
        GaussianOutcome(
            loc=torch.zeros(BATCH_SIZE, 3), scale=torch.zeros(BATCH_SIZE, 3)
        )
    with pytest.raises(ContractError, match="at least 1"):
        make_gaussian().sample(torch.zeros(BATCH_SIZE, dtype=torch.long), 0)


def test_the_checker_refuses_a_square_batch() -> None:
    """`B == K` passes under accidental broadcasting, so it is not allowed."""
    head = GaussianOutcome(
        loc=torch.randn(NUM_TREATMENTS, NUM_TREATMENTS),
        scale=torch.ones(NUM_TREATMENTS, NUM_TREATMENTS),
    )
    with pytest.raises(ValueError, match="B != K"):
        check_outcome_distribution_contract(
            head, y=torch.randn(NUM_TREATMENTS), num_treatments=NUM_TREATMENTS
        )


# -- the mutation test for the conformance function -------------------------


class _ReversedCandidateMean(GaussianOutcome):
    """Returns the candidate axis in the wrong order — a silently wrong CATE."""

    def mean(self, t: Tensor) -> Tensor:
        out = super().mean(t)
        return out.flip(1) if t.ndim == 2 else out


class _TransposedCandidateAxis(GaussianOutcome):
    """Puts the candidate axis before the batch axis."""

    def mean(self, t: Tensor) -> Tensor:
        out = super().mean(t)
        return out.transpose(0, 1) if t.ndim == 2 else out


class _SwappedLogProbColumns(GaussianOutcome):
    """Candidate `log_prob` disagrees with the per-treatment call."""

    def log_prob(self, y: Tensor, t: Tensor) -> Tensor:
        out = super().log_prob(y, t)
        return out.roll(1, dims=1) if t.ndim == 2 else out


class _TrailingSampleAxis(GaussianOutcome):
    """Samples with the sample axis last instead of leading."""

    def sample(self, t: Tensor, n: int) -> Tensor:
        return super().sample(t, n).movedim(0, -1)


class _UnseededSample(GaussianOutcome):
    """Draws from a private generator, so a seeded run is not reproducible."""

    def sample(self, t: Tensor, n: int) -> Tensor:
        # stdlib entropy on purpose: torch's global seed is what is under test.
        generator = torch.Generator().manual_seed(random.randrange(1 << 30))
        loc = super().mean(t)
        noise = torch.randn(n, *loc.shape, generator=generator)
        return loc + noise


class _NotADistribution:
    """Has `log_prob` but no `mean` or `sample`."""

    def log_prob(self, y: Tensor, t: Tensor) -> Tensor:
        return torch.zeros(y.shape[0])


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (_ReversedCandidateMean, "disagrees with mean"),
        (_TransposedCandidateAxis, "immediately after the batch axis"),
        (_SwappedLogProbColumns, "disagrees with log_prob"),
        (_TrailingSampleAxis, r"sample\(t: \[B, K\], n\)"),
        (_UnseededSample, "not deterministic"),
    ],
)
def test_the_conformance_check_catches_a_broken_head(
    broken: type[GaussianOutcome], message: str
) -> None:
    shape = (BATCH_SIZE, NUM_TREATMENTS)
    head = broken(loc=torch.randn(shape), scale=torch.rand(shape) + 0.5)
    with pytest.raises(ContractError, match=message):
        check_outcome_distribution_contract(
            head, y=make_y(), num_treatments=NUM_TREATMENTS
        )


def test_an_object_missing_the_protocol_is_not_an_outcome_distribution() -> None:
    assert not isinstance(_NotADistribution(), OutcomeDistribution)
    with pytest.raises(ContractError, match="does not implement"):
        check_outcome_distribution_contract(
            _NotADistribution(),  # type: ignore[arg-type]
            y=make_y(),
            num_treatments=NUM_TREATMENTS,
        )


# -- treatment distributions: normalisation and the same rank rule ----------


def test_the_reference_propensity_conforms() -> None:
    head = CategoricalTreatment(logits=torch.randn(BATCH_SIZE, NUM_TREATMENTS))
    check_treatment_distribution_contract(head, num_treatments=NUM_TREATMENTS)
    assert isinstance(head, TreatmentDistribution)
    assert torch.allclose(head.log_probs, head.probs.log(), atol=1e-6)
    assert head.log_prob(torch.zeros(BATCH_SIZE, dtype=torch.long)).shape == (
        BATCH_SIZE,
    )


class _UnnormalisedTreatment:
    """Exponentiates logits without normalising — the classic silent bug."""

    def __init__(self, logits: Tensor) -> None:
        self._logits = logits

    @property
    def probs(self) -> Tensor:
        return self._logits.exp()

    def log_prob(self, t: Tensor) -> Tensor:
        index = t[:, None] if t.ndim == 1 else t
        out = torch.gather(self._logits, 1, index)
        return out.squeeze(1) if t.ndim == 1 else out


class _LogProbDisagreesWithProbs(CategoricalTreatment):
    def log_prob(self, t: Tensor) -> Tensor:
        return super().log_prob(t) - 1.0


@pytest.mark.parametrize(
    ("broken", "message"),
    [
        (_UnnormalisedTreatment, "must sum to 1"),
        (_LogProbDisagreesWithProbs, "disagrees with log"),
    ],
)
def test_the_treatment_check_catches_a_broken_head(
    broken: type[TreatmentDistribution], message: str
) -> None:
    head = broken(torch.randn(BATCH_SIZE, NUM_TREATMENTS))  # type: ignore[call-arg]
    with pytest.raises(ContractError, match=message):
        check_treatment_distribution_contract(head, num_treatments=NUM_TREATMENTS)


def test_a_mismatched_y_batch_axis_is_not_broadcast_away() -> None:
    head = make_gaussian()
    t = torch.zeros(BATCH_SIZE, dtype=torch.long)
    with pytest.raises(ContractError, match="y has batch axis"):
        head.log_prob(torch.randn(1), t)


@pytest.mark.parametrize("event_shape", EVENT_SHAPES, ids=["scalar-y", "vector-y"])
def test_a_mismatched_t_batch_axis_is_not_broadcast_away(
    event_shape: tuple[int, ...],
) -> None:
    """The failure this rejects returns a *correctly shaped* wrong answer.

    With `t: [1]`, `gather` yields row 0's parameters and ambient broadcasting
    then scores every row of `y` under them, returning a plausible `[B]`. No
    downstream shape check can see it, so it is rejected at the source.
    """
    head, y = make_gaussian(event_shape), make_y(event_shape)
    for t in (
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, NUM_TREATMENTS, dtype=torch.long),
        torch.zeros(BATCH_SIZE - 1, dtype=torch.long),
    ):
        with pytest.raises(ContractError, match="t has batch axis"):
            head.log_prob(y, t)
        with pytest.raises(ContractError, match="t has batch axis"):
            head.mean(t)
        with pytest.raises(ContractError, match="t has batch axis"):
            head.sample(t, 2)


def test_the_propensity_rejects_a_mismatched_t_batch_axis() -> None:
    head = CategoricalTreatment(logits=torch.randn(BATCH_SIZE, NUM_TREATMENTS))
    with pytest.raises(ContractError, match="t has batch axis"):
        head.log_prob(torch.zeros(1, dtype=torch.long))


def test_a_nan_scale_is_rejected() -> None:
    """`scale.min() <= 0` is false for NaN; the check must not rely on it."""
    scale = torch.ones(BATCH_SIZE, NUM_TREATMENTS)
    scale[0, 0] = float("nan")
    with pytest.raises(ContractError, match="finite and positive"):
        GaussianOutcome(loc=torch.zeros(BATCH_SIZE, NUM_TREATMENTS), scale=scale)
    scale[0, 0] = float("inf")
    with pytest.raises(ContractError, match="finite and positive"):
        GaussianOutcome(loc=torch.zeros(BATCH_SIZE, NUM_TREATMENTS), scale=scale)
