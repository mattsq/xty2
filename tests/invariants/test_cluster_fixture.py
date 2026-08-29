"""Tier 0 — the `K`-cluster fixture family (`benchmarks/common.py`).

`fixmatch.md` §6.1's DGP has two classes and a fair coin, and three separate
questions have now died on that: FlexMatch's per-class curriculum, its
class-imbalanced probe, and FreeMatch's fairness term and the sign of its
eq. (11). `cluster_population` generalises the same DGP over `K` and over the
cluster prior so those questions have somewhere to be asked.

The property that makes the family worth having is asserted first and hardest:
**at `K = 2` with a uniform prior it is the original fixture bit-for-bit.** Not
"a comparable world" — the same draws, from the same seed, in the same order.
Everything a sweep across `K` reports is therefore anchored to a fixture whose
numbers five cards already carry, and `two_cluster_population` delegating to it
cannot have moved anything. The digests below were taken from the *pre*-refactor
implementation and are what would catch it if it had.
"""

from __future__ import annotations

import hashlib

import pytest
import torch
from xty2.evaluation.benchmarks.common import (
    CLUSTER_SEPARATION,
    CLUSTER_SIGNAL,
    SIGNAL_COLUMNS,
    cluster_centres,
    cluster_population,
    two_cluster_population,
)

ROWS = 1_024
FROZEN_DRAWS = {
    (90_001, 0.02): "efd5a31a3fe25893e6df0d80a6e32c31",
    (90_001, 0.15): "06bb8d4b25089b8338ccae89d0d67c17",
    (90_003, 0.02): "473984de71b6194f51731c92eb85d685",
    (90_003, 0.15): "b7f9d4171bb2a74626b51fc26f784790",
}
"""SHA-256 prefixes of `two_cluster_population`'s draws, taken *before* it was
expressed in terms of `cluster_population`.

A digest rather than a re-implementation on purpose: a second copy of the
arithmetic would drift with the first, where a hash of the numbers the old code
actually produced cannot. Five cards' §6 results rest on these draws.
"""

SUPPORTED = (2, 3, 4, 5)


def _digest(seed: int, low: float) -> str:
    population = two_cluster_population(ROWS, seed=seed, row_offset=0, low=low)
    running = hashlib.sha256()
    for tensor in (
        population.batch.x,
        population.batch.y,
        population.true_effect,
    ):
        running.update(tensor.to(torch.float64).numpy().tobytes())
    running.update(population.batch.t.numpy().tobytes())
    return running.hexdigest()[:32]


# ---------------------------------------------------------------------------
# The two-class fixture has not moved
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("seed", "low"), sorted(FROZEN_DRAWS))
def test_the_two_class_draws_are_the_ones_five_cards_were_measured_on(
    seed: int, low: float
) -> None:
    """The whole safety argument for generalising in place, as one assertion."""
    assert _digest(seed, low) == FROZEN_DRAWS[(seed, low)], (
        "the two-cluster DGP's draws moved. Every §6 number in fixmatch, "
        "scarf, flexmatch, doublematch and freematch was measured on these "
        "rows; a change here invalidates all of them."
    )


def test_the_generalisation_at_two_classes_is_that_fixture_exactly() -> None:
    """`cluster_population(classes=2)` and `two_cluster_population` agree."""
    for seed in (90_001, 90_003):
        theirs = two_cluster_population(ROWS, seed=seed, row_offset=0)
        ours = cluster_population(ROWS, seed=seed, row_offset=0, classes=2)
        assert torch.equal(theirs.batch.x, ours.batch.x)
        assert torch.equal(theirs.batch.t, ours.batch.t)
        assert torch.equal(theirs.batch.y, ours.batch.y)
        assert torch.equal(theirs.true_effect, ours.true_effect)
        assert torch.equal(theirs.batch.row_id, ours.batch.row_id)


def test_the_two_class_centres_are_the_declared_plus_minus_signal() -> None:
    """The construction reproduces `+-0.45` in all four columns, not near it."""
    centres = cluster_centres(2)
    assert centres.shape == (2, SIGNAL_COLUMNS)
    assert torch.equal(centres[0], torch.full((SIGNAL_COLUMNS,), -CLUSTER_SIGNAL))
    assert torch.equal(centres[1], torch.full((SIGNAL_COLUMNS,), CLUSTER_SIGNAL))


def test_a_skewed_two_class_prior_is_flexmatchs_imbalanced_variant() -> None:
    """`flexmatch.md` §6.1's probe is a member of this family, not a cousin.

    That card draws `cluster = 1[u_c < 0.15]`. The descending inverse CDF puts
    `prior=(0.85, 0.15)` on exactly that rule, so the imbalanced probe and the
    `K`-sweep are the same generator with different arguments.
    """
    population = cluster_population(
        ROWS, seed=90_001, row_offset=0, classes=2, prior=(0.85, 0.15)
    )
    generator = torch.Generator().manual_seed(90_001)
    u_c = torch.rand(ROWS, generator=generator)
    expected = (u_c < 0.15).long()
    # The cluster is not in the batch, but the treatment follows it with
    # probability 1 - low, so a 98% agreement is the readable proxy.
    assert float((population.batch.t == expected).float().mean()) > 0.95
    assert 0.1 < float(population.batch.t.float().mean()) < 0.2


# ---------------------------------------------------------------------------
# The geometry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("classes", SUPPORTED)
def test_every_pair_of_classes_is_equally_separated(classes: int) -> None:
    """A regular simplex, so `K` varies the class *count* and nothing else.

    Unequal separations would confound "more classes" with "some classes are
    harder", which is exactly the reading a per-class mechanism is being
    measured for.
    """
    centres = cluster_centres(classes)
    distances = torch.cdist(centres.double(), centres.double())
    off_diagonal = distances[~torch.eye(classes, dtype=torch.bool)]
    assert torch.allclose(
        off_diagonal, torch.full_like(off_diagonal, CLUSTER_SEPARATION), atol=1e-5
    )


@pytest.mark.parametrize("classes", SUPPORTED)
def test_the_signal_is_redundant_across_all_four_columns(classes: int) -> None:
    """Load-bearing rather than tidy (`cluster_centres`).

    Every recipe on this fixture masks features, and `flexmatch.md` §5.2's
    label-preservation argument depends on a masked column leaving signal in the
    others. A construction that concentrated a class's signal in one column
    would let the *weak* view destroy the label.
    """
    magnitude = cluster_centres(classes).abs().mean(dim=0)
    assert magnitude.shape == (SIGNAL_COLUMNS,)
    assert float(magnitude.min()) > 0.5 * float(magnitude.max()), (
        f"column signal is lopsided at K = {classes}: {magnitude.tolist()!r}"
    )


@pytest.mark.parametrize("classes", [1, 0, -1, 6, 7])
def test_a_class_count_the_four_signal_columns_cannot_hold_is_refused(
    classes: int,
) -> None:
    with pytest.raises(ValueError, match="simplex"):
        cluster_centres(classes)


# ---------------------------------------------------------------------------
# The draws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("classes", SUPPORTED)
def test_the_cluster_prior_and_the_assignment_are_the_declared_ones(
    classes: int,
) -> None:
    """`p(t = c | c) = 1 - low`, with `low` split evenly over the rest."""
    low = 0.1
    population = cluster_population(
        40_000, seed=5, row_offset=0, classes=classes, low=low
    )
    shares = torch.bincount(population.batch.t, minlength=classes).float()
    shares /= shares.sum()
    # A uniform cluster prior plus a symmetric assignment gives a uniform
    # treatment marginal, to sampling error.
    assert torch.allclose(shares, torch.full((classes,), 1.0 / classes), atol=0.02), (
        shares.tolist()
    )


def test_a_skewed_prior_reaches_the_treatment_marginal_it_declares() -> None:
    prior = (0.55, 0.25, 0.13, 0.07)
    population = cluster_population(
        40_000, seed=5, row_offset=0, classes=4, prior=prior, low=0.02
    )
    shares = torch.bincount(population.batch.t, minlength=4).float()
    shares /= shares.sum()
    assert torch.allclose(shares, torch.tensor(prior), atol=0.02), shares.tolist()


def test_the_outcome_multiplier_is_categorical_and_not_a_dose() -> None:
    """`effects` is a declared tuple, and the default reduces at `K = 2`.

    A multiplier rising with `t` would be a dose-response model wearing a
    categorical costume, which `BACKLOG.md` §15.9 and `DESIGN.md` §11.4's
    `continuous-t` row both put outside v1.
    """
    effects = (0.0, 1.0, 0.4, 1.6)
    population = cluster_population(
        20_000, seed=5, row_offset=0, classes=4, effects=effects, low=0.02
    )
    means = [
        float(population.batch.y[population.batch.t == level].mean())
        for level in range(4)
    ]
    # Ordered by the declared multiplier, not by the treatment index.
    assert means[0] < means[2] < means[1] < means[3]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior": (0.5, 0.5, 0.0)},
        {"prior": (0.5, 0.6)},
        {"prior": (1.0,)},
        {"effects": (0.0,)},
        {"low": 0.0},
        {"low": 1.0},
    ],
)
def test_an_ill_formed_declaration_is_refused(kwargs: dict[str, object]) -> None:
    classes = len(kwargs.get("prior", (0, 0, 0))) if "prior" in kwargs else 2  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        cluster_population(
            32,
            seed=1,
            row_offset=0,
            classes=classes,
            **kwargs,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# The view check every card on this fixture owes (`flexmatch.md` §5.2)
# ---------------------------------------------------------------------------


def _flip_rate(
    mask_rate: float, classes: int, rows: int = 60_000
) -> tuple[float, float]:
    """`P(Bayes label flips)` under masking, and the clean Bayes error.

    Closed form up to the Monte Carlo draw, as `flexmatch.md` §5.2's is: the
    signal columns are an isotropic Gaussian around one of `K` centres, and
    `FeatureMask` replaces a masked column with a constant carrying no
    information about the cluster, so the Bayes rule conditions on the visible
    columns alone.
    """
    deviation = 0.6
    generator = torch.Generator().manual_seed(90_001)
    centres = cluster_centres(classes).double()
    labels = torch.randint(classes, (rows,), generator=generator)
    x = centres[labels] + deviation * torch.randn(
        rows, SIGNAL_COLUMNS, generator=generator, dtype=torch.float64
    )
    visible = (
        torch.rand(rows, SIGNAL_COLUMNS, generator=generator) >= mask_rate
    ).double()

    def posterior(mask: torch.Tensor) -> torch.Tensor:
        squared = ((x[:, None, :] - centres[None]) ** 2 * mask[:, None, :]).sum(-1)
        return (-squared / (2.0 * deviation**2)).argmax(-1)

    clean = posterior(torch.ones_like(visible))
    return (
        float((clean != posterior(visible)).double().mean()),
        float((clean != labels).double().mean()),
    )


@pytest.mark.parametrize("classes", SUPPORTED)
def test_the_strong_view_stays_label_preserving_relative_to_what_is_achievable(
    classes: int,
) -> None:
    """The criterion `flexmatch.md` §5.2 states, on the yardstick `K` requires.

    That card's rule is an *absolute* 90% Bayes-label retention, chosen at
    `K = 2` and flagged there as "not robust to its own constant". It does not
    survive more classes: at a fixed pairwise separation the declared strong
    view flips 18% of Bayes labels at `K = 4`, and no layered mask at all clears
    10% at `K = 5`. That is not the view becoming careless — it is a fixed
    budget being a harsher standard the further chance-level sits from 100%.

    Measured against what is *achievable* instead, the view is as
    label-preserving at every `K` as the one that card reviewed: the flip rate
    stays within a quarter of the clean Bayes error, where `K = 2` sits at 1.13.
    """
    weak, strong = 0.10, 1.0 - 0.9 * 0.8
    weak_flips, bayes_error = _flip_rate(weak, classes)
    strong_flips, _ = _flip_rate(strong, classes)
    assert weak_flips < strong_flips, "the strong view must be the stronger one"
    assert strong_flips / bayes_error < 1.25, (
        f"K = {classes}: the strong view flips {strong_flips:.3f} of Bayes "
        f"labels against an achievable {bayes_error:.3f}"
    )
    assert weak_flips / bayes_error < 0.5


# ---------------------------------------------------------------------------
# Class adjacency: unequal confidence at equal frequency
# ---------------------------------------------------------------------------

ADJACENT = {"groups": (3, 1), "within": 0.5}
"""`freematch.md` §6.4's `k4_adjacent`: three crowded classes and one apart."""


def _bayes_confidence(classes: int, **centres: object) -> tuple[float, ...]:
    """Mean Bayes posterior of the predicted class, per class, uniform prior."""
    deviation, rows = 0.6, 120_000
    generator = torch.Generator().manual_seed(90_001)
    means = cluster_centres(classes, **centres).double()  # type: ignore[arg-type]
    labels = torch.randint(classes, (rows,), generator=generator)
    x = means[labels] + deviation * torch.randn(
        rows, SIGNAL_COLUMNS, generator=generator, dtype=torch.float64
    )
    squared = ((x[:, None, :] - means[None]) ** 2).sum(-1)
    posterior = (-squared / (2.0 * deviation**2)).softmax(-1)
    predicted = posterior.argmax(-1)
    return tuple(
        float(posterior[predicted == level].max(-1).values.mean())
        for level in range(classes)
    )


def test_groups_of_one_are_the_regular_simplex_every_caller_already_gets() -> None:
    """The regression guard: adjacency is opt-in and changes nothing by default."""
    for classes in SUPPORTED:
        assert torch.equal(
            cluster_centres(classes), cluster_centres(classes, groups=(1,) * classes)
        )


def test_adjacency_crowds_one_group_and_isolates_the_other() -> None:
    """`groups=(3, 1)` at `within=0.5`: a tight triangle and a distant fourth."""
    centres = cluster_centres(4, **ADJACENT).double()  # type: ignore[arg-type]
    distances = torch.cdist(centres, centres)
    crowded = distances[:3, :3][~torch.eye(3, dtype=torch.bool)]
    assert torch.allclose(
        crowded, torch.full_like(crowded, 0.5 * CLUSTER_SEPARATION), atol=1e-5
    )
    isolated = distances[3, :3]
    assert float(isolated.min()) > 2.0 * float(crowded.max())


def test_adjacency_gives_unequal_confidence_at_equal_frequency() -> None:
    """The property the fixture exists for, and the contrast that proves it.

    A regular simplex has every class equally confident, so a mechanism reading
    per-class confidence has nothing to read however many classes there are or
    however skewed the prior is — `freematch.md` §6.4 measures exactly that.
    Crowding three classes and isolating a fourth separates confidence from
    frequency, which is the one arrangement eq. (9)'s `p_bar / h_bar` responds
    to.
    """
    regular = _bayes_confidence(4)
    assert max(regular) - min(regular) < 0.01, (
        f"the regular simplex should be confidence-flat, got {regular!r}"
    )

    adjacent = _bayes_confidence(4, **ADJACENT)
    crowded, isolated = adjacent[:3], adjacent[3]
    assert max(crowded) - min(crowded) < 0.02, (
        f"the crowded classes should match each other, got {crowded!r}"
    )
    assert isolated - max(crowded) > 0.15, (
        f"the isolated class should be markedly more confident, got {adjacent!r}"
    )


def test_adjacency_leaves_the_class_frequencies_alone() -> None:
    """Equal frequency is the whole point; the prior is a separate argument."""
    population = cluster_population(
        40_000,
        seed=5,
        row_offset=0,
        classes=4,
        low=0.02,
        **ADJACENT,  # type: ignore[arg-type]
    )
    shares = torch.bincount(population.batch.t, minlength=4).float()
    shares /= shares.sum()
    assert torch.allclose(shares, torch.full((4,), 0.25), atol=0.02), shares.tolist()


def test_the_adjacency_fixture_keeps_the_signal_redundant() -> None:
    magnitude = cluster_centres(4, **ADJACENT).abs().mean(dim=0)  # type: ignore[arg-type]
    assert float(magnitude.min()) > 0.5 * float(magnitude.max())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"groups": (3, 2)},
        {"groups": (0, 4)},
        {"groups": (-1, 5)},
        {"within": 0.0},
        {"within": 1.5},
        {"groups": (5, 1)},
    ],
)
def test_an_ill_formed_adjacency_declaration_is_refused(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        cluster_centres(4, **kwargs)  # type: ignore[arg-type]
