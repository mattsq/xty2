"""Shared deterministic mechanics for the card-defined benchmark cases."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import torch
from torch import Tensor

from xty2.core import (
    Dataset,
    FeatureSpec,
    OutcomeSpec,
    Schema,
    TrainingPopulation,
    XTYBatch,
)

Replicate = Mapping[str, float]
ReplicateFunction = Callable[[int], dict[str, float]]


def continuous_schema(features: int) -> Schema:
    """The all-continuous scalar-outcome schema used by all five cards."""
    return Schema(
        features=tuple(
            FeatureSpec(f"x{column}", "continuous") for column in range(features)
        ),
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


def configure_worker() -> None:
    """Make small CPU fits deterministic and avoid thread oversubscription."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Torch permits setting the inter-op pool only before the first
        # parallel operation. A one-worker run configures several replicates
        # in the same process, where the existing value is already one.
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)


def parallel_replicates(
    function: ReplicateFunction,
    count: int,
    *,
    workers: int,
) -> tuple[dict[str, float], ...]:
    """Run seed-indexed replicates in stable order, optionally in processes."""
    if count < 1:
        raise ValueError(f"replicate count must be positive, got {count}")
    if workers < 1:
        raise ValueError(f"worker count must be positive, got {workers}")
    resolved_workers = min(workers, count)
    if resolved_workers == 1:
        return tuple(function(index) for index in range(count))
    # Spawn rather than fork: forking after importing torch can inherit a live
    # thread pool and deadlock. Each child configures one deterministic thread.
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=resolved_workers,
        mp_context=context,
    ) as executor:
        return tuple(executor.map(function, range(count)))


def default_workers() -> int:
    """A conservative default that fits GitHub's ordinary hosted runners."""
    value = os.environ.get("XTY2_TIER2_WORKERS")
    if value is not None:
        try:
            parsed = int(value)
        except ValueError:
            raise ValueError(
                f"XTY2_TIER2_WORKERS must be a positive integer, got {value!r}"
            ) from None
        if parsed < 1:
            raise ValueError(f"XTY2_TIER2_WORKERS must be positive, got {parsed}")
        return parsed
    return min(4, os.cpu_count() or 1)


def column(rows: Sequence[Replicate], name: str) -> tuple[float, ...]:
    """Extract one finite scalar from every replicate."""
    values: list[float] = []
    for index, replicate in enumerate(rows):
        try:
            value = float(replicate[name])
        except KeyError:
            raise ValueError(
                f"replicate {index} has no metric {name!r}; it has "
                f"{sorted(replicate)!r}"
            ) from None
        if not math.isfinite(value):
            raise ValueError(
                f"replicate {index} metric {name!r} is not finite: {value!r}"
            )
        values.append(value)
    return tuple(values)


def take(batch: XTYBatch, rows: Tensor) -> XTYBatch:
    """Functional row selection preserving every optional batch field."""
    return XTYBatch(
        x=batch.x.index_select(0, rows),
        t=batch.t.index_select(0, rows),
        y=batch.y.index_select(0, rows),
        t_observed=batch.t_observed.index_select(0, rows),
        y_observed=batch.y_observed.index_select(0, rows),
        row_id=batch.row_id.index_select(0, rows),
        fold_id=(
            None if batch.fold_id is None else batch.fold_id.index_select(0, rows)
        ),
        weight=(None if batch.weight is None else batch.weight.index_select(0, rows)),
    )


@dataclass(frozen=True)
class BatchStream:
    """A re-iterable ordered batch table shared by paired fits."""

    population: XTYBatch
    indices: Tensor

    def __post_init__(self) -> None:
        if self.indices.dtype != torch.long or self.indices.ndim != 2:
            raise ValueError(
                "batch indices must be a [steps,batch_size] long tensor, got "
                f"{self.indices.dtype} {tuple(self.indices.shape)}"
            )

    def __iter__(self) -> Iterator[XTYBatch]:
        for rows in self.indices:
            yield take(self.population, rows)


def batch_indices(
    rows: int,
    *,
    steps: int,
    batch_size: int,
    seed: int,
) -> Tensor:
    """The first ``batch_size`` rows of one fresh permutation per step."""
    if not 0 < batch_size <= rows:
        raise ValueError(f"batch_size must be in [1,{rows}], got {batch_size}")
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randperm(rows, generator=generator)[:batch_size] for _ in range(steps)]
    )


def standardise_outcome(
    train: XTYBatch,
    *others: XTYBatch,
) -> tuple[float, float, tuple[XTYBatch, ...]]:
    """Population-standardise Y using all training outcomes."""
    location = train.y.mean()
    scale = train.y.std(unbiased=False)
    if not bool(torch.isfinite(scale)) or float(scale) <= 0.0:
        raise ValueError(f"training outcome scale must be positive, got {scale}")
    transformed = tuple(
        batch.replace(y=(batch.y - location) / scale) for batch in (train, *others)
    )
    return float(location), float(scale), transformed


def bool_float(value: bool) -> float:
    """Record an executable guardrail as a numeric all-replicate metric."""
    return 1.0 if value else 0.0


__all__ = [
    "BatchStream",
    "Replicate",
    "batch_indices",
    "bool_float",
    "cluster_centres",
    "cluster_population",
    "column",
    "configure_worker",
    "continuous_schema",
    "default_workers",
    "parallel_replicates",
    "standardise_outcome",
    "take",
]


@dataclass(frozen=True)
class ClusterPopulation:
    """One draw of the cluster DGP, with its analytic `t=1` vs `t=0` effect."""

    batch: XTYBatch
    true_effect: Tensor


CLUSTER_FEATURES = 6
CLUSTER_SIGNAL = 0.45
SEPARATED = 0.02
"""`p(t=1 | c=0)`; the assignment is 0.02/0.98 and a confident gate can open."""

CLUSTER_SEPARATION = 2.0 * CLUSTER_SIGNAL * 2.0
"""Centre-to-centre distance between any two clusters, held fixed across `K`.

The `K = 2` fixture puts its two centres at `+-0.45` in each of four signal
columns, so they are `2 * 0.45 * sqrt(4) = 1.8` apart. Every `K` below uses that
same pairwise distance, which is what makes `K` the *only* thing a comparison
across `K` varies: each pair of classes is exactly as separable as the two
classes of the original fixture, and what grows is the number of neighbours a
row can be confused with, not the difficulty of any one confusion.

The alternative — holding the centres' *radius* fixed — crowds the simplex as
`K` grows and would confound "more classes" with "closer classes".
"""

SIGNAL_COLUMNS = 4
"""Columns 0-3 carry the cluster signal; 4-5 are outcome-only covariates."""

_HADAMARD_4 = (
    (1.0, 1.0, 1.0, 1.0),
    (1.0, -1.0, 1.0, -1.0),
    (1.0, 1.0, -1.0, -1.0),
    (1.0, -1.0, -1.0, 1.0),
)
"""A 4x4 Hadamard matrix. Divided by 2 it is orthogonal, and it is what spreads
each cluster centre across all four signal columns (`cluster_centres`)."""


def _helmert(classes: int) -> Tensor:
    """An orthonormal basis `[K, K-1]` of the sum-zero hyperplane in `R^K`.

    The Helmert basis, in its textbook form: column `j` contrasts the first `j`
    coordinates against coordinate `j`. Closed form and deterministic, so the
    cluster centres are a stated construction rather than a stored table.
    """
    columns = []
    for split in range(1, classes):
        vector = torch.zeros(classes, dtype=torch.float64)
        vector[:split] = 1.0 / split
        vector[split] = -1.0
        columns.append(vector / vector.norm())
    return torch.stack(columns, dim=1)


def cluster_centres(classes: int, *, separation: float = CLUSTER_SEPARATION) -> Tensor:
    """`[K, 4]` cluster centres: a regular simplex, rotated to fill four columns.

    Three properties, and each is a requirement rather than an aesthetic:

    * **Equidistant.** The centres are the vertices of a regular simplex, so
      every pair of classes is exactly `separation` apart. A comparison across
      `K` therefore varies the number of classes and nothing else.
    * **Redundant across the four signal columns.** The simplex is built in
      `R^(K-1)`, padded to `R^4` and then rotated by a Hadamard matrix, which
      spreads each centre's mass over all four columns. This is load-bearing
      rather than tidy: every recipe on this fixture uses `FeatureMask` views,
      and `flexmatch.md` §5.2's label-preservation argument depends on a masked
      column leaving signal behind in the others. An unrotated simplex would
      concentrate `K = 2`'s signal in one column and make the weak view
      destroy the label.
    * **`K = 2` is the existing fixture, exactly.** The construction returns
      `[[-0.45] * 4, [+0.45] * 4]` at `K = 2` — bit-for-bit the centres of
      `fixmatch.md` §6.1 — so the two-class arm of any sweep across `K` is not
      a comparable world but literally the same one. Tier 0 asserts it.

    `K <= 5` because a regular simplex on `K` vertices needs `K - 1` dimensions
    and there are four signal columns. A card wanting more levels needs more
    signal columns, which is a different fixture and a different §6.
    """
    if not 2 <= classes <= SIGNAL_COLUMNS + 1:
        raise ValueError(
            f"cluster_centres supports 2 <= K <= {SIGNAL_COLUMNS + 1}, got "
            f"{classes}: a regular simplex on K vertices spans K - 1 "
            f"dimensions and this DGP has {SIGNAL_COLUMNS} signal columns."
        )
    if not separation > 0.0:
        raise ValueError(f"separation must be positive, got {separation}")
    centred = torch.eye(classes, dtype=torch.float64) - 1.0 / classes
    vertices = -(centred @ _helmert(classes))
    vertices = torch.nn.functional.pad(vertices, (0, SIGNAL_COLUMNS - (classes - 1)))
    scale = separation / float(torch.cdist(vertices, vertices)[0, 1])
    hadamard = torch.tensor(_HADAMARD_4, dtype=torch.float64) / 2.0
    return (vertices * scale @ hadamard.T).to(torch.float32)


def _inverse_cdf(uniform: Tensor, probabilities: Tensor, classes: int) -> Tensor:
    """Sample `[rows]` classes from `[rows, K]` probabilities and one uniform.

    The cumulative sum runs over classes in **descending** index order, which is
    an arbitrary convention with one job: at `K = 2` it reduces to the original
    fixture's `t = 1 if u < p(t = 1)`, so the two-class draws are unchanged.
    """
    cumulative = probabilities.flip(-1).cumsum(-1)
    passed = (uniform[:, None] >= cumulative).sum(-1).clamp(max=classes - 1)
    return (classes - 1 - passed).long()


def cluster_population(
    rows: int,
    *,
    seed: int,
    row_offset: int,
    low: float = SEPARATED,
    classes: int = 2,
    prior: Sequence[float] | None = None,
    effects: Sequence[float] | None = None,
) -> ClusterPopulation:
    """The `K`-cluster generalisation of `fixmatch.md` §6.1's DGP.

    At `classes=2` with a uniform prior this is that DGP **bit-for-bit** — the
    same draws in the same order for the same seed — which is what lets a sweep
    across `K` or across `prior` treat the original fixture as its own control
    rather than as a neighbouring one.

    Args:
        rows: How many rows to draw.
        seed: Seeds one generator for all four draws, in the order `u_c`,
            `eps_x`, `u_t`, `eps_y`. The order is part of the protocol.
        row_offset: Where `row_id` starts, so train and test never collide.
        low: The mass the assignment puts on the *wrong* classes in total:
            `p(t = c | c) = 1 - low`, and the remaining `low` is split evenly
            over the other `K - 1`. At `K = 2` this is the original 0.02/0.98.
        classes: `K`. Two by default, so every existing caller is unchanged.
        prior: `p(cluster = c)`, length `K`. `None` is uniform, which at
            `K = 2` is the original fair coin. A skewed prior is how a card
            asks whether a mechanism that acts on the *class marginal* has
            anything to act on (`freematch.md` §6.4).
        effects: The outcome multiplier per treatment level, length `K`; `y =
            baseline + effect * effects[t]`. `None` is `(0, 1, 1, ...)`, which
            at `K = 2` is the original `y = baseline + t * effect`. Deliberately
            **not** a dose: `DESIGN.md` §11.4's `continuous-t` row and
            `BACKLOG.md` §15.9 both put dose-response outside v1, and a
            multiplier that rose with `t` would be one wearing a categorical
            costume. A card at `K > 2` states a non-monotone tuple.

    Every treatment is observed. The label budget is the *recipe's* declaration
    (`data.missingness_mechanism`), so a benchmark that masked rows here would
    be applying a policy twice.
    """
    weights = (
        torch.full((classes,), 1.0 / classes, dtype=torch.float32)
        if prior is None
        else torch.tensor(prior, dtype=torch.float32)
    )
    if weights.shape != (classes,):
        raise ValueError(
            f"prior must have one entry per class, got {tuple(weights.shape)} "
            f"for K = {classes}"
        )
    if float(weights.min()) <= 0.0 or abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError(
            f"prior must be positive and sum to 1, got {weights.tolist()!r}"
        )
    multipliers = (
        torch.arange(classes, dtype=torch.float32).clamp(max=1.0)
        if effects is None
        else torch.tensor(effects, dtype=torch.float32)
    )
    if multipliers.shape != (classes,):
        raise ValueError(
            f"effects must have one entry per class, got "
            f"{tuple(multipliers.shape)} for K = {classes}"
        )
    if not 0.0 < low < 1.0:
        raise ValueError(f"low is a probability mass in (0, 1), got {low}")

    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, CLUSTER_FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = _inverse_cdf(u_c, weights.expand(rows, classes), classes)
    x = epsilon_x.clone()
    x[:, :SIGNAL_COLUMNS] = (
        cluster_centres(classes)[cluster] + 0.6 * epsilon_x[:, :SIGNAL_COLUMNS]
    )
    assignment = torch.full((rows, classes), low / (classes - 1))
    assignment[torch.arange(rows), cluster] = 1.0 - low
    treatment = _inverse_cdf(u_t, assignment, classes)
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    true_effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    y = baseline + true_effect * multipliers[treatment] + 0.5 * epsilon_y
    return ClusterPopulation(
        batch=XTYBatch(
            x=x,
            t=treatment,
            y=y,
            t_observed=torch.ones(rows, dtype=torch.bool),
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        true_effect=true_effect,
    )


def two_cluster_population(
    rows: int,
    *,
    seed: int,
    row_offset: int,
    low: float = SEPARATED,
) -> ClusterPopulation:
    """The DGP of `fixmatch.md` §6.1, which `scarf.md` §6.1 adopts unchanged.

    One implementation for both cards, deliberately: `scarf.md` §6.1 says it is
    "the DGP of `fixmatch.md` §6.1, unchanged, so that two cards' §6 numbers are
    about the recipes rather than about two different worlds". Two transcriptions
    of one specification is how those two worlds would quietly diverge.

    It is now `cluster_population` at `classes=2`, which is the same sentence
    applied once more: the two-class fixture and the `K`-class family were two
    transcriptions until this delegated, and Tier 0 pins the draws to a digest
    taken before the two were joined.

    The draw order — `u_c`, `eps_x`, `u_t`, `eps_y` — is part of the protocol,
    and it matches `tests/smoke/test_fixmatch.py` so that Tier 1 and Tier 2 read
    the same rows from the same seed.

    Every treatment is observed. The label budget is the *recipe's* declaration
    now (`data.missingness_mechanism`), so a benchmark that masked rows here
    would be applying a policy twice.
    """
    return cluster_population(
        rows, seed=seed, row_offset=row_offset, low=low, classes=2
    )


def training_dataset(schema: Schema, train: XTYBatch) -> Dataset:
    """The training rows under the assignment name both cards' policies use."""
    return Dataset(
        schema=schema,
        rows=train,
        assignments={"train": torch.arange(train.batch_size)},
    )


def on_the_training_scale(batch: XTYBatch, population: TrainingPopulation) -> XTYBatch:
    """Apply the outcome scaling the run **fitted**, never one refitted here.

    Refitting on the held-out rows is the leakage `FIDELITY.md` §2 names first,
    and `StageResult.population` exists so that it is not the path of least
    resistance.
    """
    location = population.statistics["y_location"]
    scale = population.statistics["y_scale"]
    return batch.replace(y=(batch.y - location) / scale)
