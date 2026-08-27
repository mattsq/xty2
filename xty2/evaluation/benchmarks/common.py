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
    """One draw of the two-cluster DGP, with its analytic treatment effect."""

    batch: XTYBatch
    true_effect: Tensor


CLUSTER_FEATURES = 6
CLUSTER_SIGNAL = 0.45
SEPARATED = 0.02
"""`p(t=1 | c=0)`; the assignment is 0.02/0.98 and a confident gate can open."""


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

    The draw order — `u_c`, `eps_x`, `u_t`, `eps_y` — is part of the protocol,
    and it matches `tests/smoke/test_fixmatch.py` so that Tier 1 and Tier 2 read
    the same rows from the same seed.

    Every treatment is observed. The label budget is the *recipe's* declaration
    now (`data.missingness_mechanism`), so a benchmark that masked rows here
    would be applying a policy twice.
    """
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, CLUSTER_FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < 0.5).float()
    sign = 2.0 * cluster - 1.0
    x = epsilon_x.clone()
    x[:, :4] = CLUSTER_SIGNAL * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = low + (1.0 - 2.0 * low) * cluster
    treatment = (u_t < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    true_effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    y = baseline + treatment * true_effect + 0.5 * epsilon_y
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
