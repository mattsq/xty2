"""Applying a declared data policy to a supplied `Dataset`.

`xty2.core.data` holds the declarations the compiler reads. This is what runs
them: `build_population` turns a `Dataset` plus a `DataSpec` into the
`TrainingPopulation` a stage steps on, and `iterate` turns that population plus
a `SamplerSpec` into the batch stream.

Two properties are structural rather than remembered, and each is a Tier 0
invariant:

**The sampler is the scheme the fixtures already used.** `UniformSampler` is
*defined* as one fresh permutation per step, first `batch_size` rows — byte for
byte what `xty2.evaluation.benchmarks.common.batch_indices` does — so adopting
it changes who owns the sampling, not what sampling is.

**The sampler has its own stream, and the stream does not depend on the model.**
Its generator is seeded by hashing the stage seed rather than by offsetting it,
so it cannot collide with the per-step view keys that walk upward from the same
number (`STREAM_STRIDE` in `executors.py`). Two recipes differing only in an
objective weight therefore draw identical `row_id` sequences, which is what
every paired ablation in this repository depends on.

**The dataset is never mutated.** Standardisation and missingness produce a new
`XTYBatch`; the `Dataset` handed in is bit-identical afterwards.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.data import (
    Dataset,
    DataSpec,
    ExternalBatches,
    QuotaSampler,
    SamplerSpec,
    TrainingPopulation,
    UniformSampler,
)
from xty2.core.errors import TrainingError
from xty2.core.rows import Rows

__all__ = ["build_population", "iterate", "sampler_seed"]


def sampler_seed(seed: int, *, label: str = "sampler") -> int:
    """The sampler's own 63-bit stream, hashed rather than offset.

    A stage walks one view key per optimiser step upward from its seed, so an
    offset would have to fit inside a stride budget and would collide the day
    that budget was raised. Hashing removes the arithmetic relationship
    entirely.
    """
    digest = hashlib.blake2b(f"{seed}:{label}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") >> 1


# ---------------------------------------------------------------------------
# Policy -> population
# ---------------------------------------------------------------------------


def build_population(
    dataset: Dataset,
    spec: DataSpec,
    *,
    seed: int,
) -> TrainingPopulation:
    """Apply `spec` to `dataset` and return the training rows.

    The order is split, then standardise, then apply missingness — statistics
    are fitted on the training split *before* anything else touches the rows,
    which is what makes `fitted_on_row_ids` the answer to "was this leaked".

    Raises:
        TrainingError: if the declared training assignment is absent or empty,
            or if a fitted scale is degenerate.
    """
    before = dataset.rows.clone()
    train_index = dataset.assignment(spec.split.train)
    if train_index.numel() == 0:
        raise TrainingError(
            f"assignment {spec.split.train!r} is empty, so there is nothing to "
            "fit the declared standardisation on"
        )
    train_rows = _take(dataset.rows, train_index)
    statistics = _fit(train_rows, spec)
    standardised = _apply(train_rows, statistics)
    rows = _apply_missingness(standardised, spec, seed=seed)
    if not dataset.rows.equal_to(before):
        raise TrainingError(
            "building a training population wrote into its Dataset. Loading is "
            "functional and no stage mutates the source data (DESIGN.md §7.1)."
        )
    return TrainingPopulation._issue(
        rows=rows,
        assignment=spec.split.train,
        statistics=statistics,
        fitted_on_row_ids=torch.unique(train_rows.row_id.detach().cpu()),
        spec_digest=spec.digest,
    )


def check_fitted_on(
    population: TrainingPopulation,
    dataset: Dataset,
    spec: DataSpec,
) -> None:
    """Verify the statistics were fitted on the declared training rows.

    The run-time half of the leakage rule, and the reason `fitted_on_row_ids`
    exists at all. `tarnet.md` §5.5 records what its absence cost: a runner
    could fit the standardisation on the wrong split and nothing in the plan
    would say so.

    Raises:
        TrainingError: if any fitted row lies outside the declared assignment.
    """
    declared = torch.unique(
        _take(dataset.rows, dataset.assignment(spec.split.train)).row_id.detach().cpu()
    )
    fitted = population.fitted_on_row_ids
    if fitted.numel() == 0:
        return
    known = torch.isin(fitted, declared)
    if bool(known.all()):
        return
    stray = fitted[~known]
    raise TrainingError(
        f"the training population's statistics were fitted on {int(stray.numel())} "
        f"row(s) outside assignment {spec.split.train!r} — first offenders "
        f"{stray[:5].tolist()!r}. Standardisation fitted off the training split "
        "is the leakage point FIDELITY.md §2 names first, and this is the check "
        "that makes the claim falsifiable rather than declared."
    )


def _fit(train: XTYBatch, spec: DataSpec) -> dict[str, Tensor]:
    """Fit the declared statistics on the training rows, and only those."""
    statistics: dict[str, Tensor] = {}
    if spec.preprocess.features == "zscore":
        location = train.x.mean(dim=0)
        scale = train.x.std(dim=0, unbiased=False)
        _require_positive(scale, "feature")
        statistics["x_location"] = location
        statistics["x_scale"] = scale
    if spec.preprocess.outcome == "zscore":
        location = train.y.mean(dim=0)
        scale = train.y.std(dim=0, unbiased=False)
        _require_positive(scale, "outcome")
        statistics["y_location"] = location
        statistics["y_scale"] = scale
    return statistics


def _require_positive(scale: Tensor, what: str) -> None:
    if not bool(torch.isfinite(scale).all()) or float(scale.min()) <= 0.0:
        raise TrainingError(
            f"the declared {what} standardisation has a degenerate scale "
            f"{scale.tolist()!r}: a constant column cannot be z-scored. Declare "
            f"'none' for {what} standardisation, or drop the column."
        )


def _apply(batch: XTYBatch, statistics: dict[str, Tensor]) -> XTYBatch:
    """Apply fitted statistics functionally."""
    changes: dict[str, Tensor | None] = {}
    if "x_scale" in statistics:
        changes["x"] = (batch.x - statistics["x_location"]) / statistics["x_scale"]
    if "y_scale" in statistics:
        changes["y"] = (batch.y - statistics["y_location"]) / statistics["y_scale"]
    return batch.replace(**changes) if changes else batch


def _apply_missingness(batch: XTYBatch, spec: DataSpec, *, seed: int) -> XTYBatch:
    """Induce the declared treatment missingness, keyed by `row_id`.

    Exact rather than per-row Bernoulli: the rows are ranked by a hash of their
    `row_id` and the lowest `floor(rate * N)` are made missing. A declared 50%
    is then 50% of the rows rather than 50% in expectation, which is what every
    fixture doing this by hand already produced and what a small Tier 1
    population needs in order not to wobble.
    """
    if spec.missingness.mechanism == "observed":
        return batch
    already_missing = int(batch.t_missing.sum())
    if already_missing:
        raise TrainingError(
            f"the declared mechanism induces treatment missingness, but "
            f"{already_missing} of the {batch.batch_size} training rows arrive "
            "with it already. Composing the two would make the declared budget "
            "a statement about neither: declare mechanism='observed' to consume "
            "the data's own mask, or supply labelled rows."
        )
    if spec.missingness.observed is not None:
        budget = spec.missingness.observed
        if budget > batch.batch_size:
            raise TrainingError(
                f"the declared budget of {budget} labelled rows exceeds the "
                f"{batch.batch_size}-row training population. A budget nothing "
                "can meet is a declaration the run would quietly ignore."
            )
        count = batch.batch_size - budget
    else:
        count = int(batch.batch_size * float(spec.missingness.rate or 0.0))
    if count == 0:
        return batch
    order = torch.argsort(_row_hashes(batch.row_id, seed=seed))
    observed = batch.t_observed.clone()
    observed[order[:count]] = False
    return batch.replace(t_observed=observed)


def _row_hashes(row_id: Tensor, *, seed: int) -> Tensor:
    """A stable pseudo-random key per row id, independent of row order."""
    key = sampler_seed(seed, label="missingness")
    values = [
        int.from_bytes(
            hashlib.blake2b(f"{key}:{int(row)}".encode(), digest_size=8).digest(),
            "big",
        )
        >> 11
        for row in row_id.tolist()
    ]
    return torch.tensor(values, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Population -> batches
# ---------------------------------------------------------------------------


def iterate(
    population: TrainingPopulation,
    sampler: SamplerSpec,
    *,
    steps: int,
    seed: int,
) -> Iterator[XTYBatch]:
    """Yield exactly `steps` batches drawn as `sampler` declares.

    Raises:
        TrainingError: for an `ExternalBatches` declaration, which has no rows
            of its own to draw, or for a quota the population cannot fill.
    """
    if isinstance(sampler, ExternalBatches):
        raise TrainingError(
            "ExternalBatches declares that the caller supplies the batches; "
            "there is nothing here to draw from."
        )
    if steps < 1:
        raise TrainingError(f"a stage draws at least one batch, got steps={steps!r}")
    generator = torch.Generator(device=population.rows.device)
    generator.manual_seed(sampler_seed(seed))
    for _ in range(steps):
        yield _take(population.rows, _draw(population, sampler, generator))


def _draw(
    population: TrainingPopulation,
    sampler: UniformSampler | QuotaSampler,
    generator: torch.Generator,
) -> Tensor:
    """One step's row positions."""
    if isinstance(sampler, UniformSampler):
        return _draw_uniform(
            population.batch_size,
            sampler.batch_size,
            replacement=sampler.replacement,
            generator=generator,
            what="the training population",
        )

    parts: list[Tensor] = []
    for quota in sampler.quotas:
        eligible = population.eligible(quota.rows)
        if quota.stratify is None:
            parts.append(
                eligible[
                    _draw_uniform(
                        eligible.numel(),
                        quota.size,
                        generator=generator,
                        what=f"the {quota.rows} rows",
                    )
                ]
            )
            continue
        treatments = population.rows.t.index_select(0, eligible)
        for level in sorted({int(value) for value in treatments.tolist()}):
            level_rows = eligible[treatments == level]
            parts.append(
                level_rows[
                    _draw_uniform(
                        level_rows.numel(),
                        quota.size,
                        generator=generator,
                        what=f"the {quota.rows} rows with t = {level}",
                    )
                ]
            )
    return torch.cat(parts)


def _draw_uniform(
    available: int,
    size: int,
    *,
    generator: torch.Generator,
    what: str,
    replacement: bool = False,
) -> Tensor:
    """`size` positions out of `available`.

    `randperm(available)[:size]` — the definition `UniformSampler` is pinned
    to, and the reason adopting it moves no arithmetic.
    """
    if replacement:
        if available == 0:
            raise TrainingError(f"{what} is empty, so no batch can be drawn")
        return torch.randint(available, (size,), generator=generator)
    if size > available:
        raise TrainingError(
            f"the sampler asks for {size} rows but {what} holds {available}. A "
            "quota that cannot be filled is an error rather than a short batch, "
            "for the same reason a batch source running dry is: a stage that "
            "silently stepped on fewer rows than its card says is a difference "
            "nothing downstream would show. Repeating a shuffled pass to fill "
            "it — what FixMatch's labelled loader does at 40 labels and B = 64 "
            "— would put one row in a batch twice, which batch.row_id forbids "
            "(DESIGN.md §7.1, ledger key `batch-row-repetition`)."
        )
    return torch.randperm(available, generator=generator)[:size]


def _take(batch: XTYBatch, rows: Tensor) -> XTYBatch:
    """Gather ``rows`` through the batch's structural operation."""
    return batch.index_select(rows)


def eligible_populations(sampler: SamplerSpec) -> tuple[Rows, ...]:
    """The row populations a sampler draws from, for the compiler's §7.0 check."""
    if isinstance(sampler, ExternalBatches):
        return ()
    return sampler.rows
