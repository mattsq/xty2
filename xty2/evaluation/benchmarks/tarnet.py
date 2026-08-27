"""TARNet on the pinned IHDP/NPCI archive named by its card section 6."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from functools import partial
from pathlib import Path

import numpy as np
import torch

from xty2.core import Dataset, GaussianOutcome, Port, XTYBatch, compile
from xty2.evaluation.benchmarks.common import (
    column,
    configure_worker,
    continuous_schema,
    parallel_replicates,
)
from xty2.evaluation.causal import (
    absolute_ate_error,
    average_treatment_effect,
    candidate_treatment_means,
    sqrt_pehe,
    treatment_contrast,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.recipes import tarnet
from xty2.training import run_stage

_COMMIT = "0377b0c8c822845d335540d4be6003024a65d3c8"
_FILES = {
    "ihdp_npci_1-100.train.npz": (
        f"https://raw.githubusercontent.com/clinicalml/cfrnet/{_COMMIT}/"
        "data/ihdp_npci_1-100.train.npz",
        "750697c71b4f8d7a3aafff771b56a4ac4cd83ec649bf69afb04f8a5aee41a240",
    ),
    "ihdp_npci_1-100.test.npz": (
        f"https://raw.githubusercontent.com/clinicalml/cfrnet/{_COMMIT}/"
        "data/ihdp_npci_1-100.test.npz",
        "a70a8acbcc4e8deb677cc9bf9e9dabeb17caaa37cdbb1d7ba06be7ffb929c41c",
    ),
}
_STEPS = 3_000
_VAL_FRACTION = 0.30
# The batch size used to live here, beside the step count, because nothing in
# the recipe could hold it. It is `tarnet.BATCH_SIZE` now, bound to
# `optimisation.batch_size` and printed in the plan (card §5 deviation 5).
_MODEL_SEED = 130_000


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten-replicate, explicitly partial IHDP diagnostic."""
    spec.bind(
        {
            "dataset": "IHDP",
            "variant": (
                "1000 realisations, Hill (2011) / NPCI setting A; binary "
                "treatment; fully observed t"
            ),
            "split": (
                "63/27/10 train/validation/test; within-sample metric over "
                "train plus validation"
            ),
            "metric": "sqrt_PEHE_in_sample",
            "published": "0.88",
            "tolerance": "0.10",
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(
            f"TARNet card reviewed 10 seeds, got {spec.seed_count}; amend the "
            "card before changing the benchmark"
        )
    data = _ensure_data(Path(cache_root) / "ihdp")
    replicate = partial(_replicate, train_path=str(data["train"]))
    rows = parallel_replicates(replicate, spec.seed_count, workers=workers)
    pehe = column(rows, "sqrt_pehe")
    ate_error = column(rows, "absolute_ate_error")
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.interval(
                "sqrt_PEHE_in_sample",
                pehe,
                0.78,
                0.98,
                unit="outcome units",
            ),
            MetricResult.information(
                "absolute_ATE_error",
                ate_error,
                unit="outcome units",
            ),
        ),
        interpretation=(
            "This evaluates the implemented TARNet extension against the IHDP "
            "within-sample estimand, with the paper's target retained unchanged."
        ),
        protocol_deviation=(
            "The pinned reference repository ships 100 of the declared 1,000 "
            "IHDP realisations. The reviewed card also requests only ten seeds, "
            "so P12 runs realisations 1-10 with one deterministic fit each. That "
            "cannot establish the published 1,000-realisation centre and is "
            "recorded as deviating even if its ten-run mean lies inside the "
            "numeric tolerance."
        ),
    )


def _replicate(index: int, *, train_path: str) -> dict[str, float]:
    configure_worker()
    seed = _MODEL_SEED + index
    with np.load(train_path) as archive:
        if archive["x"].shape[2] <= index:
            raise ValueError(
                f"IHDP archive has only {archive['x'].shape[2]} realisations; "
                f"cannot run index {index}"
            )
        x = torch.from_numpy(np.asarray(archive["x"][:, :, index], dtype=np.float32))
        treatment = torch.from_numpy(np.asarray(archive["t"][:, index], dtype=np.int64))
        outcome = torch.from_numpy(
            np.asarray(archive["yf"][:, index], dtype=np.float32)
        )
        true_effect = torch.from_numpy(
            np.asarray(
                archive["mu1"][:, index] - archive["mu0"][:, index],
                dtype=np.float32,
            )
        )
    rows = int(x.shape[0])
    split_generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(rows, generator=split_generator)
    validation_rows = int(_VAL_FRACTION * rows)
    fit_rows = permutation[: rows - validation_rows]
    fit_treatment = treatment.index_select(0, fit_rows)
    frequencies = torch.bincount(fit_treatment, minlength=2).float()
    frequencies /= frequencies.sum()
    weights = 1.0 / (2.0 * frequencies.index_select(0, treatment))
    population = XTYBatch(
        x=x,
        t=treatment,
        y=outcome,
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
        weight=weights,
    )
    # The recipe declares the split, the standardisation and the batch size
    # now, so the runner supplies rows and the assignment its SplitSpec names
    # and stops owning any of the three. The `fit` assignment is what
    # `TrainingPopulation.fitted_on_row_ids` is checked against.
    data = Dataset(
        schema=continuous_schema(x.shape[1]),
        rows=population,
        assignments={
            "fit": fit_rows,
            "validation": permutation[rows - validation_rows :],
        },
    )
    torch.manual_seed(seed + 2)
    compiled = compile(tarnet(continuous_schema(x.shape[1])))
    run_stage(compiled, "joint_fit", data, seed=seed + 3)
    with torch.no_grad():
        outcome_distribution = compiled.state("joint_fit", population).default[
            Port.Y_GIVEN_XT
        ]
        if not isinstance(outcome_distribution, GaussianOutcome):
            raise TypeError("TARNet benchmark expected its Gaussian outcome head")
        means = candidate_treatment_means(
            outcome_distribution,
            batch_size=rows,
            num_treatments=2,
            device=population.t.device,
        )
        estimated_effect = treatment_contrast(means)
        pehe = float(sqrt_pehe(estimated_effect, true_effect))
        estimated_ate = average_treatment_effect(estimated_effect)
        true_ate = float(true_effect.mean())
    return {
        "sqrt_pehe": pehe,
        "absolute_ate_error": absolute_ate_error(estimated_ate, true_ate),
    }


def _ensure_data(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for name, (url, expected) in _FILES.items():
        destination = root / name
        if destination.exists() and _sha256(destination) == expected:
            resolved["train" if ".train." in name else "test"] = destination
            continue
        temporary = destination.with_suffix(destination.suffix + ".download")
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary.write_bytes(response.read())
        actual = _sha256(temporary)
        if actual != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(
                f"downloaded {url} with sha256 {actual}, expected {expected}"
            )
        os.replace(temporary, destination)
        resolved["train" if ".train." in name else "test"] = destination
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run"]
