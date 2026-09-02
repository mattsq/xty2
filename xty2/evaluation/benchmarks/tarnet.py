"""TARNet on the pinned IHDP/NPCI archive named by its card section 6."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.request
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch

from xty2.core import CompiledRun, Dataset, GaussianOutcome, Port, XTYBatch, compile
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
from xty2.training import MinimumValidationSelection, run_stage

_FILES = {
    "ihdp_npci_1-1000.train.npz": (
        "https://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip",
        "713e492c5f571d4260784f0ec6f3892f53a27b61310e3cef2e15733f941dd729",
        "b7dbb5e26324b3b23c90ac177e1f1c411ab8562b3fc9b78d9a4a308819f54cce",
    ),
    "ihdp_npci_1-1000.test.npz": (
        "https://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip",
        "99cf9ee79e0677c2b32ceeb2d8dd04e7cb1bcebb7dab18db2ef9db798961e8a5",
        "7dc2a2e34059a9dc9e596879770413826190e9d33613549de16ab6436ec148d2",
    ),
}
_STEPS = 3_000
_VAL_FRACTION = 0.30
_CHECKPOINT_INTERVAL = 200
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
    """Run the full 1,000-realisation paper-faithful IHDP reproduction."""
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
            "seeds": "1000",
            "checkpoint_selection": (
                "minimum validation objective every 200 optimiser steps"
            ),
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 1_000:
        raise ValueError(
            f"TARNet card reviewed 1,000 seeds, got {spec.seed_count}; amend the "
            "card before changing the benchmark"
        )
    data = _ensure_data(Path(cache_root) / "ihdp")
    replicate = partial(_replicate, train_path=str(data["train"]))
    rows = parallel_replicates(replicate, spec.seed_count, workers=workers)
    pehe = column(rows, "sqrt_pehe")
    ate_error = column(rows, "absolute_ate_error")
    selected_steps = column(rows, "selected_step")
    validation_objectives = column(rows, "validation_objective")
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
            MetricResult.information(
                "selected_checkpoint_step", selected_steps, unit="optimiser steps"
            ),
            MetricResult.information(
                "selected_validation_objective", validation_objectives
            ),
        ),
        interpretation=(
            "This evaluates the paper-faithful outcome-only TARNet objective "
            "with validation-objective checkpoint selection."
        ),
        protocol_deviation=None,
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
    validation = population.index_select(permutation[rows - validation_rows :])

    def validation_objective(run: CompiledRun) -> float:
        # The selector passes the same compiled run. Keeping the closure typed
        # locally avoids making benchmark-specific validation rows part of the
        # recipe or loader contracts.
        assert run is compiled
        state = compiled.state("joint_fit", validation)
        distribution = state.default[Port.Y_GIVEN_XT]
        if not isinstance(distribution, GaussianOutcome):
            raise TypeError("TARNet validation expected its Gaussian outcome head")
        prediction = distribution.mean(validation.t)
        squared_error = (prediction - validation.y).square()
        if squared_error.ndim > 1:
            squared_error = squared_error.sum(dim=tuple(range(1, squared_error.ndim)))
        if validation.weight is None:
            raise TypeError("TARNet validation requires treatment-group weights")
        risk = (validation.weight * squared_error).mean()
        regularisation = risk.new_zeros(())
        for name, parameter in compiled.graph["tarnet_head"].named_parameters():
            if name.endswith("weight") and parameter.ndim >= 2:
                regularisation = regularisation + 0.5e-4 * parameter.square().sum()
        return float(risk + regularisation)

    selection = MinimumValidationSelection(
        every=_CHECKPOINT_INTERVAL,
        score=validation_objective,
    )
    result = run_stage(
        compiled,
        "joint_fit",
        data,
        seed=seed + 3,
        selection=selection,
    )
    if result.selection is None:
        raise RuntimeError("TARNet benchmark did not select a validation checkpoint")
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
        "selected_step": float(result.selection.step),
        "validation_objective": result.selection.score,
    }


def _ensure_data(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, Path] = {}
    for name, (url, archive_expected, payload_expected) in _FILES.items():
        destination = root / name
        if destination.exists() and _sha256(destination) == payload_expected:
            resolved["train" if ".train." in name else "test"] = destination
            continue
        archive = destination.with_suffix(destination.suffix + ".zip.download")
        payload = destination.with_suffix(destination.suffix + ".download")
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                archive.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            archive_actual = _sha256(archive)
            if archive_actual != archive_expected:
                raise ValueError(
                    f"downloaded {url} with sha256 {archive_actual}, expected "
                    f"{archive_expected}"
                )
            with zipfile.ZipFile(archive) as compressed:
                if compressed.namelist() != [name]:
                    raise ValueError(
                        f"downloaded {url} contains {compressed.namelist()!r}, "
                        f"expected exactly [{name!r}]"
                    )
                with compressed.open(name) as source, payload.open("wb") as output:
                    shutil.copyfileobj(source, output)
            payload_actual = _sha256(payload)
            if payload_actual != payload_expected:
                raise ValueError(
                    f"extracted {name} with sha256 {payload_actual}, expected "
                    f"{payload_expected}"
                )
            os.replace(payload, destination)
        finally:
            archive.unlink(missing_ok=True)
            payload.unlink(missing_ok=True)
        resolved["train" if ".train." in name else "test"] = destination
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["run"]
