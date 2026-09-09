"""Approved fixture-specific VICReg views; these require oracle DGP knowledge."""

import hashlib
from dataclasses import dataclass
from unittest.mock import patch

import torch
from torch import Tensor

from xty2.core import CompiledRun, Schema, TrainingPopulation, ViewSpec, XTYBatch
from xty2.training import ProgramResult, run_program
from xty2.training.executors import BatchSources
from xty2.views import FeatureCorruption

PRESERVATION_TOLERANCE = 2e-5


def targets(x: Tensor) -> Tensor:
    """DGP targets on original-scale rows, never observed labels."""
    propensity = 0.02 + 0.96 * torch.sigmoid(2.5 * x[:, :4].sum(-1))
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1)
    effect = 1 + 0.5 * torch.tanh(x[:, 2])
    return torch.stack((propensity, baseline, baseline + effect), dim=-1)


@dataclass(frozen=True)
class OracleSymmetry:
    """Fixture-specific symmetry. Not a general-purpose tabular augmentation."""

    def validate(self, schema: Schema) -> None:
        if schema.feature_names != tuple(f"x{i}" for i in range(6)):
            raise ValueError("oracle symmetry requires the six-column fixture")
        if not all(spec.mutable for spec in schema.features):
            raise ValueError("oracle symmetry requires mutable fixture columns")

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        self.validate(schema)
        return frozenset(("x0", "x1", "x3", "x4", "x5"))

    def describe(self) -> str:
        return "OracleSymmetry(reflect=(3,5,0,-8),p=0.5; sign_x4,p=0.5; donor_x5)"

    def apply(
        self,
        batch: XTYBatch,
        schema: Schema,
        *,
        generator: torch.Generator,
        population: TrainingPopulation | None = None,
    ) -> XTYBatch:
        self.validate(schema)
        if population is None:
            raise ValueError("oracle symmetry requires training-only statistics")
        location = population.statistics["x_location"]
        scale = population.statistics["x_scale"]
        original = batch.x * scale + location
        transformed = original.clone()
        v = original.new_tensor((3.0, 5.0, 0.0, -8.0))
        reflect = torch.rand(batch.batch_size, generator=generator) < 0.5
        displacement = 2 * (original[:, :4] @ v)[:, None] * v / v.square().sum()
        transformed[:, :4] -= reflect[:, None] * displacement
        flip = torch.rand(batch.batch_size, generator=generator) < 0.5
        transformed[:, 4] = torch.where(flip, -original[:, 4], original[:, 4])
        # Assign only affected columns: x2 stays bit-identical after scaling.
        scaled = batch.x.clone()
        for column in (0, 1, 3, 4):
            scaled[:, column] = (transformed[:, column] - location[column]) / scale[
                column
            ]
        result = FeatureCorruption(rate=1.0, columns=("x5",)).apply(
            batch.replace(x=scaled), schema, generator=generator, population=population
        )
        error = (targets(result.x * scale + location) - targets(original)).abs().max()
        if float(error) > PRESERVATION_TOLERANCE:
            raise RuntimeError(f"oracle changed a DGP target by {float(error)}")
        return result


def fit_with_trace(
    run: CompiledRun, data: BatchSources, seed: int
) -> tuple[ProgramResult, dict[str, str]]:
    # Actual executor view draws, not a replay of what declarations would draw.
    original = ViewSpec.apply
    rows = hashlib.sha256()
    views = hashlib.sha256()
    calls = 0

    def traced(
        view: ViewSpec,
        batch: XTYBatch,
        schema: Schema,
        *,
        rng_key: int,
        draw: int = 0,
        population: TrainingPopulation | None = None,
    ) -> XTYBatch:
        nonlocal calls
        result = original(
            view, batch, schema, rng_key=rng_key, draw=draw, population=population
        )
        rows.update(batch.row_id.numpy().tobytes())
        views.update(result.x.numpy().tobytes())
        calls += 1
        return result

    with patch.object(ViewSpec, "apply", traced):
        result = run_program(run, data, seed=seed)
    return result, {
        "rows": rows.hexdigest(),
        "views": views.hexdigest(),
        "calls": str(calls),
    }
