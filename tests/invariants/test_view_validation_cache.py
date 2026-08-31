"""Tier 0 — compiled views reuse schema-dependent validation metadata."""

from __future__ import annotations

import torch

from xty2.core import FeatureSpec, Schema, TrainingPopulation, XTYBatch
from xty2.views import ViewSpec


class _CountingTransform:
    def __init__(self) -> None:
        self.validate_calls = 0
        self.affected_calls = 0

    def validate(self, schema: Schema) -> None:
        self.validate_calls += 1
        schema.feature("feature")

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        self.affected_calls += 1
        schema.feature("feature")
        return frozenset({"feature"})

    def apply(
        self,
        batch: XTYBatch,
        schema: Schema,
        *,
        generator: torch.Generator,
        population: TrainingPopulation | None = None,
    ) -> XTYBatch:
        del schema, generator, population
        return batch.clone()

    def describe(self) -> str:
        return "counting transform"


def _schema() -> Schema:
    return Schema(
        features=(
            FeatureSpec(
                "feature",
                "continuous",
                bounds=(0.0, 2.0),
                perturbation_scale=1.0,
            ),
        ),
        treatment_cardinality=2,
    )


def _batch() -> XTYBatch:
    return XTYBatch(
        x=torch.ones(4, 1),
        t=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        y=torch.zeros(4),
        t_observed=torch.ones(4, dtype=torch.bool),
        y_observed=torch.ones(4, dtype=torch.bool),
        row_id=torch.arange(4, dtype=torch.long),
    )


def test_validation_metadata_is_reused_after_compile_time_validation() -> None:
    schema = _schema()
    transform = _CountingTransform()
    view = ViewSpec(
        name="counted_x",
        transforms=(transform,),
        preserves=frozenset({"t", "y", "t_observed", "y_observed", "row_id"}),
    )

    view.validate(schema)
    assert transform.validate_calls == 1
    assert transform.affected_calls == 1

    view.apply(_batch(), schema, rng_key=1)
    view.apply(_batch(), schema, rng_key=2)

    assert transform.validate_calls == 1
    assert transform.affected_calls == 1
