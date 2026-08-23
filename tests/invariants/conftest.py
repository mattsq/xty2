"""Shared fixtures for the Tier 0 invariants.

`BATCH_SIZE != NUM_TREATMENTS` everywhere on purpose: a test with `B == K`
passes under accidental broadcasting and proves nothing (`FIDELITY.md` §3).
"""

from typing import Any

import pytest
import torch
from torch import Tensor
from xty2.core import FeatureSpec, OutcomeSpec, Schema, XTYBatch

BATCH_SIZE = 7
NUM_TREATMENTS = 3
NUM_FEATURES = 4


@pytest.fixture(autouse=True)
def _deterministic_rng() -> None:
    torch.manual_seed(20260822)


def make_schema(**overrides: Any) -> Schema:
    """A four-column schema exercising bounds, mutability and derivation."""
    defaults: dict[str, Any] = {
        "features": (
            FeatureSpec(
                "mass", "continuous", bounds=(0.0, 100.0), perturbation_scale=0.5
            ),
            FeatureSpec("speed", "continuous", perturbation_scale=1.0),
            FeatureSpec("momentum", "continuous", derived_from=("mass", "speed")),
            FeatureSpec("site", "categorical", mutable=False),
        ),
        "treatment_cardinality": NUM_TREATMENTS,
        "outcome": OutcomeSpec(),
    }
    return Schema(**(defaults | overrides))


def make_batch(**overrides: Any) -> XTYBatch:
    """A valid batch: rows 0-3 have an observed treatment, rows 4-6 do not."""
    t_observed = torch.zeros(BATCH_SIZE, dtype=torch.bool)
    t_observed[:4] = True
    defaults: dict[str, Tensor | None] = {
        "x": torch.randn(BATCH_SIZE, NUM_FEATURES),
        "t": torch.arange(BATCH_SIZE, dtype=torch.long) % NUM_TREATMENTS,
        "y": torch.randn(BATCH_SIZE),
        "t_observed": t_observed,
        "y_observed": torch.ones(BATCH_SIZE, dtype=torch.bool),
        "row_id": torch.arange(100, 100 + BATCH_SIZE, dtype=torch.long),
    }
    return XTYBatch(**(defaults | overrides))


@pytest.fixture
def schema() -> Schema:
    return make_schema()


@pytest.fixture
def batch() -> XTYBatch:
    return make_batch()
