"""Tier 2 — the variational treatment bound against exact marginalisation."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_variational_treatment_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("variational_treatment")
