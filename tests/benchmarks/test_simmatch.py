"""Tier 2 — SimMatch's paired semantic-instance propagation target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_simmatch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("simmatch")
