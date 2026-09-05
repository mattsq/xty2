"""Tier 2 — CoMatch's paired FixMatch mechanism target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_comatch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("comatch")
