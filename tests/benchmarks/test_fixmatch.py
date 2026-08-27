"""Tier 2 — FixMatch's paired `lambda_u = 0` mechanism target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_fixmatch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("fixmatch")
