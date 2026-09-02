"""Tier 2 — FreeMatch's paired self-adaptive-versus-constant target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_freematch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("freematch")
