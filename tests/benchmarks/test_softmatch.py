"""Tier 2 — SoftMatch's paired continuous-weight target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_softmatch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("softmatch")
