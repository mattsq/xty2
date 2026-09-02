"""Tier 2 — DoubleMatch's paired `w_s = 0` mechanism target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_doublematch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("doublematch")
