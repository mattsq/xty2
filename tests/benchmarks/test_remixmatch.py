"""Tier 2 — ReMixMatch's paired crowd and alignment mechanism target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_remixmatch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("remixmatch")
