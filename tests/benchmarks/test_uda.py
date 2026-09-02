"""Tier 2 — UDA's four paired mechanism arms."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_uda_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("uda")
