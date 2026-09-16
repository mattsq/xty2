"""Tier 2: complete BYOL four-arm mechanism study."""

from tests.benchmarks._runner import assert_recorded_benchmark


def test_byol_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("byol")
