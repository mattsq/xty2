"""Tier 2 — TARNet IHDP result recorded in its card."""

from tests.benchmarks._runner import assert_recorded_benchmark


def test_tarnet_matches_its_recorded_tier2_status() -> None:
    assert_recorded_benchmark("tarnet")
