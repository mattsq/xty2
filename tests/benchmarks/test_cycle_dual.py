"""Tier 2 — cycle-dual posterior-imputation target recorded in its card."""

from tests.benchmarks._runner import assert_recorded_benchmark


def test_cycle_dual_matches_its_recorded_tier2_status() -> None:
    assert_recorded_benchmark("cycle_dual")
