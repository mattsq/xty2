"""Tier 2 — CNFlow conditional-density target recorded in its card."""

from tests.benchmarks._runner import assert_recorded_benchmark


def test_cnflow_matches_its_recorded_tier2_status() -> None:
    assert_recorded_benchmark("cnflow")
