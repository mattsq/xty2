"""Tier 2 — SSDML staged-imputation target recorded in its card."""

from tests.benchmarks._runner import assert_recorded_benchmark


def test_ssdml_matches_its_recorded_tier2_status() -> None:
    assert_recorded_benchmark("ssdml")
