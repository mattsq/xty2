"""Tier 2 — Mean Teacher mechanism target recorded in its card."""

from tests.benchmarks._runner import assert_recorded_benchmark


def test_mean_teacher_matches_its_recorded_tier2_status() -> None:
    assert_recorded_benchmark("mean_teacher")
