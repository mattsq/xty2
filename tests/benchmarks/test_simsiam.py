"""Tier 2: complete SimSiam four-arm mechanism study."""

from tests.benchmarks._runner import assert_recorded_benchmark


def test_simsiam_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("simsiam")
