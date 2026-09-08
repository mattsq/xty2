"""Tier 2 — VICReg's four-arm paired mechanism study."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_vicreg_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("vicreg")
