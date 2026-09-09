"""Tier 2 — Barlow Twins' four-arm paired mechanism study."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_barlow_twins_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("barlow_twins")
