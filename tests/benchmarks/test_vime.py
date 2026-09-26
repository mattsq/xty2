"""Tier 2 — VIME-self's paired pretext and frozen-transfer mechanism target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_vime_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("vime")
