"""Tier 2 — SCARF's paired pretrained/unpretrained mechanism target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_scarf_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("scarf")
