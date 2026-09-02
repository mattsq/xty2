"""Tier 2 — FlexMatch's paired curriculum-versus-constant target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_flexmatch_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("flexmatch")
