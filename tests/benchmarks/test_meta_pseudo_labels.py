"""Tier 2 — Meta Pseudo Labels' paired feedback mechanism target."""

from __future__ import annotations

from tests.benchmarks._runner import assert_recorded_benchmark


def test_meta_pseudo_labels_matches_its_recorded_reproduction() -> None:
    assert_recorded_benchmark("meta_pseudo_labels")
