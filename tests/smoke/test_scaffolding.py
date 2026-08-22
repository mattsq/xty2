"""Tier 1 — placeholder until the first recipe lands (P5).

Tier 1 is a synthetic-DGP wiring test per `docs/FIDELITY.md` §3; there is no
recipe to fit yet. This asserts only that the tier is selectable, so the PR job
that claims to run Tier 1 is running something.
"""

import pytest


def test_tier_marker_is_applied_by_directory(request: pytest.FixtureRequest) -> None:
    assert request.node.get_closest_marker("tier1") is not None
