"""Test-suite wiring shared by all three tiers (`docs/FIDELITY.md` §3).

Two things happen here and nowhere else:

* each test directory maps to its tier marker, so `-m tier0` and a path select
  the same set and neither can drift from the other;
* a tier with no tests yet exits 0 rather than pytest's exit code 5. Tier 2 is
  empty until P12, and its nightly job has to be able to go green in the
  meantime. The mapping below is the guard: a run that collects nothing from a
  directory that *does* contain tests still fails, because those tests are
  collected.
"""

from pathlib import Path

import pytest

TIER_BY_DIRECTORY = {
    "invariants": "tier0",
    "smoke": "tier1",
    "benchmarks": "tier2",
}

TESTS_ROOT = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the tier marker implied by each test's directory."""
    for item in items:
        relative = Path(str(item.path)).relative_to(TESTS_ROOT)
        tier = TIER_BY_DIRECTORY.get(relative.parts[0])
        if tier is not None:
            item.add_marker(tier)


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Treat "no tests collected" as success — see the module docstring."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED:
        session.exitstatus = pytest.ExitCode.OK
