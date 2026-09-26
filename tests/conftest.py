"""Test-suite wiring shared by all three tiers (`docs/FIDELITY.md` §3).

Two things happen here and nowhere else:

* small CPU tests use one PyTorch thread per process;
* each test directory maps to its tier marker, so marker selection and an
  explicit directory path agree (the default paths omit Tier 2).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

TIER_BY_DIRECTORY = {
    "invariants": "tier0",
    "smoke": "tier1",
    "benchmarks": "tier2",
}

TESTS_ROOT = Path(__file__).parent


@pytest.fixture(scope="session", autouse=True)
def _small_cpu_test_threads() -> Iterator[None]:
    """Avoid parallel kernel startup for the suite's small CPU tensors.

    Most smoke modules already did this separately. Apply it to invariants and
    new recipes too, while restoring the caller's setting after pytest exits.
    Tier 2 subprocesses configure their own thread pools independently.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply the tier marker implied by each test's directory."""
    for item in items:
        relative = Path(str(item.path)).relative_to(TESTS_ROOT)
        tier = TIER_BY_DIRECTORY.get(relative.parts[0])
        if tier is not None:
            item.add_marker(tier)
