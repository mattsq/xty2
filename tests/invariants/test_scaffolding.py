"""Tier 0 — the repo scaffolding itself (P0).

These are not framework invariants; they are the checks that the harness the
framework will be built in is actually wired up. They exist so that the CI jobs
added in P0 run something real before P1 lands any code.
"""

from pathlib import Path

import pytest
import xty2

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_package_imports_and_reports_a_version() -> None:
    assert xty2.__version__
    assert xty2.__version__ != "0.0.0+unknown", "package is not installed"


def test_package_is_typed() -> None:
    package_root = Path(xty2.__file__).parent
    assert (package_root / "py.typed").is_file()


@pytest.mark.parametrize("tier_directory", ["invariants", "smoke", "benchmarks"])
def test_every_tier_directory_exists(tier_directory: str) -> None:
    assert (REPO_ROOT / "tests" / tier_directory).is_dir()


def test_tier_marker_is_applied_by_directory(request: pytest.FixtureRequest) -> None:
    assert request.node.get_closest_marker("tier0") is not None
