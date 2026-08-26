"""Thin pytest adapter around the reusable Tier 2 runner."""

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from xty2.evaluation.benchmarks.common import default_workers
from xty2.evaluation.runner import run_recipe

ROOT = Path(__file__).parents[2]


def assert_recorded_benchmark(recipe: str) -> None:
    """Run one card protocol and require its recorded status to remain true."""
    commit = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    write_ledger = os.environ.get("XTY2_WRITE_TIER2_LEDGERS") == "1"
    run_recipe(
        recipe,
        repo_root=ROOT,
        output_root=ROOT / "runs" / "tier2",
        cache_root=ROOT / "runs" / "datasets",
        commit=commit,
        date=datetime.now(UTC).date().isoformat(),
        workers=default_workers(),
        write_ledger=write_ledger,
        check_card=True,
    )


__all__ = ["assert_recorded_benchmark"]
