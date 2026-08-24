"""Command-line entry point for card-driven Tier 2 benchmark execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from xty2.evaluation.benchmarks import RECIPES, benchmark_function
from xty2.evaluation.benchmarks.common import default_workers
from xty2.evaluation.reporting import (
    BenchmarkResult,
    assert_result_matches_card,
    load_reproduction_spec,
    update_card,
)


def run_recipe(
    recipe: str,
    *,
    repo_root: Path,
    output_root: Path,
    cache_root: Path,
    commit: str,
    date: str,
    workers: int,
    write_ledger: bool = False,
    check_card: bool = False,
) -> BenchmarkResult:
    """Run one recipe from its card and persist the full replicate evidence."""
    repo_root = Path(repo_root)
    card = repo_root / "docs" / "recipes" / f"{recipe}.md"
    spec = load_reproduction_spec(card, recipe=recipe)
    function = benchmark_function(recipe)
    result = function(spec, commit, date, workers, Path(cache_root))
    result.write_json(Path(output_root) / f"{recipe}.json")
    if write_ledger:
        update_card(card, result)
    if check_card:
        assert_result_matches_card(result, card)
    return result


def main(argv: list[str] | None = None) -> int:
    """Run selected recipes, write JSON, and optionally update/check cards."""
    parser = _parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output).resolve()
    cache_root = Path(args.cache).resolve()
    commit = args.commit or _git_commit(repo_root)
    date = args.date or datetime.now(UTC).date().isoformat()
    recipes = RECIPES if args.recipe == "all" else (args.recipe,)
    for recipe in recipes:
        result = run_recipe(
            recipe,
            repo_root=repo_root,
            output_root=output_root,
            cache_root=cache_root,
            commit=commit,
            date=date,
            workers=args.workers,
            write_ledger=args.write_ledger,
            check_card=args.check_card,
        )
        print(json.dumps(_summary(result), sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=("Run the Tier 2 protocol declared in each recipe card section 6")
    )
    parser.add_argument("--recipe", choices=("all", *RECIPES), default="all")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output", type=Path, default=repo_root / "runs" / "tier2")
    parser.add_argument("--cache", type=Path, default=repo_root / "runs" / "datasets")
    parser.add_argument("--commit")
    parser.add_argument("--date")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument(
        "--write-ledger",
        action="store_true",
        help="update the card status, ledger row and deviation explanation",
    )
    parser.add_argument(
        "--check-card",
        action="store_true",
        help="fail when the fresh result disagrees with the recorded card status",
    )
    return parser


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if not commit:
        raise RuntimeError("git returned an empty commit identifier")
    return commit


def _summary(result: BenchmarkResult) -> dict[str, object]:
    return {
        "recipe": result.recipe,
        "status": result.status,
        "replicates": result.replicates,
        "metrics": {
            metric.name: {
                "mean": metric.mean,
                "stderr": metric.stderr,
                "criterion": metric.criterion,
                "passed": metric.passed,
            }
            for metric in result.metrics
        },
    }


if __name__ == "__main__":  # pragma: no cover - exercised by the nightly command
    raise SystemExit(main())


__all__ = ["main", "run_recipe"]
