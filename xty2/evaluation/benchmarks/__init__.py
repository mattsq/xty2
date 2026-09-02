"""The card-defined Tier 2 benchmark cases (`docs/PLAN.md` P12)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from xty2.evaluation.reporting import BenchmarkResult, ReproductionSpec

BenchmarkFunction = Callable[[ReproductionSpec, str, str, int, Path], BenchmarkResult]


def benchmark_function(recipe: str) -> BenchmarkFunction:
    """Resolve one benchmark only after the runner has parsed its card."""
    if recipe == "tarnet":
        from xty2.evaluation.benchmarks.tarnet import run

        return run
    if recipe == "cnflow":
        from xty2.evaluation.benchmarks.cnflow import run

        return run
    if recipe == "mean_teacher":
        from xty2.evaluation.benchmarks.mean_teacher import run

        return run
    if recipe == "cycle_dual":
        from xty2.evaluation.benchmarks.cycle_dual import run

        return run
    if recipe == "ssdml":
        from xty2.evaluation.benchmarks.ssdml import run

        return run
    if recipe == "scarf":
        from xty2.evaluation.benchmarks.scarf import run

        return run
    if recipe == "fixmatch":
        from xty2.evaluation.benchmarks.fixmatch import run

        return run
    if recipe == "doublematch":
        from xty2.evaluation.benchmarks.doublematch import run

        return run
    if recipe == "flexmatch":
        from xty2.evaluation.benchmarks.flexmatch import run

        return run
    if recipe == "freematch":
        from xty2.evaluation.benchmarks.freematch import run

        return run
    raise KeyError(
        f"unknown Tier 2 recipe {recipe!r}; expected one of {list(RECIPES)!r}"
    )


RECIPES = (
    "tarnet",
    "cnflow",
    "mean_teacher",
    "cycle_dual",
    "ssdml",
    "scarf",
    "fixmatch",
    "doublematch",
    "flexmatch",
    "freematch",
)

__all__ = ["RECIPES", "BenchmarkFunction", "benchmark_function"]
