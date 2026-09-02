"""The card-defined Tier 2 benchmark cases (`docs/PLAN.md` P12).

Eight, since `scarf`, `fixmatch` and `doublematch` acquired modules. The first
two cards' §6.3 recorded the absence as the reason their status could not pass
`smoke-passing`: "the Tier 2 runner has one module per recipe and this recipe
has none". `doublematch.md` §6.2 recorded the same thing in its own words — a
ten-seed ledger left unrun, with a card kept at `draft` because of it. A
declared protocol nothing can execute is a target in the same sense a tolerance
nobody measures against is one.
"""

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
)

__all__ = ["RECIPES", "BenchmarkFunction", "benchmark_function"]
