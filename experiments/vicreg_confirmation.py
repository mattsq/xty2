"""Run the approved fresh-seed contract with per-seed checkpoints."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from xty2.core import compile
from xty2.evaluation.benchmarks import vicreg as benchmark
from xty2.evaluation.benchmarks.common import continuous_schema
from xty2.evaluation.reporting import load_reproduction_spec
from xty2.evaluation.vicreg_views import OracleSymmetry
from xty2.recipes import vicreg


def replicate(index: int, output: str) -> dict[str, float]:
    row = benchmark._replicate(index)
    path = Path(output) / f"seed-{290000 + 100 * index}.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(f"completed seed {290000 + 100 * index}", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("confirmation requires a clean committed source tree")
    if args.output.exists():
        raise RuntimeError("refusing to overwrite confirmation evidence")
    args.output.mkdir(parents=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    environment = {
        "commit": commit,
        "tree": tree,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "workers": args.workers,
        "seed_bases": [290000 + 100 * i for i in range(10)],
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    recipe = vicreg(
        continuous_schema(6),
        first_transforms=(OracleSymmetry(),),
        second_transforms=(OracleSymmetry(),),
    )
    (args.output / "plan.txt").write_text(compile(recipe).plan.render())
    spec = load_reproduction_spec(Path("docs/recipes/vicreg.md"), recipe="vicreg")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        rows = tuple(
            executor.map(partial(replicate, output=str(args.output)), range(10))
        )
    # Score the exact saved outputs with the canonical benchmark, without
    # repeating training. The hook replaces only replicate execution.
    with patch.object(benchmark, "parallel_replicates", return_value=rows):
        result = benchmark.run(
            spec,
            commit,
            datetime.now(UTC).date().isoformat(),
            args.workers,
            args.output,
        )
    result.write_json(args.output / "vicreg.json")
    for metric in result.metrics:
        if metric.relation != "info":
            print(metric.name, metric.summary(), "pass=", metric.passed, flush=True)
    print(f"status={result.status}", flush=True)


if __name__ == "__main__":
    main()
