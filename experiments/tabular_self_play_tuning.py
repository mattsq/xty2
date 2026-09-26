"""Validation-only tuning for the tabular self-play confirmation.

Stage 1 selects one learner budget from ``BUDGETS`` with the uniform policy,
pooled over fixtures. Stage 2 selects, per fixture, the fixed task policy in
``POLICIES`` at that budget. Both stages use validation BCE, tuning seeds
disjoint from ``CONFIRMATION_SEEDS``, and never evaluate the test split.
Adaptive-controller settings are not tuned. Run
``python -m experiments.tabular_self_play_tuning --output DIR``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from experiments.tabular_self_play import (
    CONFIRMATION_SEEDS,
    FIXTURES,
    POLICIES,
    Config,
    provenance,
    run_jobs,
)

TUNING_SEEDS = tuple(320_000 + 100 * i for i in range(5))
BUDGETS = (64, 256, 1024)
assert not set(TUNING_SEEDS) & set(CONFIRMATION_SEEDS)


def mean_bce(rows: list[dict[str, object]], **match: object) -> float:
    values = [
        cast(float, r["bce"])
        for r in rows
        if all(r[key] == value for key, value in match.items())
    ]
    if not values:
        raise ValueError(match)
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("tuning requires a clean committed source tree")
    if args.output.exists():
        parser.error("refusing to overwrite tuning evidence")
    base = Config(endpoint="validation", fixed_probes=False)
    budget_rows: list[dict[str, object]] = []
    for steps in BUDGETS:
        cfg = Config(**(asdict(base) | {"steps": steps}))
        rows = run_jobs(
            [
                (kind, seed, "uniform", cfg, None)
                for kind in FIXTURES
                for seed in TUNING_SEEDS
            ],
            args.workers,
        )
        budget_rows += [r | {"steps": steps} for r in rows]
    by_budget = {steps: mean_bce(budget_rows, steps=steps) for steps in BUDGETS}
    steps = min(BUDGETS, key=lambda s: (by_budget[s], s))
    cfg = Config(**(asdict(base) | {"steps": steps}))
    policy_rows = run_jobs(
        [
            (kind, seed, "tuned", cfg, policy)
            for kind in FIXTURES
            for seed in TUNING_SEEDS
            for policy in POLICIES
        ],
        args.workers,
    )
    table = {
        kind: {
            policy: mean_bce(policy_rows, fixture=kind, policy=policy)
            for policy in POLICIES
        }
        for kind in FIXTURES
    }
    # Ties break toward the earlier, simpler entry in POLICIES.
    selected = {
        kind: min(POLICIES, key=lambda p: (table[kind][p], POLICIES.index(p)))
        for kind in FIXTURES
    }
    none = run_jobs(
        [(kind, seed, "none", cfg, None) for kind in FIXTURES for seed in TUNING_SEEDS],
        args.workers,
    )
    args.output.mkdir(parents=True)
    for rows in (budget_rows, policy_rows, none):
        for row in rows:
            row.pop("trace")
    (args.output / "environment.json").write_text(
        json.dumps(provenance(args.workers) | {"seeds": list(TUNING_SEEDS)}, indent=2)
        + "\n"
    )
    (args.output / "selection.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg) | {"fixed_probes": True},
                "budget_bce": {str(k): v for k, v in by_budget.items()},
                "policy_bce": table,
                "none_bce": {kind: mean_bce(none, fixture=kind) for kind in FIXTURES},
                "selected": selected,
            },
            indent=2,
        )
        + "\n"
    )
    (args.output / "rows.json").write_text(
        json.dumps({"budget": budget_rows, "policy": policy_rows, "none": none}) + "\n"
    )
    print(json.dumps({"steps": steps, "budget_bce": by_budget, "selected": selected}))


if __name__ == "__main__":
    main()
