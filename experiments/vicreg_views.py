"""Paired oracle-view diagnostic; see the committed protocol before using it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import platform
import subprocess
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from xty2.core import (
    CompiledRun,
    Port,
    Recipe,
    Schema,
    TrainingPopulation,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks import vicreg as benchmark
from xty2.evaluation.benchmarks.common import (
    SEPARATED,
    configure_worker,
    continuous_schema,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.reporting import MetricResult
from xty2.evaluation.vicreg_views import (
    PRESERVATION_TOLERANCE,
    OracleSymmetry,
    fit_with_trace,
    targets,
)
from xty2.recipes import vicreg
from xty2.training import STREAM_STRIDE, ProgramResult
from xty2.views import FeatureCorruption

POLICIES = ("marginal", "oracle")
ARMS = ("full", "no_variance", "no_covariance")
PROTOCOL = Path("docs/experiments/2026-09-08-vicreg-views-protocol.md")


def policy_recipe(schema: Schema, policy: str) -> Recipe:
    if policy == "marginal":
        transform = FeatureCorruption(rate=0.6, columns=None)
        return vicreg(
            schema, first_transforms=(transform,), second_transforms=(transform,)
        )
    if policy != "oracle":
        raise ValueError(policy)
    return vicreg(
        schema,
        first_transforms=(OracleSymmetry(),),
        second_transforms=(OracleSymmetry(),),
    )


def held_out_views(
    schema: Schema,
    batches: Sequence[XTYBatch],
    population: TrainingPopulation,
    policy: str,
    base: int,
) -> tuple[XTYBatch, ...]:
    transform = (
        FeatureCorruption(rate=0.6, columns=None)
        if policy == "marginal"
        else OracleSymmetry()
    )
    return tuple(
        transform.apply(
            batch,
            schema,
            population=population,
            generator=torch.Generator().manual_seed(base + 20_000 + 2 * b + branch),
        )
        for b, batch in enumerate(batches)
        for branch in range(2)
    )


def view_diagnostics(
    batches: Sequence[XTYBatch],
    views: Sequence[XTYBatch],
    population: TrainingPopulation,
) -> dict[str, float]:
    changes: list[Tensor] = []
    counts: list[Tensor] = []
    displacements: list[Tensor] = []
    pair_differences: list[Tensor] = []
    for i, view in enumerate(views):
        batch = batches[i // 2]
        clean = benchmark._original_scale(batch.x, population)
        transformed = benchmark._original_scale(view.x, population)
        changes.append((targets(transformed) - targets(clean)).abs())
        counts.append((view.x != batch.x).float().sum(-1))
        displacements.append((view.x - batch.x).square().mean(-1))
        if i % 2:
            pair_differences.append((view.x != views[i - 1].x).any(-1).float())
    values = torch.cat(changes)
    return {
        "propensity_shift": float(values[:, 0].mean()),
        "outcome_mean_shift": float(values[:, 1:].mean()),
        "max_target_error": float(values.max()),
        "changed_coordinates": float(torch.cat(counts).mean()),
        "standardised_feature_MSE": float(torch.cat(displacements).mean()),
        "distinct_view_row_fraction": float(torch.cat(pair_differences).mean()),
    }


def loss_curve(
    run: CompiledRun, schema: Schema, views: Sequence[XTYBatch]
) -> dict[str, float]:
    """Exact held-out objective under fixed embedding rescalings; no refitting."""
    collected: dict[str, list[float]] = {}
    with torch.no_grad():
        for b in range(16):
            z = [
                run.graph.evaluate(
                    views[2 * b + branch],
                    schema=schema,
                    only=("mlp_encoder", "vicreg_expander"),
                )[Port.X_PROJ]
                for branch in range(2)
            ]
            if not all(isinstance(value, Tensor) for value in z):
                raise TypeError("expected tensor embeddings")
            left, right = z
            assert isinstance(left, Tensor) and isinstance(right, Tensor)
            for factor in (0.5, 1.0, 2.0, 4.0):
                a, c = left * factor, right * factor
                invariance = (a - c).square().mean()
                deviations = [
                    (item.var(0, correction=1) + 1e-4).sqrt() for item in (a, c)
                ]
                variance = sum(torch.relu(1 - item).mean() / 2 for item in deviations)
                covariance = a.new_zeros(())
                mask = ~torch.eye(a.shape[1], dtype=torch.bool)
                for item in (a, c):
                    centred = item - item.mean(0)
                    cov = centred.T @ centred / (item.shape[0] - 1)
                    covariance += cov[mask].square().sum() / item.shape[1]
                values = {
                    "invariance": float(invariance),
                    "variance": float(variance),
                    "covariance": float(covariance),
                    "total": float(25 * invariance + 25 * variance + covariance),
                    "spread": float(sum(item.mean() / 2 for item in deviations)),
                }
                for name, value in values.items():
                    collected.setdefault(f"scale_{factor:g}_{name}", []).append(value)
    return {name: math.fsum(values) / len(values) for name, values in collected.items()}


def replicate(index: int, output: str) -> dict[str, float]:
    configure_worker()
    base = 190_000 + 100 * index
    schema = continuous_schema(6)
    train = two_cluster_population(1024, seed=base + 1, row_offset=0, low=SEPARATED)
    test = two_cluster_population(2048, seed=base + 2, row_offset=10_000, low=SEPARATED)
    data = training_dataset(schema, train.batch)
    runs: dict[str, CompiledRun] = {}
    results: dict[str, ProgramResult] = {}
    traces: dict[str, dict[str, str]] = {}
    initial: dict[str, Tensor] | None = None
    metrics: dict[str, float] = {}
    variants = [(policy, arm) for policy in POLICIES for arm in ARMS]
    variants.append(("marginal", "no_pretrain"))
    for policy, arm in variants:
        key = f"{policy}_{arm}" if arm != "no_pretrain" else arm
        torch.manual_seed(base + 6)
        recipe = benchmark._arm(policy_recipe(schema, policy), arm)
        state = {
            name: value.clone() for name, value in recipe.system.state_dict().items()
        }
        if initial is None:
            initial = state
        if any(not torch.equal(value, initial[name]) for name, value in state.items()):
            raise RuntimeError("initial states differ")
        run = compile(recipe)
        result, trace = fit_with_trace(
            run,
            {stage.name: data for stage in recipe.program},
            base + 10_000 + (STREAM_STRIDE if arm == "no_pretrain" else 0),
        )
        runs[key], results[key], traces[key] = run, result, trace
        metrics.update(
            (f"{key}_{name}", value)
            for name, value in benchmark._downstream(run, result, test).items()
        )
        if arm != "no_pretrain":
            if trace["calls"] != "2000":
                raise RuntimeError(f"expected two actual views per step: {trace}")
            for term in result.stage("pretrain").records[-1].terms:
                metrics[f"{key}_terminal_{term.name}"] = term.value
                metrics.update(
                    (f"{key}_terminal_{term.name}_{name}", value)
                    for name, value in term.diagnostics.items()
                )
        print(f"base={base} finished {key}", flush=True)
    assert initial is not None
    for policy in POLICIES:
        policy_runs = {arm: runs[f"{policy}_{arm}"] for arm in ARMS}
        policy_results = {arm: results[f"{policy}_{arm}"] for arm in ARMS}
        policy_runs["no_pretrain"] = runs["no_pretrain"]
        policy_results["no_pretrain"] = results["no_pretrain"]
        benchmark._require_one_stream(policy_runs, policy_results, data, base=base)
        benchmark._require_pretraining_touched_only_what_it_declares(
            policy_runs, policy_results, initial
        )
        if any(traces[f"{policy}_{arm}"] != traces[f"{policy}_full"] for arm in ARMS):
            raise RuntimeError("actual within-policy row/view draws differ")
    if traces["marginal_full"]["rows"] != traces["oracle_full"]["rows"]:
        raise RuntimeError("actual cross-policy row streams differ")
    population = benchmark._fit_population(results["no_pretrain"])
    batches = benchmark._held_out_batches(test, population)
    for eval_policy in POLICIES:
        views = held_out_views(schema, batches, population, eval_policy, base)
        diagnostics = view_diagnostics(batches, views, population)
        if diagnostics["distinct_view_row_fraction"] < 0.95:
            raise RuntimeError("views are insufficiently distinct")
        if (
            eval_policy == "oracle"
            and diagnostics["max_target_error"] > PRESERVATION_TOLERANCE
        ):
            raise RuntimeError("held-out target information was not preserved")
        metrics.update((f"{eval_policy}_views_{k}", v) for k, v in diagnostics.items())
        for policy in POLICIES:
            for arm in ARMS:
                key = f"{policy}_{arm}"
                embeddings = benchmark._embeddings(
                    runs[key], results[key], schema, views, arm=arm
                )
                metrics.update(
                    (f"{key}_eval_{eval_policy}_{name}", value)
                    for name, value in embeddings.items()
                )
                if arm == "full":
                    metrics.update(
                        (f"{key}_eval_{eval_policy}_{name}", value)
                        for name, value in loss_curve(runs[key], schema, views).items()
                    )
    for policy in POLICIES:
        stem = f"{policy}_full_eval_{policy}_"
        metrics[f"{policy}_spread"] = metrics[stem + "embedding_spread"]
        metrics[f"{policy}_variance_gap"] = (
            metrics[stem + "embedding_spread"]
            - metrics[f"{policy}_no_variance_eval_{policy}_embedding_spread"]
        )
        metrics[f"{policy}_covariance_gap"] = (
            metrics[f"{policy}_no_covariance_eval_{policy}_embedding_redundancy"]
            - metrics[stem + "embedding_redundancy"]
        )
        metrics[f"{policy}_outcome_cost"] = (
            metrics[f"{policy}_full_outcome_NLL"] - metrics["no_pretrain_outcome_NLL"]
        )
    for eval_policy in POLICIES:
        metrics[f"policy_spread_gap_eval_{eval_policy}"] = (
            metrics[f"oracle_full_eval_{eval_policy}_embedding_spread"]
            - metrics[f"marginal_full_eval_{eval_policy}_embedding_spread"]
        )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("nonfinite metric")
    payload = {"base": base, "metrics": metrics, "actual_streams": traces}
    Path(output, f"seed-{base}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"base={base} saved; spread={metrics['oracle_spread']:.5f}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("seed-*.json")):
        raise RuntimeError("use an empty output directory; no silent partial reruns")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("experiment must run from a clean committed tree")
    metadata = {
        "source_commit": head,
        "source_tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True
        ).strip(),
        "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "workers": args.workers,
        "bases": [190_000 + 100 * i for i in range(10)],
    }
    Path(args.output, "environment.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    for policy in POLICIES:
        plan = compile(policy_recipe(continuous_schema(6), policy)).plan.render()
        Path(args.output, f"plan-{policy}.txt").write_text(plan + "\n")
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        rows = list(
            executor.map(partial(replicate, output=str(args.output)), range(10))
        )
    required = []
    for policy in POLICIES:
        for name, target in (
            ("spread", 0.5),
            ("variance_gap", 0.1),
            ("covariance_gap", 0.01),
        ):
            key = f"{policy}_{name}"
            required.append(
                MetricResult.lower_bound(key, (row[key] for row in rows), target)
            )
        key = f"{policy}_outcome_cost"
        required.append(MetricResult.upper_bound(key, (row[key] for row in rows), 0.05))
    summary = {
        **metadata,
        "required": [metric.as_json() for metric in required],
        "diagnostics": [
            MetricResult.information(name, (row[name] for row in rows)).as_json()
            for name in sorted(rows[0])
        ],
    }
    Path(args.output, "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for metric in required:
        print(
            f"{metric.name}: {metric.mean:.6f} +/- {metric.stderr:.6f}; "
            f"pass={metric.passed}"
        )


if __name__ == "__main__":
    main()
