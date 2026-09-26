"""Bounded real-X self-play experiment; no xty2 recipe or executor contract.

The pilot, validation tuning stage and confirmation protocol are described in
docs/proposals/tabular-self-play-ssl.md §8 and the dated documents under
docs/experiments/. ``python -m experiments.tabular_self_play_tuning`` selects
fixed task policies on the validation split. ``python -m
experiments.tabular_self_play --output DIR`` runs the frozen confirmation on
the test split from a clean committed tree.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import platform
import subprocess
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Task:
    target: int
    context: str


CONTEXTS = ("full", "sparse", "jitter", "replace")
TASKS = tuple(Task(i, context) for i in range(6) for context in CONTEXTS)
FIXTURES = ("dependent", "interaction", "independent")
ADAPTIVE_ARMS = ("alignment", "current_loss", "shuffled")
FIXED_ARMS = ("uniform", "fixed_dependent", "tuned")
ARMS = ("none", *FIXED_ARMS, *ADAPTIVE_ARMS)
ENDPOINTS = ("validation", "test")
CONFIRMATION_SEEDS = tuple(330_000 + 100 * i for i in range(10))
SELECTION = Path("docs/experiments/results/tabular-self-play-tuning/selection.json")


def policy_weights(name: str) -> Tensor:
    """Normalised fixed task distributions; the tuning grid is POLICIES."""
    if name == "uniform":
        weights = torch.ones(len(TASKS))
    elif name == "fixed_dependent":
        # The pilot's informative-feature heuristic, kept for continuity.
        weights = torch.tensor([3.0 if task.target < 4 else 1.0 for task in TASKS])
    elif name.startswith("context:"):
        context = name.removeprefix("context:")
        if context not in CONTEXTS:
            raise ValueError(name)
        weights = torch.tensor([float(task.context == context) for task in TASKS])
    elif name.startswith("target:"):
        target = int(name.removeprefix("target:"))
        if not 0 <= target < 6:
            raise ValueError(name)
        weights = torch.tensor([float(task.target == target) for task in TASKS])
    else:
        raise ValueError(name)
    return weights / weights.sum()


POLICIES = (
    "uniform",
    "fixed_dependent",
    *(f"context:{context}" for context in CONTEXTS),
    *(f"target:{target}" for target in range(6)),
)


@dataclass(frozen=True)
class Config:
    steps: int = 64
    batch_size: int = 64
    probe_every: int = 8
    warmup: int = 8
    seed_count: int = 3
    train_rows: int = 1024
    validation_rows: int = 512
    test_rows: int = 512
    labeled_rows: int = 64
    endpoint: str = "validation"
    # Fixed arms discard probes; skipping them leaves training bit-identical.
    fixed_probes: bool = True


@dataclass(frozen=True)
class Split:
    x: Tensor
    y: Tensor
    validation_x: Tensor
    validation_y: Tensor
    test_x: Tensor
    test_y: Tensor

    def endpoint(self, name: str) -> tuple[Tensor, Tensor]:
        if name == "validation":
            return self.validation_x, self.validation_y
        if name == "test":
            return self.test_x, self.test_y
        raise ValueError(name)


class Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
        )
        self.head = nn.Linear(16, 6)

    def forward(self, x: Tensor, visibility: Tensor) -> Tensor:
        return cast(Tensor, self.head(self.encoder(torch.cat((x, visibility), dim=1))))


def fixture(kind: str, seed: int, cfg: Config) -> Split:
    """Draw disjoint train/validation/test rows; scale with training X alone."""
    rng = np.random.default_rng(seed)
    n = cfg.train_rows + cfg.validation_rows + cfg.test_rows
    latent = rng.normal(size=(n, 2))
    noise = rng.normal(size=(n, 6))
    if kind == "dependent":
        x = np.column_stack(
            (
                latent[:, 0] + 0.2 * noise[:, 0],
                latent[:, 0] + 0.2 * noise[:, 1],
                latent[:, 1] + 0.2 * noise[:, 2],
                latent[:, 1] + 0.2 * noise[:, 3],
                noise[:, 4],
                noise[:, 5],
            )
        )
        signal = latent[:, 0] + latent[:, 1]
    elif kind == "interaction":
        x = noise.copy()
        x[:, 2] = x[:, 0] * x[:, 1] + 0.2 * noise[:, 2]
        x[:, 3] = np.tanh(x[:, 0] - x[:, 1]) + 0.2 * noise[:, 3]
        signal = x[:, 2] + x[:, 3]
    elif kind == "independent":
        x = noise.copy()
        signal = x[:, 0] + x[:, 1]
    else:
        raise ValueError(kind)
    y = (signal + 0.3 * rng.normal(size=n) > 0).astype(np.float32)
    train = slice(0, cfg.train_rows)
    validation = slice(cfg.train_rows, cfg.train_rows + cfg.validation_rows)
    test = slice(cfg.train_rows + cfg.validation_rows, n)
    mean, scale = x[train].mean(0), x[train].std(0)
    x = np.clip((x - mean) / np.maximum(scale, 1e-6), -5, 5).astype(np.float32)
    return Split(
        *(
            torch.from_numpy(array[part].copy())
            for part in (train, validation, test)
            for array in (x, y)
        )
    )


def view(x: Tensor, pool: Tensor, task: Task, *, seed: int) -> tuple[Tensor, Tensor]:
    """Withhold the target before any context corruption or model input."""
    gen = torch.Generator().manual_seed(seed)
    visible = torch.ones_like(x)
    visible[:, task.target] = 0
    if task.context == "sparse":
        keep = torch.rand(x.shape, generator=gen) > 0.5
        visible *= keep.float()
        empty = visible.sum(dim=1) == 0
        visible[empty, (task.target + 1) % x.shape[1]] = 1
    observed = x.clone()
    if task.context == "jitter":
        observed += 0.1 * torch.randn(x.shape, generator=gen)
    elif task.context == "replace":
        donors = torch.randint(len(pool), (len(x),), generator=gen)
        corrupt = torch.rand(x.shape, generator=gen) < 0.3
        observed = torch.where(corrupt, pool[donors], observed)
    observed *= visible
    return observed, visible


def task_loss(
    model: Predictor, x: Tensor, pool: Tensor, task: Task, seed: int
) -> Tensor:
    observed, visible = view(x, pool, task, seed=seed)
    return F.mse_loss(model(observed, visible)[:, task.target], x[:, task.target])


def alignment(
    model: Predictor,
    optimizer: torch.optim.AdamW,
    historical: list[tuple[Tensor, ...]],
    step: int,
    gradients: tuple[Tensor, ...],
) -> float:
    """Eq. 2, including AdamW second moments and the floor(e/2) checkpoint."""
    old = historical[step // 2]
    score = 0.0
    for p, past, grad in zip(model.parameters(), old, gradients, strict=True):
        state = optimizer.state[p]
        if not state:
            continue
        beta2 = optimizer.param_groups[0]["betas"][1]
        bias = 1 - beta2 ** int(state["step"].item())
        vhat = state["exp_avg_sq"] / bias
        preconditioner = optimizer.param_groups[0]["lr"] / (
            vhat.sqrt() + optimizer.param_groups[0]["eps"]
        )
        score += float((grad * preconditioner * (past - p.detach())).sum())
    return score


def probabilities(logits: Tensor, arm: str, policy: str | None = None) -> Tensor:
    if arm in ("uniform", "fixed_dependent"):
        return policy_weights(arm)
    if arm == "tuned":
        if policy is None:
            raise ValueError("the tuned arm needs a selected policy")
        return policy_weights(policy)
    if arm in ADAPTIVE_ARMS:
        return 0.2 / len(TASKS) + 0.8 * torch.softmax(logits, dim=0)
    raise ValueError(arm)


def encode(model: Predictor, x: Tensor) -> Tensor:
    with torch.no_grad():
        return cast(Tensor, model.encoder(torch.cat((x, torch.ones_like(x)), dim=1)))


def readout(
    model: Predictor, split: Split, endpoint: str, cfg: Config
) -> dict[str, float]:
    """Frozen-encoder endpoints on one declared evaluation split.

    The logistic probe sees only the first ``labeled_rows`` training labels.
    Every arm shares these labels, the probe and its penalty.
    """
    model.eval()
    x, y = split.endpoint(endpoint)
    train_z = encode(model, split.x[: cfg.labeled_rows])
    z = encode(model, x)
    weights = fit_logistic_probe(
        torch.cat((train_z, torch.ones(len(train_z), 1)), dim=1),
        split.y[: cfg.labeled_rows],
    )
    logits = torch.cat((z, torch.ones(len(z), 1)), dim=1) @ weights
    with torch.no_grad():
        # Clean single-column reconstruction on unseen rows: full context, no
        # corruption; the fixed seed is unused by the full view.
        masked = [
            float(task_loss(model, x, split.x, Task(target, "full"), 0))
            for target in range(6)
        ]
    spread = z.std(dim=0)
    singular = torch.linalg.svdvals(z - z.mean(dim=0))
    share = singular / singular.sum().clamp_min(1e-12)
    entropy = -(share * share.clamp_min(1e-12).log()).sum()
    return {
        "bce": float(F.binary_cross_entropy_with_logits(logits, y)),
        "masked_mse": sum(masked) / len(masked),
        "feature_std": float(spread.mean()),
        "dead_fraction": float((spread < 1e-6).float().mean()),
        "effective_rank": float(entropy.exp()),
    }


def fit_logistic_probe(z: Tensor, labels: Tensor) -> Tensor:
    """Fit L2-regularized Bernoulli likelihood with a free intercept.

    The last column of ``z`` is the intercept. Newton steps are deterministic,
    and all arms see the same frozen features and 64 training labels.
    """
    penalty = 0.1 * torch.eye(z.shape[1], dtype=z.dtype)
    penalty[-1, -1] = 0
    weights = torch.zeros(z.shape[1], dtype=z.dtype)
    for _ in range(20):
        probabilities = torch.sigmoid(z @ weights)
        gradient = z.T @ (probabilities - labels) + penalty @ weights
        curvature = probabilities * (1 - probabilities)
        hessian = z.T @ (curvature[:, None] * z) + penalty
        weights -= torch.linalg.solve(hessian, gradient)
    return weights


def raw_readout(split: Split, endpoint: str, cfg: Config) -> float:
    """Reference: the same probe on standardized raw X, without an encoder."""
    x, y = split.endpoint(endpoint)
    train = torch.cat((split.x[: cfg.labeled_rows], torch.ones(cfg.labeled_rows, 1)), 1)
    weights = fit_logistic_probe(train, split.y[: cfg.labeled_rows])
    logits = torch.cat((x, torch.ones(len(x), 1)), dim=1) @ weights
    return float(F.binary_cross_entropy_with_logits(logits, y))


def run_arm(
    kind: str, seed: int, arm: str, cfg: Config, policy: str | None = None
) -> dict[str, object]:
    split = fixture(kind, seed + 1, cfg)
    x = split.x
    torch.manual_seed(seed + 2)
    model = Predictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    history = [tuple(p.detach().clone() for p in model.parameters())]
    logits = torch.zeros(len(TASKS))
    counts = [0] * len(TASKS)
    traces: list[dict[str, object]] = []
    probes = 0
    # The no-pretraining reference keeps the paired initial encoder.
    steps = 0 if arm == "none" else cfg.steps
    probing = arm in ADAPTIVE_ARMS or cfg.fixed_probes
    for step in range(steps):
        # Stateless streams make row order, probe batches and view noise paired.
        gen = torch.Generator().manual_seed(seed + 1000 + step)
        probe_idx = torch.randint(len(x), (cfg.batch_size,), generator=gen)
        update_idx = torch.randint(len(x), (cfg.batch_size,), generator=gen)
        if probing and step >= cfg.warmup and step % cfg.probe_every == 0:
            rewards, signs, norms, losses = [], [], [], []
            for i, task in enumerate(TASKS):
                loss = task_loss(
                    model, x[probe_idx], x, task, seed + 100_000 + step * 100 + i
                )
                grads = torch.autograd.grad(loss, tuple(model.parameters()))
                grads = tuple(g.detach() for g in grads)
                signed = alignment(model, optimizer, history, step, grads)
                rewards.append(
                    abs(signed) if arm != "current_loss" else float(loss.detach())
                )
                signs.append(signed)
                norms.append(math.sqrt(sum(float(g.square().sum()) for g in grads)))
                losses.append(float(loss.detach()))
                probes += 1
            values = torch.tensor(rewards)
            permutation = torch.arange(len(TASKS))
            if arm == "shuffled":
                permutation = torch.randperm(len(TASKS), generator=gen)
                values = values[permutation]
            if arm in ADAPTIVE_ARMS:
                standardized = (values - values.mean()) / values.std().clamp_min(1e-8)
                logits = (0.8 * logits + 0.2 * standardized).clamp(-5, 5)
            traces.append(
                {
                    "step": step,
                    "rewards": rewards,
                    "reward_permutation": permutation.tolist(),
                    "assigned_rewards": values.tolist(),
                    "signed": signs,
                    "losses": losses,
                    "gradient_norms": norms,
                    "probabilities": probabilities(logits, arm, policy).tolist(),
                }
            )
        p = probabilities(logits, arm, policy)
        chosen = int(torch.multinomial(p, 1, generator=gen))
        counts[chosen] += 1
        optimizer.zero_grad(set_to_none=True)
        loss = task_loss(
            model, x[update_idx], x, TASKS[chosen], seed + 200_000 + step * 100 + chosen
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        history.append(tuple(param.detach().clone() for param in model.parameters()))
    return {
        "fixture": kind,
        "seed": seed,
        "arm": arm,
        "policy": policy,
        "endpoint": cfg.endpoint,
        **readout(model, split, cfg.endpoint, cfg),
        "raw_bce": raw_readout(split, cfg.endpoint, cfg),
        "update_gradients": steps,
        "probe_gradients": probes,
        "counts": counts,
        "trace": traces,
    }


def _rank(values: Tensor) -> Tensor:
    return values.argsort().argsort().to(torch.float64)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = _rank(torch.tensor(a)), _rank(torch.tensor(b))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))


def trace_summary(result: Mapping[str, object]) -> dict[str, object]:
    """Selection and reward diagnostics for one run's probe trace.

    Rank correlations are averaged over probe rounds. ``reward_vs_norm`` near
    one means the controller is selecting gradient scale; ``reward_vs_loss``
    near one means it reduces to a current-loss controller.
    """
    counts = torch.tensor(cast(list[int], result["counts"]), dtype=torch.float64)
    frequency = counts / counts.sum().clamp_min(1)
    by_target = [float(frequency[i * 4 : i * 4 + 4].sum()) for i in range(6)]
    by_context = [float(frequency[j::4].sum()) for j in range(4)]
    trace = cast(list[dict[str, list[float]]], result["trace"])
    summary: dict[str, object] = {
        "target_frequency": by_target,
        "context_frequency": by_context,
        "max_task_frequency": float(frequency.max()),
    }
    if trace:
        final = torch.tensor(trace[-1]["probabilities"], dtype=torch.float64)
        entropy = -torch.xlogy(final, final).sum() / math.log(len(TASKS))
        rounds = len(trace)
        summary |= {
            "final_normalised_entropy": float(entropy),
            "reward_vs_norm": sum(
                spearman(t["rewards"], t["gradient_norms"]) for t in trace
            )
            / rounds,
            "reward_vs_loss": sum(spearman(t["rewards"], t["losses"]) for t in trace)
            / rounds,
            "positive_alignment_share": sum(
                sum(s > 0 for s in t["signed"]) for t in trace
            )
            / (rounds * len(TASKS)),
        }
    return summary


def condense(trace: Sequence[Mapping[str, list[float]]]) -> list[dict[str, object]]:
    """Per-round marginals of a probe trace, small enough to keep for every run."""
    rounds: list[dict[str, object]] = []
    for entry in trace:
        p = torch.tensor(entry["probabilities"], dtype=torch.float64)
        rewards = torch.tensor(entry["rewards"], dtype=torch.float64)
        rounds.append(
            {
                "step": entry["step"],
                "target_probability": [
                    float(p[i * 4 : i * 4 + 4].sum()) for i in range(6)
                ],
                "context_probability": [float(p[j::4].sum()) for j in range(4)],
                "reward_mean": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "reward_vs_norm": spearman(entry["rewards"], entry["gradient_norms"]),
                "reward_vs_loss": spearman(entry["rewards"], entry["losses"]),
                "positive_alignment_share": sum(s > 0 for s in entry["signed"])
                / len(entry["signed"]),
            }
        )
    return rounds


def paired(
    rows: Sequence[Mapping[str, object]], kind: str, arm: str, metric: str = "bce"
) -> list[float]:
    """Per-seed metric of ``arm`` minus uniform, in seed order."""

    def values(name: str) -> dict[int, float]:
        return {
            cast(int, r["seed"]): cast(float, r[metric])
            for r in rows
            if r["fixture"] == kind and r["arm"] == name
        }

    arm_values, base = values(arm), values("uniform")
    return [arm_values[s] - base[s] for s in sorted(base)]


def _job(job: tuple[str, int, str, Config, str | None]) -> dict[str, object]:
    torch.set_num_threads(1)
    kind, seed, arm, cfg, policy = job
    row = run_arm(kind, seed, arm, cfg, policy)
    row["summary"] = trace_summary(row)
    return row


def run_jobs(
    jobs: Sequence[tuple[str, int, str, Config, str | None]], workers: int
) -> list[dict[str, object]]:
    if workers <= 1:
        return [_job(job) for job in jobs]
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        return list(executor.map(_job, jobs))


def provenance(workers: int) -> dict[str, object]:
    return {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"], text=True
        ).strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "workers": workers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    args = parser.parse_args()
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        parser.error("confirmation requires a clean committed source tree")
    if args.output.exists():
        parser.error("refusing to overwrite confirmation evidence")
    selection = json.loads(args.selection.read_text())
    cfg = replace(Config(**selection["config"]), endpoint="test", fixed_probes=True)
    tuned = cast(dict[str, str], selection["selected"])
    jobs = [
        (kind, seed, arm, cfg, tuned[kind] if arm == "tuned" else None)
        for kind in FIXTURES
        for seed in CONFIRMATION_SEEDS
        for arm in ARMS
    ]
    rows = run_jobs(jobs, args.workers)
    for row in rows:
        trace = cast(list[dict[str, list[float]]], row["trace"])
        row["rounds"] = condense(trace)
        # Full candidate-level traces are kept for the first seed only.
        if row["seed"] != CONFIRMATION_SEEDS[0]:
            row.pop("trace")
    args.output.mkdir(parents=True)
    (args.output / "environment.json").write_text(
        json.dumps(
            provenance(args.workers) | {"seeds": list(CONFIRMATION_SEEDS)}, indent=2
        )
        + "\n"
    )
    (args.output / "results.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "selection": selection["selected"],
                "tasks": [asdict(t) for t in TASKS],
                "results": rows,
            }
        )
        + "\n"
    )
    for kind in FIXTURES:
        for arm in ARMS:
            values = [
                cast(float, r["bce"])
                for r in rows
                if r["fixture"] == kind and r["arm"] == arm
            ]
            print(f"{kind:12} {arm:16} {np.mean(values):.4f}", flush=True)


if __name__ == "__main__":
    main()
