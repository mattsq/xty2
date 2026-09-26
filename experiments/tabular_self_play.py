"""Bounded real-X self-play pilot; no xty2 recipe or executor contract is implied.

Run ``python -m experiments.tabular_self_play --output results.json``. The
predeclared protocol lives in docs/proposals/tabular-self-play-ssl.md §8.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
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


TASKS = tuple(
    Task(i, context)
    for i in range(6)
    for context in ("full", "sparse", "jitter", "replace")
)
ARMS = ("uniform", "fixed_dependent", "alignment", "current_loss", "shuffled")


@dataclass(frozen=True)
class Config:
    steps: int = 64
    batch_size: int = 64
    probe_every: int = 8
    warmup: int = 8
    seed_count: int = 3
    train_rows: int = 1024
    test_rows: int = 512
    labeled_rows: int = 64


class Predictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
        )
        self.head = nn.Linear(16, 6)

    def forward(self, x: Tensor, visibility: Tensor) -> Tensor:
        return cast(Tensor, self.head(self.encoder(torch.cat((x, visibility), dim=1))))


def fixture(kind: str, seed: int, cfg: Config) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Draw disjoint populations, and fit scaling using training X alone."""
    rng = np.random.default_rng(seed)
    n = cfg.train_rows + cfg.test_rows
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
    mean, scale = x[: cfg.train_rows].mean(0), x[: cfg.train_rows].std(0)
    x = np.clip((x - mean) / np.maximum(scale, 1e-6), -5, 5).astype(np.float32)
    return (
        torch.from_numpy(x[: cfg.train_rows].copy()),
        torch.from_numpy(y[: cfg.train_rows].copy()),
        torch.from_numpy(x[cfg.train_rows :].copy()),
        torch.from_numpy(y[cfg.train_rows :].copy()),
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


def probabilities(logits: Tensor, arm: str) -> Tensor:
    if arm == "uniform":
        return torch.full((len(TASKS),), 1 / len(TASKS))
    if arm == "fixed_dependent":
        # Deliberately fixed, informative-feature heuristic; not a tuned control.
        weights = torch.tensor([3.0 if task.target < 4 else 1.0 for task in TASKS])
        return weights / weights.sum()
    return 0.2 / len(TASKS) + 0.8 * torch.softmax(logits, dim=0)


def readout(
    model: Predictor,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    cfg: Config,
) -> float:
    """Identical frozen logistic probe and restricted labels for each arm."""
    model.eval()
    with torch.no_grad():
        visible = torch.ones_like(x_train)
        train_z = model.encoder(torch.cat((x_train, visible), dim=1))
        test_z = model.encoder(torch.cat((x_test, torch.ones_like(x_test)), dim=1))
        train_z = torch.cat(
            (train_z[: cfg.labeled_rows], torch.ones(cfg.labeled_rows, 1)), dim=1
        )
        test_z = torch.cat((test_z, torch.ones(len(x_test), 1)), dim=1)
        weights = fit_logistic_probe(train_z, y_train[: cfg.labeled_rows])
        return float(F.binary_cross_entropy_with_logits(test_z @ weights, y_test))


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


def run_arm(kind: str, seed: int, arm: str, cfg: Config) -> dict[str, object]:
    x, y, test_x, test_y = fixture(kind, seed + 1, cfg)
    torch.manual_seed(seed + 2)
    model = Predictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    history = [tuple(p.detach().clone() for p in model.parameters())]
    logits = torch.zeros(len(TASKS))
    counts = [0] * len(TASKS)
    traces: list[dict[str, object]] = []
    probes = 0
    for step in range(cfg.steps):
        # Stateless streams make row order, probe batches and view noise paired.
        gen = torch.Generator().manual_seed(seed + 1000 + step)
        probe_idx = torch.randint(len(x), (cfg.batch_size,), generator=gen)
        update_idx = torch.randint(len(x), (cfg.batch_size,), generator=gen)
        if step >= cfg.warmup and step % cfg.probe_every == 0:
            rewards, signs, norms = [], [], []
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
                probes += 1
            values = torch.tensor(rewards)
            permutation = torch.arange(len(TASKS))
            if arm == "shuffled":
                permutation = torch.randperm(len(TASKS), generator=gen)
                values = values[permutation]
            if arm in ("alignment", "current_loss", "shuffled"):
                standardized = (values - values.mean()) / values.std().clamp_min(1e-8)
                logits = (0.8 * logits + 0.2 * standardized).clamp(-5, 5)
            traces.append(
                {
                    "step": step,
                    "rewards": rewards,
                    "reward_permutation": permutation.tolist(),
                    "assigned_rewards": values.tolist(),
                    "signed": signs,
                    "gradient_norms": norms,
                    "probabilities": probabilities(logits, arm).tolist(),
                }
            )
        p = probabilities(logits, arm)
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
        "test_bce": readout(model, x, y, test_x, test_y, cfg),
        "update_gradients": cfg.steps,
        "probe_gradients": probes,
        "counts": counts,
        "trace": traces,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    cfg = Config(steps=args.steps, seed_count=args.seeds)
    if cfg.steps <= cfg.warmup or cfg.seed_count < 1:
        parser.error("steps must exceed warmup and seeds must be positive")
    torch.set_num_threads(1)
    rows = [
        run_arm(kind, 310_000 + 100 * i, arm, cfg)
        for kind in ("dependent", "interaction", "independent")
        for i in range(cfg.seed_count)
        for arm in ARMS
    ]
    if args.output.exists():
        parser.error("refusing to overwrite existing results")
    args.output.write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "tasks": [asdict(t) for t in TASKS],
                "results": rows,
            },
            indent=2,
        )
        + "\n"
    )
    for kind in ("dependent", "interaction", "independent"):
        for arm in ARMS:
            values = [
                cast(float, r["test_bce"])
                for r in rows
                if r["fixture"] == kind and r["arm"] == arm
            ]
            print(f"{kind:12} {arm:16} {np.mean(values):.4f}")


if __name__ == "__main__":
    main()
