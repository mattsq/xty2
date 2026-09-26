"""The two probes behind `2026-09-26-vime-tier2-audit.md`.

1. The Bayes-optimal Eq. 6 predictor on card section 6's own held-out draw,
   computed exactly from the fixture's generating process.
2. An independent plain-PyTorch transcription of `vime_self.py`, at the
   reference width `d` and at width 64, for several step budgets.

Run from the repository root:
`uv run python docs/experiments/results/vime-68a3425/audit.py`.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import replace

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from xty2.core import Program
from xty2.evaluation.benchmarks import vime as benchmark
from xty2.evaluation.benchmarks.common import CLUSTER_SIGNAL, configure_worker
from xty2.evaluation.benchmarks.scarf import fixture
from xty2.recipes.vime import MASK_PROBABILITY

NOISE = 0.6
"""The fixture's within-cluster standard deviation on `x0..x3`."""


def _normal(value: Tensor, mean: float, scale: float) -> Tensor:
    return torch.exp(-0.5 * ((value - mean) / scale) ** 2) / (
        scale * math.sqrt(2 * math.pi)
    )


def bayes(index: int) -> dict[str, float]:
    """`E[x | x~]` and `P(changed | x~)` under the true DGP, on the Tier 2 draw."""
    world = fixture(index)
    recipe = benchmark._vime(world)
    recipe = replace(recipe, program=Program((replace(recipe.program[0], steps=1),)))
    _, population = benchmark._pretext(world, recipe)
    corrupted, clean = benchmark._held_out_corruption(world, population)
    location = population.statistics["x_location"].double()
    scale = population.statistics["x_scale"].double()
    observed = corrupted.x.double() * scale + location
    truth = clean.x.double() * scale + location
    changed = clean.x != corrupted.x
    p = MASK_PROBABILITY

    dependent = observed[:, :4]
    centres = (-CLUSTER_SIGNAL, CLUSTER_SIGNAL)
    marginal = sum(0.5 * _normal(dependent, c, NOISE) for c in centres)
    log_likelihood, kept = [], []
    for centre in centres:
        keep = (1 - p) * _normal(dependent, centre, NOISE)
        cell = keep + p * marginal
        log_likelihood.append(cell.log().sum(dim=1))
        kept.append(keep / cell)
    posterior = torch.softmax(torch.stack(log_likelihood, dim=1), dim=1)
    mean_dependent = sum(
        posterior[:, k : k + 1] * (kept[k] * dependent + (1 - kept[k]) * centres[k])
        for k in range(2)
    )
    changed_dependent = sum(posterior[:, k : k + 1] * (1 - kept[k]) for k in range(2))
    # An independent cell's donor has its own marginal, so the corruption is
    # undetectable there: `P(changed | x~) = p` and `E[x | x~] = (1 - p) x~`.
    mean_independent = (1 - p) * observed[:, 4:]
    prediction = torch.cat([mean_dependent, mean_independent], dim=1)
    changed_score = torch.cat(
        [changed_dependent, torch.full_like(mean_independent, p)], dim=1
    )
    column_mean = (population.rows.x.double().mean(dim=0) * scale + location).expand_as(
        truth
    )
    return {
        "dependent_ratio": benchmark._block_ratio(
            prediction, column_mean, truth, changed, benchmark.DEPENDENT_BLOCK
        ),
        "independent_ratio": benchmark._block_ratio(
            prediction, column_mean, truth, changed, benchmark.INDEPENDENT_BLOCK
        ),
        "mask_auroc": benchmark.auroc(changed_score.flatten(), changed.flatten()),
    }


def plain(width: int, steps: int) -> dict[str, float]:
    """`vime_self.py` in plain PyTorch on replicate 0's rows; not xty2's objects."""
    torch.manual_seed(0)
    world = fixture(0)
    rows = world.train.batch.x
    low, high = rows.min(dim=0).values, rows.max(dim=0).values
    train = (rows - low) / (high - low)
    test = (world.test.batch.x - low) / (high - low)
    generator = torch.Generator().manual_seed(7)
    mask = torch.rand(test.shape, generator=generator) < MASK_PROBABILITY
    donors = torch.randint(0, train.shape[0], test.shape, generator=generator)
    corrupted = torch.where(mask, train.gather(0, donors), test)
    changed = corrupted != test

    encoder = nn.Sequential(nn.Linear(6, width), nn.ReLU())
    mask_head, feature_head = nn.Linear(width, 6), nn.Linear(width, 6)
    parameters = [
        *encoder.parameters(),
        *mask_head.parameters(),
        *feature_head.parameters(),
    ]
    optimiser = torch.optim.RMSprop(parameters, lr=1e-3, alpha=0.9, eps=1e-7)
    for _ in range(steps):
        batch = train[torch.randperm(train.shape[0])[:128]]
        selected = torch.rand(batch.shape) < MASK_PROBABILITY
        replacement = train.gather(0, torch.randint(0, train.shape[0], batch.shape))
        tilde = torch.where(selected, replacement, batch)
        z = encoder(tilde)
        loss = F.binary_cross_entropy_with_logits(
            mask_head(z), (batch != tilde).float()
        ) + 2.0 * F.mse_loss(torch.sigmoid(feature_head(z)), batch)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        z = encoder(corrupted)
        logits, reconstruction = mask_head(z), torch.sigmoid(feature_head(z))
    column_mean = train.mean(dim=0).expand_as(test)
    return {
        "dependent_ratio": benchmark._block_ratio(
            reconstruction, column_mean, test, changed, benchmark.DEPENDENT_BLOCK
        ),
        "independent_ratio": benchmark._block_ratio(
            reconstruction, column_mean, test, changed, benchmark.INDEPENDENT_BLOCK
        ),
        "mask_auroc": benchmark.auroc(logits.flatten(), changed.flatten()),
    }


def main() -> None:
    configure_worker()
    oracle = [bayes(index) for index in range(10)]
    for key in ("dependent_ratio", "independent_ratio", "mask_auroc"):
        values = [row[key] for row in oracle]
        stderr = statistics.stdev(values) / math.sqrt(len(values))
        print(f"bayes {key}: {statistics.mean(values):.4f} +/- {stderr:.4f}")
    for width, steps in ((6, 80), (6, 16_000), (64, 16_000)):
        scored = {key: round(value, 3) for key, value in plain(width, steps).items()}
        print(f"plain width={width} steps={steps}: {scored}")


if __name__ == "__main__":
    main()
