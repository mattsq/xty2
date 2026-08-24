"""The synthetic DGP Tier 1 fits against (`FIDELITY.md` §3).

A tiny data-generating process we write, so that the true `p(t|x)`,
`p(y|x,t)` and CATE are **analytic** and a recipe's estimates can be compared
against something rather than against themselves. It is deliberately linear in
`x`: Tier 1 asks "is this recipe connected to the data at all", and a DGP that
is hard to fit answers that question with noise.

Two details are here on purpose and are worth not undoing.

**Treatments are missing completely at random, and the hidden values are
replaced with wrong ones.** `DESIGN.md` §1.1 says a row with `t_observed`
false carries an arbitrary valid class index and that reading it is a bug. An
arbitrary index that happened to be the *true* one would let exactly that bug
pass every assertion below, so the mask is applied by rotating the treatment
to a different class. Any code that reads `t` where it may not is then
measurably wrong rather than invisibly wrong.

**The outcome noise is small relative to the gap between arms.** That is what
makes `y` informative about a missing `t`, which is the mechanism
`MissingTreatmentMarginalNLL` exploits and the thing the load-bearing Tier 1
assertion is measuring.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import Tensor
from xty2.core import FeatureSpec, OutcomeSpec, Schema, XTYBatch

NUM_FEATURES = 5
NUM_TREATMENTS = 3
OUTCOME_NOISE = 0.3
"""`sigma`. Small against the arm gaps below, so `y` identifies the arm."""


@dataclass(frozen=True)
class SyntheticXTY:
    """A linear-Gaussian `p(t|x) p(y|x,t)`, with every truth available.

    Attributes:
        propensity_weights: `[D, K]`; `p(t|x) = softmax(x @ W)`.
        outcome_weights: `[K, D]`; `E[y|x,t=k] = x @ w_k + b_k`.
        outcome_bias: `[K]`; the intercepts, which are the ATE up to the mean
            of `x` — and `x` is standard normal, so `b_k - b_0` *is* the true
            ATE of arm `k` against arm 0.
        noise: The standard deviation of `y` around its arm mean.
    """

    propensity_weights: Tensor
    outcome_weights: Tensor
    outcome_bias: Tensor
    noise: float = OUTCOME_NOISE

    @classmethod
    def build(cls, seed: int = 0) -> SyntheticXTY:
        """A fixed DGP. Seeded so a Tier 1 failure is reproducible."""
        generator = torch.Generator().manual_seed(seed)
        return cls(
            propensity_weights=torch.randn(
                NUM_FEATURES, NUM_TREATMENTS, generator=generator
            ),
            outcome_weights=torch.randn(
                NUM_TREATMENTS, NUM_FEATURES, generator=generator
            ),
            # Well-separated arms: the ATEs are 1.5 and -2.0 against arm 0.
            outcome_bias=torch.tensor([0.5, 2.0, -1.5]),
        )

    @property
    def schema(self) -> Schema:
        """A schema with `D` plain continuous columns and a scalar outcome."""
        return Schema(
            features=tuple(
                FeatureSpec(f"x{index}", "continuous") for index in range(NUM_FEATURES)
            ),
            treatment_cardinality=NUM_TREATMENTS,
            outcome=OutcomeSpec(),
        )

    # -- the truths --------------------------------------------------------

    def true_log_propensity(self, x: Tensor) -> Tensor:
        """`[N, K]` log `p(t|x)`."""
        return torch.log_softmax(x @ self.propensity_weights, dim=-1)

    def true_means(self, x: Tensor) -> Tensor:
        """`[N, K]` `E[y|x,t=k]` for every arm — the counterfactual means."""
        return x @ self.outcome_weights.T + self.outcome_bias

    def true_cate(self, x: Tensor) -> Tensor:
        """`[N, K]` `E[y|x,t=k] - E[y|x,t=0]`. Column 0 is zero by definition."""
        means = self.true_means(x)
        return means - means[:, :1]

    def true_ate(self) -> Tensor:
        """`[K]` the population ATE against arm 0. Exact: `E[x] = 0`."""
        return self.outcome_bias - self.outcome_bias[0]

    # -- sampling ----------------------------------------------------------

    def draw(self, rows: int, *, seed: int, missing_rate: float = 0.0) -> XTYBatch:
        """One split, as a single `XTYBatch`.

        Args:
            rows: `N`.
            seed: Seeds a local generator, so a draw is reproducible without
                touching the global RNG the executor seeds.
            missing_rate: Fraction of rows whose treatment is hidden,
                completely at random. The hidden values are replaced with
                wrong ones — see the module docstring.
        """
        generator = torch.Generator().manual_seed(seed)
        x = torch.randn(rows, NUM_FEATURES, generator=generator)
        probabilities = self.true_log_propensity(x).exp()
        t = torch.multinomial(probabilities, num_samples=1, generator=generator)
        t = t.squeeze(1)
        means = self.true_means(x)
        y = means.gather(1, t[:, None]).squeeze(1) + self.noise * torch.randn(
            rows, generator=generator
        )
        observed = torch.rand(rows, generator=generator) >= missing_rate
        # Rotated, not kept: a hidden treatment that is still the true one
        # would make an illegal read of `batch.t` pass every assertion.
        stored = torch.where(observed, t, (t + 1) % NUM_TREATMENTS)
        return XTYBatch(
            x=x,
            t=stored,
            y=y,
            t_observed=observed,
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(rows, dtype=torch.long),
        )


def minibatches(split: XTYBatch, size: int, *, seed: int) -> Iterator[XTYBatch]:
    """Endless uniform minibatches drawn without replacement within a batch.

    Without replacement because `XTYBatch` requires unique `row_id`s — they
    key the provenance an artifact records (`DESIGN.md` §7.1) — and a batch
    holding one row twice would be a batch whose checkpoint claims fewer rows
    than it saw.
    """
    generator = torch.Generator().manual_seed(seed)
    rows = split.batch_size
    while True:
        index = torch.randperm(rows, generator=generator)[:size]
        yield XTYBatch(
            x=split.x[index],
            t=split.t[index],
            y=split.y[index],
            t_observed=split.t_observed[index],
            y_observed=split.y_observed[index],
            row_id=split.row_id[index],
        )


__all__ = [
    "NUM_FEATURES",
    "NUM_TREATMENTS",
    "OUTCOME_NOISE",
    "SyntheticXTY",
    "minibatches",
]
