# VIME Tier 2 run and target audit

This is a dated diagnostic record. It is not normative. It changes no bound,
no §4 value and no ledger row. The card's §6 audit note carries the proposed
amendments for review.

`CLAUDE.md` says to treat a negative result as an implementation failure until
the method has been audited equation by equation. The first Tier 2 replicate
missed the dependent-block bound. The audit asked two questions in order:

1. Is the implementation faithful?
2. Can any model trained on the published objective meet the bound?

The answers are yes and no.

## The recorded run

- **Source:** committed source `68a342575676`, tree
  `5b015c9d553b9cfd0f6f7c1cf6db9505f8d28cd2`, run in a clean detached worktree
  with `python -m xty2.evaluation.runner --recipe vime --workers 4
  --write-ledger`.
- **Hash reachability:** that commit was then amended to add this record and
  the card's ledger row, so its hash is not reachable. The landed commit's
  `xty2/` subtree is the one that ran. Its `tests/` subtree adds one Tier 0
  test that reads this recorded evidence.
- **Environment:** Python 3.11.15, PyTorch 2.14.0 and NumPy 2.4.6 on Linux
  x86-64, with four worker processes and one deterministic Torch thread each.
- **Evidence:** [the result with every per-seed value](results/vime-68a3425/vime.json),
  [the compiled plan](results/vime-68a3425/plan.txt),
  [the environment](results/vime-68a3425/environment.json) and
  [the audit probes](results/vime-68a3425/audit.py).

| Metric (10 seeds) | Mean ± stderr | Card bound | Outcome |
|---|---|---|---|
| dependent-block ratio | 1.129 ± 0.047 | < 0.95 | fails on every seed |
| independent-block ratio | 1.232 ± 0.094 | >= 0.98 | passes (see below) |
| held-out outcome NLL ratio | 1.004 ± 0.009 | <= 1.05 | passes |
| mask-estimation AUROC | 0.502 ± 0.002 | informational | within 0.01 of 0.5 on every seed |
| held-out treatment NLL ratio | 1.017 ± 0.017 | informational | — |
| terminal `l_m` | 0.664 ± 0.006 | — | above the 0.611 base-rate entropy |

The three ablations behave like the full arm:

| Ablation | Result |
|---|---|
| Fixed single draw | dependent ratio 1.126, AUROC 0.501 |
| Mask estimation only | AUROC 0.501 |
| Reconstruction only | dependent ratio 1.093 |

No ablation separates from the full arm. That is expected when none of the
pretext heads has moved far from its initialisation.

## 1. The implementation is faithful

An independent plain-PyTorch transcription of `vime_self.py` is in the audit
script. It uses one ReLU layer of width `d`, a linear mask head, a sigmoid
feature head, RMSprop with `rho = 0.9` and `eps = 1e-7`, BCE plus `2.0 × MSE`,
and a per-cell Bernoulli(0.3) mask with marginal donors. On replicate 0's rows
it reproduces the recipe's behaviour:

| Width, steps | Dependent ratio | Independent ratio | Mask AUROC |
|---|---|---|---|
| 6, 80 | 1.00 | 1.23 | 0.49 |
| 6, 16,000 | 1.08 | 1.49 | 0.50 |
| 64, 16,000 | 1.32 | 1.53 | 0.57 |

The xty2 recipe with only `pretrain.steps` raised gives the same pattern on
three seeds:

- The mask AUROC stays at 0.50 at every budget.
- The independent ratio rises toward 1.45 by 16,000 steps.

The two implementations use different initial draws and batch streams, so
they agree in pattern rather than in value. Width was varied only in the
plain transcription. There, raising the step count leaves the mask AUROC at
0.50, and raising the width is what moves it.

## 2. The bound is not attainable by the published objective

Card §6.1 derives `0.64` as the best ratio for one masked cell of `x0..x3`
"with `c` known exactly". That derivation assumes the predictor knows which
cell was masked. Eq. 6 trains `s_r` on every cell, and 70% of the cells are
visible copies. Its minimiser is the posterior mean `E[x_j | x̃]`, which copies
a cell in proportion to how likely it is to be clean.

On the independent block, the donor is drawn from the column's own marginal,
so corruption cannot be detected: `P(changed | x̃) = p_m` exactly. The
posterior mean is then `(1 − p_m) x̃_j + p_m μ_j`. On a corrupted cell its
error is `(1 − p_m)(d − μ) − (x − μ)` for an independent donor `d`, with
variance `(1 + (1 − p_m)²) σ²`. The ratio to the column mean is therefore
`1.49` at `p_m = 0.3`. Tier 0 checks this closed form by simulation
(`tests/invariants/test_vime_benchmark.py`).

On the dependent block, the audit script computes the exact posterior over the
cluster and the mask from the fixture's generating process. It uses card §6's
own held-out draw (seed `base + 7`) and the training column mean:

| Bayes-optimal Eq. 6 predictor (10 seeds) | Mean ± stderr |
|---|---|
| dependent-block ratio | 1.247 ± 0.007 |
| independent-block ratio | 1.488 ± 0.022 |
| mask AUROC | 0.588 ± 0.002 |

Three consequences follow:

- The `< 0.95` bound fails for the best possible model. The recorded 1.129
  sits below the Bayes value. That is consistent with an under-trained head
  copying less than the posterior mean does, not with it having learned more.
- The `>= 0.98` canary passes for the Bayes-optimal model (1.49) and for a pure
  copy (2.0). It cannot detect leakage.
- The width 64 run lands near the Bayes values on every metric, which is the
  behaviour a faithful implementation should show when capacity and budget
  allow.

## What would make §6 informative

These are proposals for review. The card's §6 audit note records them, and
nothing here implements them.

1. **Change the reference point from the column mean to the Bayes-optimal
   predictor.** Both endpoints are computable on this fixture: the identity
   copy and the analytic posterior mean. A normalised held-out Eq. 6 loss
   between them measures how much of the learnable structure the pretext task
   captured. It rewards detection and imputation together, as Eq. 6 does.
2. **Decide what the width and budget are for.** The reference script's rule,
   width `d` and ten epochs, gives 6 units and 80 steps here. With them, the
   encoder does not learn corruption detection at any budget, and does not
   learn the base rate at 80 steps. If the question is "the reference script
   on this fixture", the recorded result already answers it. If the question
   is "the method", supplement §5's validation ranges (widths up to `3d`,
   depths up to 5) and a step budget become §5 deviations to review before a
   rerun.
