# VIME Tier 2 runs and fixture audit

This is a dated diagnostic record. It is not normative; the card is. It covers
two Tier 2 runs on the same day:

1. **First run:** the original fixture. `deviating`, and the audit below shows
   no model trained on Eq. 6 could have passed it.
2. **Second run:** the amended fixture that the repository owner directed
   (see "The fixture change"). `reproduced`.


`CLAUDE.md` says to treat a negative result as an implementation failure until
the method has been audited equation by equation. The first Tier 2 replicate
missed the dependent-block bound. The audit asked two questions in order:

1. Is the implementation faithful?
2. Can any model trained on the published objective meet the bound?

The answers are yes and no.

## The first run

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
  copy (2.0), so it is loose. It still does its job: on an independent column a
  model can fall below 1 only with information about the clean value, which is
  leakage. An earlier version of this record said it "cannot detect leakage";
  that was wrong.
- The width 64 run lands near the Bayes values on every metric, which is the
  behaviour a faithful implementation should show when capacity and budget
  allow.

## The fixture change

The repository owner's position is that a fixture on which the method's own
optimum fails the bound is a fixture defect, not a model defect. The two
proposals the first version of this record made were resolved that way:

- **Reference point.** The Bayes-optimal Eq. 6 predictor is now part of the
  benchmark (`bayes_pretext`), reported beside the model on every draw, and
  the bound is placed against it.
- **Width and budget.** The reference script's width `d` and ten epochs are
  kept. What changes is the fixture they run on.

A pilot on the disjoint seed stream `700,000 + 100 i` varied the two fixture
properties that the audit had implicated:

- the within-cluster noise of `x0..x3`, which decides how visible a
  replaced cell is;
- the size of the unlabelled pool, which decides how many steps ten epochs is.

Three pilot seeds per cell, model at width 6 and ten epochs:

| Noise | Rows (steps) | Model dependent ratio | Bayes dependent ratio |
|---|---|---|---|
| 0.6 | 1,024 (80) | 1.04 | 1.29 |
| 0.6 | 16,384 (1,280) | 0.96 | 1.26 |
| 0.3 | 16,384 (1,280) | 0.70 | 0.73 |
| 0.2 | 1,024 (80) | 0.99 | 0.53 |
| 0.2 | 4,096 (320) | 0.91 | 0.52 |
| 0.2 | 16,384 (1,280) | 0.54 | 0.49 |

Neither change alone is enough:

- At noise 0.6 there is nothing learnable. The optimum itself is above 1.
- At 80 steps the model cannot learn what is there.

The card's §6.1 now declares noise 0.2 and 16,384 training rows. The paper's
own datasets are of that size or larger, so ten epochs over them is the
paper's regime. An eight-seed pilot of the full replicate on that fixture gave
a dependent ratio of `0.565 ± 0.013` (maximum `0.632`), an independent ratio
of `1.036 ± 0.005` (minimum `1.020`) and an outcome ratio of `0.974 ± 0.013`.
The bound `< 0.75` was then declared as the midpoint between the column mean
(1.0) and the optimum (about 0.49), before the Tier 2 stream (`190,000 +
100 i`) was run.

## The second run

- **Source:** committed source `46d3e4c7dc53`, tree
  `dbe7b13f5ed73cd1b3df3f1eeb63ec55b03dc05c`, run the same way. That commit was
  amended to add the evidence, so its hash is not reachable; the landed
  commit's `xty2/` subtree is the one that ran.
- **Plan:** it differs from the first run's in exactly two lines, the
  pretraining steps (80 → 1,280) and the split protocol's fixture name.
- **Evidence:** [result](results/vime-46d3e4c/vime.json),
  [plan](results/vime-46d3e4c/plan.txt),
  [environment](results/vime-46d3e4c/environment.json).

| Metric (10 seeds) | Mean ± stderr | Bound | Outcome |
|---|---|---|---|
| dependent-block ratio | 0.579 ± 0.009 | < 0.75 | passes; every seed 0.54–0.62 |
| independent-block ratio | 1.067 ± 0.007 | >= 0.98 | passes; every seed above 1.04 |
| held-out outcome NLL ratio | 0.985 ± 0.010 | <= 1.05 | passes |
| Bayes-optimal dependent ratio | 0.507 ± 0.011 | informational | — |
| mask-estimation AUROC | 0.525 ± 0.007 | informational | Bayes 0.698 |
| held-out treatment NLL ratio | 0.92 ± 0.18 | informational | ranges 0.39–1.86 |

The model reaches 0.579 against an optimum of 0.507, so it captures about
85% of the learnable reduction on the dependent block. On the independent
block it stays near the mean, so it detects nothing there.

The ablations:

- **Fixed single draw:** 0.581. It does not separate from fresh draws, so
  deviation 2 does not matter on this metric.
- **Reconstruction only:** 0.573. The dependent block is learned through
  `l_r`; `l_m` adds nothing measurable here.
- **Mask estimation only:** AUROC 0.536. At width 6 mask estimation stays
  weak, with or without `l_r`.

The treatment ratio is too noisy to support a direction.
