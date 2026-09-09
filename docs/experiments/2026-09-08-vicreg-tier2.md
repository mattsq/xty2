# VICReg Tier 2 study and spread audit, 2026-09-08

The ten-seed study meets **three of four** required targets and is `deviating`.
Full-arm embedding spread is `0.268395 +/- 0.00169` against a predeclared
`>= 0.5`. The three attribution and guardrail targets pass. This is
project-local mechanism evidence on a tabular fixture, not a reproduction of
Bardes et al.'s image benchmarks.

The miss is audited below rather than absorbed. `CLAUDE.md` says to treat a
negative result as an implementation failure until the method has been audited
equation by equation, and card §6.4 says a miss "requires an audit and a
recorded result, not automatic threshold relaxation". No threshold, seed,
constant or guardrail definition was changed.

## Protocol and execution

- Source commit: `a9fec5cc9623`, clean checkout.
- Protocol digest: `12d4b692ce7c1f3c1432addaaedf78ee8a81062855b784d25db7d97f6dcca911`.
- Ten replicates: base seeds `190000 + 100*i`, `i=0,...,9`; four arms each.
- 1,000 pretraining steps per pretrained arm and 3,000 fitting steps per arm:
  30,000 pretraining steps and 120,000 fitting steps in total; four workers.
- Linux x86_64 CPU; Python 3.11.15, PyTorch 2.14.0+cu130 (CPU tensors only),
  NumPy 2.4.6, pytest 9.1.1; one thread per worker, deterministic algorithms on.
- No protocol deviation.
- [Raw JSON](results/vicreg-a9fec5cc9623.json) carries every per-seed value.

Codex's Tier 1 record was taken on a `2.14.0+cpu` build and this study on a
`2.14.0+cu130` one, so the two disagree in the fourth decimal on shared
quantities. Both are dated, non-normative records of their own run.

## Required results

| Metric | Mean +/- stderr | Bound | Pass |
|---|---:|---:|:---:|
| `full_arm_embedding_spread` | 0.268395 +/- 0.0016896 | >= 0.5 | no |
| `variance_ablation_spread_gap` | 0.258219 +/- 0.0017140 | >= 0.1 | yes |
| `covariance_ablation_redundancy_gap` | 473.643 +/- 0.48522 | >= 0.01 | yes |
| `pretraining_outcome_NLL_cost` | -0.021196 +/- 0.0064447 | <= 0.05 | yes |

Per-seed full-arm spread: 0.2705, 0.2645, 0.2621, 0.2609, 0.2671, 0.2764,
0.2703, 0.2768, 0.2671, 0.2684. The miss is not a seed effect: the largest
replicate is 0.2768 and the target is 0.5.

## What the passing targets say

Both attribution targets clear their bounds by two orders of magnitude of their
own standard error, so the two terms are doing what equation (6) says they do.

- **Variance.** Zeroing `mu` collapses the embedding completely: spread
  `0.010177 +/- 0.00017`, which is `sqrt(0 + 1e-4)` to four figures, with the
  fraction of dimensions below standard deviation 0.1 at exactly 1.0 on every
  seed and diagonal covariance energy at `1.18e-07 +/- 1.18e-07`. On some seeds
  that energy is exactly zero in float32, which is why §6.4 scopes its
  zero-denominator rejection to the full and no-covariance arms.
- **Covariance.** Zeroing `nu` raises redundancy from `37.07 +/- 0.48` to
  `510.72 +/- 0.03` and, more tellingly, drives the top covariance eigenvalue's
  share of total energy from `0.1552 +/- 0.0027` to `0.99981 +/- 0.000017`. The
  no-covariance arm reaches a *higher* spread (`0.4454 +/- 0.0103`) than the
  full arm by inflating one direction rather than many, which is exactly the
  redundancy the term exists to remove and the reason spread alone is not a
  sufficient statistic for a healthy embedding.
- **Transfer.** Pretraining costs nothing on the factual outcome fit:
  `-0.0212 +/- 0.0064` nat/row, i.e. the full arm is slightly *better* than no
  pretraining. Treatment NLL is `-0.077 +/- 0.069` nat/row and its sign is not
  established. ATE error is `0.653 +/- 0.078` (full) against `0.564 +/- 0.108`
  (no pretraining); card §2 does not claim treatment-effect recovery and these
  overlap.

## Audit of the spread miss

Each diagnostic below departs from the reviewed recipe on exactly one axis, was
run at base seed 190000, and is **not** the card's protocol. They are reported
as terminal `spread_first` from the pretraining log.

| Diagnostic | Terminal spread |
|---|---:|
| Reviewed recipe, 1,000 steps | 0.2646 |
| Reviewed recipe, 10,000 steps | 0.2737 |
| Reviewed recipe, lr 1e-2 | 0.2033 |
| Covariance coefficient zeroed | 0.3907 |
| **Invariance coefficient zeroed** | **0.8246** |
| Invariance and covariance zeroed | 1.9716 |
| Corruption rate 0.2 instead of 0.6 | 0.4110 |
| Corruption rate 0.0, identical views | 0.8260 |

1. **Not the step budget.** Ten times the declared budget buys 0.009 of spread.
   The trajectory is flat from roughly step 100 onward and the gradient norm
   keeps falling, so this is an equilibrium and not an unfinished warm-up.
2. **Not the learning rate.** Adam at 1e-2 reaches a *lower* spread.
3. **Not primarily the covariance term.** Zeroing `nu` outright reaches 0.391,
   still short of 0.5.
4. **It is the invariance term against deviation 3's view strength.** Zeroing
   `lambda` reaches 0.825, and setting the corruption rate to 0 — which makes
   the two branches identical and the invariance term identically zero — also
   reaches 0.826. The two agree because they are the same intervention: with
   `FeatureCorruption(rate=0.6)` on six columns of which four carry cluster
   signal, the branches share too little for the encoder to make them agree, so
   the only way left to reduce `mean_{i,j}(Z-Z')^2` is to shrink the embedding.
   The variance hinge opposes that, and the balance sits at spread 0.26.

The card's own §6.4 diagnostic measures the same damage directly: the study's
corruption moves the analytic `p(t=1|x)` by `0.2353 +/- 0.0017`, and its
control — a corruption confined to the two signal-free columns — moves it by
exactly `0` on every replicate, so the number is cluster damage and not noise.

Scaling arithmetic agrees with the mechanism. At the terminal state the two
branches' embeddings are nearly unrelated relative to their own spread
(`inv / 2 * spread^2` is 0.797, against 0.968 with the invariance term absent
and 0.296 at corruption rate 0.2). Multiplying the embedding by `s` to reach
unit standard deviation would multiply the invariance penalty by `s^2` and the
covariance penalty by `s^4`, while the variance hinge can return at most
`25 * 0.735`. Descent therefore stops well before `gamma = 1`.

## Reading

Three conclusions, in decreasing confidence.

1. The implementation is not the cause. The three reductions, both branches'
   gradients and the expander topology are pinned to the author source by Tier
   0 against a float64 loop oracle, the two attribution targets separate the
   terms cleanly, and the spread responds to `lambda`, to `nu` and to the view
   rate in the directions the equations predict.
2. The 0.5 threshold was set before any measurement existed. Card §6.4 says so
   in its own words — "provisional scientific targets for review, not measured
   tolerances" — and §7 records that no measured spread was available when it
   was written. A spread target near `gamma` is reasonable for the published
   setting and was not re-derived against a six-column tabular fixture whose
   two views retain little mutual information.
3. The binding constraint is a declared deviation, not a defect. Deviation 3
   already predicted it: "Can erase treatment signal; report view damage and
   transfer outcomes." It did erase it, the damage is now measured, and the
   transfer outcome is nonetheless within its guardrail.

## Not done here

Changing the corruption rate, the expander width, the batch size or the 0.5
threshold would each be a card amendment requiring review before implementation,
and retuning a tolerance after seeing a result is what `FIDELITY.md` §3 forbids.
The next packet is that review. If it lowers the spread target, the number to
justify is a project-local one; if it lowers the corruption rate, the arms above
say roughly what to expect and every §6 number must be re-measured, because a
different view is a different study.
