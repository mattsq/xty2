# BYOL deviation audit and re-run

The 2026-09-16 Tier 2 run returned `deviating`: `ema_outcome_nll_gain` was
0.00386575 ± 0.00407, a lower one-standard-error bound of -0.000209. This
experiment asks which of three things produced that — a permitted variation
mattering more than realised, an implementation error, or an evaluation that
could not pass — and records the three amendments that follow.

It does not change the fixture, the DGP, the architecture or the optimiser.
It changes the target decay (§5 row 7), the downstream protocol (§5 row 4) and
which statistics are scored (§6.4), each prospectively and each with the
superseded version retained. It tests a local tabular adaptation, not ImageNet
reproduction or causal identification.

## 1. Source audit: no implementation error

Re-derived every mechanic against the reference pinned by the card,
[`82a347438fd93bdd4bc01764258223ae583e5cbf`](https://github.com/google-deepmind/deepmind-research/tree/82a347438fd93bdd4bc01764258223ae583e5cbf/byol).

| Mechanic | Source anchor | Finding |
|---|---|---|
| Loss, floor and reduction | `helpers.regression_loss`, `l2_normalize`; eq. (2) | Two directional squared distances summed then meaned over rows; floor on the squared norm; targets detached projections. Matches. |
| Head topology | `utils/networks.MLP` | Biased hidden Linear, affine BN at `decay_rate=0.9`/`eps=1e-5`, ReLU, bias-free output Linear, no output BN. Matches. |
| Optimiser | `utils/optimizers.lars`, `scale_by_lars`, `exclude_bias_and_norm` | Decay added to the gradient, then trust ratio, then momentum accumulation, then `-lr`; rank-<2 parameters excluded from both filters and still receiving the momentum step; both zero-norm guards are `where`, not clamps. Matches. |
| LR schedule | `utils/schedules.learning_schedule` | Linear warm-up then a half cosine over `total - warmup` reaching zero. Agrees with `WarmupCosine` at every step, including `step == warmup`, where the source crosses branches and both sides read exactly 1. Matches. |
| Teacher update | `byol_experiment._update_fn`; eqs. (3), (1) | Optimiser step precedes one parameter EMA; target BN state comes from the target's own two forwards, never from buffer EMA. Matches. |
| Initial copy | Appendix A; card §5 row 5 | Exact copy at construction, the disclosed variant. Matches. |
| View caching | `loss_fn` | Two view draws per step, shared between the online and target passes. Matches. |

No mechanism error was found, and Tier 0 already held independent oracles for
each of the above. **Hypothesis (b) is cleared.** What follows is why the run
deviated anyway.

## 2. The instrument could not carry the claim

**Dynamic range.** On this fixture, predicting the training mean scores about
1.376 nat and the Bayes-optimal conditional mean about 1.049, so the endpoint
spans about 0.327 nat. The recorded run's four arms all landed within 0.004 nat
of each other and reached the same downstream training loss to three decimals —
0.4797, 0.4800, 0.4802, 0.4798 — despite pre-transfer encoder effective ranks
of 3.24, 2.94, 6.68 and 22.02. The measured EMA gain was 1.18% of the endpoint's
range and the whole-of-pretraining effect 1.09%.

**A controlled protocol probe.** Six replicates on bases 620000-620500, holding
the fixture, seeds, pairing and pretraining fixed and varying only whether the
transferred encoder is trainable. The probe reproduces the recorded Tier 2
endpoint bit-for-bit on all twelve (seed, arm) points it overlaps, so it extends
the recorded run rather than replacing it.

| Held-out outcome NLL (mean, n=6) | step 0 | 100 | 300 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|---|
| fine-tuned, full | 1.40393 | 1.19492 | 1.11772 | 1.21493 | 1.21718 | 1.20425 |
| fine-tuned, no-pretraining | 1.40393 | 1.16464 | 1.11665 | 1.21000 | 1.20928 | 1.20511 |
| frozen, full | 1.40393 | 1.15647 | 1.10466 | 1.10633 | 1.09584 | 1.11826 |
| frozen, no-pretraining | 1.40393 | 1.15201 | 1.12043 | 1.13678 | 1.15062 | 1.15561 |

The whole-of-pretraining contrast `N_no_pretrain - N_full` at the reported step
3000 is **+0.0373 ± 0.0095 (5/6 seeds, t=3.94)** frozen against
**+0.0009 ± 0.0051 (3/6, t=0.17)** fine-tuned; at step 1000 it is
+0.0305 ± 0.0072 on 6/6 against -0.0049. Three mechanisms:

- the fine-tune moves the encoder **1.17** of its own initial norm, against the
  **0.171** pretraining moved it — 6.8 times further — and leaves the pretrained
  and random encoders 0.972 apart at the end;
- effective rank collapses to about 1.5 within 100 downstream steps in every
  arm, from 3.12 pretrained or 21.62 random, then re-grows to the same ~3.2;
- 3000 steps is past the optimum. Every fine-tuned arm peaks near step 300
  (≈1.118) and is 0.087 nat worse by step 3000. The frozen protocol's worst
  checkpoint beats the fine-tuned protocol's best.

**Power.** The observed effect-to-spread ratio is 0.300, at which
`mean - stderr > 0` has **49.3%** power at ten replicates; 87% would need about
fifty. Of the other three declared gates, the cost budget passed with the
measured effect fourteen times inside its tolerance and the rank guard asks for
1.1 where an untrained encoder scores 22. One coin-flip gate carried the status.

**The mechanism was separating the arms all along.** Recomputed from the
recorded run with no re-fit:

| Contrast, full vs zero-decay | mean ± SE | lower 1-SE | seeds | t |
|---|---|---|---|---|
| `A_zero_decay - A_full` | +0.009726 ± 0.001119 | +0.008606 | 10/10 | 8.69 |
| `R_full - R_zero_decay` | +0.304 ± 0.125 | +0.179 | 8/10 | 2.44 |
| `ema_outcome_nll_gain` | +0.003866 ± 0.004075 | -0.000209 | 6/10 | 0.95 |

Both pre-transfer statistics clear a `mean - stderr > 0` bound on the data the
card already had. These are `simsiam.md` §6.4's instruments, adopted there after
its own audit found spread-based bounds unable to move, with downstream NLL kept
as a budget and the warning that "if full and controls are indistinguishable on
both axes, the attribution claim has not reproduced even if downstream NLL is
good." This card took SimSiam's §6.2 fixture but not its §6.4 instrument policy.

## 3. Two findings that bound what any downstream gate could show

**The zero-decay arm is not the paper's collapse control.** Its
`target_online_lag` is exactly 0.0 on all ten recorded seeds and its target BN
statistics equal the online ones, so the target is bit-identical to the online
network at every forward and the arm is the teacher-free update rule — stop
gradient plus predictor, no momentum encoder. Table 5(a)'s τ=0 row (0.3% against
72.5% top-1, 300 epochs, batch 4096) is a catastrophic collapse that does not
occur in this regime, and the predeclared effect size was anchored on it.

**The EMA had almost nothing to do.** `configs/byol.py` keys every preset to an
epoch budget and passes `target_ema` a horizon of
`num_epochs * train_images_per_epoch // batch_size`:

| Source preset | max_steps | base_ema | time constant | Σ(1−τ) |
|---|---|---|---|---|
| 40 epochs | 12511 | 0.97 | 33.3 | 187.7 |
| 100 epochs | 31278 | 0.99 | 100.0 | 156.4 |
| 300 epochs | 93835 | 0.99 | 100.0 | 469.2 |
| 1000 epochs | 312784 | 0.996 | 250.0 | 625.6 |
| **this card, 2026-09-16** | **1000** | **0.996** | **250.0** | **2.00** |

Deviation 3 read "1000 epochs" as "1000 steps", a factor of 313, while keeping
the base from the row tied to the long horizon. The target consequently ends
0.86 in parameter norm behind the online network on every recorded replicate: a
frozen early snapshot, not a slowly moving average.

## 4. Amendments

1. **§5 row 7 (new).** The base is derived, not inherited:
   `1 - base_local = (1 - base_source) * steps_source / steps_local`, taking the
   source row this card's own budget reaches — 1000 steps of batch 128 over 1024
   rows is 125 epochs, so `_EMA_PRESETS[100] = 0.99` over 31278 steps — giving
   `1 - 0.01 * 31278 / 1000 = 0.68722` and local Σ(1−τ) of 156.5 against that
   row's 156.4. The selected 1000-epoch row is unreachable at this horizon at
   any base: the same rule sends it to -0.251, outside the `[0, 1)` an EMA
   update requires. A `source_ema` arm keeps 0.996 so the change is measured.
   The choice of source row is a disclosed judgement (§7): the 40-epoch row
   would give 0.62467 and the 300-epoch row 0.06165.
2. **§5 row 4 (amended).** The transferred encoder is frozen for the downstream
   fit, which is the source's own linear-evaluation posture (§3.3) and narrows
   rather than widens the deviation. §6.2 checks the backbone bit-identical
   after the stage.
3. **§6.4 (amended).** `ema_alignment_gap` and `ema_rank_gap` carry the
   attribution at each statistic's own zero; the 0.05 cost budget and the 1.1
   rank guard are unchanged; `ema_outcome_nll_gain` becomes informational. The
   honest risk is that these two statistics were chosen after seeing them
   separate. The guards are that their thresholds are each statistic's own zero
   rather than a level read off a measurement, and that §6.2 re-runs on
   630000+100*i, disjoint from every seed this audit saw.

## 5. Execution and provenance

Run from committed source `b2c3d632a8638549ef01b205aa8e134f0c50b91a`:

```bash
uv run python -m xty2.evaluation.runner \
  --recipe byol --workers 4 \
  --output docs/experiments/results/byol-tier2-2026-09-17
```

Ten bases are `630000 + 100*i` for `i=0..9`. Each replicate runs all five arms
with 1024 training rows, 2048 held-out rows, 40 observed treatments, batch 128,
1000 pretraining steps where applicable and 3000 downstream steps over the XTY
heads alone. Warmup is 10 steps, the marginal ramp 1000. Diagnostics use 16
fixed batch pairs; encoder rank and view alignment are read from the terminal
pretraining checkpoint on all 2048 clean held-out rows, in eval mode on frozen
training BN buffers, before transfer.

Actual initial tensors, execution streams, view draws, masks, fitted scales,
teacher updates and stage transfers are checked during fitting, and the frozen
backbone is checked after it. Differences are formed within seed before sample
standard errors with `ddof=1`.

Environment: Python 3.11.15, Torch 2.14.0+cu130, NumPy 2.4.6, CPU, four workers.

## 6. Results

**Status: `deviating`.** All fifty fits completed with finite diagnostics. Both
attribution gates fail, and one fails with its sign reversed.

| Gate | Mean ± SE | One-SE bound | Decision |
|---|---|---|---|
| `ema_alignment_gap` | -0.001139 ± 0.000142 | -0.001281 | fail |
| `ema_rank_gap` | -0.883912 ± 0.080480 | -0.964391 | fail |
| `pretraining_outcome_nll_cost` | -0.050577 ± 0.004392 | -0.046185 | pass |
| `encoder_effective_rank` | 2.220651 ± 0.079634 | 2.141017 | pass |

Per arm, read from the terminal pretraining checkpoint except the last column:

| Arm | View alignment `A` | Encoder rank `R` | Target-online lag | Downstream NLL |
|---|---|---|---|---|
| full (base 0.68722) | 0.997849 | 2.2207 | 0.01077 | 1.101720 |
| source-ema (base 0.996) | 0.987225 | 3.1147 | 0.86350 | 1.104234 |
| zero-decay | 0.996710 | 3.1046 | 0.00000 | 1.104537 |
| no-predictor | 0.999948 | 2.3187 | 0.04039 | 1.117911 |
| no-pretraining | n/a | 22.2923 | n/a | 1.152296 |

The run changed three things at once, and the `source_ema` control arm is what
separates them. Three findings.

### 6.1 The frozen transfer worked

`pretraining_outcome_nll_cost` is -0.050577 ± 0.004392: pretraining *helps* the
endpoint by 0.051 nat, against 0.0036 under the fine-tuned protocol — a factor of
fourteen, where §2's six-seed probe predicted +0.037. The endpoint has real
dynamic range for the first time on this card.

`ema_outcome_nll_gain`, now informational, is +0.002817 ± 0.001557, a lower
one-standard-error bound of +0.001260. The statistic this card withdrew as a gate
would have passed under the protocol that replaced it. That is recorded because
it is awkward rather than despite it, and it does not restore the gate: the
withdrawal was argued from the instrument's dynamic range and its 49.3% power,
not from its verdict, and re-adopting a statistic on the one run where it happens
to pass is exactly what §6's disclosure rule exists to prevent.

### 6.2 One instrument replicated, one did not

Measured at the inherited base — which is what the 2026-09-16 run measured, and
what the `source_ema` arm reproduces on the new stream:

| Contrast at base 0.996 | 620000 stream | 630000 stream |
|---|---|---|
| alignment gap `A_zero_decay - A_source_ema` | +0.009726 ± 0.001119, 10/10, t=8.69 | **+0.009485 ± 0.000706, 10/10, t=13.44** |
| rank gap `R_source_ema - R_zero_decay` | +0.304 ± 0.125, 8/10, t=2.44 | **+0.010102 ± 0.073859, 5/10, t=0.14** |

The alignment gap returns the same number on a stream chosen after it was
selected, which is as much as a prospective re-run can give it. The rank gap does
not: +0.304 was a property of the old seed stream, not of the mechanism.
Withdrawing row 7 would not rescue `ema_rank_gap`, because at the inherited base
it still misses its bound.

### 6.3 Deviation 7 is falsified by its own control

| Paired contrast, full (0.68722) vs source-ema (0.996) | mean ± SE | seeds | t |
|---|---|---|---|
| `A_full - A_source_ema` (negative would favour the re-derivation) | +0.010624 ± 0.000748 | 10/10 | 14.20 |
| `R_source_ema - R_full` (negative would favour it) | +0.894013 ± 0.035187 | 10/10 | 25.41 |
| `N_source_ema - N_full` (downstream, informational) | +0.002515 ± 0.003254 | 6/10 | 0.77 |

At the re-derived base the target has a 3.2-step time constant and ends 0.01077
behind the online network — within a hair of the teacher-free rule — and on both
attribution instruments the full arm is *worse* than zero-decay, which is why
`ema_alignment_gap` comes back negative. Preserving the EMA time constant as a
fraction of the horizon was the wrong invariant: the run says the 0.86 lag at
base 0.996 is doing the work the mechanism needs, and §5 row 7's own reasoning —
that a target this far behind is "a frozen early snapshot, not a slowly moving
average" — is what the measurement contradicts.

The `source_ema` arm was declared in the amendment precisely so this would be
measurable rather than assumed. It did its job.

### 6.4 What is not done here

Nothing is rescored, no threshold moved, and the base is left where the
amendment put it. Two questions go to the next prospective amendment, and both
need a stream disjoint from 630000-630900:

1. whether to withdraw §5 row 7 and return the base to 0.996, for which the
   evidence is §6.3 above;
2. what replaces `ema_rank_gap`, now that it is known not to replicate. The
   alignment gap is the surviving attribution instrument, and one instrument is
   a thinner claim than the amendment assumed.

The [machine-readable result](results/byol-tier2-2026-09-17/byol.json) retains
all replicate values and the BN, norm, spread, residual, gradient and
target-lag diagnostics, beside its
[environment manifest](results/byol-tier2-2026-09-17/environment.json).
