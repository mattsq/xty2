# Recipe spec card: tarnet

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity; §6 only for
> benchmark work. The missing-treatment variant is a separate recipe,
> [`tarnet_extension`](tarnet_extension.md).

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Estimating individual treatment effect: generalization bounds and algorithms](https://proceedings.mlr.press/v70/shalit17a.html) |
| Authors, year | Uri Shalit, Fredrik D. Johansson, David Sontag, 2017 |
| Version used | ICML/PMLR version and supplement |
| Reference implementation | [`clinicalml/cfrnet` @ `0377b0c8c822845d335540d4be6003024a65d3c8`](https://github.com/clinicalml/cfrnet/tree/0377b0c8c822845d335540d4be6003024a65d3c8) |
| Reference impl. runnable? | Not attempted; it targets Python 2 and TensorFlow 0.12. |

## 2. Estimand and claim

- **Estimand:** `m_k(x) = E[Y(k) | X=x]`; for IHDP,
  `tau(x) = m_1(x) - m_0(x)`.
- **Method claim:** TARNet is the `alpha = 0` CFR model: a shared representation
  and treatment-specific outcome hypotheses trained only by weighted factual
  prediction loss.
- **Scope:** this recipe is the published fully observed-treatment baseline. It
  has no propensity model and makes no missing-treatment claim.

## 3. Equations and mapping

For continuous IHDP outcomes, TARNet minimizes

$$
\frac{1}{n}\sum_i w_i\left(h_{t_i}(\Phi(x_i))-y_i\right)^2
+\lambda R(h),\qquad
w_i=\frac{t_i}{2u}+\frac{1-t_i}{2(1-u)}.
$$

| Paper symbol | xty2 mapping |
|---|---|
| `Phi(x)` | `MLPEncoder`, `X_REPR`, row-wise L2 normalized |
| `h_0`, `h_1` | independent arms in `TARNetHead`, `Y_GIVEN_XT` |
| weighted factual squared error | `ObservedOutcomeMSE`, with `XTYBatch.weight` applied before population reduction |
| `lambda R(h)` | optimizer L2 weight decay on outcome-head matrices only |

The head emits a fixed-scale `GaussianOutcome` to satisfy the common outcome
distribution protocol, but TARNet trains its mean with exact MSE, not Gaussian
NLL.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_mse: none
  detached_targets: n/a
  gradient_clipping: none
  marginal_nll_grad_path: n/a

teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    joint_fit.observed_outcome_mse: population
  eligible_rows:
    joint_fit.observed_outcome_mse: t_observed
  weights:
    joint_fit.observed_outcome_mse: 1.0
  schedules:
    joint_fit.observed_outcome_mse: constant 1.0
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a

optimisation:
  optimiser: adam(betas=(0.9, 0.999), eps=1e-8)
  lr: 0.001
  lr_schedule: staircase 1.0 * 0.97^floor(step/100)
  weight_decay: 0.0001 (components tarnet_head only; norm and bias exempt)
  batch_size: 100
  labelled_unlabelled_ratio: n/a
  total_steps_or_epochs: 3000 optimiser steps

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]
    tarnet_head: K independent heads, each [100, 100, 100]
  activation:
    mlp_encoder: elu
    tarnet_head: elu
  normalisation:
    mlp_encoder: row_l2
    tarnet_head: none
  dropout:
    mlp_encoder: 0.0
    tarnet_head: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0

data:
  standardisation: x: none fitted on 'fit'
  outcome_scaling: y: none fitted on 'fit'
  treatment_encoding: n/a
  split_protocol: the archive's own realisation, fit/validation split 70/30 by seeded permutation as the reference loader does; training rows are assignment 'fit'
  missingness_mechanism: observed: the dataset's own t_observed mask, unchanged
```

The maximum budget is 3,000 steps. For the IHDP evidence contract, predictions
are retained every 200 steps and the minimum weighted validation MSE plus
hypothesis L2 penalty selects the reported checkpoint, as specified in the
paper supplement and pinned evaluator.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Generalise two heads to categorical `K`. | The common xty2 treatment contract is categorical. | None for binary IHDP. |
| 2 | `judgement` | — | Expose the deterministic means through a fixed-scale Gaussian distribution object. | Common outcome-port contract. | None: training uses exact MSE. |

### 5.1 Framework additions made for this card

`MinimumValidationSelection` adds periodic validation scoring while preserving
one continuous optimizer trajectory and restores the selected state before the
immutable checkpoint is emitted. The executor records the full search trace
but gives the checkpoint the selected step count and corresponding row
provenance. This is reusable by any source method whose evidence protocol
selects a student checkpoint by a scalar validation criterion.

### 5.2 Reference-code conflict

The paper specifies the control weight `1 / (2 * (1-u))`. The pinned code writes
`(1-t)/(2*1-p_t)`, which Python evaluates as `(1-t)/(2-p_t)`. The recipe follows
the published equation. An exact emulation of that apparent reference bug must
be labelled separately and cannot silently replace this baseline.

## 6. Reproduction target

```yaml
reproduction:
  dataset: IHDP
  variant: 1000 realisations, Hill (2011) / NPCI setting A; binary treatment; fully observed t
  split: 63/27/10 train/validation/test; within-sample metric over train plus validation
  metric: sqrt_PEHE_in_sample
  published: 0.88
  published_source: Shalit et al. (2017), Table 1, TARNet within-sample IHDP
  tolerance: 0.10
  seeds: 1000
  checkpoint_selection: minimum validation objective every 200 optimiser steps
  report: mean_and_stderr
```

The paper averages 1,000 realizations. The reference repository's GitHub demo
links a 100-realisation subset, while its README points to the full archives at
`https://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip` and
`https://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip`. The evidence run
uses and checksum-pins those full archives and executes all 1,000 realisations.

### 6.1 Result ledger

| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-24 | `d060df351f2fe8bac6d951c3757506c684d8b408` | sqrt_PEHE_in_sample | 1.66989 +/- 0.178 outcome units | no; superseded extension result |
| 2026-08-27 | `1a10fb039e5f` | sqrt_PEHE_in_sample | 1.46891 +/- 0.117 outcome units | no; superseded extension result |
| 2026-09-01 | `fcc14ebb8734` | sqrt_PEHE_in_sample | 0.767486 +/- 0.0353 outcome units | no |
| 2026-09-01 | `fcc14ebb8734` | sqrt_PEHE_in_sample | 0.800349 +/- 0.013 outcome units | yes |

## 7. Unknowns

| Unspecified or unavailable | Our choice | Basis |
|---|---|---|
| Hosting of the full 1,000-realisation archive | Download the full train/test archives linked by the pinned reference README and checksum-pin their extracted NPZ payloads. | This is the exact full-data route the authors provide for reproducing the paper rather than the 100-realisation GitHub demo. |
| Exact dispersion label in Table 1 | Report sample standard error explicitly. | Avoid inferring an unlabeled statistic. |
| Checkpoint grid | Every 200 steps, including the final 3,000-step state. | `pred_output_delay=200` in the pinned IHDP configuration. |

## 8. Review

| | Who | Date |
|---|---|---|
| Original card reviewed | mattsq | 2026-08-23 |
| Fidelity corrections requested | mattsq | 2026-09-01 |
