# Recipe spec card: tarnet_extension

**Status:** `smoke-passing`

> **Agent route:** read §2–§5 to implement or audit fidelity; this extension
> has no published Tier 2 target.

## 1. Provenance

This is the xty2 missing-treatment extension formerly exposed as `tarnet`.
Its backbone comes from [Shalit et al. (2017)](https://proceedings.mlr.press/v70/shalit17a.html),
but its propensity and marginal-likelihood objectives are not part of TARNet.

## 2. Estimand and claim

The recipe estimates treatment-specific outcome means while additionally
learning `p(t|x)` so missing treatments can be marginalized exactly. It is an
xty2 method variant and carries no claim to reproduce the paper's 0.88 result.

## 3. Equations and mapping

It combines unit-scale Gaussian observed-outcome NLL, observed-treatment NLL,
and exact missing-treatment marginal NLL. `MLPEncoder` is shared by the
`TARNetHead` and `CategoricalPropensity` components.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: n/a
  gradient_clipping: none
  marginal_nll_grad_path: both

teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
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
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: elu
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: row_l2
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'fit'
  outcome_scaling: y: none fitted on 'fit'
  treatment_encoding: n/a
  split_protocol: the archive's own realisation, fit/validation split 70/30 by seeded permutation
  missingness_mechanism: observed: the dataset's own t_observed mask, unchanged
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What differs | Why | Expected effect |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Adds a categorical propensity head and observed-treatment NLL. | Required for missing-treatment marginalization. | Changes the shared representation even on complete data. |
| 2 | `judgement` | — | Adds exact missing-treatment marginal NLL, ramped to 0.5. | Core extension claim. | Active only where treatment is missing. |
| 3 | `judgement` | — | Uses unit-Gaussian NLL rather than MSE. | Common likelihood interface. | Outcome gradient is half MSE before mixing with other terms. |
| 4 | `judgement` | — | Generalizes binary heads to categorical `K`. | Common treatment contract. | None for `K=2`. |

### 5.1 Framework additions made for this card

None beyond the data boundary, sampler, and marginal-likelihood machinery
already documented by their original consumer cards.

## 6. Reproduction target

No published numeric target applies. Tier 1 compares exact marginalization with
the paired complete-case ablation on a controlled missing-treatment DGP.

## 7. Unknowns

The propensity architecture and marginal-loss schedule are xty2 choices, not
quantities specified by Shalit et al.

## 8. Review

Separated from the paper-faithful baseline at the user's request on 2026-09-01.
