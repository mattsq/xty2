# Recipe spec card: tarnet

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Estimating individual treatment effect: generalization bounds and algorithms](https://proceedings.mlr.press/v70/shalit17a.html) |
| Authors, year | Uri Shalit, Fredrik D. Johansson, David Sontag, 2017 |
| DOI / arXiv | [arXiv:1606.03976v5](https://arxiv.org/abs/1606.03976v5); [10.48550/arXiv.1606.03976](https://doi.org/10.48550/arXiv.1606.03976); PMLR 70:3076–3085 |
| Version used | ICML/PMLR version; arXiv v5, 2017-05-16. This is the version that names the `alpha = 0` model TARNet. |
| Reference implementation | [`clinicalml/cfrnet` @ `0377b0c8c822845d335540d4be6003024a65d3c8`](https://github.com/clinicalml/cfrnet/tree/0377b0c8c822845d335540d4be6003024a65d3c8), the last commit before the ICML publication |
| Reference impl. runnable? | Not attempted. It targets Python 2, TensorFlow 0.12.0-rc1 and NumPy 1.11.3. |

## 2. Estimand and claim

- **Estimand:** treatment-specific means `m_k(x) = E[Y(k) | X=x]`; pairwise contrasts are CATEs under the usual causal assumptions.
- **Method claim:** TARNet shares an encoder across treatment-specific outcome heads. The published binary model is the `alpha = 0` CFR ablation.
- **xty2 extension:** a categorical propensity and exact missing-treatment likelihood make the model trainable when `t` is absent. The paper does not claim this extension, categorical `K`, or identification without consistency, overlap, and exchangeability.

## 3. Equations and mapping

### 3.1 As published

The paper defines the conditional effect and its estimate as

$$
\tau(x) = m_1(x) - m_0(x), \qquad
\hat\tau_f(x) = f(x, 1) - f(x, 0),
$$

and evaluates it with Eq. (1):

$$
\epsilon_{\mathrm{PEHE}}(f)
= \int_{\mathcal X}
  \left(\hat\tau_f(x) - \tau(x)\right)^2 p(x)\,dx.
$$

TARNet is Eq. (3) with `alpha = 0`:

$$
\min_{h,\Phi:\lVert\Phi\rVert=1}
\frac{1}{n}\sum_{i=1}^{n}
w_i\,L\!\left(h(\Phi(x_i),t_i),y_i\right)
+ \lambda R(h),
\qquad
w_i = \frac{t_i}{2u} + \frac{1-t_i}{2(1-u)},
\quad
u = \frac{1}{n}\sum_i t_i.
$$

For IHDP, `L` is squared error. The network uses a shared `Phi`, followed by
separate hypotheses `h_0` and `h_1`; a row updates only the head selected by its
observed treatment (paper section 4 and Figure 1).

### 3.2 Mapping to xty2

xty2 keeps the shared encoder and candidate-treatment heads, represents each head as a unit-scale Gaussian, and adds propensity, observed-treatment, and missing-treatment objectives. The stage is single-pass and has no views, teachers, or posterior artifact.

| Paper / P5 symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `x` | Raw covariates | `X_RAW` | virtual source node |
| `Phi(x)` | Shared representation, row-wise L2 normalised | `X_REPR` | `mlp_encoder` |
| `{h_k(Phi(x))}_{k=0}^{K-1}` | Treatment-specific outcome means | `Y_GIVEN_XT` | `tarnet_head` |
| `p_phi(y | x,t)` | Fixed-scale Gaussian around the selected `h_k` | `Y_GIVEN_XT` | `tarnet_head`, consumed by `ObservedOutcomeNLL` |
| `p_theta(t | x)` | Categorical treatment distribution | `T_GIVEN_X` | `categorical_propensity`, consumed by `ObservedTreatmentNLL` |
| `L_marg` | Exact observed-data likelihood for a missing treatment | `Y_GIVEN_XT`, `T_GIVEN_X` | `MissingTreatmentMarginalNLL(grad_path="both")` |
| `w_i` | Treatment-group sample weight on complete cases | n/a (batch field) | `XTYBatch.weight`, applied inside `ObservedOutcomeNLL` before `population` reduction |
| `L_P5` | Single weighted objective mix | both predicted ports | stage `joint_fit` with three `Weighted` objectives |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none  # paper Algorithm 1; P5 trains every required port
  detached_targets: n/a                       # no consistency target
  gradient_clipping: none                     # ref impl cfr_net_train.py:294-298 leaves clipping commented out
  marginal_nll_grad_path: both                # xty2 P5 choice; not present in the paper

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
    joint_fit.observed_outcome_nll: 1.0        # paper Eq. (3), after w_i is applied per row
    joint_fit.observed_treatment_nll: 1.0      # xty2 P5 joint-likelihood choice
    joint_fit.missing_treatment_marginal_nll: 0.5  # provisional xty2 P5 choice
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a

optimisation:
  optimiser: adam(betas=(0.9, 0.999), eps=1e-8)  # paper section 5; TF defaults in pinned ref impl
  lr: 0.001                                      # ref impl configs/example_ihdp.txt
  lr_schedule: staircase 1.0 * 0.97^floor(step/100)  # ref impl cfr_net_train.py:280-282
  weight_decay: 0.0001 (components tarnet_head only; norm and bias exempt)  # ref impl cfr/cfr_net.py:259-261, 305-306
  batch_size: 100                                # ref impl configs/example_ihdp.txt; UniformSampler
  labelled_unlabelled_ratio: n/a                 # UniformSampler enforces no quota; TARNet has no unlabelled stream
  total_steps_or_epochs: 3000 optimiser steps    # ref impl configs/example_ihdp.txt

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                 # paper section 5, IHDP
    tarnet_head: K independent heads, each [100, 100, 100]  # paper section 5 generalised from K=2
    categorical_propensity: linear X_REPR -> K   # provisional xty2 P5 choice
  activation:
    mlp_encoder: elu                             # paper section 5
    tarnet_head: elu                             # paper section 5
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: row_l2                         # final representation; no batch norm; ref impl example config
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0                             # ref impl keep probability 1.0
    tarnet_head: 0.0                             # ref impl keep probability 1.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0  # pinned ref impl cfr/cfr_net.py
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'fit'      # IHDP archive x is passed through unchanged by ref loader
  outcome_scaling: y: none fitted on 'fit'      # ref loader applies none
  treatment_encoding: n/a                       # XTYBatch contract supplies integer classes 0..K-1
  split_protocol: the archive's own realisation, fit/validation split 90/10 by seeded permutation as the reference loader does; training rows are assignment 'fit'
  missingness_mechanism: observed: the dataset's own t_observed mask, unchanged  # TARNet's data arrives labelled
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Add `categorical_propensity`, `ObservedTreatmentNLL` and `MissingTreatmentMarginalNLL`. | P5 is intended to prove exact treatment marginalisation through the Phase-A stack; published TARNet has no such terms. | The marginal term is inactive on fully observed IHDP, but the propensity loss still changes the shared encoder. Direction is unknown. |
| 2 | `judgement` | — | Generalise two treatment heads to `K` categorical heads. | xty2 v1 is categorical-treatment-first and the candidate-treatment contract must work for `K != B`. | None when the IHDP benchmark runs with `K=2`; no published target exists for `K>2`. |
| 3 | `judgement` | — | Represent each deterministic squared-error head as a unit-scale Gaussian and train by NLL. | `Y_GIVEN_XT` must satisfy the distribution protocol so the same marginal objective can later consume a flow head unchanged. | For complete cases the NLL differs from squared error only by a positive scale and constant, so it has the same optimum. It changes the scale of the added mixture likelihood. |
| 4 | `judgement` | — | Simulate 50% treatment missing completely at random in Tier 1. | This is the P5 acceptance condition in `PLAN.md` and `FIDELITY.md`, not part of the paper. | Not applicable to the fully observed published target; load-bearing for the smoke comparison. |
| 5 | `withdrawn` | — | ~~Declare the split, standardisation and missingness policy on this card and enforce it in the Tier 1 fixture and the P12 runner, rather than in the recipe.~~ **Withdrawn.** The recipe declares a `DataSpec`, `optimisation.batch_size` binds `100`, and the three `data.*` keys carry values in `plan.hyperparameters`. | The original entry was correct and is now paid: xty2 has a loader. What it bought is the guarantee this row said was missing — the standardisation is fitted on the declared `fit` assignment by the compiled program, and `TrainingPopulation.fitted_on_row_ids` is checked against that assignment at run time, so a runner that fitted it on the wrong split fails rather than passing silently. The 50% Tier 1 MCAR remains the *fixture's*, which is deviation 4's business and not a property of TARNet, so the recipe declares `mechanism: observed`. | The section 6 result below was measured under the pre-loader batch stream and is **invalidated** by this change, not merely re-labelled: the recipe now owns the batch size and its sampler seed is derived from the stage seed, so the stream moved. The sampling *scheme* is identical — one fresh permutation per step, first 100 rows, asserted against the old helper in `tests/invariants/test_loading.py` — so the number should move within its existing error bars, but that is a prediction and the nightly run is what settles it. |

### 5.1 Framework additions made for this card

`tarnet` introduced the declarative data boundary (`DataSpec`, `Dataset`, and
`TrainingPopulation`) and `UniformSampler`. Runtime checks bind preprocessing to
the declared fit rows and the batch stream to the recipe. **Named second
consumer:** VIME, whose mask/impute statistics must be fitted on the declared
training population rather than the current batch, checked the boundary's row
identity and fitted-statistic shape. `UniformSampler` is reversible policy, not
load-bearing vocabulary.

### Tier 2 outcome

On 2026-08-27, commit `1a10fb039e5f` produced a `deviating` result: This evaluates the implemented TARNet extension against the IHDP within-sample estimand, with the paper's target retained unchanged. The pinned reference repository ships 100 of the declared 1,000 IHDP realisations. The reviewed card also requests only ten seeds, so P12 runs realisations 1-10 with one deterministic fit each. That cannot establish the published 1,000-realisation centre and is recorded as deviating even if its ten-run mean lies inside the numeric tolerance. Failed target(s): sqrt_PEHE_in_sample was 1.46891 +/- 0.117 outcome units against 0.78 <= mean <= 0.98 outcome units.

## 6. Reproduction target

The nightly target is the published IHDP within-sample `sqrt(PEHE)` interval. The fixed contract below is authoritative.

```yaml
reproduction:
  dataset: IHDP
  variant: 1000 realisations, Hill (2011) / NPCI setting A; binary treatment; fully observed t
  split: 63/27/10 train/validation/test; within-sample metric over train plus validation
  metric: sqrt_PEHE_in_sample
  published: 0.88
  published_source: Shalit et al. (2017), Table 1, TARNet within-sample IHDP
  tolerance: 0.10
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Result ledger


| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-24 | `d060df351f2fe8bac6d951c3757506c684d8b408` | sqrt_PEHE_in_sample | 1.66989 +/- 0.178 outcome units | no |
| 2026-08-27 | `1a10fb039e5f` | sqrt_PEHE_in_sample | 1.46891 +/- 0.117 outcome units | no |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Propensity-head architecture | One linear map from `X_REPR` to `K` logits. | Smallest parameterisation that exercises `T_GIVEN_X`; the paper has no propensity head. |
| Marginal-likelihood weight and warm-up | Ramp from `0.0` to `0.5` over the first 1,000 of 3,000 steps. | Reviewed P5 choice. A ramp is required by the plan, but neither paper nor reference code governs it. |
| Outcome likelihood scale | Fixed Gaussian standard deviation `1.0`; it is not learned. | Preserves the paper's treatment-specific means and makes NLL equivalent to MSE up to scale and a constant on complete cases. |
| Categorical extension of the paper's binary sample weight | `w_i = 1 / (K p(t_i))` on observed-treatment rows. | Reduces to the paper's Eq. (3) weight for `K=2` and gives each observed treatment class equal total mass. |
| Exact interpretation of the paper's `+/-` values | Treat the published centre `0.88` as the target and report xty2 mean and standard error explicitly. | The table does not label the dispersion statistic in its caption. |
| Whether the published `0.88` remains attainable with the auxiliary propensity loss | Keep the target and predeclare the deviation rather than widen tolerance after seeing a run. | This is the honest falsifiable test of whether the P5 extension preserves the TARNet backbone. |
| How the card's ten seeds map onto its 1,000-realisation IHDP variant | P12 uses realisations 1–10 from the `clinicalml/cfrnet` archive pinned in §1, with deterministic seeds `130000 + r` for split, batches and fit. It records the outcome as `deviating` regardless of the ten-run centre because the pinned repository ships only 100 realisations and ten runs cannot establish the published 1,000-realisation mean. | Fixes the sampling decision before any P12 result is observed and prevents a small subset from being silently presented as the paper's experiment. |
| Executable ownership of split, standardisation and missingness declarations | The compiled program owns all three: `DataSpec` on the recipe, checked against the supplied `Dataset` at run time. | This row used to record a framework limitation living in section 7, which `FIDELITY.md` section 5.1 now forbids; deviation 5 held it, and deviation 5 is withdrawn. What is left here is genuinely an unknown the paper does not settle: the reference loader's 90/10 fit/validation split is transcribed from its code rather than from the paper. |
| How component-valued architecture keys appear in `plan.hyperparameters` | Aggregate each `architecture.*` key as a mapping keyed by component name. | The first card-to-plan diff must expose the three modules separately; a scalar or last-write-wins binding would satisfy presence while hiding a discrepancy. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | mattsq | 2026-08-23 |
| Plan diffed against §3.2 and §4 | | |
