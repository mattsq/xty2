# Recipe spec card: tarnet

**Status:** `smoke-passing`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

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

The paper is the authority for the estimand, TARNet architecture and reported
number. The pinned implementation is the authority where the paper gives only a
range or a qualitative description. The semi-supervised likelihood introduced
below is an xty2 P5 extension, not a claim about the published method.

## 2. Estimand and claim

- **Estimand:** For the paper's binary treatment,
  `tau(x) = E[Y(1) - Y(0) | X=x] = m_1(x) - m_0(x)`. The P5 component contract
  generalises this to `K` categorical treatments by estimating every
  `m_k(x) = E[Y(k) | X=x]`; effects are contrasts of candidate-treatment means.
- **Claim:** TARNet jointly learns a shared representation and treatment-specific
  outcome heads from factual outcomes. It is the `alpha = 0` ablation of CFR,
  without an IPM balance penalty. On IHDP, the paper reports within-sample
  `sqrt(PEHE) = 0.88 +/- 0.02` and out-of-sample `0.95 +/- 0.02` over 1,000
  outcome realisations (Table 1).
- **Not claimed:** The paper does not model `p(t | x)`, does not train when
  treatment is missing, does not give a categorical-`K` result, and does not
  claim that TARNet is balanced or that its effects are identified without
  consistency, overlap and strong ignorability. The P5 propensity head and
  exact missing-treatment likelihood are xty2 additions.

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

#### P5 semi-supervised extension

Let `m_i` indicate that treatment is observed. P5 fits the following three
terms in one stage. These equations are xty2's observed-data likelihood, not
equations from Shalit et al.:

$$
\mathcal L_y
= -\frac{1}{B}\sum_{i:m_i=1}
  w_i\log p_\phi(y_i\mid x_i,t_i),
$$

$$
\mathcal L_t
= -\frac{1}{B}\sum_{i:m_i=1}
  \log p_\theta(t_i\mid x_i),
$$

$$
\mathcal L_{\mathrm{marg}}
= -\frac{1}{B}\sum_{i:m_i=0}
  \log\sum_{k=0}^{K-1}
  p_\theta(t=k\mid x_i)\,
  p_\phi(y_i\mid x_i,t=k),
$$

with

$$
\mathcal L_{\mathrm{P5}}(s)
= \mathcal L_y + \mathcal L_t
+ \lambda_{\mathrm{marg}}(s)\mathcal L_{\mathrm{marg}},
\qquad
\lambda_{\mathrm{marg}}(s)
= 0.5\min\!\left(\frac{s}{1000},1\right).
$$

`population` reduction supplies the `1/B` factors. The marginal term has no
stop-gradient: it trains the encoder, outcome heads and propensity head. For
`K=2`, the paper's group weights are carried on observed rows by
`XTYBatch.weight`; for general `K`, use `w_i = 1 / (K p(t_i))`. A missing row's
arbitrary stored treatment value is never used to construct either weights or
candidate treatments.

#### Component and objective mapping

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

No view, posterior, balance loss, representation penalty, teacher, second
realisation or second stage belongs in P5.

## 4. Mechanics checklist

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
  batch_size: n/a                                # external BatchSource; ref impl uses 100
  labelled_unlabelled_ratio: n/a                 # no fixed quota; Tier 1 applies 50% MCAR at dataset level
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
  standardisation: n/a                          # caller-owned; IHDP archive x is passed through unchanged by ref loader
  outcome_scaling: n/a                          # caller-owned; ref loader applies none
  treatment_encoding: n/a                       # XTYBatch contract supplies integer classes 0..K-1
  split_protocol: n/a                           # Tier 1 fixture and P12 benchmark runner own their splits
  missingness_mechanism: n/a                    # Tier 1 fixture applies 50% treatment MCAR; paper has no missing t
```

The `n/a` entries under batching and data are a deliberate P5 boundary, not
missing research. `DESIGN.md` section 11 leaves the loader/sampler out until a
recipe needs an enforced per-batch quota. The gradient executor consumes a
caller-supplied `BatchSource`, so this recipe cannot honestly claim that it
enforces batch size, splitting or preprocessing. The Tier 1 fixture and the P12
benchmark runner must record those external mechanics alongside their results.

## 5. Deviations from the paper

| # | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|
| 1 | Add `categorical_propensity`, `ObservedTreatmentNLL` and `MissingTreatmentMarginalNLL`. | P5 is intended to prove exact treatment marginalisation through the Phase-A stack; published TARNet has no such terms. | The marginal term is inactive on fully observed IHDP, but the propensity loss still changes the shared encoder. Direction is unknown. |
| 2 | Generalise two treatment heads to `K` categorical heads. | xty2 v1 is categorical-treatment-first and the candidate-treatment contract must work for `K != B`. | None when the IHDP benchmark runs with `K=2`; no published target exists for `K>2`. |
| 3 | Represent each deterministic squared-error head as a unit-scale Gaussian and train by NLL. | `Y_GIVEN_XT` must satisfy the distribution protocol so the same marginal objective can later consume a flow head unchanged. | For complete cases the NLL differs from squared error only by a positive scale and constant, so it has the same optimum. It changes the scale of the added mixture likelihood. |
| 4 | Simulate 50% treatment missing completely at random in Tier 1. | This is the P5 acceptance condition in `PLAN.md` and `FIDELITY.md`, not part of the paper. | Not applicable to the fully observed published target; load-bearing for the smoke comparison. |

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
  seeds: 10
  report: mean_and_stderr
```

This target is queued for Tier 2; P5 does not claim it from a smoke fit. Because
deviation 1 remains active even when treatment is fully observed, a miss
outside tolerance must become `deviating` with an explanation unless the card is
amended and reviewed before the run.

### 6.1 Result ledger

| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

### 6.2 P5 single-stage acceptance

The first implementation is deliberately one graph and one stage:

1. The graph contains exactly `mlp_encoder` (`X_RAW -> X_REPR`),
   `tarnet_head` (`X_REPR -> Y_GIVEN_XT`) and `categorical_propensity`
   (`X_REPR -> T_GIVEN_X`). The compiled plan has one identity/student forward
   pass in that topological order, with the two heads branching from
   `mlp_encoder`.
2. Stage `joint_fit` has `rows="all"`, trains all three components for 3,000
   steps, and contains exactly the three objectives and row populations in
   section 4. No view, second realisation, stage transition or conditional is
   permitted.
3. `tarnet_head` passes the full candidate-treatment conformance suite for
   `log_prob`, `mean` and `sample` with `B != K`; `categorical_propensity`
   passes normalisation and observed/candidate `log_prob` checks.
4. On the Tier 1 analytic DGP, all four `FIDELITY.md` section 3 assertions pass:
   loss decreases; propensity beats the held-out marginal-frequency baseline;
   estimated treatment contrasts lie in the declared wide band around the
   analytic ATEs; and, with 50% treatment MCAR, the marginal recipe beats the
   otherwise identical complete-case ablation on held-out `sqrt(PEHE)`. The
   comparison uses the same data, initial parameters, optimiser steps and seeds;
   only `MissingTreatmentMarginalNLL` is removed from the ablation.
5. The card-to-plan test checks every non-`n/a` section 4 key against
   `plan.hyperparameters`. A mutation deleting one real binding from the recipe
   must fail and name the missing canonical key. Architecture mappings must be
   rendered component-by-component so the plan diff shows all three modules,
   rather than collapsing them to one opaque value.
6. Tier 0 and Tier 1, `ruff check .`, `ruff format --check .` and
   `mypy --strict` are green. The card may then move to `smoke-passing`; Tier 2
   remains queued.

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
| Executable ownership of split, standardisation and missingness declarations | Tier 1 fixture now; P12 benchmark runner later. They stay `n/a` in the compiled recipe until a real recipe requires an enforced loader policy. | `DESIGN.md` section 11 YAGNI ledger and the current `BatchSource` executor boundary. |
| How component-valued architecture keys appear in `plan.hyperparameters` | Aggregate each `architecture.*` key as a mapping keyed by component name. | The first card-to-plan diff must expose the three modules separately; a scalar or last-write-wins binding would satisfy presence while hiding a discrepancy. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | mattsq | 2026-08-23 |
| Plan diffed against §3.2 and §4 | | |
