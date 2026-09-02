# Recipe spec card: uda

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity; §6 is the
> predeclared evidence contract. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848) |
| Authors, year | Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le; 2019 / NeurIPS 2020 |
| DOI / arXiv | [arXiv:1904.12848](https://arxiv.org/abs/1904.12848); [10.48550/arXiv.1904.12848](https://doi.org/10.48550/arXiv.1904.12848) |
| Version used | arXiv v6, 2020-11-05. Core objective: §2.2 eq. (1); confidence masking and sharpening: §2.4; Training Signal Annealing (TSA): appendix A.1; image results: §4.2 / appendix B.2; TSA ablation: appendix B.1 table 8. |
| Reference implementation | [`google-research/uda`](https://github.com/google-research/uda) @ [`960684e363251772a5938451d4d2bc0f1da9e24b`](https://github.com/google-research/uda/tree/960684e363251772a5938451d4d2bc0f1da9e24b), especially `image/main.py`, `image/data.py`, `image/preprocess.py`, `image/utils.py`, and `image/scripts/run_cifar10_gpu.sh`. |
| Reference impl. runnable? | Not attempted. It targets a TensorFlow 1.x GPU/TPU image stack; this card relies on source inspection. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities `p(t | x)` plus the
  project-local treatment-specific outcome means. UDA changes how the treatment
  classifier uses rows without observed treatment labels: a detached weak-view
  distribution is temperature-sharpened and confidence-gated, then used as the
  target of a directed consistency loss on a stronger view. TSA separately
  suppresses already-easy labelled examples using a threshold that rises during
  training.
- **Claim:** UDA argues that high-quality, label-preserving augmentation makes
  consistency training substantially more effective than simple noise, with
  confidence masking and low-temperature targets as stabilisers. TSA is proposed
  to reduce overfitting when supervised data are scarce. Appendix B.1 table 8
  reports Yelp-5 error `50.81` without TSA versus `41.35` with the exponential
  schedule; appendix B.2 table 9 reports CIFAR-10 error `4.32 ± 0.08` for the
  4,000-label RandAugment setting.
- **Nearest shipped baseline:** `fixmatch`. Both use a weak target view, a strong
  prediction view, and confidence masking. UDA keeps a **soft** target and
  sharpens it with temperature `0.4`; FixMatch takes an argmax hard target. UDA
  also supplies TSA, which FixMatch explicitly does not use.
- **Variant selected here:** low-label UDA with exponential TSA, confidence
  threshold `0.8`, and target temperature `0.4`. The released CIFAR shell script
  does not enable TSA; selecting it here is an explicit paper-supported variant
  choice because the backlog asks the UDA acceptance work to isolate TSA and
  sharpening.
- **Not claimed:** no published image result is reproduced; no causal
  identification follows from consistency training; no claim is made that TSA
  or sharpening must help every tabular problem; BERT transfer, back-translation,
  out-of-domain filtering, ImageNet scaling, and entropy minimisation are out of
  scope.

## 3. Equations and mapping

### 3.1 As published

For labelled distribution `p_L(x)`, unlabelled distribution `p_U(x)`, model
`p_θ(y|x)`, perfect target `f*(x)`, augmentation distribution `q(x_hat|x)`, and a
fixed copy `θ_tilde` of the current parameters, UDA minimises (§2.2 eq. (1))

$$
\mathcal J(\theta)
= \mathbb E_{x_1\sim p_L}[-\log p_\theta(f^*(x_1)\mid x_1)]
+ \lambda\mathbb E_{x_2\sim p_U}\mathbb E_{\hat x\sim q(\hat x\mid x_2)}
  [\mathrm{CE}(p_{\tilde\theta}(y\mid x_2)
  \|p_\theta(y\mid \hat x))].
\tag{1}
$$

`θ_tilde` is the current model under stop-gradient, not an EMA teacher. The
paper uses `lambda=1` for most experiments.

Section 2.4 masks the unsupervised term by the **unsharpened** weak confidence:

$$
\frac{1}{|B|}\sum_{x\in B}
I\!\left(\max_{y'}p_{\tilde\theta}(y'\mid x)>\beta\right)
\mathrm{CE}\!\left(p^{sharp}_{\tilde\theta}(y\mid x)
\|p_\theta(y\mid \hat x)\right),
$$

with

$$
p^{sharp}_{\tilde\theta}(y\mid x)
=\frac{\exp(z_y/\tau)}{\sum_{y'}\exp(z_{y'}/\tau)}.
$$

For CIFAR-10 and SVHN, `beta=0.8`; for CIFAR-10, SVHN, and ImageNet,
`tau=0.4`. The pinned image implementation gates `softmax(ori_logits)` before
applying `ori_logits / tau` to the target. It zeros rejected rows and then takes
`reduce_mean`, so rejected rows remain in the unsupervised denominator.

Appendix A.1 defines TSA. A labelled row is suppressed when the model's
probability of its **true class** exceeds

$$
\eta_t=\alpha_t\left(1-\frac{1}{K}\right)+\frac{1}{K},
$$

where, for total steps `T`,

$$
\alpha_t^{log}=1-e^{-5t/T},\quad
\alpha_t^{linear}=t/T,\quad
\alpha_t^{exp}=e^{5(t/T-1)}.
$$

This card selects `exp`. The reference implementation divides the masked
supervised loss by `max(number retained, 1)`. That denominator is deliberately
*different* from the UDA consistency denominator.

### 3.2 Mapping to xty2

One `joint_fit` gradient stage. Paper classes map to treatment levels, so the
classifier is `T_GIVEN_X` from `CategoricalPropensity`. The ordinary project-local
outcome likelihood and exact missing-treatment marginal remain present.

Two reversible objective additions and one shared value object are required:

1. `ConfidenceMaskedConsistencyLoss`: directed KL between two `T_GIVEN_X`
   realisations, with detached weak target, target temperature, confidence gate
   computed before sharpening, and the source full-eligible-row denominator.
   Existing `ConsistencyLoss` has directed KL and stop-gradient but no gate or
   target temperature; `PseudoLabelTreatmentNLL` intentionally hardens by
   argmax.
2. `TrainingSignalAnnealedTreatmentNLL`: observed-treatment cross-entropy with a
   scheduled correct-class ceiling. Its source arithmetic dynamically retains a
   subset of the declared `t_observed` rows and returns the mean over that
   retained subset. The mixer reduction remains the existing closed value
   `mean`; the dynamic denominator is objective arithmetic and must be printed
   by `plan_details()`, not invented as a new mixer reduction.
3. `UDAConfidenceThresholds`: one immutable policy holding the fixed
   unsupervised gate and the exponential TSA schedule. Both objectives bind the
   same policy to `losses.confidence_threshold`, so the card's one canonical
   key cannot hide either of UDA's two confidence rules. The policy carries
   `scale=5` and `T=3000` explicitly rather than leaving source arithmetic as a
   silent objective default.

No new port, global row-population token, executor, artifact, stage type, or
state lifecycle is requested.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_θ(y|x)` | class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| labelled `x_1, f*(x_1)` | observed class | `T_GIVEN_X @ weak_x` | `TrainingSignalAnnealedTreatmentNLL`, rows `t_observed` |
| `η_t` | TSA ceiling | — | `UDAConfidenceThresholds.tsa_ceiling()` inside that objective |
| weak/original `x_2` | detached consistency target | `T_GIVEN_X @ weak_x` | left side of `ConfidenceMaskedConsistencyLoss` |
| `x_hat` | stronger augmented row | `T_GIVEN_X @ strong_x` | right side of `ConfidenceMaskedConsistencyLoss` |
| `θ_tilde` | fixed current target params | `T_GIVEN_X @ weak_x, params=student` | `stop_grad="target"`; no training teacher |
| `tau=0.4` | target softmax temperature | — | `target_temperature=0.4` |
| `beta=0.8`, exponential TSA | the two confidence rules | — | shared `thresholds=UDA_THRESHOLDS` |
| eq. (1) consistency | soft directed KL / cross-entropy | weak + strong `T_GIVEN_X` | `ConfidenceMaskedConsistencyLoss(divergence="kl")`, rows `t_missing` |
| `lambda=1` | unsupervised weight | — | `Weighted(..., weight=1.0)` |
| ordinary image augmentation | weak perturbation | — | `ViewSpec("weak_x", FeatureMask(p=0.1))` |
| RandAugment + Cutout | strong perturbation | — | `ViewSpec("strong_x", FeatureMask(p=0.1), FeatureMask(p=0.1))` (deviation 2) |
| `B=64`, `mu=7` | batch composition | — | `QuotaSampler`: 64 `t_observed`, 448 `t_missing` |
| evaluation moving average | reported model | — | `TeacherSpec(decay=0.9999, role="evaluation", ema_applies_to_buffers=True)` |
| — | project-local outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — | project-local exact marginalisation | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

Load-bearing arithmetic:

- the UDA gate reads untempered weak confidence;
- only the weak target branch is detached;
- UDA consistency divides by all eligible missing-treatment rows after masking;
- TSA averages only the dynamically retained observed-treatment rows.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.tsa_observed_treatment_nll: none
    joint_fit.uda_consistency: p(t|x) @ view=weak_x params=student
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # paper §2.2 / ref impl stop_gradient
  gradient_clipping: none                     # ref impl image/main.py
  marginal_nll_grad_path: both                # project-local P5 choice

teacher:
  ema_decay: 0.9999                           # ref impl image/main.py
  ema_applies_to_buffers: true                # ref impl utils.get_all_variable includes BN moving stats
  teacher_in_train_mode: false                # evaluation role only
  teacher_requires_grad: false

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.tsa_observed_treatment_nll: mean       # objective value is retained-row mean; plan_details declares dynamic denominator
    joint_fit.uda_consistency: mean                  # rejected rows are zero but remain in denominator
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.tsa_observed_treatment_nll: t_observed
    joint_fit.uda_consistency: t_missing
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.tsa_observed_treatment_nll: 1.0
    joint_fit.uda_consistency: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.tsa_observed_treatment_nll: constant 1.0 # the TSA ceiling is gate arithmetic, represented by losses.confidence_threshold and plan_details
    joint_fit.uda_consistency: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 optimiser steps
  temperature: 0.4
  sharpening: softmax_temperature
  confidence_threshold: uda(unsupervised=0.8, tsa=exp_schedule(scale=5, steps=3000))

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True) # ref impl image/main.py
  lr: 0.03                                    # ref impl / CIFAR GPU script
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))
  weight_decay: 0.0005 (all trainable components; all parameters) # ref impl utils.decay_weights
  batch_size: 512
  labelled_unlabelled_ratio: 7.0
  total_steps_or_epochs: 3000 optimiser steps # source CIFAR script uses 500000; deviation 3

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
  standardisation: x: none fitted on 'train'
  outcome_scaling: y: zscore fitted on 'train'
  treatment_encoding: n/a # XTYBatch contract supplies integer classes 0..K-1; propensity emits K probabilities
  split_protocol: one fixed project-local DGP, split train/test by the section 6.1 fixture; no CIFAR/SVHN protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply UDA to `p(t|x)` and compose it with the causal outcome stack. | The paper studies classification; xty2 asks whether the same missing-label mechanism helps treatment prediction without replacing the reviewed outcome likelihood. | Published image error is not comparable; §6 uses paired project-local arms. |
| 2 | `judgement` | — | Replace crop/flip and RandAugment+Cutout with a 10% weak feature mask and a strong view that composes two independent 10% masks. | Image transforms have no tabular semantics. A reviewed 50% second mask flipped the fixture's Bayes label on 15.8% of rows, violating §6's data-policy guardrail; 10% retains a strictly stronger composed view while holding the seed-locked flip rate below 5%. | Weaker perturbation diversity than the initial draft; §6 measures both flip rates before training. |
| 3 | `judgement` | — | Train 3,000 optimiser steps rather than the CIFAR script's 500,000. | This is a small synthetic mechanism test. | May understate long-horizon effects; every arm shares the budget. |
| 4 | `judgement` | — | Use the reviewed MLP/TARNet/propensity stack rather than Wide-ResNet. | Architecture swaps are orthogonal components in xty2. | No architecture-dependent published accuracy comparison is valid. |
| 5 | `judgement` | — | Run UDA consistency only on `t_missing`; the released image unsupervised stream is built from the full training population. | `t_missing` is the statistical unlabelled population in this adaptation and avoids deliberately reusing observed identities as unlabelled rows. | Slightly less consistency data than source-style reuse; direction unknown. |
| 6 | `judgement` | — | Use exactly 64 observed treatments. | This is the existing low-label fixture and makes TSA relevant. | Stronger overfitting pressure may increase TSA's effect. |
| 7 | `judgement` | — | Enable exponential TSA although the released CIFAR shell script leaves `--tsa` unset. | TSA is a paper mechanic aimed at scarce-label regimes, and the backlog explicitly asks its acceptance test to isolate TSA. | §6 includes a no-TSA arm so the effect remains attributable. |
| 8 | `judgement` | — | Retain the project-local missing-treatment marginal term at weight `0.5`. | UDA is being tested as an addition to the causal stack. | Can help independently; every arm holds it fixed. |

### 5.1 Framework additions made for this card

All three additions are fidelity-bearing but reversible. None adds load-bearing
framework vocabulary, so `DESIGN.md` §11.2 does not require a named second
consumer.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `UDAConfidenceThresholds` | fidelity-bearing, reversible | both UDA objectives | not required | UDA has a fixed confidence gate and a scheduled true-class ceiling under one canonical card key. One shared value makes disagreement impossible and prints `beta`, schedule family, scale, and horizon together. |
| `ConfidenceMaskedConsistencyLoss` | fidelity-bearing, reversible | UDA | not required | Existing consistency lacks UDA's target temperature and gate; FixMatch's objective hardens the target. |
| `TrainingSignalAnnealedTreatmentNLL` | fidelity-bearing, reversible | UDA | not required | Static row populations and scalar objective schedules cannot express a model-dependent true-class gate with the source retained-row denominator. |

Implementation must amend this card and stop again if TSA's dynamic retained-row
count cannot be represented without changing the generic `LossTerm` contract;
that would be framework vocabulary, not a recipe-local detail.

## 6. Reproduction target

The primary attribution is full UDA versus the same fit with the UDA consistency
weight set to zero, with TSA retained in both. Two additional arms isolate target
sharpening and TSA.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in §6.1
  variant: four paired arms — full UDA; no-consistency (lambda_uda=0); no-sharpening (tau=1); no-TSA (ordinary observed-treatment NLL); all other mechanics paired
  split: 1024 train rows with exactly 64 observed treatments; 2048 held-out rows with every treatment observed
  metric: held-out treatment NLL for student and evaluation EMA; held-out outcome NLL guardrail; UDA gate coverage/confidence and target entropy; TSA retained fraction and ceiling
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: full/no-consistency held-out treatment-NLL ratio < 1.0 in mean by at least one stderr for student and EMA; outcome NLL <= 1.05x no-consistency; tau=0.4 target entropy < tau=1 on matched weak logits; tau must not change gate membership on fixed logits; report the learned TSA retained fraction without imposing a direction; report sharpening and TSA effects without retroactively choosing their sign
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

For replicate `r=0..9`, use base seed `s_r=94000+100r`; training generation
uses `s_r+1`, held-out generation `s_r+2`, parameter initialisation `s_r+6`, and
stage seed `s_r+10000`.

```text
cluster c = 1[u_c < 0.5]
x[0:4]   = 0.45 * (2c - 1) + 0.6 epsilon[0:4]
x[4:6]   = epsilon[4:6]
p(t=1|c) = 0.02 + 0.96c
t         = 1[u_t < p(t=1|c)]
baseline  = 0.5x0 - 0.3x1 + 0.2(x4^2 - 1)
effect    = 1 + 0.5 tanh(x2)
y         = baseline + t * effect + 0.5 epsilon_y
```

Exactly 64 training treatments are observed under a seeded MCAR permutation;
all held-out treatments and every outcome are observed. Assert that both
training treatment levels appear in the observed subset; fail rather than
reseeding. Fit outcome scaling on the complete training population. Every arm
uses the same quota stream: 64 observed and 448 missing rows per step.

Before training, report weak/strong Bayes-label flip rate, treatment prevalence
by split/missingness, the untrained weak-confidence distribution, and the initial
TSA retained fraction.

### 6.2 Predeclared evidence

**Tier 0 (invariants).**

1. UDA consistency matches a direct tensor calculation of
   `KL(softmax(z_weak/tau) || softmax(z_strong))` with the weak branch detached.
2. The confidence gate is computed from `softmax(z_weak)` before temperature;
   changing `tau` on fixed logits changes target entropy but not membership.
3. Rejected UDA rows contribute zero while remaining in the denominator; an
   all-rejected eligible set returns zero.
4. Gradient reaches strong logits and not weak target logits or the gate.
5. `tau=1` exactly recovers the ordinary weak softmax; `tau<1` cannot increase
   entropy and strictly decreases it for non-uniform rows.
6. TSA gates on probability assigned to the true observed treatment, not argmax.
7. The exponential ceiling exactly matches
   `exp(5*(step/T-1))*(1-1/K)+1/K`, including its small step-zero offset, and is
   monotone non-decreasing to `1`.
8. TSA averages over retained rows, clamped at one retained count; an all-dropped
   batch returns zero. The card still binds mixer reduction `mean`.
9. The UDA and TSA denominator conventions are distinct and plan-visible.
10. The evaluation EMA is not read by either training objective.
11. `plan.hyperparameters` matches every non-`n/a` §4 key; `plan_details()`
    prints gate source, target temperature, both denominator conventions, TSA
    schedule, and strict comparison operators.
12. The recipe compiles as one ordinary gradient stage and introduces no state,
    artifact, port, executor, or global row-population token.

**Tier 1 (one-seed smoke and mechanism arms).**

1. Run full, no-consistency, no-sharpening, and no-TSA from identical initial
   parameters and paired batch/view RNG; all losses remain finite and the full
   arm beats marginal-frequency treatment NLL.
2. Report UDA coverage, accepted confidence, weak target entropy, consistency
   loss, and held-out treatment NLL. Do not require raw consistency loss to fall
   monotonically while its gate opens.
3. No-consistency changes only the UDA weight to zero. This is the primary
   attribution arm.
4. No-sharpening changes only `tau` to `1`; step-zero gate membership must be
   identical to full.
5. No-TSA replaces only the TSA objective with ordinary
   `ObservedTreatmentNLL` under the same weak view.
6. Diagnostic only: run an always-accept UDA gate to expose low-confidence
   target instability. No performance direction is asserted.
7. Diagnostic only: replace strong augmentation with an independent weak draw.
   Compare it with the no-consistency arm as sensitivity to augmentation
   strength. A gain still has a weak-to-weak consistency signal and cannot be
   attributed to the marginal term, which is fixed in both arms.
8. Report view label-flip rates beside every result. Require the composed strong
   view to flip more labels than weak but at most 5% on the seed-locked fixture;
   a larger rate is a data-policy failure, not evidence against UDA.

**Tier 2.** Run the four predeclared arms over all ten replicates. Only the
primary full-versus-no-consistency NLL target plus outcome guardrail determines
`reproduced` versus `deviating`; sharpening and TSA signs remain reported
mechanism effects.

**What has run.** Tier 0: `tests/invariants/test_uda.py` (`32 passed`). Tier 1:
`tests/smoke/test_uda.py` (`8 passed`), including all four 3,000-step paired
arms and the always-accept and weak-to-weak diagnostics. On seed 94000, held-out
treatment NLL (student / EMA) was full `0.332 / 0.541`, no-consistency
`0.314 / 0.573`, no-sharpening `0.295 / 0.604`, and no-TSA `0.408 / 0.481`,
against the observed-frequency `0.705`. Full-arm gate coverage went `0 → 0.886`,
terminal accepted confidence was `0.960`, target entropy went `0.693 → 0.059`,
and consistency loss went `0 → 0.167`; terminal TSA retention was `1.0` at a
`0.9992` ceiling. Weak/strong label-flip rates were `2.1% / 4.3%`, inside the
predeclared guardrail. All recorded values were finite.

**Tier 2, ten replicates.** The primary attribution holds in both directions
the tolerance names. Against the `lambda_uda = 0` arm, UDA's held-out treatment
NLL ratio was `0.946 +/- 0.020` on the evaluation EMA and `0.952 +/- 0.041` on
the student, each inside `1.0` by more than one standard error; UDA led the EMA
in eight replicates of ten and the student in seven of ten. Absolute EMA NLL was
`0.593 +/- 0.011` against `0.628 +/- 0.010`, and student `0.323 +/- 0.006`
against `0.345 +/- 0.015`, on a marginal-frequency baseline of
`0.696 +/- 0.001`. The outcome guardrail was `0.9996 +/- 0.0002`, so the
consistency term did not buy treatment accuracy out of the outcome likelihood.
Both `tau` clauses held on every replicate: at step 0, where the arms share weak
logits, `tau = 0.4` strictly lowered target entropy against `tau = 1`, and gate
coverage and accepted confidence were bit-identical across the two, which is the
gate reading untempered probabilities.

Tier 1's seed-94000 student direction did not survive the wider sample — it is
one replicate of ten, and the two arms' students are `1.029` there against a
`0.952` mean — so §6.2's refusal to read a single seed's sign was the right
call, in the direction that happened to favour the method.

**Mechanism arms, sign reported not chosen.** Against the same
no-consistency denominator, dropping sharpening (`tau = 1`) gave
`0.982 +/- 0.016` EMA and `0.919 +/- 0.035` student, so on this fixture
sharpening is not what earns the gain and the student arm is nominally better
without it. Replacing TSA with the ordinary observed-treatment NLL gave
`0.761 +/- 0.011` EMA and `1.070 +/- 0.047` student — the largest effect
measured here, and one that points in opposite directions for the two parameter
sets. Neither sign was predeclared and neither is claimed. TSA retained
`0.909 +/- 0.031` of labelled rows at step 0 and `0.988 +/- 0.006` at the end,
under a terminal ceiling of `0.9992`: on a 64-label, `K = 2` fixture the
exponential schedule suppresses very little, which bounds how much appendix
A.1's mechanic can be doing.

**One fixture broke §6.2's view guardrail.** Across the ten fixtures the
composed strong view flipped `4.79% +/- 0.16%` of Bayes labels against the weak
view's `2.60% +/- 0.17%`, strong above weak everywhere as required — but seed
`94100` flipped `5.96%`, over the 5% ceiling rule 8 sets. The card calls a
breach "a data-policy failure, not evidence against UDA", and §6.2 names the
primary NLL target and outcome guardrail as the only things that set status, so
this is reported here rather than gating the result; the Tier 2 metric is
informational for that reason. It is a live question for review whether rule 8's
ceiling should bind all ten fixtures rather than Tier 1's one, which would make
deviation 2's `0.1` second mask too strong for this fixture family at its
upper tail.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-02 | `a808089` | student_treatment_NLL_ratio<br>ema_treatment_NLL_ratio<br>held_out_outcome_NLL_ratio<br>sharpening_lowers_target_entropy<br>sharpening_leaves_the_gate_unchanged | 0.952354 +/- 0.0411<br>0.946062 +/- 0.0196<br>0.999556 +/- 0.000187<br>1 +/- 0<br>1 +/- 0 | yes |

## 7. Unknowns

| Unspecified or source-dependent choice | Our choice | Basis |
|---|---|---|
| Which TSA schedule belongs to this low-label variant. | `exp_schedule`. | Appendix A.1 recommends it under stronger overfitting pressure; appendix B.1 table 8 gives the best Yelp-5 result. |
| Paper prose describes the start as `1/K`, but the exponential formula starts slightly above it. | Follow the formula/code exactly. | Appendix A.1 and pinned `get_tsa_threshold`. |
| Does the confidence gate read sharpened or ordinary weak probabilities? | Ordinary weak probabilities. | Paper §2.4 and pinned `image/main.py`. |
| Is `θ_tilde` an EMA teacher? | No; current student under stop-gradient. | Paper eq. (1) and pinned code. EMA is separate evaluation machinery. |
| Do rejected UDA rows leave the denominator? | No; they remain as zeros in the full eligible-row mean. | Pinned image implementation masks then `reduce_mean`s. |
| Does TSA use that same denominator? | No; mean over retained labelled rows, clamped to one. | Pinned `anneal_sup_loss`. |
| Should consistency also use observed-treatment rows because source image preprocessing uses the full train pool? | No; `t_missing` only. | Deliberate adaptation, deviation 5. |
| Exact tabular weak/strong transforms. | Reuse FixMatch-family feature masks. | Controlled comparison; no paper-prescribed tabular transform exists. |
| Weight-decay reach. | All trainable parameters, including biases. | Pinned `utils.decay_weights` sums all trainable variables. |
| EMA buffer treatment when the adapted graph has no BN. | Keep `ema_applies_to_buffers=true`; operationally inert here. | Pinned `utils.get_all_variable` explicitly includes BN moving statistics. |
| Confidence comparison operator. | Strict `> 0.8`. | Paper prose and pinned `tf.greater`. |
| TSA comparison operator. | Suppress only when true-class probability is strict `>` the ceiling. | Appendix A.1 and pinned `tf.greater`. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | ChatGPT | 2026-09-01 |
| Plan diffed against §3.2 and §4 | ChatGPT | 2026-09-01 |
