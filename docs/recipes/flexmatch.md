# Recipe spec card: flexmatch

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling](https://arxiv.org/abs/2110.08263) |
| Authors, year | Bowen Zhang, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, Takahiro Shinozaki; 2021 |
| DOI / arXiv | [arXiv:2110.08263](https://arxiv.org/abs/2110.08263); NeurIPS 2021 |
| Version used | The ar5iv rendering of arXiv:2110.08263, fetched 2026-08-27. The rendering does not carry a version label and one was **not** verified, so this is recorded as what it is rather than as "v3". §2 defines the background and eqs. (1)–(4); §3.1 gives eqs. (5)–(10); §3.2 the threshold warm-up and eq. (11); §3.3 the non-linear mapping and eq. (12); Algorithm 1 is the procedure; §4 the hyperparameters. |
| Reference implementation | [`TorchSSL`](https://github.com/TorchSSL/TorchSSL), the authors' own codebase — **not consulted**. This session's GitHub access is scoped to `mattsq/xty2`, so no line of it was read, first- or second-hand. Every row below is sourced from the paper, and Algorithm 1 is the source wherever the prose leaves a procedural gap. |
| Reference impl. runnable? | Not attempted. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities plus treatment-specific outcome means for causal contrasts.
- **Method claim:** FlexMatch replaces FixMatch's constant pseudo-label gate with per-class thresholds derived from sticky row marks accumulated across prior steps.
- **Scope:** the image benchmark and augmentations are not reproduced. The causal stack, tabular views, and paired benchmark are project-local.

## 3. Equations and mapping

### 3.1 As published

Notation is §2's, which is FixMatch's: `X = {(x_b, y_b)}` a batch of `B`
labelled examples, `U = {u_b}` a batch of `mu B` unlabelled ones,
`p_m(y | .)` the model's predicted class distribution, `H(., .)` cross-entropy,
`omega(.)` weak augmentation, `Omega(.)` strong augmentation, `tau` the fixed
threshold and `q_b = p_m(y | omega(u_b))`, `\hat q_b = arg max(q_b)`.

Section 2, the supervised term (eq. 10) and FixMatch's unsupervised term:

$$
\mathcal{L}_s = \frac{1}{B}\sum_{b=1}^{B} H\!\left(y_b, p_m(y \mid \omega(x_b))\right)
\tag{10}
$$

$$
\frac{1}{\mu B}\sum_{b=1}^{\mu B}
  \mathbb{1}\!\left(\max(p_m(y \mid \omega(u_b))) > \tau\right)
  H\!\left(\hat p_m(y \mid \omega(u_b)),\, p_m(y \mid \Omega(u_b))\right)
\tag{3}
$$

Section 2 also states the ideal that motivates the method — "the most ideal
approach would be calculating evaluation accuracies for each class and use them
to scale the threshold" — which is unavailable without labels:

$$
\mathcal{T}_t(c) = a_t(c) \cdot \tau
\tag{4}
$$

Section 3.1, Curriculum Pseudo Labeling. The learning effect of class `c` at
time `t`, over the `N` unlabelled examples:

$$
\sigma_t(c) = \sum_{n=1}^{N}
  \mathbb{1}\!\left(\max(p_{m,t}(y \mid u_n)) > \tau\right) \cdot
  \mathbb{1}\!\left(\arg\max(p_{m,t}(y \mid u_n)) = c\right)
\tag{5}
$$

$$
\beta_t(c) = \frac{\sigma_t(c)}{\max_c \sigma_t}
\tag{6}
$$

$$
\mathcal{T}_t(c) = \beta_t(c) \cdot \tau
\tag{7}
$$

$$
\mathcal{L}_{u,t} = \frac{1}{\mu B}\sum_{b=1}^{\mu B}
  \mathbb{1}\!\left(\max(q_b) > \mathcal{T}_t(\arg\max(q_b))\right)
  H\!\left(\hat q_b,\, p_m(y \mid \Omega(u_b))\right)
\tag{8}
$$

$$
\mathcal{L}_t = \mathcal{L}_s + \lambda \mathcal{L}_{u,t}
\tag{9}
$$

Section 3.2, threshold warm-up. The denominator gains the count of unlabelled
rows that have never been marked, which "can be regarded as the number of
unlabeled data that have not been used", so that "at the beginning of the
training, all estimated learning effects gradually rise from 0 until the number
of unused unlabeled data is no longer predominant":

$$
\beta_t(c) = \frac{\sigma_t(c)}
  {\max\{\max_c \sigma_t,\; N - \sum_c \sigma_t\}}
\tag{11}
$$

Section 3.3, the non-linear mapping. "The mapping function `M` should be
monotonically increasing and have a maximum no larger than `1/tau` [...] we
consider the mapping function to have a range from 0 to 1 so that the flexible
thresholds range from 0 to `tau`. A monotone increasing convex function lets the
thresholds grow slowly when `beta_t(c)` is small, and become more sensitive as
`beta_t(c)` gets larger. Hence, we intuitively choose a convex function with the
above-mentioned properties `M(x) = x / (2 - x)` for our experiments."

$$
\mathcal{T}_t(c) = \mathcal{M}(\beta_t(c)) \cdot \tau,
\qquad \mathcal{M}(x) = \frac{x}{2 - x}
\tag{12}
$$

Algorithm 1 is what actually runs, and three of its lines are load-bearing for
this port because they are the only statement of the procedure:

```text
 1  Input: X = {(x_m, y_m)}, U = {u_n : n in (1..N)}
 2  \hat u_n = -1 : n in (1..N)      # all unlabelled data start "unused"
 3  while not reach the maximum iteration do
 4    for c = 1 to C do
 5      sigma(c) = sum_n 1(\hat u_n = c)              # estimated learning effect
 6      if max sigma(c) < sum_n 1(\hat u_n = -1) then
 7        Calculate beta(c) using Eq. (11)            # unused data dominate
 8      else
 9        Calculate beta(c) using Eq. (6)
10      end if
11      Calculate T(c) using Eq. (7)
12    end for
13    for b = 1 to mu B do
14      if p_m(y | omega(u_b)) > tau then
15        \hat u_b = arg max q_b                      # update the mark
16      end if
17    end for
18    Compute the loss via Eq. (8), (10) and (9).
19  end while
```

Three readings that the mapping depends on:

* **Line 5 replaces eq. (5)'s sum over the dataset with a sum over stored
  marks.** Eq. (5) is stated as a fresh evaluation of every unlabelled row at
  time `t`, which no implementation can afford per step; §3.1 says instead that
  "every time the prediction confidence of an unlabeled data `u_n` is above the
  fixed threshold `tau`, the data, and its predicted class are marked and will
  be used for calculating `beta_t(c)` at the next time step". The marks are
  **sticky**: line 15 overwrites a mark, and nothing sets one back to `-1`, so a
  row that clears `tau` once counts towards `sigma` for the rest of the run.
* **The mark is set at the fixed `tau`, the loss is gated at `T_t(c)`.** Line 14
  reads `tau`; eq. (8) reads `T_t(arg max q_b)`. They are different gates and
  the card's §4 states both.
* **Thresholds are computed before the marks are updated.** Lines 4–12 precede
  lines 13–17, and eq. (8) at line 18 uses the `T(c)` from line 11. So a row's
  gate never depends on the other rows of its own batch — only on batches
  already seen. That is what keeps `batch_coupled` false, and it is the whole
  of the difference between "carries state across steps" and "couples rows
  within a step".

### 3.2 Mapping to xty2

A stage-local `CurriculumStatus` stores one mark per training row. Thresholds are computed before the current batch updates marks; marking uses fixed `tau`, while loss gating uses the class threshold. The strong view uses 20%, not FixMatch's 50%, additional masking.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_m(y \| x)` | model's class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| `omega(.)` | weak augmentation | — | `ViewSpec("weak_x")`, `FeatureMask(p=0.1)`, two draws |
| `Omega(.)` | strong augmentation | — | `ViewSpec("strong_x")`, `FeatureMask(p=0.1)` then `FeatureMask(p=0.2)` — **not** `fixmatch`'s 0.5; deviation 2 and §5.2 |
| eq. (10) `L_s` | supervised cross-entropy on weak views | `T_GIVEN_X @ weak_x draw=1` | `ObservedTreatmentNLL(realisation=Realisation("weak_x", draw=1))`, rows `t_observed`, `reduction="mean"` |
| `q_b` | artificial label distribution | `T_GIVEN_X @ weak_x draw=0` | `CurriculumPseudoLabelTreatmentNLL.target` |
| `\hat q_b` | hard pseudo-label | — | `sharpening="hard"` inside that objective |
| `p_m(y \| Omega(u_b))` | strong-view prediction | `T_GIVEN_X @ strong_x` | `CurriculumPseudoLabelTreatmentNLL.prediction` |
| Alg. 1 line 2, `\hat u_n` | the per-row mark, `-1` for unused | — | `CurriculumStatus`, the objective's per-stage state; `[N]` long, initialised to `-1`, keyed by `XTYBatch.row_id` |
| Alg. 1 line 5, eq. (5) | `sigma_t(c)` from the marks | — | `CurriculumStatus.learning_effect()` |
| eq. (11) / eq. (6) | `beta_t(c)`, with and without warm-up | — | `CurriculumThreshold.warm_up`, resolved inside `CurriculumStatus.thresholds()` |
| eq. (12), `M(x) = x/(2-x)` | the convex mapping | — | `CurriculumThreshold.mapping="convex"` |
| eq. (7)/(12) `T_t(c)` | the per-class threshold | — | the `[K]` vector `CurriculumStatus.thresholds()` returns; logged as `threshold_min` / `threshold_max` |
| Alg. 1 line 14 | mark update, at the fixed `tau` | — | `CurriculumThreshold.tau`, applied by `CurriculumStatus.mark()` |
| eq. (8) `L_u,t` | per-class-gated pseudo-label cross-entropy | `T_GIVEN_X` at both views | `CurriculumPseudoLabelTreatmentNLL`, rows `all`, `reduction="mean"` |
| eq. (9) `lambda` | unlabelled loss weight | — | `Weighted(..., weight=1.0)`, `Constant` |
| `N` | size of the unlabelled set | — | the rows of `TrainingPopulation` this objective is entitled to, counted once at stage start — every training row, since `rows` is `all` |
| `mu`, `B` | batch composition | — | `QuotaSampler(Quota("t_observed", 64), Quota("t_missing", 448))`, imported from `fixmatch` |
| `eta cos(7 pi k / 16 K)` | rate schedule (FixMatch §2.4) | — | `CosineDecay(steps=3000, phase=7/16)` |
| EMA of parameters | the model §4 reports from | — | `TeacherSpec(decay=0.999, role="evaluation")`; no objective reads it |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests. The declared strong view composes `FeatureMask(p=0.1)` with `FeatureMask(p=0.2)` (deviation 2).

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.curriculum_pseudo_label_treatment_nll: p(t|x) @ view=weak_x params=student
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # eq. (8): the target realisation is detached; the label is arg max(q_b) and the gate a step function, both constants w.r.t. theta
  gradient_clipping: none                     # neither paper names any; retained P5 choice
  marginal_nll_grad_path: both                # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.999                            # §4: "an exponential moving average with the momentum of 0.999"
  ema_applies_to_buffers: false               # the declared graph has no buffers; stated so a component that grew one would be a card change
  teacher_in_train_mode: false                # the EMA copy is an evaluation classifier
  teacher_requires_grad: false                # never an optimiser target
  # role = evaluation. Nothing reads this EMA during training: eq. (8)'s label
  # and Alg. 1's marks both come from the current network.

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: mean     # eq. (10) divides by B
    joint_fit.curriculum_pseudo_label_treatment_nll: mean   # eq. (8) divides by mu*B
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.curriculum_pseudo_label_treatment_nll: all    # FixMatch footnote 2, inherited: U is every row
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.curriculum_pseudo_label_treatment_nll: 1.0    # lambda = 1, §4
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.curriculum_pseudo_label_treatment_nll: constant 1.0   # the curriculum is in the threshold, not in lambda
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a                            # CPL adjusts a threshold; it does not sharpen a soft target
  sharpening: hard                            # eq. (8): H(\hat q_b, .) with \hat q_b = arg max(q_b)
  confidence_threshold: curriculum(tau=0.95, warm_up=true, mapping=convex)   # tau from §4; warm-up eq. (11) and Alg. 1 lines 6-9; mapping eq. (12), M(x) = x/(2-x)

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)    # §4: "SGD with a momentum of 0.9"; Nesterov is FixMatch's and §4 states it adopts FixMatch's settings
  lr: 0.03                                       # §4
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))  # FixMatch §2.4, adopted by §4; K = our 3000 steps
  weight_decay: 0.0005 (all trainable components; norm and bias exempt)  # §4's CIFAR-10 value; scope follows fixmatch.md deviation 8
  batch_size: 512                                # B + mu B = 64 + 448, derived from the QuotaSampler's quotas
  labelled_unlabelled_ratio: 7.0                 # mu, §4; derived from the same quotas
  total_steps_or_epochs: 3000                   # optimiser steps, never epochs. The paper's total is 2^20; see deviation 3

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                 # retained reviewed P5 TARNet backbone
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
    mlp_encoder: 0.0                             # perturbation comes from the two explicit input views
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
  treatment_encoding: n/a                       # XTYBatch supplies integer classes 0..K-1
  split_protocol: one fixed project-local DGP, split train/test by the section 6 fixture; no CIFAR/SVHN/STL protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id  # deviation 8
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply FlexMatch to categorical treatment assignment `p(t \| x)` and compose it with a causal outcome likelihood (`ObservedOutcomeNLL`) and exact marginalisation (`MissingTreatmentMarginalNLL`). | The paper studies image classes. The project-local question is whether a per-class curriculum threshold recovers a missing *treatment* label better than a fixed one, and whether it composes with the reviewed P5 stack rather than replacing it. | No comparison to a published image error rate is valid. The marginal term trains `p(t \| x)` on exactly the rows the curriculum is deciding about, so the two mechanisms interact; §6 measures the pair against `fixmatch`, which carries the same interaction. |
| 2 | `judgement` | — | Replace flip-and-shift (weak) and RandAugment + Cutout (strong) with schema-aware feature masking: 10% weak, and 10% followed by **20%** strong. The weak view is `fixmatch`'s to the byte; the strong one keeps its shape at a different rate, which is the one place this recipe departs from that one outside the gate. | There is no image structure in a tabular XTY batch. FixMatch §2.3 asks a strong augmentation to be severe *and label-preserving*, and §5.2 below shows `fixmatch`'s 50% is not: it flips the Bayes-optimal label on 16.8% of rows. That is invisible under a constant gate and load-bearing under a curriculum whose thresholds start at zero. §5.2 states the criterion and the measurement that picks 20%. | Directly defines the invariance being learned, and it is the difference between the mechanism engaging on five seeds of five and on two (§6.2). §6's pair holds it fixed on both arms, so it bounds what the numbers describe rather than confounding what they compare. |
| 3 | `judgement` | — | Train for 3,000 optimiser steps rather than the paper's `2^20`. | The reviewed project-local budget, shared with every other xty2 recipe so that a difference is attributable to the recipe. The cosine schedule's `K` is set to the same 3,000, so the shape of the decay is exact even though its length is not. | The paper's headline convergence claim (§4.3) is about reaching a result *sooner*; a 3,000-step budget is where CPL's early behaviour matters most and its late behaviour matters least. §6 records the trajectory rather than the endpoint alone. |
| 4 | `judgement` | — | Retain the P5 TARNet architecture (encoder, outcome head, propensity) rather than a Wide ResNet. | Holding the causal stack fixed is what makes the CPL addition attributable, and it is `fixmatch.md` deviation 6 and `mean_teacher.md` deviation 10. | The project-local result validates wiring and mechanism, not image-scale accuracy. |
| 5 | `judgement` | — | Retain P5's `Ramp(0.0, 0.5, 1000)` on the marginal-likelihood term while the CPL weight stays constant. | The ramp belongs to the reviewed P5 term, not to FlexMatch; eq. (9) states a fixed `lambda` and the curriculum lives in the threshold. | Identical to `fixmatch`'s arrangement, so the pair in §6 shares it. |
| 6 | `judgement` | — | Adopt FixMatch's optimiser (SGD, `eta = 0.03`, `beta = 0.9`, Nesterov, cosine decay) rather than P5's Adam stack. | §4 states FlexMatch adopts FixMatch's settings, and `fixmatch.md` deviation 9 already made this choice for the recipe this one is paired against. Retaining Adam would break the pair as well as the paper. | Comparisons to `tarnet`'s or `mean_teacher`'s recorded numbers do not hold; the comparison to `fixmatch` does, which is the one §6 makes. |
| 7 | `framework-limitation` | `augmentation-vocabulary` | No adaptive augmentation: the strong view's strength is fixed, where FixMatch's own reference runs CTAugment and FlexMatch inherits the framework. | The argument is `fixmatch.md` deviation 10's and is not restated: learning per-operation magnitudes presupposes a set of tabular operations with magnitudes worth learning over, and `FeatureMask`, `BoundedJitter` and `FeatureCorruption` are three operations with one scalar each. FlexMatch adds nothing to the case either way — its contribution is the threshold, not the augmentation. | Removes whatever adaptivity buys, equally from both arms of §6's pair, so it is a limit on what the numbers describe rather than a confound within them. |
| 8 | `framework-limitation` | `batch-row-repetition` | Set the §6 label budget to 64 rather than a scarcer regime, holding `B = 64` and `mu = 7` at the paper's values. | `XTYBatch.row_id` must be unique (`DESIGN.md` §7.1), so a labelled quota of `B` cannot be drawn from a population smaller than `B` without repeating a row, and the scarcest budget expressible is `B` itself. The alternative — lowering `B` — would deviate from a number the paper states. | Slightly more supervision than the label-scarce regime where §4.1 reports FlexMatch's largest gains, which is the regime this card would most like to be in. It moves both arms of the pair equally. |
| 9 | `judgement` | — | Record §6.1's imbalanced variant as a §6.3 measurement rather than a second Tier 2 target or a second Tier 1 arm. | It exists to exercise the half of CPL the primary fixture leaves inert — with `K = 2` and a balanced assignment both classes reach `beta = 1` together (§2's third limitation). Making it a reproduction target would double the nightly cost of a recipe whose declared claim is the paired one; making it a Tier 1 arm would add minutes of CI on every PR for a number `FIDELITY.md` §3 says is not a result anyway. | None on the §6 metric. It is what §6.3 is allowed to claim that changes: a direction on a few seeds, not a target. |

### 5.1 Framework additions made for this card

`StatefulObjective` adds a per-stage lifecycle;
`CurriculumPseudoLabelTreatmentNLL`, `CurriculumThreshold`, and
`CurriculumStatus` keep the FlexMatch mechanism local. **Named second consumer:**
FreeMatch uses the same executor-owned reset boundary and validates the
`TrainingPopulation | None` initialisation shape, while extending its own SAT
and SAF objectives to share one named sibling state. Existing stateless recipes
receive an empty state mapping and are unchanged.

### 5.2 Strong-view measurement

FixMatch requires a severe, label-preserving strong view. On 200,000 draws from the §6.1 Gaussian-mixture DGP, 50% additional masking flips the Bayes-optimal label on 16.8% of rows; 20% flips 7.4%. The declared 20% rate is the most severe tested candidate that preserves at least 90% of Bayes labels.

| strong view | effective rate | `P(Bayes label flips)` |
|---|---:|---:|
| weak only, 0.1 | 0.10 | 0.026 |
| 0.1 then 0.1 | 0.19 | 0.050 |
| **0.1 then 0.2 (declared)** | **0.28** | **0.074** |
| 0.1 then 0.3 | 0.37 | 0.102 |
| 0.1 then 0.5 (FixMatch) | 0.55 | 0.168 |

The 90% criterion is a judgement and is sensitive: 0.1 then 0.2 is selected only for budgets in roughly `(89.9%, 92.6%]`. The invariant test recomputes the boundary rates.

## 6. Reproduction target

The planned pair holds the 20% strong view fixed and compares curriculum gating with a constant threshold.

The result remains scoped by the open framework debt in §5.7 and §5.8: it
measures a fixed tabular augmentation at the 64-label minimum the current
sampler can express, not inherited adaptive augmentation or the paper's
scarcer-label regime.

```yaml
reproduction:
  dataset: fixmatch.md §6.1's project-local seed-locked two-cluster XTY DGP (6 features, K=2), unmodified
  variant: paired fit against a constant-gate arm — this recipe with eq. (8) replaced by eq. (3) at tau = 0.95, so both arms share deviation 2's views — same seeds and same batches
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio on the EMA parameters, FlexMatch over the constant-gate arm; the paper's mask rate and impurity, and the per-class threshold trajectory, as guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: ratio < 1.0 in mean on both the EMA and the trained parameters, by at least one standard error; terminal mask rate above 0.2; impurity of retained labels < 0.15; held-out outcome NLL within 1.05x of the constant-gate arm; max_c T(c) >= 0.9 tau by the end of the run and min_c T(c) < 0.5 tau at step 0
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGPs

The primary pair uses the same 1,024/2,048-row, 64-label MCAR fixture and seed
streams as `fixmatch.md` §6.1, restated here so this contract is self-contained:

```text
c          = 1[u_c < 0.5]
x[0:4]     = 0.45 * (2c - 1) + 0.6 epsilon[0:4]
x[4:6]     = epsilon[4:6]
p(t=1 | c) = 0.02 + 0.96c
t          = 1[u_t < p(t=1 | c)]
baseline   = 0.5x0 - 0.3x1 + 0.2(x4^2 - 1)
effect     = 1 + 0.5 tanh(x2)
y          = baseline + t * effect + 0.5 epsilon_y
```

Use base seed `90000 + 100r` for `r=0..9`, training-population outcome
standardisation, and the paired `B=64` observed / `448` missing quota stream.
The diagnostic-only imbalanced variant changes only
`c = 1[u_c < 0.15]`; it is not a second Tier 2 target.

### 6.2 Evidence summary

At the declared 3,000-step budget over five seeds, curriculum gating achieved
EMA treatment-NLL ratio `0.977 +/- 0.014` against the constant gate and led in
four seeds. Every declared-view run laid its first mark between steps 32 and 76,
ended with 0.98–1.00 of rows marked, and reached the maximum threshold `tau`.
Using FixMatch's stronger 0.5 view trapped three of five seeds at the ungated
warm-up; a `tau=1` permanently ungated ablation isolated the view, rather than
ungated self-training itself, as the cause. These script-level five-seed
measurements motivate the guardrails but cannot set status; the Tier 2 ledger
below is the result-bearing evidence.

At the declared ten seeds, FlexMatch achieved EMA treatment-NLL ratio
`0.961 +/- 0.010` and trained-parameter ratio `0.965 +/- 0.011`, leading the
constant gate in nine seeds of ten on each. Every predeclared guardrail passed:
terminal mask rate was `0.848 +/- 0.008`, retained-label impurity
`0.058 +/- 0.002`, and all runs reached `max_c T(c) = tau` after starting at
zero. The outcome-NLL ratio was `1.000 +/- 0.0001`.

### 6.3 Result ledger


| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-02 | `a382e53` | ema_treatment_NLL_ratio<br>trained_treatment_NLL_ratio<br>terminal_mask_rate<br>retained_label_impurity<br>held_out_outcome_NLL_ratio<br>terminal_threshold_max<br>initial_threshold_min | 0.961311 +/- 0.00971<br>0.964709 +/- 0.0113<br>0.847559 +/- 0.00837<br>0.0581345 +/- 0.00224<br>0.999657 +/- 0.000107<br>0.95 +/- 0<br>0 +/- 0 | yes |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Whether FlexMatch inherits FixMatch's footnote 2 — that `U` contains the labelled rows too, without their labels — and therefore what `N` counts | It does. `U` is every training row, `N = \|train\| = 1024`, and eq. (8)'s eligible set is `all` | Convention, and consistency: §1 defines FlexMatch as CPL applied to FixMatch, §2 restates FixMatch's framework unchanged and §4 adopts its settings, so the framework is inherited whole. `fixmatch.md` §3.2 reached the same reading for the same term |
| What `M` is in the *reported* results — §3.3 proposes `x/(2-x)` "for our experiments" and Alg. 1 line 11 cites eq. (7), the identity mapping | The convex `x/(2-x)` of eq. (12) | The paper's own words: §3.3 chooses it for its experiments, and §4.4's ablation reports the convex function best and the concave worst. Algorithm 1's citation of eq. (7) is read as the general form eq. (12) specialises, since §3.3 says eq. (7) "can be seen as a special case by setting `M` to the identity function" |
| Whether the threshold warm-up is on in the reported results | On | Algorithm 1 lines 6–9 make it part of the procedure rather than an option, and §3.2 introduces it as a correction to eq. (6) rather than as an alternative to it |
| Whether a mark is ever cleared — a row that clears `tau` once and later stops clearing it | It is not. Marks are sticky and only ever overwritten by another class (Alg. 1 line 15) | The paper's own procedure: line 15 is the only write, and there is no line that restores `-1`. It also matters, because sticky marks make `sigma` monotone in aggregate and therefore make the thresholds' rise monotone-ish, which is the curriculum the method is named for |
| The tabular analogue of "label-preserving" (FixMatch §2.3) — the paper states the property, not a budget for it | Keep the Bayes-optimal label on at least 90% of rows, and subject to that be as severe as possible. §5.2 computes the table; the rule picks `0.1 then 0.2` | Convention, and an explicitly chosen budget. The paper's image augmentations flip a class essentially never, so any tabular threshold is a concession and this one is ours. **It is not robust to itself**: §5.2 reports that the rule selects a different strength outside (89.9%, 92.6%], a window two and a half points wide. What does not turn on it is that `fixmatch`'s 0.55 fails at 16.8%, and that both surviving candidates are safe by any reading |
| Which candidate strong-view strengths to tabulate at all, and when the 90% was fixed | The `0.1 then p` family for `p` in 0.1 to 0.5; the budget was set after the table | Recorded rather than smoothed over, because it is the weakest joint in §5.2. The range came from a diagnostic *training* run — the first implementation locked and weakening the strong view unlocked it — and the 90% was chosen with the table already in hand. So "training-free" describes the *measurement*, which is a property of the DGP and the view and recomputed by Tier 0, and **not** the provenance of the constant applied to it. A reviewer who wants the constant defended independently is asking the right question and this card does not answer it |
| Whether Nesterov momentum is used | Yes | §4 gives "SGD with a momentum of 0.9" and states that FlexMatch adopts FixMatch's hyperparameters; FixMatch's table 4 states Nesterov. Guess where the two documents leave a gap, and the same guess `fixmatch` already runs, which is what keeps §6's pair a pair |
| The strict vs non-strict comparison at the gate: eqs. (5) and (8) and Alg. 1 line 14 all write `>`, where FixMatch's eq. (4) writes `>=` | `>`, as FlexMatch writes it | The paper. It differs from `fixmatch`'s objective in a set of measure zero — the exact tie `max(q) == T` — and is recorded only so that a reader comparing the two objectives' source does not read it as a transcription error |
| What happens when every class has been marked and `max_c sigma_t = 0` cannot occur, versus the degenerate `N = 0` | `N = 0` is rejected at stage start rather than divided by | Convention. A training population with no rows is a fixture error and the objective says so, rather than producing `0/0` thresholds that would silently admit everything |
| Whether the EMA copy or the current network supplies `q_b` and the marks | The current network | Algorithm 1 makes no mention of an EMA in lines 13–17, and §4's EMA sentence is about evaluation. The same reading `fixmatch.md` §7 records |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Codex | 2026-09-02 |
| Plan diffed against §3.2 and §4 | Codex | 2026-09-02 |
