# Recipe spec card: uda

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to review implementation scope; §6 is the
> predeclared evidence contract. Stop after review: no UDA implementation is
> authorised while this card remains `draft`.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848) |
| Authors, year | Qizhe Xie, Zihang Dai, Eduard Hovy, Minh-Thang Luong, Quoc V. Le; 2019 / NeurIPS 2020 |
| DOI / arXiv | [arXiv:1904.12848](https://arxiv.org/abs/1904.12848); arXiv DOI [10.48550/arXiv.1904.12848](https://doi.org/10.48550/arXiv.1904.12848) |
| Version used | arXiv v6, 2020-11-05. The core objective is §2.2 eq. (1); confidence masking and temperature sharpening are §2.4; Training Signal Annealing (TSA) is appendix A.1; CIFAR/SVHN results are §4.2 and appendix B.2; TSA ablation is appendix B.1 table 8. |
| Reference implementation | [`google-research/uda`](https://github.com/google-research/uda) @ [`960684e363251772a5938451d4d2bc0f1da9e24b`](https://github.com/google-research/uda/tree/960684e363251772a5938451d4d2bc0f1da9e24b), especially `image/main.py`, `image/data.py`, `image/preprocess.py`, `image/utils.py`, and `image/scripts/run_cifar10_gpu.sh`. |
| Reference impl. runnable? | Not attempted. It targets TensorFlow 1.13-era GPU/TPU image training; this card relies on source inspection rather than executing that stack. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities `p(t | x)` plus the
  project-local treatment-specific outcome means. UDA changes how the treatment
  classifier uses rows without observed treatment labels: a detached weak-view
  distribution is sharpened and confidence-gated, then used as the target of a
  directed consistency loss on a stronger view. TSA separately gates easy
  labelled examples with a threshold that rises through training.
- **Method claim:** high-quality, label-preserving augmentation makes
  consistency training substantially more useful than simple noise. The paper
  further reports that confidence masking and low-temperature targets are useful
  stabilisers, and that TSA can materially reduce overfitting when labelled data
  are scarce relative to unlabelled data. In the paper's Yelp-5 ablation,
  exp-schedule TSA reduces error from `50.81` without TSA to `41.35` (appendix
  B.1 table 8). On CIFAR-10 with 4,000 labels, the released image configuration
  reports `4.32 ± 0.08` error with RandAugment (appendix B.2 table 9).
- **Nearest shipped baseline and controlled difference:** `fixmatch`. Both use
  one weak target view, one strong prediction view, a confidence gate, the same
  `B=64, mu=7` image reference batch ratio, and the same project-local causal
  stack. UDA keeps a **soft** target, sharpens it with temperature `0.4`, and
  minimises directed KL/cross-entropy; FixMatch takes an argmax hard target.
  This card also includes TSA, which FixMatch does not have.
- **Variant selected here:** the low-label UDA variant with exp-schedule TSA,
  confidence threshold `0.8`, and target temperature `0.4`. The official CIFAR
  shell script does not enable TSA, while appendix A.1 introduces it for the
  low-data regime. Selecting TSA is therefore an explicit paper-supported
  variant choice, not an accidental inheritance from the image script.
- **Not claimed:**
  - No published image-classification number is reproduced. The paper's image
    classes become treatment levels in a six-feature tabular fixture, and
    RandAugment has no schema-preserving tabular analogue.
  - No causal identification follows from consistency training or TSA. A
    generated treatment target remains a training target, not an observed
    treatment and not a missingness assumption.
  - No claim is made that TSA or temperature sharpening must help every tabular
    problem. §6 isolates both mechanics so a gain from the full recipe cannot
    be attributed to them without evidence.
  - No claim is made about out-of-domain filtering, BERT transfer, ImageNet
    scaling, back-translation, TF-IDF replacement, or entropy minimisation.
    Those are outside this card.

## 3. Equations and mapping

### 3.1 As published

For labelled distribution `p_L(x)`, unlabelled distribution `p_U(x)`, model
`p_theta(y | x)`, perfect target function `f*(x)`, augmentation distribution
`q(x_hat | x)`, and a fixed copy `theta_tilde` of the current parameters, UDA
minimises (§2.2 eq. (1))

$$
\min_\theta \mathcal J(\theta)
= \mathbb E_{x_1\sim p_L}
  [-\log p_\theta(f^*(x_1)\mid x_1)]
+ \lambda\,
  \mathbb E_{x_2\sim p_U}
  \mathbb E_{\hat x\sim q(\hat x\mid x_2)}
  [\mathrm{CE}(p_{\tilde\theta}(y\mid x_2)
  \|p_\theta(y\mid \hat x))].
\tag{1}
$$

The paper states that `theta_tilde` is a fixed copy of the current parameters:
the target branch is stop-gradient, not an EMA teacher. It uses `lambda=1` for
most experiments.

Section 2.4 adds confidence masking. For an unlabelled minibatch `B`, only rows
whose **unsharpened** weak prediction has maximum probability above `beta`
contribute:

$$
\frac{1}{|B|}\sum_{x\in B}
I\!\left(\max_{y'}p_{\tilde\theta}(y'\mid x)>\beta\right)
\mathrm{CE}\!\left(
  p^{(sharp)}_{\tilde\theta}(y\mid x)
  \|p_\theta(y\mid \hat x)
\right).
$$

The sharpened target is

$$
p^{(sharp)}_{\tilde\theta}(y\mid x)
=\frac{\exp(z_y/\tau)}{\sum_{y'}\exp(z_{y'}/\tau)}.
$$

For CIFAR-10, SVHN, and ImageNet, the paper sets `tau=0.4`; for CIFAR-10 and
SVHN it sets `beta=0.8`. The released image code confirms that the confidence
gate reads `softmax(ori_logits)` before temperature scaling, while the KL target
uses `ori_logits / tau`. Rejected rows remain in the `|B|` denominator because
the implementation masks per-row KL and then takes its mean.

Appendix A.1 defines Training Signal Annealing. At step `t`, a labelled example
is removed from the supervised loss when its predicted probability of the
correct class exceeds `eta_t`. With `K` classes,

$$
\eta_t = \alpha_t\left(1-\frac{1}{K}\right)+\frac{1}{K}.
$$

For total training steps `T`, the three published schedules are

$$
\alpha_t^{log}=1-\exp(-5t/T),\qquad
\alpha_t^{linear}=t/T,\qquad
\alpha_t^{exp}=\exp(5(t/T-1)).
$$

This card selects `exp`, the strongest Yelp-5 result in appendix B.1 table 8.
The released code removes rows whose correct-label probability is **greater
than** the threshold and divides supervised loss by the number retained,
clamped to at least one. Thus TSA and UDA's confidence gate intentionally use
different denominators.

### 3.2 Mapping to xty2

One `joint_fit` gradient stage. The paper's classes map to treatment levels, so
`p_theta(y|x)` becomes `T_GIVEN_X`. The ordinary outcome likelihood and exact
missing-treatment marginal remain project-local companions. Two new objectives
are proposed because the existing objects deliberately encode different
arithmetic:

1. `ConfidenceMaskedConsistencyLoss` implements the detached,
   temperature-sharpened, confidence-gated directed KL. `ConsistencyLoss`
   already provides directed KL and stop-gradient, but has neither target
   temperature nor a gate; `PseudoLabelTreatmentNLL` intentionally hardens by
   argmax and rejects soft UDA targets.
2. `TrainingSignalAnnealedTreatmentNLL` implements supervised cross-entropy with
   the scheduled correct-label ceiling and retained-row denominator. Static row
   populations and ordinary objective-weight schedules cannot express a gate
   that depends on the model's current probability for the true class.

Both additions are objective-local and reversible. No new port, row population,
executor, artifact, state lifecycle, or stage type is requested.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_theta(y | x)` | class distribution | `T_GIVEN_X` | `CategoricalPropensity` over the reviewed `MLPEncoder` backbone |
| labelled `x_1, f*(x_1)` | observed treatment label | `T_GIVEN_X @ weak_x` | `TrainingSignalAnnealedTreatmentNLL`, rows `t_observed` |
| `eta_t` | TSA ceiling on correct-label probability | — | exp schedule inside `TrainingSignalAnnealedTreatmentNLL`, from `1/K` toward `1` over 3,000 optimiser steps |
| weak/original unlabelled `x_2` | detached consistency target | `T_GIVEN_X @ weak_x` | left/target side of `ConfidenceMaskedConsistencyLoss` |
| `x_hat ~ q(x_hat | x_2)` | strongly augmented unlabelled row | `T_GIVEN_X @ strong_x` | right/prediction side of `ConfidenceMaskedConsistencyLoss` |
| `theta_tilde` | stop-gradient current parameters | `T_GIVEN_X @ weak_x, params=student` | `stop_grad="left"`; no training teacher is introduced |
| `tau=0.4` | target softmax temperature | — | `target_temperature=0.4` in `ConfidenceMaskedConsistencyLoss` |
| `beta=0.8` | gate on unsharpened weak confidence | — | `confidence_threshold=0.8` in `ConfidenceMaskedConsistencyLoss` |
| `CE(p_sharp || p_strong)` | directed soft consistency term | `T_GIVEN_X @ weak_x,strong_x` | `ConfidenceMaskedConsistencyLoss(divergence="kl")`, rows `t_missing` |
| `lambda=1` | unsupervised loss weight | — | `Weighted(..., weight=1.0)`, constant |
| simple image augmentation | ordinary training perturbation | — | `ViewSpec("weak_x")`, one schema-preserving `FeatureMask(p=0.1)` |
| RandAugment + Cutout | stronger unlabelled perturbation | — | `ViewSpec("strong_x")`, weak mask plus `FeatureMask(p=0.5)`; project-local substitution, deviation 2 |
| image `B=64`, `mu=7` | labelled/unlabelled batch ratio | — | `QuotaSampler(Quota("t_observed",64), Quota("t_missing",448))` |
| moving-average evaluation model | source implementation reports EMA weights | — | `TeacherSpec(decay=0.9999, role="evaluation", ema_applies_to_buffers=True)`; no objective reads it |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | exact missing-treatment likelihood | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

Three arithmetic details are load-bearing.

- **Gate before sharpening.** The confidence mask is computed from the ordinary
  weak softmax. Lowering `tau` may change the target entropy but must not change
  which rows pass the gate.
- **Target only is detached.** UDA's `theta_tilde` is the current model under
  stop-gradient, not the evaluation EMA. The strong branch trains normally.
- **The denominators differ.** UDA consistency divides by all eligible
  unlabelled rows after zeroing rejected ones. TSA divides by the number of
  labelled rows retained by its ceiling. Conflating the two makes the effective
  objective weights change in the wrong direction.

## 4. Mechanics checklist

This YAML is the intended executable fidelity contract. Review must reconcile
its non-`n/a` entries with the future plan before status can move to `reviewed`.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.tsa_observed_treatment_nll: none
    joint_fit.uda_consistency: p(t|x) @ view=weak_x params=student
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # paper §2.2: theta_tilde is fixed; ref impl tf.stop_gradient(ori_logits_tgt)
  gradient_clipping: none                     # image reference code does not clip
  marginal_nll_grad_path: both                # reviewed project-local P5 choice

teacher:
  ema_decay: 0.9999                           # ref impl image/main.py moving_average_decay
  ema_applies_to_buffers: true                # ref impl utils.get_all_variable includes BN moving mean/variance; adapted graph has no BN, but retain the source contract
  teacher_in_train_mode: false                # evaluation role only
  teacher_requires_grad: false

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.tsa_observed_treatment_nll: mean_retained    # sum(masked CE) / max(sum(mask), 1), ref impl
    joint_fit.uda_consistency: mean                        # mean over every eligible unlabelled row after zeroing rejected rows
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.tsa_observed_treatment_nll: t_observed
    joint_fit.uda_consistency: t_missing
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.tsa_observed_treatment_nll: 1.0
    joint_fit.uda_consistency: 1.0             # lambda, paper §2.2 and README guidance
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.tsa_observed_treatment_nll: tsa_exp ceiling 1/K -> 1 over 3000 optimiser steps
    joint_fit.uda_consistency: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 optimiser steps
  temperature: 0.4                            # tau, paper §2.4; applied only to detached weak target logits
  sharpening: softmax_temperature             # target remains soft; no argmax
  confidence_threshold: uda(unsupervised=0.8, tsa=exp(1/K -> 1))

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True) # ref impl image/main.py
  lr: 0.03                                    # image default and CIFAR GPU script
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))  # source cos((7/8)*(pi/2)*progress), adapted to 3000 steps
  weight_decay: 0.0005 (all trainable parameters, including bias) # ref impl utils.decay_weights
  batch_size: 512                             # 64 labelled + 448 unlabelled rows in the adapted quota batch
  labelled_unlabelled_ratio: 7.0              # mu, released CIFAR GPU script
  total_steps_or_epochs: 3000                 # optimiser steps; source CIFAR script uses 500000 (deviation 3)

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]               # retained reviewed P5 backbone
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
  treatment_encoding: integer classes 0..K-1 through the XTYBatch contract
  split_protocol: one fixed project-local DGP, split train/test by §6.1; no CIFAR/SVHN protocol applies
  missingness_mechanism: treatment MCAR to exactly 64 labelled training rows, keyed by row_id
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply UDA to categorical treatment assignment `p(t | x)` and compose it with `ObservedOutcomeNLL` and `MissingTreatmentMarginalNLL`. | The paper is a classification method; xty2's research question is whether the same missing-label mechanism helps treatment prediction while coexisting with the causal outcome stack. | Published image error is not comparable. The marginal likelihood may either complement or compete with consistency, so §6 uses a matched project-local ablation. |
| 2 | `judgement` | — | Replace crop/flip and RandAugment+Cutout with schema-preserving feature masks: weak `p=0.1`, strong = weak plus `p=0.5`. | Image transforms are not meaningful on the tabular fixture. A tabular augmentation must preserve row identity and treatment semantics, which the §6 fixture measures before training. | Likely smaller and less diverse perturbations than RandAugment; this can reduce the consistency benefit. |
| 3 | `judgement` | — | Train 3,000 optimiser steps rather than the released CIFAR GPU script's 500,000. | The project benchmark is a mechanism test on a small synthetic DGP, not an image reproduction. The matched arms must finish in the ordinary benchmark budget. | May understate long-horizon gains; all §6 arms share the same budget. |
| 4 | `judgement` | — | Use the reviewed MLP/TARNet/propensity stack rather than Wide-ResNet-28-2. | Architecture swaps are deliberately orthogonal to recipe identity in xty2. | No published architecture-dependent accuracy comparison is valid. |
| 5 | `judgement` | — | The consistency objective runs only on `t_missing` rows, while the released image preprocessing creates the unsupervised stream from the full training set, including images that may also be in the labelled subset. | In the XTY adaptation, `t_missing` is the statistical unlabelled population. Reusing observed-treatment rows as if unlabelled would overweight those identities and complicate paired row provenance without testing the missing-treatment mechanism. | Slightly less consistency data than source-style full-population reuse; direction otherwise unknown. |
| 6 | `judgement` | — | Use exactly 64 observed treatments rather than CIFAR's 250–4,000 labelled-image settings. | This is the existing project fixture and deliberately creates the low-label regime in which TSA is relevant. | Makes overfitting pressure stronger and may increase TSA's effect. |
| 7 | `judgement` | — | Enable exp-schedule TSA in the named `uda` recipe although the released CIFAR GPU script leaves `--tsa` unset. | TSA is part of the paper, appendix A.1 identifies it specifically for low-data SSL, and the backlog asks the UDA acceptance work to isolate it. The recipe therefore selects a paper-supported low-label variant rather than the exact CIFAR command. | May improve or hurt the adapted fixture. §6 includes a one-mechanic no-TSA arm so the effect is attributable. |
| 8 | `judgement` | — | Retain the project's ramped missing-treatment marginal term at weight `0.5`. | This term is part of the causal stack UDA is being tested against, not part of the paper. Removing it would answer a different xty2 question. | Can improve treatment and outcome estimation independently of UDA; every arm holds it fixed. |

### 5.1 Framework additions made for this card

The card proposes two reversible, fidelity-bearing objective objects. It asks for
no new load-bearing vocabulary, so `DESIGN.md` §11.2 does not require a named
second consumer.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `ConfidenceMaskedConsistencyLoss` — directed KL between two `T_GIVEN_X` realisations with detached target, target temperature, gate on untempered target confidence, and full-population denominator | fidelity-bearing, reversible | UDA | not required | Existing `ConsistencyLoss` has directed KL but no sharpening or confidence gate; `PseudoLabelTreatmentNLL` intentionally hardens by argmax. UDA's soft target and gate are source mechanics, not optional flags on FixMatch. |
| `TrainingSignalAnnealedTreatmentNLL` — observed-label NLL masked by a scheduled ceiling on the current correct-class probability, reducing over retained rows | fidelity-bearing, reversible | UDA | not required | Neither a static row population nor an objective-weight schedule can express appendix A.1. The gate depends on the model's current prediction for each row and uses a different denominator from UDA consistency. |

**Existing contracts deliberately reused.** `Realisation`, `T_GIVEN_X`,
`CategoricalPropensity`, `ViewSpec`, `FeatureMask`, `QuotaSampler`, `Weighted`,
ordinary schedule timing, evaluation-only `TeacherSpec`, the project-local
outcome objectives, and the existing stage/executor are sufficient. If
implementation discovers that either proposed objective needs state, a new port,
a dynamic sampler, or a new executor, amend this card and stop again rather than
hiding that capability in the recipe function.

## 6. Reproduction target

The project-local acceptance target asks one primary question: does UDA's
confidence-gated soft consistency improve treatment prediction over the same
low-label fit without that consistency term? TSA is held on in that primary
pair. Two additional one-mechanic arms separately remove target sharpening and
TSA so their effects cannot be folded into the UDA result.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in §6.1
  variant: four paired arms — full UDA; no-consistency (lambda_uda=0, TSA retained); no-sharpening (tau=1, TSA retained); no-TSA (ordinary observed-treatment NLL, tau=0.4); identical initialisation, batches, views, optimiser, schedules and project-local causal terms otherwise
  split: 1024 train rows with exactly 64 observed treatments; 2048 held-out rows with every treatment observed
  metric: held-out treatment NLL and accuracy for student and evaluation EMA; held-out outcome NLL as guardrail; UDA gate coverage/confidence and target entropy; TSA retained fraction and ceiling trajectory
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: primary full-vs-no-consistency held-out treatment NLL ratio < 1.0 in mean by at least one standard error for both student and EMA; held-out outcome NLL <= 1.05x no-consistency; tau=0.4 target entropy < tau=1 target entropy on matched weak logits; changing tau must not change gate membership; TSA retained fraction must begin below 1 and rise over training; report full-vs-no-sharpening and full-vs-no-TSA effects without selecting either direction after seeing results
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

For replicate `r = 0..9`, use base seed `s_r = 94000 + 100r`; generate the
1,024-row training population with `s_r+1`, the 2,048-row held-out population
with `s_r+2`, initialise model parameters with `s_r+6`, and train with stage seed
`s_r+10000`. This repeats the shipped FixMatch family fixture in full so the
benchmark contract does not depend on a section of another card.

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
all held-out treatments and every outcome are observed. Assert that the fixed
observed set contains both treatment levels and fail the fixture rather than
reseeding if it does not. Fit outcome standardisation on the complete training
population. Every arm receives the same paired quota stream of 64 observed and
448 missing rows per optimiser step.

Before training, measure rather than assume:

1. Bayes-optimal treatment-label flip rate under `weak_x` and `strong_x`.
2. Treatment prevalence in train, observed-train, missing-train, and held-out
   populations.
3. The untrained weak-view confidence distribution, because a gate that accepts
   almost no rows is a different experiment from one that gradually opens.
4. The initial TSA retained fraction at its source-faithful exp threshold. For
   `K=2`, the reference formula starts slightly above `0.5`, not exactly at it.

### 6.2 Predeclared evidence

**Tier 0 (invariants).**

1. `ConfidenceMaskedConsistencyLoss` matches a direct tensor calculation of
   `KL(softmax(z_weak/tau) || softmax(z_strong))` with the weak branch detached.
2. The confidence gate is computed from `softmax(z_weak)` before temperature
   scaling. Changing `tau` on fixed logits changes target entropy and KL but
   leaves the accepted-row mask bit-identical.
3. A rejected UDA row contributes zero while remaining in the denominator. An
   all-rejected eligible set returns exactly zero rather than NaN or an empty
   mean.
4. Gradient reaches the strong logits and not the weak target logits. The
   confidence mask itself carries no gradient.
5. At `tau=1`, the target distribution is exactly the ordinary weak softmax. At
   `tau<1`, target entropy is no larger, with strict inequality for any
   non-uniform row.
6. `TrainingSignalAnnealedTreatmentNLL` computes the probability of the **true
   observed treatment**, not the model's argmax class, and drops rows only when
   that probability is greater than the current ceiling.
7. TSA exp threshold matches
   `exp(5*(step/steps-1))*(1-1/K)+1/K`, is monotone non-decreasing, and reaches
   `1` at the final step. The step-zero value matches the reference formula.
8. TSA divides by retained labelled rows, clamped to one. An all-dropped batch
   returns zero; doubling the number of dropped rows does not dilute the loss on
   the rows that remain.
9. UDA's gate and TSA's gate therefore have deliberately different reduction
   invariants. A test that swaps their denominators must fail.
10. The evaluation EMA is not a required realisation of either training
    objective. Replacing the detached current weak branch with teacher params
    changes the plan and is rejected by the card/plan check.
11. `plan.hyperparameters` contains every non-`n/a` §4 value and plan details
    print the UDA gate source (`untempered weak`), target temperature, full-row
    denominator, TSA schedule family, TSA comparison (`>`), and retained-row
    denominator.
12. The recipe adds no stateful objective, artifact, port, executor, or new row
    population; compilation remains a single ordinary gradient stage.

**Tier 1 (smoke fit and mechanism arms).**

1. Run `full`, `no-consistency`, `no-sharpening`, and `no-TSA` for one seed with
   identical initial parameters and batch/view RNG keys. Confirm the supervised
   and outcome terms remain finite and the full arm beats marginal-frequency
   treatment NLL.
2. Report UDA gate coverage, accepted confidence, weak target entropy, KL, and
   held-out treatment NLL trajectories. A rising gate may increase the raw
   consistency loss because more rows become chargeable; do not require the raw
   loss itself to fall monotonically.
3. **No consistency:** set only the UDA consistency weight to zero. TSA, views,
   marginal likelihood, optimiser, and batches remain unchanged. This is the
   primary attribution arm.
4. **No sharpening:** set only `tau=1`. Gate membership must remain identical
   step-for-step to full under paired weak logits until parameter trajectories
   diverge; at step zero it must be exactly identical.
5. **No TSA:** replace only the TSA objective with ordinary
   `ObservedTreatmentNLL` under the same weak view and mean reduction. UDA
   consistency remains unchanged.
6. **Gate off diagnostic:** set `beta=-infinity` / an explicit always-accept
   constructor mode in the benchmark object, not in the named recipe. Report
   whether low-confidence targets are responsible for any instability; no
   direction is predeclared.
7. **Strong-view collapse diagnostic:** replace `strong_x` with `weak_x draw=1`
   while keeping independent RNG. If UDA still appears to gain, inspect whether
   the project-local marginal term rather than augmentation consistency is doing
   the work.
8. Report the Bayes-label flip rate of both views beside every fit result. A
   strong transform with materially higher flip rate invalidates the intended
   label-preserving interpretation and must be treated as a data-policy failure,
   not evidence against UDA.

**Tier 2 (fixed ten-replicate target).** Run the four YAML arms over all ten
replicates. The status may become `reproduced` only if the primary full-vs-no-
consistency treatment-NLL target and outcome guardrail pass. The sharpening and
TSA contrasts are reported as predeclared mechanism effects; their signs are not
retroactively promoted to acceptance criteria.

**What has run.** Nothing. This is a draft card. No implementation or benchmark
result exists, and no number in this section may be read as empirical evidence
for xty2.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | — | — |

## 7. Unknowns

| Unspecified or source-dependent choice | Our choice | Basis |
|---|---|---|
| Which TSA schedule belongs to the named low-label UDA variant. | `exp_schedule`. | Appendix A.1 says exp is most suitable when overfitting pressure is high; appendix B.1 table 8 gives it the best Yelp-5 result. The backlog explicitly asks the UDA work to isolate TSA. |
| The paper prose says TSA rises from `1/K`, while the published exp formula and reference code start at `exp(-5)*(1-1/K)+1/K`. | Follow the formula/code exactly, including the small positive offset at step 0. | `get_tsa_threshold` in the pinned image reference implementation and appendix A.1 figure 5. |
| Whether the confidence gate should read the sharpened target or the ordinary weak prediction. | Gate the ordinary weak prediction. | Paper §2.4 writes the indicator with `p_theta_tilde(y|x)` and defines sharpening inside the CE target; pinned `image/main.py` computes `largest_prob` from `softmax(ori_logits)` while sharpening uses `ori_logits / uda_softmax_temp`. |
| Whether the detached target is an EMA teacher. | No. Use current student parameters under stop-gradient. | Paper eq. (1) calls `theta_tilde` a fixed copy of current `theta`; pinned code uses `tf.stop_gradient` on current weak logits. The separate EMA is applied after optimisation for evaluation checkpoints. |
| Whether rejected UDA rows should be removed from the denominator. | No. Zero them then mean over all eligible unlabelled rows. | Pinned image and text implementations multiply per-example KL by a mask and then take `reduce_mean`. |
| Whether TSA uses the same denominator convention. | No. Divide by retained labelled rows, clamped to one. | Pinned `anneal_sup_loss` uses `sum(masked loss) / max(sum(mask),1)`. |
| Whether UDA consistency should also train observed-treatment rows in the XTY adaptation because the released image unsupervised files contain the full train population. | No; use `t_missing` only. | Deliberate adaptation in deviation 5. This makes the objective correspond to the statistical unlabelled population and keeps row provenance simple. |
| Exact tabular weak/strong transforms. | Reuse the shipped FixMatch family masks: weak `FeatureMask(0.1)`; strong weak-plus-`FeatureMask(0.5)`. | There is no paper-prescribed tabular transform. Reuse gives the closest controlled comparison to the existing hard-target recipe; §6 measures label flips before training. |
| Weight decay reach. | All trainable parameters, including biases; the adapted graph has no BatchNorm parameters. | Pinned `utils.decay_weights` sums `tf.nn.l2_loss` over every `tf.trainable_variable`. |
| EMA buffer treatment in an architecture with no BN. | Declare `ema_applies_to_buffers=true` even though it is operationally inert here. | Pinned `utils.get_all_variable` explicitly includes BN moving mean and variance. Keeping the source contract prevents a future architecture swap from silently changing evaluation semantics. |
| Comparison operator at the UDA confidence threshold. | Strict `> 0.8`. | Paper §2.4 says “greater than”; pinned image code uses `tf.greater`. This intentionally differs from FixMatch's source-port `>=` choice. |
| Comparison operator at the TSA ceiling. | Drop a row only when correct-label probability is strictly `>` the ceiling. | Appendix A.1 prose and pinned `tf.greater` implementation. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
