# Recipe spec card: remixmatch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity; §6 is the
> predeclared evidence contract. This card is stopped for review
> (`CLAUDE.md` hard rule 1): no `xty2/recipes/remixmatch.py` exists yet, and
> §5.1 asks review two questions before any code is written.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring](https://arxiv.org/abs/1911.09785) |
| Authors, year | David Berthelot, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Kihyuk Sohn, Han Zhang, Colin Raffel; 2019 / ICLR 2020 |
| DOI / arXiv | [arXiv:1911.09785](https://arxiv.org/abs/1911.09785); [10.48550/arXiv.1911.09785](https://doi.org/10.48550/arXiv.1911.09785) |
| Version used | arXiv v2, 2020-02-13, read through the ar5iv HTML rendering. Algorithm 1 gives the per-batch procedure; §3.1.2 gives distribution alignment; §3.2.1 augmentation anchoring; §3.2.2 CTAugment; §3.3 eqs. (3)–(4) and the hyperparameters; §4.1 table 1 the CIFAR-10/SVHN results; §4.4 table 3 the ablation; appendix C the CTAugment transformation list; appendix D the alignment measurement. Section references in this card are to *this* card unless prefixed "the paper's". |
| Reference implementation | [`google-research/remixmatch`](https://github.com/google-research/remixmatch) @ [`f7061ebf055227cbeb5c6fced1ce054e0ceecfcd`](https://github.com/google-research/remixmatch/tree/f7061ebf055227cbeb5c6fced1ce054e0ceecfcd), read directly at that commit: `remixmatch_no_cta.py` (`guess_label`, `model`, the flag defaults), `cta/cta_remixmatch.py` (the headline entry point and *its* flag defaults), `libml/layers.py` (`MixMode`, `PMovingAverage`, `PData`, `interleave`), `libml/ctaugment.py` (`CTAugment`), `cta/lib/train.py` (`train_step`, the CTAugment feedback loop) and `libml/utils.py` (`model_vars`). Every §7 row marked "ref impl" names the file and line it comes from. |
| Reference impl. runnable? | Not attempted. It trains a WideResNet-28-2 on CIFAR-10 for 2^26 images under TensorFlow 1.x; this card relies on source inspection at the pinned commit. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities `p(t | x)` plus the
  project-local treatment-specific outcome means. ReMixMatch changes how rows
  without an observed treatment train the classifier: one weakly augmented
  *anchor* prediction per row is corrected towards the labelled marginal
  (distribution alignment), sharpened, and then used as the target of `K`
  strongly augmented copies of the same row; every labelled and unlabelled
  entry is then MixUp'd against a shuffled partner drawn from the pooled set
  before the two cross-entropy terms are charged. Two auxiliary terms — an
  unmixed copy of the unlabelled loss and a four-class self-supervised pretext
  loss — are added at weight `0.5` each.
- **Claim:** ReMixMatch is between 5x and 16x more data-efficient than
  MixMatch. Its evidence is the paper's table 1 (CIFAR-10 error `6.27 ± 0.34`
  at 250 labels against MixMatch's `11.08 ± 0.87`; SVHN `3.10 ± 0.50` against
  `3.78 ± 0.26`) and its table 3 single-split ablation, which is the part this
  card is actually testing a port of: from `5.94`, removing distribution
  alignment costs `1.34` points, removing the pre-mixup loss `0.72`, removing
  the rotation loss `0.14`, dropping to `K = 1` costs `1.38`, and replacing the
  unlabelled cross-entropy with MixMatch's L2 costs `11.34`.
- **Nearest shipped baseline:** `uda`. Both sharpen a detached weak-view
  distribution and charge it against a stronger view of the same row. Three
  controlled differences: ReMixMatch has **no confidence gate at all**, it
  corrects the target's *marginal* before sharpening, and it charges the loss at
  synthetic MixUp rows rather than at the augmented rows themselves.
  `softmatch` is the nearest card for the alignment mechanic alone — its
  Uniform Alignment (eq. 8) is this paper's operation with `p(y)` fixed to
  `u(K)` and applied to the *weight's* confidence rather than to the label.
- **Variant selected here:** the paper's headline configuration, which is the
  `cta/cta_remixmatch.py` entry point: `redux='1st'` (augmentation anchoring as
  described in §3.2.1), `use_dm=True`, `use_xe=True`, `K = 8`, `T = 0.5`,
  `alpha = 0.75`, `lambda_U = 1.5`, `lambda_U1 = lambda_r = 0.5`, `mixmode
  'xxy.yxy'`, Adam at `2e-3`, EMA `0.999`. Note that the *no-CTA* entry point
  defaults `redux='swap'` instead, which is a different method (§7); this card
  follows the CTA entry point's default and not that one.
- **Not claimed:**
  - **No published number is reproduced.** Every result above is image
    classification and no dataset in the paper carries a treatment. §6 is a
    project-local mechanism target.
  - **Nothing about CTAugment.** Deviation 3 omits it; a null result here is
    not evidence about the paper's augmentation controller, and a positive one
    does not validate it.
  - **Nothing about few-shot collapse.** §4.3's claim that the rotation loss
    prevents collapse at 40 labels is not tested: this fixture's label budget is
    64 and its pretext task is a substitute (deviation 4).
  - **No inference claim for pseudo-labelled rows.** An aligned, sharpened
    label is a training signal, not an identified treatment (`BACKLOG.md` §7.4).
  - **Nothing about MixUp as a general xty2 view.** §5.1's mixing vocabulary is
    built for this card's eq. (3) and is checked against MixMatch's shape; it is
    not a claim that synthetic rows are safe anywhere else in the framework, and
    §3.2 states the four places it is forbidden.

## 3. Equations and mapping

### 3.1 As published

Algorithm 1, for a labelled batch `X = {(x_b, p_b)}` and an unlabelled batch
`U = {u_b}`, both of size `B`, with sharpening temperature `T`, augmentation
count `K` and MixUp Beta parameter `alpha`:

```text
 3:  x̂_b   = StrongAugment(x_b)                                  // labelled rows are strongly augmented
 4:  û_b,k = StrongAugment(u_b);  k in {1..K}
 5:  ũ_b   = WeakAugment(u_b)                                     // the anchor
 6:  q_b   = p_model(y | ũ_b; θ)
 7:  q_b   = Normalize(q_b × p(y) / p̃(y))                        // distribution alignment
 8:  q_b   = Normalize(q_b^{1/T})                                 // sharpening
10:  X̂  = ((x̂_b, p_b))
11:  Û₁ = ((û_b,1, q_b))                                          // first strong copy, no MixUp
12:  Û  = ((û_b,k, q_b);  k in 1..K)
13:  Û  = Û ∪ ((ũ_b, q_b))                                        // the weak copies join Û
14:  W  = Shuffle(Concat(X̂, Û))
15:  X' = (MixUp(X̂_i, W_i))
16:  U' = (MixUp(Û_i, W_{i+|X̂|}))
17:  return X', U', Û₁
```

`Normalize(x)_i = x_i / Σ_j x_j`. `p̃(y)` is "the moving average of the model's
predictions on unlabeled examples over the last 128 batches" and `p(y)` is
"estimated based on the labeled examples seen during training" (§3.1.2).
`MixUp(a, b)` draws `λ ~ Beta(alpha, alpha)`, sets `λ' = max(λ, 1 - λ)` and
returns `λ' a + (1 - λ') b` applied to features and targets alike
(ref impl `libml/layers.py:184-195`); `λ' ≥ 1/2` is what makes the mixed entry
"closer to" its first source.

The loss is §3.3 eqs. (3)–(4):

$$
\sum_{x,p\in\mathcal{X}'}\mathrm{H}\big(p,\ p_{\rm model}(y\mid x;\theta)\big)
+\lambda_{\mathcal{U}}\sum_{u,q\in\mathcal{U}'}
\mathrm{H}\big(q,\ p_{\rm model}(y\mid u;\theta)\big)
\tag{3}
$$

$$
+\ \lambda_{\hat{\mathcal{U}}_1}\sum_{u,q\in\hat{\mathcal{U}}_1}
\mathrm{H}\big(q,\ p_{\rm model}(y\mid u;\theta)\big)
+\ \lambda_r\sum_{u\in\hat{\mathcal{U}}_1}
\mathrm{H}\big(r,\ p_{\rm model}(r\mid \mathrm{Rotate}(u,r);\theta)\big)
\tag{4}
$$

with `r ~ Uniform{0, 90, 180, 270}` and `λ_r = λ_U1 = 0.5`, `λ_U = 1.5`,
`T = 0.5`, `alpha = 0.75`, `K = 8`. The reference reduces all four terms with
`reduce_mean` over their own entries and ramps `λ_U` and `λ_U1` — but not
`λ_r` — linearly from zero over the first `1024 kimg` of a `65536 kimg` run,
i.e. over the first `1.5625%` of training (`remixmatch_no_cta.py:70,114`).

CTAugment (§3.2.2): each transformation parameter is binned; bin weights `m`
start at `1`. For training augmentation, bins are sampled from
`Categorical(Normalize(m̂))` where `m̂_i = m_i` if `m_i > 0.8` and `0`
otherwise. To update, a labelled example is augmented with uniformly sampled
bins, and `ω = 1 - (1/2L) Σ |p_model(y | x̂; θ) - p|` is folded in as
`m_i = ρ m_i + (1 - ρ) ω` with `ρ = 0.99`. The reference measures `ω` with the
**EMA** classifier (`cta/lib/train.py:71-76` reads `ops.classify_op`, which
`remixmatch_no_cta.py:170` builds under `ema_getter`).

### 3.2 Mapping to xty2

One `joint_fit` gradient stage. The paper's classes are treatment levels, so
`p_model(y | ·)` is `T_GIVEN_X` from `CategoricalPropensity` over `MLPEncoder`,
and the project-local outcome likelihood and exact missing-treatment marginal
stay beside it unchanged, as in every card of this family.

A quota batch already contains both populations, so `X̂` and the strong copies
of `Û` are the *same* realisation read at two row populations. That is what
makes eq. (3)'s pooled shuffle expressible at all, and it fixes the pool
membership exactly:

| Pool member | Realisation | Rows | Source line |
|---|---|---|---|
| `X̂` | `strong_x @ draw=0` | `t_observed` | Alg. 1 line 3 |
| `Û₁` | `strong_x @ draw=0` | `t_missing` | Alg. 1 line 4, `k=1` |
| `Û₂..Û_K` | `strong_x @ draw=1..K-1` | `t_missing` | Alg. 1 line 4 |
| `Û_{K+1}` | `weak_x` | `t_missing` | Alg. 1 line 13 |

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_model(y\|x;θ)` | class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| `WeakAugment` | the anchor transform | — | `ViewSpec("weak_x", (FeatureMask(p=0.1),), draws=1)` — `fixmatch.md`'s reviewed weak view (deviation 2) |
| `StrongAugment` | CTAugment | — | `ViewSpec("strong_x", (FeatureMask(p=0.1), FeatureMask(p=0.5)), draws=8)` (deviations 2, 3) |
| `q_b`, Alg. line 6 | anchor prediction | `T_GIVEN_X @ weak_x` | read detached by `AnchoredLabelGuess` state (§5.1) |
| `p̃(y)`, line 7 | 128-batch prediction window | — | `AnchoredLabelGuess` FIFO window, capacity 128, written with the *unaligned* anchor mean |
| `p(y)`, line 7 | labelled marginal | — | `AnchoredLabelGuess` EMA over observed one-hot `t`, decay `0.999` (`PData.update`) |
| `Normalize(q × p/p̃)` | alignment | — | `AnchoredLabelGuess.aligned`, with the reference's `1e-6` in both numerator and denominator |
| `q_b^{1/T}`, line 8 | sharpening | — | the same state, `T = 0.5`, as a power on probabilities *after* alignment |
| `W`, lines 14–16 | pooled shuffle and MixUp | — | `MixSpec("mixed", members=<the four rows above>, alpha=0.75, rule="max")` (§5.1) |
| eq. (3), first term | mixed labelled cross-entropy | `T_GIVEN_X @ mixed(X̂)` | `MixedTargetTreatmentNLL`, rows `t_observed`, `reduction="mean"`, weight `1.0` |
| eq. (3), second term | mixed unlabelled cross-entropy | `T_GIVEN_X @ mixed(Û_·)` | `MixedTargetTreatmentNLL`, rows `t_missing`, `reduction="mean"`, weight `1.5` ramped |
| eq. (4), first term | pre-mixup unlabelled loss | `T_GIVEN_X @ strong_x draw=0` | `AnchoredTargetTreatmentNLL`, rows `t_missing`, `reduction="mean"`, weight `0.5` ramped |
| `Rotate(u, r)` | the pretext transform | — | `ViewSpec("pretext_x", (ColumnRoll(shifts=(0,1,2,3), blocks="quarters"),), draws=1)` (deviation 4) |
| eq. (4), second term | rotation loss | `X_REPR -> PRETEXT_GIVEN_X` | `PretextTransformNLL` over `PretextHead`, rows `t_missing`, `reduction="mean"`, weight `0.5` |
| `λ_U`, `λ_U1`, `λ_r` | loss weights | — | `Weighted(..., 1.5 / 0.5 / 0.5)`; the first two ramped, the third not |
| `B`, source `mu = 1` | batch composition | — | `QuotaSampler(Quota("t_observed", 64), Quota("t_missing", 64))` |
| EMA `0.999` | reported model | — | `TeacherSpec(decay=0.999, role="evaluation", ema_applies_to_buffers=False)` |
| — | project-local outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed`, `reduction="population"` |
| — | project-local exact marginalisation | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing`, ramped weight `0.5` |

**Load-bearing arithmetic.** Six things a reviewer should check the code
against, because each is a place this method differs from its neighbours:

1. **There is no confidence gate.** `losses.confidence_threshold` is `n/a` and
   that is a *stated* value, not an omission: every unlabelled row trains at
   every step at full weight. This is the first card in the family for which
   that is true, `softmatch` having replaced the gate with a weight rather than
   deleted it.
2. **Alignment precedes sharpening**, and sharpening is a power on
   probabilities, `Normalize(q^{1/T})`, not `softmax(z/T)`. The two coincide
   only when `q` is an unmodified softmax of `z`, which after alignment it is
   not.
3. **`p̃(y)` is a 128-entry moving *window*, not an EMA**, and it is written
   with the *unaligned* anchor mean (`p_model.update(guess.p_model)`,
   `remixmatch_no_cta.py:154`). `softmatch`'s `ConfidenceGaussian` gets both of
   those the other way round, deliberately.
4. **The labelled rows are strongly augmented** (Alg. line 3). The labelled
   term is not an ordinary `ObservedTreatmentNLL` under a weak view.
5. **`λ' = max(λ, 1 - λ)`** per entry, so a mixed entry is always at least half
   its first source. This is what licenses the identity rule below.
6. **The pretext head reads `X_REPR`, and the pretext realisation reaches
   nothing else.** In the source the rotated images feed `classifier_rot` only.

**What a mixed entry is, and is not** (`BACKLOG.md` §15.1's four questions,
answered here because §5.1 builds the vocabulary that raises them):

- **Identity.** A mixed entry inherits the `row_id` of its first source. `λ' ≥
  1/2` makes that the dominant contributor by construction, and Tier 0 asserts
  the bound rather than trusting the sampler. `row_id` uniqueness therefore
  still holds within each mixed member.
- **Provenance.** A mixed realisation may not feed a pseudo-label action, an
  artifact, a teacher update, or evaluation. The compiler refuses all four;
  this is the concrete answer to "artifact-join semantics", and it is refused
  rather than defined because nothing in this card needs it.
- **Targets.** The mixed target is built by the objective from the same
  `(π, λ')` the features were mixed with. It is never `batch.t`: at a mixed
  realisation `batch.t` and `batch.t_observed` describe the first source alone
  and are unreadable, exactly as `core/loss.py`'s `treatment_at` makes hidden
  treatments unreadable. Tier 0 asserts an objective that reads them fails.
- **Populations.** The *pool* spans both populations — that is the mechanic,
  since at `mu = 1` and `K = 8` roughly nine in ten of a labelled row's
  partners are unlabelled — but each mixed member keeps the row population of
  its own source, so `Stage.rows`/`Objective.rows` composition (`DESIGN.md`
  §7.0) is unchanged.

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with
the recipe and tests once the card is `reviewed`.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.mixed_labelled_treatment_nll: p(t|x) @ view=weak_x params=student     # the partner's q half of the mixed target
    joint_fit.mixed_unlabelled_treatment_nll: p(t|x) @ view=weak_x params=student
    joint_fit.premixup_treatment_nll: p(t|x) @ view=weak_x params=student
    joint_fit.pretext_transform_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # ref impl: ly = tf.stop_gradient(guess.p_target)
  gradient_clipping: none                     # ref impl builds Adam().minimize with no clipping
  marginal_nll_grad_path: both                # project-local P5 choice

teacher:
  ema_decay: 0.999                            # §3.3; ref impl model(..., ema=0.999)
  ema_applies_to_buffers: false               # ref impl `utils.model_vars` returns TRAINABLE_VARIABLES only
  teacher_in_train_mode: false                # evaluation role only
  teacher_requires_grad: false

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.mixed_labelled_treatment_nll: mean        # reduce_mean over X'
    joint_fit.mixed_unlabelled_treatment_nll: mean      # reduce_mean over the concatenated U', all K+1 members at once
    joint_fit.premixup_treatment_nll: mean              # reduce_mean over U_hat_1
    joint_fit.pretext_transform_nll: mean               # reduce_mean over U_hat_1
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.mixed_labelled_treatment_nll: t_observed
    joint_fit.mixed_unlabelled_treatment_nll: t_missing
    joint_fit.premixup_treatment_nll: t_missing
    joint_fit.pretext_transform_nll: t_missing
    joint_fit.missing_treatment_marginal_nll: t_missing
    # The mixing pool additionally *reads* both populations across four declared
    # members (§3.2). Membership is declared on the MixSpec, not by widening any
    # objective's eligible set.
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.mixed_labelled_treatment_nll: 1.0         # eq. (3): the labelled term enters with coefficient 1
    joint_fit.mixed_unlabelled_treatment_nll: 1.5       # lambda_U, §3.3
    joint_fit.premixup_treatment_nll: 0.5               # lambda_U1, §3.3
    joint_fit.pretext_transform_nll: 0.5                # lambda_r, §3.3
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.mixed_labelled_treatment_nll: constant 1.0
    joint_fit.mixed_unlabelled_treatment_nll: ramp 0.0 -> 1.5 over 47 optimiser steps   # 1.5625% of the budget; deviation 6
    joint_fit.premixup_treatment_nll: ramp 0.0 -> 0.5 over 47 optimiser steps
    joint_fit.pretext_transform_nll: constant 0.5       # ref impl does NOT ramp w_rot
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 optimiser steps
  temperature: 0.5                            # T, §3.3
  sharpening: probability_power_temperature   # Normalize(q^(1/T)) applied AFTER alignment, not softmax(z/T)
  confidence_threshold: n/a                   # stated: ReMixMatch has no gate on any term

optimisation:
  optimiser: adam(beta1=0.9, beta2=0.999, eps=1e-8)   # ref impl tf.train.AdamOptimizer(lr) defaults
  lr: 0.002                                   # §3.3; ref impl FLAGS.set_default('lr', 0.002)
  lr_schedule: constant                       # the reference declares none
  weight_decay: 0.00004 decoupled per step (encoder, propensity and pretext head; weight matrices only, biases exempt)  # ref impl wd *= lr then v <- v*(1-wd) for 'kernel' vars
  batch_size: 128                             # B + mu*B = 64 + 64, derived from the QuotaSampler's quotas
  labelled_unlabelled_ratio: 1.0              # the source feeds xt_in [batch] and y_in [batch, K+1]; mu = 1, not 7
  total_steps_or_epochs: 3000 optimiser steps # source runs 2^26 images = 1,048,576 steps at batch 64 (deviation 6)

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
    pretext_head: linear X_REPR -> 4          # ref impl classifier_rot: one dense layer on `embeds`
  activation:
    mlp_encoder: elu
    tarnet_head: elu
    categorical_propensity: linear logits
    pretext_head: linear logits
  normalisation:
    mlp_encoder: row_l2
    tarnet_head: none
    categorical_propensity: none
    pretext_head: none
  dropout:
    mlp_encoder: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
    pretext_head: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
    pretext_head: glorot_normal, bias=0       # ref impl kernel_initializer=tf.glorot_normal_initializer()
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits
    pretext_head: 4 softmax logits

data:
  standardisation: x: none fitted on 'train'
  outcome_scaling: y: zscore fitted on 'train'
  treatment_encoding: n/a   # XTYBatch supplies integer classes 0..K-1; the guess state one-hots observed t itself
  split_protocol: one fixed project-local DGP, split train/test by the section 6.1 fixture; no CIFAR-10, SVHN or STL-10 protocol applies (deviation 7); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id
```

**On the paper-governed values the vocabulary has no key for.** `K = 8`,
`alpha = 0.75`, the 128-batch window, the `p(y)` EMA decay `0.999`, the `1e-6`
alignment epsilon, `redux='1st'` and `use_xe=True` are governed by the source
and bound by no key in `FIDELITY.md` §2. Adding keys for them is a framework
change (`DESIGN.md` §9.1) and none of them is a cross-card concept, so they
take the route `comatch.md` §4 established: constructor arguments of the shared
value objects with no defaults, printed by `plan_details()` so that they enter
the plan digest and the review surface.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Predict the *treatment* rather than an image class, and keep the reviewed causal stack (outcome NLL, exact marginalisation over missing `t`) beside eqs. (3)–(4). | `p(t \| x)` is a categorical classifier over `K` levels, so `p_model(y \| x; θ)` maps exactly; the paper's downstream task is the classifier this project needs as a propensity. `uda.md` §5.1, `comatch.md` §5.1 and `paws.md` make the same call, which keeps this card comparable with them. | No published number applies. §6 is paired and project-local, and every arm holds the outcome stack identical so the ReMixMatch terms stay attributable. |
| 2 | `judgement` | — | Replace flip/crop with `FeatureMask(0.1)` as the weak anchor transform, and the CTAugment operation set with `FeatureMask(0.1) + FeatureMask(0.5)` taken as eight independent draws of one strong `ViewSpec`. | There is no image geometry in a tabular XTY batch. The weak/strong pair is `fixmatch.md`'s reviewed one, unchanged, so a difference against the shipped family is a difference in objectives rather than in views. Eight draws of one family is what `ViewSpec.draws` already means. | The `K` copies are less diverse than eight CTAugment policies, which weakens exactly the mechanic §3.2.1 argues for. §6.2 reports the Bayes-label flip rate of both views before training and predeclares a `K = 1` arm so the value of extra copies is measured rather than assumed. |
| 3 | `framework-limitation` | `augmentation-vocabulary` | No CTAugment: no operation vocabulary with magnitude bins, no `m̂_i = m_i · 1[m_i > 0.8]` sampling rule, and no `ω = 1 - (1/2L) Σ\|p_model - p\|` feedback from probe predictions into bin weights at `ρ = 0.99`. Strong-view strength is a fixed pair of mask rates. | xty2 has no augmentation operation vocabulary to bin or to sample from (`DESIGN.md` §11.4), and CTAugment additionally needs a view whose parameters are a function of *model predictions on labelled probes* — the view contract is a pure function of `(batch, rng_key)` (`core/views.py`), and the source reads its EMA classifier to compute `ω`. `fixmatch.md` §5.10, `comatch.md` §5.3 and five other cards pay for the first half; this card is the first to name the controller half, which is the part `DESIGN.md` §11.4's ledger row already says it is not building. | Unknown sign, and larger here than for the gated cards: §4.4's "no strong aug." row costs `6.57` points, the second-largest entry in table 3. A null §6 result is consistent with the strong view being too weak to make eight copies informative, which is why §6.2 measures per-copy target agreement rather than only the end metric. |
| 4 | `judgement` | — | Replace the rotation pretext task with a four-class **column-roll** pretext task: cyclically roll the feature vector by `r ∈ {0,1,2,3}` positions, in quarter blocks of the unlabelled quota, and predict `r` from `X_REPR`. Weight, head placement, target rows (`Û₁`), reduction and the absence of a warm-up ramp are the source's. | Rotating a tabular row is undefined, but the *structure* of eq. (4)'s second term is not: a bijective, information-preserving re-indexing of the input coordinates whose identity must be recoverable from the shared representation. A column roll is that, and it is the smallest such group action available. Dropping the term instead would omit the mechanic §4.3 reports as necessary against collapse, which `CLAUDE.md` forbids doing by intuition. | Identifiability is a property of the fixture, not of the method: if the feature columns are exchangeable in distribution the roll is unlearnable and the term is inert noise. §6.2 makes above-chance pretext accuracy a *measured* guardrail with a card-amendment trigger, in the shape `uda.md` §6.2 rule 8 used for view flip rates. |
| 5 | `judgement` | — | Use the reviewed MLP/TARNet/propensity stack rather than WideResNet-28-2, adding only the one-layer pretext head the source adds. | Architecture swaps are orthogonal components in xty2 (`BACKLOG.md` §9), and holding the causal stack fixed is what makes the mechanism attributable across this family's cards. | No architecture-dependent published accuracy comparison is valid. |
| 6 | `judgement` | — | Train 3,000 optimiser steps rather than the source's 1,048,576, and re-base the `λ_U` / `λ_U1` ramp on the same *fraction* of the budget (`1.5625%`, so 47 steps) rather than on its step count (16,384, which exceeds the budget). | Every card here fixes a project-local step budget so that a difference between arms is attributable to the arm. Keeping the ramp's step count would leave both unlabelled weights below their nominal values for the whole run — the ramp would silently become the experiment. `comatch.md` §5.7 re-bases the cosine schedule for the same reason. | The 47-step ramp is short in absolute terms, so the unlabelled terms engage while the anchor is still near-uniform. §6.2 reports the anchor's entropy and `p̃(y)` trajectory so an unstable early target is visible rather than inferred. |
| 7 | `judgement` | — | One fixed project-local DGP (§6.1) with a skewed training treatment prior and a balanced held-out prior; no CIFAR-10, SVHN or STL-10 protocol, no label-fraction splits, no error rates. | The paper's evidence is three image benchmarks and none carries a treatment. The skew is deliberate: `softmatch.md` §6.4 records that a balanced `K = 2` target made its alignment mechanic nearly inert and could not validate the published method, and distribution alignment is one of the two mechanics this paper is named after. | §6 is a mechanism target and says so. The skewed-to-balanced shape is what gives `p(y)/p̃(y)` work to do; it also means a gain attributable to alignment does not transfer to a balanced-prior problem. |
| 8 | `judgement` | — | Keep the project-local missing-treatment marginal term at ramped weight `0.5` in every arm. | ReMixMatch is being tested as an addition to the causal stack, not as a replacement for it. | It can help independently of eqs. (3)–(4); every arm holds it fixed, so it cannot explain a paired difference. |
| 9 | `judgement` | — | Set the label budget to 64 observed treatments under MCAR, drawn as a quota of 64 from a population of exactly 64, where the source samples 250 CIFAR-10 labels with replacement. | This is the shipped fixture's budget and it keeps the card comparable with `fixmatch`, `uda`, `softmatch` and `comatch`. Because the quota equals the population, the labelled half of every batch is the same 64 rows in a different order; the source's labelled loader instead re-draws from 250. | The labelled term sees no sampling noise, so `p(y)`'s EMA converges to the exact observed marginal within a few hundred steps. §6.2 reports it against the fixture's true prior rather than assuming the estimate is unbiased. |

### 5.1 Framework additions made for this card

Six additions. **Two are load-bearing vocabulary and carry named second
consumers**; four are reversible objects of the kind `DESIGN.md` §11.2 says to
build for one card. This card is stopped for review largely because of the
first two rows.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `MixSpec` + the mixed realisations it plans + `MixingPlan` (the per-step `(π, λ')` the compiler hands to objectives reading a mixed realisation) | fidelity-bearing, **load-bearing vocabulary** — it is the first declaration that produces a *synthetic row* (`BACKLOG.md` §15.1) | the two `MixedTargetTreatmentNLL` terms | **MixMatch** (`BACKLOG.md` §2.2), whose Algorithm 1 lines 12–13 build the identical `W = Shuffle(Concat(X̂, Û))` and mix against it | Eq. (3) charges the model at rows that are convex combinations of two augmented entries drawn from a pool spanning both populations *and* `K+1` view draws. No arrangement of views, realisations or row populations expresses that: a `ViewSpec` transform sees one batch under one draw, so a within-draw pool would mix labelled rows only with labelled rows, where in the source roughly nine of ten of a labelled row's partners are unlabelled. **Shape check against MixMatch**, in the two places it could go wrong: (i) MixMatch's pool has `K = 2` unlabelled members and *no* weak member, so members are an explicit ordered tuple of `(view, draw, rows)` rather than "all draws of a view"; (ii) MixMatch's guessed label is the *average* over its `K` copies rather than an anchor, so the plan carries only `(π, λ')` and the target itself stays the objective's business. |
| `PRETEXT_GIVEN_X` port + `PretextHead` component + `PretextTransformNLL` + the `ColumnRoll` view transform | fidelity-bearing, **load-bearing vocabulary** (a port; `DESIGN.md` §2) | eq. (4)'s second term | **S4L** (`BACKLOG.md` §2.1), whose §3 "S4L-Rotation" attaches the same four-class self-supervised head to the same shared representation beside a supervised loss | Eq. (4)'s second term predicts a property of the *transform*, not of the row, so no existing port's value contract fits: `T_GIVEN_X` is `K` treatment probabilities and `RECONSTRUCTION` is `[B, D]` features. **Shape check against S4L**: S4L also runs its pretext head on *labelled* rows and reports an exemplar variant with a different class count, so the port is a categorical distribution of declared cardinality rather than a hard-coded four, and the objective takes its row population like any other rather than assuming `t_missing`. |
| `AnchoredLabelGuess` — stage-local objective state owning the 128-entry `p̃(y)` window, the `p(y)` EMA, and the once-per-step preparation of `q_b` | fidelity-bearing, reversible | all three treatment terms of eqs. (3)–(4), through the sibling read | not required (reversible) | `q_b` is a function of the last 128 batches, so it is not computable from one batch — `flexmatch.md`'s argument for a per-class counter, and `comatch.md`'s for a bank. Three objectives consume it, so preparation must be idempotent within a step and independent of declaration order, which is `freematch.md` §5.1's sibling-read mechanism reused unchanged. It is deliberately *not* `softmatch`'s `ConfidenceGaussian`: that object's alignment target is `u(K)` and never touches the pseudo-label, where this one's target is a learned `p(y)` and the alignment *is* the label. |
| `MixedTargetTreatmentNLL` — cross-entropy against a target mixed with the same `(π, λ')` as the features | fidelity-bearing, reversible | eq. (3), both terms | not required (reversible) | The target is a convex combination of a one-hot and a guessed distribution, taken across the pool; no existing objective can build it, and none may read `batch.t` at a mixed realisation (§3.2). |
| `AnchoredTargetTreatmentNLL` — ungated soft cross-entropy against the aligned, sharpened anchor | fidelity-bearing, reversible | eq. (4), first term | not required (reversible) | `ConfidenceMaskedConsistencyLoss` gates and sharpens from raw logits; ReMixMatch neither gates nor sharpens from logits (§3.2, arithmetic 1–2). Setting UDA's threshold to accept everything would leave the wrong sharpening path in place. |
| `Sharpening` gains the value `"probability_power_temperature"` | fidelity-bearing, reversible | this card | not required (reversible) | The closed literal holds `hard`, `softmax_temperature` and `none`. `Normalize(q^{1/T})` after alignment is a fourth, genuinely distinct operation, and naming it is what stops the card cross-check reading it as `uda`'s. |

**Two questions this card asks review to settle before code is written.**

1. **Is the mixing pool worth its vocabulary?** Row 1 is the `BACKLOG.md`
   §15.1 stress test, and it is the largest framework addition any card in this
   family has proposed. The alternative is a `framework-limitation` naming a new
   ledger key and a within-draw pool — which would be smaller, but would change
   what eq. (3) means, and `FIDELITY.md` §4.1 forbids disguising a missing
   fidelity-bearing abstraction as a permanent deviation to keep a diff small.
   This card takes §11.2's decision table at its word (fidelity-bearing plus
   load-bearing means *build it, with a second consumer's shape checked first*)
   and proposes to build it. Review may disagree.
2. **Is a column roll an acceptable stand-in for a rotation?** Deviation 4
   preserves the term's structure and substitutes its transform group. If review
   prefers the term omitted, it becomes a `framework-limitation` needing a new
   ledger key, the `PRETEXT_GIVEN_X` row above disappears, and §6.2's pretext
   arms go with it.

If implementation finds that `MixingPlan` cannot reach an objective without
changing the generic `LossTerm`/`TrainContext` contract, that is framework
vocabulary beyond what this table declares: amend the card and stop again
(`FIDELITY.md` §1).

## 6. Reproduction target

Two required paired comparisons, on one fixture, with initial parameters,
fixture, label budget, component graph, optimiser, schedules, declared views,
seeds and batch stream held identical across arms. The first asks whether the
crowd earns anything at all; the second is the paper's own table 3 ablation of
the mechanic the paper is named after, run on a fixture built so that mechanic
is not inert.

```yaml
reproduction:
  dataset: project-local seed-locked cluster XTY DGP (6 features, K=4, long-tailed train prior, balanced held-out prior), specified in 6.1
  variant: four paired fits - full ReMixMatch; supervised-only (lambda_U = lambda_U1 = lambda_r = 0); no distribution alignment (use_dm = false); no MixUp (alpha -> the identity pool); all other mechanics paired
  split: 1024 train rows with exactly 64 observed treatments; 2048 held-out rows with every treatment observed and a balanced prior
  metric: held-out balanced macro treatment NLL for student and evaluation EMA; held-out outcome NLL guardrail; terminal L1 distance between the model's predicted marginal and the true training prior; pretext accuracy; per-copy target agreement; mixed-entry lambda' distribution
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: >
    full/supervised-only held-out balanced macro treatment-NLL ratio < 1.0 in mean by at least one standard error, for the student and the evaluation EMA alike;
    full/no-alignment ratio < 1.0 in mean by at least one standard error on the same metric;
    held-out outcome NLL <= 1.05x the supervised-only arm;
    terminal |p_model_marginal - p_true|_1 strictly smaller with alignment than without, in mean by at least one standard error;
    every mixed entry's lambda' in [0.5, 1.0];
    pretext accuracy above 0.25 chance by at least one standard error - a miss voids the deviation-4 substitution and triggers a card amendment rather than counting against the method;
    no-MixUp and K=1 signs are reported, not predeclared
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP and the declared arms

**Fixture.** `benchmarks/common.py`'s `cluster_population` at `K = 4`, exactly
as `softmatch.md` §6.1 declares it: regular-simplex centres at pairwise
separation `1.8` rotated across the four signal columns, assignment mass `0.98`
on the generating cluster, outcome multipliers `(0.0, 1.0, 0.4, 1.6)`, training
prior `(0.55, 0.25, 0.13, 0.07)` and a uniform held-out prior. Replicate seeds
`s_r = 90000 + 100 r` for `r in 0..9`, with the family's stream offsets:
training generation `s_r+1`, held-out `s_r+2`, initialisation `s_r+6`, stage
seed `s_r+10000`. Exactly 64 training treatments are observed under a seeded
MCAR permutation; all held-out treatments and every outcome are observed. Fit
outcome scaling on the complete training population.

Two departures from `softmatch.md`'s use of the same fixture, both forced by
the source and not by convenience: the quota is `B = 64` observed and `mu*B =
64` missing (the source's `mu = 1`, not the family's 7), and the optimiser is
Adam at `2e-3` with no schedule rather than SGD with cosine. Both are held
identical across all four arms.

**Arms.**

| Arm | What it is | Tier |
|---|---|---|
| `remixmatch` | this card at the §4 declarations | 2, ten seeds |
| `supervised_only` | `lambda_U = lambda_U1 = lambda_r = 0`; the mixed labelled term, outcome NLL and marginal term remain | 2, ten seeds, paired |
| `no_alignment` | `use_dm = false`: Alg. line 7 skipped, line 8 unchanged — the paper's table 3 "No dist. alignment" row | 2, ten seeds, paired |
| `no_mixup` | the pool is the identity (`λ' = 1` for every entry), so eq. (3) is charged at the augmented entries themselves | 2, ten seeds, paired, reported not gating |
| `K = 1` | one strong copy; the paper's table 3 row costing `1.38` points | 1, one seed |
| `no_pretext` | `lambda_r = 0`; the paper's table 3 row costing `0.14` points | 1, one seed |
| `no_premixup` | `lambda_U1 = 0`; the paper's table 3 row costing `0.72` points | 1, one seed |
| `mean_redux` | the guess is the mean over the `K+1` copies rather than the anchor — MixMatch's rule, and the reference's `redux='mean'` | 1, one seed |

### 6.2 Predeclared evidence

**Tier 0 (invariants).**

1. Alignment matches a direct tensor calculation of
   `Normalize(q × (1e-6 + p(y)) / (1e-6 + p̃(y)))`, and is the identity when
   `p̃(y) = p(y)`.
2. `p̃(y)` is a 128-entry FIFO **window**, not an EMA: after 129 steps of a
   constant stream it holds exactly the constant, and after 128 steps of a
   changed stream no trace of the first stream remains.
3. `p̃(y)` is written with the *unaligned* anchor mean, and `p(y)` with the
   observed one-hot mean at decay `0.999`. Reading the aligned target into
   either buffer is what the assertion exists to catch.
4. Sharpening is `Normalize(q^{1/T})` on probabilities and is applied after
   alignment; at `T = 1` it is the identity, and it cannot increase entropy.
   `softmax(z/T)` on the same logits is asserted *different* once alignment has
   moved `q`, so the two sharpening paths cannot be confused.
5. Every mixed entry satisfies `λ' ∈ [0.5, 1]`, features and targets use one
   `(π, λ')`, and the entry's `row_id` is its first source's.
6. `batch.t` and `batch.t_observed` are unreadable at a mixed realisation; an
   objective that reads them raises rather than silently reading the first
   source's treatment.
7. The compiler refuses a mixed realisation feeding a pseudo-label action, an
   artifact, a teacher update, or evaluation.
8. The pool has exactly `|X̂| + |Û| = 64 + 64*(K+1)` entries at the declared
   membership, and each member's row population is its own source's.
9. `q` carries no gradient into any of the three treatment terms; the mixed
   features do carry gradient; the pretext realisation reaches only
   `PRETEXT_GIVEN_X`.
10. The guess preparation is idempotent within a step and the four losses are
    bit-identical under any declaration order of the objectives that read it.
11. No objective in this recipe applies a confidence gate; a batch of
    arbitrarily low-confidence anchors still trains every eligible row.
12. `plan.hyperparameters` matches every non-`n/a` §4 key, and `plan_details()`
    prints `K`, `alpha`, the window capacity, the `p(y)` decay, the alignment
    epsilon, `redux` and the mixing rule.
13. The state is fresh per stage execution, so a paired arm cannot inherit
    another arm's window.

**Tier 1 (one-seed smoke and mechanism arms).**

1. **Before training:** Bayes-label flip rate under the weak and the strong
   view (`BACKLOG.md` §6), and the pretext task's identifiability — a
   logistic probe on the untrained representation is *not* enough, so report
   4-way accuracy after a short pretext-only fit. At chance, deviation 4 fails
   and the card is amended rather than the result reinterpreted.
2. All four losses finite and falling; the full arm beats marginal-frequency
   treatment NLL.
3. Report the anchor entropy, `p̃(y)`, `p(y)`, and the mean agreement between
   the anchor's target and each of the `K` copies' own predictions, per step.
4. Each of the four one-seed arms in §6.1 against the full arm, changing one
   declaration each. No sign is predeclared for any of them; `K = 1`,
   `no_premixup` and `no_pretext` have published directions in table 3 and are
   reported against those directions as commentary, not as acceptance.
5. `mean_redux` is the arm that tests §3.2.1's actual claim — that anchoring
   beats averaging under strong augmentation. It is Tier 1 because table 3 does
   not ablate it directly.

**Tier 2.** Run the four ten-seed arms. Only the two required ratios plus the
outcome and alignment guardrails set `reproduced` versus `deviating`;
`no_mixup`, the pretext accuracy and the per-copy agreement are reported
mechanism measurements whose signs were not chosen in advance.

**What has run.** Nothing. This card is `draft` and stopped for review; no
recipe, tests or benchmark exist yet.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | — | — |

## 7. Unknowns

| Unspecified or source-dependent choice | Our choice | Basis |
|---|---|---|
| The two entry points disagree about `redux`: `remixmatch_no_cta.py:215` defaults `'swap'` (each copy's target is the *next* copy's own prediction), `cta/cta_remixmatch.py:72` defaults `'1st'` (the anchor's, as §3.2.1 describes). | `'1st'`. | The paper's text and figure 2 describe anchoring, and the CTA entry point is the one producing the headline numbers. Recorded here because `'swap'` is a materially different method sharing the name. |
| Is `p(y)` known a priori or estimated? §3.1.2 says estimated from labelled examples seen during training; `PData` prefers a dataset constant when one exists, which for CIFAR-10 it does. | Estimate it: an EMA over the observed one-hot treatments of the labelled quota at decay `0.999`. | `libml/layers.py:162-164`, the `has_update` branch. Using the fixture's true prior instead would hand the method a quantity the paper says it estimates. |
| Is `p̃(y)` an EMA or a window? | A 128-entry FIFO window, mean-then-renormalise. | §3.1.2 says "moving average ... over the last 128 batches"; `PMovingAverage` is a `[128, K]` buffer with a shift-and-append update. |
| Which distribution is written into the `p̃(y)` window — the raw prediction or the aligned target? | The raw prediction (`guess.p_model`). | `remixmatch_no_cta.py:154`. The rectified distribution is tracked separately and used only for plotting (line 88). |
| Does alignment or sharpening come first? | Alignment, then sharpening. | Alg. 1 lines 7–8; `guess_label` lines 51–59. |
| Is the labelled batch weakly or strongly augmented? | Strongly. | Alg. 1 line 3; the reference feeds `xt_in` from the same strong pool. |
| Which copy does the pre-mixup term charge? | The first strong copy, `Û₁`. | `remixmatch_no_cta.py:115` charges `logits_y[1]` against `ly[:batch]`. |
| Which copy does the pretext term transform? | The same first strong copy. | `remixmatch_no_cta.py:94`, `random_rotate(y_in[:, 1])`. |
| Is the pretext transform sampled i.i.d. per row? | No: the eligible rows are split into four equal blocks, one transform each. | `random_rotate` builds `l` as four concatenated constant quarters. |
| Are the unlabelled weights ramped, and over what? | `λ_U` and `λ_U1` ramp linearly over the first `1.5625%` of training; `λ_r` does not ramp. | `remixmatch_no_cta.py:70,114` against `warmup_kimg=1024` and `train_kimg = 1<<16`; `w_rot` is multiplied by no clip. |
| Cross-entropy or Brier for the unlabelled term? | Cross-entropy. | `use_xe=True` default, and §3.2.1's paragraph on replacing MixMatch's MSE. Table 3 puts the L2 variant `11.34` points worse. |
| Does the EMA cover normalisation buffers? | Parameters only. | `utils.model_vars` returns `TRAINABLE_VARIABLES`. Operationally inert in this graph, which has no BatchNorm. |
| Weight-decay reach. | Decoupled `lr * wd = 4e-5` per step on weight matrices of the classifier components, biases exempt. | `remixmatch_no_cta.py:69,158`: `wd *= lr`, then `v <- v*(1-wd)` for `model_vars('classify')` variables whose name contains `kernel`. Note this is not Adam's `weight_decay` argument. |
| Does the pretext head decay? | Yes. | Its scope `classify_rot` matches the `'classify'` prefix filter in the same line. |
| What is a tabular analogue of `Rotate`? | A cyclic column roll by `r ∈ {0,1,2,3}`, over a declared block of columns of one kind, bounds and mutability. | No paper-prescribed answer exists; deviation 4 states the substitution and §6.2 measures whether it is learnable at all. |
| Exact tabular weak/strong transforms. | Reuse the FixMatch-family feature masks. | Controlled comparison with the shipped family; deviation 2. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
