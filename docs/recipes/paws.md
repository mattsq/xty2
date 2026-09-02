# Recipe spec card: paws

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples](https://arxiv.org/abs/2104.13963) |
| Authors, year | Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Armand Joulin, Nicolas Ballas, Michael Rabbat; 2021 (ICCV 2021) |
| DOI / arXiv | [arXiv:2104.13963](https://arxiv.org/abs/2104.13963); ICCV 2021 |
| Version used | arXiv v3, 2021-07-30, read through the ar5iv HTML rendering. Section references in this card are to *this* card unless prefixed "the paper's". The paper's §3.1–§3.2 define the method and every symbol used below; its §4 states the two assumptions and the non-collapse argument; its §5 gives the ImageNet defaults; its §7 ablates the support-set composition and the me-max term; its appendix C gives the CIFAR-10 setting this card's hyperparameters are taken from, because CIFAR-10 is the small-data end of the paper and the only one whose numbers are within an order of magnitude of a tabular fixture. |
| Reference implementation | [`facebookresearch/suncet`](https://github.com/facebookresearch/suncet) @ `731547d727b8c94d06c08a7848b4955de3a70cea`, read directly at that commit: `src/losses.py` (`init_paws_loss`), `src/paws_train.py` (`train_step`, `init_model`, `init_opt`), `src/data_manager.py` (`ClassStratifiedSampler`, `TransCIFAR10`), `src/utils.py` (`WarmupCosineSchedule`) and `configs/paws/cifar10_train.yaml`. Every §7 row marked "reference implementation" names the file it comes from. |
| Reference impl. runnable? | Not attempted — it trains a WideResNet on CIFAR-10 for 600 epochs and nothing in this card depends on running it. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities `p(t | x)` and treatment-specific outcome means, after an encoder pretrained by predicting view assignments against a class-stratified set of *labelled* support rows.
- **Method claim:** a soft pseudo-label for an unlabelled view is the support set's label distribution weighted by temperature-scaled cosine similarity; sharpening the target and sampling the support set class-balanced is enough to make the two views agree without collapsing, with no confidence gate, no negatives, and no parametric classifier in the loss.
- **Not claimed:**
  - **No published number is reproduced.** The paper's evidence is ImageNet (75.5% / 66.5% top-1 at 10% / 1% labels) and CIFAR-10 (96.0% ± 0.2 paws-nn at 4,000 labels, appendix C). No dataset in it carries a treatment, and §6 below is a project-local mechanism target.
  - **Nothing about skewed propensity.** me-max maximises the entropy of the average prediction, which for `K` treatments is a pull toward a uniform *marginal*. Where `p(t = 1)` is far from `1/K` that is a misspecified prior, not a regulariser. §6.2 predeclares the skewed fixture that measures it; the card does not claim PAWS survives it.
  - **No inference claim for the pseudo-labelled rows.** A soft support-set label is a training signal, not an identified treatment (`BACKLOG.md` §7.4).
  - **Not a claim about instance-only pretraining.** `scarf` is `deviating` on this fixture's sibling. PAWS is the label-aware comparison `BACKLOG.md` §1 asks for, and a difference between the two cards is a difference of *fixture* as much as of method until §6.1's arms are run.

## 3. Equations and mapping

### 3.1 As published

Let `f_θ` be the encoder, `x̂_i` and `x̂_i^+` the anchor and positive views of an
unlabelled image, `z, z^+ ∈ R^{n×d}` their representations, and
`z_S ∈ R^{m×d}` the representations of a mini-batch of `m` labelled support
images with one-hot labels `y_S` (the paper's §3.2).

The similarity classifier is a soft nearest-neighbours assignment:

$$
\pi_d(z_i, z_{\mathcal S}) =
\sum_{(z_{s_j},\,y_j)\,\in\,z_{\mathcal S}}
\left(
\frac{d(z_i, z_{s_j})}{\sum_{z_{s_k}\in z_{\mathcal S}} d(z_i, z_{s_k})}
\right) y_j
$$

with `d(a, b) = exp( aᵀb / (‖a‖‖b‖ τ) )`, "the exponential temperature-scaled
cosine". For `l2`-normalised representations this is
`p_i := π_d(z_i, z_S) = σ_τ(z_i z_Sᵀ) y_S`, where `σ_τ` is the softmax with
temperature `τ > 0`.

The sharpening function, with temperature `T > 0`:

$$
[\rho(p_i)]_k := \frac{[p_i]_k^{1/T}}{\sum_{j=1}^{K}[p_i]_j^{1/T}},
\qquad k = 1,\dots,K
$$

With `p̄₂ := (1/2n) Σ_i (ρ(p_i) + ρ(p_i^+))`, the two-view objective is

$$
\frac{1}{2n}\sum_{i=1}^{n}
\Big( H(\rho(p_i^+), p_i) + H(\rho(p_i), p_i^+) \Big) - H(\bar p_2)
\tag{1}
$$

For the multi-crop objective, the reference implementation forms the me-max
average from every prediction view,

$$
\bar p_8 := \frac{1}{8n}\sum_{i=1}^{n}\sum_{k=1}^{8}\rho(p_i^k),
$$

not from the two large targets alone. The objective over two large and six
small views is

$$
\frac{1}{8n}\sum_{i=1}^{n}
\Big(
H(\rho(p_i^1), p_i^2) + H(\rho(p_i^2), p_i^1)
+ \sum_{k=3}^{8} H\big(\tfrac{\rho(p_i^1)+\rho(p_i^2)}{2},\, p_i^k\big)
\Big) - H(\bar p_8)
\tag{4}
$$

Six statements of the source are load-bearing for the mapping and are quoted
rather than paraphrased, because each is a place a reimplementation silently
differs.

- **The target is a constant of `θ`.** "Note that we only differentiate the
  cross-entropy loss terms with respect to the predictions `p_i` and `p_i^+`,
  and not the sharpened targets `ρ(p_i)` and `ρ(p_i^+)`." The stop-gradient is
  therefore attached to a *role*, not to a view: the same view supplies a
  gradient-carrying prediction in one term and a detached target in the other.
- **The support set is class-balanced by construction.** "To construct the
  support mini-batch in each iteration, we first sample a subset of classes,
  and then sample an equal number of images from each sampled class." The paper's
  §4 restates it as Assumption 1 — "Each mini-batch of labeled support samples
  contains an equal number of instances from each of the sampled classes" —
  and the non-collapse argument rests on it together with Assumption 2, that
  the target is sharpened and so is not uniform. Sharpening is not a
  convenience: "Empirically, we have observed that training without sharpening
  can result in collapsing solutions."
- **The support set is drawn from the unlabelled set, not beside it.** "Note
  that the images in the support set `S` may overlap with the images in the
  dataset `D`." The reference builds the unsupervised loader over the whole
  training split and the supervised loader over the labelled subset of the same
  split (`src/data_manager.py`, `TransCIFAR10(supervised=False)` skips the
  `keep_file` that selects the labelled subset), so a labelled image can be an
  anchor and a support sample in the same step.
- **Support sampling is with replacement across steps.** "Notably, we sample
  images with replacement. Therefore, while images in the same support
  mini-batch in a given iteration are always unique, some of the images may be
  re-sampled in the subsequent iteration's support mini-batch."
- **The support labels are smoothed.** "For the sampled support images, we also
  apply label smoothing with a smoothing factor of 0.1." `y_S` is therefore not
  one-hot in the implementation of eq. (1), which matters because `y_S` is the
  *output* space of the classifier: smoothing floors every `p_i`.
- **me-max is worth about eleven points where labels are scarcest.** The
  paper's §7, table 6: 63.8% with the regulariser against 52.9% without at 1%
  of ImageNet labels, and 73.9% against 73.6% at 10%. Its support-composition
  ablation (table 4) also fixes the direction of one design choice this card
  cannot exercise at `K = 2`: "Sampling more classes and fewer samples per class
  is better than the contrary."

Defaults, from appendix C and `configs/paws/cifar10_train.yaml` (CIFAR-10;
the paper's §5 ImageNet defaults appear in §7 below where they differ):
`τ = 0.1`, `T = 0.25`, me-max on, support mini-batch of 640 images as 10 classes × 64
images, each support image under 2 views, 256 unlabelled images per step, two
large crops at scale (0.75, 1.0) and six small crops at scale (0.3, 0.75),
label smoothing 0.1, WideResNet-28-2 trunk with a 3-layer 128-wide projection
head and no prediction head, LARS (`trust_coefficient=0.001`) over SGD with
momentum 0.9 and weight decay `1e-6`, learning rate warmed linearly from 0.8 to
3.2 over the first 10 of 600 epochs and cosine-decayed to 0.032 thereafter.

### 3.2 Mapping to xty2

A `pretrain` stage runs eq. (4) over `MLPEncoder → ProjectionHead` (`X_PROJ`),
with a `Quota` stratified on `t` supplying the support rows and a second quota
supplying the unlabelled anchors. A `joint_fit` stage initialises from it,
drops the projection head, and fits the reviewed causal stack. The support
labels are the *observed treatments* of the support rows, so the "classes" of
the paper are the levels of `t`.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `f_θ` | encoder | `X_RAW -> X_REPR` | `MLPEncoder` (the reviewed P5 backbone; deviation 5) |
| the projection head | the space `π_d` is computed in — the reference's `h`, since `use_pred_head: false` makes the prediction head the identity | `X_REPR -> X_PROJ` | `ProjectionHead`, 3 layers of 128, ReLU, `l2`-normalised output (deviation 5) |
| `x̂_i^1, x̂_i^2` | the two large views | `X_PROJ @ paws_large_x draw=0,1` | `ViewSpec("paws_large_x", draws=2)` over `FeatureCorruption(rate=0.25)` (deviation 2) |
| `x̂_i^3..8` | the six small crops | `X_PROJ @ paws_small_x draw=0..5` | `ViewSpec("paws_small_x", draws=6)` over `FeatureCorruption(rate=0.5)` (deviation 2) |
| `x_S`, `z_S` | the support images and their representations | `X_PROJ @ paws_large_x draw=0,1` at the support rows | `Quota("t_observed", 16, stratify="t")`, read under both large draws — the reference's `supervised_views = 2` |
| `y_S` | the support labels | — | `batch.t` at the support rows, one-hot with label smoothing 0.1; `t_observed` is true there by construction of the quota |
| `d(a,b)`, `σ_τ` | temperature-scaled cosine and its softmax | — | `SupportSetClassifier(temperature=0.1, label_smoothing=0.1, support_rows="t_observed")`, the shared value object §5.1 introduces |
| `p_i = π_d(z_i, z_S)` | the soft pseudo-label of one view | — | that classifier, evaluated inside both objectives below |
| `ρ(·)`, `T` | target sharpening | — | `sharpening=0.25` on the consistency objective, binding `losses.sharpening` |
| eq. (4), first two terms | large-view cross-entropies, targets swapped | `X_PROJ` at both large draws | `SupportSetPseudoLabelConsistency`, rows `t_missing`, `reduction="mean"` |
| eq. (4), the `k = 3..8` sum | small views against the mean large target | `X_PROJ` at all eight draws | the same objective: a small view's target is `(ρ(p^1) + ρ(p^2)) / 2` |
| "not the sharpened targets" | the stop-gradient | — | internal to that objective; `detaches` is empty because both large realisations *also* appear as predictions (§5.1) |
| `- H(p̄)` | me-max | `X_PROJ` at all eight draws | `MeanEntropyMaximisation`, rows `t_missing`, weight 1.0, `reduction="mean"` |
| `n` / `m` | unlabelled and support batch sizes | — | `QuotaSampler(Quota("t_missing", 128), Quota("t_observed", 16, stratify="t"))`; `n = 128`, `m = 32` rows read under 2 views |
| `argmax_k [π_d(z, z_S)]_k` (paws-nn) | the paper's test-time classifier | `X_PROJ` | not an objective: a §6 evaluation over the full labelled training pool as support |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed`, stage `joint_fit` |
| — (project-local) | treatment likelihood | `T_GIVEN_X` | `ObservedTreatmentNLL`, rows `t_observed`, stage `joint_fit` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing`, stage `joint_fit` |
| "the head is discarded" (implied) | what survives pretraining | — | `Stage("joint_fit", initialise_from="pretrain")`; `projection_head` is in no forward pass and in no `trainable` list of that stage |

Two rows of that table are the whole reason this card is not another `%Match`
variant, and they are worth stating as prose because a reviewer scanning the
YAML will look for them and not find them.

- **There is no gate and no `arg max`.** `losses.confidence_threshold` is `n/a`
  and `losses.sharpening` is a temperature rather than `hard`. Every anchor row
  trains at every step, weighted by nothing.
- **There is no parametric classifier in the pretraining loss.**
  `CategoricalPropensity` produces `T_GIVEN_X` for the causal stack in
  `joint_fit` and is not in the `pretrain` graph at all. The distribution over
  treatments that eq. (4) trains against is a similarity-weighted average of
  observed treatment labels, so `p(t | x)` is represented twice in this recipe
  by two different mechanisms, and §6 measures both.

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    pretrain.support_set_pseudo_label_consistency: none       # both large realisations are predictions; the target-role detach is recorded below
    pretrain.mean_entropy_maximisation: none                  # ref impl: me-max reads the *anchor* probs, with gradient
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets:
    pretrain.support_set_pseudo_label_consistency: target     # every target role in eq. (4) is detached; no whole realisation is
  gradient_clipping:
    pretrain: none                                            # paper and ref impl name none
    joint_fit: none
  marginal_nll_grad_path:
    joint_fit.missing_treatment_marginal_nll: both            # reviewed P5 choice; project-local addition

teacher:
  ema_decay: n/a                                              # PAWS maintains no EMA and no momentum encoder; the target is the same network, detached
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    pretrain.support_set_pseudo_label_consistency: mean       # eq. (4)'s 1/8n: the mean over anchor rows of the mean over that row's eight views
    pretrain.mean_entropy_maximisation: mean                  # a single batch scalar; `sum` or `population` would multiply -H(p_bar) by a row count it does not carry
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    pretrain.support_set_pseudo_label_consistency: t_missing  # the anchors. Support rows enter through `support_rows` and carry gradient without being eligible (§5.1)
    pretrain.mean_entropy_maximisation: t_missing
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    pretrain.support_set_pseudo_label_consistency: 1.0        # ref impl `src/paws_train.py`: loss = ploss + me_max
    pretrain.mean_entropy_maximisation: 1.0                   # eq. (1) and (4): -H(p_bar) enters with coefficient 1
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    pretrain.support_set_pseudo_label_consistency: constant 1.0   # the paper ramps neither term
    pretrain.mean_entropy_maximisation: constant 1.0
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature:
    pretrain.support_set_pseudo_label_consistency: 0.1        # tau, appendix C and configs/paws/cifar10_train.yaml
    pretrain.mean_entropy_maximisation: 0.1                   # the same value object, so the two cannot drift (§5.1)
  sharpening:
    pretrain.support_set_pseudo_label_consistency: 0.25       # T; soft, not `hard` — rho(.) is a temperature, not an arg max
    pretrain.mean_entropy_maximisation: 0.25                  # p_bar averages *sharpened* predictions
  confidence_threshold: n/a                                   # PAWS has no gate: every anchor row trains at every step

optimisation:
  optimiser:
    pretrain: adam(betas=(0.9, 0.999), eps=1e-08)             # deviation 4: the paper's LARS-over-SGD is a large-batch technique for a batch we do not run
    joint_fit: adam(betas=(0.9, 0.999), eps=1e-08)
  lr:
    pretrain: 0.001
    joint_fit: 0.001
  lr_schedule:
    pretrain: warmup_cosine start=0.25 ref=1.0 final=0.01 warmup=17 steps=1000   # the CIFAR ratios 0.8/3.2, 0.032/3.2 and 10/600 epochs, re-based on the step budget (deviation 6)
    joint_fit: constant 1.0
  weight_decay:
    pretrain: 1e-06 (all trainable components; biases exempt)  # appendix C's value; ref impl `init_opt` excludes parameters whose name carries 'bias' or 'bn'
    joint_fit: 1e-06 (all trainable components; biases exempt)
  batch_size: 160                                             # m + n = 32 + 128, derived from the quotas — see the note below on the stratified derivation
  labelled_unlabelled_ratio: 4.0                              # 128 anchors per 32 support rows; derived, never asserted
  total_steps_or_epochs:
    pretrain: 1000                                            # optimiser steps, never epochs (deviation 6)
    joint_fit: 3000

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                              # retained reviewed P5 TARNet backbone; deviation 5
    projection_head: [128, 128, 128]                          # ref impl `init_model`: fc1/fc2/fc3 at hidden_dim = output_dim = 128 for wide_resnet28w2
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: elu
    projection_head: relu                                     # ref impl: ReLU after fc1 and fc2, nothing after fc3
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: row_l2
    projection_head: row_l2                                   # deviation 5: the paper's head has BatchNorm on its hidden layers and no output normalisation; pi_d l2-normalises anyway
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    projection_head: 0.0                                      # appendix C: "WideResNet-28-2 trunk without dropout"
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    projection_head: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    projection_head: 128-d embedding, l2-normalised; no prediction head       # configs/paws/cifar10_train.yaml: `use_pred_head: false`
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'train'                  # the §6.1 DGP draws standardised features
  outcome_scaling: y: zscore fitted on 'train'                # held-out rows take the same fitted transform, never a refitted one
  treatment_encoding: n/a                                     # XTYBatch supplies integer classes 0..K-1; the support labels are one-hot with smoothing 0.1 inside the classifier
  split_protocol: one fixed project-local DGP, split train/test by the §6.1 fixture; no CIFAR-10 or ImageNet protocol applies (deviation 7); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id   # §6.1, and deviation 3 for what bounds it
```

**On the two derived keys.** `optimisation.batch_size` and
`optimisation.labelled_unlabelled_ratio` are derived from the quotas rather
than declared (`DESIGN.md` §7.3), and this is the first card whose sampler
stratifies. `QuotaSampler.batch_size` sums `Quota.size`
(`xty2/core/data.py`), while a stratified quota draws `size` rows *per level*
(`xty2/training/loading.py`), so today the derivation would print `144` and
`mu = 8` where the sampler runs `160` and `mu = 4`. The values above are the
sampler's. §5.1 records the correction this card requires before Tier 0 can
compare them.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Assign the *treatment* label rather than an image class, and fine-tune into the reviewed xty2 causal stack (outcome NLL, treatment NLL, exact marginalisation) rather than the paper's SuNCEt fine-tuning phase and paws-nn readout. | The paper's downstream task is supervised classification. The project-local question `BACKLOG.md` §1 asks is whether *class* structure — here, treatment structure — can be carried by a representation at all, after `scarf` showed instance-only structure did not reach the end task on the sibling fixture. `p(t \| x)` is a classifier, so the analogue is exact for the metric §6 leads with. The paper's own readout is not discarded: paws-nn is kept as a §6 guardrail, which is the only place a number of the paper's *shape* survives. | No published number applies. The comparison is internal: the same `joint_fit` stage, seeds and batches, with and without the pretrained initialisation. |
| 2 | `judgement` | — | Replace SimCLR augmentation and multi-crop with schema-aware empirical-marginal corruption: two large views at `rate=0.25`, six small views at `rate=0.5`. | There is no image geometry in a tabular XTY batch. `FeatureCorruption` is the reviewed tabular operation (`scarf.md` §5.2) and, unlike masking, it replaces a cell with a value the column actually takes, so a corrupted row stays in the data's support — which matters more here than for a gated method, because *every* anchor row trains. The relation the paper's design carries is that small views are lossier than large ones; the two rates preserve it. The crop-scale-to-corruption-rate mapping is ours and is in §7. | Directly defines the invariance learned. A view that destroys the treatment-predictive columns makes eq. (4) train the model toward the support set's marginal; §6.2 predeclares the Bayes-label flip rate of both rates on the fixture, measured before training, as the guardrail — `flexmatch.md` §5.2 is the precedent for why an undeclared strong view is a trap. |
| 3 | `judgement` | — | Draw 32 distinct support rows (16 per treatment) and 128 distinct missing-treatment anchors from disjoint populations. The paper's CIFAR run draws 640 distinct support images and 256 anchors from one training split, so a labelled image may also be an anchor. | The 64-label project fixture cannot supply 640 unique support images. The source is explicit that support images are unique within an iteration; its with-replacement rule applies only across iterations. Repeating a labelled row, or counting extra augmentations of it as extra support samples, would therefore be a second deviation rather than a faithful workaround. Disjoint populations preserve the meaning of observed labels and unique `row_id`; the reduced support size is the unavoidable project-local adaptation and is tested directly in §6.2. | The anchors are never also support rows and `π_d` uses 32 rather than 640 distinct supports, increasing its variance. §6.2 predeclares support-size sensitivity at 8 and 16 rows per level. Both arms share the fixture choice. |
| 4 | `judgement` | — | Adam at `1e-3` rather than LARS (`trust_coefficient=0.001`) over SGD at `lr = 3.2` with momentum 0.9. The warmup-then-cosine *shape* is kept, re-based on the step budget with the paper's ratios. | LARS exists to make a 4,096-image (ImageNet) or 256-image (CIFAR) batch trainable at a learning rate three orders of magnitude above ours; its layer-wise trust ratio is inseparable from that batch size, and `lr = 3.2` is not a number any other optimiser can be handed. This is the opposite call from `fixmatch.md` §5.9, and deliberately: FixMatch's table 7 reports the optimiser as part of its published finding, and PAWS reports no such comparison. Adam at `1e-3` is the pretraining optimiser `scarf.md` §4 already fixes, which is what makes the two pretraining objectives comparable at all. | The pretraining trajectory is ours, not the paper's. §6's pair shares the optimiser, so the PAWS *mechanism* stays attributable; a comparison against the paper's convergence rate ("4× to 12× less training than the previous best methods") is not available and is not claimed. |
| 5 | `judgement` | — | Retain the P5 encoder (3 layers of 200, ELU, row-`l2`) rather than WideResNet-28-2. Take the projection head's shape from the paper — 3 layers of 128, ReLU, affine output, no prediction head — but with `row_l2` output and **no BatchNorm** on its hidden layers. | Holding the causal stack fixed is what makes an addition attributable, and is the same decision `scarf.md` §5.3 and `fixmatch.md` §5.6 record. The head is not part of that stack, so its shape is the paper's. BatchNorm is the one part that is not: a running statistic makes a component's output depend on the *composition* of the batch it is evaluated in, and this recipe plans eight forward passes over one batch whose rows are two populations drawn by quota. `fixmatch.md` §5.11 records that the equivalence of per-realisation passes to the reference's single interleaved pass holds "only while no component holds batch-coupled state"; adding BN here would break it for the first time, silently, in the card that also introduces the most passes. | Removes whatever the head's normalisation buys. Since `π_d` `l2`-normalises both sides regardless, the effect is on the gradient scale reaching the head rather than on the loss (`ProjectionHead`'s docstring measures the same thing for SCARF). §6.2's collapse guardrail is what would show it going wrong. |
| 6 | `judgement` | — | Fixed budgets of 1,000 pretraining and 3,000 fine-tuning optimiser steps, rather than 600 CIFAR epochs. The warmup keeps the paper's 10/600 fraction (17 steps) and the cosine is re-based on the same 1,000. | Every card in this repository fixes a project-local step budget so that a difference between recipes is attributable to the recipe, and §6's target is a *paired* comparison in which both arms get the same budget either way. `scarf.md` §5.4 argues this at length and would be the row to revisit if the argument ever stops holding. Re-basing the schedule rather than dropping it is `fixmatch.md` §5.3's move, and here it costs nothing: the ratios are unit-free. | Pretraining length is chosen by us. The paper's CIFAR-10 result is a 600-epoch number and its ablation table 3 reports gains still accruing at 200; 1,000 steps is a different regime and §6 states its target in those terms. |
| 7 | `judgement` | — | One fixed project-local DGP (§6.1); no CIFAR-10 or ImageNet protocol, no 4,000-label split, no top-1 accuracy. | The paper's evidence is two image benchmarks under label-fraction splits, and neither carries a treatment. Reproducing that shape is a question about data plumbing, not about whether the mechanism is assembled correctly. | §6 is a mechanism target and says so. It is evidence against this port being miswired, not for the paper's claim. |

### 5.1 Framework additions made for this card

Four additions, all in the reversible half of `DESIGN.md` §11.2's table except
where noted. The stratified `Quota` this recipe needs is **not** among them: it
already exists, and `xty2/core/data.py` says it exists "because PAWS draws a
class-balanced support batch". This card is the consumer it was designed
against, and the first to run it.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `SupportSetClassifier` — a frozen value object holding `(temperature, label_smoothing, support_rows)` and computing `π_d` | fidelity-bearing, reversible | both `pretrain` objectives | not required (reversible) | eq. (4)'s loss and its me-max term must be computed from *the same* `p_i`. Two objectives each carrying their own `temperature` would let a card declare two, and per-objective card keys are keyed by `<stage>.<objective>`, so `DESIGN.md` §9.1's "conflicting values in one scope are errors" would not catch it. One shared object is the only thing that makes them provably equal. |
| An objective that reads predictions at a **second declared row population** (`support_rows`), outside its own `RowIndex` | fidelity-bearing, load-bearing | `SupportSetPseudoLabelConsistency`, `MeanEntropyMaximisation` | **SimMatch** (`BACKLOG.md` §2.9) | `DESIGN.md` §1.3 says an objective "gather[s] both data and predictions with that index", and eq. (4) cannot: its anchors are `t_missing` and its support is `t_observed`. `InfoNCEContrastive`'s note — negatives come from the eligible rows, because taking them from elsewhere would be "reading rows the objective is not entitled to by another route" — is exactly the rule this widens, so it is widened by *declaration*: the second population is a field, prints in the plan, and is intersected with `Stage.rows` the way `Objective.rows` is. **Shape check:** SimMatch "instantiated a labeled memory buffer to fully leverage the ground truth labels on instance-level"; it needs the same thing PAWS needs — an unlabelled row scored against *labelled* instances — but persisted across steps rather than drawn into the batch. So the contract must name the population it reads and must not name the batch: a field called `support_rows` survives the memory-bank case, a method called `support_in_batch` would not. Lifetime stays out of it, where `BACKLOG.md` §15.4 wants it. |
| `WarmupCosine` schedule: linear `start -> 1.0` over `warmup` steps, then cosine to `final`, as `src/utils.py`'s `WarmupCosineSchedule` | fidelity-bearing, reversible | `pretrain.lr_schedule` | not required (reversible) | `DESIGN.md` §6 has `Ramp` and `CosineDecay` and no way to compose them, and the ledger's `lr-schedules` row says the family is built "when a reviewed card names one". This card names one. §11.2 Q1/Q2 then answer `build`, and `FIDELITY.md` §4.1 forbids the alternative — writing a permanent deviation to keep a diff small — so this is an addition rather than a §5 row. |
| A correction, not an addition: `QuotaSampler.batch_size` and `.labelled_unlabelled_ratio` must count a stratified quota as `size × levels` | fidelity-bearing, reversible | this card | — | The derivation sums `Quota.size` while the loader draws `size` per level, so the plan would print `144` and `mu = 8` for a sampler running `160` and `mu = 4` (§4). No shipped recipe stratifies, so nothing is measured wrong today and no plan digest moves; the card that first stratifies is the one that has to fix it. The number of levels is `schema.treatment_cardinality`, which `QuotaSampler` does not hold — so the fix belongs where the compiler already resolves hyperparameters, not in the dataclass. |

**Support identity and the fixture limit** (deviation 3). The two large
augmentations of each support row are both used by the classifier, but they do
not turn one base row into two distinct support samples. The source requires
unique support images inside an iteration and permits replacement only across
iterations. Consequently PAWS neither requires nor discharges the
`batch-row-repetition` ledger item. A faithful support-heavy run needs a larger
labelled population, not repeated `row_id` values or a per-view expansion.

## 6. Reproduction target

The pair compares PAWS pretraining with no pretraining on a fixed project-local
DGP, holding `joint_fit` identical. The paper's own readout (paws-nn) and its
collapse conditions are carried as mechanism guardrails, so a null result can be
attributed to the representation rather than to the fine-tuning stage.
The paws-nn NLL scores the pre-`argmax` probabilities `π_d`; accuracy alone
applies the paper's `argmax` decision rule.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in 6.1
  variant: paired fit against the identical joint_fit stage with no pretraining, same seeds and same batches
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio, pretrained over unpretrained; paws-nn held-out treatment NLL over the full labelled pool as support, and terminal H(p_bar), as mechanism guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: NLL ratio < 1.0 in mean by at least one standard error; held-out outcome NLL within 1.05x of the unpretrained arm; paws-nn held-out treatment NLL below the fixture's marginal-prior NLL; terminal H(p_bar) at least 0.95 * log(K); mean cosine similarity of a row to its own second large view at least 0.2 above its mean similarity to the other anchors of the batch
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

Use `fixmatch.md` §6.1's generator, seed streams and 64-label MCAR budget
unchanged. The pretraining and no-pretraining arms share the population, the
observed-treatment mask, the `joint_fit` batches, the initial downstream graph
and the evaluation stream. Outcome standardisation is fitted on the complete
training population; all 2,048 held-out treatments and every outcome are
observed.

Two fixture facts this recipe adds, both checked rather than assumed:

- **The stratified quota must be fillable.** The support quota is 16 rows per
  treatment level. The fixture's marginal is `p(t = 1) ≈ 0.5`, so the scarcer
  level holds about 32 of the 64 observed treatments; the fixture asserts at
  least 16 per level in every replicate. A replicate that failed the assertion
  would be a card amendment — a smaller quota, or a larger budget — and never a
  re-seed.
- **Batch composition.** Every step draws 16 rows per treatment level from
  `t_observed` (32 rows, each read under both large views) and 128 rows from
  `t_missing`, in both arms; the unpretrained arm draws the same `joint_fit`
  batches and simply never runs `pretrain`.

### 6.2 Predeclared evidence

The criteria below were fixed while the card was `draft`. Tier 0 and Tier 1
pass at the implementation on PR #23, and the ten-seed downstream target has
since been run; §6.3 records it.

**Tier 0 (invariants).**

1. The stratified quota returns exactly 16 rows per treatment level, with no
   repeated `row_id`, and the support and anchor row sets are disjoint.
2. The sharpened targets carry no gradient, and both large realisations do —
   the role-level detach of §3.1, asserted rather than inferred from `detaches`
   being empty.
3. A small view's target equals `(ρ(p^1) + ρ(p^2)) / 2` for that row, and a
   large view's target is the *other* large view's, not its own.
4. The two objectives, given one `SupportSetClassifier`, compute bit-identical
   `p_i`.
5. `MeanEntropyMaximisation` returns `-H(p̄)` for a hand-computed batch,
   trains (its gradient is non-zero at a non-uniform `p̄`), and is unchanged by
   the row count of the batch under `reduction="mean"`.
6. Both objectives declare `batch_coupled=True`, and a stage holding either is
   refused `ExternalBatches`.
7. `plan.hyperparameters` matches every non-`n/a` key of §4, including the
   corrected `batch_size = 160` and `labelled_unlabelled_ratio = 4.0`.
8. Label smoothing floors the support label matrix at `s/K` and caps it at
   `1 - s + s/K`, so `π_d`'s output is bounded away from 0 and 1.

**Tier 1 (smoke fit).**

1. **Before training:** the Bayes-optimal treatment label's flip rate under
   `FeatureCorruption(0.25)` and `FeatureCorruption(0.5)` on the fixture
   (`BACKLOG.md` §6), reported for both view rates. This is a measurement, not
   an assertion; the rates in §4 were fixed before it was taken, and a flip
   rate that makes the small views uninformative is a card amendment.
2. Eq. (4)'s value falls, and the support classifier's held-out treatment
   accuracy rises, over the 1,000 pretraining steps.
3. **The me-max ablation.** The same fit with `mean_entropy_maximisation`
   weighted 0. Predeclared expectation, from the paper's table 6: lower terminal
   `H(p̄)`, and a higher chance of the collapse `ProjectionHead`'s docstring
   describes. We report it whichever way it goes.
4. **The skewed-propensity fixture.** A binary prior-skew fixture with
   `p(t = 1) = 0.15`, where me-max's pull toward a uniform marginal is
   misspecified. It uses a 160-label MCAR diagnostic budget, rather than the
   primary fixture's 64: at 64 labels the minority has about 10 examples and
   cannot fill the predeclared 16-per-level support quota. The replicate must
   still assert at least 16 observed rows per level before fitting. Report the
   terminal `p̄`, paws-nn error rate, and whether the me-max ablation *helps*.
   This is the card's own falsification test and §2 already refuses the claim
   it would settle; the larger diagnostic label budget is not used by the
   primary arm.
5. **Support-set size sensitivity.** The same fit at 8 and at 16 rows per
   level, to bound how much of any effect is deviation 3's small support rather
   than the mechanism.

**Tier 1 measurements, seed 90,010 (2026-08-29).** These are single-seed wiring
measurements, with no standard error and no reproduction claim.

| Arm | Eq. (4), first 50 → last 50 | paws-nn NLL, before → after | Accuracy, before → after | Terminal `H(p̄)` / predicted marginal |
|---|---:|---:|---:|---:|
| primary, 16 supports/level | 0.6134 → 0.5076 | 0.3429 → 0.2771 | 0.9092 → 0.9253 | 0.6908 / (0.4723, 0.5277) |
| me-max weight 0 | 0.6033 → 0.5101 | 0.3429 → 0.2786 | 0.9092 → 0.9214 | 0.6889 / (0.4735, 0.5265) |
| 8 supports/level | 0.6780 → 0.5163 | 0.3429 → 0.2877 | 0.9092 → 0.9175 | 0.6931 / (0.4708, 0.5292) |
| skewed, me-max on | 0.6204 → 0.3743 | 0.2570 → 0.2619 | 0.9033 → 0.9575 | 0.6923 / (0.7853, 0.2147) |
| skewed, me-max weight 0 | 0.6639 → 0.2246 | 0.2570 → 0.1222 | 0.9033 → 0.9619 | 0.4751 / (0.8345, 0.1655) |

The large/small corruption flip rates on the primary fixture were 0.1133 and
0.2422. The primary mechanism passes: consistency falls, paws-nn improves, and
entropy stays above `0.95 log 2`. The me-max ablation barely changes the
balanced fixture. On the skewed fixture it is plainly harmful: disabling the
uniform-marginal prior cuts NLL by more than half and moves the predicted
minority share from 0.215 toward the true 0.15. That is the falsification §2
predeclared, not a failure of the implementation.

**Tier 2, ten seeds.** Against the unpretrained arm, PAWS pretraining reached a
held-out `p(t | x)` NLL ratio of `0.934 +/- 0.025` and led in eight seeds of
ten, at an outcome-NLL ratio of `1.011 +/- 0.010` — inside the 1.05 within
which the card is willing to let the outcome head pay for the representation.
Every guardrail passed. paws-nn scored `0.286 +/- 0.004` nat/row against the
fixture's marginal-prior `0.698 +/- 0.002`, a ratio of `0.410 +/- 0.006`, at
`0.909 +/- 0.002` accuracy; terminal `H(p̄)` was `0.6922 +/- 0.0003`, within
0.001 of `log 2`; and a row's similarity to its own second large view exceeded
its similarity to the batch's other anchors by `0.242 +/- 0.013`. That last is
the only criterion a single replicate missed — one seed of ten gave 0.170 —
and the card states it on the mean, which clears 0.2 by three standard errors.

The paired guardrails are what make the downstream ratio readable: the
representation is not collapsed by either of the paper's two conditions, and it
classifies held-out treatments far better than the prior it was given, so the
0.934 is about the representation reaching `joint_fit` rather than about a
degenerate encoder being harmless. It remains a mechanism result on one
project-local DGP, and §2 still refuses every published number.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-02 | `049cec4` | held_out_treatment_NLL_ratio<br>held_out_outcome_NLL_ratio<br>paws_nn_over_marginal_prior_NLL<br>terminal_marginal_entropy<br>positive_view_alignment_gap | 0.934255 +/- 0.0246<br>1.01086 +/- 0.0101<br>0.410086 +/- 0.00573<br>0.692213 +/- 0.00033 nat<br>0.241944 +/- 0.0135 | yes |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Whether the *support* representations carry gradient. §3.2 says only that the targets do not. | They do. The support embeddings are trained through `π_d`'s denominator and its label-weighted numerator, on the anchor side only. | **Reference implementation:** `src/paws_train.py` slices `anchor_supports = z[:num_support]` with no detach, while `target_supports = h[:num_support].detach()`. So the labelled rows train the encoder even though eq. (1) is a mean over unlabelled rows, and `losses.eligible_rows` therefore does not describe which rows receive gradient. Recorded because it is invisible in the loss expression and is the difference between PAWS and a frozen-prototype method. |
| Which space `π_d` is computed in — the trunk's or the head's. | The projection head's output, `X_PROJ`, at pretraining and at the paws-nn evaluation alike. | **Reference implementation:** `encoder(imgs, return_before_head=True)` returns `(h, z)` where the head in question is the *prediction* head; the CIFAR config sets `use_pred_head: false`, so `z = h` and both are the projection head's 128-d output. The paper's linear-probe evaluation reads the trunk, but paws-nn is the readout this card keeps, and it is the training-time classifier. |
| How to map crop scale to a tabular corruption rate. Two large crops at scale (0.75, 1.0) and six small at (0.3, 0.75) are areas, not feature counts. | `rate=0.25` large, `rate=0.5` small. | Ours, by preserved-information fraction: a large crop keeps 75–100% of the image, a small one 30–75%, and the corruption rate is the fraction of columns replaced. The mapping is crude in one known direction — colour distortion and blur perturb the whole image on top of the crop, so the paper's views are lossier than their scale alone implies, and our rates are conservative. Fixed before any result was observed; §6.2 measures what they do to the Bayes label. |
| The number of classes per support batch, when `K = 2` and both levels are always present. | All of them. `Quota(stratify="t")` draws every level the eligible rows hold. | The CIFAR config does the same thing (`classes_per_batch: 10` of 10), so this is the paper's own small-`K` behaviour rather than a simplification. It does mean the one direction the paper's table 4 establishes — "sampling more classes and fewer samples per class is better" — cannot be exercised here, and is not evidence this card can borrow. |
| Whether `- H(p̄)` is differentiated, or is a diagnostic that only the targets feel. | Differentiated, through the anchor predictions. | **Reference implementation:** `rloss` is computed from `sharpen(probs)` — the anchor branch, outside the `no_grad` block that produces the targets — and returned into `loss = ploss + me_max`. A detached me-max would be a constant and `DESIGN.md` §4 would reject the objective outright, which is a useful cross-check that the reading is right. |
| The `targets[targets < 1e-4] *= 0` clamp: it is in the code and in no equation. | Keep it, in `plan_details()`. | **Reference implementation:** `src/losses.py`, commented "numerical stability". At `T = 0.25` the sharpening raises probabilities to the fourth power, so a class the classifier gives 0.02 lands near `2e-7` and is zeroed — the clamp bites on ordinary rows, not just pathological ones, and it leaves the target *not* summing to one. Keeping it is fidelity to the arithmetic that produced the paper's numbers; recording it is what stops the next reader treating a target row as a distribution. |
| The projection head's initialisation. | The reviewed CFRNet initialisation this repository's components use: `normal(std=0.1/sqrt(fan_in))`, zero bias. | Project convention (`scarf.md` §7 makes the same call). The paper names none, and `π_d` normalises its inputs, so the initialisation's scale affects the first steps and not the geometry. |
| Whether the fine-tuning stage reuses the pretraining optimiser state. | No — each stage constructs its own optimiser. | `DESIGN.md` §7.0: a stage begins from the recipe's initial graph state overlaid with the named checkpoint, and a `Checkpoint` carries parameters and buffers, not optimiser moments. |
| How many rows the pretraining sees relative to the fit. | The same 1,024 training rows, through two quotas. | The paper pretrains and fine-tunes on one training split, and the label scarcity here is on `t`, so every row is available to both stages. |
| Whether the eight views must all be computed for the support rows, which need only the two large ones. | Yes, and it is waste rather than a difference. | xty2 plans one forward pass per realisation over the whole batch (`DESIGN.md` §2.1), so the six small draws are computed for the 32 support rows and never read. The loss is unaffected; the cost is 192 unused row-evaluations per step. Recorded because the obvious "fix" — slicing the batch per realisation — would make forward passes depend on row populations, which is exactly the coupling deviation 5 refuses for BatchNorm. |
| No published target applies to a tabular causal adaptation. | A seed-locked project-local mechanism target with a paired no-pretraining ablation, plus the paper's own paws-nn readout and its two non-collapse conditions as guardrails. | The same discipline as `scarf.md` §6 and `fixmatch.md` §6: predeclare the DGP, the pairing and the tolerance before running anything, and make the mechanism — not a borrowed number — the thing that can fail. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Codex | 2026-08-29 |
| Plan diffed against §3.2 and §4 | Codex | 2026-08-29 |
