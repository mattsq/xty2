# Recipe spec card: doublematch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2, §3.2, and §4 to implement; §5 for departures;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [DoubleMatch: Improving Semi-Supervised Learning with Self-Supervision](https://arxiv.org/abs/2205.05575) |
| Authors, year | Erik Wallin, Lennart Svensson, Fredrik Kahl, Lars Hammarstrand; 2022 (ICPR 2022) |
| DOI / arXiv | [arXiv:2205.05575](https://arxiv.org/abs/2205.05575) |
| Version used | arXiv v1, 2022-05-11. §III defines the method — eq. (1) the supervised term, eq. (2) the pseudo-label term, eq. (3) the self-supervised term, eq. (4) the total, eq. (5) the rate schedule, eq. (6) the weight decay — and Algorithm 1 states one training step. §III-A gives the augmentations, §IV-D the hyperparameters, §V-A the ablation over the similarity function and §V-B the one over the pseudo-label term. |
| Reference implementation | [`walline/doublematch`](https://github.com/walline/doublematch) @ `6ea49949a5f412b2c4cb0bf078320751cad543e9`, **read directly** in the session that wrote this card: `doublematch.py` (the whole method), `libml/models.py` (what `embeds` is), and `README.md` (the per-dataset flags the paper's §IV-D states in prose). Built on the FixMatch codebase, and the parts this recipe inherits are the parts `fixmatch.md` describes. |
| Reference impl. runnable? | Not attempted. TensorFlow 1.x compat, CTAugment, and a CIFAR-scale WideResNet; nothing about the tabular port turns on running it. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities plus treatment-specific outcome means for causal contrasts.
- **Method claim:** DoubleMatch adds detached weak-feature versus projected strong-feature cosine consistency to FixMatch's supervised and pseudo-label losses.
- **Scope:** the image architecture and published benchmark are not reproduced. The TARNet outcome stack and tabular views are project-local.

## 3. Equations and mapping

### 3.1 As published

`B` is the labelled batch, `mu` the unlabelled ratio, `H` the cross-entropy,
`alpha` the weak augmentation and `beta` the strong one.

> Eq. (1): `l_l = (1/B) sum_{i=1..B} H(y_i, p_i)`, with `p_i` the predicted
> class distribution for weakly-augmented labelled image `i`.
>
> Eq. (2): `l_p = (1/(mu B)) sum_{i=1..mu B} 1{max(w_i) > tau} H(argmax(w_i), q_i)`,
> where `w_i` is the prediction on the weakly-augmented unlabelled image `i` and
> `q_i` the prediction on the strongly-augmented one. "We let the prediction on
> the weakly augmented image act as the teacher, meaning we consider `w_i` to be
> constant when back-propagating through this loss term."
>
> Eq. (3): `l_s = -(1/(mu B)) sum_{i=1..mu B} h(v_i) . z_i / (||h(v_i)|| ||z_i||)
> = -(1/(mu B)) sum_i cos(h(v_i), z_i)`, where `z_i` is the output of the
> penultimate layer for the *weakly* augmented image `i`, `v_i` the same for the
> *strongly* augmented one, and `h` a trainable dimension-preserving linear
> projection head. "Again, the prediction on the weakly augmented image acts as
> the teacher, so we consider `z_i` as constant when evaluating the gradient
> w.r.t. this loss term."
>
> Eq. (4): `l = l_l + l_p + w_s l_s`.
>
> Eq. (5): `eta = eta_0 cos(gamma pi k / (2 K))`, `k` the current step, `K` the
> total, `gamma in (0,1)` tuned per dataset.
>
> Eq. (6): `l_w = w_d (1/2) (||theta_f||^2 + ||theta_g||^2 + ||theta_h||^2)`,
> over the backbone `f`, the prediction head `g` and the projection head `h`.

Algorithm 1 fixes the order and the sharing: one weak pass and one strong pass
over the unlabelled batch produce `z_i` and `v_i`; `w_i = stopgrad(g(z_i))` and
`q_i = g(v_i)` are read off the *same* two passes, and `h` is applied to `v_i`
only. Eq. (3) therefore costs one linear layer and no extra augmentation or
forward pass, which is the paper's "minimal computational overhead".

Two notes on the transcription, both from reading the reference:

* The implemented self-supervised loss is `mean(1 - cos)`, not `mean(-cos)`.
  The `+1` is constant in the parameters and contributes no gradient, so the two
  train identically; they log differently. This port implements **eq. (3) as
  published**, for the same reason `InfoNCEContrastive` keeps the `- log n` that
  SCARF's denominator implies: the number in the log should be the number the
  paper's expression evaluates to. §7 records the offset so that a run compared
  against a reference training curve is not read as broken.
* Eq. (2) is written with a strict `>`; the reference gates on `>=`
  (`doublematch.py`, `mask = reduce_max(...) >= confidence`), which is what
  `PseudoLabelTreatmentNLL` does and what `fixmatch.md` already records.

### 3.2 Mapping to xty2

The recipe reuses FixMatch's weak/strong views, gate, sampler, optimiser, and EMA. `ProjectionHead` consumes strong features while `CosineFeatureConsistency` compares them with detached weak features from the same two forward passes.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `f` | backbone, up to the penultimate layer | `X_RAW -> X_REPR` | `MLPEncoder`, the shared P5 backbone, at torch's default initialisation rather than CFRNet's (deviation 9) |
| `g` | prediction head, the final layer of the classifier | `X_REPR -> T_GIVEN_X` | `CategoricalPropensity` |
| `h` | dimension-preserving linear projection head | `X_REPR -> X_PROJ` | `ProjectionHead(widths=(200,), normalisation="none")` — one affine layer, no activation and no output normalisation |
| `alpha(.)` | weak augmentation | — | `ViewSpec("weak_x")`, `FeatureMask(p=0.1)`, two draws |
| `beta(.)` | strong augmentation | — | `ViewSpec("strong_x")`, `FeatureMask(p=0.1)` then `FeatureMask(p=0.5)` |
| eq. (1) `l_l` | supervised cross-entropy on weak views | `T_GIVEN_X @ weak_x draw=1` | `ObservedTreatmentNLL`, rows `t_observed`, `reduction="mean"` |
| `w_i` | artificial label distribution | `T_GIVEN_X @ weak_x draw=0` | `PseudoLabelTreatmentNLL.target` |
| `argmax(w_i)`, `1{max(w_i) >= tau}` | hard label and gate | — | `sharpening="hard"`, `threshold=0.95` |
| `q_i` | strong-view prediction | `T_GIVEN_X @ strong_x` | `PseudoLabelTreatmentNLL.prediction` |
| eq. (2) `l_p` | gated pseudo-label cross-entropy | `T_GIVEN_X` at both views | `PseudoLabelTreatmentNLL`, rows `all`, `reduction="mean"` |
| `z_i` | penultimate features, weak view, **detached** | `X_REPR @ weak_x draw=0` | `CosineFeatureConsistency.target`, `stop_grad="target"` |
| `h(v_i)` | projected penultimate features, strong view | `X_PROJ @ strong_x` | `CosineFeatureConsistency.prediction` |
| eq. (3) `l_s` | negative cosine similarity, **every** unlabelled row | `X_PROJ` and `X_REPR` | `CosineFeatureConsistency`, rows `all`, `reduction="mean"` |
| `w_s` | self-supervised loss weight | — | `Weighted(..., weight=0.5)`, `Constant` (§7) |
| eq. (5) `eta_0 cos(gamma pi k / 2K)` | rate schedule, `gamma = 7/8` | — | `CosineDecay(steps=3000, phase=7/16)` |
| eq. (6) `l_w` | weight decay over `f`, `g` **and** `h` | — | `WeightDecay(5e-4, on_norm_and_bias=False, components=None)` |
| EMA of parameters (§IV) | the model the paper reports from | — | `TeacherSpec(decay=0.999, role="evaluation")` |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.pseudo_label_treatment_nll: p(t|x) @ view=weak_x params=student   # eq. (2): w_i is constant
    joint_fit.cosine_feature_consistency: x_repr @ view=weak_x params=student   # eq. (3): z_i is constant
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # both unlabelled terms detach the weak-view side; Alg. 1 lines 11 and 14
  gradient_clipping: none                     # paper names none; retained P5 choice
  marginal_nll_grad_path: both                # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.999                            # §IV: "an exponential moving average of the model parameters (with momentum 0.999)"
  ema_applies_to_buffers: false               # ref impl EMAs the trainable variables
  teacher_in_train_mode: false                # an evaluation copy; no-op for this architecture, declared anyway
  teacher_requires_grad: false                # never an optimiser target
  # role = evaluation. Nothing reads it during training: eqs. (2) and (3) both
  # take their targets from the current network (Alg. 1).

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: mean       # eq. (1) divides by B
    joint_fit.pseudo_label_treatment_nll: mean   # eq. (2) divides by mu B
    joint_fit.cosine_feature_consistency: mean   # eq. (3) divides by mu B
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.pseudo_label_treatment_nll: all    # FixMatch footnote 2: U includes the labelled rows without their labels
    joint_fit.cosine_feature_consistency: all    # eq. (3) is over every unlabelled row, gated by nothing
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.pseudo_label_treatment_nll: 1.0    # eq. (4) states no weight on l_p; ref impl flag wu defaults to 1
    joint_fit.cosine_feature_consistency: 0.5    # w_s, §IV-D and ref impl README: CIFAR-10's smallest label budget (§7)
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.pseudo_label_treatment_nll: constant 1.0
    joint_fit.cosine_feature_consistency: constant 0.5     # eq. (4): w_s is a constant, not a ramp
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a                            # eq. (3) is a cosine; the lambda of eq. (8) belongs to the softmax variant §V-A rejects
  sharpening: hard                            # eq. (2): argmax(w_i)
  confidence_threshold: 0.95                  # tau, §IV-D

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)    # §III-B and §IV-D
  lr: 0.03                                       # eta_0, §IV-D
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))   # eq. (5) with gamma = 7/8, K = our 3000 steps
  weight_decay: 0.0005 (all trainable components; norm and bias exempt)   # w_d, §IV-D; eq. (6) covers f, g and h, and the ref impl decays the `kernel` variables of the `classify` scope, which is all three
  batch_size: 512                                # B + mu B = 64 + 448, derived from the QuotaSampler's quotas
  labelled_unlabelled_ratio: 7.0                 # mu, §IV-D; derived from the same quotas
  total_steps_or_epochs: 3000                    # optimiser steps. The paper runs 352,000 (22,000 kimg at B = 64); see deviation 3

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                 # retained reviewed P5 TARNet backbone
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K   # the paper's g is one dense layer on the penultimate features
    projection_head: [200]                       # h: dimension-preserving, d -> d, one layer
  activation:
    mlp_encoder: elu
    tarnet_head: elu
    categorical_propensity: linear logits
    projection_head: relu                        # inert: a one-layer head applies no activation (Tier 0 asserts the module is a single Linear)
  normalisation:
    mlp_encoder: row_l2                          # the shared P5 backbone, unchanged
    tarnet_head: none
    categorical_propensity: none
    projection_head: none                        # eq. (3) normalises both sides itself; h is affine and nothing else
  dropout:
    mlp_encoder: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
    projection_head: 0.0
  initialisation:
    mlp_encoder: torch Linear default Kaiming-uniform   # deviation 9: CFRNet's 0.1/sqrt(fan_in) leaves ||f(x)|| at 0.011 and eq. (3)'s gradient carries 1/||.||
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
    projection_head: torch Linear default Kaiming-uniform    # uniform +/- 1/sqrt(fan_in); ref impl uses Glorot normal; §7
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'train'    # the §6 DGP draws standardised features
  outcome_scaling: y: zscore fitted on 'train'
  treatment_encoding: n/a                       # XTYBatch supplies integer classes 0..K-1
  split_protocol: one fixed project-local DGP, split train/test by the section 6 fixture; no CIFAR/SVHN/STL protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id  # deviation 7
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply DoubleMatch to categorical treatment assignment `p(t \| x)` and compose it with a causal outcome likelihood and exact marginalisation over the missing treatment. | The paper studies image classes. The project-local question is whether the rejected rows can be made to train the representation the propensity head reads, and whether that composes with the reviewed P5 stack. Identical to `fixmatch.md` deviation 1, and deliberately so: the two cards differ in one term. | No published image number applies. §6 measures the paired `w_s = 0` ablation, which by §III of the paper is FixMatch, so the comparison isolates eq. (3) and nothing else. |
| 2 | `judgement` | — | Replace the flip/translate weak augmentation and the CTA+Cutout strong one with schema-aware feature masking at 10% and 10%-then-50%. | There is no image structure in a tabular XTY batch. Inherited verbatim from `fixmatch.md` deviation 2, which is what makes the two cards' numbers comparable. | Defines the invariance eq. (3) is trained to hold. A strong view that destroyed the treatment-predictive columns would make the term train the encoder toward a degenerate direction; §6's collapse guardrail is what would show it. |
| 3 | `judgement` | — | Train for 3,000 optimiser steps rather than the paper's 352,000, with eq. (5)'s `K` set to the same 3,000. | The shared project-local budget, so that a difference between recipes is attributable to the recipe. | This is the deviation that costs the most here, and it is stated plainly: the paper's headline claim is about *training speed* (Fig. 3), which a fixed shared budget cannot measure. §2 therefore does not claim it. |
| 4 | `judgement` | — | Retain the P5 TARNet architecture — a 3x200 ELU encoder, a linear propensity head, the outcome head — rather than a WideResNet. `d` is 200, not the paper's 128/256/512. | Holding the causal stack fixed is what makes the DoubleMatch addition attributable, and is the same decision `fixmatch.md` deviation 6 and `mean_teacher.md` deviation 10 record. | One consequence was specific to *this* term and was not cosmetic: the encoder's `row_l2` made `z_i` a unit vector where the reference's `embeds` are unnormalised, and `h` is affine, so the term was evaluated on a differently-scaled input than the reference's. This row predicted an interaction and got it half right: §6.2 measured it, deviations 9 and 10 are what came of it, and what mattered turned out to be the encoder's *scale* rather than the normalisation this row pointed at. What remains here is the width and depth, which change nothing about eq. (3)'s form. |
| 5 | `judgement` | — | Retain P5's `Ramp(0.0, 0.5, 1000)` on the marginal-likelihood term while both DoubleMatch weights stay constant. | The ramp belongs to the reviewed P5 term. Eq. (4) states `w_s` as a constant and the reference exposes it as a scalar flag, so neither published term is ramped. | Early steps are dominated by the supervised terms and by eq. (3) — which, unlike eq. (2), is at full strength from step 0 because it has no gate to open. That ordering is the paper's and is what §6.2 watches. |
| 6 | `framework-limitation` | `augmentation-vocabulary` | No adaptive augmentation: the strong view's strength is fixed, where the reference stacks CTAugment. | Identical in substance to `fixmatch.md` deviation 10 — CTAugment learns per-operation magnitudes online from labelled probe images, and its operations have no tabular meaning. The prerequisite is a tabular operation set with magnitudes worth learning over; `FeatureMask`, `BoundedJitter` and `FeatureCorruption` are one scalar each. This card adds a second card paying for that row rather than a new argument. | Removes whatever adaptivity buys, from both unlabelled terms equally. It also removes a confound: the strong view's strength is a declared constant, so eq. (3)'s target is a fixed invariance rather than a moving one. |
| 7 | `framework-limitation` | `batch-row-repetition` | Set the §6 label budget to 64 rather than the 40 of the paper's scarcest CIFAR-10 setting, holding `B = 64` and `mu = 7` at the paper's values. | `XTYBatch.row_id` is unique because artifacts and provenance are keyed by it, so a labelled quota of `B` cannot be drawn from a 40-label population without repeating a row. Lowering `B` instead would deviate from a number the paper reports rather than from one this card chose. Same wall, same reasoning and same ledger key as `fixmatch.md` deviation 12. | Slightly more supervision than the paper's scarcest regime, applied equally to both arms of §6's pair. It moves the comparison's baseline, not the mechanism under test. |
| 8 | `judgement` | — | Do not implement the MSE (eq. 7) or softmax cross-entropy (eq. 8) alternatives to the cosine similarity. | §V-A evaluates all three and reports the cosine as clearly best, retuning `w_s` for each. Implementing the rejected two would be building an ablation nobody has asked for (`DESIGN.md` §11.2, Q1: no card §4 key moves). | None. If a later card wants the comparison on tabular data, the objective grows a `similarity` field then, and §V-A's numbers are the prior. |
| 9 | `judgement` | — | Initialise the encoder at torch's `nn.Linear` default rather than CFRNet's `normal std=0.1/sqrt(fan_in)`, which `fixmatch`, `tarnet` and `mean_teacher` all keep. Its `row_l2` output normalisation is **retained**, unchanged. | Eq. (3)'s gradient carries a `1/\|\|.\|\|` factor — `F.normalize`'s backward does, on whichever side is trained — so the term is only as well-scaled as the representation it is handed. The paper hands it a batch-normalised WideResNet's pooled activations, which are order 1 by construction. CFRNet's initialisation leaves this encoder's pre-normalisation activations at a norm of **0.011**, and `row_l2` passes `1/0.011` upstream: the term arrives about ninety times louder than the paper's, drives every row to one direction inside ten steps, and the run never recovers (§6.2). Torch's default puts the representation back at order 1 and the pathology disappears. This is the minimal change that restores the scale the mechanic is written for; it is *not* what an earlier version of this card did (see the row below). | It is the one place the causal stack is not held fixed against P5, so `fixmatch`'s and `tarnet`'s recorded numbers are **not** comparable to §6's — the same cost `fixmatch.md` deviation 9 records for adopting the paper's optimiser. It is not free either: at `w_s = 0` this initialisation is worse for the propensity than CFRNet's on this fixture (§6.2), so the recipe pays for a representation eq. (3) can use. §6's pair is within this recipe, so eq. (3) stays attributable. |
| 10 | `withdrawn` | — | ~~Drop CFRNet's `row_l2` from the encoder's output, on the grounds that a cosine target is the whole geometry of a unit-sphere representation and the term therefore takes the shortest route to collapse.~~ **Withdrawn.** The diagnosis was wrong and an adversarial review of this packet caught it. Holding `row_l2` fixed and changing only the initialisation removes the collapse entirely (§6.2), so the unit sphere is not what causes it; and the unnormalised encoder this row shipped *still* enters full collapse — concentration 0.9999 by step 43, 135 steps above 0.99 — and merely climbs back out around step 180. It fixed the symptom late rather than the cause. | The mechanism is scale, not geometry, and the row above is what replaced this one. Kept, struck through, because the wrong version was pushed and because the way it failed is the useful part: a plausible mechanism was written into a `judgement`, into `BACKLOG.md` as guidance for five later cards, and into a library docstring, without the one control — hold the geometry, change the scale — that would have refuted it. | The numbers this row was justified by are still in §6.2, labelled, next to the ones that refute it. |

### 5.1 Framework impact

`CosineFeatureConsistency` is a recipe-local objective over existing ports. The draft remains unsigned; its diagnostic history is recoverable from Git rather than carried in the active card.

## 6. Reproduction target

The planned pair changes only the feature-consistency weight from `0.5` to `0` and checks likelihood, gate, alignment, and collapse diagnostics.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in fixmatch.md 6.1 and reused unchanged
  variant: paired fit against an otherwise identical w_s = 0 ablation, same seeds and same batches
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio on both the EMA and the trained parameters, DoubleMatch over the w_s = 0 ablation
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: ratio < 1.0 in mean on both the EMA and the trained parameters (fixmatch.md 6.2's rule, and 6.2 below is why this card may not weaken it); held-out outcome NLL within 1.05x of the ablation; terminal alignment (mean cos(h(v), z)) above 0.5 while target concentration stays below 0.9 at *every* step, not merely at the end - the architecture deviation 10 withdrew passes a terminal reading of that guardrail and fails a trajectory one (6.2)
  seeds: 10
  report: mean_and_stderr
```

### 6.3 Result ledger


| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | not run | — |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| `w_s` for a tabular XTY fixture. §IV-D gives eleven values across four datasets and says a well-tuned `w_s` "will be largely correlated with the number of labeled training data" (0.5 / 1 / 5 for CIFAR-10 at 40 / 250 / 4,000 labels; the reference's README pairs them). | `w_s = 0.5` | The paper's value at the label *count* nearest ours (64 observed treatments), and the conservative end of the range it implies. **Transferred, not tuned**, and the distinction matters because a sweep did happen: five values of `w_s` were run against the shared P5 encoder and all five collapsed, which is evidence about deviations 9 and 10 rather than a search for a weight. No sweep has been run at the declared architecture, so 0.5 is the paper's number and not this fixture's best. Running one is the obvious follow-up; §6's tolerance is stated so that the claim does not rest on it. |
| `gamma` for a dataset that is not one of the four. §III-B: "we suggest tuning `gamma` for different datasets in order to minimize overfitting". | `gamma = 7/8`, i.e. `CosineDecay(phase=7/16)` | The value the paper uses for three of its four datasets, and identical to FixMatch's fixed schedule — which keeps this recipe's rate trajectory byte-identical to `fixmatch`'s, so §6's pair differs in eq. (3) alone. |
| The initialisation of `h`. | `torch.nn.Linear`'s default, uniform on `+/- 1/sqrt(fan_in)` | ref impl: `tf.layers.dense(..., kernel_initializer=tf.glorot_normal_initializer())`, which for a `d -> d` layer is normal with `std = 1/sqrt(d)`. Of the two initialisations xty2 offers, torch's default has `std = 0.577/sqrt(d)` and CFRNet's has `0.1/sqrt(d)`; the former is within a factor of two of the reference and the latter is ten times smaller. Deliberately *not* the CFRNet initialisation the outcome and propensity heads keep. It coincides with the encoder's since deviation 9, which arrived later and for an unrelated reason — this row was decided on the reference's kernel initialiser, that one on a measured gradient scale. |
| Whether `h` carries a bias. | Yes | ref impl: `tf.layers.dense` defaults to `use_bias=True`, and nothing in §III says otherwise; `ProjectionHead`'s `nn.Linear` does too. |
| How that bias is **initialised**. The paper says nothing; the two frameworks disagree. | `nn.Linear`'s default, `U(+/- 1/sqrt(fan_in))` | ref impl: `tf.layers.dense` defaults to `bias_initializer=zeros`, so the reference's `h` starts at **zero** bias and this port's does not. Measured at init on this fixture: `\|b\| = 0.551` against `\|Wz\| = 0.006`, so `h(v)` is 99.99% bias and `cos(h(v), b) = 0.9999`. Two consequences are stated rather than left to be found. `prediction_concentration` reads ~1.0 for a freshly initialised head and is therefore uninformative early in a run, unlike its target-side twin. And matching the reference here is not a free correction: zeroing the bias removes the floor that `\|h(v)\|` accidentally provides, the `1/\|h(v)\|` factor in the cosine's gradient is then unbounded, and the run diverges on this fixture (§6.2). It is the reference's value that would need the gradient-scale work, not ours — which is a debt this card is recording rather than paying. |
| Whether eq. (6)'s weight decay reaches `h`. | Yes, and biases are exempt | Eq. (6) writes `\|\|theta_h\|\|^2` explicitly, and the ref impl creates the projection head inside the `classify` variable scope and sums `l2_loss` over variables whose name carries `kernel` — so matrices are decayed, biases are not, exactly as `fixmatch.md` records for the classifier. |
| The `+1` offset. Eq. (3) is `-cos`; the reference computes `mean(1 - cos)`. | Implement eq. (3), `-cos` | The paper is the primary source and the offset contributes no gradient. Recorded here because a run compared against a reference training curve will sit exactly 1.0 lower, and that is not a bug. |
| Which xty2 quantity is "the output from the penultimate layer". | `X_REPR`, unnormalised | ref impl `libml/models.py`: `embeds` is the global-average-pooled activation and `logits = dense(embeds)`. `MLPEncoder -> X_REPR -> CategoricalPropensity` is the same two-node structure. The paper does not discuss the *geometry* of that activation because a WideResNet fixes it: BatchNorm on every block makes a rank-one batch impossible, which is the state §6.2 finds this encoder passing through. What a WideResNet also fixes is the *scale* of that activation, which is what deviation 9 restores here by other means; a normalisation layer inside the encoder would be the other answer and is a bigger change than this card needs. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
