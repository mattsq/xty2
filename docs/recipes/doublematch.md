# Recipe spec card: doublematch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> The recipe, the objective and the Tier 0/Tier 1 tests exist and pass, so the
> *code* is at what this vocabulary calls `smoke-passing`. The status stays
> `draft` because §8 is unsigned.
>
> `CLAUDE.md` rule 1 was **not** followed, and it is worth being precise about
> what that cost. The card was written before the code, but no reviewer read it
> in between, and the implementation pass then rewrote 198 lines of it —
> including adding an architecture change, which is exactly the decision the
> rule exists to put in front of an independent reader. An
> adversarial review was run against the finished packet instead, and it
> overturned three claims this card had made about its own central measurement:
> that the collapse was caused by the unit-sphere geometry (it is caused by the
> initialisation scale — §5.10 is the withdrawn row and §5.9 what replaced it),
> that the recipe it then declared does not collapse (it does, for 135 steps),
> and that the Tier 1 test guarding the difference could tell the two
> architectures apart (it could not). All three are corrected below and
> attributed where they were wrong, and the architecture the packet first
> shipped is `§5.10`, kept and struck through. A card whose
> load-bearing judgement needed an adversarial pass to survive is a card whose
> §8 should be signed by someone who was not in that loop.

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

Reading the implementation settled four things the paper's prose leaves open,
and each is marked `ref impl` in §7 rather than folded into the transcription:
the projection head is a biased `tf.layers.dense` with a Glorot-normal kernel;
its parameters live inside the `classify` variable scope, so eq. (6)'s weight
decay reaches them exactly as the paper says; the implemented self-supervised
loss is `1 - cos` where eq. (3) is `-cos`; and `embeds` is the
global-average-pooled penultimate activation, with the classifier being one
dense layer on top of it — which is the structure this port maps onto.

## 2. Estimand and claim

- **Estimand:** unchanged from `fixmatch.md`. The FixMatch half estimates the
  categorical propensity `p(t | x)`; the retained causal stack estimates
  `p(y | x, t=k)` and its means `mu_k(x)`, whose contrasts identify conditional
  treatment effects under consistency, positivity and conditional
  exchangeability. **The self-supervised term estimates nothing.** It is a
  regulariser on `X_REPR`, and it is the only term in this recipe with no
  statistical quantity of its own — which is precisely why it can be applied to
  rows no likelihood is entitled to.
- **Claim:** DoubleMatch adds a third loss to FixMatch: the strongly-augmented
  row's penultimate features, passed through a trainable linear head, are
  trained to point in the same direction as the *weakly* augmented row's
  detached features (eq. 3), on **every** unlabelled row rather than only the
  ones the confidence gate retains. The paper claims this reaches higher
  accuracy in roughly a third of FixMatch's training steps, that it is new
  state of the art on CIFAR-100 at every label budget, and — §V-B — that with
  enough labels the *pseudo-label* term can be dropped entirely at almost no
  cost. This card claims only that the mechanism is faithfully assembled around
  `p(t | x)` and `X_REPR` in xty2, and that on the fixed project-local target in
  §6 the term does what eq. (3) says it does — it reaches a real alignment on
  every row, including the ones the gate rejects, without the representation
  collapsing at any point in the run. That last clause is a trajectory-wide
  measurement (§6.2) and it took two attempts to earn: the architecture an
  earlier version of this card declared spends 135 steps at a concentration
  above 0.99 before recovering, and the Tier 1 test guarding it averaged the
  last hundred steps of a 3,000-step run and reported no collapse at all. An
  adversarial review of this packet caught both, and §5 deviation 9 is what
  replaced the wrong fix. Whether the term *improves* held-out treatment
  prediction over the same fit with `w_s = 0` (which is FixMatch's loss, by the
  paper's own sentence in §III, on this recipe's encoder) is **not** claimed on
  the evidence in §6.2: two initialisation draws of the declared architecture
  put the EMA comparison on opposite sides of 1.0, and that is a ten-seed
  question.
- **Not claimed:** no image number is claimed, and no training-time claim is
  made at all: §6 fixes the step budget for attributability (deviation 3), so
  "reaches the same accuracy in a third of the steps" is a claim this protocol
  is built not to be able to test. Nor is it claimed that this recipe beats
  `fixmatch` — measured on §6.1's fixture it does not, because the encoder
  initialisation eq. (3) needs (deviation 9) is worse for the propensity head
  than CFRNet's, by more than eq. (3) recovers (§6.2). The
  claim is about the term inside the architecture the term requires.

Three things worth stating before the mechanics, because they are why this
recipe was picked rather than the next one in the sequence.

**This is the row-population test `BACKLOG.md` §2.6 asks for, and the framework
answers it with nothing new.** The backlog's sketch is three terms with three
different eligible sets, and the semantic test is that "row eligibility for one
loss must not determine eligibility for all losses". `Objective.rows` has been
per-objective since P3, so the recipe below states it directly. What
DoubleMatch actually cost was one objective — no port, no executor, no row
population, no schedule type (§5.1).

**It is the natural follow-up to `scarf.md` §6.2, not merely the next
composite.** SCARF's port found that contrastive pretraining learned a
representation whose frozen probe carried treatment structure, while the
end-to-end gain the paper's protocol predicts was absent at every fine-tuning
budget tried — with the mechanism diagnosed as instance discrimination
spending its capacity pushing apart exactly the same-treatment rows the
propensity head wants together. Eq. (3) is the same family of idea with that
term deleted: it has **no negatives**. If the SCARF diagnosis is right, a
negative-free feature-consistency loss should not pay the same cost. §6.2
records how far the measurement got: the term learns without collapsing the
representation, and the propensity comparison it was supposed to settle came
out mixed.

**The two structural limitations of `fixmatch.md` §2 are inherited unchanged**,
because this recipe inherits eqs. (1) and (2) unchanged. The confidence gate is
still in tension with positivity and still does not fail safe under overlap;
the EMA is still a reporting device whose number can improve while the network
it averages gets worse. DoubleMatch narrows neither. What it changes is what
happens to the rejected rows, and only that.

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

Four mapping decisions carry the fidelity of this port.

**`z_i` is the same weak pass the pseudo-label reads, not a third view.**
Algorithm 1 computes `z_i = f(alpha(x~_i))` once and derives both `w_i` and
eq. (3)'s target from it. The recipe therefore points `CosineFeatureConsistency`
at draw 0 of `weak_x` — the draw `PseudoLabelTreatmentNLL` already targets — so
the two terms share one forward pass, which is what makes the paper's overhead
claim true here as well. Pointing it at draw 1 would compile, cost a pass, and
silently be a different method.

**`h` is applied to the strong branch only, and that asymmetry is the loss.**
Eq. (3) is not symmetric: gradient flows through `h(v_i)` and stops at `z_i`.
So `CosineFeatureConsistency` names a `prediction_port` and a `target_port`
separately — it is the first objective in the repository whose two sides are
different ports — and derives `detaches` from `stop_grad` in the usual way.
Without the predictor the term would be an ordinary consistency loss with a
known failure mode: `SimSiam`'s ablation, which the paper cites for exactly this
design, is that dropping the predictor collapses the representation.

**Rows for eq. (2) and eq. (3) are both `all`.** FixMatch's footnote 2 puts every
labelled row into `U` as well, without its label, and DoubleMatch inherits the
framework and the codebase. Both unlabelled terms therefore average over the
same population, which keeps their two denominators equal to each other, as
they are in the paper. Both are `(1 + mu) B = 512` rather than `mu B = 448`,
because in xty2 one batch holds both populations: the ratio `l_p : l_s` is the
paper's exactly, and `w_s l_s : l_l` is off by `512/448 = 1.14`. Inherited from
`fixmatch.md`'s reading of footnote 2 and applied to both unlabelled terms
identically, so it is one convention rather than two.

**Eq. (3) needs no gate, and that is the whole method.** The paper's title claim
is that the rejected rows still train something. In xty2 that is not a mechanism
at all: `PseudoLabelTreatmentNLL` masks per row inside its own arithmetic,
`CosineFeatureConsistency` does not, and both are handed the same `RowIndex`.
`BACKLOG.md` §2.6 predicted this would be the interesting test; it is the
uninteresting one, and §5.1 records that as the result rather than as an
absence.

## 4. Mechanics checklist

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

`architecture.activation[projection_head]` is declared and inert: §III's `h` is
one layer, and a head of one layer inserts no activation module, so the field
names a transform that never runs. It is declared rather than left to a default
because §9.1 admits no silent defaults, and Tier 0 asserts the built module is a
single `nn.Linear` so the declaration cannot quietly become load-bearing. This
is not a §7 unknown — the paper is explicit that `h` is linear — and it is not a
deviation, because the computed function is exactly the paper's.

One line of that block is the only place this recipe departs from the shared
P5 backbone, and it is deviation 9: `mlp_encoder` is initialised at torch's
default rather than CFRNet's. Everything else — widths, activation,
normalisation, dropout, the outcome head, the propensity head — is the stack
`fixmatch` and `tarnet` run.

The two `ViewSpec`s are `fixmatch`'s: §III-A says "We follow one of the
augmentation schemes used in FixMatch". Their *scalars* are imported —
`WEAK_MASK_RATE`, `STRONG_MASK_RATE` and `PRESERVED_FIELDS` have one home — but
the two `ViewSpec` constructions, the transform ordering and `draws=2` are
restated here, so there are two places to edit one reviewed decision. Tier 0
compares the compiled plans of the two recipes line by line, which is what
catches a divergence; exporting the specs themselves would be better and is not
done.
`weak_x` is `FeatureMask(p=0.1)` with two draws; `strong_x` is that transform
with `FeatureMask(p=0.5)` layered on. Both preserve `t`, `y`, `t_observed`,
`y_observed`, `row_id`, `fold_id` and `weight`, and neither claims to preserve
`x`. A schema with derived features must supply recompute rules or the views are
rejected at compile time.

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

### 5.1 Framework additions made for this card

**One objective, and nothing else.** `CosineFeatureConsistency`
(`xty2/objectives/feature_consistency.py`) is the only thing this recipe added
to the repository. No port, no component, no view transform, no schedule type,
no executor, no row population, no card key. An objective is the extension
point `DESIGN.md` §11 names for exactly this case — step 4 of `BACKLOG.md`'s
post-P12 workflow, "add the smallest missing component/objective/view if
necessary" — rather than a framework concept, so this section's table is
**None.** by its own definition, and the paragraphs below are the answer.

The one shape decision inside it is worth recording because it is new
vocabulary in a small way: the objective names `prediction_port` and
`target_port` **separately**, where `ConsistencyLoss` and `InfoNCEContrastive`
each take one `port` and two realisations. Eq. (3) forces it — the two sides of
the cosine are `X_PROJ` and `X_REPR`, and no rearrangement of one port over two
realisations expresses that. It stays inside the objective, so nothing else in
the framework has to be written against it.

Two things this card was expected to need and did not, recorded because the
expectation is in `BACKLOG.md` and a silent absence would read as an oversight:

* **A row-population mechanism for "rejected by the gate".** §2.6 sketches
  `t_missing & confident` as a row selector. It is not one here, for the same
  reason `fixmatch.md` gives: the gate is a per-row mask *inside*
  `PseudoLabelTreatmentNLL`, so eq. (2)'s denominator can keep counting the
  rows it rejects (which the reference does, and which makes the gate act as
  the curriculum §2.2 of FixMatch says it does). A `confident` population
  would be a second, contradictory implementation of the same gate.
* **A second projection-head component.** `ProjectionHead` was built for SCARF
  with `widths`, `activation`, `normalisation` and `dropout` as declared
  fields. DoubleMatch's `h` is `widths=(200,)`, `normalisation="none"`, and the
  activation is inert at one layer. That is a component reused across two
  recipes with different papers behind them, which is the outcome §11.2 hopes
  for; the inert field is the price and §7 records it.

## 6. Reproduction target

**This target is measured without two mechanics the paper states** (§5.6,
CTAugment; §5.7, the 40-label budget), and neither can be discharged from this
card — both are ledger debts shared with `fixmatch`. Neither touches eq. (3),
which is the term under test.

No published number can validate this port: the inputs, the labels, the
architecture, the budget and the metric all differ, and the estimand is a
treatment assignment rather than an image class. The target is a fixed
project-local *mechanism* target.

The DGP is `fixmatch.md` §6.1's, unchanged and reused deliberately: same
fixture, same budget, same seeds, same optimiser, same quota. The `w_s = 0` arm
of this pair is this recipe with eq. (3) switched off, which §III of the paper
says is FixMatch — "our loss function is identical to that used in FixMatch when
`w_s = 0`" — and Tier 0 asserts exactly how far that goes: dropping the term and
the projection head leaves a plan that differs from `fixmatch`'s in five lines.
Four are identity — its name, its card, and one sentence of prose about the
fixture that the plan prints twice. The fifth is deviation 9. The pair is therefore *within* this recipe rather
than across the two, and eq. (3) is the only thing that varies inside it.

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

### 6.1 Fixed DGP

As `fixmatch.md` §6.1, in full and without modification: the two-cluster
generator, the 0.02/0.98 cluster-conditional propensity, the 64-row MCAR
treatment budget, the z-scored outcome fitted on the training rows, and the
`B = 64` / `mu B = 448` quota per step. Replicates are indexed `r in {0..9}`
with base seed `s_r = 90000 + 100 r`, and both arms share every stream.

The one thing this card adds to that fixture is what it measures on it. Eq. (3)
has no gate, so mask rate and impurity say nothing about whether it is working;
the two diagnostics that do are the ones `CosineFeatureConsistency` emits:

* **alignment** — mean `cos(h(v_i), z_i)` over the eligible rows. Not a
  diagnostic at all: it is exactly `-value`, so the term's own logged number is
  the alignment and a second copy of it would be noise.
* **`prediction_concentration`** and **`target_concentration`** — the norm of
  the mean unit vector on each side, and the objective's only two diagnostics.
  This is the collapse detector, and it exists because the loss cannot be one:
  a representation that maps every row to the same direction attains the *best
  possible* `l_s`, and eq. (3) has no negatives to punish it. Isotropic
  embeddings score about `1/sqrt(n)`; a fully collapsed batch scores 1. The
  two sides are reported separately because they fail differently, and §6.2 is
  where that stopped being a hypothetical.

### 6.2 What the Tier 1 fixture already shows

Tier 1 is not Tier 2 and these are **single-seed** numbers, not a result. They
are recorded because they wrote deviations 9 and 10, and because the way they
were first read is a caution worth keeping.

Provenance matters here and is stated per block, because an earlier version of
this section mixed two of these paths without saying so. **Block A** is the
recipe as `doublematch(schema)` builds it. **Block B** replaces the encoder
after construction, which consumes RNG in a different order and therefore
initialises everything differently; it is the only way to hold one encoder
field fixed and vary another, so it is what the comparisons use. Numbers from
the two blocks are comparable *within* a block and not across.

**Block B — the measurement that chose the encoder.** Three encoder
configurations, each paired against its own `w_s = 0` ablation, same fixture,
same seeds, 3,000 steps:

| encoder | `w_s` | EMA NLL | student NLL | `l_s` | max target conc. | steps above 0.99 |
|---|---|---|---|---|---|---|
| `row_l2` + CFRNet — the shared P5 backbone (600 steps, **Block A path**; the collapse is total by step 10, and Tier 1's 400-step Block B arm reproduces it) | 0.5 | 0.6962 | — | -1.0000 | 1.000 | every step |
| `row_l2` + **torch** — *declared* | 0.5 | 0.3398 | 0.3703 | -0.640 | 0.666 | 0 |
| `row_l2` + **torch** | 0 | 0.3540 | 0.4014 | -0.027 | 0.616 | 0 |
| `none` + CFRNet — what deviation 10 shipped | 0.5 | 0.3213 | 0.4708 | -0.749 | **0.9999** | **135** |
| `none` + CFRNet | 0 | 0.3514 | 0.5093 | -0.060 | 0.894 | 0 |
| `none` + torch | 0.5 | 0.4035 | 0.4538 | -0.703 | 0.656 | 0 |
| `none` + torch | 0 | 0.3876 | 0.4388 | -0.018 | 0.596 | 0 |

The frequency baseline on this fixture is 0.6931, so the first row is a
propensity that has learned nothing: `-log 2`. Four things are worth reading
off that table rather than summarising away.

* **The unit sphere is not what collapses the representation.** Holding
  `row_l2` fixed and changing only the initialisation removes the collapse
  completely — max concentration 0.666, never a step above 0.99. The claim
  deviation 10 was built on ("a cosine target is the whole geometry of a
  unit-sphere representation, so agreement and collapse are the same move")
  does not survive that control, which nobody ran until an adversarial review
  of this packet asked for it.
* **What collapses it is scale.** Measured at initialisation on this fixture's
  1,024 training rows: CFRNet's `0.1/sqrt(fan_in)` leaves the encoder's
  pre-normalisation activations at a norm of **0.011**, torch's default at
  **1.94**. `F.normalize`'s backward carries `1/\|\|.\|\|`, so under `row_l2`
  the first configuration hands eq. (3)'s gradient to the encoder about ninety
  times amplified. The paper never meets this because a batch-normalised
  WideResNet's pooled activations are order 1 by construction.
* **Dropping the normalisation does not fix it, it postpones it.** The
  configuration deviation 10 shipped still reaches a concentration of 0.9999 by
  step 43 and stays above 0.99 for 135 steps with the gate shut. It escapes
  around step 180. A Tier 1 test that averaged the last hundred steps of a
  3,000-step run reported that as "does not collapse"; the review caught both
  the test and the sentence it was there to support.
* **Eq. (3)'s benefit is architecture-dependent and small.** On the declared
  encoder it is worth 4.0% (EMA) and 7.7% (student); on `none + CFRNet`, 8.6%
  and 7.6%; on `none + torch` it *costs* 4.1% and 3.4%. One seed each. Nothing
  here supports a claim about eq. (3) that does not name the encoder.

**Block A — the recipe as declared**, built normally, same fixture and seeds,
3,000 steps:

| arm | EMA NLL | student NLL | outcome NLL | `l_s` | target conc. | pred. conc. | mask rate |
|---|---|---|---|---|---|---|---|
| `w_s = 0.5` | 0.3225 | **0.3556** | 1.1787 | -0.6131 | 0.194 | 0.410 | 0.745 |
| `w_s = 0` | **0.3185** | 0.3807 | 1.1788 | +0.0066 | 0.246 | 0.880 | 0.746 |

**The two readings disagree, and they disagree the other way round from
Block B.** Here eq. (3) is 1.3% *worse* on the EMA and 6.6% better on the
network under it; in Block B, on the same architecture and a different
initialisation draw, it was 4.0% better on the EMA and 7.7% better on the
student. Two draws, two orderings on the metric §6 declares. That is the
strongest evidence in this card for the thing it keeps saying: at this budget,
on this fixture, a single-seed directional claim about eq. (3) is not
supportable, and §6's ten-seed mean with a standard error is the only thing
that will settle it. Tier 1 asserts none of it.

What *is* stable across every configuration measured: the term learns
(alignment 0.61-0.75 where the same term left undescended sits within 0.07 of
zero), the outcome likelihood is untouched to four decimal places, and the gate
is unmoved (mask rate 0.745 against 0.746). Those are what Tier 1 asserts.

Read the undescended arm's alignment for what it is, which an earlier draft did
not: `h` is trained by eq. (3) and by nothing else, so at `w_s = 0` it keeps its
initialisation, its output is dominated by its own random bias (§7), and
`cos(h(v), z) ~ 0` is close to guaranteed. It shows the logging path is inert
without the term, not that the other four terms would have failed to align the
representation. The control that would show the latter has not been run.

**The comparison this card is careful not to make.** The shared P5 stack —
`row_l2` + CFRNet, `w_s = 0`, which is `fixmatch`'s recipe up to a weight-zero
term — gives a held-out EMA NLL of **0.3066** on the same fixture and budget,
better than either arm of Block A: 5.2% better than the declared recipe and
3.9% better than its own ablation. Deviation 9's initialisation costs more than
eq. (3) returns, so *DoubleMatch does not beat FixMatch here*. §2 claims the
mechanism inside the architecture the mechanism requires, and not this. Every
percentage in this section is relative to the arm being compared against, which
an earlier draft got wrong in the one place where the other convention was the
flattering one.

**What this says about `scarf.md` §6.2, which is why the card picked this
method.** SCARF's contrastive pretraining left the end-to-end gain absent at
every budget tried, diagnosed as instance discrimination pushing apart the
same-treatment rows the propensity head wants together. Eq. (3) is the same
family with the negatives deleted, so the prediction was that it would not pay
the same cost. What the tables support is weaker: the negative-free term learns
without collapsing on the declared encoder, which is the failure such an
objective is supposed to have. Whether deleting the negatives buys a better
propensity is exactly the question the disagreeing readings leave open, and it
needs the Tier 2 the ledger below says has not run.

### 6.3 Result ledger

No Tier 2 runner exists for this card yet, so the target above has never been
run at ten seeds and the status line stays where it is. That is the same
sequence `scarf` and `fixmatch` went through — the recipe and its Tier 0/Tier 1
tests land first, and the benchmark runner is its own packet — and §6.2 says
what the single-seed evidence does and does not support in the meantime.

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
