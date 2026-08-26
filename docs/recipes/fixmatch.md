# Recipe spec card: fixmatch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> The recipe, the objective and the Tier 0/Tier 1 tests exist and pass, so the
> *code* is at what this vocabulary calls `smoke-passing`. The status stays
> `draft` because section 8 is unsigned: this card and its implementation were
> written in one pass rather than card-first-then-review, which `CLAUDE.md`
> rule 1 asks for. A reviewer moving section 8 to signed is what moves this
> line; nothing here should be read as having been reviewed.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685) |
| Authors, year | Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, Colin Raffel; 2020 |
| DOI / arXiv | [arXiv:2001.07685](https://arxiv.org/abs/2001.07685); NeurIPS 2020 |
| Version used | arXiv v2, 2020-11-25. Section 2 defines the method; section 2.2 gives eq. (3) and eq. (4); section 2.3 the augmentations; section 2.4 the optimiser and rate schedule; section 4 the shared hyperparameters; appendix A algorithm 1; appendix B.1 table 4; appendix B.2 eq. (5) and eq. (6). |
| Reference implementation | [`google-research/fixmatch`](https://github.com/google-research/fixmatch) (official TensorFlow). **Not pinned to a commit and not read in this pass** — the session that wrote this card had no network access to repositories other than `mattsq/xty2`. Every value below is therefore sourced from the paper itself, and section 7 records the two places where that is not enough. |
| Reference impl. runnable? | Not attempted. |

Because no implementation was consulted, this card deliberately claims less than
`mean_teacher.md` does: where the paper is silent, section 7 says so and names
our choice as a choice, rather than borrowing an unverified line of code.
Fortunately FixMatch is unusually completely specified in print — algorithm 1
and table 4 together fix the loss, the gate, the ratio and every optimiser
constant.

## 2. Estimand and claim

- **Estimand:** The FixMatch part estimates the categorical propensity
  `p(t | x)` — here, which treatment a row received. The surrounding causal
  stack, retained unchanged from P5, also estimates the treatment-conditional
  outcome distribution `p(y | x, t=k)` and its means `mu_k(x)`; contrasts of
  those means identify conditional treatment effects under consistency,
  positivity and conditional exchangeability.
- **Claim:** FixMatch trains a classifier on unlabelled rows by taking the
  argmax of its own prediction under a *weak* augmentation as a hard label,
  keeping that label only where the predicted probability clears a threshold
  `tau`, and applying cross-entropy to it against the model's prediction under a
  *strong* augmentation (eq. 4). The paper claims state-of-the-art image
  SSL error rates from that combination alone, without sharpening, distribution
  alignment, an unlabelled-loss ramp or a self-supervised branch. This card
  claims only that the mechanism is faithfully assembled around `p(t | x)` in
  xty2, that the gate behaves as section 2.2 predicts (mask rate rises as the
  model sharpens), and that on the fixed project-local target in section 6 it
  improves held-out treatment prediction without damaging the outcome stack.
- **Not claimed:** No image number is claimed. Two limitations are structural
  rather than incidental and are stated here rather than buried in section 5:
  1. **The confidence gate is in tension with positivity, and does not fail
     safe.** FixMatch retains a row only when `max_k p(t=k | x) >= 0.95`. Causal
     identification of `E[Y | do(t)]` needs the opposite — overlap, `p(t=k | x)`
     bounded away from 0 and 1. Where overlap genuinely holds, no calibrated
     model can clear the gate, so the natural guess is that eq. (4) simply goes
     quiet. It does not. Measured on the overlapping version of section 6's DGP
     (section 6.2), the fit becomes confident past what a 0.15/0.85 assignment
     supports: about half the rows clear the gate and a fifth of the labels it
     keeps are wrong. FixMatch is therefore a *predictive* propensity mechanism
     that this card composes with a causal stack, and it is one whose gate
     measures the model's confidence rather than the data's. Section 6's DGP
     deliberately lives in the near-deterministic-assignment corner where that
     confidence is warranted. This is a finding of the port; it is recorded
     rather than designed around, and `tests/smoke/test_fixmatch.py` asserts
     both halves of it.
  2. **A pseudo-label on `t` is not a label on `y`.** The pseudo-labelled rows
     train `p(t | x)` only. They never enter `ObservedOutcomeNLL`, so no
     inferred treatment is ever used as if it had been observed, and the
     `DESIGN.md` section 7.2 leakage rule is not engaged at all: nothing here
     reads `Y_RAW` to produce a treatment label.

## 3. Equations and mapping

### 3.1 As published

For an `L`-class problem, `X = {(x_b, p_b) : b in (1, ..., B)}` is a batch of
`B` labelled examples with one-hot labels `p_b`, and
`U = {u_b : b in (1, ..., mu*B)}` a batch of `mu*B` unlabelled examples.
`p_m(y | x)` is the model's predicted class distribution and `H(p, q)` the
cross-entropy. `alpha(.)` is weak augmentation and `A(.)` strong augmentation
(section 2).

Section 2.1, pseudo-labelling, with `q_b = p_m(y | u_b)`:

$$
\frac{1}{\mu B}\sum_{b=1}^{\mu B}
  \mathbb{1}\!\left(\max(q_b) \ge \tau\right) H(\hat q_b, q_b)
\tag{2}
$$

where `\hat q_b = arg max(q_b)` and `tau` is the threshold; "for simplicity, we
assume that arg max applied to a probability distribution produces a valid
'one-hot' probability distribution".

Section 2.2, the supervised term on *weakly* augmented labelled examples:

$$
\ell_s = \frac{1}{B}\sum_{b=1}^{B} H\!\left(p_b, p_m(y \mid \alpha(x_b))\right)
\tag{3}
$$

and the unsupervised term, with `q_b = p_m(y | alpha(u_b))` and
`\hat q_b = arg max(q_b)`:

$$
\ell_u = \frac{1}{\mu B}\sum_{b=1}^{\mu B}
  \mathbb{1}\!\left(\max(q_b) \ge \tau\right)
  H\!\left(\hat q_b, p_m(y \mid \mathcal{A}(u_b))\right)
\tag{4}
$$

"The loss minimized by FixMatch is simply `\ell_s + \lambda_u \ell_u` where
`\lambda_u` is a fixed scalar hyperparameter denoting the relative weight of the
unlabeled loss."

Appendix B.2 defines the two quantities this card uses as diagnostics:

$$
\text{impurity} =
\frac{\sum_b \mathbb{1}(\max(q_b) \ge \tau)\,\mathbb{1}(y_b \neq \hat q_b)}
     {\sum_b \mathbb{1}(\max(q_b) \ge \tau)}
\tag{5}
$$

$$
\text{mask rate} = \frac{1}{\mu B}\sum_{b=1}^{\mu B}
  \mathbb{1}(\max(q_b) \ge \tau)
\tag{6}
$$

Two sentences of section 2.2 are load-bearing for the mapping and are quoted
rather than paraphrased. On the absence of a ramp: "we note that it is typical
in modern SSL algorithms to increase the weight of the unlabeled loss term
(`\lambda_u`) during training [...] We found that this was unnecessary for
FixMatch, which may be due to the fact that `max(q_b)` is typically less than
`tau` early in training." And footnote 2, on which rows `U` contains: "In
practice, we include all labeled data as part of unlabeled data without their
labels when constructing `U`."

### 3.2 Mapping to xty2

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_m(y \| x)` | model's class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| `alpha(.)` | weak augmentation | — | `ViewSpec("weak_x")`, `FeatureMask(p=0.1)` |
| `A(.)` | strong augmentation | — | `ViewSpec("strong_x")`, `FeatureMask(p=0.5)` |
| `x_b, p_b` | labelled example and its one-hot label | `T_GIVEN_X @ weak_x` | `ObservedTreatmentNLL(realisation=weak_x)`, rows `t_observed` |
| eq. (3) `\ell_s` | supervised cross-entropy on weak views | `T_GIVEN_X @ weak_x` | as above, `reduction="mean"` |
| `q_b` | artificial label distribution | `T_GIVEN_X @ weak_x` | `PseudoLabelTreatmentNLL.target` |
| `\hat q_b = arg max(q_b)` | hard pseudo-label | — | `sharpening="hard"` inside that objective |
| `1(max(q_b) >= tau)` | confidence gate | — | `threshold=0.95` inside that objective |
| `p_m(y \| A(u_b))` | strong-view prediction | `T_GIVEN_X @ strong_x` | `PseudoLabelTreatmentNLL.prediction` |
| eq. (4) `\ell_u` | gated pseudo-label cross-entropy | `T_GIVEN_X` at both views | `PseudoLabelTreatmentNLL`, rows `all`, `reduction="mean"` |
| `\lambda_u` | unlabelled loss weight | — | `Weighted(..., weight=1.0)`, `Constant` |
| eq. (6) mask rate | fraction retained | — | `coverage` diagnostic of that objective |
| eq. (5) impurity | error rate of retained labels | — | not an objective; measured against ground truth in the section 6 fixture, where the true `t` exists |
| `eta cos(7 pi k / 16 K)` | rate schedule (section 2.4) | — | `CosineDecay(steps=3000, phase=7/16)` on `OptimiserSpec.lr_schedule` |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

Three mapping decisions carry the fidelity of this port and are argued rather
than asserted.

**Rows for eq. (4) are `all`, not `t_missing`.** Footnote 2 says `U` contains
every labelled example as well, without its label. In xty2 a single batch holds
both populations, so the faithful eligible set for the pseudo-label term is
`all`. This is not a convenience: on a row with an observed treatment the term
pulls the propensity toward its own argmax while `ObservedTreatmentNLL` pulls it
toward the truth, and that interaction is part of the published method.

**Both FixMatch terms use `reduction="mean"`, the causal stack keeps
`population`.** Eq. (3) divides by `B`, the labelled batch, and eq. (4) by
`mu*B`, the unlabelled batch — each term averages over *its own* population, so
the ratio between them is exactly `lambda_u` regardless of how a batch happens
to be composed. `mean` over the term's own rows is that, exactly
(`DESIGN.md` section 6.1). The two retained P5 terms keep `population`, which is
what TARNet's eq. (3) states and what their reviewed cards already say. xty2 has
no loader and therefore no `mu`; using each paper's own normalisation for its
own terms is what makes that absence harmless instead of silent.

**The gate's denominator counts the rows it rejects.** Eq. (4) divides by
`mu*B`, not by the number of retained rows, so a batch in which nothing clears
`tau` contributes zero rather than an average over an empty set. The objective
therefore multiplies by a 0/1 mask and reduces over *every* eligible row; the
compiler prints that choice as a stable `plan_details` line, because no port,
realisation, row population or card key would otherwise reveal it.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.pseudo_label_treatment_nll: target (T_GIVEN_X @ weak_x)
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # eq. (4): joint_fit.pseudo_label_treatment_nll detaches T_GIVEN_X @ weak_x; the label is arg max(q_b), a constant w.r.t. theta
  gradient_clipping: none                     # paper names none; retained P5 choice
  marginal_nll_grad_path: both                # reviewed P5 choice; project-local addition

teacher:
  ema_decay: n/a                              # see deviation 5: the paper's EMA is an evaluation device, and xty2 rejects a teacher no objective reads
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: mean     # eq. (3) divides by B, the labelled batch
    joint_fit.pseudo_label_treatment_nll: mean # eq. (4) divides by mu*B, the unlabelled batch
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed   # eq. (3) is the labelled term
    joint_fit.pseudo_label_treatment_nll: all      # footnote 2: U includes the labelled rows without their labels
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.pseudo_label_treatment_nll: 1.0      # lambda_u = 1, table 4
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.pseudo_label_treatment_nll: constant 1.0   # section 2.2 explicitly rejects a lambda_u ramp
    joint_fit.missing_treatment_marginal_nll: linear ramp 0.0 -> 0.5 over 1000 optimiser steps
  temperature: n/a                            # table 1: FixMatch post-processes by pseudo-labelling, not by sharpening
  sharpening: hard                            # eq. (4): arg max(q_b), a one-hot label
  confidence_threshold: 0.95                  # tau, table 4; ablated in appendix B.2 table 5

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)    # section 2.4 and table 4: beta = 0.9, Nesterov True
  lr: 0.03                                       # eta, table 4
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))  # section 2.4: eta cos(7 pi k / 16 K), with K = our 3000 steps
  weight_decay: 0.0005 (all trainable components; all parameters)  # table 4 CIFAR-10 value; appendix B.9 "L2 penalty of all weights"
  batch_size: n/a                                # external BatchSource; section 6 fixes 256 for validation
  labelled_unlabelled_ratio: n/a                 # mu = 7 is a loader quota xty2 has no loader for (DESIGN.md section 11); see deviation 4
  total_steps_or_epochs: 3000 optimiser steps    # K = 2^20 in the paper; see deviation 3. The cosine schedule uses this K.

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
    categorical_propensity: K softmax logits   # one head, read under both views and by both treatment objectives

data:
  standardisation: n/a                          # caller-owned; section 6 records the fixed choice
  outcome_scaling: n/a                          # caller-owned; section 6 records the fixed choice
  treatment_encoding: n/a                       # XTYBatch contract supplies integer classes 0..K-1; propensity emits K probabilities
  split_protocol: n/a                           # Tier 1 fixture and the Tier 2 runner own splits
  missingness_mechanism: n/a                    # section 6 fixes treatment MCAR; the recipe consumes t_observed
```

The recipe declares two distinct `ViewSpec` instances over the same schema-aware
transform. `weak_x` is `FeatureMask(p=0.1, columns=None, value=0.0)` and
`strong_x` is `FeatureMask(p=0.5, columns=None, value=0.0)`; both preserve `t`,
`y`, `t_observed`, `y_observed`, `row_id`, `fold_id` and `weight`, and neither
claims to preserve `x`. `columns=None` means every mutable feature; immutable
columns stay bit-identical and schema bounds are enforced. A schema with derived
features must supply recompute rules or the view is rejected at compile time.
`fixmatch(schema, recompute_rules=(...))` passes the same explicit rules to both
views.

The strong view is computed for the whole batch even though eq. (4) only ever
reads it on rows the gate retains. That is arithmetically identical and costs
one masked forward pass; the alternative — a view restricted to a row
population — would be a new framework concept with one consumer.

## 5. Deviations from the paper

| # | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|
| 1 | Apply FixMatch to categorical treatment assignment `p(t \| x)` and compose it with a causal outcome likelihood (`ObservedOutcomeNLL`) and exact marginalisation (`MissingTreatmentMarginalNLL`). | The paper studies image classes. The project-local question is whether a missing *treatment* label can be recovered by the same mechanism, and whether it composes with the reviewed P5 stack rather than replacing it. | No comparison to a published image error rate is valid. The marginal term also trains `p(t \| x)` on exactly the rows the gate is deciding about, so the two mechanisms interact; section 6 measures the pair against a `lambda_u = 0` ablation of the same fit. |
| 2 | Replace flip-and-shift (weak) and RandAugment/CTAugment + Cutout (strong) with 10% and 50% schema-aware feature masking. | There is no image structure in a tabular XTY batch. `FeatureMask` is the already-validated tabular perturbation, and masking at two strengths preserves the paper's weak/strong *relation*, which section 5 of the paper shows is what matters. | Directly defines the invariance being learned. A strong view that destroys the treatment-predictive columns would make eq. (4) train the model toward its own errors; the paired ablation and the impurity guardrail are what would show it. |
| 3 | Train for 3,000 optimiser steps rather than `K = 2^20`. | The reviewed project-local budget, shared with every other xty2 recipe so that a difference is attributable to the recipe. The cosine schedule's `K` is set to the same 3,000, so the *shape* of section 2.4's decay is exact even though its length is not. | The paper's mask rate reaches 98% only after a very long run (table 5). At 3,000 steps we should expect a lower terminal mask rate; the section 6 target is stated in those terms and not in the paper's. |
| 4 | No labelled/unlabelled batch quota: `mu = 7` is not enforced. | xty2 has no loader, and the YAGNI ledger (`DESIGN.md` section 11) makes `optimisation.labelled_unlabelled_ratio` a key nothing could check until one exists. Per-term `mean` reduction removes the need: each term averages over its own population, so `lambda_u` is the relative weight the paper states. | The gradient contribution of eq. (4) relative to eq. (3) is `lambda_u`, as published. What differs is variance: with few labelled rows in a batch, eq. (3) is noisier than in a run that fixes `B = 64` labelled examples per step. |
| 5 | No EMA of model parameters. | Section 2.4 uses the EMA only to *report* final performance; it is not part of the training signal. xty2 rejects a `TeacherSpec` that no objective or action reads (`compile()`, "configures a TeacherSpec but no active objective requires a teacher realisation"), so an evaluation-only EMA would have to become a framework concept with one consumer. `mean_teacher` already owns the teacher surface, and adding a second consumer for an evaluation device is not what the two-consumer rule is for. | Reported metrics are those of the raw student. The paper's own comparison is EMA-to-EMA; ours is student-to-student across the paired ablation, which keeps the comparison internally consistent. |
| 6 | Retain the P5 TARNet architecture (encoder, outcome head, propensity) rather than a Wide ResNet. | Holding the causal stack fixed is what makes the FixMatch addition attributable, and is the same decision `mean_teacher.md` deviation 10 records. | The project-local result validates wiring and mechanism, not image-scale accuracy. |
| 7 | Retain P5's `Ramp(0.0, 0.5, 1000)` on the marginal-likelihood term while the FixMatch weight stays constant. | The ramp belongs to the reviewed P5 term, not to FixMatch; section 2.2 rejects a ramp for `lambda_u` and this recipe honours that for the term the sentence is about. | Early steps are dominated by the two supervised terms, which is also when the gate is closed. The two mechanisms therefore switch on in the order the paper describes. |
| 8 | Weight decay reaches biases as well as matrices. | Appendix B.9 describes weight decay as "adding L2 penalty of all weights to model loss", and no line of the paper exempts biases. The declared architecture has no parameterised normalisation, so this affects biases only. | Negligible at 5e-4 on a three-layer MLP; recorded because `FIDELITY.md` section 2 treats the exemption as a real difference and it would otherwise be a framework default. |
| 9 | Adopt the paper's optimiser (SGD, `eta = 0.03`, `beta = 0.9`, Nesterov, cosine decay) rather than P5's Adam stack, which `mean_teacher.md` deviation 10 retained. | Section 2.4 and appendix B.3 make the optimiser part of FixMatch's published finding — table 7 reports Adam at a materially worse error rate — so retaining Adam would deviate from an explicit result of the paper in order to match a project convention. The cost is that the outcome head now trains under FixMatch's optimiser too. | This is the one place where the causal stack is not held fixed against P5. Section 6's paired ablation shares the optimiser, so the FixMatch *mechanism* remains attributable; comparisons to `tarnet`'s or `mean_teacher`'s recorded numbers do not. |

## 6. Reproduction target

The published CIFAR-10/100, SVHN, STL-10 and ImageNet error rates cannot
validate this port: the inputs, the labels, the architecture and the metric all
differ, and the estimand is a treatment assignment rather than an image class.
The target below is a completely fixed project-local *mechanism* target, in the
same form as `mean_teacher.md` section 6. Passing it supports the limited claim
in section 2; it must not be described as reproducing Sohn et al.

The DGP is deliberately placed in the near-deterministic-assignment corner
described in section 2, "Not claimed": with cluster-conditional propensities of
0.02 and 0.98, a well-fitted `p(t | x)` can exceed `tau = 0.95` at all, so the
gate has something to open on. `tests/smoke/test_fixmatch.py` runs the same
recipe a second time on the overlapping version of this DGP (0.15/0.85) and
asserts what happens there; section 6.2 records the measurement.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in 6.1
  variant: paired fit against an otherwise identical lambda_u = 0 ablation, same seeds and same batches
  split: 1024 train rows with 40 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio, FixMatch over the lambda_u = 0 ablation; paper mask rate (eq. 6) and impurity (eq. 5) as guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: ratio < 1.0 in mean; terminal mask rate above 0.2; impurity of retained labels < 0.15; held-out outcome NLL within 1.05x of the ablation
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

Per row, with independent uniforms `u_c, u_t` and normals `eps_x [6], eps_y`:

```text
cluster   c = 1[u_c < 0.5]
x[:, 0:4] = 0.45 * (2c - 1) + 0.6 * eps_x[:, 0:4]    # redundant cluster signal
x[:, 4:6] = eps_x[:, 4:6]                            # outcome-only covariates
p(t=1|c)  = 0.02 + 0.96 * c                          # near-deterministic assignment
t         = 1[u_t < p(t=1|c)]
baseline  = 0.5*x[:,0] - 0.3*x[:,1] + 0.2*(x[:,4]^2 - 1)
effect    = 1.0 + 0.5*tanh(x[:,2])
y         = baseline + t * effect + 0.5 * eps_y
```

Treatment observation is MCAR: exactly 40 of the 1,024 training rows keep their
`t`. Forty is a choice, and the reason is the paper's: FixMatch's claim is about
the label-scarce regime, and at 205 labels this DGP is already solved by eq. (3)
alone, leaving eq. (4) nothing to add. The 0.45 cluster signal is set for the
same reason — strong enough that a confident model is possible, weak enough that
40 labels do not settle the question by themselves. Both were fixed before any
paired result was read, and both are what the tolerance above is stated against.

The outcome is standardised by the training mean and standard deviation and the
same constants are applied to the held-out rows. Batches are 256 rows drawn by a
seed-locked permutation stream shared by both arms of the pair, so the two fits
differ only in `lambda_u`.

Outcome-side guardrails are stated as non-inferiority against the ablation
rather than as absolute bands. Under a 0.02/0.98 propensity the counterfactual
arm within a cluster is nearly unobserved, so `sqrt_PEHE` is not identified at
this sample size; claiming a band for it would be claiming a number the design
cannot support. This is the cost of the DGP that lets the gate open at all, and
it is the same tension section 2 records.

### 6.2 What the Tier 1 fixture already shows

Tier 1 is not Tier 2 and these are single-seed numbers from
`tests/smoke/test_fixmatch.py`, not a result. They are recorded because they are
the evidence behind section 2's second claim, and because one of them corrected
this card:

| | separable (0.02/0.98), 3,000 steps | overlapping (0.15/0.85), 600 steps |
|---|---|---|
| held-out `p(t\|x)` NLL, FixMatch | 0.265 | 0.786 |
| held-out `p(t\|x)` NLL, `lambda_u = 0` | 0.327 | — |
| marginal-frequency baseline | 0.700 | 0.706 |
| terminal mask rate (eq. 6) | 0.833 | 0.495 |
| impurity of retained labels (eq. 5) | 0.048 | 0.197 |
| held-out outcome NLL, FixMatch / ablation | 1.179 / 1.179 | — |

An earlier draft of this card asserted that under overlap the gate would stay
shut and eq. (4) would be "inert". That is what a *calibrated* model would do,
and it is wrong: the fit becomes confident beyond what a 0.15/0.85 assignment
supports, half the rows clear the 0.95 gate, and one retained label in five is
wrong. The mechanism does not degrade quietly in the overlap regime — it
manufactures the confidence it is gated on, and then trains on it. Section 2 was
rewritten to say that, and the Tier 1 test asserts it.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

**Not yet run.** The Tier 2 runner (`xty2/evaluation/benchmarks/`) has one
module per recipe and this recipe has none; adding it is a separate, reviewed
piece of work, exactly as `mean_teacher.md` section 6 waited for P12. Until then
this card's status may not go past `smoke-passing`, and the block above is a
declared protocol rather than a result.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Eq. (4) and eq. (6) gate on `max(q_b) >= tau`; algorithm 1 line 6 writes `max(q_b) > tau`. | `>=`. | Two of the three statements, including the definition of the reported mask rate, use `>=`; the difference is measure-zero on continuous logits and is recorded only because a card that silently picked one would be hiding a real inconsistency in the source. |
| Whether the gate multiplier and the label are treated as constants of `theta`. | Both. The whole target realisation is detached before the arg max and the comparison. | `arg max` has no gradient anywhere, so the label is a constant by construction; the indicator is a step function whose derivative is zero almost everywhere. Detaching states in the plan what the arithmetic already does, and `DESIGN.md` section 4 requires a stop-gradient to be declared rather than discovered. |
| No tabular augmentation is defined; section 2.3 is entirely image-specific. | `FeatureMask(p=0.1)` weak, `FeatureMask(p=0.5)` strong. | `FeatureMask` is the reviewed schema-aware transform, already used at `p=0.1` by `mean_teacher`; reusing that value for the weak view makes the two recipes' weak augmentation identical, and 0.5 is a deliberate step in strength rather than a tuned value. Fixed before any result was observed. |
| Whether a *tabular* strong view should also perturb continuous columns (`BoundedJitter`). | No: masking only. | `BoundedJitter` requires an explicit column list and a `perturbation_scale` on each `FeatureSpec`, which would make the recipe a function of the schema's contents — logic in a recipe (`CLAUDE.md` rule 3). A jitter-based strong view is a legitimate second card, not a silent addition to this one. |
| `K = 2^20` steps and `mu = 7` cannot both be honoured with no loader and a 3,000-step budget. | Keep `tau`, `lambda_u`, `eta`, `beta`, Nesterov, weight decay and the cosine *shape*; re-base `K` on 3,000 steps; drop `mu`. | Deviations 3 and 4. The constants that define the mechanism are kept; the two that describe the schedule of a very long image run are re-based, in the unit the executor counts. |
| The paper reports final performance with an EMA of parameters (decay 0.999, stated for ImageNet in appendix B.9) but does not use it in training. | No EMA. | Deviation 5. |
| Table 4's weight decay is 0.0005 for WRN-28-2 and 0.001 for WRN-28-8; neither network is ours. | 0.0005. | The CIFAR-10 / SVHN / STL-10 column, which is the paper's default; appendix B.6 warns that the value matters most in the low-label regime and that being an order of magnitude out is what costs accuracy, so the default is the defensible choice for an architecture the paper never studied. |
| No published target applies to a tabular causal adaptation. | A seed-locked project-local mechanism target with a paired `lambda_u = 0` ablation, plus the paper's own mask rate and impurity as guardrails. | The same discipline as `mean_teacher.md` section 6: predeclare the DGP, the pairing and the tolerance before running anything, and make the mechanism — not a borrowed number — the thing that can fail. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
