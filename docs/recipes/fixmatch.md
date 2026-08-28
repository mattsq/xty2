# Recipe spec card: fixmatch

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2, §3.2, and §4 to implement; §5 for departures;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence](https://arxiv.org/abs/2001.07685) |
| Authors, year | Kihyuk Sohn, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, Colin Raffel; 2020 |
| DOI / arXiv | [arXiv:2001.07685](https://arxiv.org/abs/2001.07685); NeurIPS 2020 |
| Version used | arXiv v2, 2020-11-25. Section 2 defines the method; section 2.2 gives eq. (3) and eq. (4); section 2.3 the augmentations; section 2.4 the optimiser and rate schedule; section 4 the shared hyperparameters; appendix A algorithm 1; appendix B.1 table 4; appendix B.2 eq. (5) and eq. (6). |
| Reference implementation | [`google-research/fixmatch`](https://github.com/google-research/fixmatch) (official TensorFlow), consulted **second-hand**: not read directly and not pinned to a commit — this session had no network access to repositories other than `mattsq/xty2` — but a structural summary of `fixmatch.py`, `libml/augment.py`, `libml/train.py` and `cta/lib/train.py` was supplied in-session and is the source of the four rows marked "reference implementation" in section 7. |
| Reference impl. runnable? | Not attempted. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities plus treatment-specific outcome means for causal contrasts.
- **Method claim:** weak-view predictions above `tau` become hard stop-gradient targets for strong-view predictions; supervised and pseudo-label losses train together.
- **Scope:** the paper studies images. Tabular masking, the causal outcome stack, and missing-treatment marginalisation are project-local; the benchmark is therefore a paired mechanism test.

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

Two independently drawn weak views separate the supervised and pseudo-label paths. The strong view layers 50% masking on the weak transform. A quota sampler binds `B=64, mu=7`; cosine decay and evaluation EMA match the declared optimisation contract.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_m(y \| x)` | model's class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| `alpha(.)` | weak augmentation | — | `ViewSpec("weak_x")`, `FeatureMask(p=0.1)`, declaring two draws |
| `A(.)` | strong augmentation | — | `ViewSpec("strong_x")`, `FeatureMask(p=0.1)` then `FeatureMask(p=0.5)` — the weak transform with more corruption layered on, as the reference does |
| `x_b, p_b` | labelled example and its one-hot label | `T_GIVEN_X @ weak_x draw=1` | `ObservedTreatmentNLL(realisation=Realisation("weak_x", draw=1))`, rows `t_observed` |
| eq. (3) `\ell_s` | supervised cross-entropy on weak views | `T_GIVEN_X @ weak_x draw=1` | as above, `reduction="mean"` — a *second* sample of `alpha`, per footnote 2 (§7) |
| `q_b` | artificial label distribution | `T_GIVEN_X @ weak_x draw=0` | `PseudoLabelTreatmentNLL.target` |
| `\hat q_b = arg max(q_b)` | hard pseudo-label | — | `sharpening="hard"` inside that objective |
| `1(max(q_b) >= tau)` | confidence gate | — | `threshold=0.95` inside that objective |
| `p_m(y \| A(u_b))` | strong-view prediction | `T_GIVEN_X @ strong_x` | `PseudoLabelTreatmentNLL.prediction` |
| eq. (4) `\ell_u` | gated pseudo-label cross-entropy | `T_GIVEN_X` at both views | `PseudoLabelTreatmentNLL`, rows `all`, `reduction="mean"` |
| `\lambda_u` | unlabelled loss weight | — | `Weighted(..., weight=1.0)`, `Constant` |
| eq. (6) mask rate | fraction retained | — | `coverage` diagnostic of that objective, and §6.2's terminal figure off the trained network |
| eq. (5) impurity | error rate of retained labels | — | not an objective; measured against ground truth in the section 6 fixture, where the true `t` exists |
| `eta cos(7 pi k / 16 K)` | rate schedule (section 2.4) | — | `CosineDecay(steps=3000, phase=7/16)` on `OptimiserSpec.lr_schedule` |
| EMA of parameters (section 2.4) | the model the paper reports from | — | `TeacherSpec(decay=0.999, role="evaluation")`; no objective reads it |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.pseudo_label_treatment_nll: p(t|x) @ view=weak_x params=student
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # eq. (4): joint_fit.pseudo_label_treatment_nll detaches T_GIVEN_X @ weak_x; the label is arg max(q_b), a constant w.r.t. theta
  gradient_clipping: none                     # paper names none; retained P5 choice
  marginal_nll_grad_path: both                # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.999                            # reference implementation's ema flag; appendix B.9 states 0.999 where the paper gives a number at all
  ema_applies_to_buffers: false               # ref impl EMAs the trainable variables, so BatchNorm moving statistics are not shadowed
  teacher_in_train_mode: false                # the EMA copy is an evaluation classifier; no-op for this architecture, declared anyway
  teacher_requires_grad: false                # never an optimiser target
  # role = evaluation. Nothing reads this EMA during training: eq. (4)'s label
  # comes from the current network (section 7). It exists to be reported with,
  # and the compiler rejects an objective that takes it as a target.

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
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps   # linear, in optimiser steps
  temperature: n/a                            # table 1: FixMatch post-processes by pseudo-labelling, not by sharpening
  sharpening: hard                            # eq. (4): arg max(q_b), a one-hot label
  confidence_threshold: 0.95                  # tau, table 4; ablated in appendix B.2 table 5

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)    # section 2.4 and table 4: beta = 0.9, Nesterov True
  lr: 0.03                                       # eta, table 4
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))  # section 2.4: eta cos(7 pi k / 16 K), with K = our 3000 steps
  weight_decay: 0.0005 (all trainable components; norm and bias exempt)  # table 4 CIFAR-10 value; ref impl sums l2_loss over `kernel` variables only
  batch_size: 512                                # B + mu B = 64 + 448, derived from the QuotaSampler's quotas
  labelled_unlabelled_ratio: 7.0                 # mu, eq. (5) and table 4; derived from the same quotas, never asserted
  total_steps_or_epochs: 3000                   # optimiser steps, never epochs. K = 2^20 optimiser updates in the paper (train_kimg 2^16 << 10, global step incremented by B=64); see deviation 3. The cosine schedule uses this K.

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
  standardisation: x: none fitted on 'train'    # the section 6 DGP draws standardised features
  outcome_scaling: y: zscore fitted on 'train'  # held-out rows take the same fitted transform, never a refitted one
  treatment_encoding: n/a                       # XTYBatch contract supplies integer classes 0..K-1; propensity emits K probabilities
  split_protocol: one fixed project-local DGP, split train/test by the section 6 fixture; no CIFAR/SVHN protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id  # deviation 12
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply FixMatch to categorical treatment assignment `p(t \| x)` and compose it with a causal outcome likelihood (`ObservedOutcomeNLL`) and exact marginalisation (`MissingTreatmentMarginalNLL`). | The paper studies image classes. The project-local question is whether a missing *treatment* label can be recovered by the same mechanism, and whether it composes with the reviewed P5 stack rather than replacing it. | No comparison to a published image error rate is valid. The marginal term also trains `p(t \| x)` on exactly the rows the gate is deciding about, so the two mechanisms interact; section 6 measures the pair against a `lambda_u = 0` ablation of the same fit. |
| 2 | `judgement` | — | Replace flip-and-shift (weak) and RandAugment/CTAugment + Cutout (strong) with schema-aware feature masking: 10% weak, and 10% followed by 50% strong. | There is no image structure in a tabular XTY batch. `FeatureMask` is the already-validated tabular perturbation, and masking at two strengths preserves the paper's weak/strong *relation*, which section 5 of the paper shows is what matters. The layering mirrors the reference, where the strong branch is an independently sampled ordinary augmentation with CTAugment and Cutout added on top rather than substituted in. | Directly defines the invariance being learned. A strong view that destroys the treatment-predictive columns would make eq. (4) train the model toward its own errors; the paired ablation and the impurity guardrail are what would show it. **The 50% is not defensible as label-preserving** — `flexmatch.md` §5.2 measures it flipping the Bayes-optimal label on 16.8% of rows, and the prose after this table records that section 6.3's ledger was produced under it and has not been re-measured under a view that is. The weak/strong *relation* this row was written to preserve is preserved; the strength it picked is not. |
| 3 | `judgement` | — | Train for 3,000 optimiser steps rather than `K = 2^20`. | The reviewed project-local budget, shared with every other xty2 recipe so that a difference is attributable to the recipe. The cosine schedule's `K` is set to the same 3,000, so the *shape* of section 2.4's decay is exact even though its length is not. | The paper's mask rate reaches 98% only after a very long run (table 5). At 3,000 steps we should expect a lower terminal mask rate; the section 6 target is stated in those terms and not in the paper's. |
| 4 | `withdrawn` | — | ~~No labelled/unlabelled batch quota: `mu = 7` is not enforced.~~ **Withdrawn.** The stage declares `QuotaSampler(Quota(t_observed, 64), Quota(t_missing, 448))`, so every step mixes eq. (3)'s `B` rows with eq. (4)'s `mu B` exactly as the paper states. | xty2 has a loader. Both card keys are *derived* from the quotas rather than declared beside them, so the plan prints the ratio the sampler runs — a recipe drawing 64 and 64 cannot claim `mu = 7`. The old row's reasoning about `mean` reduction was right about the gradient and wrong about the variance, which is the half this repays. | The per-step variance of eq. (3) now matches the paper's, where before it moved with whatever the caller's batch happened to contain. The section 6 numbers below were measured without the quota and are **invalidated**: this is new arithmetic, not a rewiring, and the paired ablation has to be re-measured rather than re-labelled. |
| 5 | `withdrawn` | — | ~~No EMA of model parameters.~~ **Withdrawn.** The stage now maintains the paper's EMA at decay 0.999 and section 6 evaluates it. | The original entry was a framework limitation, not a judgement: `compile()` rejected a `TeacherSpec` no objective reads, on the reasoning that an unused EMA copy is a silent no-op. For FixMatch it is the model the paper reports. `TeacherSpec.role` now distinguishes the two, and the recipe declares `role="evaluation"` — see section 5.1, which records that this was built with one consumer and why. | Section 6 now evaluates the EMA, which is what the paper reports; the paired ablation is EMA-to-EMA, so the comparison stays internally consistent. The training signal is unchanged either way: eq. (4)'s label comes from the current network, which is the difference between this method and Mean Teacher. |
| 6 | `judgement` | — | Retain the P5 TARNet architecture (encoder, outcome head, propensity) rather than a Wide ResNet. | Holding the causal stack fixed is what makes the FixMatch addition attributable, and is the same decision `mean_teacher.md` deviation 10 records. | The project-local result validates wiring and mechanism, not image-scale accuracy. |
| 7 | `judgement` | — | Retain P5's `Ramp(0.0, 0.5, 1000)` on the marginal-likelihood term while the FixMatch weight stays constant. | The ramp belongs to the reviewed P5 term, not to FixMatch; section 2.2 rejects a ramp for `lambda_u` and this recipe honours that for the term the sentence is about. | Early steps are dominated by the two supervised terms, which is also when the gate is closed. The two mechanisms therefore switch on in the order the paper describes. |
| 8 | `withdrawn` | — | ~~Weight decay reaches biases as well as matrices.~~ **Withdrawn.** The recipe now exempts biases and norm parameters, which is what the reference does. | An earlier version of this card read appendix B.9's "L2 penalty of all weights" literally and decayed everything. The implementation sums `l2_loss` over the variables whose name carries `kernel` only, so matrices are decayed and biases are not. The paper's prose is the looser source and the code settles it; this is no longer a deviation. | Section 6.2's numbers were re-measured after the change and moved in the third decimal place — the declared architecture has no parameterised normalisation, so only biases were affected. |
| 9 | `judgement` | — | Adopt the paper's optimiser (SGD, `eta = 0.03`, `beta = 0.9`, Nesterov, cosine decay) rather than P5's Adam stack, which `mean_teacher.md` deviation 10 retained. | Section 2.4 and appendix B.3 make the optimiser part of FixMatch's published finding — table 7 reports Adam at a materially worse error rate — so retaining Adam would deviate from an explicit result of the paper in order to match a project convention. The cost is that the outcome head now trains under FixMatch's optimiser too. | This is the one place where the causal stack is not held fixed against P5. Section 6's paired ablation shares the optimiser, so the FixMatch *mechanism* remains attributable; comparisons to `tarnet`'s or `mean_teacher`'s recorded numbers do not. |
| 10 | `framework-limitation` | `augmentation-vocabulary` | No adaptive augmentation: the strong view's strength is fixed, where the reference runs CTAugment. | CTAugment learns per-operation magnitude ratings online from labelled *probe* images, scored by the EMA classifier's probability on the true class — a second learning system with its own state, and one whose operations (posterise, solarise, shear, rotate) have no tabular meaning. The paper's own RandAugment variant, FixMatch (RA), performs within noise of the CTAugment one on CIFAR-10 (table 2), so a fixed-strength strong view is the honest tabular analogue of the simpler published variant. | Removes whatever adaptivity buys. It also removes a confound: the strong view's strength is a declared constant this card can be held to, rather than a trajectory. A second consumer is not what this one is short of: ReMixMatch — where CTAugment was introduced — is an obvious one, and the pair still could not be built, because learning magnitudes presupposes a set of tabular operations with magnitudes worth learning over. `FeatureMask` has one scalar and `BoundedJitter` one more. The prerequisite is a tabular augmentation vocabulary (SCARF's corruption, SubTab's feature subsets, VIME's masking are the backlog's candidates), and only after that does the adaptive controller have anything to control. `scarf` has since built the first of those three, and it moves this row without discharging it: `FeatureCorruption` carries one magnitude, so the vocabulary is now three operations with one scalar each rather than two. |
| 12 | `framework-limitation` | `batch-row-repetition` | Set the section 6 label budget to 64 rather than the paper's smallest regime of 40, holding `B` and `mu` at the paper's values. | Section 4's smallest CIFAR-10 setting is 40 labels *in total* against `B = 64` per step: the reference iterates the labelled set as an endlessly repeating shuffle, so one step sees some labelled rows twice. `XTYBatch.row_id` must be unique because artifacts and provenance are keyed by it (`DESIGN.md` section 7.1), so a repeated row cannot go in a batch and the scarcest budget expressible here is `B` itself. The alternative — lowering `B` to fit the budget — would deviate from a number the paper reports rather than from a number this card chose. | Slightly more supervision than the paper's scarcest regime, on a fixture whose numbers were never comparable to CIFAR-10 anyway (deviation 1). It moves the section 6 paired comparison, and it moves both arms equally: the ablation shares the budget. What it does not touch is the mechanism under test, since the gate reads the model's confidence rather than the label count. |
| 11 | `judgement` | — | Four separate forward passes (identity, two draws of weak, strong) rather than one concatenated and interleaved pass. | The reference fuses the three streams into a single call and interleaves them first, so that each device's BatchNorm population sees a mixture rather than one stream. That is a BatchNorm/multi-GPU device trick, not part of the objective. xty2 plans one pass per realisation, which is arithmetically identical **only while no component holds batch-coupled state**. | None for the declared architecture: `row_l2` normalises each row on its own and the graph carries no buffers at all, which `tests/invariants/test_fixmatch.py` asserts rather than assumes. A component that later grows a running statistic would silently make the two schemes differ, and that test is what would fail first. |

### 5.1 Framework impact

`fixmatch` introduced multi-draw views, `PseudoLabelTreatmentNLL`, `QuotaSampler`, cosine scheduling, and evaluation-only teacher EMA. The exact additions and unresolved abstraction debts remain in the deviation ledger.

## 6. Reproduction target

The fixed project-local pair compares the full recipe with `lambda_u=0` and records likelihood, gate, impurity, outcome, and alignment diagnostics. It is reproduced with the fixed-strength augmentation debt in §5.10 and the minimum-label-budget debt in §5.12 still open; both arms share those limitations.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in 6.1
  variant: paired fit against an otherwise identical lambda_u = 0 ablation, same seeds and same batches
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio on the EMA parameters, FixMatch over the lambda_u = 0 ablation; paper mask rate (eq. 6) and impurity (eq. 5), measured on the trained network, as guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: ratio < 1.0 in mean on both the EMA and the trained parameters (see 6.2); terminal mask rate above 0.2; impurity of retained labels < 0.15; held-out outcome NLL within 1.05x of the ablation
  seeds: 10
  report: mean_and_stderr
```

### 6.3 Result ledger


| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-27 | `1a10fb039e5f` | ema_treatment_NLL_ratio<br>trained_treatment_NLL_ratio<br>terminal_mask_rate<br>retained_label_impurity<br>held_out_outcome_NLL_ratio | 0.886981 +/- 0.024<br>0.868972 +/- 0.0367<br>0.78418 +/- 0.0158<br>0.0518273 +/- 0.00233<br>0.999837 +/- 0.000108 | yes |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Eq. (4) and eq. (6) gate on `max(q_b) >= tau`; algorithm 1 line 6 writes `max(q_b) > tau`. | `>=`. | **Reference implementation:** the code compares the row maximum `>= confidence`. That agrees with eq. (4) and with the mask-rate definition in eq. (6), leaving algorithm 1's strict inequality as the odd one out. The difference is measure-zero on continuous logits; it is recorded because a card that silently picked one would be hiding a real inconsistency in the source. |
| Whether the gate multiplier and the label are treated as constants of `theta`. | Both. The whole target realisation is detached before the arg max and the comparison. | **Reference implementation:** the weak-view softmax is wrapped in a stop-gradient before the arg max and the threshold are taken, so the detach covers the gate as well as the label. That is also what the arithmetic forces — `arg max` has no gradient and the indicator's derivative is zero almost everywhere — but `DESIGN.md` section 4 requires the stop-gradient to be *declared*, and now it is declared to match the code rather than to match an inference. |
| Which network produces the artificial label — the trained one, or the EMA copy the paper evaluates with. | The trained one, under the same parameters every other term uses. | **Reference implementation:** the weak-view logits come from the ordinary classifier call in training mode; the EMA copy exists only for evaluation (and, separately, to score CTAugment's probes). This is the difference between FixMatch and Mean Teacher, and getting it backwards would have made this recipe a worse-specified `mean_teacher`. The EMA is still declared, as `role="evaluation"`: it is what section 6 reports from, and nothing trains against it. |
| Whether weight decay reaches biases. Appendix B.9 says "L2 penalty of all weights". | Matrices only; biases and any norm parameters exempt. | **Reference implementation:** the penalty is summed over variables whose name carries `kernel`. The same summary settles a second thing the card would otherwise have had to argue: the penalty is added to the loss with `l2_loss`'s built-in 1/2, so its gradient is `wd * W` — exactly what torch's SGD `weight_decay` adds, and unlike decoupled (AdamW-style) decay. |
| No tabular augmentation is defined; section 2.3 is entirely image-specific. | `FeatureMask(p=0.1)` weak; the same mask followed by `FeatureMask(p=0.5)` strong. | `FeatureMask` is the reviewed schema-aware transform, already used at `p=0.1` by `mean_teacher`; reusing that value for the weak view makes the two recipes' weak augmentation identical, and the added 0.5 is a deliberate step in strength rather than a tuned value. Fixed before any result was observed. Cutout — the one strong operation the reference appends unconditionally — is a contiguous blanked region, and feature masking *is* its tabular form, so it is not modelled separately. |
| Whether the layering matters once the transform is a constant-fill mask. | It does not, and the card says so rather than implying otherwise. | Two independent masks with the same fill compose to a single mask of rate `1 - (1-0.1)(1-0.5) = 0.55`, so `strong_x` is observationally one 55% mask. The two-transform spelling is kept because it states the *relation* the reference implements — strong is weak plus more — which a lone `p=0.55` would hide. In the image setting the layering is not degenerate, since CTAugment's operations do not commute with crop and flip. |
| Whether a labelled row's weak view may be shared between eq. (3) and eq. (4)'s target. The paper's footnote 2 puts labelled rows into `U` "without their labels", which in the reference means a second, independently drawn batch and therefore an independently sampled `alpha`. | **Not shared.** `weak_x` declares `draws=2`: eq. (4)'s target reads draw 0 and eq. (3) reads draw 1, which reproduces the reference's two independent samples. | xty2 used to compute a view once per batch per name, so a second independent draw meant a second declaration — a `ViewSpec` whose only content was "not the other one". Section 5.1 records the axis that replaced that, and that it was taken with one consumer. The plan now prints `weak_x (2 independent draws)` and a fourth forward pass, and asking for a draw the view does not declare is a compile error naming the view. |
| Whether a *tabular* strong view should also perturb continuous columns (`BoundedJitter`). | No: masking only. | `BoundedJitter` requires an explicit column list and a `perturbation_scale` on each `FeatureSpec`, which would make the recipe a function of the schema's contents — logic in a recipe (`CLAUDE.md` rule 3). A jitter-based strong view is a legitimate second card, not a silent addition to this one. |
| `K = 2^20` steps and `mu = 7` cannot both be honoured with no loader and a 3,000-step budget. What `K` counts is also not stated in the paper. | Keep `tau`, `lambda_u`, `eta`, `beta`, Nesterov, weight decay and the cosine *shape*; re-base `K` on 3,000 steps; drop `mu`. | Deviations 3 and 4. **Reference implementation** for the unit: the global step counts labelled examples and advances by `B` per update, and the target is `train_kimg << 10` with `train_kimg = 2^16`, i.e. `2^26 / 64 = 2^20` optimiser updates. Both `k` and `K` in the rate schedule are that same counter, so the ratio is unit-free and re-basing it on 3,000 optimiser steps preserves the schedule's shape exactly. |
| The paper gives no EMA decay for the CIFAR-scale runs; 0.999 appears in the reference implementation's defaults and in appendix B.9's ImageNet list. | 0.999, unchanged from the paper's value. | The alternative would be re-basing it the way deviation 3 re-bases the cosine's `K`, and inventing a number the paper never states is the worse of the two. The consequence is recorded rather than absorbed: at 0.999 the EMA's horizon is about 1,000 steps, which is a thousandth of the paper's `2^20`-step run and a *third* of ours. Section 6.2 measures what that does. |
| Table 4's weight decay is 0.0005 for WRN-28-2 and 0.001 for WRN-28-8; neither network is ours. | 0.0005. | The CIFAR-10 / SVHN / STL-10 column, which is the paper's default; appendix B.6 warns that the value matters most in the low-label regime and that being an order of magnitude out is what costs accuracy, so the default is the defensible choice for an architecture the paper never studied. |
| No published target applies to a tabular causal adaptation. | A seed-locked project-local mechanism target with a paired `lambda_u = 0` ablation, plus the paper's own mask rate and impurity as guardrails. | The same discipline as `mean_teacher.md` section 6: predeclare the DGP, the pairing and the tolerance before running anything, and make the mechanism — not a borrowed number — the thing that can fail. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
