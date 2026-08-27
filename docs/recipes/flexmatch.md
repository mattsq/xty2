# Recipe spec card: flexmatch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> Written card-first, before any code, per `CLAUDE.md` rule 1. The recipe, the
> objective and the Tier 0/Tier 1 tests now exist and pass, so the *code* is at
> what `FIDELITY.md` §1.1 calls `smoke-passing`. The status stays `draft`
> because §8 is unsigned and because §6.2 records a result the card should be
> re-reviewed in the light of: on this fixture the published procedure does not
> leave its own warm-up, and §2's claim was amended after the measurement — the
> amendment is marked where it is.

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

That last row is a difference from `fixmatch.md` worth stating rather than
smoothing over. That card resolved four §7 rows from a second-hand structural
summary of the reference. This one has no such source, so where the paper is
silent the choice is a convention or a guess and §7 says which. Nothing below
cites a reference implementation, and no row should be read as though it did.

## 2. Estimand and claim

- **Estimand:** The FlexMatch part estimates the categorical propensity
  `p(t | x)` — which treatment a row received. The surrounding causal stack is
  `fixmatch`'s, retained unchanged: the treatment-conditional outcome
  distribution `p(y | x, t=k)` and its means `mu_k(x)`, whose contrasts identify
  conditional treatment effects under consistency, positivity and conditional
  exchangeability.
- **Claim:** Curriculum Pseudo Labeling (CPL) replaces FixMatch's single fixed
  threshold `tau` with a **per-class** threshold `T_t(c)` that rises with how
  well the model has learned class `c`, where "learning effect" is estimated
  from how many unlabelled rows the model currently assigns to `c` above `tau`
  (eq. 5). The paper claims state-of-the-art image SSL error rates from that
  substitution alone — "without introducing additional parameters or
  computations" — with the largest gains where labels are scarcest and the
  classes hardest, and convergence to a better result in roughly a fifth of
  FixMatch's training time (§4.3).

  **This card's own claim was amended after §6.2 was measured, and the original
  is struck through rather than deleted, because the difference between the two
  is the result.** ~~That on the fixed project-local targets in §6 it improves
  held-out treatment prediction against an otherwise identical `fixmatch`
  without damaging the outcome stack.~~ What it claims now is two things. First,
  that the mechanism is faithfully assembled around `p(t | x)`: Tier 0 asserts
  the arithmetic against `PseudoLabelTreatmentNLL` at `beta(c) = 1`, and every
  part of the curriculum is measured doing what §3 says it does — marks
  accumulating, `sigma` rising, `T(c)` reaching `tau` for the best-learned class
  and staying below it for the other — in the `lambda = 0` arm of §6.2. Second,
  that on this fixture the *descended* term never lets that start: the recipe
  ends worse than the `fixmatch` it is paired against and worse than the
  marginal frequencies of its own labelled rows. §6.2 is the measurement and the
  first limitation below is the mechanism.
- **Not claimed:** No image number is claimed. Four limitations are structural
  and are stated here rather than left to be discovered:
  1. **Every threshold is zero at initialisation, and eq. (8) therefore accepts
     the whole batch.** This is not an implementation accident, it is what
     Algorithm 1 computes: at `t = 0` no row is marked, so `sigma_0(c) = 0` for
     every `c`, eq. (11)'s denominator is the unused count `N`, `beta_0(c) = 0`
     and `T_0(c) = M(0) * tau = 0`. A gate at 0 admits every row, so the run
     opens by training `p(t | x)` on the hard arg max of an untrained network,
     over the whole batch. FixMatch's opens on nothing. §3.3 of the paper is
     explicit that the warm-up exists so that "all estimated learning effects
     gradually rise from 0", so the low-threshold phase is the mechanism, not a
     boundary case of it; what the paper does not say is what that phase costs
     when it starts from zero rather than from a small number. §6.2 measures
     it, and it is the first thing to look at if this recipe underperforms.
  2. **The confidence gate is still in tension with positivity, and CPL
     sharpens the tension rather than resolving it.** `fixmatch.md` §2 records
     the finding: under genuine overlap the fit becomes confident past what the
     assignment supports, so the gate opens on labels that are wrong about a
     fifth of the time. A per-class threshold that *lowers* `tau` for the class
     the model has learned least admits more of exactly those rows. This is a
     predictive propensity mechanism composed with a causal stack, and §6's DGP
     lives in the near-deterministic-assignment corner where the confidence is
     warranted.
  3. **Two classes is the regime where CPL has least to do.** `sigma_t` is a
     per-class count and `beta_t` normalises by its maximum, so with `K = 2` and
     a near-balanced assignment both classes reach `beta = 1` together and
     `T_t(c)` collapses back to `tau` for both. What survives on such a fixture
     is the warm-up trajectory of (1), not the per-class differentiation the
     paper's CIFAR-100 and STL-10 gains come from. §6.1 therefore declares a
     deliberately **class-imbalanced** diagnostic variant, because a card that
     measured only the balanced fixture would be reporting on the half of the
     method that is inert.
  4. **A pseudo-label on `t` is not a label on `y`.** As in `fixmatch`: the
     pseudo-labelled rows train `p(t | x)` only, never `ObservedOutcomeNLL`, so
     no inferred treatment is used as if observed and the `DESIGN.md` §7.2
     leakage rule is not engaged.

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

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_m(y \| x)` | model's class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| `omega(.)` | weak augmentation | — | `ViewSpec("weak_x")`, `FeatureMask(p=0.1)`, two draws |
| `Omega(.)` | strong augmentation | — | `ViewSpec("strong_x")`, `FeatureMask(p=0.1)` then `FeatureMask(p=0.5)` |
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

Four mapping decisions carry the fidelity of this port and are argued rather
than asserted.

**`N` is the whole training population, and the marks are indexed by
`row_id`.** FlexMatch does not restate FixMatch's footnote 2 ("we include all
labeled data as part of unlabeled data without their labels when constructing
`U`"), but it defines itself as CPL applied to FixMatch and §2 restates
FixMatch's framework unchanged, so the footnote is inherited along with it —
which is also what `fixmatch.md` §3.2 concluded for the same reason. `U` is
therefore every training row, `N` is `|train|`, and eq. (8)'s eligible set is
`all`. Indexing the marks by `row_id` rather than by position is not a detail:
a `QuotaSampler` draw is a fresh subset each step, so the only stable identity a
row has across steps is the one `DESIGN.md` §7.1 already guarantees is unique.

**The state is per stage execution, not per recipe and not an artifact.** It is
created when a stage starts and discarded when it ends, so two runs of one
compiled recipe are identical and a paired ablation that shares an objective
instance between arms cannot leak marks from one arm into the other. It is not
checkpointed: `BACKLOG.md` §11.3 asks for "explicit objective/stage state"
first and a first-class stateful artifact only once two recipes need the same
lifecycle, and nothing in FlexMatch resumes a run mid-curriculum.

**Two gates, and the objective computes both.** The mark is set at `tau`
(Alg. 1 line 14) and the loss is gated at `T_t(c)` (eq. 8), on the same weak-view
confidences in the same step. Collapsing them — marking at `T_t(c)` — would make
`sigma` self-reinforcing: a class whose threshold had fallen would mark more
rows, which would raise its own `beta`. The paper's two gates are what stop
that, and the objective's `plan_details()` prints both so a reader of the plan
cannot mistake which is which.

**The denominator counts the rows it rejects.** Eq. (8) divides by `mu B`, as
FixMatch's eq. (3) does, so a batch in which nothing clears its class threshold
contributes zero rather than an average over an empty set. Identical to
`fixmatch.md` §3.2's reading, and identical arithmetic: a 0/1 mask, then a mean
over every eligible row.

## 4. Mechanics checklist

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

`losses.confidence_threshold` holds a **policy object** rather than a scalar,
and that is a deliberate reading of the closed vocabulary rather than a way
around it. `card_keys.py` refuses two fields bound to one canonical key with the
instruction "bind one field, holding a tuple if the paper states several numbers
together", and FlexMatch's gate is exactly that case: `tau`, the warm-up and the
mapping are three halves of one rule and no one of them describes the gate. The
alternative — widening `FIDELITY.md` §2 with a `losses.threshold_policy` key —
would be a framework change made to avoid writing a value object, and it would
leave `confidence_threshold` naming a number this recipe does not have.

The views are `fixmatch`'s, unchanged and imported: `weak_x` is
`FeatureMask(p=0.1)` with two draws, `strong_x` is that transform followed by
`FeatureMask(p=0.5)`, both preserving `t`, `y`, `t_observed`, `y_observed`,
`row_id`, `fold_id` and `weight`. `flexmatch(schema, recompute_rules=(...))`
passes the same explicit rules to both, as `fixmatch` does.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply FlexMatch to categorical treatment assignment `p(t \| x)` and compose it with a causal outcome likelihood (`ObservedOutcomeNLL`) and exact marginalisation (`MissingTreatmentMarginalNLL`). | The paper studies image classes. The project-local question is whether a per-class curriculum threshold recovers a missing *treatment* label better than a fixed one, and whether it composes with the reviewed P5 stack rather than replacing it. | No comparison to a published image error rate is valid. The marginal term trains `p(t \| x)` on exactly the rows the curriculum is deciding about, so the two mechanisms interact; §6 measures the pair against `fixmatch`, which carries the same interaction. |
| 2 | `judgement` | — | Replace flip-and-shift (weak) and RandAugment + Cutout (strong) with schema-aware feature masking: 10% weak, and 10% followed by 50% strong. | There is no image structure in a tabular XTY batch, and the pair must be `fixmatch`'s exactly or §6's comparison measures the views as well as the gate. | Defines the invariance being learned, identically in both arms of the pair. |
| 3 | `judgement` | — | Train for 3,000 optimiser steps rather than the paper's `2^20`. | The reviewed project-local budget, shared with every other xty2 recipe so that a difference is attributable to the recipe. The cosine schedule's `K` is set to the same 3,000, so the shape of the decay is exact even though its length is not. | The paper's headline convergence claim (§4.3) is about reaching a result *sooner*; a 3,000-step budget is where CPL's early behaviour matters most and its late behaviour matters least. §6 records the trajectory rather than the endpoint alone. |
| 4 | `judgement` | — | Retain the P5 TARNet architecture (encoder, outcome head, propensity) rather than a Wide ResNet. | Holding the causal stack fixed is what makes the CPL addition attributable, and it is `fixmatch.md` deviation 6 and `mean_teacher.md` deviation 10. | The project-local result validates wiring and mechanism, not image-scale accuracy. |
| 5 | `judgement` | — | Retain P5's `Ramp(0.0, 0.5, 1000)` on the marginal-likelihood term while the CPL weight stays constant. | The ramp belongs to the reviewed P5 term, not to FlexMatch; eq. (9) states a fixed `lambda` and the curriculum lives in the threshold. | Identical to `fixmatch`'s arrangement, so the pair in §6 shares it. |
| 6 | `judgement` | — | Adopt FixMatch's optimiser (SGD, `eta = 0.03`, `beta = 0.9`, Nesterov, cosine decay) rather than P5's Adam stack. | §4 states FlexMatch adopts FixMatch's settings, and `fixmatch.md` deviation 9 already made this choice for the recipe this one is paired against. Retaining Adam would break the pair as well as the paper. | Comparisons to `tarnet`'s or `mean_teacher`'s recorded numbers do not hold; the comparison to `fixmatch` does, which is the one §6 makes. |
| 7 | `framework-limitation` | `augmentation-vocabulary` | No adaptive augmentation: the strong view's strength is fixed, where FixMatch's own reference runs CTAugment and FlexMatch inherits the framework. | The argument is `fixmatch.md` deviation 10's and is not restated: learning per-operation magnitudes presupposes a set of tabular operations with magnitudes worth learning over, and `FeatureMask`, `BoundedJitter` and `FeatureCorruption` are three operations with one scalar each. FlexMatch adds nothing to the case either way — its contribution is the threshold, not the augmentation. | Removes whatever adaptivity buys, equally from both arms of §6's pair, so it is a limit on what the numbers describe rather than a confound within them. |
| 8 | `framework-limitation` | `batch-row-repetition` | Set the §6 label budget to 64 rather than a scarcer regime, holding `B = 64` and `mu = 7` at the paper's values. | `XTYBatch.row_id` must be unique (`DESIGN.md` §7.1), so a labelled quota of `B` cannot be drawn from a population smaller than `B` without repeating a row, and the scarcest budget expressible is `B` itself. The alternative — lowering `B` — would deviate from a number the paper states. | Slightly more supervision than the label-scarce regime where §4.1 reports FlexMatch's largest gains, which is the regime this card would most like to be in. It moves both arms of the pair equally. |
| 9 | `judgement` | — | Record §6.1's two diagnostic variants as one-off single-seed runs in §6.2, rather than adding a second Tier 2 target or a second Tier 1 arm. | Each exists to rule out one explanation of §6.2's result — that the balanced two-class regime leaves CPL inert (§2's third limitation), and that the fixture is simply too hard for the model to reach `tau`. Making either a reproduction target would multiply the nightly cost of a recipe whose declared claim is the paired one; making either a Tier 1 arm would add minutes of CI for a number `FIDELITY.md` §3 says is not a result anyway. | None on the §6 metric. It is what §6.2 is allowed to claim that changes: a direction on one seed, not a target. |

### 5.1 Framework additions made for this card

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| **Objective state with a stage lifecycle** — `StatefulObjective` (an objective may declare `initial_state(population)`), `TrainContext.objective_states` and the accessor `TrainContext.objective_state(name, kind)`, plus the executor building one state per stage execution | fidelity-bearing, **load-bearing vocabulary** (`DESIGN.md` §4 is the contract every objective is written against) | This card | **FreeMatch** (`BACKLOG.md` §2.5), whose "self-adaptive global and class-specific thresholds" are an exponential moving average of the batch's mean confidence and mean probability vector, carried from step to step: the same lifecycle — initialised at stage start, updated from each batch *after* that batch's gate is applied, never checkpointed. **It checked the shape.** FreeMatch's state is an EMA over batches and needs no row table and no `N`, so a hook shaped `initial_state(population: TrainingPopulation)` — requiring a population — would have forced it to accept one it does not use, and a stage fed by `ExternalBatches` has none to give. Hence `TrainingPopulation \| None`, with the objective deciding whether it can run without one. **SoftMatch** (§2.5) is the third: a running mean and variance of the confidence, same lifecycle again | Alg. 1 lines 2, 5 and 15 are the method. `sigma_t(c)` is a count over marks accumulated across every step so far, so without state that survives a step, `T_t(c)` cannot be computed at all and card §4's `losses.confidence_threshold` degenerates to FixMatch's constant — which is not a deviation from FlexMatch, it *is* FixMatch. §11.2 Q1 has no other answer here |
| `CurriculumPseudoLabelTreatmentNLL`, `CurriculumThreshold` and `CurriculumStatus` (`xty2/objectives/curriculum.py`) | fidelity-bearing, reversible | This card | — (not the load-bearing quadrant; nothing outside this recipe is written against them) | An objective is the ordinary extension point `BACKLOG.md` step 4 names. It is a **separate** objective rather than a `threshold: float \| Policy` union on `PseudoLabelTreatmentNLL`, because `BACKLOG.md` §15.2 says so directly — "initially keep each mechanism local and explicit [...] only consider a reusable mediator/policy abstraction if at least two real recipes need the same lifecycle and semantics" — and FreeMatch is the second recipe that would show the shape. The cost is that the two objectives share the arg-max/mask/mean arithmetic; the duplication is the price §15.2 asks for and `fixmatch`'s objective is untouched, plan and digest included |

Neither addition changes what an existing recipe computes. `TrainContext` gains
one field with an empty default, so every existing construction of it is
unchanged; no existing objective declares `initial_state`, so the executor
builds an empty mapping for every existing stage and no plan, digest or recorded
result moves. Tier 0 asserts the second half of that directly rather than
claiming it.

**§6.2 found the mechanism self-defeating on this fixture, and that does not
retract either addition.** The question §11.2 Q1 asks is whether the absence of
an abstraction forces a deviation from a mechanic a card's §4 names, not whether
the mechanic turns out to help: without objective state, `T_t(c)` is not
computable at all and §4's `losses.confidence_threshold` degenerates to
FixMatch's constant, which would not have been a *deviation from* FlexMatch so
much as a failure to implement it — and the null result would have been
unobtainable rather than merely negative. The lifecycle is also what made the
result legible: §6.2's `lambda = 0` arm is only a control because the state is
per stage execution, so the two arms cannot contaminate one another, and Tier 0
asserts exactly that. FreeMatch remains the named second consumer, and
`BACKLOG.md` §2.5 now records what this card learned that it should measure.

Two things this card was expected to need and did not:

* **A new card key.** `losses.confidence_threshold` takes the policy object,
  for the reason §4 gives. The vocabulary in `FIDELITY.md` §2 is unchanged.
* **A stateful *sampler*.** The `stateful-sampler` ledger row exists because a
  sampler that reads the model would destroy the property every paired ablation
  here rests on — that two recipes differing in one field draw identical rows.
  CPL reads the model and changes which rows *train*, not which rows are
  *drawn*, so it needs nothing from that row and §6's pair still shares a batch
  stream exactly.

## 6. Reproduction target

The published CIFAR-10/100, SVHN, STL-10 and ImageNet error rates cannot
validate this port: the inputs, the labels, the architecture and the metric all
differ, and the estimand is a treatment assignment rather than an image class.
The target below is a fixed project-local *mechanism* target in the form
`fixmatch.md` §6 uses.

**This result will be measured without two mechanics the paper's framework
states**, and `DESIGN.md` §11.6 wants that beside the number rather than in the
gap between sections: §5.7 (`augmentation-vocabulary`) fixes the strong view's
strength where the reference runs CTAugment, and §5.8
(`batch-row-repetition`) sets the label budget at 64 rather than in the scarcer
regime where §4.1 reports FlexMatch's largest gains. Both apply equally to both
arms of the pair, so they limit what the numbers describe rather than confound
what they compare.

```yaml
reproduction:
  dataset: fixmatch.md §6.1's project-local seed-locked two-cluster XTY DGP (6 features, K=2), unmodified
  variant: paired fit against `fixmatch` — the same recipe with eq. (8)'s per-class gate replaced by eq. (3)'s fixed one — same seeds and same batches
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio on the EMA parameters, FlexMatch over FixMatch; the paper's mask rate and impurity, and the per-class threshold trajectory, as guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: ratio < 1.0 in mean on both the EMA and the trained parameters, by at least one standard error; terminal mask rate above 0.2; impurity of retained labels < 0.15; held-out outcome NLL within 1.05x of the FixMatch arm; max_c T(c) >= 0.9 tau by the end of the run and min_c T(c) < 0.5 tau at step 0
  seeds: 10
  report: mean_and_stderr
```

The last tolerance is the mechanism guardrail and it is the one that would fail
first if the state were dead. A `T(c)` stuck at `tau` is FixMatch wearing this
card's name; a `T(c)` stuck at 0 is unfiltered self-training. The pair of
bounds asserts the trajectory Algorithm 1 describes — start at 0, rise with the
marks — rather than the endpoint alone.

**Everything above was written before the run, and none of it is being
retuned.** §6.2 measured a single-seed ratio of 2.9 against a declared
tolerance of "below 1.0 by at least one standard error", and the mechanism
guardrail's `max_c T(c) >= 0.9 tau` is not met either: `T(c)` never leaves zero.
So the expected Tier 2 outcome for this card is `deviating`, with §6.2 as the
written explanation `FIDELITY.md` §1.1 asks for. `FIDELITY.md` §3 is explicit
that a tolerance widened after seeing the result is itself a deviation, and a
target rewritten to the number that came out would destroy the only thing this
section is for. It stays as declared; the status line moves, not the target.

### 6.1 Fixed DGPs

**Primary.** `fixmatch.md` §6.1's DGP in full and without modification, so that
the pair differs in the gate and in nothing else — the mechanism, the seeds, the
64-label MCAR budget, the `B = 64` / `mu B = 448` quota, the outcome
standardisation fitted on the training rows, and the replicate seeds
`s_r = 90000 + 100 r` for `r in {0..9}`. Restating it here would be a second
thing to keep true.

**Two diagnostic variants, §6.2 only.** Each changes exactly one constant of
the primary DGP and nothing else. Neither is a target and neither is asserted by
a test — they are one-off runs, recorded because each rules out one explanation
of §6.2's result that the primary fixture alone cannot.

```text
imbalanced     cluster c = 1[u_c < 0.15]        # was 0.5
separable      x[:, 0:4] = 2.0 * (2c - 1) + ... # was 0.45
```

*Imbalanced* moves `p(t = 1)` to about 0.16 so that one class is roughly five
times scarcer than the other. This is the regime §2's third limitation is
about: with `K = 2` and a balanced assignment both classes reach `beta = 1`
together and `T_t(c)` collapses to `tau`, so a card measuring only the primary
fixture would be reporting on the half of CPL that is inert. Deviation 9 says
why it is not a second Tier 2 target.

*Separable* quadruples the cluster signal, which makes `p(t | x)` close to
trivial — `fixmatch` reaches a mask rate of 1.0 and a held-out NLL of 0.099 on
it in 1,500 steps. It exists to answer "is the fixture simply too hard for the
curriculum to get started on", and it answers no.

### 6.2 What the Tier 1 fixture shows

Tier 1 is not Tier 2 and these are single-seed numbers from
`tests/smoke/test_flexmatch.py`, not a result (`FIDELITY.md` §3). They are
recorded because §2's first limitation was a prediction this fixture could
falsify, and it did not.

Three columns, one initialisation, one batch stream, 3,000 steps on the primary
fixture. The middle one is what makes this a measurement rather than a story:
it is **this recipe**, with eq. (8) computed and logged on every step and
descended on none. The first two are the arms
`tests/smoke/test_flexmatch.py` runs and asserts on every PR; the `fixmatch`
column is a one-off run of that recipe on the same fixture and the same seeds,
kept out of this module's Tier 1 because it duplicates a fit
`tests/smoke/test_fixmatch.py` already performs on both.

| | `flexmatch` (declared) | `flexmatch`, `lambda = 0` | `fixmatch` |
|---|---|---|---|
| held-out `p(t\|x)` NLL — EMA / trained | 0.883 / 0.900 | 0.355 / 0.372 | 0.307 / 0.358 |
| marginal-frequency baseline | 0.693 | 0.693 | 0.693 |
| eq. (10) labelled CE — first 100 → last 100 steps | 0.805 → **0.801** | 0.421 → 0.149 | 0.421 → 0.098 |
| gate coverage — step 0 / last 100 | 1.000 / 1.000 | 1.000 / 0.890 | 0.000 / 0.717 |
| first step at which a row clears `tau` | **never** | 97 | n/a |
| terminal marked fraction, `sum_c sigma / N` | **0.000** | 0.847 | n/a |
| terminal `T(c)` — min / max | 0.000 / 0.000 | 0.427 / 0.950 | constant 0.950 |
| step at which `max_c T(c)` reaches `tau` | **never** | 412 | n/a |
| mask rate at `tau`, trained network | 0.000 | 0.610 | 0.757 |
| impurity of retained labels | n/a — none retained | 0.045 | 0.055 |
| held-out outcome NLL | 1.230 | 1.180 | 1.180 |

**The curriculum has an absorbing state at zero, and this fixture is in it.**
CPL raises a class's threshold only after rows clear the *fixed* `tau`
(Alg. 1 line 14), and until one does, `T(c) = 0` and eq. (8) trains `p(t | x)`
on the hard arg max of every row in the batch. On this fixture that term — a
mean over 512 rows against eq. (10)'s 64, at the same weight — pins the
propensity below `tau` for all 3,000 steps, so no mark is ever laid, so the
threshold never leaves zero. Positive feedback with a fixed point at the bottom
of its own range.

The `lambda = 0` column is what makes that a claim about the mechanism rather
than about the wiring. The same objective, the same state, the same diagnostics,
descended by nothing: rows start clearing `tau` at step 97, `sigma` rises
monotonically to 85% of the population, the better-learned class reaches
`T(c) = tau` at step 412 while the other sits at 0.43, and coverage falls from
1.00 to 0.89 as the gate closes on the difference. Every part of §3 does exactly
what it says. Tier 0 makes the other half of the same point without training
anything: at `beta(c) = 1` the objective's value, `n` and coverage equal
`PseudoLabelTreatmentNLL`'s on the same batch, to the bit.

The lock is not confined to the unlabelled term, which is the detail that says
how strong it is. Eq. (10)'s labelled cross-entropy in the declared arm goes
0.805 → 0.801 over 3,000 steps and stays **above `log 2 = 0.693`**: the
propensity ends confidently wrong on the very rows whose treatment it was
given. Both other arms take the same term to under 0.15 from the same
initialisation and the same batches.

Two one-off diagnostics rule out the two obvious alternative explanations. On
the **imbalanced** variant (3,000 steps) the lock is identical — no mark, no
threshold, held-out NLL 0.911 against a 0.445 baseline, where `fixmatch` reaches
0.252 — so it is not an artefact of the balanced two-class regime that §2's
third limitation warned would leave CPL inert. On the **separable** variant
(1,500 steps), where the propensity is close to trivial and `fixmatch` reaches
0.099 with a mask rate of 1.0, FlexMatch still marks nothing and still ends at
0.828. So it is not that the task is too hard for the model to reach `tau`; it
is that this term, undescended by a gate, prevents it.

**What this is and is not evidence about.** It is evidence about CPL composed
with a 3×200 MLP, two classes, a 64/448 quota and `lambda = 1` — a regime where
the unfiltered term outweighs the labelled one by construction. It is not
evidence against the paper, whose gains are reported on 10- and 100-class image
benchmarks with a Wide ResNet, where the same first steps spread the same
damage over ten times as many classes and the supervised term is far stronger
relative to it. The honest summary is that FlexMatch's warm-up trades FixMatch's
"the gate is the curriculum" (§2.2 of that paper) for a curriculum that has to
be started by the very confidence the gate was protecting, and that this
project's backbone cannot pay the entry price.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | — | Not run |

**No Tier 2 runner exists for this card yet**, and that is stated here rather
than left to be inferred from an empty table. `xty2/evaluation/benchmarks/` has
one module per recipe that has been Tier 2'd and none for this one, so §6's
target is declared and unmeasured — the same position `doublematch` shipped in.
Writing that runner is what turns §6.2's single seed into the ten-seed result
§6 asks for, and it is the next thing this card needs; the status line stays
`draft` until it has one and a reviewer has signed §8.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Whether FlexMatch inherits FixMatch's footnote 2 — that `U` contains the labelled rows too, without their labels — and therefore what `N` counts | It does. `U` is every training row, `N = \|train\| = 1024`, and eq. (8)'s eligible set is `all` | Convention, and consistency: §1 defines FlexMatch as CPL applied to FixMatch, §2 restates FixMatch's framework unchanged and §4 adopts its settings, so the framework is inherited whole. `fixmatch.md` §3.2 reached the same reading for the same term |
| What `M` is in the *reported* results — §3.3 proposes `x/(2-x)` "for our experiments" and Alg. 1 line 11 cites eq. (7), the identity mapping | The convex `x/(2-x)` of eq. (12) | The paper's own words: §3.3 chooses it for its experiments, and §4.4's ablation reports the convex function best and the concave worst. Algorithm 1's citation of eq. (7) is read as the general form eq. (12) specialises, since §3.3 says eq. (7) "can be seen as a special case by setting `M` to the identity function" |
| Whether the threshold warm-up is on in the reported results | On | Algorithm 1 lines 6–9 make it part of the procedure rather than an option, and §3.2 introduces it as a correction to eq. (6) rather than as an alternative to it |
| Whether a mark is ever cleared — a row that clears `tau` once and later stops clearing it | It is not. Marks are sticky and only ever overwritten by another class (Alg. 1 line 15) | The paper's own procedure: line 15 is the only write, and there is no line that restores `-1`. It also matters, because sticky marks make `sigma` monotone in aggregate and therefore make the thresholds' rise monotone-ish, which is the curriculum the method is named for |
| Whether Nesterov momentum is used | Yes | §4 gives "SGD with a momentum of 0.9" and states that FlexMatch adopts FixMatch's hyperparameters; FixMatch's table 4 states Nesterov. Guess where the two documents leave a gap, and the same guess `fixmatch` already runs, which is what keeps §6's pair a pair |
| The strict vs non-strict comparison at the gate: eqs. (5) and (8) and Alg. 1 line 14 all write `>`, where FixMatch's eq. (4) writes `>=` | `>`, as FlexMatch writes it | The paper. It differs from `fixmatch`'s objective in a set of measure zero — the exact tie `max(q) == T` — and is recorded only so that a reader comparing the two objectives' source does not read it as a transcription error |
| What happens when every class has been marked and `max_c sigma_t = 0` cannot occur, versus the degenerate `N = 0` | `N = 0` is rejected at stage start rather than divided by | Convention. A training population with no rows is a fixture error and the objective says so, rather than producing `0/0` thresholds that would silently admit everything |
| Whether the EMA copy or the current network supplies `q_b` and the marks | The current network | Algorithm 1 makes no mention of an EMA in lines 13–17, and §4's EMA sentence is about evaluation. The same reading `fixmatch.md` §7 records |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
