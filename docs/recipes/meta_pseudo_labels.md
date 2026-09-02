# Recipe spec card: meta_pseudo_labels

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to review the optimisation boundary. This card
> deliberately stops before implementation; §6 predeclares the evidence that
> would be required after the executor contract is reviewed.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) |
| Authors, year | Hieu Pham, Zihang Dai, Qizhe Xie, Minh-Thang Luong and Quoc V. Le; 2021 |
| DOI / arXiv | [arXiv:2003.10580v4](https://arxiv.org/abs/2003.10580v4); [10.48550/arXiv.2003.10580](https://doi.org/10.48550/arXiv.2003.10580); CVPR 2021, pp. 11557–11568 |
| Version used | arXiv v4, 2021-03-01. Core optimisation: §2 eqs. (1)–(3); hard-label estimator: appendix A eqs. (10)–(12); UDA composition: appendix B algorithm 1; stabilisation and hyperparameters: appendix C.3–C.4. |
| Reference implementation | [`google-research/google-research/meta_pseudo_labels`](https://github.com/google-research/google-research/tree/ec13eb6661a7b9500016cc6d7e3ab940c2dbf184/meta_pseudo_labels) @ `ec13eb6661a7b9500016cc6d7e3ab940c2dbf184`, especially [`training_utils.py`](https://github.com/google-research/google-research/blob/ec13eb6661a7b9500016cc6d7e3ab940c2dbf184/meta_pseudo_labels/training_utils.py). The paper is authoritative where the pinned code and v4 disagree about hard versus soft pseudo-labels; see §7. |
| Reference impl. runnable? | No on the project machine. The released path is TensorFlow 1.x and TPU-only; source inspection was used. |

## 2. Estimand and claim

- **Estimand:** the categorical treatment distribution `p(t | x)` learned by
  the final student. The mechanism estimand is the paired change in held-out
  treatment NLL caused by allowing a labelled-batch student feedback signal to
  update the pseudo-label teacher, holding teacher UDA, data, batches, views,
  architecture, optimiser steps and random seeds fixed.
- **Claim:** Meta Pseudo Labels (MPL) trains an independent teacher to choose
  pseudo-labels that improve a student's loss on labelled examples after one
  student update. The paper presents this as a one-step approximation to a
  bilevel objective and reports that MPL improves an already strong UDA teacher
  on low-label image benchmarks. In the selected project-local experiment the
  falsifiable claim is narrower: centred cosine feedback improves the final
  student's treatment NLL over the otherwise identical zero-feedback arm.
- **Nearest shipped baseline:** `uda`. The outer teacher reuses the reviewed
  `uda.md` treatment-classification graph, weak/strong views, confidence gate,
  sharpening and TSA policy. The controlled addition is an independently
  parameterised student plus the labelled-feedback update of the teacher.
- **Important terminology:** the backlog's “student validation loss” is
  shorthand. The paper samples a labelled training minibatch on every
  iteration; it does not reserve or tune on a validation split for this update.
  This card therefore calls it the **labelled feedback loss**.
- **Not claimed:** no published image accuracy is reproduced; no causal effect
  or outcome distribution is estimated; no theorem transfers from the paper's
  image classification setting to treatment labels; the labelled feedback
  batch is not an unbiased model-selection set; and a gain would not establish
  that longer unrolls, exact higher-order differentiation, or MPL's final
  labelled fine-tuning are beneficial.

## 3. Equations and mapping

### 3.1 As published

With teacher `T`, student `S`, parameters `theta_T`, `theta_S`, unlabelled batch
`x_u`, labelled batch `(x_l, y_l)`, and batch-mean cross-entropy `CE`, the
pseudo-label student objective is (§2 eq. (1))

$$
\theta_S^{\mathrm{PL}}
=\arg\min_{\theta_S}\;
\mathbb E_{x_u}\left[
  \mathrm{CE}\left(T(x_u;\theta_T),S(x_u;\theta_S)\right)
\right]
\equiv \arg\min_{\theta_S}\mathcal L_u(\theta_T,\theta_S).
\tag{1}
$$

MPL makes the teacher minimise the labelled loss of the student that results
from that inner fit (§2 eq. (2)):

$$
\min_{\theta_T}\;\mathcal L_l\!\left(\theta_S^{\mathrm{PL}}(\theta_T)\right),
\qquad
\theta_S^{\mathrm{PL}}(\theta_T)
=\arg\min_{\theta_S}\mathcal L_u(\theta_T,\theta_S).
\tag{2}
$$

The paper replaces the full inner optimisation with one student step (§2 eq.
(3)):

$$
\min_{\theta_T}\quad
\mathcal L_l\!\left(
  \theta_S-\eta_S\nabla_{\theta_S}
  \mathcal L_u(\theta_T,\theta_S)
\right).
\tag{3}
$$

For the hard-label variant selected by v4, sample
`y_hat_u ~ T(x_u; theta_T)`, update the student once,

$$
\theta'_S=\theta_S-\eta_S\nabla_{\theta_S}
\mathrm{CE}(\widehat y_u,S(x_u;\theta_S)),
$$

and use appendix A eq. (12):

$$
\nabla_{\theta_T}\mathcal L_l
=h\,\nabla_{\theta_T}
  \mathrm{CE}(\widehat y_u,T(x_u;\theta_T)),
\tag{12}
$$

$$
h=\eta_S
\left(
\nabla_{\theta'_S}\mathrm{CE}(y_l,S(x_l;\theta'_S))^\top
\nabla_{\theta_S}\mathrm{CE}(\widehat y_u,S(x_u;\theta_S))
\right).
$$

Appendix C.3 replaces the dot product in `h` by gradient cosine similarity and
centres it by a moving-average baseline. Appendix B algorithm 1 adds the
teacher's labelled loss and UDA loss to the MPL feedback gradient; the student
continues to learn only from teacher pseudo-labels during this phase.

### 3.2 Mapping to xty2

One proposed `meta_gradient` stage owns two independent instances of the same
port graph, named `outer_teacher` and `inner_student`. Within one optimiser step
the executor must, in this order:

1. evaluate both roles at their pre-update parameters and sample one hard
   treatment per eligible missing-treatment row from the teacher;
2. compute and retain the student's pseudo-label gradient at `theta_S`;
3. apply exactly one student optimiser update to obtain `theta'_S`;
4. compute the student's observed-treatment gradient at `theta'_S` on the
   labelled rows from the same quota batch;
5. form `h_raw` as the cosine of steps 2 and 4, form
   `h = stop_gradient(h_raw - b)`, then update
   `b <- 0.99 b + 0.01 h_raw`;
6. update the teacher once from its TSA-labelled loss, UDA consistency loss,
   and `h * CE(y_hat_u, T(x_u; theta_T))`.

An ordinary sequence of xty2 stages cannot express this. A checkpoint edge is
available only after a stage finishes, whereas eq. (12) needs the pre-update
student gradient, the post-update student gradient, and the current teacher
score in one atomic iteration. `TeacherSpec` is also inapplicable: it creates a
non-trainable EMA copy after a student update, not an independently optimised
outer model.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component / executor action |
|---|---|---|---|
| `T(x_u; theta_T)` | trainable teacher treatment distribution | `T_GIVEN_X @ strong_x, role=outer_teacher` | `MLPEncoder` + `CategoricalPropensity`; hard categorical sample owned by `meta_gradient` |
| `S(x_u; theta_S)` | pre-update student treatment distribution | `T_GIVEN_X @ strong_x, role=inner_student` | `MLPEncoder` + `CategoricalPropensity`; `SampledTeacherTreatmentNLL`, rows `t_missing` |
| `theta'_S` | student after exactly one pseudo-label update | same student ports at ephemeral post-update state | executor-owned one-step state; never an artifact or recipe callback |
| `L_l(theta'_S)` | labelled feedback loss | `T_GIVEN_X @ weak_x, role=inner_student, state=post_update` | `ObservedTreatmentNLL`, rows `t_observed`; gradient used only to form `h` |
| `h_raw` | alignment of post-update labelled and pre-update pseudo-label gradients | — | `MetaFeedbackCoefficient(kind="cosine")` inside `meta_gradient` |
| `b` | variance-reduction baseline | — | stage-local scalar state, initial value `0`, EMA decay `0.99` |
| `h * CE(y_hat_u,T(x_u))` | score-function teacher feedback | `T_GIVEN_X @ strong_x, role=outer_teacher` | `MetaPseudoLabelScore`, rows `t_missing`; sampled labels and `h` detached |
| `g_T,supervised` | teacher labelled objective | `T_GIVEN_X @ weak_x, role=outer_teacher` | shipped `TrainingSignalAnnealedTreatmentNLL`, rows `t_observed` |
| `g_T,UDA` | teacher weak-to-strong consistency | weak + strong `T_GIVEN_X`, role `outer_teacher` | shipped `ConfidenceMaskedConsistencyLoss`, rows `t_missing` |
| — | tabular weak view | — | shipped `ViewSpec("weak_x", FeatureMask(p=0.1))` |
| `RandAugment(x_u)` | stronger label-preserving view | — | shipped `ViewSpec("strong_x", FeatureMask(p=0.1), FeatureMask(p=0.1))` (deviation 2) |

No outcome component or objective appears in this graph. `Y_RAW` must therefore
be unreachable from both roles, and `used_y=false` is an executable invariant.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    meta_train.student_pseudo_label_nll: sampled teacher treatment and outer_teacher graph
    meta_train.student_labelled_feedback_nll: none with respect to post-update inner_student; resulting coefficient h detached before teacher update
    meta_train.teacher_meta_score: sampled treatment and h; gradient reaches outer_teacher score only
    meta_train.teacher_tsa_nll: none
    meta_train.teacher_uda_consistency: weak outer_teacher target only
  detached_targets:
    student_pseudo_label_nll: hard categorical sample
    teacher_meta_score: hard categorical sample and centred feedback coefficient
    teacher_uda_consistency: weak soft target
  gradient_clipping:
    inner_student: global_norm 5.0
    outer_teacher: global_norm 5.0
  marginal_nll_grad_path: n/a

teacher:
  ema_decay: n/a                         # the MPL teacher is an independent trainable role, not TeacherSpec
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: false           # no EMA TeacherSpec is constructed; outer_teacher trainability is declared by meta_gradient

losses:
  reduction:
    meta_train.student_pseudo_label_nll: mean
    meta_train.student_labelled_feedback_nll: mean
    meta_train.teacher_meta_score: mean
    meta_train.teacher_tsa_nll: mean
    meta_train.teacher_uda_consistency: mean
  eligible_rows:
    meta_train.student_pseudo_label_nll: t_missing
    meta_train.student_labelled_feedback_nll: t_observed
    meta_train.teacher_meta_score: t_missing
    meta_train.teacher_tsa_nll: t_observed
    meta_train.teacher_uda_consistency: t_missing
  weights:
    meta_train.student_pseudo_label_nll: 1.0
    meta_train.student_labelled_feedback_nll: 0.0  # gradient probe only; it is not applied to the student optimiser
    meta_train.teacher_meta_score: 1.0
    meta_train.teacher_tsa_nll: 1.0
    meta_train.teacher_uda_consistency: 1.0
  schedules:
    meta_train.student_pseudo_label_nll: constant 1.0
    meta_train.student_labelled_feedback_nll: constant 0.0
    meta_train.teacher_meta_score: constant 1.0
    meta_train.teacher_tsa_nll: constant 1.0
    meta_train.teacher_uda_consistency: constant 1.0
  temperature:
    hard_teacher_sample: 1.0
    teacher_uda_target: 0.4
  sharpening:
    hard_teacher_sample: none
    teacher_uda_target: softmax_temperature
  confidence_threshold:
    hard_teacher_sample: none
    teacher_uda: uda(unsupervised=0.8, tsa=exp_schedule(scale=5, steps=3000))

optimisation:
  optimiser:
    inner_student: sgd(momentum=0.9, nesterov=True)
    outer_teacher: sgd(momentum=0.9, nesterov=True)
  lr:
    inner_student: 0.03
    outer_teacher: 0.03
  lr_schedule:
    inner_student: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))
    outer_teacher: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))
  weight_decay:
    inner_student: 0.0005 (all parameters)
    outer_teacher: 0.0005 (all parameters)
  batch_size: 512
  labelled_unlabelled_ratio: 7.0
  total_steps_or_epochs: meta_train 3000 atomic student-then-teacher optimiser steps

architecture:
  widths_depths:
    outer_teacher.mlp_encoder: [200, 200, 200]
    outer_teacher.categorical_propensity: linear 200 -> K
    inner_student.mlp_encoder: [200, 200, 200]
    inner_student.categorical_propensity: linear 200 -> K
  activation:
    outer_teacher.mlp_encoder: elu
    outer_teacher.categorical_propensity: linear logits
    inner_student.mlp_encoder: elu
    inner_student.categorical_propensity: linear logits
  normalisation:
    outer_teacher.mlp_encoder: row_l2
    outer_teacher.categorical_propensity: none
    inner_student.mlp_encoder: row_l2
    inner_student.categorical_propensity: none
  dropout:
    outer_teacher: 0.0
    inner_student: 0.0
  initialisation:
    outer_teacher: independent normal std=0.1/sqrt(fan_in), bias=0
    inner_student: independent normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    outer_teacher.categorical_propensity: K softmax logits
    inner_student.categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'train'
  outcome_scaling: n/a                    # Y_RAW is unreachable
  treatment_encoding: integer classes 0..K-1; hard samples use one categorical draw per t_missing row
  split_protocol: project-local seed-locked train/held-out fixture in §6.1; no validation split; labelled feedback rows come from the training quota
  missingness_mechanism: treatment MCAR to exactly 64 observed training rows, keyed by row_id
```

`losses.weights.meta_train.student_labelled_feedback_nll=0.0` means the value is
a gradient probe, not an ordinary mixer contribution. The proposed executor
must print that distinction and reject any declaration that both probes and
applies this loss, because the paper's student does not learn directly from
labelled rows during MPL training.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply MPL to categorical treatment prediction on a fixed tabular DGP and omit the outcome stack. | The experiment is the backlog's optimisation-boundary test. Adding outcome likelihoods would introduce a second interaction before the meta-gradient executor is validated. | No published image number applies. The result speaks only to held-out treatment NLL. |
| 2 | `judgement` | — | Replace image RandAugment with UDA's shipped 10% weak mask and composed 10% + 10% strong mask. | Image transforms have no tabular semantics. Reusing the reproduced UDA views keeps the nearest baseline fixed and retains its predeclared label-flip check. | Narrows the invariance family; both paired arms share it exactly. |
| 3 | `judgement` | — | Reuse `uda.md`'s teacher policy and hyperparameters instead of paper table 8's CIFAR values: 3,000 steps, lr `0.03`, temperature `0.4`, confidence `0.8`, exponential TSA, no label smoothing, and a 64:448 quota. | The card tests one new mechanic against the shipped baseline. Simultaneously changing UDA would make a gain unattributable. | Reduces training horizon and changes feedback scale; the paired comparison remains interpretable but is not a paper reproduction. |
| 4 | `judgement` | — | Use independent tabular MLP propensity graphs rather than two WideResNet-28-2 classifiers. | Architecture is an orthogonal component and the project does not ingest images. | Changes optimisation geometry, so even the sign of MPL's effect is empirical. |
| 5 | `judgement` | — | Stop after joint MPL training; do not fine-tune the student on labelled rows. | The paper's §3.2 fine-tuning is a later supervised mechanism. The primary question is whether labelled feedback improves the pseudo-label learning path itself. | Likely understates final supervised accuracy; avoids washing out or creating the paired MPL difference. |
| 6 | `judgement` | — | Use one hard categorical sample per missing row and paper-v4's score-function estimator, not the pinned code's detached soft distribution used as both target and teacher logits. | v4 §2 and appendix A explicitly select hard samples. The authors' public issue discussion identifies the old soft path as inconsistent with the derived feedback formula. | Adds sampling variance but preserves the teacher feedback gradient described by eqs. (10)–(12). |
| 7 | `judgement` | — | Define baseline order as `h = h_raw - b_t`, then update `b_{t+1}=0.99b_t+0.01h_raw`; reset `b_0=0` for every execution. | Appendix C.3 specifies a moving baseline but not initialisation or whether the current observation enters before subtraction. The order is fixed here for paired reproducibility. | The first few feedback magnitudes differ from an update-then-subtract reading; both arms share all non-meta updates. |
| 8 | `judgement` | — | Treat appendix B's UDA argument as `x_u`, matching its prose and the reference implementation, rather than the displayed `x_l` in algorithm 1 line 15. | UDA is introduced there as the teacher's unlabelled-data objective; `x_l` would duplicate a labelled augmentation term and contradict the surrounding text. | Preserves the shipped UDA comparison. |

No paper mechanic is intentionally omitted because of the current framework.
The missing executor is fidelity-bearing and in scope after review under
`DESIGN.md` §11.2; it is therefore proposed below rather than misclassified as
a `framework-limitation`.

### 5.1 Framework additions made for this card

These are proposed additions, not implemented ones. Card review decides their
shape before code is written.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `meta_gradient` executor: two independently trainable role-tagged graphs, separate optimisers, an atomic one-step inner update, pre/post gradient probes, ordered outer update, stage-local scalar state, and role-tagged checkpoint output | fidelity-bearing, load-bearing vocabulary | this card | The validation-driven-weighting family named in `BACKLOG.md` §10.1, concretely Ren et al., *Learning to Reweight Examples for Robust Deep Learning* (2018): its clean validation loss is evaluated after a one-step weighted inner update and differentiated to the outer example weights. The checked common shape is explicit inner/outer parameter ownership, bounded unroll length, pre/post state, and an outer loss; the executor must not assume the outer parameters are a second neural network. | Eq. (3) and appendix B require this ordering inside every iteration. Ordinary stages expose only completed checkpoints, and `TeacherSpec` is frozen EMA state. Hiding the loop in a recipe function would remove update order, gradient reach and optimiser ownership from the plan. |
| `SampledTeacherTreatmentNLL`, `MetaPseudoLabelScore`, and `MetaFeedbackCoefficient` | fidelity-bearing, reversible | this card | not required (ordinary objective/executor helpers; no shared vocabulary) | Existing pseudo-label objectives use an argmax from the same graph. MPL requires a seeded categorical sample from one role, reused by the other role and by a score-function term, plus a gradient-alignment scalar. Keeping these small objects explicit makes the sampled target, detach points and coefficient visible to Tier 0. |

The executor contract must remain narrower than a general differentiable
programming system: exactly one inner step, exactly one outer step, no arbitrary
stage DAG, no implicit higher-order unroll, and no user callback that mutates
parameters. Widening any of those is a new card/design decision.

## 6. Reproduction target

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in §6.1 and inherited from uda.md
  variant: paired MPL feedback versus h=0; same initial teacher/student parameters, batches, views, hard-label RNG and UDA objectives
  split: 1024 train rows with exactly 64 observed treatments; 2048 held-out rows with every treatment observed
  metric: held-out inner-student treatment NLL ratio, MPL over h=0; outer-teacher NLL, sampled-label accuracy, h trajectory and prediction concentration as diagnostics
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: student NLL ratio < 1.0 in mean by at least one stderr; outer-teacher NLL <= 1.10x h=0; finite losses and gradients in every replicate; terminal student prediction concentration < 0.95
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

Reuse `uda.md` §6.1 without changing a draw. For replicate `r=0..9`, base seed
is `94000+100r`; training generation uses `s_r+1`, held-out generation `s_r+2`,
outer-teacher initialisation `s_r+6`, inner-student initialisation `s_r+7`,
stage/view RNG `s_r+10000`, and hard-label sampling `s_r+20000`.

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

Exactly 64 training treatments are observed by the inherited seeded MCAR
permutation; all held-out treatments and every outcome are observed. Outcomes
remain in `XTYBatch` only to assert that neither role reaches `Y_RAW`. Both arms
use the identical ordered quota stream of 64 observed and 448 missing rows.

### 6.2 Predeclared evidence

**Tier 0 (invariants).**

1. The two role graphs have identical declarations and disjoint parameter and
   optimiser state; changing one role cannot mutate the other.
2. A seeded categorical draw is reproducible, has one class per missing row,
   and is reused bit-identically by the student NLL and teacher score term.
3. The student update changes only `inner_student`; teacher objectives change
   only `outer_teacher`.
4. The pseudo-label gradient in `h_raw` is evaluated at pre-update student
   parameters and the labelled gradient at post-update parameters. Swapping
   either state fails an analytic two-parameter fixture.
5. `h_raw` equals a direct flattened-gradient cosine, lies in `[-1,1]`, and is
   zero under the defined zero-norm convention. `h` and sampled labels are
   detached before the teacher score gradient.
6. Baseline state starts at zero, follows the declared pre-subtraction update
   order, resets between executions and cannot leak between paired arms.
7. Setting the meta-score weight to zero leaves student updates, teacher TSA,
   teacher UDA, batches, views and RNG consumption bit-identical.
8. The labelled feedback loss contributes no student optimiser gradient even
   though its gradient is computed for `h`.
9. The teacher's UDA gate, target temperature, denominator and TSA arithmetic
   reproduce the shipped UDA invariants under `role=outer_teacher`.
10. Both graphs have `used_y=false`; no outcome component is trainable or
    checkpointed.
11. An ordinary `gradient` executor rejects the role-tagged meta objectives;
    `meta_gradient` rejects an EMA `TeacherSpec`, more or fewer than one inner
    step, shared optimiser state, or a missing update-order declaration.
12. The compiled plan and digest print both roles, both optimisers, the six-step
    order in §3.2, hard-sample temperature/RNG, gradient probe states, baseline
    lifecycle, and every non-`n/a` §4 key.

**Tier 1 (one-seed smoke).**

1. Run MPL and `h=0` from identical role parameters and paired streams for the
   full 3,000-step budget; require finite losses, gradients, `h`, probabilities
   and checkpoints.
2. Require both students to beat the held-out observed-frequency NLL. Do not
   require MPL to beat `h=0` on one seed.
3. Report student and teacher NLL, hard-label accuracy, entropy, feedback mean,
   standard deviation and sign fraction, baseline value, UDA coverage and TSA
   retained fraction.
4. Require non-degenerate feedback: at least one finite non-zero `h` and both
   teacher and student parameters move from initialisation. This is wiring
   evidence, not a performance result.
5. Diagnostic only: on a two-step toy batch, compare the implemented teacher
   update with a slow direct score-function calculation using the same sampled
   actions. Do not substitute a soft-label higher-order derivative.

**Tier 2.** Run the paired arms over all ten replicates. Only the student NLL
ratio, teacher guardrail, finiteness and concentration ceiling determine
`reproduced` versus `deviating`. The signs of hard-label accuracy and feedback
trajectory differences are reported rather than chosen after the run.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | — | — |

## 7. Unknowns

| Unspecified or conflicting source detail | Our choice | Basis |
|---|---|---|
| Paper v4 samples hard pseudo-labels, while the pinned public code uses a detached soft distribution for both the student target and the teacher MPL cross-entropy. | Follow v4's hard categorical sample and appendix A score-function derivation. | §2 after eq. (3), appendix A eqs. (10)–(12), and the authors' discussion in [`google-research/google-research#534`](https://github.com/google-research/google-research/issues/534). |
| Appendix C.3 says cosine similarity is crucial, while the pinned code estimates feedback with the before/after labelled-loss difference. | Use direct gradient cosine. | The card's paper-version rule and appendix C.3's explicit implementation instruction. |
| Moving-baseline decay, initial value and update order. | Scalar `b_0=0`, decay `0.99`; subtract the old baseline, then update it. | Pinned code supplies coefficient `0.01`; the remaining order is fixed in deviation 7. |
| Zero-norm cosine convention. | `h_raw=0` if either flattened gradient norm is zero. | Finite, neutral feedback; must be explicit for early or saturated batches. |
| Whether the labelled feedback batch is a validation set. | No. It is the observed-treatment quota from the training batch and may also train the teacher's supervised objective. | §2 alternating update and appendix B algorithm 1. |
| Which teacher view generates student pseudo-labels. | Teacher and student both use the same `strong_x` realisation for the pseudo-label step. | Pinned `training_utils.py` uses teacher `u_aug` logits and student `s_on_u` evaluated on `u_images_aug`; weak/strong remain separate for teacher UDA. |
| Whether hard samples are sharpened or confidence-gated. | Neither; draw from the ordinary temperature-1 teacher distribution on every missing row. | Eq. (10)'s expectation under `T(x_u;theta_T)` and algorithm 1 line 5. UDA's gate and temperature govern only the teacher's auxiliary consistency term. |
| Label smoothing in table 8 and the pinned CIFAR command. | None in this project-local card. | Controlled inheritance from `uda.md`; deviation 3 owns the departure. |
| Does the student receive a labelled loss during joint training? | No. The labelled loss is a gradient probe only. | Appendix B prose and algorithm 1; the paper explicitly says the student learns only from teacher pseudo-labels in this phase. |
| Does the student receive final labelled fine-tuning? | No in this card. | Deviation 5 isolates the meta-gradient mechanism. |
| Checkpoint/resume of the moving baseline and two optimisers. | Unsupported in the first implementation; runs are uninterrupted and reset from seed. | This card requires deterministic full executions, not bit-identical mid-stage resume. A future resume claim would engage `checkpointed-objective-state`. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
