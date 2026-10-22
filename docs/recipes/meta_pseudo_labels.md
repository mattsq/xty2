# Recipe spec card: meta_pseudo_labels

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to review or change the optimisation boundary.
> The implementation is `xty2/recipes/meta_pseudo_labels.py`; §6 owns its
> smoke and reproduction evidence.

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

The sign is load-bearing and the pinned code carries the opposite one. To
first order
`CE(y_l,S(x_l;theta'_S)) - CE(y_l,S(x_l;theta_S)) = -h`, so a positive `h`
means the student's step *reduced* the labelled loss and the teacher should
raise the probability of `y_hat_u`. The pinned implementation forms
`dot_product = cross_entropy['s_on_l_new'] - shadow` and adds
`cross_entropy['mpl'] * dot_product` to the teacher loss
(`training_utils.py:472` and `:496`), which is `-h`, not `h`. This card follows
eq. (12); §7 carries the discrepancy so that the pinned code cannot be used as
a sign check without it.

Appendix C.3 replaces the dot product in `h` by the gradients' **cosine
distance** — the paper's own words — and centres the result by a moving-average
baseline. Eq. (12) is a similarity, and reading "distance" literally as
`1 - cos` would invert the feedback, so §7 records the reading taken here.
Appendix B algorithm 1 adds the teacher's labelled loss and UDA loss to the MPL
feedback gradient; the student continues to learn only from teacher
pseudo-labels during this phase.

### 3.2 Mapping to xty2

One `meta_gradient` stage owns two independent instances of the same
port graph, named `outer_teacher` and `inner_student`. Within one optimiser step
the executor must, in this order:

1. evaluate both roles at their pre-update parameters and sample one hard
   treatment per eligible missing-treatment row from the teacher;
2. compute and retain the student's pseudo-label gradient at `theta_S`;
3. apply exactly one student optimiser update to obtain `theta'_S`;
4. compute the student's observed-treatment gradient at `theta'_S` on the
   labelled rows from the same quota batch;
5. form `h_raw` as the cosine similarity of steps 2 and 4, update
   `b <- 0.99 b + 0.01 h_raw`, then form `h = stop_gradient(h_raw - b)`
   against that already-updated baseline, which is the pinned order
   (`training_utils.py:479-483`);
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
| `h_raw` | alignment of post-update labelled and pre-update pseudo-label gradients | — | `MetaFeedbackCoefficient(kind="cosine_similarity")` inside `meta_gradient` |
| `b` | variance-reduction baseline | — | stage-local scalar state, initial value `0`, EMA decay `0.99` |
| `h * CE(y_hat_u,T(x_u))` | score-function teacher feedback | `T_GIVEN_X @ strong_x, role=outer_teacher` | `MetaPseudoLabelScore`, rows `t_missing`; sampled labels and `h` detached |
| `g_T,supervised` | teacher labelled objective | `T_GIVEN_X @ weak_x, role=outer_teacher` | shipped `TrainingSignalAnnealedTreatmentNLL`, rows `t_observed` |
| `g_T,UDA` | teacher weak-to-strong consistency | weak + strong `T_GIVEN_X`, role `outer_teacher` | shipped `ConfidenceMaskedConsistencyLoss`, rows `t_missing` |
| — | tabular weak view | — | shipped `ViewSpec("weak_x", FeatureMask(p=0.1))` |
| `RandAugment(x_u)` | stronger label-preserving view | — | shipped `ViewSpec("strong_x", FeatureMask(p=0.1), FeatureMask(p=0.1))` (deviation 2) |

No outcome component or objective appears in this graph. `Y_RAW` must therefore
be unreachable from both roles. The executable form of that is
`ComponentGraph.port_depends_on_raw_outcome` returning `False` for every port
either role reads: this recipe produces no artifact, so the `used_y` provenance
field on `PseudoLabels` has no carrier here and is not the check.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    meta_train.student_pseudo_label_nll: p(t|x) @ view=strong_x params=student role=outer_teacher
    meta_train.student_labelled_feedback_nll: none                          # gradient probe reads post-update inner_student
    meta_train.teacher_meta_score: none                                     # sampled target and h are listed under detached_targets
    meta_train.teacher_tsa_nll: none                                     # inherited from uda.md
    meta_train.teacher_uda_consistency: p(t|x) @ view=weak_x params=student role=outer_teacher
  detached_targets:
    student_pseudo_label_nll: hard categorical sample and outer_teacher graph
    teacher_meta_score: hard categorical sample and centred feedback coefficient  # appendix A eq. (12); ref impl training_utils.py:483
    teacher_uda_consistency: target                                     # inherited from uda.md
  gradient_clipping:
    inner_student: none          # ref impl clips by params.grad_bound, whose flag_utils.py:100 default 1e9 is inert; uda.md declares none
    outer_teacher: none          # same call site, training_utils.py:500; flag_utils.py:132 defines teacher_grad_bound=20 but this commit does not read it
  marginal_nll_grad_path: n/a    # no marginal term; deviation 1 omits the outcome stack

teacher:
  ema_decay: n/a                         # the MPL teacher is an independent trainable role, not TeacherSpec. No evaluation EMA either: deviation 9
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a             # no TeacherSpec is constructed; outer_teacher is independently trainable

losses:
  reduction:
    meta_train.student_pseudo_label_nll: mean          # ref impl training_utils.py:415 divides by the fixed unlabelled count
    meta_train.student_labelled_feedback_nll: mean     # ref impl training_utils.py:421 divides by the fixed labelled count
    meta_train.teacher_meta_score: mean                # ref impl training_utils.py:490
    meta_train.teacher_tsa_nll: mean                   # inherited from uda.md; retained-row mean is objective arithmetic
    meta_train.teacher_uda_consistency: mean           # inherited from uda.md; rejected rows stay in the denominator
  eligible_rows:
    meta_train.student_pseudo_label_nll: t_missing     # algorithm 1 line 6
    meta_train.student_labelled_feedback_nll: t_observed  # algorithm 1 line 7
    meta_train.teacher_meta_score: t_missing           # algorithm 1 line 8
    meta_train.teacher_tsa_nll: t_observed             # algorithm 1 line 9
    meta_train.teacher_uda_consistency: t_missing      # algorithm 1 line 10; deviation 8
  weights:
    meta_train.student_pseudo_label_nll: 1.0           # algorithm 1 line 6
    meta_train.student_labelled_feedback_nll: 0.0      # gradient probe only; it is not applied to the student optimiser
    meta_train.teacher_meta_score: 1.0                 # ref impl training_utils.py:496 adds it unweighted
    meta_train.teacher_tsa_nll: 1.0                    # ref impl training_utils.py:495
    meta_train.teacher_uda_consistency: 1.0            # uda_weight default 1.0, flag_utils.py:146; deviation 10 drops its ramp
  schedules:
    meta_train.student_pseudo_label_nll: constant 1.0
    meta_train.student_labelled_feedback_nll: constant 0.0
    meta_train.teacher_meta_score: constant 1.0        # the feedback magnitude is h, not a schedule
    meta_train.teacher_tsa_nll: constant 1.0           # the TSA ceiling is gate arithmetic under losses.confidence_threshold
    meta_train.teacher_uda_consistency: constant 1.0   # ref impl ramps over uda_steps; deviation 10
  temperature:
    student_pseudo_label_nll: 1.0       # eq. (10) samples from T(x_u; theta_T) itself
    teacher_uda_consistency: 0.4        # inherited from uda.md; deviation 3
  sharpening:
    student_pseudo_label_nll: none      # eq. (10); deviation 6
    teacher_uda_consistency: softmax_temperature   # inherited from uda.md
  confidence_threshold:
    student_pseudo_label_nll: none      # algorithm 1 line 5 gates nothing
    teacher_tsa_nll: uda(unsupervised=0.8, tsa=exp_schedule(scale=5, steps=3000))
    teacher_uda_consistency: uda(unsupervised=0.8, tsa=exp_schedule(scale=5, steps=3000))  # inherited from uda.md; deviation 3

optimisation:
  optimiser:
    inner_student: sgd(momentum=0.9, nesterov=True)    # inherited from uda.md; deviation 3
    outer_teacher: sgd(momentum=0.9, nesterov=True)    # inherited from uda.md; deviation 3
  lr:
    inner_student: 0.03                                # inherited from uda.md; deviation 3 (ref impl mpl_student_lr=0.1, flag_utils.py:157)
    outer_teacher: 0.03                                # inherited from uda.md; deviation 3 (ref impl mpl_teacher_lr=0.1, flag_utils.py:161)
  lr_schedule:
    inner_student: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))   # inherited from uda.md; no warmup or wait steps, deviation 3
    outer_teacher: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))   # inherited from uda.md; deviation 3
  weight_decay:
    inner_student: 0.0005 (all trainable components; all parameters)     # inherited from uda.md; deviation 3
    outer_teacher: 0.0005 (all trainable components; all parameters)     # inherited from uda.md; deviation 3
  batch_size: 512                                      # inherited from uda.md; deviation 3
  labelled_unlabelled_ratio: 7.0                       # 64:448, inherited from uda.md; deviation 3
  total_steps_or_epochs: 3000                           # atomic student-then-teacher steps; inherited from uda.md

architecture:
  widths_depths:
    outer_teacher.mlp_encoder: [200, 200, 200]         # inherited from uda.md; deviation 4
    outer_teacher.categorical_propensity: linear 200 -> K
    inner_student.mlp_encoder: [200, 200, 200]         # deviation 4; the paper gives teacher and student the same architecture
    inner_student.categorical_propensity: linear 200 -> K
  activation:
    outer_teacher.mlp_encoder: elu                     # inherited from uda.md
    outer_teacher.categorical_propensity: linear logits
    inner_student.mlp_encoder: elu
    inner_student.categorical_propensity: linear logits
  normalisation:
    outer_teacher.mlp_encoder: row_l2                  # inherited from uda.md
    outer_teacher.categorical_propensity: none
    inner_student.mlp_encoder: row_l2
    inner_student.categorical_propensity: none
  dropout:
    outer_teacher.mlp_encoder: 0.0                     # inherited from uda.md
    outer_teacher.categorical_propensity: 0.0
    inner_student.mlp_encoder: 0.0
    inner_student.categorical_propensity: 0.0
  initialisation:
    outer_teacher.mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    outer_teacher.categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
    inner_student.mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    inner_student.categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    outer_teacher.categorical_propensity: K softmax logits   # eq. (1) is a categorical cross-entropy
    inner_student.categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'train'           # inherited from uda.md §6.1
  outcome_scaling: y: none fitted on 'train'           # DataSpec contract; Y_RAW remains unreachable from both role graphs
  treatment_encoding: integer classes 0..K-1; one categorical draw per t_missing row   # XTYBatch contract; eq. (10)
  split_protocol: one fixed project-local DGP, split train/test by the section 6.1 fixture; no image protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id
```

`losses.weights.meta_train.student_labelled_feedback_nll=0.0` means the value is
a gradient probe, not an ordinary mixer contribution. The executor
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
| 7 | `judgement` | — | Follow the pinned update-then-subtract order — `b_{t+1}=0.99b_t+0.01h_raw`, then `h=h_raw-b_{t+1}` — and additionally fix `b_0=0` with a reset at every execution. | Appendix C.3 specifies a moving baseline but neither its order nor its initialisation; `training_utils.py:479-483` fixes the order under `tf.control_dependencies`, so only `b_0` and the reset are ours to choose. The reference variable carries no initialiser, and a paired comparison needs a deterministic start. | Feedback magnitudes over the first few steps depend on `b_0`; the arms share initial state and exogenous streams, while later endogenous trajectories may diverge after feedback. |
| 8 | `judgement` | — | Treat appendix B's UDA argument as `x_u`, matching its prose and the reference implementation, rather than the displayed `x_l` in algorithm 1 line 10. | UDA is introduced there as the teacher's unlabelled-data objective; `x_l` would duplicate a labelled augmentation term and contradict the surrounding text. | Preserves the shipped UDA comparison. |
| 9 | `judgement` | — | Report the raw student and teacher parameters and construct no evaluation EMA, dropping `uda.md`'s `TeacherSpec(decay=0.9999, role="evaluation")`. | MPL's own `ema_decay` default is `0`, that is no moving average (`flag_utils.py:113`), so the inherited EMA is a UDA-side choice rather than a paper mechanic. Keeping it would add a second reported parameter set to a card whose question is one gradient path. | `uda.md`'s headline Tier 2 number is its EMA NLL, so §6 here is not directly comparable with that column; the paired arms are unaffected. |
| 10 | `judgement` | — | Hold the teacher's UDA consistency weight at a constant `1.0` rather than ramping it linearly over `uda_steps` as `training_utils.py:489-491` does. | `uda.md` is the nearest shipped baseline and already declares a constant weight; introducing MPL and a weight ramp in one card would make a difference unattributable. | Stronger early consistency pressure on the teacher than the reference schedule; both arms share it. |
| 11 | `judgement` | — | Replace `uda.md`'s two-cluster Tier 2 fixture with §6.1's minority trap: three clusters, a 5% minority beside a large cluster of the other treatment. Tier 1 keeps the two-cluster fixture. | The inherited fixture gives the teacher no confidently wrong region for feedback to correct: by step 3,000 the feedback no longer rewards better pseudo-labels (§6.4, §6.5). | Moves the Tier 2 target from overall student NLL to paired minority-region accuracy, with a declared weight-20 dose arm reported as information. §6 is no longer comparable with `uda.md`'s. |

No paper mechanic is intentionally omitted because of the current framework.
The executor is fidelity-bearing and was added under `DESIGN.md` §11.2 rather
than hiding the mechanic behind a `framework-limitation`.

### 5.1 Framework additions made for this card

These additions are implemented by the compiler and the bounded
`meta_gradient` executor. The second consumers below constrained their shape;
they remain design checks rather than speculative implementations.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `meta_gradient` executor: two independently trainable role-tagged graphs, separate optimisers, an atomic one-step inner update, pre/post gradient probes, ordered outer update, stage-local scalar state, and role-tagged checkpoint output | fidelity-bearing, load-bearing vocabulary | this card | The validation-driven-weighting family named in `BACKLOG.md` §10.1, concretely Ren et al., *Learning to Reweight Examples for Robust Deep Learning* (2018): its clean validation loss is evaluated after a one-step weighted inner update and differentiated to the outer example weights. The checked common shape is explicit inner/outer parameter ownership, bounded unroll length, pre/post state, and an outer loss; the executor must not assume the outer parameters are a second neural network. One shape difference is not covered and review should settle it: Ren et al. differentiates with respect to a per-example weight vector and therefore needs per-example inner gradients, whereas step 2 here retains a single aggregate gradient. The contract below fixes the update order, not the arity of the retained probe. | Eq. (3) and appendix B require this ordering inside every iteration. Ordinary stages expose only completed checkpoints, and `TeacherSpec` is frozen EMA state. Hiding the loop in a recipe function would remove update order, gradient reach and optimiser ownership from the plan. |
| `Realisation.role`: a named parameter set beyond the closed `params` literal, so one graph declaration can be evaluated under `outer_teacher` and `inner_student` weights in one step | fidelity-bearing, load-bearing vocabulary | this card | `BACKLOG.md` §11.2 and §15.6, concretely Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (2017): "we train an ensemble of `M` networks independently, with random initialisation", which is `M` named parameter sets over one declaration with per-set initialisation seeds and no EMA relationship. The checked common shape is a named, independently initialised, independently optimised parameter set that the plan prints and the compiler sorts; what MPL adds beyond an ensemble is per-role optimisers, and an ensemble adds aggregation this card does not need. | `Realisation.params` is `Literal["student","teacher"]` and its own guard says a third parameter set "is a new realisation axis: it needs a reviewed card whose §4 checklist cannot be honoured without it". This is that card: §4 declares two optimisers, two learning rates and two initialisations, and `teacher` already means a non-trainable EMA copy. |
| `Realisation.state`: a within-step position marker, here `post_update`, so the labelled gradient probe names the parameters it is evaluated at | fidelity-bearing, load-bearing vocabulary | this card | The same Ren et al. (2018) row below: its clean validation loss is evaluated strictly after the weighted inner step, so it needs the same pre/post distinction on one parameter set. Review may instead prefer to keep `state` out of the realisation key and carry the position in `meta_gradient`'s declared six-step order; the card asks for the axis because Tier 0 rule 13 requires the plan and digest to print which state each probe reads, and an executor-private position is not plan-visible. | Eq. (12) reads one parameter set at two states inside one iteration. Without a marker the two student gradients are indistinguishable in the plan, which is precisely the confusion rule 4 exists to catch. |
| `SampledTeacherTreatmentNLL`, `MetaPseudoLabelScore`, and `MetaFeedbackCoefficient` | fidelity-bearing, reversible | this card | not required (ordinary objective/executor helpers; no shared vocabulary) | Existing pseudo-label objectives use an argmax from the same graph. MPL requires a seeded categorical sample from one role, reused by the other role and by a score-function term, plus a gradient-alignment scalar. Keeping these small objects explicit makes the sampled target, detach points and coefficient visible to Tier 0. |

The executor contract must remain narrower than a general differentiable
programming system: exactly one inner step, exactly one outer step, no arbitrary
stage DAG, no implicit higher-order unroll, and no user callback that mutates
parameters. Widening any of those is a new card/design decision.

### Tier 2 outcome

On 2026-10-21, commit `c7921229b9fa` produced a `deviating` result: This is the predeclared project-local Meta Pseudo Labels mechanism target: whether appendix A's centred hard-label score-function feedback improves the inner student's treatment NLL over the otherwise identical h=0 execution. It is not a reproduction of a published image-accuracy number; the tabular DGP, views, architecture, and 3,000-step budget are the reviewed deviations listed in the card. Within noise of the target: student_treatment_NLL_ratio was 0.995541 +/- 0.00577 against mean <= 1, by at least one stderr, inside its target by 0.00446 — less than its own standard error, so the run does not distinguish it from a miss.

## 6. Reproduction target

```yaml
reproduction:
  dataset: project-local minority-trap XTY DGP (6 features, K=2): cluster_population(classes=3, groups=(2,1), within=0.6, noise=0.4, prior=(0.475,0.05,0.475)) relabelled by (0,1,1), specified in §6.1
  variant: paired MPL (meta-score weight 1.0) versus the forced-zero-feedback arm, plus a declared weight-20 dose arm reported as information
  metric: paired gain in inner-student accuracy on the held-out minority region (Bayes cluster posterior picks the minority cluster), MPL weight 1.0 minus h=0; h=0 minority accuracy as the fixture-validity guardrail; finiteness, outer-teacher NLL ratio and class-mass concentration as before; the dose arm's gains, both NLL ratios and the h diagnostics as information
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: minority-accuracy gain >= 0.10 in mean by at least one stderr; h=0 minority accuracy <= 0.60 (the trap binds); outer-teacher NLL <= 1.10x the zero-feedback arm; finite losses and gradients in every replicate; terminal student class-mass concentration < 0.95
  seeds: 100
  report: mean_and_stderr
```

**The zero-feedback arm.** The control is `h := 0` after `h_raw` is formed, not
the removal of the meta-score objective. Both arms pair the exogenous batch,
view and hard-label RNG streams and run the same probe work. Their endogenous
values are identical only through the first pre-feedback step. Once MPL changes
the teacher, later teacher probabilities, categorical samples, student updates,
feedback and UDA/TSA terms may diverge by design. With a fixed state and sample,
the zero-feedback teacher gradient is the same as deleting the meta-score term.

**Class-mass concentration.** Let
`p_bar[k] = mean_i p(t=k | x_i)` on held-out rows under the terminal inner
student. The guardrail is `max_k p_bar[k] < 0.95`: it allows confident but
balanced predictions while rejecting collapse onto a single class.

### 6.1 Fixed DGP

**Tier 2 fixture: the minority trap (§6.5, deviation 11).** For replicate
`r=0..99`, the base seed is `s_r = 94000+100r`; training generation uses `s_r+1`, held-out
generation `s_r+2`, outer-teacher initialisation `s_r+6`, inner-student
initialisation `s_r+7`, stage/view RNG `s_r+10000`, and hard-label sampling
`s_r+20000`. The offsets collide with nothing within this card while `r<100`:
`s_r+20000 = s_{r+100}+10000`, so a hundred replicates is the ceiling. Both the
training (1,024 rows) and held-out (2,048 rows) draws are
`cluster_population(classes=3, groups=(2, 1), within=0.6, noise=0.4,
prior=(0.475, 0.05, 0.475), labels=(0, 1, 1))`:

```text
cluster c ~ Categorical(0.475, 0.05, 0.475)
x[0:4]    = mu_c + 0.4 epsilon[0:4]      mu = cluster_centres(3, groups=(2,1), within=0.6)
x[4:6]    = epsilon[4:6]
d ~ Categorical: p(d = c | c) = 0.98, the remaining 0.02 split over the other two clusters
t         = (0, 1, 1)[d]
baseline  = 0.5x0 - 0.3x1 + 0.2(x4^2 - 1)
effect    = 1 + 0.5 tanh(x2)
y         = baseline + t * effect + 0.5 epsilon_y
```

Clusters 0 and 1 form the tight group, `1.08` apart; cluster 2 is `1.8` from
both. The **minority region** is the set of held-out rows whose exact cluster
posterior (`cluster_posterior`) picks cluster 1, about 4% of rows. The Bayes
rule (`treatment_posterior`) is right on about `0.83` of it, and its treatment
NLL is about `0.146` overall. View label-flip rates are reported against that
Bayes rule.

**Tier 1 fixture: unchanged.** Tier 1 keeps seed `94000` on `uda.md` §6.1's
two-cluster draw, without changing a draw. It checks wiring, not the §6 claim.

Inherited from `uda.md` §6.1 by both fixtures: the row counts, the
seeded MCAR permutation, the assertion that both treatment levels appear among
the observed rows, the shared quota stream, and the pre-training report of view
label-flip rates and treatment prevalence. Not inherited: outcome scaling,
which `uda.md` fits on the training population. This card applies no outcome
scaling and keeps `Y_RAW` unreachable from both role graphs.

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
remain in `XTYBatch` only to assert that neither role reaches `Y_RAW`. Every arm
uses the identical ordered quota stream of 64 observed and 448 missing rows.
The equations above the Tier 1 note describe the two-cluster draw only.

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
5. `h_raw` equals a direct flattened-gradient cosine **similarity**, lies in
   `[-1,1]`, and is zero under the defined zero-norm convention. `h` and
   sampled labels are detached before the teacher score gradient.
6. The feedback sign matches eq. (12) and not the pinned code's `dot_product`.
   On a fixture where the student's one step provably lowers the labelled loss,
   `h_raw > 0` and the teacher update strictly raises `T(y_hat_u | x_u)`;
   negating `h` fails. This is asserted directly because the pinned
   implementation's coefficient is `-h` (§7).
7. Baseline state starts at zero, follows the pinned update-then-subtract order
   so that `h` reads the already-updated `b`, resets between executions and
   cannot leak between paired arms.
8. Before the first feedback update, paired arms are bit-identical and forcing
   `h := 0` produces the same teacher gradient as deleting the meta-score term.
   Across the run, batch, view and hard-label RNG streams remain paired and
   consume identically, but endogenous values may diverge after teacher
   feedback changes the MPL arm.
9. The labelled feedback loss contributes no student optimiser gradient even
   though its gradient is computed for `h`.
10. The teacher's UDA gate, target temperature, denominator and TSA arithmetic
    reproduce the shipped UDA invariants under `role=outer_teacher`.
11. `port_depends_on_raw_outcome` is `False` for every port either role reads;
    no outcome component is trainable or checkpointed. This recipe produces no
    artifact, so there is no `used_y` field to assert on.
12. An ordinary `gradient` executor rejects the role-tagged meta objectives;
    `meta_gradient` rejects an EMA `TeacherSpec`, more or fewer than one inner
    step, shared optimiser state, or a missing update-order declaration.
13. The compiled plan and digest print both roles, both optimisers, the six-step
    order in §3.2, hard-sample temperature/RNG, gradient probe states, baseline
    lifecycle, and every non-`n/a` §4 key.

**Tier 1 (one-seed smoke).**

The implemented 3,000-step paired smoke test exercises the following checks at
seed `94000` and passes; it is not the ten-seed Tier 2 result.

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
5. Report weak and composed-strong Bayes-label flip rates beside every result,
   as `uda.md` §6.2 rule 8 requires of the views deviation 2 inherits. Require
   strong above weak. Do **not** treat the 5% ceiling as met by inheritance:
   `uda.md`'s own Tier 2 records `4.79% +/- 0.16%` across these ten fixtures
   with seed `94100` at `5.96%`, over that ceiling, and calls whether the
   ceiling should bind all ten fixtures a live review question. This card
   reuses that fixture family unchanged, so it inherits the breach; report the
   per-replicate rates and let review settle the ceiling. A breach is a
   data-policy finding, not evidence about MPL.
6. Diagnostic only: on a two-step toy batch, compare the implemented teacher
   update with a slow direct score-function calculation using the same sampled
   actions. Do not substitute a soft-label higher-order derivative.

**Tier 1 measurements, seed 94,000 (2026-09-04).** Both arms completed all
3,000 steps with finite losses, gradients and role checkpoints. MPL versus
`h=0` student NLL was `0.2816` versus `0.2920` (ratio `0.9642`); teacher NLL
was `0.3388` versus `0.3447` (ratio `0.9828`), against observed-frequency NLL
`0.7046`. Student class-mass concentration was `0.5486` versus `0.5292`.
In the MPL arm, `h` had mean `0.0028`, population standard deviation `0.5145`,
positive fraction `0.5107`, and non-zero fraction `1.0`; the final baseline was
`0.0838`. Final hard-label accuracy/entropy were `0.8125`/`0.6895`, UDA gate
coverage was `0.8839`, and TSA retained fraction was `1.0`. The paired weak and
strong view label-flip rates were `2.15%` and `4.30%`, respectively. These are
single-seed wiring measurements, not evidence for the Tier 2 claim.

**Tier 2.** Run the three paired arms of §6.5 over all 100 replicates of the
minority trap. Only the weight-1.0 arm's paired minority-accuracy gain, the
`h=0` trap guardrail, the teacher NLL guardrail, finiteness and the class-mass
concentration ceiling determine `reproduced` versus `deviating`. The dose arm,
the student NLL ratio and the feedback diagnostics are reported, not chosen
after the run.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-04 | `06f2cb8d0ff5` | student_treatment_NLL_ratio<br>outer_teacher_treatment_NLL_ratio<br>all_losses_gradients_and_checkpoints_finite<br>terminal_student_class_mass_concentration | 0.983341 +/- 0.00652<br>0.983566 +/- 0.0139<br>1 +/- 0<br>0.521234 +/- 0.00482 | yes |
| 2026-10-21 | `52f645a7a619` | student_treatment_NLL_ratio<br>outer_teacher_treatment_NLL_ratio<br>all_losses_gradients_and_checkpoints_finite<br>terminal_student_class_mass_concentration | 0.997683 +/- 0.00994<br>1.01487 +/- 0.00424<br>1 +/- 0<br>0.521526 +/- 0.00638 | no |
| 2026-10-21 | `c7921229b9fa` | student_treatment_NLL_ratio<br>outer_teacher_treatment_NLL_ratio<br>all_losses_gradients_and_checkpoints_finite<br>terminal_student_class_mass_concentration | 0.995541 +/- 0.00577<br>1.00832 +/- 0.00875<br>1 +/- 0<br>0.526084 +/- 0.00204 | no |

### 6.4 Amendment: the ten-seed verdict depends on the machine

Both rows in §6.3 are retained. Each was produced honestly under the reviewed
protocol, and their disagreement is the evidence this amendment reasons from.

**What the nightly showed.** At the single commit `52f645a7a619`, the nightly
suite passed this card on some runners and failed it on others. Every failing
run reported exactly `0.997683 +/- 0.00994`, the second ledger row, which was
recorded on a third machine (Intel Xeon, AVX-512) from the committed tree. The
passing runs took about 12 minutes instead of 8.5, which points to a second
runner CPU class. The status is a pure function of seeds and code on one
machine, but not across machines. Different BLAS and vector-kernel rounding
perturbs the last bits, and the MPL arm's endogenous trajectory amplifies them
over 3,000 steps. Seed `94000` shows the size of that amplification. The Tier 1
note in §6.2 records a student ratio of `0.9642`; the run behind the second
row gives `1.0195` for the same seed.

Student NLL ratios behind the second row, per replicate `r` (base seed
`94000+100r`):

| r | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| ratio | 1.0195 | 1.0157 | 0.9922 | 0.9418 | 0.9543 | 1.0491 | 0.9992 | 0.9928 | 0.9952 | 1.0170 |

Six replicates improve and four worsen. The per-replicate standard deviation is
`0.031` here and `0.021` behind the first row. Pooling the two rows puts the
effect near a ratio of `0.990`. That is about one standard error from the
threshold at ten seeds, so the reviewed tolerance (mean below `1` by at least
one stderr) has no stable answer at this sample size.

**What was audited before calling this a property of the method.** The
executor was re-read against §3.2 and eq. (12). It forms the pre-update
pseudo-label gradient before the one student step, and the labelled gradient at
`theta'_S` after it. It takes their cosine similarity, updates the baseline
before subtracting it (`training_utils.py:479-483`), and descends on
`h * CE(y_hat_u, T(x_u))` with `h` and the sample detached, which is eq. (12)'s
sign. The meta-score enters the teacher total at the `mean` reduction and the
§4 weight `1.0`, alongside the TSA and UDA terms. Tier 0 rule 6 already fixes
the sign on an analytic fixture. No departure from the card was found.

**A dose diagnostic.** If the feedback carried usable information, amplifying
it should move the student ratio consistently. The meta-score weight of the MPL
arm alone was scaled, with the `h=0` arm unchanged, over the same ten
replicates on the same machine. This is a diagnostic, not a recipe change; §4
keeps weight `1.0`.

| Meta-score weight | Student NLL ratio | Teacher NLL ratio |
|---|---|---|
| 1 (the card) | `0.9977 +/- 0.0099` | `1.0149 +/- 0.0042` |
| 5 | `0.9854 +/- 0.0121` | `1.0662 +/- 0.0240` |
| 20 | `1.0121 +/- 0.0212` | `1.0603 +/- 0.0190` |

The spread grows with the dose and the direction does not follow it. The
teacher worsens at every dose. Its average `h` stays near zero, with a
population standard deviation near `0.5` and a positive fraction near `0.51`.
On this fixture and budget, one centred scalar per 448-row batch behaves like
noise, so any effect is too small for ten seeds to resolve. No hyperparameter
was searched to recover a pass.

**Protocol change — approved 2026-10-21.** Raise `seeds` from `10` to
`100` and leave the metric, tolerance, fixture and every offset unchanged.
Replicates `r=10..99` extend §6.1's generating rule `s_r = 94000+100r` beyond
`uda.md`'s ten fixtures. The view-label-flip report continues per replicate.
`100` is the largest count that keeps §6.1's claim that the new offsets collide
with nothing. The hard-label seed `s_r+20000` equals the view seed
`s_{r+100}+10000`, so replicate `r+100` must not exist. At the larger observed
spread, the standard error falls to about `0.0031`, about three standard errors
from the threshold at the pooled `0.990`. Two runner classes should then agree
on the status, whichever way it goes. The cost is about 1.4–2 hours per
nightly against the job's 300-minute limit. The §6 YAML, §6.1's replicate
range and the benchmark's bound seed count changed together; the 100-seed
ledger row was measured from that committed protocol.

**The 100-seed result.** Commit `c7921229b9fa`, on the same machine as the
second row, gave `0.995541 +/- 0.00577` and stays `deviating`. Its first ten
replicates reproduce the second row exactly (`0.997683`), so the extension
changed no earlier draw. The standard error is nearly twice the `0.0031`
projected above. The per-replicate distribution has heavier tails than the
first ten suggested: standard deviation `0.058`, range `0.789` to `1.345`.
The median ratio is `0.9995` and 51 of 100 replicates fall below `1`. The
teacher ratio is `1.0083 +/- 0.0088`, with 44 below `1`. On this fixture the
evidence is that MPL's feedback has no consistent effect on either role.

The projection's purpose is not met. The margin to the tolerance (`0.0045`) is
still smaller than one standard error, so the reporting layer flags the run as
within noise. A runner class whose rounding moves the mean by about `0.0013`
would record `reproduced`. That claim was the projection's, not the
protocol's, and this row replaces it: whether the two nightly runner classes
now agree is an empirical question for the next nightly, not a settled one.

### 6.5 Amendment: a fixture where the teacher is confidently wrong

**Approved on 2026-10-22 and implemented.** The §6 YAML, §6.1, §5 row 11 and
the benchmark module now describe the minority trap. §6.3's first three rows
and §6.4 are the record of the two-cluster fixture. The text below is the
proposal as reviewed, with the draft §6 block removed now that §6 carries it.
Every number below comes from scratch screening on one machine (Intel Xeon,
AVX-512), at the 3,000-step budget and §4 hyperparameters unless a row says
otherwise. Seeds use the §6.1 stream, `s_r = 94000+100r`.

**Why the current fixture cannot answer the question.** The paper's account of
what the feedback buys is a teacher that stops reinforcing its own early
mistakes (appendix D.3, figure 4; the TwoMoon example in §3.1 and figure 2).
The paper's own isolated evidence for it is small: a plain pseudo-label teacher
gains `+0.32` points from MPL (appendix D.4, table 10), and the UDA → UDA+MPL
gain in figure 3 is a single ImageNet-10% run. On the current fixture the UDA
teacher learns the cluster structure within a few hundred steps, so there is
no such mistake to correct. An oracle probe made this measurable. Freeze a
teacher/student pair, draw 200 hard-label samples on one fixed 448-row batch,
and compute `h_raw` and the realised one-step labelled-loss drop for each draw.

| Snapshot (current fixture, seeds 0–3) | corr(`h_raw`, realised drop) | corr(`h_raw`, draw quality under the Bayes posterior) |
|---|---|---|
| step 1 | `+0.70` … `+0.96` | `+0.52` … `+0.77` |
| step 3,000 | `+0.91` … `+0.96` | `-0.06` … `+0.24` |

The first column confirms the estimator: `h_raw` tracks the realised
improvement at every snapshot, so eq. (12) is implemented as derived. The
second shows the feedback rewarding better pseudo-labels early and almost not
at all by the end of training. Sweeping the shipped `cluster_population` family
(K up to 5, class groups, noise 0.3–0.6, the paper's 64:128 batch geometry)
gave student ratios between `0.987` and `1.026` with no consistent sign.

**The proposed fixture: a minority trap.** Draw
`cluster_population(classes=3, groups=(2, 1), within=0.6, noise=0.4,
prior=(0.475, 0.05, 0.475))` and map clusters to treatments by `(0, 1, 1)`:
`K=2`, with label noise `low=0.02` at the cluster level, as now. Cluster 1 is a
5% minority. It sits `1.08` from the large cluster 0, whose label is the other
class, and shares its label with the distant cluster 2. Its labels are
recoverable: the Bayes classifier is right on `0.83` of the minority region
(rows whose Bayes cluster posterior picks cluster 1, about 4% of rows), with
Bayes NLL `0.146` overall. Under the inherited MCAR budget, about 3 of the 64
labels fall in it. UDA's sharpened, confidence-gated consistency pulls the
minority into its neighbour's class, and the student copies the teacher. The
trap binds: across 16 seeds the `h=0` student is right on `0.13`–`0.54` of the
minority region (mean `0.29`). At step 3,000 the probe finds the feedback still
informative on all four seeds probed: corr(`h_raw`, draw quality) is
`+0.12`–`+0.21`, and corr(`h_raw`, minority labels correct) is `+0.10`–`+0.26`.

**What the screen found on it.** The paired metric is the student's accuracy
on the minority region, MPL minus `h=0`:

| Arm | Seeds | Minority-accuracy gain | Seeds up / down | Median student NLL ratio |
|---|---|---|---|---|
| §4 as written (meta-score weight `1.0`) | 8 | `+0.027` (median `+0.012`) | 6 / 2 | `0.994` |
| weight `1.0`, UDA weight ramped `0 → 1` over the run (reverting deviation 10) | 16 | `+0.028 +/- 0.043` (median `+0.011`) | 10 / 4 | `1.011` |
| weight `1.0`, all schedules stretched to 12,000 steps | 8 | `+0.02` | 3 / 2 | `0.993` |
| meta-score weight `5` | 8 | `+0.09` | 6 / 2 | `0.958` |
| meta-score weight `20` | 16 | `+0.073 +/- 0.039` (median `+0.033`) | 12 / 3 | `0.967` |

On the current fixture the same weight-20 dose only widened the spread (§6.4).
Here it corrects the trap, and the teacher's minority accuracy rises with it
(`+0.076 +/- 0.041`, 12 of 16 seeds). At the paper's weight, neither a 4×
budget nor the reference's UDA ramp makes the correction reliable: the
consistency term outweighs the feedback throughout training. The student NLL
ratio is the wrong headline for this claim. The trap is about 4% of rows, and
one replicate escaped it completely (`0.19 → 0.79`) while its overall NLL got
worse.

**The proposed study: three paired arms on the minority trap.** Every arm
shares parameters at initialisation, batches, views and hard-label streams, as
§6 already requires of two arms.

1. `h=0`, the reviewed control, unchanged.
2. MPL as §4 declares it, meta-score weight `1.0`. **This arm alone determines
   the status.**
3. A declared dose arm, MPL with meta-score weight `20`. Its metrics are
   reported as information and never determine the status. The recipe keeps
   weight `1.0`: the dose arm is built in the benchmark module, the same way
   the `h=0` arm is, so `plan.hyperparameters` is unchanged.

The approved block is §6's, applied verbatim except that `dataset` now points
to §6.1.

**Why `0.10`, and why it was chosen after the screen.** The claim being tested
is that the feedback repairs a teacher that is confidently wrong. Closing a
fifth of the gap between the `h=0` student (about `0.29`) and Bayes (`0.83`) is
about `0.11`; `0.10` rounds it down. The screen measured a per-replicate
standard deviation of `0.065`–`0.172` for this gain. At 100 seeds the larger
value gives a standard error near `0.017`. That puts the expected weight-1
result (about `+0.03`) four standard errors below the bar, and the expected
dose-arm result (about `+0.07`) about two below. The predicted status is
therefore `deviating`, decided by the paper's weight with a margin the two
runner classes should agree on, while the dose arm records that the mechanism
works when it outweighs UDA. This threshold was set after seeing the scratch
data, and review should weigh it as such. The alternative is a
no-practical-effect claim at `0` (a mean gain at or below `0`), which would sit
about 1.6 standard errors from the expected weight-1 result and could flip by
machine as the current target did.

**What implementing it added.**

- A `labels` argument to `cluster_population`
  (`xty2/evaluation/benchmarks/common.py`) mapping clusters to treatment
  levels, with Tier 0 tests. The default `None` is the identity, so every
  existing caller keeps its exact draws.
- A shared helper returning the Bayes cluster posterior for this family, so
  the minority region is defined once.
- A third arm and the new metrics in `xty2/evaluation/benchmarks/meta_pseudo_labels.py`.
- §5 row 11, with §6.1 rewritten to describe both fixtures.
- Tier 1 keeps seed `94000` on the current fixture; it checks wiring, not this
  claim.

The cost is about 100 minutes per nightly on the recording machine (three
arms, 100 replicates, four workers), against the job's 300-minute limit. The
seed stream and its ceiling of 100 are unchanged. The new target's first ledger
row is measured from the committed protocol and recorded in the commit that
follows it.

## 7. Unknowns

| Unspecified or conflicting source detail | Our choice | Basis |
|---|---|---|
| Paper v4 samples hard pseudo-labels, while the pinned public code uses a detached soft distribution for both the student target and the teacher MPL cross-entropy. | Follow v4's hard categorical sample and appendix A score-function derivation. | §2 after eq. (3), appendix A eqs. (10)–(12), and the authors' discussion in [`google-research/google-research#534`](https://github.com/google-research/google-research/issues/534). |
| Appendix C.3 writes "we compute `h` using the gradients' **cosine distance**", but eq. (12) is a similarity, and reading "distance" as `1 - cos` would invert the feedback. | Read it as cosine **similarity**: `h_raw = cos(grad_l, grad_u)`, positive when the student's step helped. | Eq. (12)'s derivation, which the prose is stabilising rather than replacing. Tier 0 rule 6 asserts the resulting sign so the reading cannot silently flip. |
| The pinned code's feedback coefficient is not the paper's `h`. `training_utils.py:472` forms `dot_product = CE_new - CE_old`, which equals `-h` to first order, and adds `mpl * dot_product` to the teacher loss at `:496`. | Follow eq. (12)'s sign, not the code's. The pinned term is inert anyway: its target is `stop_gradient(softmax(logits['u_aug']))` against those same logits (`:484-485`), so its gradient is `softmax(z) - stop_gradient(softmax(z)) = 0`. | Appendix A eqs. (10)-(12) and [`google-research/google-research#534`](https://github.com/google-research/google-research/issues/534). Recorded because the pinned code is otherwise this card's tie-breaker and would supply the wrong sign. |
| Moving-baseline initial value and reset. | Scalar `b_0=0`, decay `0.99`, reset at every execution. | The pinned code supplies the `0.01` coefficient and, via `tf.control_dependencies([moving_dot_product_update])` at `:481`, the update-then-subtract order, which deviation 7 now follows. Its `moving_dot_product` variable carries no initialiser, so `b_0` and the reset are genuinely ours. |
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
| Card reviewed (status → `reviewed`) | PR #40 review plus implementation audit | 2026-09-04 |
| Plan diffed against §3.2 and §4 | Codex | 2026-09-04 |
