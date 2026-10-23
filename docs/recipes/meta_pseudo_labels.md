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
  the final student, which is the student after the paper's labelled
  fine-tuning (§3.1). The mechanism estimand is the paired change in the final
  student's held-out accuracy on §6.1's minority region, the rows where the
  teacher is confidently wrong. It is caused by allowing a labelled-batch
  student feedback signal to update the pseudo-label teacher, holding teacher
  UDA, data, batches, views, architecture, optimiser steps and random seeds
  fixed. The paired change in overall held-out treatment NLL is reported
  alongside it.
- **Claim:** Meta Pseudo Labels (MPL) trains an independent teacher to choose
  pseudo-labels that improve a student's loss on labelled examples after one
  student update. The paper presents this as a one-step approximation to a
  bilevel objective and reports that MPL improves an already strong UDA teacher
  on low-label image benchmarks, which the paper attributes to the teacher
  correcting its own confirmation bias (appendix D.3). In the selected
  project-local experiment the falsifiable claim is narrower: at the §4
  meta-score weight, centred cosine feedback raises the final student's
  accuracy on a minority cluster the UDA teacher absorbs into its neighbour's
  class by at least `0.10` over the otherwise identical zero-feedback arm
  (§6.5). The card's earlier claim, an improvement in overall treatment NLL on
  `uda.md`'s two-cluster fixture, was tested and not supported (§6.4).
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

After joint training the paper fine-tunes the student. §2, final paragraph:
"we can take a student model that has converged after training with Meta
Pseudo Labels and finetune it on labeled data to improve its accuracy". §3.2,
*Implementation details*, fixes the procedure for the low-label benchmarks:
"SGD with a fixed learning rate of 10⁻⁵ and a batch size of 512, running for
2,000 steps for ImageNet-10% and 1,000 steps for CIFAR-10 and SVHN", and "we
return the model at the final checkpoint". Each fine-tuning step is

$$
\theta_S \leftarrow \theta_S - 10^{-5}\,
\nabla_{\theta_S}\mathrm{CE}\left(y_l, S(x_l;\theta_S)\right),
$$

starting from the student's parameters at the end of joint training. The
teacher takes no further part.

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

Fine-tuning is an ordinary second stage. `finetune` is a `gradient` stage that
starts from the `inner_student` checkpoint of `meta_train`
(`initialise_from="meta_train"`, `initialise_role="inner_student"`) and fits
the labelled NLL on `weak_x`. The teacher's checkpoint is not read. A
`meta_gradient` stage writes one checkpoint per role, so the role is named
rather than inferred (§5.1).

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
| `CE(y_l, S(x_l))` after joint training | the student's labelled fine-tuning loss | `T_GIVEN_X @ weak_x`, default parameters overlaid from `inner_student` | stage `finetune`: `ObservedTreatmentNLL` named `student_finetune_nll`, rows `t_observed` |
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
    finetune.student_finetune_nll: none                                    # ref impl Supervised step, training_utils.py:140-147
  detached_targets:
    student_pseudo_label_nll: hard categorical sample and outer_teacher graph
    teacher_meta_score: hard categorical sample and centred feedback coefficient  # appendix A eq. (12); ref impl training_utils.py:483
    teacher_uda_consistency: target                                     # inherited from uda.md
  gradient_clipping:
    inner_student: none          # ref impl clips by params.grad_bound, whose flag_utils.py:100 default 1e9 is inert; uda.md declares none
    outer_teacher: none          # same call site, training_utils.py:500; flag_utils.py:132 defines teacher_grad_bound=20 but this commit does not read it
    finetune: none               # Supervised step clips at grad_bound, training_utils.py:152, inert at the flag_utils.py:100 default 1e9
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
    finetune.student_finetune_nll: mean                # ref impl training_utils.py:140-145, a SUM divided by train_batch_size
  eligible_rows:
    meta_train.student_pseudo_label_nll: t_missing     # algorithm 1 line 6
    meta_train.student_labelled_feedback_nll: t_observed  # algorithm 1 line 7
    meta_train.teacher_meta_score: t_missing           # algorithm 1 line 8
    meta_train.teacher_tsa_nll: t_observed             # algorithm 1 line 9
    meta_train.teacher_uda_consistency: t_missing      # algorithm 1 line 10; deviation 8
    finetune.student_finetune_nll: t_observed          # paper §3.2 "finetune the student on the labeled dataset"
  weights:
    meta_train.student_pseudo_label_nll: 1.0           # algorithm 1 line 6
    meta_train.student_labelled_feedback_nll: 0.0      # gradient probe only; it is not applied to the student optimiser
    meta_train.teacher_meta_score: 1.0                 # ref impl training_utils.py:496 adds it unweighted
    meta_train.teacher_tsa_nll: 1.0                    # ref impl training_utils.py:495
    meta_train.teacher_uda_consistency: 1.0            # uda_weight default 1.0, flag_utils.py:146; deviation 10 drops its ramp
    finetune.student_finetune_nll: 1.0                 # the Supervised step's only loss term, training_utils.py:147
  schedules:
    meta_train.student_pseudo_label_nll: constant 1.0
    meta_train.student_labelled_feedback_nll: constant 0.0
    meta_train.teacher_meta_score: constant 1.0        # the feedback magnitude is h, not a schedule
    meta_train.teacher_tsa_nll: constant 1.0           # the TSA ceiling is gate arithmetic under losses.confidence_threshold
    meta_train.teacher_uda_consistency: constant 1.0   # ref impl ramps over uda_steps; deviation 10
    finetune.student_finetune_nll: constant 1.0
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
    finetune: sgd(momentum=0.0, nesterov=False)        # paper §3.2 "SGD"; common_utils.py:452-455 for optim_type='sgd'; §7
  lr:
    inner_student: 0.03                                # inherited from uda.md; deviation 3 (ref impl mpl_student_lr=0.1, flag_utils.py:157)
    outer_teacher: 0.03                                # inherited from uda.md; deviation 3 (ref impl mpl_teacher_lr=0.1, flag_utils.py:161)
    finetune: 1e-05                                    # paper §3.2 "a fixed learning rate of 10^-5"; not rescaled by batch/256, §7
  lr_schedule:
    inner_student: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))   # inherited from uda.md; no warmup or wait steps, deviation 3
    outer_teacher: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))   # inherited from uda.md; deviation 3
    finetune: constant 1.0                                              # paper §3.2 "fixed"
  weight_decay:
    inner_student: 0.0005 (all trainable components; all parameters)     # inherited from uda.md; deviation 3
    outer_teacher: 0.0005 (all trainable components; all parameters)     # inherited from uda.md; deviation 3
    finetune: 0.0005 (all trainable components; all parameters)          # Supervised step's L2 term, training_utils.py:147; §7
  batch_size:
    meta_train: 512                                    # inherited from uda.md; deviation 3
    finetune: 64                                       # paper §3.2 says 512; the whole labelled pool, deviation 12
  labelled_unlabelled_ratio:
    meta_train: 7.0                                    # 64:448, inherited from uda.md; deviation 3
    finetune: None                                     # one labelled quota; the sampler draws no labelled/unlabelled pair
  total_steps_or_epochs:
    meta_train: 3000                                   # atomic student-then-teacher steps; inherited from uda.md
    finetune: 1000                                     # paper §3.2, CIFAR-10 and SVHN

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
  treatment_encoding:
    student_pseudo_label_nll: integer classes 0..K-1; one categorical draw per t_missing row   # XTYBatch contract; eq. (10)
  split_protocol: one fixed project-local DGP, split train/test by the section 6.1 fixture; no image protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id, drawn once per program   # §6.7: fine-tuning reads joint training's labelled set
```

`losses.weights.meta_train.student_labelled_feedback_nll=0.0` means the value is
a gradient probe, not an ordinary mixer contribution. The executor
must print that distinction and reject any declaration that both probes and
applies this loss, because the paper's student does not learn directly from
labelled rows during MPL training.

The `finetune` stage trains the same two components the `inner_student` role
declares, so `architecture.*` keys need no `finetune` scope: they are printed
once per role and the stage overlays that role's parameters.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply MPL to categorical treatment prediction on a fixed tabular DGP and omit the outcome stack. | The experiment is the backlog's optimisation-boundary test. Adding outcome likelihoods would introduce a second interaction before the meta-gradient executor is validated. | No published image number applies. The result speaks only to held-out treatment NLL. |
| 2 | `judgement` | — | Replace image RandAugment with UDA's shipped 10% weak mask and composed 10% + 10% strong mask. | Image transforms have no tabular semantics. Reusing the reproduced UDA views keeps the nearest baseline fixed and retains its predeclared label-flip check. | Narrows the invariance family; both paired arms share it exactly. |
| 3 | `judgement` | — | Reuse `uda.md`'s teacher policy and hyperparameters instead of paper table 8's CIFAR values: 3,000 steps, lr `0.03`, temperature `0.4`, confidence `0.8`, exponential TSA, no label smoothing, and a 64:448 quota. | The card tests one new mechanic against the shipped baseline. Simultaneously changing UDA would make a gain unattributable. | Reduces training horizon and changes feedback scale; the paired comparison remains interpretable but is not a paper reproduction. |
| 4 | `judgement` | — | Use independent tabular MLP propensity graphs rather than two WideResNet-28-2 classifiers. | Architecture is an orthogonal component and the project does not ingest images. | Changes optimisation geometry, so even the sign of MPL's effect is empirical. |
| 5 | `withdrawn` | — | ~~Stop after joint MPL training; do not fine-tune the student on labelled rows.~~ **Withdrawn.** The program now ends with the paper's §3.2 fine-tuning stage (§3.1, §3.2, §6.6). | The paper makes fine-tuning part of the method (§2, final paragraph), and the card's estimand is the final student. Omitting a paper mechanic on the intuition that it is inert was the departure; §6.6's screen measured it instead. | §6's status metric now reads the fine-tuned students. At the paper's rate, the 8-seed screen moved the paired gain from `+0.027` to `+0.026`. |
| 6 | `judgement` | — | Use one hard categorical sample per missing row and paper-v4's score-function estimator, not the pinned code's detached soft distribution used as both target and teacher logits. | v4 §2 and appendix A explicitly select hard samples. The authors' public issue discussion identifies the old soft path as inconsistent with the derived feedback formula. | Adds sampling variance but preserves the teacher feedback gradient described by eqs. (10)–(12). |
| 7 | `judgement` | — | Follow the pinned update-then-subtract order — `b_{t+1}=0.99b_t+0.01h_raw`, then `h=h_raw-b_{t+1}` — and additionally fix `b_0=0` with a reset at every execution. | Appendix C.3 specifies a moving baseline but neither its order nor its initialisation; `training_utils.py:479-483` fixes the order under `tf.control_dependencies`, so only `b_0` and the reset are ours to choose. The reference variable carries no initialiser, and a paired comparison needs a deterministic start. | Feedback magnitudes over the first few steps depend on `b_0`; the arms share initial state and exogenous streams, while later endogenous trajectories may diverge after feedback. |
| 8 | `judgement` | — | Treat appendix B's UDA argument as `x_u`, matching its prose and the reference implementation, rather than the displayed `x_l` in algorithm 1 line 10. | UDA is introduced there as the teacher's unlabelled-data objective; `x_l` would duplicate a labelled augmentation term and contradict the surrounding text. | Preserves the shipped UDA comparison. |
| 9 | `judgement` | — | Report the raw student and teacher parameters and construct no evaluation EMA, dropping `uda.md`'s `TeacherSpec(decay=0.9999, role="evaluation")`. | MPL's own `ema_decay` default is `0`, that is no moving average (`flag_utils.py:113`), so the inherited EMA is a UDA-side choice rather than a paper mechanic. Keeping it would add a second reported parameter set to a card whose question is one gradient path. | `uda.md`'s headline Tier 2 number is its EMA NLL, so §6 here is not directly comparable with that column; the paired arms are unaffected. |
| 10 | `judgement` | — | Hold the teacher's UDA consistency weight at a constant `1.0` rather than ramping it linearly over `uda_steps` as `training_utils.py:489-491` does. | `uda.md` is the nearest shipped baseline and already declares a constant weight; introducing MPL and a weight ramp in one card would make a difference unattributable. | Stronger early consistency pressure on the teacher than the reference schedule; both arms share it. |
| 11 | `judgement` | — | Replace `uda.md`'s two-cluster Tier 2 fixture with §6.1's minority trap: three clusters, a 5% minority beside a large cluster of the other treatment. Tier 1 keeps the two-cluster fixture. | The inherited fixture gives the teacher no confidently wrong region for feedback to correct: by step 3,000 the feedback no longer rewards better pseudo-labels (§6.4, §6.5). | Moves the Tier 2 target from overall student NLL to paired minority-region accuracy, with a declared weight-20 dose arm reported as information. §6 is no longer comparable with `uda.md`'s. |
| 12 | `framework-limitation` | `batch-row-repetition` | Fine-tune on all 64 labelled rows each step, rather than the paper's batch of 512. | §3.2's batch of 512 is larger than the 64-row labelled pool, so a faithful batch repeats `row_id` values, which `XTYBatch` forbids (`DESIGN.md` §7.1). The reference draws its labelled batch from a repeated, shuffled stream (`data_utils.py:504-516`). | With a mean reduction, a full-pool step is the exact expectation of the gradient of a 512-row draw from the same 64 rows. Only the reference's sampling noise is lost, at a learning rate of `1e-5`. |

No paper mechanic is intentionally omitted because of the current framework.
The executor is fidelity-bearing and was added under `DESIGN.md` §11.2 rather
than hiding the mechanic behind a `framework-limitation`. One paper value is
changed because of the framework: the fine-tuning batch (row 12).

### 5.1 Framework additions made for this card

These additions are implemented by the compiler and the bounded
`meta_gradient` executor. The second consumers below constrained their shape;
they remain design checks rather than speculative implementations.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `meta_gradient` executor: two independently trainable role-tagged graphs, separate optimisers, an atomic one-step inner update, pre/post gradient probes, ordered outer update, stage-local scalar state, and role-tagged checkpoint output | fidelity-bearing, load-bearing vocabulary | this card | The validation-driven-weighting family named in `BACKLOG.md` §10.1, concretely Ren et al., *Learning to Reweight Examples for Robust Deep Learning* (2018): its clean validation loss is evaluated after a one-step weighted inner update and differentiated to the outer example weights. The checked common shape is explicit inner/outer parameter ownership, bounded unroll length, pre/post state, and an outer loss; the executor must not assume the outer parameters are a second neural network. One shape difference is not covered and review should settle it: Ren et al. differentiates with respect to a per-example weight vector and therefore needs per-example inner gradients, whereas step 2 here retains a single aggregate gradient. The contract below fixes the update order, not the arity of the retained probe. | Eq. (3) and appendix B require this ordering inside every iteration. Ordinary stages expose only completed checkpoints, and `TeacherSpec` is frozen EMA state. Hiding the loop in a recipe function would remove update order, gradient reach and optimiser ownership from the plan. |
| `Realisation.role`: a named parameter set beyond the closed `params` literal, so one graph declaration can be evaluated under `outer_teacher` and `inner_student` weights in one step | fidelity-bearing, load-bearing vocabulary | this card | `BACKLOG.md` §11.2 and §15.6, concretely Lakshminarayanan et al., *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* (2017): "we train an ensemble of `M` networks independently, with random initialisation", which is `M` named parameter sets over one declaration with per-set initialisation seeds and no EMA relationship. The checked common shape is a named, independently initialised, independently optimised parameter set that the plan prints and the compiler sorts; what MPL adds beyond an ensemble is per-role optimisers, and an ensemble adds aggregation this card does not need. | `Realisation.params` is `Literal["student","teacher"]` and its own guard says a third parameter set "is a new realisation axis: it needs a reviewed card whose §4 checklist cannot be honoured without it". This is that card: §4 declares two optimisers, two learning rates and two initialisations, and `teacher` already means a non-trainable EMA copy. |
| `Realisation.state`: a within-step position marker, here `post_update`, so the labelled gradient probe names the parameters it is evaluated at | fidelity-bearing, load-bearing vocabulary | this card | The same Ren et al. (2018) row below: its clean validation loss is evaluated strictly after the weighted inner step, so it needs the same pre/post distinction on one parameter set. Review may instead prefer to keep `state` out of the realisation key and carry the position in `meta_gradient`'s declared six-step order; the card asks for the axis because Tier 0 rule 13 requires the plan and digest to print which state each probe reads, and an executor-private position is not plan-visible. | Eq. (12) reads one parameter set at two states inside one iteration. Without a marker the two student gradients are indistinguishable in the plan, which is precisely the confusion rule 4 exists to catch. |
| `Stage.initialise_role`, and `run_program` executing `meta_gradient` stages with explicit `MetaGradientSeeds` | fidelity-bearing, load-bearing vocabulary | this card's `finetune` stage | Ren et al. (2018), as in the first row: the classifier trained in the inner role of its bilevel loop is the model that is evaluated or trained further. The checked common shape is that the field selects one role's checkpoint and assumes nothing about the other role. The field is required when the source is a `meta_gradient` stage and rejected otherwise, so no consumer picks a role by default. | Paper §2 and §3.2 fine-tune the student after joint training. `initialise_from` read a stage's single checkpoint, a `meta_gradient` stage writes one per role (`DESIGN.md` §7.1), and `run_program` refused `meta_gradient` stages. Running the fine-tuning outside the program would remove it from the plan. |
| `MissingnessSpec.drawn`: `"per_stage"` (default) or `"per_program"`, which keys every stage's induced mask by the program seed | fidelity-bearing, load-bearing vocabulary | this card's data policy | The staged fine-tuning programs in `BACKLOG.md` §2.12, concretely SimCLRv2 and Noisy Student, which fine-tune on the labelled set an earlier stage was given. The checked common shape is a data-policy statement, printed under a card key, that every stage of a program reads one mask; it says nothing about which stages consume labels. | Paper §3.2 fine-tunes "on the labeled dataset" joint training was given. `run_program` keyed each stage's mask by its own seed, so the two stages shared 2–6 of 64 labelled rows (§6.7). An opt-in field fixes this card without changing PAWS, SCARF or VIME's draws. |
| `SampledTeacherTreatmentNLL`, `MetaPseudoLabelScore`, and `MetaFeedbackCoefficient` | fidelity-bearing, reversible | this card | not required (ordinary objective/executor helpers; no shared vocabulary) | Existing pseudo-label objectives use an argmax from the same graph. MPL requires a seeded categorical sample from one role, reused by the other role and by a score-function term, plus a gradient-alignment scalar. Keeping these small objects explicit makes the sampled target, detach points and coefficient visible to Tier 0. |

The executor contract must remain narrower than a general differentiable
programming system: exactly one inner step, exactly one outer step, no arbitrary
stage DAG, no implicit higher-order unroll, and no user callback that mutates
parameters. Widening any of those is a new card/design decision.

### Tier 2 outcome

On 2026-10-22, commit `6494ee9b4227` produced a `deviating` result: This is the predeclared project-local Meta Pseudo Labels mechanism target of card section 6.5: whether appendix A's centred hard-label score-function feedback, at the section 4 meta-score weight, repairs a teacher that UDA's consistency leads to absorb a 5% minority cluster into its neighbour's class, measured as the student's paired accuracy gain on that region over the otherwise identical h=0 execution. A declared weight-20 dose arm is reported as information and does not determine the status. It is not a reproduction of a published image-accuracy number; the tabular DGP, views, architecture, and 3,000-step budget are the reviewed deviations listed in the card. Failed target(s): student_minority_accuracy_gain was 0.00551841 +/- 0.0065 against mean >= 0.1, by at least one stderr.

## 6. Reproduction target

```yaml
reproduction:
  dataset: project-local minority-trap XTY DGP (6 features, K=2): cluster_population(classes=3, groups=(2,1), within=0.6, noise=0.4, prior=(0.475,0.05,0.475)) relabelled by (0,1,1), specified in §6.1
  variant: paired MPL (meta-score weight 1.0) versus the forced-zero-feedback arm, each followed by the paper's labelled fine-tuning of the student, plus a declared weight-20 dose arm and a declared fine-tuning learning rate of 1e-3, both reported as information
  metric: paired gain in fine-tuned student accuracy on the held-out minority region (Bayes cluster posterior picks the minority cluster), MPL weight 1.0 minus h=0; fine-tuned h=0 minority accuracy as the fixture-validity guardrail; finiteness over both stages, outer-teacher NLL ratio and fine-tuned student class-mass concentration; the gains before fine-tuning, the dose arm, the 1e-3 fine-tuning stage, the NLL ratios and the h diagnostics as information
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: fine-tuned minority-accuracy gain >= 0.10 in mean by at least one stderr; fine-tuned h=0 minority accuracy <= 0.60 (the trap binds); outer-teacher NLL <= 1.10x the zero-feedback arm; finite losses and gradients in both stages of every replicate; fine-tuned student class-mass concentration < 0.95
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
Each arm then fine-tunes its own student (§3.2). The fine-tuning stage sees the
same rows and views in every arm, because its batch is the whole labelled pool
and its view stream comes from a stage seed the arms share.

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

**One labelled set.** The induced mask is drawn once per program from the
program seed `s_r+10000` (§6.7), so `meta_train`, `finetune` and the
information stage all see the same 64 labelled rows, the set the one-stage
protocol used.

**Fine-tuning streams.** `run_program` runs `meta_train` at stage seed
`s_r+10000`, the stream the one-stage protocol used, with the role and
hard-label seeds above. `finetune` runs at `s_r+10000+1,000,000`
(`STREAM_STRIDE`), and the declared `1e-3` fine-tuning stage of §6.6 at
`s_r+10000+2,000,000`. Each spends one view key per step, so their keys lie
above `1,000,000` and below `2,124,000`, clear of every other stream in this
card.

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
14. `finetune` starts from the `inner_student` checkpoint: at a zero learning
    rate its output equals that checkpoint and differs from `outer_teacher`'s,
    and a non-zero rate moves it. `run_program` reproduces `run_meta_gradient`'s
    role checkpoints at the same seeds.
15. A stage that initialises from a `meta_gradient` stage must name one of its
    roles; a role is rejected without `initialise_from`, against a non-meta
    source, or on a stage without objectives. `run_program` requires
    `MetaGradientSeeds` for exactly its meta-gradient stages. The plan prints
    the role beside `initialise from`.
16. Under `drawn="per_program"` both stages see the same 64 labelled rows.
    Under `"per_stage"`, `meta_train` keeps the same rows and `finetune` sees
    a different set, so the shared-set assertion is not vacuous. The plan
    prints the scope, and an `observed` mechanism rejects one (§6.7).

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

**Tier 2.** Run the three paired arms of §6.5, each followed by the paper's
fine-tuning (§6.6), over all 100 replicates of the minority trap. Only the
weight-1.0 arm's paired minority-accuracy gain after fine-tuning, the `h=0`
trap guardrail, the teacher NLL guardrail, finiteness and the class-mass
concentration ceiling determine `reproduced` versus `deviating`. The gains
before fine-tuning, the dose arm, the `1e-3` fine-tuning stage, the student NLL
ratios and the feedback diagnostics are reported, not chosen after the run.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-04 | `06f2cb8d0ff5` | student_treatment_NLL_ratio<br>outer_teacher_treatment_NLL_ratio<br>all_losses_gradients_and_checkpoints_finite<br>terminal_student_class_mass_concentration | 0.983341 +/- 0.00652<br>0.983566 +/- 0.0139<br>1 +/- 0<br>0.521234 +/- 0.00482 | yes |
| 2026-10-21 | `52f645a7a619` | student_treatment_NLL_ratio<br>outer_teacher_treatment_NLL_ratio<br>all_losses_gradients_and_checkpoints_finite<br>terminal_student_class_mass_concentration | 0.997683 +/- 0.00994<br>1.01487 +/- 0.00424<br>1 +/- 0<br>0.521526 +/- 0.00638 | no |
| 2026-10-21 | `c7921229b9fa` | student_treatment_NLL_ratio<br>outer_teacher_treatment_NLL_ratio<br>all_losses_gradients_and_checkpoints_finite<br>terminal_student_class_mass_concentration | 0.995541 +/- 0.00577<br>1.00832 +/- 0.00875<br>1 +/- 0<br>0.526084 +/- 0.00204 | no |
| 2026-10-22 | `6494ee9b4227` | student_minority_accuracy_gain<br>zero_feedback_student_minority_accuracy<br>outer_teacher_treatment_NLL_ratio<br>all_losses_gradients_and_checkpoints_finite<br>terminal_student_class_mass_concentration | 0.00551841 +/- 0.0065<br>0.339695 +/- 0.0163<br>0.990021 +/- 0.0107<br>1 +/- 0<br>0.529258 +/- 0.00203 | no |

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

The proposal estimated about 100 minutes per nightly. The first run took 2 h
32 min on the recording machine (three arms, 100 replicates, four workers),
against the job's 300-minute limit. The
seed stream and its ceiling of 100 are unchanged. The new target's first ledger
row is measured from the committed protocol and recorded in the commit that
follows it.

**The first 100-seed result (`6494ee9b4227`, 2026-10-22): `deviating`, by a
wide margin.** The status-determining gain is `0.0055 +/- 0.0065`, about 15
standard errors below the `0.10` bar, so the verdict should not depend on the
runner class. The trap binds: the `h=0` student is right on `0.340 +/- 0.016`
of the minority region, against Bayes `0.835 +/- 0.004`, and that region is
`4.2%` of held-out rows. The teacher guardrail holds (`0.990 +/- 0.011`).
Every arm finished finite. View label-flip rates against the Bayes rule are
`2.1%` weak and `4.4%` strong.

| Arm | Student minority gain | Seeds up / down | Teacher minority gain | Student NLL ratio (below 1) |
|---|---|---|---|---|
| MPL, §4 weight `1.0` (status) | `+0.0055 +/- 0.0065`, median `0` | 44 / 30 | `+0.030 +/- 0.0096` | `0.991 +/- 0.007` (56/100) |
| Dose arm, weight `20` (information) | `+0.069 +/- 0.014`, median `+0.031` | 67 / 22 | `+0.044 +/- 0.015` | `0.939 +/- 0.013` (73/100) |

At the paper's weight, the teacher moves toward the minority's label by about
three standard errors, but the student does not follow: one centred scalar per
448-row batch, competing with a full-weight UDA term, shifts the teacher's
minority probabilities too little to change the student's hard-label fit.
When the feedback outweighs UDA (the dose arm), the student escapes the trap
on two-thirds of replicates and its overall NLL improves by about 6%. Both
conclusions need the 100 seeds. The 16-seed screen overstated the weight-1
gain (`+0.028`) and understated how consistently the dose arm improves NLL.
The card's §2 question has a mechanism answer: the feedback carries the
information MPL needs and corrects the error it was designed for, but at the
paper's weight, on this fixture and budget, too weakly to change the student.

### 6.6 Amendment: the paper's labelled fine-tuning of the student

**Approved on 2026-10-23 and implemented.** Review settled the three questions
at the end of this section. Fine-tuning uses plain SGD. The paper's stated rate
of `1e-5` determines the status. Because that rate is not expected to move the
student, a declared fine-tuning rate of `1e-3` runs beside it and is reported
as information. The status metric moves to the fine-tuned student. §3, §4, §5
rows 5 and 12, §5.1, the §6 YAML, §6.1, §6.2 and §7 now describe the two-stage
program. The text below is the proposal as reviewed. The screening numbers come
from a scratch script on one machine (Intel Xeon, four workers), seeds
`s_r = 94000+100r` for `r = 0…7`, run against the committed `6494ee9b4227`
protocol.

**The `1e-3` information stage.** The benchmark adds a third stage to every
arm, `finetune_information`, which is `finetune` with the learning rate
`1e-3` and initialises from the same `inner_student` checkpoint. The recipe
keeps `1e-5`, so `plan.hyperparameters` is unchanged, as with the dose arm.
The value was chosen after the screen: it is the largest screened rate that did
not raise the `h=0` NLL (`0.2379` against `0.2365`), while `0.03` overfitted.
It answers whether a fine-tuning step that does move the student changes the
paired gain. It does not determine the status.

**What the sources specify.** The paper makes fine-tuning part of the method,
not an evaluation option. §2, final paragraph: "as the student in Meta Pseudo
Labels only learns from unlabeled data with pseudo labels generated by the
teacher, we can take a student model that has converged after training with
Meta Pseudo Labels and finetune it on labeled data to improve its accuracy".
§3.2, *Implementation details*, fixes the procedure for the low-label
benchmarks: "we finetune the student on the labeled dataset. For this
finetuning phase, we use SGD with a fixed learning rate of 10⁻⁵ and a batch
size of 512, running for 2,000 steps for ImageNet-10% and 1,000 steps for
CIFAR-10 and SVHN", with no held-out validation set: "we return the model at
the final checkpoint". Appendix D.1, figure 3, reports that "finetuning only
slightly improves over UDA+MPL". Appendix C.4's LARS optimiser applies only to
the ImageNet+JFT models of §4, which this card does not adapt.

The pinned code has no fine-tuning command. Its path is the `Supervised`
step (`training_utils.py:91-180`) started from an MPL checkpoint through
`--load_checkpoint` (`main.py:300-333`), which restores variables by name.
The student is built under the `model` scope (`training_utils.py:401`) and the
teacher under `teacher` (`:396`), so this restores the student and drops the
teacher. That step uses the mean cross-entropy over the labelled batch
(`:140-145`, a sum divided by `train_batch_size`), adds L2 weight decay
(`:147`), clips at the inert `grad_bound` default of `1e9` (`:152`;
`flag_utils.py:100`), and applies an EMA only when `ema_decay` is non-zero
(`:162`; `flag_utils.py:113` defaults to `0`). The labelled CIFAR-10-4K stream
repeats the 4,000 rows and then shuffles them (`data_utils.py:504-516`). Each
image gets a random flip and crop (`data_utils.py:472-481`), which `uda.md` §3.2
maps to `weak_x`.

**The proposed program.** Stage `meta_train` stays as reviewed. A second
`gradient` stage, `finetune`, follows it:

| Key | Proposed value | Source |
|---|---|---|
| initial parameters | the `inner_student` checkpoint of `meta_train`; `outer_teacher` is discarded | §2 final paragraph; `main.py:300-333` with the scopes at `training_utils.py:396,401` |
| objective | `ObservedTreatmentNLL` on `T_GIVEN_X @ weak_x`, rows `t_observed`, weight `1.0`, reduction `mean` | §3.2 "finetune the student on the labeled dataset"; `training_utils.py:140-147`; flip-and-crop maps to `weak_x` |
| optimiser | plain SGD: no momentum, no Nesterov | §3.2 says "SGD", where appendix C.4 names "Nesterov Momentum" for training; `common_utils.py:452-455` builds plain gradient descent for `optim_type='sgd'` |
| learning rate | `1e-5`, constant | §3.2 |
| weight decay | `0.0005`, all parameters | `training_utils.py:147`; this value is §4's inherited one (deviation 3) |
| gradient clipping | none | `training_utils.py:152` with the inert `1e9` bound |
| label smoothing, EMA | none | deviation 3; `flag_utils.py:113` |
| batch | all 64 labelled rows, every step | framework-limitation `batch-row-repetition`; see below |
| steps | `1,000` | §3.2, CIFAR-10 and SVHN |

**One new framework limitation.** The paper's batch of 512 is larger than the
64-row labelled pool, so a faithful batch repeats `row_id` values. That is the
open `DESIGN.md` §11.4 ledger item `batch-row-repetition`. The proposal uses
the whole labelled pool as each step's batch. With a mean reduction, that step
is the exact expectation of the gradient of a 512-row batch drawn from the
same 64 rows, so only the sampling noise of the reference is lost. If
approved, this becomes a new `framework-limitation` row in §5, and this card
joins that ledger row's payers.

**One new framework addition.** `run_program` refuses `meta_gradient` stages,
and `Stage.initialise_from` reads a stage's single checkpoint, whereas
`meta_gradient` writes one checkpoint per role (`DESIGN.md` §7.1). The
proposal needs both:

- a `Stage` field naming which role of a `meta_gradient` source checkpoint to
  overlay. It is required when the source is a `meta_gradient` stage, must name
  one of that stage's roles, and is rejected otherwise. The plan prints it
  beside `initialise from`;
- `run_program` executing a `meta_gradient` stage in sequence, with the role
  and hard-label seeds passed in explicitly as `run_meta_gradient` takes them
  now.

Under `DESIGN.md` §11.2 this is fidelity-bearing, because without it §2's
fine-tuning cannot run, and it is load-bearing vocabulary, because it is a
`Stage` field. The named second consumer is the same as §5.1's: Ren et al.
(2018), whose trained classifier is the inner role of a bilevel loop and is
the model that is evaluated or trained further. The check is that the field
selects one role's parameters and assumes nothing about the other role. If
approved, it becomes a §5.1 row.

**What the screen found.** Each arm's terminal inner student was copied and
fine-tuned on its 64 labelled rows, full batch and clean features, for 1,000
steps. The table gives the paired minority-accuracy gain, MPL weight `1.0`
minus `h=0`, over 8 seeds:

| Fine-tuning | Paired gain, mean (sd) | Seeds up / down | `h=0` minority accuracy | `h=0` student NLL | `h=0` parameter shift (L2, mean) |
|---|---|---|---|---|---|
| none (the current card) | `+0.027` (`0.061`) | 6 / 2 | `0.270` | `0.2365` | — |
| §3.2: SGD, lr `1e-5` | `+0.026` (`0.064`) | 5 / 2 | `0.275` | `0.2362` | `0.004` |
| §3.2 plus weight decay `5e-4` | identical to the row above to four decimals | | | | |
| plain SGD, lr `1e-3` (information only) | `-0.014` (`0.114`) | 3 / 4 | `0.365` | `0.2379` | `0.10` |
| Nesterov, lr `0.03`, the joint-training optimiser (information only) | `+0.108` (`0.340`) | 6 / 2 | `0.529` | `0.4013` | `7.2` |

At the paper's settings, fine-tuning barely moves the student: the parameters
move by at most `0.008` in L2 norm, no seed's minority accuracy changes by more
than `0.014`, and the NLL changes in the fourth decimal. This agrees with
appendix D.1's "only slightly". The rate cannot be rescaled to give fine-tuning
more effect. The paper's `1e-5` is about `3e-5` of its own student learning
rate (`0.3` for CIFAR-10, table 8), whereas against this card's `0.03` it is
already `3e-4` of the training rate. A fine-tuning step strong enough to free
the minority (the last row) overfits the 64 labels: the `h=0` NLL rises from
`0.24` to `0.40`, and the paired gain becomes noise with a standard deviation
of `0.34`. Those two rows are not part of the proposal.

**The proposed study.** Same fixture, seeds, arms and pairing as §6.5. Each of
the three arms runs `meta_train` and then `finetune`. The fine-tuning stage
consumes the labelled rows identically in every arm, because its batch is the
whole labelled pool. Its weak view draws from the stage seed, which the arms
share.

1. The status metric becomes the paired minority-accuracy gain of the
   **fine-tuned** student, MPL weight `1.0` minus `h=0`. The bar, `0.10` in
   mean by at least one standard error, is unchanged. §2's estimand is "the
   final student", and under the paper's method the final student is the
   fine-tuned one.
2. The `h=0` trap guardrail, `<= 0.60`, is read from the fine-tuned `h=0`
   student, so the fixture is still tested on the model that is reported.
3. The teacher NLL ratio, finiteness and class-mass guardrails are unchanged.
   Finiteness also covers the fine-tuning stage.
4. The §6.5 metrics of the students before fine-tuning stay as information.
   This keeps the ledger comparable with the `6494ee9b4227` row.

The predicted status is `deviating`, with no material change from §6.5: the
expected gain is about `+0.005`, as before. The point of the amendment is
fidelity. Deviation 5 is the last paper mechanic the card omits, and the
project's rule is to implement a component faithfully before concluding that
it does not matter. The screen predicts the outcome but does not substitute
for the 100-seed run.

**What implementing it would touch.** `xty2/core/recipe.py` and the compiler
(the role-selected initialisation and its plan line); `run_program` in
`xty2/training/executors.py`; Tier 0 tests for both, each seen failing; the
recipe (a second stage, with no new logic); §3.2, §4 (a `finetune` block, and
the new stage-scoped keys that a two-stage program gives every existing key),
§5 (row 5 marked `withdrawn`, the new row), §5.1, §6, §6.1 and §7; the
`batch-row-repetition` ledger row in `DESIGN.md`; the Tier 1 smoke test, so
that it runs the two-stage program; and the benchmark module. Fine-tuning adds
3,000 full-batch steps on 64 rows to each replicate, which is small against the
9,000 joint steps.

**Questions settled in review.**

- Should plain SGD stand? The paper's "SGD" and the code's `optim_type='sgd'`
  both point to plain gradient descent, but the README's training commands
  use `momentum`, and no published command shows fine-tuning.
- Should `1e-5` be the effective rate, or the flag value that
  `common_utils.py:402` would scale by `512/256` to `2e-5`? The proposal takes
  the stated value. The screen's parameter shift suggests that either value is
  inert.
- Is moving the status metric to the fine-tuned student acceptable, given
  that the screen predicts the same verdict?

### 6.7 Amendment: one labelled set for the whole program

**Approved on 2026-10-23 (option 1) and implemented.** `MissingnessSpec` takes
`drawn="per_program"`, and this recipe's data policy declares it. The plan
prints it under `data.missingness_mechanism` as "drawn once per program". The
default stays `"per_stage"`, so no other recipe's draws or plan digests change.
`meta_train` is stage 0, whose seed is the program seed, so its mask, and with
it every pre-fine-tuning metric, is unchanged from the `6494ee9b4227`
protocol. Tier 0 asserts that both stages see the same 64 labelled rows, and
that the per-stage contrast differs only in `finetune`. The text below is the
question as raised.

Implementing §6.6 exposed a framework behaviour that the approved card did not
list. `run_program` builds each stage's training population from that stage's
own seed, and the induced MCAR mask is a hash of `row_id` under that seed
(`xty2/training/loading.py:196-232`). The two stages therefore see different
labelled rows. For replicates 0–4, `finetune`'s 64 labelled rows share only 2–6
rows with `meta_train`'s. As built, fine-tuning trains on about 60 labels that
joint training treated as missing. That nearly doubles the label budget and
contradicts paper §3.2, which fine-tunes on "the labeled dataset" the method
was given. The benchmark's paired-rows guard caught it: the `finetune` and
`finetune_information` stages of one arm fine-tuned on different rows.
§6.6's screen is unaffected, because it fine-tuned on `meta_train`'s own
labelled rows.

No shipped recipe has met this, because none uses labels in two stages under
an induced mask. PAWS, SCARF and VIME pre-train on unlabelled or all rows and
use labels only in `joint_fit`; `tarnet`, `ssdml` and `cycle_dual` declare no
induced mask. The behaviour still changes what those programs' earlier stages
treat as unlabelled.

**Options.**

1. **Recommended: an opt-in data-policy field that draws the mask once per
   program.** For example, `MissingnessSpec(..., drawn="per_program")` uses
   stage 0's seed for every stage. The current behaviour stays the default
   (`"per_stage"`), so no other recipe's draws change. The plan prints the
   choice under `data.missingness_mechanism`, so it is a card key, and §4
   would bind it. `meta_train` keeps its current mask, because it is stage 0,
   so the pre-fine-tuning metrics stay comparable with the `6494ee9b4227`
   row. Second consumer: the staged fine-tuning programs in `BACKLOG.md` §2.12
   (SimCLRv2, Noisy Student), which fine-tune on the same labelled set the
   earlier stage was given.
2. A stage-level field that reuses an earlier stage's whole training
   population, for example `Stage.population_from="meta_train"`. This is
   equally opt-in, but it states a data-policy fact as a property of a stage.
3. Draw the mask once per program for every recipe. This is arguably the right
   default, but it changes the populations, and so the recorded Tier 2
   results, of PAWS, SCARF and VIME. Each would need re-running in the same
   change, which is outside this card.

When the question was raised, the branch implemented §6.6 with the per-stage
behaviour, and the benchmark's paired-rows guard refused to run.

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
| Does the student receive final labelled fine-tuning? | Yes: stage `finetune`, from the `inner_student` checkpoint, for 1,000 steps. | Paper §2, final paragraph, and §3.2; deviation 5 is withdrawn (§6.6). |
| Fine-tuning optimiser: plain SGD or momentum? | Plain SGD, no momentum. | Paper §3.2 says "SGD", where appendix C.4 names "Nesterov Momentum" for training, and `common_utils.py:452-455` builds plain gradient descent for `optim_type='sgd'`. No published command shows fine-tuning; the README's training commands use `momentum`. Settled in review (§6.6). |
| Is `10^-5` the effective rate, or a flag value the code scales by `train_batch_size/256` (`common_utils.py:402`)? | The effective rate, `1e-5`. | The paper states the rate, not the flag. A scaled `2e-5` would be equally inert: at `1e-5` the screen moved parameters by at most `0.008` (§6.6). |
| Which fine-tuning step count? | 1,000. | §3.2 gives 1,000 for CIFAR-10 and SVHN, the benchmarks the low-label fixture stands in for, and 2,000 for ImageNet-10%. |
| Weight decay and label smoothing during fine-tuning. | Weight decay `0.0005`; no label smoothing. | The reference's `Supervised` step adds its L2 term (`training_utils.py:147`); the value is §4's inherited one. Label smoothing follows deviation 3. |
| Which view does fine-tuning read? | `weak_x`. | The reference's labelled CIFAR-10 stream applies a random flip and crop (`data_utils.py:472-481`, `:504-516`), which `uda.md` §3.2 maps to `weak_x`. |
| Checkpoint/resume of the moving baseline and two optimisers. | Unsupported in the first implementation; runs are uninterrupted and reset from seed. | This card requires deterministic full executions, not bit-identical mid-stage resume. A future resume claim would engage `checkpointed-objective-state`. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | PR #40 review plus implementation audit | 2026-09-04 |
| Plan diffed against §3.2 and §4 | Codex | 2026-09-04 |
