# Recipe spec card: simmatch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** review §2–§5 before implementation. This card stops at the
> review boundary required by `CLAUDE.md`; no SimMatch code belongs in this PR.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [SimMatch: Semi-supervised Learning with Similarity Matching](https://arxiv.org/abs/2203.06915) |
| Authors, year | Mingkai Zheng, Shan You, Lang Huang, Fei Wang, Chen Qian, Chang Xu; 2022 (CVPR 2022) |
| DOI / arXiv | [10.1109/CVPR52688.2022.01407](https://doi.org/10.1109/CVPR52688.2022.01407); [arXiv:2203.06915](https://arxiv.org/abs/2203.06915) |
| Version used | arXiv v2, 2022-03-17. The method is §3, equations (1)–(12), and algorithm 1; CIFAR settings are §4.1; ImageNet settings are §4.2; mechanism ablations are §4.3, tables 5–7 and figure 5. Section references in this card are to this card unless prefixed “the paper’s”. |
| Reference implementation | [`mingkai-zheng/SimMatch`](https://github.com/mingkai-zheng/SimMatch) @ [`f10254b72ae9f4b968fe8ea91d74749e8c247237`](https://github.com/mingkai-zheng/SimMatch/tree/f10254b72ae9f4b968fe8ea91d74749e8c247237): `simmatch.py`, `models/simmatch.py`, `data/transforms.py`, `utils/lr_schedule.py`, and `script/train.sh`. The independent [`microsoft/Semi-supervised-learning`](https://github.com/microsoft/Semi-supervised-learning) SimMatch port @ `1ef4cbebcc0b368158315aeb425053858cf6c845` is used only to resolve the paper’s small-bank temporal-ensemble path where the authors’ public repository provides ImageNet code and links the CIFAR code through Google Drive. |
| Reference impl. runnable? | Not attempted. The supplied ImageNet entry point assumes Slurm, distributed CUDA, and a local ImageNet split; the card depends on reading it, not on executing it. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities `p(t | x)` and treatment-specific outcome means, fitted in one joint stage. SimMatch changes the treatment target by letting a labelled memory of instance embeddings refine semantic pseudo-labels and letting semantic probabilities refine instance-similarity targets.
- **Mechanism claim:** weak and strong views should agree in both class-probability space and similarity-to-labelled-instances space. A labelled memory makes those two spaces communicate: unfolding a semantic distribution over the labels attached to memory entries calibrates the instance target, while aggregating instance similarity by those labels refines the semantic target. The paper reports that removing semantic refinement (`alpha = 1`) costs 1.8 ImageNet top-1 points at 100 epochs, removing calibrated instance targets costs 9.4 points, and replacing the instance-matching loss with InfoNCE or SwAV costs 8.2 or 12.0 points (figure 5 and tables 5–7).
- **Nearest shipped baseline and controlled difference:** `fixmatch`. Both use a weak semantic target, a confidence gate, a strong-view prediction, the same quota shape, and the same project-local causal stack. The primary §6 pair holds SimMatch’s soft target, distribution alignment, projection head, labelled memory, instance-consistency loss, views, batches, optimiser, schedule, and seeds fixed; its ablation removes only the two propagation operations in equations (8) and (10). This separates “semantic and instance labels improve one another” from the less specific claim that another loss or a softer FixMatch target helps.
- **Not claimed:**
  - No published image-classification number is reproduced. The paper’s CIFAR and ImageNet classes become treatment levels in a six-feature tabular fixture, so §6 is a project-local mechanism test.
  - No representation-transfer claim. The paper’s table 3 freezes an ImageNet ResNet-50 and fits downstream linear classifiers; this card evaluates the treatment and outcome heads trained on one XTY population.
  - No causal identification from pseudo-labels. The memory-refined distribution is a training target, not an observed treatment and not an assumption that makes missing treatments identified.
  - No large-bank scalability claim. This card deliberately selects the paper’s small-`K` temporal-ensemble variant, not its ImageNet student-teacher variant.
  - No claim that uniform distribution alignment is valid under arbitrary treatment prevalence. The primary fixture is marginally balanced; §6 includes a skewed-propensity diagnostic and reports the failure if alignment moves predictions away from the true marginal.

## 3. Equations and mapping

### 3.1 As published

For a labelled batch `X = {(x_b, y_b)}_{b=1..B}`, encoder `F`, classifier
`phi`, and weak augmentation `T_w`, the supervised loss is

$$
h = F(T_w(x)), \qquad p = \phi(h), \qquad
\mathcal L_s = \frac{1}{B}\sum_b H(y_b, p_b).
\tag{1}
$$

For `mu B` unlabelled rows, let `p_b^w` and `p_b^s` be the semantic
distributions under weak and strong views. Distribution alignment divides by a
moving average of weak predictions and renormalises. The semantic consistency
term is

$$
\mathcal L_u = \frac{1}{\mu B}\sum_b
\mathbf 1\!\left(\max \operatorname{DA}(p_b^w) > \tau\right)
H\!\left(\operatorname{DA}(p_b^w), p_b^s\right).
\tag{2}
$$

Unlike FixMatch, `DA(p^w)` remains a soft target: the paper explicitly says it
is neither sharpened nor converted to one-hot form.

Let `g` project the encoder representation to an L2-normalised embedding. A
labelled memory contains `K` embeddings `z_i` and their ground-truth labels.
Weak and strong similarity distributions over those same entries are

$$
q^w_{bi} =
\frac{\exp(\operatorname{sim}(z_b^w,z_i)/t_w)}
{\sum_{k=1}^{K}\exp(\operatorname{sim}(z_b^w,z_k)/t_w)},
\tag{3}
$$

$$
q^s_{bi} =
\frac{\exp(\operatorname{sim}(z_b^s,z_i)/t_s)}
{\sum_{k=1}^{K}\exp(\operatorname{sim}(z_b^s,z_k)/t_s)},
\tag{4}
$$

with cosine similarity (a dot product after L2 normalisation). Before label
propagation, instance consistency would be

$$
\mathcal L_{in} = \frac{1}{\mu B}\sum_b H(q_b^w,q_b^s),
\tag{5}
$$

and the complete objective is

$$
\mathcal L = \mathcal L_s + \lambda_u\mathcal L_u
+ \lambda_{in}\mathcal L_{in}.
\tag{6}
$$

SimMatch couples the two target spaces using the known label `ell_i` of every
memory entry. “Unfolding” copies each semantic class probability to all memory
entries of that class,

$$
p^{unfold}_{bi} = p^w_{b,\ell_i},
\tag{7}
$$

then calibrates and renormalises the weak instance target,

$$
\hat q_{bi} =
\frac{q^w_{bi}p^{unfold}_{bi}}
{\sum_{k=1}^{K}q^w_{bk}p^{unfold}_{bk}}.
\tag{8}
$$

In the reverse direction, “aggregation” sums the *uncalibrated* weak instance
similarities belonging to each class,

$$
q^{agg}_{bj} = \sum_{i=1}^{K}\mathbf 1(\ell_i=j)q^w_{bi},
\tag{9}
$$

and smooths the semantic target,

$$
\hat p_{bj} = \alpha p^w_{bj} + (1-\alpha)q^{agg}_{bj}.
\tag{10}
$$

`hat p` replaces `DA(p^w)` in equation (2), including in its confidence gate,
and `hat q` replaces `q^w` in equation (5). Algorithm 1 detaches both targets.

For a large labelled memory the paper uses an EMA teacher,

$$
F_t \leftarrow mF_t + (1-m)F_s,
\tag{11}
$$

to write slowly changing support embeddings. For small `K`, which is the path
selected here, it instead temporally smooths each memory slot,

$$
Q_i \leftarrow \operatorname{normalize}
\left(mQ_i + (1-m)z_i^{current}\right).
\tag{12}
$$

The normalisation and the distinction between the previous slot and the
current embedding are made explicit from the reference ports; the printed
equation reuses `z_t` on both sides and is otherwise not executable.

### 3.2 Mapping to xty2

One `joint_fit` stage. The paper’s classes are the levels of `t`, so its
classifier is `CategoricalPropensity`. The stage carries the ordinary xty2
outcome terms unchanged. `SimilarityMatchingTreatmentNLL` owns one piece of
stage-local state, and `LabeledMemoryInstanceConsistency` reads that named
sibling state. Both ask the state to prepare the previous-bank targets and
observe the current support embeddings; preparation and observation are each
idempotent within a step, so objective declaration order cannot change either
loss.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `F` | encoder | `X_RAW -> X_REPR` | `MLPEncoder`, the reviewed P5 backbone (deviation 4) |
| `phi` | semantic classifier | `X_REPR -> T_GIVEN_X` | `CategoricalPropensity` |
| `g` | nonlinear projection head | `X_REPR -> X_PROJ` | `ProjectionHead(widths=(200, 128), activation="relu", normalisation="row_l2")` |
| `T_w` | weak augmentation | `weak_x` | `ViewSpec("weak_x", transforms=(FeatureMask(p=0.1),), draws=1)` |
| `T_s` | strong augmentation | `strong_x` | `ViewSpec("strong_x", transforms=(FeatureMask(p=0.1), FeatureMask(p=0.5)), draws=1)` |
| `p^w` | aligned weak semantic distribution | `T_GIVEN_X @ weak_x` | detached input to `SimilarityMatchingTreatmentNLL`; a 32-step moving average in its state performs DA |
| `p^s` | strong semantic distribution | `T_GIVEN_X @ strong_x` | prediction side of that objective; gradient reaches encoder and propensity |
| `z^w` | weak projected embedding | `X_PROJ @ weak_x` | detached for target construction; observed-row values are written to the labelled memory after both losses read it |
| `z^s` | strong projected embedding | `X_PROJ @ strong_x` | prediction side of `LabeledMemoryInstanceConsistency`; gradient reaches encoder and projection head |
| `Q_f, Q_l`, `K` | one feature slot and one immutable known label per observed training row | — | `LabeledSimilarityMemory`, initialised from `TrainingPopulation`; slots are keyed by sorted observed `row_id`, not FIFO |
| equations (3), (4), `t_w=t_s=0.1` | weak/strong distributions over memory slots | — | `SimilarityMatchingSpec(instance_temperature=0.1, ...)`, shared by both objectives |
| equations (7), (8) | semantic-to-instance unfolding and calibrated target | — | prepared in the shared state; consumed by `LabeledMemoryInstanceConsistency` |
| equations (9), (10), `alpha=0.9` | instance-to-semantic aggregation and smoothed target | — | prepared in the same state; consumed by `SimilarityMatchingTreatmentNLL` |
| equation (1) | labelled cross-entropy | `T_GIVEN_X @ weak_x` | `ObservedTreatmentNLL(realisation=weak_x)`, rows `t_observed`, `reduction="mean"` |
| equation (2) | gated soft semantic consistency | `T_GIVEN_X @ weak_x,strong_x` | `SimilarityMatchingTreatmentNLL`, rows `t_missing`, threshold `0.95`, `reduction="mean"` |
| equation (5) | instance consistency | `X_PROJ @ weak_x,strong_x` | `LabeledMemoryInstanceConsistency(owner="similarity_matching_treatment_nll")`, rows `t_missing`, no gate, `reduction="mean"` |
| equation (12), `m=0.7` | small-bank temporal ensemble | — | detached random-access update inside `LabeledSimilarityMemory`, after the step’s targets are prepared |
| one-epoch warm-up | do not read random initial slots | — | `SimilarityMatchingSpec(warmup_steps=2)`; both propagation and instance loss are off at steps 0 and 1 (§7) |
| evaluation EMA | model reported for the CIFAR experiments | — | `TeacherSpec(decay=0.999, role="evaluation")`; no objective reads it |
| `B=64`, `mu=7` | batch composition | — | `QuotaSampler(Quota("t_observed", 64), Quota("t_missing", 448))` |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed`, `reduction="population"` |
| — (project-local) | exact marginalisation over missing treatment | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing`, ramped weight |

Four details are load-bearing.

- **The bank contains labelled instances, not recent unlabelled predictions.**
  It has exactly one slot for every observed training row and is updated by
  stable row identity. That is different from CoMatch’s insertion-ordered FIFO
  and from PAWS’s current-batch support set.
- **The two propagation directions are different arithmetic.** Equation (8)
  multiplies then renormalises in instance space. Equation (10) convexly mixes
  in class space. A generic “label smoothing” helper would erase the method.
- **The semantic target stays soft.** The confidence gate selects rows but
  neither the selected semantic target nor the instance target is hardened.
- **The bank is read before it is written.** Current support embeddings may
  train through equation (1), but they cannot alter the targets charged in the
  same optimiser step.

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the future recipe and tests.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.similarity_matching_treatment_nll: p(t|x) @ view=weak_x params=student, x_proj @ view=weak_x params=student, labelled memory
    joint_fit.labeled_memory_instance_consistency: x_proj @ view=weak_x params=student, labelled memory
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets:
    joint_fit.similarity_matching_treatment_nll: target                # hat p and its gate are constants of theta
    joint_fit.labeled_memory_instance_consistency: target              # hat q is constant; q^s and the strong branch train
  gradient_clipping: none                                              # paper and both reference ports name none
  marginal_nll_grad_path: both                                         # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.999                                                     # evaluation only; §7
  ema_applies_to_buffers: false                                        # parameter EMA; the declared graph has no BN
  teacher_in_train_mode: false                                         # evaluation role
  teacher_requires_grad: false
  # role = evaluation. This is not equation (11): the selected small-K path
  # trains both weak and strong targets with the current student.

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: mean                             # equation (1), denominator B
    joint_fit.similarity_matching_treatment_nll: mean                 # equation (2), denominator mu*B including rejected rows
    joint_fit.labeled_memory_instance_consistency: mean               # equation (5), denominator mu*B
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.similarity_matching_treatment_nll: t_missing            # paper §4.1 uses the rest of train as U
    joint_fit.labeled_memory_instance_consistency: t_missing
    joint_fit.missing_treatment_marginal_nll: t_missing
    # Both SimMatch objectives additionally read t_observed as support_rows;
    # widening eligible_rows would incorrectly charge the unlabelled losses on them.
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0                              # equation (6)
    joint_fit.similarity_matching_treatment_nll: 1.0                  # lambda_u, paper §4.1
    joint_fit.labeled_memory_instance_consistency: 1.0                # lambda_in, paper §4.1
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.similarity_matching_treatment_nll: constant 1.0              # steps 0..1 use unpropagated p^w, but the semantic loss remains active
    joint_fit.labeled_memory_instance_consistency: constant 1.0            # source warm-up makes the objective return zero at steps 0..1
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: simmatch(instance_weak=0.1, instance_strong=0.1)        # equations (3), (4); paper §4.1
  sharpening: none                                                     # paper §3.1: soft DA(p^w), not argmax
  confidence_threshold: 0.95                                          # tau, equation (2), paper §4.1

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)                          # paper §4.1
  lr: 0.03                                                             # paper §4.1
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))       # same FixMatch schedule and rebasing as the shipped fixmatch card
  weight_decay: 0.0005 (all trainable components; norm and bias exempt) # paired FixMatch value; deviation 4 and §7
  batch_size: 512                                                      # 64 + 448, derived from quotas
  labelled_unlabelled_ratio: 7.0                                       # mu, derived from quotas
  total_steps_or_epochs: 3000                                          # optimiser steps; deviation 3

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]
    projection_head: [200, 128]                                        # hidden width = encoder width, output dim 128
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: elu
    projection_head: relu
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: row_l2
    projection_head: row_l2                                            # cosine is a dot product of unit vectors
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    projection_head: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    projection_head: torch Linear default Kaiming-uniform
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: "x: none fitted on 'train'"                         # §6.1 draws standardised features
  outcome_scaling: "y: zscore fitted on 'train'"
  treatment_encoding: n/a                                              # XTYBatch integer levels; memory labels are those observed values
  split_protocol: fixed project-local DGP in §6.1; no CIFAR or ImageNet split applies (deviation 10); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id
```

`alpha=0.9`, memory momentum `m=0.7`, the 32-step alignment window, the
two-step warm-up, and `K=N_observed=64` have no canonical `FIDELITY.md` §2
keys. They are required constructor arguments of one shared frozen
`SimilarityMatchingSpec`; `plan_details()` must print all five and the memory
key/update policy. This follows the reviewed CoMatch and PAWS precedent without
adding five one-recipe card keys.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Predict treatment rather than image class and add the reviewed outcome likelihood plus exact missing-treatment marginalisation to equation (6). | The paper’s categorical classifier maps directly to `p(t\|x)`. The causal terms are held identical in every §6 arm, so they cannot explain the controlled difference. | No published accuracy applies; §6 measures whether the ported propagation improves treatment targets and held-out treatment NLL without harming outcome NLL. |
| 2 | `judgement` | — | Replace crop/flip with `FeatureMask(0.1)` and the paper’s strong image view with `FeatureMask(0.1)` followed by `FeatureMask(0.5)`. | There is no image geometry in an XTY table. These are exactly the reviewed weak and strong views of `fixmatch.md`, so the nearest-baseline comparison does not move the invariance target. | The strong view can erase treatment-predictive columns. §6 measures Bayes-label flip rates before training and refuses to interpret a destructive view as evidence about SimMatch. |
| 3 | `judgement` | — | Run 3,000 optimiser steps and re-base the cosine schedule on that budget rather than the paper’s CIFAR-scale run. | This is the shared project-local budget. The primary pair and every ablation receive identical steps and batches. | The bank and projection head may not converge on the source’s timescale. §6 reports target-quality and memory diagnostics throughout training so an unfinished curriculum is visible. |
| 4 | `judgement` | — | Retain the P5 MLP causal stack rather than WRN-28-2, but add the source-shaped two-layer, 128-dimensional, L2-normalised projection head. Use the shipped FixMatch optimiser policy, including its bias/norm decay exemptions. | Holding the causal stack and optimiser fixed is what makes the propagation addition attributable. The projection head is method-specific, so its depth, hidden-width rule, activation, output width, and normalisation follow the authors’ code. | The learned geometry is project-local. Equation (8) can fail because the MLP space is poor even when its arithmetic is right; §6’s bank-label nearest-neighbour and same-row cross-view diagnostics distinguish that case. |
| 5 | `framework-limitation` | `batch-row-repetition` | Use 64 observed treatments rather than the paper’s 40-label CIFAR-10 regime, preserving `B=64` and `mu=7`. | The source repeats labelled rows inside a batch when the pool is smaller than `B`; `XTYBatch.row_id` must be unique. This is the same unresolved boundary paid by FixMatch and CoMatch. | More distinct supervision per step than the source’s smallest regime. It moves every §6 arm equally and rules out a claim about the 40-label result. |
| 6 | `judgement` | — | Select the paper’s small-`K` temporal-ensemble bank (equation 12, `m=0.7`) rather than the large-`K` EMA-teacher bank (equation 11, `m=0.999`). | The fixed fixture has 64 labelled support rows and two treatment levels. The paper says a teacher is unnecessary when `K` is small; using it here would import a scaling intervention into the loss target for no scaling problem. `mean_teacher` remains the controlled card for teacher-produced targets. | The weak target follows current student parameters while support slots lag with momentum 0.7. Results do not support the ImageNet student-teacher variant. |
| 7 | `framework-limitation` | `checkpointed-objective-state` | A resumed run cannot restore the labelled feature bank or DA window; the fixed §6 protocol is one uninterrupted stage execution. | xty2 stage-local objective state deliberately is not a checkpoint artifact. The authors’ bank and labels are registered buffers and `load_state_dict` restores them, so silently restarting them would change subsequent targets. | None in the declared uninterrupted run. Interrupted/resumed equivalence is out of scope until the debt is repaid and must not be claimed. |
| 8 | `framework-limitation` | `augmentation-vocabulary` | No CTAugment/RandAugment or image-colour stack; strong-view strength is fixed. | The paper follows FixMatch’s image augmentation. xty2 has no learned tabular operation vocabulary with comparable magnitudes, and the same ledger row already records the limitation for neighbouring cards. | Unknown sign. Target corruption and cross-view alignment are reported; no claim is made that feature masking reproduces the source augmentation distribution. |
| 9 | `judgement` | — | Use decay 0.999 for an evaluation-only EMA and report both student and EMA. | The paper says its CIFAR result uses an EMA but gives no small-dataset decay. `0.999` is the authors’ public default and the shipped FixMatch choice. Reporting both prevents the unresolved value from becoming result selection. | May smooth or lag a 3,000-step run. The primary tolerance must pass for both parameter sets. |
| 10 | `judgement` | — | Use the fixed §6.1 XTY DGP rather than CIFAR-10/100 or ImageNet, and test the propagation mechanism against a matched ablation rather than borrowing top-1 accuracy. | None of the paper’s datasets has treatment, outcome, or missing-treatment semantics. A published-number target would test a second data stack instead of this recipe’s mapping. | Evidence is limited to correct wiring and usefulness on the declared fixture. |

### 5.1 Framework additions made for this card

The card proposes four reversible, fidelity-bearing objects. It requests no new
port, row population, executor, artifact kind, or stage type. The one missing
lifecycle capability is left as typed debt in deviation 7 rather than smuggled
into a component buffer.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `SimilarityMatchingSpec` — frozen shared values for temperatures, `alpha`, bank momentum, DA window, warm-up, threshold, and support population | fidelity-bearing, reversible | both SimMatch objectives | not required | The two objectives must use the same bank arithmetic and source constants. One value object makes equality inspectable and puts otherwise keyless values in the plan digest. |
| `LabeledSimilarityMemory` — stage-local state with one random-access feature slot and known label per observed training `row_id`, plus a DA queue | fidelity-bearing, reversible | both SimMatch objectives through one owner | not required; CoMatch already constrains the general lifecycle but its FIFO key space must not be widened into this one | Equations (7)–(12) cannot be computed from one batch. `TrainingPopulation` already supplies the stable rows and labels, and `StatefulObjective` already owns reset and sibling reads. This object is recipe-local arithmetic over those contracts, not new framework vocabulary. |
| `SimilarityMatchingTreatmentNLL` — equation (2) with the aggregated soft target `hat p`; owner of the memory | fidelity-bearing, reversible | this card | not required | Existing `PseudoLabelTreatmentNLL` hardens a target read directly from `T_GIVEN_X`; neither is true here. A flag would put two target-generating algorithms inside one objective. |
| `LabeledMemoryInstanceConsistency` — equations (3)–(5), reading calibrated `hat q` from the named owner | fidelity-bearing, reversible | this card | not required | `InfoNCEContrastive` uses the identity as its target. The paper’s table 7 says that substitution loses 8.2 points, so the labelled distribution over memory slots is the mechanic rather than an interchangeable contrastive loss. |

**Existing contracts deliberately reused.** `X_PROJ` and `ProjectionHead` come
from SCARF/PAWS; `support_rows` comes from PAWS and was already exercised by
CoMatch; objective state, `TrainingPopulation`, idempotent sibling reads, quota
sampling, distribution-alignment precedent, soft sharpening `none`, and
evaluation-only EMA all exist. Implementation should amend this card and stop
again if any of those shapes proves insufficient.

## 6. Reproduction target

The primary pair isolates the two arrows that make SimMatch different. The full
arm uses equations (8) and (10). The no-propagation arm sets `hat q=q^w` and
`hat p=p^w`, while retaining the same soft semantic loss, DA, projection head,
labelled memory, instance loss, temporal bank update, gate, quotas, views,
optimiser, schedule, initial parameters, seeds, and batch stream.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in 6.1
  variant: paired full SimMatch against no semantic-instance propagation; identical seeds, initialisation, batches, optimiser, schedule, views and all non-propagation mechanics
  split: 1024 train rows with exactly 64 observed treatments; 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio full over no-propagation, for student and evaluation EMA; online hidden-label NLL ratios for hat p over p^w and aggregate(hat q) over aggregate(q^w); outcome NLL, gate rate, bank coverage and representation alignment as guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: held-out treatment-NLL ratio < 1.0 in mean by at least one standard error for both student and EMA; terminal hat-p target-NLL ratio < 1.0; terminal aggregate-hat-q target-NLL ratio < 1.0; held-out outcome NLL within 1.05x of the ablation; terminal gate rate >= 0.5; bank coverage = 1.0 before the first propagated target; mean same-row weak/strong cosine at least 0.2 above mean cross-row cosine
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

For replicate `r = 0..9`, use base seed `s_r = 90000 + 100r`; generate the
1,024-row training population with `s_r+1`, the 2,048-row held-out population
with `s_r+2`, initialise model parameters with `s_r+6`, and train with stage
seed `s_r+10000`. This is the shipped FixMatch fixture, repeated here so this
card’s benchmark contract does not depend on a deleted or amended section of
another card.

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

Exactly 64 training treatments are observed under a seeded MCAR permutation;
all held-out treatments and every outcome are observed. Assert that the fixed
observed support contains both treatment levels and fail the fixture rather
than reseeding if it does not: conditioning the missingness draw on `t` would
no longer be MCAR. Fit outcome standardisation on the complete training
population. Every step draws the same paired quota stream, `B=64` observed and
`mu B=448` missing rows, in every arm.

Before training, measure rather than assume:

1. Bayes-optimal treatment-label flip rate under the weak and strong views.
2. Observed support count by treatment and the labelled-memory size `K=64`.
3. The true marginal treatment prevalence in train and held-out populations;
   the primary DA target is uniform only because this generator is balanced in
   expectation, not by method guarantee.

### 6.2 Predeclared evidence

**Tier 0 (invariants).**

1. On a hand-built two-class, four-slot bank, equations (7)–(10) match direct
   tensor calculations; both outputs are row-normalised.
2. `q^agg` in equation (9) uses the original `q^w`, not the calibrated `hat q`;
   changing one slot’s semantic factor changes `hat q` but leaves `q^agg`
   unchanged.
3. Slots are keyed by sorted observed `row_id`. Permuting a batch does not move
   its slot; a missing or duplicate support identity is rejected; a hidden
   treatment can never enter `Q_l`.
4. Targets read the previous-step bank. The current support embedding is
   detached, temporally mixed, L2-normalised, and visible only on the next
   optimiser step.
5. Preparation and observation are idempotent within one step. Reversing the
   two objective declarations gives bit-identical losses and the bank updates
   exactly once.
6. Steps 0 and 1 use `p^w`, set instance loss to zero, and still fill/update the
   bank. Step 2 is the first propagated target and sees coverage 1.0.
7. `hat p`, `hat q`, weak probabilities, weak embeddings, memory contents, and
   the confidence indicator carry no gradient. Strong semantic logits and
   strong embeddings do carry gradient.
8. Rejected semantic rows remain in equation (2)’s `mu B` denominator; an
   all-rejected batch returns exactly zero without an empty mean. Equation (5)
   still charges all missing rows.
9. With `alpha=1`, `hat p=p^w` exactly. With unfolding disabled, `hat q=q^w`
   exactly. Enabling both recovers the full object without changing any other
   plan field.
10. Distribution alignment leaves a stationary uniform stream unchanged and
    uses the current plus at most 31 prior batch means.
11. The state is fresh per stage execution and cannot run behind
    `ExternalBatches` or `cross_fit`, because it requires a population and has
    unresolved reset semantics across folds.
12. `plan.hyperparameters` matches every non-`n/a` §4 key, and
    `plan_details()` prints `alpha`, both temperatures, bank momentum, DA
    window, warm-up steps, memory key space, update order, and `K`.

**Tier 1 (smoke fit and mechanism arms).**

1. Run the full and no-propagation pair for one seed. Both unsupervised losses
   fall after warm-up; the full arm beats marginal-frequency treatment NLL;
   `hat p` and class-aggregated `hat q` are scored against the fixture’s hidden
   treatments whether they improve or not.
2. **Instance-to-semantic off:** `alpha=1`, equation (8) retained. This is the
   paper’s `w/o hat p` arm; predeclared expectation from figure 5/table 5:
   worse than full.
3. **Semantic-to-instance off:** set `hat q=q^w`, equation (10) retained. This
   is the paper’s `w/o hat q` arm; predeclared expectation from table 5: worse,
   with a larger effect than arm 2.
4. **Instance loss off:** `lambda_in=0`, propagation arithmetic otherwise
   unchanged. This tests whether a detached, initially untrained projection
   space can appear to improve semantic targets by chance.
5. **No temporal memory:** set bank momentum to zero, so the next step reads
   the last current support embedding. Report target NLL and weak/strong
   alignment; no direction is asserted because the paper does not ablate `m`.
6. **Permuted memory labels:** apply one fixed non-identity permutation to
   `Q_l` after warm-up while retaining features and class counts. Both target
   improvements should disappear. This is a wiring control, not a useful
   method arm.
7. **Distribution alignment off**, on the balanced fixture and on a diagnostic
   variant with `p(t=1)=0.15`. Report predicted and true marginals. No skewed-
   fixture performance direction is required; a uniformising operation can be
   misspecified there.
8. Report bank slot norm, class coverage, nearest-neighbour label agreement,
   gate rate, retained semantic-target impurity, same-row weak/strong cosine,
   and cross-row cosine trajectories for every arm.

**Tier 2 (fixed ten-replicate target).** Run only the full and no-propagation
pair under the YAML contract above. The Tier 1 arms diagnose a failure but do
not enter the acceptance metric and cannot be selected after seeing Tier 2.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | — | — |

## 7. Unknowns

| Unspecified or inconsistent in source | Our choice | Basis |
|---|---|---|
| Equation (12) prints the same time-indexed embedding on both sides. | `Q_i <- normalize(0.7 Q_i + 0.3 z_i_current)`. | The prose calls this temporal ensembling; SemiLearn’s small-dataset port implements exactly this update and normalises the result. Without the previous slot the equation is an identity up to scale. |
| Whether equation (9) aggregates `q^w` or the calibrated `hat q`. | Aggregate the original `q^w`. | Literal equation (9) and the authors’ code: `aggregated_prob.scatter_add(..., teacher_prob_orig)`, while the scaled distribution is a different tensor used only by the instance loss. |
| Whether equation (8)’s unfolding uses the raw or distribution-aligned semantic prediction. | The aligned `p^w`. | Algorithm 1 computes `p^w=DA(phi_t(h^w))` before propagation; the authors’ code calls distribution alignment before gathering semantic factors by memory label. |
| The paper does not mention a propagation warm-up. | Disable propagation and instance loss for the first two optimiser steps, while writing the bank. | Both reference ports disable them for epoch 0. On §6’s 960-row missing population with `448` missing rows and dropped incomplete batches, one source-style epoch is `floor(960/448)=2` steps. This is printed in the plan rather than hidden behind the word “epoch”. |
| Initial memory contents. | Random unit vectors, labels fixed from the observed population; no target reads them before every feature slot has been replaced once. | The authors and SemiLearn initialise random normalised features. Initial labels are zero in code and overwritten by indexed support batches; deriving immutable `Q_l` from `TrainingPopulation` is safer and arithmetically identical by the first read because the quota contains all 64 observed rows. |
| Read/write order. | Clone/read the prior bank, prepare both targets, then write current detached support embeddings once. | The authors’ `forward` clones `self.bank`, computes both losses, then calls `_update_bank`. Current supports must not leak into their own targets. |
| Whether weak and strong instance temperatures can differ. | Represent both fields, bind both to `0.1`, and reject a plan that silently aliases only one. | The paper has one `t`; the authors’ code exposes `tt` and `st` separately and the published command sets both to `0.1`. Separate fields preserve the code’s actual degree of freedom while the shared value object proves equality here. |
| Whether the semantic confidence comparison is `>` or `>=`. | `>=`. | Equation (2) and algorithm 1 print `>`; the authors’ code uses `.ge(threshold)`. The difference is measure-zero for continuous logits, and code is the arithmetic that produced the reported run. |
| Whether the gate uses `p^w` or propagated `hat p`. | Propagated `hat p`. | Algorithm 1 writes `1(max hat p > tau)H(hat p,p^s)`; the authors’ code constructs `prob_ku` first and then computes `max_probs` and the mask. This is essential: agreement between spaces is intended to raise or lower confidence. |
| Whether temporal momentum `m=0.7` is the same `m` as the evaluation EMA. | No. Bank momentum is `0.7`; evaluation EMA is `0.999`. | Paper §4.1 names `m=0.7` immediately after selecting the temporal-ensemble bank. The same section separately says final performance uses an EMA without stating its decay; the authors’ public default is `0.999`. One key must not silently control both lifecycles. |
| Projection-head width for the small-bank tabular port. | Hidden width equals encoder width `200`; output width `128`. | The authors’ `ResNet` head is `trunk_width -> trunk_width -> dim`, with `dim=128`. This transfers the structural rule, not a ResNet-specific width. |
| Weight decay scope for a projection head absent from the shipped FixMatch graph. | Use the FixMatch arm’s `0.0005` and bias/norm exemptions for every trainable component. | The CIFAR paper says it follows FixMatch; the downloadable CIFAR source is not pinned in Git. The authors’ ImageNet code instead applies global `1e-4`. Matching the controlled arm takes priority over pretending either image setting is uniquely authoritative here. |
| Checkpoint and resume semantics for the memory and DA window. | Unsupported; fresh per uninterrupted stage execution. | Deviation 7 and the new `checkpointed-objective-state` ledger row. The source restores registered buffers; xty2 objective state is not an artifact today. |
| No published target applies to this adaptation. | Fixed paired mechanism target in §6, plus the paper’s two propagation ablations and a label-permutation control. | This makes both directions of the claimed interaction falsifiable before implementation and prevents choosing whichever component happens to help after seeing results. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
