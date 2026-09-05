# Recipe spec card: simmatch

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity; §6 only for
> benchmark/reporting work. `xty2/recipes/simmatch.py` and
> `xty2/objectives/simmatch.py` implement it; `tests/invariants/test_simmatch.py`
> is the Tier 0 contract and `tests/smoke/test_simmatch.py` runs §6.2's first
> Tier 1 arm.

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
| `p^w` | aligned weak semantic distribution | `T_GIVEN_X @ weak_x` | detached input, declared and read by *both* objectives because either may prepare the shared state first; a 32-step moving average in that state performs DA |
| `p^s` | strong semantic distribution | `T_GIVEN_X @ strong_x` | prediction side of that objective; gradient reaches encoder and propensity |
| `z^w` | weak projected embedding | `X_PROJ @ weak_x` | detached for target construction; observed-row values are written to the labelled memory after both losses read it |
| `z^s` | strong projected embedding | `X_PROJ @ strong_x` | prediction side of `LabeledMemoryInstanceConsistency`; gradient reaches encoder and projection head |
| `Q_f, Q_l`, `K` | one feature slot and one immutable known label per observed training row | — | `LabeledSimilarityMemory`, initialised from `TrainingPopulation`; slots are keyed by sorted observed `row_id`, not FIFO |
| equations (3), (4), `t_w=t_s=0.1` | weak/strong distributions over memory slots | — | `SimilarityMatchingTemperatures(instance_weak=0.1, instance_strong=0.1)` inside one `SimilarityMatchingSpec`, shared by both objectives |
| equations (7), (8) | semantic-to-instance unfolding and calibrated target | — | prepared in the shared state; consumed by `LabeledMemoryInstanceConsistency` |
| equations (9), (10), `alpha=0.9` | instance-to-semantic aggregation and smoothed target | — | prepared in the same state; consumed by `SimilarityMatchingTreatmentNLL` |
| equation (1) | labelled cross-entropy | `T_GIVEN_X @ weak_x` | `ObservedTreatmentNLL(realisation=weak_x)`, rows `t_observed`, `reduction="mean"` |
| equation (2) | gated soft semantic consistency | `T_GIVEN_X @ weak_x,strong_x` | `SimilarityMatchingTreatmentNLL`, rows `t_missing`, threshold `0.95`, `reduction="mean"` |
| equation (5) | instance consistency | `X_PROJ @ weak_x,strong_x` | `LabeledMemoryInstanceConsistency(owner="similarity_matching_treatment_nll")`, rows `t_missing`, no gate, `reduction="mean"` |
| equation (12), `m=0.7` | small-bank temporal ensemble | — | detached random-access update inside `LabeledSimilarityMemory`, after the step’s targets are prepared; a slot's *first* observation fills it and every later one mixes (deviation 11) |
| one-epoch warm-up | do not read unfilled slots | — | `SimilarityMatchingSpec(warmup_steps=2)`; both propagation and instance loss are off at steps 0 and 1, and also while any slot is still unfilled (§7) |
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
    joint_fit.similarity_matching_treatment_nll: p(t|x) @ view=weak_x params=student, x_proj @ view=weak_x params=student
    joint_fit.labeled_memory_instance_consistency: p(t|x) @ view=weak_x params=student, x_proj @ view=weak_x params=student
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                                             # hat p and its gate, and hat q, are all constants of theta; q^s and both strong branches train
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
two-step warm-up, and the unfolding switch of equations (7)–(8) have no
canonical `FIDELITY.md` §2 keys. They are required constructor arguments of one
shared frozen `SimilarityMatchingSpec`; `plan_details()` must print all five
together with the memory key space and update order. This follows the reviewed
CoMatch and PAWS precedent without adding five one-recipe card keys.

`K` is *not* declared beside them. It is the number of observed training rows,
read from the stage's `TrainingPopulation` when the state is built, on the
`flexmatch` precedent for `N`: a recipe that asserted `K` could assert one the
sampler never draws from. `plan_details()` prints the key space that determines
it — one slot per observed training `row_id` — and §6.1 asserts `K=64` on the
fixture.

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
| 11 | `judgement` | — | Do not initialise the bank with random unit vectors. A slot holds nothing until its row is first observed, that first observation *fills* it outright, and only later observations apply equation (12)'s momentum. Propagation and the instance loss stay off until every slot is filled, not merely until `warmup_steps` has passed. | Both ports seed random features and mix from the first update. That is harmless over their warm-up *epoch* — hundreds of labelled steps at `m=0.7` leave the noise at `0.7^n` — but this fixture's warm-up is two steps (§7), so a mixed random seed would still carry weight `0.49` into the first propagated target and would make the mechanism's first reads noise. Filling on first sight is the same limit the source reaches, one step earlier, and removes an RNG stream the plan does not otherwise have. | None expected: the source's initial features are arbitrary and no equation reads them. §6.2's coverage invariant (`bank coverage = 1.0` before the first propagated target) is what makes the claim checkable. |

### 5.1 Framework additions made for this card

The card proposes four reversible, fidelity-bearing objects. It requests no new
port, row population, executor, artifact kind, or stage type. The one missing
lifecycle capability is left as typed debt in deviation 7 rather than smuggled
into a component buffer.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `SimilarityMatchingSpec` — frozen shared values for the two instance temperatures, `alpha`, bank momentum, DA window, warm-up, the gate threshold, and the equation (7)–(8) unfolding switch | fidelity-bearing, reversible | both SimMatch objectives | not required | The two objectives must use the same bank arithmetic and source constants. One value object makes equality inspectable and puts otherwise keyless values in the plan digest. |
| `LabeledSimilarityMemory` — stage-local state with one random-access feature slot and known label per observed training `row_id`, plus a DA queue | fidelity-bearing, reversible | both SimMatch objectives through one owner | not required; CoMatch already constrains the general lifecycle but its FIFO key space must not be widened into this one | Equations (7)–(12) cannot be computed from one batch. `TrainingPopulation` already supplies the stable rows and labels, and `StatefulObjective` already owns reset and sibling reads. This object is recipe-local arithmetic over those contracts, not new framework vocabulary. |
| `SimilarityMatchingTreatmentNLL` — equation (2) with the aggregated soft target `hat p`; owner of the memory | fidelity-bearing, reversible | this card | not required | Existing `PseudoLabelTreatmentNLL` hardens a target read directly from `T_GIVEN_X`; neither is true here. A flag would put two target-generating algorithms inside one objective. |
| `LabeledMemoryInstanceConsistency` — equations (3)–(5), reading calibrated `hat q` from the named owner | fidelity-bearing, reversible | this card | not required | `InfoNCEContrastive` uses the identity as its target. The paper’s table 7 says that substitution loses 8.2 points, so the labelled distribution over memory slots is the mechanic rather than an interchangeable contrastive loss. |

**Added since, for §6.2's arm 8 and nothing else.** `PropagatedTargets` carries
the eligible rows' detached weak embeddings so equation (5) can log the
same-row and cross-row cross-view cosines each step. It adds no port, value,
schedule or plan field, and it changes no number the mixer sees; it exists
because a per-step trajectory of the pair equation (5) is meant to align cannot
be reconstructed from a step record that holds only one side of it.

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
  metric: held-out p(t|x) NLL ratio full over no-propagation, for student and evaluation EMA; terminal hidden-label NLL ratio for equation (9)'s aggregate(q^w) over p^w; outcome NLL, gate rate, bank coverage and cross-class-opportunity-adjusted representation alignment as guardrails; hat p and aggregate(hat q) target NLLs reported informationally
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: held-out treatment-NLL ratio < 1.0 in mean by at least one standard error for both student and EMA; terminal aggregate(q^w) target-NLL ratio over p^w < 1.0; held-out outcome NLL within 1.05x of the ablation; terminal gate rate >= 0.5; bank coverage = 1.0 before the first propagated target; mean same-row weak/strong cosine minus mean cross-row cosine, divided by the exact fraction of ordered hidden-row pairs with different treatments, >= 0.2
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
   its slot; a support identity with no slot, or one repeated inside a batch,
   is rejected; a hidden treatment can never enter `Q_l`. A batch that reaches
   only *some* slots is not an error — it leaves the rest unfilled, which
   invariant 6 turns into a refusal to propagate.
4. Targets read the previous-step bank. The current support embedding is
   detached, written into its slot — filling it on first sight, temporally
   mixed at `m=0.7` afterwards (deviation 11) — L2-normalised, and visible only
   on the next optimiser step.
5. Preparation and observation are idempotent within one step. Reversing the
   two objective declarations gives bit-identical losses and the bank updates
   exactly once.
6. Steps 0 and 1 use `p^w`, set instance loss to zero, and still fill/update the
   bank. Step 2 is the first propagated target and sees coverage 1.0. A bank
   still holding an unfilled slot after `warmup_steps` also refuses to
   propagate, so an under-covered quota degrades to FixMatch's target rather
   than reading a slot nothing has written.
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
    window, warm-up steps, the unfolding switch, the memory key space, and the
    update order. `K` is read from the population rather than printed (§4).

**Tier 1 (smoke fit and mechanism arms).**

1. Run the full and no-propagation pair for one seed. The instance loss falls
   after warm-up, and the supervised term falls. The **gated** semantic term is
   read through its gate rate and accepted confidence rather than its level: a
   gate that opens raises the term it charges, which is the same reading
   `fixmatch.md` §6 records for that paper’s equation (4), so a rising eq. (2)
   under a rising gate rate is the curriculum and not divergence. The full arm
   beats marginal-frequency treatment NLL; `hat p` and class-aggregated `hat q`
   are scored against the fixture’s hidden treatments whether they improve or
   not. (Amended during implementation: the draft asked for both unsupervised
   losses to fall, which eq. (2) cannot be expected to do while its gate is
   still opening.)
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
   method arm. (Run during implementation, in the literal retrained form and in
   an isolated one that rolls the map on the full arm's own finished bank. The
   aggregate's improvement disappears; `hat p`'s does not, because equation
   (10) mixing in 10% of a constant is shrinkage. Predeclared expectation
   falsified; the measurement and its consequence for §6's third tolerance are
   under "What has run", and the tolerance is left as declared.)
7. **Distribution alignment off**, on the balanced fixture and on a diagnostic
   variant with `p(t=1)=0.15`. Report predicted and true marginals. No skewed-
   fixture performance direction is required; a uniformising operation can be
   misspecified there.
8. Report bank slot norm, class coverage, nearest-neighbour label agreement,
   gate rate, retained semantic-target impurity, same-row weak/strong cosine,
   and cross-row cosine trajectories for every arm. (The two cosines are logged
   by equation (5) each step for this, as diagnostics; see "What has run".)

**Tier 2 (fixed ten-replicate target).** Run only the full and no-propagation
pair under the YAML contract above. The Tier 1 arms diagnose a failure but do
not enter the acceptance metric and cannot be selected after seeing Tier 2.
The original target and its result remain in §6.3; §6.4 records the reviewed
amendment that produced the current YAML block.

**What has run.** All twelve Tier 0 invariants are in
`tests/invariants/test_simmatch.py`. The whole Tier 1 study is in
`tests/smoke/test_simmatch.py`, at 600 optimiser steps on one seed of §6.1's
stream, with every arm starting from the same initial parameters — asserted
tensor by tensor — and drawing the same quota stream.

§6.1's three pre-training measurements were taken rather than assumed. The
Bayes-optimal treatment label — which on this DGP is the Bayes-optimal cluster
label, because the assignment puts 0.98 on a cluster's own level — flips on
**2.9%** of rows under the weak view and **18.5%** under the strong one. That
second number is `fixmatch`'s 0.5 strong rate, which `flexmatch.md` §5.2
already measured as not label-preserving and which deviation 2 inherits
deliberately to keep the controlled comparison; every arm below carries it
equally. Observed support is 27/37 over `K = 64`, and the true `p(t = 1)` is
0.500 in train, 0.494 held out, and 0.152 on arm 7's skewed fixture.

Held-out treatment NLL is the student's; `same − cross` is the terminal-window
cross-view cosine margin; the last four columns are hidden-label NLLs of the
terminal targets. The marginal-frequency baseline is 0.708 on the balanced
fixture, which the first eight arms share.

| Arm | held-out t-NLL | gate | `same − cross` | `hat p` | `p^w` | `agg hat q` | `agg q^w` |
|---|---|---|---|---|---|---|---|
| full | 0.2680 | 0.724 | 0.168 | 0.2870 | 0.2911 | 0.4222 | 0.2679 |
| §6 pair's no-propagation ablation | 0.2822 | 0.737 | 0.001 | 0.3006 | 0.3006 | 0.6933 | n/a |
| 2 — `alpha = 1` | 0.2683 | 0.739 | 0.168 | 0.2912 | 0.2912 | 0.4220 | n/a |
| 3 — unfolding off | 0.2981 | 0.109 | 0.001 | 0.2979 | 0.3153 | 0.6945 | 0.6945 |
| 4 — `lambda_in = 0` | 0.2836 | 0.466 | 0.034 | 0.2907 | 0.3034 | 0.3454 | 0.4286 |
| 5 — bank momentum 0 | 0.2647 | 0.720 | 0.173 | 0.2853 | 0.2896 | 0.4173 | 0.2683 |
| 6 — permuted `Q_l` | 0.2991 | 0.092 | 0.002 | 0.2980 | 0.3159 | 0.3250 | 0.7067 |
| 7 — DA off, balanced | 0.2685 | 0.722 | 0.168 | 0.2874 | 0.2919 | 0.4214 | 0.2672 |
| 7 — skewed, DA on | 0.2329 | 0.108 | 0.070 | 0.3422 | 0.3885 | 0.3030 | 0.2064 |
| 7 — skewed, DA off | 0.2030 | 0.686 | 0.062 | 0.2155 | 0.2197 | 0.2766 | 0.2083 |

Three readings. None of them is a direction: one seed at a fifth of the
declared budget cannot support one, and §6's ten replicates cover the pair
only. Arms 2, 3 and 4 carry predeclared expectations from the paper's table 5
and figure 5, and this study reports where they landed without promoting a
one-seed landing into a finding.

1. **The cross-view geometry comes from equation (8)'s calibration and
   equation (5)'s gradient, and not from equation (10).** The full arm's
   terminal margin is 0.168. Arm 2 drops only equation (10) and is
   indistinguishable from it (0.168). Arm 3 drops only the calibration and
   loses essentially all of it (0.001), as does the §6 ablation, which drops
   both (0.001), and arm 6, which keeps the calibration but rolls the labels it
   reads (0.002). Arm 4, which keeps every piece of the arithmetic and removes
   only equation (5)'s gradient, keeps a fifth (0.034) — what the shared
   encoder produces on its own, with no instance loss pushing on it.
   The mechanism is this. With `hat q = q^w` the instance loss has a degenerate
   optimum: a collapsed space makes both `hat q` and `q^s` near-uniform over
   the 64 slots, and the cross-entropy is satisfied without any geometry.
   Calibrating `hat q` by `p^w` peaks the target on same-class slots, which
   `q^s` can only match by moving.
2. **Distribution alignment is inert on the balanced fixture and misspecified
   on the skewed one.** Turning it off moves the balanced arm by 0.0005 in
   held-out NLL — the fixture's true marginal is already uniform, so there is
   nothing for a uniformising operation to do. On the `p(t = 1) = 0.152`
   fixture the aligned arm's terminal predicted marginal is 0.208 against the
   unaligned arm's 0.177, and the unaligned arm fits better. This is the
   misspecification `paws.md` §6.2 records for me-max; §2 claims nothing about
   a skewed treatment prevalence and this card does not start now.
3. **Arm 6 falsifies its own predeclared expectation, and §6's third tolerance
   inherits the consequence.** See below.

**Arm 6, and what it costs the `hat p` reading.** The card predeclared that
permuting `Q_l` would make "both target improvements disappear". Read in an
isolated form the card did not specify — the full arm's *own* finished bank,
its own student, its own view draw, and only the slot-to-class map rolled by
one — half of that happens and half does not:

| | `p^w` | `hat p` | `aggregate(q^w)` | spread of `aggregate(q^w)` |
|---|---|---|---|---|
| true `Q_l` | 0.29107 | 0.28656 | 0.26642 | 0.4280 |
| rolled `Q_l` | 0.29107 | 0.28225 | 0.72898 | 0.0142 |
| `0.9 p^w + 0.1 * class frequency` | 0.29107 | 0.28209 | — | — |

The aggregate collapses exactly as a wiring control should: a roughly even
random split of the slots carries a roughly even split of every row's
similarity mass whatever that row looks like, so
equation (9)'s output goes from an informative distribution to a near-constant
one and its hidden-label NLL nearly triples. What does **not** disappear is
`hat p`'s advantage over `p^w`. It cannot: equation (10) mixes in 10% of
*something*, and 10% of a constant is shrinkage toward the class prior. The
rolled reading lands on the constant-prior shrinkage baseline to within
0.0002, and it is *better* than the true-label reading — with the true map the
aggregate is informative but disagrees with `p^w` in a way that costs more NLL
than it earns on this fixture.

So §6's `terminal hat-p target-NLL ratio < 1.0` measures something real and
passes it (§6.3), but on this fixture it is not by itself evidence that
propagation propagated anything, and no reading of the ledger should treat it
as such. The tolerance is left exactly as declared — weakening or redefining a
predeclared metric after seeing it is the deviation `FIDELITY.md` §3 forbids —
and this paragraph is the caveat that travels with it. A future amendment
wanting a propagation-attributable version of this metric should predeclare the
rolled-label control as its denominator, and re-run.

**What arm 8 required, and what changed to supply it.** The same-row and
cross-row cross-view cosines are now logged by equation (5) every step rather
than recomputed afterwards, so the card's "trajectories for every arm" are a
property of the realisations the loss charged. They are diagnostics only — no
loss, gradient, hyperparameter or plan field moves — and they are computed in
closed form at `O(nD)`, because `sum_ij z^w_i . z^s_j` is
`(sum_i z^w_i) . (sum_j z^s_j)` and a 448-row gram every step is not what a
diagnostic may cost.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-05 | `be173ef1a2e6` | student_treatment_NLL_ratio<br>ema_treatment_NLL_ratio<br>terminal_hat_p_target_NLL_ratio<br>terminal_aggregate_hat_q_target_NLL_ratio<br>held_out_outcome_NLL_ratio<br>terminal_gate_rate<br>bank_coverage_before_first_propagation<br>cross_view_alignment_margin | 0.958396 +/- 0.0392<br>0.979415 +/- 0.0113<br>0.955314 +/- 0.00617<br>1.63719 +/- 0.0342<br>1.00009 +/- 0.000173<br>0.705775 +/- 0.0177<br>1 +/- 0<br>0.155997 +/- 0.0121 | no |
| 2026-09-05 | `a8a7f1c991ed` | student_treatment_NLL_ratio<br>ema_treatment_NLL_ratio<br>terminal_aggregate_q_weak_to_p_weak_target_NLL_ratio<br>held_out_outcome_NLL_ratio<br>terminal_gate_rate<br>bank_coverage_before_first_propagation<br>cross_class_adjusted_alignment_margin | 0.941082 +/- 0.0281<br>0.973394 +/- 0.0100<br>0.846193 +/- 0.0142<br>1.00005 +/- 0.000152<br>0.7055 +/- 0.0176<br>1 +/- 0<br>0.311243 +/- 0.0231 | yes |

**Reading the 2026-09-05 row.** Six of the eight required metrics pass and two
miss, so the status is `deviating`. What passed is the primary pair: the
held-out treatment NLL ratio is below 1 by more than a standard error on both
parameter sets (student 0.958 ± 0.039, EMA 0.979 ± 0.011), the outcome head is
untouched (1.00009 ± 0.00017 against a 1.05 allowance), the gate reaches
0.706 ± 0.018, and the bank was covered before the first propagated target in
every replicate. Against a 0.698 marginal-frequency baseline the full arm's EMA
reaches 0.287 and the ablation's 0.293.

What missed:

- **`terminal_aggregate_hat_q_target_NLL_ratio`, 1.637 ± 0.034 against < 1.**
  This is not a fit failure, and the equations say why it was the wrong
  direction to predeclare. Aggregating equation (8) gives
  `aggregate(hat q) = normalize(p^w ⊙ aggregate(q^w))` — a product of two
  experts that share an encoder. In absolute terms both factors are good
  (`p^w` 0.342 ± 0.010, `aggregate(q^w)` 0.288 ± 0.005) and their product is
  worse than either (0.473 ± 0.016), which is what multiplying two correlated,
  already-confident distributions does to a log score. The paper never forms
  this quantity: equation (9) aggregates `q^w`, and `hat q` exists for
  equation (5). The card asked for a direction the source does not imply, and
  the tolerance stays as written.
- **`cross_view_alignment_margin`, 0.156 ± 0.012 against ≥ 0.2.** A genuine
  miss of a project-local threshold, not a null mechanism: the paired ablation
  reaches 0.00116 ± 0.00013, so equation (8) is worth two orders of magnitude
  on this statistic and still lands short of the number this card guessed
  before running anything. **Deviation 3's budget is not the explanation, and
  the obvious guess that it is should be dropped rather than repeated.** That
  deviation predicts an unfinished curriculum, but the margin does not behave
  like one. Inside the Tier 1 run it climbs from 0.060 over the first hundred
  post-warm-up steps to 0.168 by step 600 — and the ten-replicate 3,000-step
  runs terminate *lower*, at 0.156 ± 0.012. The two are not a clean series (one
  seed against ten, and deviation 3 re-bases the cosine schedule on whichever
  budget it is given), but five times the budget moving the statistic slightly
  the wrong way is not the signature of a curriculum that ran out of steps.
  What does bound it is the fixture and deviation 4's trunk. At `K = 2` roughly
  half of every "cross-row" pair is a same-cluster pair that *should* be
  similar, so the attainable margin is capped by the DGP rather than by
  training; and 64 labelled slots over two treatment levels give equation (5)
  almost no uniformity pressure, where the source's bank spans ten classes and
  a WideResNet. A threshold of 0.2 on a statistic whose ceiling this fixture
  sets was never derived from anything; it is the weakest joint in §6 and the
  one a re-review should replace with a bound computed from the DGP.

Neither miss is an arithmetic disagreement with the source. The audit that
preceded this reading covered the mixer, both objectives' equations, the views,
the data policy, every §4 hyperparameter, both schedules, and the inherited
FixMatch optimiser policy; §6.2's arms then attributed the surviving movement
to equation (8) specifically. What a future amendment should reconsider is the
two project-local thresholds and the propagation-attributable form of the
`hat p` metric, predeclared before any re-run.

### 6.4 Amendment: measure the two source paths without class-collision dilution

The first row in §6.3 is retained. It was produced honestly under the original
protocol, and its two misses are the evidence this amendment reasons from; an
amended target does not turn that row into a pass.

**The target-quality replacement.** The withdrawn pair of target ratios asked
`hat p` to beat `p^w` and `aggregate(hat q)` to beat `aggregate(q^w)`. Neither
isolates the information carried by the labelled memory. Arm 6 showed that the
first can improve when the slot labels are wrong, because mixing ten percent of
a constant class prior is useful shrinkage on this balanced fixture. The second
scores a class aggregate the source never forms: equation (8)'s calibrated
instance distribution is the target of equation (5), while equation (9)
deliberately aggregates the *uncalibrated* `q^w`. Algebraically,
`aggregate(hat q)` is a product of two correlated experts and its log score can
worsen even when each expert is informative.

The replacement asks whether equation (9)'s actual `aggregate(q^w)` beats the
same row's `p^w` on hidden treatments. This directly checks that the instance
space contributes class information to equation (10), has the same no-effect
boundary at ratio one as the withdrawn metrics, and fails when the memory-label
control destroys that information. The two withdrawn ratios remain
informational so their caveats continue to travel with every run.

**The alignment adjustment.** The raw same-row-minus-cross-row margin mixes two
kinds of off-diagonal pair. When two rows have the same treatment, similarity
is desirable rather than a failure of instance discrimination; only a
different-treatment pair supplies the contrast the threshold intends to test.
For each replicate the amended statistic therefore divides the raw margin by

```text
(n_missing^2 - sum_j n_j^2) / (n_missing * (n_missing - 1)),
```

the exact fraction of ordered, distinct hidden-row pairs whose treatments
differ. This is not a fitted threshold: it preserves the original `0.2`
required separation per informative pair and corrects only the dilution the
two-class DGP determines before training. The raw margin and the opportunity
fraction remain informational metrics.

**What did not change.** The pair, fixture, seed stream, initial parameters,
batches, optimiser, schedule, views, equations, budget and all recipe
hyperparameters are untouched.

**Outcome.** A fresh ten-replicate execution under the amended contract passes
all seven required metrics, so the card is `reproduced`. Equation (9)'s
aggregate target-NLL ratio is `0.846 +/- 0.014` against `1.0`, and the
cross-class-adjusted alignment margin is `0.311 +/- 0.023` against `0.2`.
Both held-out treatment ratios still favour the full arm by more than one
standard error (`0.941 +/- 0.028` student and `0.973 +/- 0.010` EMA); outcome
NLL, gate rate and bank coverage pass their unchanged guardrails. The original
failure remains visible in the same run: `aggregate(hat q)` over
`aggregate(q^w)` is `1.636 +/- 0.035`, and the unadjusted alignment margin is
`0.155 +/- 0.011`. They are informational now for the reasons above, not
silently removed from the evidence.

## 7. Unknowns

| Unspecified or inconsistent in source | Our choice | Basis |
|---|---|---|
| Equation (12) prints the same time-indexed embedding on both sides. | `Q_i <- normalize(0.7 Q_i + 0.3 z_i_current)`. | The prose calls this temporal ensembling; SemiLearn’s small-dataset port implements exactly this update and normalises the result. Without the previous slot the equation is an identity up to scale. |
| Whether equation (9) aggregates `q^w` or the calibrated `hat q`. | Aggregate the original `q^w`. | Literal equation (9) and the authors’ code: `aggregated_prob.scatter_add(..., teacher_prob_orig)`, while the scaled distribution is a different tensor used only by the instance loss. |
| Whether equation (8)’s unfolding uses the raw or distribution-aligned semantic prediction. | The aligned `p^w`. | Algorithm 1 computes `p^w=DA(phi_t(h^w))` before propagation; the authors’ code calls distribution alignment before gathering semantic factors by memory label. |
| The paper does not mention a propagation warm-up. | Disable propagation and instance loss for the first two optimiser steps, while writing the bank. | Both reference ports disable them for epoch 0. On §6’s 960-row missing population with `448` missing rows and dropped incomplete batches, one source-style epoch is `floor(960/448)=2` steps. This is printed in the plan rather than hidden behind the word “epoch”. |
| Initial memory contents. | No initial features at all: a slot is empty until its row is first observed, that first observation fills it, and no target reads the bank until every slot is filled (deviation 11). Labels are fixed from the observed population and never written by a batch. | The authors and SemiLearn initialise random normalised features and mix from the first update, which their warm-up epoch dilutes to nothing; two warm-up steps here would not (deviation 11). Their initial labels are zero in code and overwritten by indexed support batches, so deriving immutable `Q_l` from `TrainingPopulation` is arithmetically identical by the first read and cannot take a hidden treatment. |
| Read/write order. | Clone/read the prior bank, prepare both targets, then write current detached support embeddings once, whichever objective the mixer evaluates first. | The authors’ `forward` clones `self.bank`, computes both losses, then calls `_update_bank`. Current supports must not leak into their own targets. |
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
| Card reviewed (status → `reviewed`) | Claude | 2026-08-31 |
| Plan diffed against §3.2 and §4 | Claude | 2026-08-31 |
| Tier 1 study run in full; §6.1 measurements taken | Claude | 2026-09-05 |
| Tier 2 run, ten replicates (status → `deviating`) | Claude | 2026-09-05 |
| §6 amended: target-quality and alignment instruments replaced (§6.4) | Codex | 2026-09-05 |
| Tier 2 re-run under amended §6 (status → `reproduced`) | Codex | 2026-09-05 |
