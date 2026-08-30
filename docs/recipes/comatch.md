# Recipe spec card: comatch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [CoMatch: Semi-supervised Learning with Contrastive Graph Regularization](https://arxiv.org/abs/2011.11183) |
| Authors, year | Junnan Li, Caiming Xiong, Steven C.H. Hoi; 2021 (ICCV 2021) |
| DOI / arXiv | [arXiv:2011.11183](https://arxiv.org/abs/2011.11183); ICCV 2021 |
| Version used | arXiv v2, 2021-03-03, read through the ar5iv HTML rendering. The paper's §3.1 gives eqs. (3)–(5); §3.2 gives eqs. (6)–(12) and the memory bank; §3.3 gives the EMA/momentum-queue variant, eq. (13); §4.1 gives the CIFAR-10 and STL-10 protocol and results; §4.3 the ablations; appendix A table 5 the full hyperparameter set; appendix C algorithm 1 the per-iteration pseudo-code. Section references in this card are to *this* card unless prefixed "the paper's". |
| Reference implementation | [`salesforce/CoMatch`](https://github.com/salesforce/CoMatch) @ `a64ccf5af11017a1a07267b21e5899f4e8157801`, read directly at that commit: `Train_CoMatch.py` (`train_one_epoch`, `main`), `WideResNet.py` (the projection head), `datasets/cifar.py` (`load_data_train`, the three transforms) and `utils.py` (`WarmupCosineLrScheduler`). Every §7 row marked "reference implementation" names the file it comes from. The CIFAR entry point is the one this card follows; `imagenet/Train_CoMatch.py` implements the §3.3 variant and is cited only where the two differ. |
| Reference impl. runnable? | Not attempted — it trains a WideResNet-28-2 on CIFAR-10 for 512 epochs and nothing in this card depends on running it. |

## 2. Estimand and claim

- **Estimand:** categorical treatment probabilities `p(t | x)` and treatment-specific outcome means for causal contrasts, fitted in one joint stage in which the label head and the embedding head train each other.
- **Method claim:** a pseudo-label refined by its neighbours *in the embedding space* (eq. 8) is more accurate than the classifier's own prediction, and a contrastive loss whose target graph is built from those pseudo-labels (eqs. 9–11) learns embeddings that are useful for the classification task rather than merely instance-discriminative. The two representations "interact with each other and jointly evolve". Evidence: CIFAR-10 at 40 labels, 93.09 ± 1.39 against FixMatch-with-DA's 86.98 ± 3.40 (the paper's table 1); ImageNet at 1% labels, 66.0 top-1 against 53.4 (table 2); ablations at 100 ImageNet epochs from a 57.1% default — `T = 1` (self-loops only, i.e. ordinary instance discrimination) costs 2.8 points, and `alpha = 1` (no memory smoothing, i.e. mean-teacher-style pseudo-labelling) costs 2.1 (figure 4a, 4c).
- **Not claimed:**
  - **No published number is reproduced.** Every result above is image classification with `K = 10` or `K = 1000` classes; no dataset in the paper carries a treatment. §6 is a project-local mechanism target measured against a `fixmatch` arm, which is the paper's own baseline.
  - **Nothing about representation transfer.** The paper's tables 3 and 4 (VOC07, Places205, COCO) are about a CNN backbone transferring to other vision tasks. This card fits one stack on one fixture and evaluates it there.
  - **Nothing about open-set SSL.** The paper argues out-of-distribution rows get low-confidence pseudo-labels and are pushed away by the contrastive term. The §6.1 fixture has no OOD population and the claim is not tested (`BACKLOG.md` §18.4).
  - **No inference claim for the pseudo-labelled rows.** A memory-smoothed soft label is a training signal, not an identified treatment (`BACKLOG.md` §7.4).
  - **Nothing about the graph at large `K`.** With `K = 2` treatments, `q_b · q_j` is a function of two scalars near the simplex edge, so the pseudo-label graph's structure is much coarser than a 10-class one: at `T = 0.8` an edge means "both rows confidently agree". §6.2 predeclares edge density and purity as measurements precisely because the mechanism's headroom here is smaller than the paper's, and a null result must be readable as *that* rather than as a wiring failure.

## 3. Equations and mapping

### 3.1 As published

Let `f` be the encoder, `h` the classification head with `p(y|x) = h(f(x))`, and
`g` the projection head with `z(x) = g(f(x))` normalised to the unit sphere.
A step draws `X = {(x_b, y_b)}_{b=1..B}` labelled and `U = {u_b}_{b=1..mu B}`
unlabelled rows (the paper's §3.1).

The three losses are

$$
\mathcal{L}_x = \frac{1}{B}\sum_{b=1}^{B}
\mathrm{H}\big(y_b,\ p(y \mid \mathrm{Aug_w}(x_b))\big)
\tag{3}
$$

$$
\mathcal{L}_u^{cls} = \frac{1}{\mu B}\sum_{b=1}^{\mu B}
\mathbb{1}(\max q_b \ge \tau)\,
\mathrm{H}\big(q_b,\ p(y \mid \mathrm{Aug_s}(u_b))\big)
\tag{4}
$$

$$
\mathcal{L} = \mathcal{L}_x
+ \lambda_{cls}\mathcal{L}_u^{cls}
+ \lambda_{ctr}\mathcal{L}_u^{ctr}
\tag{5}
$$

**Memory-smoothed pseudo-labelling.** Every row of `X` and `U` contributes a
class probability — `p^w = y` for a labelled row, `p^w = h∘f(Aug_w(u))` for an
unlabelled one — and an embedding `z^w = g∘f(Aug_w(·))`. Unlabelled
probabilities first pass distribution alignment, `p^w = Normalize(p^w / p̃^w)`
against a moving average `p̃^w` of `p^w`, which "prevents the model's prediction
from collapsing to certain classes". A memory bank
`MB = {(p_k^w, z_k^w)}_{k=1..K}` holds the last `K` weakly-augmented rows,
labelled and unlabelled alike, first-in-first-out. The pseudo-label minimises

$$
J(q_b) = (1-\alpha)\sum_{k=1}^{K} a_k \lVert q_b - p^w_k \rVert_2^2
+ \alpha \lVert q_b - p^w_b \rVert_2^2
\tag{6}
$$

$$
a_k = \frac{\exp(z^w_b \cdot z^w_k / t)}{\sum_{k=1}^{K}\exp(z^w_b \cdot z^w_k / t)}
\tag{7}
$$

and because `a` is normalised the minimiser is available in closed form:

$$
q_b = \alpha p^w_b + (1-\alpha)\sum_{k=1}^{K} a_k p^w_k
\tag{8}
$$

**Graph-based contrastive learning.** The pseudo-label graph and the embedding
graph over the `mu B` unlabelled rows, with `z_b`, `z'_b` the embeddings of two
strong augmentations:

$$
W^q_{bj} = \begin{cases}
1 & b = j\\
q_b \cdot q_j & b \ne j \text{ and } q_b \cdot q_j \ge T\\
0 & \text{otherwise}
\end{cases}
\tag{9}
$$

$$
W^z_{bj} = \begin{cases}
\exp(z_b \cdot z'_b / t) & b = j\\
\exp(z_b \cdot z_j / t) & b \ne j
\end{cases}
\tag{10}
$$

Both are row-normalised, `Ŵ_{bj} = W_{bj} / sum_j W_{bj}`, and the loss is the
row-wise cross-entropy between them:

$$
\mathcal{L}_u^{ctr} = \frac{1}{\mu B}\sum_{b=1}^{\mu B}
\mathrm{H}\big(\hat{W}^q_b,\ \hat{W}^z_b\big)
\tag{11}
$$

which decomposes into a self-loop term — "a self-supervised contrastive loss …
a form of consistency regularization" — and an off-diagonal term that "gathers
samples from the same class into clusters, which achieves entropy
minimization" (eq. 12).

The paper's §3.3 adds, *for scale only*, an EMA model
`θ̄ ← m θ̄ + (1-m) θ` (eq. 13) and a momentum queue of `K` strongly-augmented
EMA embeddings, so that `W^q` and `W^z` become `mu B × K` and a large graph
fits on eight GPUs. Deviation 5 records why this card implements §3.2 and not
§3.3.

Six statements are load-bearing for the mapping, and each is a place a
reimplementation silently differs. Four come from the reference implementation
rather than the paper, and are quoted as code.

- **The unlabelled population is disjoint from the labelled one.** `load_data_train`
  splits each class into `inds_x, inds_u = indices[:n_labels], indices[n_labels:]`
  and builds the two loaders over the two halves. This is the opposite of
  FixMatch's footnote 2, which puts every labelled row into `U` as well, and it
  is why §4's `eligible_rows` reads `t_missing` where `fixmatch.md` reads `all`.
- **The pseudo-label is soft.** "Different from FixMatch, our soft pseudo-labels
  `q_b` are not converted to hard labels for entropy minimization. Instead, we
  achieve entropy minimization by optimizing the contrastive loss." In code,
  `loss_u = -sum(log_softmax(logits_u_s0) * probs, dim=1) * mask`: the target is
  the full smoothed distribution and the `arg max` appears only inside the gate.
- **The bank is read before it is written.** `Train_CoMatch.py` computes `A`,
  then `probs`, then `scores/mask`, and only then writes
  `queue_feats[ptr:ptr+n] = feats_w`. A row's own entry is therefore never in
  its own affinity, and the affinity is over the previous `K` rows — up to five
  steps old at the CIFAR setting, with no EMA smoothing them.
- **Smoothing is warmed up.** `if epoch>0 or it>args.queue_batch` gates eq. (8):
  for the first `queue_batch = 5` steps of training, `q_b = p^w_b` and the
  method is FixMatch-with-DA and soft labels. Nothing in the paper says this.
- **Distribution alignment is a 32-batch moving average, not an EMA.**
  `prob_list.append(probs.mean(0)); if len(prob_list)>32: prob_list.pop(0)`, then
  `prob_avg = stack(prob_list).mean(0)`. The current batch is inside its own
  denominator. The bank stores `probs_orig`, the aligned but *unsmoothed*
  probabilities, which is what eq. (6)'s `p^w_k` says and worth stating because
  storing `q` instead would make the smoothing recursive.
- **Both strong embeddings carry gradient.** `sim = exp(feats_u_s0 @ feats_u_s1.T / t)`
  has no `detach` on either side; the only stop-gradient in the contrastive term
  is the whole of `Q`, which is built inside `torch.no_grad()`. There is no
  "target branch" here in the BYOL sense — the paper's figure 2 marks `sg` on
  the pseudo-label path, not on `z'`.

Defaults, from appendix A table 5 and `Train_CoMatch.py`'s argument parser
(CIFAR-10; the ImageNet column appears in §7 where it differs): `B = 64`,
`mu = 7`, `lambda_cls = 1`, `lambda_ctr = 1`, `alpha = 0.9`, `K = 2560`,
`t = 0.2`, `tau = 0.95`, `T = 0.8`, a WideResNet-28-2 trunk with a 2-layer
projection head to 64 dimensions, SGD with Nesterov momentum 0.9 and weight
decay `5e-4` excluding parameters whose name carries `bn`, learning rate 0.03
under `cos(7 pi k / 16 K)` with no warmup, 512 epochs of 1,024 iterations, and an
evaluation-only EMA at `m = 0.999`.

### 3.2 Mapping to xty2

One `joint_fit` stage. CoMatch is not a pretrain-then-fine-tune method: the
classification head and the projection head train in the same step and each
supplies the other's target, which is the "co-training" the name is about, and
the reason this card has no `initialise_from` edge where `scarf.md` and
`paws.md` do. The "classes" of the paper are the levels of `t`, so `h` is the
reviewed `CategoricalPropensity` and eq. (3) is `ObservedTreatmentNLL`.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `f` | encoder | `X_RAW -> X_REPR` | `MLPEncoder` (the reviewed P5 backbone; deviation 6) |
| `h` | classification head, `p(y\|x)` | `X_REPR -> T_GIVEN_X` | `CategoricalPropensity` (deviation 1: the "classes" are treatments) |
| `g` | projection head, `z = g(f(x))` | `X_REPR -> X_PROJ` | `ProjectionHead(widths=(200, 64), activation="leaky_relu", normalisation="row_l2")` (§5.1; deviation 6) |
| `Aug_w` | weak augmentation | `weak_x` | `ViewSpec("weak_x", transforms=(FeatureMask(p=0.1),), draws=1)` — `fixmatch.md`'s reviewed weak view (deviation 2) |
| `Aug_s`, `Aug'_s` | the two strong augmentations | `strong_x @ draw=0`, `strong_x @ draw=1` | `ViewSpec("strong_x", transforms=(FeatureMask(p=0.1), FeatureMask(p=0.5)), draws=2)` (deviations 2, 3) |
| `p^w_b` (unlabelled) | aligned weak prediction | `T_GIVEN_X @ weak_x` | read by `MemorySmoothedPseudoLabelTreatmentNLL`, detached |
| `p^w_b = y_b` (labelled) | the ground-truth row of the bank | — | `batch.t` one-hot at `label_rows="t_observed"`; the second declared row population `paws.md` §5.1 introduced |
| `z^w` | weak embedding | `X_PROJ @ weak_x` | the same objective, detached; written to the bank for both populations |
| `DA(·)`, `p̃^w` | distribution alignment | — | the 32-batch moving average inside `MemorySmoothedLabels` state |
| `MB`, `K` | memory bank | — | `MemorySmoothedLabels` state: FIFO of `(p^w, z^w)`, capacity 2,560 (§5.1) |
| `a_k`, `t` | affinity, eq. (7) | — | `MemorySmoothedLabelGraph(temperature=0.2, alpha=0.9, capacity=2560, edge_threshold=0.8, alignment_window=32, warmup_batches=5)`, the shared value object §5.1 introduces |
| `q_b`, eq. (8) | the memory-smoothed pseudo-label | — | computed once per step by that objective; the sibling reads it through `context.objective_state` (`DESIGN.md` §4) |
| eq. (3) | labelled cross-entropy | `T_GIVEN_X @ weak_x` | `ObservedTreatmentNLL(realisation=weak_x)`, rows `t_observed`, `reduction="mean"` |
| eq. (4) | gated soft pseudo-label loss | `T_GIVEN_X @ strong_x draw=0` | `MemorySmoothedPseudoLabelTreatmentNLL(threshold=0.95, sharpening="none", stop_grad="target")`, rows `t_missing`, `reduction="mean"` |
| `W^q`, eq. (9), `T` | pseudo-label graph | — | inside `PseudoLabelGraphContrastive`, from the sibling's `q`; detached by construction |
| `W^z`, eq. (10) | embedding graph | `X_PROJ @ strong_x draw=0,1` | the same objective; both realisations train |
| eqs. (11), (12) | graph cross-entropy | `X_PROJ @ strong_x draw=0,1` | `PseudoLabelGraphContrastive(owner=<the objective above>)`, rows `t_missing`, `reduction="mean"`, `batch_coupled=True` |
| eq. (13), the momentum queue | the §3.3 scale variant | — | not implemented; deviation 5 |
| `m = 0.999` | the evaluation EMA | — | `TeacherSpec(decay=0.999, role="evaluation")`, as `fixmatch.md` §5.5 |
| `B`, `mu` | batch composition | — | `QuotaSampler(Quota("t_observed", 64), Quota("t_missing", 448))`; both card keys derived, never asserted |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed`, `reduction="population"` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing`, ramped weight |

Three rows of that table are the whole reason this card is not another `%Match`
variant, and a reviewer scanning the YAML will look for them there and not find
them.

- **The label target is a function of the embeddings.** Eq. (8) is the only
  place in this repository where a pseudo-label depends on `X_PROJ`. Every
  other gated method (`fixmatch`, `flexmatch`, `freematch`, `doublematch`)
  derives its target from `T_GIVEN_X` alone, and `doublematch` — the nearest
  shipped card — adds a representation term that *coexists* with the label term
  rather than feeding it. The controlled difference §6 measures is exactly this
  edge.
- **The contrastive target is a function of the labels.** Eq. (9) is the
  reverse edge. `InfoNCEContrastive` is not reused: its target is the identity
  matrix, which is eq. (9) at `T = 1`, and the paper reports that
  configuration as a 2.8-point *loss*. The `T = 1` arm in §6.2 is therefore
  both an ablation of CoMatch and a matched comparison against `scarf`'s
  objective on this fixture.
- **There is no `arg max` anywhere in the loss.** `losses.sharpening` is
  `none`: the gate keeps or drops a row, and what is charged is the full
  distribution `q_b`. This is the first xty2 objective to do that, and it is
  the paper's stated reason for needing the contrastive term at all — entropy
  minimisation moves from the target to `L_u^ctr`.

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: p(t|x) @ view=weak_x; x_proj @ view=weak_x; the whole memory bank   # ref impl: the block is inside torch.no_grad()
    joint_fit.pseudo_label_graph_contrastive: q, and therefore W^q, read from the sibling state   # both strong embeddings train; ref impl detaches neither side of `sim`
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets:
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: target       # eq. (4): q_b is a constant of theta
    joint_fit.pseudo_label_graph_contrastive: target                   # eq. (11): W^q is a constant of theta; `detaches` is empty because no *required* realisation is detached (§5.1)
  gradient_clipping:
    joint_fit: none                                                    # paper and ref impl name none
  marginal_nll_grad_path:
    joint_fit.missing_treatment_marginal_nll: both                     # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.999                                                     # ref impl `--ema-m`; evaluation only (deviation 5)
  ema_applies_to_buffers: false                                        # ref impl `ema_model_update` EMAs parameters and *copies* buffers, so norm statistics are never shadowed
  teacher_in_train_mode: false                                         # `ema_model.eval()` at construction
  teacher_requires_grad: false                                         # `param_k.requires_grad = False`
  # role = evaluation. No objective reads it: eq. (8) reads the current model
  # (deviation 5). The compiler rejects an objective that takes it as a target.

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: mean                             # eq. (3) divides by B, the labelled batch
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: mean         # eq. (4) divides by mu*B, including the rows the gate rejected
    joint_fit.pseudo_label_graph_contrastive: mean                     # eq. (11) divides by mu*B
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed                       # eq. (3)
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: t_missing    # U is disjoint from X (ref impl `load_data_train`) - not `all` as in fixmatch
    joint_fit.pseudo_label_graph_contrastive: t_missing                # the graph is mu*B x mu*B over the same rows
    joint_fit.missing_treatment_marginal_nll: t_missing
    # The bank additionally *writes* t_observed rows (their one-hot t and weak
    # embedding), declared as `label_rows` rather than by widening the two
    # eligible sets - the second row population of `paws.md` §5.1.
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0                              # eq. (5): L_x enters with coefficient 1
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: 1.0          # lambda_cls, table 5
    joint_fit.pseudo_label_graph_contrastive: 1.0                      # lambda_ctr, table 5 (CIFAR-10; 5 for STL-10, 10 for ImageNet 1%)
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: constant 1.0 # the paper ramps neither unsupervised weight
    joint_fit.pseudo_label_graph_contrastive: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature:
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: 0.2          # t, eq. (7); table 5
    joint_fit.pseudo_label_graph_contrastive: 0.2                      # t, eq. (10) - the same value object, so the two cannot drift (§5.1)
  sharpening:
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: none         # §3.1: the soft pseudo-label is not converted to a hard label and is not temperature-sharpened
    joint_fit.pseudo_label_graph_contrastive: n/a
  confidence_threshold:
    joint_fit.memory_smoothed_pseudo_label_treatment_nll: 0.95         # tau, eq. (4); table 5
    joint_fit.pseudo_label_graph_contrastive: 0.8                      # T, eq. (9): the edge threshold on q_b . q_j; table 5, ablated in figure 4a

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)                          # ref impl `main`: SGD(..., momentum=0.9, nesterov=True)
  lr: 0.03                                                             # §4.1
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))       # `WarmupCosineLrScheduler` with warmup_iter=0: cos(7 pi k / 16 K), K = our 3000 steps (deviation 7)
  weight_decay: 0.0005 (all trainable components; norm and bias exempt) # §4.1; ref impl puts parameters whose name carries 'bn' in a zero-decay group
  batch_size: 512                                                      # B + mu B = 64 + 448, derived from the QuotaSampler's quotas
  labelled_unlabelled_ratio: 7.0                                       # mu, table 5; derived from the same quotas, never asserted
  total_steps_or_epochs: 3000                                          # optimiser steps, never epochs; the paper runs 512 * 1024 (deviation 7). The cosine schedule uses this K.

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                                       # retained reviewed P5 TARNet backbone; deviation 6
    projection_head: [200, 64]                                         # ref impl `WideResnet.__init__`: fc1 keeps the trunk width, fc2 maps to low_dim = 64
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K                         # the paper's h is "a fully-connected layer followed by softmax"
  activation:
    mlp_encoder: elu
    projection_head: leaky_relu                                        # ref impl: nn.LeakyReLU(negative_slope=0.1) between fc1 and fc2, nothing after fc2 (§5.1)
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: row_l2
    projection_head: row_l2                                            # ref impl `Normalize(2)` on the head output; eqs. (7) and (10) are dot products of unit vectors
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0                                                   # perturbation comes from the three explicit input views
    projection_head: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    projection_head: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    projection_head: 64-d embedding, l2-normalised                     # ref impl `low_dim = 64` for CIFAR (128 for ImageNet)
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits                           # one head, read under the weak and the first strong view

data:
  standardisation: x: none fitted on 'train'                           # the §6.1 DGP draws standardised features
  outcome_scaling: y: zscore fitted on 'train'                         # held-out rows take the same fitted transform, never a refitted one
  treatment_encoding: n/a                                              # XTYBatch supplies integer classes 0..K-1; the bank one-hots the observed t itself
  split_protocol: one fixed project-local DGP, split train/test by the §6.1 fixture; no CIFAR-10, STL-10 or ImageNet protocol applies (deviation 9); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id   # §6.1, and deviation 4 for what bounds it
```

**On the four paper-governed values the vocabulary has no key for.**
`alpha = 0.9` (eq. 8), the bank capacity `K = 2560` (§3.2), the alignment
window of 32 batches and the 5-batch smoothing warm-up are governed by the
source and bound by no key in `FIDELITY.md` §2. Adding keys for them is a
framework change (`DESIGN.md` §9.1), and two of the four would not survive the
argument for one: the alignment window and the warm-up are
reference-implementation constants that appear in no equation and no table. They take the route `DESIGN.md` §4 already provides: they are
constructor arguments of the shared value object with no defaults, and
`plan_details()` prints all four, so they enter the plan digest and the §5.1
review surface. `label_smoothing` in `paws.md` §4 is the precedent.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Predict the *treatment* rather than an image class, and add the reviewed xty2 causal stack (outcome NLL, exact marginalisation over missing `t`) to eqs. (3)–(5). | `p(t \| x)` is a categorical classifier over `K` levels, so `h` maps exactly; the paper's downstream task is the classification head itself, which is the head this project needs for a propensity. The outcome terms are the project's, not the paper's, and they are held identical to the `fixmatch` arm so the CoMatch mechanism stays attributable. `paws.md` §5.1 and `fixmatch.md` §5.1 make the same call. | No published number applies. The §6 comparison is internal and paired against `fixmatch` — the paper's own baseline — on one fixture, one seed stream and one batch stream. |
| 2 | `judgement` | — | Replace crop/flip with `FeatureMask(0.1)` as `Aug_w`, and RandAugment/SimCLR colour distortion with `FeatureMask(0.1) + FeatureMask(0.5)` as both `Aug_s` and `Aug'_s`, taken as two independent draws of one `ViewSpec`. | There is no image geometry in a tabular XTY batch. The weak/strong pair is `fixmatch.md`'s reviewed one, unchanged, because §6 compares against that card and a difference in views would confound the comparison it exists to make. Using one strong *family* for both strong views is the paper's own ImageNet configuration — "We use the same strong augmentation for `Aug_s` and `Aug'_s`" — so the CIFAR-only asymmetry of RandAugment against colour distortion is the part that is dropped, and it is dropped by deviation 3. | Defines what invariance eq. (11)'s self-loop term learns. Two draws of one family make the self-loop easier than the paper's CIFAR pairing and the off-diagonal term relatively more load-bearing. §6.2 predeclares the Bayes-label flip rate of both views, measured before training. |
| 3 | `framework-limitation` | `augmentation-vocabulary` | No adaptive or compositional augmentation: strong-view strength is a fixed pair of mask rates, where the reference runs RandAugment(2, 10) for `Aug_s` and a SimCLR crop/jitter/grayscale stack for `Aug'_s`. | Both are image-operation vocabularies with magnitudes; neither has a tabular meaning, and xty2 has no operation vocabulary to sample from (`DESIGN.md` §11.4). `fixmatch.md` §5.10 and `doublematch.md` §5.6 pay for the same absence, and this card's dependence is stronger than theirs, because CoMatch's contrastive term consumes the *difference* between two strong views rather than only their distance from the weak one. | Unknown sign. A strong view that is too weak makes the self-loop term trivial and the graph term dominate; too strong and the pseudo-label the graph is built from degrades first. §6.2 measures the flip rate and the terminal edge density so the failure mode is identifiable rather than inferred. |
| 4 | `framework-limitation` | `batch-row-repetition` | Set the §6 label budget to 64 rather than the paper's 40-label CIFAR-10 regime, holding `B = 64` and `mu = 7` at the paper's values. | The paper's headline gain is largest exactly where labels are scarcest (table 1: +6.11 at 40 labels, +1.68 at 80). The reference iterates the labelled set as an endlessly repeating shuffle, so a step sees some labelled rows twice; `XTYBatch.row_id` must be unique because artifacts and provenance are keyed by it (`DESIGN.md` §7.1), so the scarcest budget expressible here is `B` itself. `fixmatch.md` §5.12 is the same row for the same reason, which is what makes the two arms comparable. | Measured in the least favourable regime for the mechanism. A null result at 64 labels is not evidence against the paper's 40-label claim, and §6 states its target in those terms. |
| 5 | `judgement` | — | Implement §3.2 — the in-batch `mu B × mu B` graph, with the memory bank read from the current model — and not §3.3's EMA model and `mu B × K` momentum queue. Keep the reference's evaluation-only EMA at `m = 0.999`. | §3.3 opens by stating its own purpose: a batch large enough to contain several rows per class "would exceed the memory capacity of 8 commodity GPUs", so the EMA and the queue exist to make `K = 30000` affordable at 1,000 classes. At `K = 2` treatments and 448 unlabelled rows per step, the in-batch graph is the paper's own small-data configuration and the CIFAR entry point of the reference implements it exactly. Adopting §3.3 anyway would add an EMA into the *loss* — a second published mechanic, with its own decay to tune — for a scaling problem this fixture does not have. | The EMA is a smoother of the pseudo-label path; the paper's figure 4c shows `alpha = 1` (pure EMA prediction, no smoothing) is 2.1 points worse, not better, so the omitted mechanic is not the one carrying its gain. `mean_teacher.md` is the shipped card for a teacher-smoothed target if the §6.2 diagnostics implicate target noise. |
| 6 | `judgement` | — | Retain the P5 encoder, outcome head and propensity (3 × 200 ELU, `row_l2`; `K` heads of 3 × 100; linear propensity) rather than WideResNet-28-2. Take the projection head's shape from the reference — 2 layers, hidden width = trunk width, 64-d `l2`-normalised output, LeakyReLU(0.1) — and give it no BatchNorm. | Holding the causal stack fixed is what makes an addition attributable, and is the decision `fixmatch.md` §5.6, `scarf.md` §5.3 and `paws.md` §5.5 record. The head is not part of that stack, so its shape is the reference's. BatchNorm is the one part that is not: this recipe plans four forward passes over a batch of two row populations, and `fixmatch.md` §5.11 records that per-realisation passes equal the reference's interleaved pass "only while no component holds batch-coupled state". Both CoMatch objectives are already batch-coupled through the graph; adding a batch-coupled *component* would make the loss depend on quota composition as well. | The embedding geometry is ours. Since eqs. (7) and (10) normalise both sides, the head's normalisation affects the gradient scale reaching it rather than the loss (`ProjectionHead`'s docstring measures this for SCARF); §6.2's collapse guardrail is what would show it going wrong. |
| 7 | `judgement` | — | Train for 3,000 optimiser steps rather than 512 epochs of 1,024 iterations, with the cosine schedule's `K` re-based on the same 3,000 so the *shape* of `cos(7 pi k / 16 K)` is exact. | Every card here fixes a project-local step budget so a difference between recipes is attributable to the recipe, and §6's target is a paired comparison in which both arms get the same budget either way. `fixmatch.md` §5.3 makes the identical move with the identical schedule, which is what lets the two arms share a learning-rate trajectory exactly. | The paper's own selling point is efficiency — it trains CoMatch for half the baseline's epochs — but 3,000 steps is a different regime from 524,288, and the natural curriculum §3.2 describes (a sparse graph densifying as confidence rises) may not have completed. §6.2 reports the edge-density trajectory so a truncated curriculum is visible rather than assumed away. |
| 8 | `judgement` | — | Keep the reference's bank *formula*, `K = queue_batch * (mu + 1) * B = 5 * 8 * 64 = 2560`, which lands on the paper's published `K = 2560` at our batch shape — even though our training population is 1,024 rows, so the bank holds roughly two and a half copies of it, where the paper's holds 5% of CIFAR-10. | The alternative is to rescale `K` to a fixed fraction of the population (about 51 rows), which changes the number the paper published and the neighbourhood size eq. (7) averages over, to preserve a ratio the paper never names. The reference derives `K` from the batch, not from the dataset, and five steps of history is what `queue_batch = 5` means. Keeping the formula keeps both the published constant and its stated derivation; the population ratio is a fixture consequence and is measured rather than hidden. | The affinity in eq. (7) will frequently include *other draws of the same row*, up to five steps stale. That is a softer smoothing than the paper's, biased toward each row's own recent prediction — which is `alpha` by another route. §6.2 predeclares a `K = 512` (one step of history) arm to bound how much of any effect this is. |
| 9 | `judgement` | — | One fixed project-local DGP (§6.1); no CIFAR-10, STL-10 or ImageNet protocol, no label-fraction splits, no top-1 accuracy. | The paper's evidence is three image benchmarks and none carries a treatment. Reproducing that shape is a question about data plumbing, not about whether the mechanism is assembled correctly. | §6 is a mechanism target and says so. It is evidence against this port being miswired, not for the paper's claim. |

### 5.1 Framework additions made for this card

Six additions. Three are reversible objects of the kind §11.2 says to build for
one card; one is load-bearing and carries a named second consumer; two are
one-line widenings of existing closed vocabularies. Three things this recipe
needs are **not** additions and are listed after the table, because each is a
place an earlier card designed against exactly this one.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `MemorySmoothedLabels` — objective state holding the FIFO bank of `(p^w, z^w)` and the 32-batch alignment window, owned by `MemorySmoothedPseudoLabelTreatmentNLL` | fidelity-bearing, **load-bearing vocabulary** (it is the first state that stores *per-row tensors from earlier batches*, which is what `BACKLOG.md` §15.4 asks a contract for) | `MemorySmoothedPseudoLabelTreatmentNLL`, and `PseudoLabelGraphContrastive` through the sibling read | **SimMatch** (`BACKLOG.md` §2.9), whose labelled and unlabelled memory banks are written once per step and read by both its instance-similarity and its semantic-similarity terms | Eq. (8) sums over the last `K` rows, so no arrangement of ports, realisations or row populations computes it from one batch — the same argument `flexmatch.md` made for a per-class counter, one step further. **Shape check against SimMatch**, in the two places it could have gone wrong: (i) SimMatch's banks are keyed by *instance*, so the contract stores an opaque row-major block with an insertion pointer and no `row_id` index — a bank keyed by `row_id` would have fitted CoMatch and broken SimMatch's labelled bank, which holds one entry per class prototype; (ii) SimMatch reads its bank from *both* terms, so the update must be idempotent within a step exactly as `SelfAdaptiveThresholds.observe` is, and for the same reason: declaration order must not change the loss. §7 records the seven lifecycle answers `BACKLOG.md` §15.4 requires. |
| `MemorySmoothedLabelGraph` — a frozen value object holding `(temperature, alpha, capacity, edge_threshold, alignment_window, warmup_batches)`, passed to both objectives | fidelity-bearing, reversible | both `joint_fit` unsupervised objectives | not required (reversible) | Eq. (7)'s affinity and eq. (10)'s embedding graph are the *same* `t` in the source (`args.temperature`), and `losses.temperature` is keyed by `<stage>.<objective>`, so two objectives each carrying their own literal could declare two values and `DESIGN.md` §9.1's "conflicting values in one scope are errors" would not catch it. One shared object is the only thing that makes them provably equal, and it is where the four keyless values of §4 live. `paws.md` §5.1's `SupportSetClassifier` is the precedent and the same argument. |
| `MemorySmoothedPseudoLabelTreatmentNLL` — eq. (4) with a soft, memory-smoothed target | fidelity-bearing, reversible | this card | not required (reversible) | `PseudoLabelTreatmentNLL` takes its target from a declared realisation and hardens it. Neither half survives here: the target is a function of the bank as well as of `T_GIVEN_X @ weak_x`, and it stays soft. Widening the existing objective would put a conditional inside `compute` and a `q`-source union in its `requires`, which is the shape `CLAUDE.md` rule 3 forbids in recipes and `DESIGN.md` §4 discourages in objectives. |
| `PseudoLabelGraphContrastive` — eqs. (9)–(11) | fidelity-bearing, reversible | this card | not required (reversible) | `InfoNCEContrastive` is eq. (11) with `W^q = I`, which the paper's figure 4a reports as a 2.8-point regression, so the graph is the mechanic rather than a decoration on one. Keeping the two objects separate is what makes the `T = 1` arm in §6.2 a *matched* comparison between them rather than a flag. |
| `Sharpening` gains the value `"none"` | fidelity-bearing, reversible | this card | not required (reversible) | The literal is `Literal["hard"]` today, and its own docstring anticipates a second value arriving with a card that needs one. `none` is the paper's post-processing in table-1 terms: neither pseudo-labelling nor sharpening. `losses.sharpening: none` is a *stated* value, distinct from `n/a`, which is what stops the card cross-check reading "soft target" as "key not applicable". |
| `ProjectionHead.activation` gains `"leaky_relu"` (slope 0.1) | fidelity-bearing, reversible | this card | not required (reversible) | The reference's head is `Linear -> LeakyReLU(0.1) -> Linear -> l2`. Substituting ReLU would be a deviation row against a mechanic we can express in one line, which `FIDELITY.md` §4.1 names directly ("do not disguise it as a permanent deviation to keep a diff small"). The slope is a constructor argument with no default, printed in the plan. `paws.md` keeps `relu` because its reference uses ReLU; the two cards now say what their sources say. |

**Three things that are not additions.**

- **The second row population.** The bank writes `t_observed` rows (their
  one-hot `t` and weak embedding) while the two objectives are eligible over
  `t_missing`. That is `paws.md` §5.1's `support_rows` field, and this card is
  the second consumer it was designed against. The shape survives the move: the
  field names a *population*, not a batch slice, and here it is read for a
  write-side effect rather than for a similarity denominator.
- **The sibling state read.** `PseudoLabelGraphContrastive` names the objective
  that owns `q` and reads it through `context.objective_state`, which is
  `DESIGN.md` §4's sentence and `freematch.md` §5.1's addition. That card named
  SimMatch as its second consumer and CoMatch as its third; this is the third,
  and the shape holds unchanged — including the idempotency requirement, which
  here also fixes *read-before-write* ordering, since eq. (8) must see the bank
  as of the previous step whichever objective the mixer reaches first.
- **`X_PROJ`.** `scarf.md` §5.1 named CoMatch as the port's second consumer,
  against "a projected embedding that a similarity graph is built over". This
  is that consumer, and the port needed no change.

## 6. Reproduction target

The pair compares CoMatch against `fixmatch` on a fixed project-local DGP,
holding the fixture, the label budget, the causal stack, the optimiser, the
schedule, the views, the seeds and the batch stream identical. FixMatch is the
paper's own baseline (its table 1 and table 2), so the arm is the comparison the
source makes, transplanted to a fixture where the classes are treatments. The
paper's own ablation axes — `T`, `alpha`, `lambda_ctr`, `K` — are carried as
mechanism guardrails so that a null result can be attributed to a component
rather than to the port as a whole.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in 6.1
  variant: paired fit against the shipped fixmatch recipe, same seeds, same fixture, same batches, same optimiser and schedule
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio, comatch over fixmatch, reported for the student and for the evaluation EMA; terminal mask rate, retained-label impurity, mean edges per row and embedding alignment as mechanism guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: treatment-NLL ratio < 1.0 in mean by at least one standard error, on the EMA and the student alike; held-out outcome NLL within 1.05x of the fixmatch arm; terminal mask rate at least 0.5; mean edges per row strictly between 1.0 (self-loops only) and 0.5 * mu * B; mean cosine similarity of a row to its own second strong view at least 0.2 above its mean similarity to the other unlabelled rows of the batch
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

Use `fixmatch.md` §6.1's generator, seed streams and 64-label MCAR budget
unchanged, so the two arms differ only in the recipe. Both arms draw the same
paired `Quota` stream — `B = 64` observed and `mu B = 448` missing rows per
step — fit outcome standardisation on the complete training population, and
evaluate on the same 2,048 held-out rows with every treatment observed.

Two fixture facts this recipe adds, both checked rather than assumed:

- **The bank holds more rows than the population has.** `K = 2560` against 1,024
  training rows (deviation 8). The fixture asserts the ratio rather than
  ignoring it, and §6.2 reports the mean number of bank entries that are earlier
  draws of the anchor's own row — the quantity that separates "smoothing over
  neighbours" from "smoothing over my own past".
- **The graph must not be trivially dense or trivially empty.** At `K = 2` and
  `T = 0.8` an edge means both rows confidently agree. A run whose terminal edge
  count per row is 1.0 has learnt nothing the self-loop did not already say, and
  one whose count approaches `mu B / 2` has collapsed to two blocks; the
  tolerance above brackets both, and the trajectory is reported either way.

### 6.2 Predeclared evidence

The criteria below are fixed while the card is `draft`. Nothing is implemented,
so §6.3 is empty and stays empty until Tier 0 and Tier 1 pass.

**Tier 0 (invariants).**

1. Eq. (8) reads the bank as of the previous step: an anchor's own entry is
   never in its own affinity, asserted by stepping a hand-built bank twice.
2. The bank is FIFO with capacity `K`: after `ceil(K / 512) + 1` steps it holds
   exactly `K` rows, the oldest are gone, and the pointer wraps once.
3. The shared update is idempotent within a step, and the loss is bit-identical
   under both declaration orders of the two objectives — `freematch.md`'s
   assertion, re-run for a bank rather than for three scalars.
4. `q` carries no gradient, both strong embeddings do, and `W^q`'s rows sum to
   1 wherever the row has any edge.
5. The gate counts rejected rows in the denominator: a batch where nothing
   clears `tau` returns exactly zero, not a mean over an empty set.
6. At `T = 1`, `W^q` is the identity and `PseudoLabelGraphContrastive` equals
   `InfoNCEContrastive` on the same two realisations up to the additive `log n`
   that objective's `1/n` normaliser contributes, and up to the `1e-7` log
   floor. This is the assertion that makes §6.2's arm 6 a matched comparison
   against `scarf`'s objective rather than a claim about one.
7. At `alpha = 1`, `q` equals the aligned weak prediction exactly, for any bank
   contents.
8. Distribution alignment leaves a uniform batch mean unchanged and is
   idempotent on a stationary stream.
9. Both objectives declare `batch_coupled=True`, and a stage holding either is
   refused `ExternalBatches`; the stateful objective is refused `cross_fit`.
10. `plan.hyperparameters` matches every non-`n/a` key of §4, and
    `plan_details()` prints `alpha`, `K`, the alignment window and the warm-up.
11. The state is fresh per stage execution: two runs of one compiled recipe are
    identical, and a paired arm cannot inherit the other's bank.

**Tier 1 (smoke fit).**

1. **Before training:** the Bayes-optimal treatment label's flip rate under the
   weak and the strong view on the fixture (`BACKLOG.md` §6), reported for
   both. A measurement, not an assertion; a flip rate that makes the strong
   view uninformative is a card amendment.
2. All three CoMatch losses fall; the mask rate rises; the retained pseudo-label
   impurity, scored against the fixture's hidden `t`, does not rise.
3. Held-out treatment NLL beats the marginal-frequency baseline and the paired
   `fixmatch` arm's, at one seed, as a wiring check only.
4. **`lambda_ctr = 0`.** The graph term off, everything else held. Isolates
   memory smoothing from graph regularisation. Predeclared expectation from
   figure 4b: worse, and the gap is the graph's contribution.
5. **`alpha = 1`.** No smoothing; eq. (8) collapses to the aligned weak
   prediction and the recipe becomes soft-label FixMatch-with-DA plus a graph
   term. Predeclared expectation from figure 4c: worse. This arm and arm 4
   together decide whether either edge of the co-training loop is load-bearing
   at `K = 2`.
6. **`T = 1`.** Self-loops only; the contrastive term becomes instance
   discrimination and the recipe becomes "FixMatch + SCARF's objective".
   Predeclared expectation from figure 4a: worse than `T = 0.8`. This is the
   arm that tests `BACKLOG.md` §1's question directly — whether class structure
   and instance structure cooperate — and it is reported whichever way it goes.
7. **`K = 512`.** One step of history rather than five, bounding deviation 8.
8. **Distribution alignment off.** The paper's baseline table separates FixMatch
   from FixMatch-with-DA by 0.9 points at 40 labels; on a fixture whose true
   `p(t = 1) ≈ 0.5` the alignment target is nearly correct, so this arm mostly
   measures whether DA is doing anything here at all. Report the terminal
   predicted marginal with and without.
9. **A skewed-propensity fixture**, `p(t = 1) = 0.15`, run for arm 8 only.
   Distribution alignment divides by a running marginal, so where the true
   marginal is far from uniform it is the same misspecification `paws.md` §6.2
   found for me-max. This card predeclares the measurement and refuses the
   claim in §2 rather than asserting a direction.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | — | — |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Eq. (10)'s off-diagonal is `exp(z_b · z_j / t)` — the *same* strong view on both sides — but the code computes `exp(feats_u_s0 @ feats_u_s1.T / t)`, whose `(b, j)` entry is `z_b · z'_j`, the other strong view. | The code: the whole matrix is cross-view, so the diagonal and the off-diagonal come from one product and the graph is not symmetric. | **Reference implementation:** `Train_CoMatch.py`, `sim = torch.exp(torch.mm(feats_u_s0, feats_u_s1.t())/args.temperature)`. The equation's diagonal case already reads `z_b · z'_b`, so the code's reading makes eq. (10) one expression instead of two, and it is the arithmetic that produced the published numbers. Recorded because it changes what the off-diagonal term pushes apart — augmented copies rather than anchors. |
| Whether memory smoothing runs from the first step. | No: `q_b = p^w_b` for the first 5 optimiser steps, then eq. (8). | **Reference implementation:** `if epoch>0 or it>args.queue_batch`, with `queue_batch = 5`. The bank is zero-initialised, so an ungated eq. (7) would take a softmax over 2,560 zero vectors and mix in a uniform average of zero probabilities. Keeping the gate is fidelity; recording it matters more here than in the paper's 512-epoch run, because 5 of our 3,000 steps is a larger fraction than 5 of 524,288. |
| What "a moving-average `p̃^w`" means precisely. | The unweighted mean of the last 32 batch means of `p^w`, including the current batch. | **Reference implementation:** `prob_list.append(probs.mean(0)); if len(prob_list)>32: prob_list.pop(0)`. Not an EMA, and not a running mean over all history. The window length 32 appears in no equation and in no table. |
| Whether the bank stores the pseudo-label `q` or the aligned prediction `p^w`. | `p^w` — aligned, unsmoothed — for unlabelled rows, and the one-hot label for labelled rows. | **The paper** (`MB = {(p_k^w, z_k^w)}`) and the **reference implementation** (`probs_w = torch.cat([probs_orig, onehot])`) agree. Recorded because storing `q` would make eq. (8) recursive across steps, which is a different estimator with the same equation number. |
| Whether the labelled rows are in `U` for eq. (4), as FixMatch's footnote 2 has them. | No. `L_u^cls` and `L_u^ctr` are over `t_missing` only; labelled rows enter the bank and eq. (3) and nothing else. | **Reference implementation:** `load_data_train` partitions each class into a labelled and an unlabelled half and builds two loaders. This is the single largest structural difference from `fixmatch.md` §4, where the same key reads `all`. |
| The seven lifecycle questions `BACKLOG.md` §15.4 asks of a bank: key space, capacity, eviction, update order, device, reset, checkpoint. | Key space: none — an opaque row-major block, insertion-ordered, no `row_id` index. Capacity: 2,560 rows of `(K probs, 64-d embedding)`. Eviction: FIFO by pointer, wrapping. Update order: read for eq. (8), then write the whole step's `B + mu B` rows. Device: the batch's. Reset: fresh per stage execution, zero-initialised, with the 5-step warm-up standing in for "not yet full". Checkpoint: never — the state is not an artifact. | `DESIGN.md` §4 and the reference implementation, which stores the queue in its epoch checkpoint but restores nothing from it (`Train_CoMatch.py` saves `queue` in `save_obj` and `main` always constructs a fresh one). The paper is silent on all seven. |
| Whether the memory bank is `K` *rows* or `K` *steps*. | `K = 2560` rows, derived as `queue_batch * (mu + 1) * B` and carrying 5 steps at this batch shape. | **Reference implementation:** `args.queue_size = args.queue_batch*(args.mu+1)*args.batchsize`, which reproduces the paper's published `K = 2560` exactly. Deviation 8 records what that means against a 1,024-row population. |
| The `1e-7` inside `log(sim_probs + 1e-7)`: it is in the code and in no equation. | Keep it, and print it in `plan_details()`. | **Reference implementation:** `loss_contrast = - (torch.log(sim_probs + 1e-7) * Q).sum(1)`. At `t = 0.2` an orthogonal pair contributes `exp(0) = 1` against a diagonal near `exp(5)`, so row-normalised entries reach `1e-3`–`1e-4` routinely and the floor is inert on ordinary rows — but it is the arithmetic that produced the numbers, and `LOG_FLOOR` already exists in `xty2/objectives/adaptive_threshold.py` for the same reason. |
| Whether `W^q`'s self-loop is set before or after thresholding. | Before, and unconditionally: the diagonal is 1 whatever `T` is. | **The paper's** eq. (9) case split and the **reference's** `Q.fill_diagonal_(1)` after the matrix product and before `pos_mask`. This is why `T = 1` degrades to instance discrimination rather than to an empty graph. |
| The projection head's width when the trunk is not a WideResNet. | Hidden width = the trunk's output width (200), output 64. | **Reference implementation:** `fc1 = nn.Linear(64*k, 64*k)`, `fc2 = nn.Linear(64*k, low_dim)` — the hidden layer *is* the trunk width by construction, so the ratio rather than the number is what transfers. `low_dim = 64` is the CIFAR value; ImageNet uses 128. |
| Whether weight decay reaches the projection head. | Yes; it is exempt only for norm and bias parameters. | **Reference implementation:** the optimiser splits on `'bn' in name`, so every matrix including `fc1`/`fc2` is decayed. `fixmatch.md` §5.8 settled the same question the same way. |
| Whether `lambda_ctr = 1` transfers, given that the paper varies it by dataset (1 for CIFAR-10, 5 for STL-10, 10 for ImageNet 1%, 2 for 10%) and says "fewer labeled samples require a larger `lambda_ctr`". | 1 — the CIFAR-10 value, unswept. | The CIFAR-10 column of table 5 is the small-data configuration this card follows throughout, and sweeping a weight on a project-local fixture would be tuning on the thing §6 measures. §6.2 arm 4 measures the term's contribution at weight 1 and at 0; a sweep, if one is ever justified, needs its own predeclared protocol and a split that is not §6's. |
| Whether the evaluation EMA or the student is the reported model. | Both, as `fixmatch.md` §6 does. | The paper reports the EMA on CIFAR-10 and STL-10 and the student on ImageNet, "for fair comparison with baselines". Reporting one of the two would make the choice look like a result. |
| No published target applies to a tabular causal adaptation. | A seed-locked project-local mechanism target with a paired `fixmatch` arm and the paper's own four ablation axes as guardrails. | The same discipline as `fixmatch.md` §6, `scarf.md` §6 and `paws.md` §6: predeclare the DGP, the pairing and the tolerance before running anything, and make the mechanism — not a borrowed number — the thing that can fail. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
