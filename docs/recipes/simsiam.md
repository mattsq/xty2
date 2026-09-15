# Recipe spec card: simsiam

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 for benchmark and reporting work.

Selected from BACKLOG.md §5.1. Implementation was requested after review of
PR #55. The callable, component topology and paired protocol implement this
card; section 6 records the measured result. The 2026-09-15 fidelity re-audit
withdrew deviations 3 and 7, replaced section 6.4's instrument and reran
section 6 on a fresh seed stream; section 5 keeps both withdrawn rows and
section 6.1 keeps the failed row they explain.

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566v1) |
| Authors, year | Xinlei Chen and Kaiming He, 2020; subsequently CVPR 2021 |
| DOI / arXiv | arXiv:2011.10566 |
| Version used | v1, 2020-11-20; equations (1)–(4), Algorithm 1, §3 and §4.1–§4.2 |
| Reference implementation | [facebookresearch/simsiam](https://github.com/facebookresearch/simsiam/tree/a7bc1772896d0dad0806c51f0bb6f3b16d290468) @ `a7bc1772896d0dad0806c51f0bb6f3b16d290468` |
| Reference impl. runnable? | Not attempted; inspected `simsiam/builder.py:SimSiam` and `main_simsiam.py:train, adjust_learning_rate`. |

Source links: [architecture and detach](https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/simsiam/builder.py),
[loss and optimisation](https://github.com/facebookresearch/simsiam/blob/a7bc1772896d0dad0806c51f0bb6f3b16d290468/main_simsiam.py).
Source-code symbol names, rather than moving line numbers, identify the mapping.

## 2. Estimand and claim

- **Estimand:** paired effects of target stop-gradient and a learned predictor
  on how close the optimiser gets to the cosine's trivial optimum and on the
  transferred representation's rank, followed by the effect of pretraining
  on held-out factual outcome NLL in a missing-treatment problem.
- **Published claim:** stop-gradient prevents collapse in the paper's image
  experiments without negative pairs or a momentum encoder; the predictor
  ablation also fails. This is empirical evidence, not a universal guarantee.
- **Local claim to test:** removing either mechanic lets the optimiser reach the
  loss's degenerate solution and costs the transferred representation rank,
  while the full method does neither and does not materially harm downstream
  outcome fit.
- **Not claimed:** ImageNet accuracy reproduction, superiority to VICReg,
  Barlow Twins or SCARF, general-purpose tabular augmentation, or improved causal
  identification. Treatment-effect RMSE is diagnostic, not established by NLL.
  Nor is any absolute rank claimed: this fixture's invariant content under
  section 6.2's views is at most five-dimensional, so a rank of a few is what a
  working representation looks like here and a large one would be random
  features rather than structure. The bounds are paired for that reason.

The nearest architectural baseline is VICReg's two-stage transfer protocol.
The primary controls here change one SimSiam mechanic each. Cross-method
rankings would confound objective, architecture and optimisation choices.

## 3. Equations and mapping

### 3.1 As published

The paper's `f` includes backbone and projection MLP; `h` is the predictor.
For two views, `z_i=f(x_i)`, `p_i=h(z_i)`:

```text
(1) D(p1,z2) = -(p1 / ||p1||_2) · (z2 / ||z2||_2)
(2) L = D(p1,z2)/2 + D(p2,z1)/2
(3) D(p1,stopgrad(z2))
(4) L = D(p1,stopgrad(z2))/2 + D(p2,stopgrad(z1))/2
```

Algorithm 1 averages over rows. The reference `train` uses negative
`CosineSimilarity(dim=1)`, with two means and a factor of one half.
Detach applies to the target of each directional loss, not globally to either
view. Both shared branches train through their prediction paths.

### 3.2 Mapping to xty2

The mapping below is implemented by `xty2.recipes.simsiam`.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| x1,x2 | independent transforms of the same rows | `X_RAW` at `corrupted_a,b` | two explicit `ViewSpec` instances; benchmark supplies existing `OracleSymmetry` |
| backbone in f | transferable representation | `X_RAW -> X_REPR` | `MLPEncoder`, four 256-wide layers |
| projection in f | target embedding z | `X_REPR -> X_PROJ` | `SimSiamProjector`, three 256-wide layers |
| h | prediction p | `X_PROJ -> X_PRED` | `SimSiamPredictor`, 256 -> 64 -> 256 |
| D(p1,sg(z2)) | forward direction | `X_PRED@a, X_PROJ@b` | `CosineFeatureConsistency`, name `simsiam_a_to_b`, weight 0.5 |
| D(p2,sg(z1)) | reverse direction | `X_PRED@b, X_PROJ@a` | same class, name `simsiam_b_to_a`, weight 0.5 |
| downstream backbone | transferred encoder | identity `X_REPR` | `Stage(joint_fit, initialise_from=pretrain)` |
| project-local outcome | factual conditional distribution | `Y_GIVEN_XT` | `TARNetHead`, `ObservedOutcomeNLL` |
| project-local propensity | treatment probabilities | `T_GIVEN_X` | `CategoricalPropensity`, `ObservedTreatmentNLL` |
| project-local missing-t likelihood | exact enumeration | `T_GIVEN_X,Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path=both)` |

Use existing gradient stages, `UniformSampler(128)`, `OptimiserSpec`,
`Constant`, `Ramp`, `DataSpec` and `TrainingPopulation`.
No teacher, queue, historical target, synthetic row or pseudo-label artifact.

Pretraining trains encoder, projector and predictor only. The two losses consume
the same cached view pair. Each averages all eligible rows; the mixer must not
divide by dimension or by batch size again. The scalar cosine objective is
rowwise, but BN makes the forward pass batch-dependent: bind batch size, reject
training batches below two rows, and never use concatenated-view BN in place of
two separate forwards. Each module processes view a before b; every BN buffer
updates twice per step, not once per objective. Stop-gradient leaves BN buffer
updates intact. A frozen teacher is not an equivalent implementation.

Fine-tuning trains the encoder and downstream heads on identity rows only.
The projector and predictor have no downstream forward pass. Copy the encoder's
complete state; retain identical initial downstream head states in all arms.
Start a fresh optimiser, resetting all moments. Preserve actual row IDs, masks,
scaling statistics and downstream batches across paired arms.

The existing cosine uses separate `F.normalize` calls with epsilon 1e-12.
Retain that path explicitly (deviation 6); log vector norms as well as cosine.
Extend its explicit `stop_grad` policy from `target` to `target | none`
for the no-stop-gradient control. `detaches`, computation and plan details must
agree; existing DoubleMatch declarations remain `target`.

That objective requires a norm check before it is pointed at a new pair of
ports, because its gradient carries `1 / ||prediction||`: its first consumer
collapsed under this term at a pre-normalisation representation norm of 0.011,
arriving about ninety times louder than the paper's, and stopped collapsing
only once a different initialisation put that representation back at order 1
(`doublematch.md` deviation 9, evidence in its section 6.2). The same
normalisation was retained across both. The failure does not transfer here, and
the reason is structural rather than incidental. The projector terminates in
non-affine BN, so the detached target carries `||z||` near `sqrt(d) = 16` by
construction, independent of the encoder initialisation in section 4; and two
BN layers sit between the encoder and `||p||`, so an encoder scale like 0.011
cannot reach the cosine the way it did there. The source's output BN carries
that argument (paper section 4.4), so removing or making it affine changes the
scale claim and not only the topology. This is a construction argument, not a
measurement: report realised norms for both ports per arm, and treat the
section 6.4 near-zero fractions as the check that it held.

The 2026-09-15 audit measured that argument and found it sound at `X_PROJ` and
incomplete before it. The terminal non-affine BN does pin the detached target
near `sqrt(d)`, and every arm's near-zero fraction is exactly zero. But the
check read only the objective's two ports, and the encoder scale it declined to
look at is what the projector's *own* BatchNorm layers consume: under the
initialiser section 4 then bound, the representation arrived at `3.8e-4` and the
two hidden layers divided by `sqrt(eps)` rather than by a batch standard
deviation. Deviation 7 is withdrawn and the encoder now takes the paper's own
initialiser, so all three layers normalise — `0.89`, `0.99989` and `0.99991` of
`var/(var+eps)`, with `X_REPR` arriving at `0.57` to `0.73`.
`tests/invariants/test_simsiam.py` asserts that and fails it under the withdrawn
initialiser; the [audit](../experiments/2026-09-15-simsiam-fidelity-audit.md)
carries the rest. The standing rule the episode leaves behind: a norm check on
this objective reads every port between the encoder and the cosine, not only the
two the objective names.

The no-predictor arm uses the same two losses with `X_PROJ` on both sides
and opposite views, retaining stop-gradient. Omit the dead predictor from its
compiled graph/trainables, while preserving all common initial tensors and RNG
streams. Do not approximate this arm with a zero learning rate on a random
predictor, which is a different ablation.

## 4. Mechanics checklist

Source architecture: pinned `SimSiam.__init__`; source loss:
`SimSiam.forward` and `train`. Comments identify project departures.

```yaml
gradients:
  stop_gradients:
    pretrain.simsiam_a_to_b: x_proj @ view=corrupted_b params=student
    pretrain.simsiam_b_to_a: x_proj @ view=corrupted_a params=student
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target
  gradient_clipping:
    pretrain: none
    joint_fit: none
  marginal_nll_grad_path:
    joint_fit.missing_treatment_marginal_nll: both
teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a
losses:
  reduction:
    pretrain.simsiam_a_to_b: mean
    pretrain.simsiam_b_to_a: mean
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    pretrain.simsiam_a_to_b: all
    pretrain.simsiam_b_to_a: all
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    pretrain.simsiam_a_to_b: 0.5
    pretrain.simsiam_b_to_a: 0.5
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    pretrain.simsiam_a_to_b: constant 0.5
    pretrain.simsiam_b_to_a: constant 0.5
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a
optimisation:  # pretrain from the source; joint_fit is deviation 4's local protocol
  optimiser:
    pretrain: sgd(momentum=0.9, nesterov=False)   # paper section 4 baseline settings; `main_simsiam.main_worker`
    joint_fit: adam(betas=(0.9, 0.999), eps=1e-08)
  lr:
    pretrain: 0.025                               # base lr 0.05 * batch/256, the source's linear scaling rule at batch 128
    joint_fit: 0.001
  lr_schedule:
    pretrain: cosine anneal 0.5 * (1 + cos(pi * min(step/1000, 1)))   # `adjust_learning_rate`, its epoch counter reading this card's step horizon (deviation 3)
    joint_fit: constant 1.0
  weight_decay:
    pretrain: 0.0001 (all trainable components; all parameters)       # supplement A: "for all parameter layers, including the BN scales and biases"
    joint_fit: none
  batch_size:
    pretrain: 128                                 # deviation 3; the source's default is 512
    joint_fit: 128
  labelled_unlabelled_ratio: n/a
  total_steps_or_epochs:
    pretrain: 1000
    joint_fit: 3000
architecture:
  widths_depths:  # deviation 1; topology from builder.py
    mlp_encoder: [256, 256, 256, 256]
    simsiam_projector: [256, 256, 256]
    simsiam_predictor: [64, 256]
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: relu
    simsiam_projector: hidden relu; output linear
    simsiam_predictor: hidden relu; output linear
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: none
    simsiam_projector: hidden BN affine=true; output BN affine=false; all eps=1e-5 momentum=0.1 track_running_stats=true
    simsiam_predictor: hidden BN affine=true eps=1e-5 momentum=0.1 track_running_stats=true; output none
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    simsiam_projector: 0.0
    simsiam_predictor: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:  # deviation 7, withdrawn; paper supplement A
    mlp_encoder: torch Linear default Kaiming-uniform
    simsiam_projector: torch Linear reset_parameters; hidden bias=false; final bias initialised then frozen; BN affine weight=1,bias=0,running_mean=0,running_var=1
    simsiam_predictor: torch Linear reset_parameters; hidden bias=false; final bias=true; BN weight=1,bias=0,running_mean=0,running_var=1
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits
data:  # deviations 2 and 4
  standardisation: "x: zscore fitted on 'train'"
  outcome_scaling: "y: zscore fitted on 'train'"
  treatment_encoding: n/a
  split_protocol: fixed two-cluster DGP; disjoint train and held-out populations; no test-based selection; training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 40 labelled rows, keyed by row_id
```

Section 4 specifies the full arm. Control plans must explicitly bind the changed
stop-gradient policy or prediction ports. Constructor-level epsilon and the
exact BN/bias topology must appear in plan details and Tier 0 value checks.
Projector final bias freezing follows the pinned builder, including its initial
value; do not silently copy Barlow Twins' all-bias-free projector.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | | Tabular MLP replaces ResNet-50; projector width 256 replaces 2048 and predictor bottleneck 64 replaces 512. Retain three-layer projector, output non-affine BN, two-layer predictor and 4:1 bottleneck ratio. | Bounded local experiment. The encoder initialisation clause this row used to carry moved to deviation 7, which is now withdrawn: the encoder takes the paper's initialiser and only the widths and backbone family still depart. | Capacity, norms and collapse dynamics can differ; no image score transfers. |
| 2 | `judgement` | | Explicit caller-supplied views; the study imports VICReg's OracleSymmetry. | Preserve analytic propensity and conditional outcome means using known fixture symmetries. These are privileged DGP operations, not learned augmentations. | Tests a valid-view mechanism, not a general tabular policy. |
| 3 | `judgement` | | **Narrowed 2026-09-15.** Pretraining now runs the source's optimiser — SGD, momentum 0.9, base LR 0.05 under its own linear scaling rule (0.025 at batch 128), weight decay 1e-4 on every parameter layer including BN scales and biases, and `adjust_learning_rate`'s half-cosine anneal. What still departs: batch 128 rather than 512, a 1000-step rather than 100-epoch horizon with the anneal re-based on it, single-process float32, and no synchronised BN (one device). | The row previously read "Adam 0.001, constant LR, no decay" and predicted that optimisation might decide whether either ablation collapses. The 2026-09-15 audit ran that miss and found it does: this row and deviation 7 together suppressed the source's ordering, and returning either alone did not restore it, so the implicated half is withdrawn. The remainder is the local compute budget and the single-process scope of `DESIGN.md` section 0. | Batch size and horizon change the anneal's shape per step and the BN batch statistics; paper section 4.3 reports batch 128 costing about 0.8 points on ImageNet, which does not transfer but does say the axis is live. No image score transfers either way. |
| 4 | `judgement` | | Fit XTY heads for 3000 steps, with the explicit supervised and marginal-NLL weights in §4, rather than source frozen image linear evaluation. Train-only scaling and 40 MCAR labels. | Isolate transfer to the repository's task. | Evaluates a different task and estimand; report local bounds only. |
| 5 | `judgement` | | The predictor shares the encoder's annealed rate rather than holding a fixed one. `OptimiserSpec` carries one schedule per stage, and the source's `--fix-pred-lr` puts the predictor in a second parameter group at the constant `init_lr`. | This reproduces the source's **baseline** row rather than its `fix_pred_lr` variant: paper Table 1 reads "baseline lr with cosine decay 67.7" against "(c) lr not decayed 68.1". Running the row the paper's headline number comes from is the better-defined choice, and it is what the framework expresses today. | Table 1(c) puts the two within 0.4 points on ImageNet. Predictor tracking may still differ here; retained as an optimisation limitation of the local claim rather than a mechanic omitted. |
| 6 | `judgement` | | Use existing separate L2 normalisations with epsilon 1e-12 rather than reference nn.CosineSimilarity's default epsilon 1e-8. | Matches paper Algorithm 1's separate normalisation form and existing objective; pins numerical handling instead of silently inheriting it. | Differences near zero norms and in gradients must be tested and reported; ordinary nonzero cosine agrees. |
| 7 | `withdrawn` | | **Withdrawn 2026-09-15, reviewed and implemented.** The encoder took `normal std=0.1/sqrt(fan_in)` inherited from the sibling tabular encoders rather than the paper's own initialiser, and now takes the paper's: the source's convolution and fc layers use the default PyTorch `U(-sqrt(k), sqrt(k))`, `k = 1/fan_in` (supplement A), which is `MLPEncoder`'s `TORCH_LINEAR_INITIALISATION`. | Inherited from `vicreg` and `barlow_twins` without re-deriving it against this paper and this architecture, which is what section 3.2's norm check was for. At four 256-wide layers the inherited constant is `std = 0.00625`, the size supplement A names when it warns that such models "may not converge". | Retained as history. While it stood, `X_REPR` reached the projector at a mean row norm of `3.8e-4` instead of the pooled ResNet feature's order one, and the projector's two hidden BatchNorms performed `7.5e-6` and `0.078` of their declared normalisation against `0.999` at the output layer — paper Table 3(a)/3(b) rather than its 3(c) default, an axis the paper measures at 33 accuracy points. After withdrawal the same three layers measure `0.89`, `0.99989` and `0.99991`, and `X_REPR` arrives at `0.57` to `0.73`. `tests/invariants/test_simsiam.py` asserts the restored regime and fails it under the withdrawn initialiser. |

No source mechanic is omitted because of an unavailable abstraction. The
frozen final projector bias is stored as a buffer: its sampled value and
forward arithmetic match the source, it has no gradient or optimiser slot,
and it travels in the checkpoint. This fits the executor's component-level
trainable-parameter contract without weakening that contract for other recipes.

### 5.1 Framework additions made for this card

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `X_PRED` tensor port, shape [B,d], no normalisation guarantee | Fidelity-bearing, load-bearing | SimSiam | BYOL, BACKLOG.md §5.1; [paper §3.1 and Figure 1](https://arxiv.org/abs/2006.07733), prediction q_theta(z_theta) matched to target projection z_xi. Prediction and projection coexist with backbone features; teacher/student remain realisations. | Both directional losses need z and p while downstream uses backbone features. Existing X_PROJ cannot hold both simultaneously under one realisation. |
| SimSiamProjector and SimSiamPredictor components | Fidelity-bearing, reversible | SimSiam | n/a | Existing projector topologies do not express this source's output BN and predictor bottleneck exactly. |
| Extend CosineFeatureConsistency stop_grad to target or none, explicit at construction | Fidelity-bearing, reversible | SimSiam controls; preserve DoubleMatch target policy | n/a | The paper's no-stop-gradient control must change actual autograd, declared detaches and plan identity together. |
| `CosineAnneal` schedule: `0.5 * (1 + cos(pi * min(step/steps, 1)))`, as `main_simsiam.adjust_learning_rate` computes it | Fidelity-bearing, reversible | `pretrain.lr_schedule` | not required (reversible) | Deviation 3's narrowing needs the source's rate schedule, and `DESIGN.md` section 6 had no curve of this shape: `CosineDecay` is FixMatch's partial turn, which at the midpoint reads 0.707 where this reads 0.5, and `WarmupCosine` carries the same half cosine only behind a warm-up of at least one step, which would shift the horizon and print a warm-up the source has not got. The `lr-schedules` ledger row builds the family "when a reviewed card names one", `DESIGN.md` section 11.2 Q1/Q2 then answer build, and `FIDELITY.md` section 4.1 forbids the alternative of writing a permanent deviation to keep a diff small. This follows `paws.md`'s `WarmupCosine` precedent exactly. |

Rejected mappings: overwriting X_PROJ loses the target; using X_REPR for the
projection loses the transferred backbone; making the predictor an objective
parameter hides trainable component ownership; using a teacher changes the
method; inventing a role or view to mean a network layer misuses Realisation.

X_PRED is registered in DESIGN.md §2 and tensor-port validation, with executable
compilation/lineage checks. No new executor, row population, artifact kind,
shared SSL framework or card-key category is needed.

### 5.2 Tier 2 evidence history

This card has two recorded runs and both stay in the §6.1 ledger. The
2026-09-14 run under the then-current §4 and §6.4 failed both attribution
bounds on every seed; the
[2026-09-15 re-audit](../experiments/2026-09-15-simsiam-fidelity-audit.md)
found the transcription faithful, the instrument unable to express the claim it
scored, and deviations 3 and 7 jointly suppressing the source's ordering. Those
three amendments and the
[rerun that followed them](../experiments/2026-09-15-simsiam-corrected-tier2.md)
are what the current status rests on. The failed row is retained because the
amendments are only legible beside it.

## 6. Reproduction target

This is a project-local mechanism study, not a published-number reproduction.
All bounds were declared before the run they score, on a seed stream disjoint
from every seed any earlier run or audit has touched. Neither an ablation
collapse nor a downstream gain is assumed. What informed the choice of
*statistic* is disclosed in section 6.4; what set each *threshold* is the
statistic's own zero, not a measured value.

```yaml
reproduction:
  dataset: shared two_cluster_population DGP; six features; K=2; OracleSymmetry views
  variant: full versus no-stop-gradient, no-predictor and no-pretraining
  split: 1024 train with 40 observed treatments; 2048 fully observed held-out rows
  metric: two paired view-alignment gaps; two paired encoder effective-rank gaps; factual outcome NLL cost
  published: none - project-local tabular adaptation
  published_source: n/a
  tolerance: all five one-standard-error bounds in section 6.4
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |
| 2026-09-15 | `946601def994` | stop_gradient_alignment_gap<br>predictor_alignment_gap<br>stop_gradient_rank_gap<br>predictor_rank_gap<br>pretraining_outcome_nll_cost | 0.0372959 +/- 0.00256<br>0.0377957 +/- 0.00255<br>1.06106 +/- 0.0998<br>0.226683 +/- 0.164<br>0.00938544 +/- 0.00382 nat/row | yes |
| 2026-09-14 | `c14090a43e2e` | full_projection_spread<br>stop_gradient_spread_gap<br>predictor_spread_gap<br>pretraining_outcome_nll_cost | 0.896711 +/- 0.013111<br>-0.066665 +/- 0.011058<br>-0.078406 +/- 0.012552<br>-0.029322 +/- 0.032332 | no; spread and NLL pass, both attribution gaps fail |

### 6.2 Fixed DGP and paired execution

Import `two_cluster_population`, `continuous_schema`, `training_dataset`
and `on_the_training_scale` from `xty2.evaluation.benchmarks.common`.
Adopt the separated generator with `low=SEPARATED` explicitly, unchanged.
Import `OracleSymmetry` and its preservation tolerance from
`xty2.evaluation.vicreg_views`; do not copy the DGP or transform.
See [VICReg's valid-view contract](vicreg.md#62-fixed-dgp-and-paired-execution).

Tier 2 replicate i=0..9 uses base=520000+100*i: training seed base+1,
held-out seed base+2, model seed base+6, execution seed base+10000.
Use 1024 training rows with offset 0 and 2048 held-out rows with offset 10000.
All outcomes are observed. Fit preprocessing once on training rows and share
the actual MCAR treatment mask across arms and stages; masked treatment values
never train. The shared population uses the downstream execution seed
`base+10000+STREAM_STRIDE` for its one MCAR draw. The study checks every arm's
data policy before supplying this fitted object to the existing loader boundary.

Run four arms: full, both target detaches removed, predictor removed, and no
pretraining. All pretraining arms get 1000 steps and all downstream arms get
3000 steps. No-pretraining removes the stage and inheritance edge; use the
existing STREAM_STRIDE offset so its downstream stream matches stage index 1.
This control estimates the effect of the extra pretraining phase, not efficiency
at equal total compute.

Assert common initial encoder/projector tensors, identical downstream head
states, actual training row/view traces, mask identities and all fitted scales.
Account for changed component construction with isolated initialisation or
copied common states, not just equal global seeds. Check parameters and buffers
at transfer. The no-predictor arm must not consume extra or fewer view draws.

Evaluate the final pretraining checkpoint before fine-tuning on 16 disjoint
held-out batches of 128. Use eval mode and frozen training BN statistics.
View seeds for batch b are base+20000+2*b and base+20001+2*b; realise views once
and share across arms. No BN recalibration, checkpoint selection or test fitting.
Clean identity rows evaluate downstream predictions.

### 6.3 Tier 0 and Tier 1 evidence

Tier 0 must include:

- Independent scalar cosine and autograd oracles at nonzero and near-zero
  norms; both directions weighted by 0.5. Check unequal widths, rank, empty
  rows, non-finite inputs and B=1 training BN rejection.
- Directional gradient tests: no target gradient for one loss, nonzero
  prediction gradient, and encoder gradients through both branches after
  summation. Detaching both complete branches must fail.
- Source topology and frozen bias checks; two BN updates per step, separate
  view statistics, cache reuse, and no predictor output BN. No-predictor
  symmetric detached cosine must have half the undetached cosine gradient
  on a nondegenerate fixed tensor fixture.
- Explicit card/plan values, X_PRED lineage, no Y_RAW dependency in pretraining,
  no teacher, optimiser reset and head/encoder transition state equality.
- The stage's mixed pretraining total against `main_simsiam.train`'s own
  `-(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5` through
  `nn.CosineSimilarity`, on a nondegenerate fixture at the fixture's own seed.
- `CosineAnneal` against `adjust_learning_rate`'s expression written out, and
  against `CosineDecay`, which it must not coincide with away from the endpoints.
- Every projector BatchNorm in its normalising regime and `X_REPR` arriving at
  order one, the two checks deviation 7's withdrawal is worth.
- That section 6.4's retired `S` maps an embedding of exact rank one and an
  isotropic one to the same value, which is why it carries no bound.
- Observed failures for mutants: remove either half-weight, remove target
  detach, detach prediction, swap a target view, retarget a direction at its own
  view, concatenate BN batches, change output BN affine, retain predictor in
  no-predictor arm, restore the withdrawn encoder initialiser, drop `S`'s row
  normalisation, and alter a treatment mask while preserving its observed count.

Tier 1 uses bases 42, 142 and 242 with the same seed offsets and pairing
rules, 512 training rows, 512 held-out rows, 40 observed treatments,
128 pretraining and 256 downstream steps; batch size remains 128.
Evaluate four held-out batches. Require finite losses/parameters/diagnostics,
nonzero intended gradients and actual parameter updates, exact pairing and
transition assertions, and finite downstream predictions in every seed.
Report alignment, encoder and projection rank, spread, norms and downstream
metrics for all arms, and the untrained encoder's rank beside them.
Do not assert that an ablation collapses or that an arm wins after this shorter
run. Any new directional Tier 1 assertion requires evidence on every declared
seed and a reviewed amendment.

### 6.4 Metrics and acceptance

Two statistics carry the attribution, both read from the terminal pretraining
checkpoint on section 6.2's fixed held-out batches, in eval mode on frozen
training BN buffers.

**View alignment** `A` is the mean of the two directional cosines between the
prediction and the other view's projection, averaged over rows, both directions
and all held-out batches, with epsilon 1e-12. It is the objective's own value
negated, so `A = 1` is the loss at its minimum possible value of -1. Paper
section 4.1: "Without stop-gradient, the optimizer quickly finds a degenerated
solution and reaches the minimum possible loss of -1." An ablation sitting
closer to that optimum than the full arm is that sentence, measured.

**Encoder effective rank** `R` is `exp` of the Shannon entropy of the centred
covariance's normalised eigenvalues of `X_REPR`, averaged over the same batches.
It is read at `X_REPR` and not at `X_PROJ` for two reasons: it is the port the
downstream stage inherits, matching the source's linear evaluation on backbone
features, and it is the only one no BatchNorm stands in front of.

Per replicate define:

| Required metric | Definition | Acceptance over ten replicates |
|---|---|---|
| stop_gradient_alignment_gap | A_no_stop - A_full | mean - stderr > 0.0 |
| predictor_alignment_gap | A_no_predictor - A_full | mean - stderr > 0.0 |
| stop_gradient_rank_gap | R_full - R_no_stop | mean - stderr > 0.0 |
| predictor_rank_gap | R_full - R_no_predictor | mean - stderr > 0.0 |
| pretraining_outcome_nll_cost | factual NLL_full - factual NLL_no_pretrain on clean held-out rows | mean + stderr <= 0.05 nat/row |

Use sample stderr with ddof=1 divided by sqrt(10); form within-seed differences
before summarising. Every threshold except the last is the statistic's own zero:
these are paired contrasts, so no level had to be chosen and none was read off a
measurement. The 0.05 NLL tolerance follows the local transfer question and is
explicitly chosen here; it is not a SimSiam paper number.

**Why all four attribution bounds are paired, and why there is no absolute
one.** The bound this set replaces, `full_projection_spread >= 0.5`, was an
absolute guard that could not fail; the audit note below is its post-mortem. A
paired set needs no separate collapse guard, because uniform collapse defeats it
directly: if every arm degenerates, all alignments approach 1 and all ranks
approach each other, and all four gaps go to zero rather than passing. The
reverse failure — a full arm that simply never trains — is caught in-run rather
than by a bound, by section 6.2's executed checks that every trainable component
changed and that the transferred checkpoint is the one that was fitted.
`initial_encoder_effective_rank`, the untrained encoder on the same views, is
reported beside `R` so that any collapse is visible; it is deliberately not a
bound, since a random 4-layer projection of six inputs has an effective rank of
about 19 and concentrating onto this fixture's invariant subspace is what a
working representation should do.

**What informed the choice of statistic.** Both were picked from the
2026-09-15 audit's five-seed probe on bases 42-442, which established that they
are well formed and that the gaps are reachable; the Tier 2 stream at
520000+100*i is disjoint from those seeds and from 310000-310900. That
disclosure is the point: the statistics are chosen, the thresholds are not.

Non-finite values are failures, never dropped replicates. Log each arm and
each seed, not only passing contrasts. If full and controls are indistinguishable
on both axes, the attribution claim has not reproduced even if downstream NLL is
good. Audit gradients, topology, BN, initialisation, views and optimiser fidelity
before proposing a prospective amendment; retain the failed result.

Informational metrics: clean treatment NLL, factual NLL, conditional-mean
treatment-effect RMSE after inverse outcome scaling, projection and predictor
spread and their two gaps, zero/near-zero vector fractions (norm <= 1e-8), raw
norms, concentration, projection effective rank and the initial encoder
diagnostics. Use analytic conditional means, not a noisy realised
potential-outcome difference, as the treatment-effect target.

For each held-out view's projection matrix Z, the retained spread diagnostic is
S(Z)=sqrt(d)*mean_j std_i(U_ij) with each row normalised to U at epsilon 1e-12,
correction=0, d=256, averaged over both views and the fixed held-out batches. It
is reported and no longer bounded, for the reason below.

#### Audit finding, 2026-09-15: why S no longer carries a bound

This is the post-mortem of the three spread-based bounds the table above
replaced. It is retained because the replacement is only as good as the reason
for it, and because the same trap is available to any card that reads a
statistic downstream of a non-affine BatchNorm.

`S` divides every row by its own norm, so it is invariant to the embedding's
scale and what remains is how evenly the channels share the direction. The
projector terminates in a non-affine BatchNorm, which equalises exactly that.
Measured on a deliberately collapsed fixture in `tests/invariants/test_simsiam.py`:
an embedding of **exact rank one** — every row a multiple of a single vector —
scores `S = 0.79` bare and `S = 0.99` behind that BatchNorm, and an isotropic
embedding of rank 98 scores the same 0.99. The first bound, `S >= 0.5`,
therefore cannot fail for any embedding leaving a functioning output BatchNorm,
and the two gap bounds are differences of two numbers both pinned near one,
whose residual variation is row-norm heterogeneity rather than collapse: over
the recorded run's thirty (arm, seed) points the correlation between mean
projection row norm and `S` is 0.88.

The decisive check was that correcting the recipe does not rescue them. In the
audit's paired probe cell where both deviation 3 and deviation 7 are returned to
the paper's values — and the full arm's encoder effective rank is above both
ablations' on every one of five seeds — the two spread gaps are still negative,
at -0.018 and -0.032. The instrument had to change whatever else did, which is
why deviations 3 and 7 and this section all moved together.

Paper section 4.1 reads this quantity at zero for its collapsed run, which
requires the pre-BatchNorm output to be constant across rows; in eval mode on
frozen buffers that is reachable. It is not reached here. The pre-BatchNorm
per-channel variance measured in every arm of the recorded run is 0.5 to 1.0,
five orders of magnitude above `eps = 1e-5`, so `var/(var+eps) = 0.99997` and
the BatchNorm is squarely in its normalising regime. The card imported the
paper's diagnostic without the state that makes it move.

One diagnostic this section already required did separate the arms on the
failed run, in the paper's own direction and on every seed: view alignment
reached 0.9998 and 0.99996 in the two ablations against 0.974 in the full arm,
which is Figure 2 (left)'s "reaches the minimum possible loss of -1" in this
fixture's terms. Centred covariance effective rank did not separate them there —
1.566 against 1.911 and 1.632, the wrong way round — but does once deviations 3
and 7 are corrected. Those are the two statistics the table above now bounds,
on a seed stream disjoint from every seed the audit saw.

The benchmark binds the protocol values to this card and lands with its
ten-seed results from a committed tree. The blank ledger placeholder is retained.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Suitable XTY transformations | Explicit views; import OracleSymmetry for this fixture only | Existing validated target-preservation contract; deviation 2 |
| Tabular capacity, initialisation, training budget and head losses | Exact §4 values | Declared local design, not source defaults; deviations 1, 3 and 4 |
| Zero-norm numerical convention in the printed equations | Separate normalisation, epsilon 1e-12 | Existing objective and Algorithm 1 form; difference from pinned runtime disclosed in deviation 6 |
| A transferable numeric noncollapse tolerance on this DGP | Prospective S >= 0.5 and positive paired gaps | Isotropic reference plus attribution question; not calibrated on results |
| Whether the local Adam protocol preserves source ablation behaviour | Answered: it does not, and it is withdrawn | The 2026-09-15 audit's paired probe; deviations 3 and 7 both participated, and neither returned alone restored the source's ordering |
| A collapse statistic this architecture can actually move | Answered: view alignment and encoder effective rank; `S` cannot | Section 6.4 and its audit note; `S` scores 0.99 on an embedding of exact rank one behind the terminal non-affine BatchNorm |
| How the source's per-epoch rate schedule maps to a step budget | The half cosine re-based on the 1000-step horizon | `adjust_learning_rate` is a function of `epoch/epochs`; `CosineAnneal(steps=1000)` is that function of `step/steps` (deviation 3) |
| Whether the predictor should hold a fixed rate | No: the paper's baseline row decays both | Table 1 reads 67.7 for the baseline against 68.1 for `fix_pred_lr`; deviation 5 |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Codex implementation audit under the user's PR #55 request; no independent human approval claimed | 2026-09-14 |
| Plan diffed against §3.2 and §4 | Codex; actual compiled plan and executable value/topology checks | 2026-09-14 |
| Fidelity re-audit after the failed §6 result | Claude Code, requested by the repository owner; recorded in `docs/experiments/2026-09-15-simsiam-fidelity-audit.md` | 2026-09-15 |
| Deviations 3 and 7 withdrawn, §2's claim restated, §6.4's bounds replaced | Repository owner, on the audit's two proposals and its own question about the optimiser; implemented in the same change | 2026-09-15 |
| Plan re-diffed against §3.2 and §4 after those amendments | `tests/invariants/test_simsiam.py`, over every answered §4 entry and the actual compiled plan | 2026-09-15 |

The actual compiled plan is included in the PR and checked against sections
3.2 and 4 by `tests/invariants/test_simsiam.py`.
