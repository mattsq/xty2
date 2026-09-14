# Recipe spec card: simsiam

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 for benchmark and reporting work.

Selected from BACKLOG.md §5.1. Implementation was requested after review of
PR #55. The callable, component topology and paired protocol implement this
card; section 6 records the measured result without changing prospective bounds.

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
  on terminal representation spread, followed by the effect of pretraining
  on held-out factual outcome NLL in a missing-treatment problem.
- **Published claim:** stop-gradient prevents collapse in the paper's image
  experiments without negative pairs or a momentum encoder; the predictor
  ablation also fails. This is empirical evidence, not a universal guarantee.
- **Local claim to test:** the two mechanics retain spread under the declared
  tabular views, without materially harming downstream outcome fit.
- **Not claimed:** ImageNet accuracy reproduction, superiority to VICReg,
  Barlow Twins or SCARF, general-purpose tabular augmentation, or improved causal
  identification. Treatment-effect RMSE is diagnostic, not established by NLL.

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
optimisation:  # deviation 3, applies to every paired arm
  optimiser:
    pretrain: adam(betas=(0.9, 0.999), eps=1e-08)
    joint_fit: adam(betas=(0.9, 0.999), eps=1e-08)
  lr:
    pretrain: 0.001
    joint_fit: 0.001
  lr_schedule:
    pretrain: constant 1.0
    joint_fit: constant 1.0
  weight_decay:
    pretrain: none
    joint_fit: none
  batch_size:
    pretrain: 128
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
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
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
| 1 | `judgement` | | Tabular MLP replaces ResNet-50; projector width 256 replaces 2048 and predictor bottleneck 64 replaces 512. Retain three-layer projector, output non-affine BN, two-layer predictor and 4:1 bottleneck ratio. | Bounded local experiment. Encoder initialisation follows the existing local encoder rather than residual-network initialisation. | Capacity, norms and collapse dynamics can differ; no image score transfers. |
| 2 | `judgement` | | Explicit caller-supplied views; the study imports VICReg's OracleSymmetry. | Preserve analytic propensity and conditional outcome means using known fixture symmetries. These are privileged DGP operations, not learned augmentations. | Tests a valid-view mechanism, not a general tabular policy. |
| 3 | `judgement` | | Adam 0.001, constant LR for encoder and predictor, no decay, batch 128, 1000 pretrain steps, single-process float32. | Match the local transfer budget; deliberately depart from source SGD, momentum 0.9, base LR 0.05 scaled by batch/256, decay 1e-4 and cosine training. No claim that SGD or cosine is unavailable. | Optimisation may determine whether either ablation collapses; a miss audits this choice before changing a bound. |
| 4 | `judgement` | | Fit XTY heads for 3000 steps, with the explicit supervised and marginal-NLL weights in §4, rather than source frozen image linear evaluation. Train-only scaling and 40 MCAR labels. | Isolate transfer to the repository's task. | Evaluates a different task and estimand; report local bounds only. |
| 5 | `judgement` | | No separate fixed-predictor-LR experiment; both networks use the same constant schedule. | This is a declared tabular protocol, not the source Table 1(c) variant, whose encoder LR still decays. | Predictor tracking may change; retain as an optimisation limitation of the claim. |
| 6 | `judgement` | | Use existing separate L2 normalisations with epsilon 1e-12 rather than reference nn.CosineSimilarity's default epsilon 1e-8. | Matches paper Algorithm 1's separate normalisation form and existing objective; pins numerical handling instead of silently inheriting it. | Differences near zero norms and in gradients must be tested and reported; ordinary nonzero cosine agrees. |

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

Rejected mappings: overwriting X_PROJ loses the target; using X_REPR for the
projection loses the transferred backbone; making the predictor an objective
parameter hides trainable component ownership; using a teacher changes the
method; inventing a role or view to mean a network layer misuses Realisation.

X_PRED is registered in DESIGN.md §2 and tensor-port validation, with executable
compilation/lineage checks. No new executor, row population, artifact kind,
shared SSL framework or card-key category is needed.

### Tier 2 outcome

The ten-seed run from committed source `c14090a43e2e` passed the full-spread
and downstream-NLL bounds but failed both attribution bounds. Every seed's
full-minus-ablation spread gap was negative. The original bounds remain
unchanged. The [complete result and fidelity audit](../experiments/2026-09-14-simsiam-tier2.md)
record all arms, per-seed values, rank/norm diagnostics and the source environment.
Passing spread did not imply high rank: the full projection's mean effective
rank was 1.566 out of 256. No ImageNet reproduction or causal identification
claim follows from this local result.

## 6. Reproduction target

This is a project-local mechanism study, not a published-number reproduction.
All bounds were declared prospectively, not calibrated on measured results.
Neither an ablation collapse nor a downstream gain is assumed.

```yaml
reproduction:
  dataset: shared two_cluster_population DGP; six features; K=2; OracleSymmetry views
  variant: full versus no-stop-gradient, no-predictor and no-pretraining
  split: 1024 train with 40 observed treatments; 2048 fully observed held-out rows
  metric: terminal normalised projection spread; two paired spread gaps; factual outcome NLL cost
  published: none - project-local tabular adaptation
  published_source: n/a
  tolerance: all four one-standard-error bounds in section 6.4
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |
| 2026-09-14 | `c14090a43e2e` | full_projection_spread<br>stop_gradient_spread_gap<br>predictor_spread_gap<br>pretraining_outcome_nll_cost | 0.896711 +/- 0.013111<br>-0.066665 +/- 0.011058<br>-0.078406 +/- 0.012552<br>-0.029322 +/- 0.032332 | no; spread and NLL pass, both attribution gaps fail |

### 6.2 Fixed DGP and paired execution

Import `two_cluster_population`, `continuous_schema`, `training_dataset`
and `on_the_training_scale` from `xty2.evaluation.benchmarks.common`.
Adopt the separated generator with `low=SEPARATED` explicitly, unchanged.
Import `OracleSymmetry` and its preservation tolerance from
`xty2.evaluation.vicreg_views`; do not copy the DGP or transform.
See [VICReg's valid-view contract](vicreg.md#62-fixed-dgp-and-paired-execution).

Tier 2 replicate i=0..9 uses base=310000+100*i: training seed base+1,
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
- Observed failures for mutants: remove either half-weight, remove target
  detach, detach prediction, swap a target view, concatenate BN batches,
  change output BN affine, retain predictor in no-predictor arm, and alter a
  treatment mask while preserving its observed count.

Tier 1 uses bases 42, 142 and 242 with the same seed offsets and pairing
rules, 512 training rows, 512 held-out rows, 40 observed treatments,
128 pretraining and 256 downstream steps; batch size remains 128.
Evaluate four held-out batches. Require finite losses/parameters/diagnostics,
nonzero intended gradients and actual parameter updates, exact pairing and
transition assertions, and finite downstream predictions in every seed.
Report spread, alignment, norms and downstream metrics for all arms.
Do not assert that an ablation collapses or that an arm wins after this shorter
run. Any new directional Tier 1 assertion requires evidence on every declared
seed and a reviewed amendment.

### 6.4 Metrics and acceptance

For each held-out view's projection matrix Z, normalise each row with epsilon
1e-12 to U and define S(Z)=sqrt(d)*mean_j std_i(U_ij), correction=0, d=256.
Average over both views and the fixed held-out batches. No BN refitting occurs.
Constant nonzero outputs have S=0; isotropic directions have S near 1.
Also report encoder and predictor spread, raw norms, concentration and centred
covariance effective rank. Projection spread alone cannot rule out rank loss.

Per replicate define:

| Required metric | Definition | Acceptance over ten replicates |
|---|---|---|
| full_projection_spread | S_full | mean - stderr >= 0.5 |
| stop_gradient_spread_gap | S_full - S_no_stop | mean - stderr > 0.0 |
| predictor_spread_gap | S_full - S_no_predictor | mean - stderr > 0.0 |
| pretraining_outcome_nll_cost | factual NLL_full - factual NLL_no_pretrain on clean held-out rows | mean + stderr <= 0.05 nat/row |

Use sample stderr with ddof=1 divided by sqrt(10); form within-seed differences
before summarising. The terminal full spread target is half the isotropic
reference, a provisional noncollapse guard rather than a learned threshold.
The 0.05 NLL tolerance follows the local transfer question and is explicitly
chosen here; it is not a SimSiam paper number. Both positive spread-gap
requirements test attribution without requiring exact zero-spread ablations.

Non-finite values are failures, never dropped replicates. Log each arm and
each seed, not only passing contrasts. If full and controls all retain spread,
the attribution claim has not reproduced even if downstream NLL is good.
Audit gradients, topology, BN, initialisation, views and optimiser fidelity
before proposing a prospective amendment; retain the failed result.

Informational metrics: clean treatment NLL, factual NLL, conditional-mean
treatment-effect RMSE after inverse outcome scaling, view alignment, zero/near-zero
vector fractions (norm <= 1e-8), raw norms, concentration and effective rank.
Use analytic conditional means, not a noisy realised potential-outcome
difference, as the treatment-effect target.

The benchmark binds the protocol values to this card and lands with its
ten-seed results from a committed tree. The blank ledger placeholder is retained.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Suitable XTY transformations | Explicit views; import OracleSymmetry for this fixture only | Existing validated target-preservation contract; deviation 2 |
| Tabular capacity, initialisation, training budget and head losses | Exact §4 values | Declared local design, not source defaults; deviations 1, 3 and 4 |
| Zero-norm numerical convention in the printed equations | Separate normalisation, epsilon 1e-12 | Existing objective and Algorithm 1 form; difference from pinned runtime disclosed in deviation 6 |
| A transferable numeric noncollapse tolerance on this DGP | Prospective S >= 0.5 and positive paired gaps | Isotropic reference plus attribution question; not calibrated on results |
| Whether the local Adam protocol preserves source ablation behaviour | Unknown until the paired study | Treat a negative result as an audit trigger; do not promise reproduced status |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Codex implementation audit under the user's PR #55 request; no independent human approval claimed | 2026-09-14 |
| Plan diffed against §3.2 and §4 | Codex; actual compiled plan and executable value/topology checks | 2026-09-14 |

The actual compiled plan is included in the PR and checked against sections
3.2 and 4 by `tests/invariants/test_simsiam.py`.
