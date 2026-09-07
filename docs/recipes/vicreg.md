# Recipe spec card: vicreg

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 for benchmark and reporting work.

This card selects BACKLOG.md §5.1. It stops before implementation for review.

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906v3) |
| Authors, year | Adrien Bardes, Jean Ponce, Yann LeCun; ICLR 2022 |
| DOI / arXiv | arXiv:2105.04906 |
| Version used | v3, 2022-01-28; §4.1 equations (1)–(7), §4.2 architecture and training |
| Reference implementation | [facebookresearch/vicreg](https://github.com/facebookresearch/vicreg/tree/4e12602fd495af83efd1631fbe82523e6db092e0) @ `4e12602fd495af83efd1631fbe82523e6db092e0`; `main_vicreg.py`, symbols `VICReg.forward`, `Projector`, `get_arguments` |
| Reference impl. runnable? | Not attempted. Author source inspected; no ImageNet or GPU reproduction claimed. |

## 2. Estimand and claim

- **Estimand:** paired changes in embedding collapse/redundancy and held-out
  treatment and factual outcome NLL after representation pretraining.
- **Claim to test:** on the fixed XTY fixture, explicit variance preservation
  maintains embedding spread, covariance regularisation reduces redundant
  dimensions, and transferring the encoder does not materially harm outcome fit.
- **Nearest shipped baseline:** `scarf.md`, the same two-stage pattern. The
  controlled attribution arms below differ by exactly one VICReg coefficient.
  A comparison with shipped SCARF would change several ingredients and is not
  the primary causal contrast.
- **Not claimed:** ImageNet reproduction, superiority to SCARF, improved causal
  identification, or a guarantee of downstream gains. Factual NLL alone does
  not validate treatment-effect recovery. The published evidence concerns image
  representation transfer, not missing-treatment XTY estimation.

## 3. Equations and mapping

### 3.1 As published

Use the paper's notation: two batches of embeddings `Z`, `Z'`, each containing
`n` vectors of dimension `d`, with column `z^j` and batch mean `bar(z)`.
The following is the mathematical specification in §4.1:

```text
(1) v(Z) = (1/d) sum_j max(0, gamma - S(z^j, epsilon))
(2) S(x, epsilon) = sqrt(Var(x) + epsilon)
(3) C(Z) = (1/(n-1)) sum_i (z_i - bar(z))(z_i - bar(z))^T
(4) c(Z) = (1/d) sum_{i != j} C(Z)_{ij}^2
(5) s(Z,Z') = (1/n) sum_i ||z_i - z'_i||_2^2
(6) ell(Z,Z') = lambda*s + mu*(v(Z)+v(Z')) + nu*(c(Z)+c(Z'))
(7) L = sum_{I in D} sum_{t,t' sampled from T} ell(Z^I,Z'^I)
```

**Selected numerical variant: pinned author code, not literal equation (6).**
`VICReg.forward` computes elementwise mean MSE (dividing equation (5) by `d`),
averages the two variance penalties, and sums the two covariance penalties:

```text
L_code = 25 * mean_{i,j}(Z_ij - Z'_ij)^2
       + 25 * (v(Z) + v(Z')) / 2
       + 1 * (c(Z) + c(Z'))
gamma = 1; epsilon = 1e-4; sample variance uses correction=1
```

Equivalently, the printed equation's coefficients are `(25/d, 12.5, 1)`.
Do not retain `(25,25,1)` while silently switching reductions. Both branches
receive gradients, including through batch centring, variance and covariance.
There is no target detach, output L2 normalisation, predictor, queue or teacher.
This contract is single-process: the local full batch is the statistics batch.

### 3.2 Mapping to xty2

All names marked **new** are proposed, not current exports.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| x, x' | independent transformations of the same rows | `X_RAW` at `corrupted_a`, `corrupted_b` | two `ViewSpec`s, each `FeatureCorruption(rate=0.6, columns=None)` |
| f_theta | shared encoder | `X_RAW -> X_REPR` | `MLPEncoder`, four 256-wide ReLU layers |
| h_phi | shared expander | `X_REPR -> X_PROJ` | **new** `VICRegExpander`, widths (512,512,512), hidden BN/ReLU, linear output |
| s/d | unnormalised elementwise MSE | both `X_PROJ` realisations | **new** `EmbeddingInvariance`, rows all |
| (v+v')/2 | batch spread penalty | both `X_PROJ` realisations | **new** `EmbeddingVariance`, gamma=1, epsilon=1e-4, correction=1 |
| c+c' | within-view off-diagonal covariance penalty | both `X_PROJ` realisations | **new** `EmbeddingCovariance`, correction=1, divide by embedding dimension |
| downstream f_theta | transferred, trainable encoder | `X_REPR` at identity | `Stage(joint_fit, initialise_from=pretrain)` |
| project-local p(y given x,t) | outcome model | `Y_GIVEN_XT` | `TARNetHead`, `ObservedOutcomeNLL` |
| project-local p(t given x) | treatment model | `T_GIVEN_X` | `CategoricalPropensity`, `ObservedTreatmentNLL` |
| project-local missing-t likelihood | exact enumeration | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path=both)` |

Both stages use the existing gradient executor, `UniformSampler(128)`,
`OptimiserSpec`, `Constant`, and `Ramp`. Pretraining trains only `mlp_encoder`
and `vicreg_expander`. Fine-tuning trains only `mlp_encoder`, `tarnet_head` and
`categorical_propensity`; the expander has no downstream forward pass. A fresh
optimiser is created at the transition. The heads retain their initial states.

Both views preserve row identity, treatments, outcomes, availability masks,
fold ids and weights. Corruption samples exactly `floor(0.6*M)` mutable columns
per row without replacement, using independent training-population donors per
cell. It does not guarantee treatment-label preservation. Pretraining reads
neither treatment nor outcome values; downstream objectives use identity only.
Derived-column recompute rules remain explicit, as in SCARF.

Variance and covariance are `batch_coupled=True`, require an internally declared
batch size, and reject `n < 2`. All three losses use the same two cached draws.
Each returns its own scalar batch mean convention above with mixer reduction
`mean`; there is no additional division by batch size or dimension. Sample
weights do not reweight these batch statistics; this fixture has unit weights.
Raw constituent losses and each branch's standard deviations/covariance energy
are diagnostics. Gamma, epsilon and correction must be explicit constructor
fields and appear in plan details; no expansion of the closed card-key schema
is proposed. Coefficients bind through `Weighted` to the canonical weight keys.

## 4. Mechanics checklist

Comments identify source decisions or the explicit project-local choices in §5.

```yaml
gradients:
  stop_gradients:
    pretrain.embedding_invariance: none
    pretrain.embedding_variance: none
    pretrain.embedding_covariance: none
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: n/a  # author VICReg.forward
  gradient_clipping: {pretrain: none, joint_fit: none}  # deviation 4
  marginal_nll_grad_path:
    joint_fit.missing_treatment_marginal_nll: both
teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: false
losses:
  reduction:
    pretrain.embedding_invariance: mean
    pretrain.embedding_variance: mean
    pretrain.embedding_covariance: mean
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    pretrain.embedding_invariance: all
    pretrain.embedding_variance: all
    pretrain.embedding_covariance: all
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:  # pinned author forward; downstream deviation 5
    pretrain.embedding_invariance: 25.0
    pretrain.embedding_variance: 25.0
    pretrain.embedding_covariance: 1.0
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    pretrain.embedding_invariance: constant 25.0
    pretrain.embedding_variance: constant 25.0
    pretrain.embedding_covariance: constant 1.0
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a
optimisation:  # deliberate matched tabular protocol, deviation 4
  optimiser:
    pretrain: adam(betas=(0.9, 0.999), eps=1e-08)
    joint_fit: adam(betas=(0.9, 0.999), eps=1e-08)
  lr: {pretrain: 0.001, joint_fit: 0.001}
  lr_schedule: {pretrain: constant 1.0, joint_fit: constant 1.0}
  weight_decay: {pretrain: none, joint_fit: none}
  batch_size: 128
  labelled_unlabelled_ratio: n/a
  total_steps_or_epochs: {pretrain: 1000, joint_fit: 3000}  # optimiser steps
architecture:  # deviations 2 and 5; expander topology follows author Projector
  widths_depths:
    mlp_encoder: [256, 256, 256, 256]
    vicreg_expander: [512, 512, 512]
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: relu
    vicreg_expander: hidden relu; output linear
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: none
    vicreg_expander: hidden BatchNorm1d(eps=1e-5, momentum=0.1, affine=true, track_running_stats=true); output none
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    vicreg_expander: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    vicreg_expander: torch Linear reset_parameters; hidden bias=true; output bias=false; BN weight=1,bias=0,running_mean=0,running_var=1
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits
data:  # deviations 3 and 5; §6.2
  standardisation: "x: zscore fitted on train only"
  outcome_scaling: "y: zscore fitted on train only; evaluation on that scale"
  treatment_encoding: n/a
  split_protocol: fixed two-cluster DGP; disjoint train and held-out populations; no test-based selection
  missingness_mechanism: MCAR exactly 40 observed training treatments keyed by row_id
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Use author-code MSE and variance reductions instead of literal equations (5)–(6). | Reproduce an executable, pinned numerical variant; discrepancy is explicit in §3.1. | Changes relative gradient scales substantially, especially as dimension changes. |
| 2 | `judgement` | — | Four-layer tabular MLP and 512-wide expander replace ResNet-50 and 8192-wide expander; retain the author's hidden BN/ReLU and bias-free output topology. Encoder uses the SCARF initialisation, expander uses author defaults. | Study an expanding embedding (512 > 256) on the existing tabular problem. This choice survives unlimited framework capacity. | Different capacity and conditioning; no ImageNet number applies. |
| 3 | `judgement` | — | Two independent empirical feature corruptions at rate 0.6 replace image transformations. | A tabular representation experiment using existing training-population transforms. The SCARF rate is a declared starting choice, not a VICReg default or guaranteed label-preserving operation. | Can erase treatment signal; report view damage and transfer outcomes. |
| 4 | `judgement` | — | Adam 0.001, no decay/clipping/schedule, batch 128, 1000 pretrain and 3000 fit steps; single-process float32. | Match the local SCARF-scale protocol rather than §4.2's ImageNet LARS, batch 2048 and epoch schedule. No claim that LARS is unavailable. | Batch statistics, optimisation and compute differ; validate mechanism directly. |
| 5 | `judgement` | — | Fine-tune the XTY heads, with outcome/treatment weights 1 and missing-treatment weight ramping to 0.5 over 1000 steps; train-only scaling, 40 labels, synthetic evaluation. | Hold the local downstream stack fixed across arms. These are project policies, not paper constants. | Tests transfer under missing treatments; likelihood is not causal identification evidence. |

### 5.1 Framework additions made for this card

Proposed for implementation after review; none are implemented by this PR.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| Three independent embedding objectives in §3.2 | Fidelity-bearing, reversible | proposed VICReg | n/a; no new vocabulary | Preserve separately ablatable terms and their exact reductions using existing ports. |
| `VICRegExpander` component | Fidelity-bearing, reversible | proposed VICReg | n/a; no new vocabulary | Current `ProjectionHead` lacks hidden BN and a bias-free final layer. A separate component preserves existing recipes and carries the author's topology. |

No new port, executor, row population, artifact or framework debt is proposed.
Existing `X_PROJ` is an embedding tensor, not a promise of unit norm.

## 6. Reproduction target

This is a predeclared project-local mechanism study. All required bounds are
provisional scientific targets for review, not measured tolerances or claims
that VICReg must achieve them. A miss requires an audit and a recorded result,
not automatic threshold relaxation.

```yaml
reproduction:
  dataset: shared two_cluster_population DGP, 6 features and K=2; section 6.2
  variant: full VICReg versus paired zero-variance, zero-covariance and no-pretraining arms
  split: 1024 train with 40 observed treatments; 2048 fully observed held-out rows
  metric: terminal embedding spread, paired variance and redundancy effects, factual outcome NLL difference; treatment NLL informational
  published: none - project-local tabular adaptation
  published_source: n/a
  tolerance: all required one-standard-error bounds in section 6.4
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

### 6.2 Fixed DGP and paired execution

Import `two_cluster_population`, `continuous_schema`, `training_dataset` and
`on_the_training_scale` from `xty2.evaluation.benchmarks.common`. Adopt
`fixmatch.md` §6.1's separated two-cluster equations unchanged, as SCARF does;
do not transcribe the generator. Use its default `low=SEPARATED` explicitly.
Only the observed-treatment count and the batch size follow this card.

Tier 2 replicate `i=0..9` uses `base=190000+100*i`: train seed `base+1`, test
seed `base+2`, model seed `base+6`, execution seed `base+10000`. Row offsets are
0 and 10000. All outcomes are observed. Apply one shared MCAR mask through
`DataSpec`; hidden training treatments are available to evaluation only.

The four arms are full, variance weight exactly zero, covariance weight exactly
zero, and no pretraining. Preserve components, initial tensors, data, row
streams, view draws, step counts and downstream objectives across the first
three. Evaluate all four downstream fits. The no-pretraining arm removes the
pretrain stage and its inheritance edge; offset its execution seed by the
existing `STREAM_STRIDE` so its downstream stream matches stage index 1.
Assert equality of actual sampled row ids and view draws, not merely seeds.
Heads are identical before fine-tuning, and optimiser moments are reset.

Evaluate pretraining embeddings at its final checkpoint, before fine-tuning,
on 16 disjoint held-out batches of 128 rows, in eval mode with frozen training
BN buffers. Make two corruption draws per batch from the training population
with independent generators `base+20000+2*b` and `base+20001+2*b` for batch
`b=0..15`; reuse these realised views across arms. Fine-tuned predictions use
the clean held-out rows. No checkpoint selection or test-time fitting occurs.

### 6.3 Tier 0 and Tier 1 contracts

Tier 0 must compare values and input gradients to an independent scalar/loop
oracle of §3.1 on non-degenerate float64 tensors with `B=5, d=3`, unequal
branch variances, nonzero means and nonzero off-diagonals. Check `n-1`, `1/d`,
both variance branches, summed covariance branches, epsilon inside sqrt, and
the elementwise MSE reduction. Check finite zero-variance values without
claiming a nonzero gradient at exact collapse; reject batches below two rows.
Check view exchange symmetry, shift invariance of variance/covariance, and
nonzero gradients on both branches of the full loss. The expander must permit
negative, non-unit outputs and have the specified BN/bias placement. Verify
cached-draw sharing, training-only donor/scaler provenance, downstream head
isolation in pretraining, encoder transfer and absence of expander execution
in fine-tuning. Compile-plan/card checks cover every applicable §4 leaf.

Mutation evidence must include: `n-1 -> n`, MSE sum instead of mean, removing
the variance half factor, one branch detached, covariance diagonal included,
one view resampled per objective, output L2 normalisation, and an expander
parameter leaking into fine-tuning. Each relevant assertion must be seen fail.

Tier 1 uses bases `[191, 293, 397]`, the same stream-offset rules and fixture
sizes, 200 pretrain and 300 downstream steps. These are explicit smoke-budget
overrides, not Tier 2 defaults. Require finite losses/gradients, valid class
probabilities, changed encoder parameters, and correct transition/provenance
on every seed. Run the three pretraining arms and no-pretraining arm. Report
spread, redundancy, treatment NLL and outcome NLL, but promote no directional
smoke assertion until it holds on every declared seed. A low total loss is
insufficient because constant embeddings minimise the invariance term.

### 6.4 Metrics and acceptance

For each evaluation batch and branch define `a = mean_j sqrt(Var_j + 1e-4)`
with correction 1. Define redundancy `r = sum_offdiag C_ij^2 /
sum_j C_jj^2`; reject a zero denominator for full/no-covariance arms rather
than awarding a collapsed embedding perfect decorrelation. Average each
statistic over the two views and then the 16 batches, yielding one observation
per seed. Also report encoder and expander norms, fraction of dimensions with
standard deviation below 0.1, and covariance eigenvalue concentration.

Required targets, each assessed using the sample standard error over the ten
paired seed observations (`sd/sqrt(10)`):

1. Full-arm spread: `mean(a_full) - SE(a_full) >= 0.5`.
2. Variance attribution: `mean(a_full-a_no_variance) - SE >= 0.1`.
3. Covariance attribution: `mean(r_no_covariance-r_full) - SE >= 0.01`.
4. Outcome guardrail: `mean(NLL_full-NLL_no_pretrain) + SE <= 0.05` nats/row,
   using factual Gaussian outcome NLL on the same training-standardised scale.

The first two thresholds distinguish substantial spread from near collapse;
the third asks for a measurable reduction in relative redundant energy; the
fourth budgets a small additive predictive cost without requiring positive NLL
denominators. They are project-local effect-size choices, not inherited SCARF
tolerances. All metrics must be finite in every replicate. Report all per-seed
values, both arms' absolute values and their paired differences.

Treatment NLL differences, true mean treatment-effect error, and view-induced
changes in the DGP's Bayes treatment probabilities are informational. Compute
the latter using the original-scale corrupted features and the analytic
propensity specified by `fixmatch.md` §6.1, never the fitted classifier. Report
them before interpreting a downstream failure as objective failure. Do not
claim an interaction effect from these one-term ablations.

Run Tier 2 from a committed implementation. Benchmark registration, complete
results and ledger/status update must land together under `CLAUDE.md`; this
draft adds no callable, benchmark registration or fabricated execution plan.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Printed reductions differ from author code | Pin the code convention and retain both formulas in §3.1 | Executable reference; deviation 1 records the departure. |
| Variance estimator correction | Sample variance, correction=1 | Author `Tensor.var` convention and covariance denominator. |
| Hidden linear biases, final bias, BN defaults and expander initialisation | Hidden bias on, final bias off, default PyTorch Linear/BN initialisation and explicit BN settings in §4 | Author `Projector`; make library defaults explicit. |
| Appropriate tabular views, width and optimiser | Two empirical corruptions, 512-wide expander, Adam fixed budget | Deliberate experiment choices in §5, not purported source defaults. |
| No published XTY acceptance thresholds | Fixed effect-size targets in §6.4, review before execution | Distinguish attribution from transfer; no measured spread available yet. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
