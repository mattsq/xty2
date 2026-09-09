# Recipe spec card: barlow_twins

**Status:** `smoke-passing`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

**Agent route:** read §2–§5 before implementation; §6 defines acceptance.
`xty2.recipes.barlow_twins` compiles, Tier 0 passes, and §6.3's three-seed
Tier 1 study runs in `tests/smoke/test_barlow_twins.py`; its numbers are in
[the Tier 1 note](../experiments/2026-09-09-barlow-twins-smoke.md). No §6
Tier 2 result exists, so no bound in §6.4 has been assessed at its full budget.

## 1. Provenance

| Field | Value |
|---|---|
| Paper | Barlow Twins: Self-Supervised Learning via Redundancy Reduction |
| Authors, year | Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, Stéphane Deny, 2021 |
| DOI / arXiv | [arXiv:2103.03230](https://arxiv.org/abs/2103.03230) |
| Version used | [ICML 2021 proceedings, PMLR 139:12310–12320](https://proceedings.mlr.press/v139/zbontar21a/zbontar21a.pdf), equations (1)–(2), Algorithm 1 and §2.2 |
| Reference implementation | [facebookresearch/barlowtwins](https://github.com/facebookresearch/barlowtwins/tree/8e8d284ca0bc02f88b92328e53f9b901e86b4a3c) @ `8e8d284ca0bc02f88b92328e53f9b901e86b4a3c`; `main.py`, parser and `BarlowTwins` |
| Reference impl. runnable? | Not attempted; source inspected. No ImageNet training claimed. |

## 2. Estimand and claim

- **Estimand:** paired change in held-out embedding redundancy from adding the
  off-diagonal penalty, conditional on retaining cross-view diagonal alignment;
  paired factual outcome NLL cost after encoder transfer.
- **Claim to test:** on the fixed, target-preserving XTY fixture, the complete
  objective learns aligned, nonconstant embeddings, reduces redundancy relative
  to its diagonal-only ablation, and transfers without material outcome harm.
- **Nearest shipped baseline:** [VICReg](vicreg.md), with the same encoder,
  widths, explicit views, sampler, budgets and downstream stack. The decisive
  attribution comparison is Barlow Twins versus its own zero-off-diagonal arm.
  VICReg is contextual: its objective and hidden projector biases both differ.
- **Not claimed:** ImageNet reproduction, superiority to VICReg, general tabular
  augmentation validity, causal identification, or statistical independence of
  decorrelated features. The source evaluates image representations; the XTY
  claim is a project-local mechanism experiment.

## 3. Equations and mapping

### 3.1 As published

With batch-centred embeddings `Z^A`, `Z^B`, batch index `b`, feature indices
`i,j`, the paper's equations are:

```text
(1) L_BT = sum_i (1-C_ii)^2 + lambda * sum_i sum_{j != i} C_ij^2
(2) C_ij = sum_b z^A_bi z^B_bj /
           (sqrt(sum_b (z^A_bi)^2) * sqrt(sum_b (z^B_bj)^2))
```

**Selected variant: pinned author code.** `main.py:BarlowTwins.forward` uses
non-affine batch normalisation then the cross-product divided by batch size.
For this single-process contract, compute equivalent differentiable,
stateless arithmetic inside each objective:

```text
mu^v_j = mean_b Z^v_bj
var^v_j = mean_b (Z^v_bj - mu^v_j)^2             # correction=0
U^v_bj = (Z^v_bj - mu^v_j) / sqrt(var^v_j + epsilon)
C = (U^A)^T U^B / B
D = sum_i (1-C_ii)^2
O = sum_{i != j} C_ij^2
L = D + 0.0051 * O; epsilon = 1e-5
```

The parser sets 0.0051, whereas paper §2.2 prints 0.005. Algorithm 1's
unspecified `std` correction must not override the executable BatchNorm
convention. Epsilon is inside each branch's square root. No division by `d`
or `d*(d-1)` belongs in either training term. Every ordered off-diagonal entry
is included. Both branches and their means/variances receive gradients.

This is cross-view correlation, not within-view covariance or row cosine
similarity. No teacher, predictor, stop-gradient, negatives or memory bank is
used. Exact constant embeddings have finite loss `D=d, O=0` and can have zero
gradient; do not promise escape from exact collapse.

### 3.2 Mapping to xty2

The three objects marked **new** below were added by the implementing PR and
are now exports; the remaining rows were already shipped.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| Y^A, Y^B | independent transforms of the same rows | `X_RAW` at `corrupted_a`, `corrupted_b` | two `ViewSpec`s, explicit `first_transforms` and `second_transforms` tuples |
| encoder part of f_theta | shared representation | `X_RAW -> X_REPR` | `MLPEncoder`, four 256-wide ReLU layers |
| projector part of f_theta | shared embedding | `X_REPR -> X_PROJ` | **new** `BarlowTwinsProjector`, widths (512,512,512), all Linear biases off, hidden BN/ReLU |
| D | diagonal alignment term | both `X_PROJ` realisations | **new** `CrossCorrelationDiagonal` |
| O | off-diagonal redundancy term | both `X_PROJ` realisations | **new** `CrossCorrelationOffDiagonal` |
| transferred encoder | downstream trainable representation | `X_REPR` at identity | `Stage(joint_fit, initialise_from=pretrain)` |
| project-local p(y given x,t) | outcome density | `Y_GIVEN_XT` | `TARNetHead`, `ObservedOutcomeNLL` |
| project-local p(t given x) | propensity | `T_GIVEN_X` | `CategoricalPropensity`, `ObservedTreatmentNLL` |
| project-local missing-t likelihood | enumerate K treatments | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path=both)` |

Use existing gradient stages, `UniformSampler(128)`, `OptimiserSpec`, `Weighted`,
`Constant`, `Ramp` and `DataSpec`. Pretraining trains only `mlp_encoder` and
`barlow_twins_projector`. Fine-tuning trains only the encoder and XTY heads;
transfer encoder parameters/buffers, reset optimiser moments, retain the heads'
initial tensors, and never execute the projector downstream.

Both new objectives are `batch_coupled=True`, eligible on all rows, require
`B >= 2`, and reject `ExternalBatches`. They consume the same cached draws and
may share a pure correlation helper, but need no mutable shared state. They
return `LossTerm(value=D or O, n=B)` with mixer reduction `mean`, treating each
as a batch statistic, with no second reduction. Sample weights do not alter
statistics; the declared experiment uses unit weights. Each scalar retains
its source coordinate-sum convention, as VICReg's covariance term retains its
own internal coordinate reduction.

Normalisation epsilon and correction are explicit constructor arguments with
`REQUIRED` defaults and recorded in `plan_details()`, along with the exact
coordinate reductions. Existing closed card keys bind coefficients and batch
size. Views preserve identity, masks, treatment/outcome values, folds and
weights; pretraining never reads labels. There is no implicit corruption policy.
The benchmark supplies `OracleSymmetry`; it does not enter recipe assembly.

## 4. Mechanics checklist

Source arithmetic and topology are §3.1 and pinned `main.py:BarlowTwins`.
The optimiser, dimensions and downstream choices are deliberate §5 adaptations.
Every applicable leaf must bind by value to the future compiled plan.

```yaml
gradients:
  stop_gradients:
    pretrain.cross_correlation_diagonal: none
    pretrain.cross_correlation_off_diagonal: none
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: n/a
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
    pretrain.cross_correlation_diagonal: mean
    pretrain.cross_correlation_off_diagonal: mean
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    pretrain.cross_correlation_diagonal: all
    pretrain.cross_correlation_off_diagonal: all
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    pretrain.cross_correlation_diagonal: 1.0
    pretrain.cross_correlation_off_diagonal: 0.0051
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    pretrain.cross_correlation_diagonal: constant 1.0
    pretrain.cross_correlation_off_diagonal: constant 0.0051
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a
optimisation:
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
  widths_depths:
    mlp_encoder: [256, 256, 256, 256]
    barlow_twins_projector: [512, 512, 512]
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: relu
    barlow_twins_projector: hidden relu; output linear
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: none
    barlow_twins_projector: hidden BatchNorm1d(eps=1e-5, momentum=0.1, affine=true, track_running_stats=true); output none
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    barlow_twins_projector: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    barlow_twins_projector: torch Linear reset_parameters; all linear bias=false; BN weight=1,bias=0,running_mean=0,running_var=1
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits
data:
  standardisation: x: zscore fitted on 'train'
  outcome_scaling: y: zscore fitted on 'train'
  treatment_encoding: n/a
  split_protocol: fixed two-cluster DGP; disjoint train and held-out populations; no test-based selection; training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 40 labelled rows, keyed by row_id
```

All step counts are optimiser steps. All outcomes are observed in this
experiment; general use must intersect outcome-loss eligibility with
`y_observed`. Loss normalisation is stateless §3.1 arithmetic, separate from
the projector's hidden BN described in the architecture block.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | | Select code lambda=0.0051 and BatchNorm epsilon/correction, rather than rounded paper lambda=0.005 and unspecified pseudocode std. | Pin an executable numerical variant. | Changes penalty balance and low-variance behaviour. |
| 2 | `judgement` | | Tabular four-layer encoder and 512-wide projector replace ResNet-50 and 8192-wide projector. Preserve all-bias-free source projector topology. | Match local VICReg capacity; no image pipeline is needed to answer this experiment. | Dimension and finite-batch rank change attainable redundancy; source accuracy does not transfer. |
| 3 | `judgement` | | Explicit supplied views; benchmark imports target-preserving oracle symmetries. | Reuse the validated fixture contract, independently of framework capacity. | Privileged DGP knowledge makes this fixture-specific; no general augmentation claim. |
| 4 | `judgement` | | Adam 0.001, no decay or schedule, 128 rows, 1000 pretrain/3000 downstream steps, single-process float32. | Matched local budget, rather than source LARS/ImageNet schedule. | Different optimisation and batch statistics may fail the targets. Retain source lambda despite changed d; do not assume it is optimal. |
| 5 | `judgement` | | XTY fine-tuning with train-only scaling and missing-treatment marginalisation replaces image linear evaluation. | Hold observed weights 1 and missing weight ramp 0 to 0.5 over 1000 steps fixed across arms, as in VICReg. | Measures factual transfer with 40 labels, not ImageNet accuracy or causal identification. |
| 6 | `judgement` | | Stateless loss normalisation reproduces training-mode output BN arithmetic; discard the author's output-BN running buffers. | The output normaliser has no inference consumer. The projector's hidden BN retains ordinary buffers. | Identical single-process training loss/gradients; diagnostics explicitly recompute held-out batch statistics. |

### 5.1 Framework additions made for this card

Both rows were approved at review and are implemented.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `BarlowTwinsProjector` | Fidelity-bearing, reversible | `barlow_twins` | n/a; existing ports | `VICRegExpander` hard-codes hidden biases; preserve source bias placement without changing VICReg's validated initialisation. |
| Two cross-correlation objectives and a pure arithmetic helper | Fidelity-bearing, reversible | `barlow_twins` | n/a; existing objective contract | Preserve source reductions and expose the off-diagonal ablation independently. |

No new port, executor, row population, artifact kind or framework debt. Nothing
is omitted because the framework cannot express it.

## 6. Reproduction target

```yaml
reproduction:
  dataset: fixed two-cluster XTY DGP, low=SEPARATED
  variant: full Barlow Twins; diagonal-only; no pretraining; contextual VICReg
  split: 1024 train, 40 observed treatments; 2048 fully observed held-out rows
  metric: diagonal alignment; active dimensions; paired redundancy gap; factual outcome NLL cost
  published: none - project-local tabular mechanism adaptation
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
`on_the_training_scale` from `xty2.evaluation.benchmarks.common`, with
`low=SEPARATED` explicit. Import `OracleSymmetry` from
`xty2.evaluation.vicreg_views` and adopt the unchanged view contract in
[VICReg §6.2](vicreg.md#62-fixed-dgp-and-paired-execution). Do not transcribe
either generator or transformation. Its reflection, sign flip and empirical
nuisance redraw preserve the analytic propensity and conditional outcome means.
Verify maximum target error <=2e-5, nonidentity, independent branch draws,
unchanged row metadata, and training-only donor/scaler provenance.

For replicate `i=0..9`, use `base=390000+100*i`; train seed `base+1`, test
`base+2`, model seed `base+6`, execution seed `base+10000`, row offsets 0 and
10000. One shared MCAR mask supplies exactly 40 observed treatments. Hidden
treatments are evaluation-only. Use the same fitted training scale in all arms.

Run full, off-diagonal weight exactly zero, no-pretraining and contextual
VICReg arms. The first two differ only in that weight; keep the zero-weight
objective declared. Match initial encoder/projector/head tensors, actual row
streams and cached view draws, not merely seed arguments. The no-pretraining
arm removes the stage and inheritance edge and adjusts the execution seed by
the existing `STREAM_STRIDE` to match stage index 1's downstream stream.
Copy identical encoder and head initial states into contextual VICReg, since
its different projector consumes RNG differently. Its coefficients are the
current card's (25,25,1); no acceptance metric requires beating it.

Train each arm once per seed at §4's budget. Pretraining uses training mode;
evaluate its final checkpoint before fine-tuning, with hidden BN in eval mode
and frozen training buffers. Form 16 disjoint held-out batches of 128 rows and
two views with RNG seeds `base+20000+2*b`, `base+20001+2*b`, `b=0..15`, shared
across arms. Recompute §3.1's stateless correlation per held-out batch. This
does not fit persistent model state. Fine-tuned predictions use clean inputs.
No checkpoint selection, early stopping, sweeps or test-based retuning.

### 6.3 Tier 0 and Tier 1 contracts

Tier 0: independently compute values and input gradients with scalar loops on
nondegenerate float64 `B=5,d=3` arrays: unequal variances, nonzero means,
asymmetric cross-correlation, and one near-constant column to expose epsilon.
Check correction=0, denominator B, both branch gradients, all ordered
off-diagonals, and summed coordinate reductions. Check view exchange symmetry,
common row-permutation invariance and independent shifts; do not assert exact
scale invariance with nonzero epsilon. At constant inputs expect finite D=d,
O=0; reject B<2. Losses must change when one branch's row pairing changes.

Verify bias-free projector linears, hidden affine BN/ReLU, signed non-unit
output, cached-draw identity, no labels read in pretraining, stage ownership,
encoder transfer, fresh optimiser, unchanged initial heads and no downstream
projector execution. Check applicable §4 leaves by value, epsilon/correction
in plan details, training-only view provenance and `ExternalBatches` rejection.

See each relevant assertion fail under mutants: sample variance instead of
population variance; B-1 denominator; feature-mean rather than sum reduction;
within-view covariance instead of cross-correlation; diagonal included in O;
upper triangle only; one branch detached; epsilon outside sqrt; hidden bias
enabled; fresh draw per objective; projector leaking into fine-tuning.
Every mutant above was applied to the shipped source and seen to fail at
least one Tier 0 assertion.

Tier 1 bases `[419, 523, 631]` use the same offsets, split sizes and four arms,
with 200 pretrain/300 downstream steps and the unchanged 1000-step marginal
ramp. Assert finite losses/gradients, normalised treatment probabilities,
changed encoder parameters and valid transitions on every seed. Report §6.4
metrics without directional smoke assertions or a reproduction status claim.

### 6.4 Metrics, feasibility and acceptance

For each held-out batch compute `a=mean_i C_ii`,
`r=sum_{i!=j} C_ij^2 / (d*(d-1))`, and active fraction `f`: the fraction of
coordinates whose raw embedding population variance exceeds `100*epsilon`
in **both** branches. Average each over 16 batches to obtain one value per
seed. The variance cutoff ensures normalisation is not dominated by epsilon.

Required bounds use ten independent seed observations, sample
`SE=sd(ddof=1)/sqrt(10)`, and paired differences before computing SE:

1. Full diagonal alignment: `mean(a_full)-SE >= 0.5`.
2. Full active dimensions: `mean(f_full)-SE >= 0.9`.
3. Redundancy attribution: `mean(r_diagonal_only-r_full)-SE >= 0.01`.
4. Outcome transfer guardrail: `mean(NLL_full-NLL_no_pretrain)+SE <= 0.05`
   nats/row, using factual Gaussian NLL on the shared training-standardised
   outcome scale, after the full downstream budget.

These are prospective project-local effect-size choices, not published
numbers or measured spread. The first two exclude an off-diagonal improvement
obtained by collapsing the full arm. The third asks for one percentage point
of mean squared off-diagonal correlation improvement. The outcome margin is
an explicitly retained VICReg policy, not a mathematical consequence of this
objective. No claim of an interaction effect follows from one-term ablation.

At B=128,d=512, a centred cross-product has rank at most 127: C=I is impossible.
Do not require zero loss or zero off-diagonal energy. Barlow Twins' approximate
scale invariance also makes VICReg's raw spread >=0.5 an unsuitable inherited
target; use the epsilon-relative activity guard above instead. Report raw
variances/norms, diagonal error D/d, redundancy, covariance eigenvalue
concentration, and both branches' variances to distinguish scale from rank.

All required metrics must be finite for every seed. Report every arm's absolute
values and paired differences; treatment NLL, conditional-mean treatment-effect
error and contextual VICReg comparisons are informational. A failed bound
triggers a fidelity/fixture audit and `deviating`, not a wider tolerance.
Bind all protocol scalars to the benchmark configuration and digest. Run from
a committed implementation; benchmark registration, results and status/ledger
update land together. A successful status refers only to this local mechanism.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Paper/code lambda and std conventions disagree | Code 0.0051, population variance and epsilon=1e-5 | Pinned parser and output BatchNorm; deviation 1. |
| Appropriate tabular view distribution | Explicit caller views and imported oracle fixture symmetries | VICReg's accepted contract; deviation 3, no automatic generalisation. |
| Appropriate small-width/batch optimisation and thresholds | Fixed local budget, unchanged source lambda and prospective §6.4 bounds | Controlled experiment, not an optimality claim; deviations 2/4. |
| Source projector bias/BN defaults not fully described in prose | Every linear bias off; explicit PyTorch BN defaults and reset initialisation | Pinned `BarlowTwins.__init__`, not the superficially similar VICReg component. |
| Implementation comments misread centring and source sections | Both paper and code centre embeddings; nonzero epsilon still distinguishes code from eq. (2). Optimisation is §2.2; projector ablations are in §4. | Paper §2.1 assumes centred embeddings, Algorithm 1 centres them explicitly, and the pinned output BN adds epsilon. Corrected the comments without changing the selected arithmetic. |
| Test-time embedding-statistic convention | Frozen hidden BN, fresh stateless output normalisation per fixed evaluation batch | Measure the training correlation functional without changing inference state. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Repository owner | 2026-09-09 |
| Plan diffed against §3.2 and §4 | Claude | 2026-09-09 |
| Recipe implemented, Tier 0 passing (status → `implemented`) | Claude | 2026-09-09 |
| Source, plan and implementation reviewed; input validation corrected | Codex | 2026-09-09 |
| Tier 1 study run on bases 419/523/631 (status → `smoke-passing`) | Claude | 2026-09-09 |
| Tier 1 pairing/scaler/buffer checks strengthened and effect diagnostic added | Codex | 2026-09-09 |

Drafted from pinned author source on 2026-09-09. Review accepted the
prospective §6.4 bounds, the explicit oracle-view scope and §5.1's two
additions, and made one amendment, which is presentation rather than method:
§4's `data.standardisation` and `data.outcome_scaling` were written as quoted
YAML strings, and the card/plan cross-check compares the cell as written, so
the surrounding quotes were a mismatch against the identical value `DataSpec`
composes. They are now written the way `vicreg.md` §4 writes them, and all
sixty-six non-`n/a` §4 leaves are compared by value rather than by presence.

Status is now `smoke-passing`. §6.3's Tier 1 packet ran on bases 419, 523 and
631 at the declared 200/300-step overrides with all four §6.2 arms, and it
asserts wiring only: finite losses and gradients, a normalised propensity, the
encoder transfer and stage transition, the arms' shared initial tensors, row
streams and cached view draws, and no projector in fine-tuning. §6.4's metrics
are reported beside it and none of its four bounds is assessed, because each is
a ten-seed statement at §4's full budget. Eight mutants were injected one at a
time and each was seen to fail a named assertion; the experiment note lists
them with the measured arms.

No benchmark module, `RECIPES` entry or §6.1 ledger row is added, since
`CLAUDE.md` requires those three to land with a result rather than ahead of
one. §6.1's placeholder row is therefore still the template's empty row.

The implementation review added rejection of scalar, zero-width and mismatched
embedding shapes: mismatched widths previously let the diagonal term silently
score only the shared prefix of a rectangular cross-correlation. All three
regressions failed on the original implementation. An independent training-mode
`BatchNorm1d` oracle now checks both terms' values and input gradients, including
the near-constant column. It rejects a sample-variance mutant and a
correct-value/half-gradient mutant. Source-comment corrections also distinguish
collapse's `D=d` from a maximum, epsilon-attenuated self-correlations from exact
ones, and mean raw variance diagnostics from §6.4's active-coordinate fraction.
These corrections preserve §3–§6's reviewed method and acceptance bounds.

The [Tier 1 review](../experiments/2026-09-09-barlow-twins-smoke-review.md)
checks actual treatment-mask identities across arms, all four fitted scaling
statistics, and checkpoint buffers as well as parameters. It adds the §6.4
conditional-mean treatment-effect RMSE diagnostic in original outcome units.
All three declared seeds pass; the recipe and acceptance bounds are unchanged.
