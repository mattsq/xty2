# Recipe spec card: byol

**Status:** `draft`

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 for benchmark and reporting work.

Selected from [BACKLOG.md §5.1](../BACKLOG.md). This packet specifies a
prospective study and stops for card review. No callable or result exists yet.

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733v3) |
| Authors, year | Jean-Bastien Grill et al., 2020 |
| DOI / arXiv | arXiv:2006.07733 |
| Version used | v3, 2020-09-10; §3.1, §3.3, §5 and Appendix A |
| Reference implementation | [google-deepmind/deepmind-research/byol](https://github.com/google-deepmind/deepmind-research/tree/82a347438fd93bdd4bc01764258223ae583e5cbf/byol) @ `82a347438fd93bdd4bc01764258223ae583e5cbf` |
| Reference impl. runnable? | Not attempted; source inspected, no claim of reference execution. |

Pinned source anchors, all relative to that commit: `byol_experiment.py`
(`loss_fn`, `_update_fn`, `_make_initial_state`), `configs/byol.py`
(`get_config`, 1000-epoch preset), and `utils/{networks,helpers,schedules,optimizers}.py`
(`MLP`, `regression_loss`, `target_ema`, `learning_schedule`, `lars`).

## 2. Estimand and claim

- **Estimand:** the paired change in held-out factual outcome NLL when a
  cross-view representation learner uses a slowly moving target instead of
  copying the online parameters after every update, with architecture, loss,
  initialisation, batches, views and downstream training fixed.
- **Published claim:** useful image representations can be learned by predicting
  target embeddings without negative pairs. The target update and predictor
  are empirically studied, not a general non-collapse theorem.
- **Local claim to test:** scheduled EMA improves downstream outcome NLL over
  the zero-decay control, retains nonconstant encoder features, and costs no
  more than 0.05 nat/row against training without pretraining. These are
  prospective hypotheses; an EMA advantage is not assumed.
- **Not claimed:** ImageNet reproduction, better causal identification,
  universal prevention of collapse, a usable general tabular augmentation
  policy, or an advantage at equal total compute.

The nearest shipped card is [SimSiam](simsiam.md). Its teacher-free target
and predictor make the mechanism comparison useful, but changing directly from
its published topology to BYOL changes several factors. The primary control
therefore uses BYOL's own topology with only target decay changed. Calling that
arm a reproduction of SimSiam would be incorrect. The no-predictor arm is a
separate diagnostic, not a second primary superiority claim.

## 3. Equations and mapping

### 3.1 As published

For views v=t(x), v'=t'(x), the paper uses y_theta=f_theta(v),
z_theta=g_theta(y_theta), and z'_xi=g_xi(f_xi(v')). Its equations are:

```text
(1) xi <- tau * xi + (1-tau) * theta
(2) L_theta,xi = || normalized(q_theta(z_theta)) - normalized(z'_xi) ||_2^2
              = 2 - 2 <q_theta(z_theta), z'_xi> / (||q_theta(z_theta)|| ||z'_xi||)
    L_BYOL = L_theta,xi + L_tilde_theta,xi       [swap v and v']
(3) theta <- optimizer(theta, grad_theta L_BYOL, eta)
```

Only theta receives gradients; (1) follows (3). The implementation averages
the summed directional squared distances over rows. It does not divide the
sum by two. Targets are projections, never target predictions.

The pinned `helpers.l2_normalize` divides each vector by
sqrt(max(sum(vector**2), 1e-12)). Its squared-distance implementation also
defines behaviour below that floor, where the second equality in (2) need
not hold. Preserve that implementation, including the sum over coordinates.

### 3.2 Mapping to xty2

Names marked proposed below are implementation scope, not existing APIs.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| v,v' | two views of one source row | `X_RAW` in views `byol_a`, `byol_b` | two `ViewSpec`s; explicit caller transforms, study uses `OracleSymmetry` |
| f_theta | online backbone | `X_RAW -> X_REPR` | existing `MLPEncoder`, four 256-wide ReLU layers |
| g_theta | online projection | `X_REPR -> X_PROJ` | proposed `BYOLProjector`, 256 -> 4096 -> 256 |
| q_theta | online predictor | `X_PROJ -> X_PRED` | proposed `BYOLPredictor`, 256 -> 4096 -> 256 |
| f_xi,g_xi | target backbone/projector | teacher realisations of `X_REPR`, `X_PROJ` | existing stage-owned `TeacherSpec` / `EMATeacher` |
| L_theta,xi | a predicts target b | online `X_PRED@byol_a`, teacher `X_PROJ@byol_b` | proposed `NormalizedSquaredFeatureConsistency`, weight 1 |
| L_tilde_theta,xi | b predicts target a | online `X_PRED@byol_b`, teacher `X_PROJ@byol_a` | same objective class, distinct name, weight 1 |
| tau | target update schedule | n/a | proposed `CosineEMADecay`, §4 |
| optimizer | online update | n/a | proposed LARS option in `OptimiserSpec`, existing gradient executor |
| transferred f_theta | downstream encoder | identity `X_REPR` | `Stage(joint_fit, initialise_from=pretrain)` |
| local outcome | factual conditional density | `Y_GIVEN_XT` | `TARNetHead`, `ObservedOutcomeNLL` |
| local propensity | categorical treatment law | `T_GIVEN_X` | `CategoricalPropensity`, `ObservedTreatmentNLL` |
| local missing-t likelihood | exact treatment enumeration | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path=both)` |

Use two ordinary gradient stages, `UniformSampler(128)`, `Weighted`,
`WarmupCosine`, `Ramp`, `Constant` and explicit `DataSpec` preprocessing and
MCAR missingness. Pretraining trains only encoder/projector/predictor and
reads neither outcomes nor treatment values. Fine-tuning trains only the
online encoder and XTY heads, with a fresh Adam optimiser and no teacher.
Preserve all row identifiers, supervision masks, weights and split fields in
both views. No synthetic rows, bank, pseudo-label artifact or new executor.

The target graph may contain unused copied predictor/head modules because
`EMATeacher` copies the graph. They must not run or influence the loss. Copy
online state at teacher construction, then compute both online views and both
target views once each, sum losses, update online parameters, and update target
parameters exactly once. BN forwards happen in a-then-b order with separate
view statistics and cached reuse. Never update the teacher between directions.
Target BN statistics belong to target forwards, not to parameter EMA. In
downstream inference export the online encoder, not the target or projector.

## 4. Mechanics checklist

This block defines the full local arm. References in comments distinguish
source settings from the explicit adaptations in §5. All applicable keys must
bind into the future compiled plan; this draft cannot yet be compiled.

```yaml
gradients:
  stop_gradients: target projection in each direction; no online prediction detach # eqs (1)-(3)
  detached_targets: true # loss_fn
  gradient_clipping: none # reference lars pipeline
  marginal_nll_grad_path: both # local downstream, deviation 4

teacher:
  ema_decay: 1-(1-0.996)*(1+cos(pi*k/1000))/2; k=0..999 # section 3.3, schedules.target_ema; horizon deviation 3
  ema_applies_to_buffers: false # _update_fn returns separately forwarded target_state
  teacher_in_train_mode: true # loss_fn is_training=True for both networks
  teacher_requires_grad: false # eq (3)

losses:
  reduction: pretrain mean over rows after coordinate sum; downstream population # loss_fn; deviation 4
  eligible_rows: pretrain all train rows; outcome t/y observed; treatment t observed; marginal t missing/y observed
  weights: pretrain directions 1 each; downstream outcome 1, treatment 1, marginal 0 to 0.5 # loss_fn; deviation 4
  schedules: pretrain constant; downstream marginal linear ramp over first 1000 steps # deviation 4
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a

optimisation:
  optimiser: pretrain LARS momentum=0.9 eta=0.001 no Nesterov; downstream Adam betas=(0.9,0.999) eps=1e-8 # configs/byol.py; deviation 4
  lr: pretrain 0.2*128/256=0.1; downstream 0.001 # 1000-epoch preset; deviations 3,4
  lr_schedule: pretrain WarmupCosine start=0 warmup=10 steps=1000; downstream constant # learning_schedule, compressed horizon
  weight_decay: pretrain 1.5e-6 excluding biases and BN from both decay and LARS adaptation; downstream none # 1000-epoch preset
  batch_size: 128 for both stages # deviation 3
  labelled_unlabelled_ratio: no fixed quota; uniform sampling of shared population with 40 observed treatments # deviation 4
  total_steps_or_epochs: pretrain 1000 optimiser steps; joint_fit 3000 optimiser steps # deviations 3,4

architecture:
  widths_depths: encoder [256,256,256,256]; projector [4096,256]; predictor [4096,256]; outcome [100,100,100]; linear propensity # deviation 1; networks.MLP
  activation: encoder ReLU; projector/predictor hidden ReLU only; outcome ELU; propensity linear logits
  normalisation: encoder none; each projector/predictor hidden affine BN eps=1e-5 momentum=0.1; no output BN; heads none # networks.MLP; deviations 1,6
  dropout: none # source networks; local backbone
  initialisation: encoder/projector/predictor torch Linear default; BN scale=1 bias=0; target copies online; heads CFRNET_INITIALISATION # deviations 5,6
  output_parameterisation: raw 256-vector projection/prediction; outcome K means with fixed Gaussian scale=1; K softmax propensity logits

data:
  standardisation: zscore features fitted once on training population only # deviation 4
  outcome_scaling: train-only zscore; primary NLL on common training scale; original-scale metrics diagnostic # deviation 4
  treatment_encoding: integers 0..K-1 with K=2 # local fixture
  split_protocol: disjoint generated train/held-out populations; final checkpoint; no test selection # section 6.2
  missingness_mechanism: treatment MCAR to exactly 40 training rows, shared row_id-keyed mask; all outcomes observed # deviation 4
```

LARS must match `optimizers.lars`: add the filtered weight decay to the
gradient, apply eta*parameter_norm/update_norm when both norms are positive
(otherwise multiplier 1), then momentum accumulation, then multiply by -LR.
Excluded bias and BN parameters still receive ordinary momentum updates.
Do not substitute AdamW, damped trust ratios or decay outside this sequence.

The target schedule consumes zero-based pretraining step k, including the
post-update EMA at k=999. Evaluate it in sufficient precision to keep every
used value below 1. The unused mathematical endpoint k=1000 is 1 and must
not be passed to the existing teacher, which rejects decay=1. This is an
indexing contract, not a licence to clamp or alter the curve.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | | Four-layer 256-wide unnormalised tabular backbone replaces ResNet-50. Preserve source projector/predictor widths and two-layer topology. | Local XTY experiment. | Changes representation scale and capacity; inspect norm and BN regime before interpreting failure. |
| 2 | `judgement` | | Independent symmetric OracleSymmetry views replace asymmetric image augmentation distributions. | Analytically valid transformations on this fixture. | Privileged DGP knowledge limits transfer; cannot establish general tabular usefulness. |
| 3 | `judgement` | | Batch 128, 1000 steps and 10 warmup steps replace batch 4096, 1000 epochs and 10 warmup epochs. Retain the selected source preset's LR scaling, decay, LARS and EMA base. | Bounded single-process experiment; warmup retains its 1% horizon fraction. | The number of EMA time constants and optimiser trajectory differ substantially; 0.996 is a tested hypothesis here, not an inherited guarantee. |
| 4 | `judgement` | | Shared train scaling, 40 MCAR labels and 3000-step XTY fine-tuning replace image linear evaluation. Omit the reference's detached online monitoring classifier. | Measure downstream missing-treatment prediction. The source classifier never trains the backbone. | No image accuracy or causal identification claim follows. |
| 5 | `judgement` | | Initialise target as an exact online copy, following Appendix A, rather than the code's independent RNG draws. | Select the paper's initialisation and make the EMA ablation paired. The code comments report no significant difference. | Changes early targets; must be recorded and checked, not conflated with decay. |
| 6 | `judgement` | | Explicit PyTorch Linear initialisation and single-device PyTorch BN replace Haiku defaults and cross-replica BN. | Fixed local backend. We would retain this adaptation with an unlimited framework. Hidden BN keeps source epsilon and old-statistic decay 0.9; no final BN. | Initial norms and running-variance estimators can alter dynamics. Both arms share them; source numerical equivalence is not claimed. |

### 5.1 Framework additions made for this card

These are proposed for implementation after review, not implemented by this
documentation PR. No source mechanic is being omitted for a framework gap;
the reversible additions below are required before declaring `implemented`.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| BYOLProjector and BYOLPredictor | Fidelity-bearing, reversible | proposed BYOL | n/a | Existing SimSiam output BN and bottleneck topology are different. Use biased hidden linear, affine BN, ReLU, then bias-free final linear, as networks.MLP. |
| NormalizedSquaredFeatureConsistency | Fidelity-bearing, reversible | proposed BYOL | n/a | Existing negative cosine lacks source squared-norm floor and literal squared-distance values near zero. Keep ports, row scopes and target detach explicit. |
| CosineEMADecay schedule | Fidelity-bearing, reversible | proposed BYOL | n/a | Existing Schedule/TeacherSpec can carry the curve but no existing primitive represents its affine complement. Bind base, horizon and index convention in plan details. |
| LARS optimiser option | Fidelity-bearing, reversible | proposed BYOL | n/a | OptimiserSpec currently supports SGD/Adam families; preserve source trust-ratio and decay ordering in that existing boundary. Bind eta, momentum and exclusions in optimisation metadata. |

No new port or executor contract is proposed. BYOL is the named second
consumer that SimSiam's `X_PRED` addition anticipated. Do not add an EMA
framework, generic SSL engine, checkpoint system or new card-key category.
If implementation discovers another missing mechanic, amend and review this
card first; if a mechanic is omitted, add typed debt and reconcile DESIGN §11.4.

## 6. Reproduction target

This is a project-local mechanism study. All bounds below are unmeasured and
prospective. They are not paper numbers or values copied from SimSiam results.

```yaml
reproduction:
  dataset: common.two_cluster_population with low=SEPARATED; six features; K=2; OracleSymmetry views
  variant: scheduled EMA versus zero-decay target, no predictor, and no pretraining
  split: 1024 train, 40 observed treatments; 2048 fully observed held-out rows
  metric: ema_outcome_nll_gain; pretraining_outcome_nll_cost; encoder_effective_rank
  published: none - project-local tabular adaptation
  published_source: n/a
  tolerance: all three one-standard-error bounds in section 6.4
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

### 6.2 Fixed DGP and paired execution

Import `two_cluster_population`, `continuous_schema`, `training_dataset` and
`on_the_training_scale` from `xty2.evaluation.benchmarks.common`, and
`OracleSymmetry` with its preservation tolerance from
`xty2.evaluation.vicreg_views`. Adopt SimSiam §6.2's separated fixture unchanged,
including `low=SEPARATED`; do not transcribe the generator or transforms.
Assert analytic treatment probabilities and conditional outcome means are
unchanged by the views before any fit. Oracle information is evaluation-only.

Tier 2 replicate i=0..9 uses base=620000+100*i; training seed base+1,
held-out seed base+2, model seed base+6, execution seed base+10000.
Training row offsets start at 0; held-out offsets start at 10000.
Fit preprocessing once and share the actual 40-label mask across arms, using
the downstream execution seed base+10000+STREAM_STRIDE for that draw. Hidden
treatment values must never reach either fit stage.

| Arm | Change from full | Interpretation |
|---|---|---|
| full | none | scheduled EMA and learned predictor |
| zero-decay | tau=0 at every update | target copies the updated online parameters; same stop-gradient and independently forwarded target BN |
| no-predictor | route online X_PROJ directly to loss; remove predictor from trainables | predictor mechanism diagnostic, retaining EMA |
| no-pretraining | omit pretrain and inheritance edge | effect of the additional pretraining phase; not compute-matched |

Each pretraining arm gets 1000 steps. Every arm gets 3000 downstream steps.
The no-pretraining arm uses the existing STREAM_STRIDE offset to match stage
index 1's downstream stream. Copy common initial tensors and downstream head
states or use isolated RNG streams; equal global seeds alone are insufficient.
Assert actual batch/view traces, fitted scales and masks match across arms.
The zero-decay target is a separate graph and is not a shared online forward:
its BN buffers remain separately owned. Copying online buffers after each
step would introduce another intervention and is prohibited.

At pretrain end, before fine-tuning, evaluate the online encoder in eval mode
on all 2048 clean held-out rows, with frozen training BN statistics and no
recalibration. Directional loss diagnostics use 16 batches of 128 and views
seeded base+20000+2*b and base+20001+2*b; realise once and share across arms.
Downstream evaluation uses clean identity rows at the final checkpoint.
No early stopping, test-based selection, or additional seed search.

### 6.3 Tier 0 and Tier 1 evidence

Tier 0 requirements before implementation status:

- Independent scalar and autograd oracles for squared distances, both
  directional weights and the 1e-12 squared-norm floor, including zero and
  near-zero vectors. A cosine-only shortcut must fail these tests.
- Nonzero online encoder/projector/predictor gradients and absent target
  gradients; target outputs use X_PROJ, not X_PRED. Check B != K, empty eligible
  scopes, shape errors, non-finite inputs and singleton training BN rejection.
- Hand-computed two-step LARS updates including bias/BN exclusions, zero
  norms, momentum and weight decay ordering; independent schedule checks at
  k=0, warmup boundary, midpoint and k=999.
- Exact initial teacher copy, one EMA update after the optimiser, both views
  using the same target parameters, independently updated BN buffers, one
  forward per network/view and no teacher mutation from diagnostic reads.
- Card/plan value agreement, absence of label/outcome lineage in pretraining,
  only encoder transfer to active downstream computations, fresh downstream
  optimiser, shared head initialisation and matched streams across controls.
- Demonstrate effective-rank handling on constant, rank-one and isotropic
  tensors. NaN or infinite metrics fail, including in an ablation arm.

When implementing these assertions, observe failures under mutants that halve
the loss, detach the prediction, use target predictions, EMA-update buffers,
update the teacher between directions, move weight decay after adaptation,
apply adaptation to BN, or shift the EMA schedule by one step. Record mutants
in the implementation commit; this draft does not claim those tests exist.

Tier 1 uses seeds 42, 43 and 44, 512 training rows, 512 held-out rows, 40
observed treatments, 100 pretraining and 200 downstream steps; batch 128.
Rebase warmup to 1 step, EMA horizon to 100 and marginal ramp to 200, explicitly
binding smoke overrides. Run all four arms. Require finite losses/gradients,
source update identities, actual target movement and successful transfer on
every seed. Record norms, BN variance relative to epsilon, rank, predictor
residual and outcome NLL. Do not assert EMA superiority or ablation collapse
in a wiring test. Any proposed directional assertion needs a reviewed card
amendment and evidence on every declared smoke seed.

### 6.4 Metrics, tolerances and interpretation

For each replicate let N_arm be mean held-out factual outcome NLL on the
common training-standardised outcome scale. Define:

1. `ema_outcome_nll_gain = N_zero_decay - N_full`; require mean - SE > 0.
2. `pretraining_outcome_nll_cost = N_full - N_no_pretraining`; require
   mean + SE < 0.05 nat/row. This is a declared local non-inferiority budget.
3. `encoder_effective_rank`: centre the full arm's [2048,256] pre-fine-tuning
   encoder matrix by column; take singular values s and probabilities
   p_j=s_j^2/sum(s^2); report exp(-sum(p_j*log(p_j))) with 0*log(0)=0.
   If sum(s^2)<=1e-12, return 0. Require mean - SE > 1.1. This excludes
   constant and essentially rank-one solutions without requiring 256
   independent factors from the fixture's limited invariant information.

Compute each paired difference within seed, then sample SE=sd/sqrt(10),
ddof=1. Report every arm and replicate, including failures; do not pool rows
as independent replicates. All three criteria must pass for the local
`reproduced` status under FIDELITY §3; a nominal pass within its uncertainty
is `deviating`. These are repository evidence gates, not a multiple-testing
controlled statistical claim.

Report no-predictor differences, treatment NLL, treatment-effect RMSE,
normalised feature spread, rank of every arm, feature norms, directional
residuals and target-online lag as diagnostics. Rank alone does not show
useful learning; the paired outcome criterion supplies that test. Do not
require zero-decay or no-predictor collapse: neither is guaranteed here.
If EMA adds no measurable benefit, audit source fidelity and record the
negative result. Any changed fixture, threshold, optimiser or seed stream
requires a prospective amendment with the failed protocol retained.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Paper initial copy versus code independent target initialisation | Select Appendix A's copy and disclose deviation from pinned code | Explicit variant resolution, §5.5 |
| Numerical normalisation at zero | Pinned helper's squared-norm floor 1e-12 and literal squared distance | Source code resolves equation's undefined zero case |
| PyTorch equivalent of unpinned Haiku Linear defaults | Explicit torch Linear defaults; no numerical equivalence claim | Backend judgement §5.6, not an assumed source initialiser |
| Appropriate tabular widths, view policy and short-horizon EMA base | Fixed §4 settings and §6 fixture; no post-hoc tuning | Prospective local choices; departures §5.1–§5.4 |
| Whether moving targets help this low-dimensional fixture | Unresolved; measured by the declared zero-decay pair | This is the experiment, not a source fact |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |

Review must settle the proposed LARS and loss contracts, target initialisation,
BN policy and the prospective attribution bound. No execution-plan rendering
is available before implementation; do not present a schematic as compiler
output. Implement only after this draft is reviewed.
