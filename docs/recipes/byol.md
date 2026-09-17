# Recipe spec card: byol

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 for benchmark and reporting work.

Selected from [BACKLOG.md §5.1](../BACKLOG.md). `xty2.recipes.byol` implements
§3–§4 and `tests/invariants/test_byol.py` is its core Tier 0 suite.
`tests/smoke/test_byol.py` implements the §6.3 Tier 1 study, with arm and
diagnostic invariants in `tests/invariants/test_byol_study.py`. Tier 2 is
registered in `xty2.evaluation.benchmarks.byol`, with its nightly adapter and
protocol/result invariants.

The 2026-09-16 ten-seed run is `deviating` and is retained in §6.1 with the
protocol that produced it. The 2026-09-17 audit
([report](../experiments/2026-09-17-byol-audit.md)) found no source-mechanism
error and established instead that the instrument carrying the EMA claim could
not carry it. Three amendments follow from that audit — §5 row 7 re-derives the
target decay against this card's own horizon, §5 row 4 restores the source's
frozen-backbone evaluation, and §6.4 moves the attribution onto pre-transfer
statistics — and §6.2 re-runs them on a seed stream disjoint from every seed the
audit saw. See also the
[2026-09-16 review and evidence](../experiments/2026-09-16-byol-tier2.md).

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
- **Local claim to test** (amended 2026-09-17, §6.4): a slowly moving target
  holds the representation further from the objective's degenerate optimum than
  a target that copies the online parameters — measured before transfer, by view
  alignment and encoder effective rank — while retaining nonconstant features and
  costing no more than 0.05 nat/row against training without pretraining. These
  are prospective hypotheses; an EMA advantage is not assumed. The withdrawn
  claim, that scheduled EMA improves *downstream* outcome NLL over the zero-decay
  control, is retained as an informational statistic and in §6.1's 2026-09-16 row.
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

Every name below exists; the four added for this card are §5.1's.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| v,v' | two views of one source row | `X_RAW` in views `byol_a`, `byol_b` | two `ViewSpec`s; explicit caller transforms, study uses `OracleSymmetry` |
| f_theta | online backbone | `X_RAW -> X_REPR` | existing `MLPEncoder`, four 256-wide ReLU layers |
| g_theta | online projection | `X_REPR -> X_PROJ` | `BYOLProjector`, 256 -> 4096 -> 256 |
| q_theta | online predictor | `X_PROJ -> X_PRED` | `BYOLPredictor`, 256 -> 4096 -> 256 |
| f_xi,g_xi | target backbone/projector | teacher realisations of `X_REPR`, `X_PROJ` | existing stage-owned `TeacherSpec` / `EMATeacher` |
| L_theta,xi | a predicts target b | online `X_PRED@byol_a`, teacher `X_PROJ@byol_b` | `NormalizedSquaredFeatureConsistency`, weight 1 |
| L_tilde_theta,xi | b predicts target a | online `X_PRED@byol_b`, teacher `X_PROJ@byol_a` | same objective class, distinct name, weight 1 |
| tau | target update schedule | n/a | `CosineEMADecay`, §4 |
| optimizer | online update | n/a | `OptimiserSpec(name='lars')` and `core.optimisation.LARS`, existing gradient executor |
| transferred f_theta | downstream encoder | identity `X_REPR` | `Stage(joint_fit, initialise_from=pretrain)` |
| local outcome | factual conditional density | `Y_GIVEN_XT` | `TARNetHead`, `ObservedOutcomeNLL` |
| local propensity | categorical treatment law | `T_GIVEN_X` | `CategoricalPropensity`, `ObservedTreatmentNLL` |
| local missing-t likelihood | exact treatment enumeration | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path=both)` |

Use two ordinary gradient stages, `UniformSampler(128)`, `Weighted`,
`WarmupCosine` with an explicit final multiplier of 0, `Ramp`, `Constant` and
explicit `DataSpec` preprocessing and MCAR missingness. Pretraining trains only encoder/projector/predictor and
reads neither outcomes nor treatment values. The downstream stage transfers the
online encoder and holds it **frozen**, declaring only the XTY heads trainable,
with a fresh Adam optimiser and no teacher (§5 row 4, as amended: this is the
source's linear-evaluation posture, §3.3). An encoder that moved during
`joint_fit` would be a different protocol, so §6.2 checks the transferred
backbone is bit-identical after the stage rather than trusting the declaration.
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

This block defines the full local arm and is the compiled plan, key for key:
`pretrain` is the source's, `joint_fit` is deviation 4's local protocol.
Comments cite the pinned file the value comes from, or the §5 row that changed
it.

```yaml
gradients:
  stop_gradients:  # eqs (1)-(3); `loss_fn`'s jax.lax.stop_gradient on the target projection only
    pretrain.byol_a_to_b: x_proj @ view=byol_b params=teacher
    pretrain.byol_b_to_a: x_proj @ view=byol_a params=teacher
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target
  gradient_clipping:
    pretrain: none                                # the reference lars chain clips nothing
    joint_fit: none
  marginal_nll_grad_path:
    joint_fit.missing_treatment_marginal_nll: both   # deviation 4
teacher:
  # `schedules.target_ema` with max_steps 1000, evaluated at the zero-based
  # pretraining step. The base is 0.68722, deviation 7's re-derivation of
  # `_EMA_PRESETS` against this horizon, not the inherited 0.996. The scalar
  # bound here is the decay at the last applied step k=999; the curve itself is
  # the `decay_schedule=` field of the plan's teacher line,
  # `1 - (1 - 0.68722) * 0.5 * (1 + cos(pi * min(k/1000, 1)))`.
  # See the indexing note below for why the k=1000 endpoint is not this number.
  ema_decay:
    pretrain: 0.9999992282469186
  ema_applies_to_buffers:
    pretrain: false                               # `_update_fn` returns a separately forwarded target_state
  teacher_in_train_mode:
    pretrain: true                                # `loss_fn` applies both networks with is_training=True
  teacher_requires_grad:
    pretrain: false                               # eq. (3)
losses:
  reduction:  # `loss_fn`: one jnp.mean over rows of the summed directional distances
    pretrain.byol_a_to_b: mean
    pretrain.byol_b_to_a: mean
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    pretrain.byol_a_to_b: all
    pretrain.byol_b_to_a: all
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:  # `repr_loss = a + b`, not (a + b) / 2; deviation 4 for joint_fit
    pretrain.byol_a_to_b: 1.0
    pretrain.byol_b_to_a: 1.0
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    pretrain.byol_a_to_b: constant 1.0
    pretrain.byol_b_to_a: constant 1.0
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps   # deviation 4
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a
optimisation:
  optimiser:
    pretrain: lars(momentum=0.9, eta=0.001, adaptation on parameters of rank two or more)   # `optimizer_config`; the filter is `optimizers.exclude_bias_and_norm`
    joint_fit: adam(betas=(0.9, 0.999), eps=1e-08)                                          # deviation 4
  lr:
    pretrain: 0.1                                 # _LR_PRESETS[1000] = 0.2, scaled by batch/256 in `learning_schedule`; deviations 3, 4
    joint_fit: 0.001
  lr_schedule:
    pretrain: warmup cosine 0.0 -> 1.0 over 10 steps, then cosine -> 0.0 at 1000 steps   # `learning_schedule`, whose post-warmup half cosine reaches zero at the horizon; deviation 3
    joint_fit: constant 1.0
  weight_decay:
    pretrain: 1.5e-06 (all trainable components; norm and bias exempt)   # _WD_PRESETS[1000]; `exclude_bias_and_norm` also exempts them from the trust ratio
    joint_fit: none
  batch_size:
    pretrain: 128                                 # deviation 3; the source's preset is 4096
    joint_fit: 128
  labelled_unlabelled_ratio: n/a                  # UniformSampler enforces no quota
  total_steps_or_epochs:
    pretrain: 1000                                # deviation 3: optimiser steps, not the source's 1000 epochs
    joint_fit: 3000                               # deviation 4; heads only, frozen encoder
architecture:
  widths_depths:  # deviation 1 for the encoder; `configs/byol.py` sizes for both heads
    mlp_encoder: [256, 256, 256, 256]
    byol_projector: [4096, 256]
    byol_predictor: [4096, 256]
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: relu
    byol_projector: hidden relu; output linear
    byol_predictor: hidden relu; output linear
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:  # bn_config decay_rate 0.9 is torch momentum 0.1; deviations 1, 6
    mlp_encoder: none
    byol_projector: hidden BN affine=true eps=1e-5 momentum=0.1 track_running_stats=true; output none
    byol_predictor: hidden BN affine=true eps=1e-5 momentum=0.1 track_running_stats=true; output none
    tarnet_head: none
    categorical_propensity: none
  dropout:  # no source network uses any
    mlp_encoder: 0.0
    byol_projector: 0.0
    byol_predictor: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:  # deviations 5, 6; `networks.MLP` biases the hidden layer and not the output
    mlp_encoder: torch Linear default Kaiming-uniform
    byol_projector: torch Linear reset_parameters; hidden bias=true; output bias=false; BN weight=1,bias=0,running_mean=0,running_var=1
    byol_predictor: torch Linear reset_parameters; hidden bias=true; output bias=false; BN weight=1,bias=0,running_mean=0,running_var=1
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits
data:  # deviations 2 and 4
  standardisation: "x: zscore fitted on 'train'"
  outcome_scaling: "y: zscore fitted on 'train'"
  treatment_encoding: n/a                         # XTYBatch supplies integer classes 0..K-1, K=2 on this fixture
  split_protocol: fixed two-cluster DGP; disjoint train and held-out populations; no test-based selection; training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 40 labelled rows, keyed by row_id
```

LARS must match `optimizers.lars`: add the filtered weight decay to the
gradient, apply eta*parameter_norm/update_norm when both norms are positive
(otherwise multiplier 1), then momentum accumulation, then multiply by -LR.
Excluded bias and BN parameters still receive ordinary momentum updates.
Do not substitute AdamW, damped trust ratios or decay outside this sequence.

The target schedule consumes zero-based pretraining step k, and its horizon is
the stage's own step budget, so the executor applies it at k=0..999 including
the post-update EMA at k=999. The mathematical endpoint k=1000 is exactly 1 —
a frozen target — and is never applied. `CosineEMADecay.nominal` therefore
reports the decay at k=999 rather than that endpoint, which is what the
`ema_decay` key above binds: `TeacherSpec` and `EMATeacher` both reject a decay
outside [0, 1), and those checks are about decays that run. This is an indexing
contract, not a licence to clamp or alter the curve, and Tier 0 checks the
horizon against the stage's `steps` rather than trusting the two to agree.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | | Four-layer 256-wide unnormalised tabular backbone replaces ResNet-50. Preserve source projector/predictor widths and two-layer topology. | Local XTY experiment. | Changes representation scale and capacity; inspect norm and BN regime before interpreting failure. |
| 2 | `judgement` | | Independent symmetric OracleSymmetry views replace asymmetric image augmentation distributions. | Analytically valid transformations on this fixture. | Privileged DGP knowledge limits transfer; cannot establish general tabular usefulness. |
| 3 | `judgement` | | Batch 128, 1000 steps and 10 warmup steps replace batch 4096, 1000 epochs and 10 warmup epochs. Retain the selected source preset's LR scaling, decay and LARS. **Amended 2026-09-17:** the EMA base is no longer among the values this row inherits; row 7 derives it. | Bounded single-process experiment; warmup retains its 1% horizon fraction. | The optimiser trajectory still differs substantially. The row's original claim that 0.996 was "a tested hypothesis here" understated the size of the change: see row 7. |
| 4 | `judgement` | | Shared train scaling, 40 MCAR labels and a 3000-step XTY head fit replace image linear evaluation. Omit the reference's detached online monitoring classifier. **Amended 2026-09-17:** the transferred encoder is frozen for that fit, where the original row fine-tuned it. | Measure downstream missing-treatment prediction. The source classifier never trains the backbone, and paper §3.3 evaluates a frozen one; the original row departed from both without saying so. | No image accuracy or causal identification claim follows. Freezing narrows this row toward the source: see §6.4's note on what the fine-tuned variant measured. |
| 5 | `judgement` | | Initialise target as an exact online copy, following Appendix A, rather than the code's independent RNG draws. | Select the paper's initialisation and make the EMA ablation paired. The code comments report no significant difference. | Changes early targets; must be recorded and checked, not conflated with decay. |
| 6 | `judgement` | | Explicit PyTorch Linear initialisation and single-device PyTorch BN replace Haiku defaults and cross-replica BN. | Fixed local backend. We would retain this adaptation with an unlimited framework. Hidden BN keeps source epsilon and old-statistic decay 0.9; no final BN. | Initial norms and running-variance estimators can alter dynamics. Both arms share them; source numerical equivalence is not claimed. |
| 7 | `judgement` | | Target decay base 0.68722 replaces the inherited `_EMA_PRESETS[1000] = 0.996`. Derived, not tuned: `configs/byol.py` keys every preset to an epoch budget and passes `target_ema` a horizon of `num_epochs * train_images_per_epoch // batch_size`, so the base is a function of run length. Preserving the one invariant the curve has — the EMA time constant as a fraction of the pretraining horizon, `1 - base_local = (1 - base_source) * steps_source / steps_local` — and selecting the source row this card's own budget reaches (1000 steps of batch 128 over 1024 rows is 125 epochs, so `_EMA_PRESETS[100] = 0.99` over 31278 steps) gives `1 - 0.01 * 31278 / 1000`. | Row 3 shortened the horizon by a factor of 313 while keeping a base the source ties to the long horizon. Measured consequence at 0.996: integrated tracking `sum(1 - tau)` of **2.00** over the whole run against 156-626 in every source preset, and a target that ends 0.86 in parameter norm behind the online network on all ten replicates of the 2026-09-16 run. That is a frozen early snapshot, not a slowly moving average. | Restores the source's tracking regime (local integrated tracking 156.5 against the 100-epoch row's 156.4). The selected 1000-epoch row is unreachable here at any base — the same rule sends it to -0.251, outside the `[0, 1)` an EMA update requires — which is why the row is reselected rather than rescaled. The `source_ema` arm of §6.2 keeps 0.996 so the change is measured rather than assumed. |

### 5.1 Framework additions made for this card

Each addition below is built, reversible, and reaches `plan.hyperparameters`
through an existing card key. No source mechanic is omitted for a framework
gap, so §5 carries no `framework-limitation` row and this card owes the
`DESIGN.md` §11.4 ledger nothing.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `BYOLProjector` and `BYOLPredictor` (`components/byol.py`) | Fidelity-bearing, reversible | byol | n/a | SimSiam's output BN and bottleneck topology are a different network. These are `networks.MLP`: biased hidden linear, affine BN, ReLU, then bias-free final linear. |
| `NormalizedSquaredFeatureConsistency` (`objectives/feature_consistency.py`) | Fidelity-bearing, reversible | byol | n/a | The existing negative cosine has neither the source squared-norm floor nor its literal squared-distance values near zero. Ports, row scopes and target detach stay explicit. |
| `CosineEMADecay` (`core/schedules.py`) | Fidelity-bearing, reversible | byol | n/a | `Schedule`/`TeacherSpec` can carry a scheduled decay, but no primitive represented the affine complement of a half cosine. Base, horizon and index convention are bound in the plan's teacher line. |
| `lars` in `OptimiserSpec`, and `core.optimisation.LARS` | Fidelity-bearing, reversible | byol | n/a | `OptimiserSpec` built the SGD and Adam families only. The trust-ratio and decay ordering live inside that existing boundary, and eta, momentum and the adaptation exclusion are rendered into `optimisation.optimiser`. |

No new port or executor contract was added. BYOL is the named second consumer
that SimSiam's `X_PRED` addition anticipated. Do not add an EMA framework,
generic SSL engine, checkpoint system or new card-key category. If further work
discovers a missing mechanic, amend and review this card first; if a mechanic is
omitted, add typed debt and reconcile `DESIGN.md` §11.4.

### Tier 2 outcome, 2026-09-16 (superseded protocol, retained)

On 2026-09-16, commit `49df876e8ad9` produced a `deviating` result: Project-local BYOL mechanism study, not ImageNet reproduction. All four arms share actual common initial tensors, fitted scales, masks and batch streams; pretraining arms share actual view draws. EMA changes only target parameter decay; target BN owns its state. Contrasts are formed within each of the ten predeclared seeds. Rank is measured on all 2048 clean held-out rows before transfer, without BN recalibration. OracleSymmetry uses privileged DGP knowledge; no general tabular or causal-identification claim follows. Within noise of the target: ema_outcome_nll_gain was 0.00386575 +/- 0.00407 against mean - stderr > 0, inside its target by 0.00387 — less than its own standard error, so the run does not distinguish it from a miss.

## 6. Reproduction target

This is a project-local mechanism study. The 2026-09-17 amendment changes which
statistics are scored and adds two; every threshold is still the statistic's own
zero or the unchanged 0.05 and 1.1 declared before the first run, and none was
read off a measurement. The statistics were chosen on the 2026-09-17 audit, which
saw bases 620000-620500; §6.2's stream is disjoint from those. That disclosure is
the point, and it is the discipline `simsiam.md` §6.4 already applies: the
statistics are chosen, the thresholds are not. These are not paper numbers or
values copied from SimSiam results.

```yaml
reproduction:
  dataset: common.two_cluster_population with low=SEPARATED; six features; K=2; OracleSymmetry views
  variant: scheduled EMA versus zero-decay target, the inherited 1000-epoch EMA base, no predictor, and no pretraining
  split: 1024 train, 40 observed treatments; 2048 fully observed held-out rows; frozen-encoder transfer
  metric: ema_alignment_gap; ema_rank_gap; pretraining_outcome_nll_cost; encoder_effective_rank
  published: none - project-local tabular adaptation
  published_source: n/a
  tolerance: all four one-standard-error bounds in section 6.4
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-16 | `49df876e8ad9` | ema_outcome_nll_gain<br>pretraining_outcome_nll_cost<br>encoder_effective_rank | 0.00386575 +/- 0.00407<br>-0.00357281 +/- 0.00355 nat/row<br>3.24223 +/- 0.115 | no |

The 2026-09-16 row is the protocol §5 rows 3 and 4 and §6.4 have since amended:
base_ema 0.996, a fine-tuned encoder, and the superiority gate on downstream NLL.
It is retained, not superseded; §6.4's note says what its instrument measured.

### 6.2 Fixed DGP and paired execution

Import `two_cluster_population`, `continuous_schema`, `training_dataset` and
`on_the_training_scale` from `xty2.evaluation.benchmarks.common`, and
`OracleSymmetry` with its preservation tolerance from
`xty2.evaluation.vicreg_views`. Adopt SimSiam §6.2's separated fixture unchanged,
including `low=SEPARATED`; do not transcribe the generator or transforms.
Assert analytic treatment probabilities and conditional outcome means are
unchanged by the views before any fit. Oracle information is evaluation-only.

Tier 2 replicate i=0..9 uses base=630000+100*i, disjoint from the 620000-620900
stream of the 2026-09-16 run and of the 2026-09-17 audit; training seed base+1,
held-out seed base+2, model seed base+6, execution seed base+10000.
Training row offsets start at 0; held-out offsets start at 10000.
Fit preprocessing once and share the actual 40-label mask across arms, using
the downstream execution seed base+10000+STREAM_STRIDE for that draw. Hidden
treatment values must never reach either fit stage.

| Arm | Change from full | Interpretation |
|---|---|---|
| full | none | scheduled EMA at §5 row 7's base, and a learned predictor |
| source-ema | base_ema 0.996 | deviation 7's own control: the inherited 1000-epoch row at this horizon. Informational, never a gate |
| zero-decay | tau=0 at every update | target copies the updated online parameters; same stop-gradient and independently forwarded target BN. Note this makes the target bit-identical to the online network at every forward, so the arm is the teacher-free update rule, not the paper's Table 5(a) collapse |
| no-predictor | route online X_PROJ directly to loss; remove predictor from trainables | predictor mechanism diagnostic, retaining EMA |
| no-pretraining | omit pretrain and inheritance edge | effect of the additional pretraining phase; not compute-matched. With the encoder frozen this arm is a random-feature probe |

Each pretraining arm gets 1000 steps. Every arm gets 3000 downstream steps, over
the XTY heads alone: the transferred encoder is frozen (§5 row 4, as amended) and
§6.2's checks require it bit-identical after the stage.
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

Tier 0 is `tests/invariants/test_byol.py`, and the requirements predeclared for
implementation status are met there:

- Independent scalar and autograd oracles for the squared distances, both
  directional weights and the 1e-12 squared-norm floor, at ordinary scale,
  below the floor and at exactly zero. Both a `2 - 2cos` shortcut and torch's
  own `normalize`, whose floor is on the norm rather than on the square, are
  shown to disagree below the floor and to agree above it.
- Nonzero online encoder/projector/predictor gradients, absent target
  gradients, and targets read from `X_PROJ`: the compiler plans no `X_PRED`
  pass under teacher parameters at all. Empty eligible scopes, rank and width
  errors, non-finite inputs and singleton training BN rejection are covered
  here; `B != K` candidate scoring is a repository-wide contract that
  `tests/invariants/test_objectives.py` holds for the downstream terms.
- Hand-computed two-step LARS updates including the bias and BN exclusions,
  both zero-norm guards, momentum accumulation and weight-decay ordering, plus
  independent oracles for `target_ema` and `learning_schedule` at k=0, the
  warmup boundary, the midpoint, k=999 and every step between.
- Exact initial teacher copy, one EMA update after the optimiser step, both
  directions reading one target parameter set, target BN statistics owned by
  target forwards, two target forwards per step, and a teacher unchanged by
  having been read.
- Card/plan value agreement across every answered §4 key, absence of outcome
  lineage in the pretrained ports, encoder-only transfer with the projector and
  predictor absent from downstream passes, and a fresh downstream optimiser.

The Tier 1 packet adds `tests/invariants/test_byol_study.py`: effective rank
on constant, rank-one, isotropic, anisotropic and below-floor tensors,
non-finite rejection, smoke schedule/arm binding and exact common initial
tensors. Actual batch/view streams and shared fitted scales/masks are checked
inside `xty2.evaluation.byol_study.study` during each four-arm fit.

The mutants recorded in the implementing commit, each observed failing the
oracle named above: halve the loss; detach the prediction; point the target at
`X_PRED`; floor the norm instead of the square; EMA the target's buffers; move
the target twice in one step, which is the observable form of updating it
between directions; move weight decay after the trust ratio; apply the trust
ratio to biases and BN; and shift the EMA horizon by one step.

Tier 1 is implemented in `tests/smoke/test_byol.py`. It uses seeds 42, 43 and 44, 512
training rows, 512 held-out rows, 40 observed treatments, 100 pretraining and
200 downstream steps; batch 128. Rebase warmup to 1 step, the EMA horizon to
100 and the marginal ramp to 200, explicitly binding smoke overrides — the EMA
horizon is the stage's step budget, so shortening one without the other runs a
prefix of the curve. Run all four arms. Require finite losses/gradients, source
update identities, actual target movement and successful transfer on every
seed. Record norms, BN variance relative to epsilon, rank, predictor residual
and outcome NLL. Do not assert EMA superiority or ablation collapse in a wiring
test. Any proposed directional assertion needs a reviewed card amendment and
evidence on every declared smoke seed.

The 2026-09-16 smoke run passed all twelve fits under PyTorch 2.2.2 on CPU.
All losses, gradients and diagnostics were finite. Actual initial tensors,
batch/view traces, shared fitted scales/masks, target updates and downstream
transfer passed their checks. These wiring results do not populate the Tier 2
ledger or test §6.4's superiority bounds. Rounded diagnostic values:

| Seed | Arm | Outcome NLL | Clean encoder rank | Encoder norm | Predictor residual |
|---|---|---|---|---|---|
| 42 | full | 1.158474 | 6.094008 | 1.475127 | 0.960369 |
| 42 | zero-decay | 1.152634 | 2.782090 | 1.617800 | 0.247584 |
| 42 | no-predictor | 1.155998 | 10.714819 | 1.054903 | 0.683423 |
| 42 | no-pretraining | 1.149140 | 22.461118 | 0.674942 | n/a |
| 43 | full | 1.189714 | 8.616748 | 1.545943 | 1.086880 |
| 43 | zero-decay | 1.208598 | 4.056369 | 1.636624 | 0.306093 |
| 43 | no-predictor | 1.215760 | 12.316316 | 1.027578 | 0.784320 |
| 43 | no-pretraining | 1.202031 | 22.577982 | 0.745315 | n/a |
| 44 | full | 1.143018 | 5.686602 | 1.615051 | 1.025360 |
| 44 | zero-decay | 1.146747 | 2.969043 | 1.803678 | 0.227279 |
| 44 | no-predictor | 1.148738 | 9.477957 | 1.035043 | 0.751790 |
| 44 | no-pretraining | 1.140496 | 21.061956 | 0.714665 | n/a |

Rank and norm use all 512 clean held-out rows before downstream fitting; the
no-pretraining row measures the untrained encoder. Residual is the sum of the
two directional normalised squared distances, averaged across four fixed
held-out batch pairs, using target projections (online projections in the
no-predictor arm's prediction slot). The test prints full per-seed JSON,
including BN variance/epsilon and target-online lag. Across seeds, full-arm
projector running variance/epsilon is 23.56–29.92 online and 10.55–12.01 target;
predictor variance/epsilon is 8465.36–9175.60. No BN recalibration is performed.

Additional observed failing mutants: remove the rank's energy floor;
ignore state-pairing mismatches; bypass LARS finite-value validation;
retain hidden treatment payloads; freeze
the target; update it twice; EMA its buffers; remove LARS momentum. The last
four run the smoke execution and fail its update oracle. Direct LARS construction now rejects
NaN and infinite learning rates, momentum and eta, matching the existing
`OptimiserSpec` validation; finite recipe settings and updates are unchanged.
Missing treatment payloads are replaced with valid dummy class zero before
either stage's forward pass, while preserving the shared missingness mask.

Validation environment: `uv run` could not install locked Torch 2.13.0 on
Intel macOS, so checks used `uv run --no-sync` with the existing Torch 2.2.2
environment. Lint and formatting pass. Full strict mypy reports eight existing
errors under that environment, reproduced on a clean archive of `c32a9cc`;
the added files introduce no additional type errors.
The final BYOL-focused run passed 72 tests. The broader Tier 0/Tier 1 run
was intentionally interrupted after 26 minutes with 1747 passing tests and
no reported failures; all invariants had completed, and existing FixMatch
smoke tests were running. This is not a claim that the full smoke suite passed.

### 6.4 Metrics, tolerances and interpretation

Attribution is carried by two statistics read from the **terminal pretraining
checkpoint**, in eval mode on frozen training BN buffers, over §6.2's fixed
held-out view batches — before any transfer. Downstream NLL is a budget, not an
attribution instrument. That division is `simsiam.md` §6.4's, adopted here for
the reason its own audit gives, and the 2026-09-17 note below records why this
card had to arrive at it a second time rather than inheriting it.

**View alignment** `A` is the mean of the two directional inner products between
the online prediction of one view and the **target** projection of the other,
each side normalised by `sqrt(max(sum(v^2), 1e-12))`, averaged over rows, both
directions and all held-out batch pairs. Above that floor it is the cosine; the
floor is the source's own (§3.1), which is why `A` is written from
`squared_norm_floor_normalize` rather than from `cosine_similarity`. `A = 1` is
the objective at its minimum. An arm sitting closer to that optimum than the full
arm has stopped telling rows apart — the paper's non-collapse statement,
measured. `A` differs from SimSiam's identically named statistic in reading a
teacher projection, so the two cards' `A` values are not comparable.

**Encoder effective rank** `R`: centre the arm's [2048, 256] pre-transfer encoder
matrix by column; take singular values s and probabilities `p_j = s_j^2/sum(s^2)`;
report `exp(-sum(p_j*log(p_j)))` with `0*log(0) = 0`. If `sum(s^2) <= 1e-12`,
return 0.

For each replicate let `N_arm` be mean held-out factual outcome NLL on the common
training-standardised outcome scale. Per replicate define:

| Required metric | Definition | Acceptance over ten replicates |
|---|---|---|
| ema_alignment_gap | A_zero_decay - A_full | mean - stderr > 0.0 |
| ema_rank_gap | R_full - R_zero_decay | mean - stderr > 0.0 |
| pretraining_outcome_nll_cost | N_full - N_no_pretrain | mean + stderr < 0.05 nat/row |
| encoder_effective_rank | R_full | mean - stderr > 1.1 |

The first two thresholds are each statistic's own zero: they are paired contrasts,
so no level had to be chosen and none was read off a measurement. The last two are
unchanged from the pre-2026-09-16 declaration. Compute each paired difference
within seed, then sample SE = sd/sqrt(10), ddof=1. Report every arm and replicate,
including failures; do not pool rows as independent replicates. All four criteria
must pass for the local `reproduced` status under FIDELITY §3; a nominal pass
within its uncertainty is `deviating`. These are repository evidence gates, not a
multiple-testing controlled statistical claim.

`R` is a collapse guard and not a quality ordering. The untrained encoder scores
about 22 on this fixture and concentrating onto its invariant subspace is what a
working representation should do, so a *higher* `R` is not a better arm — which is
why `ema_rank_gap` is paired against a specific ablation and why the no-predictor
arm's rank is reported without a bound.

Report `ema_outcome_nll_gain = N_zero_decay - N_full`, the three `source_ema`
contrasts, no-predictor differences, treatment NLL, treatment-effect RMSE,
normalised feature spread, rank of every arm, feature norms, directional residuals
and target-online lag as diagnostics. If full and controls are indistinguishable
on both attribution axes, the claim has not reproduced even if downstream NLL is
good. Do not require zero-decay or no-predictor collapse: neither is guaranteed
here, and §6.2's arm table says why zero-decay in particular is not the paper's
collapse control. If EMA adds no measurable benefit, audit source fidelity and
record the negative result. Any changed fixture, threshold, optimiser or seed
stream requires a prospective amendment with the failed protocol retained.

#### Audit finding, 2026-09-17: why downstream NLL no longer carries the claim

This is the post-mortem of `ema_outcome_nll_gain`, the superiority gate the table
above replaces. It is retained because the replacement is only as good as the
reason for it.

The 2026-09-16 run returned 0.00386575 +/- 0.00407, a lower one-standard-error
bound of -0.000209, and `deviating`. An equation-by-equation re-audit against the
pinned source found no mechanism error, and Tier 0's oracles already covered every
mechanic it checked. Three measurements explain the result instead.

**The instrument had almost no dynamic range, because the encoder was fine-tuned.**
On this fixture, predicting the training mean scores about 1.376 nat and the Bayes
-optimal conditional mean about 1.049, so the endpoint spans about 0.327 nat. The
recorded run's four arms all landed within 0.004 nat of each other and reached the
same downstream training loss to three decimals - 0.4797, 0.4800, 0.4802, 0.4798 -
despite pre-transfer encoder effective ranks of 3.24, 2.94, 6.68 and **22.02**. A
protocol under which a random rank-22 encoder and a pretrained rank-3 one converge
to the same place is not measuring the representation. The audit's six-seed probe,
which reproduces the recorded endpoint bit-for-bit on all twelve (seed, arm)
points and varies only the downstream protocol, puts a number on it: the whole-of-
pretraining contrast `N_no_pretrain - N_full` is +0.0373 +/- 0.0095 with a frozen
encoder and +0.0009 +/- 0.0051 with the fine-tuned one. The fine-tune moves the
encoder 1.17 of its own initial norm - 6.8 times further than pretraining moved it -
collapses every arm's rank to about 1.5 within 100 steps whatever it started from,
and leaves the pretrained and random encoders 0.97 apart at the end. §5 row 4 is
amended for that reason, and the frozen protocol is also simply better here: its
worst checkpoint beats the fine-tuned protocol's best, which peaks near step 300
and is 0.087 nat worse by the reported step 3000.

**The gate was a coin flip at the declared seed count.** The observed effect-to-
spread ratio is 0.300, at which `mean - stderr > 0` has 49.3% power at ten
replicates; 87% would need about fifty. Two of the other three declared gates
could not realistically fail: the cost budget passed with the measured effect
fourteen times inside its tolerance, and the rank guard asks for 1.1 where an
untrained encoder scores 22.

**The mechanism was separating the arms all along, on the statistics above.**
Recomputed from the recorded run with no re-fit, `A_zero_decay - A_full` is
+0.009726 +/- 0.001119 on 10 of 10 seeds (t = 8.69) and `R_full - R_zero_decay` is
+0.304 +/- 0.125 on 8 of 10 (t = 2.44). Both clear the bounds the table now
declares. The instrument had to change, and the honest risk is that these two were
chosen after seeing that: the guard is that their thresholds are each statistic's
own zero rather than a level, and that §6.2 re-runs on a disjoint stream.

Two further findings belong here because they bound what any downstream gate could
have shown. The zero-decay target is bit-identical to the online network at every
forward - `target_online_lag` is exactly 0.0 on all ten recorded seeds and its
target BN statistics equal the online ones - so that arm is the teacher-free
update rule, whose published gap to BYOL is small, and not the paper's Table 5(a)
tau=0 row (0.3% against 72.5% top-1 at 300 epochs, batch 4096), which is a
catastrophic collapse that does not occur in this regime. And the EMA itself had
almost nothing to do: §5 row 7 records the integrated tracking of 2.00 against the
source's 156-626. Neither is repaired by changing the instrument, which is why
rows 4 and 7 are amended alongside it and why the frozen-protocol probe did not,
on its own, resolve the EMA contrast either.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Paper initial copy versus code independent target initialisation | Select Appendix A's copy and disclose deviation from pinned code | Explicit variant resolution, §5.5 |
| Numerical normalisation at zero | Pinned helper's squared-norm floor 1e-12 and literal squared distance | Source code resolves equation's undefined zero case |
| PyTorch equivalent of unpinned Haiku Linear defaults | Explicit torch Linear defaults; no numerical equivalence claim | Backend judgement §5.6, not an assumed source initialiser |
| Appropriate tabular widths, view policy and short-horizon EMA base | Fixed §4 settings and §6 fixture; no post-hoc tuning | Prospective local choices; departures §5.1–§5.4 |
| Whether moving targets help this low-dimensional fixture **downstream** | Unresolved, and the 2026-09-16 protocol could not resolve it | Mean gain 0.003866 ± 0.004075 SE at 49.3% gate power; §6.4's 2026-09-17 note. Neither superiority nor equivalence is established |
| Which source preset row a 1000-step horizon should take its EMA base from | `_EMA_PRESETS[100]`, the largest row the local 125-epoch budget reaches, translated by the time-constant/horizon invariant | §5 row 7. The selected 1000-epoch row is unreachable at this horizon at any base; the 40-epoch row would give 0.62467 and the 300-epoch row 0.06165, so the choice of row is a disclosed judgement, not a derivation |
| Whether the local endpoint can carry any representation-quality claim under a fine-tuned encoder | No; it is a budget only | Measured 40-fold attenuation against the frozen protocol, §6.4's 2026-09-17 note |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Claude Code, at the repository owner's request on PR #57, against the pinned source files; no independent human approval claimed | 2026-09-15 |
| Two automated review findings on PR #57 resolved | Claude Code; the EMA schedule's `TeacherSpec` validity and the unstated `WarmupCosine` final multiplier, both amended in §4 | 2026-09-15 |
| Plan diffed against §3.2 and §4 | `tests/invariants/test_byol.py`, over the actual compiled plan and every answered §4 entry | 2026-09-15 |
| Recipe implemented, Tier 0 passing (status → `implemented`) | Claude Code | 2026-09-15 |
| Source audit and Tier 1 implementation; all declared smoke seeds pass | Codex; LARS finite-value validation corrected, paired execution and diagnostics added | 2026-09-16 |
| Source re-audit and complete Tier 2 execution (status → `deviating`) | Codex; all forty fits complete; two of three unchanged gates pass; strict upper-bound reporting and saved-result verification added | 2026-09-16 |
| Deviation audit of the `deviating` result; §5 rows 3, 4 amended, row 7 added, §6.4 instrument replaced, §6.2 stream moved | Claude Code, at the repository owner's request; no source-mechanism error found; the fine-tuned endpoint, the gate's power and the zero-decay arm's identity measured | 2026-09-17 |

Four amendments were made at the 2026-09-15 review, none of them to the method.
The 2026-09-17 amendments are described in §6.4's audit note and §5 rows 3, 4 and
7; one of them, row 7, *is* a change to the method, which is why it carries its
own row and its own `source_ema` control arm.

The first two answer PR #57's review comments. A `CosineEMADecay` faithful to
`schedules.target_ema` reaches exactly 1 at its horizon, and `TeacherSpec`
rejects any decay outside `[0, 1)` before the stage runs, so the draft's
schedule would have compiled into nothing. The endpoint is real — a frozen
target — but it is outside the applied domain, because the executor updates a
teacher at steps `0 .. steps - 1`. `nominal` therefore reports the decay at the
last applied step and §4 binds that number, which keeps both range checks
meaningful without clamping the curve the draft explicitly forbade clamping.
Separately, §4's `lr_schedule` line named `WarmupCosine` without its `final`
multiplier, which has no default. The pinned `learning_schedule` runs
`_cosine_decay` over `total_steps - warmup_steps` all the way down, so the
value is 0, and §4 now says so.

The third is the one the draft could not have: §4 was written as prose and is
now the two-level form the other cards use, so that `FIDELITY.md` §1.2's
cross-check compares every answered leaf by value against `plan.hyperparameters`
rather than merely by presence.

The fourth corrects two keys the draft answered that nothing binds.
`labelled_unlabelled_ratio` is `n/a` under `UniformSampler`, which enforces no
quota, and `treatment_encoding` is `n/a` because `XTYBatch` supplies integer
classes; both had prose answers that would have claimed a plan entry that does
not exist.

The 2026-09-16 Tier 2 implementation leaves the claim and all §6.4 thresholds
unchanged. Mean EMA gain is 0.0038657546 nat/row with SE 0.0040746716; its lower
one-SE bound is -0.0002089170, so superiority is not demonstrated. Six of ten
paired gains are positive. Pretraining cost is -0.0035728097 ± 0.0035531770
nat/row and full-arm encoder rank is 3.2422294 ± 0.1147409: both guards pass.
The source audit found no additional method error; it checked the loss,
topology, LARS, schedules, teacher/BN state and local deviations. Do not turn
this uncertain local gain into a claim that EMA is useless or that BYOL fails
on its original task. All arm and per-seed diagnostics, environment details
and source provenance are retained in the linked experiment report.
