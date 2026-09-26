# Recipe spec card: vime

**Status:** `implemented`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. §2.1 says how this card relates to
> the adaptive-task proposal; the controller itself is not in this card.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) |
| Authors, year | Jinsung Yoon, Yao Zhang, James Jordon, Mihaela van der Schaar; 2020 (NeurIPS 2020) |
| DOI / arXiv | No arXiv version. NeurIPS 2020 proceedings, paper file `7d97667a3e056acab9aaf653807b4a03-Paper.pdf`. |
| Version used | The NeurIPS 2020 camera-ready PDF (11 pages). Section 4.1 defines the self-supervised method (Eqs. 3–6); section 5 states the min-max preprocessing. The supplementary material, especially §§3, 5–6 and 8, was also checked. It reports pretext-task diagnostics, validation-based architecture and hyperparameter selection, and a mask-estimation ablation. The fixed defaults below come from the linked reference script, not a universal recommendation of the paper. |
| Reference implementation | [jsyoon0823/VIME](https://github.com/jsyoon0823/VIME) @ `996c58cf4c570061b30c38ecf2a754a9af85aafd` (2020-10-26), linked from the paper's section 5. Files: `vime_self.py`, `vime_utils.py`, `main_vime.py`, `supervised_models.py`, `data_loader.py`. Keras 2.3.1 on TensorFlow 1.15 (`requirements`). |
| Reference impl. runnable? | not attempted. It pins TensorFlow 1.15, which does not install on this repository's Python. The code was read, not run; §7 names each value taken from it. |

## 2. Estimand and claim

- **Estimand:** treatment-specific outcome means, fitted by the project-local
  causal heads on top of a VIME-pretrained, **frozen** encoder.
- **Method claim (VIME-self only):** corrupt each cell with probability `p_m`
  by a draw from that column's empirical training marginal (Eq. 3), train one
  encoder and two estimator heads to recover which cells were replaced (Eq. 5)
  and the original row (Eq. 6), weighted `l_m + α·l_r` (Eq. 4). Keep the
  encoder, discard both heads, and fit a downstream predictor on the encoder's
  output.
- **Not claimed:**
  - VIME-semi (paper section 4.2, `vime_semi.py`) is out of scope. It is a
    separate consistency method over `K` augmentations and would need its own
    card.
  - The paper's genomics, clinical and MNIST numbers are not reproduced. The
    §6 target is a mechanism target on a project-local fixture.
  - Nothing here claims that masked-feature pretraining identifies causal
    structure or improves effect estimation. §6 treats the downstream
    outcome fit as a guardrail and not as the method's claim.

### 2.1 Relation to the adaptive-task proposal

[`proposals/tabular-self-play-ssl.md`](../proposals/tabular-self-play-ssl.md)
§5 step 1 asks for a reviewed fixed-task masked-feature card before any
adaptive task controller is piloted. This card is that step. It supplies:

- the **fixed, published masked-feature policy** the proposal's §4 arm 2
  ("a credible strong baseline") needs; and
- the shared encoder, estimator heads, corruption view and loss objects that
  the proposal's §3 candidate tasks would reuse.

It does **not** contain the controller, the candidate task set, the
gradient-alignment reward of Cowsik et al. Eq. 2, or any stateful sampling.
The proposal keeps those in an external experiment driver, and `DESIGN.md`
§11.4 `stateful-sampler` stays unbuilt. A conditional (masked-cells-only)
reconstruction loss is the proposal's object, not VIME's; §5.1 names it only
as the second consumer that constrains the shape of the objective built here.

## 3. Equations and mapping

### 3.1 As published

Section 4.1, in the paper's notation. `D_u` is the unlabelled set,
`x ∈ R^d` a row, `e` the encoder, `s_m` the mask vector estimator and `s_r`
the feature vector estimator.

> Mask: `m = [m_1, …, m_d]ᵀ ∈ {0,1}^d`, `m_j ~ Bern(m_j | p_m)` independently.

> Eq. (3): `x̃ = g_m(x, m) = m ⊙ x̄ + (1 − m) ⊙ x`, where the `j`-th feature of
> `x̄` is sampled from the empirical distribution
> `p̂_{X_j} = (1/N_u) Σ_{i=N_l+1}^{N_l+N_u} δ(x_j = x_{i,j})` — "the empirical
> marginal distribution of each feature".

> Eq. (4): `min_{e, s_m, s_r} E_{x~p_X, m~p_m, x̃~g_m(x,m)} [ l_m(m, m̂) + α · l_r(x, x̂) ]`,
> with `m̂ = (s_m ∘ e)(x̃)` and `x̂ = (s_r ∘ e)(x̃)`.

> Eq. (5): `l_m(m, m̂) = −(1/d) Σ_{j=1}^{d} [ m_j log (s_m∘e)_j(x̃) + (1 − m_j) log(1 − (s_m∘e)_j(x̃)) ]`.

> Eq. (6): `l_r(x, x̂) = (1/d) Σ_{j=1}^{d} (x_j − (s_r∘e)_j(x̃))²`.
> "For categorical variables, we modified Equation 6 to cross-entropy loss."

> Section 5: "We use Min-max scaler to normalize the data between 0 and 1."

Four properties are load-bearing and easy to lose:

- **The mask is Bernoulli per cell, not a fixed count.** Each row corrupts a
  Binomial(`d`, `p_m`) number of cells. SCARF's `FeatureCorruption` corrupts
  exactly `floor(c·M)`. Same mean, different variance, different task
  (`scarf.md` §3.1 makes the same point in the other direction).
- **Eq. 6 sums over all `d` features**, masked or not. The unmasked cells are
  visible to the encoder, so part of `l_r` is an identity copy. That is the
  published objective; a masked-cells-only loss is a different one.
- **Both heads read the same `z = e(x̃)`.** The encoder sees only the corrupted
  row. The clean row reaches the loss as a target and never as an input.
- **Only `e` survives.** "It is the only part we will utilize in the
  downstream tasks." In `main_vime.py` the downstream MLP is fitted on
  `vime_self_encoder.predict(x)`, so the encoder is frozen downstream.

Reference implementation (`vime_self.py`, `vime_utils.py`, `main_vime.py`):

- `e`: `Dense(int(dim), activation='relu')` — one layer of width `d`.
- `s_m`: `Dense(dim, activation='sigmoid', name='mask')`.
- `s_r`: `Dense(dim, activation='sigmoid', name='feature')` — sigmoid because
  the inputs are min-max scaled to `[0, 1]`.
- `model.compile(optimizer='rmsprop', loss={'mask': 'binary_crossentropy',
  'feature': 'mean_squared_error'}, loss_weights={'mask': 1, 'feature': alpha})`.
- `main_vime.py` defaults: `p_m = 0.3`, `alpha = 2.0`; `vime_self` runs
  `batch_size = 128`, `epochs = 10`.
- `mask_generator` draws `m` with `np.random.binomial(1, p_m, x.shape)`;
  `pretext_generator` builds `x̄` by an independent row permutation of each
  column, and returns the label `m_new = 1 * (x != x_tilde)`.
- Both are called **once**, before `model.fit`. The corrupted table is fixed for
  all ten epochs.

### 3.2 Mapping to xty2

Two stages. `pretrain` realises `x̃` through a view and trains the encoder and
both estimators on `X_RAW` alone. `joint_fit` initialises from `pretrain`,
leaves the encoder out of `trainable`, and fits the reviewed causal heads.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `x` | the clean row (target only) | `X_RAW` | the source node under the `identity` realisation |
| `m`, `x̄`, `g_m` | Bernoulli mask, marginal draws, pretext generator | `X_RAW @ vime_corrupted` | `ViewSpec("vime_corrupted")` over `BernoulliMarginalCorruption(p=0.3)` (new, §5.1) |
| `p̂_{X_j}` | empirical marginal of column `j` in `D_u` | — | the training assignment's column `j`, drawn i.i.d. per corrupted cell (deviation 7) |
| `x̃` | the corrupted row | `X_RAW @ vime_corrupted` | the only input `e` sees in `pretrain` |
| `e` | encoder | `X_RAW -> X_REPR` | `MLPEncoder`, widths `[d]`, ReLU, no normalisation, Glorot-uniform (new option, §5.1) |
| `s_m` | mask vector estimator | `X_REPR -> FEATURE_MASK_LOGITS` | `MaskEstimatorHead` (new, §5.1): one affine layer to `d` logits |
| `s_r` | feature vector estimator | `X_REPR -> RECONSTRUCTION` | `FeatureEstimatorHead` (new, §5.1): one affine layer to `d` values, then sigmoid |
| `m` as a label | which cells changed | — | computed inside `MaskEstimationBCE` as `1[x != x̃]` from `X_RAW` at both realisations, as `pretext_generator` does (§7) |
| `l_m` (Eq. 5) | per-cell BCE, mean over `d` | `FEATURE_MASK_LOGITS @ vime_corrupted`, `X_RAW` at both realisations | `MaskEstimationBCE` (new), weight 1, rows `all`, `reduction="mean"` |
| `l_r` (Eq. 6) | per-cell squared error, mean over `d` | `RECONSTRUCTION @ vime_corrupted`, `X_RAW @ identity` | `FeatureReconstruction(cells="all")` (new), weight `α = 2.0`, rows `all`, `reduction="mean"` |
| `α` | loss trade-off | — | the `LossMixer` weight on `FeatureReconstruction` |
| min-max scaling | preprocessing | — | `PreprocessSpec` `x: minmax fitted on 'train'` (new option, §5.1) |
| "only `e`" is kept | heads discarded | — | `joint_fit` omits `mask_estimator` and `feature_estimator` from every forward pass and `trainable` list |
| frozen downstream `e` | `vime_self_encoder.predict` | — | `joint_fit.trainable` omits `mlp_encoder` |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `TARNetHead` + `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | treatment likelihood | `T_GIVEN_X` | `CategoricalPropensity` + `ObservedTreatmentNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    pretrain.mask_estimation_bce: none            # Eq. 4 minimises over e, s_m, s_r jointly
    pretrain.feature_reconstruction: none
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets:
    pretrain.mask_estimation_bce: n/a             # the label 1[x != x~] is a function of X_RAW, which has no parameters
    pretrain.feature_reconstruction: n/a          # the target x is X_RAW
  gradient_clipping:
    pretrain: none                                # vime_self.py: Keras 'rmsprop' with no clipnorm/clipvalue
    joint_fit: none
  marginal_nll_grad_path:
    joint_fit.missing_treatment_marginal_nll: both   # reviewed P5 choice; project-local addition

teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    pretrain.mask_estimation_bce: mean            # Eq. 5's 1/d per row, then Keras's batch mean
    pretrain.feature_reconstruction: mean         # Eq. 6's 1/d per row, then Keras's batch mean
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    pretrain.mask_estimation_bce: all             # D_u: self-supervised, reads no label of any kind
    pretrain.feature_reconstruction: all
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    pretrain.mask_estimation_bce: 1.0             # vime_self.py loss_weights={'mask': 1, ...}
    pretrain.feature_reconstruction: 2.0          # alpha; main_vime.py --alpha default 2.0
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    pretrain.mask_estimation_bce: constant 1.0
    pretrain.feature_reconstruction: constant 2.0
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a

optimisation:
  optimiser:
    pretrain: rmsprop(rho=0.9, eps=1e-07, momentum=0.0, centered=false)   # Keras 2.3.1 RMSprop defaults; vime_self.py optimizer='rmsprop'
    joint_fit: adam(betas=(0.9, 0.999), eps=1e-08)                        # supervised_models.py:130 optimizer='adam' (Keras defaults); see deviation 4
  lr:
    pretrain: 0.001                               # Keras 2.3.1 RMSprop default
    joint_fit: 0.001                              # Keras Adam default, as supervised_models.py uses it
  lr_schedule:
    pretrain: constant 1.0                        # no schedule in vime_self.py
    joint_fit: constant 1.0
  weight_decay:
    pretrain: none                                # no regulariser on any Dense layer
    joint_fit: none
  batch_size:
    pretrain: 128                                 # main_vime.py vime_self_parameters['batch_size']
    joint_fit: 128                                # deviation 4: SCARF's stage, adopted unchanged
  labelled_unlabelled_ratio: n/a                  # UniformSampler enforces no quota; pretrain reads no labels
  total_steps_or_epochs:
    pretrain: 80                                  # optimiser steps = 10 epochs (main_vime.py) x 1024 rows / 128; deviation 3
    joint_fit: 3000                               # deviation 4

architecture:
  widths_depths:
    mlp_encoder: [d]                              # vime_self.py: Dense(int(dim)); d = 6 on the section 6 fixture
    mask_estimator: linear d -> d
    feature_estimator: linear d -> d
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: relu                             # vime_self.py activation='relu'
    mask_estimator: linear logits                 # sigmoid folded into the BCE; section 7
    feature_estimator: sigmoid                    # vime_self.py activation='sigmoid' on 'feature'
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: none
    mask_estimator: none
    feature_estimator: none
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    mask_estimator: 0.0
    feature_estimator: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: glorot_uniform, bias=0           # Keras Dense default kernel/bias initialisers
    mask_estimator: glorot_uniform, bias=0
    feature_estimator: glorot_uniform, bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    mask_estimator: d Bernoulli logits
    feature_estimator: d values in (0, 1)
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: x: minmax fitted on 'train'    # section 5: "Min-max scaler to normalize the data between 0 and 1"
  outcome_scaling: y: zscore fitted on 'train'    # project-local, SCARF section 6.1 unchanged
  treatment_encoding: n/a                         # XTYBatch contract supplies integer classes 0..K-1
  split_protocol: scarf.md section 6.1 fixture unchanged; training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 40 labelled rows, keyed by row_id  # scarf.md section 6.1
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Fit the reviewed xty2 causal stack (outcome NLL, treatment NLL, exact marginalisation over missing `t`) on the frozen encoder, instead of the reference two-layer softmax MLP (`supervised_models.py:82–142`). | The project-local question is whether the representation helps the treatment-scarce XTY problem, as for `scarf.md` §5.1. The encoder stays frozen, which is what `main_vime.py` does. | No published number applies. The §6 comparison is internal and paired. |
| 2 | `judgement` | — | Draw a fresh mask and fresh marginal replacements for every batch at every step, rather than corrupting the table once before `model.fit` as the reference code does. | Eq. 4 is an expectation over `m ~ p_m` and `x̃ ~ g_m(x, m)` for each `x`. A single fixed draw is a finite-sample approximation of it that the paper does not describe. Fresh draws are also what an xty2 view does by construction. The fixed-draw variant is an ablation in §6, not the default. | Fresh draws give the encoder more distinct corruptions per row over 10 epochs (10 instead of 1). The direction on the §6 metric is not predicted. |
| 3 | `judgement` | — | 10 epochs are expressed as 80 optimiser steps drawn by `UniformSampler(batch_size=128)`: each step is a fresh without-replacement batch, not one pass of a shuffled epoch. | The paper does not name epoch semantics; Keras `fit` supplies them. The step count matches `main_vime.py` exactly on 1,024 rows. Every card here fixes a step budget, and both §6 arms share the stream. | Rows are visited 10 times in expectation instead of exactly 10 times. Expected to be small; not measured. |
| 4 | `judgement` | — | `joint_fit` is SCARF's stage unchanged (batch 128, 3,000 steps, Adam at 0.001), apart from the frozen encoder. The reference MLP uses batch 100 and up to 100 epochs with early stopping (patience 50) on a 10% validation split. | Adopting SCARF's §6.1 protocol unchanged lets §6 import its module and compare against its arms on the same stream. Adam at 0.001 agrees with the reference downstream optimiser. Early stopping is the `early-stopping` ledger item, and a fixed budget is what we would choose here even if it existed (`scarf.md` §5.4 makes the same call). | Downstream numbers are a property of this stage, not of VIME. Only the paired ratio in §6 is interpreted. |
| 5 | `judgement` | — | `FeatureReconstruction` and `MaskEstimationBCE` refuse schemas with categorical or ordinal features, rather than switching Eq. 6 to cross-entropy for them. | The §6 fixture is all-continuous, so the categorical branch cannot affect any result this card reports. The proposal's first experiment also asks for an all-continuous fixture. Refusing is honest; a silent squared error on class codes would not be. | None on §6. A card on a mixed schema must implement the categorical branch first. |
| 6 | `judgement` | — | Corruption is restricted to columns the schema marks `mutable`, and `d` in Eqs. 5–6 still counts every feature. | `FeatureSpec.mutable=False` is absolute in xty2 (`DESIGN.md` §5). An immutable column is then never masked, so its `m_j` is always 0 and its reconstruction is always an identity copy, which is what Eqs. 5–6 give when the mask never selects it. | None on §6: every fixture column is mutable. |
| 7 | `judgement` | — | Pretrain on `X` from every training row and draw donors from that same training assignment, including rows with observed `t`. The paper defines its marginal over the separate unlabeled set `D_u`. | Here "unlabeled" means unused `t` and `y` in the pretext task; the 40 observed-treatment rows still supply covariates. This matches the project's split contract and keeps donor statistics train-only. | The donor empirical marginal and pretraining sample differ slightly from a literal `D_u` partition. No validation or held-out rows enter either. |

### 5.1 Framework additions made for this card

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `BernoulliMarginalCorruption(p, columns=None)` view: per-cell Bernoulli mask, replacement drawn from the `TrainingPopulation` column, requires the population as `FeatureCorruption` does | fidelity-bearing, reversible | `vime` | — (reversible; VIME-semi reuses `g_m` unchanged, and the proposal's §3 "marginal replacement" corruption axis) | Eq. 3 is a Bernoulli mask. `FeatureCorruption` is a fixed count and `FeatureMask` fills a constant; either would change the task (§3.1). |
| `Port.FEATURE_MASK_LOGITS`, shape `[B, D]`: "one replaced-cell logit per feature" | fidelity-bearing, **load-bearing vocabulary** | `vime` (`s_m`) | TabTransformer-RTD (Huang et al. 2020, arXiv:2012.06678v1, §2 pre-training): "RTD replaces the original feature by a random value of that feature. Here, the loss is minimized for a binary classifier that tries to predict whether or not the feature has been replaced." Same shape: one binary logit per column, per row, with no class axis. Not yet in `BACKLOG.md`; this pass adds it to §5.2 there. | `PRETEXT_GIVEN_X` is one categorical distribution per row over transform classes, and `D` independent Bernoullis are not that. Reusing it would mislabel the quantity in every plan. `RECONSTRUCTION` already exists and is reused for `s_r`. |
| `MaskEstimatorHead`, `FeatureEstimatorHead` components | fidelity-bearing, reversible | `vime` | — | `s_m`, `s_r` (§3.2). |
| `MaskEstimationBCE` (Eq. 5) and `FeatureReconstruction(cells=...)` (Eq. 6) objectives | fidelity-bearing, reversible | `vime` with `cells="all"` | The proposal's §3 conditional feature prediction needs `cells="masked"`: the loss on cells whose original value did not reach the encoder. Checked shape: both read the same two `X_RAW` realisations and one `RECONSTRUCTION`, and differ only in which cells enter the per-row mean. | Eq. 6 has no existing objective. Designing the `cells` argument now keeps the proposal from forking the objective. Only `"all"` is built until a reviewed consumer needs `"masked"`. |
| `rmsprop` in `OptimiserSpec` (`rho`, `eps`, `momentum`, `centered`) | fidelity-bearing, reversible | `vime` | — | `vime_self.py` compiles with `'rmsprop'`. Keras's `rho=0.9`, `eps=1e-7` differ from PyTorch's defaults (`alpha=0.99`, `eps=1e-8`), so every knob is bound explicitly. |
| `minmax` in `Standardisation`, fitted on the training assignment with row-id provenance | fidelity-bearing, reversible | `vime` | — | Section 5, and the sigmoid `s_r` assumes targets in `[0, 1]`. |
| `glorot_uniform, bias=0` initialisation option for `MLPEncoder` and the new heads | fidelity-bearing, reversible | `vime` | — | Keras `Dense` default. The paper names none, so the reference code is the source. |

## 6. Reproduction target

A mechanism target. It asks whether the published pretext task learns
conditional structure where the fixture has some, and not where it has none.
It does not ask for a downstream gain. Reconstruction and mask-estimation
metrics inspect the fitted pretext heads immediately after `pretrain`, before
they are discarded. They are diagnostics of the pretext model, not predictions
from the frozen downstream encoder alone; the untrained-encoder arm has no
trained reconstruction head and contributes only to the downstream comparison.

```yaml
reproduction:
  dataset: scarf.md section 6.1 fixture (fixmatch.md 6.1 generator, 6 features, K=2), imported unchanged
  variant: paired VIME pretraining against an untrained encoder with the same initialisation, both frozen under the identical joint_fit stage, same seeds and same batches
  split: 1024 train rows with 40 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out reconstruction MSE on corrupted cells of the dependent block x0..x3, as a ratio to imputing the training column mean; the same ratio on the independent block x4..x5 as a leakage canary; held-out outcome NLL ratio as an adaptation guardrail; mask-estimation AUROC and treatment NLL ratio are informational
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: dependent-block ratio < 0.95 in mean; independent-block ratio >= 0.98 in mean; held-out outcome NLL within 1.05x of the untrained-encoder arm
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

Adopt `scarf.md` §6.1 unchanged, and import its benchmark module's fixture
and seed streams (`xty2/evaluation/benchmarks/scarf.py`) rather than
transcribing them. Feature min-max scaling replaces SCARF's z-score and is
fitted on the complete training population. Held-out rows take the same
fitted transform, never a refitted one.

The fixture suits this question because it contains both kinds of column.
`x0..x3` share the cluster indicator `c`, so a masked cell there is
predictable from the visible cells of the same block. `x4, x5` are
independent noise, so no function of the visible cells can beat the column
mean for them. The corrupted-cell MSE is measured on held-out rows under one
fixed evaluation draw of the corruption per replicate, and the baseline
predicts the training column mean on the same cells.

- **Dependent-block ratio < 0.95.** With `c` known exactly, the
  best achievable ratio for one masked cell of `x0..x3` is
  `0.36 / (0.45² + 0.36) = 0.64`. Inferring `c` from the visible cells
  raises that bound. `0.95` asks only that the pretext task learned some of
  the dependence. It is a prospective bound and has not been measured.
- **Independent-block ratio >= 0.98.** An average materially below 1 on `x4, x5`
  warrants a leakage and evaluation-protocol audit. Finite test draws can
  also yield a ratio below 1 by chance; this gate alone does not prove leakage.
- **Outcome NLL guardrail.** A frozen encoder of width 6 is a bottleneck,
  and a frozen random ReLU layer may lose information. The guardrail asks
  only that VIME pretraining does not make the frozen-encoder fit worse than
  the frozen-random one by more than 5%.

Ablations run in the same study and are reported, not gated: the fixed
single-draw corruption of the reference code (deviation 2); `α = 0` (mask
estimation alone); and `l_m` at weight 0 (reconstruction alone). For each
pretraining arm retain its diagnostic heads only long enough to score the
held-out pretext metrics, and exclude those heads from `joint_fit`. Use the
same fixed held-out corruption draw for all pretext arms in a replicate.
The frozen random encoder arm is evaluated only on downstream NLL.

### 6.2 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Whether `x̄_j` is drawn with replacement from the marginal, or by permuting the column. | I.i.d. draws with replacement from the training population's column, per `(row, feature)` cell. | Eq. 3 defines `x̄_j` as a sample from `p̂_{X_j}`. `pretext_generator` permutes each column over the whole table, which is without-replacement sampling over one pass. The two agree in distribution per cell. |
| Whether the mask label is the drawn `m` or the cells that actually changed. | `1[x != x̃]`, computed from `X_RAW` at both realisations. | `pretext_generator` returns `m_new = 1 * (x != x_tilde)`, and that is what `vime_self.py` trains on. Eq. 5 writes `m`. On a continuous fixture they differ only when a donor equals the row's own value (probability about `1/1024` per masked cell), so the reference code's definition is kept and no new batch field is needed to export `m`. |
| `p_m` and `α`. | `p_m = 0.3`, `α = 2.0`. | `main_vime.py` argparse defaults. Supplement §6 reports sensitivity ranges and validation-based selection, not a universal recommended pair. These values are the reference script's defaults; tuning is deliberately omitted from this fixed-fixture protocol. |
| Architecture of `e`, `s_m`, `s_r`. | One ReLU layer of width `d`; one sigmoid affine layer each. | `vime_self.py`. Supplement §5 describes validation-based selection of widths `{d/3, d/2, d, 2d, 3d}` and depths `{1, 2, 3, 4, 5}` for the experiments; this card uses the fixed released script. |
| Optimiser and its constants. | RMSprop, `lr = 0.001`, `rho = 0.9`, `eps = 1e-7`, no momentum. | `vime_self.py` `optimizer='rmsprop'` with Keras 2.3.1 defaults. |
| Whether BCE is computed on probabilities or logits. | On logits (`binary_cross_entropy_with_logits`), with the sigmoid folded in. | Keras clips probabilities to `[1e-7, 1 − 1e-7]` before the log. The two agree except where the sigmoid saturates beyond that clip, where Keras's gradient vanishes and ours does not. |
| Initialisation. | Glorot-uniform kernels, zero biases. | Keras `Dense` defaults in the reference code. |
| Pretraining length in steps. | 80 steps. | 10 epochs (`main_vime.py`) × 1,024 rows / 128. The paper's datasets are far larger, so 10 epochs there is many more steps. 80 steps may be too few to learn the dependence on this fixture. That is the faithful value, and §6 measures it rather than assuming it. |
| Whether the downstream encoder is frozen or fine-tuned for VIME-self. | Frozen. | `main_vime.py` fits the MLP on `vime_self_encoder.predict(x_train)`. The paper's text says only that `e` "is the only part we will utilize". |
| Pretext diagnostics after the heads are discarded. | Score immediately after pretraining and before constructing the downstream stage; keep diagnostic outputs out of `joint_fit`. | Supplement §3 reports mask AUROC and reconstruction MSE. Neither metric can be computed from the retained encoder alone. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Codex | 2026-09-26 |
| Plan diffed against §3.2 and §4 | | |
