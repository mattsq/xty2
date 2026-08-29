# Recipe spec card: ssdml

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Double/Debiased Machine Learning for Treatment and Structural Parameters](https://arxiv.org/abs/1608.00060) supplies the orthogonal ATE score and cross-fitting argument. The semi-supervised treatment-imputation stage is project-local. |
| Authors, year | Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey and James Robins, 2018 |
| DOI / arXiv | [arXiv:1608.00060v7](https://arxiv.org/abs/1608.00060v7); [10.1111/ectj.12097](https://doi.org/10.1111/ectj.12097) |
| Version used | The Econometrics Journal 21(1), 2018, together with arXiv v7, 2024-11-03. |
| Reference implementation | Official [`DoubleML/doubleml-for-py` @ `1808b07a13cc8c61f508c1ed6aec658ea32a2807`](https://github.com/DoubleML/doubleml-for-py/tree/1808b07a13cc8c61f508c1ed6aec658ea32a2807), especially [`DoubleMLIRM`](https://github.com/DoubleML/doubleml-for-py/blob/1808b07a13cc8c61f508c1ed6aec658ea32a2807/doubleml/irm/irm.py). Project-local predecessor: [`mattsq/XTYLearner` `ss_dml.py` @ `35734ec2d5a62d54a59eca38d1e31423da31e1ea`](https://github.com/mattsq/XTYLearner/blob/35734ec2d5a62d54a59eca38d1e31423da31e1ea/xtylearner/models/ss_dml.py). |
| Reference impl. runnable? | The official implementation was inspected but not executed in the card pass. The old XTYLearner wrapper is not a valid reference run: it constructs `DoubleMLSSM` without the selection indicator required by that estimator and then calls an undocumented `set_external_data(X_full=...)` surface. |

## 2. Estimand and claim

- **Estimand:** the binary-treatment average treatment effect under consistency, positivity, and exchangeability.
- **Method claim:** DML2 evaluates the orthogonal interactive-regression score out of fold. xty2 prepends project-local out-of-fold hard treatment imputation.
- **Scope:** the imputed-treatment extension is not covered by the DML theorem. The reported standard error is diagnostic only; this is not DoubleML's sample-selection model.

## 3. Equations and mapping

### 3.1 As published

For binary `D`, covariates `X` and outcome `Y`, the interactive regression model
uses

$$
g_0(d,x)=\mathbb E[Y\mid D=d,X=x],
\qquad m_0(x)=P(D=1\mid X=x),
$$

and targets

$$
\theta_0=\mathbb E[g_0(1,X)-g_0(0,X)].
$$

The doubly robust ATE score used by the paper and official IRM implementation
is

$$
\psi(W;\theta,\eta)
= g(1,X)-g(0,X)
+\frac{D[Y-g(1,X)]}{m(X)}
-\frac{(1-D)[Y-g(0,X)]}{1-m(X)}
-\theta.
$$

DML fits the nuisances on each fold complement and evaluates the score only on
the held-out fold. The DML2 estimate is the value that makes the mean of all
held-out scores zero.

This differs from DoubleMLSSM's missing-at-random sample-selection score, which
contains a distinct selection indicator `S` and selection propensity
`pi(d,x)`. No such variable exists in the legacy `ss_dml.py` construction or in
the P11 missing-treatment problem.

### 3.2 Mapping to xty2

A cross-fit propensity stage emits missing-treatment labels. An `array_fit` stage then fits fold-complement ridge outcome models and logistic propensity, clips propensity to `[0.025, 0.975]`, and aggregates held-out AIPW scores.

| Paper / stage quantity | Meaning | xty2 Port / artifact | xty2 Objective / Component / action |
|---|---|---|---|
| `m_0(x)` in label stage | Treatment propensity used for missing-label prediction | `T_GIVEN_X` | `mlp_encoder`, `categorical_propensity` |
| `-log p(t_i | x_i)` | Fit treatment predictor where treatment is observed | `T_GIVEN_X` | `ObservedTreatmentNLL` |
| `argmax p(t | x)` | Hard out-of-fold treatment labels | `PseudoLabels` | `PseudoLabelAction(port=T_GIVEN_X, rows="t_missing")` |
| `g_0(0,x), g_0(1,x)` | DML outcome nuisances | array checkpoint tensor state | `SSDMLATEAction` ridge fits |
| `m_0(x)` in DML action | DML score propensity nuisance | array checkpoint tensor state | `SSDMLATEAction` logistic-ridge fit |
| `psi(W;theta,eta)` | Held-out orthogonal ATE score | array checkpoint tensor state | `SSDMLATEAction` score aggregation |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    propensity_labels.observed_treatment_nll: none
  detached_targets: n/a
  gradient_clipping:
    propensity_labels: none
    dml_ate: n/a
  marginal_nll_grad_path: n/a

teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    propensity_labels.observed_treatment_nll: mean
  eligible_rows:
    propensity_labels.observed_treatment_nll: t_observed
  weights:
    propensity_labels.observed_treatment_nll: 1.0
  schedules:
    propensity_labels.observed_treatment_nll: constant 1.0
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a  # P10 action is unconditional hard argmax

optimisation:
  optimiser:
    propensity_labels: adam(betas=(0.9, 0.999), eps=1e-8)
    dml_ate: closed-form ridge solves plus Newton/IRLS logistic ridge
  lr:
    propensity_labels: 0.001
    dml_ate: n/a
  lr_schedule:
    propensity_labels: constant 1.0
    dml_ate: n/a
  weight_decay:
    propensity_labels: none
    dml_ate: ridge penalty 0.001 on nuisance slopes; intercept exempt
  batch_size: n/a  # cross-fit source and array action consume finite external batches
  labelled_unlabelled_ratio: n/a  # no per-batch quota
  total_steps_or_epochs:
    propensity_labels: 500 optimiser steps per fold
    dml_ate: five held-out fold evaluations; IRLS maximum 100 iterations

architecture:
  widths_depths:
    mlp_encoder: [64, 64]
    categorical_propensity: linear 64 -> 2
    dml_ate: two linear ridge outcome nuisances and one logistic ridge propensity per fold
  activation:
    mlp_encoder: relu
    categorical_propensity: linear logits
    dml_ate: identity outcome links; logistic propensity link
  normalisation:
    mlp_encoder: none
    categorical_propensity: none
    dml_ate: none inside action
  dropout:
    mlp_encoder: 0.0
    categorical_propensity: 0.0
    dml_ate: n/a
  initialisation:
    mlp_encoder: torch Linear default Kaiming-uniform
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
    dml_ate: deterministic zero-start IRLS; ridge solves have no iterative initialisation
  output_parameterisation:
    categorical_propensity: two softmax logits
    dml_ate: scalar ATE plus held-out nuisance and influence-score tensors

data:
  standardisation:
    propensity_labels: none  # fixed DGP is already standard normal; general callers own upstream encoding
    dml_ate: z-score X from each fold complement and freeze for its held-out fold
  outcome_scaling: none
  treatment_encoding: binary integer 0/1 with t_observed mask; K must equal 2
  split_protocol: supplied fold_id with exactly five non-empty folds; every nuisance prediction held out
  missingness_mechanism: n/a  # dataset property; section 6 fixes treatment MCAR
```

## 5. Deviations from the paper and project-local source

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Implement the IRM/AIPW ATE score, not `DoubleMLSSM`. | `DoubleMLSSM` is for outcome sample selection and requires a separate `S`; the legacy wrapper supplies none. | Replaces a non-runnable/misidentified baseline with the estimator matching the declared ATE problem. |
| 2 | `judgement` | — | Prepend cross-fitted hard treatment imputation using `p(t | x)`. | P11 must exercise treatment-missing semi-supervision and the P10 cross-fit artifact path. | Can add efficiency when labels are accurate and bias when they are not; the DML theorem does not cover it. |
| 3 | `judgement` | — | Do not claim the DML standard error after pseudo-labelling. | Orthogonality protects nuisance estimation under the paper's observed-data model, not classification error in an imputed treatment. | No effect on the point estimate; prevents invalid inferential claims. |
| 4 | `judgement` | — | Use deterministic ridge linear/logistic nuisances rather than the old source's random forests and `DoubleML` dependency. | The array artifact must be portable tensor state, and numpy is already a core dependency. On the oracle true-treatment data the outcome nuisance is linear, but the bounded propensity and hard-label join leave the staged nuisance fits deliberately misspecified. | Lower capacity; the oracle ablation checks the array estimator separately, while the staged tolerance measures this specific misspecified composition rather than a general-purpose learner comparison. |
| 5 | `judgement` | — | Use ordinary AIPW weights with propensity clipping, not `normalize_ipw=True`. | Normalized IPW was copied from the unrelated SSM wrapper; the IRM score above is the declared method. | Clipping trades small bias for bounded variance; lack of normalization changes finite-sample weighting. |
| 6 | `framework-limitation` | `repeated-cross-fitting` | Run one fixed five-fold split and no repeated sample splitting. | Repeating the split and aggregating across repetitions is part of the estimator's recommended procedure, not an optional refinement, so this is a mechanic the source states and we do not implement. The framework is what stops us: an `XTYBatch` carries one `fold_id`, and the artifact contract, the fold-disjointness check and the checkpoint provenance are each written against a single fold assignment (`DESIGN.md` section 11.4, `repeated-cross-fitting`). | Higher Monte Carlo variance than repeated DML: the section 6 number is one draw of the split, and its stderr is over seeds rather than over partitions, so it understates the spread the published procedure averages out. |
| 7 | `judgement` | — | Preserve observed treatments and hard-fill only missing rows. | This is the P10 functional join contract. | Avoids classifier error on gold labels; the semi-supervised gain/loss comes only from formerly missing rows. |

### 5.1 Framework additions made for this card

`SSDMLATEAction` uses the existing array executor and returns portable tensor state. The open framework debt is repeated cross-fitting; one fixed five-fold assignment is used.

### Tier 2 outcome

On 2026-08-24, commit `d060df351f2fe8bac6d951c3757506c684d8b408` produced a `deviating` result: This matches the predeclared project-local staged-imputation IRM target. It validates deterministic executor/artifact plumbing and does not transfer the DML paper's inference claim to hard labels. Failed target(s): staged_absolute_ATE_error was 0.423247 +/- 0.0131 against mean <= 0.3.

## 6. Reproduction target

The target measures the staged point estimator on a fixed project-local IRM DGP.

```yaml
reproduction:
  dataset: fixed project-local staged-imputation IRM DGP
  variant: binary treatment; 50% treatment MCAR; staged hard imputation; five-fold DML2
  split: one independent 4000-row estimation population per replicate; five held-out folds
  metric: absolute_ATE_error
  published: n/a
  published_source: n/a; project-local P11 mechanism target
  tolerance: 0.30 from analytic ATE 1.0
  seeds: 20
  report: mean_and_stderr
```

### 6.1 Fixed staged-imputation IRM DGP

For replicate `r = 0..19`, use base seed `120000 + 100r` and one independent
4,000-row population. With six independent standard-normal covariates,

$$
e(x)=0.05+0.90\operatorname{sigmoid}
(1.25x_1-0.75x_2+0.5x_3),\qquad
T=\mathbb 1\{U_T<e(x)\},
$$

$$
\mu_0(x)=x_1+0.5x_2-0.25x_3,\qquad
\tau(x)=1+0.25x_4,
$$

$$
Y=\mu_0(X)+T\tau(X)+\epsilon_Y,\qquad
\epsilon_Y\sim\mathcal N(0,1).
$$

The analytic ATE is 1.0. Treatment is 50% MCAR and
`fold_id = row_id mod 5`. Standardise `X` on each fold complement and do not
standardise `Y`. The staged join is deliberately misspecified:

$$
\widetilde T=MT+(1-M)h(X),\qquad
P(\widetilde T=1\mid X=x)=0.5e(x)+0.5h(x),
$$

which is generally not logistic-linear even before the joined outcome strata
are considered. Score `|\hat\theta-1|`; require finite held-out predictions,
propensities in `[0.025,0.975]`, verified out-of-fold `used_y=false` provenance,
immutable source data, deterministic tensor state, and a generator-truth
treatment ablation with mean absolute ATE error at most 0.10.

### 6.2 Result ledger


| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-24 | `d060df351f2fe8bac6d951c3757506c684d8b408` | staged_absolute_ATE_error<br>oracle_treatment_absolute_ATE_error<br>out_of_fold_without_y<br>finite_complete_clipped_state<br>deterministic_array_state<br>source_batch_unchanged | 0.423247 +/- 0.0131<br>0.0334739 +/- 0.00624<br>1 +/- 0<br>1 +/- 0<br>1 +/- 0<br>1 +/- 0 | no |

## 7. Unknowns

| Unspecified in source | Our choice | Basis |
|---|---|---|
| The DML paper assumes observed treatment and does not specify semi-supervised imputation. | Cross-fit `p(t | x)`, emit hard labels only for missing rows, and explicitly withdraw the inference claim. | Smallest composition of the P10 artifacts with a real array estimator. |
| The old `ss_dml.py` calls `DoubleMLSSM`, which has incompatible data semantics. | Treat it as failed archaeology, not as the algorithmic authority. | Official DoubleML documentation requires `s_col` and defines SSM as sample selection/outcome attrition. |
| Nuisance learner family | Ridge linear outcome models and ridge logistic propensity, penalty `0.001`, intercept exempt. | Deterministic numpy implementation and portable tensor state. The true-treatment outcome regression is linear, but section 6 explicitly documents propensity misspecification and the further joined-data misspecification from hard labels. |
| Propensity stabilisation | Clip held-out `m_hat` to `[0.025,0.975]`; do not normalize IPW. | Fixed before results; bounds the AIPW residual terms without importing the old SSM option. |
| Logistic solver | Newton/IRLS, zero coefficients, maximum 100 iterations, relative parameter tolerance `1e-8`; fail loudly on non-convergence. | Makes array fitting deterministic and turns convergence into an executable condition. |
| Repeated cross-fitting | One supplied five-fold partition. | Section 5.6 owns this: it is a framework limitation blocked on `repeated-cross-fitting`, and a limitation is never a section 7 basis (`FIDELITY.md` section 5.1). |
| Statistical effect of row-wise cross-fitted hard labels on DML inference | Unknown; report point-estimate error only. | No reviewed source theorem covers this exact composition. |
| Published reproduction number | None for the P11 extension. | Use the fully fixed section 6 target and retain `published: n/a`. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status -> `reviewed`) | Matt | 2026-08-24 |
| Plan diffed against section 3.2 and section 4 | Codex | 2026-08-24 |
