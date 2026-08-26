# Recipe spec card: ssdml

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

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

The old name is misleading. In DoubleML, `SSM` means **sample selection
model**, where an outcome-selection variable `S` and nuisance
`pi(d,x)=P(S=1 | D=d,X=x)` are part of the score. It does not mean
semi-supervised DML. The old wrapper discards rows whose treatment is missing,
does not supply `S`, and therefore neither implements DoubleML's SSM contract
nor uses the unlabelled rows. P11 keeps the public recipe name `ssdml` but bases
the estimator on the appropriate interactive-regression ATE score and records
the treatment-imputation extension as a deviation.

## 2. Estimand and claim

- **Estimand:** For binary treatment, the average treatment effect
  `theta_0 = E[Y(1)-Y(0)]` under consistency, positivity and conditional
  exchangeability given `X`.
- **Claim:** With fully observed treatment, the DML interactive-regression
  score is Neyman-orthogonal and cross-fitting limits overfit/regularisation
  bias from the outcome and propensity nuisance estimators. P11 additionally
  tests a staged semi-supervised point estimator: cross-fit `p(t | x)` on rows
  with observed treatment, hard-label missing treatments out of fold, then run
  an explicit array-based cross-fitted ATE action on the joined batch.
- **Not claimed:** The DML paper does not analyse hard pseudo-treatment labels.
  Its root-N normality and confidence-interval results are not transferred to
  this extension. The first P11 implementation reports a point estimate and a
  diagnostic influence-score standard error only. It is not DoubleMLSSM, does
  not handle outcome attrition or nonignorable selection, does not estimate
  CATE, and supports only `K=2`.

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

P11 uses two explicit stages:

1. `propensity_labels`, `executor="cross_fit"`: `mlp_encoder` and
   `categorical_propensity` produce `T_GIVEN_X`. `ObservedTreatmentNLL` fits
   them on each fold complement's `t_observed` rows. The action emits hard
   argmax labels for held-out `t_missing` rows. Because the source is
   `p(t | x)`, artifact provenance must derive `used_y=false`; actual row sets
   must still earn `prediction_mode="out_of_fold"`.
2. `dml_ate`, `executor="array_fit"`: the stage consumes
   `propensity_labels`, receives the finite functionally joined batch once and
   invokes `SSDMLATEAction`. The action uses the supplied non-negative
   `fold_id` values. For each fold it fits, on the complement:
   - separate ridge outcome regressions `g_0(0,x)` and `g_0(1,x)`;
   - a ridge logistic propensity `m_0(x)`;
   - then evaluates the ATE score on the held-out fold after clipping
     `m_hat` to `[0.025, 0.975]`.

The action returns immutable tensor state containing `ate`,
`diagnostic_standard_error`, held-out `influence_score`, `g0_hat`, `g1_hat`,
`m_hat`, `row_id` and `fold_id`. The checkpoint's
`trained_on_row_ids` is produced by the executor from the finite input; the
action cannot assert its own provenance. No opaque sklearn or DoubleML object
is pickled.

| Paper / stage quantity | Meaning | xty2 Port / artifact | xty2 Objective / Component / action |
|---|---|---|---|
| `m_0(x)` in label stage | Treatment propensity used for missing-label prediction | `T_GIVEN_X` | `mlp_encoder`, `categorical_propensity` |
| `-log p(t_i | x_i)` | Fit treatment predictor where treatment is observed | `T_GIVEN_X` | `ObservedTreatmentNLL` |
| `argmax p(t | x)` | Hard out-of-fold treatment labels | `PseudoLabels` | `PseudoLabelAction(port=T_GIVEN_X, rows="t_missing")` |
| `g_0(0,x), g_0(1,x)` | DML outcome nuisances | array checkpoint tensor state | `SSDMLATEAction` ridge fits |
| `m_0(x)` in DML action | DML score propensity nuisance | array checkpoint tensor state | `SSDMLATEAction` logistic-ridge fit |
| `psi(W;theta,eta)` | Held-out orthogonal ATE score | array checkpoint tensor state | `SSDMLATEAction` score aggregation |

## 4. Mechanics checklist

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

| # | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|
| 1 | Implement the IRM/AIPW ATE score, not `DoubleMLSSM`. | `DoubleMLSSM` is for outcome sample selection and requires a separate `S`; the legacy wrapper supplies none. | Replaces a non-runnable/misidentified baseline with the estimator matching the declared ATE problem. |
| 2 | Prepend cross-fitted hard treatment imputation using `p(t | x)`. | P11 must exercise treatment-missing semi-supervision and the P10 cross-fit artifact path. | Can add efficiency when labels are accurate and bias when they are not; the DML theorem does not cover it. |
| 3 | Do not claim the DML standard error after pseudo-labelling. | Orthogonality protects nuisance estimation under the paper's observed-data model, not classification error in an imputed treatment. | No effect on the point estimate; prevents invalid inferential claims. |
| 4 | Use deterministic ridge linear/logistic nuisances rather than the old source's random forests and `DoubleML` dependency. | The array artifact must be portable tensor state, and numpy is already a core dependency. On the oracle true-treatment data the outcome nuisance is linear, but the bounded propensity and hard-label join leave the staged nuisance fits deliberately misspecified. | Lower capacity; the oracle ablation checks the array estimator separately, while the staged tolerance measures this specific misspecified composition rather than a general-purpose learner comparison. |
| 5 | Use ordinary AIPW weights with propensity clipping, not `normalize_ipw=True`. | Normalized IPW was copied from the unrelated SSM wrapper; the IRM score above is the declared method. | Clipping trades small bias for bounded variance; lack of normalization changes finite-sample weighting. |
| 6 | Run one fixed five-fold split and no repeated sample splitting. | P10 carries one `fold_id`; repeated splitting is not needed to prove the executor seam. | Higher Monte Carlo variance than repeated DML. |
| 7 | Preserve observed treatments and hard-fill only missing rows. | This is the P10 functional join contract. | Avoids classifier error on gold labels; the semi-supervised gain/loss comes only from formerly missing rows. |

### Tier 2 outcome

On 2026-08-24, commit `d060df351f2fe8bac6d951c3757506c684d8b408` produced a `deviating` result: This matches the predeclared project-local staged-imputation IRM target. It validates deterministic executor/artifact plumbing and does not transfer the DML paper's inference claim to hard labels. Failed target(s): staged_absolute_ATE_error was 0.423247 +/- 0.0131 against mean <= 0.3.

## 6. Reproduction target

The paper's Monte Carlo tables assume fully observed treatment. They cannot
validate the P11 imputation extension. P12 must run the fixed project-local
mechanism target below and report it separately from the paper's inferential
claims.

### 6.1 Fixed staged-imputation IRM DGP

Run twenty replicates `r in {0, ..., 19}` with base seed
`s_r = 120000 + 100*r`. Each independent estimation population contains 4,000
rows. Draw `X`, `U_T`, `epsilon_Y` and `U_M` independently in that order. Let
`X in R^6` have independent standard-normal columns,

$$
e(x)=0.05+0.90\operatorname{sigmoid}
  (1.25x_1-0.75x_2+0.5x_3),
\qquad T=\mathbb 1\{U_T<e(x)\},
$$

$$
\mu_0(x)=x_1+0.5x_2-0.25x_3,
\qquad \tau(x)=1+0.25x_4,
$$

$$
Y=\mu_0(X)+T\tau(X)+\epsilon_Y,
\qquad \epsilon_Y\sim\mathcal N(0,1).
$$

The analytic ATE is `1.0`. In training, set
`t_observed = 1{U_M < 0.50}` independently. Assign `fold_id = row_id mod 5`.
Standardise `X` using the complement's mean and standard deviation inside each
fold fit; apply those frozen statistics to that fold's held-out rows. Do not
standardise `Y`.

This is not a correctly specified ridge/logistic benchmark. Even before
imputation, `0.05 + 0.90*sigmoid(linear(x))` is not representable as a single
logistic-linear propensity because its tails are bounded at `0.05` and `0.95`.
Hard imputation creates a second misspecification. If `M` is the observation
indicator, `rho=P(M=1)=0.5` and `h(x)` is the hard classifier decision, then the
idealised joined treatment satisfies

$$
\widetilde T=MT+(1-M)h(X),
\qquad
P(\widetilde T=1\mid X=x)=\rho e(x)+(1-\rho)h(x),
$$

which is generally not logistic-linear. Conditioning $Y$ on
$(X,\widetilde T)$ also mixes true-treatment and imputed strata, so the original
linear potential-outcome equations do not make the joined-data outcome
nuisances linear. The staged target therefore tests deterministic executor and
artifact plumbing under explicit nuisance misspecification; its tolerance is a
fixed mechanism target, not a consequence of the DML theorem or correct
specification.

The primary metric is absolute ATE error `|theta_hat - 1.0|`. Secondary
guardrails are: finite held-out predictions for every row; all propensities in
`[0.025,0.975]`; `used_y=false` and verified out-of-fold provenance for the
label artifact; no source-batch mutation; and the same seed produces
bit-identical tensor state. For every replicate, also run the same
`SSDMLATEAction` with generator-truth treatment in place of the joined labels.
Report its ATE error as an `oracle_treatment` ablation and require its
twenty-seed mean absolute error to be at most `0.10`. Report the staged-minus-
oracle estimate difference without a pass threshold. The ablation separates a
broken array estimator from error introduced by the documented pseudo-label
composition; it does not make the staged estimator inferentially valid.

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
| Repeated cross-fitting | One supplied five-fold partition. | P10 executor owns one actual fold assignment; repeated splitting has no second consumer. |
| Statistical effect of row-wise cross-fitted hard labels on DML inference | Unknown; report point-estimate error only. | No reviewed source theorem covers this exact composition. |
| Published reproduction number | None for the P11 extension. | Use the fully fixed section 6 target and retain `published: n/a`. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status -> `reviewed`) | Matt | 2026-08-24 |
| Plan diffed against section 3.2 and section 4 | Codex | 2026-08-24 |
