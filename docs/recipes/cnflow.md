# Recipe spec card: cnflow

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Neural Spline Flows](https://arxiv.org/abs/1906.04032) supplies the density parameterisation. `cnflow` itself is a project-local composition, not a method proposed in that paper. |
| Authors, year | Conor Durkan, Artur Bekasov, Iain Murray and George Papamakarios, 2019 |
| DOI / arXiv | [arXiv:1906.04032v2](https://arxiv.org/abs/1906.04032v2); [10.48550/arXiv.1906.04032](https://doi.org/10.48550/arXiv.1906.04032); NeurIPS 2019 |
| Version used | arXiv v2, 2019-12-02. Sections 2–3 define the flow and rational-quadratic transform; section 5 and Appendix B supply the architectural defaults cited below. |
| Reference implementation | Project-local source: [`mattsq/XTYLearner` `cnflow_model.py` @ `35734ec2d5a62d54a59eca38d1e31423da31e1ea`](https://github.com/mattsq/XTYLearner/blob/35734ec2d5a62d54a59eca38d1e31423da31e1ea/xtylearner/models/cnflow_model.py). The categorical-context correction entered in [PR #227](https://github.com/mattsq/XTYLearner/pull/227) / merge `bbc84a62f43a231830d4e95e629666ee167a612f`; the spline variant entered in [PR #260](https://github.com/mattsq/XTYLearner/pull/260) / merge `c5303ea859fb3034f8471a879ae3aff6eddcef5a`. Flow primitives: [`bayesiains/nflows` v0.14](https://github.com/bayesiains/nflows), [10.5281/zenodo.4296287](https://doi.org/10.5281/zenodo.4296287). |
| Reference impl. runnable? | Not attempted in the P7 card pass. Its focused tests passed when the cited changes merged; the repository now pins an older Torch range and carries unrelated legacy dependencies. |

## 2. Estimand and claim

- **Estimand:** the full `p(y | x,t=k)` and its conditional mean for each categorical treatment.
- **Method claim:** a conditional rational-quadratic spline flow supplies exact density evaluation and sampling; candidate treatment is conditioner context, never an invertible event coordinate.
- **Scope:** this causal composition is project-local. Neural Spline Flows supplies the density primitive, not a treatment-effect, missing-label, or identification claim.

## 3. Equations and mapping

### 3.1 As published

Durkan et al. define a normalising flow by their Eq. (1),

$$
\mathbf y = f(\mathbf z), \qquad \mathbf z \sim \pi(\mathbf z),
$$

and obtain the data density by the change of variables in Eq. (2),

$$
p(\mathbf y)
= \pi\!\left(f^{-1}(\mathbf y)\right)
  \left|\det\frac{\partial f^{-1}}{\partial \mathbf y}\right|.
$$

Their rational-quadratic transform (Eq. (4)) replaces the affine elementwise
transform inside an autoregressive flow. It is monotone, differentiable and
analytically invertible, with linear tails outside `[-B, B]`. Section 3.2 calls
a stack of these autoregressive transforms an RQ-NSF (AR). The paper uses a
standard-normal base, eight bins and `B=3` across its experiments (section 5),
and two residual blocks in the tabular autoregressive conditioners (Appendix
B.1). Those facts govern the flow primitive, not the causal objective below.

### 3.2 Mapping to xty2

Each candidate treatment is one-hot context for `conditional_flow`. The existing propensity and observed/marginal likelihood objectives train one stage. Conditional means use 100 fixed antithetic base draws so candidate and single-treatment calls agree deterministically.

| Paper / P7 symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `x_i` | Raw covariates | `X_RAW` | virtual source node |
| `h_psi(x_i)` | Shared covariate context | `X_REPR` | `mlp_encoder` |
| `e_k` | Categorical candidate treatment used only as conditioner context | n/a; supplied through the distribution call | `conditional_flow` encodes candidate class indices internally |
| `p_phi(y | x,t)` | Conditional RQ-NSF outcome density | `Y_GIVEN_XT` | `conditional_flow` |
| `p_theta(t | x)` | Categorical treatment distribution | `T_GIVEN_X` | existing `categorical_propensity` |
| `L_y` | Complete-case flow NLL | `Y_GIVEN_XT` | existing `ObservedOutcomeNLL` |
| `L_t` | Complete-case propensity NLL | `T_GIVEN_X` | existing `ObservedTreatmentNLL` |
| `L_marg` | Exact missing-treatment observed-data NLL | `Y_GIVEN_XT`, `T_GIVEN_X` | existing `MissingTreatmentMarginalNLL(grad_path="both")` |
| `L_P7` | Single objective mix | both predicted ports | stage `joint_fit` with three `Weighted` objectives |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: n/a
  gradient_clipping: none                     # old XTYLearner cnflow used none; NSF Appendix B.1 clipping is not transferred to this different recipe
  marginal_nll_grad_path: both                # P7 abstraction test; the same existing objective trains both heads

teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 1.0
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: constant 1.0
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a

optimisation:
  optimiser: adam(betas=(0.9, 0.999), eps=1e-8)  # xty2 P7 choice; old benchmark harness used Adam
  lr: 0.001                                      # old XTYLearner benchmark harness default
  lr_schedule: constant 1.0                      # old XTYLearner benchmark harness used no scheduler
  weight_decay: none                             # old XTYLearner model and harness set none
  batch_size: n/a                                # external BatchSource; old benchmark used 10
  labelled_unlabelled_ratio: n/a                 # no enforced per-batch quota
  total_steps_or_epochs: 3000 optimiser steps    # xty2 P7 choice, matching the P5 smoke budget

architecture:
  widths_depths:
    mlp_encoder: [128, 128]                      # old XTYLearner cond_net
    conditional_flow: 5 RQ-NSF autoregressive transforms, each hidden=128 with 2 residual blocks, 8 bins, tails="linear", tail_bound=3; random permutation after each transform
    categorical_propensity: linear 128 -> K
  activation:
    mlp_encoder: relu                            # old XTYLearner cond_net
    conditional_flow: relu                       # nflows ResMADE conditioner default
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: none
    conditional_flow: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    conditional_flow: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: torch Linear default Kaiming-uniform  # old XTYLearner make_mlp default
    conditional_flow: nflows 0.14 defaults
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0  # existing shared component
  output_parameterisation:
    conditional_flow: StandardNormal base -> 5 conditional RQ-NSF(AR) transforms with explicit linear tails outside [-3, 3] over flattened continuous Y; categorical t is one-hot context; 100 fixed-antithetic draws approximate mean
    categorical_propensity: K softmax logits

data:
  standardisation: n/a                          # caller-owned; benchmark records preprocessing
  outcome_scaling: n/a                          # caller-owned; benchmark records preprocessing
  treatment_encoding: one-hot K-vector appended to X_REPR as flow context; never part of the flow event
  split_protocol: n/a                           # Tier 1 fixture and P12 runner own their splits
  missingness_mechanism: n/a                    # Tier 1 fixture applies exactly 50% treatment MCAR
```

## 5. Deviations from the paper and project-local source

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Condition an outcome flow on covariates and categorical treatment, and add a propensity head plus missing-treatment likelihood. | Durkan et al. provide the density primitive, not a causal recipe. These additions are the P7 purpose. | No direct comparison with the paper's unconditional UCI likelihood numbers is valid. |
| 2 | `judgement` | — | Use five autoregressive spline transforms with random permutations and hidden width 128, rather than the paper's ten steps with LU linear transforms and dataset-specific widths. | These are the pinned XTYLearner defaults; P7 ports the project-local recipe rather than reproducing an NSF table. | Lower capacity and cheaper sampling; direction on the project-local benchmark is unknown. |
| 3 | `judgement` | — | Keep the old XTYLearner `tails="linear"` setting and use `tail_bound=3` for every continuous outcome dimension instead of its inherited `nflows` default of 1. | `nflows` v0.14 otherwise defaults `tails=None`, which selects the bounded spline and makes `tail_bound` ineffective. The paper defines linear tails and reports `B=3` as robust in `[1,5]`; both arguments must therefore be explicit. | Preserves unrestricted support and likely improves flexibility for standardised outcomes outside `[-1,1]`; direction otherwise unknown. |
| 4 | `judgement` | — | Use RQ-NSF autoregressive transforms for every continuous event dimension. Old XTYLearner switched to affine MAF when `D_y > 1`. | One declared parameterisation avoids an outcome-rank branch and tests the same distribution contract for scalar and vector outcomes. | More flexible but slower for vector outcomes. |
| 5 | `judgement` | — | Replace the old missing-label `logsumexp_k p(y|x,k)` with the existing propensity-weighted `logsumexp_k p(t=k|x)p(y|x,k)`. | The old expression omits the treatment prior and is not the observed-data likelihood. P7 exists to prove the generic objective can supply the correct calculation unchanged. | Changes both optimum and gradient path on missing-treatment data; expected to improve a correctly specified DGP, not guaranteed under misspecification. |
| 6 | `judgement` | — | Omit training-time outcome jitter, the optional MMD branch and optional inverse-propensity weighting from `CNFlowModel.loss`. | Losses must remain independent of parameterisation. MMD was inactive by default; jitter would make repeated `log_prob` calls disagree; row weights already belong to `XTYBatch.weight`. | MMD omission has no default effect; jitter omission may reduce regularisation; explicit row weights retain their normal effect. |
| 7 | `judgement` | — | Approximate `mean` with 100 fixed antithetic base draws rather than 100 fresh samples per prediction. | Candidate-treatment means must agree column-wise and evaluation must be reproducible. | Reduces Monte Carlo variance between calls; a small deterministic integration error remains. |
| 8 | `judgement` | — | Do not expose the old likelihood-only `predict_treatment_proba(x,y)` as `T_GIVEN_XY`. | That quantity omits the treatment prior, and no P7 objective consumes a posterior. The real posterior component first has a consumer in P11. | None for the P7 graph or losses. |

### 5.1 Framework additions made for this card

`conditional_flow` is a new `OutcomeDistribution` implementation behind the existing `Y_GIVEN_XT` port. No new port, objective protocol, or execution mode is required.

### Tier 2 outcome

On 2026-08-24, commit `d060df351f2fe8bac6d951c3757506c684d8b408` produced a `deviating` result: This is the predeclared project-local conditional-density validation target. Matching it validates the CNFlow recipe's limited claim and is not a reproduction of Durkan et al. Failed target(s): paired_d_conditional_NLL was 1.88504 +/- 0.225 nat/row against mean <= -0.1 nat/row.

## 6. Reproduction target

The target is a paired project-local density comparison, not a Neural Spline Flows paper reproduction.

```yaml
reproduction:
  dataset: xty2 analytic non-Gaussian outcome DGP
  variant: section 6.1 scalar-Y equations; six Gaussian X; binary confounded T; centred heteroskedastic two-component outcome mixture
  samples: {train: 4096, validation: 2048, test: 4096}
  missingness: exactly 2048 training treatments MCAR; all outcomes and all validation/test treatments observed
  preprocessing: raw X; population-standardise Y from all training outcomes; evaluate in original Y units
  pairing: identical populations, missingness mask, ordered batches and bit-identical initial shared parameters
  training: batch_size=256; 3000 final-checkpoint Adam steps; validation is diagnostic only
  primary_metric: test conditional outcome NLL p(Y|X,T), explicitly not joint or missing-treatment marginal NLL
  guardrail: test sqrt_PEHE against analytic tau(X)
  published: n/a                         # project-local recipe; no published causal result exists
  published_source: n/a                  # threshold is predeclared below, not attributed to Durkan et al.
  tolerance: mean paired d_NLL <= -0.10 nat/row; mean paired d_PEHE <= 0.10
  seeds: r=0..9 with base 70000+100*r and fixed stream offsets from sections 6.1-6.2
  report: per-model means plus paired-difference means and sample stderrs over 10 replicates
```

### 6.1 Fixed DGP

For replicate `r = 0..9`, use base seed `70000 + 100r` and independent
train/validation/test populations of 4,096/2,048/4,096 rows. Let
`X_j iid ~ N(0,1)` for `j=1..6` and

$$
e(x)=0.1+0.8\operatorname{sigmoid}(1.25x_1-x_2+0.5x_3),\qquad
T=\mathbb 1\{U_T<e(x)\},
$$

$$
b(x)=0.8x_1+0.5(x_2^2-1)-0.6x_3x_4+0.3\sin(2x_5),\quad
\tau(x)=1+0.5\tanh(x_1),
$$

$$
q_t(x)=0.2+0.6\operatorname{sigmoid}(0.75x_5-0.5x_6+0.5t),\quad
\sigma_t(x)=0.15+0.10\operatorname{sigmoid}(x_6)+0.05t,
$$

$$
S=\mathbb 1\{U_S<q_T(x)\},\qquad
Y=b(x)+T\tau(x)+1.5\{S-q_T(x)\}+\sigma_T(x)\epsilon.
$$

Thus `mu_t(x)=b(x)+t*tau(x)` and the exact conditional density is

$$
q_t\,\mathcal N(\mu_t+1.5(1-q_t),\sigma_t^2)
+(1-q_t)\,\mathcal N(\mu_t-1.5q_t,\sigma_t^2).
$$

Exactly 2,048 training treatments are missing under a seeded MCAR permutation;
all outcomes and validation/test treatments are observed. Use raw `X` and
population-standardise `Y` from all training outcomes, evaluating means and
likelihoods back in original units.

### 6.2 Fixed fit and evidence

Compare the declared flow against the P5 three-layer Gaussian TARNet head while
holding the encoder, propensity, objectives, optimiser, ordered 256-row batches,
and shared initial parameters fixed. Train 3,000 deterministic CPU Adam steps;
validation is diagnostic only and the final test checkpoint is fixed. The
primary difference is conditional `p(Y|X,T)` NLL, with analytic-`tau(X)`
`sqrt(PEHE)` as guardrail. The predeclared pass rule is
`mean(d_NLL) <= -0.10` and `mean(d_PEHE) <= 0.10` over ten paired replicates.
The immutable result at commit `d060df351f2fe8bac6d951c3757506c684d8b408`
was `d_NLL = 1.88504 +/- 0.225` and
`d_PEHE = -0.226847 +/- 0.0637`: better effect error did not offset the failed
density target.

### 6.3 Result ledger


| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-24 | `d060df351f2fe8bac6d951c3757506c684d8b408` | paired_d_conditional_NLL<br>paired_d_sqrt_PEHE | 1.88504 +/- 0.225 nat/row<br>-0.226847 +/- 0.0637 outcome units | no |

## 7. Unknowns

| Unspecified in source | Our choice | Basis |
|---|---|---|
| There is no paper defining `cnflow` as a causal method. | Treat XTYLearner as the recipe source and NSF as the flow primitive; state every causal addition as project-local. | Avoids transferring claims from an unconditional density paper. |
| The old code set `tails="linear"` but did not make the spline tail bound explicit. | Preserve `tails="linear"` and set `B=3`; both constructor arguments are machine-checked. | `nflows` v0.14 defaults `tails=None`, so stating only `B` does not request the unconstrained wrapper. Durkan et al. define linear tails and fix `B=3`. |
| The old code used spline transforms only for scalar outcomes. | Use the same RQ-NSF(AR) parameterisation for any flattened continuous event shape. | The transform supports it, removes an architecture branch and exercises the protocol more strongly. |
| The old code estimated means with 100 new random draws. | 100 fixed antithetic standard-normal draws, generated independently of the global RNG and shared by every candidate. | Preserves the reference budget while satisfying deterministic, column-consistent means. |
| Flow sampling returns a library-specific axis order. | The distribution wrapper always returns `[n, B, *Dy]` or `[n, B, C, *Dy]`. | `DESIGN.md` section 3.1 is authoritative at the port boundary. |
| The old model's training budget was owned by whichever trainer called it. | 3,000 Adam steps at `1e-3`, constant rate, no decay or clipping. | Matches P5's smoke budget and the old benchmark learning rate while keeping the P7 stage explicit. |
| No published result exists. | Predeclare the complete seed-locked non-Gaussian DGP, paired fit and exact evaluation calculation in section 6, and do not call it a paper reproduction. | A directional capability test is more honest than borrowing an unrelated NSF UCI number or a noisy legacy ledger row, and the threshold cannot be tuned by changing an underspecified benchmark. |
| Whether a project-local result needs a new card status. | Leave the status vocabulary unchanged in P7 and flag the question for the Gate 1 review. | Changing the fidelity system is not required to test the outcome-head abstraction. |
| Whether the shared encoder should retain TARNet's ELU/L2 configuration. | Use the XTYLearner cnflow encoder: two ReLU layers of width 128 with no normalisation. | The graph component is shared; its recipe configuration remains method-specific and is explicitly visible in the plan. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status -> `reviewed`) | mattsq | 2026-08-23 |
| Plan diffed against section 3.2 and section 4 | Codex | 2026-08-23 |
