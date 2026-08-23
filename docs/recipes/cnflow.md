# Recipe spec card: cnflow

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

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

The authority order is deliberate. The old XTYLearner implementation defines
what the recipe name means: a conditional outcome density with categorical
treatment in the context. Durkan et al. define the flow machinery, but do not
propose this causal or semi-supervised recipe. P7 therefore tests an xty2
abstraction boundary; it is not presented as a paper reproduction.

## 2. Estimand and claim

- **Estimand:** The full treatment-conditional outcome distribution
  `p(y | x, t=k)` for each categorical treatment `k`, and its conditional mean
  `mu_k(x) = E[Y | X=x, T=k]`. Under consistency, positivity and conditional
  exchangeability, contrasts `mu_k(x) - mu_j(x)` identify CATEs and the fitted
  distributions represent conditional potential-outcome marginals.
- **Claim:** A conditional rational-quadratic neural spline flow can evaluate
  an exact density and draw exact samples while allowing a more flexible
  outcome distribution than P5's fixed-scale Gaussian. In P7, placing the class
  label in the conditioner makes every candidate `t` evaluable without treating
  a discrete label as a continuous flow coordinate. That is the only claim this
  packet needs in order to test the `OutcomeDistribution` boundary.
- **Not claimed:** Durkan et al. do not study treatment effects, missing
  treatments or causal identification. The old XTYLearner recipe has no
  peer-reviewed method claim or published benchmark. A flexible conditional
  density does not repair confounding, non-overlap or MNAR treatment labels;
  neither likelihood nor CATE accuracy follows from the flow being invertible.

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

#### Conditional and semi-supervised construction

For candidate treatment `k`, P7 conditions every transform on

$$
c_{ik} = [h_\psi(x_i),\; e_k],
$$

where `e_k` is a `K`-class one-hot vector. Treatment is never concatenated to
`y` and never enters the invertible event space. With `g_phi` denoting the
data-to-base transform used for density evaluation,

$$
z_{ik} = g_\phi(y_i; c_{ik}), \qquad
\log p_\phi(y_i\mid x_i,t=k)
= \log\pi(z_{ik})
  + \log\left|\det\frac{\partial g_\phi(y_i;c_{ik})}{\partial y_i}\right|.
$$

Let `m_i` indicate that treatment is observed. The single P7 stage fits the
same three independent objectives used by P5:

$$
\mathcal L_y
= -\frac{1}{B}\sum_{i:m_i=1}
  w_i\log p_\phi(y_i\mid x_i,t_i),
$$

$$
\mathcal L_t
= -\frac{1}{B}\sum_{i:m_i=1}
  \log p_\theta(t_i\mid x_i),
$$

$$
\mathcal L_{\mathrm{marg}}
= -\frac{1}{B}\sum_{i:m_i=0}
  \log\sum_{k=0}^{K-1}
  p_\theta(t=k\mid x_i)\,
  p_\phi(y_i\mid x_i,t=k),
$$

and

$$
\mathcal L_{\mathrm{P7}}
= \mathcal L_y + \mathcal L_t + \mathcal L_{\mathrm{marg}}.
$$

All three terms use `population` reduction, which supplies the `1/B` factors.
The marginal term has no stop-gradient and therefore trains the encoder, flow
and propensity head. This is a coherent observed-data likelihood, not the old
XTYLearner loss: that code used an unnormalised uniform sum over treatments
when `t` was missing and did not include `p_theta(t | x)`.

The flow has no analytic conditional mean. `mean(t)` uses 100 fixed,
antithetic standard-normal base draws shared across every row and candidate,

$$
\widehat\mu_k(x_i)
= \frac{1}{100}\sum_{q=1}^{100} f_\phi(z_q;c_{ik}).
$$

The fixed common draws make the candidate and one-treatment calls numerically
identical, as required by the executable contract, while `sample(t, n)` still
draws from Torch's global RNG and is reproducible under the caller's seed. The
card calls this an approximation wherever it says "mean"; it does not silently
promote a Monte Carlo estimate to an analytic moment.

#### Component and objective mapping

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

No view, consistency loss, outcome jitter, MMD penalty, posterior
`T_GIVEN_XY`, balance penalty, teacher, second realisation or second stage
belongs in P7.

## 4. Mechanics checklist

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

`conditional_flow` supports continuous `OutcomeSpec` values only. It flattens
the declared trailing event shape for `nflows` and restores it at the
distribution boundary. A categorical outcome fails when the graph is built.
The `n/a` batching and preprocessing entries remain outside the current
executor for the same reason documented in the TARNet card: the recipe consumes
an external `BatchSource` and cannot honestly claim to enforce them.

## 5. Deviations from the paper and project-local source

| # | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|
| 1 | Condition an outcome flow on covariates and categorical treatment, and add a propensity head plus missing-treatment likelihood. | Durkan et al. provide the density primitive, not a causal recipe. These additions are the P7 purpose. | No direct comparison with the paper's unconditional UCI likelihood numbers is valid. |
| 2 | Use five autoregressive spline transforms with random permutations and hidden width 128, rather than the paper's ten steps with LU linear transforms and dataset-specific widths. | These are the pinned XTYLearner defaults; P7 ports the project-local recipe rather than reproducing an NSF table. | Lower capacity and cheaper sampling; direction on the project-local benchmark is unknown. |
| 3 | Keep the old XTYLearner `tails="linear"` setting and use `tail_bound=3` for every continuous outcome dimension instead of its inherited `nflows` default of 1. | `nflows` v0.14 otherwise defaults `tails=None`, which selects the bounded spline and makes `tail_bound` ineffective. The paper defines linear tails and reports `B=3` as robust in `[1,5]`; both arguments must therefore be explicit. | Preserves unrestricted support and likely improves flexibility for standardised outcomes outside `[-1,1]`; direction otherwise unknown. |
| 4 | Use RQ-NSF autoregressive transforms for every continuous event dimension. Old XTYLearner switched to affine MAF when `D_y > 1`. | One declared parameterisation avoids an outcome-rank branch and tests the same distribution contract for scalar and vector outcomes. | More flexible but slower for vector outcomes. |
| 5 | Replace the old missing-label `logsumexp_k p(y|x,k)` with the existing propensity-weighted `logsumexp_k p(t=k|x)p(y|x,k)`. | The old expression omits the treatment prior and is not the observed-data likelihood. P7 exists to prove the generic objective can supply the correct calculation unchanged. | Changes both optimum and gradient path on missing-treatment data; expected to improve a correctly specified DGP, not guaranteed under misspecification. |
| 6 | Omit training-time outcome jitter, the optional MMD branch and optional inverse-propensity weighting from `CNFlowModel.loss`. | Losses must remain independent of parameterisation. MMD was inactive by default; jitter would make repeated `log_prob` calls disagree; row weights already belong to `XTYBatch.weight`. | MMD omission has no default effect; jitter omission may reduce regularisation; explicit row weights retain their normal effect. |
| 7 | Approximate `mean` with 100 fixed antithetic base draws rather than 100 fresh samples per prediction. | Candidate-treatment means must agree column-wise and evaluation must be reproducible. | Reduces Monte Carlo variance between calls; a small deterministic integration error remains. |
| 8 | Do not expose the old likelihood-only `predict_treatment_proba(x,y)` as `T_GIVEN_XY`. | That quantity omits the treatment prior, and no P7 objective consumes a posterior. The real posterior component first has a consumer in P11. | None for the P7 graph or losses. |

## 6. Reproduction target

There is no paper-level `cnflow` result to reproduce. Substituting an RQ-NSF
UCI density number would test a different, unconditional model; calling the old
single-run XTYLearner ledger entry "published" would be equally misleading.
P12 must instead run the fully fixed project-local discrimination benchmark
below. Nothing in the DGP, preprocessing, fit, checkpoint selection or metric
may be changed after observing results.

### 6.1 Fixed DGP

Run ten paired replicates indexed by `r in {0, ..., 9}`. Replicate `r` has base
seed `s_r = 70000 + 100*r`. All generation uses CPU `torch.Generator` instances
and `float32`. Independent train, validation and test populations have 4,096,
2,048 and 4,096 rows and use seeds `s_r+1`, `s_r+2` and `s_r+3`. Within each
population, draw tensors in this exact order: `X`, `U_T`, `U_S`, `epsilon`.
For every row, with six independent standard-normal covariates,

$$
X_j \overset{\mathrm{iid}}{\sim} \mathcal N(0,1), \qquad j=1,\ldots,6,
$$

$$
e(x)=0.1+0.8\,\operatorname{sigmoid}
\left(1.25x_1-x_2+0.5x_3\right), \qquad
T=\mathbb 1\{U_T<e(x)\}, \quad U_T\sim\operatorname{Uniform}(0,1),
$$

$$
b(x)=0.8x_1+0.5(x_2^2-1)-0.6x_3x_4+0.3\sin(2x_5),
\qquad
\tau(x)=1+0.5\tanh(x_1),
$$

and, for candidate treatment `t in {0,1}`,

$$
q_t(x)=0.2+0.6\,\operatorname{sigmoid}(0.75x_5-0.5x_6+0.5t),
\qquad
\sigma_t(x)=0.15+0.10\,\operatorname{sigmoid}(x_6)+0.05t.
$$

Finally draw `U_S ~ Uniform(0,1)` and `epsilon ~ N(0,1)`, independently, and
set

$$
S=\mathbb 1\{U_S<q_T(x)\}, \qquad
Y=b(x)+T\tau(x)+1.5\{S-q_T(x)\}+\sigma_T(x)\epsilon.
$$

Thus the analytic conditional mean is
`mu_t(x) = b(x) + t*tau(x)`, the CATE is `tau(x)`, and the exact conditional
density is

$$
q_t\,\mathcal N\!\left(\mu_t+1.5(1-q_t),\sigma_t^2\right)
+(1-q_t)\,\mathcal N\!\left(\mu_t-1.5q_t,\sigma_t^2\right).
$$

Every outcome is observed. In the training population only, generate
`torch.randperm(4096)` with seed `s_r+4`; the first 2,048 indices have missing
treatment and the rest have observed treatment. Validation and test treatments
remain fully observed. This is exact 50% training-treatment MCAR and is
independent of `X`, `T` and `Y`.

Use `X` without preprocessing. From all 4,096 training outcomes, including
rows whose treatment is hidden, compute the population moments
`y_bar = mean(Y)` and `s_y = sqrt(mean((Y-y_bar)^2))`. Both fits consume
`Z=(Y-y_bar)/s_y`; the same training moments transform validation/test
outcomes and convert predicted means back to original outcome units. No row or
treatment-group weights are supplied.

### 6.2 Fixed fit and evaluation

The flow fit is exactly the section 4 recipe. The comparator uses the same
two-layer 128-wide ReLU encoder, categorical propensity head, three objectives,
objective weights, Adam settings and no weight decay, but replaces
`conditional_flow` with the P5-style `tarnet_head`: two independent treatment
heads, each `[100, 100, 100]` with ELU activation, no normalisation or dropout,
`normal std=0.1/sqrt(fan_in), bias=0` initialisation, and
`Normal(mean_k, 1.0)` output in standardised-outcome units. Reset the global
CPU Torch RNG to `s_r+6`, initialise the shared encoder and propensity once,
and copy them bit-for-bit to both fits. Reset it to `s_r+7` immediately before
initialising the flow head (including its permutations) and to `s_r+8`
immediately before initialising the Gaussian outcome head.

Before either fit, use seed `s_r+5` to precompute 3,000 batches. Batch `j` is
the first 256 indices of a fresh `torch.randperm(4096)` call; both fits consume
the identical ordered batch-index table. Run both fits on CPU with Torch
deterministic algorithms enabled, train for exactly 3,000 optimiser steps and
evaluate the final checkpoint. The validation population is logged only as a
diagnostic: it cannot select a checkpoint, tune a parameter or change this
protocol. The test population is evaluated once after both fits.

For model `M`, the primary per-replicate metric is the **conditional outcome
NLL** on the fully labelled test population,

$$
\operatorname{NLL}^{(r)}_M
=-\frac{1}{4096}\sum_i
\log p_{M,Y}(Y_i\mid X_i,T_i),
\qquad
\log p_{M,Y}(y\mid x,t)
=\log p_{M,Z}((y-\bar y)/s_y\mid x,t)-\log s_y.
$$

This is neither joint outcome-treatment NLL nor missing-treatment marginal
NLL. The guardrail is test-set

$$
\sqrt{\operatorname{PEHE}}^{(r)}_M
=\sqrt{\frac{1}{4096}\sum_i
\left[\{\widehat\mu_{M,1}(X_i)-\widehat\mu_{M,0}(X_i)\}
-\tau(X_i)\right]^2},
$$

in original outcome units. The flow uses its fixed 100-draw antithetic mean;
the Gaussian comparator uses its analytic head means. Define paired
differences `d_NLL^(r) = NLL_flow^(r) - NLL_Gaussian^(r)` and
`d_PEHE^(r) = sqrt_PEHE_flow^(r) - sqrt_PEHE_Gaussian^(r)`. The benchmark
passes only when the ten-replicate means satisfy
`mean(d_NLL) <= -0.10` nat/row and `mean(d_PEHE) <= 0.10`. Report each model's
mean plus the paired differences; every standard error is the sample standard
deviation across replicates divided by `sqrt(10)`.

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

Passing this target can validate the project-local claim but cannot turn it
into a reproduction of Durkan et al.; the result ledger must say that plainly.
The eventual status vocabulary may need a project-local `validated` state. P7
does not add one: that is a fidelity-system decision, not flow implementation.

### 6.3 Result ledger

| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

### 6.4 P7 and Gate 1 acceptance

1. The graph contains exactly `mlp_encoder` (`X_RAW -> X_REPR`),
   `conditional_flow` (`X_REPR -> Y_GIVEN_XT`) and the existing
   `categorical_propensity` (`X_REPR -> T_GIVEN_X`). Stage `joint_fit` contains
   exactly the three existing objectives and row populations in section 4. The
   compiled plan has one identity/student forward pass and no view.
2. `conditional_flow` passes the unchanged candidate-treatment conformance
   suite for `log_prob`, `mean` and `sample` with `B != K`, for scalar and vector
   continuous outcomes. Tests also show that permuting only candidate `t`
   changes context without changing the flow event dimension.
3. `MissingTreatmentMarginalNLL` is imported and reused without modification.
   `xty2/objectives/marginal.py` must not appear in the P7 changed-file list,
   and an exact marginal calculation through the real flow equals an explicit
   loop over `k` including propensity log-probabilities.
4. The recipe contains declarations only and no conditional. Outcome-kind and
   event-shape handling live inside the component/distribution contract, not in
   `xty2/recipes/cnflow.py`.
5. On the Tier 1 analytic DGP, the four `FIDELITY.md` assertions pass: loss
   decreases; propensity beats the held-out frequency baseline; ATE is in the
   declared wide band; and the 50%-MCAR marginal recipe beats its paired
   complete-case ablation. Those are wiring assertions, not the section 6
   non-Gaussian validation target.
6. The card-to-plan cross-check covers every non-`n/a` section 4 key, including
   `data.treatment_encoding`, the flow architecture and the per-objective
   settings. A mutation that removes the flow head or replaces categorical
   context with a flow coordinate, removes `tails="linear"`, or changes
   `tail_bound=3` fails a named invariant.
7. Gate 1 is recorded in the implementation PR: no objective changed; the
   recipe contains no conditional; and the real P5 review discrepancy already
   caught in PR #7 (component-scoped decay and per-component architecture plan
   values) is cited rather than inventing a synthetic discrepancy after the
   fact. Any failed condition blocks P8.
8. Tier 0 and Tier 1, `ruff check .`, `ruff format --check .`, `mypy --strict`
   and `git diff --check` are green. The card may then move to `smoke-passing`;
   the section 6 result remains for P12.

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
| Card reviewed (status -> `reviewed`) | | |
| Plan diffed against section 3.2 and section 4 | | |
