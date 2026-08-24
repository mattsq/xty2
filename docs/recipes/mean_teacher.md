# Recipe spec card: mean_teacher

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html) |
| Authors, year | Antti Tarvainen and Harri Valpola, 2017 |
| DOI / arXiv | [arXiv:1703.01780](https://arxiv.org/abs/1703.01780); [NeurIPS 2017 paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/68053af2923e00204c3ca7c6a3150cf7-Paper.pdf) |
| Version used | arXiv v6, 2018-04-16, together with the final NeurIPS 2017 proceedings version. Section 2 defines Mean Teacher; sections 3.3-3.4 and Appendix B govern the cited mechanics. |
| Reference implementation | [`CuriousAI/mean-teacher` @ `546348ff863c998c26be4339021425df973b4a36`](https://github.com/CuriousAI/mean-teacher/tree/546348ff863c998c26be4339021425df973b4a36), especially [`pytorch/main.py`](https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/main.py), [`losses.py`](https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/mean_teacher/losses.py), and [`ramps.py`](https://github.com/CuriousAI/mean-teacher/blob/546348ff863c998c26be4339021425df973b4a36/pytorch/mean_teacher/ramps.py). Project-local predecessor: [`mattsq/XTYLearner` `mean_teacher.py` @ `35734ec2d5a62d54a59eca38d1e31423da31e1ea`](https://github.com/mattsq/XTYLearner/blob/35734ec2d5a62d54a59eca38d1e31423da31e1ea/xtylearner/models/mean_teacher.py). |
| Reference impl. runnable? | Not attempted in the P9 card pass. Both repositories were inspected at the pinned commits; the official code targets an obsolete PyTorch/CUDA stack and the project-local source carries unrelated legacy dependencies. |

The authority order is deliberate. Tarvainen and Valpola define the method;
their pinned implementation resolves operational details that the paper leaves
implicit. The old XTYLearner module is evidence for the intended project-local
use -- semi-supervised treatment inference -- but it is not allowed to add
unreviewed mechanisms to Mean Teacher. In particular, its `q(t | x, y)` input,
OOD confidence weighting, continuous-treatment variance branch and dual-head
logit penalty are separate methods or later extensions, not part of P9.

## 2. Estimand and claim

- **Estimand:** The Mean Teacher part estimates the categorical propensity
  distribution `p(t | x)`. The surrounding causal stack also estimates the
  treatment-conditional outcome distribution `p(y | x, t=k)` and its means
  `mu_k(x)`. Under consistency, positivity and conditional exchangeability,
  contrasts of those means identify conditional treatment effects.
- **Claim:** Mean Teacher regularises a classifier by matching a student's
  prediction under one perturbation to an exponential-moving-average teacher's
  prediction under an independent perturbation. The paper reports better
  semi-supervised image-classification error than its Pi-model and Temporal
  Ensembling baselines, and shows that the teacher is normally a better final
  predictor than the unaveraged student. P9 claims only that this mechanism is
  faithfully assembled around `p(t | x)` and improves held-out perturbation
  consistency without materially damaging propensity or treatment-effect
  metrics on the fixed project-local target in section 6.
- **Not claimed:** The paper does not study tabular data, treatment assignment,
  missing treatment labels, outcome models or causal identification. Consistency
  does not repair confounding, non-overlap, MNAR labels or an augmentation that
  changes semantics. P9 does not claim the paper's SVHN/CIFAR/ImageNet numbers,
  does not use `y` to predict `t`, and does not provide out-of-fold protection;
  the posterior and leakage guard first have consumers in P10-P11.

## 3. Equations and mapping

### 3.1 As published

The paper's section 2 gives the following two unnumbered displayed equations.
With student weights `theta`, teacher weights `theta'`, independent noise
realisations `eta` and `eta'`, and classifier output `f`, its consistency cost
is

$$
J(\theta)
= \mathbb E_{x,\eta,\eta'}
  \left[
    \left\|f(x,\theta',\eta')-f(x,\theta,\eta)\right\|_2^2
  \right].
$$

After the student update at training step `s`, the teacher is updated by

$$
\theta'_s
= \alpha\theta'_{s-1} + (1-\alpha)\theta_s.
$$

Figure 2 and section 2 make the total training criterion a supervised
classification cost on labelled examples plus a weighted consistency cost on
all examples,

$$
\mathcal L_{\mathrm{MT}}(\theta)
= \mathcal L_{\mathrm{class}}(\theta)
  + \lambda_s J(\theta).
$$

The teacher is constant with respect to optimisation. The paper samples the
two noises independently at each step, normally uses MSE between softmax
probabilities, and ramps `lambda_s` early because teacher targets are initially
poor. The pinned implementation's ramp is

$$
r(s;S)
= \exp\!\left[-5\left(1-\min(s/S,1)\right)^2\right],
$$

which starts at `exp(-5)` and equals one from step `S` onward. It divides the
sum of squared probability differences by the number of classes, which is the
per-row class mean used by xty2's existing `ConsistencyLoss(divergence="mse")`.

### 3.2 Mapping to xty2

Let

```text
r_s = Realisation(view="student_x", params="student")
r_T = Realisation(view="teacher_x", params="teacher")
r_0 = Realisation(view="identity", params="student")
```

The two named views independently apply the same schema-aware feature mask.
For `K` treatments, P9 uses the lower end of the reference implementation's
recommended MSE range and sets the final consistency coefficient to `K`:

$$
\lambda_s
= K\,r(s;40).
$$

The single stage minimises

$$
\mathcal L_{\mathrm{P9}}
= \mathcal L_y^{r_0}
 + \mathcal L_t^{r_s}
 + \lambda_{\mathrm{marg},s}\mathcal L_{\mathrm{marg}}^{r_0}
 + \lambda_s\mathcal L_{\mathrm{cons}}^{r_s,r_T},
$$

where the first three terms are the reviewed P5 likelihood assembly,
`lambda_marg,s` is the reviewed linear ramp from `0` to `0.5` over 1,000
optimiser steps, and

$$
\mathcal L_{\mathrm{cons}}^{r_s,r_T}
= \frac{1}{B}\sum_{i=1}^B
  \frac{1}{K}\sum_{k=0}^{K-1}
  \left[
    p_{\theta}(t=k\mid \widetilde x_i^{(s)})
    -p_{\theta'}(t=k\mid \widetilde x_i^{(T)})
  \right]^2.
$$

The teacher probability is detached. `population` reduction makes every term
use `1/B`, including supervised terms whose ineligible rows contribute zero;
this matches the reference implementation's division by total minibatch size.
The supervised treatment loss consumes `r_s`, not `r_0`, so P9 is the second
consumer that justifies adding an explicit `realisation` field to
`ObservedTreatmentNLL`. Its default remains `r_0`, preserving P5 and P7.

| Paper / P9 symbol | Meaning | xty2 Port | xty2 Objective / Component / View |
|---|---|---|---|
| `x` | Raw covariates | `X_RAW` | virtual source node |
| `f(...; theta)` shared features | Student representation | `X_REPR` at `r_0` and `r_s` | existing `mlp_encoder` |
| `f(...; theta)` outcome branch | Project-local treatment-specific outcome distribution | `Y_GIVEN_XT` at `r_0` | existing `tarnet_head` |
| `f(...; theta)` class probabilities | Student propensity distribution | `T_GIVEN_X` at `r_0` and `r_s` | existing `categorical_propensity` |
| `f(...; theta')` | EMA-teacher propensity distribution | `T_GIVEN_X` at `r_T` | P8 teacher realisation of the same graph |
| `eta` | Student perturbation | view `student_x` | existing `FeatureMask(p=0.1, columns=None, value=0.0)` |
| `eta'` | Independent teacher perturbation | view `teacher_x` | a distinct identically configured `FeatureMask`; the view name gives it an independent RNG stream |
| `L_class` | Supervised treatment classification on the noisy student | `T_GIVEN_X` at `r_s` | `ObservedTreatmentNLL(realisation=r_s)` |
| Project-local `L_y` | Complete-case outcome NLL | `Y_GIVEN_XT` at `r_0` | existing `ObservedOutcomeNLL` |
| Project-local `L_marg` | Exact missing-treatment observed-data NLL | `Y_GIVEN_XT`, `T_GIVEN_X` at `r_0` | existing `MissingTreatmentMarginalNLL(grad_path="both")` |
| `J(theta)` | Student/teacher probability MSE on every row | `T_GIVEN_X` at `r_s`, `r_T` | existing `ConsistencyLoss(divergence="mse", stop_grad="right", rows="all")` |
| `lambda_s` | Gaussian/sigmoid consistency ramp | n/a | new `SigmoidRamp(end=K, steps=40)` schedule with the exact formula above |
| `alpha` | Teacher parameter EMA | every graph component's parameters | existing `TeacherSpec`; executor updates after the student optimiser step |

No posterior `T_GIVEN_XY`, pseudo-label artifact, confidence gate, OOD score,
continuous-treatment branch, dual classifier head, logit-distance objective,
outcome view or second stage belongs in P9.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
    joint_fit.consistency: right (teacher_x, teacher parameters)
  detached_targets: joint_fit.consistency detaches T_GIVEN_X at teacher_x/teacher  # paper section 2; ref impl losses.py and main.py
  gradient_clipping: none                     # project-local P5 choice; old XTYLearner mean_teacher applied none
  marginal_nll_grad_path: both                # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.99                             # paper section 3.4 early phase; official CIFAR ResNet ref; old XTYLearner default
  ema_applies_to_buffers: false               # pinned PyTorch update_ema_variables iterates parameters only
  teacher_in_train_mode: true                 # pinned PyTorch train() calls ema_model.train()
  teacher_requires_grad: false                # paper section 2 treats teacher as constant; ref impl detaches it

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
    joint_fit.consistency: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
    joint_fit.consistency: all                 # paper Figure 2: labelled and unlabelled examples
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
    joint_fit.consistency: K                   # lower end of official README's K..K^2 MSE guidance
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: linear ramp 0.0 -> 0.5 over 1000 optimiser steps
    joint_fit.consistency: K * exp(-5*(1-min(step/40,1))^2), 40 optimiser steps
  temperature: n/a                            # probability MSE; no sharpening
  sharpening: n/a
  confidence_threshold: n/a                   # canonical Mean Teacher uses all examples

optimisation:
  optimiser: adam(betas=(0.9, 0.999), eps=1e-8)  # retained P5 causal-stack setting
  lr: 0.001                                      # retained P5 causal-stack setting
  lr_schedule: staircase 1.0 * 0.97^floor(step/100)  # retained P5 causal-stack setting
  weight_decay: 0.0001 (components tarnet_head only; norm and bias exempt)  # retained P5 setting
  batch_size: n/a                                # external BatchSource; section 6 fixes 256 for validation
  labelled_unlabelled_ratio: n/a                 # no enforced batch quota; every row may enter consistency
  total_steps_or_epochs: 3000 optimiser steps    # retained P5 budget; ramp units are therefore unambiguous

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                 # retained reviewed P5 TARNet backbone
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: elu
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: row_l2
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0                             # perturbation comes from the two explicit input views
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: one strongly coupled K-logit head used by supervised and consistency objectives

data:
  standardisation: n/a                          # caller-owned; section 6 records the fixed choice
  outcome_scaling: n/a                          # caller-owned; section 6 records the fixed choice
  treatment_encoding: n/a                       # XTYBatch contract supplies integer classes 0..K-1; propensity emits K probabilities
  split_protocol: n/a                           # Tier 1 fixture and P12 runner own splits
  missingness_mechanism: n/a                    # section 6 fixes treatment MCAR; recipe consumes t_observed
```

The recipe declares two distinct `ViewSpec` instances. Each contains exactly
`FeatureMask(p=0.1, columns=None, value=0.0)` and preserves `t`, `y`,
`t_observed`, `y_observed`, `row_id`, `fold_id` and `weight`. It does not claim
to preserve `x`. `columns=None` means all mutable features; immutable columns
remain bit-identical and schema bounds are enforced. A schema with derived
features must provide the recompute rules required by `DESIGN.md` section 5 or
the view is rejected at compile time -- stale derived values are never accepted
as an implicit approximation.
`mean_teacher(schema, recompute_rules=(...))` supplies the same explicit rules
to both views; the empty tuple is valid only when masking cannot stale a derived
column.

The batching and preprocessing `n/a` entries are executable boundaries, not
missing research. P9 does not add a sampler merely to copy the image
reference's labelled/unlabelled minibatch ratio. The fixed smoke and validation
runners record their external batches, masks and transformations alongside the
result.

## 5. Deviations from the paper and project-local source

| # | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|
| 1 | Apply Mean Teacher to categorical treatment propensity `p(t | x)` and place it beside a causal outcome likelihood. | The paper studies image classes; P9 is the project-local causal composition named in `PLAN.md`. | No comparison to a published image error rate is valid. |
| 2 | Use only `x` for treatment prediction, not the old XTYLearner module's concatenated `(x, y)`. | `q(t | x, y)` is a posterior with an outcome-reachability/leakage consequence. P10-P11 own that port and its out-of-fold guard. | May be less predictive of missing treatment than the old module, but preserves the declared causal propensity estimand. |
| 3 | Add `ObservedOutcomeNLL` and `MissingTreatmentMarginalNLL`, and retain P5's marginal ramp. | P9 must compose Mean Teacher with the already reviewed causal stack and exercise multi-objective mixing plus two independent ramps. | Both terms train the shared encoder; their interaction with consistency is measured by the gradient trace and direction on metrics is not assumed. |
| 4 | Replace image translation/flips, Gaussian input noise and dropout with two independent 10% feature masks. | `FeatureMask` is the already validated schema-aware tabular perturbation; the paper says useful input augmentation is domain-specific. | Directly defines the invariance being measured; inappropriate masking can bias propensity and is guarded by NLL non-inferiority. |
| 5 | Use a strongly coupled single propensity head for both supervised and consistency losses. | The paper's primary formulation uses one output. Dual heads and a logit-distance loss are an optional ablation and would add a component/objective not needed to prove P9. | Avoids an extra degree of freedom; the paper reports that strong coupling performs well. |
| 6 | Use final consistency weight `K` and a 40-step sigmoid ramp, rather than benchmark-specific image weights and an 80-epoch paper ramp. | `K` is the lower end of the official implementation's MSE guidance; 40 steps is the pinned old XTYLearner default and is explicit in optimiser-step units. | Faster and smaller regularisation than the image runs; the schedule mutation and fixed target make this falsifiable. |
| 7 | Use constant EMA decay `0.99`, with no startup bias correction or later switch to `0.999`. | P8's reviewed teacher contract takes one decay. `0.99` is used by the official CIFAR ResNet run and old project source; a decay schedule would be a separate mechanism. | Shorter teacher memory late in training; may reduce final averaging benefit. |
| 8 | EMA only parameters while the teacher stays in train mode. | This is the pinned official PyTorch behaviour. The P9 MLP has no stateful normalisation or dropout, but the policy remains explicit and invariant-tested. | No numerical effect for the declared architecture; prevents a silent policy change if it later gains buffers. |
| 9 | Omit OOD weighting, confidence thresholds, continuous-treatment variance, calibrated Gaussian noise and outcome perturbation from old XTYLearner. | They are later project additions with different claims and require new objective/view semantics. The reviewed card boundary is canonical Mean Teacher on categorical `p(t | x)`. | Removes possible robustness gains and failure modes; none is part of the P9 target. |
| 10 | Reuse the P5 TARNet architecture, optimiser and 3,000-step budget rather than a ConvNet/ResNet image configuration. | Holding the causal stack fixed makes the P9 addition attributable and keeps the paired ablation meaningful. | The project-local result validates wiring, not paper-level image accuracy. |

## 6. Reproduction target

The published SVHN, CIFAR-10 and ImageNet results cannot honestly validate this
tabular causal adaptation: their inputs, labels, architectures and metrics are
all different. P12 must instead run the completely fixed project-local
mechanism target below. Passing it supports the limited P9 claim in section 2;
it must not be called a reproduction of Tarvainen and Valpola.

### 6.1 Fixed clustered DGP

Run ten paired replicates indexed by `r in {0, ..., 9}` with base seed
`s_r = 90000 + 100*r`. All generation uses CPU `torch.Generator` instances and
`float32`. Independent train, validation and test populations contain 4,096,
2,048 and 4,096 rows and use seeds `s_r+1`, `s_r+2` and `s_r+3`. Within each
population draw, in order, `U_C`, `epsilon_X`, `U_T` and `epsilon_Y`.
The two uniform tensors are independent draws with
`U_C, U_T ~ Uniform(0, 1)`. The six entries of `epsilon_X` and the scalar
`epsilon_Y` are independent `Normal(0, 1)` draws. All of these variables are
mutually independent within a row and independent across rows.

For each row, set `C = 1{U_C < 0.5}` and `S = 2C-1`. Define redundant cluster
features

$$
X_j = 0.8S + 0.6\epsilon_j,\quad j=1,\ldots,4,
\qquad X_5=\epsilon_5,\quad X_6=\epsilon_6.
$$

Treatment has overlap within both clusters,

$$
e(C)=0.15+0.70C,
\qquad T=\mathbb 1\{U_T<e(C)\},
$$

and the continuous outcome is

$$
b(X)=0.5X_1-0.3X_2+0.2(X_5^2-1),
\qquad \tau(X)=1+0.5\tanh(X_3),
$$

$$
Y=b(X)+T\tau(X)+0.5\epsilon_Y.
$$

Every outcome is observed. In the training population only, generate
`torch.randperm(4096)` with seed `s_r+4`; the first 205 indices have observed
treatment and the remaining 3,891 have missing treatment. Validation and test
treatments remain fully observed. This is an exactly fixed, approximately 5%
treatment-labelled MCAR design. Use raw `X`. Compute the population mean and
standard deviation of all training outcomes, including treatment-missing rows,
standardise train/validation/test `Y` with those moments for fitting, and return
predicted means to original outcome units for effect metrics. Supply no row
weights.

### 6.2 Fixed paired fit and evaluation

Compare the section 4 recipe with a zero-consistency ablation. The ablation is
identical -- graph, views, teacher, all four objectives, EMA, optimiser,
initialisation, batches and RNG keys -- except its consistency schedule is
`Constant(0.0)`. Keeping the zero-weight objective and teacher present isolates
the effect of the scheduled consistency gradient rather than conflating it with
EMA evaluation or extra forward passes.

Reset the global CPU Torch RNG to `s_r+6`, initialise one graph, and copy its
initial student and teacher state bit-for-bit to both fits. With seed `s_r+5`,
precompute 3,000 batches; batch `j` is the first 256 indices of a fresh
`torch.randperm(4096)`. Both fits consume the identical ordered index table and
the same per-step `rng_key = s_r + 10000 + j`, so corresponding student and
teacher views are bit-identical between fits but independent within a fit. Run
on CPU with deterministic algorithms enabled for exactly 3,000 optimiser steps
and evaluate final checkpoints. Validation is diagnostic only and cannot select
a checkpoint or alter this protocol.

On the test population, generate sixteen paired view realisations with keys
`s_r+20000+q`, `q in {0, ..., 15}`. For fit `M`, define held-out perturbation
disagreement

$$
C_M^{(r)}
= \frac{1}{16\cdot4096\cdot K}
  \sum_{q,i,k}
  \left[
    p_{M,\mathrm{student}}(k\mid v_s^{(q)}(X_i))
    -p_{M,\mathrm{teacher}}(k\mid v_T^{(q)}(X_i))
  \right]^2.
$$

The primary paired metric is `R_C^(r) = C_MT^(r) / C_zero^(r)`. Guardrails are
the identity-view final-teacher treatment NLL and test `sqrt(PEHE)` against the
analytic `tau(X)`. Define paired differences as Mean Teacher minus ablation.
The target passes only when

```text
mean(R_C) <= 0.90
mean(d_treatment_NLL) <= 0.02 nat/row
mean(d_sqrt_PEHE) <= 0.10 outcome units.
```

Report each fit's mean, every paired metric, and sample standard errors across
the ten replicates. If `C_zero` is exactly zero in any replicate, the target is
invalid rather than automatically passing.

```yaml
reproduction:
  dataset: xty2 analytic redundant-cluster treatment DGP
  variant: section 6.1; six continuous X; binary overlapping T; continuous heterogeneous-effect Y
  split: independent 4096 train / 2048 validation / 4096 test; exactly 205 train treatments observed
  metric: held-out masked-view student/teacher probability-MSE ratio; treatment NLL and sqrt_PEHE guardrails
  published: n/a
  published_source: n/a                     # project-local mechanism target, not an image-classification reproduction
  tolerance: mean ratio <= 0.90; mean d_NLL <= 0.02 nat/row; mean d_sqrt_PEHE <= 0.10
  seeds: r=0..9 with base 90000+100*r and fixed stream offsets in sections 6.1-6.2
  report: per-fit means plus paired means and sample stderrs over 10 replicates
```

### 6.3 Result ledger

| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-24 | `d060df351f2fe8bac6d951c3757506c684d8b408` | consistency_MSE_ratio<br>paired_d_treatment_NLL<br>paired_d_sqrt_PEHE | 0.316756 +/- 0.0303<br>-0.00555395 +/- 0.00797 nat/row<br>0.054138 +/- 0.00626 outcome units | yes |

### 6.4 P9 acceptance

1. The graph contains exactly the reviewed P5 `mlp_encoder`, `tarnet_head` and
   `categorical_propensity`. The recipe contains declarations only. It declares
   exactly `student_x` and `teacher_x`, with the transforms and preservation
   sets in section 4; a fixed key yields reproducible views and the two names
   yield independent RNG streams.
2. The compiled stage plans exactly three realisations: identity/student for
   outcome and marginal likelihood, `student_x`/student for supervised
   treatment plus consistency, and `teacher_x`/teacher for consistency. No
   objective re-invokes the graph and no fourth forward pass appears.
3. `joint_fit` contains exactly the four objectives, row scopes, reductions and
   schedules in section 4. `ObservedTreatmentNLL(realisation=...)` is the only
   supervised-objective surface added; its default-realisation behaviour and
   P5/P7 plans remain byte-stable.
4. The teacher begins as an exact student copy, has `requires_grad=False`,
   receives no gradient, uses the explicit parameter/buffer/mode policy, and is
   updated only after each student optimiser step. Candidate and identity
   evaluation work for both parameter sets.
5. A named Tier 1 mutation test replaces the reviewed 40-step consistency
   schedule with `SigmoidRamp(end=K, steps=400)`. The smoke trace asserts the
   realised consistency weights at steps `0`, `20` and `40` are respectively
   `K*exp(-5)`, `K*exp(-1.25)` and `K`, as well as asserting the plan's
   40-step description. The real recipe passes and the 400-step mutant fails
   that assertion. This is the required proof that the ramp length came from
   this card rather than a default; a loss-decrease assertion alone is not a
   substitute.
6. One non-CI P9 smoke run enables `GradientProbe.periodic(200)`. The
   implementation PR body records the resulting per-objective gradient norms
   and pairwise cosines at every probed step, including the treatment versus
   consistency and marginal versus consistency pairs. The probe remains off by
   default in the recipe and normal CI.
7. The Tier 1 analytic fixture shows total loss decreases, propensity beats the
   held-out marginal-frequency baseline, treatment contrasts remain in the
   declared wide analytic band, and the scheduled fit has lower held-out
   perturbation disagreement than the zero-consistency paired ablation. These
   are wiring assertions, not the ten-seed section 6 target.
8. The card-to-plan cross-check covers every non-`n/a` section 4 key, both view
   descriptions, every realisation, teacher policy, component-scoped
   architecture, and exact schedule formula/length. Mutations removing either
   view, detaching the student, EMA-updating buffers or using the identity view
   on both sides fail named invariants.
9. Tier 0 and Tier 1, `ruff check .`, `ruff format --check .`, `mypy --strict`
   and `git diff --check` are green. The card may then move to
   `smoke-passing`; section 6 remains queued for P12.

## 7. Unknowns

| Unspecified in source | Our choice | Basis |
|---|---|---|
| The paper has no causal or tabular recipe. | Mean Teacher regularises `p(t | x)` inside the reviewed P5 likelihood graph; it never consumes `y` for treatment prediction. | Smallest P9 composition that exercises the required framework surfaces without pre-empting P10-P11. |
| The paper says augmentation is domain-specific. | Independent 10% zero feature masks over every mutable column. | Existing schema-aware view plus the old XTYLearner primer's tabular feature-dropout starting point; fixed before observing P9 results. |
| Published consistency weights and ramp lengths are dataset-specific and expressed in epochs. | Final weight `K`; exact sigmoid/Gaussian ramp over 40 optimiser steps. | Official README's `K..K^2` MSE guidance plus old XTYLearner's pinned 40-step default; optimiser-step units are executable. |
| The paper changes EMA decay from `0.99` to `0.999` after ramp-up in its ConvNet evaluation, while official runs vary by dataset. | Constant `0.99`, no startup correction. | Supported official/old-project value and the reviewed P8 constant-decay surface; an EMA schedule waits for a second consumer. |
| Batch composition strongly affects Mean Teacher. | No recipe-enforced quota; fixed validation uses uniformly sampled batches from a dataset with exactly 205 labelled treatments. | Current `BatchSource` boundary. A sampler is not added until a card requires a per-batch invariant; zero-eligible rows are already safe. |
| The old XTYLearner module uses separate classification and consistency heads. | One propensity head. | Paper's primary strongly coupled formulation and section 3.4 ablation; avoids an otherwise unconsumed head-distance objective. |
| Teacher buffer behaviour is not stated in the paper. | Parameters-only EMA, teacher train mode. | Pinned official PyTorch implementation. The declared P9 network has no stateful buffers, so invariants protect the policy even though this card's numbers do not distinguish it. |
| The paper evaluates EMA predictions but does not define a causal outcome-evaluation policy. | Report the full EMA teacher graph for section 6 propensity and outcome guardrails; retain both teacher and student artifacts. | Makes the evaluated model match the teacher claim and uses P8's whole-graph teacher contract without adding promotion semantics. |
| No published target applies to the tabular causal adaptation. | Predeclare a seed-locked clustered DGP, a zero-consistency paired ablation, a direct consistency metric and predictive/effect guardrails. | Tests the limited P9 mechanism claim without borrowing an unrelated image number or tuning the target after a run. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status -> `reviewed`) | mattsq | 2026-08-24 |
| Plan diffed against section 3.2 and section 4 | Codex | 2026-08-24 |
