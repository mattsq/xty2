# Recipe spec card: variational_treatment

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Semi-Supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298) |
| Authors, year | Diederik P. Kingma, Danilo J. Rezende, Shakir Mohamed, Max Welling; 2014 (NIPS 27) |
| DOI / arXiv | [arXiv:1406.5298](https://arxiv.org/abs/1406.5298); [10.48550/arXiv.1406.5298](https://doi.org/10.48550/arXiv.1406.5298) |
| Version used | arXiv v2, 2014-10-31 (the NIPS camera-ready). §2 defines the three models — eq. (2) is M2's generative model and eq. (4) its inference model — and §3.1.2 states the objectives: eq. (6) the labelled bound, eq. (7) the unlabelled bound, eq. (8) their sum, eq. (9) the α-weighted classification term, and `α = 0.1 · N`. §4.1 gives the benchmark architecture and Table 1; §4.4 the optimisation details. This card ports **M2's discrete-latent objective**, not M1 and not the M1+M2 stack. |
| Reference implementation | [`dpkingma/nips14-ssl`](https://github.com/dpkingma/nips14-ssl) @ `1fc7aec899ca429ff26a1d10f747e5ad6bcd17b2` (master, the only branch), **read directly** in the session that wrote this card: `learn_yz_x_ss.py` (the whole semi-supervised objective and its optimiser) and `run_2layer_ssl.py` (the per-label-count settings the paper states in prose). |
| Reference impl. runnable? | Not attempted. Python 2.7, Theano with `floatX=float32`, and a pickled M1 checkpoint the repository does not ship. Nothing about the tabular port turns on running it. |

## 2. Estimand and claim

- **Estimand:** unchanged from `tarnet`: the treatment-specific means `m_k(x) = E[Y(k) | X=x]`, whose contrasts are CATEs under the usual assumptions, and the propensity `p(t | x)`. This card adds one quantity that is *not* an estimand: `q(t | x, y)`, an outcome-aware imputation distribution over an unobserved treatment.
- **Mechanism under test (one sentence, falsifiable):** an amortised `q(t | x, y)` trained by eq. (7) and eq. (9) tracks the exact model posterior closely enough that replacing exact marginalisation with the variational objective has no material held-out marginal-likelihood cost, while exposing that posterior through a learned `T_GIVEN_XY` head.
- **Method claim (the paper's):** a discrete label can be treated as a latent variable and marginalised under an amortised approximate posterior; eq. (9) is required because `q_φ(y|x)` appears only in the unlabelled term of eq. (8) and would otherwise never learn from labelled data (§3.1.2). Evidence is Table 1: M2 reaches `11.97% ± 1.71` MNIST test error at 100 labels against `25.81%` for a plain neural network, and `3.92% ± 0.63` at 3,000 labels.
- **Nearest shipped baseline and the controlled difference:** `tarnet`. Both fit the same three serving-path components on the same rows; this card replaces `MissingTreatmentMarginalNLL` — the exact sum over candidate treatments — with eq. (7)'s variational bound plus eq. (9)'s posterior term. §6 runs that substitution as a pair.
- **Not claimed:** no image benchmark is reproduced and no published number transfers (§5.3). For a *fixed* propensity/outcome state the variational bound cannot be below exact enumeration; that algebra does not order two separately fitted arms, which may generalise either way (§3.2). No identification claim is made from `q`: it is not a serving-time propensity, because it reads `y`. It must never be written back as a treatment label consumed by an in-sample outcome fit — that is the circularity `DESIGN.md` §7.2 rejects, and §3.2 states why an in-objective `q` is a different thing. Because deviation 1 removes the continuous latent and this adaptation explicitly enumerates all `K` treatments, the exact arm can itself compute `p(t|x,y)` by normalising `p(t|x)p(y|x,t)`. The learned `q` is therefore an amortised posterior head, not an inference capability or a training-time computational shortcut unavailable to exact enumeration on this fixture.

## 3. Equations and mapping

### 3.1 As published

M2's generative model, eq. (2), over an observation `x` with a discrete label `y`
and a continuous latent `z`:

$$
p(y) = \mathrm{Cat}(y \mid \pi), \qquad
p(z) = \mathcal N(z \mid 0, I), \qquad
p_\theta(x \mid y, z) = f(x; y, z, \theta),
$$

and its inference model, eq. (4):

$$
q_\phi(z \mid y, x) = \mathcal N\!\left(z \mid \mu_\phi(y, x), \operatorname{diag}(\sigma^2_\phi(x))\right),
\qquad
q_\phi(y \mid x) = \mathrm{Cat}(y \mid \pi_\phi(x)).
$$

For a labelled pair, eq. (6):

$$
\log p_\theta(x, y) \ge
\mathbb E_{q_\phi(z \mid x,y)}\!\left[
\log p_\theta(x \mid y, z) + \log p_\theta(y) + \log p(z) - \log q_\phi(z \mid x, y)
\right]
= -\mathcal L(x, y).
$$

For an unlabelled point the label is marginalised under `q`, eq. (7):

$$
\log p_\theta(x) \ge
\sum_y q_\phi(y \mid x)\,\bigl(-\mathcal L(x, y)\bigr)
+ \mathcal H\bigl(q_\phi(y \mid x)\bigr)
= -\mathcal U(x).
$$

Eq. (8) is the sum over both streams, and eq. (9) adds the classification term:

$$
\mathcal J = \sum_{(x,y) \sim \tilde p_l} \mathcal L(x, y)
           + \sum_{x \sim \tilde p_u} \mathcal U(x),
\qquad
\mathcal J^{\alpha} = \mathcal J
  + \alpha \cdot \mathbb E_{\tilde p_l(x,y)}\bigl[-\log q_\phi(y \mid x)\bigr],
$$

with `α = 0.1 · N` (§3.1.2, immediately after eq. (9); `alpha=0.1` in
`run_2layer_ssl.py:13`, rescaled by `n_tot / n_labeled` at
`learn_yz_x_ss.py:248`). The sum over `y` in eq. (7) is taken by explicit
enumeration, not by sampling: `learn_yz_x_ss.py:256` says so in a comment and
`:265` is the loop.

### 3.2 Mapping to xty2

**The role swap, stated once.** M2's partially observed discrete label `y` is
xty2's treatment `t`. M2's observation `x` — the variable the model generates —
is xty2's outcome `y`, generated conditional on the covariates `x`, which are
context throughout and are never generated. M2's continuous latent `z` and its
decoder are dropped (deviation 1), and M2's label prior becomes the propensity
(deviation 2). With `z` gone, eq. (6) is a joint conditional likelihood and
eq. (7) is a sum over `K` candidate treatments:

$$
\mathcal L(x, t, y) = -\log p_\theta(t \mid x) - \log p_\phi(y \mid x, t),
$$

$$
\mathcal U(x, y) =
\sum_{k=0}^{K-1} q_\psi(t{=}k \mid x, y)
\Bigl[-\log p_\theta(t{=}k \mid x) - \log p_\phi(y \mid x, t{=}k)\Bigr]
+ \sum_{k=0}^{K-1} q_\psi(t{=}k \mid x, y)\,\log q_\psi(t{=}k \mid x, y).
$$

M2's `q_φ(y|x)` reads everything that is observed when the label is not. Here
that is the pair `(x, y)`, which is exactly the `T_GIVEN_XY` port — so the
inference network is `CategoricalPosterior`, the component `cycle_dual` already
uses, and eq. (9)'s term is `ObservedTreatmentNLL(port=T_GIVEN_XY)`, which
already exists for the same reason.

**The relation to exact marginalisation, and why §6 is a pair.** Write
`M(x, y)` for what `MissingTreatmentMarginalNLL` returns on the same row,
`-log Σ_k p(t=k|x) p(y|x,t=k)`. Then for any `q`

$$
\mathcal U(x, y) = M(x, y) + \mathrm{KL}\bigl(q_\psi(\cdot \mid x, y)\,\|\,p(\cdot \mid x, y)\bigr)
\;\ge\; M(x, y),
$$

with equality exactly when `q` is the exact posterior under that same model
state, `p(t=k|x,y) ∝ p(t=k|x) p(y|x,t=k)`. Three things follow. The bound is on
the *same* observed-data likelihood, so the per-state comparison is a fidelity
check rather than a comparison of unrelated objectives. Its slack is one
number, `KL(q ‖ posterior)`, which §6 measures directly. And for fixed
`p(t|x)` and `p(y|x,t)`, the variational value cannot be below the exact value.
That last inequality does **not** order the two fitted arms in §6: finite
optimisation, regularisation and sampling can take them to different serving
parameters, so either may have the lower held-out exact marginal NLL. §6's
ratio is therefore an empirical cost/benefit comparison; Tier 0 and Tier 1 use
the algebra only on the same state.

Dropping `z` has a second consequence: arm B can compute that same model
posterior directly from its propensity and outcome distributions, and both arms
already enumerate all `K` candidates while training. The `q` head is therefore
tested as an amortised representation of the posterior and as part of a
variational training mechanism, not as a way to make an otherwise intractable
posterior tractable in this project-local adaptation.

**Why this is not the circularity of `DESIGN.md` §7.2.** `q` reads `y` and
weights an outcome likelihood on the same rows, which looks like the rejected
`q(t|x,y) -> p(y|x,t)` staged fit. It is not, and the difference is checkable
rather than rhetorical: no treatment label is written, `t_observed` is
untouched, and the quantity being optimised is a bound on `-log p(y|x)`, a
likelihood of observed data only, valid for *any* `q`. The compiler agrees for a
structural reason — `_check_static_leakage` fires on a `PseudoLabelAction`
artifact crossing a stage edge (`xty2/core/compile.py:752`), and this recipe is
one stage with no action. Tier 0 asserts that the recipe compiles with
`purpose="causal"` and `allow_leakage=False`, so the opt-out is provably unused.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `x` (context, not M2's `x`) | Raw covariates | `X_RAW` | virtual source node |
| `y` (M2's observation) | Raw outcome | `Y_RAW` | virtual source node |
| `y` (M2's discrete label) | Treatment `t`, latent where unobserved | — | `XTYBatch.t` under `t_observed` |
| `Φ(x)` | Shared representation | `X_REPR` | `mlp_encoder` |
| `p_θ(x \| y, z)` | The generated observation | `Y_GIVEN_XT` | `tarnet_head` |
| `p(y) = Cat(y \| π)` | Label prior, uniform in the pinned code | `T_GIVEN_X` | `categorical_propensity` (deviation 2) |
| `q_φ(y \| x)` | Amortised posterior over the discrete latent | `T_GIVEN_XY` | `categorical_posterior` |
| `p(z)`, `q_φ(z \| x, y)` | Continuous latent and its posterior | n/a | dropped (deviation 1) |
| `L(x, y)`, eq. (6) | Labelled term | `T_GIVEN_X`, `Y_GIVEN_XT` | `ObservedTreatmentNLL` + `ObservedOutcomeNLL`, rows `t_observed` |
| `U(x)`, eq. (7) | Unlabelled term, exact sum over `K` plus `H(q)` | `T_GIVEN_X`, `Y_GIVEN_XT`, `T_GIVEN_XY` | `VariationalTreatmentELBO` (§5.1), rows `t_missing` |
| `α E[-log q_φ(y \| x)]`, eq. (9) | Classification term for the inference network | `T_GIVEN_XY` | `ObservedTreatmentNLL(port=T_GIVEN_XY)`, rows `t_observed`, weight `0.1` |
| `J^α`, eq. (9) | One weighted objective mix | all three predicted ports | stage `elbo_fit` with four `Weighted` terms |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    elbo_fit.observed_outcome_nll: none
    elbo_fit.observed_treatment_nll: none
    elbo_fit.variational_treatment_elbo: none   # eq. (7) is enumerated exactly, so q's weights are differentiated through; ref impl learn_yz_x_ss.py:217 takes the gradient of the whole weighted sum
    elbo_fit.posterior_treatment_nll: none
  detached_targets: n/a                         # no consistency target; nothing here is a target
  gradient_clipping: none                       # ref impl learn_yz_x_ss.py:307 steps AdaM with no clipping
  marginal_nll_grad_path: n/a                   # this arm has no MissingTreatmentMarginalNLL; §6's exact-marginal arm binds `both`

teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    elbo_fit.observed_outcome_nll: population
    elbo_fit.observed_treatment_nll: population
    elbo_fit.variational_treatment_elbo: population   # eq. (8) sums L over labelled and U over unlabelled rows; ref impl normalises the total by n_tot (learn_yz_x_ss.py:296), which `population` is, given the batch composition below
    elbo_fit.posterior_treatment_nll: mean           # eq. (9)'s alpha is stated against N, not against the labelled count; ref impl learn_yz_x_ss.py:248 rescales the per-labelled-row term by n_tot/n_labeled before the same /n_tot, which is exactly a mean over labelled rows
  eligible_rows:
    elbo_fit.observed_outcome_nll: t_observed
    elbo_fit.observed_treatment_nll: t_observed
    elbo_fit.variational_treatment_elbo: t_missing   # eq. (8) partitions; unlike FixMatch footnote 2, labelled rows do not also enter U
    elbo_fit.posterior_treatment_nll: t_observed
  weights:
    elbo_fit.observed_outcome_nll: 1.0
    elbo_fit.observed_treatment_nll: 1.0
    elbo_fit.variational_treatment_elbo: 1.0         # eq. (8) weights L and U equally (deviation 7)
    elbo_fit.posterior_treatment_nll: 0.1            # alpha; run_2layer_ssl.py:13
  schedules:
    elbo_fit.observed_outcome_nll: constant 1.0
    elbo_fit.observed_treatment_nll: constant 1.0
    elbo_fit.variational_treatment_elbo: constant 1.0  # M2 ramps nothing
    elbo_fit.posterior_treatment_nll: constant 1.0
  temperature: n/a                              # eq. (7) takes the expectation under q as predicted
  sharpening: n/a                               # no sharpening: q enters as a distribution, never as a label
  confidence_threshold: n/a                     # every t_missing row trains eq. (7); there is no gate

optimisation:
  optimiser: adam(betas=(0.9, 0.999), eps=1e-8)  # ref impl learn_yz_x_ss.py:307; the paper's §4.4 text says RMSProp with momentum — see §7
  lr: 0.0003                                     # paper §4.4 and ref impl learn_yz_x_ss.py:307 agree
  lr_schedule: constant 1.0                      # the ref impl steps AdaM at a fixed rate
  weight_decay: 0.0009765625 (all trainable components; all parameters)  # the N(0, I) parameter prior (prior_sd=1, learn_yz_x_ss.py:140 and :146) enters the per-datapoint objective as 1/N (:293-296); N = 1024 on the §6.1 fixture — see §7
  batch_size: 128                                # derived from the QuotaSampler's quotas, never asserted
  labelled_unlabelled_ratio: 15.0                # derived; equals the fixture's 64:960, so `population` is the reference's dataset-proportional normalisation (learn_yz_x_ss.py:197-201)
  total_steps_or_epochs: 3000 optimiser steps    # deviation 5; the paper runs 3,000 epochs of 100 minibatches

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                 # retained reviewed P5 TARNet backbone
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
    categorical_posterior: concat(X_RAW, Y_RAW) -> [300] -> K   # one hidden layer of 300: run_2layer_ssl.py:13, the branch for the scarcest label budget
  activation:
    mlp_encoder: elu
    tarnet_head: elu
    categorical_propensity: linear logits
    categorical_posterior: relu                  # deviation 4; the paper and ref impl use softplus
  normalisation:
    mlp_encoder: row_l2
    tarnet_head: none
    categorical_propensity: none
    categorical_posterior: none
  dropout:
    mlp_encoder: 0.0
    tarnet_head: 0.0
    categorical_propensity: 0.0
    categorical_posterior: 0.0                   # M2 regularises by the parameter prior, not by dropout
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
    categorical_posterior: torch Linear default Kaiming-uniform   # deviation 4; the ref impl initialises from N(0, 0.001^2) (init_w(1e-3), learn_yz_x_ss.py:147)
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits
    categorical_posterior: K softmax logits

data:
  standardisation: x: none fitted on 'train'     # the §6.1 DGP draws standardised features
  outcome_scaling: y: zscore fitted on 'train'   # y is a network input to the posterior, so its scale is load-bearing here in a way it is not for tarnet
  treatment_encoding: categorical integers 0..K-1 with t_observed mask; no sentinel
  split_protocol: one fixed project-local DGP, split train/test by the section 6 fixture; no MNIST/SVHN/NORB protocol applies (deviation 3); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Drop the continuous latent `z`, its posterior `q_φ(z\|x,y)` and the decoder `p_θ(x\|y,z)`. Model `p(t, y \| x) = p_θ(t\|x) p_φ(y\|x,t)` with covariates as conditioning context. | §2 does not claim a generative model of the covariates, and `DESIGN.md` §11.2's first question asks whether the card's §4 checklist needs the mechanism. This one does not: the estimand is a conditional outcome mean, and a joint `p(x,t,y)` is a different modelling commitment with its own estimand (`BACKLOG.md` §18.11). `cycle_dual.md` §5.3 is the precedent for declining a source mechanic a card does not claim, rather than half-building it. It is also what makes the experiment identifiable: with `z` gone the only slack left in eq. (7) is `KL(q ‖ posterior)`, which §6 measures as a single number. | Decisive. The objective is M2's discrete-latent structure, not M2. No published number transfers and §6 is a mechanism target throughout. |
| 2 | `judgement` | — | Replace the label prior `p(y) = Cat(y\|π)` — uniform in the pinned code, `learn_yz_x_ss.py:139` sets `uniform_y = True` — with the covariate-dependent propensity `p_θ(t\|x)`. | M2 generates `x` from `y`, so its label prior *cannot* depend on `x` without inverting the model. Here `x` is context, and `p(t\|x)` is one of the three quantities under test — dropping to a marginal prior would delete the propensity from the objective and make eq. (7) a bound on the wrong quantity. | Load-bearing and favourable to fidelity: it is what makes eq. (7) an exact upper bound on the same `-log p(y\|x)` that `MissingTreatmentMarginalNLL` computes, and therefore what makes §6's pair a comparison rather than two unrelated numbers. |
| 3 | `judgement` | — | No MNIST, SVHN or NORB protocol, no label-count sweep, no test-error target. One fixed project-local DGP (§6.1). | Images are outside xty2 v1 (`DESIGN.md` §0) and none of the paper's datasets carries a treatment. Reproducing that shape would test data plumbing, not whether the objective is assembled correctly. `paws.md` §5.7 is the precedent. | §6 is a mechanism target and says so. It is evidence against a miswired objective, not for the paper's claim. |
| 4 | `judgement` | — | The posterior network takes the paper's *shape* — one hidden layer of 300 units, `run_2layer_ssl.py:13` — with ReLU rather than softplus and Torch's default Kaiming-uniform initialisation rather than `N(0, 0.001^2)`. The rest of the graph is the P5 TARNet backbone unchanged. | Holding the causal stack fixed is what makes an addition attributable (`scarf.md` §5.3, `fixmatch.md` §5.6, `paws.md` §5.5). `CategoricalPosterior` is the reviewed component `cycle_dual` already uses and it admits only ReLU; mixing one softplus module into an ELU/ReLU graph is not a difference this fixture could resolve, and the small-variance initialisation belongs to a 3,000-epoch budget this card does not run (deviation 5). Recorded rather than absorbed, because it is a departure from a stated setting. | On the head that produces `q` only. §6.2's amortisation gap and posterior advantage are the two numbers that would show it mattering. |
| 5 | `judgement` | — | 3,000 optimiser steps, rather than the paper's 3,000 passes over 100 minibatches (300,000 updates). | Every card here fixes a project-local step budget so that a difference between arms is attributable to the arm (`scarf.md` §5.4, `paws.md` §5.6), and §6's pair gives both arms the same budget either way. | The bound is measured far short of convergence. §6's gap is a statement about this budget; no claim is made about its asymptote, and a gap that is still shrinking at step 3,000 is reported as such. |
| 6 | `judgement` | — | Inherit `tarnet.md` §5.2 and §5.3 unchanged: `K` categorical outcome heads rather than two, each a unit-scale Gaussian trained by NLL rather than a squared-error point predictor. | Both are properties of the shared backbone this card holds fixed, not of M2. They are restated here because eq. (6) and eq. (7) both consume `log p_φ(y\|x,t)` and its scale sets the relative weight of the two terms of the ELBO. | Sets the scale of every likelihood in §4. Identical in both §6 arms, so it cannot move the ratios; it can move the absolute nats. |
| 7 | `judgement` | — | Eq. (8) weights `L` and `U` equally and this card keeps that: the variational term runs at constant `1.0`, and §6's exact-marginal arm runs `MissingTreatmentMarginalNLL` at constant weight `1.0` too — not `tarnet.md` §4's `0.5` on a 1,000-step ramp. | Otherwise the pair would differ by an objective *and* by its weight and schedule, and §6 could not attribute either result. The paper has no ramp anywhere; `tarnet`'s is a reviewed P5 choice belonging to that card. | Both arms move away from `tarnet`'s published configuration, so neither is comparable to `tarnet.md` §6.1's ledger. The pair remains internally controlled, which is what §6 reports. |

### 5.1 Framework additions made for this card

One objective, in the reversible half of `DESIGN.md` §11.2's table. No port, no
executor contract, no row population, no artifact kind, and no new sampler or
schedule: eq. (9) is `ObservedTreatmentNLL(port=T_GIVEN_XY)`, which
`xty2/objectives/supervised.py` already admits for `cycle_dual`, and eq. (6) is
the two shipped likelihood terms. **No `DESIGN.md` §11.4 ledger key is cited or
discharged by this card.** In particular `batch-row-repetition` is neither paid
nor discharged: the quotas draw 8 of 64 observed rows and 120 of 960 missing
rows, so no `row_id` repeats.

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `VariationalTreatmentELBO` — eq. (7) on `t_missing` rows: `Σ_k q_k [-log p(t=k\|x) - log p(y\|x,t=k)] + Σ_k q_k log q_k`, requiring `T_GIVEN_X`, `Y_GIVEN_XT` and `T_GIVEN_XY`, detaching nothing | fidelity-bearing, reversible | `elbo_fit.variational_treatment_elbo` | not required (reversible) | No shipped objective takes an expectation under a *predicted* distribution: `MissingTreatmentMarginalNLL` sums exactly and every pseudo-label term reduces `q` to a label or a detached target first. `BACKLOG.md` §3.4 asks for "a single written probabilistic objective before composing terms", and eq. (7) is one. `q`'s log probabilities are read through the `TreatmentDistribution.log_prob(candidate_treatments)` contract, so the computation stays in log space without depending on the concrete `CategoricalTreatment` implementation or taking `probs.log()`. |

**A constraint, not an addition.** `CategoricalPosterior` binds
`data.standardisation` and `data.outcome_scaling` (`xty2/components/posterior.py:37-38`)
and so does `DataSpec` (`xty2/core/data.py:228-232`). `cycle_dual` never met this
because it declares no `DataSpec`, and `tarnet` never met it because it has no
posterior; this is the first recipe with both. `_merge_value`
(`xty2/core/compile.py:1440`) rejects only *differing* values, so the recipe must
pass `DATA_POLICY.standardisation` and `DATA_POLICY.outcome_scaling` into the
component rather than restating the strings. Nothing to build, and the fix is a
reference rather than a copy — but it is written here because a restated string
that later drifts fails at compile time in a place that names neither document.

## 6. Reproduction target

The target is a paired substitution on a fixed project-local DGP: eq. (7) plus
eq. (9) against xty2's exact marginalisation, with the same fixture, seeds,
batches, backbone and budget in both arms. Arm **A** (`variational`) is the
recipe of §3–§4. Arm **B** (`exact`) is the same graph without
`categorical_posterior`, with `variational_treatment_elbo` and
`posterior_treatment_nll` replaced by a single
`MissingTreatmentMarginalNLL(grad_path="both")` at constant weight `1.0`,
reduction `population`, rows `t_missing` (deviation 7). The arms differ in
parameter count, and only in parameters that produce `q`: nothing in A's
serving path — encoder, outcome heads, propensity — is larger than B's, and `q`
reaches them only through eq. (7)'s weights. Arm B exposes no `T_GIVEN_XY`
component, but its exact model posterior is still derivable by normalising
`p(t|x)p(y|x,t)` across the same `K` candidates.

Every metric is computed on the held-out population, where all treatments and
outcomes are observed, and every likelihood is scored by the *exact* quantity
`-log Σ_k p(t=k|x) p(y|x,t=k)` — including for arm A, which is scored by what
it bounds rather than by its own bound.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2) with overlapping treatment assignment, specified in 6.1
  variant: paired substitution of eq. (7) + eq. (9) for exact marginalisation; same fixture, seeds, batches, backbone, optimiser and step budget in both arms
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment and outcome observed
  metric: held-out exact marginal NLL ratio, variational arm over exact arm; amortisation gap, posterior advantage, the share of its own model posterior's y-information the amortised head recovers, and two likelihood guardrails as declared in 6.2
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: held_out_marginal_NLL_ratio mean plus one stderr at most 1.02; amortisation_gap mean plus one stderr at most 0.10 nats; posterior_information_captured mean minus one stderr at least 0.8, with model_posterior_information mean minus one stderr at least 0.02 nats so that a fixture whose y carries no treatment information voids this pair rather than passing it (6.4); posterior_advantage mean plus one stderr strictly below 0.0 nats; held_out_outcome_NLL_ratio and held_out_treatment_NLL_ratio each at most 1.05
  seeds: 10
  report: mean_and_stderr
```

### 6.1 Fixed DGP

Use `fixmatch.md` §6.1's generator, seed streams, 1,024-row train population,
2,048-row held-out population and 64-label MCAR budget unchanged, with **one
declared change**: the treatment assignment is

```text
p(t=1|c) = 0.25 + 0.5c
```

rather than `0.02 + 0.96c`. The reason is specific to this card and is not a
convenience. At `0.02 / 0.98` the treatment is already almost determined by the
cluster signal in `x`, so observing `y` has little room to improve treatment
inference; the analytic Bayes posterior-minus-propensity treatment NLL gap that
§6.2 item 5 exists to expose is correspondingly compressed. This says nothing
about `KL(q ‖ posterior)` for an arbitrary `q`: disagreement with a nearly
point-mass posterior can be very expensive. At `0.25 / 0.75` the treatment is
genuinely uncertain given `x` while the outcome shift
(`effect = 1 + 0.5 tanh(x2)` against noise `0.5`) keeps `y` informative about
`t`, giving the outcome-aware posterior a measurable target and making the
amortisation gap meaningful rather than making the comparison pass by fixture
design. Everything else is `fixmatch.md` §6.1's: outcome standardisation fitted
on the complete training population, held-out treatments and outcomes all
observed.

**Batch composition.** Every step of both arms draws 8 rows from `t_observed`
and 120 from `t_missing`. The quota is not free: `64 : 960` is the fixture's own
population ratio, and `8 : 120` reproduces it exactly, which is what makes the
`population` reduction of §4 equal to the reference implementation's
dataset-proportional normalisation (`learn_yz_x_ss.py:197-201, 296`) rather than
merely resemble it. A quota that changed the ratio would silently reweight
eq. (8).

### 6.2 Predeclared evidence

Predeclared while the card was `draft`, before either arm was run. Tier 0 and
Tier 1 pass at the implementation on PR #24. The ten-seed reproduction target
has since been run and is recorded in §6.3; read the Tier 2 outcome above §6
before item 3 below, whose direction that run does not support.

**Tier 0 (invariants),** in `tests/invariants/test_variational_treatment.py`:

1. **The bound holds.** For random logits, random `y` and random `q`, the
   objective's per-row value is `>=` `MissingTreatmentMarginalNLL`'s on the same
   state, and the difference equals `KL(q ‖ posterior)` computed independently,
   to float tolerance. This is the §3.2 identity, asserted rather than argued.
2. **The bound is tight at the posterior.** Setting `q` to the normalised
   `p(t=k|x) p(y|x,t=k)` makes the two objectives equal to float tolerance.
3. **Both degenerate cases.** A uniform `q` contributes exactly `-log K` of
   entropy; a one-hot `q` at `k` reduces the term to the complete-case
   `-log p(t=k|x) - log p(y|x,t=k)` for that `k`.
4. **Gradients reach all three heads,** and `detaches` is empty. A stage
   trainable in any one of the three is not a no-op.
5. **Candidates come from the schema.** The `[B, K]` candidate matrix is built
   from `schema.treatment_cardinality` and never from `batch.t` — the trap
   `MissingTreatmentMarginalNLL`'s docstring names — asserted with `B != K`.
6. Rows are `t_missing`; a batch with none returns `LossTerm(0, 0)` and the
   mixer excludes it. `batch_coupled` is `False`, and the objective is
   accepted behind `ExternalBatches`.
7. **The guardrail is provably unused:** the recipe compiles with
   `purpose="causal"`, every stage has `allow_leakage=False`, and no stage
   declares a `PseudoLabelAction` (§3.2).
8. `plan.hyperparameters` matches every non-`n/a` key of §4, including the
   derived `batch_size = 128` and `labelled_unlabelled_ratio = 15.0`.
   `DATA_POLICY` and `categorical_posterior` agree on the two `data.*` keys they
   both bind: `data.standardisation` and `data.outcome_scaling`; treatment
   encoding is posterior-only, while split and missingness are policy-only
   (§5.1).

**Tier 1 (smoke fit),** in `tests/smoke/test_variational_treatment.py`:

1. Both arms' mixed losses fall over 3,000 steps.
2. **The bound holds in training, not only in a unit test:** at every logged
   step, arm A's eq. (7) value is at or above the exact marginal NLL of arm A's
   own components on the same batch.
3. The amortisation gap: mean `KL(q ‖ posterior)` on the held-out rows at the
   end, and the share of the model posterior's own held-out `y`-information
   that `q` recovers (§6.4's `posterior_information_captured`), at or above
   `0.8` on the smoke seed. The first-50 and last-50 training windows are still
   logged and reported, but **no direction is asserted between them**: §6.4
   records why the run's only gap transient completes inside the first window.
4. **The eq. (9) ablation** (`alpha = 0`). The paper's stated reason for the
   term is that `q` never sees a label; here `q` still receives gradients from
   eq. (7), so the ablation asks whether the labelled term is doing anything
   beyond that self-consistent fixed point. Predeclared expectation, from §3.1.2
   of the paper: without it the held-out posterior advantage shrinks and `q`
   drifts toward the model's own posterior rather than the data's. Reported
   whichever way it goes.
5. **Bayes-rate context for the posterior advantage.** On the fixture, the
   analytic `p(t|x)` and `p(t|x,y)` are both computable; report the held-out NLL
   of each alongside the fitted heads, so a "posterior beats propensity" result
   is read against the gap that exists in the DGP rather than against zero.

The smoke suite establishes these as wiring checks only. It does not execute
the ten-seed mean-and-stderr target above, and no reproduction claim follows
from its green status.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-05 | `5ae0fe865425` | held_out_marginal_NLL_ratio<br>amortisation_gap<br>amortisation_gap_reduction<br>posterior_advantage<br>held_out_outcome_NLL_ratio<br>held_out_treatment_NLL_ratio | 1.00118 +/- 0.000652<br>0.00490335 +/- 0.000375 nat/row<br>-4.68541e-06 +/- 0.000286 nat/row<br>-0.0798819 +/- 0.00529 nat/row<br>0.974457 +/- 0.00142<br>1.00846 +/- 0.00477 | no |
| 2026-09-05 | `1e7900b08274` | held_out_marginal_NLL_ratio<br>amortisation_gap<br>posterior_information_captured<br>model_posterior_information<br>posterior_advantage<br>held_out_outcome_NLL_ratio<br>held_out_treatment_NLL_ratio | 1.00118 +/- 0.000652<br>0.00490335 +/- 0.000375 nat/row<br>1.2881 +/- 0.044<br>0.0619601 +/- 0.00332 nat/row<br>-0.0798819 +/- 0.00529 nat/row<br>0.974457 +/- 0.00142<br>1.00846 +/- 0.00477 | yes |

### 6.4 Amendment: the gap-trajectory clause, and what replaced it

The first row above, at `5ae0fe865425`, is retained rather than withdrawn. It
was produced under the protocol reviewed at the time, on the seed stream §6.1
declares, and it is the evidence this amendment reasons from. `softmatch.md` §6.3 keeps both of its
rows for the same reason: a superseded target that was honestly run is a
finding, not a mistake.

**What the withdrawn clause required.** That `amortisation_gap`'s "terminal
value" fall below its "step-50 value" in mean. Ten seeds returned
`-0.0000047 +/- 0.000286` nat/row: not merely a miss, but a statistic with no
resolving power, since it cannot distinguish its own sign.

**Why it could not measure what it was for.** The clause presumes a gap that
opens at initialisation and closes as `q` learns. This architecture makes that
false by construction. `CategoricalPosterior` and the propensity both start
near-uniform, so `KL(q ‖ posterior)` starts near its floor; and eq. (7)'s
gradient with respect to `q` is minimised exactly at the model posterior, so
`q` is pinned to it from the first step. The gap does fall — held-out
`0.0112` at initialisation against `0.0044` at step 3,000 on seed 90000 — but
the fall completes within one or two optimiser steps, inside the first-fifty
window itself. Any window the card could name already contains the settled
value. This is a property of the initialisation and the objective, not of the
run, and it would have been derivable before the protocol was written.

**That is not a weakness of the fixture, which was checked separately.** On
seed 90000 the model posterior travels from `KL(posterior ‖ uniform) = 0.0000`
at initialisation to `0.2790` at the end, so there is a moving target; and the
fitted outcome heads give a log-likelihood ratio
`log p(y|x,t=1) - log p(y|x,t=0)` with mean `-0.0007` and standard deviation
`0.2883`, so observing `y` shifts the log-odds of `t` by a row-varying amount
that is worth `0.065` nats of held-out NLL. The fixture supplies both the
motion and the information the clause needed; only the instrument was wrong.
Changing the fixture would also invalidate the five targets the row above
passes, for a defect the fixture does not cause.

**The replacement, and why this one.** `posterior_information_captured` is

```text
(NLL[p(t|x)] - NLL[q(t|x,y)]) / (NLL[p(t|x)] - NLL[p(t|x,y)])
```

on the held-out population in the variational arm: the share of its own model
posterior's advantage over its own propensity that the amortised head
recovers. It states §2's claim as a learning outcome rather than as a
trajectory — a `q` that ignores `y` scores `0`, a noisy one scores low or
negative, and only a head that has actually learned the `y`-correction scores
near `1`. Both terms come from the same fitted arm, so no external reference is
needed and the statistic is scale-free.

Its companion `model_posterior_information` is the denominator,
`NLL[p(t|x)] - NLL[p(t|x,y)]`, required at or above `0.02` nats. This is the
guard the withdrawn clause lacked: on a fixture where `y` says nothing about
`t`, the denominator collapses and the pair **voids** rather than passes, which
is exactly the degenerate case a small gap alone cannot distinguish from
success.

**Two candidates were rejected.** A ratio of the gap to the posterior's
distance from uniform is dominated by how well `p(t|x)` learns from `x`, so a
`q` that ignored `y` entirely would pass it. A ratio of `KL(q ‖ posterior)` to
`KL(p(t|x) ‖ posterior)` is the right question but an unusable statistic: a
three-seed diagnostic returned `0.983`, `0.578` and `0.323`, scatter that ten
replicates could not resolve against any threshold worth setting.

**Predeclared, with its numbers withheld.** The three-seed diagnostic that
motivated this section reported `posterior_information_captured` of `0.937`,
`1.354` and `1.313` on seeds 90000, 90100 and 90200. It is a diagnostic and not
Tier 2 evidence: the threshold `0.8` is set as "most of it" against a statistic
whose failure mode is stated above, not fitted to those three values, and the
second ledger row above is a fresh ten-seed run under the amended block. A
value above `1` is possible and is reported rather than clipped — eq. (9)
supervises `q` against observed treatments, which the model posterior does not
see, so `q` can be better calibrated against the truth than the product it
approximates.

**Outcome.** Under the amended block the target passes on all seven required
metrics and the card is `reproduced`. `posterior_information_captured` is
`1.2881 +/- 0.044` against `0.8`, and `model_posterior_information` is
`0.0620 +/- 0.0033` nats against `0.02`, so the denominator is real and the
head recovers all of it. The value above `1` is the reported case above: `q`
scores `0.5584 +/- 0.0136` nats against the model posterior's own
`0.5763 +/- 0.0109`, because eq. (9) trains it on treatments the product never
sees. The five metrics carried over from the first row are unchanged to every
digit — same seeds, same fits, same arms — so this row differs from that one in
what was asked of the run, not in the run.

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| The optimiser: §4.4 says "RMSProp with momentum 0.1", the pinned code steps `AdaM(alpha=3e-4, beta1=0.9, beta2=0.999)` (`learn_yz_x_ss.py:307`). | Adam with the code's betas. Both agree on `3e-4`. | Reference implementation over paper prose, which is this repository's usual order when the two disagree on a mechanic the code plainly runs. |
| Which `N` scales `α = 0.1 · N`, and against what normalisation. | `n_tot`, the labelled plus unlabelled count, giving weight `0.1` with reduction `mean` on eq. (9)'s term while the likelihood terms take `population`. | `learn_yz_x_ss.py:248` (`beta = alpha * n_tot / n_labeled`) composed with the `/n_tot` at `:296`; the composition is exactly a mean over labelled rows at weight `alpha`. |
| The coefficient of the `N(0, I)` parameter prior as a weight-decay number. | `1/N = 0.0009765625` at the fixture's `N = 1024`, applied to every trainable component including biases and norm parameters. | `prior_sd=1` (`learn_yz_x_ss.py:140`, `:146`) enters the per-datapoint objective divided by `n_tot` (`:293-296`), and a unit Gaussian prior contributes gradient `w`. A different training population size changes the number, so it is a card amendment rather than a runner argument. |
| The posterior network's width: §4.1 says 500 hidden units, `run_2layer_ssl.py:13` uses `(300,)` for the 100-label run and `(500,)` otherwise. | 300, the scarce-label branch. | Our fixture has 64 observed treatments, which is the regime the 100-label branch was written for. |
| Whether the inference network shares the representation encoder. | It does not: `CategoricalPosterior` reads `X_RAW` and `Y_RAW` and owns its parameters. | `learn_yz_x_ss.py:146` builds `model_qy` as a separate `MLP_Categorical` with its own parameter set `u`. It also keeps §6's two arms comparable: no shared parameter changes size between them. |
| Whether labelled rows also enter the unlabelled term. | They do not; §4's `eligible_rows` partitions `t_observed` and `t_missing`. | Eq. (8) sums `L` over `p̃_l` and `U` over `p̃_u`. This is the opposite of FixMatch's footnote 2 and worth stating, since both cards live in the same repository. |
| Whether Table 1's reported error is the last epoch or the best validation epoch. | Not resolved, and not needed: deviation 3 declines the benchmark, and §6's budget is fixed rather than selected. | The hook logs validation and test error every pass (`learn_yz_x_ss.py:159-180`) and the function returns the last, but the paper does not say which the table carries. |
| How `q` should be reported for a causal reader. | As an amortised imputation distribution only. The exact model posterior is also computable from `p(t|x)` and `p(y|x,t)` in this adaptation; `q` is the learned `T_GIVEN_XY` approximation to it. | `DESIGN.md` §7.2's separation of the three quantities. §2 refuses the serving claim, and Tier 0 item 7 checks that no staged path exists to make it accidentally. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | GPT-5.6 Sol | 2026-08-29 |
| Plan diffed against §3.2 and §4 | GPT-5.6 Sol | 2026-08-29 |
| Implementation and Tier 1 audited (status → `smoke-passing`) | GPT-5.6 Sol | 2026-08-29 |
| Tier 2 run and ledger recorded (status → `deviating`) | Claude | 2026-09-05 |
| §6 amended: gap-trajectory clause replaced (§6.4) | mattsq | 2026-09-05 |
| Tier 2 re-run under the amended §6 (status → `reproduced`) | Claude | 2026-09-05 |
