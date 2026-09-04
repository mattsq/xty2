# Recipe spec card: softmatch

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2–§5 to implement or audit fidelity;
> §6 only for benchmark/reporting work.

> Written card-first, before any code, per `CLAUDE.md` rule 1. The reviewed
> card now has a recipe, objective, Tier 0 invariants, a Tier 1 paired fit and
> a Tier 2 runner whose ten-seed result is recorded in §6.3. That result is
> `deviating`: seven of the eight declared targets pass — the recipe beats the
> constant gate on held-out treatment NLL and raises quantity — and the one
> that fails is §6's quality guardrail, which is the clause §6 wrote so that it
> could fail on its own and the cost §2's first limitation predicts. §6.2's
> five-seed `no_ua`/`matched` directions remain outstanding.
>
> It is also the card `freematch.md` §5.3 named in advance. That section asked
> whether the three gated pseudo-label objectives should collapse into one
> policy union, answered "wait", and said the recipe that would settle it is
> "one whose gate is a fourth kind — SoftMatch's continuous weight, which is
> not a gate at all but a per-row multiplier". §5.3 below is that answer, and
> the paper turns out to supply the unification argument the repository was
> missing.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-supervised Learning](https://arxiv.org/abs/2301.10921) |
| Authors, year | Hao Chen, Ran Tao, Yue Fan, Yidong Wang, Jindong Wang, Bernt Schiele, Xing Xie, Bhiksha Raj, Marios Savvides; 2023 |
| DOI / arXiv | [arXiv:2301.10921](https://arxiv.org/abs/2301.10921); ICLR 2023 |
| Version used | [arXiv:2301.10921v2](https://arxiv.org/abs/2301.10921v2), revised 2023-03-15 and fetched 2026-09-04. §2.1 gives the framework and eqs. (1), (2); §2.2 gives definitions 2.1–2.3, eqs. (3), (4) and table 1; §3.1 gives eqs. (5)–(7); §3.2 gives UA and eqs. (8), (9); §4.1 the classic-image protocol; §4.5 the ablations; appendix A.1 the quantity/quality derivations; appendix A.2 Algorithm 1; appendix A.3.1 table 6; appendix A.5 tables 10–12. |
| Reference implementation | The authors' paper-era [TorchSSL SoftMatch at `03193a1`](https://github.com/TorchSSL/TorchSSL/blob/03193a1b7883727db1ce9c092e083091e18aedbb/models/softmatch/softmatch.py), used for the paper's classic-image and ablation experiments, and [USB SoftMatch at `5c9ee21`](https://github.com/microsoft/Semi-supervised-learning/blob/5c9ee2148ab1a30e63fa05414f12033c6952d1fc/semilearn/algorithms/softmatch/softmatch.py), the first post-v2 SoftMatch revision. The redirect repository is [`Hhhhhhao/SoftMatch`](https://github.com/Hhhhhhao/SoftMatch). §7 records the two code paths' disagreements instead of silently selecting one. |
| Reference impl. runnable? | Not run end-to-end: both historical stacks require their image datasets and environments. The pinned SoftMatch, weighting, alignment, optimiser and schedule sources were inspected directly. |

The paper is authoritative for the method. Where it is explicit, this card
follows it even if one code path differs. Where it is ambiguous, the
paper-era TorchSSL path used for §4.1 is the tie-breaker; USB is corroborating
evidence and its disagreements are recorded in §7.

## 2. Estimand and claim

- **Estimand:** As `fixmatch`, `flexmatch` and `freematch`: the categorical
  propensity `p(t | x)`, composed with the retained causal stack — the
  treatment-conditional outcome distribution `p(y | x, t=k)` and its means
  `mu_k(x)`, whose contrasts identify conditional treatment effects under
  consistency, positivity and conditional exchangeability.
- **Claim:** SoftMatch replaces the pseudo-label *gate* with a pseudo-label
  *weight*. §2.1 rewrites the whole family as one weighted cross-entropy
  (eq. 2) in which FixMatch's indicator is a step function `lambda(p)`, and
  §2.2 defines the two quantities that step function trades off: the
  **quantity** `f(p) = E[lambda(p)]` (eq. 3) and the **quality**
  `g(p) = E_{lambda-bar}[1(p_hat = y^u)]` (eq. 4). §3.1 replaces the step with
  a truncated Gaussian on the confidence (eq. 5) whose mean and variance are
  EMAs of the model's own batch confidence statistics (eqs. 6, 7), and §3.2
  adds Uniform Alignment, which divides each prediction by a running estimate
  of the model's marginal before the weight is computed, without touching the
  hard label it is charged against (eqs. 8, 9). The paper claims
  state-of-the-art error rates on CIFAR-10/100, SVHN, STL-10, ImageNet and
  text benchmarks, and — the claim this card actually cares about — that both
  quantity *and* quality rise, so that the trade-off itself is removed (§4.4,
  figs. 2(b), 2(c)).

  This card claims that the mechanism is faithfully assembled around
  `p(t | x)` in xty2, and that on the fixed project-local target in §6 it
  improves held-out treatment prediction against an otherwise identical
  `fixmatch` constant-gate arm **while raising quantity without a
  proportionate loss of quality**. §6's tolerance is written so that the last
  clause can fail on its own: the paper's own definitions of quantity and
  quality (eqs. 3, 4) are computed on both arms of the pair, and a run that
  wins on NLL while the impurity of the weighted labels rises past the declared
  bound is `deviating`, not a pass.
- **Not claimed:** No image or text number is claimed. Five limitations are
  structural and are stated here rather than left to be discovered:

  1. **Every row trains at every step, and at `K = 2` the first step trains
     every row at almost full weight.** Eq. (5) is strictly positive everywhere, so
     unlike every earlier card in this family SoftMatch never rejects a row —
     the paper's own bound is `f(p) >= lambda_max / 2` (appendix A.1). Worse
     for a two-class fixture: `mu_hat_0 = 1/C = 0.5` (§3.1) and a two-class
     softmax has `max(p) >= 0.5` for every row. Algorithm 1 and both pinned
     implementations fold the first batch into the EMAs before weighting it,
     so `mu_hat_1` is slightly above `0.5`: rows below that updated mean receive
     weights just below `lambda_max`, while the initial variance near `1.0`
     makes the difference tiny. The earlier draft incorrectly skipped that
     first update and claimed an exactly flat first step. FixMatch's
     ungated warm-up (`fixmatch.md` §2) and FreeMatch's (`freematch.md` §2)
     are phases; SoftMatch's is the design. The consequence is that the
     Bayes-label flip rate of the strong view is charged against `p(t | x)` on
     every row for the whole run, which is why deviation 2 takes `flexmatch`'s
     strong view rather than `fixmatch`'s and why the training-free check
     `flexmatch.md` §5.2 introduced is a precondition of this card rather than
     an afterthought.
  2. **The weight is self-referential, in the same way FreeMatch's threshold
     is and for the same reason.** `mu_hat_t` and `sigma_hat_t` are moments of
     the model's own confidence, so a model that becomes confident past what
     the assignment supports recentres the Gaussian on that confidence and
     keeps full weight. `freematch.md` §2's second limitation applies verbatim;
     §6's DGP lives in the near-deterministic-assignment corner where the
     confidence is warranted, and this card claims nothing outside it.
  3. **Quantity and quality are defined against a label this recipe does not
     have.** Eq. (4) needs `y^u`, the true label of an unlabelled row. On the
     rows this term trains, `t` is by definition missing, so quality is
     measurable only against the §6 fixture's own ground truth, off the
     training path and after the fact. Quantity (eq. 3) needs nothing and is
     logged during training; quality is a benchmark metric only.
  4. **`K = 2` with a balanced treatment marginal leaves Uniform Alignment
     with almost nothing to do.** UA divides by a running estimate of the
     model's mean prediction and renormalises towards `u(C)`; on a fixture
     whose marginal is 0.5/0.5 the ratio is near 1 and eq. (9) is close to
     eq. (5). This is the same finding `freematch.md` §2's third limitation
     records for SAF, reached by a different route, and §6.2 predeclares the
     paper's own `w/o UA` ablation (§4.5) as a Tier 1 arm so that the question
     is answered with a number rather than asserted.
  5. **A pseudo-label on `t` is not a label on `y`.** As in `fixmatch`,
     `flexmatch` and `freematch`: weighted rows train `p(t | x)` only, never
     `ObservedOutcomeNLL`, so no inferred treatment is used as if observed and
     the `DESIGN.md` §7.2 leakage rule is not engaged. A *weight* changes
     nothing about that rule — it is the objective that decides which heads a
     row reaches, not the size of its contribution.

## 3. Equations and mapping

### 3.1 As published

Notation is §2.1's. `D_L` and `D_U` are the labelled and unlabelled sets;
`B_L` and `B_U` are the labelled and unlabelled batch sizes; `omega(.)` is weak
and `Omega(.)` strong augmentation; `H(.,.)` is cross-entropy; `C` is the
number of classes; `p` abbreviates `p(y | omega(x^u))` and `p_hat` its one-hot
`argmax(p)`.

Section 2.1, the framework and the unified weighted form of the unsupervised
term:

$$
\mathcal{L}_{s} = \frac{1}{B_{L}}\sum_{i=1}^{B_{L}}
  \mathcal{H}\!\left(\mathbf{y}_{i},\, \mathbf{p}(\mathbf{y}\mid \mathbf{x}_{i}^{l})\right)
\tag{1}
$$

$$
\mathcal{L}_{u} = \frac{1}{B_{U}}\sum_{i=1}^{B_{U}}
  \lambda(\mathbf{p}_{i})\,
  \mathcal{H}\!\left(\hat{\mathbf{p}}_{i},\, \mathbf{p}(\mathbf{y}\mid \Omega(\mathbf{x}_{i}^{u}))\right)
\tag{2}
$$

with `lambda(p)` the sample weighting function with range `[0, lambda_max]`,
and the joint objective `L = L_s + L_u`.

Section 2.2, the two quantities the weighting function trades off
(definitions 2.1 and 2.2), where `lambda_bar` is the normalised weight:

$$
f(\mathbf{p}) = \mathbb{E}_{\mathcal{D}_{U}}[\lambda(\mathbf{p})] \in [0, \lambda_{\max}]
\tag{3}
$$

$$
g(\mathbf{p}) = \sum_{i}^{N_{U}} \mathbb{1}(\hat{\mathbf{p}}_{i} = \mathbf{y}_{i}^{u})
  \frac{\lambda(\mathbf{p}_{i})}{\sum_{j}^{N_{U}}\lambda(\mathbf{p}_{j})}
  = \mathbb{E}_{\bar\lambda(\mathbf{p})}\!\left[\mathbb{1}(\hat{\mathbf{p}} = \mathbf{y}^{u})\right] \in [0, 1]
\tag{4}
$$

Table 1 instantiates eq. (2) for the family: naive pseudo-labelling is
`lambda = lambda_max`; confidence thresholding is
`lambda = lambda_max * 1(max(p) >= tau)`; SoftMatch is eq. (5).

Section 3.1, the truncated Gaussian weight and the estimates of its two
parameters:

$$
\lambda(\mathbf{p}) = \begin{cases}
\lambda_{\max}\exp\!\left(-\frac{(\max(\mathbf{p})-\mu_{t})^{2}}{2\sigma_{t}^{2}}\right), & \text{if } \max(\mathbf{p}) < \mu_{t},\\
\lambda_{\max}, & \text{otherwise.}
\end{cases}
\tag{5}
$$

$$
\hat\mu_{b} = \frac{1}{B_{U}}\sum_{i=1}^{B_{U}}\max(\mathbf{p}_{i}),
\qquad
\hat\sigma_{b}^{2} = \frac{1}{B_{U}}\sum_{i=1}^{B_{U}}\left(\max(\mathbf{p}_{i})-\hat\mu_{b}\right)^{2}
\tag{6}
$$

$$
\hat\mu_{t} = m\,\hat\mu_{t-1} + (1-m)\,\hat\mu_{b},
\qquad
\hat\sigma_{t}^{2} = m\,\hat\sigma_{t-1}^{2} + (1-m)\,\frac{B_{U}}{B_{U}-1}\,\hat\sigma_{b}^{2}
\tag{7}
$$

with `mu_hat_0 = 1/C` and `sigma_hat_0^2 = 1.0` (§3.1).

Section 3.2, Uniform Alignment and the final weighting function:

$$
\operatorname{UA}(\mathbf{p}) = \operatorname{Normalize}\!\left(
  \mathbf{p}\cdot\frac{\mathbf{u}(C)}{\hat{\mathbb{E}}_{B_{U}}[\mathbf{p}]}\right),
\qquad \operatorname{Normalize}(\cdot) = \frac{(\cdot)}{\sum(\cdot)}
\tag{8}
$$

$$
\lambda(\mathbf{p}) = \begin{cases}
\lambda_{\max}\exp\!\left(-\frac{(\max(\operatorname{UA}(\mathbf{p}))-\hat\mu_{t})^{2}}{2\hat\sigma_{t}^{2}}\right), & \text{if } \max(\operatorname{UA}(\mathbf{p})) < \hat\mu_{t},\\
\lambda_{\max}, & \text{otherwise.}
\end{cases}
\tag{9}
$$

Algorithm 1 (appendix A.2) is the procedure, and its ordering is load-bearing
for this port:

```text
 1  Input: C, labelled batch {x_i, y_i}, unlabelled batch {u_i}, EMA momentum m
 2  Define: p_i = p(y | omega(u_i))
 3    L_s = (1/B_L) sum H(y_i, p(y | omega(x_i)))            (eq. 1, weak view)
 4    mu_b     from THIS batch's max(p_i)                    (eq. 6)
 5    sigma_b^2 from THIS batch's max(p_i)                   (eq. 6)
 6    mu_hat_t     = m mu_hat_{t-1} + (1-m) mu_b             (eq. 7)
 7    sigma_hat_t^2 = m sigma_hat_{t-1}^2 + (1-m)(B_U/(B_U-1)) sigma_b^2
 8  for i = 1 to B_U do
 9      lambda(p_i) from max(UA(p_i)) against mu_hat_t, sigma_hat_t   (eq. 9)
10  end for
11    L_u = (1/B_U) sum lambda(p_i) H(p_hat_i, p(y | Omega(u_i)))     (eq. 2)
12  Return: L_s + L_u
```

Four readings the mapping depends on:

* **The statistics are updated from the current batch *before* that batch is
  weighted.** Lines 4–7 precede line 9. As in FreeMatch and unlike FlexMatch, a
  row's weight depends on the confidences of the other rows in its own batch,
  so the objective is `batch_coupled = True`.
* **Line 9 compares an *aligned* confidence against moments estimated from
  *unaligned* confidence.** Lines 4–5 read `max(p_i)`; line 9 reads
  `max(UA(p_i))`. This is followed literally and recorded in §7.6, because it
  is the one place where the two halves of §3 meet and the paper never says
  the two quantities are on the same scale.
* **UA changes the weight and not the label.** §3.2 is explicit: "UA avoids
  this issue by exploiting original predictions to compute pseudo-labels and
  normalized predictions to compute sample weights". `p_hat` in eq. (2) is
  `argmax(p)`, never `argmax(UA(p))`. This is the stated difference from
  Distribution Alignment, and it is the single easiest thing to get wrong.
* **`lambda_max` and `w_u` are the same number.** Eq. (2) carries `lambda_max`
  inside the sum; the joint objective is `L_s + L_u` with no separate
  unsupervised weight; Algorithm 1 line 9 writes `1.0` where eqs. (5) and (9)
  write `lambda_max`. Since `lambda_max` multiplies every row identically, it
  factors out of the objective exactly, and §3.2 binds it to `losses.weights`
  so there is one place to change it.

### 3.2 Mapping to xty2

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p(y \| x)` | model's class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| `omega(.)` | weak augmentation | — | `ViewSpec("weak_x")`, `FeatureMask(p=0.1)`, two draws |
| `Omega(.)` | strong augmentation | — | `ViewSpec("strong_x")`, `FeatureMask(p=0.1)` then `FeatureMask(p=0.2)` — `flexmatch`'s, not `fixmatch`'s; deviation 2 |
| eq. (1) `L_s` | supervised cross-entropy on weak views | `T_GIVEN_X @ weak_x draw=1` | `ObservedTreatmentNLL(realisation=Realisation("weak_x", draw=1))`, rows `t_observed`, `reduction="mean"` |
| `p_i` | weak-view artificial label distribution | `T_GIVEN_X @ weak_x draw=0` | `SoftWeightedTreatmentNLL.target` |
| `p_hat_i` | hard pseudo-label `argmax(p_i)` | — | `sharpening="hard"` inside that objective |
| `p(y \| Omega(u_i))` | strong-view prediction | `T_GIVEN_X @ strong_x` | `.prediction` on that objective |
| eq. (6), (7) `mu_hat_t` | EMA of the mean top-class confidence | — | `ConfidenceGaussian.mean`, the objective's per-stage state |
| eq. (6), (7) `sigma_hat_t^2` | EMA of the unbiased confidence variance | — | `ConfidenceGaussian.variance` |
| eq. (8) `E_hat[p]` | EMA of the mean predicted class distribution | — | `ConfidenceGaussian.marginal`, `[K]` |
| `m` | the decay of eq. (7) and of eq. (8)'s estimate | — | `TruncatedGaussianWeighting(decay=0.999)`, bound to `losses.confidence_threshold` |
| §4.1 "divide the estimated variance by 4" | the `2 sigma` range | — | `TruncatedGaussianWeighting(n_sigma=2)`; §7.2 |
| eq. (8) `UA(.)`, `u(C)` | uniform alignment and its target | — | `TruncatedGaussianWeighting(alignment="uniform")`; `alignment="none"` is the paper's §4.5 `w/o UA` arm |
| eq. (9) `lambda(p_i)` | the per-row weight | — | computed inside `SoftWeightedTreatmentNLL`; logged as `quantity` (eq. 3), with `mu_hat_t` and `sigma_hat_t` beside it |
| eq. (2) `L_u` | softly weighted pseudo-label cross-entropy | `T_GIVEN_X` at both views | `SoftWeightedTreatmentNLL`, rows `all`, `reduction="mean"` |
| eq. (2), (5) `lambda_max` | the weight ceiling, `= w_u` | — | `Weighted(..., weight=1.0)`, `Constant` |
| `B_L`, `B_U` | batch composition | — | `QuotaSampler(Quota("t_observed", 64), Quota("t_missing", 448))`, imported from `fixmatch` |
| `eta_0 cos(7 pi k / 16 K)` | rate schedule (table 6) | — | `CosineDecay(steps=3000, phase=7/16)` |
| table 6 "Model EMA Momentum" | the model §4.1 evaluates | — | `TeacherSpec(decay=0.999, role="evaluation")`; no objective reads it |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

Four mapping decisions carry the fidelity of this port.

**One objective, one state, and no sibling read.** SoftMatch has a single
unsupervised term. `mu_hat_t`, `sigma_hat_t^2` and `E_hat[p]` are three
statistics of one batch quantity, all decayed at one `m`, all read by eq. (9)
and by nothing else, so they are one `ConfidenceGaussian` owned by one
objective. The `DESIGN.md` §4 sibling-state read that `freematch` needed is
not engaged here, and the idempotence obligation that comes with it does not
arise: there is exactly one writer and exactly one reader, in one `compute`.

**The per-row weight needs no framework change, because it is where the mask
already was.** `PseudoLabelTreatmentNLL` computes a 0/1 mask inside `compute`
and multiplies the per-row cross-entropy by it; `LossTerm.value` is then the
mean of that product over eligible rows (`DESIGN.md` §4). Eq. (2) is the same
sentence with `lambda(p_i) in (0, 1]` in place of `1(max(p) >= tau)`. In
particular this does **not** touch `batch.weight`, which reaches
`ObservedOutcomeNLL` alone — `pseudo_label.py` records that a recipe wanting
weighted pseudo-labels "is asking for a framework-wide decision about which
objectives weights reach", and this card is not asking for one. The weight
here is a function of the model's predictions computed by the objective, not a
property of a row supplied by the data.

**The denominator counts every row, so `rows = all` and `reduction = mean`.**
Eq. (2) divides by `B_U`, and SoftMatch's `lambda` is inside that sum, so a
batch of uniformly unconfident rows contributes a small number rather than an
average over a small set — identical arithmetic to `fixmatch`, `flexmatch` and
`freematch`, with a continuous multiplier instead of a binary one. `U` is
every training row for the reason `fixmatch.md` §3.2 gives and §7.7 records:
SoftMatch §2.1 restates that framework and §4.1 runs it in TorchSSL.

**`lambda_max` lives in the mixer's weight, not in the policy.** Eq. (2)
multiplies every row by it, `Weighted` multiplies the whole term by `w`, and
the two are the same operation. Binding it to `losses.weights` keeps the
policy object to the three numbers that actually shape the curve (`m`,
`n_sigma`, and the alignment target) and makes the paper's own `L = L_s + L_u`
readable in the plan as `weight=1.0`.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.soft_weighted_treatment_nll: p(t|x) @ view=weak_x params=student
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target        # eq. (2) reads p only through an arg max and, via eqs. (6)-(9), through EMA buffers and a max; nothing on that side carries a gradient
  gradient_clipping: none         # the paper names none; retained P5 choice
  marginal_nll_grad_path: both    # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.999                # table 6, "Model EMA Momentum"; §4.1 "We record the EMA of model parameters for evaluation with a momentum of 0.999"
  ema_applies_to_buffers: false   # the declared graph has no buffers; stated so a component that grew one would be a card change
  teacher_in_train_mode: false    # the EMA copy is an evaluation classifier
  teacher_requires_grad: false    # never an optimiser target
  # role = evaluation. Nothing reads this EMA during training: eqs. (5)-(9) are
  # all statements about the current network.

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: mean      # eq. (1) divides by B_L
    joint_fit.soft_weighted_treatment_nll: mean # eq. (2) divides by B_U
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.soft_weighted_treatment_nll: all  # FixMatch footnote 2, inherited through §2.1; §7.7
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.soft_weighted_treatment_nll: 1.0  # lambda_max; Algorithm 1 line 9 and L = L_s + L_u
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.soft_weighted_treatment_nll: constant 1.0   # the adaptation is in lambda(p), not in a ramp; table 1 lists ramp-up as a different lambda
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a                # SoftMatch never sharpens a soft target: eq. (2) charges a hard arg max, and UA is a normalisation, not a temperature
  sharpening: hard                # eq. (2): H(p_hat_i, .) with p_hat = argmax(p)
  confidence_threshold: truncated_gaussian(decay=0.999, n_sigma=2, alignment=uniform)  # eqs. (5)-(9); mu_hat_0 = 1/K, sigma_hat_0^2 = 1.0, m = 0.999 (§4.1), 2 sigma range (§4.1, §A.5)

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)   # §4.1 and table 6 give SGD with momentum 0.9; Nesterov is TorchSSL's, which §4.1 says every experiment ran in (§7.5)
  lr: 0.03                                      # §4.1 and table 6
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))  # table 6's eta_0 cos(7 pi k / 16 K); K = our 3000 steps
  weight_decay: 0.0005 (all trainable components; norm and bias exempt)  # table 6's CIFAR-10/STL-10 value; scope follows fixmatch.md deviation 8
  batch_size: 512                               # B_L + B_U = 64 + 448, derived from the QuotaSampler's quotas
  labelled_unlabelled_ratio: 7.0                # §4.1: "B_U is set to 7 times of B_L for all datasets"; derived from the same quotas
  total_steps_or_epochs: 3000                   # optimiser steps, never epochs. The paper's total is 2^20; see deviation 3

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                # retained reviewed P5 TARNet backbone
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
    mlp_encoder: 0.0                            # perturbation comes from the two explicit input views
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'train'
  outcome_scaling: y: zscore fitted on 'train'
  treatment_encoding: n/a                       # XTYBatch supplies integer classes 0..K-1
  split_protocol: one fixed project-local DGP, split train/test by the section 6 fixture; no CIFAR/SVHN/STL protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id  # deviation 8
```

`losses.confidence_threshold` holds a **policy object**, as it does in
`flexmatch.md` §4 and `freematch.md` §4 and for the same reason: `card_keys.py`
refuses two fields bound to one canonical key, so a rule with several numbers
in it is bound once, whole. This is the third card to read the key that way and
the first whose rule contains no threshold at all in any form — SoftMatch's
`lambda(p)` has a *breakpoint* at `mu_hat_t`, but a row below it still trains.
The key is therefore now carrying "the rule that decides how much each
artificial label counts", which is what eq. (2) says it was all along; §5.3
takes that up.

The **weak** view is `fixmatch`'s, unchanged and imported: `FeatureMask(p=0.1)`
with two draws (Algorithm 1 line 3 puts the labelled rows through `omega` too,
so the second draw is used exactly as in `freematch`). The **strong** one is
`flexmatch`'s: `FeatureMask(p=0.1)` then `FeatureMask(p=0.2)`, imported so the
reviewed value has one home. Deviation 2 is why. Both views preserve `t`, `y`,
`t_observed`, `y_observed`, `row_id`, `fold_id` and `weight`, and
`softmatch(schema, recompute_rules=(...))` passes the same explicit rules to
both. §4's YAML has no key for a view, so a Tier 0 test must compare this
paragraph's two transforms against the compiled plan, as
`tests/invariants/test_freematch.py` does.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply SoftMatch to categorical treatment assignment `p(t \| x)` and compose it with a causal outcome likelihood (`ObservedOutcomeNLL`) and exact marginalisation (`MissingTreatmentMarginalNLL`). | The paper studies image and text classes. The project-local question is whether a *weight* recovers a missing treatment label better than a *gate*, and whether it composes with the reviewed P5 stack rather than replacing it. | No comparison to a published error rate is valid. The marginal term trains `p(t \| x)` on exactly the rows the weight is scaling, so the two mechanisms interact; §6 measures the pair against the constant gate, which carries the same interaction. |
| 2 | `judgement` | — | Replace crop-and-flip (weak) and RandAugment (strong) with schema-aware feature masking: 10% weak, and 10% followed by 20% strong — `flexmatch`'s strong view, not `fixmatch`'s 50%. | There is no image structure in a tabular XTY batch. `flexmatch.md` §5.2 measures the label-preservation half of FixMatch §2.3's requirement on this exact DGP: at an effective 0.55 the view flips the Bayes-optimal label on 16.8% of rows, at 0.28 on 7.4%. §2's first limitation is why this card cannot inherit the 0.55: SoftMatch assigns a strictly positive weight to *every* row and full weight to every row at or above `mu_hat_t`, so a flipped label is never masked out, at any point in training, on any row. | Directly defines the invariance being learned, and this recipe is the most exposed of the family to getting it wrong. §6's pair holds it fixed on both arms, so it bounds what the numbers describe rather than confounding what they compare. |
| 3 | `judgement` | — | Train for 3,000 optimiser steps rather than the paper's `2^20`. | The reviewed project-local budget, shared with every other xty2 recipe so that a difference is attributable to the recipe. The cosine schedule's `K` is set to the same 3,000, so the shape of the decay is exact even though its length is not. | §4.4 reports SoftMatch's advantage as largest "especially for the first 50k iterations", so a short budget is where the weighting should matter most. It also means `mu_hat_t` and `sigma_hat_t^2` see 3,000 updates at `m = 0.999`, roughly three EMA horizons; §6 records their trajectory rather than the endpoint alone. |
| 4 | `judgement` | — | Retain the P5 TARNet architecture (encoder, outcome head, propensity) rather than WRN-28-2. | Holding the causal stack fixed is what makes the addition attributable, and it is `fixmatch.md` deviation 6, `flexmatch.md` deviation 4 and `freematch.md` deviation 4. | The project-local result validates wiring and mechanism, not image-scale accuracy. |
| 5 | `judgement` | — | Retain P5's `Ramp(0.0, 0.5, 1000)` on the marginal-likelihood term while the unsupervised weight stays constant. | The ramp belongs to the reviewed P5 term, not to SoftMatch; §2.2 and table 1 are explicit that a ramped `lambda` is a *different* weighting function, so ramping `w_u` here would silently port a method the card is not porting. | Identical to `fixmatch`'s, `flexmatch`'s and `freematch`'s arrangement, so the pair in §6 shares it. |
| 6 | `judgement` | — | Port only the §4.1 configuration: all-class Gaussian estimation with UA. Do not implement §4.5's per-class Gaussians, its fixed `mu = 0.95, sigma^2 = 0.01` arm, or its linear, quadratic and truncated-Laplacian weighting functions. | Those are the paper's ablations of its own choice, not the method. §4.5 selects all-class + UA, and `FIDELITY.md` §4.1's "port methods lazily, in response to a reviewed need" applies: a second weighting family arrives with the card that needs it. The one ablation that *is* implemented is `alignment="none"`, because it is expressible in the declared policy field and §6.2 predeclares it as an arm. | None on the primary metric. It bounds what §6 can attribute: a failure of this recipe is a failure of the truncated Gaussian with all-class estimation, not of every `lambda` in §4.5. |
| 7 | `framework-limitation` | `augmentation-vocabulary` | No RandAugment and no augmentation-strength vocabulary: the strong view's strength is a fixed scalar. | The argument is `fixmatch.md` deviation 10's and is not restated. SoftMatch adds nothing to the case either way — its contribution is the weighting function, and table 6 runs the same RandAugment the earlier cards already deviate from. | Removes whatever augmentation diversity buys, equally from both arms of §6's pair, so it is a limit on what the numbers describe rather than a confound within them. |
| 8 | `framework-limitation` | `batch-row-repetition` | Set the §6 label budget to 64 rather than a scarcer regime, holding `B_L = 64` and `B_U = 448` at the paper's values. | `XTYBatch.row_id` must be unique (`DESIGN.md` §7.1), so a labelled quota of `B_L` cannot be drawn from a population smaller than `B_L` without repeating a row, and the scarcest budget expressible is `B_L` itself. The alternative — lowering `B_L` — would deviate from a number the paper states. | Table 2's largest margins are the 40-label CIFAR-10, 400-label CIFAR-100 and 40-label STL-10 settings, where quantity matters most because so few rows clear a fixed gate. At 64 labels over `K = 2` this fixture is not in that regime, so the mechanic is being measured where it has least to gain. It moves both arms of the pair equally. |
| 9 | `judgement` | — | Declare the `alignment="none"` comparison a Tier 1 arm rather than a second Tier 2 target. | §2's fourth limitation predicts UA is near-inert at `K = 2` on a balanced marginal, and `freematch.md` deviation 10 made the same call for SAF for the same reason: making it a reproduction target would multiply the nightly cost of a card whose declared claim is the paired one. | None on the §6 metric. What changes is what §6.2 may claim about UA: a direction on the declared Tier 1 seeds, not a target. |

### 5.1 Framework additions made for this card

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `TruncatedGaussianWeighting` (the policy), `ConfidenceGaussian` (the state) and `SoftWeightedTreatmentNLL` (eqs. 2, 5–9), in `xty2/objectives/soft_weighting.py` | fidelity-bearing, reversible | This card | — (not the load-bearing quadrant; nothing outside this recipe is written against them) | An objective is the ordinary extension point `BACKLOG.md` step 4 names. Eq. (2)'s multiplier is continuous and eqs. (6)–(9) need per-stage state, which no existing objective has: `PseudoLabelTreatmentNLL` takes a `float`, `CurriculumPseudoLabelTreatmentNLL` a per-row mark table, and `SelfAdaptiveThresholdTreatmentNLL` an EMA pair it converts into a hard per-class gate. §5.3 is the review of whether the four should now be one |

**No code is added to `core/`, and no card key is added.** The three
capabilities this objective needs — stage-local state through
`initial_state(TrainingPopulation | None)`, `batch_coupled = True`, and a
per-row multiplier applied inside `compute` — all exist and are all in use.
`ConfidenceGaussian` is built from `K` alone and ignores the population
argument, exactly as `SelfAdaptiveThresholds` does; that signature is now
load-bearing on its third consumer.

**`E_hat[p]` is FreeMatch's `p~_t` under another name, and that is worth
recording rather than hiding.** Eq. (8)'s estimate is an EMA of the mean
predicted class distribution over the unlabelled batch, decayed at 0.999 — the
same statistic, the same decay and the same initialisation as `freematch`
eq. (6). The two recipes then use it for different things (a per-class
threshold scale there, a per-row renormalisation here), so this is duplication
of a *statistic*, not of a mechanism. `BACKLOG.md` §15.2 says to keep each
mechanism local until two recipes need the same lifecycle *and* semantics;
the lifecycle now matches on two cards and the semantics do not. What would
settle it is a third consumer that needs the running marginal for a third
purpose — ReMixMatch's distribution alignment (`BACKLOG.md` §2.3) is the
obvious candidate, and its target is the *labelled* marginal rather than
`u(C)`, which is the shape question a promoted object would have to answer.
Recorded so that card inherits the question rather than rediscovering it.

### 5.2 Two things this card is expected not to need

* **A row-weight contract.** The framework's `batch.weight` reaches
  `ObservedOutcomeNLL` alone, and `pseudo_label.py` records that widening it is
  a framework-wide decision. Eq. (2)'s `lambda(p_i)` is not that: it is
  computed by the objective from the model's own predictions, in the place the
  0/1 mask is computed today. If review disagrees, the alternative is a debt
  row, not a quiet widening.
* **A stateful sampler.** `stateful-sampler` exists because a sampler reading
  the model would destroy the paired batch stream every ablation here rests on.
  SoftMatch reads the model and changes how much each row *counts*, not which
  rows are *drawn*, so §6's pair still shares a batch stream exactly.

### 5.3 The answer to `freematch.md` §5.3

That section asked whether the family's gated pseudo-label objectives should
collapse into one policy union, answered "wait", and named this card as the
one that would settle it. Three things changed:

* **The shared surface is not a gate.** `freematch.md` §5.3 framed the question
  as a union over three *gates* and rejected it partly because
  `batch_coupled` — a `bool` the compiler reads — would end up behind a policy
  field, making a declaration depend on a value. That objection stands, and
  SoftMatch is now the second `batch_coupled = True` member, so it is no longer
  a two-against-one asymmetry that a union could paper over.
* **The paper supplies the abstraction the repository was missing.** §2.1's
  eq. (2) and table 1 are exactly the unification: naive pseudo-labelling,
  ramp-up, confidence thresholding, adaptive thresholding and SoftMatch are one
  weighted cross-entropy differing only in `lambda(p)`. If xty2 ever collapses
  these four objectives, this is the shape to collapse them into — a
  **weighting function**, not a threshold policy — and the card key
  `losses.confidence_threshold` is already being used to carry it.
* **It is still not worth doing in this card.** Building it would rewrite four
  objectives, three of which are `reproduced`, and `DESIGN.md` §11.6 is
  explicit that new fidelity-bearing vocabulary moves plans, digests and
  benchmark identity. This card needs none of it to be faithful. The
  recommendation is therefore: keep the fourth objective local, and revisit
  when a fifth consumer arrives with a `lambda` that none of the four can
  express — `BACKLOG.md` §2.5's SequenceMatch is the named candidate — or when
  a card needs two recipes' weighting rules to be interchangeable at runtime,
  which nothing does today.

### Tier 2 outcome

On 2026-09-04, commit `7e55fdc8e921` produced a `deviating` result: This is the predeclared project-local SoftMatch mechanism target: whether continuous Gaussian weights improve missing-treatment classification while raising weighted quantity without losing the declared pseudo-label quality. It is not an image benchmark. Failed target(s): weighted_impurity_vs_1.25x_gate was 0.00881344 +/- 0.00175 against mean <= 0, by at least one stderr.

## 6. Reproduction target

The published CIFAR-10/100, SVHN, STL-10, ImageNet and text error rates cannot
validate this port: the inputs, the labels, the architecture and the metric all
differ, and the estimand is a treatment assignment rather than an image class.
The target below is a fixed project-local *mechanism* target in the form
`fixmatch.md` §6 and `freematch.md` §6 use.

**This result will be measured without two mechanics the paper's framework
states**, and `DESIGN.md` §11.6 wants that beside the number rather than in the
gap between sections: §5.7 (`augmentation-vocabulary`) fixes the strong view's
strength where the reference runs RandAugment, and §5.8
(`batch-row-repetition`) sets the label budget at 64 rather than in the
scarce-label regime where table 2 reports SoftMatch's largest margins. Both
apply equally to every arm.

```yaml
reproduction:
  dataset: fixmatch.md §6.1's project-local seed-locked two-cluster XTY DGP (6 features, K=2), unmodified
  variant: paired fit against a constant-gate arm — this recipe with eq. (2)'s lambda replaced by FixMatch's indicator at tau = 0.95, so both arms share deviation 2's views — same seeds and same batches
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio on the EMA parameters, SoftMatch over the constant-gate arm; the paper's own quantity f(p) (eq. 3) and quality g(p) (eq. 4) on the same batches as the trade-off guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: ratio < 1.0 in mean on both the EMA and the trained parameters, by at least one standard error; terminal quantity f(p) above the paired arm's terminal mask rate by at least one standard error; terminal lambda-weighted impurity 1 - g(p) no worse than 1.25x the paired arm's retained-label impurity and below 0.15 absolutely; held-out outcome NLL within 1.05x of the paired arm; terminal mu_t above 1/K by at least one standard error and terminal sigma_t^2 below its 1.0 initialisation
  seeds: 10
  report: mean_and_stderr
```

Three clauses of that tolerance are the point of the card and each can fail:

* **The quantity clause is the paper's central claim in its own units.** For
  the constant-gate arm `lambda` is `1(max(p) >= tau)`, so `f(p)` *is* the mask
  rate and the two arms are compared on one functional (table 1's own
  observation). It is falsifiable and it may well fail: appendix A.1 guarantees
  only `f(p) >= lambda_max / 2`, while `fixmatch.md` §6.3 records a terminal
  mask rate of 0.784 ± 0.016 on this near-separable fixture. A gate can beat
  half.
* **The quality clause is the other half of the trade-off**, and it is the
  clause that stops a win on NLL from being reported as a vindication of the
  method's claim. SoftMatch enrols every row, so some rise in impurity is
  expected; 1.25x is the predeclared bound on "some".
* **The `sigma_t^2` clause is the mechanism guardrail.** `sigma_hat_0^2 = 1.0`
  is enormous next to the range of a confidence, so the weighting function
  starts almost flat and only becomes selective as the variance collapses. A
  run whose variance does not fall has trained an unweighted pseudo-label term
  for 3,000 steps under SoftMatch's name, which is exactly the failure §2's
  first limitation describes.

**This target is written before any of it is run, and must not be retuned.**
`FIDELITY.md` §3 is explicit that a tolerance adjusted after seeing a result is
itself a deviation.

### 6.1 Fixed DGP and the declared arms

**Fixture.** `fixmatch.md` §6.1's DGP in full and without modification, so that
the pair differs in the weighting function and in nothing else — the mechanism,
the seeds, the 64-label MCAR budget, the `B_L = 64` / `B_U = 448` quota, the
outcome standardisation fitted on the training rows, and the replicate seeds
`s_r = 90000 + 100 r` for `r in {0..9}`. Restating it here would be a second
copy to keep in step; `freematch.md` §6.1 takes the same route.

**Arms.**

| Arm | What it is | Tier |
|---|---|---|
| `softmatch` | this card at the §4 declarations | 2, ten seeds |
| `constant` | `fixmatch`'s eq. (4) at `tau = 0.95`, this card's views, same batches | 2, ten seeds, paired |
| `no_ua` | `alignment="none"`, i.e. eq. (5) in place of eq. (9) — the paper's §4.5 `w/o UA` ablation | 1, five seeds (deviation 9) |
| `matched` | the constant gate at the `mu_hat_t` this recipe converges to | 1, five seeds — `freematch.md` §6.4 found this arm recovers 29–56% of the apparent gain there, so it is declared here in advance rather than after the fact |

### 6.2 Predeclared evidence

`BACKLOG.md`'s triage rule 6 asks for all three tiers before implementation.

**Tier 0 — invariants.** Five assertions, all training-free:

1. Eqs. (6) and (7) against hand-computed batch moments and a hand-rolled EMA,
   including the `B_U / (B_U - 1)` correction, over three synthetic steps.
2. **The gate limit.** With Uniform Alignment disabled (or its running marginal
   uniform) and `ConfidenceGaussian.variance` pinned near zero,
   `SoftWeightedTreatmentNLL` equals `PseudoLabelTreatmentNLL` at
   `tau = mu_hat_t` to within floating-point tolerance. Table 1 says confidence
   thresholding is the degenerate case; this asserts it.
3. **The ungated limit.** With the variance pinned large, the same term equals
   an unweighted pseudo-label cross-entropy over all eligible rows — table 1's
   first column.
4. **UA is the identity at initialisation and inert under a uniform marginal**,
   and `alignment="none"` reproduces eq. (5) exactly.
5. **Per-row bounds.** `lambda in (0, 1]` for every row, `lambda = 1` exactly
   for every row with `max(UA(p)) >= mu_hat_t`, and `plan_details()` reports the
   denominator convention and the alignment target so both enter the digest.

Plus the two the family already runs on every card: the declared views compared
against the compiled plan, and the card-key cross-check against
`plan.hyperparameters`.

**Tier 1 — smoke.** A short paired fit on the §6.1 fixture asserting that the
unsupervised term is non-degenerate (terminal `f(p)` strictly inside
`(0.5, 1.0)`), that `mu_hat_t` rises above `1/K` and `sigma_hat_t^2` falls, and
that held-out treatment NLL beats the `constant` arm on the smoke seed. The
`no_ua` and `matched` arms of §6.1 run here, five seeds, reported as directions.

**The five-seed `no_ua` and `matched` study is declared and not yet run.**
`tests/smoke/test_softmatch.py` fits the `softmatch`/`constant` pair on the
smoke seed and asserts only that the other two arms are *expressible* — the
`alignment="none"` policy and the gate matched to the `mu_hat_t` this recipe
reaches. Recorded here rather than left in the gap between a declaration and a
test, because `freematch.md` §6.2's directions are the shape this section is
promising and nothing has produced them yet.

**Tier 2 — the block above**, nightly, ten seeds.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-09-04 | `7e55fdc8e921` | ema_treatment_NLL_ratio<br>trained_treatment_NLL_ratio<br>terminal_quantity_advantage<br>weighted_impurity_vs_1.25x_gate<br>weighted_pseudo_label_impurity<br>held_out_outcome_NLL_ratio<br>terminal_mu_hat<br>terminal_sigma_squared | 0.923445 +/- 0.00715<br>0.937315 +/- 0.0118<br>0.104669 +/- 0.00945<br>0.00881344 +/- 0.00175<br>0.0754915 +/- 0.00248<br>0.999709 +/- 9.5e-05<br>0.916051 +/- 0.00281<br>0.0598245 +/- 0.000441 | no |

## 7. Unknowns

| # | Unspecified in paper | Our choice | Basis |
|---|---|---|---|
| 1 | `lambda_max` is symbolic in eqs. (2), (5) and (9) and never given a value in the text; Algorithm 1 line 9 writes `1.0` and §2.1's objective is `L = L_s + L_u` with no separate `w_u`. | `lambda_max = 1.0`, carried by `losses.weights` rather than by the policy (§3.2). | Algorithm 1 line 9, read as the paper's own instantiation of the symbol. Convention: this also matches FixMatch's and FreeMatch's `w_u = 1`, which §4.1's TorchSSL setup inherits. |
| 2 | §4.1 says "divide the estimated variance `sigma_hat_t` by 4 for `2 sigma` of the Gaussian function", using the standard-deviation symbol with the word "variance". | Divide the **variance**: the exponent's denominator is `2 * sigma_hat_t^2 / n_sigma^2` with `n_sigma = 2`, so a confidence one `sigma_hat_t` below `mu_hat_t` is a two-sigma event for the weight. | Both pinned implementations compute `2 * variance / n_sigma^2`, resolving the wording directly; the paper's appendix A.5 names the same setting `2 sigma`. |
| 3 | Eq. (8) gives neither the momentum nor the initial value of `E_hat[p]`. | Use `m = 0.999` and initialise the running marginal to `u(C)`. Fold the first batch in before its weights are computed. | Table 6 and both implementations use 0.999. TorchSSL initialises the running marginal to uniform and updates it before alignment; USB instead replaces a `None` state with the first batch mean. The classic-image path is the tie-breaker, and uniform is the neutral pre-batch value. |
| 4 | Algorithm 1 omits the `E_hat[p]` update, so its position within a step is unstated. | Update from this batch's weak-view predictions **before** computing eq. (9), i.e. between lines 7 and 8. | Both pinned implementations update the running marginal before aligning and weighting the current batch. |
| 5 | Neither §4.1 nor table 6 says whether the SGD momentum is Nesterov. | Nesterov. | Convention, and consistency: §4.1 states every classic-image experiment was run in TorchSSL, whose shared optimiser is Nesterov SGD, and `fixmatch.md` §7, `flexmatch.md` §7 and `freematch.md` §4 make the same call so that §6's pair differs in the weighting function alone. |
| 6 | Algorithm 1 estimates `mu_b` and `sigma_b^2` from `max(p_i)` (lines 4–5) but compares `max(UA(p_i))` against them (line 9). | Follow Algorithm 1 literally: unaligned confidence for the moments, aligned confidence for the weight. | TorchSSL agrees with the algorithm. USB aligns first and therefore updates the Gaussian from aligned confidence; that later code path conflicts with eqs. (6), (7) and Algorithm 1, so it is not selected. |
| 7 | §2.1 defines `D_L` and `D_U` as separate sets and does not restate FixMatch's footnote 2 about labelled rows also appearing in the unlabelled batch. | `rows = all`: every training row is eligible for eq. (2). | §2.1 states the framework as FixMatch's and UDA's and §4.1 runs it in TorchSSL, so the inclusion comes with it; `fixmatch.md` §3.2, `flexmatch.md` §3.2 and `freematch.md` §3.2 take the same reading for the same term, and §6's pair would not be comparable if this card took a different one. |
| 8 | Eq. (5) and Algorithm 1 line 9 both use a strict `<` for the exponential branch, so the breakpoint itself takes `lambda_max`; nothing in the paper contradicts it. | `<` exactly as written. | The source. Noted only because `fixmatch.md` §7 records the equivalent inconsistency in FixMatch, and its absence here is worth recording as an absence rather than as an oversight. |
| 9 | The paper defines UA's target as `u(C)`, but paper-era TorchSSL aligns to an EMA of predictions on the labelled batch; USB defaults to a fixed uniform target. | Use the paper's fixed `u(C)` target. | Eq. (8), §3.2 and appendix A.5 are explicit, and USB corroborates them. TorchSSL's labelled-prediction target is the paper's ablated `p_hat_L(y)`-style alternative, not Uniform Alignment as published. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | Codex | 2026-09-04 |
| Plan diffed against §3.2 and §4 | Codex | 2026-09-04 |
