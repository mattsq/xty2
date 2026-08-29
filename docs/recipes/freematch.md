# Recipe spec card: freematch

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> Written card-first, before any code, per `CLAUDE.md` rule 1.
>
> This is the card `flexmatch.md` §5.1 named in advance. That card took objective
> state with a stage lifecycle on one consumer and wrote **FreeMatch** into its
> §5.1 as the second, with a specific prediction about the shape. §5.1 below
> records which half of that prediction held and which half was wrong, because a
> named second consumer is only worth naming if somebody comes back and checks.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning](https://arxiv.org/abs/2205.07246) |
| Authors, year | Yidong Wang, Hao Chen, Qiang Heng, Wenxin Hou, Yue Fan, Zhen Wu, Jindong Wang, Marios Savvides, Takahiro Shinozaki, Bhiksha Raj, Bernhard Schölkopf, Xing Xie; 2023 |
| DOI / arXiv | [arXiv:2205.07246](https://arxiv.org/abs/2205.07246); ICLR 2023 |
| Version used | The ar5iv rendering of arXiv:2205.07246, fetched 2026-08-28. The rendering carries no version label and one was **not** verified, so this is recorded as what it is. §2 motivates the threshold; §3 gives the preliminaries and eqs. (3), (4); §4.1 gives SAT and eqs. (5)–(8); §4.2 gives SAF and eqs. (9)–(12); §5.1 the hyperparameters; §5.3 the ablations; Appendix C is Algorithm 1 and Appendix D tables 5 and 6 are the hyperparameter settings. |
| Reference implementation | [`TorchSSL`](https://github.com/TorchSSL/TorchSSL) / `USB`, the authors' own codebases — **not consulted**. This session's GitHub access is scoped to `mattsq/xty2`, and `flexmatch.md` set the precedent of not routing around that scope by a second-hand reading. Every row below is sourced from the paper, and Algorithm 1 is the source wherever the prose leaves a procedural gap. |
| Reference impl. runnable? | Not attempted. |

That row costs this card something specific and §7 says where: eq. (11)'s sign
and the empty-histogram-bin convention of eq. (9) are the two places a reference
would have settled a question the paper leaves open, and both are decided here
on the paper's own prose instead.

## 2. Estimand and claim

- **Estimand:** As `fixmatch` and `flexmatch`: the categorical propensity
  `p(t | x)`, composed with the retained causal stack — the treatment-conditional
  outcome distribution `p(y | x, t=k)` and its means `mu_k(x)`, whose contrasts
  identify conditional treatment effects under consistency, positivity and
  conditional exchangeability.
- **Claim:** FreeMatch replaces the hand-set threshold with two statistics of
  the model's own predictions on unlabelled data. A **global** threshold `tau_t`
  is the EMA of the mean top-class confidence (eq. 5); a **local** vector
  `p~_t` is the EMA of the mean predicted class distribution (eq. 6); the gate
  is their product under maximum normalisation, `tau_t(c) = MaxNorm(p~_t(c)) *
  tau_t` (eq. 7). A second term, self-adaptive fairness, pushes the marginal of
  the model's predictions on the retained rows towards its own running marginal
  after both are normalised by the histogram of predicted labels (eqs. 9–11).
  The paper claims state-of-the-art error rates across CIFAR-10/100, SVHN,
  STL-10 and ImageNet, with the largest margins in the barely-supervised
  settings, and faster convergence than FixMatch and FlexMatch (§5.2).

  This card claims that the mechanism is faithfully assembled around `p(t | x)`
  in xty2 — Tier 0 asserts eq. (7)'s arithmetic against a hand-computed EMA, and
  asserts that a state pinned at `p~ = uniform, tau = tau*` reduces eq. (8) to
  `PseudoLabelTreatmentNLL` at `tau*` exactly — that `tau_t` and `tau_t(c)` move
  over training the way §4.1 prescribes, and that on the fixed project-local
  target in §6 it improves held-out treatment prediction against an otherwise
  identical constant-gate arm without damaging the outcome stack.

  **What §6.2 measures of that is stated there and nowhere rounded up.** It
  reports **all six** clauses of §6's tolerance, met, at **five** of the
  declared ten seeds, from a script rather than a Tier 2 runner — so the claim
  above is supported on this fixture at half the declared seed count and
  `reproduced` is not available to it under `FIDELITY.md` §1.1 (§6.3).

  Two things the numbers do **not** support, stated here rather than left in
  §6.2 for a reader who stops early. The gain is **not** attributed to
  self-adaptivity as against the lower threshold it produces — the terminal
  `tau_t` is 0.922 where the comparison arm is fixed at 0.95, and a constant
  gate at 0.92 was not run. And **self-adaptive fairness contributes nothing
  measurable here**: the paper's own `w_f = 0` ablation is within 0.0001 ±
  0.0012 of the full recipe, because on a near-balanced two-class fixture the
  term sits at its own floor (§2's third limitation, measured in §6.2).
- **Not claimed:** No image number is claimed. Five limitations are structural
  and are stated here rather than left to be discovered:

  1. **The gate is open on the whole batch at step 0, and with `K = 2` it is
     open on the whole batch by arithmetic rather than by accident.** Eq. (5)
     and eq. (6) initialise `tau_0 = 1/C` and `p~_0(c) = 1/C`, so
     `MaxNorm(p~_0) = 1` and `tau_0(c) = 1/C = 0.5`. A two-class softmax has
     `max(q_b) >= 0.5` for every row, so eq. (8)'s indicator `max(q_b) >
     tau_0(argmax q_b)` fails only at an exact tie. FreeMatch therefore opens by
     training `p(t | x)` on the hard arg max of an untrained network over the
     entire batch — the same ungated phase `flexmatch.md` §2 records, reached by
     a different route and, at `K = 2`, reached certainly rather than probably.

     `BACKLOG.md` §2.5 gives the instruction this card is following: "Any
     descendant that opens the gate early — SoftMatch's continuous weights at
     low confidence, FreeMatch's self-adaptive `tau` in its own warm-up, Dash's
     decaying threshold — is exposed to the same trap and should run that check
     first." The check is `flexmatch.md` §5.2's, it is training-free, and
     deviation 2 is what it selects.
  2. **A self-adaptive threshold is self-referential, and that is a sharper
     version of the tension `fixmatch.md` §2 recorded.** `tau_t` is the EMA of
     the model's own confidence, so a model that becomes confident past what the
     assignment supports raises its own bar and then meets it. Under genuine
     overlap `fixmatch.md` §6.2 found the gate manufactures the confidence it is
     gated on; here the threshold is *defined* as that confidence. §6's DGP
     lives in the near-deterministic-assignment corner where the confidence is
     warranted, and this card claims nothing outside it.
  3. **`K = 2` leaves SAT's local half with something to do and SAF's with
     almost nothing.** This differs from `flexmatch.md` §2's third limitation
     and the difference is worth being precise about. FlexMatch's `beta(c)`
     normalises by `max_c sigma`, so at `K = 2` both classes reach 1 together
     and the per-class threshold collapses to `tau`; FreeMatch's `MaxNorm(p~)`
     puts the *larger* class at 1 and the smaller strictly below it whenever
     `p~` is not exactly uniform, which it never is after the first step. So the
     local threshold is live at `K = 2`. SAF is the opposite: on a fixture whose
     treatment marginal is already near-uniform, a term that pushes the
     predicted marginal towards uniformity has little to move, and §6 declares
     the `w_f = 0` ablation the paper itself runs (§5.3) so that the question is
     answered with a number.
  4. **Eq. (11) is implemented against the paper's stated purpose rather than
     against its sign, and that is deviation 7.** As written, `L_f = -H(A, B)`
     minimised drives `B` away from `A` and towards a corner of the simplex —
     the opposite of the "diverse predictions" §1, §2, §4.2 and §6 all say the
     term is for. §5's deviation 7 sets out the derivative that settles it and
     §7 records that the reference implementation, which would have decided the
     question directly, was not consulted. **The literal reading is expressible
     in this recipe's own declared fields** — `w_f = -0.05` negates the term —
     so the deviation is cheap to test rather than merely argued.
  5. **A pseudo-label on `t` is not a label on `y`.** As in `fixmatch` and
     `flexmatch`: pseudo-labelled rows train `p(t | x)` only, never
     `ObservedOutcomeNLL`, so no inferred treatment is used as if observed and
     the `DESIGN.md` §7.2 leakage rule is not engaged.

## 3. Equations and mapping

### 3.1 As published

Notation is §3's, which is FixMatch's and UDA's. `D_L` and `D_U` are the
labelled and unlabelled sets; `B` is the labelled batch size and `mu` the ratio
of unlabelled to labelled; `omega(.)` is weak and `Omega(.)` strong
augmentation; `H(.,.)` is cross-entropy; `q_b := p_m(y | omega(u_b))` and
`Q_b := p_m(y | Omega(u_b))`, with `\hat q_b` and `\hat Q_b` their hard one-hot
arg maxes; `C` is the number of classes.

Section 3, the supervised term and the thresholded unsupervised term FreeMatch
inherits:

$$
\mathcal{L}_{s} = \frac{1}{B}\sum_{b=1}^{B} \mathcal{H}\!\left(y_b,\, p_m(y \mid \omega(x_b))\right)
\tag{3}
$$

$$
\mathcal{L}_{u} = \frac{1}{\mu B}\sum_{b=1}^{\mu B}
  \mathbb{1}(\max(q_b) > \tau)\cdot \mathcal{H}(\hat q_b,\, Q_b)
\tag{4}
$$

Section 4.1, self-adaptive thresholding. The global threshold, the local
per-class estimate, their combination, and the unsupervised objective they gate:

$$
\tau_t = \begin{cases}\frac{1}{C}, & t = 0\\
\lambda \tau_{t-1} + (1-\lambda)\frac{1}{\mu B}\sum_{b=1}^{\mu B}\max(q_b), & \text{otherwise}\end{cases}
\tag{5}
$$

$$
\tilde p_t(c) = \begin{cases}\frac{1}{C}, & t = 0\\
\lambda \tilde p_{t-1}(c) + (1-\lambda)\frac{1}{\mu B}\sum_{b=1}^{\mu B} q_b(c), & \text{otherwise}\end{cases}
\tag{6}
$$

$$
\tau_t(c) = \operatorname{MaxNorm}(\tilde p_t(c))\cdot \tau_t
  = \frac{\tilde p_t(c)}{\max\{\tilde p_t(c) : c \in [C]\}}\cdot \tau_t
\tag{7}
$$

$$
\mathcal{L}_{u} = \frac{1}{\mu B}\sum_{b=1}^{\mu B}
  \mathbb{1}\!\left(\max(q_b) > \tau_t(\arg\max(q_b))\right)\cdot \mathcal{H}(\hat q_b,\, Q_b)
\tag{8}
$$

Section 4.2, self-adaptive fairness. The batch quantities, the EMA of the
predicted-label histogram, and the objective:

$$
\overline p = \frac{1}{\mu B}\sum_{b=1}^{\mu B}\mathbb{1}\!\left(\max(q_b) \ge \tau_t(\arg\max(q_b))\right) Q_b,
\qquad
\overline h = \operatorname{Hist}_{\mu B}\!\left(\mathbb{1}\!\left(\max(q_b) \ge \tau_t(\arg\max(q_b))\right)\hat Q_b\right)
\tag{9}
$$

$$
\tilde h_t = \lambda \tilde h_{t-1} + (1-\lambda)\operatorname{Hist}_{\mu B}(\hat q_b)
\tag{10}
$$

$$
\mathcal{L}_{f} = -\mathcal{H}\!\left(\operatorname{SumNorm}\!\left(\frac{\tilde p_t}{\tilde h_t}\right),\;
                                      \operatorname{SumNorm}\!\left(\frac{\overline p}{\overline h}\right)\right),
\qquad \operatorname{SumNorm}(\cdot) = \frac{(\cdot)}{\sum(\cdot)}
\tag{11}
$$

$$
\mathcal{L} = \mathcal{L}_{s} + w_u \mathcal{L}_{u} + w_f \mathcal{L}_{f}
\tag{12}
$$

Algorithm 1 (Appendix C) is the procedure, and its ordering is load-bearing for
this port:

```text
 1  Input: C, labelled batch X, unlabelled batch U, w_u, w_f, EMA decay lambda
 2    Compute L_s for labelled data                            (eq. 3)
 3    Update the global threshold  tau_t   from THIS batch's q_b     (eq. 5)
 4    Update the local threshold   p~_t    from THIS batch's q_b     (eq. 6)
 5    Update the histogram         h~_t    from THIS batch's \hat q_b (eq. 10)
 6  for c = 1 to C do
 7      tau_t(c) = MaxNorm(p~_t(c)) * tau_t                     (eq. 7)
 8  end for
 9    Compute L_u  with the tau_t(c) just computed              (eq. 8)
10    Compute \bar p over the retained rows                     (eq. 9)
11    Compute \bar h over the retained rows                     (eq. 9)
12    Compute L_f                                               (eq. 11)
13  Return: L_s + w_u * L_u + w_f * L_f                         (eq. 12)
```

Three readings the mapping depends on:

* **The statistics are updated from the current batch *before* that batch is
  gated.** Lines 3–5 precede line 7, which precedes line 9. This is the
  opposite of FlexMatch, whose Algorithm 1 computes every threshold from marks
  laid down by earlier steps and writes this step's marks afterwards. The
  consequence is not cosmetic: a row's gate depends on the confidences of the
  other rows in its own batch, so both FreeMatch terms are `batch_coupled`
  where `CurriculumPseudoLabelTreatmentNLL` is not. `flexmatch.md` §5.1
  predicted the opposite and §5.1 below records the correction.
* **`L_u` and `L_f` are two terms over one set of statistics.** `tau_t(c)`
  appears in eq. (8) and again in eq. (9)'s indicator; `p~_t` appears in eq. (7)
  and again in eq. (11); `h~_t` appears only in eq. (11) but is an EMA over the
  same `q_b`. Eq. (12) gives them separate weights, so they are two objectives,
  and §3.2 says how one set of statistics reaches both.
* **Eq. (8) writes `>` and Algorithm 1 line 9 writes `>=`; eq. (9) writes
  `>=`.** §7 records the reading and it differs from the objective's gate in a
  set of measure zero.

### 3.2 Mapping to xty2

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `p_m(y \| x)` | model's class distribution | `T_GIVEN_X` | `CategoricalPropensity` over `MLPEncoder` |
| `omega(.)` | weak augmentation | — | `ViewSpec("weak_x")`, `FeatureMask(p=0.1)`, two draws |
| `Omega(.)` | strong augmentation | — | `ViewSpec("strong_x")`, `FeatureMask(p=0.1)` then `FeatureMask(p=0.2)` — `flexmatch`'s, not `fixmatch`'s; deviation 2 |
| eq. (3) `L_s` | supervised cross-entropy on weak views | `T_GIVEN_X @ weak_x draw=1` | `ObservedTreatmentNLL(realisation=Realisation("weak_x", draw=1))`, rows `t_observed`, `reduction="mean"` |
| `q_b` | artificial label distribution | `T_GIVEN_X @ weak_x draw=0` | `SelfAdaptiveThresholdTreatmentNLL.target`, and `SelfAdaptiveFairness.target` |
| `\hat q_b` | hard pseudo-label | — | `sharpening="hard"` inside that objective |
| `Q_b` | strong-view prediction | `T_GIVEN_X @ strong_x` | `.prediction` on both objectives |
| eq. (5) `tau_t` | global threshold, EMA of mean confidence | — | `SelfAdaptiveThresholds.tau`, the objective's per-stage state |
| eq. (6) `p~_t` | local per-class estimate, EMA of mean probability | — | `SelfAdaptiveThresholds.marginal` |
| eq. (10) `h~_t` | EMA of the weak-view predicted-label histogram | — | `SelfAdaptiveThresholds.histogram` |
| `lambda` | the decay of eqs. (5), (6) and (10) | — | `SelfAdaptiveThreshold(decay=0.999)`, bound to `losses.confidence_threshold` |
| eq. (7) `tau_t(c)` | the gate | — | `SelfAdaptiveThresholds.thresholds()`, `[K]`; logged as `threshold_min` / `threshold_max`, with `tau_global` beside them |
| eq. (8) `L_u` | self-adaptively gated pseudo-label cross-entropy | `T_GIVEN_X` at both views | `SelfAdaptiveThresholdTreatmentNLL`, rows `all`, `reduction="mean"` |
| eq. (9) `\bar p`, `\bar h` | retained-row mean probability and label histogram, strong view | `T_GIVEN_X @ strong_x` | computed inside `SelfAdaptiveFairness` |
| eq. (11) `L_f` | the fairness term | `T_GIVEN_X` at both views | `SelfAdaptiveFairness`, rows `all`, `reduction="mean"`; the sign is deviation 7 |
| eq. (12) `w_u` | unsupervised weight | — | `Weighted(..., weight=1.0)`, `Constant` |
| eq. (12) `w_f` | fairness weight | — | `Weighted(..., weight=0.05)`, `Constant` |
| `mu`, `B` | batch composition | — | `QuotaSampler(Quota("t_observed", 64), Quota("t_missing", 448))`, imported from `fixmatch` |
| `eta_0 cos(7 pi k / 16 K)` | rate schedule (§5.1) | — | `CosineDecay(steps=3000, phase=7/16)` |
| EMA of parameters | the model §5.1 evaluates | — | `TeacherSpec(decay=0.999, role="evaluation")`; no objective reads it |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

Four mapping decisions carry the fidelity of this port.

**One state, two objectives, and the reader names the owner.** Eq. (12) gives
`L_u` and `L_f` separate weights, and `losses.weights` is per objective, so
folding them into one term would hide `w_f` from the plan and from the
per-objective log. They are therefore two `Weighted` objectives — and they share
one `SelfAdaptiveThresholds`, because `tau_t`, `p~_t` and `h~_t` are one set of
statistics that eqs. (7), (8), (9) and (11) all read. The threshold objective
declares `initial_state`; the fairness objective declares a field naming it and
reads that state through `TrainContext.objective_state`. `DESIGN.md` §4 gains
one sentence for this and no code (§5.1).

The threshold objective also carries `num_treatments`, which is `C` and is not
a card key: it is a property of the schema, a component takes it the same way
(`CategoricalPropensity`), and the state is `[C]` wide and is built before the
first batch exists. `compute` checks it against `ctx.schema` for the reason
`DESIGN.md` §3.1 gives — a term that read `C` off the head's own output would
agree with a head that had the wrong `C`.

**The shared update is idempotent within a step, so declaration order cannot
change a number.** Algorithm 1 lines 3–5 run once per iteration. Whichever of
the two objectives the mixer computes first performs that update; the second
finds the step already recorded and reads the same `tau_t`, `p~_t` and `h~_t`.
Tier 0 asserts it by swapping the two objectives in the stage and comparing the
loss to the last bit. Without that property the recipe would carry an invisible
dependency on the order two lines appear in, which is precisely the "logic in
the recipe" `CLAUDE.md` rule 3 exists to forbid.

Idempotence only *makes* the order inert if the two terms are entitled to the
same rows, since otherwise the first to run would decide which set the averages
were taken over. Both declare `all`, for the footnote-2 reason below; and
`SelfAdaptiveThresholds.observe` refuses a repeat at one step whose row count
differs, rather than silently ignoring it. Equal counts over different sets
would still pass, so the declaration is the guard and the check is the cheap
half of it.

**`U` is every training row, so eq. (8)'s rows are `all` and so are eq. (11)'s.**
FreeMatch §3 restates FixMatch's and UDA's framework and §5.1 adopts their
settings, so FixMatch's footnote 2 — every labelled row is also in `U`, without
its label — is inherited along with it. This is the reading `fixmatch.md` §3.2
and `flexmatch.md` §3.2 already took for the same term. Unlike FlexMatch, no `N`
is counted: eqs. (5), (6) and (10) are averages over the batch, so the state
needs no population at all.

**The denominator counts the rows it rejects.** Eq. (8) divides by `mu B`, as
eqs. (3) and (4) do, so a batch in which nothing clears its class threshold
contributes zero rather than an average over an empty set. Identical arithmetic
to `fixmatch` and `flexmatch`: a 0/1 mask, then a mean over every eligible row.
Eq. (9)'s two quantities divide by `mu B` as well, and since eq. (11) takes
their ratio the factor cancels — which is why the convention that *does* matter
there is the empty-bin one, not the denominator (§7).

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.self_adaptive_threshold_treatment_nll: p(t|x) @ view=weak_x params=student
    joint_fit.self_adaptive_fairness: p(t|x) @ view=weak_x params=student
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: target                    # eqs. (8) and (11) both read q_b only through an arg max, a step function and an EMA buffer; nothing on that side carries a gradient
  gradient_clipping: none                     # the paper names none; retained P5 choice
  marginal_nll_grad_path: both                # reviewed P5 choice; project-local addition

teacher:
  ema_decay: 0.999                            # §5.1: "an exponential moving average with the momentum of 0.999 of the training model to conduct inference"
  ema_applies_to_buffers: false               # the declared graph has no buffers; stated so a component that grew one would be a card change
  teacher_in_train_mode: false                # the EMA copy is an evaluation classifier
  teacher_requires_grad: false                # never an optimiser target
  # role = evaluation. Nothing reads this EMA during training: eqs. (5), (6),
  # (8), (10) and (11) are all statements about the current network.

losses:
  reduction:
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: mean     # eq. (3) divides by B
    joint_fit.self_adaptive_threshold_treatment_nll: mean   # eq. (8) divides by mu*B
    joint_fit.self_adaptive_fairness: mean     # eq. (11) is one scalar per batch; see below
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.self_adaptive_threshold_treatment_nll: all    # FixMatch footnote 2, inherited: U is every row
    joint_fit.self_adaptive_fairness: all
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.self_adaptive_threshold_treatment_nll: 1.0    # w_u = 1, §5.1 and table 5
    joint_fit.self_adaptive_fairness: 0.05                  # w_f, table 5's "other settings" value; §7
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.self_adaptive_threshold_treatment_nll: constant 1.0   # the adaptation is in the threshold, not in w_u
    joint_fit.self_adaptive_fairness: constant 0.05
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature: n/a                            # FreeMatch adjusts a threshold; it never sharpens a soft target
  sharpening: hard                            # eq. (8): H(\hat q_b, .) with \hat q_b = arg max(q_b)
  confidence_threshold: self_adaptive(decay=0.999)   # eqs. (5)-(7); tau_0 = p~_0(c) = h~_0(c) = 1/K and lambda = 0.999 from table 5

optimisation:
  optimiser: sgd(momentum=0.9, nesterov=True)    # §5.1: "SGD with a momentum of 0.9"; Nesterov is FixMatch's, which TorchSSL runs for every algorithm in the comparison (§7)
  lr: 0.03                                       # §5.1 and table 6
  lr_schedule: cosine 1.0 * cos(pi * 0.4375 * min(step/3000, 1))  # §5.1's eta_0 cos(7 pi k / 16 K); K = our 3000 steps
  weight_decay: 0.0005 (all trainable components; norm and bias exempt)  # table 6's CIFAR-10 value; scope follows fixmatch.md deviation 8
  batch_size: 512                                # B + mu B = 64 + 448, derived from the QuotaSampler's quotas
  labelled_unlabelled_ratio: 7.0                 # mu, table 5; derived from the same quotas
  total_steps_or_epochs: 3000                    # optimiser steps, never epochs. The paper's total is 2^20; see deviation 3

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
    categorical_propensity: K softmax logits
data:
  standardisation: x: none fitted on 'train'
  outcome_scaling: y: zscore fitted on 'train'
  treatment_encoding: n/a                       # XTYBatch supplies integer classes 0..K-1
  split_protocol: one fixed project-local DGP, split train/test by the section 6 fixture; no CIFAR/SVHN/STL protocol applies (deviation 1); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 64 labelled rows, keyed by row_id  # deviation 9
```

`losses.confidence_threshold` holds a **policy object**, exactly as
`flexmatch.md` §4 argues and for the same reason: `card_keys.py` refuses two
fields bound to one canonical key and instructs the author to bind one field
holding the whole rule. FreeMatch's gate is a rule with no scalar in it at all —
`tau_t(c)` is a function of the training history and the only number a recipe
sets is `lambda` — so a `float` field here would have nothing to hold. This is
the second card to read the key that way, which is the evidence `flexmatch.md`
§4's reading was a reading of the vocabulary and not a way around it.

**`losses.reduction: mean` for the fairness term is a statement, not a
default.** Eq. (11) is one scalar per batch and has no per-row decomposition:
`\bar p` and `\bar h` are batch aggregates. The objective therefore returns that
scalar with `n = |eligible rows|`, and `mean` is the mode under which a term's
value enters the total unscaled (`DESIGN.md` §6.1) — which is what eq. (12)'s
`w_f L_f` says. `sum` or `population` would multiply a batch-level quantity by a
row count.

The **weak** view is `fixmatch`'s, unchanged and imported: `FeatureMask(p=0.1)`
with two draws. The **strong** one is `flexmatch`'s: `FeatureMask(p=0.1)` then
`FeatureMask(p=0.2)`, imported from that recipe so the reviewed value has one
home. Deviation 2 is why it is that one and not `fixmatch`'s 0.5. Both views
preserve `t`, `y`, `t_observed`, `y_observed`, `row_id`, `fold_id` and `weight`,
and `freematch(schema, recompute_rules=(...))` passes the same explicit rules to
both, as `fixmatch` and `flexmatch` do.

§4's YAML has no key for a view, so nothing above is covered by the card-key
cross-check; `tests/invariants/test_freematch.py` compares this paragraph's two
transforms against the compiled plan instead, because this is the line that went
stale on `flexmatch` once already.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Apply FreeMatch to categorical treatment assignment `p(t \| x)` and compose it with a causal outcome likelihood (`ObservedOutcomeNLL`) and exact marginalisation (`MissingTreatmentMarginalNLL`). | The paper studies image classes. The project-local question is whether a self-adaptive threshold recovers a missing *treatment* label better than a fixed one, and whether it composes with the reviewed P5 stack rather than replacing it. | No comparison to a published image error rate is valid. The marginal term trains `p(t \| x)` on exactly the rows the threshold is deciding about, so the two mechanisms interact; §6 measures the pair against a constant gate, which carries the same interaction. |
| 2 | `judgement` | — | Replace crop-and-flip (weak) and RandAugment (strong) with schema-aware feature masking: 10% weak, and 10% followed by 20% strong — `flexmatch`'s strong view, not `fixmatch`'s 50%. | There is no image structure in a tabular XTY batch. FixMatch §2.3 asks a strong augmentation to be severe *and* label-preserving, and `flexmatch.md` §5.2 measures a tabular analogue of the second half on this exact DGP: at an effective 0.55 the view flips the Bayes-optimal label on 16.8% of rows, at 0.28 on 7.4%. §2's first limitation is why this card cannot inherit the 0.55 the way `fixmatch` could: `tau_0(c) = 1/K = 0.5` makes eq. (8) ungated on the whole batch at `K = 2`, which is the configuration `flexmatch.md` §6.2 measured locking on three initialisation seeds of five. | Directly defines the invariance being learned. §6's pair holds it fixed on both arms, so it bounds what the numbers describe rather than confounding what they compare. Adopting `fixmatch`'s 0.5 instead is predicted to lock the run at a chance-level propensity on some seeds; §6.2 records whether it did. |
| 3 | `judgement` | — | Train for 3,000 optimiser steps rather than the paper's `2^20`. | The reviewed project-local budget, shared with every other xty2 recipe so that a difference is attributable to the recipe. The cosine schedule's `K` is set to the same 3,000, so the shape of the decay is exact even though its length is not. | The paper's convergence claim (§5.2) is that FreeMatch reaches a better result *sooner*, so a short budget is where SAT's early behaviour matters most. §6 records the trajectory rather than the endpoint alone. |
| 4 | `judgement` | — | Retain the P5 TARNet architecture (encoder, outcome head, propensity) rather than Wide ResNet-28-2. | Holding the causal stack fixed is what makes the SAT/SAF addition attributable, and it is `fixmatch.md` deviation 6, `flexmatch.md` deviation 4 and `mean_teacher.md` deviation 10. | The project-local result validates wiring and mechanism, not image-scale accuracy. |
| 5 | `judgement` | — | Retain P5's `Ramp(0.0, 0.5, 1000)` on the marginal-likelihood term while both FreeMatch weights stay constant. | The ramp belongs to the reviewed P5 term, not to FreeMatch; eq. (12) states fixed `w_u` and `w_f`. | Identical to `fixmatch`'s and `flexmatch`'s arrangement, so the pair in §6 shares it. |
| 6 | `judgement` | — | Do not adopt the two SVHN-specific techniques of §5.1: a two-epoch warm-up on labelled data only, and clamping `tau_t` to `[0.9, 0.95]`. | §5.1 introduces both for SVHN alone, as a fix for a dataset where "using a low threshold at early training stage impedes the model to cluster the unlabeled data". They are not part of the method as stated in §4, and adopting the clamp in particular would replace the mechanism this card is porting with a hand-set threshold wearing its name. | Leaves §2's first limitation live rather than patched. If the ungated warm-up turns out to be what hurts on this fixture, the clamp is the paper's own answer and a card amendment, not a silent addition. |
| 7 | `judgement` | — | Implement eq. (11) **without its leading minus**: `L_f = H(SumNorm(p~/h~), SumNorm(\bar p/\bar h))`, minimised. | The sign as written contradicts the purpose the paper states for the term in four places (§1 "encourage the model for diverse predictions", §2's summary, §4.2 "encourage the model to make diverse predictions for each class", §6's related work on entropy maximisation inducing fairness). The arithmetic settles it: with `A` detached and `B` on the simplex, `d/dB_1 [sum_c A_c log B_c]` vanishes at `B = A` and points *away* from it either side, so minimising `-H(A, B)` drives `B` to a corner — entropy minimisation of the marginal. Minimising `+H(A, B)` has `B = A` as its minimum, which is what §4.2's sentence "encourages the expectation of the output probability for each mini-batch to be close to a marginal class distribution of the model" describes. §7 records that a reference implementation would have settled this directly and was not consulted. | Bounded either way at this weight, but opposite in direction. **The literal reading needs no code**: `w_f = -0.05` on this objective is eq. (11) exactly as printed, so §6 declares it as a third arm rather than arguing about it. |
| 8 | `framework-limitation` | `augmentation-vocabulary` | No adaptive augmentation and no RandAugment: the strong view's strength is a fixed scalar. | The argument is `fixmatch.md` deviation 10's and is not restated. FreeMatch adds nothing to the case either way — its contribution is the threshold and the fairness term, not the augmentation — and its §5.1 runs the same RandAugment the earlier cards already deviate from. | Removes whatever augmentation diversity buys, equally from both arms of §6's pair, so it is a limit on what the numbers describe rather than a confound within them. |
| 9 | `framework-limitation` | `batch-row-repetition` | Set the §6 label budget to 64 rather than a scarcer regime, holding `B = 64` and `mu = 7` at the paper's values. | `XTYBatch.row_id` must be unique (`DESIGN.md` §7.1), so a labelled quota of `B` cannot be drawn from a population smaller than `B` without repeating a row, and the scarcest budget expressible is `B` itself. The alternative — lowering `B` — would deviate from a number the paper states. | This is the deviation that costs this card the most. §5.2's largest margins, and the setting §4.2 says SAF exists for ("especially under the settings where labeled data are rare"), are the 10-label and 40-label CIFAR-10 regimes. At 64 labels over `K = 2` the recipe is nowhere near barely supervised, so SAF is being measured outside the regime it was designed for. It moves both arms of the pair equally. |
| 10 | `judgement` | — | Record §6.4's `K`-sweep and skewed fixtures as a diagnostic measurement rather than as a second Tier 2 target or a second Tier 1 arm. | They exist to exercise the half of FreeMatch the primary fixture leaves inert — §2's third limitation — and the same reasoning `flexmatch.md` deviation 9 applies to its imbalanced probe applies here: making them reproduction targets would multiply the nightly cost of a card whose declared claim is the paired one, and making them Tier 1 arms would add minutes of CI on every PR for a number `FIDELITY.md` §3 says is not a result anyway. | None on the §6 metric. What changes is what §6.4 is allowed to claim: a direction on five seeds, not a target. |

### 5.1 Framework additions made for this card

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `SelfAdaptiveThreshold` (the policy), `SelfAdaptiveThresholds` (the state), `SelfAdaptiveThresholdTreatmentNLL` (eq. 8) and `SelfAdaptiveFairness` (eq. 11), in `xty2/objectives/adaptive_threshold.py` | fidelity-bearing, reversible | This card | — (not the load-bearing quadrant; nothing outside this recipe is written against them) | An objective is the ordinary extension point `BACKLOG.md` step 4 names. They are **separate** objectives rather than a policy union on `PseudoLabelTreatmentNLL` for the reason `BACKLOG.md` §15.2 gives and `DESIGN.md` §4.2 has already applied once: keep each mechanism local and explicit until a third recipe shows the shape. §5.3 below is the review of whether that third recipe has now arrived |
| **One sentence of `DESIGN.md` §4: an objective may read the state of a *named sibling* in the same stage.** No code | fidelity-bearing, **load-bearing vocabulary** (§4 is the contract every objective is written against) | This card | **SimMatch** (`BACKLOG.md` §2.9), whose "labeled feature and label memory banks" are written once per step and read by *both* its instance-similarity and its semantic-similarity terms, which §2.9 says have separate weights — the same shape as `tau_t`, `p~_t` and `h~_t` read by eq. (8) and eq. (11). **CoMatch** (§2.7) is the third: its memory-smoothed pseudo-label graph is one structure consumed by the pseudo-label branch and the contrastive branch. **The shape was checked against SimMatch in the one place it could have gone wrong**: a bank that two terms read must not depend on which of them the mixer reaches first, so the contract requires the shared update to be **idempotent within a step** rather than requiring a declaration order. `SelfAdaptiveThresholds.observe` records the step it last folded in and returns early on a repeat, and Tier 0 asserts the loss is bit-identical under both declaration orders | Eq. (12) gives `L_u` and `L_f` separate weights and `losses.weights` is per objective, so one merged objective would drop `w_f` from the plan, from the per-objective log and from the card-key cross-check — a `framework-limitation` row against a key §4 names, which is exactly the `DESIGN.md` §11.2 Q1 trigger. The alternative that keeps two objectives without shared state is for `SelfAdaptiveFairness` to carry its own `lambda` and its own EMAs, which duplicates eqs. (5), (6) and (10) and binds `losses.confidence_threshold` twice |

**No code was added to `core/`.** `TrainContext.objective_state(name, kind)`
already takes the name as an argument and already type-checks the result, so a
sibling read is a use of the existing accessor. What changed is its docstring
and `DESIGN.md` §4, because the contract *said* "this objective's own state" and
a capability nobody wrote down is one the next agent has to rediscover. Nothing
existing gains a code path: no plan, digest or recorded result moves, and Tier 0
checks that on every shipped recipe.

**What `flexmatch.md` §5.1 predicted, and what it got wrong.** That card named
FreeMatch as the second consumer of `StatefulObjective` and made two claims
about the shape. The first held exactly: "FreeMatch's state is an EMA over
batches and needs no row table and no `N`, so a hook shaped
`initial_state(population: TrainingPopulation)` — requiring a population — would
have forced it to accept one it does not use". `SelfAdaptiveThresholds` is built
from `K` alone and ignores the population argument, so the
`TrainingPopulation | None` signature is load-bearing on the first day it had a
second consumer.

The second claim was wrong. It said FreeMatch's state is "updated from each
batch *after* that batch's gate is applied". Algorithm 1 lines 3–5 update
*before* line 7 computes the gate and line 9 applies it, so a row's threshold
depends on the confidences of the other rows in its own batch. Both FreeMatch
objectives therefore declare `batch_coupled = True`, where
`CurriculumPseudoLabelTreatmentNLL` declares `False`.

The repository contained both answers at once. That objective's own docstring
said "A threshold computed from *this* batch's confidences — FreeMatch's
self-adaptive one — would answer true", which is right; `flexmatch.md` §5.1,
written in the same packet, said the update comes after the gate, which implies
false. Nobody reconciled them because nothing forced it: a prediction about a
card that does not exist yet costs nothing to get wrong. This is what the
§11.2 obligation buys — not that the guess is right, but that somebody has to
come back and score it. The consequence is
real and it is enforced: a stage holding either objective may not declare
`ExternalBatches` (`DESIGN.md` §7), so this recipe could not hand `mu B` back to
a caller even if a later edit tried.

Nothing about the lifecycle needed to change to accommodate that, which is the
mildly reassuring half: a second consumer found the prediction about *ordering*
wrong and the prediction about *shape* right, and the shape is the part §11.2
asks a card to check.

### 5.2 Two things this card was expected to need and did not

* **A new card key.** `losses.confidence_threshold` takes the policy object,
  for the reason §4 gives. The vocabulary in `FIDELITY.md` §2 is unchanged, and
  this is now the second recipe to bind a rule rather than a number to it.
* **A stateful sampler.** The `stateful-sampler` ledger row exists because a
  sampler that reads the model would destroy the property every paired ablation
  here rests on. SAT reads the model and changes which rows *train*, not which
  rows are *drawn*, so §6's pair still shares a batch stream exactly.

### 5.3 Is it time for a shared thresholding abstraction?

`BACKLOG.md` §15.2 asks this question directly and asks for it to be answered
with evidence rather than with aesthetics: "Only consider a reusable
mediator/policy abstraction if at least two real recipes need the same lifecycle
and semantics. The evidence should be duplicated or unreviewable
implementation, not aesthetic similarity."

There are now three gated pseudo-label objectives — `PseudoLabelTreatmentNLL`,
`CurriculumPseudoLabelTreatmentNLL` and `SelfAdaptiveThresholdTreatmentNLL` —
and they share the arg-max, the mask, the `-log p` and the mean-over-eligible
rows. That is real duplication and it is the third instance, so the answer is
not automatically "wait" any more. It is still **wait**, for a reason specific
to what the three differ in rather than to what they share:

* their **gates** are three different functions of three different things (a
  constant; a count over per-row marks accumulated across steps; two EMAs over
  batch statistics), and
* their **coupling** differs: two are `batch_coupled = False` and this one is
  `True`, which is a compile-time property the framework acts on.

A union type over those three gates would put a `bool` that the compiler reads
behind a policy field, so the *declaration* the compiler needs would depend on
the *value* a recipe passed. That is a worse shape than the duplication, and the
recipe that would settle it is one whose gate is a fourth kind — SoftMatch's
continuous weight, which is not a gate at all but a per-row multiplier, and
which `BACKLOG.md` §2.5 lists next. Recorded here so the next card inherits the
question rather than the conclusion.

## 6. Reproduction target

The published CIFAR-10/100, SVHN, STL-10 and ImageNet error rates cannot
validate this port: the inputs, the labels, the architecture and the metric all
differ, and the estimand is a treatment assignment rather than an image class.
The target below is a fixed project-local *mechanism* target in the form
`fixmatch.md` §6 and `flexmatch.md` §6 use.

**This result will be measured without two mechanics the paper's framework
states**, and `DESIGN.md` §11.6 wants that beside the number rather than in the
gap between sections: §5.8 (`augmentation-vocabulary`) fixes the strong view's
strength where the reference runs RandAugment, and §5.9
(`batch-row-repetition`) sets the label budget at 64 rather than in the
barely-supervised regime where §5.2 reports FreeMatch's largest margins and
where §4.2 says SAF earns its keep. Both apply equally to every arm, so they
limit what the numbers describe rather than confound what they compare.

```yaml
reproduction:
  dataset: fixmatch.md §6.1's project-local seed-locked two-cluster XTY DGP (6 features, K=2), unmodified
  variant: paired fit against a constant-gate arm — this recipe with eq. (8) replaced by FixMatch's eq. (4) at tau = 0.95 and the fairness term removed, so both arms share deviation 2's views — same seeds and same batches
  split: 1024 train rows with 64 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio on the EMA parameters, FreeMatch over the constant-gate arm; the paper's mask rate and the impurity of retained labels, and the tau_t trajectory, as guardrails
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: ratio < 1.0 in mean on both the EMA and the trained parameters, by at least one standard error; terminal mask rate above 0.2; impurity of retained labels < 0.15; held-out outcome NLL within 1.05x of the constant-gate arm; tau_t above 0.8 and strictly above 1/K at the end of the run
  seeds: 10
  report: mean_and_stderr
```

The last clause is the mechanism guardrail and it is written to be **falsifiable
by a dead mechanism**, which the equivalent clause on `flexmatch.md` §6 was not:
that card declared `min_c T(c) < 0.5 tau at step 0`, which Algorithm 1 makes
true for every possible run. `tau_t > 1/K` at the end can fail — `tau_t` is an
EMA of the model's own confidence, so a run whose propensity never becomes
confident leaves it pinned near its initialisation, which is exactly the failure
mode §2's first limitation describes.

**The target above was written before any of it was run, and it has not been
retuned.** `FIDELITY.md` §3 is explicit that a tolerance adjusted after seeing a
result is itself a deviation, and that cuts both ways.

### 6.1 Fixed DGP and the declared arms

**Fixture.** `fixmatch.md` §6.1's DGP in full and without modification, so that
the pair differs in the gate and in nothing else — the mechanism, the seeds, the
64-label MCAR budget, the `B = 64` / `mu B = 448` quota, the outcome
standardisation fitted on the training rows, and the replicate seeds
`s_r = 90000 + 100 r` for `r in {0..9}`. Restating it here would be a second
thing to keep true. It is the same fixture `flexmatch.md` §6.1 uses, so the two
cards' primary numbers are directly comparable.

**Four arms**, all sharing that fixture, those seeds and that batch stream:

| Arm | What it is | What it isolates |
|---|---|---|
| `constant` | eq. (8) replaced by FixMatch's eq. (4) at `tau = 0.95`, no fairness term | the §6 denominator; identical to `flexmatch.md` §6's comparison arm |
| `sat` | this recipe with `w_f = 0` | SAT alone, which is the paper's own §5.3 ablation |
| `freematch` | this recipe as declared, `w_f = 0.05` | SAT + SAF |
| `literal` | this recipe with `w_f = -0.05` | eq. (11) exactly as printed; deviation 7's alternative, expressible with no code change |
| `amplified` | this recipe with `w_f = +1.0` | the same term at twenty times the paper's weight |
| `amplified_literal` | this recipe with `w_f = -1.0` | eq. (11) as printed, likewise amplified |

The last two arms are §6.4's, not §6's, and they exist because of an asymmetry
in the two readings rather than to tune anything. Deviation 7's sign is bounded
below — `H(A, B)` is minimised at `B = A` — and eq. (11)'s printed sign is not:
minimising `-H(A, B)` rewards driving `B` towards a corner of the simplex without
limit. At `w_f = 0.05` the two are within each other's error bars on the primary
fixture, so the question is unresolvable there whatever the sign. At a weight
where the term can actually move the optimisation, a bounded objective and an
unbounded one must behave differently, and *that* is a measurement rather than
an argument. A recipe declaring `w_f = 1.0` is not FreeMatch and is not offered
as one; it is the instrument.

The `literal` arm is not a target. It exists because deviation 7 is an argument
about a sign, and an argument about a sign that could have been a measurement is
a bad trade in this repository.

**Two diagnostic fixtures, §6.4 only.** §2's third limitation says a balanced
two-class fixture leaves SAF with nothing to do, and §6.2 measured exactly that.
`benchmarks/common.py`'s `cluster_population` generalises `fixmatch.md` §6.1's
DGP over the number of clusters and over the cluster prior, and **at `K = 2`
with a uniform prior it is that DGP bit-for-bit** — the same draws, from the
same seed, in the same order, pinned by a digest taken before the two were
joined. So the primary fixture is not a neighbour of the sweep below; it is its
`K = 2` member, and Tier 0 asserts it.

| Fixture | `K` | cluster prior | Bayes error | `P(max q >= 0.95)` | weak-view flips | strong-view flips |
|---|---|---|---|---|---|---|
| **primary (§6.1)** | 2 | `(0.5, 0.5)` | 0.067 | 0.705 | 0.026 | 0.075 |
| `k4` | 4 | `(0.25, 0.25, 0.25, 0.25)` | 0.154 | 0.424 | 0.064 | 0.181 |
| `k4_skewed` | 4 | `(0.55, 0.25, 0.13, 0.07)` | 0.123 | 0.521 | 0.047 | 0.129 |
| `k4_adjacent` | 4 | `(0.25, 0.25, 0.25, 0.25)` | 0.306 | 0.155 | 0.076 | 0.202 |

`k4_adjacent` is the exception to the sentence below and the reason it is
declared last: its classes are **not** equidistant. Classes 0-2 sit in one tight
group 0.9 apart and class 3 sits 1.87 from all three, so the Bayes-optimal
confidence is 0.632/0.632/0.634 on the crowded classes and **0.872** on the
isolated one — an unequal per-class confidence at a *uniform* class frequency.
That is the "class adjacency" FreeMatch §4.1 gives as a reason to want a
per-class threshold, and §6.4 needs it for a reason that only became clear once
`k4_skewed` had been run. Tier 0 asserts both halves: that the regular simplex
is confidence-flat, and that this one is not.

Every other cluster centre is a vertex of a regular simplex at a fixed pairwise
separation of 1.8 — the primary fixture's — rotated so each class's signal is
spread across all four signal columns. Equidistance is what makes `K` the only
thing the sweep varies; the rotation is what keeps a masked column from
destroying the label, which `flexmatch.md` §5.2's argument depends on. The
assignment is `p(t = c | c) = 0.98` with the remaining 0.02 split evenly, which
at `K = 2` is that card's 0.02/0.98. The outcome multiplier at `K = 4` is
`(0.0, 1.0, 0.4, 1.6)`, deliberately non-monotone in `t`: a multiplier rising
with the treatment index would be a dose-response model wearing a categorical
costume, and `BACKLOG.md` §15.9 puts that outside v1.

**The declared views had to be re-justified, and the criterion they are judged
against had to change.** `flexmatch.md` §5.2 requires a strong view to keep the
Bayes-optimal label on at least **90%** of rows, and that card flags the 90% as
"not robust to its own constant". It is not robust to `K` either: at a fixed
separation the declared strong view flips 18% of Bayes labels at `K = 4`, and at
`K = 5` **no** layered mask clears 10% at all — the weak view alone already
flips 7.9%. That is not the view becoming careless. It is an absolute budget
being a harsher standard the further chance-level sits from 100%, and holding it
would have meant either a "strong" view barely stronger than the weak one, or a
fixture so separated that the propensity is solved before the gate opens (the
failure `fixmatch.md` §6.1 says it chose 0.45 to avoid).

So the criterion is stated **relative to what is achievable**: the strong view's
flip rate must stay within a quarter of the clean Bayes error. The primary
fixture sits at 1.13, and the table above is 1.18 and 1.05 — so the views carry
over across `K` unchanged, and are as label-preserving on each fixture as the
one `flexmatch` reviewed. `tests/invariants/test_cluster_fixture.py` recomputes
the whole column on every PR rather than quoting it.

**What §6.4 is asking, declared before it was run.** Three questions, each of
which §6.2 established this card's primary fixture cannot answer:

1. **Does SAF do anything when the class marginal is not already uniform?**
   §6.2 bounded its effect at `-0.0001 +- 0.0012` and showed why — the term
   sits at its own floor. `k4_skewed` is the fixture that was expected to
   change that.
2. **Is deviation 7's sign question measurable?** The `literal` arm on a fixture
   where SAF is live is the experiment §6.2 said was needed and could not run.
   `k4_adjacent` and the two amplified arms are what make it decidable if the
   first two fixtures leave it at the noise floor.
3. **Does the `k4` / `k4_skewed` contrast separate "more classes" from
   "skewed marginal"?** SAT's local half has more to differentiate at `K = 4`
   whatever the prior; SAF's was expected to move only under skew.

**Question 1's premise turned out to be wrong, and `k4_adjacent` is the
correction rather than a fourth guess.** Eq. (9) divides mean probability mass
by predicted-label frequency, so `p_bar / h_bar` is the mean confidence *per
predicted class*. Skewing the prior moves numerator and denominator together —
a rare class collects disproportionate spill from other rows, which pushes the
ratio up, and is predicted less confidently, which pushes it down — and on a
fixture where rarity and low confidence coincide, as they generally do, the two
cancel. What the ratio responds to is per-class *confidence* at fixed
*frequency*, which is what `k4_adjacent` supplies and what no amount of skew
can. §6.4 records the closed form and the measurement together.

A result on (1) or (2) is a **direction on five seeds**, never a target
(deviation 10). A null on (2) at a fixture where §6.4 can show the term is
*live* would be informative where §6.2's null was not — and that distinction is
the whole reason the `fairness_support` and `marginal_entropy` diagnostics are
logged.

### 6.2 What the Tier 1 fixture shows

Tier 1 is not Tier 2 and none of this is a `reproduced` claim: there is no
Tier 2 runner for this card (§6.3), the seed count is **five** rather than the
declared ten, and the metrics were computed by a script rather than by
`evaluation/benchmarks/`. What it is, is the declared budget, the declared
fixture, the declared replicate seeds and the four declared arms.

Each replicate `r` uses `s_r = 90000 + 100 r`, with the DGP at `s_r + 1`, every
arm initialised from `s_r + 6` and fitted under stage seed `s_r + 10000`, so the
four arms differ in the gate and the fairness weight and in nothing else.

**The primary metric, five seeds, 3,000 steps.**

| `r` | 0 | 1 | 2 | 3 | 4 | mean ± stderr |
|---|---|---|---|---|---|---|
| held-out `p(t\|x)` NLL, EMA — `constant` | 0.295 | 0.338 | 0.333 | 0.315 | 0.300 | 0.316 ± 0.009 |
| held-out `p(t\|x)` NLL, EMA — `freematch` | 0.285 | 0.317 | 0.322 | 0.308 | 0.287 | 0.304 ± 0.008 |
| ratio, EMA | 0.967 | 0.938 | 0.966 | 0.979 | 0.957 | **0.961 ± 0.007** |
| ratio, trained network | 0.953 | 0.931 | 0.951 | 0.988 | 0.936 | **0.952 ± 0.010** |
| marginal-frequency baseline | 0.694 | 0.694 | 0.694 | 0.693 | 0.693 | 0.694 |

Both ratios are below 1.0 by more than four standard errors and FreeMatch is
ahead on five seeds of five on both, which is the direction §2 claims. The two
readings agreeing is worth one sentence, because two cards in this family have
found them disagreeing — `fixmatch.md` §6.2 on the overlapping fixture and
`doublematch.md` §6.2 across two initialisation draws — and a result resting on
the EMA alone is a result about a reporting device. That they agree here is a
fact about these five runs, not a property established of the method.

**Every clause of §6's tolerance, on those five seeds.**

| Clause | Measured | Met? |
|---|---|---|
| ratio < 1.0 in mean on the EMA, by ≥ 1 stderr | 0.961 ± 0.007 | yes, by 5.7 |
| ratio < 1.0 in mean on the trained network, by ≥ 1 stderr | 0.952 ± 0.010 | yes, by 4.9 |
| terminal mask rate above 0.2 | 0.822 ± 0.013 | yes |
| impurity of retained labels < 0.15 | 0.047 ± 0.002 | yes |
| held-out outcome NLL within 1.05x of `constant` | 0.9999 ± 0.0001, max 1.0003 | yes |
| `tau_t` above 0.8 and strictly above `1/K` at the end | 0.922 ± 0.005 | yes |

So the declared target is met on every clause at five of the declared ten
seeds. That is not `reproduced` and §6.3 says why; it is the strongest thing
Tier 1 evidence is allowed to be.

**The mechanism, on the same five runs.** `tau_t` starts at `1/K = 0.5`
exactly, so eq. (8) opens ungated on the whole batch — coverage 1.0 at step 0
against the constant arm's 0.0 — and ends at 0.922, having been earned rather
than declared. Eq. (7)'s local half is live throughout: the terminal
`min_c tau_t(c)` is 0.858 against a `max_c` of 0.922, so `MaxNorm` separates the
two classes by 0.064 where FlexMatch's `beta` would have collapsed both to `tau`
(§2's third limitation). Terminal coverage is 0.854 against the constant arm's
0.804, so the self-adaptive gate admits *more* rows — which is the direction
§4.1 says SAT is for. It does so at an impurity of 0.047 ± 0.002 against the
constant arm's 0.049 ± 0.002; those overlap within a standard error, so the
honest statement is that admitting the extra rows did not cost purity, not that
it bought any.

**Self-adaptive fairness contributes nothing here, and the reason is
measurable rather than inferred.** The paper's own §5.3 ablation is the `sat`
arm, `w_f = 0`:

| paired difference in held-out `p(t\|x)` NLL (EMA), five seeds | mean ± stderr |
|---|---|
| `freematch` (`w_f = 0.05`) − `sat` (`w_f = 0`) | **−0.0001 ± 0.0012** |
| `literal` (`w_f = −0.05`, eq. 11 as printed) − `sat` | **+0.0001 ± 0.0010** |

Against a `sat` − `constant` difference of −0.0123, SAF moves the metric by
under 1% of what SAT moves it by, with error bars an order of magnitude smaller
than that effect. So this is a bound, not merely a failure to detect one.

The diagnostic says why, and it is exactly §2's third limitation: the terminal
marginal entropy of the retained rows' predictions is **0.6919 ± 0.0006**
against a maximum of `log 2 = 0.6931` — a deficit of 0.0012 nats, which for
two classes puts the marginal within about 0.025 of even. `H(A, B)` with `A`
uniform is minimised at exactly that point, so the term sits within a thousandth
of its own floor for the whole run and the gradient it has to contribute is the
gradient of a loss at its minimum. On a near-balanced two-class fixture a term
that pushes the predicted marginal towards uniformity has nothing to push.

**Which means this fixture cannot settle deviation 7, and saying so is the
point of having run the `literal` arm.** The two signs differ by −0.0001 and
+0.0001 against the same baseline; both are zero. That is not evidence that the
sign does not matter — it is evidence that *on a fixture where the term is
inert, the sign of an inert term is unmeasurable*. Deviation 7 therefore stands
on §5's argument alone, and the experiment that would test it needs a fixture
where the marginal is genuinely skewed. `flexmatch.md` §6.1's class-imbalanced
variant (`p(t = 1) ≈ 0.16`) is the obvious candidate and is not run here; §7's
first row is where that debt lives.

**One thing this section deliberately does not say.** It does not claim the
gain is FreeMatch's *self-adaptivity* rather than the lower effective threshold
it happens to produce. The terminal `tau_t` is 0.922 against the constant arm's
0.95, and a constant gate at 0.92 was not run. Separating "the threshold
adapts" from "0.92 beats 0.95 on this fixture" needs a third constant-gate arm,
and `BACKLOG.md`'s note on this card is where that is recorded as the next
measurement rather than as a result.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | — | Not run |

§6.2's numbers are **not** in the table above, and that is the rule rather than
an oversight: `FIDELITY.md` §3 makes the ledger a record of Tier 2 runs, and
§6.2 is five seeds from a script.

**No Tier 2 runner exists for this card**, and that is stated here rather than
left to be inferred from an empty table. `xty2/evaluation/benchmarks/` has one
module per recipe that has been Tier 2'd and none for this one, so §6's target
is declared and unmeasured at the declared ten seeds — the position `doublematch`
and `flexmatch` also ship in. The status line stays `draft` until it has one and
a reviewer has signed §8.

### 6.4 The diagnostic fixtures

*Declared in §6.1 above and not yet measured. This section is written when the
numbers exist; the questions it must answer are fixed before the run, and the
answers are directions on five seeds rather than targets (deviation 10).*

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| The sign of eq. (11). As printed, `L_f = -H(A, B)` minimised is entropy *minimisation* of the modulated marginal, which contradicts the purpose §1, §2, §4.2 and §6 all state for the term | `L_f = +H(A, B)`, minimised — deviation 7 | The paper's own prose against the paper's own sign, plus the derivative in deviation 7. **This is the row a reference implementation would have closed**, and §1 records why one was not read. §6.1's `literal` arm was declared to measure it and §6.2 ran it — and it came back a null on **both** signs, because the term is inert on a near-balanced two-class fixture. So the choice is still argued rather than measured, and the outstanding experiment is the same one §6.2 names: this recipe on `flexmatch.md` §6.1's class-imbalanced variant, where the marginal SAF acts on is genuinely skewed |
| What `Hist` does when a class has no retained row: eq. (9)'s `\bar h(c)` is then 0 and `\bar p(c)/\bar h(c)` is undefined | Classes with an empty `\bar h` bin are **excluded from both SumNorms**, and a step in which fewer than two classes survive contributes `L_f = 0` | Convention. The two candidates are exclusion and "treat `1/\bar h(c)` as 0", and the second is worse under the sign deviation 7 settles: it makes `B(c) = 0` for an absent class, so `-A(c) log B(c)` is an arbitrarily large penalty carrying no gradient — a constant added to the loss for a class the term cannot act on. Exclusion keeps the term differentiable everywhere and makes it inert rather than explosive when the retained set is single-class. The objective logs `fairness_support` so a run in which this is biting is visible rather than inferred |
| `h~_0`, the initial value of eq. (10)'s EMA. Eqs. (5) and (6) state `t = 0` cases and eq. (10) does not | `h~_0(c) = 1/C`, uniform | Consistency with eqs. (5) and (6), which give every other statistic in the same triple a uniform initialisation, and with the quantity: `h~` is a normalised histogram, so the uninformative value is the uniform one. A zero initialisation would make eq. (11)'s `p~/h~` undefined at the first step |
| Whether the `t = 0` cases of eqs. (5) and (6) mean "initialise, then update from step 1" or "initialise and also fold in step 0's batch" | Initialise, and skip the update at step 0 | Eq. (5)'s piecewise form is explicit that `tau_0` *is* `1/C`, which folding step 0's batch in would contradict. At `lambda = 0.999` the two readings differ by about `10^-3` of one batch's mean confidence, so this is recorded for exactness rather than because anything turns on it |
| Whether FreeMatch inherits FixMatch's footnote 2 — that `U` contains the labelled rows too, without their labels | It does. `U` is every training row and eq. (8)'s eligible set is `all` | Convention, and consistency: §3 restates FixMatch's and UDA's framework unchanged and §5.1 adopts their hyperparameters. `fixmatch.md` §3.2 and `flexmatch.md` §7 reached the same reading for the same term |
| The strict vs non-strict comparison at the gate: eq. (8) writes `>`, Algorithm 1 line 9 writes `>=`, and eq. (9)'s indicator writes `>=` | `>` in both objectives, as eq. (8) writes it, applied identically in eq. (9)'s indicator so that the two terms retain the same rows | The main text over the appendix, and internal consistency over transcription: eqs. (8) and (9) gating *different* row sets in the same step would be a substantive difference, where `>` against `>=` differs in a set of measure zero (the exact tie `max(q) == tau_t(c)`). Recorded so that a reader comparing this objective's source against `PseudoLabelTreatmentNLL`'s `>=` does not read it as a transcription error |
| Which `w_f` applies. Table 5 gives 0.01 for CIFAR-10 (10 labels), CIFAR-100 (400), STL-10 (40), ImageNet (100k) and all of SVHN, and 0.05 "for other settings" | 0.05 | The 0.01 list is the barely-supervised end of each dataset. At 64 labels over `K = 2` — 32 per class — this fixture is not in that regime, so the "other settings" value applies. §5.9 records that the fixture cannot reach the regime where the other value would |
| Whether Nesterov momentum is used | Yes | §5.1 gives "SGD with a momentum of 0.9" and states that every algorithm in the comparison is trained "using the unified codebase TorchSSL with the same backbones and hyperparameters"; FixMatch's table 4 states Nesterov. The same guess `fixmatch` and `flexmatch` already run, which is what keeps §6's pair a pair |
| Whether the EMA copy or the current network supplies `q_b`, `Q_b` and the statistics of eqs. (5), (6) and (10) | The current network | Algorithm 1 makes no mention of an EMA in lines 2–12, and §5.1's EMA sentence is about the testing phase. The same reading `fixmatch.md` §7 and `flexmatch.md` §7 record |
| Whether the two SVHN techniques of §5.1 — a labelled-only warm-up and clamping `tau_t` to `[0.9, 0.95]` — belong to the method | They do not; deviation 6 | §5.1 introduces both for SVHN alone and as a response to a dataset-specific failure. §4, which defines FreeMatch, contains neither |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
