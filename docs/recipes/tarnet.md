# Recipe spec card: tarnet

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **On the status.** The implementation exists and both tiers pass: Tier 0
> covers the components, the compiled plan and the §4 cross-check, and Tier 1
> fits the synthetic DGP and clears all four assertions of `FIDELITY.md` §3.
> The status is nevertheless `draft`, because the ladder in `FIDELITY.md` §1.1
> runs through `reviewed`, and `reviewed` means *a human has approved this
> card*. Nobody has. Setting `smoke-passing` here would claim a review that
> did not happen, which is the one thing a status is for. When a reviewer
> signs §8, it goes straight to `smoke-passing`.

> **Reviewer, read this first.** The paper could not be re-read while this card
> was written: the authoring environment has no network egress, so `arxiv.org`
> and the reference implementation were both unreachable. Every §4 entry is
> therefore one of three things, and each is labelled: **(paper)** — stated in
> the published text and recalled with high confidence; **(verify)** — believed
> to be in the paper but *not* re-read, and needing a page reference before this
> card leaves `draft`; **(ours)** — not in the paper at all, with the choice and
> its basis recorded in §7. Nothing marked **(verify)** may be treated as a
> paper citation until someone has checked it against the PDF.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | Estimating individual treatment effect: generalization bounds and algorithms |
| Authors, year | Uri Shalit, Fredrik D. Johansson, David Sontag — 2017 (ICML) |
| DOI / arXiv | arXiv:1606.03976 |
| Version used | **unverified** — the arXiv listing was unreachable at authoring time; fix the version at review |
| Reference implementation | https://github.com/clinicalml/cfrnet — **not pinned, not consulted** |
| Reference impl. runnable? | not attempted |

TARNet is the *α = 0* member of the paper's CFR family: the shared
representation and the per-arm outcome heads, without the IPM balance
penalty. It is reported as its own row in the results table, which is what
makes it a reproduction target rather than an ablation we invented.

## 2. Estimand and claim

- **Estimand:** the conditional average treatment effect
  `τ(x) = E[Y | x, t = 1] − E[Y | x, t = 0]`, and the ATE as its mean over the
  population. Both are read off the outcome head's treatment-wise means, which
  is why the `mean` half of the candidate-treatment contract
  (`DESIGN.md` §3.1) is load-bearing here and not only `log_prob`.
- **Claim:** with a shared representation `Φ` and two arm-specific outcome
  heads trained on the factual outcome alone, individual-effect error
  (`√εPEHE`) on IHDP is competitive with the balanced (CFR) variants and
  better than the tree and matching baselines. The paper's generalisation
  bound motivates the balance term; TARNet is the arm of the experiment that
  shows how much of the result survives without it.
- **Not claimed:**
  - nothing about missing or partially observed treatments. The paper's
    setting has `t` observed on every row. Our `MissingTreatmentMarginalNLL`
    term and the propensity head it needs are **our extension** (§5), and no
    published number exists for them.
  - nothing about the *treatment* model. TARNet has no `p(t | x)`.
  - no calibration or density claim about `p(y | x, t)`: the paper fits a
    conditional mean under squared loss, not a likelihood.

## 3. Equations and mapping

### 3.1 As published

The CFR objective, in the paper's notation (equation number **(verify)** — it
is the displayed objective in the *Algorithm* section):

> minimise over `h, Φ`:
>
> `(1/n) Σ_i w_i · L( h(Φ(x_i), t_i), y_i )  +  λ · R(h)  +  α · IPM_G( {Φ(x_i)}_{i: t_i=0}, {Φ(x_i)}_{i: t_i=1} )`
>
> with `w_i = t_i / (2u) + (1 − t_i) / (2(1 − u))`, `u = P(t = 1)` **(verify)**,
> and `L` the squared loss for a continuous outcome **(paper)**.
>
> **TARNet is this objective at `α = 0`** **(paper)**, so the IPM term is absent
> rather than weighted to zero.

`h(Φ, t)` is not a function of a treatment *feature*: the paper uses one
hypothesis head per treatment arm over the shared `Φ(x)` **(paper)**, which is
the whole point of the architecture and is exactly the candidate-treatment
contract of `DESIGN.md` §3.1.

### 3.2 Mapping to xty2

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `Φ(x)` | shared representation | `X_REPR` | `mlp_encoder` |
| `h(Φ, t)` | per-arm outcome head | `Y_GIVEN_XT` | `tarnet_head` |
| `L(h(Φ, t_i), y_i)` | factual squared loss | `Y_GIVEN_XT` | `ObservedOutcomeNLL` |
| `λ · R(h)` | L2 regularisation | — | `OptimiserSpec.weight_decay` |
| `α · IPM_G(...)` | balance penalty | — | **absent by construction** — TARNet is `α = 0`, so there is no `RepresentationBalance` term in this recipe |
| `w_i` | treated/control reweighting | — | **not implemented** (§5, deviation 3) |
| — (no paper symbol) | `p(t \| x)` | `T_GIVEN_X` | `categorical_propensity`, `ObservedTreatmentNLL` — **our extension** (§5, deviation 1) |
| — (no paper symbol) | `−log Σ_k p(t=k\|x) p(y\|x,t=k)` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL` — **our extension** (§5, deviation 2) |

Views: none. Realisations: `view=identity params=student` only.

## 4. Mechanics checklist

> Labels: **(paper)** recalled from the published text; **(verify)** believed to
> be in the paper but not re-read; **(ours)** our choice, recorded in §7.
>
> The `data` block is `n/a` **for the recipe**, and that is a statement about
> where the decision lives rather than a claim that it does not matter. A recipe
> is a component graph and a program; nothing in it loads, splits, standardises
> or hides data, and `DESIGN.md` §11 refuses to add a card key the framework
> cannot check. The Tier 2 data protocol is §6 and belongs to P12; the Tier 1
> missingness mechanism is the synthetic DGP in `tests/smoke/synthetic.py`.
> When P12 lands a benchmark runner that owns these decisions, this block stops
> being `n/a` and the runner binds the keys.

```yaml
gradients:
  stop_gradients: none                    # (ours) every objective declares detaches = {}
  detached_targets: n/a                   # no consistency loss in this recipe
  gradient_clipping: none                 # (ours) the paper states no clipping
  marginal_nll_grad_path: both            # (ours) our term; plain likelihood gradient

teacher:
  ema_decay: n/a                          # no teacher parameters in this recipe
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction: mean                         # (paper) per objective; see the note below
  eligible_rows: per objective            # t_observed, t_observed, t_missing
  weights: 1.0, 1.0, ramped to 1.0        # (ours) per objective
  schedules: marginal term ramps 0 -> 1   # (ours) linear over the first 1000 steps
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a

optimisation:
  optimiser: adam                         # (verify) the paper trains with Adam
  lr: 0.001                               # (ours) unspecified in the text - §7
  lr_schedule: constant                   # (ours) unspecified in the text - §7
  weight_decay: 0.0001, norm and bias exempt   # (ours) the paper's lambda, value grid-searched - §7
  batch_size: n/a                         # no loader exists (DESIGN.md §11) - §7
  labelled_unlabelled_ratio: n/a          # same; and the paper has no unlabelled rows
  total_steps_or_epochs: 3000 steps       # (ours) steps, not epochs - §7

architecture:
  widths_depths: representation 3x200, heads 3x100   # (paper)
  activation: elu                         # (paper) exponential-linear units
  normalisation: none                     # (ours) - §7
  dropout: 0.0                            # (ours) - §7
  initialisation: xavier_normal           # (ours) - §7
  output_parameterisation: gaussian per-arm mean, unit scale   # (ours) - §7

data:
  standardisation: n/a                    # see the note above §4
  outcome_scaling: n/a
  treatment_encoding: n/a
  split_protocol: n/a
  missingness_mechanism: n/a
```

**On `losses.reduction`.** The paper's objective is a mean over the whole
training set, and in the paper's setting every row has an observed treatment —
so `mean` over `t_observed` *is* the paper's reduction, exactly, for
`ObservedOutcomeNLL`. That equality stops holding the moment treatments go
missing, which is our setting and not the paper's: with 50% of `t` missing,
`mean` over `t_observed` averages over half the batch. We keep `mean` for all
three terms and record the consequence here rather than in a comment: the
outcome and marginal terms are then each a mean over *their own* population,
so the semi-supervised term's effective weight does not shrink as coverage
does. `population` would have been the other defensible reading. This is the
field `DESIGN.md` §6.1 says produces a model that trains, looks reasonable and
weights its semi-supervised term differently from the paper, so it is stated
rather than inherited.

## 5. Deviations from the paper

| # | What we do differently | Why | Expected effect on the §6 metric |
|---|---|---|---|
| 1 | We add a propensity head `p(t\|x)` on the **shared** representation and train it with `ObservedTreatmentNLL`, so its gradient reaches `Φ`. TARNet has no treatment model. | The framework's purpose is the partially-observed-treatment setting, and exact marginalisation (§4.1 of `DESIGN.md`) needs `p(t\|x)`. Putting it on the shared trunk rather than on raw `x` is what makes it a *recipe* rather than two unrelated models. | Real and unquantified. `Φ` is fit to predict `t` as well as `y`, which is a form of the balancing the IPM term was meant to *prevent* being needed. Direction unknown; magnitude expected small at weight 1.0. **This is the deviation most likely to move `√εPEHE`, and it is why §6 may land at `deviating`.** |
| 2 | We add `MissingTreatmentMarginalNLL`, ramped 0 → 1 over the first 1000 steps. | The recipe exists inside a semi-supervised framework; without it P5 would prove nothing about the machinery the whole design is arranged around. | **None on §6.** The IHDP reproduction has `t` observed on every row, so the term has zero eligible rows on every batch and is excluded from the total by the zero-eligible-row rule. It is validated by the Tier 1 assertion instead (`FIDELITY.md` §3). |
| 3 | We do **not** apply the per-row treated/control reweighting `w_i`. | `ObservedOutcomeNLL` is unweighted (`DESIGN.md` §4); `XTYBatch.weight` exists but no objective consumes it, and adding that consumption is a framework change with one consumer. | IHDP is roughly 19% treated, so the factual loss under-weights the treated arm relative to the paper's objective. Expected to *increase* `√εPEHE` slightly. If §6 lands outside tolerance, this is the first thing to implement. |
| 4 | The factual loss is a Gaussian negative log-likelihood with unit scale, not a squared error. | The `Y_GIVEN_XT` port carries a distribution, not a point prediction (`DESIGN.md` §2). | None on the optimum: `−log N(y; μ, 1) = ½(y − μ)² + ½log 2π`, so the gradient differs from the paper's squared loss by a constant factor of ½, absorbed into the learning rate. It is not a no-op for deviation 2, though: the marginalisation term weights arms by `p(y \| x, t=k)`, so the unit scale is a modelling choice there and not a nuisance constant (§7). |
| 5 | No IPM / balance term. | This is not a deviation from **TARNet**; it is what TARNet *is* (`α = 0`). Recorded so that a reader diffing this card against the CFR rows of the same table does not read the omission as an error. | None. |

## 6. Reproduction target

```yaml
reproduction:
  dataset: IHDP
  variant: "1000 realisations, Hill (2011) setting A, as distributed with Shalit et al. 2017"
  split: "the authors' train/validation/test split, 63/27/10"
  metric: sqrt_PEHE_in_sample
  published: 0.88
  published_source: "Shalit et al. 2017, Table 1 — TARNET row, within-sample"
  tolerance: 0.10
  seeds: 10
  report: mean_and_stderr
```

`t` is fully observed in this target, by the paper's construction. The
semi-supervised half of the recipe is therefore **not** validated by Tier 2 and
must not be reported as if it were: what validates it is the Tier 1 assertion
that marginalisation beats the complete-case baseline at 50% missing `t`.

The published value and its source are copied from the worked example in
`FIDELITY.md` §3, which quotes the same table. **(verify)** both against the
paper before this card leaves `draft` — a reproduction target taken from our own
documentation rather than from the source is exactly the circularity §6 exists
to prevent.

### 6.1 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| — | — | — | not yet run (P12) | — |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Learning rate | `1e-3` | Convention for Adam on a network of this size. The paper grid-searches it; we did not reproduce the search. |
| Learning-rate schedule | constant (`Constant(1.0)` multiplier) | The paper states no schedule we can cite. The reference implementation is believed to decay the rate; unverified, so a constant is the honest choice and a decay is a future deviation, not a silent one. |
| Weight-decay coefficient and reach | `1e-4`, exempting biases and norm parameters | The paper's `λ` is grid-searched, so no single value is *the* paper's. `1e-4` is the middle of the usual range. Exempting one-dimensional parameters is the standard reading of "weight decay" and is the half `FIDELITY.md` §2 says is invisible in a diff. |
| Training length | 3000 optimiser steps | Steps, not epochs (`FIDELITY.md` §2). Chosen as a plausible budget for the IHDP training-set size; **this is a guess and must be revisited if §6 lands short.** |
| Dropout | `0.0` | The paper's architecture description does not include dropout. |
| Normalisation | `none` | Likewise. The reference implementation is believed to offer a representation-normalisation option; unverified. |
| Initialisation | `xavier_normal` | The paper is believed to initialise from a fan-in-scaled zero-mean normal **(verify)**; Xavier-normal is the closest standard rule. `torch_default` (Kaiming-uniform) would be the alternative. |
| Outcome parameterisation | Gaussian, per-arm mean, **unit** scale | The paper has no likelihood at all — it minimises squared error. Unit scale is the parameterisation that makes the NLL proportional to that error (§5, deviation 4). A learned scale would change the arm weighting inside the marginalisation term, so this is a modelling decision and not a formatting one. |
| Propensity-head shape | the same `3x100` as the outcome arms | The paper has no propensity head. Mirroring the hypothesis heads keeps one `widths_depths` line describing the whole stack, which is what the closed card-key vocabulary can express. |
| Weight on `ObservedTreatmentNLL` | `1.0` | Our extension; no paper value exists. `1.0` puts it on the same footing as the factual term and makes the deviation-1 effect maximally visible rather than hidden behind a small weight. |
| Marginal-term ramp | linear `0 → 1` over the first 1000 steps | Our extension. A ramp rather than a constant because the term is meaningless while both heads are at initialisation: `p(y \| x, t=k)` is then nearly flat in `k`, and the term is close to `−log Σ_k p(t=k\|x) · c`, which trains the propensity toward whatever the outcome noise favours. |
| Gradient path of the marginal term | `both` | Our extension; the plain likelihood gradient, which is the default reading when no paper stops one side. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
