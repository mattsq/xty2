# Recipe spec card: cycle_dual

**Status:** `reproduced`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) and [DualGAN: Unsupervised Dual Learning for Image-to-Image Translation](https://arxiv.org/abs/1704.02510) supply the cycle/dual inspiration. `cycle_dual` is a project-local causal adaptation, not a method proposed by either paper. |
| Authors, year | Jun-Yan Zhu, Taesung Park, Phillip Isola and Alexei A. Efros, 2017; Zili Yi, Hao Zhang, Ping Tan and Minglun Gong, 2017 |
| DOI / arXiv | [arXiv:1703.10593v7](https://arxiv.org/abs/1703.10593v7); [arXiv:1704.02510v4](https://arxiv.org/abs/1704.02510v4) |
| Version used | CycleGAN arXiv v7, 2020-08-24, especially equations (1)-(4); DualGAN arXiv v4, 2018-10-09. |
| Reference implementation | Project-local source: [`mattsq/XTYLearner` `cycle_dual.py` @ `35734ec2d5a62d54a59eca38d1e31423da31e1ea`](https://github.com/mattsq/XTYLearner/blob/35734ec2d5a62d54a59eca38d1e31423da31e1ea/xtylearner/models/cycle_dual.py). Paper-level CycleGAN reference: [`junyanz/pytorch-CycleGAN-and-pix2pix` @ `2a7afba2895d52556dd5dfe07e8555ef657ced6f`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/2a7afba2895d52556dd5dfe07e8555ef657ced6f). |
| Reference impl. runnable? | Not attempted in the P11 card pass. The project-local source was inspected at the pinned commit. |

The authority order is deliberate. The two papers establish what cycle
consistency means in unpaired image translation. The pinned XTYLearner source
defines the historical tabular adaptation and recipe name. Neither source
establishes a causal estimator for missing treatment labels. The P11 card owns
that adaptation and states the departures explicitly.

## 2. Estimand and claim

- **Estimand:** The first stage estimates the categorical treatment posterior
  `q(t | x, y)`. The second estimates candidate-treatment outcome means
  `mu_k(x) = E[Y | X=x, T=k]`; under consistency, positivity and conditional
  exchangeability, contrasts `mu_k(x) - mu_j(x)` identify conditional treatment
  effects.
- **Claim:** This recipe tests a narrow staged construction: fit an
  outcome-dependent treatment posterior on observed treatment labels, emit hard
  out-of-fold labels for missing treatments, then fit a treatment-conditional
  outcome model on the functionally joined data. The posterior should exploit
  outcome information that `p(t | x)` cannot, while P10 provenance prevents an
  in-sample posterior-to-outcome refit.
- **Not claimed:** CycleGAN and DualGAN do not justify this causal adaptation.
  Out-of-fold pseudo-labelling prevents direct training-row reuse but does not
  make a misspecified posterior, hard labels, or a missing-not-at-random
  mechanism harmless. The recipe does not claim valid confidence intervals,
  does not identify effects without the ordinary causal assumptions, does not
  implement continuous treatment, and does not reproduce an image-translation
  result.

## 3. Equations and mapping

### 3.1 As published and as inherited

CycleGAN defines two mappings `G: X -> Y` and `F: Y -> X`. Its equation (2) is

$$
\mathcal L_{\mathrm{cyc}}(G,F)
= \mathbb E_x\lVert F(G(x))-x\rVert_1
+ \mathbb E_y\lVert G(F(y))-y\rVert_1.
$$

Its full equation (3) adds adversarial objectives in both directions,

$$
\mathcal L(G,F,D_X,D_Y)
= \mathcal L_{\mathrm{GAN}}(G,D_Y,X,Y)
+ \mathcal L_{\mathrm{GAN}}(F,D_X,Y,X)
+ \lambda\mathcal L_{\mathrm{cyc}}(G,F),
$$

with the minimax problem in equation (4). DualGAN uses the same broad idea: a
primal translator and inverse translator form a closed reconstruction loop over
two unpaired domains.

The old XTYLearner source is a separate, unnumbered adaptation. For observed
indicator `M_i`, it predicts

$$
q_\phi(t\mid x_i,y_i), \qquad
\tilde t_i =
\begin{cases}
t_i,&M_i=1,\\
\arg\max_k q_\phi(k\mid x_i,y_i),&M_i=0,
\end{cases}
$$

then forms `G_Y(x, t)` and `G_X(t, y)`, two detached reconstruction cycles, a
supervised posterior term, direct reconstruction terms and posterior entropy.
In the notation of that source its total is

$$
\mathcal L_{\mathrm{old}}
= \lambda_{\mathrm{sup}}
  (\mathcal L_{\mathrm{sup},X}
   +\mathcal L_{\mathrm{sup},Y}
   +\mathcal L_{\mathrm{sup},T})
+\mathcal L_{\mathrm{rec},X}+\mathcal L_{\mathrm{rec},Y}
+\lambda_{\mathrm{cyc}}
  (\mathcal L_{\mathrm{cyc},X}+\mathcal L_{\mathrm{cyc},Y})
+\lambda_{\mathrm{ent}}\mathcal L_{\mathrm{ent}}.
$$

That code computes missing-row treatment labels and the outcome reconstruction
inside one loss call. It records no producing fold and therefore cannot support
the P10 causal provenance decision.

### 3.2 Mapping to xty2

P11 keeps the dual statistical directions that exercise the v1 framework and
makes their transition explicit:

1. `posterior_labels`, `executor="cross_fit"`:
   `categorical_posterior` maps `(X_RAW, Y_RAW)` to `T_GIVEN_XY`.
   `ObservedTreatmentNLL(port=T_GIVEN_XY)` fits it on `t_observed` rows. For
   every actual `fold_id`, the executor starts from the same initial state,
   fits on the complement and applies `PseudoLabelAction(T_GIVEN_XY)` only to
   held-out `t_missing` rows. The artifact must derive `used_y=true` and earn
   `prediction_mode="out_of_fold"` from its row sets.
2. `outcome_fit`, `executor="gradient"`: the stage consumes
   `posterior_labels`. The join preserves observed treatments, fills only the
   originally missing rows and marks those joined treatments available in the
   fresh batch. `mlp_encoder` and `tarnet_head` then minimise

$$
\mathcal L_y
= -\frac{1}{N}\sum_i
  \log p_\theta(y_i\mid x_i,\tilde t_i).
$$

The production recipe is causal and safe by construction. P11 also carries a
mutation test made from this real recipe: replacing
`posterior_labels.executor="cross_fit"` with `"gradient"` while leaving the
artifact edge and outcome stage unchanged must make `compile()` raise the
`q(t|x,y) -> p(y|x,t)` circular-fit error. The unsafe form is never registered.

| Source symbol / operation | Meaning | xty2 Port / artifact | xty2 Objective / Component / action |
|---|---|---|---|
| `q_phi(t | x,y)` / `C(X,Y)` | Treatment posterior | `T_GIVEN_XY` | `categorical_posterior` |
| `-log q_phi(t_i | x_i,y_i)` | Supervised posterior fit | `T_GIVEN_XY` | `ObservedTreatmentNLL(port=T_GIVEN_XY, name="observed_posterior_nll")` |
| `argmax q_phi` | Hard missing-treatment labels | `PseudoLabels` | `PseudoLabelAction(port=T_GIVEN_XY, rows="t_missing")` under `cross_fit` |
| `G_Y(x,t)` | Treatment-conditional outcome distribution | `X_RAW -> X_REPR -> Y_GIVEN_XT` | `mlp_encoder`, `tarnet_head` |
| `-log p_theta(y | x,tilde t)` | Outcome fit after functional label join | `Y_GIVEN_XT` plus `posterior_labels` input | `ObservedOutcomeNLL` |

No view, teacher, inverse `G_X`, adversarial discriminator or reconstruction
port is created by this card.

## 4. Mechanics checklist

```yaml
gradients:
  stop_gradients:
    posterior_labels.observed_posterior_nll: none
    outcome_fit.observed_outcome_nll: none
  detached_targets: n/a
  gradient_clipping:
    posterior_labels: none
    outcome_fit: none
  marginal_nll_grad_path: n/a

teacher:
  ema_decay: n/a
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    posterior_labels.observed_posterior_nll: mean
    outcome_fit.observed_outcome_nll: mean
  eligible_rows:
    posterior_labels.observed_posterior_nll: t_observed
    outcome_fit.observed_outcome_nll: t_observed  # all rows after the functional join
  weights:
    posterior_labels.observed_posterior_nll: 1.0
    outcome_fit.observed_outcome_nll: 1.0
  schedules:
    posterior_labels.observed_posterior_nll: constant 1.0
    outcome_fit.observed_outcome_nll: constant 1.0
  temperature: n/a
  sharpening: n/a
  confidence_threshold: n/a  # P10 action is unconditional hard argmax

optimisation:
  optimiser:
    posterior_labels: adam(betas=(0.9, 0.999), eps=1e-8)
    outcome_fit: adam(betas=(0.9, 0.999), eps=1e-8)
  lr:
    posterior_labels: 0.001
    outcome_fit: 0.001
  lr_schedule:
    posterior_labels: constant 1.0
    outcome_fit: constant 1.0
  weight_decay:
    posterior_labels: none
    outcome_fit: none
  batch_size: n/a  # external BatchSource; section 6 fixes it for validation
  labelled_unlabelled_ratio: n/a  # no enforced per-batch quota
  total_steps_or_epochs:
    posterior_labels: 500 optimiser steps per fold
    outcome_fit: 1000 optimiser steps

architecture:
  widths_depths:
    categorical_posterior: concat(X_RAW, Y_RAW) -> [128, 128] -> K
    mlp_encoder: [128, 128]
    tarnet_head: K independent heads, each [128, 128]
  activation:
    categorical_posterior: relu
    mlp_encoder: relu
    tarnet_head: relu
  normalisation:
    categorical_posterior: none
    mlp_encoder: none
    tarnet_head: none
  dropout:
    categorical_posterior: 0.0
    mlp_encoder: 0.0
    tarnet_head: 0.0
  initialisation:
    categorical_posterior: torch Linear default Kaiming-uniform
    mlp_encoder: torch Linear default Kaiming-uniform
    tarnet_head: torch Linear default Kaiming-uniform
  output_parameterisation:
    categorical_posterior: K softmax logits
    tarnet_head: K means; fixed Gaussian scale=1.0

data:
  standardisation:
    posterior_labels: z-score X and Y with training-population statistics
    outcome_fit: z-score X and Y with the same frozen training-population statistics
  outcome_scaling: inverse-transform each candidate-treatment mean with the frozen training Y mean and standard deviation before scoring
  treatment_encoding: categorical integers 0..K-1 with t_observed mask; no sentinel
  split_protocol: n/a  # fold_id belongs to the supplied batch; section 6 fixes five folds
  missingness_mechanism: n/a  # dataset property; section 6 fixes MCAR
```

## 5. Deviations from the papers and project-local source

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Treat `cycle_dual` as a staged posterior/outcome recipe, not a literal CycleGAN or DualGAN. | The papers concern unpaired image domains and make no treatment-effect claim. P11 is specifically assigned to the posterior, pseudo-label transition and leakage guardrail. | No comparison with either paper's image metrics is valid. |
| 2 | `judgement` | — | Replace in-loss, in-sample `argmax q(t|x,y)` with an immutable out-of-fold side table. | The old operation is the exact circular staged fit rejected by `DESIGN.md` section 7.2 when its labels train an outcome model. | May reduce apparent training fit; should make held-out behaviour more credible. |
| 3 | `judgement` | — | Omit `G_X(t,y)`, both explicit reconstruction cycles, adversarial discriminators and entropy minimisation. | A treatment-conditioned covariate reconstruction is a semantic output no v1 recipe produces, and building the cycles, discriminators and entropy term around it would make this a literal CycleGAN — which deviation 1 has already declined for this card. That scope decision is the reason, not a count of consumers: `DESIGN.md` section 11.2 asks whether a card's section 4 checklist needs the mechanism, and this card's does not, because it does not claim the GAN objective. | This is a materially smaller method; the old XTYLearner loss is not expected to be numerically reproduced. |
| 4 | `judgement` | — | Use a reused TARNet Gaussian outcome head rather than the old deterministic `G_Y` MLP. | Candidate-treatment means and likelihoods must satisfy the existing `Y_GIVEN_XT` distribution contract. | MSE and fixed-scale Gaussian NLL have the same optimum up to scale and a constant on complete rows; the architecture still differs. |
| 5 | `judgement` | — | Support categorical treatment only. | Continuous treatment is outside xty2 v1 scope. | No categorical effect for the declared benchmark; legacy continuous-treatment behaviour is absent. |
| 6 | `judgement` | — | Use all hard pseudo-labels with no entropy or confidence threshold. | Neither paper states a confidence gate, and section 4 accordingly marks `losses.confidence_threshold` as `n/a` — so no paper mechanic is being dropped here, which is what `DESIGN.md` section 11.2 Q1 asks. That `PseudoLabelAction` emits argmax only is a consequence of the same absence rather than its cause; a gate has existed on the objective path since `fixmatch`, and this card still declines one. | Noisy labels can bias the outcome fit; the fixed benchmark is intended to reveal that rather than tune it away. |

## 6. Reproduction target

There is no published causal `cycle_dual` result. P12 must run the fixed
project-local benchmark below and must not describe it as a reproduction of
CycleGAN or DualGAN.

### 6.1 Fixed posterior-imputation DGP

Run ten replicates `r in {0, ..., 9}` with base seed
`s_r = 110000 + 100*r`. Independent train, validation and test populations have
2,048, 1,024 and 2,048 rows. Draw, in order, `X`, `U_T`, `epsilon_Y` and `U_M`
from independent CPU generators. Let `X in R^4` have independent standard-normal
columns and

$$
e(x)=\operatorname{sigmoid}(0.8x_1-0.5x_2+0.25x_3),
\qquad T=\mathbb 1\{U_T<e(x)\},
$$

$$
\mu_0(x)=0.5x_1-0.25x_2+0.25(x_3^2-1),
\qquad \tau(x)=1.5+0.5\tanh(x_1),
$$

$$
Y=\mu_0(X)+T\tau(X)+0.5\epsilon_Y.
$$

The analytic ATE is `1.5`. In the training population set
`t_observed = 1{U_M < 0.30}` independently of `(X,T,Y)`. Validation and test
treatments remain available to the evaluator but are never passed as observed
to a fitted posterior. Assign `fold_id = row_id mod 5`. Standardise `X` and `Y`
with training-population mean and standard deviation only; transform validation
and test with those frozen statistics. The outcome head therefore emits means
in standardised-Y units. Before scoring, inverse-transform every candidate mean
with the frozen training statistics,

$$
\hat\mu_t^{\mathrm{original}}(x)
= \bar Y_{train}+s_{Y,train}\hat\mu_t^{\mathrm{standardised}}(x).
$$

Compute treatment contrasts only after this inverse transform. Equivalently,
multiply each standardised contrast by `s_Y_train`; the additive mean cancels.

The primary metric, entirely on the original outcome scale, is

$$
\left|\frac{1}{N_{test}}\sum_i
  [\hat\mu_1(x_i)-\hat\mu_0(x_i)]-1.5\right|.
$$

Secondary guardrails are: the emitted training labels report
`prediction_mode="out_of_fold"` and `used_y=true`; hidden-treatment accuracy on
the originally missing training rows is at least `0.75`; the unsafe executor
mutation fails compilation; and every observed treatment is unchanged by the
artifact join.

```yaml
reproduction:
  dataset: fixed project-local posterior-imputation DGP
  variant: binary treatment; 70% treatment MCAR; five folds
  split: independent 2048 train / 1024 validation / 2048 test per replicate
  metric: absolute_ATE_error
  published: n/a
  published_source: n/a; project-local P11 mechanism target
  tolerance: 0.35 from analytic ATE 1.5
  seeds: 10
  report: mean_and_stderr
```

### 6.2 Result ledger

| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-24 | `d060df351f2fe8bac6d951c3757506c684d8b408` | absolute_ATE_error<br>hidden_treatment_accuracy<br>out_of_fold_and_used_y<br>observed_treatments_preserved<br>source_batch_unchanged<br>unsafe_recipe_rejected | 0.0860431 +/- 0.0198 outcome units<br>0.908346 +/- 0.00224<br>1 +/- 0<br>1 +/- 0<br>1 +/- 0<br>1 +/- 0 | yes |
| 2026-08-27 | `40265928e87a` | absolute_ATE_error<br>hidden_treatment_accuracy<br>out_of_fold_and_used_y<br>observed_treatments_preserved<br>source_batch_unchanged<br>unsafe_recipe_rejected | 0.0860431 +/- 0.0198 outcome units<br>0.908346 +/- 0.00224<br>1 +/- 0<br>1 +/- 0<br>1 +/- 0<br>1 +/- 0 | yes |

## 7. Unknowns

| Unspecified in source | Our choice | Basis |
|---|---|---|
| There is no published causal method called `cycle_dual`. | Preserve the registry name for the smallest safe dual posterior/outcome program and label it project-local. | `PLAN.md` P11 and the pinned XTYLearner entry. |
| The old source does not separate fitting and label generation or record folds. | Five-way cross-fitting over the batch's actual `fold_id`; one fresh initial state per fold. | P10 executor and provenance contract. |
| The old posterior network's exact optimiser budget belonged to an external trainer. | Two ReLU layers of width 128, 500 Adam steps per fold at `1e-3`. | Preserves the old hidden widths and fixes the previously implicit training mechanics before results. |
| The old outcome network concatenated one-hot treatment to `X`; the v1 reusable head has treatment-specific arms. | Reuse `mlp_encoder` plus `tarnet_head`, both with two 128-wide layers and no normalisation or dropout. | Avoids a new outcome parameterisation at Gate 2; the deviation is explicit. |
| No confidence policy is specified. | Hard argmax for every missing row. | Existing P10 action; confidence filtering has no second reviewed consumer. |
| Whether row-wise out-of-fold prediction is sufficient for causal validity under an outcome-dependent imputer. | Treat it as a leakage control, not an identification theorem; report point-estimate error and no confidence interval claim. | `DESIGN.md` section 7.2 plus the absence of a source theorem for this adaptation. |
| No paper-level benchmark transfers to this method. | Use the seed-locked section 6 DGP and retain `published: n/a`. | Same project-local-target policy used by `cnflow` and `mean_teacher`. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status -> `reviewed`) | Matt | 2026-08-24 |
| Plan diffed against section 3.2 and section 4 | Codex | 2026-08-24 |
