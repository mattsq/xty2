# Recipe spec card: scarf

**Status:** `deviating`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> **Agent route:** read §2, §3.2, and §4 to implement; §5 for departures;
> §6 only for benchmark/reporting work. Historical diagnosis lives in Git.

---

## 1. Provenance

| Field | Value |
|---|---|
| Paper | [SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption](https://arxiv.org/abs/2106.15147) |
| Authors, year | Dara Bahri, Heinrich Jiang, Yi Tay, Donald Metzler; 2021 (ICLR 2022) |
| DOI / arXiv | [arXiv:2106.15147](https://arxiv.org/abs/2106.15147); ICLR 2022 (spotlight) |
| Version used | arXiv v2, 2022-03-15, read through the ar5iv HTML rendering. Section 3 defines the method and gives algorithm 1; section 4 gives the encoder, head and optimisation defaults; the ablations fix the corruption rate, the temperature and the batch size. |
| Reference implementation | **None available.** The authors released no code with the paper. Third-party ports exist and were deliberately **not** consulted: an unofficial reimplementation is not evidence about the paper, and citing one as though it settled an ambiguity is the failure `FIDELITY.md` §1 exists to stop. Every row of section 4 is therefore sourced from the paper, and everything the paper leaves open is in section 7 marked as our own choice. |
| Reference impl. runnable? | n/a — none exists. |

## 2. Estimand and claim

- **Estimand:** treatment-specific outcome means after self-supervised encoder pretraining.
- **Method claim:** corrupt exactly `floor(cM)` columns from empirical training marginals, contrast clean against corrupted embeddings with one-directional InfoNCE, discard the projection head, and fine-tune the encoder.
- **Scope:** SCARF's classification result is not reproduced. The causal heads, missing-treatment likelihood, and benchmark are project-local.

## 3. Equations and mapping

### 3.1 As published

The paper numbers no equations in section 3; its method is stated as
algorithm 1, transcribed here in its own notation. Given unlabelled training
data `X = {x^(i)}` with `M` features, corruption rate `c`, temperature `tau`,
batch size `N`, encoder `f` and pre-train head `g`:

> for `j` in `1..M`: let `X̂_j` be the empirical marginal distribution of
> feature `j` — "the uniform distribution over the values that feature takes on
> across the training dataset";
> let `q = floor(c * M)` be the number of features to corrupt;
> for each minibatch and each `i` in `1..N`:
>   draw `I_i` uniformly from the subsets of `{1..M}` of size `q`;
>   set `x̃^(i)_j = v`, `v ~ X̂_j`, for `j` in `I_i`, and `x̃^(i)_j = x^(i)_j`
>   otherwise;
>   `z^(i) = g(f(x^(i)))`, `z̃^(i) = g(f(x̃^(i)))`;
>   `s_{i,j} = z^(i)ᵀ z̃^(j) / (||z^(i)||_2 ||z̃^(j)||_2)`;

$$
\mathcal{L}_{\text{cont}} := \frac{1}{N}\sum_{i=1}^{N}
  -\log \frac{\exp(s_{i,i}/\tau)}
             {\frac{1}{N}\sum_{k=1}^{N}\exp(s_{i,k}/\tau)}
$$

> update `f` and `g` by SGD on `L_cont`; return `f`.

Four properties of that block are load-bearing and are quoted rather than
paraphrased, because each is a place a reimplementation silently differs.

- **The similarity matrix is cross-view and one-directional.** `s_{i,j}` pairs
  the *uncorrupted* `z^(i)` against the *corrupted* `z̃^(j)`. There is no
  second term contrasting `z̃^(i)` against `z^(j)`: the loss is not
  symmetrised, unlike SimCLR's NT-Xent.
- **The denominator includes `k = i`** — the positive pair appears in the
  normaliser — **and carries a `1/N`**, which the SimCLR form does not. Since
  `N` is fixed within a step, the `1/N` is an additive `log N` on the value and
  contributes no gradient; it is implemented anyway, because the number this
  recipe logs should be the number the paper's expression evaluates to.
- **`q` is a count, not a rate per feature.** Exactly `floor(c * M)` features
  are corrupted in every row, drawn without replacement. A per-feature
  Bernoulli(`c`) mask — the obvious reading of "corrupt 60% of features", and
  what `FeatureMask` does — has the same mean and a different variance, and is
  a different augmentation.
- **`g` is discarded.** "After pre-training, `g` is discarded and a
  classification head `h` is applied on top of the learned `f` and both `f` and
  `h` are subsequently fine-tuned for classification." The encoder is *not*
  frozen downstream.

Defaults, from section 4 and the ablations: `c = 0.6` ("we see that performance
is stable when the rate is in the range 50%-80%. We thus recommend a default
setting of 60%"), `tau = 1` ("while prior work considers temperature an
important hyperparameter that needs to be tuned, we see that a default of 1
[...] works the best in our setting"), `f` = 4 layers of width 256 with ReLU,
`g` = 2 layers of width 256 with ReLU whose output is `l2`-normalised onto the
unit hypersphere, Adam at its default learning rate 0.001, batch size 128.

### 3.2 Mapping to xty2

The pretrain stage maps clean and corrupted realisations through `MLPEncoder` and `ProjectionHead` to `X_PROJ`; `InfoNCEContrastive` is batch-coupled. `joint_fit` initializes from pretraining, omits the projection head, and fine-tunes the encoder with the causal heads.

| Paper symbol | Meaning | xty2 Port | xty2 Objective / Component |
|---|---|---|---|
| `x^(i)` | an uncorrupted row | `X_RAW` | the virtual source node under the `identity` realisation |
| `x̃^(i)` | its corrupted copy | `X_RAW @ corrupted_x` | `ViewSpec("corrupted_x")` over `FeatureCorruption(rate=0.6)` |
| `X̂_j` | feature `j`'s empirical marginal | — | the batch's own column `j`, resampled row-wise (deviation 2) |
| `q = floor(c M)` | features corrupted per row | — | `FeatureCorruption`, computed from the schema's mutable columns |
| `f` | encoder | `X_RAW -> X_REPR` | `MLPEncoder` (the reviewed P5 backbone; deviation 3) |
| `g` | pre-train head | `X_REPR -> X_PROJ` | `ProjectionHead`, 2 layers of 256, ReLU, `l2`-normalised |
| `z^(i)` | embedding of the row | `X_PROJ @ identity` | `InfoNCEContrastive.anchor` |
| `z̃^(i)` | embedding of its corruption | `X_PROJ @ corrupted_x` | `InfoNCEContrastive.contrast` |
| `s_{i,j}` | cosine similarity | — | inside that objective; a `plan_details` line |
| `tau` | temperature | — | `InfoNCEContrastive(temperature=1.0)`, binding `losses.temperature` |
| `L_cont` | the contrastive loss | `X_PROJ` at both realisations | `InfoNCEContrastive`, stage `pretrain`, rows `all`, `reduction="mean"` |
| "return `f`" / "`g` is discarded" | what survives pretraining | — | `Stage("joint_fit", initialise_from="pretrain")`; `projection_head` is in no forward pass of that stage and in no `trainable` list |
| "both `f` and `h` are fine-tuned" | the encoder is not frozen | — | `joint_fit.trainable` names `mlp_encoder` |
| — (project-local) | outcome likelihood | `Y_GIVEN_XT` | `ObservedOutcomeNLL`, rows `t_observed` |
| — (project-local) | treatment likelihood | `T_GIVEN_X` | `ObservedTreatmentNLL`, rows `t_observed` |
| — (project-local) | exact marginalisation over missing `t` | `T_GIVEN_X`, `Y_GIVEN_XT` | `MissingTreatmentMarginalNLL(grad_path="both")`, rows `t_missing` |

## 4. Mechanics checklist

This YAML is the executable fidelity contract. Keep its keys synchronized with the recipe and tests.

```yaml
gradients:
  stop_gradients:
    pretrain.info_nce_contrastive: none          # algorithm 1 descends both branches; SCARF is not BYOL/SimSiam
    joint_fit.observed_outcome_nll: none
    joint_fit.observed_treatment_nll: none
    joint_fit.missing_treatment_marginal_nll: none
  detached_targets: n/a                          # no consistency target; the contrastive loss has no target side
  gradient_clipping:
    pretrain: none                               # paper names none
    joint_fit: none
  marginal_nll_grad_path:
    joint_fit.missing_treatment_marginal_nll: both   # reviewed P5 choice; project-local addition

teacher:
  ema_decay: n/a                                 # SCARF maintains no EMA
  ema_applies_to_buffers: n/a
  teacher_in_train_mode: n/a
  teacher_requires_grad: n/a

losses:
  reduction:
    pretrain.info_nce_contrastive: mean          # L_cont is (1/N) sum over the batch, i.e. the term's own rows
    joint_fit.observed_outcome_nll: population
    joint_fit.observed_treatment_nll: population
    joint_fit.missing_treatment_marginal_nll: population
  eligible_rows:
    pretrain.info_nce_contrastive: all           # self-supervised: no label of any kind is read
    joint_fit.observed_outcome_nll: t_observed
    joint_fit.observed_treatment_nll: t_observed
    joint_fit.missing_treatment_marginal_nll: t_missing
  weights:
    pretrain.info_nce_contrastive: 1.0           # the only term in its stage
    joint_fit.observed_outcome_nll: 1.0
    joint_fit.observed_treatment_nll: 1.0
    joint_fit.missing_treatment_marginal_nll: 0.5
  schedules:
    pretrain.info_nce_contrastive: constant 1.0
    joint_fit.observed_outcome_nll: constant 1.0
    joint_fit.observed_treatment_nll: constant 1.0
    joint_fit.missing_treatment_marginal_nll: ramp 0.0 -> 0.5 over 1000 steps
  temperature:
    pretrain.info_nce_contrastive: 1.0           # section 4 ablation: "a default of 1 ... works the best in our setting"
  sharpening: n/a                                # no pseudo-label is formed anywhere in this recipe
  confidence_threshold: n/a

optimisation:
  optimiser:
    pretrain: adam(betas=(0.9, 0.999), eps=1e-08)     # "the Adam optimizer using the default learning rate of 0.001"
    joint_fit: adam(betas=(0.9, 0.999), eps=1e-08)    # same optimiser for fine-tuning, section 4
  lr:
    pretrain: 0.001
    joint_fit: 0.001
  lr_schedule:
    pretrain: constant 1.0                       # the paper names no schedule
    joint_fit: constant 1.0
  weight_decay:
    pretrain: none                               # the paper names none
    joint_fit: none
  batch_size: 128                                # N, the paper's; and L_cont's negative count is N - 1
  labelled_unlabelled_ratio: n/a                 # UniformSampler enforces no quota; pretraining reads no labels at all
  total_steps_or_epochs:
    pretrain: 1000                               # optimiser steps, never epochs; see deviation 4
    joint_fit: 3000

architecture:
  widths_depths:
    mlp_encoder: [200, 200, 200]                 # retained reviewed P5 TARNet backbone; deviation 3
    projection_head: [256, 256]                  # g: "2 layers", "hidden dimension 256"
    tarnet_head: K independent heads, each [100, 100, 100]
    categorical_propensity: linear X_REPR -> K
  activation:
    mlp_encoder: elu
    projection_head: relu                        # section 4: ReLU
    tarnet_head: elu
    categorical_propensity: linear logits
  normalisation:
    mlp_encoder: row_l2
    projection_head: row_l2                      # "the pre-train head network l2-normalizes the outputs so that they lie on the unit hypersphere"
    tarnet_head: none
    categorical_propensity: none
  dropout:
    mlp_encoder: 0.0
    projection_head: 0.0                         # the paper names none
    tarnet_head: 0.0
    categorical_propensity: 0.0
  initialisation:
    mlp_encoder: normal std=0.1/sqrt(fan_in), bias=0
    projection_head: normal std=0.1/sqrt(fan_in), bias=0
    tarnet_head: normal std=0.1/sqrt(fan_in), bias=0
    categorical_propensity: normal std=0.1/sqrt(fan_in), bias=0
  output_parameterisation:
    tarnet_head: K means; fixed Gaussian scale=1.0
    categorical_propensity: K softmax logits

data:
  standardisation: x: none fitted on 'train'     # the section 6 DGP draws standardised features
  outcome_scaling: y: zscore fitted on 'train'   # held-out rows take the same fitted transform, never a refitted one
  treatment_encoding: n/a                        # XTYBatch contract supplies integer classes 0..K-1
  split_protocol: one fixed project-local DGP, split train/test by the section 6 fixture; no OpenML-CC18 protocol (deviation 7); training rows are assignment 'train'
  missingness_mechanism: treatment MCAR to a budget of 40 labelled rows, keyed by row_id  # section 6
```

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Fine-tune into the reviewed xty2 causal stack (outcome NLL, treatment NLL, exact marginalisation over missing `t`) rather than the paper's classification head `h`. | The paper's downstream task is supervised classification. The project-local question is whether SCARF's representation helps the *treatment*-scarce XTY problem, which is the closest analogue of the semi-supervised regime section 4 of the paper reports its largest gains in. The `p(t \| x)` head is a classifier, so the analogue is exact for the metric section 6 leads with. | No published number applies. The comparison is internal: the same stage, same seeds and same batches, with and without the pretrained initialisation. |
| 2 | `withdrawn` | — | ~~The empirical marginal `X̂_j` is taken over the **batch** the view is transforming, not over the training dataset.~~ **Withdrawn.** `FeatureCorruption` draws each replacement from the training population's column. | This was the open question §5.1 put to a reviewer, and the loader is what settled it: `TrainingPopulation` exists now, so a transform has a training set to read and the argument for deferring — that building the population from one transform's evidence would fix the shape wrongly — no longer applies. `ViewTransform.apply` takes the population, and `FeatureCorruption` **requires** it rather than falling back to the batch, so the deviation cannot return as a silent default. | The tail the old row named is reachable: a value held by fewer than one row in `B` can now be drawn, which `tests/invariants/test_scarf.py` asserts directly. §6.2's numbers were measured under the batch-local draw and are re-measured with the rest of this card. |
| 3 | `judgement` | — | Retain the reviewed P5 encoder — 3 layers of 200 with ELU and row-`l2` normalisation — rather than the paper's 4 layers of 256 with ReLU. Take the pre-train head `g` from the paper (2 layers of 256, ReLU, `l2`-normalised). | Holding the causal stack fixed across cards is what makes an addition attributable, and is the same decision `mean_teacher.md` deviation 10 and `fixmatch.md` deviation 6 record. `g` is not part of that stack — it exists only because SCARF does — so it is taken as published. | Both arms of section 6's pair share the encoder, so the comparison is unaffected. An absolute comparison against the paper's numbers was never available. |
| 4 | `judgement` | — | Fixed budgets of 1,000 pretraining and 3,000 fine-tuning optimiser steps, rather than "a max number of pre-train epochs of 1000" with "early stopping with patience 3 on the validation loss" and a max of 200 fine-tuning epochs early-stopped on validation classification error. | This row was typed `framework-limitation` in the card's first draft, on the true observation that a `Stage` runs `steps` optimiser steps (`DESIGN.md` §7) and has nowhere to put a validation split. That is the wrong test. `FIDELITY.md` §5's is "would we choose the same again given an infinite framework", and we would: every card in this repository fixes a project-local step budget so that a difference between recipes is attributable to the recipe (`fixmatch.md` §5.3 is the same call), and section 6's target is a *paired* comparison in which both arms get the same budget either way. Section 6.2 also measured the fine-tuning half directly — four budgets from 150 to 3,000 steps — and the result does not turn on it. Typing a decision we would make again as a debt would have put a creditor on the ledger who is owed nothing, which is its own kind of dishonesty. | Pretraining length is chosen by us rather than by the data. Section 6.2's budget sweep covers the fine-tuning half; the pretraining half is not swept, and the `L_cont` trace bottoming near step 300 says a validation-stopped run would have stopped earlier than 1,000. |
| 5 | `judgement` | — | Corruption is restricted to columns the schema marks `mutable`, and `M` counts those columns only. | `FeatureSpec.mutable=False` is absolute in xty2 (`DESIGN.md` §5) and a view that overrode it would be able to produce rows the schema declares impossible. On the section 6 schema every column is mutable, so `M` is the paper's `M` there. | None on the section 6 fixture. On a schema with immutable columns, fewer features are corrupted than `floor(0.6 * M_all)` — recorded so that a later card on such a schema does not read the rate as if it applied to every column. |
| 6 | `withdrawn` | — | ~~Nothing enforces a batch size, and SCARF's loss depends on it: the number of negatives is `N - 1`.~~ **Withdrawn.** Both stages declare `UniformSampler(batch_size=128)` and `optimisation.batch_size` binds the paper's `N`. | xty2 has a loader. The guard is stronger than the binding on its own: `InfoNCEContrastive` declares itself `batch_coupled`, and a stage holding a batch-coupled term is *refused* the `ExternalBatches` declaration at compile time — so this recipe could not hand the number back to a caller even if a later edit tried. A caller feeding 16-row batches is no longer expressible. | The section 6 results below were measured under the pre-loader batch stream and are **invalidated pending re-measurement**: the recipe now draws its own batches from a seed-derived stream, and it also owns the 40-label budget and the outcome scaling the fixture used to apply. The sampling scheme is unchanged (`tests/invariants/test_loading.py` pins it), so the paired comparison should stand; that is a prediction, not a result. |
| 7 | `judgement` | — | No label-noise and no OpenML-CC18 protocol; one fixed project-local DGP, in section 6. | The paper's evidence is 69 datasets under three label regimes. Reproducing that shape is a Tier 2 question about data plumbing, not about whether the mechanism is assembled correctly, and no dataset in it carries a treatment. | Section 6 is a mechanism target and says so. It is not evidence for the paper's claim, only against this port being miswired. |

### 5.1 Framework impact

`scarf` introduced `X_PROJ`, `ProjectionHead`, `InfoNCEContrastive`, the required `batch_coupled` declaration, and population-aware feature corruption.

### Tier 2 outcome

On 2026-08-27, commit `40265928e87a` produced a `deviating` result: This is the predeclared project-local SCARF mechanism target: does an encoder trained only on the covariance of x help a scarce-label treatment fit. It is not a reproduction of Bahri et al., whose evidence is 69 OpenML-CC18 datasets under three label regimes and whose downstream task carries no treatment. Within noise of the target: held_out_treatment_NLL_ratio was 0.999111 +/- 0.038 against mean <= 1, by at least one stderr, inside its target by 0.000889 — less than its own standard error, so the run does not distinguish it from a miss.

## 6. Reproduction target

The pair compares SCARF pretraining with no pretraining on a fixed project-local DGP.

```yaml
reproduction:
  dataset: project-local seed-locked two-cluster XTY DGP (6 features, K=2), specified in 6.1
  variant: paired fit against the identical joint_fit stage with no pretraining, same seeds and same batches
  split: 1024 train rows with 40 observed treatments, 2048 held-out rows with every treatment observed
  metric: held-out p(t|x) NLL ratio, pretrained over unpretrained; positive-pair alignment of the pretrained encoder as a mechanism guardrail
  published: none - no published number applies to this adaptation
  published_source: n/a
  tolerance: NLL ratio < 1.0 in mean; held-out outcome NLL within 1.05x of the unpretrained arm; mean cosine similarity of a row to its own corrupted view at least 0.2 above its mean similarity to the other rows of the batch
  seeds: 10
  report: mean_and_stderr
```

### 6.3 Result ledger


| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| 2026-08-27 | `1a10fb039e5f` | held_out_treatment_NLL_ratio<br>held_out_outcome_NLL_ratio<br>terminal_alignment_minus_uniformity | 0.999111 +/- 0.038<br>1.0135 +/- 0.00984<br>0.509232 +/- 0.0148 | yes |
| 2026-08-27 | `40265928e87a` | held_out_treatment_NLL_ratio<br>held_out_outcome_NLL_ratio<br>terminal_alignment_minus_uniformity | 0.999111 +/- 0.038<br>1.0135 +/- 0.00984<br>0.509232 +/- 0.0148 | no |

## 7. Unknowns

| Unspecified in paper | Our choice | Basis |
|---|---|---|
| Whether `I_i` is drawn independently per row or once per batch. | Independently per row. | Algorithm 1 draws `I_i` inside the loop over `i`, and the subscript is the row's. Recorded because a per-batch mask is the cheaper implementation and would be invisible in a diff. |
| Whether the marginal draw for a corrupted cell is independent per `(row, feature)` or one donor row is used for all of a row's corrupted features. | Independent per `(row, feature)`. | "`x̃_j^(i) = v`, where `v ~ X̂_j`" is written per feature `j`, and `X̂_j` is a distribution over one column. Drawing one donor row instead would preserve that row's cross-feature dependence, which is exactly what the corruption is meant to destroy. |
| The initialisation of `f` and `g`. | The reviewed CFRNet initialisation this repository's other components use: `normal(std=0.1/sqrt(fan_in))`, zero bias. | Project convention. The paper names none, and `g`'s output is `l2`-normalised, so the scale of its initialisation affects the first few steps and not the geometry. |
| Whether the pre-train head's last layer is followed by an activation. "2 layers, hidden dimension 256" fixes the count and the width and not the shape of the output. | No terminal activation: `Linear -> ReLU -> Linear`, then the `l2` normalisation. | The convention every projection head this lineage descends from uses (SimCLR's `g(h) = W2 sigma(W1 h)`), and — after the alternative was implemented by accident and measured — the only one that trains. A terminal ReLU confines the embedding to the non-negative orthant, where two rows can be orthogonal but never opposed, so the loss's only route to a low off-diagonal similarity is disjoint sparse supports: 99.6% of the head's units were zero by step 1,000, alignment fell to 0.11 and `L_cont` climbed back toward the collapsed value of 0. Recorded because the failure looks like ordinary optimisation noise in a loss curve and is not. |
| Whether the pre-train head is discarded or retained for a later stage. | Discarded, and *provably* so: `projection_head` appears in no `joint_fit` forward pass and in no `trainable` list, so the compiler would reject it as dead weight if a later edit tried to train it without using it. | The paper: "after pre-training, `g` is discarded". |
| Weight decay during either phase. | None. | The paper names only "the Adam optimizer using the default learning rate of 0.001". A decay nobody stated would be a hyperparameter this card could not source. |
| Whether the fine-tuning phase re-uses the pretraining optimiser state. | No — each stage constructs its own optimiser. | `DESIGN.md` §7.0: a stage begins from the recipe's initial graph state overlaid with the named checkpoint, and a `Checkpoint` carries parameters and buffers, not optimiser moments. The paper is silent, and carrying Adam moments across a change of objective would be the surprising choice. |
| How many rows the pretraining sees relative to the fit. | The same stream: both stages draw from the same 1,024 training rows in section 6. | The paper pretrains on the same dataset it fine-tunes on (its semi-supervised setting pretrains on all rows and fine-tunes on the labelled subset). Here every row is available to both stages, because `pretrain` reads no labels and the label scarcity is on `t`. |

## 8. Review

| | Who | Date |
|---|---|---|
| Card reviewed (status → `reviewed`) | | |
| Plan diffed against §3.2 and §4 | | |
