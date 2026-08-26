# Recipe spec card: scarf

**Status:** `draft`
<!-- draft | reviewed | implemented | smoke-passing | reproduced | deviating -->

> This card was written before the recipe, the port, the component, the view
> transform and the objective it describes, and committed on its own so that
> the card-first gate of `CLAUDE.md` rule 1 has something to be a gate *on*.
> The implementation follows in the same branch. Section 8 is unsigned: nothing
> below has been reviewed, and a reviewer moving section 8 is what moves this
> status line.
>
> The code is at what `FIDELITY.md` §1.1 calls `smoke-passing`. Read section 6.2
> before reading section 6: the mechanism is assembled and works as a
> representation learner, and the **downstream target section 6 declares is not
> met** — measured over five seeds and four fine-tuning budgets, and recorded
> rather than tuned away. Sections 2 and 6.2 were rewritten around that
> measurement; the target itself is left exactly as it was declared.

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

- **Estimand:** SCARF itself estimates nothing. It is a pretraining objective
  that produces an encoder `f`; the estimand belongs to what is fitted on top
  of it. In this recipe that is the reviewed xty2 stack: the categorical
  propensity `p(t | x)`, the treatment-conditional outcome `p(y | x, t=k)` and
  its means `mu_k(x)`, whose contrasts identify conditional treatment effects
  under consistency, positivity and conditional exchangeability.
- **Claim:** the paper claims that contrasting a row against a *corrupted* copy
  of itself — a random subset of its features replaced by draws from those
  features' empirical marginals — pretrains an encoder that improves downstream
  classification on 69 OpenML-CC18 tabular datasets, and that the improvement
  is largest where labels are scarce or noisy. This card claims two much
  smaller things, and the second is smaller than the draft of this section
  claimed before section 6.2 was measured:
  1. the mechanism is faithfully assembled in xty2 — the corrupted view reaches
     the loss, the loss is the paper's expression, and the representation it
     produces separates a row from the other rows of its batch rather than
     collapsing;
  2. that representation carries structure the *treatment* head can use: with
     the encoder held fixed, `p(t | x)` fitted on it beats the same fit on an
     untrained encoder (section 6.2, mean ratio 0.87 over five seeds).

  It does **not** claim the thing section 6 declares a target for — that
  pretraining improves the end-to-end scarce-label fit the paper's own protocol
  prescribes. That was measured, at four fine-tuning budgets and five seeds,
  and it is absent. Section 6.2 has the numbers and what they appear to mean.
- **Not claimed:** no OpenML-CC18 number is claimed, and none could be: the
  datasets, the architecture, the metric and the label are all different. Two
  further limits are structural and are stated here rather than left to
  section 5:
  1. **SCARF is not a causal method and this recipe does not make it one.**
     The pretraining stage reads `X_RAW` alone. It cannot leak the outcome into
     a treatment label, because it produces no label and never touches `Y_RAW`
     — `DESIGN.md` §7.2's static rule is not engaged at all, which is a
     property of the graph here and not a promise. What it *can* do is shape a
     representation around whatever dominates the covariance of `x`, which is
     not necessarily what predicts `t` or `y`. Where the two disagree,
     pretraining is a cost rather than a benefit, and the paired ablation in
     section 6 is what would show it.
  2. **Instance discrimination is not class discrimination, and on this
     fixture they are in tension.** SCARF's loss treats every other row of the
     batch as a negative, including rows that share the cluster — and on the
     section 6 DGP the cluster is what the treatment is assigned from. The
     objective is therefore spending capacity pushing apart exactly the rows a
     propensity head wants together, which section 6.2 measures directly: the
     mean similarity between different rows falls to 0.002 while a row's
     similarity to its own corrupted copy stays near 0.48. This is a property
     of unsupervised instance discrimination rather than a fault in the port,
     and it is the reason the backlog's next contrastive cards — CoMatch,
     SimMatch, PAWS — put class or pseudo-label information into the
     contrastive graph.
  3. **The corrupted view is a statement about the feature distribution, not
     about the treatment.** Replacing a feature by a draw from its marginal
     destroys that feature's dependence with every other column, including any
     column the treatment assignment depends on. The view is legitimate for
     representation learning precisely because no target is attached to it; it
     would not be a legitimate weak/strong augmentation pair for a consistency
     loss on `p(t | x)`, and this recipe does not use it as one.

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

Three mapping decisions carry the fidelity of this port and are argued rather
than asserted.

**The anchor is the `identity` realisation, not a second corrupted view.**
Algorithm 1 embeds `x^(i)` itself, uncorrupted, and contrasts it against the
corrupted copies. That is one of the two places SCARF differs from SimCLR,
where both sides are augmented, and it is why this recipe declares one view
rather than two. A recipe that corrupted both sides would need
`ViewSpec.draws = 2` and would be a different method.

**The denominator's negatives are the *eligible* rows, and nothing else.**
`DESIGN.md` §4 gives an objective a `RowIndex` and says the term is a mean over
it. For a contrastive loss the row set does double duty: it is the set of
anchors *and* the set of candidates, because a negative drawn from a row this
objective is not entitled to would be reading outside its declared population
by another route. With this recipe's `rows: all` and a stage scope of `all` the
distinction never bites — `N` is the batch — but it is a real choice and it is
emitted as a `plan_details` line rather than left to be inferred.

**The pretraining stage's step count is the paper's `N` in disguise.** SCARF's
loss couples every row in the batch to every other, so unlike every other
objective in this repository its *value* depends on the batch size. xty2 has no
loader (`DESIGN.md` §11.4, `loader`), so the batch is whatever the caller
supplies; section 6 fixes 128 to match the paper, and section 5 deviation 6
records that nothing enforces it.

## 4. Mechanics checklist

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
  batch_size: n/a                                # external BatchSource; section 6 fixes 128, the paper's N
  labelled_unlabelled_ratio: n/a                 # pretraining reads no labels; the fit takes what the data gives
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
  standardisation: n/a                           # caller-owned; section 6 records the fixed choice
  outcome_scaling: n/a                           # caller-owned; section 6 records the fixed choice
  treatment_encoding: n/a                        # XTYBatch contract supplies integer classes 0..K-1
  split_protocol: n/a                            # Tier 1 fixture owns splits; section 6 fixes them
  missingness_mechanism: n/a                     # section 6 fixes treatment MCAR; the recipe consumes t_observed
```

The recipe declares one view. `corrupted_x` is
`FeatureCorruption(rate=0.6, columns=None)`, which corrupts
`floor(0.6 * M)` of the `M` mutable feature columns in every row, replacing
each with a value that column takes somewhere else in the same batch. It
preserves `t`, `y`, `t_observed`, `y_observed`, `row_id`, `fold_id` and
`weight`, and does not claim to preserve `x`. Immutable columns are never
touched and are not counted in `M`; a schema with derived features must supply
recompute rules or the view is rejected at compile time, and
`scarf(schema, recompute_rules=(...))` is how they arrive.

`projection_head`'s `[256, 256]` is two *layers*, and the second of them is
affine: `Linear(200, 256) -> ReLU -> Linear(256, 256)`, then the `l2`
normalisation. The count and the width come from the paper; the absence of a
terminal activation is §7's, and it is the difference between a head that
trains and one that dies.

One property of the corruption transform is worth stating because no other view
in the repository has it: **a corrupted value is always a value the column actually
took.** Bounds, kinds and any implicit support constraint hold by construction
rather than by a clamp, which is what makes the augmentation defensible on
tabular physical data where a jitter or a constant fill can produce an
impossible row.

## 5. Deviations from the paper

| # | Kind | Blocked on | What we do differently | Why | Expected effect on the section 6 metric |
|---|---|---|---|---|---|
| 1 | `judgement` | — | Fine-tune into the reviewed xty2 causal stack (outcome NLL, treatment NLL, exact marginalisation over missing `t`) rather than the paper's classification head `h`. | The paper's downstream task is supervised classification. The project-local question is whether SCARF's representation helps the *treatment*-scarce XTY problem, which is the closest analogue of the semi-supervised regime section 4 of the paper reports its largest gains in. The `p(t \| x)` head is a classifier, so the analogue is exact for the metric section 6 leads with. | No published number applies. The comparison is internal: the same stage, same seeds and same batches, with and without the pretrained initialisation. |
| 2 | `framework-limitation` | `view-population-statistics` | The empirical marginal `X̂_j` is taken over the **batch** the view is transforming, not over the training dataset. | A `ViewSpec` transform is a pure function of `(batch, rng_key)` (`DESIGN.md` §5) and there is no training-population object anywhere in xty2 for it to read: the gradient executor takes an iterable of batches. Sampling column `j` from the batch is a draw from the *batch's* empirical marginal, which is itself a uniform subsample of the training one when batches are drawn uniformly — so the corrupted value is still a real observed value of that feature, and the two distributions agree in expectation over batches. What is lost is the tail: a value held by fewer than one row in `B` cannot be drawn into the batch it is not in. | Small and in the direction of *less* corruption diversity at small batch sizes, which the paper's batch-size ablation suggests matters little above 128 — the size section 6 fixes. It would matter more on a heavy-tailed column, which the section 6 DGP does not have. |
| 3 | `judgement` | — | Retain the reviewed P5 encoder — 3 layers of 200 with ELU and row-`l2` normalisation — rather than the paper's 4 layers of 256 with ReLU. Take the pre-train head `g` from the paper (2 layers of 256, ReLU, `l2`-normalised). | Holding the causal stack fixed across cards is what makes an addition attributable, and is the same decision `mean_teacher.md` deviation 10 and `fixmatch.md` deviation 6 record. `g` is not part of that stack — it exists only because SCARF does — so it is taken as published. | Both arms of section 6's pair share the encoder, so the comparison is unaffected. An absolute comparison against the paper's numbers was never available. |
| 4 | `framework-limitation` | `early-stopping` | Fixed budgets of 1,000 pretraining and 3,000 fine-tuning optimiser steps, rather than "a max number of pre-train epochs of 1000" with "early stopping with patience 3 on the validation loss" and a max of 200 fine-tuning epochs early-stopped on validation classification error. | A `Stage` runs `steps` optimiser steps (`DESIGN.md` §7) and there is nowhere for a validation split or a monitored metric to live, so the paper's stopping rule cannot be stated. The budgets are the project-local ones every other card uses. | Pretraining length is chosen by us rather than by the data. Under-training weakens the effect section 6 measures and over-training risks the representation drifting from what the downstream fit needs; the paired design means both arms share the fine-tuning budget, so only the pretraining half of this is a confound. |
| 5 | `judgement` | — | Corruption is restricted to columns the schema marks `mutable`, and `M` counts those columns only. | `FeatureSpec.mutable=False` is absolute in xty2 (`DESIGN.md` §5) and a view that overrode it would be able to produce rows the schema declares impossible. On the section 6 schema every column is mutable, so `M` is the paper's `M` there. | None on the section 6 fixture. On a schema with immutable columns, fewer features are corrupted than `floor(0.6 * M_all)` — recorded so that a later card on such a schema does not read the rate as if it applied to every column. |
| 6 | `framework-limitation` | `loader` | Nothing enforces a batch size, and SCARF's loss depends on it: the number of negatives is `N - 1`. | xty2 has no loader, so `optimisation.batch_size` is a key nothing can check and the field does not exist on `Stage`. Section 6 fixes 128 in the fixture, which is where the paper's ablation finds the curve flat. | The recipe as declared is correct at any batch size and *means* something slightly different at each. A caller feeding 16-row batches would be running a much easier contrastive task than the card describes. |
| 7 | `judgement` | — | No label-noise and no OpenML-CC18 protocol; one fixed project-local DGP, in section 6. | The paper's evidence is 69 datasets under three label regimes. Reproducing that shape is a Tier 2 question about data plumbing, not about whether the mechanism is assembled correctly, and no dataset in it carries a treatment. | Section 6 is a mechanism target and says so. It is not evidence for the paper's claim, only against this port being miswired. |

### 5.1 Framework additions made for this card

| Added | Quadrant (§11.2) | Consumers today | Named second consumer | Why now |
|---|---|---|---|---|
| `Port.X_PROJ`, `[B, H]` — the embedding space a contrastive loss is computed in | fidelity-bearing, **load-bearing vocabulary** | This card, through `ProjectionHead` and `InfoNCEContrastive` | **CoMatch** (`BACKLOG.md` §2.7, high-priority tranche). Li, Xiong & Hoi §3: "We also define a non-linear projection head (a MLP) `g(·)`, which transforms a feature `f(x)` into a normalized low-dimensional embedding `z(x) = g(f(x))`" — the same port, the same producer shape, and the same discarded-after-training lifecycle. The shape was checked against it in two places: CoMatch reads `z` under *two* realisations of the same batch and pairs it with `p(y\|x)` from a classification head on the same `f`, so the port must be per-realisation (it is — `State` is keyed by realisation) and must coexist with `T_GIVEN_X` over one encoder (it does — they are separate components over `X_REPR`). CoMatch's embedding is 64-dimensional against SCARF's 256, which is why the shape contract is `[B, H]` with a free width rather than anything the schema fixes | Without it the contrastive loss would run on `X_REPR` and the pre-train head `g` — a component the paper specifies, and specifies discarding — could not exist. That is a card §4 `architecture.widths_depths` row that could not be honoured, which is §11.2 Q1 answered yes |
| `ProjectionHead` (`X_REPR -> X_PROJ`) | fidelity-bearing, reversible | This card | — (not required for this quadrant) | It is `g`. Deliberately a separate class rather than a widened `MLPEncoder`: every recorded number in this repository depends on `MLPEncoder`'s construction-time RNG consumption, and a shared base class would put a reviewed component's initialisation at risk to save forty lines of validation |
| `FeatureCorruption` view transform | fidelity-bearing, reversible | This card | — | It is the paper's corruption, and it is the whole method. `FeatureMask` is not a substitute: it fills with a constant rather than a marginal draw, and it masks each cell independently rather than exactly `floor(cM)` per row |
| `InfoNCEContrastive` objective | fidelity-bearing, reversible | This card | — | It is `L_cont`. It is the first objective in the repository whose per-row value depends on the other rows of the batch, which is why its negatives-are-the-eligible-rows rule is written into §3.2 and into `plan_details` rather than left implicit |

`X_PROJ` is the one row in the load-bearing quadrant, and the obligation
§11.2 attaches to that quadrant is discharged in the column above rather than
gestured at: the second consumer is named, the sentence of its paper that needs
the port is quoted, and the two places the shape was checked against it are
stated. Nothing else here is vocabulary a future recipe must be written
against — a transform, a component and an objective are each additions to a
registry that existing recipes never read.

Two ledger rows in `DESIGN.md` §11.4 are new with this card:
`view-population-statistics` (deviation 2) and `early-stopping` (deviation 4).
Both are written with the evidence that would change the decision, as
`FIDELITY.md` §5.1 requires of a `framework-limitation` with no row to cite.

## 6. Reproduction target

The published OpenML-CC18 accuracies cannot validate this port, for the reasons
section 2 gives. The target below is a completely fixed project-local
*mechanism* target, in the same form as `fixmatch.md` §6, and it is a **paired**
comparison: the same `joint_fit` stage, the same seeds, the same batch stream
and the same initial parameters, run once from the SCARF-pretrained encoder and
once from the recipe's initialisation. Passing it supports the limited claim in
section 2; it must not be described as reproducing Bahri et al.

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

### 6.1 Fixed DGP

The mechanism under test is "does an encoder trained only on the covariance of
`x` help a scarce-label treatment fit", so the DGP is one where the answer is
*not* trivially yes: the cluster structure that dominates `x` is also what
drives assignment, but four of six features carry it and two do not, and only
40 of 1,024 rows keep their treatment. It is deliberately the DGP of
`fixmatch.md` §6.1, unchanged, so that two cards' §6 numbers are about the
recipes rather than about two different worlds:

```text
cluster   c = 1[u_c < 0.5]
x[:, 0:4] = 0.45 * (2c - 1) + 0.6 * eps_x[:, 0:4]    # redundant cluster signal
x[:, 4:6] = eps_x[:, 4:6]                            # outcome-only covariates
p(t=1|c)  = 0.02 + 0.96 * c                          # near-deterministic assignment
t         = 1[u_t < p(t=1|c)]
baseline  = 0.5*x[:,0] - 0.3*x[:,1] + 0.2*(x[:,4]^2 - 1)
effect    = 1.0 + 0.5*tanh(x[:,2])
y         = baseline + t * effect + 0.5 * eps_y
```

Treatment observation is MCAR: exactly 40 of the 1,024 training rows keep their
`t`. The outcome is standardised by the training mean and standard deviation and
the same constants are applied to the held-out rows. Batches are **128 rows** —
the paper's `N`, and the one place this fixture departs from `fixmatch.md`
§6.1's 256, because the batch size is a hyperparameter of SCARF's loss
(deviation 6) and not merely of the loader.

The outcome-side guardrail is stated as non-inferiority against the unpretrained
arm rather than as an absolute band, for the reason `fixmatch.md` §6 gives about
this DGP: under a 0.02/0.98 propensity the counterfactual arm within a cluster
is nearly unobserved, so `sqrt_PEHE` is not identified at this sample size and
claiming a band for it would be claiming a number the design cannot support.

### 6.2 What the Tier 1 fixture already shows, including against §6

These are Tier 1 numbers from `tests/smoke/test_scarf.py` and the paired sweep
it was written from — not a Tier 2 result, and not evidence about Bahri et al.
They are recorded because they answer §6's question in the direction §6 did not
predict, and because two of them rewrote §2.

All rows use the §6.1 DGP. "Seeds" are five distinct (data, initialisation,
batch-stream) seeds; the Tier 1 fixture is the first of them. Every pair shares
its initial parameters and its batch stream, so the only difference between the
arms is the pretraining.

**The pretraining itself works.** On the fixture seed, `L_cont` falls from
`-0.234` to `-0.370` over 1,000 steps while the similarity of a row to its own
corrupted copy stays high and its similarity to the other rows of the batch goes
to nothing:

| | step 0 | step 999 |
|---|---|---|
| `L_cont` | -0.234 | -0.370 |
| alignment (a row vs its own corrupted copy) | 0.590 | 0.479 |
| uniformity (a row vs the other rows) | 0.325 | 0.002 |

The gap is 0.477 against §6's declared 0.2, and it is not the gap the network
was born with. This is also the measurement that distinguishes learning from
collapse: a collapsed encoder drives `L_cont` to exactly 0 — *higher* than the
untrained network's -0.234 — with alignment and uniformity equal, which is what
an earlier version of the projection head actually did (§7, terminal
activation).

**The declared §6 target is not met.** Held-out `p(t | x)` NLL, pretrained arm
against the no-pretraining arm, at four fine-tuning budgets:

| `joint_fit` steps | mean NLL ratio, pretrained / not | per-seed pairs |
|---|---|---|
| 150 | 1.042 | 0.252/0.245, 0.313/0.300, 0.346/0.356, 0.357/0.299, 0.323/0.334 |
| 300 | 1.004 | 0.231/0.229, 0.295/0.287, 0.380/0.368, 0.328/0.304, 0.278/0.318 |
| 1,000 | 1.084 | 0.324/0.265, 0.402/0.383, 0.499/0.458, 0.324/0.314, 0.444/0.432 |
| **3,000 (the recipe)** | **1.081** | 0.559/0.564, 0.932/0.682, 0.677/0.708, 0.605/0.578, 0.727/0.696 |

The marginal-frequency baseline is 0.70 on every seed. §6's tolerance is a mean
ratio below 1.0 and no budget reaches it, so the shorter budgets are reported
for what they rule out rather than as an alternative to be adopted: the null is
not an artefact of deviation 4's fixed 3,000-step fine-tune. The held-out
outcome NLL ratios at 3,000 steps are 0.996, 1.030, 1.023, 0.980 and 1.042, so
§6's non-inferiority guardrail holds — the pretraining is not damaging the
outcome stack, it is simply not helping the treatment one.

**The representation is not empty, though — the fine-tuning erases it.** The
same pair with the encoder *frozen* during the fit, which is the standard
linear-probe question asked with the framework's own vocabulary (it is a
measurement, not the recipe: SCARF fine-tunes `f`, and freezing it to move a
number would be the tuning this project refuses):

| | pretrained | untrained | |
|---|---|---|---|
| held-out `p(t\|x)` NLL, 5 seeds | 0.245, 0.315, 0.257, 0.283, 0.238 | 0.292, 0.338, 0.356, 0.282, 0.285 | mean ratio **0.867** |
| held-out outcome NLL, 5 seeds | 1.169, 1.181, 1.193, 1.160, 1.183 | 1.128, 1.148, 1.116, 1.112, 1.154 | pretrained worse on all five |

Four wins and one tie on the treatment side. So the contrastive encoder does
carry treatment-predictive structure — and 1,000 or 3,000 steps of gradient from
40 labels takes both arms far enough from their initialisations that where they
started stops mattering. The outcome row is the other half of the same fact and
is not a happy one: a representation shaped by the covariance of `x` alone is
not shaped for `p(y | x, t)`, and a frozen encoder gives the outcome head no way
to reshape it. Both halves are asserted in Tier 1 so that they stay visible.

**What this is not.** One DGP, one architecture, 40 labels, five seeds. SCARF's
published evidence is 69 datasets under three label regimes, and nothing here
contradicts it: what is measured is that on a fixture where the scarce label is
the cluster structure, an objective that pushes same-cluster rows apart does not
hand the propensity head anything that survives fine-tuning. §2 now says that,
and it is a concrete argument for running the class-aware contrastive cards
(CoMatch, SimMatch, PAWS) that `BACKLOG.md` §1 sequences immediately after this
one.

### 6.3 Result ledger

| Date | Commit | Metric | Value ± stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

**Not yet run.** The Tier 2 runner (`xty2/evaluation/benchmarks/`) has one
module per recipe and this recipe has none, exactly as `fixmatch.md` §6.3
records for its own. Until that module exists this card's status may not go past
`smoke-passing`, and the block above is a declared protocol rather than a
result. On §6.2's evidence the protocol as declared should be expected to
return a `deviating` outcome; the tolerance stays as written, because a
tolerance rewritten after seeing the numbers is worth nothing
(`FIDELITY.md` §3).

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
