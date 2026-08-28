# xty2 — Post-P12 research backlog

**Status:** idea backlog, not a build plan
**Reads with:** `DESIGN.md`, `FIDELITY.md`, `PLAN.md`, `CLAUDE.md`

This document is a catalogue of methods and experiments that become possible once
P0–P12 are complete. It is deliberately broader than `PLAN.md`.

It is **not** permission to keep expanding the framework. The default assumption
at Gate 2 remains that xty2 is done: new work should arrive as a reviewed spec
card plus a declarative recipe assembled from existing components, objectives,
views and executors. A new framework abstraction for *convenience* is justified
only when a second real recipe needs it. One that is load-bearing for fidelity —
without it the card's §4 checklist cannot be honoured as the paper states it —
is justified by the first card, and `DESIGN.md` §11.2 says how far the design
obligation goes. If a method cannot be expressed cleanly, that is evidence to
discuss the framework boundary, not a reason to widen it opportunistically —
and equally not a reason to ship the method with the mechanic missing.

The intended post-P12 workflow is therefore:

1. Pick an approach because there is a real experiment to run.
2. Write `docs/recipes/<name>.md` first and stop for review.
3. Try to express it with the existing xty2 vocabulary.
4. Add the smallest missing component/objective/view if necessary.
5. Add a new framework concept under `DESIGN.md` §11.2 — one consumer where it
   is load-bearing for fidelity, a second where it is convenience — and record
   it in card §5.1.
6. Anything you cannot express and do not build becomes a typed
   `framework-limitation` row in card §5 citing a `DESIGN.md` §11.4 ledger key,
   never an untyped line that reads like a design choice.
7. Require Tier 0, Tier 1 and an explicit Tier 2 target as usual.

The point of this backlog is increasingly **combinatorial** experimentation. It
should not turn into a project to port the other ~35 XTYLearner model classes.

A useful way to read the SSL literature is as three generations:

1. **Isolated mechanisms** — pseudo-labelling, entropy minimisation, VAT,
   Pi-model consistency, Mean Teacher, rotation prediction.
2. **Composite recipes** — S4L, MixMatch, UDA, ReMixMatch, FixMatch, Noisy
   Student: the method is increasingly a deliberate bundle of mechanisms.
3. **Interacting/adaptive systems** — CoMatch, SimMatch, Meta Pseudo Labels,
   FlexMatch, FreeMatch, SoftMatch, SequenceMatch and related work: one mechanism
   changes the rows, targets, thresholds, augmentation strengths or optimisation
   trajectory seen by another.

The third category is the most interesting post-P12 stress test. xty2 should
first try to express these methods using ordinary components, objectives, views,
schedules, artifacts and stages. Do **not** invent a generic `Policy` abstraction
in advance: a mediator with one caller is the top of `DESIGN.md` §11.1's
over-building column, and none of these recipes is blocked on *generality* — each
is blocked on its own specific mechanism, which is the convenience quadrant. If
the same missing concept appears independently in two real recipes, that is the
evidence for one.

---

## 1. High-priority post-P12 tranche

These are attractive early implementations because each tests whether xty2 can
express a recognisable method mostly through recombination rather than new
framework machinery.

| Approach | What it exercises | Expected fit |
|---|---|---|
| ReMixMatch | MixMatch + distribution alignment + adaptive strong augmentation + anchoring + rotation | major composite-recipe stress test |
| CoMatch | pseudo-labelling + contrastive learning + graph regularisation with mutual interaction | semantic-interaction stress test |
| Meta Pseudo Labels + UDA | teacher/student meta-update plus a full auxiliary SSL recipe | executor/optimisation boundary test |
| SimMatch | semantic and instance similarity, memory banks, mutual pseudo-label refinement | state/artifact interaction test |
| DoubleMatch | confidence-gated pseudo-labelling plus self-supervised loss for rejected rows | row-population composability |
| UDA | weak/strong consistency + confidence masking + sharpening + TSA | mostly recipe/objectives |
| MixMatch | augmentation averaging + sharpening + MixUp + pseudo-labels | deliberate row-synthesis boundary test |
| FixMatch | weak/strong views, confidence pseudo-labels, consistency | mostly recipe/objective |
| FlexMatch | FixMatch plus class-specific curriculum thresholds | adaptive pseudo-label policy |
| FreeMatch | self-adaptive global/class thresholds + fairness regularisation | adaptive objective/policy |
| SoftMatch | continuous confidence weighting + distribution alignment | objective/policy |
| PAWS | labeled support representations + view consistency + soft nearest-neighbour labels | metric/pseudo-label composition |
| SelfMatch | contrastive pretraining followed by consistency SSL | existing staged Program |
| SsCL | classification and contrastive branches that co-calibrate one another | semantic-interaction stress test |
| Semi-supervised clustering | supervised class prototypes + self-supervised clustering objective | prototype/objective interaction |
| Noisy Student | teacher -> pseudo-label -> noisy/larger student, iterated | existing Program/artifacts |
| SimCLRv2 | SSL pretrain -> supervised fine-tune -> unlabeled distillation | existing staged Program |
| SCARF | tabular corruption plus representation contrast | view + contrastive objective |
| SubTab | feature-subset views plus reconstruction | view + reconstruction |
| VIME | corruption mask prediction plus reconstruction | likely one new semantic output |
| VAT | adversarial local consistency | adversarial view/objective |
| Pi-model | consistency across stochastic realisations | recipe |
| Temporal Ensembling | consistency to historical predictions | artifact/objective |
| VICReg | invariance/variance/covariance regularisation | objective |
| Barlow Twins | redundancy-reduction representation learning | objective |
| S4L-style joint training | several SSL and supervised objectives jointly | composite recipe |
| hard/soft EM | alternate latent-treatment inference and fitting | stages/objectives |
| variational latent-treatment model | q(t|x,y), p(t|x), p(y|x,t), ELBO | existing core quantities |
| DragonNet | propensity, outcome and targeted regularisation | components/objective |
| CFRNet | outcome learning plus representation balancing | balancing objective |

### Recommended stress-test sequence

A useful sequence after P12 is:

1. **SCARF** — establish an independently validated representation objective.
2. **FixMatch** — establish confidence-gated weak/strong pseudo-labelling.
3. **DoubleMatch** — show that rows rejected by pseudo-labelling can still train
   through another objective.
4. **ReMixMatch** — deliberately crowd the recipe with interacting mechanisms.
5. **CoMatch** — test whether representation and pseudo-label quantities can
   influence one another without procedural recipe code.
6. **SimMatch** — test memory/stateful cross-space pseudo-label refinement.
7. **Meta Pseudo Labels + UDA** — test the optimisation/executor boundary.
8. **variational latent-treatment / ELBO** — exploit all three native XTY
   probabilistic quantities.
9. **explicit treatment-observation/missingness model** — deliberate statistical
   vocabulary boundary test.
10. **an S4L-style XTY composite** combining independently validated pieces and
    using raw loss, coverage, gradient norms and gradient cosines to determine
    which objectives cooperate.

The point of this sequence is not to reproduce a leaderboard progression. It is
to make each new recipe probe a different architectural claim.

**Where the sequence has got to.** Steps 1 and 2 have landed —
`docs/recipes/scarf.md` and `docs/recipes/fixmatch.md`, both `draft` and
neither Tier 2'd. SCARF's port carries a result worth reading before picking up
step 5 or 6: the contrastive pretraining works as a representation learner and
its encoder does carry treatment-predictive structure under a frozen probe,
but the end-to-end gain the paper's protocol predicts is *absent* on the
project fixture at every fine-tuning budget tried (`scarf.md` §6.2). The
mechanism appears to be the obvious one — instance discrimination treats
same-class rows as negatives, and on a fixture where the scarce label is the
cluster structure it spends its capacity pushing apart exactly the rows the
propensity head wants together. That is the gap CoMatch (step 5), SimMatch
(step 6) and PAWS exist to close, which makes them a sharper next step than
they looked before the measurement existed.

Step 3 has now landed too — `docs/recipes/doublematch.md` — and it bears on
that reading without settling it. DoubleMatch's `l_s` is SCARF's family with
the negatives deleted (a cosine to one's own other view, no contrast set), and
on the declared encoder it does learn, and without collapsing: an alignment of
0.61 where the same term left undescended sits within 0.01 of zero. Whether
that buys a better propensity came out **mixed**, and mixed in a way worth
knowing about — two initialisation draws of one architecture put the EMA
comparison on opposite sides of 1.0 (`doublematch.md` §6.2). So the
negative-free variant clears the bar SCARF's did not, and the end-to-end
question is a Tier 2 one rather than a null result.

It also produced a result nobody was looking for, and it is the reason to read
that card before writing the next one: **a cosine consistency term collapses
this project's shared encoder, and the cause is the scale of the
representation, not its geometry.** Eq. (3)'s gradient carries `1/||.||`.
CFRNet's `0.1/sqrt(fan_in)` leaves `MLPEncoder`'s pre-normalisation activations
at a norm of 0.011 — about a hundredth of what a batch-normalised backbone
hands such a term in the papers these methods come from — and `row_l2` passes
that factor upstream, so the term arrives ninety times too loud and drives the
whole batch to one direction inside ten steps at every `w_s` from 0.5 to 0.01
(`doublematch.md` §6.2). Changing only the initialisation removes it; changing
only the normalisation does not, which is what the first version of that card
got wrong and an adversarial review caught.

**Step 4's neighbourhood has now been visited too, out of order and
deliberately.** `flexmatch` is not step 4 - ReMixMatch is - but it is the
cheapest available test of the same architectural claim the sequence was built
to probe, because Curriculum Pseudo Labeling changes exactly one thing about
`fixmatch` and that one thing needs a mechanic the framework did not have: a
quantity accumulated across steps. §2.5 records what it cost (one objective, one
framework concept, no new card key) and what it found.

Its most transferable result is not about CPL at all. **A recipe with an
ungated phase is an instrument for measuring whether the project's strong view
is any good, and every recipe before it was blind to that.** `fixmatch`'s strong
view flips the Bayes-optimal label on one row in six; three recipes shipped on
it without noticing, because a confidence gate holds the term inert until the
noticing would matter. The check is training-free and cheap - the Bayes-optimal
label-flip rate of the view on the fixture, closed form for the project DGP -
and every card declaring a strong view should now carry it.

Two things follow for §5.1's cosine-shaped methods (SimMatch's instance
similarity, PAWS's soft assignments, BYOL, SimSiam, and Barlow Twins by the
same argument about its cross-correlation). Any of them meets this the moment
it reads `X_REPR` from the P5 backbone, so measure `||f(x)||` before tuning a
weight. And the cheap diagnostic that caught it is
`CosineFeatureConsistency`'s concentration pair, which any of them can reuse —
provided it is read over the whole trajectory: the architecture that card first
declared spends 135 steps fully collapsed and recovers, which a terminal
reading scores as healthy.

---

## 2. Composite SSL lineage

This section records methods whose contribution is explicitly or effectively the
combination of several reusable SSL mechanisms. These are particularly important
for xty2 because they test the premise that a named algorithm can remain a thin
recipe over independently reusable pieces.

### 2.1 S4L

S4L combines supervised learning with self-supervised pretext objectives such as
rotation prediction. The important architectural lesson is that self-supervision
is an auxiliary objective over the same underlying representation rather than a
separate monolithic model family.

**xty2 experiment:** treatment NLL + outcome NLL + missing-treatment marginal NLL
+ one or more representation pretexts, first separately and then jointly.

Reference: Zhai et al., *S4L: Self-Supervised Semi-Supervised Learning* (ICCV
2019), https://arxiv.org/abs/1905.03670

### 2.2 MixMatch

MixMatch deliberately combines:

- multiple augmentations of an unlabeled example;
- prediction averaging;
- temperature sharpening / entropy minimisation;
- pseudo-labels;
- MixUp between labeled and pseudo-labeled examples;
- supervised and unsupervised objectives.

The method is useful because it is already a statement that these mechanisms
should be composed rather than treated as mutually exclusive model families.

**xty2 boundary:** MixUp synthesises rows and often targets, which is not
obviously a `ViewSpec`. Do not pretend otherwise. If a second real recipe also
needs row synthesis, consider a first-class mechanism at that point. Note which
quadrant that is: row synthesis is load-bearing vocabulary (it changes what a
`ViewSpec` means), so the first card that genuinely cannot state its §4 without
it may still build it — against a named second consumer, per `DESIGN.md` §11.2.

Reference: Berthelot et al., *MixMatch* (NeurIPS 2019),
https://arxiv.org/abs/1905.02249

### 2.3 ReMixMatch ★

ReMixMatch is one of the strongest direct tests of the xty2 thesis. It layers
onto MixMatch:

- distribution alignment;
- augmentation anchoring from weak to strong views;
- multiple strong augmentations;
- CTAugment / adaptive augmentation;
- an additional pre-MixUp unlabeled loss;
- an S4L-style rotation-prediction objective;
- the existing MixMatch sharpening and MixUp machinery.

This is almost the canonical "many useful things at once" recipe. A successful
xty2 implementation should be declarative enough that each of the above remains
identifiable in the execution plan and diagnostics.

**Stress test:** if ReMixMatch requires procedural branching inside the recipe,
record exactly which interaction cannot be represented. Do not immediately
patch around it. Compare that missing concept against CoMatch, SimMatch and MPL
before deciding whether a framework extension has two consumers. If it turns
out ReMixMatch's §4 checklist cannot be honoured without it, `DESIGN.md` §11.2
Q1 settles it on one — the comparison is then about *shape*, not permission.

Reference: Berthelot et al., *ReMixMatch* (ICLR 2020),
https://arxiv.org/abs/1911.09785

### 2.4 UDA

Unsupervised Data Augmentation combines more than its name suggests:

- supervised cross-entropy;
- weak/strong augmentation consistency;
- detached targets;
- confidence masking;
- temperature sharpening;
- large unlabeled batches;
- EMA in the vision setup;
- Training Signal Annealing, which dynamically suppresses sufficiently easy
  labeled examples.

TSA is especially interesting in xty2 terms: it is effectively a dynamic row
eligibility rule for a supervised objective.

Reference: Xie et al., *Unsupervised Data Augmentation for Consistency Training*
(NeurIPS 2020), https://arxiv.org/abs/1904.12848

### 2.5 FixMatch and adaptive descendants

FixMatch compresses much of the previous generation into a simple core:
weak-view prediction -> confidence threshold -> hard pseudo-label -> strong-view
cross-entropy. Its descendants then modify or add mechanisms around that core:

> **Implemented.** `docs/recipes/fixmatch.md` and `xty2/recipes/fixmatch.py`.
> It cost one objective (`PseudoLabelTreatmentNLL`), one schedule type
> (`CosineDecay`, on the ledger condition in `DESIGN.md` §11.4), and two
> framework concepts taken with **one consumer** — a `draw` axis on
> `Realisation` and `TeacherSpec.role`, both recorded in that card's §5.1.
> Under the old two-consumer rule those needed a maintainer's ad-hoc
> dispensation; under `DESIGN.md` §11.2 they are the fidelity-bearing,
> reversible quadrant and are simply the rule. No new port, executor or row population: the gate is a
> per-row mask inside the objective rather than a new `Rows` value, which is
> what kept `t_missing & confident` from becoming framework vocabulary.
>
> Two findings worth carrying forward. Into FlexMatch/FreeMatch/SoftMatch: the
> confidence gate does not fail safe under overlap — it manufactures the
> confidence it is gated on (card §2, §6.2). Into anything that reports from an
> EMA: on the overlapping fixture the EMA-reported NLL beats the baseline while
> the network it averages is worse than the baseline, so an EMA number is not
> evidence the mechanism is behaving.

- **FlexMatch** — class-specific curriculum thresholds based on learning
  progress.

  > **Implemented.** `docs/recipes/flexmatch.md` and
  > `xty2/recipes/flexmatch.py`. It cost **one objective**
  > (`CurriculumPseudoLabelTreatmentNLL`, with its `CurriculumThreshold` policy
  > and `CurriculumStatus` state) and **one framework concept**: objective state
  > with a stage lifecycle (`StatefulObjective`,
  > `TrainContext.objective_states`), taken in the load-bearing quadrant with
  > FreeMatch named as the second consumer and its shape checked against it
  > (card §5.1). No new card key — the gate rule is one value bound to
  > `losses.confidence_threshold` — no new port, view, schedule, row population
  > or executor. §15.2's instruction was followed to the letter: the mechanism
  > is local and explicit, a *separate* objective rather than a policy union on
  > `PseudoLabelTreatmentNLL`, and the duplicated arg-max/mask/mean arithmetic
  > is the price §15.2 asks for until FreeMatch shows the shape.
  >
  > **Not reproduced — `draft`, no Tier 2 runner — and the way it first failed
  > is the more useful half anyway.** Be precise about the evidence, because
  > this entry is what the next agent reads to decide what is settled.
  > `flexmatch.md` §6.2 measures **one** of §6's five tolerance clauses, at
  > **five** of the declared ten seeds, from a script: a paired EMA ratio of
  > 0.977 +/- 0.014 against a constant-gate arm, ahead on four of five, with the
  > curriculum engaging on all five (first mark between steps 32 and 76, `T(c)`
  > reaching `tau`). The trained-parameter ratio, mask rate, impurity and
  > outcome NLL are unmeasured. That is enough to say the mechanism runs and
  > points the right way; it is not enough to say the method reproduces, and
  > `CLAUDE.md`'s standing rule is that only a Tier 2 result sets `reproduced`.
  > It also does *not* establish that the per-class half of CPL earns the ratio:
  > with `K = 2` both classes reach `beta = 1` together, and the
  > class-imbalanced probe in §6.3 is a null with error bars an order of
  > magnitude wider. Anyone wanting that question answered needs a fixture with
  > more treatment levels, which is the one thing this section's K = 2 project
  > DGP cannot supply at any seed count.
  >
  > Getting even the paired result required fixing a mechanic this backlog entry
  > should flag for every descendant below.
  >
  > **A gated method hides a strong view that is not label-preserving; an
  > ungated phase does not.** FixMatch §2.3 asks a strong augmentation to be
  > severe *and* label-preserving, and `fixmatch`'s tabular analogue - a 10%
  > mask followed by a 50% one, an effective 0.55 over six columns of which
  > four carry the signal - flips the *Bayes-optimal* label on 16.8% of rows.
  > `fixmatch` scores the same at 0.55 as at 0.28 across five seeds, because
  > its constant gate holds eq. (4) inert until the model is already confident.
  > FlexMatch has no such protection: its thresholds start at zero, so eq. (8)
  > is ungated until some row clears `tau`, and a strong view whose one-hot
  > target is unattainable pins the propensity below `tau` for good. At 0.55
  > that happened on **three initialisation seeds of five**; at the 0.2 that
  > clears a stated label-preservation criterion, on none.
  >
  > Three things follow for the rest of this section. **The strong view is a
  > hyperparameter of the *fixture*, not of the recipe, and it needs a
  > training-free check** - the Bayes-optimal label-flip rate under the view,
  > which `flexmatch.md` §5.2 computes in closed form for the project DGP and
  > Tier 0 recomputes. Any descendant that opens the gate early - SoftMatch's
  > continuous weights at low confidence, FreeMatch's self-adaptive `tau` in
  > its own warm-up, Dash's decaying threshold - is exposed to the same trap and
  > should run that check first. **Watch the labelled cross-entropy, not only
  > the unlabelled one**: in the trapped runs eq. (10) stayed flat above `log 2`
  > for 3,000 steps, which is a louder signal than any coverage number. And
  > **one seed is not a measurement of a warm-up**: the first draft of that card
  > reported a single initialisation draw as a property of the method, and it
  > was not even a property of the seed set - `doublematch.md` §6.2 had already
  > recorded the same lesson about initialisation draws and this card had to
  > learn it twice.

- **FreeMatch** — self-adaptive global and class-specific thresholds plus
  fairness/class-balancing regularisation.
- **SoftMatch** — continuous confidence weighting rather than a binary gate,
  together with distribution alignment.
- **ConMatch** — confidence estimation and multiple strong views.
- **SequenceMatch** — weak/medium/strong views and different consistency rules
  for high- and low-confidence examples.
- **ShrinkMatch** — reduced class-space consistency for uncertain examples.
- **InfoMatch** — pseudo-supervision + consistency + information/contrastive
  regularisation.
- **AllMatch** — adds a second training signal so rejected unlabeled examples
  can still contribute.
- **CGMatch** — partitions examples by learning state and applies different
  regularisation to easy/ambiguous/hard subsets.

This family repeatedly expresses the same principle: **if a row is unsuitable
for mechanism A, find mechanism B whose assumptions it does satisfy instead of
throwing the row away.** That principle maps naturally onto xty2 row populations.

For missing-treatment data, one observation might simultaneously be:

- too uncertain for hard pseudo-label CE;
- valid for exact marginal likelihood;
- valid for augmentation consistency;
- valid for reconstruction or contrastive SSL;
- later eligible for pseudo-labeling once its posterior sharpens.

References:

- FixMatch: https://arxiv.org/abs/2001.07685
- FlexMatch: https://arxiv.org/abs/2110.08263
- FreeMatch: https://arxiv.org/abs/2205.07246
- SoftMatch: https://arxiv.org/abs/2301.10921
- SequenceMatch: https://arxiv.org/abs/2310.15787
- ShrinkMatch: https://arxiv.org/abs/2308.06777
- InfoMatch: https://arxiv.org/abs/2404.11003
- AllMatch: https://arxiv.org/abs/2406.15763

### 2.6 DoubleMatch

> **Implemented.** `docs/recipes/doublematch.md` and
> `xty2/recipes/doublematch.py`. It cost **one objective**
> (`CosineFeatureConsistency`) and nothing else: no port, no component, no view,
> no schedule type, no executor, no row population, no card key. The
> row-eligibility test below is answered by `Objective.rows`, which has been
> per-objective since P3 — the gate stays a per-row mask inside
> `PseudoLabelTreatmentNLL`, so eq. (2) keeps counting the rows it rejects and
> eq. (3) is simply entitled to all of them.
>
> The one shape decision worth carrying forward: the objective names
> `prediction_port` and `target_port` separately, where every earlier objective
> takes one port under two realisations. DoubleMatch forces it — the two sides
> of its cosine are `X_PROJ` (strong view, through the projection head) and
> `X_REPR` (weak view, detached) — and any method with a predictor on one
> branch only (BYOL, SimSiam, SimMatch) needs the same asymmetry.
>
> Two findings are in that card's §6.2 and are summarised under the sequence
> note above: the scale-driven collapse of the shared encoder, and a mixed
> end-to-end result whose two readings disagree — and swap which of them is
> favourable between two initialisation draws of one architecture. It is the
> second card in a row to find that an EMA number and the network under it can
> point opposite ways (`fixmatch.md` §6.2 was the first).

DoubleMatch pairs FixMatch-style pseudo-supervision on confident rows with a
self-supervised representation objective that still uses the low-confidence
rows the pseudo-label branch rejects.

This is a particularly clean test of `Objective.rows`:

```text
ObservedTreatmentNLL        t_observed
PseudoLabelTreatmentLoss    t_missing & confident
SelfSupervisedLoss          all
```

The exact row-selector syntax may differ, but the semantic test is clear: row
eligibility for one loss must not determine eligibility for all losses.

Reference: *DoubleMatch: Improving Semi-Supervised Learning with Self-Supervision*,
https://arxiv.org/abs/2205.05575

### 2.7 CoMatch ★

CoMatch combines:

- pseudo-label/self-training;
- supervised classification;
- self-supervised contrastive representation learning;
- graph-based regularisation;
- weak/strong consistency.

The important difference from simple weighted multitask learning is that the
parts **interact**. Representation similarity constrains class-probability
pseudo-labels, while the pseudo-label graph determines relationships used by the
contrastive branch.

This is a valuable post-P12 test because it asks whether outputs of one learning
mechanism can become structured targets for another without embedding control
flow in the recipe.

Reference: Li et al., *CoMatch: Semi-supervised Learning with Contrastive Graph
Regularization* (ICCV 2021),
https://arxiv.org/abs/2011.11183

### 2.8 SsCL

Semi-supervised Contrastive Learning similarly maintains classification and
contrastive branches that co-calibrate one another: class predictions affect
neighbour/positive relationships and representation similarity can in turn
calibrate classification predictions.

**Architectural question:** does the shared concept with CoMatch justify a
reusable similarity/relationship mechanism, or can both remain ordinary
objectives over existing `X_REPR` and treatment distributions? Implement first;
abstract second.

Reference: https://arxiv.org/abs/2105.07387

### 2.9 SimMatch ★

SimMatch jointly maintains semantic/class similarity and instance/representation
similarity. Its full recipe includes:

- consistency across augmented views at both levels;
- pseudo-label propagation between semantic and instance spaces;
- labeled feature and label memory banks;
- EMA teacher or temporal ensembling depending on the setting.

This is an important state/artifact test. A memory bank should not automatically
become a new framework primitive. First determine whether it can be expressed as
an explicit artifact/state owned by an objective or stage. Compare the answer to
other methods that need memory before generalising.

Reference: Zheng et al., *SimMatch* (CVPR 2022),
https://arxiv.org/abs/2203.06915

### 2.10 PAWS

PAWS uses labeled support representations to construct soft class assignments
for unlabeled examples and trains augmented views to agree on those assignments.
It combines metric/nearest-neighbour ideas, self-supervised representation
learning and supervised label information without requiring a conventional
classifier-generated pseudo-label.

This may translate unusually well to tabular XTY data because the core mechanism
is representation similarity rather than image-specific augmentation structure.

Reference: Assran et al., *Semi-Supervised Learning of Visual Features by
Non-Parametrically Predicting View Assignments with Support Samples* (ICCV 2021),
https://arxiv.org/abs/2104.13963

### 2.11 Meta Pseudo Labels + UDA ★

Meta Pseudo Labels already contains a teacher/student feedback loop:

1. teacher generates pseudo-labels;
2. student updates on them;
3. student's improvement on labeled data is measured;
4. that improvement supplies a meta-gradient/update signal to the teacher.

The high-performing recipe then jointly trains the teacher with ordinary
supervised learning and **the full UDA objective** as auxiliary signals.

This is more than several losses sharing a forward pass: one learner's update
changes the optimisation signal of another learner. It is therefore a deliberate
boundary test for the `gradient` executor and linear `Program`, and one of the
strongest candidates for exposing a genuinely missing post-v1 abstraction.

Do not build bilevel/meta-learning machinery before attempting the card and
identifying the exact required execution semantics.

Reference: Pham et al., *Meta Pseudo Labels* (CVPR 2021),
https://arxiv.org/abs/2003.10580

### 2.12 SelfMatch, SimCLRv2 and Noisy Student

These show the same compositional instinct across **stages** rather than within
one simultaneous objective set.

- **SelfMatch:** contrastive self-supervised pretraining -> augmentation-
  consistency semi-supervised fine-tuning.
- **SimCLRv2:** self-supervised pretraining -> supervised fine-tuning ->
  distillation/self-training over unlabeled data.
- **Noisy Student:** supervised teacher -> pseudo-label unlabeled data -> train
  an equal-or-larger noisy student with augmentation/dropout/stochastic depth ->
  promote student to teacher -> repeat.

These should mostly validate `Program`, immutable stage artifacts and
initialisation between stages rather than require framework changes.

References:

- SelfMatch: https://arxiv.org/abs/2101.06480
- SimCLRv2: https://arxiv.org/abs/2006.10029
- Noisy Student: https://arxiv.org/abs/1911.04252

### 2.13 Semi-supervised clustering / shared prototypes

Fini et al. show a different S4L-like direction: merge supervised class
prototypes and self-supervised clustering machinery into one multi-task system,
rather than treating supervised and self-supervised representation objectives as
separate stages.

This is relevant if xty2 eventually has at least two real consumers for
prototype/cluster-assignment semantics.

Reference: Fini et al., *Semi-Supervised Learning Made Simple with
Self-Supervised Clustering* (CVPR 2023),
https://arxiv.org/abs/2306.07483

### 2.14 SemiLearn / USB as software prior art

The USB/SemiLearn project is worth studying independently of any one algorithm.
It provides a unified implementation of many SSL methods and factors out reusable
algorithmic utilities/hooks such as:

- distribution alignment;
- pseudo-label generation;
- thresholding;
- MixUp;
- EMA updates;
- SSL losses.

That is direct engineering prior art for moving reuse below the named-algorithm
level. xty2 should remain more semantically/statistically explicit, but before
inventing a new post-P12 concept check whether SemiLearn has already encountered
the same decomposition problem.

Reference: Wang et al., *USB: A Unified Semi-supervised Learning Benchmark*,
https://arxiv.org/abs/2208.07204

---

## 3. Semi-supervised treatment-label methods

### 3.1 Pseudo-labelling and self-training

- Plain self-training from `p(t|x)`.
- EMA-teacher self-training.
- Out-of-fold self-training using P10 provenance checks.
- Posterior self-training from `q(t|x,y)` with leakage protection active.
- Ensembled pseudo-labels combining `p(t|x)` and `q(t|x,y)`.
- Agreement-only pseudo-labelling when propensity and posterior agree.
- Soft pseudo-labels rather than argmax labels.
- Temperature-sharpened pseudo-labels.
- Confidence-weighted pseudo-label loss.
- Uncertainty-weighted pseudo-label loss.
- Class-balanced pseudo-labelling.
- Treatment-frequency-aware pseudo-labelling.
- Distribution alignment of pseudo-label marginals.
- Curriculum pseudo-labelling from high to lower confidence.
- Per-class curriculum thresholds.
- Self-adaptive thresholds.
- Continuous confidence weighting instead of hard rejection.
- Calibration-aware thresholds.
- Entropy-based abstention.
- Ensemble-disagreement abstention.
- Monte-Carlo uncertainty abstention.
- Pseudo-label refresh every epoch/stage.
- Frozen pseudo-label tables generated once between stages.
- Iterative pseudo-label/refit programs.
- Multiple treatment imputation rather than one hard pseudo-label.
- Posterior-sampled treatment imputation.
- Easy/ambiguous/hard row partitions with subset-specific objectives.
- Reduced-class-space consistency for uncertain rows.
- Binary/coarse consistency objectives for rows rejected by multiclass
  pseudo-labelling.

### 3.2 Consistency methods

- Weak/strong consistency.
- Weak/medium/strong consistency.
- Multi-strong-view consistency.
- VAT.
- Pi-model consistency.
- Temporal Ensembling.
- Mean Teacher variants.
- Consistency under feature masking.
- Consistency under bounded jitter.
- Consistency under feature-subset views.
- Consistency under dropout/noise realisations.
- Consistency under manifold perturbation.
- Consistency between `p(t|x)` and `q(t|x,y)`.
- Symmetric KL between propensity and posterior.
- Jensen-Shannon propensity/posterior matching.
- Consistency of treatment-wise outcome means under permissible views.
- Consistency of CATE estimates across views.
- Consistency between semantic and instance-level pseudo-labels.

### 3.3 Dynamic eligibility and adaptive mechanisms

These are deliberately listed as **mechanisms**, not a request for a generic
framework `Policy` type.

- Confidence-threshold row selection.
- Class-specific curriculum thresholds.
- Self-adaptive global/class thresholds.
- Continuous confidence weights.
- Training Signal Annealing for labeled rows.
- Distribution alignment.
- Class-fairness regularisation.
- Adaptive augmentation strength.
- Learning-state partitions (easy/ambiguous/hard).
- Memory-bank label propagation.
- Pseudo-label sharpening/temperature adaptation.
- Neighbour-graph construction from current representations.
- Relationship graphs derived from current pseudo-labels.

Implement these in the smallest local place each recipe permits. If two or more
recipes independently require the same stateful mediation semantics and local
implementations become duplicated or unreviewable, revisit the abstraction
boundary then.

### 3.4 Entropy, posterior and latent-variable objectives

- Entropy minimisation on missing-treatment rows.
- Entropy maximisation where uncertainty should be preserved.
- Posterior KL `KL[q(t|x,y) || p(t|x)]`.
- Reverse KL.
- Symmetric KL.
- Jensen-Shannon matching.
- Mutual-information maximisation between representation and treatment.
- Hard EM.
- Soft EM.
- Generalised EM with parameter-block stages.
- Variational EM.
- ELBO for latent treatment.
- Importance-weighted ELBO / IWAE-style objective.
- Posterior tempering.
- KL annealing.
- Free-bits style posterior regularisation if collapse appears.
- Importance-weighted missing-treatment marginal likelihood.

### 3.5 Multi-model SSL

- Co-training with distinct encoders/views.
- Tri-training.
- Deep mutual learning.
- Student-student consistency.
- Teacher ensembles.
- Cross-pseudo-supervision.
- Co-regularised propensity models.
- Co-regularised outcome heads.
- Meta Pseudo Labels.
- Teacher trained with auxiliary UDA/supervised objectives.

---

## 4. Explicit modelling of treatment-label availability

The v1 framework distinguishes observed and missing `t` but does not model why
`t` is observed. This is a deliberately deferred direction and may eventually
justify a new semantic quantity only after two concrete recipes need it.

Candidate approaches:

- MAR label-selection model `p(r_t=1 | x)`.
- Outcome-dependent selection `p(r_t=1 | x,y)`.
- MNAR selection depending on latent treatment.
- Selection depending on treatment uncertainty.
- Selection depending on acquisition process/site/source.
- Inverse-probability weighting for label availability.
- Doubly robust correction for treatment-label missingness.
- Joint propensity + label-observation model.
- Joint posterior + label-observation model.
- Sensitivity analysis over unverifiable missingness assumptions.
- Pattern-mixture formulations.
- Selection-model formulations.

Do not add a `LABEL_OBSERVATION_PROB`-like port just because one recipe asks for
it. Specify at least two real consumers first.

---

## 5. Self-supervised representation learning

### 5.1 Contrastive, clustering and redundancy-reduction methods

- SCARF.
- SimCLR-style contrastive pretraining.
- Supervised contrastive learning on observed-treatment rows.
- Semi-supervised contrastive learning using pseudo-labels.
- CoMatch-style contrastive graph regularisation.
- SsCL-style classification/contrastive co-calibration.
- SimMatch-style instance/semantic similarity learning.
- PAWS-style support-sample assignments.
- VICReg.
- Barlow Twins.
- BYOL-style representation prediction.
- SimSiam.
- DINO-style self-distillation.
- DeepCluster-style clustering.
- Shared supervised/self-supervised prototypes.
- Prototype consistency.
- Neighbour consistency.
- Contrastive predictive coding adapted to tabular views where meaningful.

### 5.2 Reconstruction and corruption methods

- Denoising autoencoder.
- Masked feature reconstruction.
- SCARF-style feature replacement without contrastive loss.
- VIME.
- SubTab.
- Feature dropout reconstruction.
- Block masking of related feature groups.
- Corruption-mask prediction.
- Missingness-pattern reconstruction.
- Multi-task reconstruction of raw and derived quantities.
- Treatment-conditioned reconstruction where causally appropriate.

### 5.3 Composite pretraining and joint recipes

- SCARF pretrain -> joint marginal-likelihood fit.
- Masked reconstruction -> TARNet/propensity fit.
- SCARF -> FixMatch.
- SCARF -> Mean Teacher.
- SelfMatch-style contrastive pretrain -> consistency SSL.
- SimCLRv2-style SSL pretrain -> supervised fit -> unlabeled distillation.
- Reconstruction + VAT + missing-treatment marginal likelihood.
- Contrastive learning + propensity supervision.
- Contrastive learning + outcome supervision.
- Contrastive learning + propensity + outcome + marginalisation jointly.
- Masked reconstruction + weak/strong consistency + pseudo-labelling.
- DoubleMatch-style pseudo-label loss + self-supervised loss over rejected rows.
- Contrastive pretrain -> Mean Teacher -> OOF pseudo-label refit.
- Multi-pretext S4L-style training where several SSL objectives run together.
- ReMixMatch-style rotation + consistency + pseudo-label + distribution-alignment
  bundle.
- Shared prototype/classification + clustering objective.

---

## 6. Views and augmentation strategies

These should remain schema-aware and functional. Physical/tabular constraints
matter more than reproducing image-augmentation APIs.

- Independent bounded jitter.
- Feature masking/dropout.
- Random feature replacement.
- Feature-subset views.
- Group/block masking of related variables.
- Quantisation noise.
- Gaussian noise scaled by declared measurement uncertainty.
- Noise scaled by `FeatureSpec.perturbation_scale`.
- Empirical resampling from feature marginals.
- Conditional resampling given related features.
- Copula/conditional-density resampling.
- Correlated jitter using empirical covariance.
- Manifold perturbation in `X_REPR`.
- Adversarial perturbation.
- Weak/strong augmentation pairs.
- Weak/medium/strong augmentation chains.
- Multiple independent strong views.
- Random compositions of valid transforms, analogous to tabular RandAugment.
- Adaptive augmentation strength analogous to CTAugment where a tabular analogue
  can be defined without violating schema constraints.
- Primitive-feature perturbation followed by recomputation of derived columns.
- Domain-informed perturbations preserving physical identities.
- Simulated missing-feature patterns.
- Noise based on known sensor resolution/error.
- Feature permutation where scientifically defensible.
- Counterfactual-valid transformations that preserve treatment/outcome semantics.

### 6.1 Row-mixing methods as an explicit boundary test

- MixUp.
- Manifold MixUp.
- CutMix-like feature-block interpolation if meaningful for tabular data.
- MixMatch.
- ReMixMatch.

These may not belong in `ViewSpec`: they synthesize rows and often targets,
rather than creating another realisation of one row while preserving declared
semantics. Do not force them into the existing view abstraction. If two real
recipes need row synthesis, design that concept explicitly.

---

## 7. Causal representation and treatment-effect methods

### 7.1 Neural causal representation learners

- TARNet variants.
- CFRNet with MMD balancing.
- CFRNet with Wasserstein balancing.
- DragonNet.
- DragonNet with targeted regularisation.
- DragonNet + exact missing-treatment marginalisation.
- TARNet + representation balance.
- TARNet + propensity regularisation.
- TARNet + self-supervised representation pretraining.
- Orthogonal neural networks / R-loss neural models.
- CEVAE.
- GANITE.
- Balancing-neural-network variants.
- Treatment-specific adapters or experts.

### 7.2 Meta-learners and orthogonal estimators

- S-learner.
- T-learner.
- X-learner.
- R-learner.
- DR-learner.
- AIPW.
- TMLE.
- Double/debiased ML.
- Orthogonal/R-loss estimators.
- Partialling-out estimators.
- Doubly robust pseudo-outcome regression.
- Multi-treatment extensions where compatible with categorical small-K v1.

### 7.3 Array/cross-fit estimators

Good candidates for the existing `array_fit` and `cross_fit` executors:

- Causal forests.
- Generalised random forests.
- BART-based causal models.
- Linear/lasso DML nuisance models.
- Gradient-boosted nuisance models.
- Random-forest nuisance models.
- Stacked/ensemble nuisance learners.
- Entropy balancing.
- Covariate balancing propensity scores.
- Overlap weighting.
- Propensity truncation strategies.

### 7.4 Semi-supervised causal combinations

- DragonNet + missing-treatment marginal likelihood.
- CFRNet + missing-treatment marginal likelihood.
- R-learner with representation pretraining.
- DR-learner with nuisance models trained using all eligible XTY information.
- OOF `q(t|x,y)` treatment imputation -> orthogonal causal estimator.
- Multiple-imputation treatment labels -> causal estimator pooling.
- Propensity/posterior agreement gating before causal fitting.
- SSL-pretrained encoder -> causal forest on learned representation.
- Mean Teacher propensity -> cross-fit causal nuisance pipeline.
- DoubleMatch-style representation objective + marginal likelihood + causal fit.
- CoMatch/SimMatch-style representation-pseudo-label interaction followed by
  cross-fit causal estimation, with leakage rules explicit.

---

## 8. Alternative outcome parameterisations

Once `cnflow` proves the `OutcomeDistribution` protocol, these should mostly be
component swaps. `MissingTreatmentMarginalNLL` should remain untouched.

- Homoskedastic Gaussian.
- Heteroskedastic Gaussian.
- Student-t.
- Laplace.
- Asymmetric Laplace / quantile model.
- Gaussian mixture / mixture-density network.
- Mixture of experts.
- Categorical outcome model.
- Ordinal outcome model.
- Poisson / negative-binomial count model where appropriate.
- Hurdle distributions.
- Zero-inflated distributions.
- Conditional RealNVP.
- Conditional MAF.
- Neural spline flows.
- Mixtures of flows.
- Treatment-specific flows.
- Shared flow with treatment context.
- Energy-based outcome model.
- Joint-energy models over `(x,t,y)` where a concrete recipe justifies them.

A standing regression test for this family should remain: swapping the outcome
parameterisation must not require editing the source of generic marginalisation
objectives.

---

## 9. Encoder and parameterisation swaps

These are components, not new framework concepts.

- Plain MLP.
- Residual MLP.
- Gated residual MLP.
- SELU/self-normalising MLP.
- Wide & Deep.
- Deep & Cross Network.
- FT-Transformer.
- TabTransformer.
- SAINT.
- TabNet.
- NODE.
- Feature-token transformers.
- Periodic embeddings for continuous variables.
- Piecewise-linear numerical embeddings.
- Learned spline embeddings.
- Mixture-of-experts encoder.
- Shared trunk + treatment-specific heads.
- Treatment-conditioned FiLM.
- Treatment-specific adapters.
- Low-rank treatment interactions.
- Monotonic/constrained subnetworks where domain knowledge warrants them.
- Sparse/feature-selecting encoders.

---

## 10. Multi-objective optimisation and scheduling

P3's raw-loss, coverage, gradient-norm and gradient-cosine diagnostics make this
an especially useful experimental axis.

- Fixed objective weights.
- Linear ramps.
- Sigmoid ramps.
- Warm-up periods.
- Step schedules.
- Curriculum introduction of objectives.
- Alternating objectives.
- Objective-specific update frequencies.
- Alternating parameter blocks.
- Freeze/unfreeze curricula.
- Loss normalisation by moving scale.
- Uncertainty weighting.
- GradNorm.
- PCGrad.
- CAGrad.
- MGDA.
- GradVac.
- Dynamic Weight Averaging.
- Nash-MTL.
- Gradient clipping per objective.
- Gradient clipping per component group.
- Training Signal Annealing / dynamic supervised-row eligibility.

A particularly useful experiment is PCGrad versus a conventional weighted sum
on an intentionally crowded S4L/ReMixMatch-style recipe, with gradient-cosine
traces used to test whether conflict resolution is solving a real observed
problem.

### 10.1 Bilevel/meta-learning boundary

Meta Pseudo Labels is qualitatively different from ordinary weighting: a student
update changes the signal used to update a teacher. Other future methods may do
the same.

Do not generalise the executor before a card exists. First write the exact
sequence of forward/update/evaluation/meta-update operations. If a second real
recipe needs the same execution semantics, that is evidence for a reusable
bilevel/meta-gradient executor.

---

## 11. Distillation, ensembling and uncertainty

### 11.1 Distillation

- Ordinary knowledge distillation.
- Self-distillation.
- Mean Teacher variants.
- Born-Again Networks.
- Noisy Student.
- SimCLRv2-style post-finetune unlabeled distillation.
- Snapshot distillation.
- Propensity -> posterior distillation.
- Posterior -> propensity distillation where leakage semantics permit it.
- Mutual propensity/posterior distillation.
- Representation distillation.
- Outcome-distribution distillation.

### 11.2 Ensembles and approximate posterior uncertainty

- Deep ensembles.
- Bootstrap ensembles.
- Snapshot ensembles.
- MC dropout.
- SWA.
- SWAG.
- Laplace approximation.
- Heteroskedastic predictive uncertainty.
- Ensemble disagreement as pseudo-label uncertainty.

Do not immediately widen `Realisation.params` into an arbitrary ensemble axis.
Independent recipe fits outside one compiled program may be the correct solution
until two genuine recipes need ensemble members inside the framework.

### 11.3 Memory banks and historical state

- Temporal Ensembling prediction histories.
- SimMatch labeled feature/label memory banks.
- Prototype banks.
- Neighbour queues for contrastive learning.
- Teacher-generated cached pseudo-label tables.

Treat these first as explicit objective/stage state or immutable artifacts. If
at least two recipes require the same lifecycle semantics (initialise, update,
checkpoint, restore, provenance), then consider whether a first-class stateful
artifact abstraction is warranted.

### 11.4 Calibration and abstention

- Temperature scaling.
- Isotonic regression.
- Platt-style calibration where applicable.
- Vector scaling.
- Dirichlet calibration.
- Expected calibration error diagnostics.
- Brier score calibration diagnostics.
- Calibration-aware pseudo-label thresholds.
- Conformal treatment prediction sets.
- Conformal outcome intervals.
- Uncertainty-gated pseudo-labelling.
- Disagreement-gated pseudo-labelling.

---

## 12. Domain robustness and adaptation

These may require explicit domain/group metadata not present in v1, so keep them
below the abstraction line until concrete experiments justify that data concept.

- CORAL.
- MMD domain alignment.
- DANN / gradient-reversal domain adversary.
- GroupDRO.
- IRM.
- VREx.
- Domain-specific normalisation.
- Domain-conditioned heads.
- Domain-specific adapters.
- Invariant representation learning across aircraft/fleet/site/time domains.
- Test-time entropy minimisation.
- Test-time normalisation adaptation.
- Source-free adaptation.
- Domain-generalised augmentation.
- Importance weighting under covariate shift.
- Label-shift correction.
- Treatment-prior shift correction.

---

## 13. Active learning and label acquisition

These close the loop with an external labelling process and therefore may sit
outside `Program` rather than forcing acquisition into the training abstraction.

- Entropy sampling.
- Margin sampling.
- Least-confidence sampling.
- BALD.
- Query-by-committee.
- Ensemble disagreement.
- Core-set selection.
- BADGE.
- Expected gradient length.
- Uncertainty x diversity acquisition.
- Class-balanced acquisition.
- Treatment-stratified acquisition.
- Propensity/posterior disagreement acquisition.
- Expected model-change acquisition.
- Expected reduction in causal-estimand uncertainty.
- Acquisition targeted at overlap/positivity gaps.

---

## 14. Composite experiments enabled by xty2

These are more important than faithfully porting every named method. They test
the claim that components, objectives, views and stages can be mixed without
rewriting monolithic model classes.

### 14.1 Representation + semi-supervision

- SCARF + FixMatch.
- SCARF + Mean Teacher.
- SCARF + VAT.
- VIME + exact treatment marginalisation.
- SubTab + Mean Teacher.
- Masked reconstruction + FixMatch.
- VICReg + pseudo-labelling.
- Contrastive pretraining + posterior/propensity matching.
- DoubleMatch-style self-supervision for low-confidence rows.
- PAWS-style support assignment + exact marginal likelihood.

### 14.2 Interacting representation/pseudo-label systems

- CoMatch-style representation graph + treatment pseudo-label graph.
- SsCL-style representation similarity calibrating treatment predictions and
  treatment predictions determining contrastive relationships.
- SimMatch-style instance/semantic mutual refinement.
- Shared supervised/self-supervised prototypes for treatment classes.
- Propensity/posterior agreement graph driving representation positives.

These are deliberately stronger tests than simply adding two loss values. They
ask whether one semantic quantity can shape the targets/relationships used by
another objective while keeping the recipe declarative.

### 14.3 Semi-supervision + causal regularisation

- FixMatch + CFRNet.
- Mean Teacher + CFRNet.
- FixMatch + DragonNet.
- Mean Teacher + DragonNet targeted loss.
- Exact marginalisation + representation balancing.
- Exact marginalisation + orthogonal/R-loss objective.
- Posterior/propensity agreement + targeted regularisation.
- ReMixMatch-style treatment SSL + causal outcome regularisation.

### 14.4 Staged programs

- SSL pretrain -> joint XTY fit -> EMA teacher -> OOF pseudo-labels -> refit.
- SSL pretrain -> marginal-likelihood fit -> cross-fit causal estimator.
- Joint XTY fit -> posterior pseudo-labels -> OOF refit -> targeted causal fit.
- Mean Teacher -> calibrated pseudo-label table -> DR-learner.
- Contrastive pretrain -> FixMatch -> cross-fit DML.
- VIME pretrain -> DragonNet -> TMLE/AIPW post-fit correction.
- SimCLRv2-style pretrain -> supervised XTY fit -> unlabeled distillation.
- Noisy Student-style iterative teacher/student promotion.

### 14.5 Intentionally crowded S4L/ReMixMatch-style recipes

- Reconstruction + treatment NLL + outcome NLL + missing-treatment marginal NLL.
- Contrastive representation loss + supervised treatment loss + marginal NLL.
- Reconstruction + VAT + weak/strong consistency + marginal NLL.
- Contrastive + reconstruction + Mean Teacher + pseudo-label loss.
- Posterior KL + marginal NLL + supervised treatment NLL + outcome NLL.
- Distribution alignment + weak/strong consistency + pseudo-labeling +
  self-supervised reconstruction/contrastive loss.
- Multi-view augmentation + pseudo-label sharpening + exact marginal likelihood +
  representation pretext.
- One of the above with PCGrad/GradNorm to test measured objective conflict.

### 14.6 Row-utilisation experiments

A recurring lesson of DoubleMatch, AllMatch, ShrinkMatch, SequenceMatch and
related work is that confidence should decide **which objective a row is eligible
for**, not simply whether the row is discarded.

Construct recipes where missing-treatment rows move through tiers such as:

```text
all missing-t rows
    -> marginal likelihood + representation SSL
high-confidence rows
    -> + hard/soft pseudo-label CE
ambiguous rows
    -> + coarse/reduced-space consistency
very uncertain rows
    -> no hard pseudo-label, retain reconstruction/contrastive signal
```

This is a particularly natural place to test the row-population design.

---

## 15. Deliberate framework stress tests

Some recipes are valuable specifically because they may reveal that a v1
abstraction is too narrow. Treat these as experiments, not automatic feature
requests.

### 15.1 MixUp / synthetic-row semantics

If MixMatch and ReMixMatch (or another real recipe) both require row
interpolation, consider a first-class synthetic-row mechanism rather than
pretending row mixing is a normal `ViewSpec`.

### 15.2 Stateful mediation between objectives

CoMatch, SimMatch, FlexMatch/FreeMatch and related methods introduce things such
as dynamic thresholds, relationship graphs, distribution alignment and memory
banks. Initially keep each mechanism local and explicit.

Only consider a reusable mediator/policy abstraction if at least two real recipes
need the same lifecycle and semantics. The evidence should be duplicated or
unreviewable implementation, not aesthetic similarity.

### 15.3 Bilevel/meta-gradient execution

Meta Pseudo Labels is the strongest candidate. If a second method genuinely
requires "update learner A, evaluate consequence through learner B, backpropagate
that consequence into A", consider a dedicated executor. Do not contort this
into an ordinary `LossMixer` if the semantics are actually different.

### 15.4 Memory-bank/state lifecycle

If SimMatch and another method require persistent in-training banks with shared
checkpoint/provenance semantics, consider an explicit stateful artifact
abstraction. Until then, keep memory local to the consuming objective/stage.

### 15.5 Explicit missingness mechanism

If two recipes need to model why `t` is observed, consider an explicit typed
statistical quantity for label availability and corresponding provenance rules.

### 15.6 Ensemble realisations

If two recipes need simultaneous in-program ensemble members, reconsider the
current student/teacher parameter realisation axis. Until then, train ensembles
as independent recipes/runs.

### 15.7 Prototype/similarity semantics

CoMatch, SsCL, PAWS, SimMatch and semi-supervised clustering all use
representation relationships, but not necessarily the same statistical
quantity. Do not add `SIMILARITY`, `PROTOTYPE` or graph ports merely because the
names recur. First identify two consumers with the same type/shape/meaning.

### 15.8 Domain/group metadata

If two domain-robust recipes need a stable domain variable, extend the batch or
introduce another typed mechanism only after specifying exactly what is observed,
where it is valid and how leakage should be checked.

### 15.9 Continuous/dose-response treatment

This is explicitly outside v1. It would invalidate assumptions behind exact
small-K marginalisation, candidate-treatment contracts and some estimator APIs.
Treat it as a versioned design exercise rather than another recipe.

### 15.10 Sequence/time-series inputs

Also outside v1. A sequence encoder is not merely another MLP if the input
contract, views and reconstruction semantics all change. Require concrete
sequence recipes before widening the batch/schema abstractions.

---

## 16. Backlog triage rules

When choosing from this file, prefer an item that does at least one of the
following:

- reuses an existing objective with a genuinely different parameterisation;
- reuses an existing component in a genuinely different training program;
- composes two already validated methods without framework changes;
- makes different objectives consume **different row populations** from the same
  batch rather than discarding data globally;
- makes two independently meaningful semantic quantities interact, as in
  CoMatch/SimMatch, while remaining reviewable;
- exposes a suspected objective conflict that P3 diagnostics can test;
- exercises leakage/provenance rules on a real method;
- forces a meaningful boundary decision backed by two consumers;
- has a published result suitable for a credible Tier 2 target;
- addresses a real XTY experiment rather than filling out a taxonomy.

Deprioritise items that are merely another architecture swap, have no credible
reproduction target, or require large framework expansion before they answer a
research question.

When a recipe fails to fit cleanly, classify the failure before changing the
framework:

1. **missing component/objective/view** — local extension, no framework issue;
2. **missing row-selection or scheduling mechanic** — see whether an existing
   objective/schedule can own it;
3. **synthetic-row semantics** — compare MixMatch/ReMixMatch consumers;
4. **persistent training state** — compare SimMatch/Temporal Ensembling/etc.;
5. **bilevel optimisation semantics** — compare MPL with a second real method;
6. **new statistical quantity** — require matching consumers and a `PortSpec`;
7. **new executor semantics** — require two methods that genuinely cannot be
   represented by existing executors.

---

## 17. What success looks like after P12

The desired trajectory is not:

> five recipes -> forty recipes -> another monolithic registry.

It is:

> five validated recipes -> a growing library of independently validated
> components/objectives/views -> increasingly ambitious composite experiments.

The strongest evidence that xty2 worked would be an experiment that would have
required a new model class in XTYLearner but becomes, in xty2, a small reviewed
card and a declarative recipe assembling pieces already known to work.

An even stronger test is one of the Generation-3 methods above: a recipe in
which confidence, representation similarity, pseudo-labels or teacher/student
updates change what another objective sees, while the resulting execution plan
still makes every dependency and training signal understandable. ReMixMatch,
CoMatch and Meta Pseudo Labels + UDA are the three deliberate reference stress
tests for that claim.

---

## 18. Adjacent research directions worth investigating

The SSL backlog above mostly assumes the supervision state is binary: a treatment
label is either observed or missing, and unlabeled rows contribute through
likelihoods, pseudo-labels, consistency or representation objectives. Several
adjacent literatures suggest a broader question: **what information do we
actually possess about an apparently unlabeled treatment, and what is the
statistically legitimate way to use each kind of information?**

These directions may ultimately matter more than adding another pseudo-label
variant.

### 18.1 Rich / weak supervision rather than observed-vs-missing labels

Real supervision can be weaker than an exact class label without being absent.
Useful research families include:

- partial-label learning: the true treatment is known to lie in a candidate set;
- complementary-label learning: a row is known *not* to belong to one or more
  treatment classes;
- positive-unlabeled and class-specific partially observed labels;
- pairwise same/different-treatment constraints;
- triplet or ranking constraints;
- noisy heuristic / labeling-function supervision in the Snorkel / data-
  programming tradition;
- multiple noisy annotators or operational rules with different coverage and
  accuracy;
- learning from label proportions or aggregate treatment frequencies;
- combinations of partial labels and completely unlabeled examples.

**xty2 experiment:** do not immediately collapse weak evidence to one pseudo-label.
Let separate objectives consume the evidence in its native form where possible.
For small `K`, partial-treatment likelihoods are particularly natural: sum or
normalise probability mass over the admissible candidate set rather than choosing
one class.

**Potential boundary:** v1 `XTYBatch` only carries exact observed treatment plus a
boolean mask. Do not widen it to a generic supervision object in advance. First
specify at least two concrete rich-supervision recipes and identify the minimum
typed representation they genuinely share.

### 18.2 Learning Using Privileged Information / teacher-only variables

LUPI and generalized distillation study variables available during training but
unavailable to the deployed predictor. This gives another interpretation of
`q(t|x,y)` and of operational datasets with expensive or post-hoc information.

Possible privileged information includes:

- `y` when the deployed propensity model must use `x` only;
- high-resolution sensor channels available only for a subset;
- future observations or maintenance findings available retrospectively;
- engineering calculations too expensive for production inference;
- manually derived features or expert annotations available only in training;
- richer data sources used by a teacher but absent from the student input.

Candidate programmes:

- rich-view teacher -> `x`-only student distillation;
- posterior `q(t|x,y)` -> propensity `p(t|x)` distillation with leakage rules
  explicit;
- privileged representation pretraining -> deployable representation student;
- privileged teacher + Mean Teacher / Noisy Student style iteration;
- cross-fitted privileged-information distillation for causal use.

This direction is valuable because it separates **information used to learn**
from **information required at prediction time** rather than treating every
training-only variable as forbidden leakage.

### 18.3 Missing-treatment causal inference as its own research branch

Do not reduce missing treatment to ordinary semi-supervised classification.
There is a causal-inference literature deriving estimators whose target remains a
causal estimand while treatment is missing, sometimes allowing missingness to
depend on observed outcomes.

Priority topics:

- efficient influence-function estimators for ATE with partially missing
  treatment;
- CATE and policy-learning estimators with missing treatment;
- MAR / MCCAR missing-treatment estimators;
- double- and multiply-robust estimators;
- MTRNet-style representation balancing across both treatment groups and
  treatment-observation groups;
- outcome-assisted multiple imputation of missing treatment;
- multiple-imputation pooling rather than single pseudo-label substitution;
- semi-supervised estimation when both treatment and outcome have gold-standard
  labels only on a subset and noisy surrogates exist for everyone;
- sensitivity analysis for misspecified or MNAR treatment missingness.

**Important distinction:** using `y` to infer missing `t` is not inherently
invalid for causal estimation. What matters is the estimating procedure and its
assumptions. The naive programme

```text
q(t|x,y) -> hard treatment label -> pretend observed -> ordinary outcome fit
```

is different from an estimator derived to remain valid while integrating or
imputing missing treatment using outcome information.

This branch should therefore have its own Tier 2 targets and statistical cards,
not merely reuse predictive SSL benchmarks.

### 18.4 Safe SSL and open-environment SSL

Unlabeled data can hurt. Distribution shift, unseen classes, violations of the
cluster assumption and contaminated unlabeled pools all undermine the usual
intuition that more unlabeled data is automatically useful.

Research directions:

- safe SSL methods designed not to underperform the supervised baseline;
- open-set / open-world SSL with unknown treatment classes or contaminating
  observations;
- OOD detection before pseudo-label admission;
- robust pseudo-labeling under unlabeled distribution shift;
- covariate/label/treatment-prior shift correction;
- uncertainty- or conformity-gated use of unlabeled rows;
- self-supervised objectives for suspicious rows while excluding them from
  treatment pseudo-label losses;
- per-objective robustness: a row may be unsafe for pseudo-label CE but still
  useful for reconstruction or contrastive learning.

**xty2 experiment:** make "unlabeled data hurts" a first-class benchmark outcome.
For every composite recipe, compare against the supervised-only baseline and log
which objective changes sign or becomes conflicting as contamination increases.

### 18.5 Expectation constraints, posterior regularisation and aggregate knowledge

Some supervision applies to populations or expectations rather than individual
rows. Generalized Expectation / posterior-regularisation style methods can train
from statements such as:

- treatment class frequencies should approximately match a known marginal;
- a regime should almost never contain treatment class `k`;
- treatment distribution conditional on a known group should satisfy a prior
  proportion;
- a feature/treatment relationship should be monotone or directionally
  constrained;
- aggregate totals from an operational reporting system are trusted even when
  row-level labels are not;
- domain rules specify moments or inequalities rather than labels.

Candidate objectives:

- marginal treatment-proportion matching;
- group-conditional proportion losses;
- expectation constraints on propensity predictions;
- moment matching / maximum-entropy constraints;
- posterior regularisation;
- weak monotonicity or ordering constraints;
- calibration to external fleet/site/period totals.

This is a genuinely different way to use "unlabeled" data: impose knowledge on
the distribution of predictions without inventing per-row labels.

### 18.6 Graph and manifold SSL across observations

The current view machinery mostly relates multiple versions of one row. Graph SSL
adds structure **between rows**.

Possible graphs:

- nearest neighbours in raw or learned operating-condition space;
- same-tail / nearby-flight relationships;
- route-aircraft-regime similarity;
- neighbourhoods on an estimated physical/performance manifold;
- graphs from representation similarity updated during training;
- domain/expert-defined adjacency.

Candidate methods:

- label propagation;
- graph Laplacian / smoothness regularisation;
- graph contrastive learning;
- GNN propensity or posterior components;
- graph-based pseudo-label refinement;
- co-training between row-local and neighbourhood predictors.

**Leakage warning:** graph construction can leak test rows, future observations,
repeated-aircraft information or outcome-derived similarity. Any graph artifact
needs explicit provenance and fold-aware construction before causal use.

### 18.7 Set-valued, credal and conformal pseudo-supervision

Hard pseudo-labels throw away uncertainty. Soft labels retain probabilities but
still commit to one estimated distribution. Partial-label, credal and conformal
methods suggest intermediate supervision objects.

Candidate directions:

- conformal candidate-treatment sets;
- partial-label objectives over candidate sets;
- credal pseudo-labels / sets of admissible probability distributions;
- conformal admission rules instead of fixed confidence thresholds;
- weak-supervision conformal prediction;
- semi-supervised conformal calibration using unlabeled scores;
- treatment prediction sets used as inputs to downstream multiple-imputation or
  robust causal procedures.

With small categorical `K`, this is especially attractive: a row can remain
explicitly `{1, 2}` rather than becoming class 1 at probability 0.71 because an
arbitrary threshold was crossed.

### 18.8 Automatic search over validated objectives, views and programs

Once xty2 has a meaningful library of independently validated pieces, the object
of optimisation can become the **training recipe itself**, not just ordinary
hyperparameters.

Possible search dimensions:

```text
objective subset
x view / augmentation family
objective weights
warm-up / introduction times
eligible-row policies
pseudo-label thresholds or weighting rules
pretrain-vs-joint-vs-staged ordering
parameter freezing / update frequencies
```

Relevant research includes AutoSSL, meta-learning of unlabeled-example weights,
neural architecture/search ideas applied to SSL, and automatic composition of
self-supervised pretext tasks.

Guardrails:

- search only over components/objectives/views that already have their own
  invariants or fidelity evidence;
- use a genuine outer validation split so recipe search does not become another
  route to benchmark overfitting;
- record the complete compiled program as the searchable object;
- distinguish searching weights inside one card from discovering a genuinely new
  recipe whose mechanics need their own card;
- do not make the search engine a privileged way to bypass `DESIGN.md` §11.2 —
  a search that discovers it "needs" an abstraction is not a card, and §11.2 Q1
  asks about a card.

Long term, this is one of the clearest payoffs of decomposing XTYLearner: the
search space becomes structured and inspectable instead of "choose one of forty
monolithic classes."

### 18.9 Tabular foundation models and synthetic-task pretraining

TabPFN and successors suggest a different scaling direction: pretrain across a
distribution of synthetic tabular learning problems, then amortise inference on
new datasets.

Possible xty2-adjacent work:

- benchmark TabPFN/TabICL/other tabular foundation models as propensity or
  outcome nuisance learners;
- use tabular foundation models as teachers for smaller xty2 components;
- generate synthetic semi-supervised tasks from structural causal models;
- pretrain encoders across synthetic `(X,T,Y)` DGPs with varied confounding,
  missing-treatment mechanisms, heterogeneity and noise;
- investigate an "XTY-PFN" style model amortising inference across families of
  causal/semi-supervised tasks;
- self-supervised task generation from one real table, followed by transfer to
  related tables.

This is probably a separate research programme rather than a v1 framework
extension. Treat PFN/foundation models first as external components/baselines.

### 18.10 Incomplete multi-view learning

If `x` eventually consists of multiple semantically distinct data sources rather
than one fully observed `[B,D]` tensor, the incomplete-multi-view literature is
more relevant than simply concatenating columns and imputing missing values.

Potential views:

- flight/QAR dynamics;
- flight-plan or dispatch variables;
- weather;
- loadsheet information;
- aircraft history / maintenance;
- airport/network context;
- manually reviewed engineering features.

Candidate mechanisms:

- cross-view reconstruction;
- view-specific encoders with shared latent space;
- contrastive alignment across available views;
- graph propagation between partially observed view combinations;
- privileged-view distillation;
- missing-view completion;
- consistency losses only over pairs of views actually observed for each row.

This would genuinely challenge v1's single-tensor input contract. Do not widen
`XTYBatch` until two concrete multi-view recipes require it and the semantics of
missing views are specified.

### 18.11 Fully generative semi-supervised models

The Kingma M2 / auxiliary deep generative-model lineage treats labels as latent
variables inside a joint generative model rather than attaching pseudo-labels to
a discriminative classifier.

Potential directions:

- M2-style latent-treatment generative model;
- auxiliary-variable variational models;
- explicit joint modelling of `p(x,t,y)`;
- normalising-flow or diffusion-based tabular joint models;
- generative missing-treatment imputation integrated with likelihood training;
- posterior predictive uncertainty over treatment and outcome jointly.

xty2 already has much of the probabilistic vocabulary needed for
`p(t|x)`, `q(t|x,y)` and `p(y|x,t)`. A fully generative `p(x|...)` path should
still be added only when real recipes justify the additional semantic quantities.

### 18.12 Priority order for adjacent research

If choosing research questions rather than filling a taxonomy, prioritise:

1. **Missing-treatment causal inference** — changes what is statistically valid,
   not merely which SSL trick performs best.
2. **Rich / weak treatment supervision** — exploits information currently thrown
   away by a binary observed/missing representation.
3. **Expectation and aggregate constraints** — uses domain knowledge without
   manufacturing row labels.
4. **Privileged-information learning** — formalises training-only information and
   clarifies the role of `q(t|x,y)`.
5. **Conformal / set-valued supervision** — preserves ambiguity instead of
   thresholding it away.
6. **Safe/open-environment SSL** — asks when unlabeled data should *not* influence
   a given objective.
7. **Automatic recipe search** — becomes compelling only after the component
   library has genuine independently validated breadth.
8. **Graph, multi-view and foundation-model directions** — potentially large
   programmes, best pursued when a concrete dataset/use case demands them.

The shared principle is broader than semi-supervised classification: **preserve
the form of evidence you actually have for as long as possible, and let each
objective consume only the information its statistical assumptions justify.**

---

## 19. Framework and composability research

Detailed architectural notes belong in `PRIOR_ART.md`; this section records only
framework-related research tasks that could affect the post-P12 programme.

### 19.1 Source-code studies

Study these frameworks before inventing a new xty2 abstraction in the same area:

1. **PyTorch Metric Learning** — strongest current precedent for semantic
   intermediate contracts (`miner -> distance -> loss -> reducer`) and explicit
   compatibility/conversion rules between independently reusable pieces.
2. **MosaicML Composer** — strongest execution precedent for independently
   composable, stateful interventions attached to declared training events.
3. **VISSL** — strong precedent for assembling SSL tasks from trunks, heads,
   losses, transforms, schedules and lifecycle hooks rather than one monolithic
   method class.
4. **OpenMixup** — larger-scale registry/config composition across supervised,
   self-supervised and semi-supervised research.
5. **LightlySSL** — reusable SSL primitives and persistent mechanisms such as
   memory banks, momentum models and neighbour structures.
6. **solo-learn / Dassl** — useful negative/control cases for the common
   "shared trainer + algorithm subclass" equilibrium also seen in SemiLearn.
7. **WRENCH** — relevant if rich/weak supervision becomes concrete, because it
   treats supervision/label models as a separate layer from downstream models.

For each study, record observations in `PRIOR_ART.md`, not `DESIGN.md`. Promote
only a binding decision supported by real xty2 consumers.

### 19.2 Harmony-style crowded objective composition

Harmony (TMLR 2025) deliberately combines weak supervision, discriminative
self-supervision, generative self-supervision and EMA soft targets, with multiple
objectives active at once and subsets available for ablation.

Reference: https://openreview.net/forum?id=IcOBCufqFO

This is a useful scientific template for a future XTY composite recipe:

- supervised treatment signal;
- exact missing-treatment likelihood;
- discriminative representation SSL;
- generative/reconstruction SSL;
- EMA teacher soft targets;
- optionally causal/outcome objectives.

The goal is not to reproduce Harmony's vision setup. It is to reproduce the
**experimental structure**: independently meaningful signals, arbitrary subsets,
and diagnostics that show which combinations cooperate or interfere.

### 19.3 Composition interaction and order effects

The ICLR 2025 *Composable Interventions for Language Models* work is relevant to
the science of combination even though it is outside SSL. It treats interaction
and ordering between independently developed interventions as measurable objects
and finds that composition can be non-commutative and mutually interfering.

Reference:
https://proceedings.iclr.cc/paper_files/paper/2025/hash/7f5f9a88c6516469c83d074c6f2976fb-Abstract-Conference.html

Possible xty2 research questions:

- Does objective A help alone but hurt when B is active?
- Is `pretrain(A) -> joint(B,C)` different from `joint(A,B,C)` at matched compute?
- Does the order in which objectives are introduced matter after controlling for
  total optimization steps?
- Do gradient-cosine conflicts predict actual negative transfer?
- Does a stateful mediator change another objective's effective row population
  enough to explain gains/losses?
- Are interactions stable across labeled fractions and missing-treatment
  mechanisms?

This is one reason to retain raw per-objective losses, coverage, gradient norms
and gradient cosines even when a composite recipe appears to work.

### 19.4 Framework-abstraction decision checklist

When a post-P12 recipe appears to need a new framework concept, compare the
failure against existing framework precedents before adding it:

| Suspected missing concept | Prior art to inspect first |
|---|---|
| stateful prediction transform / policy | SemiLearn, Composer |
| memory bank / neighbour state | LightlySSL, SemiLearn |
| semantic intermediate object | PyTorch Metric Learning |
| config/recipe composition | VISSL, OpenMixup |
| rich supervision object | WRENCH |
| lifecycle/event hook | Composer, VISSL, SemiLearn |
| method subclass pressure | solo-learn, Dassl, SemiLearn |

The question is not whether another framework has the abstraction. The question
is whether that framework's experience reveals a stable contract that **two real
xty2 recipes also require**.
