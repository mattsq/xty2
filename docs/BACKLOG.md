# xty2 — Post-P12 research backlog

**Status:** candidate queue, not an implementation plan
**Start here:** docs/README.md

Use this file only to choose or scope a new research packet. For an existing
recipe, use docs/RECIPES.md and its card instead. PLAN.md is completed history;
DESIGN.md and FIDELITY.md govern architecture and evidence.

A candidate becomes work only when someone:

1. states the experiment and estimand;
2. writes `docs/recipes/<name>.md` and stops for review;
3. tries the existing component, objective, view, schedule, artifact, and
   executor vocabulary first;
4. records any unimplemented paper mechanic as a typed framework-limitation;
5. adds a framework concept only under DESIGN.md §11.2; and
6. predeclares Tier 0, Tier 1, and Tier 2 evidence.

Git retains the exploratory chronology removed from this active queue.

---

## 1. High-priority post-P12 tranche

| Candidate | Why it is useful now | Principal boundary |
|---|---|---|
| CoMatch | tests whether class and instance structure can cooperate after SCARF did not improve the end task | mutual graph/pseudo-label interaction |
| PAWS | uses labelled support embeddings instead of instance negatives | stratified support sampling and metric labels |
| ReMixMatch | deliberately combines several validated mechanisms | distribution alignment, adaptive augmentation, and row mixing |
| SimMatch | refines labels across semantic and instance spaces | memory-bank lifecycle |
| Meta Pseudo Labels + UDA | makes a teacher update depend on student validation loss | meta-gradient executor |
| Variational latent treatment | uses q(t\|x,y), p(t\|x), and p(y\|x,t) together | ELBO composition |
| Explicit treatment-observation model | promotes label availability to a statistical variable | missingness semantics |

Already present: SCARF is deviating, FixMatch is reproduced, DoubleMatch is a
draft, and FlexMatch is a draft. Read their cards before proposing a close
relative; their active ledgers supersede the historical notes once carried
here.

### Recommended stress-test sequence

1. CoMatch or PAWS: test class-compatible representation learning.
2. ReMixMatch: crowd one recipe without adding procedural recipe code.
3. SimMatch: test explicit historical state.
4. Meta Pseudo Labels + UDA: test the optimisation boundary.
5. A latent-treatment ELBO: compose all three native probabilistic quantities.
6. An explicit missingness model: decide whether availability needs vocabulary.
7. A crowded XTY experiment only after its ingredients pass independently.

Each step should probe one architectural claim, not reproduce a leaderboard.

## 2. Composite SSL lineage

### 2.1 S4L

Joint supervised and self-supervised objectives are a composition test. Add no
special S4L executor unless the objectives cannot share an ordinary stage.

### 2.2 MixMatch

Augmentation averaging, sharpening, and MixUp make synthetic rows the hard part.
Treat MixUp as the §15.1 boundary test; do not disguise it as a view over the
same row identity.

### 2.3 ReMixMatch ★

ReMixMatch combines MixMatch with distribution alignment, augmentation
anchoring, rotation, and adaptive strong augmentation. It is the preferred
crowded-recipe test once its ingredients exist independently.

### 2.4 UDA

UDA needs weak/strong consistency, confidence masking, sharpening, and training
signal annealing. Most of it should fit existing views, objectives, and
schedules; its acceptance test should isolate TSA and sharpening.

### 2.5 FixMatch and adaptive descendants

FixMatch and FlexMatch are shipped cards. FreeMatch, SoftMatch, and
SequenceMatch are the next threshold-policy comparisons. Keep each objective's
state local and explicit; generalise only after two cards demonstrate the same
lifecycle and semantics.

### 2.6 DoubleMatch

DoubleMatch is a shipped draft. Its row-population claim is load-bearing:
feature consistency trains all eligible unlabelled rows, including rows rejected
by the pseudo-label gate. A future result must compare the exact zero-weight
ablation and monitor representation scale, concentration, and alignment.

### 2.7 CoMatch ★

CoMatch couples class probabilities, projected embeddings, and a similarity
graph. X_PROJ already exists; the new question is whether graph-refined targets
can remain declarative and provenance-safe.

### 2.8 SsCL

SsCL couples classification and contrastive branches. Use it only after the
individual branches have independent baselines so interaction effects are
identifiable.

### 2.9 SimMatch ★

SimMatch aligns semantic and instance similarities against a memory bank.
Specify ownership, update order, reset behaviour, and checkpoint semantics
before code.

### 2.10 PAWS

PAWS builds soft assignments from labelled support representations. Its sampler
should be expressible as a stratified Quota beside an unlabelled quota; this is
why QuotaSampler is generic rather than labelled/unlabelled-specific.

### 2.11 Meta Pseudo Labels + UDA ★

The teacher update depends on how a student trained on teacher labels performs
on labelled data. If this cannot be represented by ordinary stages, it is
evidence for the §15.3 meta-gradient boundary, not permission to hide a loop in
a recipe function.

### 2.12 SelfMatch, SimCLRv2 and Noisy Student

These are staged-program candidates: pretrain or teacher fit, artifact
production, then fine-tune or student fit. Prefer explicit artifacts and
initialise_from edges over new executors.

### 2.13 Semi-supervised clustering / shared prototypes

Prototype methods test whether a class-indexed semantic object deserves a
contract. Start recipe-local; promote only when two methods need compatible
shape, update, and lifecycle rules.

### 2.14 SemiLearn / USB as software prior art

Use these suites to cross-check method decomposition and defaults, not as an
architecture to copy. xty2 keeps recipes declarative and rejects hidden
framework-wide method branches.

## 3. Semi-supervised treatment-label methods

### 3.1 Pseudo-labelling and self-training

Compare hard, soft, confidence-gated, and calibrated labels with the same data
stream and outcome stack. Preserve observed treatments and fill only missing
rows.

### 3.2 Consistency methods

Pi-model, VAT, temporal ensembling, and teacher methods should differ through
views, targets, stop-gradient direction, schedules, and state—not bespoke
training loops.

### 3.3 Dynamic eligibility and adaptive mechanisms

Curriculum gates, hard-negative mining, and active acquisition make eligibility
depend on learned state. State that dependency explicitly and test update order.
A sampler that reads the model is a new boundary because it breaks paired batch
streams.

### 3.4 Entropy, posterior and latent-variable objectives

Entropy minimisation, EM, and ELBOs are useful because xty2 already exposes
p(t\|x), q(t\|x,y), and p(y\|x,t). Require a single written probabilistic
objective before composing terms.

### 3.5 Multi-model SSL

Co-training, tri-training, and student ensembles require named realisations or
artifacts. Do not encode model identity in ad hoc tensor keys.

## 4. Explicit modelling of treatment-label availability

A model for R = 1{t observed} is justified only by an estimand and assumptions
that use it. Distinguish MCAR/MAR/MNAR, declare which variables the mechanism
sees, and do not imply that likelihood alone identifies an MNAR problem.

## 5. Self-supervised representation learning

### 5.1 Contrastive, clustering and redundancy-reduction methods

Candidates: CoMatch, PAWS, SimMatch, VICReg, Barlow Twins, BYOL, and SimSiam.
Before tuning, measure representation norm, collapse concentration, and whether
same-treatment rows are treated as negatives. Batch-coupled objectives must
declare and bind batch size.

### 5.2 Reconstruction and corruption methods

VIME and SubTab test mask prediction, empirical-marginal replacement, subset
views, and reconstruction outputs. Reuse TrainingPopulation for fitted
statistics; add a semantic port only when a card consumes it.

### 5.3 Composite pretraining and joint recipes

Compare pretrain-then-fine-tune against joint objectives with matched steps and
streams. A staged gain does not establish that simultaneous composition helps.

## 6. Views and augmentation strategies

Every view must declare whether it preserves row identity and the treatment
label. For a synthetic DGP, measure Bayes-label flip rate before training.
Candidate operations include masks, bounded jitter, empirical corruption,
feature subsets, and domain-valid monotone transforms.

### 6.1 Row-mixing methods as an explicit boundary test

MixUp, CutMix-like tabular mixtures, and manifold interpolation synthesize a new
row. They cannot inherit one source row_id or treatment availability silently.
Resolve identity, provenance, target, and artifact-join semantics first.

## 7. Causal representation and treatment-effect methods

### 7.1 Neural causal representation learners

CFRNet, DragonNet, DragonNet targeted regularisation, and balancing penalties
should reuse outcome/propensity ports. Add only the objective or component the
card actually needs.

### 7.2 Meta-learners and orthogonal estimators

S/T/X/R-learners and doubly robust learners belong behind explicit staged or
array actions. Preserve nuisance-fit row provenance.

### 7.3 Array/cross-fit estimators

Candidates include repeated DML, causal forests, and TMLE-like updates. The
first repeated-cross-fit card must resolve the open ledger debt rather than
overloading one fold_id.

### 7.4 Semi-supervised causal combinations

Evaluate imputation, representation, and outcome effects separately. Do not
transfer a supervised estimator's inference theorem to pseudo-labelled
treatments without a new argument.

## 8. Alternative outcome parameterisations

Candidates include heteroskedastic Gaussian heads, quantile models, mixtures,
flows, and survival/count distributions. They must satisfy the
OutcomeDistribution contract and state how means, log probabilities, and
samples are evaluated.

## 9. Encoder and parameterisation swaps

Architectural swaps are components, not recipes. Record initialisation,
normalisation, width/depth, and scale diagnostics because cosine objectives are
sensitive to representation norm.

## 10. Multi-objective optimisation and scheduling

Start with scalar schedules and logged per-objective losses, gradient norms, and
cosines. Only add gradient surgery or learned weights after a paired experiment
shows fixed weighting is the limiting mechanic.

### 10.1 Bilevel/meta-learning boundary

Meta-gradients, validation-driven weights, and unrolled optimisation need an
explicit executor contract for differentiable inner state, truncation, and
provenance.

## 11. Distillation, ensembling and uncertainty

### 11.1 Distillation

Specify teacher source, temperature, target rows, stop-gradient, and whether the
teacher is fixed, EMA, or produced by another stage.

### 11.2 Ensembles and approximate posterior uncertainty

Use named realisations or checkpoints. Define aggregation and seed ownership;
do not infer an ensemble from repeated calls to one stochastic model.

### 11.3 Memory banks and historical state

Memory banks, queues, temporal predictions, and curriculum marks require
explicit objective or stage state. Declare initialisation, update order, reset,
serialization, and population key. Never use module-global mutable state.

### 11.4 Calibration and abstention

Calibration and abstention are evaluation or decision layers unless they change
training. Predeclare the calibration split and avoid tuning on the Tier 2 test
set.

## 12. Domain robustness and adaptation

Group/domain methods require declared metadata and split rules. A domain label
is not a generic batch field until two cards need the same semantics.

## 13. Active learning and label acquisition

Acquisition changes the training population over time and usually reads the
model. Treat it as an explicit stateful program or sampler boundary, with budget,
oracle, timing, and replay rules declared before implementation.

## 14. Composite experiments enabled by xty2

### 14.1 Representation + semi-supervision

Compare representation pretraining, semi-supervision, and their composition in
a factorial design.

### 14.2 Interacting representation/pseudo-label systems

CoMatch, PAWS, and SimMatch are preferred because they make representation
structure affect labels rather than merely add losses.

### 14.3 Semi-supervision + causal regularisation

Hold the semi-supervised treatment mechanism fixed while adding balance,
targeted, or orthogonal outcome terms.

### 14.4 Staged programs

Teacher/student iteration, pretraining, pseudo-label artifacts, and array
estimators should be explicit stages with named transitions.

### 14.5 Intentionally crowded S4L/ReMixMatch-style recipes

Crowded recipes are useful only after each ingredient has a baseline and the
combined card specifies ablations for interaction.

### 14.6 Row-utilisation experiments

Report which rows train each objective. Confidence rejection from one loss must
not accidentally exclude a row from an independent loss.

## 15. Deliberate framework stress tests

### 15.1 MixUp / synthetic-row semantics

Resolve row identity, masks, weights, targets, and provenance for synthetic
rows before adding a row-mixing view.

### 15.2 Stateful mediation between objectives

Initially keep each mechanism local and explicit. Consider a reusable mediator
or policy only if at least two real recipes need the same lifecycle and
semantics.

### 15.3 Bilevel/meta-gradient execution

Require a card that states the differentiable inner/outer update and why stages
cannot express it.

### 15.4 Memory-bank/state lifecycle

Specify key space, capacity, eviction, update order, device, reset, and
checkpoint behaviour.

### 15.5 Explicit missingness mechanism

Add availability vocabulary only when a statistical model consumes it; a data
split description alone is not a model.

### 15.6 Ensemble realisations

Define identity, parameter ownership, independent randomness, and aggregation.

### 15.7 Prototype/similarity semantics

Start with recipe-local tensors. Promote a semantic contract only when producer
and consumer shapes recur.

### 15.8 Domain/group metadata

Do not add multi-dataset or domain-keyed sources speculatively. The first card
must define whether metadata is covariate, supervision, stratification, or
evaluation-only.

### 15.9 Continuous/dose-response treatment

Continuous treatment changes candidate evaluation, overlap, distributions, and
card keys. Treat it as a new version boundary.

### 15.10 Sequence/time-series inputs

Temporal data changes row independence, splits, views, and leakage rules.
Require a dedicated design packet.

## 16. Backlog triage rules

A candidate is ready only if all answers are concrete:

| Question | Required answer |
|---|---|
| What estimand or mechanism is tested? | one falsifiable sentence |
| What is the nearest shipped baseline? | one card and one controlled difference |
| What existing vocabulary expresses it? | named ports/components/objectives/views/stages |
| What is genuinely missing? | card mechanic and DESIGN §11.2 quadrant |
| What evidence closes it? | Tier 0, Tier 1, and predeclared Tier 2 |
| What will not be claimed? | explicit scope and inference limits |

Reject packets whose purpose is “support method X” without an experiment.

## 17. What success looks like after P12

Success is not candidate count. It is that new methods are mostly card plus
declarations; plan digests and provenance stay stable; deviations are typed;
paired experiments isolate one mechanic; and framework additions are rare,
small, and justified by evidence.

## 18. Adjacent research directions worth investigating

These are lower priority than the tranche in §1. Promote one only with a
specific XTY experiment.

### 18.1 Rich / weak supervision rather than observed-vs-missing labels

Partial, noisy, pairwise, or aggregate treatment labels need explicit
observation semantics and targets.

### 18.2 Learning Using Privileged Information / teacher-only variables

Teacher-only variables require a declared training-only port and a test that
serving paths cannot read it.

### 18.3 Missing-treatment causal inference as its own research branch

Separate identification assumptions from implementation. Compare imputation,
likelihood, and sensitivity analyses under controlled missingness mechanisms.

### 18.4 Safe SSL and open-environment SSL

Distribution shift, OOD rows, and harmful unlabeled data need abstention or
robustness targets, not just better average likelihood.

### 18.5 Expectation constraints, posterior regularisation and aggregate knowledge

Constraints belong in explicit objectives with feasibility and dual-update
diagnostics.

### 18.6 Graph and manifold SSL across observations

Cross-row graphs are batch/population-coupled and may need state or finite-data
execution.

### 18.7 Set-valued, credal and conformal pseudo-supervision

Represent uncertainty explicitly; do not encode a set-valued target as an
ordinary hard pseudo-label.

### 18.8 Automatic search over validated objectives, views and programs

Search only over validated declarations and keep the selected plan and search
budget reproducible.

### 18.9 Tabular foundation models and synthetic-task pretraining

Treat pretrained encoders as versioned artifacts with frozen provenance and
clear fine-tuning rules.

### 18.10 Incomplete multi-view learning

Distinguish missing feature views from missing treatment labels; they require
different masks and likelihoods.

### 18.11 Fully generative semi-supervised models

A joint p(x,t,y) is a different modelling commitment from xty2's current
conditional quantities. Add it only for an explicit estimand.

### 18.12 Priority order for adjacent research

Prefer: missingness identification, safe SSL, weak supervision, then generative
or search-heavy directions. Evidence and available data may reorder them.

## 19. Framework and composability research

### 19.1 Source-code studies

Study mature libraries to identify failure modes and decomposition choices.
Record conclusions in PRIOR_ART.md; do not turn library APIs into requirements.

### 19.2 Harmony-style crowded objective composition

Test whether validated objectives cooperate under matched streams, budgets, and
diagnostics before designing a composition framework.

### 19.3 Composition interaction and order effects

Use factorial ablations, schedule/order swaps, and gradient diagnostics. A
combined gain alone does not identify which interaction helped.

### 19.4 Framework-abstraction decision checklist

Before changing core vocabulary, answer: which card mechanic is impossible,
which quadrant applies, what is the smallest contract, which second consumer
checks its shape when required, and which existing plans remain byte-stable.
