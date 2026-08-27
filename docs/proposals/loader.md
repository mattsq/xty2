# Proposal — bringing data loading into the model pipeline

**Status:** draft, for review. **Nothing described here is built.**
**Reads with:** `DESIGN.md` §7 (program), §9.1 (card-key binding), §11.2 (the
abstraction test), §11.4 (the ledger); `FIDELITY.md` §2 (the checklist), §5 (the
two kinds of deviation).
**Discharges, if accepted:** the `loader` ledger row, and with it
`fixmatch` §5.4, `scarf` §5.6 and `tarnet` §5.5.

This is the framework analogue of a spec card: written and reviewed *before* the
code, for the same reason (`FIDELITY.md` §1). It is not a description of the
repository as it stands, and it deliberately does not touch `DESIGN.md` §11.4
— the ledger row stays until the packet lands.

---

## 0. What is missing, stated as the debt it already is

The executor contract is one line:

```python
BatchSource = Iterable[XTYBatch]      # xty2/training/executors.py
```

and its docstring says why: *"Sampling, the labelled/unlabelled mix and the
epoch boundary are the caller's, because no loader exists yet and a half-built
one here would own card keys it could not check."* That was the right call when
it was made. It is now being paid for by three reviewed cards, and the payments
are not the same kind of thing:

| Card | Row | What the paper says | What we do |
|---|---|---|---|
| `fixmatch` | §5.4 | eq. (5) mixes a labelled batch of `B` with an unlabelled batch of `mu B`, `mu = 7` | whatever the caller's batch happens to contain |
| `scarf` | §5.6 | `L_cont`'s negatives are `N - 1`, so the loss's arithmetic *is* the batch size | correct at any size, meaning something different at each |
| `tarnet` | §5.5 | the published number depends on a declared split and standardisation | enforced by the fixture, invisible to the plan |

A fourth is entangled and named as such in the ledger: `view-population-statistics`
(`scarf` §5.2), where SCARF's corruption should draw from the training set's
empirical marginal and draws from the batch's instead, "since a training
population is the thing a loader would own".

These are three different failures, and the third is the one that matters most
for what this framework claims about itself:

1. **A method mechanic cannot be expressed** (`fixmatch`). The batch composition
   *is* part of FixMatch.
2. **An objective's arithmetic is unpinned** (`scarf`). `InfoNCEContrastive` is
   the repository's first objective whose per-row value depends on the other
   rows of the batch, and nothing in the compiled recipe says how many there are.
3. **A leakage point is outside the plan** (`tarnet`). `tarnet` §5.5 states the
   cost precisely: *"a later runner could fit it on the wrong split and nothing
   in the plan would say so — the leakage point `FIDELITY.md` §2 names first."*
   `DESIGN.md` §7.2 makes treatment-label leakage a compile-time rule; feature
   standardisation fitted on the test split is the same class of error, and it
   currently has no rule at all.

There is also an ordinary duplication argument, recorded but not load-bearing:
`_take`, `_BatchStream` and `_batch_indices` are reimplemented in
`tests/smoke/test_fixmatch.py`, `test_mean_teacher.py`, `test_scarf.py`,
`test_tarnet.py`, `test_cnflow.py` and again in
`xty2/evaluation/benchmarks/common.py`. Under §11.2 that is a Q1-no convenience
argument and would not justify anything on its own. It is listed so that the
reviewer can see it was not what drove this.

---

## 1. The two questions, asked in order (`DESIGN.md` §11.2)

**Q1 — is it load-bearing for fidelity?** Yes, and not as a matter of judgement:
three reviewed cards already carry `framework-limitation` rows naming this
capability, and five card §4 keys — `optimisation.batch_size`,
`optimisation.labelled_unlabelled_ratio`, `data.split_protocol`,
`data.standardisation`, `data.missingness_mechanism` — are marked `n/a` in those
cards with the reason "xty2 has no loader" written beside them. §11.2's test is
"some key in some card's §4 mechanics checklist cannot be honoured as the paper
states it". Five keys, three cards, already written down.

**Q2 — what does being wrong about it cost?** **Load-bearing vocabulary.** It
changes the executor contract (§11.2 names that explicitly), it adds a run-time
object every future runner is written against, and it moves every existing
recipe's plan and therefore its digest.

That is the bottom-right quadrant: *build it now, one consumer, and design the
shape against a named second consumer, written down before the code.* §7 below
discharges that obligation for each piece separately, because the pieces have
different second consumers.

**One thing this proposal does not do, and the fact is load-bearing:** it adds
**no card keys**. Every key it gives a home to is already in the closed
vocabulary of `FIDELITY.md` §2. A capability that needs new vocabulary to
express itself is usually the wrong size; this one is being asked to fill five
slots the checklist has had open since P0.

---

## 2. The split that makes it buildable

`scarf` §5.1 states the obstacle that has kept this deferred, and it is a real
one:

> Building it here would fix the shape of that concept on the evidence of one
> transform, and would put a tensor of training rows inside a `Recipe`, which
> is a declaration of a method rather than a container for data.

The whole design follows from taking that sentence seriously. Two things have
been conflated under the word "loader", and they are known at different times:

| | What it is | Where it lives | When it is known |
|---|---|---|---|
| **Policy** | how a population is split, standardised, made missing, and turned into batches | the `Recipe` — declarative, compiled, plan-visible, digest-covered, card-key-bound | at authoring time |
| **Data** | the actual rows | the caller — passed to the executor, exactly as `BatchSource` is today | at run time |

The recipe never holds a tensor of training rows. It holds a *policy* the
executor applies to whatever population it is handed, and the plan prints the
policy, so a card key finally has something to bind to and a reviewer can diff
it.

The second principle is the one `DESIGN.md` §7.1 already uses for artifacts, and
it is the reason this is not just a config object: **the thing that does the work
is the thing that reports it, and the report is derived rather than declared.**
A loader supplied by the caller that *reports* the batch size it used would be a
provenance claim nobody can falsify — "a guardrail that any caller can talk its
way past is not a guardrail". So the executor builds the batches, from the
compiled policy, and the plan is the cause of the run rather than a description
of it.

---

## 3. The declarations

New module `xty2/core/data.py` (declarations — read by the compiler) and
`xty2/training/loading.py` (what runs them). `core` must not import `training`,
which is why the split falls this way (`DESIGN.md` §10).

### 3.1 `Dataset` — the population, supplied by the caller

```python
@dataclass(frozen=True)
class Dataset:
    """A finite, immutable row table. Never part of a Recipe."""
    schema: Schema
    rows: XTYBatch                             # the whole population, one batch object
    assignments: Mapping[str, Tensor] = ...    # named [N] long partitions over rows
```

A population is already an `XTYBatch` — `row_id` and `fold_id` are on it — so
this wrapper earns its place only through `assignments`, and that is deliberate.
`assignments` is a **mapping of named partitions**, not a single `split` field,
because `repeated-cross-fitting` (§11.4, paid for by `ssdml` §5.6) is blocked on
exactly one thing: *"Today an `XTYBatch` has a single `fold_id` ... so a second
partition has nowhere to live."* Naming partitions rather than fixing one is what
makes a second one additive later. This proposal does **not** discharge that row
— the artifact contract and the disjointness check are still written against one
assignment — but it stops being the thing standing in the way.

`Dataset` is immutable and no code path writes to it (`DESIGN.md` §7.1).

### 3.2 `DataSpec` — on the `Recipe`

```python
@dataclass(frozen=True)
class SplitSpec:
    protocol: str = REQUIRED           # data.split_protocol
    fractions: tuple[float, ...] | None = None
    stratify: Rows | None = None
    source: Literal["declared", "derived"] = "declared"

@dataclass(frozen=True)
class PreprocessSpec:
    features: Standardisation = REQUIRED     # data.standardisation
    outcome: OutcomeScaling = REQUIRED       # data.outcome_scaling
    fit_on: str = REQUIRED                   # the assignment name, e.g. "train"

@dataclass(frozen=True)
class MissingnessSpec:
    mechanism: str = REQUIRED          # data.missingness_mechanism
    rate: float | None = None
    keyed_by: Literal["row_id"] = "row_id"

@dataclass(frozen=True)
class DataSpec:
    split: SplitSpec = REQUIRED
    preprocess: PreprocessSpec = REQUIRED
    missingness: MissingnessSpec = REQUIRED
```

Recipe-level, not stage-level: a per-stage split would be a different and much
worse concept, since the leakage argument depends on there being *one* train
assignment for the run. `Recipe` gains `data: DataSpec = REQUIRED` and, for the
first time, `CARD_KEYS` of its own; `_hyperparameters` in `core/compile.py`
gains one recipe-scope owner beside the existing stage, objective and component
scopes (§9.1).

`source="declared"` covers `tarnet`, whose IHDP archive arrives pre-split: the
policy says *the split is the one the archive carries*, which is a statement the
plan can print and a reviewer can check, where `n/a` is not.

### 3.3 `TrainingPopulation` — what the policy produces

```python
@dataclass(frozen=True)
class TrainingPopulation:
    rows: XTYBatch                     # split, standardised, missingness applied
    assignment: str                    # which partition these rows are
    statistics: Mapping[str, Tensor]   # what the preprocessing fitted
    fitted_on_row_ids: Tensor          # [M] the rows those statistics were fitted on
    spec_digest: str
```

`fitted_on_row_ids` is the point of the object and it is a straight lift of
`Checkpoint.trained_on_row_ids` (`DESIGN.md` §7.1). Standardisation fitted on the
test split is the same shape of error as a checkpoint predicting rows it trained
on, and it gets the same treatment: the claim carries the rows that make it
checkable, and the check is **run** at load rather than trusted. As with
artifacts, `TrainingPopulation` is constructed only by the loading factory, which
is private to the executor — a factory anyone can call is a factory anyone can
hand invented row ids to.

`statistics` is a plain mapping of tensors, not a service. That is a shape
decision, argued in §7.3.

### 3.4 `SamplerSpec` — on the `Stage`

```python
@dataclass(frozen=True)
class Quota:
    rows: Rows                       # the population this quota draws from
    size: int                        # rows per step
    stratify: str | None = None      # a categorical quantity; `size` is then per level

@dataclass(frozen=True)
class UniformSampler:
    batch_size: int = REQUIRED       # optimisation.batch_size
    replacement: bool = False
    CARD_KEYS = {"batch_size": "optimisation.batch_size"}

@dataclass(frozen=True)
class QuotaSampler:
    quotas: tuple[Quota, ...] = REQUIRED
    CARD_KEYS = {
        "batch_size": "optimisation.batch_size",
        "labelled_unlabelled_ratio": "optimisation.labelled_unlabelled_ratio",
    }
    # both are @property, derived from `quotas` — see below

@dataclass(frozen=True)
class ExternalBatches:
    """The caller supplies batches. Explicit, never inherited. See §10."""
```

`Rows` is reused verbatim from `core/rows.py`. The sampler therefore speaks the
same row-population vocabulary as every objective and every stage scope, and
"which rows" needs no new words — only "how many" does.

**The two `QuotaSampler` keys are properties, not constructor arguments**, for
the reason `DESIGN.md` §7.1 gives about `PseudoLabels.used_y`: a field a producer
can set is a field a producer can set wrongly. `batch_size` is the sum of the
quotas and `labelled_unlabelled_ratio` is their ratio, so the plan prints the
ratio the sampler *runs*, and `mu = 7` cannot be claimed by a recipe that draws
64 and 64.

`UniformSampler` binds `batch_size` only. A card using it marks
`labelled_unlabelled_ratio: n/a` — and that `n/a` now means "this sampler
enforces no quota", which is a fact about the method, rather than "xty2 has no
loader", which is a fact about us.

---

## 4. What compiles, and what the plan prints

`compile(recipe)` gains: the `DataSpec` and each stage's `SamplerSpec` resolved
into the plan, five more entries in `plan.hyperparameters`, and the checks in §5.
The plan grows a `data` block per run and a `sampler` line per stage:

```
recipe fixmatch  purpose=causal
data
  split            declared assignments {train, test}          [data.split_protocol]
  standardisation  x: mean/std fitted on `train`               [data.standardisation]
  outcome scaling  y: mean/std fitted on `train`, metrics on original scale
  missingness      treatment MCAR at 0.500, keyed by row_id    [data.missingness_mechanism]

stage fit
  sampler          quota, without replacement, one draw per step
                     t_observed  64                            [optimisation.batch_size = 512]
                     t_missing  448                            [optimisation.labelled_unlabelled_ratio = 7.0]
  ...
```

Everything printed is digest-covered. This is the part with a real, unavoidable
cost, and §9 states it plainly: **every existing recipe's plan digest moves.**

---

## 5. What is checked, and where

Split by what is knowable when, exactly as `DESIGN.md` §7.2 splits the leakage
rule.

**At compile time**

1. **Preprocessing scope.** A `PreprocessSpec` whose `fit_on` is not the stage's
   training assignment is rejected for `purpose="causal"`, and opted out of
   per-stage under `purpose="predictive"` with `allow_leakage=True` — the same
   opt-out, the same requirement that it be written down and appear in the plan.
   This is the rule `tarnet` §5.5 says does not exist.
2. **Empty-by-construction quotas.** A `Quota(rows="t_missing")` inside a stage
   scoped `rows="t_observed"` is a compile error, not a permanently empty draw.
   This is `DESIGN.md` §7.0's rule applied to the sampler, and it is the same
   argument: emptiness by construction is a wiring bug; emptiness in a
   particular batch is a runtime fact to log.
3. **`ExternalBatches` against a batch-coupled objective.** See §10.

**At run time**

4. **Fitted-on check.** `fitted_on_row_ids` must be a subset of the declared
   assignment's rows. Verified against the actual row ids, never a label.
5. **A quota that cannot be filled is an error, not a short batch** — matching
   the existing rule that a batch source running dry is an error rather than a
   short run.
6. **Immutability.** The `Dataset` handed in is bit-identical afterwards; the
   existing batch-immutability invariant extends to cover it.

---

## 6. The bit-identity requirement

This is the single most important engineering constraint in the proposal, and it
is what makes the packet affordable.

**`UniformSampler` must reproduce the existing fixtures' batch sequence
bit-for-bit.** Today every fixture does the same thing:

```python
torch.stack([torch.randperm(rows, generator=g)[:batch_size] for _ in range(steps)])
```

so `UniformSampler` is *defined* as one fresh permutation per step, first
`batch_size` rows, and a Tier 0 test asserts its output equals
`xty2.evaluation.benchmarks.common.batch_indices` for a fixed seed and shape.

The consequence is the `TeacherSpec.role` precedent from `fixmatch` §5.1: plans
and digests move, **no arithmetic does**, and every recorded §6 number stands as
measured. Without this property the packet would silently re-run five reviewed
recipes under new noise, which `DESIGN.md` §2.1 names as the bar any new axis has
to clear.

Two seeding rules follow, and both are Tier 0 assertions rather than intentions:

- **The sampler has its own stream.** Its seed is
  `blake2b(f"{stage_seed}:sampler")`, hashed rather than offset, so it cannot
  collide with the per-step view keys that walk upward from `stage_seed`
  (`STREAM_STRIDE`, `MAX_STAGE_STEPS`).
- **The stream does not depend on the model.** Two recipes differing only in an
  objective weight draw identical `row_id` sequences. Every paired ablation in
  the repository — `fixmatch` §6's `lambda_u = 0` arm, `scarf` §6's
  with/without-pretraining pair — depends on this today by sharing a
  pre-computed index tensor; after the change they get it from the seed instead.

---

## 7. Second-consumer shape checks

§11.2's obligation for the load-bearing quadrant: name a specific `BACKLOG.md`
card, find the sentence of its paper that needs the same thing, and check the
shape against both *before* writing the code. Four pieces, four checks.

### 7.1 `QuotaSampler` — consumer today `fixmatch`, second consumer **PAWS**

`BACKLOG.md` §2.10. PAWS *"uses labeled support representations to construct soft
class assignments for unlabeled examples"* — the support set is drawn fresh each
step and is **class-balanced**: `k` labelled rows per class, alongside the
unlabelled batch.

**This changed the shape.** A `QuotaSampler` written for FixMatch alone would
have two fields, `labelled` and `unlabelled`. PAWS needs `k` rows *per level of a
categorical quantity*, which that shape cannot express and which a third field
would not fix. Hence `quotas: tuple[Quota, ...]` with an optional `stratify`:
FixMatch is `(Quota("t_observed", 64), Quota("t_missing", 448))`, PAWS is
`(Quota("t_observed", k, stratify="t"), Quota("t_missing", B))`. ReMixMatch and
UDA reuse the FixMatch shape unchanged.

### 7.2 `Dataset.assignments` — consumer today the train/test split, second consumer **repeated cross-fitting**

`DESIGN.md` §11.4, paid for by `ssdml` §5.6: a second partition over the same
rows has nowhere to live. A mapping of named partitions is the smallest shape in
which a second one is additive rather than structural. Checked, not discharged
(§8).

### 7.3 `TrainingPopulation` — consumer today `scarf`'s marginal, second consumer **ReMixMatch distribution alignment**

`BACKLOG.md` §2.3. **This check produced a deliberate refusal, which is the more
useful outcome.** ReMixMatch's distribution alignment maintains a running average
of *the model's predictions* over the unlabelled stream — a statistic of the run,
not of the data. Had `TrainingPopulation` been shaped as "the thing that answers
statistical questions", DA would have been its second consumer and the object
would have grown a model-dependent, time-varying interior.

So `statistics` is a plain mapping of tensors fitted once by the declared
preprocessing, and nothing else. SCARF's marginal is computed by the *transform*
from `population.rows`; DA's running average belongs to whatever ReMixMatch's
card decides, and is not this object's business. The second consumer's job here
was to say where the boundary is, and it did.

### 7.4 Population-reading transforms — consumer today `FeatureCorruption`, second consumer **VIME**

`BACKLOG.md` §5.2. VIME's mask-and-impute replaces each masked cell with a draw
from the same empirical marginal over the training set — the identical read, on
the identical object. See §8 for whether this lands in the same packet.

---

## 8. What this does to the ledger (`DESIGN.md` §11.4)

| Row | Effect |
|---|---|
| `loader` | **Deleted.** This is the discharge, and per `FIDELITY.md` §5.2 deleting it turns Tier 0 red on `fixmatch` §5.4, `scarf` §5.6 and `tarnet` §5.5 until each is revisited in the same PR (§9) |
| `view-population-statistics` | **Amended or discharged** — the reviewer's call, §12. Its stated blocker ("there is no training-population object anywhere in xty2 for it to read") ceases to be true the moment `TrainingPopulation` exists, so leaving the row's text unchanged would be exactly the stale register §11.3 is about |
| `early-stopping` | **Amended, not discharged.** Half its stated obstacle — *"there is nowhere for a validation split to live"* — goes away with `Dataset.assignments`. The other half, a stage that ends on a monitored metric rather than a `steps` budget, is untouched, and nothing is paying for the row. Its "Would build when" is rewritten to say which half is left |
| `repeated-cross-fitting` | **Amended, not discharged.** §7.2 above: shape-compatible now, but the artifact contract and the fold-disjointness check are still written against one assignment, and `ssdml` §5.6 keeps paying |
| `distributed` | Unchanged. Explicitly the reason there are no workers or prefetch here |
| **`out-of-core-data`** | **New row.** Not building streaming or larger-than-memory sources. Would build when a card's dataset does not fit in one process's memory. Nobody paying |
| **`stateful-sampler`** | **New row.** Not building a sampler whose behaviour depends on training state — curriculum sampling, hard-negative mining, active-learning acquisition (`BACKLOG.md` §3.3, §13). Would build when a card's §4 states a sampling rule that reads the model. Deliberately refused here: it would make the data stream a function of the model and destroy §6's paired-ablation property, which is a cost that should be paid knowingly by the card that needs it. Nobody paying |

---

## 9. What this does to the paying cards

Two of the three are clean withdrawals. All seven cards take a digest change.

| Card | Row | Outcome | Cost |
|---|---|---|---|
| `scarf` | §5.6 | **Withdrawn.** The pretrain stage declares `UniformSampler(batch_size=128)`; §4's `optimisation.batch_size` goes from `n/a` to `128`, which is what §6 already fixes | Card amendment. §6.2's numbers stand, by §6's bit-identity property. Status is `draft`; no Tier 2 row to invalidate |
| `tarnet` | §5.5 | **Withdrawn.** The recipe declares the split, standardisation and missingness policy; three §4 keys go from `n/a` to values, and §7's row about "executable ownership" is struck | Card amendment. Its `deviating` Tier 2 result stands iff the declared policy reproduces what the P12 runner does today — which must be **demonstrated in the packet**, not assumed. If it does not, the row is re-run or explicitly invalidated (`FIDELITY.md` §5.3) |
| `fixmatch` | §5.4 | **Withdrawn.** `QuotaSampler` enforces `mu = 7` | The expensive one. This is *new arithmetic*, not a rewiring: the batch composition changes, so §6's numbers must be **re-measured**. Status is `draft` with no Tier 2 row, which is why this is affordable now and will not be later |
| `scarf` | §5.2 | Withdrawn if §12's optional half lands; otherwise unchanged, with the ledger row amended to say the population now exists | Card amendment; the open question its §5.1 puts to the reviewer is finally answerable either way |
| `cnflow`, `cycle_dual`, `mean_teacher`, `ssdml` | — | No deviation changes. Each declares a sampler (or `ExternalBatches`, §10) and a `DataSpec`; plans and digests move, arithmetic does not | The §6 numbers of two `reproduced` and two `deviating` cards stand on §6's bit-identity property. **This is the packet's main risk and its main test.** A run directory written before the change will not accept a checkpoint written after it — the mechanism working, as `fixmatch` §5.1 says of the same situation |

---

## 10. The one decision this proposal does not make

**Is `ExternalBatches` a legitimate value, permanently?**

`Stage.sampler` is `REQUIRED` either way — there is no silent default and a
recipe that omits it does not construct. The question is whether "the caller
supplies batches" stays sayable.

**Recommendation: yes, with a compile-time bar.** Three reasons and one guard.

- `FIDELITY.md` §5.3 is explicit that repaying a debt is not licence to widen the
  PR, and that where the cost is real, restating rather than churning a reviewed
  card is the correct outcome. Forcing `cnflow`, `cycle_dual`, `mean_teacher` and
  `ssdml` — four cards, two of them `reproduced` — into card amendments they do
  not need is that churn.
- `tarnet` §4 reads `batch_size: n/a  # external BatchSource; ref impl uses 100`,
  while the Tier 1 fixture and the P12 runner use their own sizes. Binding `100`
  into the recipe would *change the run* that produced its recorded number. The
  honest declaration for that stage today is "the caller supplies batches", and
  a vocabulary that cannot say a true thing pushes cards into saying false ones.
- An explicit `ExternalBatches()` in the plan is a visible, greppable admission.
  Its absence is not.

**The guard, which is what stops it becoming the escape hatch:** `ExternalBatches`
is rejected at compile time for any stage containing an objective that declares
itself batch-coupled. `Objective` gains `batch_coupled: bool` as a required
member, on the same argument §4 gives for `detaches` — *"a declaration with a
fallback is one that can be forgotten"*. `InfoNCEContrastive` is `True` today, so
`scarf`'s pretrain stage **cannot** use the hatch; every likelihood term in the
repository is `False`. The escape hatch is closed exactly where batch size is
method-bearing, which is the whole of `scarf` §5.6's complaint.

Second consumer for `batch_coupled` (§11.2 obligation, since it is a required
protocol member): **Barlow Twins** and **VICReg** (`BACKLOG.md` §1, §5.1), whose
losses are computed from a cross-correlation or covariance matrix over the batch
dimension and whose value therefore depends on batch size in exactly the way
InfoNCE's does.

**The alternative** — no `ExternalBatches`, every stage declares a real sampler —
is defensible and the reviewer may prefer it. It buys a smaller vocabulary and
one fewer required objective member, and costs four card amendments and a
re-examination of what `tarnet`'s recorded number was measured under. It is
written here rather than decided so that the reviewer is asked, in the form
`scarf` §5.1 established for an open question.

---

## 11. Deliberately out of scope

Written down so the packet cannot quietly acquire them:

- **No `torch.utils.data.DataLoader`, workers, prefetch or collate.** v1 is
  tabular, in-memory and single-process (`distributed`, `out-of-core-data`).
- **No epochs.** `Stage.steps` stays optimiser steps, for the reason
  `FIDELITY.md` §2 already gives.
- **No stateful or curriculum samplers** (`stateful-sampler`, §8).
- **No multi-dataset or domain-keyed sources** (`BACKLOG.md` §15.8).
- **No early stopping** (`early-stopping`, amended not discharged).
- **No second fold partition** (`repeated-cross-fitting`, amended not discharged).
- **No new card keys.** §1.

---

## 12. The packet, and what it has to prove

Four parts. (a) through (c) are one packet and cannot be separated — Tier 0 is
red between (a) and (c) by design.

**(a) Declarations and population.** `core/data.py`, `training/loading.py`,
`Recipe.data`, the private factory, the compile-time preprocessing-scope rule and
the run-time fitted-on check.
*Accept:* the scope rule **rejects a standardisation fitted on the test split for
a causal recipe, and accepts it under `predictive` + `allow_leakage`**; the
fitted-on check **fails on a deliberately mis-attributed statistic** — the
mutation test is the criterion, mirroring P10's, because a provenance label that
nothing can falsify is not checked.

**(b) Samplers and the executor contract.** `Stage.sampler`, `UniformSampler`,
`QuotaSampler`, `ExternalBatches`, `Objective.batch_coupled`, plan printing,
hyperparameter binding.
*Accept:* **`UniformSampler` reproduces `batch_indices` bit-for-bit** (§6) and
**every recipe's Tier 1 loss trace is unchanged to float tolerance** — the two
assertions the whole packet's affordability rests on; the sampler stream is
independent of the model (two recipes differing only in an objective weight draw
identical `row_id` sequences); an empty-by-construction quota and an
`ExternalBatches` stage holding a batch-coupled objective each raise with an
actionable message; `QuotaSampler`'s derived `labelled_unlabelled_ratio` cannot
be set by a recipe.

**(c) Collection.** Delete the `loader` ledger row; revisit `fixmatch` §5.4,
`scarf` §5.6 and `tarnet` §5.5 (§9); amend `view-population-statistics`,
`early-stopping` and `repeated-cross-fitting`; add `out-of-core-data` and
`stateful-sampler`; re-measure `fixmatch` §6; demonstrate that `tarnet`'s
declared policy reproduces the P12 runner's, or invalidate its row.
*Accept:* `tests/invariants/test_deviation_debt.py` green — which it cannot be
until every one of those is done. That is the mechanism, not a checklist.

**(d) Optional, recommended: population-reading transforms.**
`ViewTransform.apply` takes the `TrainingPopulation`; `FeatureCorruption` draws
from the training marginal; `view-population-statistics` is discharged and
`scarf` §5.2 withdrawn.
*Accept:* the corrupted value is a value the column takes **in the training
population**, including one held by a single row, which is the tail the current
batch-local draw provably cannot reach.

**The argument for folding (d) in** is `DESIGN.md` §11.3's, and it is the
argument this repository has already been burned by ignoring once: *"The moment
the second consumer is built is the moment the first consumer's debt is cheapest
to repay — the agent holding the context has, that hour, built the thing that
repays it — and it is the last moment anyone is guaranteed to look."* `scarf`
§5.2 is the open question its own card asks the reviewer to settle, and after (a)
the reason for deferring it has evaporated.

**The argument against** is scope: (d) changes a protocol every transform
implements, and (a)–(c) already move seven digests. If it is deferred, the
ledger row must be rewritten in the same PR to say what is actually blocking it
now, because "there is no training-population object" will no longer be true.
