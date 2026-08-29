# xty2 design specification

**Status:** current

This is the normative architecture. Read only the section owned by the change;
use `docs/README.md` for routing.

## 0. Problem statement

xty2 separates five concerns that XTYLearner placed inside monolithic model
classes:

1. represented quantities;
2. their parameterisations;
3. the objectives that train them;
4. data views and eligible rows;
5. training order.

The unit of reuse is a component, objective, view, or executor. A named method
is a declarative recipe assembled from them. This makes a result attributable
to a particular data policy, parameterisation, loss, view, or schedule.

### Scope of v1

| In scope | Deferred until a reviewed card needs it |
|---|---|
| small categorical treatment | continuous or dose-response treatment |
| exact treatment marginalisation | quadrature or sampled marginalisation |
| ordered stages | a general stage DAG or parallel branches |
| Python recipe functions | a config-first surface or plugins |
| in-memory tabular data | streaming, images, sequences, or text |
| single-process training | distributed execution |

Section 11 records the evidence required to widen this scope.

## 1. Core data types

### 1.1 `XTYBatch`

| Field | Contract |
|---|---|
| `x` | float tensor `[B, D]` |
| `t` | integer tensor `[B]`, values in `[0, K)` |
| `y` | tensor `[B, *Dy]` |
| `t_observed`, `y_observed` | boolean masks `[B]` |
| `row_id` | unique, stable integer identity `[B]` |
| `fold_id` | optional fold identity `[B]` |
| `weight` | optional sample weight `[B]` |

Rules:

- Missingness uses masks, never sentinel values. `t` is arbitrary but valid
  where `t_observed` is false and must not be read there.
- Transforms are functional. They return a new batch and never mutate a tensor
  reachable from the input.
- A batch cannot repeat a `row_id`. Artifacts and provenance are keyed by row
  identity; repeated sampling remains open ledger item `batch-row-repetition`.
- v1 uses one float feature matrix. Mixed types are encoded upstream and remain
  described by the schema.

### 1.2 `Schema` and `FeatureSpec`

Each feature declares `name`, `kind`, optional bounds and perturbation scale,
whether a view may mutate it, and any `derived_from` dependencies. `Schema`
also owns treatment cardinality and outcome shape.

A view that changes an input to a derived column must declare a recompute rule
for that column. The compiler rejects stale derived features, immutable-column
changes, and out-of-bounds transforms.

### 1.3 Row populations

The closed row vocabulary is `all`, `t_observed`, `t_missing`, and
`y_observed`. A stage and objective each declare a population; the effective
rows are their intersection (§7.0).

Objectives receive full-batch state plus an explicit `RowIndex`. They gather
both data and predictions with that index. A runtime empty set returns an
unweighted `LossTerm(value=0, n=0)`; the mixer excludes it and logs coverage. A
pairing empty by construction is a compile error.

## 2. Ports: semantic quantities as the common currency

| Port | Value contract |
|---|---|
| `X_RAW` | tensor `[B, D]` |
| `Y_RAW` | tensor `[B, *Dy]` |
| `X_REPR` | representation `[B, H]` |
| `X_PROJ` | projection `[B, H]` |
| `T_GIVEN_X` | treatment distribution with probabilities `[B, K]` |
| `T_GIVEN_XY` | treatment distribution with probabilities `[B, K]` |
| `Y_GIVEN_XT` | outcome distribution satisfying §3.1 |
| `JOINT_ENERGY` | one energy per candidate treatment `[B, K]` |
| `RECONSTRUCTION` | feature reconstruction `[B, D]` |

Ports are load-bearing vocabulary. Adding one requires a reviewed card that
cannot express a source mechanic without it, plus a named second consumer used
to check the shape (§11.2).

### 2.1 Realisations: the same port under different conditions

A `Realisation` is `(view, params, draw)`:

- `view` selects a declared `ViewSpec`, defaulting to `identity`;
- `params` selects `student` or `teacher`;
- `draw` selects an independent sample of that view.

`ViewSpec.draws` bounds the available draws. State is keyed by realisation, and
the compiler plans the minimum forward passes needed by objective requirements.
Draw zero preserves the original view seed and plan rendering.

### 2.2 Raw inputs are ports too

A virtual source node supplies `X_RAW` and `Y_RAW`. Components receive only a
`PortView`, never an `XTYBatch`, so `requires` is the complete model dependency
declaration. This gives the compiler full lineage and makes outcome dependence
derivable: `used_y` is true exactly when `Y_RAW` is in the producing subgraph.

There is no `T_RAW` port. Components do not consume an unmasked observed
treatment; outcome distributions receive observed or candidate treatment values
through their methods. A future component that genuinely needs observed
treatment requires a mask-safe type and corresponding invariants, not a bare
tensor port.

Objectives do receive the batch because losses are functions of data and
predictions. The raw-port restriction applies to the model graph used for
lineage and leakage checks.

## 3. Components: parameterisation only

A `Component` is an `nn.Module` with a unique identifier-like `name`, declared
`requires` and `provides`, and `forward(PortView) -> dict[Port, value]`.

The graph enforces that a component:

- reads and writes only declared ports;
- computes no loss and performs no optimisation;
- exposes parameters, buffers, and state through the ordinary module API;
- uses no reserved component names.

### 3.1 Distribution protocols

An `OutcomeDistribution` implements `log_prob(y, t)`, `mean(t)`, and
`sample(t, n)`. `y` is always passed unexpanded. The rank of `t` selects
observed or candidate evaluation.

| Method | `t` | Result |
|---|---|---|
| `log_prob(y, t)` | `[B]` | `[B]` |
| `log_prob(y, t)` | `[B, K]` | `[B, K]` |
| `mean(t)` | `[B]` | `[B, *Dy]` |
| `mean(t)` | `[B, K]` | `[B, K, *Dy]` |
| `sample(t, n)` | `[B]` | `[n, B, *Dy]` |
| `sample(t, n)` | `[B, K]` | `[n, B, K, *Dy]` |

The candidate axis follows the batch axis; the sample axis leads. Tier 0 checks
column agreement with independent calls, shapes, and seeded sampling with
`B != K`.

A `TreatmentDistribution` exposes `probs: [B, K]` and `log_prob(t)` for `[B]`
or `[B, K]`. Rows normalise to one and `log_prob` agrees with `log(probs)`.

These protocols let the same marginal objective consume TARNet and conditional
flow heads. Categorical treatment is context to a flow, never part of its event
dimension.

## 4. Objectives: losses as independent objects

An objective declares `name`, required `(Port, Realisation)` pairs, the subset
it detaches, eligible rows, whether it is batch-coupled, and
`compute(state, batch, rows, context) -> LossTerm`.

Rules:

- `LossTerm.value` is scalar, unweighted, and the mean over eligible rows;
  `LossTerm.n` is their count. The mixer owns weighting and reduction.
- An objective does not call `backward`, mutate model state, or touch
  parameters.
- `detaches` makes stop-gradient paths visible to compilation and the plan. An
  objective that detaches everything it requires is rejected.
- Stable arithmetic choices not represented by ports, rows, schedules, or card
  keys are returned by `plan_details()` and enter the plan digest.
- `batch_coupled=True` means a term depends on other rows in the batch and
  cannot run behind `ExternalBatches`.

An objective may opt into stage-local state with
`initial_state(TrainingPopulation | None)`. The executor creates one state per
stage execution and returns it through `TrainContext`. The recipe declaration
remains immutable, paired arms cannot share history, and the state is not a
checkpointed artifact. Stateful objectives are currently incompatible with
`cross_fit`, where reset semantics are unspecified.

Two objectives may share one published mechanic by naming the state-owning
sibling and reading it through `context.objective_state(owner, kind)`. The
shared update must be idempotent within an optimiser step so declaration order
cannot change the result. FreeMatch's adaptive-threshold state records its last
observed step, and Tier 0 checks both objective orders.

### 4.1 The objective that motivates the whole design

For a missing treatment and small `K`, xty2 evaluates

$$
-\log \sum_{k=0}^{K-1}
p_\theta(t=k\mid x)\,p_\phi(y\mid x,t=k)
$$

as `-logsumexp(log_pt + log_py)` over candidate treatments. The required
`gradients.marginal_nll_grad_path` card key states whether gradients reach the
propensity head, outcome head, or both.

### 4.2 v1 objective set

Current objectives are exported from `xty2/objectives/__init__.py`:

| Family | Objects |
|---|---|
| supervised likelihood | `ObservedOutcomeNLL`, `ObservedTreatmentNLL` |
| missing-treatment likelihood | `MissingTreatmentMarginalNLL` |
| consistency | `ConsistencyLoss`, `CosineFeatureConsistency` |
| pseudo-labels | `PseudoLabelTreatmentNLL`, `CurriculumPseudoLabelTreatmentNLL`, `SelfAdaptiveThresholdTreatmentNLL`, `SelfAdaptiveFairness` |
| contrastive | `InfoNCEContrastive` |

FlexMatch's curriculum objective and FreeMatch's SAT/SAF pair are stateful.
FreeMatch is the first pair to share one state. Its threshold is derived from
the current batch and applied to that same batch, so both objectives are
`batch_coupled=True`; FlexMatch's historical curriculum is not. Add new
objectives with their first reviewed consumer rather than expanding a
speculative family.

## 5. Views: augmentation separated from loss

A `ViewSpec` names functional transforms, preserved fields, available draws,
and recompute rules. Each transform is deterministic for its RNG key, reports
its affected columns, respects schema mutability and bounds, and returns a new
batch. A transform may read the training population only when its contract
requires it, as SCARF corruption does.

Objectives refer to realisations, not transform implementations. A consistency
term therefore declares the two views/parameter sets and its detached side. The
executor computes and caches each required draw once per step.

## 6. Loss mixer

`LossMixer` evaluates each `Weighted(objective, weight, reduction)`, records the
raw term, applies the scheduled weight and reduction, and sums active terms.
Schedules are pure functions of global step and serve both objective weights
and learning-rate multipliers. Current types are `Constant`, `Ramp`,
`SigmoidRamp`, `Step`, `ExponentialDecay`, and `CosineDecay`.

### 6.1 Reduction

Objectives return a mean over their eligible rows. The mixer maps that value to
the paper's convention:

| Mode | Contribution |
|---|---|
| `mean` | `value` |
| `sum` | `value * n` |
| `population` | `value * n / B` |

Weight and reduction have `REQUIRED` defaults because they are paper-governed.
An inactive `n=0` term contributes nothing.

### 6.2 Mandatory logging

Every step logs each objective's raw value, scheduled weight, weighted value,
and eligible count. Pseudo-label objectives also log acceptance coverage.
Optional diagnostics compute per-objective gradient norms and pairwise cosines
on the stage's trainable parameters. They are disabled by default and in CI.

## 7. Program: sequencing as data

A `Program` is an ordered tuple of uniquely named `Stage` declarations. A stage
declares objectives, trainable component names, rows, optional checkpoint
inheritance, optional teacher, optional action and artifact inputs, an explicit
executor, sampler, optimiser, and optimiser-step count.

Executors are explicit:

- `gradient` runs a compiled objective mix and may emit pseudo-labels;
- `array_fit` calls one functional estimator on a finite row-keyed table;
- `cross_fit` resets and fits a gradient stage once per actual fold, predicting
  only its held-out fold.

Pseudo-label actions emit immutable side tables. Consumption joins by `row_id`
into a fresh batch and never rewrites source treatment or missingness.

### 7.0 How `Stage.rows` and `Objective.rows` compose

Effective rows are `Stage.rows ∩ Objective.rows`. Structural emptiness is a
compile error; data-dependent emptiness is a logged runtime `n=0`.

Each gradient stage starts from recipe-initial parameters, then overlays only
the components contained in its named earlier checkpoint. There is no implicit
inheritance from whichever stage happened to run before it. References cannot
point forward or to self.

`steps` means optimiser steps, never epochs. `optimiser`, `steps`, and `sampler`
are required when applicable so a recipe cannot inherit method-governed policy.

### Teacher parameters

A `TeacherSpec` declares decay, buffer treatment, train/eval mode, and
`requires_grad=False`. The executor creates a complete graph copy before the
stage, evaluates it under `no_grad`, updates the student, then applies EMA to
teacher parameters and configured buffers. Integral buffers are copied rather
than averaged.

Components outside the student's trainable set have gradients disabled and run
in evaluation mode. Flags and modes are restored after the stage.

### 7.1 Artifacts and provenance

Checkpoints, teachers, pseudo-label tables, fold assignments, plans, and logs
are immutable. A run directory belongs to one compiled plan digest. No stage
mutates its input dataset.

A checkpoint records recipe, stage, fold, trained row ids, component state,
steps, seed, and plan digest. A pseudo-label artifact records row ids, labels,
the producing checkpoint for each row, and the source checkpoints.

Provenance is derived by executor-owned factories:

- `used_y` comes from `Y_RAW` reachability in the compiled graph;
- `prediction_mode` comes from actual checkpoint/prediction row sets;
- out-of-fold status requires, for every prediction, that its row was absent
  from the producing checkpoint's training rows.

Callers cannot set these as trusted labels. The consuming stage reruns the
fold-disjointness check when loading an artifact.

### 7.2 The causal guardrail, as a compile-time rule

Keep `p(t|x)`, outcome-dependent `q(t|x,y)`, and `p(y|x,t)` distinct. A staged
procedure that creates treatment labels from `q(t|x,y)` and fits an outcome
model on the same rows is circular unless predictions are held out.

At compile time, reject a later outcome-fitting consumer when all are true:

1. the producing subgraph reaches `Y_RAW`;
2. labels will be in-sample rather than produced by `cross_fit` or another
   declared held-out source;
3. the consumer's effective rows intersect those labels.

Only a recipe with `purpose="predictive"` may opt out, and the consuming stage
must set `allow_leakage=True`. Causal recipes cannot opt out.

At runtime, artifact loading checks the actual fold assignment. Compilation
rejects programs wrong by construction; loading rejects executions wrong in
fact.

### 7.3 Data: the policy is declared, the rows are supplied

The recipe owns policy; the caller owns rows:

| Object | Owns |
|---|---|
| `DataSpec` | split protocol, preprocessing, missingness |
| `Dataset` | schema, source batch, named assignments |
| `TrainingPopulation` | transformed training rows, fitted statistics, and the row ids used to fit them |

The executor applies compiled policy to the supplied dataset. Preprocessing
provenance is therefore checkable: statistics carry `fitted_on_row_ids`, and a
wrong split fails rather than being described in prose.

Each sampled gradient stage declares one of:

- `UniformSampler(batch_size)` for a fresh seeded permutation per step;
- `QuotaSampler(quotas)` for fixed population counts and optional
  stratification;
- `ExternalBatches()` when the caller intentionally owns batching.

Quota-derived batch size and labelled/unlabelled ratio enter the plan. A
batch-coupled objective cannot use `ExternalBatches`; InfoNCE was the first
consumer of that guardrail, while FreeMatch SAT/SAF exercise it because their
threshold is computed from the current batch. Array and cross-fit stages consume
one finite table and do not accept samplers. Samplers do not read model state;
stateful sampling remains deferred in §11.4.

## 8. Compiler

`compile(recipe)`:

1. validates recipe, graph, program, data, and card-key declarations;
2. checks every required port and realisation is produced;
3. orders components and plans the minimum forward passes;
4. rejects unknown or dead trainables, including paths cut by `detaches`;
5. validates views and data policy against the schema;
6. resolves row intersections and rejects structural emptiness;
7. applies the static leakage guardrail;
8. validates executor, sampler, teacher, and artifact relationships;
9. emits a stable, printable execution plan and digest.

The plan lists components, lineage, views, stages, objectives, rows, schedules,
optimisation, trainables, and resolved hyperparameters. It is the review and
provenance surface.

## 9. Registries and recipes

Despite the historical section title, current xty2 does not use a runtime model
registry. Components and objectives are ordinary exported classes; recipes are
ordinary exported Python functions. A recipe contains declarations and explicit
values only. Conditionals belong in reusable objects whose contracts can be
tested.

Migration is lazy. A method enters with a reviewed card and its smallest
faithful assembly.

### 9.1 Canonical hyperparameter binding

The card-key vocabulary is the closed YAML structure in `FIDELITY.md` §2.
Paper-governed object fields map to canonical keys through `CARD_KEYS` and carry
the unusable `REQUIRED` sentinel until a recipe supplies them.

Compilation emits `plan.hyperparameters`:

- per-objective weights, schedules, reductions, rows, and stop-gradients are
  derived and keyed by `<stage>.<objective>`;
- architecture values are grouped by component name;
- program values are grouped by stage in multi-stage recipes and remain scalar
  for a single-stage recipe;
- conflicting values in one scope are errors.

Tier 0 compares non-`n/a` card keys with this mapping. That proves explicit
agreement with the reviewed card, not correctness against the paper.

## 10. Package layout

| Path | Responsibility |
|---|---|
| `xty2/core/` | batch/schema contracts, ports, graph, recipe declarations, data, compiler |
| `xty2/components/` | encoders and distribution parameterisations |
| `xty2/views/` | schema-aware transforms |
| `xty2/objectives/` | independent loss terms |
| `xty2/training/` | mixing, executors, teachers, artifacts, loading |
| `xty2/recipes/` | declarative method assemblies |
| `xty2/evaluation/` | metrics, reports, and benchmarks |
| `xty2/estimators/` | array-style estimators |

Declarations used by the compiler live in `core`; execution imports `core`,
not the reverse.

### The five ported recipes, and what each one proves

The heading is retained for stable references to the original Gate 2. The
initial five proved the core architecture; later cards extended it under §11.2.

| Recipe | Primary contract exercised |
|---|---|
| `tarnet` | outcome/propensity heads and exact marginalisation |
| `cnflow` | distribution-protocol substitution |
| `mean_teacher` | views, teacher realisations, scheduled consistency |
| `cycle_dual` | staged outcome-dependent labels and leakage checks |
| `ssdml` | array and cross-fit execution |
| `fixmatch` | quota batches and confidence-gated pseudo-labels |
| `scarf` | population-aware corruption and contrastive pretraining |
| `doublematch` | representation consistency beside FixMatch |
| `flexmatch` | stage-local objective state and adaptive thresholds |
| `freematch` | shared state across batch-coupled adaptive objectives |

## 11. Overdesign guardrails

Framework growth follows evidence from cards, not a list of interesting
methods.

### 11.1 The two failure modes, and why counting consumers only sees one

| Failure | Symptom |
|---|---|
| over-building | new vocabulary or execution machinery without a source mechanic that needs it |
| under-building | a card ships without a source mechanic because xty2 cannot express it |

Consumer count controls convenience abstractions. It does not justify omitting
a fidelity-bearing mechanic.

### 11.2 The rule: build for one, design against two

Ask two questions:

1. Would absence force a deviation from a source mechanic named by a reviewed
   card? If yes, the change is fidelity-bearing.
2. Is the shape reversible, or is it vocabulary every future recipe must use
   such as a port, executor contract, row population, or artifact kind?

| | Reversible | Load-bearing vocabulary |
|---|---|---|
| **Fidelity-bearing** | build for the reviewed card and record it in §5.1 | build for the card, but check the shape against a named second consumer before coding |
| **Convenience** | wait for a second real consumer | wait for a second reviewed consumer and an explicit design decision |

The second consumer constrains shape; it need not be implemented. Duplicate
local code is acceptable while evidence is still forming.

### 11.3 Fidelity debt, and who collects it

A card §5 `framework-limitation` is a creditor. It cites one ledger key, and
the ledger cites the exact card row. Tier 0 reconciles both directions.

When a capability is built, delete or amend its ledger row. Every remaining
creditor then fails CI and must be revisited in the same PR. Withdraw the
deviation if the source mechanic is now implemented; otherwise restate it as a
judgement explaining why the choice survives the capability.

### 11.4 The ledger

| Key | Not building | Would build when | Who is paying |
|---|---|---|---|
| `stage-dag` | general stage DAG | a reviewed method needs parallel branches | — |
| `continuous-t` | continuous or dose-response treatment | a reviewed card requires it | — |
| `grad-surgery` | GradNorm or PCGrad | measured gradient conflict warrants intervention | — |
| `config-surface` | config-first authoring | Python recipes impede real sweeps | — |
| `plugins` | plugin/entry-point system | an external consumer exists | — |
| `distributed` | distributed training | one required recipe no longer fits one process | — |
| `out-of-core-data` | streaming or larger-than-memory datasets | a card's dataset does not fit in memory | — |
| `stateful-sampler` | sampling driven by model or training state | a card states curriculum sampling, hard-negative mining, or acquisition | — |
| `batch-row-repetition` | repeated `row_id` values inside one batch | a faithful protocol requires a quota larger than its source population and the shape is checked against PAWS-style support sampling — `paws` §5.1 now supplies that check | `fixmatch` §5.12; `doublematch` §5.7; `flexmatch` §5.8; `freematch` §5.9; `paws` §5.3 |
| `lr-schedules` | schedule families beyond the implemented types | a reviewed card names one | — |
| `augmentation-vocabulary` | a shared augmentation vocabulary and adaptive controller | multiple useful operations and magnitudes exist | `fixmatch` §5.10; `doublematch` §5.6; `flexmatch` §5.7; `freematch` §5.8 |
| `staged-gate` | confidence gating on staged pseudo-label writeback | a reviewed staged method names the gate | — |
| `repeated-cross-fitting` | several fold assignments and aggregation | a reviewed estimator requires repeated splitting and a second consumer checks the artifact shape | `ssdml` §5.6 |
| `early-stopping` | validation-metric stage termination | a reviewed protocol cannot be stated as a fixed step budget | — |
| `model-families` | unrequested XTYLearner families | one is needed for a result | — |

### 11.5 What this changes about the `fixmatch` exceptions

A capability is specific to the path that implements it. Objective-path
confidence gating does not imply staged-action gating; an implemented sampler
does not imply repeated rows are valid. Cards name the exact missing path rather
than claiming the framework generally lacks a concept.

### 11.6 What adopting this cost

New fidelity-bearing vocabulary changes plans, digests, and possibly benchmark
identity. Card amendment, invariant coverage, and evidence invalidation are part
of that change. This cost is intentional: it prevents an architectural edit
from silently changing what a recorded result means.
