# xty2 — Design Specification

**Status:** draft v1, for review
**Supersedes:** the model-class registry in `mattsq/XTYLearner`

## 0. Problem statement

XTYLearner registers ~40 monolithic model classes. Each class collapses five
independent questions into one object, so nothing composes and nothing is
independently testable:

1. What probabilistic quantities are represented?
2. How is each quantity parameterised?
3. Which losses train them?
4. Which data views and row subsets does each loss use?
5. In what order are those losses and models trained?

xty2 separates these five axes. The registry moves *down* a level: components,
objectives and views are the units of reuse; named methods (`tarnet`, `cnflow`,
`cycle_dual`, `mean_teacher`) become **recipes** that compose them.

A second, equally important goal: a bad result must be **attributable**. When a
recipe underperforms, the architecture should let you say which of {data view,
objective, parameterisation, schedule} is responsible. In XTYLearner and in
looptab, it could not.

### Scope of v1

| In | Out (v1) |
|---|---|
| Categorical treatment, K small (K≈4) | Continuous / dose-response T |
| Exact marginalisation over t | Quadrature or sampled marginalisation |
| Linear list of training stages | General stage DAG, parallel branches |
| Python-first recipe construction | Config-first (Hydra/YAML) surface |
| Tabular features with a declared schema | Images, sequences, text |
| 5 ported recipes (§10) | The other ~35 XTYLearner families |
| Single-process training | Distributed / multi-GPU |

The YAGNI ledger in §11 records what we are deliberately not building and what
evidence would justify building it.

---

## 1. Core data types

### 1.1 `XTYBatch`

```python
@dataclass(frozen=True)
class XTYBatch:
    x: Tensor                 # [B, D] float; mixed types encoded upstream
    t: Tensor                 # [B]  long, values in [0, K); arbitrary where unobserved
    y: Tensor                 # [B]  or [B, Dy]
    t_observed: Tensor        # [B]  bool
    y_observed: Tensor        # [B]  bool  (v1: assumed all-True, still carried)
    row_id: Tensor            # [B]  long, stable index into the source dataset
    fold_id: Tensor | None    # [B]  long, for cross-fitting and leakage checks
    weight: Tensor | None     # [B]  float, sample weights
```

**Rule: no sentinel values.** Missingness is carried by an explicit boolean
mask, never by `t == -1`. Sentinels leak into `nn.Embedding`, into `one_hot`,
and into loss reductions, and every such leak is silent. Where `t_observed` is
false, `t` holds an arbitrary valid class index and reading it is a bug.

**Rule: functional transforms.** `frozen=True` only prevents rebinding the
dataclass fields — it does **not** make the tensor leaves read-only. An in-place
write inside a transform would corrupt the source batch while still satisfying
the type. Transforms are therefore required to be functional: they return a new
`XTYBatch` and perform no in-place write on any tensor reachable from the input.
This is enforced rather than trusted — a Tier 0 test clones every input batch,
runs each registered transform, and asserts the original is bit-identical
afterwards.

**Rule: `x` is a single float tensor in v1.** Heterogeneous / mixed-type
features are encoded upstream into `[B, D]`, which is what gives
`RECONSTRUCTION` (§2) a well-defined shape and lets
`MaskedFeatureReconstruction` have a portable output. A structured
reconstruction contract is deferred until a recipe needs one (§11).

### 1.2 `Schema` and `FeatureSpec`

Every feature carries metadata, because generic augmentation over tabular
physical data otherwise produces impossible rows:

```python
@dataclass(frozen=True)
class FeatureSpec:
    name: str
    kind: Literal["continuous", "categorical", "ordinal"]
    bounds: tuple[float, float] | None = None
    perturbation_scale: float | None = None   # in natural units, not sigmas
    mutable: bool = True                      # may an augmentation touch it?
    derived_from: tuple[str, ...] = ()        # columns this is computed from
```

`Schema` holds the feature list, the treatment cardinality `K`, and the outcome
spec. It is resolved once and is available to the compiler, so transforms and
objectives can be validated statically rather than failing at step 4,000.

**Derived-column rule.** If a transform perturbs a column that appears in some
other column's `derived_from`, the compiler **rejects the view** unless a
recompute rule is registered for the dependent column. Silently jittering MASS
while leaving a mass-derived column stale is the tabular equivalent of a
mislabelled image.

### 1.3 Row populations

Every objective declares which rows it is entitled to:

```python
Rows = Literal["all", "t_observed", "t_missing", "y_observed"]
```

The engine computes each index set once per batch and passes it to the
objective as an explicit `RowIndex` (§4). It does **not** hand over a pre-sliced
batch: `state` holds full-batch `[B, ...]` quantities, some of them distribution
objects that are not generally sliceable, so pre-slicing the batch alone would
silently misalign predictions against rows. Batch and state share one batch
axis and one index.

**Zero-eligible-row rule.** An objective with no eligible rows in a batch
returns `LossTerm(value=0.0, n=0)` — never `NaN`, never a mean over an empty
tensor. The mixer excludes `n == 0` terms from the total and logs the coverage.
This is a mandatory invariant test (`FIDELITY.md` Tier 0), because empty-set
`NaN`s are the single most common cause of a silently dead objective.

---

## 2. Ports: semantic quantities as the common currency

Components exchange **named statistical quantities**, not architectures:

```python
class Port(str, Enum):
    # Raw inputs, supplied by the virtual source node (§2.2)
    X_RAW           = "x"
    Y_RAW           = "y"
    # Computed quantities
    X_REPR          = "x_repr"
    Y_GIVEN_XT      = "p(y|x,t)"
    T_GIVEN_X       = "p(t|x)"
    T_GIVEN_XY      = "q(t|x,y)"
    JOINT_ENERGY    = "energy(x,t,y)"
    RECONSTRUCTION  = "reconstruction"
```

`XY_REPR` appears in `SEED.md` but has no consumer among the five v1 recipes, so
by the two-consumer rule (§11) it is not declared. A port with no `PortSpec` is
a port the compiler cannot check.

Each port has a `PortSpec` fixing its **type and shape contract**, checked at
compile time and asserted in tests:

| Port | Type | Contract |
|---|---|---|
| `X_RAW` | `Tensor` | `[B, D]`, as `batch.x` |
| `Y_RAW` | `Tensor` | `[B, *Dy]`, as `batch.y` |
| `X_REPR` | `Tensor` | `[B, H]` |
| `T_GIVEN_X` | `TreatmentDistribution` | `probs: [B, K]`, rows sum to 1 |
| `T_GIVEN_XY` | `TreatmentDistribution` | `probs: [B, K]` |
| `Y_GIVEN_XT` | `OutcomeDistribution` | `log_prob(y, t)` broadcasts over t (§3.1) |
| `JOINT_ENERGY` | `Tensor` | `[B, K]`, one energy per treatment value |
| `RECONSTRUCTION` | `Tensor` | `[B, D]`, matching `X_RAW` |

Ports are the framework's vocabulary. **Adding a port is a design decision, not
an implementation detail** — it requires a second real consumer (§11).

### 2.2 Raw inputs are ports too

A component that reads `batch.x` directly cannot be checked: `requires` would
name no dependency, the compiler could not order it, and the "reads only what it
declares" rule would be an unenforceable claim. So the batch is not handed to
components at all. A **virtual source node** provides `X_RAW` and `Y_RAW` from
the batch, and every component declares them like any other dependency:

- `mlp_encoder`: `{X_RAW}` → `{X_REPR}`
- `xy_posterior`: `{X_RAW, Y_RAW}` → `{T_GIVEN_XY}`

Three things fall out of this, which is why it is worth the three extra enum
members:

1. **`requires` is the single dependency declaration.** One mechanism, one
   compiler check, no second "declared batch fields" concept.
2. **The execution plan shows full data lineage**, including which components
   touch raw `y`.
3. **`used_y` becomes derivable rather than declared.** Whether a pseudo-label
   depends on the outcome is now a property of the graph — is `Y_RAW` in the
   transitive closure of the producing subgraph — instead of a boolean somebody
   remembered to set correctly (§7.2).

**There is no `T_RAW` port**, and this is deliberate. No v1 component consumes
the observed treatment: outcome heads receive candidate `t` through
`log_prob(y, t)` (§3.1) rather than from the batch, the energy head emits one
value per treatment, and propensity and posterior heads take representations and
`Y_RAW`. A raw treatment port would therefore be a port with no consumer, which
the two-consumer rule (§11) already forbids.

It would also be the one port that cannot be typed honestly as a tensor. `t` is
only valid where `t_observed` (§1.1), and a bare `Tensor` port carries no way to
say so — a component could read arbitrary class indices on missing rows and
nothing in the API or the compiler could stop it. If a future component genuinely
needs the observed treatment, it arrives as a **masked type**, not a bare
tensor:

```python
@dataclass(frozen=True)
class MaskedTreatment:
    def where_observed(self, fill: int) -> Tensor: ...
    def observed_rows(self) -> Tensor: ...        # [n] indices
    # no attribute returns raw values without the mask applied
```

so that the mask is impossible to drop rather than merely required. Adding it
means adding the port, its `PortSpec` and its invariants together.

Objectives, by contrast, still receive the batch directly (§4), because a loss
is by definition a function of data and predictions. The static guarantee is
wanted on the *model graph*, which is what the leakage rule reasons about.

### 2.1 Realisations: the same port under different conditions

Consistency losses need `p(t|x)` under two augmentations. Mean-teacher losses
need it under two *parameter sets*. Both are the same idea: the graph is
evaluated more than once, under different conditions.

```python
@dataclass(frozen=True)
class Realisation:
    view: str = "identity"                        # a ViewSpec name
    params: Literal["student", "teacher"] = "student"

DEFAULT = Realisation()
```

State is therefore keyed by realisation. It is a small wrapper, not a bare
dict, because the default-realisation lookup is used constantly:

```python
class State:
    def __getitem__(self, r: Realisation) -> Mapping[Port, Any]: ...
    @property
    def default(self) -> Mapping[Port, Any]:   # == self[DEFAULT]
        ...
```

The compiler collects the set of realisations the objectives demand and plans
exactly that many forward passes — no more.

This is the one piece of machinery in v1 that exists for a model we have not
written yet (Mean Teacher, P9). It is included because the alternative — letting
each consistency loss re-invoke the graph itself — is what produced
uncontrolled, untestable forward passes in the previous codebase.

---

## 3. Components: parameterisation only

```python
class Component(nn.Module):          # a base class, not a Protocol — see below
    name: str
    requires: frozenset[Port]
    provides: frozenset[Port]

    def forward(self, ports: PortView) -> dict[Port, Any]: ...
```

Rules, all enforced rather than documented:

- A component reads **only** the ports in `requires`, including raw inputs
  (§2.2). It is handed a `PortView` that raises on undeclared reads, and it is
  handed nothing else — no `XTYBatch` argument exists, so there is no channel
  through which an undeclared dependency can be taken.
- A component writes **only** the ports in `provides`.
- **A component never computes a loss.** No `loss()` method exists on the
  Component base class. This is the single largest departure from XTYLearner.
- **A component is an `nn.Module`.** This is a requirement, not an
  implementation detail: stages must build optimiser param groups, freeze and
  unfreeze by component name, and maintain EMA copies, and none of that is
  expressible against a structural protocol with no parameter interface.
  `named_parameters()`, `state_dict()` and `load_state_dict()` are the contract
  the training layer depends on.
- Component names are Python identifiers. `source` is reserved for the virtual
  source node and `_components` is reserved for `ComponentGraph`'s internal
  module registry, so qualified parameter names have one unambiguous owner.

Examples: `mlp_encoder` (`X_RAW` → `X_REPR`), `tarnet_head`
(`X_REPR` → `Y_GIVEN_XT`), `categorical_propensity` (`X_REPR` → `T_GIVEN_X`),
`conditional_flow` (`X_REPR`, with categorical t as context → `Y_GIVEN_XT`),
`xy_posterior` (`X_RAW`, `Y_RAW` → `T_GIVEN_XY`).

### 3.1 Distribution protocols

```python
class OutcomeDistribution(Protocol):
    def log_prob(self, y: Tensor, t: Tensor) -> Tensor: ...
    def mean(self, t: Tensor) -> Tensor: ...
    def sample(self, t: Tensor, n: int) -> Tensor: ...

class TreatmentDistribution(Protocol):
    @property
    def probs(self) -> Tensor: ...      # [B, K]
    def log_prob(self, t: Tensor) -> Tensor: ...
```

**The candidate-treatment contract is the load-bearing detail, and it is the
single most error-prone thing in this document** — three separate mistakes were
made stating it in prose before it was written down as a signature. It is
therefore specified as a shape table, with an executable reference
implementation shipped in P1 that every outcome head is tested against.

`Y_GIVEN_XT.log_prob(y, t)`:

| `y` | `t` | returns | meaning |
|---|---|---|---|
| `[B, *Dy]` | `[B]` | `[B]` | `out[i] = log p(y_i \| x_i, t_i)` |
| `[B, *Dy]` | `[B, K]` | `[B, K]` | `out[i, k] = log p(y_i \| x_i, t[i, k])` |

`mean` and `sample` take the same `t`, and need the same contract — the
estimator and evaluation layers compute CATE from treatment-wise means, so a
head that satisfies only the `log_prob` test can still return an ambiguously
broadcast mean and corrupt every causal metric downstream:

| method | `t` | returns |
|---|---|---|
| `mean(t)` | `[B]` | `[B, *Dy]` |
| `mean(t)` | `[B, K]` | `[B, K, *Dy]` |
| `sample(t, n)` | `[B]` | `[n, B, *Dy]` |
| `sample(t, n)` | `[B, K]` | `[n, B, K, *Dy]` |

The candidate axis is inserted **immediately after the batch axis**, and the
sample axis leads. All three methods are covered by the Tier 0 conformance
test: exact column agreement for `log_prob` and `mean`, and shape plus
seed-determinism for `sample`.

Two rules make this implementable:

1. **`y` is passed unexpanded, always.** The caller never reshapes `y`; the head
   inserts the candidate axis internally (`y[:, None]` for scalar outcomes,
   `y[:, None, :]` for `[B, Dy]`). Relying on ambient tensor broadcasting does
   **not** work here: `y: [B]` against `t: [B, K]` aligns trailing dimensions
   `B` against `K` and fails for every batch where `B != K`.
2. **The rank of `t` selects the mode**, and nothing else does. `t.ndim == 1` is
   observed-treatment evaluation; `t.ndim == 2` is candidate evaluation.

Every outcome head must satisfy this, and it is a Tier 0 test.

That single contract is what lets `MissingTreatmentMarginalNLL` (§4.1) work
unchanged across a TARNet head, a Gaussian density head, a conditional flow and
an energy model. It also settles the `cnflow_model.py` problem directly: **t is
categorical context to the flow, never a dimension inside the flow.**

---

## 4. Objectives: losses as independent objects

```python
RowIndex = Tensor          # [n] long, indices into the shared batch axis

class Objective(Protocol):
    name: str
    requires: frozenset[tuple[Port, Realisation]]
    detaches: frozenset[tuple[Port, Realisation]]    # subset of `requires`
    rows: Rows

    def compute(self, state: State, batch: XTYBatch,
                rows: RowIndex, ctx: TrainContext) -> LossTerm: ...

@dataclass(frozen=True)
class LossTerm:
    value: Tensor          # scalar, UNWEIGHTED, mean over eligible rows
    n: int                 # eligible rows
    diagnostics: dict[str, float] = field(default_factory=dict)
```

Rules:

- An objective returns its **unweighted** value. Weighting is the mixer's job
  (§6). An objective that applies its own weight is a bug, because it makes the
  logged raw value incomparable across runs.
- An objective never calls `.backward()`, never mutates state, never touches
  parameters directly.
- An objective with an arithmetic choice that is not already visible through
  its ports, realisations, rows, reduction, schedule or card keys emits that
  choice through stable `plan_details()`. Those lines are part of the execution
  plan digest, so two recipes that compute different losses cannot share a
  provenance identity merely because their graph wiring is the same.
- **A stop-gradient is declared, because it is invisible in the graph.**
  `requires` says a term *reads* `p(t|x)`, and the compiler plans the forward
  pass from that — but a `.detach()` inside `compute` means no gradient ever
  reaches the component that produced it. `detaches` names that subset, so the
  dead-trainable rule (§8.4) can tell a component that is merely executed from
  one that is actually trained. Without it a stage whose sole trainable is the
  detached side compiles, trains, and makes every optimiser step a no-op — the
  dead-weight stage §8.4 exists to reject, hidden behind a detach rather than
  behind the wiring. It is a required member rather than an optional attribute
  for the reason §7.1 gives about provenance: a declaration with a fallback is
  one that can be forgotten, and forgetting this one restores the hole. Where a
  card field governs the stop-gradient — `gradients.marginal_nll_grad_path`,
  `ConsistencyLoss.stop_grad` — derive `detaches` from that field rather than
  stating it twice. An objective that detaches *everything* it requires
  contributes a constant to the total and is rejected.
- **`rows` is the eligible set, and the objective must gather by it.** `state`
  and `batch` are both full-batch and share one batch axis, so the same `rows`
  indexes both and alignment is automatic. `n == rows.numel()`. Reading
  `batch.t` at ineligible positions is a bug of the same kind as reading it
  where `t_observed` is false (§1.1); it is the objective's row declaration that
  makes such a read meaningless, not the type.
- The reduction convention is **mean over `rows`**, matching how papers state
  their losses. Papers that sum, or that average over the whole batch, are
  expressed by the mixer's reduction mode (§6.1) rather than by a different
  objective, and which one a recipe uses is a required card field.

### 4.1 The objective that motivates the whole design

With K small, missing treatments admit exact marginalisation:

$$-\log \sum_{k=1}^{K} p_\theta(t=k \mid x)\, p_\phi(y \mid x, t=k)$$

implemented in log space as
`-logsumexp_k( log_pt[:, k] + log_py[:, k] )`, with `log_pt: [B,K]` from
`T_GIVEN_X` and `log_py: [B,K]` from `Y_GIVEN_XT.log_prob(y, all_k)`.

It requires exactly two ports and no architecture. It is therefore reusable
across every outcome parameterisation without touching the training loop.

**Known missing detail, made explicit:** this term backpropagates into *both*
the propensity and the outcome head. Several papers stop-gradient one side. That
is a mandatory field in every card that uses this objective (`FIDELITY.md` §4).

### 4.2 v1 objective set

Ship only what the five ported recipes need:

`ObservedOutcomeNLL`, `ObservedTreatmentNLL`, `MissingTreatmentMarginalNLL`,
`PosteriorKLDivergence`, `MaskedFeatureReconstruction`, `ConsistencyLoss`
(parameterised by port + two realisations, covering VAT / weak-strong /
teacher-student), `EntropyMinimisation`, `SoftTreatmentNLL`,
`RepresentationBalance`, `DragonNetTargetedLoss`.

`CycleConsistency`, `OrdinalTreatmentLoss` and the open-set / diffusion families
are deferred to their first real consumer.

Past Gate 2 the same rule applies to this list itself: an objective enters when
a reviewed card needs it and not before. `PseudoLabelTreatmentNLL` — a
confidence-gated hard pseudo-label across two realisations — is the first such
addition, and arrived with `docs/recipes/fixmatch.md`. It is a *loss*, not a
stage transition: FixMatch's artificial label is a per-batch detached target,
where §7's `PseudoLabelAction` emits an immutable side table between stages.
Both exist because those are two different mechanisms, not two spellings of
one.

---

## 5. Views: augmentation separated from loss

```python
ViewSpec(
    name="strong_x",
    transforms=[FeatureMask(p=0.25),
                BoundedJitter(columns=["KTAS", "TRQ", "FF"])],
    preserves={"t", "y"},
    recompute_rules=[],  # required for any derived column made stale
)
```

- A view is a **pure function of (batch, rng_key)** — deterministic given a
  seed, so left/right views are reproducible and diffable.
- `preserves` is checked by test, not trusted: a view declaring
  `preserves={"t","y"}` that alters `t` or `y` fails Tier 0.
- Transforms consult `FeatureSpec`. `mutable=False` columns are never touched;
  `bounds` are respected; `perturbation_scale` is in natural units. Their
  returned batches are checked against `affected_columns`, so a custom
  transform cannot bypass mutability or derived-column validation by
  under-reporting its footprint.
- Views are computed once per batch per name and cached for that step.

A consistency objective names realisations, not transforms:

```python
ConsistencyLoss(port=Port.T_GIVEN_X,
                left=Realisation(view="weak_x"),
                right=Realisation(view="strong_x"),
                divergence="kl", stop_grad="left")
```

`stop_grad` is explicit and required — no default. Whether the target side is
detached is precisely the kind of one-line paper detail that gets lost.

---

## 6. Loss mixer

```python
LossMixer([
    Weighted(ObservedOutcomeNLL(),            weight=1.0),
    Weighted(ObservedTreatmentNLL(),          weight=1.0),
    Weighted(MissingTreatmentMarginalNLL(),   weight=Ramp(0.0, 0.5, steps=5_000)),
    Weighted(ConsistencyLoss(...),            weight=Ramp(0.0, 0.2, steps=10_000)),
])
```

Schedules in v1: `Constant`, `Ramp` (linear), `SigmoidRamp` (Mean Teacher's
Gaussian-shaped ramp-up), `Step`, and the staircase `ExponentialDecay` required
by TARNet's pinned reference implementation. `CosineDecay` joined them with the
`fixmatch` card, whose §2.4 states `eta cos(7 pi k / 16 K)` — the ledger
condition in §11 for a new schedule type. Schedules are pure functions of
`ctx.global_step` and are logged. The same types serve objective weights and
learning-rate multipliers.

### 6.1 Reduction

Objectives always return the **mean over eligible rows** (§4). Papers do not
always state their losses that way, so the reduction a paper prescribes is
applied by the mixer, not by the objective:

```python
Weighted(obj, weight=1.0, reduction="mean" | "sum" | "population")
```

| Mode | Contribution | Use when |
|---|---|---|
| `mean` | `value` | the paper averages over the term's own rows |
| `sum` | `value * n` | the paper sums over rows |
| `population` | `value * n / B` | the paper averages over the *whole* batch |

**Neither field has a default.** `weight` and `reduction` both bind card keys
(`FIDELITY.md` §2: `losses.weights`, `losses.reduction`), so both carry the
`REQUIRED` sentinel of §9.1 and a recipe that omits either is rejected at
construction. `mean` is the *common* choice, not an inherited one — the
paragraph below is the reason a silent one would be the worst available
outcome.

Because `LossTerm` already carries `n`, all three are recoverable from the same
objective with no change to the objective API, and the raw per-objective value
stays comparable across runs and across modes.

The distinction is not cosmetic. `sum` and `mean` differ by a factor that
*varies per batch* whenever the row population varies — which is precisely the
case for every `t_missing` objective — so choosing the wrong one produces a
model that trains, looks reasonable, and weights its semi-supervised term
differently from the paper. It is a required card field (`FIDELITY.md` §2).

The mixer is the single place that later supports alternating updates, per-
objective update frequency, loss normalisation, GradNorm and PCGrad. **None of
those are implemented in v1** — the mixer just holds the seam.

### 6.2 Mandatory logging

Per objective, per step: raw value, weight, weighted value, `n` eligible.
Per objective, periodically (default every 200 steps, configurable, off in CI):
gradient norm on shared parameters, and pairwise gradient cosine similarity
between objectives.
For pseudo-labelling recipes: coverage (fraction above threshold) and
calibration of the accepted labels.

Rationale: once several objectives are mixed, validation performance alone
cannot tell you that a term has become numerically irrelevant or that two terms
are fighting. Both happened in the previous codebase and were invisible.

---

## 7. Program: sequencing as data

Pseudo-labelling, distillation and targeted refitting are stage transitions, not
weighted losses.

```python
@dataclass(frozen=True)
class Stage:
    name: str
    objectives: tuple[Weighted, ...] = ()
    trainable: tuple[str, ...] = ()             # component names
    rows: Rows = "all"                         # stage-level row scope (§7.0)
    initialise_from: str | None = None         # a previous stage's checkpoint
    teacher: TeacherSpec | None = None          # stage-local EMA parameter set
    action: PseudoLabelAction | ArrayFitAction | None = None
    inputs: tuple[str, ...] = ()                # earlier pseudo-label artifacts
    executor: Literal["gradient", "array_fit", "cross_fit"] = "gradient"
    allow_leakage: bool = False                # predictive consumer only
    optimiser: OptimiserSpec = REQUIRED        # §9.1 sentinel, not a default
    steps: int = REQUIRED                      # optimiser steps, never epochs

@dataclass(frozen=True)
class PseudoLabelAction:
    port: Literal[Port.T_GIVEN_X, Port.T_GIVEN_XY]
    rows: Rows = "t_missing"
    realisation: Realisation = DEFAULT

class ArrayFitAction(Protocol):
    name: str
    def fit(
        self, batch: XTYBatch, rows: RowIndex, *, seed: int
    ) -> Mapping[str, Tensor]: ...

@dataclass(frozen=True)
class TeacherSpec:
    decay: float = REQUIRED
    applies_to_buffers: bool = REQUIRED
    train_mode: bool = REQUIRED
    requires_grad: Literal[False] = REQUIRED

@dataclass(frozen=True)
class Program:
    stages: tuple[Stage, ...]                  # ordered; names are unique
```

`optimiser` and `steps` are written here as `REQUIRED` rather than `None`,
amending an earlier draft that gave them optional defaults. Every field of an
`OptimiserSpec` binds a card key from the `optimisation` block of
`FIDELITY.md` §2, and so does `steps` — so a default is precisely the silent
inheritance §9.1 exists to make impossible. They are required exactly when a
stage has objectives to descend; action-only and `array_fit` stages do not
pretend to optimise. `steps` is **optimiser steps**, because "epochs on a
semi-supervised loader are ambiguous" (`FIDELITY.md` §2); a card stating epochs
converts, and writes the conversion into its §7.

`OptimiserSpec` itself lives in `core/optimisation.py` (§10) with the two value
objects it needs: `WeightDecay`, which carries the coefficient, its component
scope, *and* whether it reaches biases and norm parameters, and
`GradientClipping`, which carries the mode and the threshold. Each is one field
bound to one canonical key, as §9.1 requires — the alternative, several fields
sharing `optimisation.weight_decay`, is rejected by the binding rule. The
learning-rate schedule reuses the
`Schedule` types of §6 as a *multiplier* on `lr`, so warmup is a `Ramp` and no
schedule is `Constant(1.0)`: a string field naming a schedule nothing
implements would let a card claim one the run does not have.

### 7.0 How `Stage.rows` and `Objective.rows` compose

Both exist, so the rule must be stated: the eligible set is the
**intersection** of the stage scope and the objective's declared population.
The stage says which rows this stage may touch at all; the objective says which
of those it is entitled to.

An intersection that is empty *by construction* — a `t_observed` stage
containing a `t_missing` objective — is a **compile-time error**, not a
silently zero term. Without that rule the objective would return `n = 0` on
every batch forever, which is exactly the silently-dead-objective failure the
zero-eligible-row rule (§1.3) exists to make visible. Emptiness that arises
from the data in a particular batch remains a runtime `n = 0`, logged as
coverage.

`Program` is an **ordered list** of stages. Not a DAG. Stages run in sequence;
`initialise_from` references an earlier stage by name and cannot point forward
or to itself. Each gradient stage begins from the recipe's initial graph state,
then overlays exactly the parameters and buffers in that named immutable
checkpoint. There is no implicit "whatever the previous stage left in memory"
transition: it would make an omitted `initialise_from` change the method while
leaving the plan silent. Components absent from the named checkpoint retain
their recipe-initial values, matching the rule in §7.1 that a checkpoint carries
the trained components and only those.

This covers the full Beyer-style recipe — representation pretraining → joint
fit → generate pseudo-labels → refit → targeted causal fit — and we will not
build a scheduler until a fifth model genuinely needs one. P10 makes every
transition explicit: `inputs` names earlier pseudo-label side tables,
`initialise_from` names one earlier graph checkpoint, and `executor` names the
mechanism rather than asking the runtime to infer it from a `fit` method.

`PseudoLabelAction` is a declaration over the component graph. It names the
treatment-distribution port, realisation and row scope, and P10 emits hard
argmax labels only. The port is also the lineage root from which `used_y` is
derived. A `gradient` stage may emit labels after its fit, or an action-only
stage may predict from its named initial checkpoint. At consumption, a matching
`row_id` replaces the arbitrary treatment placeholder in a fresh batch and
marks that value available through `t_observed`; the source batch and its
missingness mask remain bit-identical. Consequently an objective whose
effective §7.0 scope still requires `t_missing` cannot train on joined labels,
which the static leakage check accounts for. A fitting gradient action captures
exactly the `steps` batches that its optimiser consumes, so a cycling training
source remains valid and is never exhausted merely to make predictions. An
action-only prediction source is finite by contract.

`array_fit` is the deliberately narrow functional seam for non-autograd
estimators: one finite row-keyed batch plus the resolved `RowIndex` and seed go
in, and a complete mapping of named tensors comes out as immutable checkpoint
state. The input batch is checked for mutation. This is explicit estimator
behaviour assembled by the recipe, not executor selection inferred from the
presence of a method.

P10's `cross_fit` path repeats a gradient stage with a `PseudoLabelAction` over
the actual non-negative `fold_id` values in a finite dataset. Every fold starts
from the same recipe-initial state plus the same named initial checkpoint,
trains on the complement, and predicts only the held-out fold. At least two
folds and unique `row_id` values are required. The fold checkpoints and their
held-out predictions are what make the §7.1 decision procedure executable;
P11 may add a second cross-fit action kind only if its real array estimator
needs one.

### Teacher parameters

A `TeacherSpec` makes `params="teacher"` realisable for that stage, under the
identity view and every declared `ViewSpec`. It is a complete, distinct copy of
the component graph made immediately before the stage's first forward pass.
All teacher parameters have `requires_grad=False`, their gradients remain
`None`, and teacher forward passes run under `torch.no_grad()`. An objective
reading a teacher target must therefore include that requirement in
`detaches`; the compiler rejects an undeclared teacher gradient path rather
than letting the runtime and printed plan disagree.

The update order is fixed: evaluate the current student and teacher, descend
the student loss, then update teacher parameters as
`teacher = decay * teacher + (1 - decay) * student`. The four choices in the
`teacher` checklist are required and plan-visible. `train_mode` independently
controls dropout and whether the teacher updates its own BatchNorm statistics.
When `applies_to_buffers=True`, floating and complex buffers receive the same
EMA update after the step and integral counters are copied exactly; when false,
the teacher buffers evolve only through a teacher forward in training mode.

Freezing is also component-wide for the student: parameters outside
`Stage.trainable` have gradients disabled and those components run in evaluation
mode, so stateful buffers cannot drift while only a downstream component is
being fitted. All parameter flags and module modes are restored after the stage.

The explicit executor field added in P10 also removes an inference we should
never have had: XTYLearner decided that "a model with `fit()` but no `loss()`
uses ArrayTrainer". The recipe instead says `executor="array_fit"`.

### 7.1 Artifacts and provenance

Stages emit **immutable artifacts** into a run directory: checkpoints, EMA
teachers, pseudo-label tables, fold assignments. **No stage mutates the source
dataset.** Pseudo-labels are a side table keyed by `row_id`, joined at load time
by the consuming stage.

Every generated artifact carries provenance. The important property is that it
is **verifiable rather than declarative** — a label saying `out_of_fold` is
worth nothing if nothing can check it:

```python
@dataclass(frozen=True)
class Checkpoint:
    recipe: str
    stage: str
    fold: int | None
    trained_on_row_ids: Tensor        # [M] exactly the rows this fit saw
    parameters: Mapping[str, Tensor]  # graph params, or named array-fit state
    buffers: Mapping[str, Tensor]     # graph buffers; empty for array_fit
    components: tuple[str, ...]
    steps: int
    seed: int
    plan_digest: str                  # sha256 of the plan this ran under

@dataclass(frozen=True)
class PseudoLabels:
    source_stage: str
    source_checkpoints: Mapping[int, Checkpoint]
    predicted_by_fold: Tensor         # [N] which checkpoint produced each row
    row_id: Tensor                    # [N] the rows predicted
    labels: Tensor

    # NOT constructor arguments — computed from the fields above
    @property
    def used_y(self) -> bool: ...            # Y_RAW reachability, from the plan
    @property
    def prediction_mode(self) -> str: ...    # from actual fold disjointness
```

**The provenance fields are properties, not parameters.** An earlier draft of
this section showed them as ordinary keyword arguments while the surrounding
text called them derived facts, which would have let a producer write
`used_y=False` on an outcome-dependent artifact, or claim `out_of_fold` without
ever being checked — a guardrail that any caller can talk its way past is not a
guardrail. Artifacts are therefore constructed only by executor factories:

```python
labels = cross_fit.emit_pseudo_labels(plan, checkpoints, predictions)
```

The factory takes the compiled plan, so `used_y` comes from graph reachability
(§2.2); it takes the checkpoints, so `prediction_mode` is computed from the
actual row sets and `out_of_fold` is *earned* by passing the disjointness check
below rather than asserted. A direct constructor call is rejected with an
`ArtifactError`.

**And the factory itself is not public.** A factory anyone can call is a
factory anyone can hand invented row ids to, which moves the hole up one level
rather than closing it: the constructor guard would then stop only the caller
who was not trying. The public surface is therefore the *executor*, and the
factory is private to it — `run_stage` returns a `Checkpoint` because it ran
the loop that produced it. This is a legibility guard and not a security
boundary: Python cannot prove a caller's provenance, and one determined to
reach the private factory can. What it buys is that no ordinary path yields an
artifact whose provenance was asserted rather than observed, and that a path
which does is a deliberate line in a diff.

A run directory holds **one** compiled recipe, and enforces it in both
directions: a second, different plan is rejected, and so is a checkpoint whose
`plan_digest` is not the plan already written there. The checkpoint carrying
the digest makes the mismatch *detectable*; refusing the write is what stops a
directory existing whose two halves each look valid and describe different
runs.

Two fields do the work. Each checkpoint records the row ids it was **fit on**;
each prediction records **which checkpoint produced it**. `out_of_fold` is then
a claim with a decision procedure:

```
for every predicted row r:
    assert r not in trained_on_row_ids[ predicted_by_fold[r] ]
```

The consuming stage runs that check when it loads the artifact, and `cross_fit`
emits the fields to make it runnable. `prediction_mode` becomes a *summary of a
verified fact* rather than an assertion the reader has to trust. `used_y` is
likewise derived: it is true iff `Y_RAW` lies in the transitive closure of the
subgraph that produced the labels (§2.2), so it cannot be set wrongly.

### 7.2 The causal guardrail, as a compile-time rule

Keep three quantities distinct and never conflate them:

- `p(t|x)` — treatment assignment / propensity
- `q(t|x,y)` — posterior used to *infer* missing treatments
- `p(y|x,t)` — outcome model

Using `q(t|x,y)` to create pseudo-labels and then fitting `p(y|x,t)` on the same
rows is a circular fit. It is coherent inside a joint likelihood or ELBO; as a
*staged* procedure it needs out-of-fold predictions or another leakage control.

This is enforced in two places, because the two halves are knowable at
different times. Trying to do it all at compile time does not work: `used_y`
and the fold assignment live on an artifact that does not exist until the
producing stage has run.

**Static, at `compile(recipe)` — from the declared graph and program:**

`Recipe` carries `purpose: Literal["causal", "predictive"]` and `Stage` carries
`allow_leakage: bool`. The compiler walks the program in order and rejects when
all of the following hold, unless the predictive opt-out below applies:

1. a stage produces treatment labels from a subgraph whose transitive closure
   contains `Y_RAW` (i.e. `used_y` will be true — derivable statically, §2.2);
2. its `executor` is not `cross_fit` and its action does not declare a held-out
   prediction source (so the labels will be in-sample);
3. a later stage consumes that artifact, trains a component providing
   `Y_GIVEN_XT`, and its row scope (§7.0) intersects the labelled rows.

`purpose="predictive"` with `allow_leakage=True` on the consuming stage opts
out. Predictive experiments may; causal estimation may not. The opt-out is
per-stage and must be written down, so it appears in the execution plan and in
the diff.

**Runtime, at artifact load** — the fold-disjointness check of §7.1, which is
the only place the *actual* row assignment can be tested. A `cross_fit` stage
whose folds overlap fails here even though it compiled, and that is the correct
division: the compiler rejects programs that are wrong by construction, the
loader catches executions that are wrong in fact.

---

## 8. Compiler

`compile(recipe) -> CompiledRun` is where the framework earns its keep. It:

1. resolves every registry string to a Python object, **once** — after this
   point the code works with ordinary objects, never string lookups;
2. checks every objective's `(port, realisation)` requirement is produced by
   some component under that realisation;
3. topologically orders components per realisation and plans the minimum number
   of forward passes;
4. checks `trainable` names exist and that every trainable component is actually
   upstream of at least one active objective **through a port that objective
   backpropagates through** (catches dead-weight stages, including the ones
   hidden behind a `detaches` declaration);
5. validates views against the schema (mutability, bounds, derived columns);
6. intersects `Stage.rows` with each `Objective.rows` and rejects any pairing
   that is empty by construction (§7.0);
7. applies the **static** half of the leakage rule (§7.2), the runtime half
   being checked at artifact load;
8. emits a **printable execution plan**.

That last point matters for agent accuracy: the plan is a human-readable
artifact listing, per stage, the forward passes, the active objectives with
their row populations and weight schedules, and the trainable parameter groups.
It can be diffed against the recipe's spec card by a reviewer in under a minute,
before any training runs.

---

## 9. Registries and recipes

Three registries, at the right level:

| Registry | Examples |
|---|---|
| Components | `mlp_encoder`, `tarnet_head`, `conditional_flow`, `categorical_posterior` |
| Objectives & views | `marginal_t_nll`, `vat`, `masked_features`, `weak_x`, `strong_x` |
| Recipes | `tarnet`, `cnflow`, `cycle_dual`, `mean_teacher`, `ssdml` |

```python
learner = xty2.create("cycle_dual", schema=schema, seed=0)
```

expands to `Recipe(system=ComponentGraph([...]), program=Program([...]),
evaluator=EvaluationSuite([...]), card="docs/recipes/cycle_dual.md")`.

**A recipe contains no logic.** It is a declarative assembly of registered
pieces plus explicit hyperparameters. If a recipe needs behaviour that is not a
component, objective or view, that is a signal to add one — and the reviewer's
job is to insist on it rather than accept an `if` statement in the recipe. This
is the primary anti-drift rule for implementation agents.

Recipes are Python functions. A thin dict/YAML loader exists for sweeps and is a
shallow layer over the Python API, not a parallel surface.

### 9.1 Canonical hyperparameter binding

A recipe is arbitrary Python, so "CI checks the recipe sets every hyperparameter
the card names" (`FIDELITY.md` §1.2) needs something to compare *against*. A
card path on the `Recipe` is not enough: nothing maps `teacher.ema_decay` to a
value, and nothing distinguishes a value the recipe set from one it inherited.
Three pieces make it checkable.

**1. The card key vocabulary is closed.** The keys in `FIDELITY.md` §2 are the
whole namespace. Adding a key is a framework change, not a per-card decision, so
cards cannot drift apart from each other or from the code.

**2. Paper-governed fields bind to a canonical key and have no usable default.**

```python
@dataclass
class EMATeacher(Component):
    decay: float = REQUIRED       # sentinel: constructing without it raises
    CARD_KEYS = {"decay": "teacher.ema_decay"}
```

The `REQUIRED` sentinel is what removes the "explicit or inherited?" ambiguity —
not by tracking provenance, but by making inheritance impossible. Fields the
paper does not govern keep ordinary defaults and no binding.

**3. `compile()` emits the resolved values as flat data.**

```python
plan.hyperparameters: dict[str, Any]     # {"teacher.ema_decay": 0.999, ...}
```

CI then asserts that every card §4 key not marked `n/a` appears in
`plan.hyperparameters` with a non-null value. The same flat dict is what makes
the printed plan diffable against the card by eye.

**Per-objective keys are derived, not bound.** Four of the §2 keys —
`losses.weights`, `losses.schedules`, `losses.reduction`, `losses.eligible_rows`
— are annotated "per objective", and a canonical key names one value. `compile()`
therefore aggregates them over the whole program into one mapping each, keyed by
`"<stage>.<objective>"`, rather than having each `Weighted` bind them and
collide. The values come from what the term actually runs with: the schedule's
nominal weight, its description, its reduction mode, and the *effective* row set
after the §7.0 intersection. They are derived because there is nothing for a
recipe to declare twice — it already supplied every one of them by constructing
the `Weighted`.

`gradients.stop_gradients` is derived the same way from each objective's
required `detaches` declaration. A plan therefore shows an explicit `none` for
every ordinary likelihood term and the detached port for a stopped path; a
global prose claim cannot hide one exceptional objective.

**Architecture keys are component-valued.** Components commonly bind the same
canonical field to different values — TARNet has `[200, 200, 200]` in the
encoder, `[100, 100, 100]` in each outcome arm, and a linear propensity head.
`compile()` aggregates every `architecture.*` binding as
`{component_name: value}`.

**Program keys are stage-valued when there is more than one stage.** Learning
rate, duration, optimiser policy and teacher policy commonly change across an
ordered program. For a multi-stage recipe, bindings owned by a stage, its
optimiser or its teacher therefore aggregate as `{stage_name: value}`; bindings
owned by an objective aggregate as `{<stage>.<objective>: value}`. A one-stage
recipe keeps the scalar surface its card already uses. Conflicts within one
scope remain errors. This is the smallest representation that lets the plan
show a real transition instead of either rejecting it or retaining whichever
stage happened to be merged last.

**What this does not do.** It checks *presence*, never *correctness*: CI can
prove the recipe sets `teacher.ema_decay`, and cannot prove 0.999 is the number
in the paper. Only the card review does that. The mechanism stops silent
omission — the looptab failure — and nothing more; treating a green cross-check
as evidence of fidelity would be exactly the over-trust this document exists to
prevent.

---

## 10. Package layout

```
xty2/
  core/        batch.py schema.py ports.py distributions.py rows.py views.py
               graph.py card_keys.py loss.py schedules.py optimisation.py
               recipe.py compile.py
  components/  encoders/ outcome/ treatment/ posterior/ density/ energy/
  views/       masking.py perturbations.py
  objectives/  supervised.py marginal.py consistency.py generative.py causal.py
  training/    program.py loss_mixer.py executors.py artifacts.py
  recipes/     tarnet.py cnflow.py cycle_dual.py mean_teacher.py ssdml.py
  evaluation/  predictive.py causal.py calibration.py policy.py
  estimators/  cate.py dml.py policy.py
docs/
  DESIGN.md FIDELITY.md PLAN.md recipes/<name>.md
tests/
  invariants/  smoke/  benchmarks/
```

`Stage`, `Recipe`, `Weighted` and the `Objective` protocol are in
`core/recipe.py` rather than beside the training layer, because they are the
compiler's *input*: they declare what a program is, and `compile()` reads them.
`training/` imports `core`, never the reverse, so putting the declarations in
the leaf layer is what keeps that one-way. The same argument puts `loss.py`
(`LossTerm`, `TrainContext`, the reduction modes of §6.1) and `schedules.py`
(§6) in `core/` rather than in `training/` beside the mixer: `Weighted` names a
schedule and the `Objective` protocol is stated in terms of a `LossTerm`, so
both are read by the compiler. `training/loss_mixer.py` holds what *runs* them.
`card_keys.py` holds the closed vocabulary of `FIDELITY.md` §2 and the
`REQUIRED` sentinel (§9.1).

### The five ported recipes, and what each one proves

| # | Recipe | Exercises |
|---|---|---|
| 1 | `tarnet` | ports, outcome head, propensity, exact marginalisation, single stage |
| 2 | `cnflow` | distribution protocol generality; categorical t as context; marginal NLL unchanged |
| 3 | `mean_teacher` | views, teacher realisation, multi-objective mixing, ramps |
| 4 | `cycle_dual` | posterior `q(t|x,y)`, staged pseudo-labelling, leakage guardrail |
| 5 | `ssdml` | `array_fit` and `cross_fit` executors, out-of-fold artifacts |

If these five compose cleanly, we have enough framework. That is the stopping
condition, not a milestone on the way to porting forty models.

---

## 11. Overdesign guardrails

**The two-consumer rule.** No abstraction enters the framework until a second
real recipe needs it. New ports, new executors, new schedule types, new
realisation axes all fall under this. `Realisation` is the one advance purchase
(§2.1) and it is documented as such.

**The YAGNI ledger.** Deliberately not built, with the evidence that would
change the decision:

| Not building | Would build when |
|---|---|
| General stage DAG | a recipe needs genuinely parallel branches, not just ordering |
| Continuous / dose-response T | the flight data or a target paper requires it |
| GradNorm / PCGrad | logged gradient cosines show sustained objective conflict |
| Config-first surface | sweeps outgrow the Python API in practice |
| Plugin / entry-point system | a consumer outside this repo exists |
| Distributed training | a single recipe stops fitting in one process |
| A loader / sampler, and with it the `optimisation.batch_size` and `labelled_unlabelled_ratio` bindings | a recipe needs a fixed labelled/unlabelled quota per batch rather than whatever the data gives. The gradient executor takes an iterable of batches; a stage field for either key would be a card key nothing could check, which §7.1 rejects for provenance and §9.1 for hyperparameters |
| LR schedules beyond `Constant`, `Ramp`, `SigmoidRamp`, `CosineDecay`, `Step` and `ExponentialDecay` (one-cycle, warm restarts) | a card names one. The rate is a schedule multiplier, so a new type serves both loss weights and the LR; `ExponentialDecay` entered with TARNet, `SigmoidRamp` with Mean Teacher and `CosineDecay` with FixMatch, the first real cards that name them |
| The other ~35 XTYLearner families | one is actually needed for a result |

**Migration is lazy by design.** No model is ported until it is next used. A
model family that nobody has run in a year is not a requirement, it is a
liability, and reproducing it faithfully costs more than it is worth until
someone needs the number.
