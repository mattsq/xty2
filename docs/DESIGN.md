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
    x: Tensor                 # [B, D] or a dict for mixed types
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

`XTYBatch` is frozen. Transforms produce new batches (see §5), they never mutate.

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

The engine computes each index set once per batch and hands the objective only
the eligible rows.

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
    X_REPR          = "x_repr"
    XY_REPR         = "xy_repr"
    Y_GIVEN_XT      = "p(y|x,t)"
    T_GIVEN_X       = "p(t|x)"
    T_GIVEN_XY      = "q(t|x,y)"
    JOINT_ENERGY    = "energy(x,t,y)"
    RECONSTRUCTION  = "reconstruction"
```

Each port has a `PortSpec` fixing its **type and shape contract**, checked at
compile time and asserted in tests:

| Port | Type | Contract |
|---|---|---|
| `X_REPR` | `Tensor` | `[B, H]` |
| `T_GIVEN_X` | `TreatmentDistribution` | `probs: [B, K]`, rows sum to 1 |
| `T_GIVEN_XY` | `TreatmentDistribution` | `probs: [B, K]` |
| `Y_GIVEN_XT` | `OutcomeDistribution` | `log_prob(y, t)` broadcasts over t (§3.1) |
| `JOINT_ENERGY` | `Tensor` | `[B, K]`, one energy per treatment value |
| `RECONSTRUCTION` | `Tensor` | same shape as `x` |

Ports are the framework's vocabulary. **Adding a port is a design decision, not
an implementation detail** — it requires a second real consumer (§11).

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

State is therefore `dict[Realisation, dict[Port, Any]]`, with `state.default`
sugar for the common case. The compiler collects the set of realisations the
objectives demand and plans exactly that many forward passes — no more.

This is the one piece of machinery in v1 that exists for a model we have not
written yet (Mean Teacher, P9). It is included because the alternative — letting
each consistency loss re-invoke the graph itself — is what produced
uncontrolled, untestable forward passes in the previous codebase.

---

## 3. Components: parameterisation only

```python
class Component(Protocol):
    name: str
    requires: frozenset[Port]
    provides: frozenset[Port]

    def forward(self, state: PortView, batch: XTYBatch) -> dict[Port, Any]: ...
```

Rules, all enforced rather than documented:

- A component reads **only** the ports in `requires`. The engine passes a
  `PortView` that raises on undeclared reads, so a component cannot quietly
  reach for `batch.x` when it declared it consumes `X_REPR`.
- A component writes **only** the ports in `provides`.
- **A component never computes a loss.** No `loss()` method exists on the
  Component protocol. This is the single largest departure from XTYLearner.
- Parameters are owned by the component and addressed by `name`, so stages can
  freeze/unfreeze by name without reaching into module internals.

Examples: `mlp_encoder` (x → `X_REPR`), `tarnet_head` (`X_REPR` → `Y_GIVEN_XT`),
`categorical_propensity` (`X_REPR` → `T_GIVEN_X`), `conditional_flow`
(`X_REPR` + categorical t as context → `Y_GIVEN_XT`), `xy_posterior`
(x, y → `T_GIVEN_XY`).

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

**The broadcast contract is the load-bearing detail.** `Y_GIVEN_XT.log_prob`
must accept `t` of shape `[B]` (→ returns `[B]`) *or* `[B, K]` (→ returns
`[B, K]`), and it is **elementwise in `t`**: `out[i, k] = log_prob(y_i, t[i, k])`.
A caller wanting all candidates passes `arange(K)` broadcast across the batch;
the port makes no assumption about what the caller put in `t`. Every outcome
head must satisfy this, and it is a Tier 0 test.

That single contract is what lets `MissingTreatmentMarginalNLL` (§4.1) work
unchanged across a TARNet head, a Gaussian density head, a conditional flow and
an energy model. It also settles the `cnflow_model.py` problem directly: **t is
categorical context to the flow, never a dimension inside the flow.**

---

## 4. Objectives: losses as independent objects

```python
class Objective(Protocol):
    name: str
    requires: frozenset[tuple[Port, Realisation]]
    rows: Rows

    def compute(self, state: State, batch: XTYBatch, ctx: TrainContext) -> LossTerm: ...

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
- The reduction convention is **mean over eligible rows**, matching how papers
  state their losses. The mixer offers `population_weighted=True` to instead
  weight by `n / B`. Which one a recipe uses is a spec-card field (§4 of the
  card), because this choice silently changes the effective weight of any
  objective whose row population varies across batches — a classic
  reimplementation discrepancy.

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

---

## 5. Views: augmentation separated from loss

```python
ViewSpec(
    name="strong_x",
    transforms=[FeatureMask(p=0.25),
                BoundedJitter(columns=["KTAS", "TRQ", "FF"])],
    preserves={"t", "y"},
)
```

- A view is a **pure function of (batch, rng_key)** — deterministic given a
  seed, so left/right views are reproducible and diffable.
- `preserves` is checked by test, not trusted: a view declaring
  `preserves={"t","y"}` that alters `t` or `y` fails Tier 0.
- Transforms consult `FeatureSpec`. `mutable=False` columns are never touched;
  `bounds` are respected; `perturbation_scale` is in natural units.
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

Schedules in v1: `Constant`, `Ramp` (linear), `Step`. Schedules are functions of
`ctx.global_step` and are logged.

The mixer is the single place that later supports alternating updates, per-
objective update frequency, loss normalisation, GradNorm and PCGrad. **None of
those are implemented in v1** — the mixer just holds the seam.

### 6.1 Mandatory logging

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
    objectives: list[Weighted] | None = None
    action: StageAction | None = None          # non-gradient stages
    trainable: list[str] = ()                  # component names
    rows: Rows = "all"
    initialise_from: str | None = None         # a previous stage's checkpoint
    inputs: list[str] = ()                     # artifacts from previous stages
    executor: Literal["gradient", "array_fit", "cross_fit"] = "gradient"
    optimiser: OptimiserSpec | None = None
    steps: int | None = None
```

`Program` is an **ordered list** of stages. Not a DAG. Stages run in sequence;
`initialise_from` and `inputs` reference earlier stages by name. This covers the
full Beyer-style recipe — representation pretraining → joint fit → build EMA
teacher → generate pseudo-labels → refit → targeted causal fit — and we will not
build a scheduler until a fifth model genuinely needs one.

Making the executor explicit also removes an inference we should never have
had: XTYLearner decided that "a model with `fit()` but no `loss()` uses
ArrayTrainer". Now the recipe says `executor="array_fit"`.

### 7.1 Artifacts and provenance

Stages emit **immutable artifacts** into a run directory: checkpoints, EMA
teachers, pseudo-label tables, fold assignments. **No stage mutates the source
dataset.** Pseudo-labels are a side table keyed by `row_id`, joined at load time
by the consuming stage.

Every generated artifact carries provenance:

```python
PseudoLabels(
    source_stage="joint_xty",
    used_y=True,                       # was q(t|x,y) involved?
    prediction_mode="out_of_fold",     # "in_sample" | "out_of_fold" | "held_out"
    fold_ids=...,
)
```

### 7.2 The causal guardrail, as a compile-time rule

Keep three quantities distinct and never conflate them:

- `p(t|x)` — treatment assignment / propensity
- `q(t|x,y)` — posterior used to *infer* missing treatments
- `p(y|x,t)` — outcome model

Using `q(t|x,y)` to create pseudo-labels and then fitting `p(y|x,t)` on the same
rows is a circular fit. It is coherent inside a joint likelihood or ELBO; as a
*staged* procedure it needs out-of-fold predictions or another leakage control.

The compiler therefore **rejects** a program in which a downstream stage trains
a component providing `Y_GIVEN_XT` on rows whose pseudo-labels have
`used_y=True` and `prediction_mode == "in_sample"` — unless the recipe declares
`purpose="predictive"` and sets `allow_leakage=True`. Predictive experiments may
opt in; causal estimation may not.

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
   upstream of at least one active objective (catches dead-weight stages);
5. validates views against the schema (mutability, bounds, derived columns);
6. applies the leakage rules of §7.2;
7. emits a **printable execution plan**.

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

---

## 10. Package layout

```
xty2/
  core/        batch.py schema.py ports.py distributions.py graph.py compile.py
  components/  encoders/ outcome/ treatment/ posterior/ density/ energy/
  views/       masking.py tabular.py perturbations.py
  objectives/  supervised.py marginal.py consistency.py generative.py causal.py
  training/    stage.py program.py loss_mixer.py schedules.py executors.py artifacts.py
  recipes/     tarnet.py cnflow.py cycle_dual.py mean_teacher.py ssdml.py
  evaluation/  predictive.py causal.py calibration.py policy.py
  estimators/  cate.py dml.py policy.py
docs/
  DESIGN.md FIDELITY.md PLAN.md recipes/<name>.md
tests/
  invariants/  smoke/  benchmarks/
```

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
| The other ~35 XTYLearner families | one is actually needed for a result |

**Migration is lazy by design.** No model is ported until it is next used. A
model family that nobody has run in a year is not a requirement, it is a
liability, and reproducing it faithfully costs more than it is worth until
someone needs the number.
