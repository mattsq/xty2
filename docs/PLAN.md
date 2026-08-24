# xty2 — Build plan

**Status:** draft v1, for review
**Reads with:** `DESIGN.md` (architecture), `FIDELITY.md` (correctness system)

## Shape of the plan

Thirteen work packets, P0 through P12 — five in Phase A, three in Phase B, four
in Phase C, one in Phase D — each sized for a single focused session, each with explicit
acceptance criteria and an explicit out-of-scope list. Packets are ordered so
that **every second or third packet is a real recipe**, not more framework. The
framework is only ever built to the depth the next recipe demands.

Two hard review gates (after P7 and after P11) ask the same question: *does the
abstraction hold, and is it already enough?* The default answer at the second
gate should be "enough" — see `DESIGN.md` §11.

---

## Phase A — core (P0–P4)

### P0 · Repo scaffolding
**Goal:** a repo an agent can work in without inventing conventions.
**Ships:** `pyproject.toml`, pytest layout (`tests/invariants`, `tests/smoke`,
`tests/benchmarks`), ruff + mypy config, CI workflow running Tier 0 + Tier 1 on
PR and Tier 2 nightly (initially both empty), and a short `CLAUDE.md` that says:
read the card first, Tier 0+1 must pass, no logic in recipes, paste the
execution plan in the PR.
**Accept:** `pytest` runs green on an empty suite; CI is green; `mypy --strict`
passes on an empty package.
**Not in scope:** any framework code.

### P1 · Core data types
**Goal:** `XTYBatch`, `Schema`, `FeatureSpec`, `Port`, `PortSpec`, distribution
protocols, row-population resolution.
**Also ships:** an **executable reference implementation** of the
candidate-treatment contract (`DESIGN.md` §3.1) covering `log_prob`, `mean` and
`sample` — a trivial Gaussian head plus the conformance test every outcome head
is later run against. `mean` matters as much as `log_prob`: CATE is computed
from treatment-wise means, so a head that passes only the `log_prob` test can
still corrupt every Tier 2 causal metric. Three separate
attempts to state this contract in prose were wrong; it stops being a prose
claim here.
**Accept:** Tier 0 tests for mask semantics (no sentinels reachable), row
populations, zero-eligible-row behaviour, batch immutability under transforms,
and every `PortSpec` shape contract. The candidate-treatment conformance test
passes with `B != K`. `Schema` validates its own feature graph (`derived_from`
is acyclic and complete).
**Not in scope:** components, losses, anything that trains.

### P2 · Component protocol, graph, compiler
**Goal:** `Component` (an `nn.Module` base class, per `DESIGN.md` §3),
`ComponentGraph`, the virtual source node supplying `X_RAW` and `Y_RAW`,
`Realisation`, `State`, `compile()`, the printable execution plan, and
`plan.hyperparameters` — the flat canonical-key dict the card cross-check
compares against (`DESIGN.md` §9.1), with `REQUIRED` sentinels on paper-governed
fields.
**Accept:** compile-time rejections all raise with actionable messages —
unsatisfied port, unsatisfied realisation, unknown trainable name, dead trainable
component, empty `Stage.rows ∩ Objective.rows`. `PortView` raises on undeclared
reads, and components take no `XTYBatch` argument at all, so raw-input
dependencies are declared or unavailable. The printed plan shows full data
lineage including which components read `Y_RAW`. Plan printing is
snapshot-tested.
**Not in scope:** leakage rules (P10), views (P6), any real component, and the
derived-column rule — it can only be violated by a transform, so it has no
expressible test surface until P6 owns it.

### P3 · Objectives, mixer, schedules, logging
**Goal:** `Objective`, `LossTerm`, `LossMixer`, `Constant`/`Ramp`/`Step`, the
full §6.2 logging surface — including per-objective gradient norms and pairwise
gradient cosines — and three objectives: `ObservedOutcomeNLL`,
`ObservedTreatmentNLL`, `MissingTreatmentMarginalNLL`.
**Accept:** exact marginalisation equals brute force over `k`; the broadcast
contract test passes for a trivial Gaussian head; unweighted values and `n` are
logged per objective; a zero-`n` term never reaches the total; gradient cosines
are verified against constructed objectives with analytically identical (≈ +1)
and analytically opposed (≈ −1) gradients, and are off by default and off in CI.
**Not in scope:** consistency losses (P6).

> Gradient cosines land here, not at P9, for two reasons: they are mixer
> machinery and belong beside the mixer, and `tarnet` (P5) already mixes three
> objectives with a ramp, so objective conflict is observable two packets before
> Mean Teacher needs it. Deterministic construction also makes them testable
> here, which "turn it on and look at a run" never was.

### P4 · Gradient executor and artifacts
**Goal:** single-stage `gradient` executor, training loop, run directory,
immutable artifact writing and loading.
**Accept:** trainable-isolation invariant passes; determinism invariant passes;
artifacts are immutable and carry provenance fields; no code path mutates a
dataset.
**Not in scope:** `Program`, multiple stages, EMA, cross-fitting.

---

## Phase B — first recipes (P5–P7)

### P5 · Recipe 1 — `tarnet` ★
**Card first.** Write `docs/recipes/tarnet.md`, get it to `reviewed`, then
implement: `mlp_encoder`, `tarnet_head`, `categorical_propensity`, single stage.
**Proves:** ports, outcome head, propensity, exact marginalisation, the whole
Phase-A stack end to end.
**Accept:** card at `smoke-passing`; Tier 1 assertions in `FIDELITY.md` §3 pass,
including *marginalisation beats complete-case at 50% missing t*; the card §4 ↔
`plan.hyperparameters` cross-check runs and **fails when a card key is deleted
from the recipe** — a cross-check that has never been seen to fail is not known
to work; Tier 2 queued.
**Watch for:** this is the first time the card→plan diff is exercised. If the
diff is not genuinely informative, fix the plan printer now, not later.

### P6 · Views
**Goal:** `ViewSpec`, transforms (`FeatureMask`, `BoundedJitter`), schema-aware
perturbation, view-keyed realisation execution, `ConsistencyLoss`.
**Accept:** `preserves` is enforced by test; `mutable=False` respected; bounds
respected; derived-column rule rejects at compile time; views are deterministic
given a seed; the compiler plans the minimum number of forward passes.
**Not in scope:** teacher parameters (P8), VAT's inner loop unless P7 needs it.

### P7 · Recipe 2 — `cnflow` ★
**Card first.** Conditional flow outcome head with **categorical t as context**.
**Proves:** the distribution protocol generalises; `MissingTreatmentMarginalNLL`
is reused *without modification*.
**Accept:** the marginal NLL objective's source file is untouched by this
packet's diff. If it is not, the abstraction has failed and that is the finding.

> ### ▲ Gate 1 — does the abstraction hold?
> Two recipes with different outcome parameterisations now share a propensity
> head, a marginalisation objective and a training loop. Confirm: no objective
> was edited for P7; no recipe contains a conditional; the plan diff caught at
> least one real discrepancy during P5–P7. If any of these fail, stop and fix
> the core before adding stages.

---

## Phase C — sequencing (P8–P11)

### P8 · Program, stages, teacher parameters
**Goal:** `Stage`, `Program` (ordered list), `initialise_from`, stage artifacts,
EMA teacher as a `params="teacher"` realisation.
**Accept:** teacher-isolation invariant passes (no grad, `requires_grad=False`,
buffer handling explicit and card-driven); stage checkpoints are immutable;
freezing by component name works.
**Not in scope:** DAG scheduling, parallel stages — ever, without new evidence.

**Implemented contract:** stages start from recipe-initial state plus the named
earlier checkpoint, never implicit preceding-stage state. EMA updates occur
after each student step. Frozen student components run in evaluation mode;
teacher parameter gradients, mode and parameter/buffer update policies are
covered by Tier 0 invariants.

### P9 · Recipe 3 — `mean_teacher` ★
**Card first.** **Proves:** views + teacher realisation + multi-objective mixing
+ ramp schedules, together.
**Accept:** gradient-cosine logging (implemented in P3) enabled for one run,
with the resulting objective-conflict trace recorded in the PR body; ramp
lengths come from the card, not from a default; Tier 1 catches a deliberately
mis-scheduled consistency weight (write that as a mutation test).

### P10 · Executors, cross-fitting, leakage rules
**Goal:** `array_fit` and `cross_fit` executors, `PseudoLabels` and `Checkpoint`
artifacts carrying `trained_on_row_ids` / `predicted_by_fold`, and both halves of
the `DESIGN.md` §7.2 guardrail.
**Accept:** *Static* — a program that pseudo-labels with `q(t|x,y)` in-sample and
then fits `p(y|x,t)` on intersecting rows is **rejected at compile time**, driven
by `Y_RAW` reachability in the producing subgraph rather than a hand-set flag,
and accepted only under `purpose="predictive"` with `allow_leakage=True` on the
consuming stage. *Runtime* — loading an `out_of_fold` artifact executes the
disjointness check of `DESIGN.md` §7.1 and **fails on a deliberately overlapping
fold assignment**. That mutation test is the acceptance criterion; a provenance
label that nothing can falsify is not "checked".

### P11 · Recipes 4 and 5 — `cycle_dual`, `ssdml` ★
**Cards first.** `cycle_dual` exercises the posterior `q(t|x,y)`, staged
pseudo-labelling and the leakage guardrail; `ssdml` exercises `array_fit` and
`cross_fit`.
**Accept:** both at `smoke-passing`; the leakage rule fires on a real recipe, not
only on a synthetic test.

> ### ▲ Gate 2 — is it enough?
> Five recipes spanning discriminative, flow, teacher-student, staged
> pseudo-labelling and array/cross-fit executors. Per `SEED.md`, if these compose
> cleanly the framework is **done**. The default decision here is to stop adding
> abstraction and start producing results. Anything proposed at this gate goes
> through the two-consumer rule.

---

## Phase D — validation (P12)

### P12 · Evaluation suite and Tier 2 runner
**Goal:** `evaluation/` (predictive, causal, calibration), the per-recipe
benchmark runner driven by card §6, the nightly workflow, and the result-ledger
writeback.
**Accept:** each of the five recipes has a Tier 2 result recorded in its card
with mean ± stderr, and a status of `reproduced` or `deviating` with a written
explanation. **This packet is what makes the other eleven trustworthy** — a
recipe that never reaches it is unvalidated by definition.

---

## Risk register

| Risk (from looptab / XTYLearner) | Mechanism that answers it | Packet |
|---|---|---|
| Paper details silently omitted | Spec card §4 checklist, written and reviewed before code; CI cross-checks that the recipe sets every key | P0, P5 |
| No ground truth; wrong code looks plausible | Tier 2 per-recipe reproduction with pre-declared tolerance, recorded in the card | P12 |
| Slow feedback — errors found only at benchmark time | Tier 0 invariants + Tier 1 synthetic smoke on every PR | P1–P4 |
| Can't attribute a bad number | Component/objective separation; per-objective raw values, `n`, grad norms and cosines | P3 |
| An objective silently dies (zeroed, empty rows, detached) | Zero-`n` invariant, coverage logging, and the Tier 1 "marginalisation beats complete-case" assertion | P3, P5 |
| Agents over-build / rewrite working code | Two-consumer rule, YAGNI ledger, "no logic in recipes", card-amendment-before-code | throughout |
| Abstraction turns out wrong late | Gate 1 after two recipes, with a falsifiable test (P7 must not edit P3's objective) | P7 |
| Leakage from `q(t\|x,y)` pseudo-labels into `p(y\|x,t)` | Compile-time rejection on artifact provenance | P10 |
| Card rot | CI parses card §4 and asserts recipe coverage; plan diff at review | P0, P5 |
| Breadth without depth (40 families, no trustworthy numbers) | Lazy migration; five recipes is the stopping condition, not a milestone | Gate 2 |

## What "done" means

Not "the framework is complete". Done is: **five recipes, each with a reviewed
card, each passing Tier 0 and Tier 1 on every PR, each with a recorded Tier 2
result that either matches its published number within a pre-declared tolerance
or explains in writing why it does not.**

A sixth model is then a card and a recipe file, and if it needs more than that,
the framework — not the model — is the thing to discuss.
