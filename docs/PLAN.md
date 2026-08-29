# xty2 build history

**Status:** P0–P12 complete. This file records sequencing decisions; it is not
the active task queue. Current work is scoped by a reviewed card, issue, or PR.

## Shape of the plan

The initial build alternated framework packets with real recipes. Gate 1 tested
whether two different outcome parameterisations shared the same contracts.
Gate 2 stopped framework growth after five recipes exercised stages and
executors. P12 then attached published-number evidence.

## Phase A — core (P0–P4)

### P0 · Repo scaffolding

Package, CI, three test tiers, and agent rules.

### P1 · Core data types

`XTYBatch`, schema, ports, row populations, and executable distribution
conformance.

### P2 · Component protocol, graph, compiler

Component graph, realisations, compile-time checks, and printable plans.

### P3 · Objectives, mixer, schedules, logging

Independent objectives, reductions, schedules, exact marginalisation, and
optional gradient diagnostics.

### P4 · Gradient executor and artifacts

Single-stage training, deterministic execution, and immutable artifacts.

## Phase B — first recipes (P5–P7)

### P5 · Recipe 1 — `tarnet`

Proved the Phase A stack and card-to-plan hyperparameter checks.

### P6 · Views

Added schema-aware transforms, deterministic draws, and consistency loss.

### P7 · Recipe 2 — `cnflow`

Proved a conditional flow could reuse the marginal objective unchanged.

Gate 1 passed: recipes stayed declarative and shared the objective/executor
contracts.

## Phase C — sequencing (P8–P11)

### P8 · Program, stages, teacher parameters

Ordered stages, explicit checkpoint inheritance, freezing, and EMA teachers.

### P9 · Recipe 3 — `mean_teacher`

Exercised views, teacher parameters, multiple objectives, and ramps.

### P10 · Executors, cross-fitting, leakage rules

Added `array_fit`, `cross_fit`, provenance-bearing pseudo-labels, and static plus
runtime leakage checks.

### P11 · Recipes 4 and 5 — `cycle_dual`, `ssdml`

Exercised staged outcome-dependent labels and out-of-fold estimation.

Gate 2 passed. New framework vocabulary now requires evidence from a reviewed
card under `DESIGN.md` §11.2.

## Phase D — validation (P12)

### P12 · Evaluation suite and Tier 2 runner

Added predictive, causal, and calibration metrics; card-driven benchmarks;
nightly execution; and result-ledger writeback.

## Risk register

| Risk | Control |
|---|---|
| A paper mechanic is omitted | reviewed card §3–§4 and plan diff |
| A plausible implementation is wrong | Tier 0 invariants, Tier 1 wiring fit, Tier 2 target |
| A hyperparameter silently defaults | closed card keys and `REQUIRED` bindings |
| A framework limitation becomes permanent | typed deviation and reconciled §11.4 ledger |
| Outcome-dependent labels leak | compile-time lineage and runtime fold checks |
| Framework breadth outruns evidence | card-first additions and §11.2 guardrails |

## What "done" means

A recipe is done only when its reviewed card, implementation, Tier 0/Tier 1
tests, and Tier 2 ledger agree. `implemented` is not a result status.
