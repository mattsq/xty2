# Fidelity

**Status:** current

`DESIGN.md` makes implementation errors attributable. This document defines how
xty2 detects them.

## 1. The core mechanism: the spec card

Every recipe has `docs/recipes/<name>.md`. Write and review the card before
implementation. If implementation reveals an unlisted mechanic, amend the card
and review it before continuing.

The authoritative form is `docs/recipes/_TEMPLATE.md`:

1. provenance and pinned sources;
2. estimand, claim, and non-claims;
3. source equations mapped to xty2 objects;
4. machine-readable mechanics;
5. typed deviations and framework additions;
6. predeclared reproduction protocol and result ledger;
7. source ambiguities and chosen resolutions;
8. review record.

### 1.1 Card status

| Status | Evidence |
|---|---|
| `draft` | card exists but is not reviewed |
| `reviewed` | implementation may begin |
| `implemented` | code exists and Tier 0 passes |
| `smoke-passing` | Tier 1 passes |
| `reproduced` | Tier 2 meets the predeclared target |
| `deviating` | Tier 2 does not meet the target, with an explanation in §5 |

Only Tier 2 can set `reproduced` or `deviating`.

### 1.2 Cards that bite instead of rot

- **Card/plan cross-check:** card §4 uses the closed vocabulary in §2.
  Paper-governed fields bind to those keys with `REQUIRED` defaults.
  `compile()` emits `plan.hyperparameters`; Tier 0 checks key presence and,
  where supported, values. This proves agreement with the card, not with the
  paper. Review establishes the latter.
- **Debt reconciliation:** a §5 `framework-limitation` cites a live
  `DESIGN.md` §11.4 key, and that ledger row cites the card. Tier 0 checks both
  directions.
- **Plan diff:** the PR includes the rendered execution plan. Review compares
  its components, lineage, views, objectives, rows, schedules, and
  hyperparameters with card §3–§4.

## 2. The mechanics checklist

Every applicable value needs a source citation. Use explicit `none` or `n/a`;
absence is not an answer. The block shape and key vocabulary are read by Tier 0.

```yaml
gradients:
  stop_gradients:        # where, and on which side. "none" must be explicit.
  detached_targets:      # is the consistency target detached?
  gradient_clipping:     # norm/value/none, and the threshold
  marginal_nll_grad_path: # does the marginalisation term train the propensity, the outcome head, or both?

teacher:
  ema_decay:
  ema_applies_to_buffers: # BN running stats too, or parameters only?
  teacher_in_train_mode:  # does the teacher see dropout / BN batch stats?
  teacher_requires_grad:  # must be false; asserted in Tier 0

losses:
  reduction:             # mean | sum | population — see DESIGN.md §6.1
  eligible_rows:         # per objective
  weights:               # per objective, with units
  schedules:             # ramp start/end/length, in steps or epochs — state which
  temperature:
  sharpening:
  confidence_threshold:

optimisation:
  optimiser:
  lr:
  lr_schedule:           # including warmup
  weight_decay:          # coefficient, component scope, and bias / norm reach
  batch_size:
  labelled_unlabelled_ratio: # per batch — a very common silent difference
  total_steps_or_epochs: # state which; epochs on a semi-supervised loader are ambiguous

architecture:
  widths_depths:
  activation:
  normalisation:         # BN vs LN vs none — and where
  dropout:
  initialisation:        # if the paper or reference code specifies it
  output_parameterisation: # e.g. does the outcome head predict mean, or mean and log-variance?

data:
  standardisation:       # fit on which split? a classic leakage point
  outcome_scaling:       # is y standardised, and are metrics reported on the original scale?
  treatment_encoding:
  split_protocol:
  missingness_mechanism:  # how t-missingness is simulated, if it is
```

## 3. Three test tiers

### Tier 0 — invariants (every PR, seconds)

Tier 0 tests repository-wide contracts independently of paper-specific results:

- exact treatment marginalisation;
- outcome/treatment distribution shapes, normalisation, and candidate scoring
  with `B != K`;
- trainable and EMA-teacher isolation;
- zero-eligible-row handling and row-scope composition;
- immutable batches, preserved fields, schema bounds, and derived columns;
- deterministic seeded views and training;
- port contracts and compile-time failures;
- artifact provenance and out-of-fold disjointness;
- card keys, deviations, and execution-plan stability.

The executable conformance checks live in `xty2/core/conformance.py`; the suite
lives in `tests/invariants/`.

### Tier 1 — smoke fit on a synthetic DGP (every PR)

Tier 1 is a wiring test, not a reproduction claim. Each recipe uses a small DGP
with analytic targets and coarse directional assertions. Depending on the
recipe, these cover loss reduction, propensity performance, treatment-effect
error, coverage, leakage, and the marginal-likelihood advantage over a matched
complete-case ablation. A smoke pass cannot set `reproduced`.

### Tier 2 — published-number reproduction (nightly)

Card §6 defines the dataset, split, metric, tolerance, and replicate count.
The benchmark binds every scalar in that YAML block and stores its digest with
the result.

Rules:

- Declare tolerance before running. Widening it later is a deviation.
- Use at least two replicates and report mean plus sample standard error.
- A required metric passes only when its mean satisfies the target by at least
  one standard error. A nominal pass inside its own noise is `deviating`.
- Record date, commit, metric, value, and tolerance result in the card ledger.
- A fresh nightly result must agree with the recorded card status.

## 4. The implementation loop for agents

1. Read the card. If absent, draft it and stop for review.
2. Implement only the components, objectives, views, data policy, and stages in
   §3–§4.
3. Amend and re-review the card before widening that set.
4. Run Tier 0, Tier 1, lint, format, and type checks.
5. Include the rendered plan in the PR and compare it with §3–§4.
6. Run Tier 2 and update the status and ledger.
7. If the change discharges a §11.4 debt, reconcile every paying card in the
   same PR.

### 4.1 Standing rules

- No silent defaults for paper-governed values.
- No conditionals in recipe assembly.
- Record departures before implementing them.
- A missing fidelity-bearing abstraction is in scope after card amendment and
  review; do not disguise it as a permanent deviation to keep a diff small.
- A falling training loss is not evidence that the method is implemented.
- Port methods lazily, in response to a reviewed need.

## 5. Deviations, and the debt they create

| Kind | Meaning | Lifecycle |
|---|---|---|
| `judgement` | a deliberate choice we would retain with an unlimited framework | permanent unless reconsidered |
| `framework-limitation` | the source mechanic is omitted because xty2 cannot express it | open debt |
| `withdrawn` | a former deviation has been implemented or rejected | retained as history |

### 5.1 The form

The first table after card §5 has this exact shape:

| # | Kind | Blocked on | What we do differently | Why | Expected effect on §6 |
|---|---|---|---|---|---|
| 1 | `judgement` | — | ... | ... | ... |

`Blocked on` is empty for `judgement` and `withdrawn`. A
`framework-limitation` cites a `DESIGN.md` §11.4 key. Framework additions made
for the card go in §5.1, including the named second consumer required for
load-bearing vocabulary under `DESIGN.md` §11.2.

A framework limitation never belongs in §7. That section is for ambiguity in
the source, not inability in xty2.

### 5.2 What the Tier 0 reconciliation asserts

1. Every row has one of the three kinds.
2. Every open limitation cites a live ledger key; other kinds cite none.
3. The ledger's **Who is paying** column exactly matches the citing card rows.
4. A `reproduced` card with open debt names that debt beside its §6 result.

Deleting a ledger row therefore breaks every card still paying it. Each card
must either withdraw the deviation or restate it as a judgement explaining why
it survives the new capability.

### 5.3 A note on scope

Debt repayment does not authorise unrelated changes. Revising another card is a
reviewed amendment, and a material recipe change invalidates or reruns its Tier
2 evidence. Silence is the only prohibited outcome.
