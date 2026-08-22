# Fidelity — how xty2 keeps implementations honest

**Status:** draft v1, for review
**Problem this solves:** in `looptab`, reimplementations silently omitted key
mechanics from the source papers, and nothing reproduced a known-good number, so
a wrong implementation looked plausible for a long time. Both failures are
addressed here, in that order — because the second is what let the first survive.

The architecture in `DESIGN.md` makes a wrong implementation *attributable*.
This document makes it *detectable*.

---

## 1. The core mechanism: the spec card

**Every recipe has a card at `docs/recipes/<name>.md`, and the card is written
and reviewed before any code is written.**

That ordering is the whole point. Reading a paper carefully and transcribing its
mechanics is a different cognitive task from writing PyTorch, and doing them
together is how details get dropped. Separating them creates a cheap human gate:
reviewing a card takes ten minutes and catches missing details *before* they are
buried in a diff.

A card has seven sections. Sections 4 and 6 are machine-readable YAML; the rest
is prose. `docs/recipes/_TEMPLATE.md` is the authoritative form.

1. **Provenance** — paper, authors, year, DOI/arXiv, *which version*, and a
   reference implementation URL pinned to a commit if one exists.
2. **Estimand and claim** — what quantity the method estimates, what the paper
   claims about it, and what it does *not* claim.
3. **Equations** — the losses transcribed in the paper's own notation with its
   equation numbers, followed by a mapping table: paper symbol → xty2
   Port / Objective / Component.
4. **Mechanics checklist** (YAML) — §2 below. Every entry cites where in the
   paper it is stated.
5. **Deviations** — a table of everything we do differently, and why. *"None"
   is a valid answer but must be written explicitly.* An empty section is not
   the same as a section asserting there are no deviations.
6. **Reproduction target** (YAML) — dataset, split protocol, metric, published
   value, tolerance, seed count.
7. **Unknowns** — things the paper does not specify, and the choice we made.
   This section is mandatory and is almost never empty. Forcing an agent to
   write "the paper does not state the encoder width; we chose 200 to match the
   reference implementation" is the difference between a documented assumption
   and a silent one.

### 1.1 Card status

Each card carries a status, and it is the recipe's real state:

| Status | Meaning |
|---|---|
| `draft` | card written, not yet reviewed |
| `reviewed` | a human has approved the card; implementation may begin |
| `implemented` | code exists, Tier 0 passes |
| `smoke-passing` | Tier 1 passes |
| `reproduced` | Tier 2 has run and matched the §6 target |
| `deviating` | Tier 2 has run and did *not* match; §5 explains why |

**A recipe is not done at `implemented`.** It is done at `reproduced` or at
`deviating` with a written explanation. A recipe that has never had Tier 2 run
against it is explicitly unvalidated, and the registry says so at import time.

### 1.2 Cards that bite instead of rot

Documentation rots because nothing checks it. Two mechanisms stop that:

- **Checklist/recipe cross-check (CI, Tier 0).** Section 4 is YAML with stable
  keys. A test parses every card and asserts that the recipe passes an explicit
  value for every key the card names. A hyperparameter the paper governs may not
  fall through to a framework default — if the card says the EMA decay is 0.999,
  the recipe must set it, not inherit it.
- **Plan diff (review).** `compile(recipe)` prints an execution plan
  (`DESIGN.md` §8). A reviewer diffs that plan against the card's §3 mapping
  table and §4 checklist. Objectives present in one and not the other are
  immediately visible.

---

## 2. The mechanics checklist

This is the list of things that get missed. It is derived from the failure mode
you actually hit, not from a general software checklist. Every card must answer
every applicable key, with a paper citation, or mark it `n/a`.

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
  reduction:             # mean over eligible rows | sum | population-weighted
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
  weight_decay:          # and whether it applies to biases / norm params
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

`labelled_unlabelled_ratio`, `ema_applies_to_buffers`, `reduction` and
`standardisation` are called out because each one produces a plausible-looking
model with wrong numbers, and none of them is visible in a diff.

---

## 3. Three test tiers

### Tier 0 — invariants (every PR, seconds)

Cheap, deterministic, and they localise breakage to a single component. These
are properties of the *framework*, not of any paper, so they are written once
and apply to everything.

- **Exact marginalisation** equals a brute-force loop over `k` in `range(K)`, to
  float tolerance.
- **`log_prob` broadcast contract**: `log_prob` is elementwise in `t`, so for
  any candidate matrix `M: [B, K]`, `log_prob(y, M)[:, k] == log_prob(y, M[:, k])`.
  Build `M` from **candidate** treatments — `arange(K)[None, :].expand(B, K)` —
  so column `k` is treatment `k` and the assertion becomes
  `log_prob(y, M)[:, k] == log_prob(y, full(B, k))`, which is exactly the call
  `MissingTreatmentMarginalNLL` makes.
  Do *not* build `M` from the observed `t` (`t[:, None].expand(B, K)`): that
  repeats `t_i` across every column, so the assertion would demand
  treatment-insensitivity and fail every correct head.
- **Normalisation**: `T_GIVEN_X.probs` rows sum to 1; `log_prob` agrees with
  `log(probs)`.
- **Trainable isolation**: after `backward()` in a stage, parameters outside
  `stage.trainable` have zero or absent gradients.
- **Teacher isolation**: EMA teacher parameters have `requires_grad=False` and
  receive no gradient.
- **Zero-eligible rows**: an objective given no eligible rows returns
  `value=0.0, n=0`, and no `NaN` reaches the total.
- **View preservation**: a view declaring `preserves={"t","y"}` leaves `t` and
  `y` bit-identical; `mutable=False` columns are untouched; `bounds` hold.
- **Determinism**: same seed → identical loss trace to float tolerance.
- **Port shape contracts**: every component's output matches its `PortSpec`.
- **Compile-time rejections**: missing port, unknown trainable name, unsatisfied
  realisation, derived-column violation, and the §7.2 leakage rule each raise.
- **Card cross-check**: §1.2 above.

### Tier 1 — smoke fit on a synthetic DGP (every PR, ~1 min per recipe)

A tiny data-generating process we write, so the true `p(t|x)`, `p(y|x,t)` and
CATE are analytic. **This is a wiring test, not a fidelity claim.** It answers
"is this recipe connected to the data at all", which Tier 0 cannot and Tier 2 is
too slow to.

Assertions are coarse and directional, chosen to be robust to seed noise:

- training loss decreases over the run;
- the propensity head beats the marginal-frequency baseline on held-out
  log-loss;
- estimated ATE lies within a wide band around the analytic ATE;
- with 50% of `t` missing at random, the recipe using exact marginalisation
  beats the complete-case baseline of the same recipe.

The last one is the load-bearing assertion: it fails if the marginalisation term
is scheduled to zero, is masked to an empty row set, or is detached — the exact
class of silent death that motivated the mixer logging in `DESIGN.md` §6.1.

### Tier 2 — published-number reproduction (nightly, per recipe)

The claim that actually matters. The dataset is **chosen per recipe** in card
§6 — whichever benchmark the source paper reported on — rather than a fixed
repo-wide suite, so we compare against a number that was actually published for
that method.

```yaml
reproduction:
  dataset: IHDP
  variant: "1000 realisations, Hill (2011) setting A"
  split: "author's train/test split, 63/27/10"
  metric: sqrt_PEHE_in_sample
  published: 0.88
  published_source: "Shalit et al. 2017, Table 1"
  tolerance: 0.10          # absolute, on the metric
  seeds: 10
  report: mean_and_stderr
```

Rules that keep this honest:

- **Tolerance is declared before the run**, in the card, at review time. A
  tolerance widened after seeing the result is a deviation and goes in §5.
- **Report the standard error.** A single-seed match on a high-variance
  benchmark is not evidence. If the published number is within our error bars
  *and* our error bars are wide enough to contain anything, say so — that is a
  `deviating` outcome with an explanation, not a `reproduced` one.
- **Record the run in the card.** Date, commit, metric, stderr. The card is the
  ledger; the CI artifact is the evidence.
- **A number that moves outside tolerance is a regression**, and the nightly job
  names the offending recipe, not just "benchmarks failed".

Tier 2's known weakness — it does not tell you *which* of data, objective,
architecture or schedule broke — is exactly what Tiers 0 and 1 and the component
graph exist to answer. When a number drifts, the first step is not to reread the
paper; it is to check which invariant or smoke assertion also moved.

---

## 4. The implementation loop for agents

This is the procedure. It is short deliberately.

1. **Read the card.** If `docs/recipes/<name>.md` does not exist, write it and
   **stop for review**. Do not write code in the same pass.
2. **Implement only what the card's §3 mapping table names.** Every component,
   objective and view you create must appear there.
3. **If you need something the card does not name, stop and amend the card.**
   Adding an unlisted objective, a helper `if` branch in a recipe, or a
   hyperparameter the card does not mention is out of scope by definition. The
   correct action is a card amendment and a second review, not a larger diff.
4. **Run Tier 0 and Tier 1.** Both must pass before the PR opens.
5. **Print the execution plan** and paste it into the PR body, so the reviewer
   can diff it against the card.
6. **Record the Tier 2 result** in the card when the nightly run completes, and
   set the status. Do not set `reproduced` from a Tier 1 pass.

### 4.1 Standing rules

- **No silent defaults for paper-governed hyperparameters.** If card §4 names
  it, the recipe sets it explicitly. Framework defaults exist only for things
  the paper does not govern.
- **No logic in recipes.** See `DESIGN.md` §9. A recipe that needs a conditional
  is telling you a component or objective is missing.
- **Deviations are written down before they are implemented**, not discovered in
  review.
- **"It trains and the loss goes down" is not evidence of anything.** Tier 1
  exists precisely so that this statement stops being used as one.
- **Do not port a model nobody has asked for.** Migration is lazy
  (`DESIGN.md` §11). Breadth is how the previous codebase acquired forty
  families and no trustworthy numbers.
