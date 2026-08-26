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
5. **Deviations** — a table of everything we do differently, why, and which of
   the two kinds it is: a `judgement` we would make again given an infinite
   framework, or a `framework-limitation` we would not (§5). *"None" is a valid
   answer but must be written explicitly.* An empty section is not the same as
   a section asserting there are no deviations.
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
against it is explicitly unvalidated, and its card's status line is where that
is recorded. It is *not* announced at import time: `DESIGN.md` §9's recipe
registry is not built — recipes are plain functions — so this sentence used to
describe a warning that does not exist. Tier 0 is what enforces status claims
in the meantime (`tests/invariants/test_deviation_debt.py` for the one about
open framework limitations).

### 1.2 Cards that bite instead of rot

Documentation rots because nothing checks it. Two mechanisms stop that:

- **Checklist/recipe cross-check (CI, Tier 0).** Section 4 is YAML drawn from a
  **closed** key vocabulary (§2), and each paper-governed field binds to one of
  those keys and is declared `REQUIRED`, so it cannot fall through to a
  framework default. `compile()` emits `plan.hyperparameters` as a flat
  `{canonical_key: value}` dict, and the test asserts every card key not marked
  `n/a` is present in it with a non-null value. The mechanics are in
  `DESIGN.md` §9.1.
  This checks **presence, not correctness** — it proves the recipe sets
  `teacher.ema_decay`, never that 0.999 is the paper's number. Card review is
  the only thing that establishes the latter, and a green cross-check must not
  be read as fidelity.
- **Debt reconciliation (CI, Tier 0).** Section 5 rows marked
  `framework-limitation` name a ledger key in `DESIGN.md` §11.4, and each
  ledger row names the cards citing it. The test reads both and asserts they
  agree. Its purpose is what happens on *discharge*: deleting a ledger row
  turns CI red on every card still naming it, so the capability's own PR is
  what forces the earlier cards to be revisited. §5 has the details, and
  `DESIGN.md` §11.3 has the argument for why this is a test rather than a
  convention.
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
- **Candidate-treatment contract** (`DESIGN.md` §3.1). `y` is passed
  **unexpanded** in both calls; the head inserts the candidate axis. With
  `M = arange(K)[None, :].expand(B, K)`:

  ```python
  out = head.log_prob(y, M)                 # [B, K]
  assert out.shape == (B, K)
  for k in range(K):
      assert allclose(out[:, k], head.log_prob(y, full((B,), k)))   # [B]
  ```

  Run it with `B != K` — a test where `B == K` passes under accidental
  broadcasting and proves nothing. Two ways to get this wrong, both of which
  were written into earlier drafts of this document and are worth stating so
  they are not rediscovered:
  - building `M` from the observed `t` (`t[:, None].expand(B, K)`) repeats
    `t_i` across every column, so the assertion demands treatment-insensitivity
    and fails every correct head;
  - expanding `y` at the call site and relying on ambient broadcasting —
    `y: [B]` against `t: [B, K]` aligns `B` against `K` and raises for ordinary
    batch sizes. Expansion is the head's job, per the contract.
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
- **Batch immutability**: clone every input batch, run each registered
  transform, assert the original is bit-identical afterwards. `frozen=True` does
  not cover in-place writes to tensor leaves (`DESIGN.md` §1.1).
- **Row-scope composition**: `Stage.rows ∩ Objective.rows` is computed as
  specified in `DESIGN.md` §7.0, and a pairing empty by construction is a
  compile error rather than a permanent `n = 0`.
- **Fold disjointness**: for an `out_of_fold` artifact, every predicted row is
  absent from `trained_on_row_ids` of the checkpoint that produced it
  (`DESIGN.md` §7.1). This is run against real artifacts, not mocked
  provenance — the point is that the label is checked, not trusted.
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
class of silent death that motivated the mixer logging in `DESIGN.md` §6.2.

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
7. **If your change discharges a `DESIGN.md` §11.4 ledger entry**, Tier 0 will
   name the cards that were paying for it. Revisit each one in the same PR:
   withdraw the deviation or restate it as a `judgement` (§5.2). This is part
   of the packet, not follow-up work.

### 4.1 Standing rules

- **No silent defaults for paper-governed hyperparameters.** If card §4 names
  it, the recipe sets it explicitly. Framework defaults exist only for things
  the paper does not govern.
- **No logic in recipes.** See `DESIGN.md` §9. A recipe that needs a conditional
  is telling you a component or objective is missing.
- **Deviations are written down before they are implemented**, not discovered in
  review — and each one is typed `judgement` or `framework-limitation` (§5).
  Typing it is not bookkeeping: writing `framework-limitation` is the moment
  you notice you are shipping something the paper does not say, and it is the
  moment to ask `DESIGN.md` §11.2's two questions instead.
- **If a missing abstraction is what stands between the recipe and its paper,
  the abstraction is in scope.** §11.2 Q1 exists so that "the framework cannot
  express it" stops being a free answer. Amend the card, say what you are
  adding and why, get the amendment reviewed — the same gate as any other card
  change — and build it. Reaching for a §5 row instead, because the diff stays
  smaller that way, is the failure this project is about.
- **"It trains and the loss goes down" is not evidence of anything.** Tier 1
  exists precisely so that this statement stops being used as one.
- **Do not port a model nobody has asked for.** Migration is lazy
  (`DESIGN.md` §11). Breadth is how the previous codebase acquired forty
  families and no trustworthy numbers.

---

## 5. Deviations, and the debt they create

Section 5 of a card is the only place a reader learns that an implementation
and its paper differ. It is therefore the place where a wrong implementation is
most likely to look finished, and until now it rendered two very different
statements in the same typeface:

| Kind | Means | Is it permanent? |
|---|---|---|
| `judgement` | We chose differently on purpose, and would choose the same again if the framework could express either. Tabular adaptations of an image method, a held-fixed architecture, a project-local step budget | Yes. It is a modelling decision and it is finished |
| `framework-limitation` | We would have implemented the paper. xty2 could not express it, so the recipe ships without it | **No.** It is provisional, and it is a debt |

Collapsing the two is how fidelity debt becomes invisible: a
`framework-limitation` written in the register of a design decision reads, to
the next reviewer, as a decision that has already been made and reviewed. It
has not been. Nobody chose it; the framework did.

### 5.1 The form

Card §5's table carries both fields, and `docs/recipes/_TEMPLATE.md` is
authoritative:

| # | Kind | Blocked on | What we do differently | Why | Expected effect on §6 |
|---|---|---|---|---|---|
| 4 | `framework-limitation` | `loader` | `mu = 7` is not enforced | … | … |
| 6 | `judgement` | — | Retain the P5 architecture rather than a Wide ResNet | … | … |

`Kind` is one of three: `judgement`, `framework-limitation`, or `withdrawn` —
a row that was one of the first two and has since been implemented, kept with
its history rather than deleted, as `fixmatch.md` §5 deviations 5 and 8 are.

`Blocked on` cites a key from the `DESIGN.md` §11.4 ledger, and is empty for a
`judgement` or a `withdrawn`. If a `framework-limitation` has no ledger row to
cite, the ledger is missing a row — write it, with the evidence that would
change the decision, in the same pass. A deviation blocked on nothing is either
a judgement in disguise or a capability nobody has costed.

**A framework limitation is never a §7 basis.** §7 is for what the paper does
not specify. If the reason we chose X is that xty2 could not do Y, that is a
deviation, and writing it as the *basis* for an unknown is the most comfortable
place in the card to hide one — it reads as a decision with a rationale rather
than as something missing. Both debts this rule turned up in the existing cards
(`tarnet` §5.5, `ssdml` §5.6) were living in §7 exactly like that.

Cards also carry a **§5.1** listing framework additions made *for that card*
under `DESIGN.md` §11.2 — what was added, its consumers today, and for the
load-bearing quadrant the named second consumer the shape was designed against.
`fixmatch.md` §5.1 is the worked example.

### 5.2 What the Tier 0 reconciliation asserts

1. Every row declares one of the three kinds.
2. Every `framework-limitation` row cites a key that exists in `DESIGN.md`
   §11.4; every `judgement` and `withdrawn` row cites nothing.
3. Every §11.4 row's **Who is paying** lists exactly the card rows that cite its
   key, in the checkable form `` `<card>` §5.<n> `` and separated by `;` —
   neither direction may drift.
4. A `reproduced` card carrying an open `framework-limitation` names that row
   in its §6. `reproduced` there is a claim that the omitted mechanic did not
   matter for the published number; it may well be true, but it is a claim, and
   it belongs in writing beside the result rather than in the gap between two
   sections.

What none of them checks is whether a row typed `judgement` deserves it. A
framework limitation someone found it easier to call a choice passes every
assertion above. That is card review's job — §5's kinds exist to make it a
question a reviewer can ask, not to answer it.

(1) is the one that does the work. Discharging a ledger entry means deleting
its row, and deleting the row breaks every card still citing it. The PR that
builds the loader cannot go green until each card that was paying for its
absence has been revisited, and revisiting means exactly one of two edits:

- **Withdraw** the deviation — implement the mechanic and strike the row
  through, as `fixmatch.md` §5 deviation 5 does; or
- **Restate** it as a `judgement`, with the reason it survives the capability
  that was supposed to end it.

"Leave it and open an issue" is not one of the two, and that is the entire
design intent. The agent who builds the capability is the only one guaranteed
to have the context for this, and is holding it at the cheapest possible
moment. `DESIGN.md` §11.3 records what happened the one time we relied on a
later agent instead.

### 5.3 A note on scope

Repaying the debt is not licence to widen the PR. Withdrawing a deviation on
someone else's card is a card amendment: it changes a reviewed artifact, so it
is reviewed, and if the recipe has a Tier 2 ledger row that row is now measured
against a different recipe and must be re-run or explicitly invalidated. Where
that cost is real, **restating as a `judgement` with the reason is the correct
outcome**, not a cop-out. What is refused is silence — the card that still
reads as though the capability had never been built.
