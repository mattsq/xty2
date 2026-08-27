# Working in xty2

Read `docs/DESIGN.md` (architecture), `docs/FIDELITY.md` (how correctness is
established) and `docs/PLAN.md` (what packet you are in) before changing code.
This file is the short version of the rules those documents make binding.

## The four rules

1. **Read the card first.** A recipe's spec card is `docs/recipes/<name>.md`.
   If it does not exist, write it from `docs/recipes/_TEMPLATE.md` and **stop
   for review** — do not write code in the same pass. If you need something the
   card does not name, amend the card and get it re-reviewed; do not widen the
   diff instead.
2. **Tier 0 and Tier 1 must pass before a PR opens.**
   `pytest tests/invariants tests/smoke`, plus `ruff check .`,
   `ruff format --check .` and `mypy --strict`.
3. **No logic in recipes.** A recipe is a declarative assembly of registered
   components, objectives and views plus explicit hyperparameters. A recipe that
   needs an `if` is telling you a component or objective is missing
   (`docs/DESIGN.md` §9).
4. **Paste the execution plan into the PR body.** `compile(recipe)` prints it;
   the reviewer diffs it against the card's §3 mapping table and §4 checklist.
   That diff is the review.

## Layout

```
xty2/core/        batch, schema, ports, distributions, graph, views, loss,
                  schedules, optimisation, data, recipe, compile
xty2/components/  parameterisations only
xty2/views/       augmentation, separated from the losses that use it
xty2/objectives/  losses as independent objects
xty2/training/    loss mixer, executors, artifacts, loading (data policy -> batches)
xty2/recipes/     named methods, no logic
xty2/evaluation/  predictive, causal, calibration metrics
xty2/estimators/  cate, dml, policy front-ends
docs/recipes/     one spec card per recipe
tests/invariants/ Tier 0 — framework invariants, seconds, every PR
tests/smoke/      Tier 1 — synthetic-DGP wiring fits, every PR
tests/benchmarks/ Tier 2 — published-number reproduction, nightly
```

Tier markers (`tier0`/`tier1`/`tier2`) are applied automatically by directory;
put a test in the directory for its tier and do not mark it by hand.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"   # torch first from the CPU wheel index in CI
uv run pytest tests/invariants tests/smoke   # what CI runs on a PR
uv run pytest -m tier0                       # same set, selected by marker
uv run ruff check . && uv run ruff format .
uv run mypy --strict
```

## Standing rules

- **No silent defaults for paper-governed hyperparameters.** If card §4 names a
  key, the recipe sets it explicitly; the field carries a `REQUIRED` sentinel so
  it cannot fall through to a framework default (`docs/DESIGN.md` §9.1).
- **Deviations are written into card §5 before they are implemented**, not
  discovered in review. "None" is a valid answer, written explicitly. Reaching
  for a §5 row because it keeps the diff smaller than the missing abstraction
  would is the failure mode this project exists to prevent.
- **"It trains and the loss goes down" is not evidence.** Tier 1 exists so that
  sentence stops being used as one; only a Tier 2 result sets `reproduced`.
- **Build for one consumer, design against two** (`docs/DESIGN.md` §11.2). Ask
  two questions of any new abstraction. *Does its absence force a deviation from
  a paper-governed mechanic named in a card §4?* If yes, one consumer is enough
  — build it. *If the shape is wrong, what does it cost?* Reversible (opt-in,
  default-preserving, changes no existing plan, digest or number) means build it
  now; load-bearing vocabulary — a port, an executor contract, a row population,
  an artifact kind — means build it now but design the shape against a **named**
  second consumer, written into card §5.1 first. An abstraction that fails the
  first question is convenience, and convenience still waits for a second real
  recipe. Deliberate omissions go in the ledger (`docs/DESIGN.md` §11.4).
- **A framework limitation is a debt, not a decision.** Card §5 rows are typed
  `judgement`, `framework-limitation` or `withdrawn`, and only the second cites
  the ledger key blocking it. A framework limit is never a §7 basis. If your change discharges a ledger entry, Tier 0 names the
  cards that were paying for it and you revisit each in the same PR — withdraw
  the deviation or restate it as a `judgement` with the reason it survives
  (`docs/FIDELITY.md` §5). Not follow-up work; part of the packet.
- **Do not port a model nobody has asked for.** Migration is lazy.
- **Stay inside the packet.** `docs/PLAN.md` gives every packet an explicit
  out-of-scope list. Work that belongs to a later packet waits for it.
