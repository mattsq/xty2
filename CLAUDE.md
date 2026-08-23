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
xty2/core/        batch, schema, ports, distributions, graph, recipe, compile
xty2/components/  parameterisations only
xty2/views/       augmentation, separated from the losses that use it
xty2/objectives/  losses as independent objects
xty2/training/    stage, program, loss mixer, schedules, executors, artifacts
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
  discovered in review. "None" is a valid answer, written explicitly.
- **"It trains and the loss goes down" is not evidence.** Tier 1 exists so that
  sentence stops being used as one; only a Tier 2 result sets `reproduced`.
- **Two-consumer rule.** No new abstraction — port, executor, schedule type,
  realisation axis — until a second real recipe needs it. Deliberate omissions
  are in the YAGNI ledger (`docs/DESIGN.md` §11); add to it rather than building
  ahead.
- **Do not port a model nobody has asked for.** Migration is lazy.
- **Stay inside the packet.** `docs/PLAN.md` gives every packet an explicit
  out-of-scope list. Work that belongs to a later packet waits for it.
