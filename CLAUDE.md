# Working in xty2

Start with [`docs/README.md`](docs/README.md). It routes tasks to the smallest
useful document set. Do not preload `DESIGN.md`, `FIDELITY.md`, `PLAN.md`, and
`BACKLOG.md` for every change.

## Hard rules

1. **Card before code.** A recipe's contract is
   `docs/recipes/<name>.md`. If the card does not exist, draft it from
   `docs/recipes/_TEMPLATE.md` and stop for review. If implementation needs an
   unlisted mechanic, amend the card and stop again.
2. **Reproduce before simplifying.** The paper and pinned reference code are
   authoritative. Record every departure in card §5 as `judgement`,
   `framework-limitation`, or `withdrawn`. A constant inherited from a sibling
   recipe is re-derived against this paper, fixture, and K before use; a
   departure from an inherited policy is its own §5 row.
3. **No logic in recipes.** Recipes assemble declared components, objectives,
   views, data policy, and stages. A conditional belongs in a reusable object,
   not `xty2/recipes/`.
4. **Make the review surface executable.** Run Tier 0, Tier 1, lint, format,
   and type checks before each commit, not once before the PR. Paste
   `compile(recipe).plan.render()` into the PR body and compare it with card
   §3–§4. Cite by equation, section, and algorithm line, and check the citation
   resolves to the value: a value right for a reason other than its citation is
   a §7 row, and line numbers in recipe comments are card content.

Treat a negative result as an implementation failure until the method has been
audited equation by equation: architecture, mixer, views, data policy,
hyperparameters, schedules, and inherited choices. Check more than one seed.
If a component may matter, implement it faithfully and ablate it later; do not
omit it by intuition.

Paper-governed hyperparameters have no silent defaults. A non-`n/a` card §4
key must reach `plan.hyperparameters`. A framework limitation names a live
`DESIGN.md` §11.4 ledger key. Discharging that key requires revisiting every
paying card in the same PR.

Stay within the issue or PR scope. `docs/PLAN.md` records the completed P0–P12
build and is not a standing instruction to reopen old packets.

## Evidence

- **A test is not done until you have seen it fail.** Perturb what each new
  assertion guards — flip a `>=`, drop a term, collapse a cache key, swap two
  realisations — and name the mutants in the commit. Two shapes are vacuous by
  construction: a test whose two sides share a constant or a call path, and a
  fixture whose degenerate values (NaN, ±inf, empty) satisfy the assertion for
  free. Never pin a hash of torch RNG draws; the stream holds only within a
  release.
- **One seed is not a direction.** A direction enters a Tier 1 assertion only
  if it holds on every replicate seed the card declares; otherwise assert the
  mechanism and leave the direction to Tier 2. Set thresholds from the measured
  spread, not from the fixture seed. Compare arms paired — same seeds, draws,
  and initialisation — since overlapping unpaired error bars are not evidence
  of no effect.
- **Tier 2 lands whole.** The benchmark module, its `RECIPES` entry, and the
  recorded result belong in one commit, run from a committed tree; a module
  ahead of its result gives the nightly a job that can only fail. Bind every §6
  scalar from the card by value and never retype a seed stream — the Tier 1
  seed is not the Tier 2 stream. Leave the result ledger's placeholder row
  exactly as `_TEMPLATE.md` writes it, wherever the card numbers that section.
- **Import what a card adopts.** Where §6.1 takes a sibling's fixture or
  protocol unchanged, import that module. A second transcription is how two
  cards' numbers stop being about their recipes.

## Commands

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest tests/invariants tests/smoke
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict
```

Run these verbatim. A path argument to `mypy --strict` narrows it below the
`files` list in `pyproject.toml`, which is what CI runs.

Tier 2 lives in `tests/benchmarks/` and runs nightly. Test tiers are assigned by
directory; do not add markers manually.

## Layout

| Path | Owns |
|---|---|
| `xty2/core/` | contracts, graph, compiler, data, recipes, schedules |
| `xty2/components/` | parameterisations |
| `xty2/views/` | schema-aware transforms |
| `xty2/objectives/` | independent losses |
| `xty2/training/` | executors, mixing, artifacts, loading, teachers |
| `xty2/recipes/` | declarative named methods |
| `xty2/evaluation/` | metrics and Tier 2 runners |
| `docs/recipes/` | reviewed method cards |
| `tests/invariants/` | Tier 0 contracts |
| `tests/smoke/` | Tier 1 wiring fits |
| `tests/benchmarks/` | Tier 2 reproduction |

A new recipe touches `xty2/recipes/__init__.py`, `RECIPES`,
`benchmark_function`, `docs/RECIPES.md`, and its card. Conflicts in the first
three are additive: keep both entries, in the order the list declares.

Prefer native read/edit tools for targeted changes. Use a mechanical script
only for a genuinely repeated transformation, and verify its diff.
