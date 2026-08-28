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
   `framework-limitation`, or `withdrawn`.
3. **No logic in recipes.** Recipes assemble declared components, objectives,
   views, data policy, and stages. A conditional belongs in a reusable object,
   not `xty2/recipes/`.
4. **Make the review surface executable.** Before opening a PR, run Tier 0 and
   Tier 1, lint, format, and type checks. Paste `compile(recipe).plan.render()`
   into the PR body and compare it with card §3–§4.

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

## Commands

```bash
uv venv && uv pip install -e ".[dev]"
uv run pytest tests/invariants tests/smoke
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict
```

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

Prefer native read/edit tools for targeted changes. Use a mechanical script
only for a genuinely repeated transformation, and verify its diff.
