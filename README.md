# xty2

Composable semi-supervised causal learning for tabular data.

xty2 separates representation, parameterisation, objectives, data views, and
training order. Named methods are declarative recipes assembled from those
parts, so a result can be traced to a specific mechanic rather than a monolithic
model class.

Each recipe also has a reviewed spec card. The card records the source method,
equations, executable hyperparameters, deviations, unknowns, and a predeclared
reproduction target. Fast invariants and synthetic smoke fits run on every PR;
published-number benchmarks run nightly.

## Documentation

Start at [`docs/README.md`](docs/README.md). It explains which document is
authoritative and which one to read for a given task. Recipe cards are indexed
at [`docs/RECIPES.md`](docs/RECIPES.md).

## Development

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -e ".[dev]"

uv run pytest tests/invariants tests/smoke
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict
```

Tier 0 is `tests/invariants/`, Tier 1 is `tests/smoke/`, and Tier 2 is
`tests/benchmarks/`. Directory-based markers are applied automatically.
