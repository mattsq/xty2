"""xty2 — composable semi-supervised causal learning for tabular data.

`xty2.core` holds the data types the rest of the framework is stated against —
the batch, the schema, the ports and their shape contracts, the distribution
protocols and the row populations (P1) — together with the component graph, the
declarative recipe surface and the compiler that turns one into a checked,
printable execution plan (P2), and the objective, weighting and descent
contracts those read (P3, P4).

`xty2.objectives` holds three of the ten losses of `DESIGN.md` §4.2, and
`xty2.training` the mixer that weights and logs them (P3) together with the
single-stage `gradient` executor and the immutable artifacts it writes (P4).
`xty2.components` and `xty2.recipes` now contain the first reviewed method:
TARNet's shared MLP, treatment-specific outcome heads, categorical propensity,
and one joint stage (P5). Views and the later recipes remain deliberately
absent until their packets. Layout is in `docs/DESIGN.md` §10 and
`docs/PLAN.md`.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - only hit when running from an unbuilt source tree
    __version__ = version("xty2")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
