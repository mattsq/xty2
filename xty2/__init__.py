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
`xty2.components` and `xty2.recipes` hold the first recipe and the three
parameterisations it composes — `mlp_encoder`, `tarnet_head` and
`categorical_propensity` (P5).

`xty2.views`, `xty2.evaluation` and `xty2.estimators` are still empty, and
that is the plan's shape rather than an accident — the framework is built only
to the depth the next recipe demands. Layout, and the packet that fills each
subpackage, is in `docs/DESIGN.md` §10 and `docs/PLAN.md`.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - only hit when running from an unbuilt source tree
    __version__ = version("xty2")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
