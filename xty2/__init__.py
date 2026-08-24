"""xty2 — composable semi-supervised causal learning for tabular data.

`xty2.core` holds the data types the rest of the framework is stated against —
the batch, the schema, the ports and their shape contracts, the distribution
protocols and the row populations (P1) — together with the component graph, the
declarative recipe surface and the compiler that turns one into a checked,
printable execution plan (P2), and the objective, weighting and descent
contracts those read (P3, P4).

`xty2.objectives` holds four of the ten losses of `DESIGN.md` §4.2, and
`xty2.training` the mixer that weights and logs them (P3) together with the
`gradient` executor and immutable artifacts (P4), then the ordered program
runner and EMA teacher parameter set (P8).
`xty2.components` and `xty2.recipes` contain five reviewed methods: TARNet's
fixed-scale treatment heads (P5), CNFlow's conditional spline outcome density
(P7), Mean Teacher's EMA propensity consistency (P9), cycle-dual's staged
outcome-dependent posterior, and SSDML's deterministic array ATE action (P11).
They compose the exact missing-treatment marginal objective, schema-aware
views, explicit gradient/cross-fit/array-fit executors, and immutable staged
artifacts. Layout is in `docs/DESIGN.md` §10 and `docs/PLAN.md`.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - only hit when running from an unbuilt source tree
    __version__ = version("xty2")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
