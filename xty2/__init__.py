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
`xty2.components` and `xty2.recipes` contain two reviewed methods: TARNet's
fixed-scale treatment heads (P5) and CNFlow's conditional spline outcome
density (P7). They share the categorical propensity, exact missing-treatment
marginal objective and one-stage executor. `xty2.views` adds schema-aware
masking and jitter, view-keyed realisations, and the consistency objective
(P6). Later recipes remain deliberately absent until their packets. Layout is
in `docs/DESIGN.md` §10 and `docs/PLAN.md`.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - only hit when running from an unbuilt source tree
    __version__ = version("xty2")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
