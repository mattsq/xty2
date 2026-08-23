"""xty2 — composable semi-supervised causal learning for tabular data.

`xty2.core` holds the data types the rest of the framework is stated against —
the batch, the schema, the ports and their shape contracts, the distribution
protocols and the row populations (P1) — together with the component graph, the
declarative recipe surface and the compiler that turns one into a checked,
printable execution plan (P2).

The subpackages around it are still empty: there is no component, objective,
view, executor or recipe yet. That is the plan's shape rather than an accident
— the framework is built only to the depth the next recipe demands. Layout, and
the packet that fills each subpackage, is in `docs/DESIGN.md` §10 and
`docs/PLAN.md`.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - only hit when running from an unbuilt source tree
    __version__ = version("xty2")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
