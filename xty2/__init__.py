"""xty2 — composable semi-supervised causal learning for tabular data.

`xty2.core` holds the data types the rest of the framework is stated against
(P1): the batch, the schema, the ports and their shape contracts, the
distribution protocols and the row populations. Everything else — components,
objectives, views, training — is still empty. Layout, and the packet that fills
each subpackage, is in `docs/DESIGN.md` §10 and `docs/PLAN.md`.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - only hit when running from an unbuilt source tree
    __version__ = version("xty2")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
