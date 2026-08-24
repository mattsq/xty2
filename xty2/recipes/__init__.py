"""Named methods, assembled from registered pieces (`DESIGN.md` §9).

A recipe is a declarative assembly of components, objectives and views plus
explicit hyperparameters, and it contains no logic: a recipe that needs an
`if` is telling you a component or an objective is missing.

One exists so far — `tarnet` (P5). The other four arrive with their packets.

There is no registry yet, and its absence is deliberate. `DESIGN.md` §9 shows
`xty2.create("cycle_dual", ...)` resolving a name to a recipe, but a registry
with one entry is a lookup table nothing looks anything up in, and the
compiler's "resolve every registry string once" step has no string to resolve.
It arrives when a second recipe makes the name-to-object step real
(two-consumer rule, §11).
"""

from xty2.recipes.tarnet import tarnet

__all__ = ["tarnet"]
