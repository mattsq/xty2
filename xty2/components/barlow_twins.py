"""Barlow Twins' projector, as the author's `BarlowTwins.__init__` builds it.

Pinned to `facebookresearch/barlowtwins@8e8d284`, `main.py`, where the stack is

```python
for i in range(len(sizes) - 2):
    layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
    layers.append(nn.BatchNorm1d(sizes[i + 1]))
    layers.append(nn.ReLU(inplace=True))
layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
```

This is deliberately not `VICRegExpander` with a wider `widths`, and the reason
is one line of the loop above. The author's *hidden* linears carry `bias=False`
as well as the output one, where VICReg's `Projector` writes a plain
`nn.Linear(f[i], f[i + 1])` and keeps the hidden biases. A bias immediately in
front of a `BatchNorm1d` is a constant the normalisation removes, so the two
stacks compute closely related functions — but they do not consume the same
construction-time RNG, because `reset_parameters` draws a bias vector for each
hidden layer VICReg has and this one does not. `vicreg`'s recorded §6 numbers
are the numbers of the tensors that draw produced (`docs/recipes/vicreg.md`
§6.1), so widening `VICRegExpander` with a bias switch would move them. A
separate component is the same argument `vicreg.py` makes for not widening
`ProjectionHead`, one recipe further along.

The author's fourth `BatchNorm1d(sizes[-1], affine=False)` is **not** part of
this component. It normalises the embedding inside the loss rather than
producing a representation, and `docs/recipes/barlow_twins.md` deviation 6
reproduces its training-mode arithmetic statelessly inside the two objectives
instead — the running buffers it would otherwise accumulate have no inference
consumer here, and the diagnostics of §6.4 recompute the statistic per held-out
batch on purpose.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from torch import nn

from xty2.components._nn import validate_dimension, validate_dropout, validate_widths
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue

BARLOW_TWINS_PROJECTOR_ACTIVATION = "hidden relu; output linear"
"""`architecture.activation`: `nn.ReLU(inplace=True)` after each hidden block,
and nothing after the output layer (author `BarlowTwins.__init__`)."""

BARLOW_TWINS_PROJECTOR_NORMALISATION = (
    "hidden BatchNorm1d(eps=1e-5, momentum=0.1, affine=true, "
    "track_running_stats=true); output none"
)
"""`architecture.normalisation`: the author code writes `nn.BatchNorm1d(sizes[i+1])`
and takes torch's defaults, which this string makes explicit rather than
inherited. "output none" is deviation 6: the author's `self.bn` is loss
arithmetic, not a layer of the projector (`docs/recipes/barlow_twins.md` §7)."""

BARLOW_TWINS_PROJECTOR_INITIALISATION = (
    "torch Linear reset_parameters; all linear bias=false; "
    "BN weight=1,bias=0,running_mean=0,running_var=1"
)
"""`architecture.initialisation`: the author code initialises nothing itself, so
this names the library defaults it therefore uses. `all linear bias=false` is
the half that separates this string from `VICREG_EXPANDER_INITIALISATION`, and
it is the author's own choice rather than a torch default."""

BATCHNORM_EPS = 1e-5
BATCHNORM_MOMENTUM = 0.1


class BarlowTwinsProjector(Component):
    """The projector: an MLP from `X_REPR` to `X_PROJ`, bias-free throughout.

    Attributes:
        widths: The projector layer widths, `(512, 512, 512)` for the card's
            tabular adaptation (deviation 2). The last entry is `d`, the
            dimension the cross-correlation matrix is square in.
        activation: Bound to `architecture.activation`; the one supported value
            is `BARLOW_TWINS_PROJECTOR_ACTIVATION`.
        normalisation: Bound to `architecture.normalisation`; the one supported
            value is `BARLOW_TWINS_PROJECTOR_NORMALISATION`.
        dropout: Bound to `architecture.dropout`. The author code has none, so
            the recipe passes `0.0` explicitly rather than defaulting to it.
        initialisation: Bound to `architecture.initialisation`; the one
            supported value is `BARLOW_TWINS_PROJECTOR_INITIALISATION`.

    Four single-valued fields are four card keys the plan carries, not four
    settings: `FIDELITY.md` §1.2 compares card §4 against `plan.hyperparameters`
    key by key, and a value the component never states is a key the cross-check
    cannot see.
    """

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "widths": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
    }

    def __init__(
        self,
        name: str = "barlow_twins_projector",
        *,
        representation_dim: int,
        widths: tuple[int, ...] = REQUIRED,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
    ) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={Port.X_PROJ})
        self.widths = widths
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        card_hyperparameters(self)

        owner = type(self).__name__
        representation_dim = validate_dimension(
            representation_dim, field="representation_dim", owner=owner
        )
        self.widths = validate_widths(self.widths, owner=owner)
        self.dropout = validate_dropout(self.dropout, owner=owner)
        if self.dropout:
            raise GraphError(
                f"{owner}.dropout must be 0.0: the author's projector has no "
                f"dropout, and a dropped unit is a zero the cross-correlation "
                f"would normalise and then charge as a correlated coordinate. "
                f"Got {self.dropout!r}."
            )
        for field, value, supported in (
            ("activation", self.activation, BARLOW_TWINS_PROJECTOR_ACTIVATION),
            ("normalisation", self.normalisation, BARLOW_TWINS_PROJECTOR_NORMALISATION),
            (
                "initialisation",
                self.initialisation,
                BARLOW_TWINS_PROJECTOR_INITIALISATION,
            ),
        ):
            if value != supported:
                raise GraphError(
                    f"{owner}.{field} supports {supported!r}, got {value!r}. The "
                    "projector topology is pinned to the author's "
                    "`BarlowTwins.__init__` (docs/recipes/barlow_twins.md §3.2); "
                    "a variant is a different component and waits for a card "
                    "that states one."
                )
        self.network = _projector(representation_dim, self.widths)
        self.output_dim = self.widths[-1]

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_PROJ: self.network(ports.tensor(Port.X_REPR))}


def _projector(input_dim: int, widths: tuple[int, ...]) -> nn.Sequential:
    """`[Linear(bias=False) -> BN -> ReLU] * (len(widths) - 1)`, then `Linear`.

    Transcribed from the author's `BarlowTwins.__init__`, which builds its
    blocks over `range(len(sizes) - 2)` on `sizes = [embedding, *projector]` and
    appends `nn.Linear(sizes[-2], sizes[-1], bias=False)`. Every linear in the
    stack is bias-free, the output one included. A single-width projector is
    therefore one bias-free affine map and no normalisation at all, which is
    what that loop degenerates to and is left reachable rather than
    special-cased.
    """
    layers: list[nn.Module] = []
    current = input_dim
    for width in widths[:-1]:
        layers.append(nn.Linear(current, width, bias=False))
        layers.append(
            nn.BatchNorm1d(
                width,
                eps=BATCHNORM_EPS,
                momentum=BATCHNORM_MOMENTUM,
                affine=True,
                track_running_stats=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))
        current = width
    layers.append(nn.Linear(current, widths[-1], bias=False))
    return nn.Sequential(*layers)


__all__ = [
    "BARLOW_TWINS_PROJECTOR_ACTIVATION",
    "BARLOW_TWINS_PROJECTOR_INITIALISATION",
    "BARLOW_TWINS_PROJECTOR_NORMALISATION",
    "BATCHNORM_EPS",
    "BATCHNORM_MOMENTUM",
    "BarlowTwinsProjector",
]
