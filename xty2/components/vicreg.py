"""VICReg's expander `h_phi`, as the author's `Projector` builds it.

`ProjectionHead` already maps `X_REPR -> X_PROJ`, and this is deliberately not
a widened version of it. The two differ in three places that the pinned
reference implementation makes load-bearing rather than stylistic
(`facebookresearch/vicreg@4e12602`, `main_vicreg.py`, symbol `Projector`):

* **Every hidden layer is followed by `BatchNorm1d` and then ReLU.** SCARF's
  head has no normalisation anywhere, and a VICReg expander without it is a
  different network: the variance term is computed on this component's output,
  so what the hidden batch statistics do to the scale of that output is part of
  the mechanism under study rather than a training convenience.
* **The output layer carries no bias.** `nn.Linear(f[-2], f[-1], bias=False)`
  in the author code. A bias there is a per-dimension constant, which the
  covariance and variance terms are both invariant to and the invariance term
  is not, so it would be a parameter only one of the three losses can move.
* **The output is not normalised.** There is no unit hypersphere in VICReg —
  `v` and `c` are statistics of an unconstrained embedding, and row-`l2` would
  bound `Var(z^j)` above and hand the variance hinge a ceiling the paper does
  not put there (`docs/recipes/vicreg.md` §5.1).

Widening `ProjectionHead` to cover all three would change what `X_PROJ` means
for `scarf`, whose recorded numbers depend on that component's construction-time
RNG consumption. A separate component leaves those alone, which is the same
argument `projection.py` makes for not sharing a base class with `MLPEncoder`.
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

VICREG_EXPANDER_ACTIVATION = "hidden relu; output linear"
"""`architecture.activation`: `nn.ReLU(True)` after each hidden block, and
nothing after the output layer (author `Projector`)."""

VICREG_EXPANDER_NORMALISATION = (
    "hidden BatchNorm1d(eps=1e-5, momentum=0.1, affine=true, "
    "track_running_stats=true); output none"
)
"""`architecture.normalisation`: the author code writes `nn.BatchNorm1d(f[i+1])`
and takes torch's defaults, which this string makes explicit rather than
inherited (`docs/recipes/vicreg.md` §7)."""

VICREG_EXPANDER_INITIALISATION = (
    "torch Linear reset_parameters; hidden bias=true; output bias=false; "
    "BN weight=1,bias=0,running_mean=0,running_var=1"
)
"""`architecture.initialisation`: the author code initialises nothing itself, so
this names the library defaults it therefore uses, and the bias placement that
is the author's own choice."""

BATCHNORM_EPS = 1e-5
BATCHNORM_MOMENTUM = 0.1


class VICRegExpander(Component):
    """`h_phi`: an MLP from `X_REPR` to `X_PROJ`, hidden BN/ReLU, affine output.

    Attributes:
        widths: The expander layer widths, `(512, 512, 512)` for the card's
            tabular adaptation. The last entry is `d`, the dimension `v` and
            `c` average over.
        activation: Bound to `architecture.activation`; the one supported value
            is `VICREG_EXPANDER_ACTIVATION`.
        normalisation: Bound to `architecture.normalisation`; the one supported
            value is `VICREG_EXPANDER_NORMALISATION`. There is deliberately no
            `row_l2` option here — see the module note.
        dropout: Bound to `architecture.dropout`. The author code has none, so
            the recipe passes `0.0` explicitly rather than defaulting to it.
        initialisation: Bound to `architecture.initialisation`; the one
            supported value is `VICREG_EXPANDER_INITIALISATION`.

    Three single-valued fields are three card keys the plan carries, not three
    settings. A field whose only accepted value is the author's is still worth
    declaring: `FIDELITY.md` §1.2 compares card §4 against `plan.hyperparameters`
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
        name: str = "vicreg_expander",
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
                f"{owner}.dropout must be 0.0: the author's Projector has no "
                f"dropout, and a dropped unit is a zero this component's "
                f"variance and covariance terms would read as an embedding "
                f"value. Got {self.dropout!r}."
            )
        for field, value, supported in (
            ("activation", self.activation, VICREG_EXPANDER_ACTIVATION),
            ("normalisation", self.normalisation, VICREG_EXPANDER_NORMALISATION),
            ("initialisation", self.initialisation, VICREG_EXPANDER_INITIALISATION),
        ):
            if value != supported:
                raise GraphError(
                    f"{owner}.{field} supports {supported!r}, got {value!r}. The "
                    "expander topology is pinned to the author's Projector "
                    "(docs/recipes/vicreg.md §3.2); a variant is a different "
                    "component and waits for a card that states one."
                )
        self.network = _expander(representation_dim, self.widths)
        self.output_dim = self.widths[-1]

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_PROJ: self.network(ports.tensor(Port.X_REPR))}


def _expander(input_dim: int, widths: tuple[int, ...]) -> nn.Sequential:
    """`[Linear -> BN -> ReLU] * (len(widths) - 1)`, then a bias-free `Linear`.

    Transcribed from the author's `Projector`, which builds its blocks over
    `range(len(f) - 2)` on `f = [embedding, *mlp_widths]` and appends
    `nn.Linear(f[-2], f[-1], bias=False)`. A single-width expander is therefore
    one bias-free affine map and no normalisation at all, which is what that
    loop degenerates to and is left reachable rather than special-cased.
    """
    layers: list[nn.Module] = []
    current = input_dim
    for width in widths[:-1]:
        layers.append(nn.Linear(current, width))
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
    "BATCHNORM_EPS",
    "BATCHNORM_MOMENTUM",
    "VICREG_EXPANDER_ACTIVATION",
    "VICREG_EXPANDER_INITIALISATION",
    "VICREG_EXPANDER_NORMALISATION",
    "VICRegExpander",
]
