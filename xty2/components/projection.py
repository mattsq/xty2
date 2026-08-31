"""The pre-train head a contrastive objective is computed on (`DESIGN.md` §2).

Deliberately a class of its own rather than a widened `MLPEncoder`. The two
would share about forty lines of validation and nothing else: they sit at
different points of the graph, and every recorded number in this repository
depends on `MLPEncoder`'s construction-time RNG consumption, which a shared
base class would put at risk to save that duplication.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch.nn.functional as F
from torch import nn

from xty2.components._nn import (
    CFRNET_INITIALISATION,
    TORCH_LINEAR_INITIALISATION,
    initialise_cfrnet,
    validate_dimension,
    validate_dropout,
    validate_widths,
)
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue


class ProjectionHead(Component):
    """An MLP from `X_REPR` to `X_PROJ` — SCARF's `g`, SimCLR's projection.

    `normalisation="row_l2"` is what SCARF means by "the pre-train head network
    l2-normalizes the outputs so that they lie on the unit hypersphere".

    What it does **not** do is change `InfoNCEContrastive`'s value: that term
    takes the cosine itself, so the loss and both its diagnostics are identical
    whichever way this flag is set (measured: they agree to 1.5e-8, float32
    rounding). Two things it does do. It fixes what the *port* carries — a
    consumer reading `X_PROJ` gets unit vectors, which is what the paper says
    the embedding space is and what a future similarity graph over it would
    otherwise have to re-establish — and it scales the gradient reaching this
    head, since the unnormalised outputs here have norm around 0.02 and
    normalising divides it back out. An earlier version of this docstring
    claimed the loss depended on it. It does not, and the claim is corrected
    rather than deleted because "the objective normalises anyway" is the reason
    someone would remove the flag.

    **The last layer is affine, with no activation after it**, which is what a
    projection head of `n` layers means and is why this does not reuse
    `relu_stack`. The difference is not stylistic. A terminal ReLU confines
    every embedding to the non-negative orthant, where two rows can be
    orthogonal and never opposed, so the only route a contrastive loss has to a
    low off-diagonal similarity is disjoint sparse supports — and it takes it:
    measured on `scarf`'s Tier 1 fixture, a terminally-activated head drove
    99.6% of its units to zero by step 1,000, taking the alignment of a row
    with its own corrupted copy down with it (`docs/recipes/scarf.md` §7).
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
        name: str = "projection_head",
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
        if self.activation not in ("relu", "elu", "leaky_relu:0.1"):
            raise GraphError(
                f"{owner}.activation supports 'relu', 'elu', or "
                f"'leaky_relu:0.1', got {self.activation!r}"
            )
        if self.normalisation not in ("row_l2", "none"):
            raise GraphError(
                f"{owner}.normalisation supports 'row_l2' or 'none', got "
                f"{self.normalisation!r}"
            )
        if self.initialisation not in (
            CFRNET_INITIALISATION,
            TORCH_LINEAR_INITIALISATION,
        ):
            raise GraphError(
                f"{owner}.initialisation supports {CFRNET_INITIALISATION!r} or "
                f"{TORCH_LINEAR_INITIALISATION!r}, got {self.initialisation!r}"
            )
        self.network = _head(
            representation_dim,
            self.widths,
            activation=self.activation,
            dropout=self.dropout,
        )
        self.output_dim = self.widths[-1]
        if self.initialisation == CFRNET_INITIALISATION:
            initialise_cfrnet(self)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        embedding = self.network(ports.tensor(Port.X_REPR))
        if self.normalisation == "row_l2":
            embedding = F.normalize(embedding, p=2.0, dim=-1)
        return {Port.X_PROJ: embedding}


def _head(
    input_dim: int, widths: tuple[int, ...], *, activation: str, dropout: float
) -> nn.Sequential:
    """`Linear -> act -> ... -> Linear`: `len(widths)` layers, affine output."""
    layers: list[nn.Module] = []
    current = input_dim
    for index, width in enumerate(widths):
        if index:
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                layers.append(nn.LeakyReLU(negative_slope=0.1))
            if dropout:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(current, width))
        current = width
    return nn.Sequential(*layers)


__all__ = ["ProjectionHead"]
