"""Shared covariate encoders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch.nn.functional as F

from xty2.components._nn import (
    CFRNET_INITIALISATION,
    elu_stack,
    initialise_cfrnet,
    validate_dimension,
    validate_dropout,
    validate_widths,
)
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue


class MLPEncoder(Component):
    """An ELU MLP from `X_RAW` to a row-normalised `X_REPR`."""

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "widths": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
    }

    def __init__(
        self,
        name: str = "mlp_encoder",
        *,
        input_dim: int,
        widths: tuple[int, ...] = REQUIRED,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
    ) -> None:
        super().__init__(name, requires={Port.X_RAW}, provides={Port.X_REPR})
        self.widths = widths
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        card_hyperparameters(self)

        owner = type(self).__name__
        input_dim = validate_dimension(input_dim, field="input_dim", owner=owner)
        self.widths = validate_widths(self.widths, owner=owner)
        self.dropout = validate_dropout(self.dropout, owner=owner)
        if self.activation != "elu":
            raise GraphError(
                f"{owner}.activation supports only 'elu', got {self.activation!r}"
            )
        if self.normalisation != "row_l2":
            raise GraphError(
                f"{owner}.normalisation supports only 'row_l2', got "
                f"{self.normalisation!r}"
            )
        if self.initialisation != CFRNET_INITIALISATION:
            raise GraphError(
                f"{owner}.initialisation supports only {CFRNET_INITIALISATION!r}, "
                f"got {self.initialisation!r}"
            )
        self.network, self.output_dim = elu_stack(
            input_dim, self.widths, dropout=self.dropout
        )
        initialise_cfrnet(self)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        representation = self.network(ports.tensor(Port.X_RAW))
        return {Port.X_REPR: F.normalize(representation, p=2.0, dim=-1)}


__all__ = ["MLPEncoder"]
