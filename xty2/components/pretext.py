"""Categorical transform-prediction head."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from torch import nn

from xty2.components._nn import validate_dimension, validate_dropout
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue


class PretextHead(Component):
    """One affine head from ``X_REPR`` to transform-class logits."""

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "widths_description": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
        "output_parameterisation": "architecture.output_parameterisation",
    }

    def __init__(
        self,
        name: str = "pretext_head",
        *,
        representation_dim: int,
        num_transforms: int,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        output_parameterisation: str = REQUIRED,
    ) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={Port.PRETEXT_GIVEN_X})
        self.representation_dim = representation_dim
        self.num_transforms = num_transforms
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        self.output_parameterisation = output_parameterisation
        card_hyperparameters(self)
        owner = type(self).__name__
        self.representation_dim = validate_dimension(
            representation_dim, field="representation_dim", owner=owner
        )
        self.num_transforms = validate_dimension(
            num_transforms, field="num_transforms", owner=owner
        )
        self.dropout = validate_dropout(dropout, owner=owner)
        if self.num_transforms < 2:
            raise GraphError("PretextHead.num_transforms must be at least 2")
        if activation != "linear logits" or normalisation != "none" or dropout != 0.0:
            raise GraphError(
                "PretextHead is affine: linear logits, no normalisation or dropout"
            )
        if initialisation != "glorot_normal, bias=0":
            raise GraphError("PretextHead supports only 'glorot_normal, bias=0'")
        if output_parameterisation != f"{self.num_transforms} softmax logits":
            raise GraphError(
                f"PretextHead output_parameterisation must be "
                f"{self.num_transforms!r} softmax logits"
            )
        self.logits = nn.Linear(self.representation_dim, self.num_transforms)
        nn.init.xavier_normal_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    @property
    def widths_description(self) -> str:
        return f"linear {self.representation_dim} -> {self.num_transforms}"

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.PRETEXT_GIVEN_X: self.logits(ports.tensor(Port.X_REPR))}


__all__ = ["PretextHead"]
