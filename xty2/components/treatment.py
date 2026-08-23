"""Categorical treatment parameterisations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from torch import nn

from xty2.components._nn import (
    CFRNET_INITIALISATION,
    initialise_cfrnet,
    validate_dimension,
    validate_dropout,
)
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.distributions import CategoricalTreatment
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue


class CategoricalPropensity(Component):
    """A linear softmax propensity head from `X_REPR` to `T_GIVEN_X`."""

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
        name: str = "categorical_propensity",
        *,
        representation_dim: int,
        num_treatments: int,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        output_parameterisation: str = REQUIRED,
    ) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={Port.T_GIVEN_X})
        self.representation_dim = representation_dim
        self.num_treatments = num_treatments
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        self.output_parameterisation = output_parameterisation
        card_hyperparameters(self)

        owner = type(self).__name__
        self.representation_dim = validate_dimension(
            self.representation_dim, field="representation_dim", owner=owner
        )
        self.num_treatments = validate_dimension(
            self.num_treatments, field="num_treatments", owner=owner
        )
        if self.num_treatments < 2:
            raise GraphError(f"{owner}.num_treatments must be at least 2")
        self.dropout = validate_dropout(self.dropout, owner=owner)
        if self.activation != "linear logits":
            raise GraphError(f"{owner}.activation supports only 'linear logits'")
        if self.normalisation != "none":
            raise GraphError(f"{owner}.normalisation supports only 'none'")
        if self.dropout != 0.0:
            raise GraphError(f"{owner} is linear and supports only dropout=0.0")
        if self.initialisation != CFRNET_INITIALISATION:
            raise GraphError(
                f"{owner}.initialisation supports only {CFRNET_INITIALISATION!r}"
            )
        if self.output_parameterisation != "K softmax logits":
            raise GraphError(
                f"{owner}.output_parameterisation supports only 'K softmax logits'"
            )
        self.logits = nn.Linear(self.representation_dim, self.num_treatments)
        initialise_cfrnet(self)

    @property
    def widths_description(self) -> str:
        return f"linear {self.representation_dim} -> {self.num_treatments}"

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        logits = self.logits(ports.tensor(Port.X_REPR))
        return {Port.T_GIVEN_X: CategoricalTreatment(logits)}


__all__ = ["CategoricalPropensity"]
