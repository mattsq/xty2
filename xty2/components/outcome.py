"""Treatment-specific outcome parameterisations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar, cast

import torch
from torch import nn

from xty2.components._nn import (
    CFRNET_INITIALISATION,
    elu_stack,
    initialise_cfrnet,
    validate_dimension,
    validate_dropout,
    validate_widths,
)
from xty2.core.card_keys import REQUIRED, card_hyperparameters, is_required
from xty2.core.distributions import GaussianOutcome
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue
from xty2.core.schema import OutcomeSpec


class _OutcomeMLP(nn.Module):
    """One independent treatment arm."""

    def __init__(
        self, input_dim: int, widths: tuple[int, ...], output_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.hidden, hidden_dim = elu_stack(input_dim, widths, dropout=dropout)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.output(self.hidden(representation)))


class TARNetHead(Component):
    """Independent outcome MLPs, one per categorical treatment value."""

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
        name: str = "tarnet_head",
        *,
        representation_dim: int,
        num_treatments: int,
        outcome: OutcomeSpec,
        widths: tuple[int, ...] = REQUIRED,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        output_parameterisation: str = REQUIRED,
    ) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={Port.Y_GIVEN_XT})
        self.widths = widths
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        self.output_parameterisation = output_parameterisation

        owner = type(self).__name__
        representation_dim = validate_dimension(
            representation_dim, field="representation_dim", owner=owner
        )
        self.num_treatments = validate_dimension(
            num_treatments, field="num_treatments", owner=owner
        )
        if self.num_treatments < 2:
            raise GraphError(f"{owner}.num_treatments must be at least 2")
        if not isinstance(outcome, OutcomeSpec):
            raise GraphError(
                f"{owner}.outcome must be an OutcomeSpec, got {type(outcome)}"
            )
        if not outcome.is_continuous:
            raise GraphError(
                f"{owner} supports only continuous outcomes because it emits a "
                "fixed-scale GaussianOutcome; use a categorical outcome head "
                "for a categorical OutcomeSpec"
            )
        self.outcome_shape = outcome.shape
        if any(type(size) is not int or size < 1 for size in self.outcome_shape):
            raise GraphError(
                f"{owner}.outcome_shape must contain positive dimensions, got "
                f"{self.outcome_shape!r}"
            )
        card_hyperparameters(self)
        self.widths = validate_widths(self.widths, owner=owner)
        self.dropout = validate_dropout(self.dropout, owner=owner)
        if self.activation != "elu":
            raise GraphError(
                f"{owner}.activation supports only 'elu', got {self.activation!r}"
            )
        if self.normalisation != "none":
            raise GraphError(f"{owner}.normalisation supports only 'none'")
        if self.initialisation != CFRNET_INITIALISATION:
            raise GraphError(
                f"{owner}.initialisation supports only {CFRNET_INITIALISATION!r}"
            )
        expected_output = "K means; fixed Gaussian scale=1.0"
        if self.output_parameterisation != expected_output:
            raise GraphError(
                f"{owner}.output_parameterisation supports only "
                f"{expected_output!r}, got {self.output_parameterisation!r}"
            )

        output_dim = math.prod(self.outcome_shape) if self.outcome_shape else 1
        self.heads = nn.ModuleList(
            _OutcomeMLP(
                representation_dim,
                self.widths,
                output_dim,
                self.dropout,
            )
            for _ in range(self.num_treatments)
        )
        initialise_cfrnet(self)

    @property
    def widths_description(self) -> object:
        """The independent-arm topology as one component-scoped card value."""
        if is_required(self.widths):
            return REQUIRED
        return f"{self.num_treatments} independent heads, each {list(self.widths)!r}"

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        representation = ports.tensor(Port.X_REPR)
        raw = torch.stack([head(representation) for head in self.heads], dim=1)
        if self.outcome_shape:
            loc = raw.reshape(raw.shape[0], self.num_treatments, *self.outcome_shape)
        else:
            loc = raw.squeeze(-1)
        return {Port.Y_GIVEN_XT: GaussianOutcome(loc=loc, scale=torch.ones_like(loc))}


__all__ = ["TARNetHead"]
