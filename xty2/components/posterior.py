"""Outcome-dependent categorical treatment posteriors."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import nn

from xty2.components._nn import (
    TORCH_LINEAR_INITIALISATION,
    relu_stack,
    validate_dimension,
    validate_dropout,
    validate_widths,
)
from xty2.core.card_keys import REQUIRED, card_hyperparameters, is_required
from xty2.core.distributions import CategoricalTreatment
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue
from xty2.core.schema import OutcomeSpec


class CategoricalPosterior(Component):
    """An MLP from concatenated raw covariates and outcome to `q(t | x, y)`."""

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "widths_description": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
        "output_parameterisation": "architecture.output_parameterisation",
        "standardisation": "data.standardisation",
        "outcome_scaling": "data.outcome_scaling",
        "treatment_encoding": "data.treatment_encoding",
    }

    def __init__(
        self,
        name: str = "categorical_posterior",
        *,
        input_dim: int,
        num_treatments: int,
        outcome: OutcomeSpec,
        widths: tuple[int, ...] = REQUIRED,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        output_parameterisation: str = REQUIRED,
        standardisation: object = REQUIRED,
        outcome_scaling: object = REQUIRED,
        treatment_encoding: object = REQUIRED,
    ) -> None:
        super().__init__(
            name,
            requires={Port.X_RAW, Port.Y_RAW},
            provides={Port.T_GIVEN_XY},
        )
        self.input_dim = input_dim
        self.num_treatments = num_treatments
        self.outcome = outcome
        self.widths = widths
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        self.output_parameterisation = output_parameterisation
        self.standardisation = standardisation
        self.outcome_scaling = outcome_scaling
        self.treatment_encoding = treatment_encoding
        card_hyperparameters(self)

        owner = type(self).__name__
        self.input_dim = validate_dimension(
            self.input_dim, field="input_dim", owner=owner
        )
        self.num_treatments = validate_dimension(
            self.num_treatments, field="num_treatments", owner=owner
        )
        if self.num_treatments < 2:
            raise GraphError(f"{owner}.num_treatments must be at least 2")
        if not isinstance(self.outcome, OutcomeSpec):
            raise GraphError(
                f"{owner}.outcome must be an OutcomeSpec, got {type(self.outcome)}"
            )
        if not self.outcome.is_continuous:
            raise GraphError(
                f"{owner} supports only continuous outcomes because Y_RAW is "
                "concatenated as a float network input"
            )
        self.widths = validate_widths(self.widths, owner=owner)
        self.dropout = validate_dropout(self.dropout, owner=owner)
        if self.activation != "relu":
            raise GraphError(f"{owner}.activation supports only 'relu'")
        if self.normalisation != "none":
            raise GraphError(f"{owner}.normalisation supports only 'none'")
        if self.initialisation != TORCH_LINEAR_INITIALISATION:
            raise GraphError(
                f"{owner}.initialisation supports only {TORCH_LINEAR_INITIALISATION!r}"
            )
        if self.output_parameterisation != "K softmax logits":
            raise GraphError(
                f"{owner}.output_parameterisation supports only 'K softmax logits'"
            )

        outcome_dim = math.prod(self.outcome.shape) if self.outcome.shape else 1
        self.hidden, hidden_dim = relu_stack(
            self.input_dim + outcome_dim,
            self.widths,
            dropout=self.dropout,
        )
        self.logits = nn.Linear(hidden_dim, self.num_treatments)

    @property
    def widths_description(self) -> object:
        """The concatenation and hidden topology as one card value."""
        if is_required(self.widths):
            return REQUIRED
        return f"concat(X_RAW, Y_RAW) -> {list(self.widths)!r} -> K"

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        x = ports.tensor(Port.X_RAW)
        y = ports.tensor(Port.Y_RAW).reshape(x.shape[0], -1)
        logits = self.logits(self.hidden(torch.cat((x, y), dim=1)))
        return {Port.T_GIVEN_XY: CategoricalTreatment(logits)}


__all__ = ["CategoricalPosterior"]
