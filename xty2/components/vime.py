"""VIME's two pretext estimators, `s_m` and `s_r` (`docs/recipes/vime.md`).

Both read the same `z = e(x~)` and emit one value per feature. They are
separate components rather than one head with two outputs because a port has
one producer (`DESIGN.md` §2) and because the card's ablations drop either
loss independently. Each is one affine layer, as `vime_self.py` builds them:
`Dense(dim, activation='sigmoid', name='mask')` and
`Dense(dim, activation='sigmoid', name='feature')`.

The mask estimator emits **logits**, not probabilities. Its sigmoid is folded
into `MaskEstimationBCE`, which computes the binary cross-entropy from logits
(`vime.md` §7). The feature estimator keeps its sigmoid, because the targets
are min-max scaled into `[0, 1]` and the port carries values, not logits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import nn

from xty2.components._nn import (
    GLOROT_UNIFORM_INITIALISATION,
    initialise_glorot_uniform,
    validate_dimension,
    validate_dropout,
)
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue


class _FeatureWiseHead(Component):
    """One affine layer from `X_REPR` to one value per feature."""

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "widths_description": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
        "output_parameterisation": "architecture.output_parameterisation",
    }
    OUTPUT: ClassVar[Port]
    ACTIVATION: ClassVar[str]
    OUTPUT_PARAMETERISATION: ClassVar[str]
    """`{d}` is replaced by the number of features."""

    def __init__(
        self,
        name: str,
        *,
        representation_dim: int,
        num_features: int,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        output_parameterisation: str = REQUIRED,
    ) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={self.OUTPUT})
        self.representation_dim = representation_dim
        self.num_features = num_features
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
        self.num_features = validate_dimension(
            num_features, field="num_features", owner=owner
        )
        self.dropout = validate_dropout(dropout, owner=owner)
        if activation != self.ACTIVATION:
            raise GraphError(
                f"{owner}.activation must be {self.ACTIVATION!r}, got {activation!r}"
            )
        if normalisation != "none" or self.dropout != 0.0:
            raise GraphError(
                f"{owner} is one affine layer: no normalisation and no dropout"
            )
        if initialisation != GLOROT_UNIFORM_INITIALISATION:
            raise GraphError(
                f"{owner} supports only {GLOROT_UNIFORM_INITIALISATION!r}, the Keras "
                f"Dense default the reference code uses; got {initialisation!r}"
            )
        expected = self.OUTPUT_PARAMETERISATION.format(d=self.num_features)
        if output_parameterisation != expected:
            raise GraphError(
                f"{owner}.output_parameterisation must be {expected!r}, got "
                f"{output_parameterisation!r}"
            )
        self.layer = nn.Linear(self.representation_dim, self.num_features)
        initialise_glorot_uniform(self)

    @property
    def widths_description(self) -> str:
        return f"linear {self.representation_dim} -> {self.num_features}"


class MaskEstimatorHead(_FeatureWiseHead):
    """`s_m`: one Bernoulli logit per feature, "was this cell replaced?"."""

    OUTPUT = Port.FEATURE_MASK_LOGITS
    ACTIVATION = "linear logits"
    OUTPUT_PARAMETERISATION = "{d} Bernoulli logits"

    def __init__(
        self,
        name: str = "mask_estimator",
        *,
        representation_dim: int,
        num_features: int,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        output_parameterisation: str = REQUIRED,
    ) -> None:
        super().__init__(
            name,
            representation_dim=representation_dim,
            num_features=num_features,
            activation=activation,
            normalisation=normalisation,
            dropout=dropout,
            initialisation=initialisation,
            output_parameterisation=output_parameterisation,
        )

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.FEATURE_MASK_LOGITS: self.layer(ports.tensor(Port.X_REPR))}


class FeatureEstimatorHead(_FeatureWiseHead):
    """`s_r`: the reconstructed row, one sigmoid value per feature."""

    OUTPUT = Port.RECONSTRUCTION
    ACTIVATION = "sigmoid"
    OUTPUT_PARAMETERISATION = "{d} values in (0, 1)"

    def __init__(
        self,
        name: str = "feature_estimator",
        *,
        representation_dim: int,
        num_features: int,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        output_parameterisation: str = REQUIRED,
    ) -> None:
        super().__init__(
            name,
            representation_dim=representation_dim,
            num_features=num_features,
            activation=activation,
            normalisation=normalisation,
            dropout=dropout,
            initialisation=initialisation,
            output_parameterisation=output_parameterisation,
        )

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {
            Port.RECONSTRUCTION: torch.sigmoid(self.layer(ports.tensor(Port.X_REPR)))
        }


__all__ = ["FeatureEstimatorHead", "MaskEstimatorHead"]
