"""SimSiam builder.py topology at a7bc177, with card-declared tabular widths."""

from typing import ClassVar

from torch import nn

from xty2.components._nn import validate_dimension, validate_widths
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue

ACTIVATION = "hidden relu; output linear"
PROJECTOR_NORMALISATION = (
    "hidden BN affine=true; output BN affine=false; all eps=1e-5 "
    "momentum=0.1 track_running_stats=true"
)
PREDICTOR_NORMALISATION = (
    "hidden BN affine=true eps=1e-5 momentum=0.1 track_running_stats=true; output none"
)
PROJECTOR_INITIALISATION = (
    "torch Linear reset_parameters; hidden bias=false; final bias initialised "
    "then frozen; BN affine weight=1,bias=0,running_mean=0,running_var=1"
)
PREDICTOR_INITIALISATION = (
    "torch Linear reset_parameters; hidden bias=false; final bias=true; "
    "BN weight=1,bias=0,running_mean=0,running_var=1"
)


class _SiameseMLP(Component):
    CARD_KEYS: ClassVar[dict[str, str]] = {
        "widths": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
    }
    INPUT: ClassVar[Port]
    OUTPUT: ClassVar[Port]
    DEPTH: ClassVar[int]
    NORMALISATION: ClassVar[str]
    INITIALISATION: ClassVar[str]

    def __init__(
        self,
        name: str,
        *,
        representation_dim: int,
        widths: tuple[int, ...] = REQUIRED,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
    ) -> None:
        super().__init__(name, requires={self.INPUT}, provides={self.OUTPUT})
        self.widths = widths
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        card_hyperparameters(self)
        owner = type(self).__name__
        current = validate_dimension(
            representation_dim, field="representation_dim", owner=owner
        )
        self.widths = validate_widths(widths, owner=owner)
        if len(widths) != self.DEPTH:
            raise GraphError(f"{owner} requires {self.DEPTH} layers")
        for value, expected in (
            (activation, ACTIVATION),
            (normalisation, self.NORMALISATION),
            (dropout, 0.0),
            (initialisation, self.INITIALISATION),
        ):
            if value != expected:
                raise GraphError(f"{owner} requires {expected!r}, got {value!r}")
        layers: list[nn.Module] = []
        for width in widths[:-1]:
            layers.extend(
                (
                    nn.Linear(current, width, bias=False),
                    nn.BatchNorm1d(
                        width,
                        eps=1e-5,
                        momentum=0.1,
                        affine=True,
                        track_running_stats=True,
                    ),
                    nn.ReLU(inplace=True),
                )
            )
            current = width
        final = nn.Linear(current, widths[-1], bias=True)
        layers.append(final)
        if self.OUTPUT == Port.X_PROJ:
            assert final.bias is not None
            final.bias.requires_grad_(False)
            # Fixed source bias travels in checkpoints, outside the optimiser.
            bias = final.bias.detach()
            del final._parameters["bias"]
            final.register_buffer("bias", bias)
            layers.append(
                nn.BatchNorm1d(
                    widths[-1],
                    eps=1e-5,
                    momentum=0.1,
                    affine=False,
                    track_running_stats=True,
                )
            )
        self.network = nn.Sequential(*layers)
        self.output_dim = widths[-1]

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        value = ports.tensor(self.INPUT)
        if self.training and value.shape[0] < 2:
            raise GraphError("SimSiam training BatchNorm requires at least two rows")
        return {self.OUTPUT: self.network(value)}


class SimSiamProjector(_SiameseMLP):
    """Three-layer projector with a frozen final bias and non-affine output BN."""

    INPUT = Port.X_REPR
    OUTPUT = Port.X_PROJ
    DEPTH = 3
    NORMALISATION = PROJECTOR_NORMALISATION
    INITIALISATION = PROJECTOR_INITIALISATION


class SimSiamPredictor(_SiameseMLP):
    """Two-layer prediction bottleneck; no output normalisation."""

    INPUT = Port.X_PROJ
    OUTPUT = Port.X_PRED
    DEPTH = 2
    NORMALISATION = PREDICTOR_NORMALISATION
    INITIALISATION = PREDICTOR_INITIALISATION
