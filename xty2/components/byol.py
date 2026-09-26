"""BYOL `utils/networks.MLP` at 82a3474, with card-declared tabular widths."""

from typing import ClassVar

from torch import nn

from xty2.components._nn import validate_dimension, validate_widths
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue

ACTIVATION = "hidden relu; output linear"
NORMALISATION = (
    "hidden BN affine=true eps=1e-5 momentum=0.1 track_running_stats=true; output none"
)
INITIALISATION = (
    "torch Linear reset_parameters; hidden bias=true; output bias=false; "
    "BN weight=1,bias=0,running_mean=0,running_var=1"
)


class _ByolMLP(Component):
    """`networks.MLP`: biased linear, affine BN, ReLU, then a bias-free linear.

    BYOL's projector and predictor are the *same* module class at the source,
    constructed twice with different sizes, so they are one class here too and
    differ only in the ports they sit between. That matters beyond tidiness:
    `docs/recipes/byol.md` §3.2 routes the online prediction of one view against
    the target projection of the other, and a topology difference between the
    two heads would be a second, unstated departure from a method whose
    published ablations turn on exactly which head is present.

    Three details are the source's rather than the obvious defaults, and each is
    one line of `MLP.__call__`:

    * **the hidden linear carries a bias and the output linear does not**
      (`with_bias=True` then `with_bias=False`). SimSiam's heads make the
      opposite choice at both layers, which is why `simsiam.py` cannot be reused
      here even though the two methods are siblings;
    * **normalisation is on the hidden layer only.** There is no output BN, so
      the projection this card feeds to the loss is un-normalised and the
      `1e-12` squared-norm floor in the objective is load-bearing rather than
      decorative;
    * **`bn_config` is `decay_rate=0.9, eps=1e-5`**, which is torch's
      `momentum=0.1, eps=1e-5`: Haiku's decay rate is what the running estimate
      keeps and torch's momentum is what it takes from the batch.

    The batch-size guard is the same one SimSiam carries. Training BN over one
    row has no variance to normalise by, and torch's own error names the module
    rather than the stage that asked for the forward.
    """

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "widths": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
    }
    INPUT: ClassVar[Port]
    OUTPUT: ClassVar[Port]

    def __init__(
        self,
        name: str,
        *,
        input_dim: int,
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
        current = validate_dimension(input_dim, field="input_dim", owner=owner)
        self.widths = validate_widths(widths, owner=owner)
        if len(self.widths) != 2:
            raise GraphError(
                f"{owner} is `networks.MLP`, one hidden layer and one output "
                f"layer, so it takes exactly two widths; got {self.widths!r}"
            )
        for value, expected in (
            (activation, ACTIVATION),
            (normalisation, NORMALISATION),
            (dropout, 0.0),
            (initialisation, INITIALISATION),
        ):
            if value != expected:
                raise GraphError(f"{owner} requires {expected!r}, got {value!r}")
        hidden, output = self.widths
        self.network = nn.Sequential(
            nn.Linear(current, hidden, bias=True),
            nn.BatchNorm1d(
                hidden,
                eps=1e-5,
                momentum=0.1,
                affine=True,
                track_running_stats=True,
            ),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, output, bias=False),
        )
        self.output_dim = output

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        value = ports.tensor(self.INPUT)
        if self.training and value.shape[0] < 2:
            raise GraphError("BYOL training BatchNorm requires at least two rows")
        return {self.OUTPUT: self.network(value)}


class BYOLProjector(_ByolMLP):
    """`g_theta` — the online projection, and under teacher parameters `g_xi`."""

    INPUT = Port.X_REPR
    OUTPUT = Port.X_PROJ


class BYOLPredictor(_ByolMLP):
    """`q_theta` — the online predictor, which the target network never runs."""

    INPUT = Port.X_PROJ
    OUTPUT = Port.X_PRED


__all__ = [
    "ACTIVATION",
    "INITIALISATION",
    "NORMALISATION",
    "BYOLPredictor",
    "BYOLProjector",
]
