"""`tarnet_head`: one outcome head per treatment arm (`DESIGN.md` §3, §3.1).

`X_REPR -> Y_GIVEN_XT`. This is the architecture TARNet is named for: the
treatment is not a *feature* of the outcome model, it selects which head
evaluates the representation. That is also exactly the candidate-treatment
contract — `log_prob(y, t)` with `t: [B, K]` asks all `K` heads at once — so
the head computes every arm's mean on every forward pass and lets the
distribution select. There is no cheaper option that satisfies the contract:
`MissingTreatmentMarginalNLL` needs all `K` arms on every row it sees, so a
head that evaluated one arm would be re-run `K` times by the objective the
whole design is arranged around.

**Unit scale, and it is a modelling choice.** The port carries a distribution
and the paper minimises squared error, so the arm mean is wrapped in a
Gaussian of unit scale: `-log N(y; mu, 1)` is `0.5 * (y - mu)^2` plus a
constant, which is the paper's loss up to a factor absorbed by the learning
rate. It is *not* a nuisance constant for the marginalisation term, which
weights arms by `p(y | x, t=k)` — the scale sets how sharply a residual
discriminates between arms. That is why the card records it under
`architecture.output_parameterisation` rather than leaving it implied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar

import torch
from torch import nn

from xty2.components.architecture import MLPArchitecture, MLPComponent, build_mlp
from xty2.core.distributions import GaussianOutcome
from xty2.core.errors import GraphError
from xty2.core.graph import PortView
from xty2.core.ports import Port, PortValue
from xty2.core.schema import Schema

OUTPUT_PARAMETERISATION = "gaussian per-arm mean, unit scale"
"""The value bound to `architecture.output_parameterisation`.

Derived from the head's identity rather than taken as a constructor argument,
for the reason `DESIGN.md` §9.1 gives for the four per-objective `losses.*`
keys: there is nothing for a recipe to declare twice. Choosing `tarnet_head`
*is* choosing this parameterisation, and a `REQUIRED` field with one legal
value would be a sentinel guarding a decision nobody can make. Swap the head
and the line in the plan changes with it.
"""


class TarnetHead(MLPComponent):
    """`K` arm-specific heads over the shared representation.

    Attributes:
        architecture: Shared with the encoder and the propensity head. Its
            `head` widths are the hidden layers of **each** arm.
    """

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        **MLPComponent.CARD_KEYS,
        "output_parameterisation": "architecture.output_parameterisation",
    }

    def __init__(
        self,
        schema: Schema,
        *,
        architecture: MLPArchitecture,
        name: str = "tarnet_head",
    ) -> None:
        super().__init__(
            name,
            architecture=architecture,
            requires={Port.X_REPR},
            provides={Port.Y_GIVEN_XT},
        )
        if not schema.outcome.is_continuous:
            raise GraphError(
                f"component {name!r} parameterises a Gaussian density and this "
                "schema declares a categorical outcome. A categorical outcome "
                "head is a different parameterisation and arrives with the "
                "recipe that needs it (DESIGN.md §11)."
            )
        self.event_shape = tuple(schema.outcome.shape)
        self.num_treatments = int(schema.treatment_cardinality)
        self.arms = nn.ModuleList(
            build_mlp(
                architecture.width,
                architecture.head,
                _event_size(self.event_shape),
                architecture,
            )
            for _ in range(self.num_treatments)
        )

    @property
    def output_parameterisation(self) -> str:
        """`architecture.output_parameterisation` — see the module docstring."""
        return OUTPUT_PARAMETERISATION

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        """`[B, H] -> p(y | x, t)` with `loc: [B, K, *Dy]`."""
        representation = ports.tensor(Port.X_REPR)
        batch_size = representation.shape[0]
        loc = torch.stack(
            [
                arm(representation).reshape(batch_size, *self.event_shape)
                for arm in self.arms
            ],
            dim=1,
        )
        return {Port.Y_GIVEN_XT: GaussianOutcome(loc=loc, scale=torch.ones_like(loc))}


def _event_size(event_shape: tuple[int, ...]) -> int:
    """How many numbers one arm predicts per row: `prod(Dy)`, and 1 for `()`."""
    return math.prod(event_shape) if event_shape else 1


__all__ = ["OUTPUT_PARAMETERISATION", "TarnetHead"]
