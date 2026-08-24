"""`categorical_propensity`: `p(t | x)` over `K` classes (`DESIGN.md` §3).

`X_REPR -> T_GIVEN_X`. A single head emitting `K` logits, wrapped in the
reference `CategoricalTreatment` so that normalisation and the rank rule of
§3.1 come from one implementation rather than from each head's own softmax.

It sits on the **shared** representation, which is a decision and not a
detail: its NLL then trains `Φ` as well, so a recipe that adds this head has
changed its outcome model too. The `tarnet` card records that as a deviation
with an expected effect (`docs/recipes/tarnet.md` §5), because it is invisible
in the graph — the plan shows the wiring and cannot show what it costs.
"""

from __future__ import annotations

from xty2.components.architecture import MLPArchitecture, MLPComponent, build_mlp
from xty2.core.distributions import CategoricalTreatment
from xty2.core.graph import PortView
from xty2.core.ports import Port, PortValue
from xty2.core.schema import Schema


class CategoricalPropensity(MLPComponent):
    """A `K`-way softmax head over the representation.

    Attributes:
        architecture: Shared with the encoder and the outcome head; its `head`
            widths are this head's hidden layers, so one card line describes
            every head in the recipe.
    """

    def __init__(
        self,
        schema: Schema,
        *,
        architecture: MLPArchitecture,
        name: str = "categorical_propensity",
    ) -> None:
        super().__init__(
            name,
            architecture=architecture,
            requires={Port.X_REPR},
            provides={Port.T_GIVEN_X},
        )
        self.num_treatments = int(schema.treatment_cardinality)
        self.net = build_mlp(
            architecture.width,
            architecture.head,
            self.num_treatments,
            architecture,
        )

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        """`[B, H] -> p(t | x)` with `probs: [B, K]`."""
        return {
            Port.T_GIVEN_X: CategoricalTreatment(self.net(ports.tensor(Port.X_REPR)))
        }


__all__ = ["CategoricalPropensity"]
