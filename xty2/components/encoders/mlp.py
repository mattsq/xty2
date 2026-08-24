"""`mlp_encoder`: the shared representation `Φ(x)` (`DESIGN.md` §3).

`X_RAW -> X_REPR`. It is the trunk every head in a recipe hangs off, and it is
the component the leakage rule cares about most: it reads raw `x` and nothing
else, so `Y_RAW` is not in its transitive closure and the plan says so.
"""

from __future__ import annotations

from xty2.components.architecture import MLPArchitecture, MLPComponent, build_mlp
from xty2.core.graph import PortView
from xty2.core.ports import Port, PortValue
from xty2.core.schema import Schema


class MLPEncoder(MLPComponent):
    """A fully-connected trunk over the feature vector.

    The output is the last **activated** hidden layer, not a projection off
    one: `Φ(x)` in TARNet is the output of the third exponential-linear layer,
    and a bare linear map on the end would be a fourth layer the card does not
    describe.

    Attributes:
        architecture: Shared with every other MLP component of the recipe; it
            supplies `representation` (the trunk's widths) and the four
            `architecture.*` fields the whole stack has in common.
    """

    def __init__(
        self,
        schema: Schema,
        *,
        architecture: MLPArchitecture,
        name: str = "mlp_encoder",
    ) -> None:
        super().__init__(
            name,
            architecture=architecture,
            requires={Port.X_RAW},
            provides={Port.X_REPR},
        )
        self.net = build_mlp(
            schema.num_features, architecture.representation, None, architecture
        )

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        """`[B, D] -> [B, H]`."""
        return {Port.X_REPR: self.net(ports.tensor(Port.X_RAW))}


__all__ = ["MLPEncoder"]
