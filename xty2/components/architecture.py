"""The MLP architecture the first recipe's three components share.

`tarnet` (P5) wires an encoder and two heads, and the paper states their
architecture as **one** line: "three fully-connected exponential-linear layers
of 200 units for the representation and three of 100 for the hypothesis". The
card key vocabulary is closed and `architecture.widths_depths` names *one*
value (`DESIGN.md` §9.1), so that line has to bind once, not three times — a
per-component binding would either collide in `plan.hyperparameters` or need a
key the vocabulary does not have.

`MLPArchitecture` is that one value. The recipe constructs it once and hands
the same instance to every component; `MLPComponent` delegates the five
`architecture.*` card keys to it, so all three components resolve them to
identical values and the merge in `compile()` accepts. Handing two different
architectures to two components is then a compile error naming the key, which
is the right failure: it means the plan and the card's one-line architecture
description have stopped describing the same network.

This sits at the root of `components/` rather than in `encoders/`, `outcome/`
or `treatment/` because all three subpackages read it. It is the same
arrangement `DESIGN.md` §7 gives for `WeightDecay` and `GradientClipping`: a
value object holding the several halves of one card field, beside the thing
that consumes it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Final, Literal, get_args

from torch import nn

from xty2.core.card_keys import REQUIRED, is_required
from xty2.core.errors import CompileError, GraphError
from xty2.core.graph import Component
from xty2.core.ports import Port

ActivationName = Literal["relu", "elu", "tanh"]
"""The activations v1 builds. Binds `architecture.activation`."""

NormalisationName = Literal["none", "layer", "batch"]
"""Where a hidden layer is normalised, or that it is not. `FIDELITY.md` §2
asks a card to say "BN vs LN vs none — and where", so `none` is a value here
and not the absence of one."""

InitialisationName = Literal["torch_default", "xavier_uniform", "xavier_normal"]
"""How linear weights are initialised. `torch_default` is Kaiming-uniform, and
naming it explicitly is the point: it is a choice a card can state, not the
silence that means nobody made one."""

ACTIVATIONS: Final[tuple[ActivationName, ...]] = get_args(ActivationName)
NORMALISATIONS: Final[tuple[NormalisationName, ...]] = get_args(NormalisationName)
INITIALISATIONS: Final[tuple[InitialisationName, ...]] = get_args(InitialisationName)


@dataclass(frozen=True)
class MLPArchitecture:
    """One architecture, shared by every MLP component of a recipe.

    Attributes:
        representation: Hidden widths of the shared trunk. Its last entry is
            the width of `X_REPR`, and the trunk's final layer carries the
            activation like every other — the representation is the output of
            an activated layer, not of a bare projection.
        head: Hidden widths of each head placed on the representation. One
            entry per hidden layer; the head's output layer is a bare linear
            map whose width is fixed by what the head produces (`K` logits,
            or one arm mean).
        activation: Applied after every hidden layer.
        normalisation: Applied to every hidden layer, between the linear map
            and the activation.
        dropout: Applied after every hidden activation. `0.0` disables it and
            is written explicitly.
        initialisation: Applied to every linear layer this architecture builds.
    """

    representation: tuple[int, ...] = REQUIRED
    head: tuple[int, ...] = REQUIRED
    activation: ActivationName = REQUIRED
    normalisation: NormalisationName = REQUIRED
    dropout: float = REQUIRED
    initialisation: InitialisationName = REQUIRED

    def __post_init__(self) -> None:
        for field, key in (
            ("representation", "architecture.widths_depths"),
            ("head", "architecture.widths_depths"),
            ("activation", "architecture.activation"),
            ("normalisation", "architecture.normalisation"),
            ("dropout", "architecture.dropout"),
            ("initialisation", "architecture.initialisation"),
        ):
            _require_set(field, getattr(self, field), key)
        object.__setattr__(self, "representation", tuple(self.representation))
        object.__setattr__(self, "head", tuple(self.head))
        _require_widths("representation", self.representation)
        _require_widths("head", self.head)
        _require_choice("activation", self.activation, ACTIVATIONS)
        _require_choice("normalisation", self.normalisation, NORMALISATIONS)
        _require_choice("initialisation", self.initialisation, INITIALISATIONS)
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, int | float):
            raise CompileError(
                f"MLPArchitecture.dropout must be a number, got {type(self.dropout)}"
            )
        if not math.isfinite(self.dropout) or not 0.0 <= float(self.dropout) < 1.0:
            raise CompileError(
                f"MLPArchitecture.dropout must be in [0, 1), got {self.dropout!r}. "
                "Dropout of 1 drops every unit; it is not a way to disable the "
                "layer, and 0.0 is."
            )

    @property
    def width(self) -> int:
        """`H`, the width of `X_REPR` — the trunk's last hidden width."""
        return int(self.representation[-1])

    @property
    def widths_description(self) -> str:
        """The whole stack as one line — the value bound to the card key.

        Rendered rather than structured for the reason `OptimiserSpec` renders
        its optimiser identity: `plan.hyperparameters` is diffed against a
        card's §4 YAML by eye, and a pair of tuples in that column is diffable
        by a machine and not by a reviewer.
        """
        return (
            f"representation {_layers(self.representation)}, heads {_layers(self.head)}"
        )

    def describe_lines(self) -> tuple[str, ...]:
        """The architecture as the plan prints it, one card field per line."""
        return (
            f"widths        {self.widths_description}",
            f"activation    {self.activation}",
            f"normalisation {self.normalisation}",
            f"dropout       {float(self.dropout)!r}",
            f"initialisation {self.initialisation}",
        )


class MLPComponent(Component):
    """A `Component` whose parameterisation is an `MLPArchitecture`.

    It exists to declare the five `architecture.*` card keys **once**. Every
    subclass resolves them through the shared architecture object, so two
    components of one recipe cannot disagree about the network the card
    describes in a single line.

    Subclasses that own a sixth key — `tarnet_head` and
    `architecture.output_parameterisation` — extend `CARD_KEYS` rather than
    replacing it.
    """

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "widths_depths": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
    }

    def __init__(
        self,
        name: str,
        *,
        architecture: MLPArchitecture,
        requires: Iterable[Port],
        provides: Iterable[Port],
    ) -> None:
        super().__init__(name, requires=requires, provides=provides)
        if not isinstance(architecture, MLPArchitecture):
            raise GraphError(
                f"component {name!r} takes an MLPArchitecture — one value object "
                "holding the whole architecture.* card block, shared by every "
                f"MLP component of a recipe — got {type(architecture)}."
            )
        self.architecture = architecture

    @property
    def widths_depths(self) -> str:
        """`architecture.widths_depths`, describing the whole stack."""
        return self.architecture.widths_description

    @property
    def activation(self) -> ActivationName:
        """`architecture.activation`."""
        return self.architecture.activation

    @property
    def normalisation(self) -> NormalisationName:
        """`architecture.normalisation`."""
        return self.architecture.normalisation

    @property
    def dropout(self) -> float:
        """`architecture.dropout`."""
        return float(self.architecture.dropout)

    @property
    def initialisation(self) -> InitialisationName:
        """`architecture.initialisation`."""
        return self.architecture.initialisation


def build_mlp(
    in_features: int,
    hidden: Sequence[int],
    out_features: int | None,
    architecture: MLPArchitecture,
) -> nn.Sequential:
    """A stack of `hidden` activated layers, optionally ending in a linear map.

    Args:
        in_features: Width of the input.
        hidden: One entry per hidden layer. Each is linear, then normalisation,
            then activation, then dropout — in that order, so the normaliser
            sees the pre-activation and dropout is not undone by it.
        out_features: Width of a final **bare** linear layer, or `None` for a
            trunk whose output is the last activated hidden layer. The trunk
            form is what the representation needs: `Φ(x)` is the output of an
            activated layer, not of a projection hanging off one.
        architecture: Supplies the activation, normalisation, dropout and
            initialisation. It is one object rather than four arguments
            because it is one card field.

    Returns:
        The stack, with `architecture.initialisation` already applied.
    """
    layers: list[nn.Module] = []
    width = in_features
    for size in hidden:
        layers.append(nn.Linear(width, size))
        layers.extend(_normaliser(architecture.normalisation, size))
        layers.append(_activation(architecture.activation))
        if architecture.dropout > 0.0:
            layers.append(nn.Dropout(float(architecture.dropout)))
        width = size
    if out_features is not None:
        layers.append(nn.Linear(width, out_features))
    stack = nn.Sequential(*layers)
    _initialise(stack, architecture.initialisation)
    return stack


def _activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "elu":
        return nn.ELU()
    return nn.Tanh()


def _normaliser(name: NormalisationName, width: int) -> tuple[nn.Module, ...]:
    if name == "layer":
        return (nn.LayerNorm(width),)
    if name == "batch":
        return (nn.BatchNorm1d(width),)
    return ()


def _initialise(stack: nn.Module, name: InitialisationName) -> None:
    """Apply `name` to every linear layer, leaving norm layers to torch.

    `torch_default` is a real branch and not a no-op by omission: it is the
    Kaiming-uniform initialisation `nn.Linear` performs in its own constructor,
    and a card naming it is naming that rule rather than declining to choose.
    """
    if name == "torch_default":
        return
    for module in stack.modules():
        if not isinstance(module, nn.Linear):
            continue
        if name == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _layers(widths: tuple[int, ...]) -> str:
    """`3x200` for a uniform stack, `[200, 100]` otherwise."""
    if len(set(widths)) == 1:
        return f"{len(widths)}x{widths[0]}"
    return "[" + ", ".join(str(width) for width in widths) + "]"


def _require_set(field: str, value: object, key: str) -> None:
    if is_required(value):
        raise CompileError(
            f"MLPArchitecture was given no {field!r}. It binds card key {key!r} "
            "and is governed by the paper, so it has no usable default — the "
            "recipe sets it explicitly (DESIGN.md §9.1, CLAUDE.md standing "
            "rules)."
        )


def _require_widths(field: str, widths: tuple[int, ...]) -> None:
    if not widths:
        raise CompileError(
            f"MLPArchitecture.{field} is empty. It is one width per layer, and "
            "a stack with no layers is a card line describing a network that "
            "does not exist."
        )
    if any(type(width) is not int or width < 1 for width in widths):
        raise CompileError(
            f"MLPArchitecture.{field} = {widths!r}; every entry is a positive "
            "integer layer width"
        )


def _require_choice(field: str, value: object, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise CompileError(
            f"MLPArchitecture.{field} = {value!r}; expected one of "
            f"{list(allowed)!r}. A fourth arrives with the card that names it "
            "(DESIGN.md §11)."
        )


__all__ = [
    "ACTIVATIONS",
    "INITIALISATIONS",
    "NORMALISATIONS",
    "ActivationName",
    "InitialisationName",
    "MLPArchitecture",
    "MLPComponent",
    "NormalisationName",
    "build_mlp",
]
