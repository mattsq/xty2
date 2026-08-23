"""Conditional density parameterisations for continuous outcomes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import ClassVar, Protocol, cast

import torch
from nflows.transforms import (  # type: ignore[import-untyped]
    CompositeTransform,
    RandomPermutation,
)
from nflows.transforms.autoregressive import (  # type: ignore[import-untyped]
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
)
from torch import Tensor, nn
from torch.nn import functional as F

from xty2.components._nn import (
    validate_dimension,
    validate_dropout,
)
from xty2.core.card_keys import REQUIRED, card_hyperparameters, is_required
from xty2.core.distributions import TreatmentMode, treatment_mode
from xty2.core.errors import ContractError, GraphError
from xty2.core.graph import Component, PortView
from xty2.core.ports import Port, PortValue
from xty2.core.schema import OutcomeSpec

NFLOWS_INITIALISATION = "nflows 0.14 defaults"
RANDOM_PERMUTATION = "random after each transform"
STANDARD_NORMAL = "StandardNormal"


class _ConditionalTransform(Protocol):
    """The typed part of the untyped `nflows` transform API used here."""

    def __call__(
        self, inputs: Tensor, context: Tensor | None = None
    ) -> tuple[Tensor, Tensor]: ...

    def inverse(
        self, inputs: Tensor, context: Tensor | None = None
    ) -> tuple[Tensor, Tensor]: ...


def _fixed_antithetic_standard_normal(samples: int, features: int) -> Tensor:
    """Fixed standard-normal pairs without advancing Torch's global RNG."""
    generator = torch.Generator(device="cpu").manual_seed(0)
    half = torch.randn(samples // 2, features, generator=generator)
    return torch.cat((half, -half), dim=0)


class ConditionalFlowOutcome:
    """One conditional flow distribution satisfying the candidate-t contract."""

    def __init__(
        self,
        *,
        representation: Tensor,
        transform: _ConditionalTransform,
        num_treatments: int,
        event_shape: tuple[int, ...],
        mean_base_samples: Tensor,
    ) -> None:
        if representation.ndim != 2:
            raise ContractError(
                "conditional-flow representation must be [B, H], got "
                f"{tuple(representation.shape)}"
            )
        self._representation = representation
        self._transform = transform
        self._num_treatments = num_treatments
        self._event_shape = event_shape
        self._event_dim = math.prod(event_shape) if event_shape else 1
        if mean_base_samples.ndim != 2 or mean_base_samples.shape[1] != self._event_dim:
            raise ContractError(
                "fixed flow mean samples must be [Q, D_y], got "
                f"{tuple(mean_base_samples.shape)} for D_y={self._event_dim}"
            )
        self._mean_base_samples = mean_base_samples

    @property
    def batch_size(self) -> int:
        """`B`, the number of conditioning rows."""
        return int(self._representation.shape[0])

    @property
    def num_treatments(self) -> int:
        """`K`, the number of categorical treatment values."""
        return self._num_treatments

    @property
    def event_shape(self) -> tuple[int, ...]:
        """The trailing shape `Dy` restored at the protocol boundary."""
        return self._event_shape

    @property
    def event_dim(self) -> int:
        """The flattened continuous flow-event dimension, which excludes `t`."""
        return self._event_dim

    @property
    def representation_dim(self) -> int:
        """The learned part of the conditioner, before one-hot treatment."""
        return int(self._representation.shape[1])

    def log_prob(self, y: Tensor, t: Tensor) -> Tensor:
        """Evaluate `log p(y|x,t)`, inserting any candidate axis internally."""
        mode, context = self._context(t)
        event = self._event(y)
        if mode == "candidate":
            candidates = int(t.shape[1])
            event = event[:, None, :].expand(-1, candidates, -1)
        flat_event = event.reshape(-1, self._event_dim)
        flat_context = context.reshape(-1, context.shape[-1])
        noise, logabsdet = self._transform(flat_event, context=flat_context)
        base_log_prob = -0.5 * (noise.square() + math.log(2.0 * math.pi)).sum(dim=-1)
        result = base_log_prob + logabsdet
        return result.reshape(self.batch_size, -1) if mode == "candidate" else result

    def mean(self, t: Tensor) -> Tensor:
        """Approximate `E[y|x,t]` with fixed common antithetic base samples."""
        mode, context = self._context(t)
        flat_context = context.reshape(-1, context.shape[-1])
        rows = int(flat_context.shape[0])
        samples = int(self._mean_base_samples.shape[0])
        base = self._mean_base_samples[None, :, :].expand(rows, -1, -1)
        repeated_context = flat_context[:, None, :].expand(-1, samples, -1)
        values, _ = self._transform.inverse(
            base.reshape(-1, self._event_dim),
            context=repeated_context.reshape(-1, flat_context.shape[-1]),
        )
        means = values.reshape(rows, samples, self._event_dim).mean(dim=1)
        leading = (
            (self.batch_size, int(t.shape[1]))
            if mode == "candidate"
            else (self.batch_size,)
        )
        return self._restore(means.reshape(*leading, self._event_dim), leading)

    def sample(self, t: Tensor, n: int) -> Tensor:
        """Draw from Torch's global RNG and put the sample axis first."""
        if type(n) is not int or n < 1:
            raise ContractError(f"n must be a positive integer, got {n!r}")
        mode, context = self._context(t)
        flat_context = context.reshape(-1, context.shape[-1])
        rows = int(flat_context.shape[0])
        base = torch.randn(
            rows,
            n,
            self._event_dim,
            dtype=self._representation.dtype,
            device=self._representation.device,
        )
        repeated_context = flat_context[:, None, :].expand(-1, n, -1)
        values, _ = self._transform.inverse(
            base.reshape(-1, self._event_dim),
            context=repeated_context.reshape(-1, flat_context.shape[-1]),
        )
        by_context = values.reshape(rows, n, self._event_dim)
        leading: tuple[int, ...]
        if mode == "candidate":
            candidates = int(t.shape[1])
            ordered = by_context.reshape(
                self.batch_size, candidates, n, self._event_dim
            ).permute(2, 0, 1, 3)
            leading = (n, self.batch_size, candidates)
        else:
            ordered = by_context.permute(1, 0, 2)
            leading = (n, self.batch_size)
        return self._restore(ordered, leading)

    def _context(self, t: Tensor) -> tuple[TreatmentMode, Tensor]:
        """Append categorical `t` to the conditioner, never to the flow event."""
        mode = treatment_mode(t, self.num_treatments, batch_size=self.batch_size)
        one_hot = F.one_hot(t, num_classes=self.num_treatments).to(
            dtype=self._representation.dtype
        )
        if mode == "observed":
            return mode, torch.cat((self._representation, one_hot), dim=-1)
        representation = self._representation[:, None, :].expand(
            -1, int(t.shape[1]), -1
        )
        return mode, torch.cat((representation, one_hot), dim=-1)

    def _event(self, y: Tensor) -> Tensor:
        """Validate unexpanded `y` and flatten only its declared event axes."""
        if y.ndim < 1 or y.shape[0] != self.batch_size:
            axis = None if y.ndim < 1 else int(y.shape[0])
            raise ContractError(
                f"y has batch axis {axis}, but this distribution was built for "
                f"{self.batch_size} rows"
            )
        if tuple(y.shape[1:]) != self.event_shape:
            raise ContractError(
                f"y has trailing shape {tuple(y.shape[1:])}, expected "
                f"{self.event_shape} — pass y unexpanded (DESIGN.md §3.1)"
            )
        return y.reshape(self.batch_size, self.event_dim)

    def _restore(self, values: Tensor, leading: tuple[int, ...]) -> Tensor:
        """Restore `Dy`, removing the artificial width-one scalar event axis."""
        if self.event_shape:
            return values.reshape(*leading, *self.event_shape)
        return values.reshape(*leading)


class ConditionalFlow(Component):
    """Conditional RQ-NSF outcome density with categorical treatment context."""

    mean_base_samples: Tensor

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "widths_description": "architecture.widths_depths",
        "activation": "architecture.activation",
        "normalisation": "architecture.normalisation",
        "dropout": "architecture.dropout",
        "initialisation": "architecture.initialisation",
        "output_parameterisation_description": "architecture.output_parameterisation",
        "treatment_encoding_description": "data.treatment_encoding",
    }

    def __init__(
        self,
        name: str = "conditional_flow",
        *,
        representation_dim: int,
        num_treatments: int,
        outcome: OutcomeSpec,
        num_transforms: int = REQUIRED,
        hidden_features: int = REQUIRED,
        num_blocks: int = REQUIRED,
        use_residual_blocks: bool = REQUIRED,
        num_bins: int = REQUIRED,
        tails: str = REQUIRED,
        tail_bound: float = REQUIRED,
        permutation: str = REQUIRED,
        activation: str = REQUIRED,
        normalisation: str = REQUIRED,
        dropout: float = REQUIRED,
        initialisation: str = REQUIRED,
        base_distribution: str = REQUIRED,
        mean_samples: int = REQUIRED,
    ) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={Port.Y_GIVEN_XT})
        self.num_transforms = num_transforms
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        self.use_residual_blocks = use_residual_blocks
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound
        self.permutation = permutation
        self.activation = activation
        self.normalisation = normalisation
        self.dropout = dropout
        self.initialisation = initialisation
        self.base_distribution = base_distribution
        self.mean_samples = mean_samples
        card_hyperparameters(self)

        owner = type(self).__name__
        self.representation_dim = validate_dimension(
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
                f"{owner} supports only continuous outcomes because an "
                "invertible spline requires a continuous event"
            )
        self.outcome_shape = outcome.shape
        self.event_dim = math.prod(self.outcome_shape) if self.outcome_shape else 1

        self.num_transforms = validate_dimension(
            self.num_transforms, field="num_transforms", owner=owner
        )
        self.hidden_features = validate_dimension(
            self.hidden_features, field="hidden_features", owner=owner
        )
        self.num_blocks = validate_dimension(
            self.num_blocks, field="num_blocks", owner=owner
        )
        self.num_bins = validate_dimension(self.num_bins, field="num_bins", owner=owner)
        self.mean_samples = validate_dimension(
            self.mean_samples, field="mean_samples", owner=owner
        )
        self.dropout = validate_dropout(self.dropout, owner=owner)
        if self.num_transforms != 5:
            raise GraphError(f"{owner}.num_transforms supports only 5")
        if self.hidden_features != 128:
            raise GraphError(f"{owner}.hidden_features supports only 128")
        if self.num_blocks != 2 or self.use_residual_blocks is not True:
            raise GraphError(f"{owner} requires exactly 2 residual blocks")
        if self.num_bins != 8:
            raise GraphError(f"{owner}.num_bins supports only 8")
        if self.tails != "linear":
            raise GraphError(
                f"{owner}.tails must be 'linear'; nflows defaults to the bounded "
                "spline when tails=None"
            )
        raw_tail_bound = cast(object, self.tail_bound)
        if (
            isinstance(raw_tail_bound, bool)
            or not isinstance(raw_tail_bound, int | float)
            or not math.isfinite(float(raw_tail_bound))
            or float(raw_tail_bound) != 3.0
        ):
            raise GraphError(
                f"{owner}.tail_bound supports only 3, got {self.tail_bound!r}"
            )
        self.tail_bound = float(raw_tail_bound)
        if self.permutation != RANDOM_PERMUTATION:
            raise GraphError(
                f"{owner}.permutation supports only {RANDOM_PERMUTATION!r}"
            )
        if self.activation != "relu":
            raise GraphError(f"{owner}.activation supports only 'relu'")
        if self.normalisation != "none":
            raise GraphError(f"{owner}.normalisation supports only 'none'")
        if self.dropout != 0.0:
            raise GraphError(f"{owner}.dropout supports only 0.0")
        if self.initialisation != NFLOWS_INITIALISATION:
            raise GraphError(
                f"{owner}.initialisation supports only {NFLOWS_INITIALISATION!r}"
            )
        if self.base_distribution != STANDARD_NORMAL:
            raise GraphError(
                f"{owner}.base_distribution supports only {STANDARD_NORMAL!r}"
            )
        if self.mean_samples != 100 or self.mean_samples % 2:
            raise GraphError(f"{owner}.mean_samples requires 100 antithetic draws")

        context_features = self.representation_dim + self.num_treatments
        transforms: list[nn.Module] = []
        for _ in range(self.num_transforms):
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=self.event_dim,
                    hidden_features=self.hidden_features,
                    context_features=context_features,
                    num_bins=self.num_bins,
                    tails=self.tails,
                    tail_bound=self.tail_bound,
                    num_blocks=self.num_blocks,
                    use_residual_blocks=self.use_residual_blocks,
                    random_mask=False,
                    activation=F.relu,
                    dropout_probability=self.dropout,
                    use_batch_norm=False,
                )
            )
            transforms.append(RandomPermutation(features=self.event_dim))
        self.transform = cast(_ConditionalTransform, CompositeTransform(transforms))
        self.register_buffer(
            "mean_base_samples",
            _fixed_antithetic_standard_normal(self.mean_samples, self.event_dim),
        )

    @property
    def widths_description(self) -> object:
        """The full spline stack as one component-scoped card value."""
        governed = (
            self.num_transforms,
            self.hidden_features,
            self.num_blocks,
            self.use_residual_blocks,
            self.num_bins,
            self.tails,
            self.tail_bound,
            self.permutation,
        )
        if any(is_required(value) for value in governed):
            return REQUIRED
        return (
            f"{self.num_transforms} RQ-NSF autoregressive transforms, each "
            f"hidden={self.hidden_features} with {self.num_blocks} residual blocks, "
            f'{self.num_bins} bins, tails="{self.tails}", '
            f"tail_bound={float(self.tail_bound):g}; random permutation after "
            "each transform"
        )

    @property
    def output_parameterisation_description(self) -> object:
        """The base, support, event and fixed-mean approximation in the plan."""
        governed = (
            self.base_distribution,
            self.num_transforms,
            self.tails,
            self.tail_bound,
            self.mean_samples,
        )
        if any(is_required(value) for value in governed):
            return REQUIRED
        return (
            f"{self.base_distribution} base -> {self.num_transforms} conditional "
            "RQ-NSF(AR) transforms with explicit linear tails outside "
            f"[-{float(self.tail_bound):g}, {float(self.tail_bound):g}] over "
            "flattened continuous Y; categorical t is one-hot context; "
            f"{self.mean_samples} fixed-antithetic draws approximate mean"
        )

    @property
    def treatment_encoding_description(self) -> str:
        """The categorical treatment belongs to context, never the event."""
        return (
            "one-hot K-vector appended to X_REPR as flow context; never part of "
            "the flow event"
        )

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        representation = ports.tensor(Port.X_REPR)
        return {
            Port.Y_GIVEN_XT: ConditionalFlowOutcome(
                representation=representation,
                transform=self.transform,
                num_treatments=self.num_treatments,
                event_shape=self.outcome_shape,
                mean_base_samples=self.mean_base_samples,
            )
        }


__all__ = [
    "NFLOWS_INITIALISATION",
    "RANDOM_PERMUTATION",
    "STANDARD_NORMAL",
    "ConditionalFlow",
    "ConditionalFlowOutcome",
]
