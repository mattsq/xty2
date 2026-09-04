"""SoftMatch's truncated-Gaussian pseudo-label weighting (paper eqs. 2, 5-9)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.data import TrainingPopulation
from xty2.core.errors import LossError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows, treatment_distribution
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population

Alignment = Literal["uniform", "none"]


@dataclass(frozen=True, repr=False)
class TruncatedGaussianWeighting:
    """The all-class SoftMatch policy from eqs. (5)-(9).

    ``n_sigma`` implements the paper's variance-range convention: the EMA
    variance is divided by ``n_sigma ** 2`` in the Gaussian denominator.
    Uniform Alignment affects only the confidence used for the weight; the
    pseudo-label always comes from the original prediction.
    """

    decay: float
    n_sigma: float
    alignment: Alignment

    def __post_init__(self) -> None:
        for name, value in (("decay", self.decay), ("n_sigma", self.n_sigma)):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise LossError(
                    f"TruncatedGaussianWeighting.{name} must be numeric, got "
                    f"{type(value)}"
                )
        if not 0.0 < float(self.decay) < 1.0:
            raise LossError(
                "TruncatedGaussianWeighting.decay must be in (0, 1), got "
                f"{self.decay!r}"
            )
        if float(self.n_sigma) <= 0.0:
            raise LossError(
                "TruncatedGaussianWeighting.n_sigma must be positive, got "
                f"{self.n_sigma!r}"
            )
        if self.alignment not in ("uniform", "none"):
            raise LossError(
                "TruncatedGaussianWeighting.alignment must be 'uniform' or "
                f"'none', got {self.alignment!r}"
            )
        object.__setattr__(self, "decay", float(self.decay))
        object.__setattr__(self, "n_sigma", float(self.n_sigma))

    def __repr__(self) -> str:
        return (
            f"truncated_gaussian(decay={self.decay:g}, "
            f"n_sigma={self.n_sigma:g}, alignment={self.alignment})"
        )

    def describe(self) -> tuple[str, ...]:
        return (
            f"mu_hat and sigma_hat^2 use EMA decay {self.decay:g} (eq. 7)",
            "sigma_hat^2 uses the unbiased B_U/(B_U-1) batch variance",
            f"Gaussian denominator = 2 * sigma_hat^2 / {self.n_sigma:g}^2",
            f"weight confidence alignment = {self.alignment}",
            "uniform alignment target = u(K); pseudo-label remains unaligned",
        )


class ConfidenceGaussian:
    """Executor-owned EMAs for confidence mean, variance, and class marginal."""

    __slots__ = (
        "_classes",
        "_last_rows",
        "_last_step",
        "_marginal",
        "_mean",
        "_policy",
        "_variance",
    )

    def __init__(self, num_treatments: int, policy: TruncatedGaussianWeighting) -> None:
        if isinstance(num_treatments, bool) or not isinstance(num_treatments, int):
            raise LossError(
                "ConfidenceGaussian.num_treatments must be an int, got "
                f"{type(num_treatments)}"
            )
        if num_treatments < 2:
            raise LossError(f"ConfidenceGaussian needs K >= 2, got {num_treatments}")
        self._classes = num_treatments
        self._policy = policy
        uniform = 1.0 / num_treatments
        self._mean = torch.tensor(uniform, dtype=torch.float64)
        self._variance = torch.tensor(1.0, dtype=torch.float64)
        self._marginal = torch.full((num_treatments,), uniform, dtype=torch.float64)
        self._last_step: int | None = None
        self._last_rows = 0

    @property
    def classes(self) -> int:
        return self._classes

    @property
    def mean(self) -> float:
        return float(self._mean)

    @property
    def variance(self) -> float:
        return float(self._variance)

    @property
    def marginal(self) -> Tensor:
        return self._marginal.clone()

    @property
    def last_observed_step(self) -> int | None:
        return self._last_step

    def observe(self, step: int, probs: Tensor) -> None:
        """Fold the current unaligned weak-view batch in before weighting it."""
        if probs.ndim != 2 or probs.shape[1] != self._classes:
            raise LossError(
                f"ConfidenceGaussian.observe needs [n, {self._classes}] "
                f"probabilities, got {tuple(probs.shape)}"
            )
        if probs.shape[0] == 0:
            return
        if probs.shape[0] < 2:
            raise LossError(
                "SoftMatch's unbiased confidence variance needs at least two "
                f"eligible rows, got {probs.shape[0]}"
            )
        if self._last_step is not None and step <= self._last_step:
            if step == self._last_step and probs.shape[0] != self._last_rows:
                raise LossError(
                    f"ConfidenceGaussian observed two row counts at step {step}: "
                    f"{self._last_rows} and {probs.shape[0]}"
                )
            return
        batch = probs.detach().to(torch.float64)
        confidence = batch.max(dim=-1).values
        decay = self._policy.decay
        self._mean = decay * self._mean + (1.0 - decay) * confidence.mean()
        self._variance = decay * self._variance + (
            (1.0 - decay) * confidence.var(unbiased=True)
        )
        self._marginal = decay * self._marginal + (1.0 - decay) * batch.mean(dim=0)
        self._last_step = step
        self._last_rows = int(batch.shape[0])

    def aligned(self, probs: Tensor) -> Tensor:
        """Eq. (8), or the declared no-UA ablation."""
        if self._policy.alignment == "none":
            return probs
        marginal = self._marginal.to(device=probs.device, dtype=probs.dtype)
        aligned = probs * ((1.0 / self._classes) / marginal)
        return aligned / aligned.sum(dim=-1, keepdim=True)

    def weights(self, probs: Tensor) -> Tensor:
        """Eq. (9), with ``lambda_max`` factored into the mixer weight."""
        confidence = self.aligned(probs).max(dim=-1).values
        mean = self._mean.to(device=probs.device, dtype=probs.dtype)
        variance = self._variance.to(device=probs.device, dtype=probs.dtype)
        delta = torch.clamp(confidence - mean, max=0.0)
        denominator = 2.0 * variance / (self._policy.n_sigma**2)
        return torch.exp(-(delta.square() / denominator))


@dataclass(frozen=True)
class SoftWeightedTreatmentNLL:
    """Eq. (2): hard weak-view labels weighted by eq. (9)."""

    port: Port
    target: Realisation
    prediction: Realisation
    num_treatments: int
    weighting: TruncatedGaussianWeighting = REQUIRED
    sharpening: Literal["hard"] = REQUIRED
    stop_grad: Literal["target"] = REQUIRED
    rows: Rows = "all"
    name: str = "soft_weighted_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "weighting": "losses.confidence_threshold",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("soft-weighted objective name", self.name, error=LossError):
            raise LossError("SoftWeightedTreatmentNLL.name must be non-empty")
        if not isinstance(self.port, Port):
            raise LossError(
                f"SoftWeightedTreatmentNLL.port must be a Port, got {type(self.port)}"
            )
        if port_spec(self.port).kind != "treatment_distribution":
            raise LossError(
                "SoftWeightedTreatmentNLL requires a treatment-distribution port"
            )
        target: object = self.target
        prediction: object = self.prediction
        if not isinstance(target, Realisation) or not isinstance(
            prediction, Realisation
        ):
            raise LossError(
                "SoftWeightedTreatmentNLL.target and prediction must be Realisations"
            )
        if self.target == self.prediction:
            raise LossError(
                "SoftWeightedTreatmentNLL needs distinct weak target and strong "
                "prediction realisations"
            )
        if not isinstance(self.weighting, TruncatedGaussianWeighting):
            raise LossError(
                "SoftWeightedTreatmentNLL.weighting must be a "
                f"TruncatedGaussianWeighting, got {type(self.weighting)}"
            )
        if self.sharpening != "hard":
            raise LossError(
                "SoftWeightedTreatmentNLL.sharpening must be 'hard', got "
                f"{self.sharpening!r}"
            )
        if self.stop_grad != "target":
            raise LossError(
                "SoftWeightedTreatmentNLL.stop_grad must be 'target', got "
                f"{self.stop_grad!r}"
            )
        if isinstance(self.num_treatments, bool) or not isinstance(
            self.num_treatments, int
        ):
            raise LossError(
                "SoftWeightedTreatmentNLL.num_treatments must be an int, got "
                f"{type(self.num_treatments)}"
            )
        if self.num_treatments < 2:
            raise LossError(
                "SoftWeightedTreatmentNLL.num_treatments must be at least 2"
            )
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(
                f"SoftWeightedTreatmentNLL {self.name!r}: {error}"
            ) from error

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.target), (self.port, self.prediction)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.target)})

    @property
    def batch_coupled(self) -> bool:
        return True

    def initial_state(self, population: TrainingPopulation | None) -> object:
        del population
        return ConfidenceGaussian(self.num_treatments, self.weighting)

    def plan_details(self) -> tuple[str, ...]:
        return (
            "label = arg max of the unaligned target realisation",
            "weight = truncated Gaussian of aligned confidence (eq. 9)",
            *self.weighting.describe(),
            "all three EMAs fold in this batch before this batch is weighted",
            "denominator = every eligible row; weights multiply inside the mean",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        gaussian = ctx.objective_state(self.name, ConfidenceGaussian)
        target = treatment_distribution(
            state, self.port, self.target, objective=self.name
        ).probs.detach()
        prediction = treatment_distribution(
            state, self.port, self.prediction, objective=self.name
        )
        if target.shape[1] != self.num_treatments:
            raise LossError(
                f"SoftWeightedTreatmentNLL expected K={self.num_treatments}, "
                f"got target shape {tuple(target.shape)}"
            )
        if self.num_treatments != ctx.schema.treatment_cardinality:
            raise LossError(
                "SoftWeightedTreatmentNLL.num_treatments disagrees with the schema"
            )
        gaussian.observe(ctx.global_step, target.index_select(0, rows))
        confidence, labels = target.max(dim=-1)
        weights = gaussian.weights(target)
        per_row = -prediction.log_prob(labels) * weights
        if per_row.shape[0] != batch.batch_size:
            raise LossError(
                f"SoftWeightedTreatmentNLL got {per_row.shape[0]} rows for a "
                f"batch of {batch.batch_size}"
            )
        diagnostics: dict[str, float] = {}
        if rows.numel():
            eligible_weights = weights.index_select(0, rows)
            aligned = gaussian.aligned(target).index_select(0, rows)
            diagnostics = {
                "quantity": float(eligible_weights.mean()),
                "weight_min": float(eligible_weights.min()),
                "weight_max": float(eligible_weights.max()),
                "confidence_mean": float(confidence.index_select(0, rows).mean()),
                "aligned_confidence_mean": float(aligned.max(dim=-1).values.mean()),
                "mu_hat": gaussian.mean,
                "sigma_squared": gaussian.variance,
            }
        return reduce_rows(per_row, rows, diagnostics=diagnostics)


__all__ = [
    "Alignment",
    "ConfidenceGaussian",
    "SoftWeightedTreatmentNLL",
    "TruncatedGaussianWeighting",
]
