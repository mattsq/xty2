"""UDA's sharpened consistency and training-signal annealing objectives.

The two objectives share one declared threshold policy because UDA has two
confidence gates under the single ``losses.confidence_threshold`` card key:
the fixed weak-target confidence gate of section 2.4 and the step-dependent
true-class ceiling of appendix A.1.  Keeping the policy as one immutable value
makes both numbers enter the plan without inventing another card key.

The denominator conventions deliberately differ.  UDA masks rejected
unlabelled rows and averages over the whole eligible population.  TSA masks
easy labelled rows and divides by the retained count, clamped at one.  Both
still return ``LossTerm.n`` as the declared eligible-row count; the exceptional
TSA denominator is objective arithmetic and is printed by ``plan_details()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import LossError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import (
    LossTerm,
    TrainContext,
    reduce_rows,
    treatment_at,
    treatment_distribution,
)
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population

UDASchedule = Literal["exp_schedule"]
UDASharpening = Literal["softmax_temperature"]
UDAStopGrad = Literal["target"]
UDADivergence = Literal["kl"]


@dataclass(frozen=True, repr=False)
class UDAConfidenceThresholds:
    """Section 2.4's fixed gate and appendix A.1's TSA ceiling.

    Every field is required: ``scale=5`` is part of the source formula and
    ``total_steps`` fixes the unit of its ``t/T``.  A recipe therefore cannot
    silently inherit either while still claiming the reviewed policy.
    """

    unsupervised: float
    tsa_schedule: UDASchedule
    scale: float
    total_steps: int

    def __post_init__(self) -> None:
        if isinstance(self.unsupervised, bool) or not isinstance(
            self.unsupervised, int | float
        ):
            raise LossError(
                "UDAConfidenceThresholds.unsupervised must be a number in "
                f"[0, 1], got {type(self.unsupervised)}"
            )
        if not 0.0 <= float(self.unsupervised) <= 1.0:
            raise LossError(
                "UDAConfidenceThresholds.unsupervised must be in [0, 1], got "
                f"{self.unsupervised!r}"
            )
        if self.tsa_schedule != "exp_schedule":
            raise LossError(
                "UDAConfidenceThresholds.tsa_schedule must be 'exp_schedule', "
                f"got {self.tsa_schedule!r}. A second schedule arrives with a "
                "reviewed recipe variant that uses it."
            )
        if isinstance(self.scale, bool) or not isinstance(self.scale, int | float):
            raise LossError(
                "UDAConfidenceThresholds.scale must be a positive number, got "
                f"{type(self.scale)}"
            )
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise LossError(
                "UDAConfidenceThresholds.scale must be finite and positive, got "
                f"{self.scale!r}"
            )
        if type(self.total_steps) is not int or self.total_steps < 1:
            raise LossError(
                "UDAConfidenceThresholds.total_steps must be an integer at "
                f"least 1, got {self.total_steps!r}"
            )
        object.__setattr__(self, "unsupervised", float(self.unsupervised))
        object.__setattr__(self, "scale", float(self.scale))

    def __repr__(self) -> str:
        return (
            f"uda(unsupervised={self.unsupervised:g}, "
            f"tsa={self.tsa_schedule}(scale={self.scale:g}, "
            f"steps={self.total_steps}))"
        )

    def tsa_ceiling(self, step: int, num_treatments: int) -> float:
        """Appendix A.1 and ``image/main.py:get_tsa_threshold`` exactly."""
        if type(step) is not int or step < 0:
            raise LossError(f"TSA step must be a non-negative int, got {step!r}")
        if type(num_treatments) is not int or num_treatments < 2:
            raise LossError(
                f"TSA needs at least two treatment levels, got {num_treatments!r}"
            )
        progress = step / self.total_steps
        coefficient = math.exp((progress - 1.0) * self.scale)
        start = 1.0 / num_treatments
        return coefficient * (1.0 - start) + start

    def describe(self) -> tuple[str, ...]:
        """Stable plan lines for the two gates the value represents."""
        return (
            f"UDA gate = max untempered target probability > {self.unsupervised:g}",
            "TSA ceiling = exp("
            f"{self.scale:g} * (step/{self.total_steps} - 1)) * (1 - 1/K) + 1/K",
        )


@dataclass(frozen=True)
class ConfidenceMaskedConsistencyLoss:
    """UDA's gated ``KL(sharpen(target) || prediction)``.

    The target gate reads the ordinary weak probabilities before sharpening,
    the target branch is detached, and rejected rows remain in the denominator.
    """

    port: Port
    target: Realisation
    prediction: Realisation
    thresholds: UDAConfidenceThresholds = REQUIRED
    target_temperature: float = REQUIRED
    sharpening: UDASharpening = REQUIRED
    stop_grad: UDAStopGrad = REQUIRED
    divergence: UDADivergence = "kl"
    rows: Rows = "t_missing"
    name: str = "uda_consistency"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "thresholds": "losses.confidence_threshold",
        "target_temperature": "losses.temperature",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str(
            "UDA consistency objective name", self.name, error=LossError
        ):
            raise LossError("ConfidenceMaskedConsistencyLoss.name must be non-empty")
        if not isinstance(self.port, Port):
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.port must be a Port, got "
                f"{type(self.port)}"
            )
        if port_spec(self.port).kind != "treatment_distribution":
            raise LossError(
                "ConfidenceMaskedConsistencyLoss compares treatment "
                f"distributions, but port {self.port!s} carries "
                f"{port_spec(self.port).kind}"
            )
        target: object = self.target
        prediction: object = self.prediction
        if not isinstance(target, Realisation) or not isinstance(
            prediction, Realisation
        ):
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.target and prediction must be "
                "Realisations"
            )
        if self.target == self.prediction:
            raise LossError(
                "ConfidenceMaskedConsistencyLoss compares one realisation with "
                "itself; UDA requires an augmented prediction view"
            )
        if not isinstance(self.thresholds, UDAConfidenceThresholds):
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.thresholds must be a "
                f"UDAConfidenceThresholds, got {type(self.thresholds)}"
            )
        if isinstance(self.target_temperature, bool) or not isinstance(
            self.target_temperature, int | float
        ):
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.target_temperature must be a "
                f"positive number, got {type(self.target_temperature)}"
            )
        if (
            not math.isfinite(float(self.target_temperature))
            or float(self.target_temperature) <= 0.0
        ):
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.target_temperature must be "
                f"finite and positive, got {self.target_temperature!r}"
            )
        object.__setattr__(self, "target_temperature", float(self.target_temperature))
        if self.sharpening != "softmax_temperature":
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.sharpening must be "
                f"'softmax_temperature', got {self.sharpening!r}"
            )
        if self.stop_grad != "target":
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.stop_grad must be 'target', "
                f"got {self.stop_grad!r}"
            )
        if self.divergence != "kl":
            raise LossError(
                "ConfidenceMaskedConsistencyLoss.divergence must be 'kl', got "
                f"{self.divergence!r}"
            )
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(
                f"ConfidenceMaskedConsistencyLoss {self.name!r}: {error}"
            ) from error

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.target), (self.port, self.prediction)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.target)})

    def plan_details(self) -> tuple[str, ...]:
        return (
            "divergence = 'kl'",
            *self.thresholds.describe()[:1],
            f"target = softmax(log(p_target) / {self.target_temperature:g})",
            "only the target realisation is detached",
            "denominator = every eligible row; rejected rows contribute 0",
        )

    @property
    def batch_coupled(self) -> bool:
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        untempered = treatment_distribution(
            state, self.port, self.target, objective=self.name
        ).probs.detach()
        prediction = treatment_distribution(
            state, self.port, self.prediction, objective=self.name
        ).probs
        _check_probability_shapes(untempered, prediction, batch, self.name)

        target = _temperature_sharpen(untempered, self.target_temperature)
        confidence = untempered.max(dim=-1).values
        accepted = confidence > self.thresholds.unsupervised
        tiny = torch.finfo(target.dtype).tiny
        per_row = (
            target * (target.clamp_min(tiny).log() - prediction.clamp_min(tiny).log())
        ).sum(dim=-1)
        per_row = per_row * accepted.to(per_row.dtype)
        return reduce_rows(
            per_row,
            rows,
            diagnostics=_consistency_diagnostics(confidence, accepted, target, rows),
        )


@dataclass(frozen=True)
class TrainingSignalAnnealedTreatmentNLL:
    """Observed-treatment NLL under appendix A.1's true-class ceiling."""

    port: Port
    realisation: Realisation
    thresholds: UDAConfidenceThresholds = REQUIRED
    name: str = "tsa_observed_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {"thresholds": "losses.confidence_threshold"}

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("TSA objective name", self.name, error=LossError):
            raise LossError("TrainingSignalAnnealedTreatmentNLL.name must be non-empty")
        if self.port not in (Port.T_GIVEN_X, Port.T_GIVEN_XY):
            raise LossError(
                "TrainingSignalAnnealedTreatmentNLL.port must be T_GIVEN_X or "
                f"T_GIVEN_XY, got {self.port!r}"
            )
        if not isinstance(self.realisation, Realisation):
            raise LossError(
                "TrainingSignalAnnealedTreatmentNLL.realisation must be a "
                f"Realisation, got {type(self.realisation)}"
            )
        if not isinstance(self.thresholds, UDAConfidenceThresholds):
            raise LossError(
                "TrainingSignalAnnealedTreatmentNLL.thresholds must be a "
                f"UDAConfidenceThresholds, got {type(self.thresholds)}"
            )

    @property
    def rows(self) -> Rows:
        return "t_observed"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.realisation)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    def plan_details(self) -> tuple[str, ...]:
        return (
            self.thresholds.describe()[1],
            "gate = retain true-class probability <= TSA ceiling; suppress on strict >",
            "denominator = retained eligible rows, clamped at 1",
        )

    @property
    def batch_coupled(self) -> bool:
        """Yes: the source retained-count denominator couples row scaling."""
        return True

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        propensity = treatment_distribution(
            state, self.port, self.realisation, objective=self.name
        )
        probs = propensity.probs
        if probs.shape[0] != batch.batch_size:
            raise LossError(
                f"TrainingSignalAnnealedTreatmentNLL {self.name!r} got "
                f"{probs.shape[0]} rows for a batch of {batch.batch_size}"
            )
        if probs.shape[1] != ctx.schema.treatment_cardinality:
            raise LossError(
                f"TrainingSignalAnnealedTreatmentNLL {self.name!r} got "
                f"{probs.shape[1]} classes but the schema declares "
                f"{ctx.schema.treatment_cardinality}"
            )
        labels = treatment_at(batch, rows)
        per_row = -propensity.log_prob(labels)
        if rows.numel() == 0:
            return LossTerm.empty(like=per_row)

        correct_probability = probs.gather(1, labels[:, None]).squeeze(1)
        ceiling = self.thresholds.tsa_ceiling(
            ctx.global_step, ctx.schema.treatment_cardinality
        )
        retained = correct_probability <= ceiling
        eligible_loss = per_row.index_select(0, rows)
        eligible_retained = retained.index_select(0, rows)
        retained_count = int(eligible_retained.sum())
        value = (eligible_loss * eligible_retained.to(eligible_loss.dtype)).sum() / max(
            retained_count, 1
        )
        return LossTerm(
            value=value,
            n=int(rows.numel()),
            diagnostics={
                "retained_fraction": retained_count / int(rows.numel()),
                "tsa_ceiling": ceiling,
            },
        )


def _temperature_sharpen(probs: Tensor, temperature: float) -> Tensor:
    """``softmax(logits / tau)`` from probabilities, exactly at ``tau=1``."""
    if temperature == 1.0:
        return probs
    tiny = torch.finfo(probs.dtype).tiny
    return torch.softmax(probs.clamp_min(tiny).log() / temperature, dim=-1)


def _check_probability_shapes(
    target: Tensor, prediction: Tensor, batch: XTYBatch, name: str
) -> None:
    if target.shape != prediction.shape:
        raise LossError(
            f"ConfidenceMaskedConsistencyLoss {name!r} got target shape "
            f"{tuple(target.shape)} and prediction shape {tuple(prediction.shape)}"
        )
    if target.ndim != 2 or target.shape[0] != batch.batch_size:
        raise LossError(
            f"ConfidenceMaskedConsistencyLoss {name!r} needs [B, K] "
            f"probabilities for B={batch.batch_size}, got {tuple(target.shape)}"
        )


def _consistency_diagnostics(
    confidence: Tensor, accepted: Tensor, target: Tensor, rows: RowIndex
) -> dict[str, float]:
    if rows.numel() == 0:
        return {}
    eligible_accepted = accepted.index_select(0, rows)
    retained = int(eligible_accepted.sum())
    accepted_confidence = 0.0
    if retained:
        accepted_confidence = float(
            confidence.index_select(0, rows)[eligible_accepted].mean()
        )
    eligible_target = target.index_select(0, rows)
    tiny = torch.finfo(target.dtype).tiny
    entropy = -(eligible_target * eligible_target.clamp_min(tiny).log()).sum(dim=-1)
    return {
        "coverage": retained / int(rows.numel()),
        "accepted_confidence": accepted_confidence,
        "target_entropy": float(entropy.mean()),
    }


__all__ = [
    "ConfidenceMaskedConsistencyLoss",
    "TrainingSignalAnnealedTreatmentNLL",
    "UDAConfidenceThresholds",
    "UDADivergence",
    "UDASchedule",
    "UDASharpening",
    "UDAStopGrad",
]
