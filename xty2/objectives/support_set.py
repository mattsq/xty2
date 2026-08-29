"""PAWS soft support-set classification and multi-view objectives.

The two objectives deliberately share one frozen ``SupportSetClassifier``.
PAWS's consistency and me-max terms must be functions of the same probabilities;
carrying separate temperature and label-smoothing literals would make drift
possible while still producing a plausible plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import LossError, PortContractError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows
from xty2.core.ports import Port
from xty2.core.rows import RowIndex, Rows, resolve_rows, validate_population

TargetRole = Literal["target"]


@dataclass(frozen=True)
class SupportSetClassifier:
    """PAWS's soft nearest-neighbour classifier ``pi_d(z, z_S)``.

    Support embeddings from every declared support realisation are concatenated;
    their observed labels are repeated in the same order and smoothed before
    the similarity weights are applied.
    """

    temperature: float = REQUIRED
    label_smoothing: float = REQUIRED
    support_rows: Rows = "t_observed"

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, int | float
        ):
            raise LossError(
                "SupportSetClassifier.temperature must be a number, got "
                f"{type(self.temperature)}"
            )
        if float(self.temperature) <= 0.0:
            raise LossError(
                "SupportSetClassifier.temperature divides cosine similarities "
                f"and must be positive, got {self.temperature!r}"
            )
        if isinstance(self.label_smoothing, bool) or not isinstance(
            self.label_smoothing, int | float
        ):
            raise LossError(
                "SupportSetClassifier.label_smoothing must be a number, got "
                f"{type(self.label_smoothing)}"
            )
        if not 0.0 <= float(self.label_smoothing) < 1.0:
            raise LossError(
                "SupportSetClassifier.label_smoothing must be in [0, 1), got "
                f"{self.label_smoothing!r}"
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "label_smoothing", float(self.label_smoothing))
        try:
            validate_population(self.support_rows)
        except Xty2Error as error:
            raise LossError(f"SupportSetClassifier: {error}") from error

    def probabilities(
        self,
        state: State,
        batch: XTYBatch,
        queries: Realisation,
        supports: tuple[Realisation, ...],
        query_rows: RowIndex,
        *,
        classes: int,
        detach: bool = False,
    ) -> Tensor:
        """Return ``[n, K]`` support-weighted class probabilities."""
        if not supports:
            raise LossError("SupportSetClassifier needs at least one support view")
        support_rows = resolve_rows(batch, self.support_rows)
        if support_rows.numel() == 0:
            raise LossError(
                "SupportSetClassifier has no support rows in this batch; its "
                f"declared population is {self.support_rows!r}"
            )
        query = _embedding(state, queries, batch, owner="SupportSetClassifier")
        support = torch.cat(
            [
                _embedding(
                    state, realisation, batch, owner="SupportSetClassifier"
                ).index_select(0, support_rows)
                for realisation in supports
            ],
            dim=0,
        )
        query = query.index_select(0, query_rows)
        labels = batch.t.index_select(0, support_rows).repeat(len(supports))
        if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= classes):
            raise LossError(
                "SupportSetClassifier support labels fall outside the schema's "
                f"0..{classes - 1} range"
            )
        smoothed = self.smoothed_labels(labels, classes=classes).to(dtype=support.dtype)
        if detach:
            query = query.detach()
            support = support.detach()
            smoothed = smoothed.detach()
        query = F.normalize(query, dim=-1)
        support = F.normalize(support, dim=-1)
        weights = torch.softmax(
            query @ support.transpose(0, 1) / self.temperature, dim=-1
        )
        return weights @ smoothed

    def smoothed_labels(self, labels: Tensor, *, classes: int) -> Tensor:
        """The reference's ``(1-s)y + s/K`` support-label matrix."""
        if classes < 2:
            raise LossError(
                f"SupportSetClassifier needs at least two classes, got {classes}"
            )
        one_hot = F.one_hot(labels, num_classes=classes).to(dtype=torch.float32)
        smoothing = self.label_smoothing
        return (1.0 - smoothing) * one_hot + smoothing / classes


@dataclass(frozen=True)
class SupportSetPseudoLabelConsistency:
    """PAWS eq. (4): two swapped large targets and six mean-target views."""

    classifier: SupportSetClassifier
    large: tuple[Realisation, Realisation]
    small: tuple[Realisation, ...]
    sharpening: float = REQUIRED
    stop_grad: TargetRole = REQUIRED
    target_floor: float = REQUIRED
    rows: Rows = "t_missing"
    name: str = "support_set_pseudo_label_consistency"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "temperature": "losses.temperature",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        _validate_objective(
            name=self.name,
            classifier=self.classifier,
            large=self.large,
            small=self.small,
            sharpening=self.sharpening,
            target_floor=self.target_floor,
            rows=self.rows,
        )
        if self.stop_grad != "target":
            raise LossError(
                "SupportSetPseudoLabelConsistency.stop_grad must be 'target': "
                "PAWS differentiates predictions and support anchors, not its "
                f"sharpened target role; got {self.stop_grad!r}"
            )
        object.__setattr__(self, "sharpening", float(self.sharpening))
        object.__setattr__(self, "target_floor", float(self.target_floor))

    @property
    def temperature(self) -> float:
        return self.classifier.temperature

    @property
    def support_rows(self) -> Rows:
        return self.classifier.support_rows

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            (Port.X_PROJ, realisation) for realisation in (*self.large, *self.small)
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        # The detach is role-level: each large realisation is also an
        # undetached prediction, so no whole (port, realisation) is detached.
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"support rows = {self.support_rows}",
            f"support views = {self.large[0]}, {self.large[1]}",
            "large targets = the other large view; small targets = mean of both",
            "target role = detached; prediction and anchor-support roles train",
            f"support label smoothing = {self.classifier.label_smoothing!r}",
            f"target probabilities below {self.target_floor!r} are set to zero",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        if rows.numel() == 0:
            like = _embedding(state, self.large[0], batch, owner=self.name)
            return LossTerm.empty(like=like)
        prediction = [
            self.classifier.probabilities(
                state,
                batch,
                realisation,
                self.large,
                rows,
                classes=ctx.schema.treatment_cardinality,
            )
            for realisation in (*self.large, *self.small)
        ]
        large_targets = [
            _sharpen(
                self.classifier.probabilities(
                    state,
                    batch,
                    realisation,
                    self.large,
                    rows,
                    classes=ctx.schema.treatment_cardinality,
                    detach=True,
                ),
                self.sharpening,
            )
            for realisation in self.large
        ]
        mean_target = (large_targets[0] + large_targets[1]) / 2.0
        targets = [large_targets[1], large_targets[0]] + [mean_target] * len(self.small)
        targets = [
            target.masked_fill(target < self.target_floor, 0.0) for target in targets
        ]
        cross_entropy = torch.stack(
            [
                -(torch.xlogy(target, probability)).sum(dim=-1)
                for target, probability in zip(targets, prediction, strict=True)
            ],
            dim=0,
        ).mean(dim=0)
        per_row = torch.zeros(
            batch.batch_size,
            dtype=cross_entropy.dtype,
            device=cross_entropy.device,
        ).index_copy(0, rows, cross_entropy)
        return reduce_rows(
            per_row,
            rows,
            diagnostics={
                "target_entropy": float(
                    _entropy(torch.stack(large_targets).mean(dim=(0, 1))).detach()
                )
            },
        )


@dataclass(frozen=True)
class MeanEntropyMaximisation:
    """PAWS me-max, ``-H(p_bar)`` over all prediction rows and views."""

    classifier: SupportSetClassifier
    views: tuple[Realisation, ...]
    support_views: tuple[Realisation, ...]
    sharpening: float = REQUIRED
    rows: Rows = "t_missing"
    name: str = "mean_entropy_maximisation"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "temperature": "losses.temperature",
        "sharpening": "losses.sharpening",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("me-max objective name", self.name, error=LossError):
            raise LossError("MeanEntropyMaximisation.name must be non-empty")
        if not isinstance(self.classifier, SupportSetClassifier):
            raise LossError(
                "MeanEntropyMaximisation.classifier must be a "
                f"SupportSetClassifier, got {type(self.classifier)}"
            )
        _validate_realisations("MeanEntropyMaximisation.views", self.views)
        _validate_realisations(
            "MeanEntropyMaximisation.support_views", self.support_views
        )
        _validate_sharpening(self.sharpening)
        object.__setattr__(self, "sharpening", float(self.sharpening))
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(
                f"MeanEntropyMaximisation {self.name!r}: {error}"
            ) from error

    @property
    def temperature(self) -> float:
        return self.classifier.temperature

    @property
    def support_rows(self) -> Rows:
        return self.classifier.support_rows

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            (Port.X_PROJ, realisation)
            for realisation in (*self.views, *self.support_views)
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"support rows = {self.support_rows}",
            f"support label smoothing = {self.classifier.label_smoothing!r}",
            f"p_bar = mean sharpened probability over {len(self.views)} views",
            "me-max reads undetached prediction and anchor-support roles",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        if rows.numel() == 0:
            like = _embedding(state, self.views[0], batch, owner=self.name)
            return LossTerm.empty(like=like)
        probabilities = [
            _sharpen(
                self.classifier.probabilities(
                    state,
                    batch,
                    realisation,
                    self.support_views,
                    rows,
                    classes=ctx.schema.treatment_cardinality,
                ),
                self.sharpening,
            )
            for realisation in self.views
        ]
        marginal = torch.stack(probabilities, dim=0).mean(dim=(0, 1))
        entropy = _entropy(marginal)
        return LossTerm(
            value=-entropy,
            n=int(rows.numel()),
            diagnostics={"marginal_entropy": float(entropy.detach())},
        )


def _validate_objective(
    *,
    name: str,
    classifier: object,
    large: object,
    small: object,
    sharpening: object,
    target_floor: object,
    rows: Rows,
) -> None:
    if not require_str("support consistency objective name", name, error=LossError):
        raise LossError("SupportSetPseudoLabelConsistency.name must be non-empty")
    if not isinstance(classifier, SupportSetClassifier):
        raise LossError(
            "SupportSetPseudoLabelConsistency.classifier must be a "
            f"SupportSetClassifier, got {type(classifier)}"
        )
    if not isinstance(large, tuple) or len(large) != 2:
        raise LossError(
            "SupportSetPseudoLabelConsistency.large must hold exactly two Realisations"
        )
    _validate_realisations("SupportSetPseudoLabelConsistency.large", large)
    if not isinstance(small, tuple):
        raise LossError("SupportSetPseudoLabelConsistency.small must be a tuple")
    _validate_realisations("SupportSetPseudoLabelConsistency.small", small)
    if len(set((*large, *small))) != len((*large, *small)):
        raise LossError("PAWS prediction realisations must be distinct")
    _validate_sharpening(sharpening)
    if isinstance(target_floor, bool) or not isinstance(target_floor, int | float):
        raise LossError(
            "SupportSetPseudoLabelConsistency.target_floor must be a number"
        )
    if not 0.0 <= float(target_floor) < 1.0:
        raise LossError(
            "SupportSetPseudoLabelConsistency.target_floor must be in [0, 1), "
            f"got {target_floor!r}"
        )
    try:
        validate_population(rows)
    except Xty2Error as error:
        raise LossError(
            f"SupportSetPseudoLabelConsistency {name!r}: {error}"
        ) from error


def _validate_realisations(label: str, realisations: object) -> None:
    if not isinstance(realisations, tuple) or not realisations:
        raise LossError(f"{label} must be a non-empty tuple of Realisations")
    if any(not isinstance(realisation, Realisation) for realisation in realisations):
        raise LossError(f"{label} must contain only Realisations")


def _validate_sharpening(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LossError(f"sharpening must be a number, got {type(value)}")
    if float(value) <= 0.0:
        raise LossError(f"sharpening must be positive, got {value!r}")


def _embedding(
    state: State, realisation: Realisation, batch: XTYBatch, *, owner: str
) -> Tensor:
    value = state[realisation][Port.X_PROJ]
    if not isinstance(value, Tensor):
        raise PortContractError(
            f"{owner} read {Port.X_PROJ} under {realisation} as an embedding "
            f"tensor, but it carries {type(value)}"
        )
    if value.ndim != 2 or value.shape[0] != batch.batch_size:
        raise LossError(
            f"{owner} got embedding shape {tuple(value.shape)} under "
            f"{realisation} for a batch of {batch.batch_size}"
        )
    return value


def _sharpen(probabilities: Tensor, temperature: float) -> Tensor:
    powered = probabilities.pow(1.0 / temperature)
    return powered / powered.sum(dim=-1, keepdim=True)


def _entropy(probabilities: Tensor) -> Tensor:
    return -(torch.xlogy(probabilities, probabilities)).sum()


__all__ = [
    "MeanEntropyMaximisation",
    "SupportSetClassifier",
    "SupportSetPseudoLabelConsistency",
    "TargetRole",
]
