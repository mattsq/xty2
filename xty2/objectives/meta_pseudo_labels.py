"""Meta Pseudo Labels' sampled score-function mechanics.

The two losses are declarations consumed only by the bounded
``meta_gradient`` executor.  Their ordinary ``compute`` method refuses direct
use because the sampled action and centred feedback coefficient must be shared
across phases of one atomic step; a normal loss mix has no such lifecycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import LossError
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, treatment_distribution
from xty2.core.ports import Port
from xty2.core.rows import RowIndex, Rows


@dataclass
class MetaFeedbackState:
    """Executor-owned moving baseline, fresh for every stage execution."""

    baseline: float


@dataclass(frozen=True)
class MetaFeedbackCoefficient:
    """Centred cosine similarity between pre/post student gradients."""

    kind: str = REQUIRED
    baseline_decay: float = REQUIRED
    baseline_initial: float = REQUIRED
    baseline_order: str = REQUIRED
    force_zero: bool = False
    name: str = "meta_feedback"

    def __post_init__(self) -> None:
        if self.kind != "cosine_similarity":
            raise LossError(
                "MetaFeedbackCoefficient.kind must be 'cosine_similarity', got "
                f"{self.kind!r}"
            )
        if not math.isfinite(float(self.baseline_decay)) or not (
            0.0 <= float(self.baseline_decay) < 1.0
        ):
            raise LossError(
                "MetaFeedbackCoefficient.baseline_decay must be finite in [0, 1)"
            )
        if not math.isfinite(float(self.baseline_initial)):
            raise LossError("MetaFeedbackCoefficient.baseline_initial must be finite")
        if self.baseline_order != "update_then_subtract":
            raise LossError(
                "MetaFeedbackCoefficient.baseline_order must be 'update_then_subtract'"
            )
        if type(self.force_zero) is not bool:
            raise LossError("MetaFeedbackCoefficient.force_zero must be bool")

    def new_state(self) -> MetaFeedbackState:
        return MetaFeedbackState(float(self.baseline_initial))

    def compute(
        self,
        pseudo_gradients: tuple[Tensor | None, ...],
        labelled_gradients: tuple[Tensor | None, ...],
        state: MetaFeedbackState,
    ) -> tuple[Tensor, Tensor, float]:
        """Return ``(h_raw, h, updated_baseline)`` in reviewed order."""
        if len(pseudo_gradients) != len(labelled_gradients):
            raise LossError(
                "meta feedback gradient tuples must have the same parameter arity"
            )
        pairs = [
            (left, right)
            for left, right in zip(pseudo_gradients, labelled_gradients, strict=True)
            if left is not None and right is not None
        ]
        if pairs:
            left = torch.cat([value.reshape(-1) for value, _ in pairs])
            right = torch.cat([value.reshape(-1) for _, value in pairs])
            denominator = left.norm() * right.norm()
            if float(denominator.detach()) == 0.0:
                raw = left.new_zeros(())
            else:
                raw = torch.dot(left, right) / denominator
                raw = raw.clamp(-1.0, 1.0)
        else:
            like = next(
                (
                    value
                    for value in (*pseudo_gradients, *labelled_gradients)
                    if value is not None
                ),
                None,
            )
            raw = torch.tensor(0.0) if like is None else like.new_zeros(())
        decay = float(self.baseline_decay)
        state.baseline = decay * state.baseline + (1.0 - decay) * float(raw.detach())
        centred = (raw - state.baseline).detach()
        if self.force_zero:
            centred = centred.new_zeros(())
        return raw.detach(), centred, state.baseline

    def describe(self) -> tuple[str, ...]:
        control = ", forced h=0 after centring" if self.force_zero else ""
        return (
            f"h_raw = cosine_similarity(pre_inner_grad, post_inner_grad){control}",
            f"baseline b0={float(self.baseline_initial):g}, decay="
            f"{float(self.baseline_decay):g}, update_then_subtract",
            "zero-norm convention = h_raw 0",
        )


@dataclass(frozen=True)
class SampledTeacherTreatmentNLL:
    """Student NLL against one detached categorical draw from the teacher."""

    port: Port
    teacher: Realisation
    student: Realisation
    temperature: float = REQUIRED
    sharpening: str = REQUIRED
    confidence_threshold: str = REQUIRED
    detached_target: str = REQUIRED
    treatment_encoding: str = REQUIRED
    rows: Rows = "t_missing"
    name: str = "student_pseudo_label_nll"

    meta_kind: ClassVar[str] = "sampled_teacher_treatment_nll"
    CARD_KEYS: ClassVar[dict[str, str]] = {
        "temperature": "losses.temperature",
        "sharpening": "losses.sharpening",
        "confidence_threshold": "losses.confidence_threshold",
        "detached_target": "gradients.detached_targets",
        "treatment_encoding": "data.treatment_encoding",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if self.port not in (Port.T_GIVEN_X, Port.T_GIVEN_XY):
            raise LossError("SampledTeacherTreatmentNLL needs a treatment port")
        if self.teacher.role == self.student.role:
            raise LossError("sampled teacher and student roles must be distinct")
        if float(self.temperature) != 1.0:
            raise LossError("hard MPL samples are drawn at temperature 1.0")
        if self.sharpening != "none" or self.confidence_threshold != "none":
            raise LossError("hard MPL samples are neither sharpened nor gated")

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.teacher), (self.port, self.student)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.teacher)})

    @property
    def batch_coupled(self) -> bool:
        return False

    def plan_details(self) -> tuple[str, ...]:
        return (
            "one categorical sample per eligible row at temperature 1",
            "sample uses the explicit hard_label_seed stream",
            "sample and teacher graph are detached from the student update",
            "no sharpening and no confidence gate",
            f"treatment encoding = {self.treatment_encoding}",
        )

    def sample(
        self,
        state: State,
        rows: RowIndex,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        probs = treatment_distribution(
            state, self.port, self.teacher, objective=self.name
        ).probs.detach()
        return torch.multinomial(
            probs.index_select(0, rows), 1, replacement=True, generator=generator
        ).squeeze(1)

    def sampled_loss(self, state: State, rows: RowIndex, sampled: Tensor) -> LossTerm:
        distribution = treatment_distribution(
            state, self.port, self.student, objective=self.name
        )
        probabilities = distribution.probs.index_select(0, rows)
        values = (
            -probabilities.gather(1, sampled[:, None])
            .squeeze(1)
            .clamp_min(torch.finfo(probabilities.dtype).tiny)
            .log()
        )
        if rows.numel() == 0:
            return LossTerm.empty(like=distribution.probs)
        return LossTerm(value=values.mean(), n=int(rows.numel()))

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del state, batch, rows, ctx
        raise LossError(
            "SampledTeacherTreatmentNLL requires executor='meta_gradient' so "
            "its sampled labels can be reused by the teacher score"
        )


@dataclass(frozen=True)
class MetaPseudoLabelScore:
    """Teacher score-function cross-entropy multiplied by detached ``h``."""

    port: Port
    teacher: Realisation
    detached_target: str = REQUIRED
    rows: Rows = "t_missing"
    name: str = "teacher_meta_score"

    meta_kind: ClassVar[str] = "meta_pseudo_label_score"
    CARD_KEYS: ClassVar[dict[str, str]] = {
        "detached_target": "gradients.detached_targets"
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if self.port not in (Port.T_GIVEN_X, Port.T_GIVEN_XY):
            raise LossError("MetaPseudoLabelScore needs a treatment port")
        if not isinstance(self.teacher, Realisation):
            raise LossError("MetaPseudoLabelScore.teacher must be a Realisation")

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.teacher)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        return False

    def plan_details(self) -> tuple[str, ...]:
        return (
            "score-function CE reuses the student's detached categorical sample",
            "centred h is detached; gradient reaches only the outer role",
        )

    def score_loss(self, state: State, rows: RowIndex, sampled: Tensor) -> LossTerm:
        distribution = treatment_distribution(
            state, self.port, self.teacher, objective=self.name
        )
        probabilities = distribution.probs.index_select(0, rows)
        values = (
            -probabilities.gather(1, sampled[:, None])
            .squeeze(1)
            .clamp_min(torch.finfo(probabilities.dtype).tiny)
            .log()
        )
        if rows.numel() == 0:
            return LossTerm.empty(like=distribution.probs)
        return LossTerm(value=values.mean(), n=int(rows.numel()))

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del state, batch, rows, ctx
        raise LossError(
            "MetaPseudoLabelScore requires executor='meta_gradient' so it can "
            "reuse the sampled actions and detached feedback coefficient"
        )


__all__ = [
    "MetaFeedbackCoefficient",
    "MetaFeedbackState",
    "MetaPseudoLabelScore",
    "SampledTeacherTreatmentNLL",
]
