"""Distribution-aligned anchor targets, pooled MixUp, and pretext loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.data import TrainingPopulation
from xty2.core.errors import LossError, PortContractError, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows, treatment_distribution
from xty2.core.mixing import MixingPlan
from xty2.core.ports import Port
from xty2.core.rows import RowIndex, Rows, resolve_rows, validate_population
from xty2.views import ColumnRoll


def _validate_target_policy(sharpening: object, stop_grad: object) -> None:
    if sharpening != "probability_power_temperature" or stop_grad != "target":
        raise LossError(
            "ReMixMatch requires probability-power sharpening and detached targets"
        )


class AnchoredLabelGuess:
    """Executor-owned ReMixMatch distribution-alignment history."""

    __slots__ = (
        "_labelled",
        "_last_rows",
        "_last_step",
        "_target",
        "_window",
        "capacity",
        "classes",
        "decay",
        "epsilon",
        "temperature",
        "use_alignment",
    )

    def __init__(
        self,
        num_treatments: int,
        *,
        capacity: int,
        labelled_decay: float,
        epsilon: float,
        temperature: float,
        use_alignment: bool,
    ) -> None:
        self.classes = num_treatments
        self.capacity = capacity
        self.decay = labelled_decay
        self.epsilon = epsilon
        self.temperature = temperature
        self.use_alignment = use_alignment
        self._window: list[Tensor] = []
        self._labelled = torch.full(
            (num_treatments,), 1.0 / num_treatments, dtype=torch.float64
        )
        self._last_step: int | None = None
        self._last_rows: tuple[int, ...] = ()
        self._target: Tensor | None = None

    @property
    def prediction_marginal(self) -> Tensor:
        if not self._window:
            return torch.full((self.classes,), 1.0 / self.classes, dtype=torch.float64)
        return torch.stack(self._window).mean(dim=0)

    @property
    def labelled_marginal(self) -> Tensor:
        return self._labelled.clone()

    @property
    def last_prepared_step(self) -> int | None:
        return self._last_step

    def prepare(
        self,
        *,
        step: int,
        probabilities: Tensor,
        batch: XTYBatch,
        rows: RowIndex,
        support_rows: RowIndex,
    ) -> Tensor:
        signature = tuple(int(value) for value in rows.tolist())
        if self._last_step is not None and step <= self._last_step:
            if step != self._last_step or signature != self._last_rows:
                raise LossError(
                    "AnchoredLabelGuess was prepared out of order or for different rows"
                )
            assert self._target is not None
            return self._target
        if probabilities.shape != (batch.batch_size, self.classes):
            raise LossError(
                f"anchor probabilities must be [{batch.batch_size}, {self.classes}], "
                f"got {tuple(probabilities.shape)}"
            )
        raw = probabilities.index_select(0, rows).detach()
        if raw.numel() == 0 or support_rows.numel() == 0:
            raise LossError(
                "AnchoredLabelGuess needs non-empty labelled and unlabelled quotas"
            )
        labels = batch.t.index_select(0, support_rows)
        observed = (
            F.one_hot(labels, num_classes=self.classes).to(torch.float64).mean(dim=0)
        )
        aligned = raw.to(torch.float64)
        if self.use_alignment:
            aligned = aligned * (
                (self._labelled.to(raw.device) + self.epsilon)
                / (self.prediction_marginal.to(raw.device) + self.epsilon)
            )
            aligned = aligned / aligned.sum(dim=-1, keepdim=True)
        sharpened = aligned.pow(1.0 / self.temperature)
        sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True)
        full = torch.zeros(
            batch.batch_size, self.classes, dtype=raw.dtype, device=raw.device
        ).index_copy(0, rows, sharpened.to(raw.dtype))
        # The source builds the guess from the old variables and assigns both
        # moving statistics in post_ops after the optimiser step.
        self._window.append(raw.mean(dim=0).to(torch.float64).cpu())
        if len(self._window) > self.capacity:
            self._window.pop(0)
        self._labelled = (
            self.decay * self._labelled + (1.0 - self.decay) * observed.cpu()
        )
        self._last_step = step
        self._last_rows = signature
        self._target = full.detach()
        return self._target


@dataclass(frozen=True)
class AnchoredTargetTreatmentNLL:
    target: Realisation
    prediction: Realisation
    num_treatments: int
    target_copies: tuple[Realisation, ...] = ()
    temperature: float = REQUIRED
    sharpening: Literal["probability_power_temperature"] = REQUIRED
    stop_grad: Literal["target"] = REQUIRED
    confidence_threshold: Literal["n/a"] = REQUIRED
    window_capacity: int = 128
    labelled_decay: float = 0.999
    alignment_epsilon: float = 1e-6
    use_alignment: bool = True
    augmentation_count: int = 8
    redux: Literal["1st", "mean"] = "1st"
    support_rows: Rows = "t_observed"
    rows: Rows = "t_missing"
    name: str = "premixup_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "temperature": "losses.temperature",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
        "confidence_threshold": "losses.confidence_threshold",
    }

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_copies", tuple(self.target_copies))
        card_hyperparameters(self)
        _validate_target_policy(self.sharpening, self.stop_grad)
        if self.confidence_threshold != "n/a":
            raise LossError("ReMixMatch has no confidence gate")
        if not self.temperature > 0 or self.window_capacity < 1:
            raise LossError("invalid ReMixMatch temperature or window capacity")
        if not 0 < self.labelled_decay < 1 or self.alignment_epsilon <= 0:
            raise LossError("invalid ReMixMatch alignment state parameters")
        if self.augmentation_count < 1 or self.redux not in ("1st", "mean"):
            raise LossError("ReMixMatch needs K >= 1 and redux in {'1st', 'mean'}")
        if self.redux == "mean" and len(self.target_copies) != self.augmentation_count:
            raise LossError("redux='mean' needs exactly K strong target copies")
        if self.redux == "1st" and self.target_copies:
            raise LossError("redux='1st' may not declare extra target copies")

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.target),
                (Port.T_GIVEN_X, self.prediction),
                *((Port.T_GIVEN_X, value) for value in self.target_copies),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        # A copy can be both an input to the detached mean guess and this
        # objective's trainable prediction. At pair granularity that is an
        # undetached read; AnchoredLabelGuess detaches only the target use.
        return frozenset(
            {
                (Port.T_GIVEN_X, self.target),
                *(
                    (Port.T_GIVEN_X, value)
                    for value in self.target_copies
                    if value != self.prediction
                ),
            }
        )

    @property
    def batch_coupled(self) -> bool:
        return True

    def initial_state(self, population: TrainingPopulation | None) -> object:
        del population
        return AnchoredLabelGuess(
            self.num_treatments,
            capacity=self.window_capacity,
            labelled_decay=self.labelled_decay,
            epsilon=self.alignment_epsilon,
            temperature=self.temperature,
            use_alignment=self.use_alignment,
        )

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"anchor={self.target}; prediction={self.prediction}; "
            f"target copies={len(self.target_copies)}",
            f"alignment window={self.window_capacity}, labelled EMA="
            f"{self.labelled_decay:g}, epsilon={self.alignment_epsilon:g}",
            f"alignment={'on' if self.use_alignment else 'off'} before "
            f"probability-power T={self.temperature:g}",
            f"K={self.augmentation_count}; redux={self.redux!r}; use_xe=True",
            "confidence gate = none",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        prediction = treatment_distribution(
            state, Port.T_GIVEN_X, self.prediction, objective=self.name
        ).probs
        target = _prepared(
            self.name,
            state,
            batch,
            rows,
            ctx,
            (self.target, *self.target_copies),
            self.redux,
            self.support_rows,
        )
        per_row = -(target * torch.log(prediction.clamp_min(1e-12))).sum(dim=-1)
        return reduce_rows(per_row, rows)


@dataclass(frozen=True)
class MixedTargetTreatmentNLL:
    owner: str
    anchor: Realisation
    predictions: tuple[Realisation, ...]
    num_treatments: int
    first_target: Literal["observed", "anchor"]
    anchor_copies: tuple[Realisation, ...] = ()
    redux: Literal["1st", "mean"] = "1st"
    support_rows: Rows = "t_observed"
    rows: Rows = "t_missing"
    name: str = "mixed_unlabelled_treatment_nll"

    def __post_init__(self) -> None:
        require_str("MixedTargetTreatmentNLL.owner", self.owner, error=LossError)
        object.__setattr__(self, "predictions", tuple(self.predictions))
        object.__setattr__(self, "anchor_copies", tuple(self.anchor_copies))
        if not self.predictions or self.first_target not in ("observed", "anchor"):
            raise LossError(
                "MixedTargetTreatmentNLL needs predictions and a target kind"
            )
        if self.redux not in ("1st", "mean"):
            raise LossError("MixedTargetTreatmentNLL has an invalid redux")
        if self.redux == "1st" and self.anchor_copies:
            raise LossError("redux='1st' may not declare extra anchor copies")
        if self.redux == "mean" and not self.anchor_copies:
            raise LossError("redux='mean' needs strong anchor copies")
        try:
            validate_population(self.rows)
            validate_population(self.support_rows)
        except Exception as error:
            raise LossError("MixedTargetTreatmentNLL has invalid rows") from error

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.anchor),
                *((Port.T_GIVEN_X, value) for value in self.anchor_copies),
                *((Port.T_GIVEN_X, value) for value in self.predictions),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.anchor),
                *((Port.T_GIVEN_X, value) for value in self.anchor_copies),
            }
        )

    @property
    def batch_coupled(self) -> bool:
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            f"anchor state owner={self.owner}; first target={self.first_target}; "
            f"redux={self.redux!r}",
            "features and targets share each synthetic realisation's "
            "(permutation, lambda)",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        targets = _prepared(
            self.owner,
            state,
            batch,
            rows if self.first_target == "anchor" else resolve_rows(batch, "t_missing"),
            ctx,
            (self.anchor, *self.anchor_copies),
            self.redux,
            self.support_rows,
        )
        losses: list[Tensor] = []
        coefficients: list[Tensor] = []
        for realisation in self.predictions:
            prediction = treatment_distribution(
                state, Port.T_GIVEN_X, realisation, objective=self.name
            ).probs
            plan = state.mixing_plan(realisation)
            if not isinstance(plan, MixingPlan):
                raise LossError(f"{self.name} received an invalid mixing plan")
            if not torch.equal(plan.first_rows, rows):
                raise LossError(
                    f"{self.name} mixing-plan rows differ from its eligible rows"
                )
            if self.first_target == "observed":
                first = F.one_hot(
                    batch.t.index_select(0, rows), self.num_treatments
                ).to(prediction.dtype)
            else:
                first = targets.index_select(0, rows)
            partner = targets.index_select(0, plan.partner_rows).clone()
            if bool(plan.partner_is_observed.any()):
                observed_positions = torch.nonzero(
                    plan.partner_is_observed, as_tuple=False
                ).flatten()
                observed_rows = plan.partner_rows.index_select(0, observed_positions)
                one_hot = F.one_hot(
                    batch.t.index_select(0, observed_rows), self.num_treatments
                ).to(prediction.dtype)
                partner.index_copy_(0, observed_positions, one_hot)
            lam = plan.coefficient.to(prediction.dtype)[:, None]
            coefficients.append(plan.coefficient.mean())
            mixed_target = lam * first + (1.0 - lam) * partner
            losses.append(
                -(
                    mixed_target
                    * torch.log(prediction.index_select(0, rows).clamp_min(1e-12))
                ).sum(dim=-1)
            )
        eligible = torch.stack(losses).mean(dim=0)
        per_row = torch.zeros(
            batch.batch_size, dtype=eligible.dtype, device=eligible.device
        ).index_copy(0, rows, eligible)
        return reduce_rows(
            per_row,
            rows,
            diagnostics={"lambda_mean": float(torch.stack(coefficients).mean())},
        )


@dataclass(frozen=True)
class PretextTransformNLL:
    realisation: Realisation
    num_transforms: int = 4
    rows: Rows = "t_missing"
    name: str = "pretext_transform_nll"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.PRETEXT_GIVEN_X, self.realisation)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    @property
    def batch_coupled(self) -> bool:
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        logits = state[self.realisation][Port.PRETEXT_GIVEN_X]
        if not isinstance(logits, Tensor):
            raise PortContractError(f"{self.name} expected tensor pretext logits")
        if logits.shape != (batch.batch_size, self.num_transforms):
            raise LossError(
                f"pretext logits must be [{batch.batch_size}, {self.num_transforms}]"
            )
        labels = ColumnRoll.labels(rows.numel(), device=batch.device)
        eligible = F.cross_entropy(
            logits.index_select(0, rows), labels, reduction="none"
        )
        per_row = torch.zeros(
            batch.batch_size, dtype=logits.dtype, device=logits.device
        ).index_copy(0, rows, eligible)
        accuracy = (
            float(
                (logits.index_select(0, rows).argmax(dim=-1) == labels).float().mean()
            )
            if rows.numel()
            else 0.0
        )
        return reduce_rows(per_row, rows, diagnostics={"accuracy": accuracy})


def _prepared(
    owner: str,
    state: State,
    batch: XTYBatch,
    rows: RowIndex,
    ctx: TrainContext,
    anchors: tuple[Realisation, ...],
    redux: Literal["1st", "mean"],
    support_rows: Rows,
) -> Tensor:
    guess = ctx.objective_state(owner, AnchoredLabelGuess)
    distributions = tuple(
        treatment_distribution(state, Port.T_GIVEN_X, anchor, objective=owner).probs
        for anchor in anchors
    )
    probabilities = distributions[0]
    if redux == "mean":
        probabilities = torch.stack(distributions).mean(dim=0)
    return guess.prepare(
        step=ctx.global_step,
        probabilities=probabilities,
        batch=batch,
        rows=rows,
        support_rows=resolve_rows(batch, support_rows),
    )


__all__ = [
    "AnchoredLabelGuess",
    "AnchoredTargetTreatmentNLL",
    "MixedTargetTreatmentNLL",
    "PretextTransformNLL",
]
