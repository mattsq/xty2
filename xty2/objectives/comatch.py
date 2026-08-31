"""CoMatch memory-smoothed labels and pseudo-label graph contrastive loss.

The two losses in CoMatch are separate weighted terms but share one piece of
per-stage state.  Either term may be evaluated first: both declare and provide
the weak prediction and embedding needed to prepare ``q`` and the state makes
that preparation idempotent within an optimiser step.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import Tensor
from torch.nn import functional as F

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.data import TrainingPopulation
from xty2.core.errors import LossError, PortContractError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows, treatment_distribution
from xty2.core.ports import Port
from xty2.core.rows import RowIndex, Rows, resolve_rows, validate_population

LOG_FLOOR = 1e-7


@dataclass(frozen=True)
class CoMatchConfidenceThresholds:
    """The two gates appendix A states together: eq. (4)'s tau and eq. (9)'s T."""

    pseudo_label: float
    edge: float

    def __post_init__(self) -> None:
        for field in ("pseudo_label", "edge"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise LossError(f"CoMatch {field} threshold must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise LossError(
                    f"CoMatch {field} threshold must be in [0, 1], got {value!r}"
                )
            object.__setattr__(self, field, float(value))

    def __repr__(self) -> str:
        return f"comatch(pseudo_label={self.pseudo_label:g}, edge={self.edge:g})"


@dataclass(frozen=True)
class MemorySmoothedLabelGraph:
    """The values shared by eqs. (7)--(11) and their reference-code lifecycle."""

    temperature: float
    alpha: float
    capacity: int
    thresholds: CoMatchConfidenceThresholds
    alignment_window: int
    unsmoothed_steps: int

    def __post_init__(self) -> None:
        if isinstance(self.temperature, bool) or not isinstance(
            self.temperature, int | float
        ):
            raise LossError("CoMatch temperature must be a number")
        if float(self.temperature) <= 0.0:
            raise LossError(
                f"CoMatch temperature must be positive, got {self.temperature!r}"
            )
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, int | float):
            raise LossError("CoMatch alpha must be a number")
        if not 0.0 <= float(self.alpha) <= 1.0:
            raise LossError(f"CoMatch alpha must be in [0, 1], got {self.alpha!r}")
        if type(self.capacity) is not int or self.capacity < 1:
            raise LossError(
                f"CoMatch memory capacity must be a positive int, got {self.capacity!r}"
            )
        if not isinstance(self.thresholds, CoMatchConfidenceThresholds):
            raise LossError(
                "MemorySmoothedLabelGraph.thresholds must be a "
                "CoMatchConfidenceThresholds value"
            )
        if type(self.alignment_window) is not int or self.alignment_window < 1:
            raise LossError(
                "CoMatch alignment_window must be a positive int, got "
                f"{self.alignment_window!r}"
            )
        if type(self.unsmoothed_steps) is not int or self.unsmoothed_steps < 0:
            raise LossError(
                "CoMatch unsmoothed_steps must be a non-negative int, got "
                f"{self.unsmoothed_steps!r}"
            )
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "alpha", float(self.alpha))


class MemorySmoothedLabels:
    """A fresh-per-stage FIFO bank and distribution-alignment window.

    The bank holds detached, raw weak probabilities (captured before
    distribution alignment), detached weak embeddings, and diagnostic row ids.
    Row ids are never affinity keys; they exist only to measure repeated draws.
    """

    __slots__ = (
        "_classes",
        "_embeddings",
        "_graph",
        "_last_rows",
        "_last_step",
        "_marginals",
        "_probabilities",
        "_pseudo_labels",
        "_repeated_draws",
        "_row_ids",
    )

    def __init__(self, classes: int, graph: MemorySmoothedLabelGraph) -> None:
        if classes < 2:
            raise LossError(
                f"MemorySmoothedLabels needs at least two classes, got {classes}"
            )
        self._classes = int(classes)
        self._graph = graph
        self._probabilities: Tensor | None = None
        self._embeddings: Tensor | None = None
        self._row_ids: Tensor | None = None
        self._marginals: deque[Tensor] = deque(maxlen=graph.alignment_window)
        self._last_step: int | None = None
        self._last_rows: Tensor | None = None
        self._pseudo_labels: Tensor | None = None
        self._repeated_draws = 0.0

    @property
    def graph(self) -> MemorySmoothedLabelGraph:
        return self._graph

    @property
    def size(self) -> int:
        return 0 if self._probabilities is None else int(self._probabilities.shape[0])

    @property
    def probabilities(self) -> Tensor:
        if self._probabilities is None:
            return torch.empty((0, self._classes))
        return self._probabilities.clone()

    @property
    def embeddings(self) -> Tensor:
        if self._embeddings is None:
            return torch.empty((0, 0))
        return self._embeddings.clone()

    @property
    def row_ids(self) -> Tensor:
        if self._row_ids is None:
            return torch.empty(0, dtype=torch.long)
        return self._row_ids.clone()

    @property
    def last_prepared_step(self) -> int | None:
        return self._last_step

    def prepare(
        self,
        *,
        step: int,
        raw_probabilities: Tensor,
        weak_embeddings: Tensor,
        batch: XTYBatch,
        eligible_rows: RowIndex,
        support_rows: RowIndex,
    ) -> Tensor:
        """Return detached current-step ``q``, then append this step to the FIFO."""
        self._validate_inputs(raw_probabilities, weak_embeddings, batch)
        eligible_ids = batch.row_id.index_select(0, eligible_rows).detach()
        if self._last_step is not None and step <= self._last_step:
            if step != self._last_step:
                raise LossError(
                    f"CoMatch state was asked to move backwards from step "
                    f"{self._last_step} to {step}"
                )
            if self._last_rows is None or not torch.equal(
                eligible_ids.to(self._last_rows.device), self._last_rows
            ):
                raise LossError(
                    "the two CoMatch objectives prepared different eligible rows "
                    f"at step {step}; shared state requires one row population"
                )
            if self._pseudo_labels is None:
                raise LossError("CoMatch state recorded a step without pseudo-labels")
            return self._pseudo_labels

        if support_rows.numel() and not bool(
            batch.t_observed.index_select(0, support_rows).all()
        ):
            raise LossError(
                "CoMatch memory support rows must all have observed treatments; "
                "using hidden treatment values here would leak labels into the bank"
            )

        raw = raw_probabilities.detach()
        weak = F.normalize(weak_embeddings.detach(), dim=-1)
        unlabelled_raw = raw.index_select(0, eligible_rows)
        self._marginals.append(unlabelled_raw.mean(dim=0).clone())
        marginal = torch.stack(tuple(self._marginals)).mean(dim=0)
        aligned = unlabelled_raw / marginal.clamp_min(torch.finfo(raw.dtype).tiny)
        aligned = aligned / aligned.sum(dim=-1, keepdim=True)

        if step < self._graph.unsmoothed_steps or self.size == 0:
            pseudo = aligned
        else:
            assert self._embeddings is not None
            assert self._probabilities is not None
            bank_embeddings = self._embeddings.to(device=weak.device, dtype=weak.dtype)
            bank_probabilities = self._probabilities.to(
                device=raw.device, dtype=raw.dtype
            )
            anchors = weak.index_select(0, eligible_rows)
            affinity = torch.softmax(
                anchors @ bank_embeddings.transpose(0, 1) / self._graph.temperature,
                dim=-1,
            )
            pseudo = self._graph.alpha * aligned + (1.0 - self._graph.alpha) * (
                affinity @ bank_probabilities
            )

        pseudo = pseudo.detach()
        support_probabilities = F.one_hot(
            batch.t.index_select(0, support_rows), num_classes=self._classes
        ).to(dtype=raw.dtype)
        write_probabilities = torch.cat((unlabelled_raw, support_probabilities), dim=0)
        write_embeddings = torch.cat(
            (
                weak.index_select(0, eligible_rows),
                weak.index_select(0, support_rows),
            ),
            dim=0,
        )
        write_row_ids = torch.cat(
            (eligible_ids, batch.row_id.index_select(0, support_rows).detach()), dim=0
        )
        self._repeated_draws = self.repeated_draws(eligible_ids)
        self._append(write_probabilities, write_embeddings, write_row_ids)
        self._last_step = step
        self._last_rows = eligible_ids.clone()
        self._pseudo_labels = pseudo
        return pseudo

    def repeated_draws(self, eligible_row_ids: Tensor) -> float:
        """Mean previous bank entries sharing an anchor's row id; diagnostic only."""
        if self._row_ids is None or eligible_row_ids.numel() == 0:
            return 0.0
        matches = (
            eligible_row_ids.detach().to(self._row_ids.device)[:, None] == self._row_ids
        )
        return float(matches.sum(dim=-1).to(torch.float32).mean())

    @property
    def last_repeated_draws(self) -> float:
        """Mean entries for the current anchors that existed before this write."""
        return self._repeated_draws

    def _append(
        self, probabilities: Tensor, embeddings: Tensor, row_ids: Tensor
    ) -> None:
        if probabilities.shape[0] == 0:
            return
        if self._probabilities is None:
            combined_probabilities = probabilities.detach()
            combined_embeddings = embeddings.detach()
            combined_row_ids = row_ids.detach()
        else:
            assert self._embeddings is not None
            assert self._row_ids is not None
            combined_probabilities = torch.cat(
                (self._probabilities.to(probabilities), probabilities.detach()), dim=0
            )
            combined_embeddings = torch.cat(
                (self._embeddings.to(embeddings), embeddings.detach()), dim=0
            )
            combined_row_ids = torch.cat(
                (self._row_ids.to(row_ids), row_ids.detach()), dim=0
            )
        keep = self._graph.capacity
        self._probabilities = combined_probabilities[-keep:].clone()
        self._embeddings = combined_embeddings[-keep:].clone()
        self._row_ids = combined_row_ids[-keep:].clone()

    def _validate_inputs(
        self, probabilities: Tensor, embeddings: Tensor, batch: XTYBatch
    ) -> None:
        if probabilities.ndim != 2 or probabilities.shape != (
            batch.batch_size,
            self._classes,
        ):
            raise LossError(
                f"CoMatch weak probabilities must be "
                f"[{batch.batch_size}, {self._classes}], got "
                f"{tuple(probabilities.shape)}"
            )
        if embeddings.ndim != 2 or embeddings.shape[0] != batch.batch_size:
            raise LossError(
                "CoMatch weak embeddings must be [B, D], got "
                f"{tuple(embeddings.shape)} for B={batch.batch_size}"
            )


@dataclass(frozen=True)
class MemorySmoothedPseudoLabelTreatmentNLL:
    """CoMatch eq. (4) with eq. (8)'s detached soft pseudo-label."""

    graph: MemorySmoothedLabelGraph
    target: Realisation
    weak_embedding: Realisation
    prediction: Realisation
    num_treatments: int
    sharpening: Literal["none"] = REQUIRED
    stop_grad: Literal["target"] = REQUIRED
    support_rows: Rows = "t_observed"
    rows: Rows = "t_missing"
    name: str = "memory_smoothed_pseudo_label_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "thresholds": "losses.confidence_threshold",
        "temperature": "losses.temperature",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        _validate_common(
            owner=type(self).__name__,
            name=self.name,
            graph=self.graph,
            target=self.target,
            weak_embedding=self.weak_embedding,
            num_treatments=self.num_treatments,
            support_rows=self.support_rows,
            rows=self.rows,
        )
        if not isinstance(self.prediction, Realisation):
            raise LossError("CoMatch prediction must be a Realisation")
        if self.prediction == self.target:
            raise LossError("CoMatch target and prediction realisations must differ")
        if self.sharpening != "none":
            raise LossError("CoMatch keeps q soft; sharpening must be 'none'")
        if self.stop_grad != "target":
            raise LossError("CoMatch detaches q; stop_grad must be 'target'")

    @property
    def thresholds(self) -> CoMatchConfidenceThresholds:
        return self.graph.thresholds

    @property
    def temperature(self) -> float:
        return self.graph.temperature

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.target),
                (Port.X_PROJ, self.weak_embedding),
                (Port.T_GIVEN_X, self.prediction),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.target),
                (Port.X_PROJ, self.weak_embedding),
            }
        )

    @property
    def batch_coupled(self) -> bool:
        return True

    def initial_state(self, population: TrainingPopulation | None) -> object:
        del population
        return MemorySmoothedLabels(self.num_treatments, self.graph)

    def plan_details(self) -> tuple[str, ...]:
        return (
            *_shared_plan(self.graph, self.support_rows),
            "label = detached soft q from eq. (8); no arg max or sharpening",
            "gate = max(q) >= pseudo-label threshold",
            "denominator = every eligible row; rejected rows contribute 0",
            f"prediction = {self.prediction}",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        prediction = treatment_distribution(
            state, Port.T_GIVEN_X, self.prediction, objective=self.name
        ).probs
        if rows.numel() == 0:
            return LossTerm.empty(like=prediction)
        labels = ctx.objective_state(self.name, MemorySmoothedLabels)
        pseudo = _prepare(
            labels,
            graph=self.graph,
            state=state,
            batch=batch,
            rows=rows,
            support_rows=self.support_rows,
            target=self.target,
            weak_embedding=self.weak_embedding,
            step=ctx.global_step,
            owner=self.name,
        )
        if prediction.shape != (batch.batch_size, self.num_treatments):
            raise LossError(
                f"{self.name} prediction has shape {tuple(prediction.shape)}, "
                f"expected [{batch.batch_size}, {self.num_treatments}]"
            )
        confidence = pseudo.max(dim=-1).values
        accepted = confidence >= self.thresholds.pseudo_label
        eligible_loss = -torch.xlogy(pseudo, prediction.index_select(0, rows)).sum(
            dim=-1
        ) * accepted.to(prediction.dtype)
        per_row = torch.zeros(
            batch.batch_size, dtype=prediction.dtype, device=prediction.device
        ).index_copy(0, rows, eligible_loss)
        diagnostics: dict[str, float] = {}
        if rows.numel():
            diagnostics["coverage"] = float(accepted.to(torch.float32).mean())
            diagnostics["accepted_confidence"] = (
                float(confidence[accepted].mean()) if bool(accepted.any()) else 0.0
            )
        return reduce_rows(per_row, rows, diagnostics=diagnostics)


@dataclass(frozen=True)
class PseudoLabelGraphContrastive:
    """CoMatch eqs. (9)--(11), using two trainable strong embedding views."""

    graph: MemorySmoothedLabelGraph
    labels: str
    target: Realisation
    weak_embedding: Realisation
    anchor: Realisation
    contrast: Realisation
    num_treatments: int
    support_rows: Rows = "t_observed"
    rows: Rows = "t_missing"
    name: str = "pseudo_label_graph_contrastive"

    def __post_init__(self) -> None:
        _validate_common(
            owner=type(self).__name__,
            name=self.name,
            graph=self.graph,
            target=self.target,
            weak_embedding=self.weak_embedding,
            num_treatments=self.num_treatments,
            support_rows=self.support_rows,
            rows=self.rows,
        )
        if not require_str("CoMatch state owner", self.labels, error=LossError):
            raise LossError("PseudoLabelGraphContrastive.labels must be non-empty")
        anchor: object = self.anchor
        contrast: object = self.contrast
        if not isinstance(anchor, Realisation) or not isinstance(contrast, Realisation):
            raise LossError("CoMatch anchor and contrast must be Realisations")
        if self.anchor == self.contrast:
            raise LossError("CoMatch needs two distinct strong realisations")

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.target),
                (Port.X_PROJ, self.weak_embedding),
                (Port.X_PROJ, self.anchor),
                (Port.X_PROJ, self.contrast),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.target),
                (Port.X_PROJ, self.weak_embedding),
            }
        )

    @property
    def batch_coupled(self) -> bool:
        return True

    def plan_details(self) -> tuple[str, ...]:
        return (
            *_shared_plan(self.graph, self.support_rows),
            f"pseudo-label state owner = {self.labels}",
            f"anchor rows (W^z row index) = {self.anchor}",
            f"contrast columns (W^z column index) = {self.contrast}",
            f"edge threshold = {self.graph.thresholds.edge!r}; diagonal = 1",
            f"log floor = {LOG_FLOOR!r}",
            "both strong embedding realisations train; W^q is detached",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        anchor = _embedding(state, self.anchor, batch, owner=self.name)
        contrast = _embedding(state, self.contrast, batch, owner=self.name)
        if rows.numel() == 0:
            return LossTerm.empty(like=anchor)
        labels = ctx.objective_state(self.labels, MemorySmoothedLabels)
        pseudo = _prepare(
            labels,
            graph=self.graph,
            state=state,
            batch=batch,
            rows=rows,
            support_rows=self.support_rows,
            target=self.target,
            weak_embedding=self.weak_embedding,
            step=ctx.global_step,
            owner=self.name,
        )
        q_similarity = pseudo @ pseudo.transpose(0, 1)
        target_graph = torch.where(
            q_similarity >= self.graph.thresholds.edge,
            q_similarity,
            torch.zeros_like(q_similarity),
        )
        target_graph.fill_diagonal_(1.0)
        target_graph = target_graph / target_graph.sum(dim=-1, keepdim=True)

        left = F.normalize(anchor.index_select(0, rows), dim=-1)
        right = F.normalize(contrast.index_select(0, rows), dim=-1)
        similarities = torch.exp(left @ right.transpose(0, 1) / self.graph.temperature)
        predicted_graph = similarities / similarities.sum(dim=-1, keepdim=True)
        eligible_loss = -(target_graph * torch.log(predicted_graph + LOG_FLOOR)).sum(
            dim=-1
        )
        per_row = torch.zeros(
            batch.batch_size,
            dtype=eligible_loss.dtype,
            device=eligible_loss.device,
        ).index_copy(0, rows, eligible_loss)

        detached_cosines = (left @ right.transpose(0, 1)).detach()
        count = int(rows.numel())
        diagnostics = {
            "edges_per_row": float((target_graph > 0).sum(dim=-1).float().mean()),
            "alignment": float(detached_cosines.diagonal().mean()),
            "repeated_bank_rows": labels.last_repeated_draws,
        }
        if count > 1:
            diagnostics["uniformity"] = float(
                detached_cosines.sum() - detached_cosines.diagonal().sum()
            ) / (count * (count - 1))
        return reduce_rows(per_row, rows, diagnostics=diagnostics)


def _prepare(
    labels: MemorySmoothedLabels,
    *,
    graph: MemorySmoothedLabelGraph,
    state: State,
    batch: XTYBatch,
    rows: RowIndex,
    support_rows: Rows,
    target: Realisation,
    weak_embedding: Realisation,
    step: int,
    owner: str,
) -> Tensor:
    if labels.graph != graph:
        raise LossError(
            f"{owner} carries a different MemorySmoothedLabelGraph from the "
            "state owner it names"
        )
    weak_probabilities = treatment_distribution(
        state, Port.T_GIVEN_X, target, objective=owner
    ).probs
    weak = _embedding(state, weak_embedding, batch, owner=owner)
    return labels.prepare(
        step=step,
        raw_probabilities=weak_probabilities,
        weak_embeddings=weak,
        batch=batch,
        eligible_rows=rows,
        support_rows=resolve_rows(batch, support_rows),
    )


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
            f"{owner} got embedding shape {tuple(value.shape)} under {realisation} "
            f"for a batch of {batch.batch_size}"
        )
    return value


def _validate_common(
    *,
    owner: str,
    name: str,
    graph: object,
    target: object,
    weak_embedding: object,
    num_treatments: object,
    support_rows: Rows,
    rows: Rows,
) -> None:
    if not require_str("CoMatch objective name", name, error=LossError):
        raise LossError(f"{owner}.name must be non-empty")
    if not isinstance(graph, MemorySmoothedLabelGraph):
        raise LossError(f"{owner}.graph must be a MemorySmoothedLabelGraph")
    if not isinstance(target, Realisation) or not isinstance(
        weak_embedding, Realisation
    ):
        raise LossError(f"{owner} weak inputs must be Realisations")
    if type(num_treatments) is not int or num_treatments < 2:
        raise LossError(f"{owner}.num_treatments must be an int >= 2")
    for population in (support_rows, rows):
        try:
            validate_population(population)
        except Xty2Error as error:
            raise LossError(f"{owner} {name!r}: {error}") from error


def _shared_plan(
    graph: MemorySmoothedLabelGraph, support_rows: Rows
) -> tuple[str, ...]:
    return (
        f"support rows written to memory = {support_rows}",
        "memory probabilities = raw weak predictions before distribution alignment; "
        "observed supports use one-hot treatment",
        f"distribution alignment = unweighted mean of last {graph.alignment_window} "
        "current-inclusive batch marginals",
        f"memory = FIFO capacity {graph.capacity}; read before write",
        f"q = {graph.alpha:g} aligned weak + {1.0 - graph.alpha:g} "
        "memory-neighbour average (eq. 8)",
        f"memory smoothing begins at step {graph.unsmoothed_steps}; steps "
        f"0..{max(graph.unsmoothed_steps - 1, 0)} are unsmoothed",
    )


__all__ = [
    "LOG_FLOOR",
    "CoMatchConfidenceThresholds",
    "MemorySmoothedLabelGraph",
    "MemorySmoothedLabels",
    "MemorySmoothedPseudoLabelTreatmentNLL",
    "PseudoLabelGraphContrastive",
]
