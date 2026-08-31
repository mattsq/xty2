"""SimMatch semantic/instance propagation over a labelled similarity memory.

`docs/recipes/simmatch.md` maps the paper's equations (1)--(12) onto one
`joint_fit` stage. Two of those equations charge losses -- eq. (2)'s gated
semantic consistency and eq. (5)'s instance consistency -- and they carry
separate coefficients under eq. (6), so they are two objectives. Everything
between them is one piece of stage-local state: the labelled bank `Q_f, Q_l`,
the distribution-alignment window, and both propagation directions, prepared
once per optimiser step and read by whichever objective the mixer evaluates
first.

The bank is *not* CoMatch's FIFO. It holds exactly one feature slot per
observed training row, keyed by that row's `row_id` and labelled by its
observed treatment, which is what makes eqs. (7)--(10) a statement about
labelled instances rather than about recent predictions (card §3.2).
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


@dataclass(frozen=True)
class SimilarityMatchingTemperatures:
    """Eqs. (3) and (4): `t_w` and `t_s`, which the source exposes separately.

    The paper writes one `t`; the authors' code carries `--tt` and `--st` and
    the published command sets both to `0.1` (card §7). Two fields keep that
    degree of freedom visible, and the recipe binding them to one number is
    then a fact the plan prints rather than one the class assumes.
    """

    instance_weak: float
    instance_strong: float

    def __post_init__(self) -> None:
        for field in ("instance_weak", "instance_strong"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise LossError(f"SimMatch {field} temperature must be a number")
            if float(value) <= 0.0:
                raise LossError(
                    f"SimMatch {field} temperature must be positive, got {value!r}"
                )
            object.__setattr__(self, field, float(value))

    def __repr__(self) -> str:
        return (
            f"simmatch(instance_weak={self.instance_weak:g}, "
            f"instance_strong={self.instance_strong:g})"
        )


@dataclass(frozen=True)
class SimilarityMatchingSpec:
    """The source constants both SimMatch objectives must agree on.

    Five of these bind no `FIDELITY.md` §2 key -- `alpha`, the bank momentum,
    the alignment window, the warm-up, and the unfolding switch -- so they take
    the route `DESIGN.md` §4 provides for keyless paper-governed values: no
    defaults, one shared frozen object, and `plan_details()` prints them
    (card §4).

    `K` is deliberately absent. It is the number of observed training rows and
    is read from the stage's `TrainingPopulation`, on the `flexmatch`
    precedent for `N`.
    """

    temperatures: SimilarityMatchingTemperatures
    alpha: float
    memory_momentum: float
    alignment_window: int
    warmup_steps: int
    threshold: float
    unfold: bool

    def __post_init__(self) -> None:
        if not isinstance(self.temperatures, SimilarityMatchingTemperatures):
            raise LossError(
                "SimilarityMatchingSpec.temperatures must be a "
                "SimilarityMatchingTemperatures value"
            )
        for field, upper in (("alpha", 1.0), ("memory_momentum", 1.0)):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise LossError(f"SimMatch {field} must be a number")
            if not 0.0 <= float(value) <= upper:
                raise LossError(f"SimMatch {field} must be in [0, 1], got {value!r}")
            object.__setattr__(self, field, float(value))
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, int | float
        ):
            raise LossError("SimMatch threshold must be a number")
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise LossError(
                f"SimMatch threshold must be in [0, 1], got {self.threshold!r}"
            )
        object.__setattr__(self, "threshold", float(self.threshold))
        if type(self.alignment_window) is not int or self.alignment_window < 1:
            raise LossError(
                "SimMatch alignment_window must be a positive int, got "
                f"{self.alignment_window!r}"
            )
        if type(self.warmup_steps) is not int or self.warmup_steps < 0:
            raise LossError(
                f"SimMatch warmup_steps must be a non-negative int, got "
                f"{self.warmup_steps!r}"
            )
        if not isinstance(self.unfold, bool):
            raise LossError(
                f"SimMatch unfold must be a bool, got {type(self.unfold)}. It is "
                "eqs. (7)-(8) on or off, which is the §6 ablation's only switch."
            )


@dataclass(frozen=True)
class PropagatedTargets:
    """One step's detached targets, shared by both objectives.

    Attributes:
        semantic: `[n, K_t]`, eq. (10)'s `hat p`, or the aligned `p^w` while
            propagation is off.
        instance: `[n, K]`, eq. (8)'s `hat q`, or `None` during warm-up, when
            eq. (5) is not charged at all.
        bank: `[K, D]` the slot features the targets were built from -- the
            *previous* step's, cloned before this step writes. Eq. (4)'s `q^s`
            is taken against these same vectors.
        aligned: `[n, K_t]`, `DA(p^w)`, kept so the semantic ablation and the
            diagnostics can compare against the unpropagated target.
        propagated: Whether eqs. (8) and (10) ran this step.
        coverage: The fraction of slots that had been filled when the targets
            were prepared.
    """

    semantic: Tensor
    instance: Tensor | None
    bank: Tensor
    aligned: Tensor
    propagated: bool
    coverage: float


class LabeledSimilarityMemory:
    """`Q_f` and `Q_l` of card §3.2: one slot per observed training row.

    Three properties do the work the card asks for.

    * **Random access by identity.** Slots are keyed by the sorted `row_id` of
      the observed training rows, so a support row lands in its own slot
      whatever order the sampler drew it in, and a slot is only ever written by
      the row it belongs to.
    * **Read before write.** `prepare` clones the bank, builds both targets
      from that clone, and only then observes the current support embeddings
      (card §7). A support embedding therefore cannot alter the target charged
      against it in the same optimiser step.
    * **Idempotent within a step.** Either objective may be evaluated first;
      the second call at the same step returns the first call's targets and
      writes nothing, so the mixer's declaration order cannot move a number.

    Labels come from the `TrainingPopulation` and are immutable, so no hidden
    treatment can reach `Q_l` -- the batch is never asked for one.
    """

    __slots__ = (
        "_classes",
        "_features",
        "_filled",
        "_labels",
        "_last_rows",
        "_last_step",
        "_marginals",
        "_slot_ids",
        "_spec",
        "_targets",
    )

    def __init__(
        self,
        *,
        classes: int,
        spec: SimilarityMatchingSpec,
        slot_ids: Tensor,
        labels: Tensor,
    ) -> None:
        if classes < 2:
            raise LossError(
                f"LabeledSimilarityMemory needs at least two classes, got {classes}"
            )
        if slot_ids.ndim != 1 or slot_ids.numel() == 0:
            raise LossError(
                "LabeledSimilarityMemory needs at least one observed training "
                f"row to key a slot by, got shape {tuple(slot_ids.shape)}"
            )
        if slot_ids.numel() > 1 and not bool((slot_ids.diff() > 0).all()):
            raise LossError(
                "LabeledSimilarityMemory slots are keyed by sorted, distinct "
                "row ids; a repeated or unordered key makes 'which slot is this "
                "row' answerable two ways"
            )
        if labels.shape != slot_ids.shape:
            raise LossError(
                f"LabeledSimilarityMemory got {labels.numel()} labels for "
                f"{slot_ids.numel()} slots"
            )
        if bool((labels < 0).any()) or bool((labels >= classes).any()):
            raise LossError(
                f"LabeledSimilarityMemory got a memory label outside 0..{classes - 1}"
            )
        self._classes = int(classes)
        self._spec = spec
        self._slot_ids = slot_ids.detach().clone()
        self._labels = labels.detach().clone()
        self._features: Tensor | None = None
        self._filled = torch.zeros(int(slot_ids.numel()), dtype=torch.bool)
        self._marginals: deque[Tensor] = deque(maxlen=spec.alignment_window)
        self._last_step: int | None = None
        self._last_rows: Tensor | None = None
        self._targets: PropagatedTargets | None = None

    @property
    def spec(self) -> SimilarityMatchingSpec:
        return self._spec

    @property
    def size(self) -> int:
        """`K`: how many labelled slots the bank keys."""
        return int(self._slot_ids.numel())

    @property
    def slot_ids(self) -> Tensor:
        return self._slot_ids.clone()

    @property
    def labels(self) -> Tensor:
        return self._labels.clone()

    @property
    def features(self) -> Tensor:
        """`[K, D]` slot features, zero where a slot is still unfilled."""
        if self._features is None:
            return torch.zeros((self.size, 0))
        return self._features.clone()

    @property
    def coverage(self) -> float:
        """The fraction of slots written at least once."""
        return float(self._filled.to(torch.float32).mean())

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
    ) -> PropagatedTargets:
        """Both detached targets for this step, then observe the supports."""
        self._validate_inputs(raw_probabilities, weak_embeddings, batch)
        eligible_ids = batch.row_id.index_select(0, eligible_rows).detach()
        cached = self._cached(step, eligible_ids)
        if cached is not None:
            return cached

        slots = self._slots_for(batch, support_rows)
        raw = raw_probabilities.detach()
        weak = F.normalize(weak_embeddings.detach(), dim=-1)
        aligned = self._align(raw.index_select(0, eligible_rows))
        bank = self._bank(like=weak)
        propagated = step >= self._spec.warmup_steps and bool(self._filled.all())

        instance: Tensor | None = None
        semantic = aligned
        if propagated:
            anchors = weak.index_select(0, eligible_rows)
            weak_similarity = torch.softmax(
                anchors @ bank.transpose(0, 1) / self._spec.temperatures.instance_weak,
                dim=-1,
            )
            # Eq. (9) aggregates the *original* `q^w`; the calibrated `hat q`
            # of eq. (8) is a different tensor and only eq. (5) reads it
            # (card §7).
            aggregated = torch.zeros_like(aligned).index_add(
                1, self._labels.to(device=aligned.device), weak_similarity
            )
            semantic = (
                self._spec.alpha * aligned + (1.0 - self._spec.alpha) * aggregated
            )
            instance = weak_similarity
            if self._spec.unfold:
                unfolded = aligned.index_select(
                    1, self._labels.to(device=aligned.device)
                )
                calibrated = weak_similarity * unfolded
                instance = calibrated / calibrated.sum(dim=-1, keepdim=True).clamp_min(
                    torch.finfo(calibrated.dtype).tiny
                )

        targets = PropagatedTargets(
            semantic=semantic.detach(),
            instance=None if instance is None else instance.detach(),
            bank=bank.detach(),
            aligned=aligned.detach(),
            propagated=propagated,
            coverage=self.coverage,
        )
        self._observe(weak.index_select(0, support_rows), slots)
        self._last_step = step
        self._last_rows = eligible_ids.clone()
        self._targets = targets
        return targets

    def _cached(self, step: int, eligible_ids: Tensor) -> PropagatedTargets | None:
        """The first objective's targets, when the second asks at the same step."""
        if self._last_step is None or step > self._last_step:
            return None
        if step != self._last_step:
            raise LossError(
                f"SimMatch state was asked to move backwards from step "
                f"{self._last_step} to {step}"
            )
        if self._last_rows is None or not torch.equal(
            eligible_ids.to(self._last_rows.device), self._last_rows
        ):
            raise LossError(
                "the two SimMatch objectives prepared different eligible rows "
                f"at step {step}; shared state requires one row population"
            )
        if self._targets is None:
            raise LossError("SimMatch state recorded a step without targets")
        return self._targets

    def _slots_for(self, batch: XTYBatch, support_rows: RowIndex) -> Tensor:
        """The slot index of every support row, or a clear error."""
        if support_rows.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        if not bool(batch.t_observed.index_select(0, support_rows).all()):
            raise LossError(
                "SimMatch memory support rows must all have observed "
                "treatments; using hidden treatment values here would leak "
                "labels into the bank"
            )
        identities = batch.row_id.index_select(0, support_rows).detach()
        if int(torch.unique(identities).numel()) != int(identities.numel()):
            raise LossError(
                "SimMatch support rows repeat a row_id inside one batch; a slot "
                "is keyed by row identity and cannot be written twice in a step"
            )
        keys = self._slot_ids.to(identities.device)
        found = keys[None, :] == identities[:, None]
        if not bool(found.any(dim=-1).all()):
            missing = identities[~found.any(dim=-1)].tolist()
            raise LossError(
                f"SimMatch support row_id(s) {missing!r} have no slot. The bank "
                "keys one slot per observed row of the stage's training "
                "population, so a support row from outside it has no identity "
                "to write (card §3.2)."
            )
        slots = found.to(torch.uint8).argmax(dim=-1)
        labels = self._labels.to(identities.device).index_select(0, slots)
        if not bool(
            torch.equal(labels, batch.t.index_select(0, support_rows).detach())
        ):
            raise LossError(
                "SimMatch memory labels disagree with the batch's observed "
                "treatments; `Q_l` is fixed from the training population and a "
                "disagreement means the batch is not that population's rows"
            )
        return slots

    def _align(self, raw: Tensor) -> Tensor:
        """Distribution alignment over the current and prior batch marginals."""
        self._marginals.append(raw.mean(dim=0).clone())
        marginal = torch.stack(tuple(self._marginals)).mean(dim=0)
        aligned = raw / marginal.clamp_min(torch.finfo(raw.dtype).tiny)
        return aligned / aligned.sum(dim=-1, keepdim=True)

    def _bank(self, *, like: Tensor) -> Tensor:
        """A *copy* of the slots as they stand, before this step writes.

        Cloned rather than referenced so that read-before-write is a property
        of this method instead of a property of whichever tensor operation
        `_observe` happens to use.
        """
        if self._features is None:
            return torch.zeros(
                (self.size, like.shape[-1]), dtype=like.dtype, device=like.device
            )
        return self._features.to(device=like.device, dtype=like.dtype).clone()

    def _observe(self, embeddings: Tensor, slots: Tensor) -> None:
        """Eq. (12), on the slots this batch supports. Fills, then mixes."""
        if self._features is None:
            self._features = torch.zeros(
                (self.size, embeddings.shape[-1]),
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
        if slots.numel() == 0:
            return
        slots = slots.to(self._features.device)
        current = embeddings.detach().to(
            device=self._features.device, dtype=self._features.dtype
        )
        previous = self._features.index_select(0, slots)
        # A slot's first observation *fills* it (card §5, deviation 11): the
        # source's random initial features survive its warm-up epoch at
        # `m^n`, and two warm-up steps here would not dilute them.
        filled = self._filled.to(slots.device).index_select(0, slots)
        momentum = torch.where(
            filled[:, None],
            torch.full_like(previous, self._spec.memory_momentum),
            torch.zeros_like(previous),
        )
        updated = F.normalize(momentum * previous + (1.0 - momentum) * current, dim=-1)
        self._features = self._features.index_copy(0, slots, updated)
        self._filled = self._filled.index_fill(0, slots.to(self._filled.device), True)

    def _validate_inputs(
        self, probabilities: Tensor, embeddings: Tensor, batch: XTYBatch
    ) -> None:
        if probabilities.ndim != 2 or probabilities.shape != (
            batch.batch_size,
            self._classes,
        ):
            raise LossError(
                f"SimMatch weak probabilities must be "
                f"[{batch.batch_size}, {self._classes}], got "
                f"{tuple(probabilities.shape)}"
            )
        if embeddings.ndim != 2 or embeddings.shape[0] != batch.batch_size:
            raise LossError(
                "SimMatch weak embeddings must be [B, D], got "
                f"{tuple(embeddings.shape)} for B={batch.batch_size}"
            )


@dataclass(frozen=True)
class SimilarityMatchingTreatmentNLL:
    """Eq. (2), gated and charged against eq. (10)'s soft `hat p`.

    Not `PseudoLabelTreatmentNLL` with another flag: that objective hardens an
    arg max read straight off `T_GIVEN_X`, and neither half of that sentence
    is true here (card §5.1).
    """

    spec: SimilarityMatchingSpec
    target: Realisation
    weak_embedding: Realisation
    prediction: Realisation
    num_treatments: int
    sharpening: Literal["none"] = REQUIRED
    stop_grad: Literal["target"] = REQUIRED
    support_rows: Rows = "t_observed"
    rows: Rows = "t_missing"
    name: str = "similarity_matching_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "threshold": "losses.confidence_threshold",
        "temperatures": "losses.temperature",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        _validate_common(
            owner=type(self).__name__,
            name=self.name,
            spec=self.spec,
            target=self.target,
            weak_embedding=self.weak_embedding,
            num_treatments=self.num_treatments,
            support_rows=self.support_rows,
            rows=self.rows,
        )
        card_hyperparameters(self)
        if not isinstance(self.prediction, Realisation):
            raise LossError("SimMatch prediction must be a Realisation")
        if self.prediction == self.target:
            raise LossError("SimMatch target and prediction realisations must differ")
        if self.sharpening != "none":
            raise LossError(
                "SimMatch keeps `hat p` soft (§3.1); sharpening must be 'none'"
            )
        if self.stop_grad != "target":
            raise LossError("SimMatch detaches `hat p`; stop_grad must be 'target'")

    @property
    def threshold(self) -> float:
        return self.spec.threshold

    @property
    def temperatures(self) -> SimilarityMatchingTemperatures:
        return self.spec.temperatures

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
        """Yes: distribution alignment folds this batch's own marginal in."""
        return True

    def initial_state(self, population: TrainingPopulation | None) -> object:
        """One slot per observed training row, labelled from that population."""
        if population is None:
            raise LossError(
                f"objective {self.name!r} needs the stage's training "
                "population: the bank keys one feature slot per observed "
                "training row and takes `Q_l` from that row's observed "
                "treatment. A stage fed by ExternalBatches has neither "
                "(card §6.2)."
            )
        rows = population.rows
        observed = torch.nonzero(rows.t_observed, as_tuple=False).flatten()
        identities = rows.row_id.index_select(0, observed)
        order = torch.argsort(identities)
        return LabeledSimilarityMemory(
            classes=self.num_treatments,
            spec=self.spec,
            slot_ids=identities.index_select(0, order),
            labels=rows.t.index_select(0, observed).index_select(0, order),
        )

    def plan_details(self) -> tuple[str, ...]:
        return (
            *_shared_plan(self.spec, self.support_rows),
            "target = detached soft `hat p` from eq. (10); no arg max or sharpening",
            f"gate = max(hat p) >= {self.spec.threshold!r}, on the propagated "
            "target (eq. 2)",
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
        memory = ctx.objective_state(self.name, LabeledSimilarityMemory)
        targets = _prepare(
            memory,
            spec=self.spec,
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
        confidence = targets.semantic.max(dim=-1).values
        accepted = confidence >= self.spec.threshold
        eligible_loss = -torch.xlogy(
            targets.semantic, prediction.index_select(0, rows)
        ).sum(dim=-1) * accepted.to(prediction.dtype)
        per_row = torch.zeros(
            batch.batch_size, dtype=prediction.dtype, device=prediction.device
        ).index_copy(0, rows, eligible_loss)
        diagnostics = {
            "coverage": float(accepted.to(torch.float32).mean()),
            "accepted_confidence": (
                float(confidence[accepted].mean()) if bool(accepted.any()) else 0.0
            ),
            "bank_coverage": targets.coverage,
            "propagated": float(targets.propagated),
            # How far eq. (10) moved the target the ablation would have used.
            "propagation_shift": float(
                (targets.semantic - targets.aligned).abs().sum(dim=-1).mean()
            ),
        }
        return reduce_rows(per_row, rows, diagnostics=diagnostics)


@dataclass(frozen=True)
class LabeledMemoryInstanceConsistency:
    """Eqs. (3)--(5): `H(hat q, q^s)` over the owner's labelled slots.

    Reads the calibrated target through the named sibling's state, the
    `freematch` sibling read (`DESIGN.md` §4). It charges nothing while the
    bank is still warming up, which is the two ports' epoch-0 behaviour stated
    in optimiser steps (card §7).
    """

    spec: SimilarityMatchingSpec
    owner: str
    target: Realisation
    weak_embedding: Realisation
    prediction: Realisation
    num_treatments: int
    support_rows: Rows = "t_observed"
    rows: Rows = "t_missing"
    name: str = "labeled_memory_instance_consistency"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "temperatures": "losses.temperature",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        _validate_common(
            owner=type(self).__name__,
            name=self.name,
            spec=self.spec,
            target=self.target,
            weak_embedding=self.weak_embedding,
            num_treatments=self.num_treatments,
            support_rows=self.support_rows,
            rows=self.rows,
        )
        card_hyperparameters(self)
        if not require_str("SimMatch state owner", self.owner, error=LossError):
            raise LossError("LabeledMemoryInstanceConsistency.owner must be non-empty")
        if not isinstance(self.prediction, Realisation):
            raise LossError("SimMatch prediction must be a Realisation")
        if self.prediction == self.weak_embedding:
            raise LossError(
                "SimMatch instance consistency needs a strong realisation to "
                "charge eq. (5) against; `z^s` and `z^w` must differ"
            )

    @property
    def temperatures(self) -> SimilarityMatchingTemperatures:
        return self.spec.temperatures

    @property
    def stop_grad(self) -> Literal["target"]:
        """Eq. (5)'s `hat q` is a constant of theta; `q^s` is not."""
        return "target"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (Port.T_GIVEN_X, self.target),
                (Port.X_PROJ, self.weak_embedding),
                (Port.X_PROJ, self.prediction),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """Both weak inputs: either objective may prepare the shared state."""
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
            *_shared_plan(self.spec, self.support_rows),
            f"memory owner = {self.owner}",
            "target = detached `hat q` from eq. (8); `q^s` is eq. (4) against "
            "the same previous-step slots",
            "no gate; denominator = every eligible row (eq. 5)",
            "returns exactly 0 while the bank is warming up or uncovered",
            f"prediction = {self.prediction}",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        strong = _embedding(state, self.prediction, batch, owner=self.name)
        if rows.numel() == 0:
            return LossTerm.empty(like=strong)
        memory = ctx.objective_state(self.owner, LabeledSimilarityMemory)
        targets = _prepare(
            memory,
            spec=self.spec,
            state=state,
            batch=batch,
            rows=rows,
            support_rows=self.support_rows,
            target=self.target,
            weak_embedding=self.weak_embedding,
            step=ctx.global_step,
            owner=self.name,
        )
        per_row = torch.zeros(
            batch.batch_size, dtype=strong.dtype, device=strong.device
        )
        if targets.instance is None:
            return reduce_rows(
                per_row,
                rows,
                diagnostics={"bank_coverage": targets.coverage, "propagated": 0.0},
            )
        anchors = F.normalize(strong.index_select(0, rows), dim=-1)
        bank = targets.bank.to(device=anchors.device, dtype=anchors.dtype)
        strong_log_similarity = torch.log_softmax(
            anchors @ bank.transpose(0, 1) / self.spec.temperatures.instance_strong,
            dim=-1,
        )
        instance = targets.instance.to(dtype=anchors.dtype)
        eligible_loss = -(instance * strong_log_similarity).sum(dim=-1)
        per_row = per_row.index_copy(0, rows, eligible_loss)
        with torch.no_grad():
            slot_labels = memory.labels.to(device=anchors.device)
            nearest = (anchors @ bank.transpose(0, 1)).argmax(dim=-1)
            agreement = slot_labels.index_select(0, nearest)
            aggregated = torch.zeros_like(targets.aligned).index_add(
                1, slot_labels, instance
            )
        diagnostics = {
            "bank_coverage": targets.coverage,
            "propagated": float(targets.propagated),
            "target_entropy": float(
                -torch.xlogy(instance, instance).sum(dim=-1).mean()
            ),
            # Does the nearest labelled slot agree with the semantic target?
            "nearest_slot_agreement": float(
                (agreement == targets.semantic.argmax(dim=-1)).to(torch.float32).mean()
            ),
            "aggregated_confidence": float(aggregated.max(dim=-1).values.mean()),
        }
        return reduce_rows(per_row, rows, diagnostics=diagnostics)


def _prepare(
    memory: LabeledSimilarityMemory,
    *,
    spec: SimilarityMatchingSpec,
    state: State,
    batch: XTYBatch,
    rows: RowIndex,
    support_rows: Rows,
    target: Realisation,
    weak_embedding: Realisation,
    step: int,
    owner: str,
) -> PropagatedTargets:
    if memory.spec != spec:
        raise LossError(
            f"{owner} carries a different SimilarityMatchingSpec from the state "
            "owner it names"
        )
    weak_probabilities = treatment_distribution(
        state, Port.T_GIVEN_X, target, objective=owner
    ).probs
    weak = _embedding(state, weak_embedding, batch, owner=owner)
    return memory.prepare(
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
    spec: object,
    target: object,
    weak_embedding: object,
    num_treatments: object,
    support_rows: Rows,
    rows: Rows,
) -> None:
    if not require_str("SimMatch objective name", name, error=LossError):
        raise LossError(f"{owner}.name must be non-empty")
    if not isinstance(spec, SimilarityMatchingSpec):
        raise LossError(f"{owner}.spec must be a SimilarityMatchingSpec")
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


def _shared_plan(spec: SimilarityMatchingSpec, support_rows: Rows) -> tuple[str, ...]:
    return (
        f"support rows written to memory = {support_rows}",
        "memory = one slot per observed training row_id, labelled from the "
        "training population; read before write",
        f"memory update = normalize({spec.memory_momentum:g} * previous slot + "
        f"{1.0 - spec.memory_momentum:g} * current detached weak embedding), "
        "replacing outright on a slot's first observation",
        f"instance temperatures = {spec.temperatures!r} (eqs. 3, 4)",
        f"hat p = {spec.alpha:g} DA(p^w) + {1.0 - spec.alpha:g} "
        "aggregate(q^w) (eq. 10)",
        f"hat q = normalize(q^w * unfold(DA(p^w))) (eqs. 7, 8); unfolding "
        f"{'on' if spec.unfold else 'off'}",
        f"distribution alignment = unweighted mean of last "
        f"{spec.alignment_window} current-inclusive batch marginals",
        f"propagation and eq. (5) begin at step {spec.warmup_steps}, and not "
        "before every slot is filled",
    )


__all__ = [
    "LabeledMemoryInstanceConsistency",
    "LabeledSimilarityMemory",
    "PropagatedTargets",
    "SimilarityMatchingSpec",
    "SimilarityMatchingTemperatures",
    "SimilarityMatchingTreatmentNLL",
]
