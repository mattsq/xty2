"""Confidence-gated hard pseudo-labelling across two realisations (§4.2, §5).

This is FixMatch eq. (4): read a treatment distribution under one realisation —
the *target*, a weak view — take its arg max as a hard label, keep the label
only where its probability clears `threshold`, and charge cross-entropy against
the same port under a second realisation, the *prediction* from a strong view.

Three properties of that sentence are decisions rather than details, and each is
made visible to the compiler rather than left inside `compute`:

* **The target is detached.** `arg max` has no gradient and the gate is a step
  function, so the label is a constant of the parameters whatever the code does.
  `stop_grad` states it anyway, because `DESIGN.md` §4 requires a stop-gradient
  to be declared — `detaches` is what tells the dead-trainable rule that only
  the prediction side is trained here.
* **The denominator counts rejected rows.** Eq. (4) divides by the whole
  unlabelled batch, not by the rows that cleared the gate, so a step where
  nothing is confident contributes zero rather than an average over an empty
  set. That is a per-row mask followed by a mean over *every* eligible row, and
  it is emitted through `plan_details` since no port, row population or card key
  would otherwise reveal which denominator ran.
* **Coverage is logged, not inferred.** `DESIGN.md` §6.2 asks pseudo-labelling
  recipes for the fraction above the threshold; it is the paper's own mask rate
  (eq. 6) and it is the number that distinguishes a working gate from an inert
  one. Calibration of the accepted labels — the paper's impurity, eq. (5) —
  needs the true `t`, which by definition is not in the batch on the rows this
  term exists for; the Tier 1 and Tier 2 fixtures measure it against their own
  ground truth instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import LossError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows, treatment_distribution
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population

Sharpening = Literal["hard"]
"""How the artificial label is post-processed.

One value so far, and it is the paper's own taxonomy rather than an accident:
FixMatch's table 1 lists its post-processing as *pseudo-labelling* — the arg max
— where UDA and MixMatch are listed as *sharpening*, a temperature applied to a
soft target. A recipe that needs the second names a second value and brings
`losses.temperature` with it (`DESIGN.md` §11).
"""

PseudoLabelStopGrad = Literal["target"]
"""Which side carries no gradient. The target always does; see the module note."""


@dataclass(frozen=True)
class PseudoLabelTreatmentNLL:
    """`1(max q >= tau) * -log p(t = argmax q | x)` across two realisations.

    Attributes:
        port: The treatment-distribution port both sides read.
        target: The realisation the artificial label comes from — FixMatch's
            weak view, `q_b = p_m(y | alpha(u_b))`.
        prediction: The realisation the label is charged against — the strong
            view, `p_m(y | A(u_b))`.
        threshold: `tau`. Binds `losses.confidence_threshold`, so it has no
            default (`DESIGN.md` §9.1).
        sharpening: How the label is formed. Binds `losses.sharpening`.
        stop_grad: Which side is detached. Binds `gradients.detached_targets`.
        rows: The population the term is entitled to. FixMatch's footnote 2
            puts every labelled row into `U` as well, so its own value is
            `all`; a recipe that wants the missing-treatment rows alone says
            `t_missing` and the compiler intersects the stage scope in.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    target: Realisation
    prediction: Realisation
    threshold: float = REQUIRED
    sharpening: Sharpening = REQUIRED
    stop_grad: PseudoLabelStopGrad = REQUIRED
    rows: Rows = "all"
    name: str = "pseudo_label_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "threshold": "losses.confidence_threshold",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("pseudo-label objective name", self.name, error=LossError):
            raise LossError("PseudoLabelTreatmentNLL.name must be non-empty")
        if not isinstance(self.port, Port):
            raise LossError(
                f"PseudoLabelTreatmentNLL.port must be a Port, got {type(self.port)}"
            )
        if port_spec(self.port).kind != "treatment_distribution":
            raise LossError(
                f"PseudoLabelTreatmentNLL labels a treatment, but port "
                f"{self.port!s} carries {port_spec(self.port).kind}. A hard "
                "pseudo-label over another port waits for the card that needs "
                "one (DESIGN.md §11)."
            )
        target: object = self.target
        prediction: object = self.prediction
        if not isinstance(target, Realisation) or not isinstance(
            prediction, Realisation
        ):
            raise LossError(
                "PseudoLabelTreatmentNLL.target and prediction must be Realisations"
            )
        if self.target == self.prediction:
            raise LossError(
                f"PseudoLabelTreatmentNLL labels {self.target} with itself. The "
                "method is a weak-view label charged against a strong-view "
                "prediction (FixMatch eq. 4); one realisation on both sides is "
                "entropy minimisation, which is a different objective."
            )
        if self.sharpening != "hard":
            raise LossError(
                f"PseudoLabelTreatmentNLL.sharpening must be 'hard', got "
                f"{self.sharpening!r}. A temperature-sharpened soft target is "
                "UDA's post-processing, not FixMatch's (paper table 1)."
            )
        if self.stop_grad != "target":
            raise LossError(
                f"PseudoLabelTreatmentNLL.stop_grad must be 'target', got "
                f"{self.stop_grad!r}. The label is an arg max and the gate a "
                "step function, so no gradient exists on that side to keep."
            )
        if isinstance(self.threshold, bool) or not isinstance(
            self.threshold, int | float
        ):
            raise LossError(
                f"PseudoLabelTreatmentNLL.threshold must be a number in [0, 1], "
                f"got {type(self.threshold)}"
            )
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise LossError(
                f"PseudoLabelTreatmentNLL.threshold is a probability and must be "
                f"in [0, 1], got {self.threshold!r}"
            )
        object.__setattr__(self, "threshold", float(self.threshold))
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(
                f"PseudoLabelTreatmentNLL {self.name!r}: {error}"
            ) from error

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.target), (self.port, self.prediction)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """The target side, derived from `stop_grad` rather than restated."""
        return frozenset({(self.port, self.target)})

    def plan_details(self) -> tuple[str, ...]:
        """Arithmetic the ports, rows and card keys do not already say."""
        return (
            "label = arg max of the target realisation",
            "gate = max prob >= threshold",
            "denominator = every eligible row; rejected rows contribute 0",
        )

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        target = treatment_distribution(
            state, self.port, self.target, objective=self.name
        ).probs.detach()
        prediction = treatment_distribution(
            state, self.port, self.prediction, objective=self.name
        )
        confidence, labels = target.max(dim=-1)
        accepted = confidence >= float(self.threshold)
        per_row = -prediction.log_prob(labels) * accepted.to(target.dtype)
        if per_row.shape[0] != batch.batch_size:
            raise LossError(
                f"PseudoLabelTreatmentNLL {self.name!r} got {per_row.shape[0]} "
                f"rows from its realisations for a batch of {batch.batch_size}"
            )
        return reduce_rows(per_row, rows, diagnostics=_gate(confidence, accepted, rows))


def _gate(
    confidence: torch.Tensor, accepted: torch.Tensor, rows: RowIndex
) -> dict[str, float]:
    """The paper's mask rate (eq. 6), plus the confidence behind it.

    Both are reported over the eligible rows rather than the whole batch, so
    that they answer "of the rows this term saw, how many did it use?" — the
    question `DESIGN.md` §6.2 asks of a pseudo-labelling recipe. `n` is already
    logged beside them, so the batch-level figure is recoverable.
    """
    if rows.numel() == 0:  # `reduce_rows` returns the zero term and drops these.
        return {}
    eligible = accepted.index_select(0, rows)
    retained = int(eligible.sum())
    mean_confidence = 0.0
    if retained:
        selected = confidence.index_select(0, rows)[eligible]
        mean_confidence = float(selected.mean())
    return {
        "coverage": retained / int(rows.numel()),
        "accepted_confidence": mean_confidence,
    }


__all__ = ["PseudoLabelStopGrad", "PseudoLabelTreatmentNLL", "Sharpening"]
