"""Directional agreement between two embedding ports (§4.2, §5).

This is DoubleMatch eq. (3): take the penultimate features of a strongly
augmented row, pass them through a trainable projection head, and train that
direction toward the *weakly* augmented row's features, which are held constant.
It is a consistency loss in feature space rather than in label space, which is
what lets a recipe apply it to rows a confidence gate has rejected.

Four properties of that sentence are decisions rather than details.

* **The two sides are different ports.** Every other objective here reads one
  port under two realisations; this one reads `X_PROJ` on the prediction side
  and `X_REPR` on the target side, because eq. (3) puts the projection head on
  the strong branch only. No arrangement of one port over two realisations says
  that, so `prediction_port` and `target_port` are named separately and the
  asymmetry lives inside this objective rather than in the graph.
* **The target is detached, and only the target.** Eq. (3) says `z_i` is
  constant when the gradient is evaluated, and the reason is structural rather
  than incidental: a cosine has a trivial global optimum — map every row to one
  direction — that costs the representation everything, and there are no
  negatives here to punish it. The stop-gradient plus the predictor is what
  SimSiam's ablation shows is load-bearing against exactly that. So
  `stop_grad` takes one value; a symmetrised or undetached variant is a
  different method and waits for the card that states one (`DESIGN.md` §11).
  Neither is sufficient, and the first consumer measured that rather than
  inheriting the reassurance: on a representation that the encoder normalises
  onto the unit sphere, this term collapses it anyway, within ten steps, at
  every weight tried (`docs/recipes/doublematch.md` §5 deviation 9 and §6.2).
  A cosine is the whole geometry of a unit-sphere embedding, so agreement and
  collapse are the same move there. Anything reading `X_REPR` from an encoder
  with `normalisation="row_l2"` meets it.
* **Nothing is gated.** The whole point of the method is that this term sees
  the rows the pseudo-label term rejects, so the row population is the
  objective's `rows` and there is no mask inside the arithmetic. What the
  denominator counts is therefore exactly what `reduce_rows` counts.
* **Collapse is a diagnostic, not a loss.** The value alone cannot distinguish
  a representation that has learned the invariance from one that has stopped
  distinguishing rows: both drive the cosine to 1. The two concentration
  numbers below are what tells them apart, and they exist because no single
  loss number can (`DESIGN.md` §6.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import REQUIRED, card_hyperparameters
from xty2.core.errors import LossError, PortContractError, Xty2Error, require_str
from xty2.core.graph import Realisation, State
from xty2.core.loss import LossTerm, TrainContext, reduce_rows
from xty2.core.ports import Port, port_spec
from xty2.core.rows import RowIndex, Rows, validate_population

FeatureStopGrad = Literal["target"]
"""Which side carries no gradient. The target does; see the module note."""


@dataclass(frozen=True)
class CosineFeatureConsistency:
    """`-cos(prediction, target)` per row, target detached.

    DoubleMatch eq. (3), transcribed in `docs/recipes/doublematch.md` §3.1. The
    published expression has no additive constant; the reference implementation
    computes `1 - cos` instead, which trains identically and logs one higher.
    This is the paper's expression, for the same reason `InfoNCEContrastive`
    keeps SCARF's `- log n`.

    Attributes:
        prediction_port: The port carrying the trained side — DoubleMatch's
            `h(v_i)`, the projected features of the strong view.
        target_port: The port carrying the detached side — `z_i`, the
            unprojected features of the weak view. May be the same port as
            `prediction_port`; the realisations must then differ.
        prediction: The realisation the prediction is read under.
        target: The realisation the target is read under.
        stop_grad: Which side is detached. Binds `gradients.detached_targets`,
            so it has no default (`DESIGN.md` §9.1).
        rows: The population this term is entitled to. DoubleMatch's is `all`.
        name: Keys the per-objective log (§6.2).
    """

    prediction_port: Port
    target_port: Port
    prediction: Realisation
    target: Realisation
    stop_grad: FeatureStopGrad = REQUIRED
    rows: Rows = "all"
    name: str = "cosine_feature_consistency"

    CARD_KEYS: ClassVar[dict[str, str]] = {"stop_grad": "gradients.detached_targets"}

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("feature consistency name", self.name, error=LossError):
            raise LossError("CosineFeatureConsistency.name must be non-empty")
        for field, port in (
            ("prediction_port", self.prediction_port),
            ("target_port", self.target_port),
        ):
            if not isinstance(port, Port):
                raise LossError(
                    f"CosineFeatureConsistency.{field} must be a Port, got {type(port)}"
                )
            if port_spec(port).kind != "tensor":
                raise LossError(
                    f"CosineFeatureConsistency takes the cosine of two "
                    f"embeddings, but {field} {port!s} carries "
                    f"{port_spec(port).kind}. A directional agreement between "
                    "distributions is a divergence, which is what "
                    "ConsistencyLoss is for (DESIGN.md §11)."
                )
        prediction: object = self.prediction
        target: object = self.target
        if not isinstance(prediction, Realisation) or not isinstance(
            target, Realisation
        ):
            raise LossError(
                "CosineFeatureConsistency.prediction and target must be Realisations"
            )
        if (self.prediction_port, self.prediction) == (self.target_port, self.target):
            raise LossError(
                f"CosineFeatureConsistency matches {self.prediction_port!s} @ "
                f"{self.prediction} with itself; every cosine would be exactly "
                "1 and the term a constant. A positive pair is two "
                "realisations of one row, or two ports of one realisation "
                "(DoubleMatch eq. 3)."
            )
        if self.stop_grad != "target":
            raise LossError(
                f"CosineFeatureConsistency.stop_grad must be 'target', got "
                f"{self.stop_grad!r}. Eq. (3) holds `z_i` constant, and without "
                "a stop-gradient the term has a trivial optimum — one direction "
                "for every row — that no negative pair is present to punish. A "
                "symmetrised variant is a different method and waits for a card "
                "that states one (DESIGN.md §11)."
            )
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(f"CosineFeatureConsistency {self.name!r}: {error}") from (
                error
            )

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset(
            {
                (self.prediction_port, self.prediction),
                (self.target_port, self.target),
            }
        )

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """The target side, derived from `stop_grad` rather than restated."""
        return frozenset({(self.target_port, self.target)})

    def plan_details(self) -> tuple[str, ...]:
        """Which side is which, and what the arithmetic is.

        `requires` is a set, so the plan renders the two `(port, realisation)`
        pairs in a canonical order and cannot show which one is trained. The
        term is not symmetric — the other assignment trains the encoder through
        the weak view and holds the projection head constant, which is a
        different method with the same declaration — so the roles are printed
        and therefore enter the plan digest.
        """
        return (
            f"prediction (trained) = {self.prediction_port!s} @ {self.prediction}",
            f"target (detached) = {self.target_port!s} @ {self.target}",
            "value = -cosine(prediction, target), per row",
            "denominator = every eligible row; nothing is gated",
        )

    @property
    def batch_coupled(self) -> bool:
        """No: the cosine pairs one row's two realisations with each other.

        The diagnostics *are* batch-coupled — a concentration is a statistic of
        the eligible rows — and that is deliberately not what this flag is
        about: it declares whether the **value** changes when the batch is
        split, and a logged number that does is not a term the optimiser sees.
        """
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        prediction = self._embedding(
            state, self.prediction_port, self.prediction, batch
        )
        target = self._embedding(state, self.target_port, self.target, batch).detach()
        if prediction.shape[-1] != target.shape[-1]:
            raise LossError(
                f"CosineFeatureConsistency {self.name!r} compares "
                f"{self.prediction_port!s} of width {prediction.shape[-1]} with "
                f"{self.target_port!s} of width {target.shape[-1]}. Eq. (3)'s "
                "projection head is dimension-preserving, so a width mismatch "
                "is a mis-declared head rather than something to broadcast."
            )
        predicted = torch.nn.functional.normalize(prediction, dim=-1)
        matched = torch.nn.functional.normalize(target, dim=-1)
        per_row = -(predicted * matched).sum(dim=-1)
        return reduce_rows(
            per_row, rows, diagnostics=_concentrations(predicted, matched, rows)
        )

    def _embedding(
        self, state: State, port: Port, realisation: Realisation, batch: XTYBatch
    ) -> Tensor:
        value = state[realisation][port]
        if not isinstance(value, Tensor):
            raise PortContractError(
                f"objective {self.name!r} read port {str(port)!r} under "
                f"{realisation} as an embedding tensor, but it carries "
                f"{type(value)}. Its PortSpec is the contract (DESIGN.md §2)."
            )
        if value.shape[0] != batch.batch_size:
            raise LossError(
                f"CosineFeatureConsistency {self.name!r} got {value.shape[0]} "
                f"rows from {realisation} for a batch of {batch.batch_size}"
            )
        return value


def _concentrations(
    predicted: Tensor, matched: Tensor, rows: RowIndex
) -> dict[str, float]:
    """How close each side is to pointing everywhere at once.

    The norm of the mean unit vector over the eligible rows: `1.0` exactly when
    every row shares one direction, and about `1/sqrt(n)` for isotropic
    embeddings. This is the collapse detector the value cannot be: eq. (3) is
    minimised at `-1`, and a collapsed encoder reaches it — a *perfect* score on
    a representation that has stopped telling rows apart. Reported for both
    sides because they fail differently. The target side collapsing is the
    encoder collapsing, which is the failure SimSiam's stop-gradient exists to
    prevent; the prediction side collapsing on its own is the projection head
    absorbing the term, which leaves the encoder untouched and the loss looking
    healthy.

    Both are taken over the term's own eligible rows, so they answer a question
    about the rows this objective saw rather than about the batch.
    """
    if rows.numel() == 0:  # `reduce_rows` returns the zero term and drops these.
        return {}
    return {
        "prediction_concentration": _concentration(predicted, rows),
        "target_concentration": _concentration(matched, rows),
    }


def _concentration(unit: Tensor, rows: RowIndex) -> float:
    return float(unit.detach().index_select(0, rows).mean(dim=0).norm())


__all__ = ["CosineFeatureConsistency", "FeatureStopGrad"]
