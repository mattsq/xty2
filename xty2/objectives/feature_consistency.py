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
  `stop_grad` is explicit: DoubleMatch retains `target`; SimSiam's declared
  no-stop-gradient control uses `none` (SimSiam card section 6).
  Neither is sufficient, and the first consumer measured that rather than
  inheriting the reassurance: it collapsed the representation within ten steps,
  at every weight from 0.5 down to 0.01, and never recovered
  (`docs/recipes/doublematch.md` §6.2). The cause is **scale**, not geometry —
  see the next note — and the control that establishes it (hold the encoder's
  normalisation fixed, change only its initialisation) is one an earlier
  version of that card asserted a mechanism without running.
* **Nothing is gated.** The whole point of the method is that this term sees
  the rows the pseudo-label term rejects, so the row population is the
  objective's `rows` and there is no mask inside the arithmetic. What the
  denominator counts is therefore exactly what `reduce_rows` counts.
* **The cosine's gradient carries `1 / ||prediction||`.** `F.normalize`'s
  backward does, and nothing here floors it beyond its `eps=1e-12`. The term is
  therefore only as well-scaled as the vectors it is handed: a producer whose
  output can approach zero norm makes this loss arbitrarily loud, and a
  producer that normalises its own output — an encoder ending in `row_l2` —
  hands the same factor to *its* upstream instead, sized by the
  pre-normalisation activation. That is not hypothetical and it is not about
  the unit sphere: `doublematch.md` §6.2 measures a representation whose norm
  is 0.011 under one initialisation and 1.94 under another, the same encoder
  and the same normalisation, and the first collapses under this term while the
  second does not. Before pointing this objective at a new pair of ports, check
  what norm they carry.
* **Collapse is a diagnostic, not a loss.** The value alone cannot distinguish
  a representation that has learned the invariance from one that has stopped
  distinguishing rows: both drive the cosine to 1. The two concentration
  numbers below are what tells them apart, and they exist because no single
  loss number can (`DESIGN.md` §6.2).
"""

from __future__ import annotations

import math
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

FeatureStopGrad = Literal["target", "none"]
"""Target detach, or the explicitly declared SimSiam ablation."""


@dataclass(frozen=True)
class CosineFeatureConsistency:
    """`-cos(prediction, target)` per row, with an explicit target-gradient policy.

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
            so it has no default (`DESIGN.md` §9.1). `none` is the SimSiam
            no-stop-gradient control.
        epsilon: Floor for each vector's norm; recorded in plan details.
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
    epsilon: float = 1e-12

    CARD_KEYS: ClassVar[dict[str, str]] = {"stop_grad": "gradients.detached_targets"}

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise LossError(
                "CosineFeatureConsistency.epsilon must be finite and positive"
            )
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
        if self.stop_grad not in ("target", "none"):
            raise LossError(
                f"CosineFeatureConsistency.stop_grad must be 'target' or 'none', got "
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
        return (
            frozenset({(self.target_port, self.target)})
            if self.stop_grad == "target"
            else frozenset()
        )

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
            f"target ({'detached' if self.stop_grad == 'target' else 'trained'}) "
            f"= {self.target_port!s} @ {self.target}",
            f"separate L2 normalisation; epsilon={self.epsilon:g}",
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
        owner = f"CosineFeatureConsistency {self.name!r}"
        prediction = _embedding(
            state, self.prediction_port, self.prediction, batch, owner
        )
        target = _embedding(state, self.target_port, self.target, batch, owner)
        if self.stop_grad == "target":
            target = target.detach()
        if prediction.shape[-1] != target.shape[-1]:
            raise LossError(
                f"CosineFeatureConsistency {self.name!r} compares "
                f"{self.prediction_port!s} of width {prediction.shape[-1]} with "
                f"{self.target_port!s} of width {target.shape[-1]}. Eq. (3)'s "
                "projection head is dimension-preserving, so a width mismatch "
                "is a mis-declared head rather than something to broadcast."
            )
        predicted = torch.nn.functional.normalize(prediction, dim=-1, eps=self.epsilon)
        matched = torch.nn.functional.normalize(target, dim=-1, eps=self.epsilon)
        per_row = -(predicted * matched).sum(dim=-1)
        return reduce_rows(
            per_row, rows, diagnostics=_concentrations(predicted, matched, rows)
        )


@dataclass(frozen=True)
class NormalizedSquaredFeatureConsistency:
    """BYOL eq. (2): the squared distance between two L2-normalised embeddings.

    `docs/recipes/byol.md` §3.1 transcribes it. The pinned
    `utils/helpers.regression_loss` is

    ```python
    normed_x, normed_y = l2_normalize(x, axis=-1), l2_normalize(y, axis=-1)
    return jnp.sum((normed_x - normed_y)**2, axis=-1)
    ```

    and this is that expression, not the `2 - 2 cos` the paper writes beside it.
    The two agree wherever both vectors have a norm, and the place they do not
    is the reason this objective exists rather than a second `stop_grad` on
    `CosineFeatureConsistency`:

    * **the floor is on the squared norm, not on the norm.** `l2_normalize`
      divides by `sqrt(max(sum(v**2), 1e-12))`, so a vector shorter than `1e-6`
      is divided by `1e-6` and comes out *shorter than unit length*. Torch's
      `F.normalize(v, eps=1e-12)` divides by `max(||v||, 1e-12)` instead, which
      is the same policy with a floor a million times smaller, and a cosine
      formulation has no floor to disagree about at all — it reads `-1` for any
      two parallel vectors however short. Below the floor the value here is a
      genuine squared distance between two sub-unit vectors and the second
      equality of eq. (2) simply does not hold. That is the source's behaviour,
      and it is what the Tier 0 oracle pins;
    * **the value is a distance, so it is non-negative and minimised at zero,**
      where the cosine term is minimised at `-1`. Weights and logged numbers do
      not transfer between the two.

    BYOL sums the two directional distances and takes one mean over rows
    (`loss_fn`: `repr_loss = a + b`, then `jnp.mean`). That is two terms of
    weight 1 under `reduction="mean"` here, and specifically *not* a half-weight
    each: halving is SimSiam's convention, and `byol.md` §6.3 requires this
    objective to be observed failing under exactly that mutant.

    The target side is detached, and under BYOL it is also a teacher realisation
    the executor evaluates under `torch.no_grad()`. The detach is still declared
    rather than inferred: the source keeps its own `jax.lax.stop_gradient` for
    the same reason, and the zero-decay arm of §6.2 — whose target parameters
    equal the online ones after every update — is a control only if the gradient
    policy is unchanged alongside it.

    Attributes:
        prediction_port: The port carrying the trained side, `q_theta(z_theta)`.
        target_port: The port carrying the target side, `z'_xi`. BYOL reads a
            projection here, never a target prediction.
        prediction: The realisation the prediction is read under.
        target: The realisation the target is read under.
        stop_grad: Which side is detached. Binds `gradients.detached_targets`,
            so it has no default (`DESIGN.md` §9.1).
        epsilon: The floor on the **squared** norm; recorded in plan details.
        rows: The population this term is entitled to. BYOL's is `all`.
        name: Keys the per-objective log (§6.2).
    """

    prediction_port: Port
    target_port: Port
    prediction: Realisation
    target: Realisation
    stop_grad: FeatureStopGrad = REQUIRED
    rows: Rows = "all"
    name: str = "normalized_squared_feature_consistency"
    epsilon: float = 1e-12

    CARD_KEYS: ClassVar[dict[str, str]] = {"stop_grad": "gradients.detached_targets"}

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise LossError(
                "NormalizedSquaredFeatureConsistency.epsilon must be finite and "
                "positive"
            )
        if not require_str("feature consistency name", self.name, error=LossError):
            raise LossError(
                "NormalizedSquaredFeatureConsistency.name must be non-empty"
            )
        for field, port in (
            ("prediction_port", self.prediction_port),
            ("target_port", self.target_port),
        ):
            if not isinstance(port, Port):
                raise LossError(
                    f"NormalizedSquaredFeatureConsistency.{field} must be a Port, "
                    f"got {type(port)}"
                )
            if port_spec(port).kind != "tensor":
                raise LossError(
                    f"NormalizedSquaredFeatureConsistency takes the squared "
                    f"distance between two embeddings, but {field} {port!s} "
                    f"carries {port_spec(port).kind}."
                )
        prediction: object = self.prediction
        target: object = self.target
        if not isinstance(prediction, Realisation) or not isinstance(
            target, Realisation
        ):
            raise LossError(
                "NormalizedSquaredFeatureConsistency.prediction and target must "
                "be Realisations"
            )
        if (self.prediction_port, self.prediction) == (self.target_port, self.target):
            raise LossError(
                f"NormalizedSquaredFeatureConsistency matches "
                f"{self.prediction_port!s} @ {self.prediction} with itself; every "
                "distance would be exactly 0 and the term a constant. BYOL's "
                "pair is one view's online prediction against the other view's "
                "target projection (eq. 2)."
            )
        if self.stop_grad not in ("target", "none"):
            raise LossError(
                "NormalizedSquaredFeatureConsistency.stop_grad must be 'target' "
                f"or 'none', got {self.stop_grad!r}. BYOL declares 'target': eq. "
                "(3) differentiates with respect to theta alone, and without the "
                "stop-gradient the pair has the trivial optimum the method has "
                "no negatives to punish."
            )
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(
                f"NormalizedSquaredFeatureConsistency {self.name!r}: {error}"
            ) from error

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
        return (
            frozenset({(self.target_port, self.target)})
            if self.stop_grad == "target"
            else frozenset()
        )

    def plan_details(self) -> tuple[str, ...]:
        """Which side is which, and the arithmetic, including the floor.

        The floor is printed because it is the whole difference between this
        term and a cosine, and because `byol.md` §4 states it as a source
        mechanic rather than as a numerical convenience.
        """
        return (
            f"prediction (trained) = {self.prediction_port!s} @ {self.prediction}",
            f"target ({'detached' if self.stop_grad == 'target' else 'trained'}) "
            f"= {self.target_port!s} @ {self.target}",
            f"separate L2 normalisation by sqrt(max(sum(v^2), {self.epsilon:g}))",
            "value = sum over coordinates of (normalised difference)^2, per row",
            "denominator = every eligible row; nothing is gated",
        )

    @property
    def batch_coupled(self) -> bool:
        """No: the distance pairs one row's two realisations with each other."""
        return False

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del ctx
        owner = f"NormalizedSquaredFeatureConsistency {self.name!r}"
        prediction = _embedding(
            state, self.prediction_port, self.prediction, batch, owner
        )
        target = _embedding(state, self.target_port, self.target, batch, owner)
        if self.stop_grad == "target":
            target = target.detach()
        if prediction.shape[-1] != target.shape[-1]:
            raise LossError(
                f"NormalizedSquaredFeatureConsistency {self.name!r} compares "
                f"{self.prediction_port!s} of width {prediction.shape[-1]} with "
                f"{self.target_port!s} of width {target.shape[-1]}. BYOL's "
                "predictor maps the projection to its own width, so a mismatch "
                "is a mis-declared head rather than something to broadcast."
            )
        predicted = squared_norm_floor_normalize(prediction, self.epsilon)
        matched = squared_norm_floor_normalize(target, self.epsilon)
        per_row = (predicted - matched).pow(2).sum(dim=-1)
        return reduce_rows(
            per_row, rows, diagnostics=_concentrations(predicted, matched, rows)
        )


def squared_norm_floor_normalize(value: Tensor, epsilon: float) -> Tensor:
    """`utils/helpers.l2_normalize`: `v * rsqrt(max(sum(v**2), epsilon))`.

    A free function rather than a method because `epsilon` floors the *squared*
    norm, which is the one thing about this objective a reader is most likely to
    assume is `F.normalize` with a different constant. Anything else that wants
    the source's normalisation — a diagnostic, a Tier 1 probe — calls this
    rather than writing the expression a second time.
    """
    squared = value.pow(2).sum(dim=-1, keepdim=True)
    return value * torch.rsqrt(squared.clamp_min(epsilon))


def _embedding(
    state: State, port: Port, realisation: Realisation, batch: XTYBatch, owner: str
) -> Tensor:
    """The rank-two, finite, batch-sized tensor a feature term reads.

    Shared by both objectives in this module because the three rejections are
    the same three rejections, and a second copy of them is a second thing to
    keep in step with `PortSpec`.
    """
    value = state[realisation][port]
    if not isinstance(value, Tensor):
        raise PortContractError(
            f"{owner} read port {str(port)!r} under "
            f"{realisation} as an embedding tensor, but it carries "
            f"{type(value)}. Its PortSpec is the contract (DESIGN.md §2)."
        )
    if value.ndim != 2 or value.shape[1] < 1:
        raise LossError(f"{owner} requires rank-two embeddings")
    if not bool(torch.isfinite(value).all()):
        raise LossError(f"{owner} requires finite embeddings")
    if value.shape[0] != batch.batch_size:
        raise LossError(
            f"{owner} got {value.shape[0]} rows from {realisation} for a batch "
            f"of {batch.batch_size}"
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


__all__ = [
    "CosineFeatureConsistency",
    "FeatureStopGrad",
    "NormalizedSquaredFeatureConsistency",
    "squared_norm_floor_normalize",
]
