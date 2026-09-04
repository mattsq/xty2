"""SoftMatch's truncated-Gaussian pseudo-label weighting (`softmatch.md`).

FixMatch keeps an artificial label when its probability clears one fixed `tau`;
FlexMatch and FreeMatch when it clears a threshold earned from the training
history. SoftMatch keeps *every* row and scales it instead: §2.1 rewrites the
whole family as one weighted cross-entropy (eq. 2) whose members differ only in
`lambda(p)`, and §3.1 replaces the indicator with a truncated Gaussian on the
confidence (eq. 5), centred and scaled by EMAs of the model's own batch
confidence moments (eqs. 6, 7). §3.2 adds Uniform Alignment (eqs. 8, 9).

Four readings of Algorithm 1 carry the fidelity of the port, and each is
visible in a declaration or in the plan rather than buried in `compute`:

* **The statistics are updated from the current batch, before that batch is
  weighted.** Lines 4-7 precede line 9, so a row's weight depends on the
  confidences of the *other* rows of its own batch and `batch_coupled` is
  `True` — as in FreeMatch, and unlike FlexMatch. Unlike FreeMatch, step 0
  folds its batch in rather than skipping it: eqs. (6) and (7) state no `t = 0`
  exception and both pinned implementations update before weighting, which is
  what makes the first step *almost* flat rather than exactly flat
  (card §2's first limitation).
* **The moments are estimated from unaligned confidence and compared against
  an aligned one.** Lines 4-5 read `max(p_i)`; line 9 reads `max(UA(p_i))`.
  Card §7's sixth unknown records that TorchSSL agrees with the algorithm and
  the later USB path does not, and that this port follows the algorithm.
* **UA changes the weight and not the label.** §3.2 is explicit that original
  predictions compute the pseudo-label and normalised ones the sample weight,
  so `labels` here is the arg max of the *unaligned* target. It is the stated
  difference from Distribution Alignment and the easiest thing to get wrong.
* **`lambda_max` is not in the policy.** It multiplies every row identically,
  so it factors out exactly into the mixer's `Weighted(..., weight=...)`, and
  card §4 binds it to `losses.weights` where a reader already looks for it.

One objective owns the three EMAs and nothing else reads them, so the sibling
state read `freematch` needed (`DESIGN.md` §4) is not engaged and the
idempotence obligation that comes with it does not arise.
"""

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
    """`mu_hat_t`, `sigma_hat_t^2` and `E_hat[p]`: the EMAs of eqs. (7) and (8).

    The state a `SoftWeightedTreatmentNLL` carries across the steps of one
    stage. Built by the executor once per stage *execution* — never held on an
    objective, so a recipe stays an immutable declaration and two runs of one
    compiled recipe are identical (`core/loss.py`, `StatefulObjective`).

    It needs no `TrainingPopulation`: every statistic is an average over the
    batch, so there is no `N` to count and no row identity to key. That is the
    third consumer of the signature `flexmatch.md` §5.1 chose for exactly this
    reason.

    All three are held in float64. They are running sums over thousands of
    steps at `m = 0.999`, where a float32 EMA loses the tail of its own history
    to rounding; `aligned` and `weights` cast at the point of use.
    """

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
        """Algorithm 1 lines 4-7, folding one batch into the three EMAs.

        Called before the same batch is weighted, and from `compute` alone —
        this state has one writer and one reader. The step guard is therefore
        not the sibling-read obligation `SelfAdaptiveThresholds` carries but a
        cheaper property: a term recomputed at one step (a diagnostic pass, a
        mixer that evaluates twice) must not decay the EMAs twice, because the
        second fold would move a number no reader of the declaration could see.
        A repeat at one step whose row count differs is refused rather than
        ignored, for the reason `freematch.md` §3.2 gives about one population.

        Args:
            step: `ctx.global_step`. Step 0 folds its batch in: eqs. (6) and
                (7) state no `t = 0` exception and Algorithm 1 updates before
                it weights, which is card §2's first limitation.
            probs: `[n, K]` **unaligned** weak-view probabilities over the rows
                the objective is entitled to, already detached. Unaligned
                because eq. (6) reads `max(p_i)`, not `max(UA(p_i))`.
        """
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

    def weights(self, probs: Tensor, *, apply_alignment: bool = True) -> Tensor:
        """Eq. (9), optionally exposing eq. (5)'s pre-UA diagnostic profile.

        ``apply_alignment=False`` does not change training. It evaluates the
        paper's all-class-without-UA ablation with this state's same Gaussian
        moments, which is also the before-UA half of appendix A.7's class-wise
        weight diagnostic.
        """
        weighted_probs = self.aligned(probs) if apply_alignment else probs
        confidence = weighted_probs.max(dim=-1).values
        mean = self._mean.to(device=probs.device, dtype=probs.dtype)
        variance = self._variance.to(device=probs.device, dtype=probs.dtype)
        delta = torch.clamp(confidence - mean, max=0.0)
        denominator = 2.0 * variance / (self._policy.n_sigma**2)
        return torch.exp(-(delta.square() / denominator))


@dataclass(frozen=True)
class SoftWeightedTreatmentNLL:
    """`lambda(p_b) * -log p(t = arg max p_b | x)` — eq. (2) at eq. (9).

    FixMatch's eq. (4) with the 0/1 gate replaced by the continuous weight of
    eqs. (5)-(9), and with the statistics update of Algorithm 1 lines 4-7
    alongside it. The weight is computed here, from the model's own
    predictions, in the place `PseudoLabelTreatmentNLL` computes its mask —
    it is not `batch.weight`, which is a property of a row supplied by the data
    and reaches `ObservedOutcomeNLL` alone (`pseudo_label.py`, card §5.2).

    Attributes:
        port: The treatment-distribution port both sides read.
        target: The realisation the artificial label comes from — the weak
            view, `p_i = p(y | omega(u_i))`. It is also the realisation the
            three EMAs are estimated from, unaligned.
        prediction: The realisation the label is charged against — the strong
            view, `p(y | Omega(u_i))`.
        num_treatments: `C`. Not paper-governed and not a card key — it is a
            property of the schema, and a component takes it the same way. It
            is a field rather than something read from the batch because
            `initial_state` runs before any batch exists, and `compute` checks
            it against `ctx.schema` for the reason `DESIGN.md` §3.1 gives.
        weighting: The whole `lambda(p)` rule. Binds
            `losses.confidence_threshold`, so it has no default
            (`DESIGN.md` §9.1) — and holds no threshold, which is why the key
            carries a policy object rather than a float (card §4).
        sharpening: How the label is formed. Binds `losses.sharpening`. Eq. (2)
            charges a hard arg max and UA is a normalisation rather than a
            temperature, so `hard` is the only reviewed value.
        stop_grad: Which side is detached. Binds `gradients.detached_targets`.
        rows: The population the term is entitled to, and the population the
            EMAs are averaged over. FixMatch's footnote 2 — inherited through
            §2.1's restatement of that framework (card §7's seventh unknown) —
            puts every labelled row into `U` as well, so this recipe's value is
            `all`.
        name: Keys the per-objective log (§6.2) and the per-stage state.
    """

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
