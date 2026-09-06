"""Curriculum Pseudo Labeling — FlexMatch's per-class threshold (`flexmatch.md`).

FixMatch keeps an artificial label when its probability clears one fixed `tau`.
FlexMatch keeps it when it clears `T_t(c)`, a threshold **per class** that rises
with how much of the unlabelled set the model currently assigns to that class
above `tau` (eq. 5-7, 11, 12). The threshold is therefore a function of the
training history rather than of the batch, which is the whole reason this module
exists beside `pseudo_label.py` instead of inside it.

Three properties of that sentence are decisions, and each is visible in the plan
rather than buried in `compute`:

* **Two gates, not one.** The *mark* is set at the fixed `tau` (algorithm 1
  line 14); the *loss* is gated at `T_t(arg max q_b)` (eq. 8). Collapsing them
  would make the learning effect self-reinforcing — a class whose threshold had
  fallen would mark more rows, which would raise its own `beta` — and the
  paper's two gates are what stop that.
* **The threshold is read before the marks are updated.** Algorithm 1 computes
  every `T(c)` (lines 4-12) before it touches a mark (lines 13-17) and then
  takes the loss with the thresholds it already has (line 18). So a row's gate
  never depends on the other rows of its own batch, which is why
  `batch_coupled` is false even though the term carries state.
* **Marks are sticky and keyed by `row_id`.** Line 15 is the only write and
  nothing restores `-1`, so a row that clears `tau` once counts towards
  `sigma` for the rest of the stage. A `QuotaSampler` draw is a fresh subset
  every step, so the only identity a row has across steps is the unique
  `row_id` of `DESIGN.md` §7.1.

The denominator convention and the row-weight convention are
`PseudoLabelTreatmentNLL`'s, deliberately: eq. (8) divides by `mu B`, the whole
unlabelled batch, so a step where nothing clears its class threshold contributes
zero rather than an average over an empty set; and `batch.weight` reaches
`ObservedOutcomeNLL` and nothing else.
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
from xty2.core.rows import RowIndex, Rows, row_mask, validate_population

UNUSED = -1
"""Algorithm 1 line 2: the mark of a row no step has yet accepted."""

CurriculumMapping = Literal["convex", "identity"]
"""`M` of eq. (12), by the paper's own two names.

`convex` is `M(x) = x / (2 - x)`, the function §3.3 "intuitively choose[s]" and
§4.4's ablation reports best. `identity` is eq. (7), which §3.3 calls "a special
case by setting `M` to the identity function" — it is here because the paper
names it and ablates it, not as a convenience default, and neither value has
one.
"""


@dataclass(frozen=True, repr=False)
class CurriculumThreshold:
    """The gate rule of eqs. (7), (11) and (12) as one declared value.

    One object rather than three fields on the objective, because
    `losses.confidence_threshold` names *the rule by which confidence gates a
    row* and `card_keys.py` refuses two fields bound to one canonical key with
    the instruction to "bind one field, holding a tuple if the paper states
    several numbers together". FlexMatch's gate is exactly that: `tau`, the
    warm-up and the mapping are three parts of one rule and no one of them
    describes it.

    Attributes:
        tau: The fixed threshold of eq. (5) and algorithm 1 line 14, and the
            ceiling `T_t(c)` rises towards in eq. (12).
        warm_up: §3.2's eq. (11) — divide by `max{max_c sigma, N - sum_c sigma}`
            rather than by `max_c sigma`, so that thresholds "gradually rise
            from 0 until the number of unused unlabeled data is no longer
            predominant". Algorithm 1's lines 6-9 branch between eq. (11) and
            eq. (6); the branch is redundant, since eq. (11)'s denominator
            *is* `max_c sigma` whenever that dominates, so `True` computes both
            lines exactly and `False` is the ablation of §3.2.
        mapping: `M` of eq. (12).
    """

    tau: float
    warm_up: bool
    mapping: CurriculumMapping

    def __post_init__(self) -> None:
        if isinstance(self.tau, bool) or not isinstance(self.tau, int | float):
            raise LossError(
                f"CurriculumThreshold.tau must be a number in [0, 1], got "
                f"{type(self.tau)}"
            )
        if not 0.0 <= float(self.tau) <= 1.0:
            raise LossError(
                f"CurriculumThreshold.tau is a probability and must be in "
                f"[0, 1], got {self.tau!r}"
            )
        object.__setattr__(self, "tau", float(self.tau))
        if not isinstance(self.warm_up, bool):
            raise LossError(
                f"CurriculumThreshold.warm_up must be a bool, got {type(self.warm_up)}"
            )
        if self.mapping not in ("convex", "identity"):
            raise LossError(
                f"CurriculumThreshold.mapping must be 'convex' or 'identity', "
                f"got {self.mapping!r}. Those are the two eq. (12) names — the "
                "paper's chosen `x / (2 - x)` and the eq. (7) special case it "
                "generalises."
            )

    def __repr__(self) -> str:
        """The form card §4 writes, so the plan and the card can be diffed.

        `repr` rather than `__str__` alone: the plan renders hyperparameters
        with `!r` and the card cross-check compares `str`, and a value that
        printed two ways would put the reviewer's two sources out of step.
        """
        warm = "true" if self.warm_up else "false"
        return f"curriculum(tau={self.tau:g}, warm_up={warm}, mapping={self.mapping})"

    def describe(self) -> tuple[str, ...]:
        """The gate as stable `plan_details` lines (`DESIGN.md` §4)."""
        beta = (
            "beta(c) = sigma(c) / max(max_c sigma, N - sum_c sigma) (eq. 11)"
            if self.warm_up
            else "beta(c) = sigma(c) / max_c sigma (eq. 6, warm-up off)"
        )
        mapped = (
            "T(c) = M(beta(c)) * tau, M(x) = x / (2 - x) (eq. 12)"
            if self.mapping == "convex"
            else "T(c) = beta(c) * tau (eq. 7, identity mapping)"
        )
        return (beta, mapped)

    def map(self, beta: Tensor) -> Tensor:
        """`M` of eq. (12) applied elementwise to `beta`."""
        if self.mapping == "identity":
            return beta
        return beta / (2.0 - beta)


class CurriculumStatus:
    """Algorithm 1's `\\hat u_n`: one mark per row, `-1` until it clears `tau`.

    The state a `CurriculumPseudoLabelTreatmentNLL` carries across the steps of
    one stage. Built by the executor from the stage's training population, once
    per stage *execution* — never held on the objective, so a recipe stays an
    immutable declaration and two runs of one compiled recipe are identical
    (`core/loss.py`, `StatefulObjective`).

    Marks are stored against `row_id` rather than against a batch position,
    because a sampler draw is a fresh subset each step; the ids are sorted once
    at construction so a lookup is a `searchsorted` rather than a scan.
    """

    __slots__ = ("_marks", "_policy", "_row_ids")

    def __init__(self, row_ids: Tensor, policy: CurriculumThreshold) -> None:
        if row_ids.ndim != 1 or row_ids.dtype != torch.long:
            raise LossError(
                f"CurriculumStatus takes a [N] long row_id tensor, got shape "
                f"{tuple(row_ids.shape)} of {row_ids.dtype}"
            )
        if row_ids.numel() == 0:
            raise LossError(
                "CurriculumStatus was built over an empty population. `N` is "
                "eq. (11)'s denominator while nothing is marked, so a zero "
                "there would make every threshold 0/0 and admit every row "
                "silently (flexmatch.md §7)."
            )
        unique = torch.unique(row_ids)
        if unique.numel() != row_ids.numel():
            raise LossError(
                "CurriculumStatus was built over repeated row_ids. A mark is "
                "keyed by row identity (DESIGN.md §7.1), so a repeat would let "
                "one row hold two marks and count twice in sigma."
            )
        self._row_ids = unique
        self._marks = torch.full_like(unique, UNUSED)
        self._policy = policy

    @property
    def size(self) -> int:
        """`N` — the unlabelled population eq. (11) normalises against."""
        return int(self._row_ids.numel())

    @property
    def marks(self) -> Tensor:
        """A copy of the marks, in ascending `row_id` order. For tests."""
        return self._marks.clone()

    def unused(self) -> int:
        """`N - sum_c sigma(c)`: rows no step has accepted (Alg. 1 line 6)."""
        return int((self._marks == UNUSED).sum())

    def learning_effect(self, num_treatments: int) -> Tensor:
        """`sigma(c)` — Algorithm 1 line 5, a count over the stored marks.

        Eq. (5) is stated as a fresh evaluation of every unlabelled row at time
        `t`, which no implementation can afford per step; §3.1 replaces it with
        exactly this count, and algorithm 1 line 5 is the replacement.
        """
        marked = self._marks[self._marks != UNUSED]
        return torch.bincount(marked, minlength=num_treatments).to(torch.long)

    def thresholds(self, num_treatments: int) -> Tensor:
        """`T(c)` for every class — eq. (7)/(12) over eq. (6)/(11).

        Zero for every class while nothing is marked, which is what algorithm 1
        computes rather than a boundary case of it: `sigma = 0` makes `beta = 0`
        and `M(0) = 0`, so the first steps of a run gate on nothing at all
        (`flexmatch.md` §2, first limitation).
        """
        sigma = self.learning_effect(num_treatments).to(torch.float64)
        denominator = float(sigma.max())
        if self._policy.warm_up:
            denominator = max(denominator, float(self.unused()))
        if denominator <= 0.0:
            # Reachable only with the warm-up off and nothing marked, where
            # eq. (6) is 0/0. Its limit under eq. (11) is zero, and that is the
            # value algorithm 1 would carry into line 11 through the branch it
            # takes at line 6.
            return torch.zeros(num_treatments, dtype=torch.float64)
        beta = sigma / denominator
        return self._policy.map(beta) * self._policy.tau

    def mark(self, row_id: Tensor, labels: Tensor, confidence: Tensor) -> None:
        """Algorithm 1 lines 13-17: write `arg max q_b` where `max q_b > tau`.

        At the **fixed** `tau`, never at the per-class threshold — see the
        module note. `row_id`, `labels` and `confidence` are the rows this
        objective was entitled to in this step, already gathered.
        """
        accepted = confidence > self._policy.tau
        if not bool(accepted.any()):
            return
        ids = row_id[accepted]
        position = torch.searchsorted(self._row_ids, ids)
        if int(position.max()) >= self.size or not bool(
            torch.equal(self._row_ids[position], ids)
        ):
            raise LossError(
                "a batch row is not in the population this CurriculumStatus "
                "was built over. The marks are keyed by `row_id`, so a stage "
                "drawing rows the population does not contain has no place to "
                "record them."
            )
        self._marks[position] = labels[accepted]


@dataclass(frozen=True)
class CurriculumPseudoLabelTreatmentNLL:
    """`1(max q > T(arg max q)) * -log p(t = arg max q | x)` — eq. (8).

    FixMatch's eq. (4) with the constant gate replaced by the per-class
    curriculum of eqs. (5)-(7), (11) and (12), and with the mark update of
    algorithm 1 lines 13-17 alongside it.

    Attributes:
        port: The treatment-distribution port both sides read.
        target: The realisation the artificial label comes from — the weak
            view, `q_b = p_m(y | omega(u_b))`.
        prediction: The realisation the label is charged against — the strong
            view, `p_m(y | Omega(u_b))`.
        threshold: The gate rule. Binds `losses.confidence_threshold`, so it
            has no default (`DESIGN.md` §9.1).
        sharpening: How the label is formed. Binds `losses.sharpening`.
        stop_grad: Which side is detached. Binds `gradients.detached_targets`.
        rows: The population the term is entitled to, and the population `N`
            is counted over. FixMatch's footnote 2 — inherited by FlexMatch
            along with the rest of its framework — puts every labelled row into
            `U` as well, so this recipe's value is `all`.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    target: Realisation
    prediction: Realisation
    threshold: CurriculumThreshold = REQUIRED
    sharpening: Literal["hard"] = REQUIRED
    stop_grad: Literal["target"] = REQUIRED
    rows: Rows = "all"
    name: str = "curriculum_pseudo_label_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "threshold": "losses.confidence_threshold",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        if not require_str("curriculum objective name", self.name, error=LossError):
            raise LossError("CurriculumPseudoLabelTreatmentNLL.name must be non-empty")
        if not isinstance(self.port, Port):
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL.port must be a Port, got "
                f"{type(self.port)}"
            )
        if port_spec(self.port).kind != "treatment_distribution":
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL labels a treatment, but "
                f"port {self.port!s} carries {port_spec(self.port).kind}. "
                "`sigma(c)` is a count per treatment level (eq. 5), so the "
                "curriculum has nothing to be per-class over otherwise."
            )
        target: object = self.target
        prediction: object = self.prediction
        if not isinstance(target, Realisation) or not isinstance(
            prediction, Realisation
        ):
            raise LossError(
                "CurriculumPseudoLabelTreatmentNLL.target and prediction must "
                "be Realisations"
            )
        if self.target == self.prediction:
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL labels {self.target} with "
                "itself. Eq. (8) is a weak-view label charged against a "
                "strong-view prediction; one realisation on both sides is "
                "entropy minimisation, which is a different objective."
            )
        if not isinstance(self.threshold, CurriculumThreshold):
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL.threshold is the gate rule "
                f"of eqs. (7), (11) and (12), a CurriculumThreshold, got "
                f"{type(self.threshold)}. A bare float is FixMatch's gate and "
                "`PseudoLabelTreatmentNLL` is the objective for it."
            )
        if self.sharpening != "hard":
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL.sharpening must be 'hard', "
                f"got {self.sharpening!r}. Eq. (8) charges `H(\\hat q_b, .)` "
                "against the arg max; CPL adjusts the threshold and never the "
                "target."
            )
        if self.stop_grad != "target":
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL.stop_grad must be "
                f"'target', got {self.stop_grad!r}. The label is an arg max "
                "and both gates are step functions, so no gradient exists on "
                "that side to keep."
            )
        try:
            validate_population(self.rows)
        except Xty2Error as error:
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL {self.name!r}: {error}"
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
            "gate = max prob > T(label), the per-class threshold (eq. 8)",
            *self.threshold.describe(),
            "sigma(c) = rows ever marked class c (Alg. 1 line 5)",
            "marks are set at the fixed tau, not at T(c) (Alg. 1 line 14)",
            "marks are per-stage state keyed by row_id, and are never cleared",
            "denominator = every eligible row; rejected rows contribute 0",
        )

    @property
    def batch_coupled(self) -> bool:
        """No — and the distinction is the point of algorithm 1's ordering.

        `T(c)` is read from marks laid down by *previous* steps and the current
        batch's marks are written afterwards, so splitting a batch in two
        changes which rows are marked when, but never one row's gate given the
        state it meets. A threshold computed from *this* batch's confidences
        answers true, and FreeMatch's does: its algorithm 1 folds the batch into
        `tau_t` at line 3 and gates that same batch at line 9
        (`SelfAdaptiveThresholdTreatmentNLL`, `docs/recipes/freematch.md` §3.1).
        """
        return False

    def initial_state(self, population: TrainingPopulation | None) -> object:
        """Algorithm 1 line 2, over the rows this term is entitled to.

        `N` is that population's size, which is what eq. (11) normalises
        against while nothing is marked, so it is read from the run's own rows
        rather than declared beside them: a recipe that asserted `N` could
        assert one the sampler never draws from.
        """
        if population is None:
            raise LossError(
                f"objective {self.name!r} needs the stage's training "
                "population: eq. (11) normalises by `N`, the size of the "
                "unlabelled set, and algorithm 1 line 2 initialises one mark "
                "per row of it. A stage fed by ExternalBatches has neither, so "
                "the curriculum cannot be computed for it."
            )
        rows = population.rows
        eligible = row_mask(self.rows, rows)
        return CurriculumStatus(rows.row_id[eligible], self.threshold)

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        status = ctx.objective_state(self.name, CurriculumStatus)
        classes = ctx.schema.treatment_cardinality
        target = treatment_distribution(
            state, self.port, self.target, objective=self.name
        ).probs.detach()
        prediction = treatment_distribution(
            state, self.port, self.prediction, objective=self.name
        )
        confidence, labels = target.max(dim=-1)
        # Read before the marks below are written, which is algorithm 1's
        # order (lines 4-12, then 13-17, then the loss at line 18 with the
        # thresholds it already had).
        thresholds = status.thresholds(classes).to(confidence.dtype)
        accepted = confidence > thresholds.index_select(0, labels)
        per_row = -prediction.log_prob(labels) * accepted.to(target.dtype)
        if per_row.shape[0] != batch.batch_size:
            raise LossError(
                f"CurriculumPseudoLabelTreatmentNLL {self.name!r} got "
                f"{per_row.shape[0]} rows from its realisations for a batch of "
                f"{batch.batch_size}"
            )
        term = reduce_rows(
            per_row,
            rows,
            diagnostics=_gate(confidence, accepted, thresholds, status, rows),
        )
        if rows.numel():
            status.mark(
                batch.row_id.index_select(0, rows),
                labels.index_select(0, rows),
                confidence.index_select(0, rows),
            )
        return term


def _gate(
    confidence: Tensor,
    accepted: Tensor,
    thresholds: Tensor,
    status: CurriculumStatus,
    rows: RowIndex,
) -> dict[str, float]:
    """The mask rate, plus the two numbers that say whether CPL is alive.

    `coverage` is FixMatch's eq. (6) over the eligible rows, so it is
    comparable to `PseudoLabelTreatmentNLL`'s line in a paired log.
    `threshold_min` / `threshold_max` are the curriculum itself — a pair stuck
    at `tau` is FixMatch wearing this objective's name, and a pair stuck at 0
    is unfiltered self-training — and `marked_fraction` is what moves them.
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
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
        "marked_fraction": 1.0 - status.unused() / status.size,
    }


__all__ = [
    "UNUSED",
    "CurriculumMapping",
    "CurriculumPseudoLabelTreatmentNLL",
    "CurriculumStatus",
    "CurriculumThreshold",
]
