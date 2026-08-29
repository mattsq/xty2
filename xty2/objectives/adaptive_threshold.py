"""FreeMatch's self-adaptive threshold and fairness terms (`freematch.md`).

FixMatch keeps an artificial label when its probability clears one fixed `tau`;
FlexMatch when it clears a per-class threshold earned by counting marks across
steps. FreeMatch keeps it when it clears `tau_t(c)`, a threshold assembled from
two exponential moving averages of the model's *own* predictions on unlabelled
data: a global `tau_t`, the mean top-class confidence (eq. 5), modulated
per class by `MaxNorm(p~_t)`, the mean predicted class distribution (eqs. 6, 7).
A second term, self-adaptive fairness, pushes the marginal of the retained rows'
predictions towards the model's own running marginal after both are normalised
by the histogram of predicted labels (eqs. 9-11).

Four properties of that are decisions, and each is visible in the plan or in a
declaration rather than buried in `compute`:

* **The statistics are updated from the current batch, before that batch is
  gated.** Algorithm 1 updates `tau_t`, `p~_t` and `h~_t` at lines 3-5, computes
  `tau_t(c)` at line 7 and only then applies it at line 9. So a row's gate
  depends on the confidences of the *other* rows of its own batch, and both
  objectives here declare `batch_coupled = True` — where
  `CurriculumPseudoLabelTreatmentNLL`, whose thresholds are read before its
  marks are written, declares `False`.
* **One state, two objectives.** Eq. (12) gives `L_u` and `L_f` separate
  weights, so they cannot be one term without dropping `w_f` from the plan and
  from the per-objective log. `tau_t`, `p~_t` and `h~_t` are one set of
  statistics that eqs. (7), (8), (9) and (11) all read, so
  `SelfAdaptiveThresholdTreatmentNLL` owns them and `SelfAdaptiveFairness`
  names it (`DESIGN.md` §4, a sibling state read).
* **The shared update is idempotent within a step.** Whichever objective the
  mixer reaches first folds the batch in; the second finds the step already
  recorded and reads the same numbers. Without that, the loss would depend on
  the order two lines appear in a recipe, which is the kind of hidden logic
  `CLAUDE.md` rule 3 exists to forbid.
* **Eq. (11) is implemented without its leading minus.** `freematch.md`
  deviation 7 is the argument; the short version is that `-H(A, B)` minimised
  drives `B` to a corner of the simplex, which is the opposite of the "diverse
  predictions" the paper states the term is for in four places. The literal
  reading needs no code: a recipe declaring `weight=-w_f` on this objective
  computes eq. (11) exactly as printed.

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
from xty2.core.rows import RowIndex, Rows, validate_population

LOG_FLOOR = 1e-12
"""Guard inside `log` for a class whose retained probability mass underflows.

Not the empty-bin convention — that is exclusion, and it happens before this is
reached (`SelfAdaptiveFairness`, `freematch.md` §7). This is only the floor that
keeps a surviving class whose mass is denormal from producing `-inf`.
"""


@dataclass(frozen=True, repr=False)
class SelfAdaptiveThreshold:
    """Eqs. (5), (6), (7) and (10) as one declared value.

    One object rather than a bare float on the objective, because
    `losses.confidence_threshold` names *the rule by which confidence gates a
    row* and FreeMatch's rule contains no threshold at all: `tau_t(c)` is a
    function of the training history and `lambda` is the only number a recipe
    sets. `card_keys.py` asks for one field holding the whole rule where a paper
    states several things together, which `flexmatch.md` §4 read the same way
    for a different rule.

    Attributes:
        decay: `lambda in (0, 1)`, the momentum of the three exponential moving
            averages — eq. (5)'s global confidence, eq. (6)'s per-class
            probability and eq. (10)'s label histogram. The paper states one
            `lambda` for all three (table 5).
    """

    decay: float

    def __post_init__(self) -> None:
        if isinstance(self.decay, bool) or not isinstance(self.decay, int | float):
            raise LossError(
                f"SelfAdaptiveThreshold.decay must be a number in (0, 1), got "
                f"{type(self.decay)}"
            )
        if not 0.0 < float(self.decay) < 1.0:
            raise LossError(
                f"SelfAdaptiveThreshold.decay is eq. (5)'s `lambda in (0, 1)`, "
                f"got {self.decay!r}. At 0 the threshold is the current batch's "
                "mean confidence and at 1 it never leaves 1/K, so neither "
                "endpoint is the mechanism."
            )
        object.__setattr__(self, "decay", float(self.decay))

    def __repr__(self) -> str:
        """The form card §4 writes, so the plan and the card can be diffed."""
        return f"self_adaptive(decay={self.decay:g})"

    def describe(self) -> tuple[str, ...]:
        """The gate as stable `plan_details` lines (`DESIGN.md` §4)."""
        return (
            f"tau_t = {self.decay:g} tau_(t-1) + "
            f"{1.0 - self.decay:g} mean_b max(q_b) (eq. 5)",
            f"p~_t = {self.decay:g} p~_(t-1) + {1.0 - self.decay:g} mean_b q_b (eq. 6)",
            "T(c) = MaxNorm(p~_t)(c) * tau_t (eq. 7)",
            "tau_0 = p~_0(c) = h~_0(c) = 1/K, and step 0 folds in no batch",
        )


class SelfAdaptiveThresholds:
    """`tau_t`, `p~_t` and `h~_t`: the three EMAs of eqs. (5), (6) and (10).

    The state a `SelfAdaptiveThresholdTreatmentNLL` carries across the steps of
    one stage, and that a `SelfAdaptiveFairness` in the same stage reads. Built
    by the executor once per stage *execution* — never held on an objective, so
    a recipe stays an immutable declaration and two runs of one compiled recipe
    are identical (`core/loss.py`, `StatefulObjective`).

    It needs no `TrainingPopulation`: every statistic is an average over the
    batch, so there is no `N` to count and no row identity to key. That is the
    property `flexmatch.md` §5.1 named this card in advance to check, and it is
    why `initial_state` takes `TrainingPopulation | None`.

    All three are held in float64. They are running sums over thousands of steps
    at `lambda = 0.999`, where a float32 EMA loses the tail of its own history
    to rounding; `thresholds()` casts at the point of use.
    """

    __slots__ = (
        "_classes",
        "_histogram",
        "_marginal",
        "_observed_rows",
        "_policy",
        "_step",
        "_tau",
    )

    def __init__(self, num_treatments: int, policy: SelfAdaptiveThreshold) -> None:
        if num_treatments < 2:
            raise LossError(
                f"SelfAdaptiveThresholds needs K >= 2, got {num_treatments}. "
                "Eq. (7) normalises by `max_c p~(c)`, and one class makes the "
                "gate the constant 1."
            )
        self._classes = int(num_treatments)
        self._policy = policy
        uniform = 1.0 / self._classes
        # Eqs. (5) and (6) state the `t = 0` case; eq. (10) does not, and
        # `freematch.md` §7 chooses the same uniform value for it.
        self._tau = torch.tensor(uniform, dtype=torch.float64)
        self._marginal = torch.full((self._classes,), uniform, dtype=torch.float64)
        self._histogram = torch.full((self._classes,), uniform, dtype=torch.float64)
        self._step: int | None = None
        self._observed_rows = 0

    @property
    def classes(self) -> int:
        """`C` — the number of treatment levels the three statistics span."""
        return self._classes

    @property
    def tau(self) -> float:
        """`tau_t`, eq. (5)."""
        return float(self._tau)

    @property
    def marginal(self) -> Tensor:
        """A copy of `p~_t`, eq. (6). `[C]` float64."""
        return self._marginal.clone()

    @property
    def histogram(self) -> Tensor:
        """A copy of `h~_t`, eq. (10). `[C]` float64."""
        return self._histogram.clone()

    @property
    def last_observed_step(self) -> int | None:
        """The step this state last folded a batch in at, or `None`."""
        return self._step

    def thresholds(self) -> Tensor:
        """`tau_t(c)` for every class — eq. (7). `[C]` float64.

        `MaxNorm` puts the most-predicted class at `tau_t` exactly and every
        other strictly below it, so unlike FlexMatch's `beta` this does not
        collapse to a single number when `K = 2` (`freematch.md` §2).
        """
        return self._marginal / self._marginal.max() * self._tau

    def observe(self, step: int, probs: Tensor) -> None:
        """Algorithm 1 lines 3-5, folding one batch into the three EMAs.

        Idempotent within a step: the second objective of the pair to call this
        in one iteration finds the step already recorded and returns, so the two
        read identical statistics whichever order the mixer computes them in.

        That is only *safe* if the two objectives are entitled to the same rows,
        because otherwise whichever the mixer reaches first decides which set the
        EMAs averaged — a difference no reader of either declaration could see.
        A repeat call at one step whose row count differs is therefore refused
        rather than ignored. Equal counts over different sets would still slip
        through; the recipe declares one population for both terms and Tier 0
        asserts it, and this is the cheap half of that guard rather than a
        replacement for it.

        Args:
            step: `ctx.global_step`. Step 0 records itself and folds nothing in,
                which is what eqs. (5) and (6) mean by their `t = 0` case.
            probs: `[n, C]` weak-view probabilities over the rows the calling
                objective is entitled to, already detached.
        """
        if self._step is not None and step <= self._step:
            if step == self._step and probs.shape[0] != self._observed_rows:
                raise LossError(
                    f"two objectives folded different row counts into one "
                    f"SelfAdaptiveThresholds at step {step}: "
                    f"{self._observed_rows} then {probs.shape[0]}. Eqs. (5), "
                    "(6) and (10) are averages over one unlabelled batch, so "
                    "the terms sharing this state have to declare one row "
                    "population (docs/recipes/freematch.md §3.2)."
                )
            return
        if probs.ndim != 2 or probs.shape[1] != self._classes:
            raise LossError(
                f"SelfAdaptiveThresholds.observe takes [n, {self._classes}] "
                f"probabilities, got shape {tuple(probs.shape)}"
            )
        if probs.shape[0] == 0:
            # Nothing to average. Deliberately not recorded as observed: a
            # later objective in the same step with rows of its own should
            # still be able to lay the update down.
            return
        self._step = step
        self._observed_rows = int(probs.shape[0])
        if step == 0:
            return
        batch = probs.detach().to(torch.float64)
        decay = self._policy.decay
        self._tau = decay * self._tau + (1.0 - decay) * batch.max(dim=-1).values.mean()
        self._marginal = decay * self._marginal + (1.0 - decay) * batch.mean(dim=0)
        counts = torch.bincount(batch.argmax(dim=-1), minlength=self._classes)
        share = counts.to(torch.float64) / float(batch.shape[0])
        self._histogram = decay * self._histogram + (1.0 - decay) * share


@dataclass(frozen=True)
class SelfAdaptiveThresholdTreatmentNLL:
    """`1(max q > tau_t(arg max q)) * -log p(t = arg max q | x)` — eq. (8).

    FixMatch's eq. (4) with the constant gate replaced by the self-adaptive
    threshold of eqs. (5)-(7), and with the statistics update of algorithm 1
    lines 3-5 alongside it.

    Attributes:
        port: The treatment-distribution port both sides read.
        target: The realisation the artificial label comes from — the weak
            view, `q_b = p_m(y | omega(u_b))`. It is also the realisation the
            three EMAs are estimated from.
        prediction: The realisation the label is charged against — the strong
            view, `Q_b = p_m(y | Omega(u_b))`.
        threshold: The gate rule. Binds `losses.confidence_threshold`, so it
            has no default (`DESIGN.md` §9.1).
        sharpening: How the label is formed. Binds `losses.sharpening`.
        stop_grad: Which side is detached. Binds `gradients.detached_targets`.
        num_treatments: `C`. Not paper-governed and not a card key — it is a
            property of the schema, and a component takes it the same way
            (`CategoricalPropensity`). It is a field rather than something read
            from the batch because `initial_state` runs before any batch
            exists, and `compute` checks it against `ctx.schema` for the reason
            `DESIGN.md` §3.1 gives: a term that took `K` from the head's own
            output would agree with a head that had the wrong `K`.
        rows: The population the term is entitled to, and the population the
            EMAs are averaged over. FixMatch's footnote 2 — inherited by
            FreeMatch along with the rest of its framework (`freematch.md`
            §3.2) — puts every labelled row into `U` as well, so this recipe's
            value is `all`.
        name: Keys the per-objective log (§6.2), and is what a
            `SelfAdaptiveFairness` in the same stage names to read the state.
    """

    port: Port
    target: Realisation
    prediction: Realisation
    num_treatments: int
    threshold: SelfAdaptiveThreshold = REQUIRED
    sharpening: Literal["hard"] = REQUIRED
    stop_grad: Literal["target"] = REQUIRED
    rows: Rows = "all"
    name: str = "self_adaptive_threshold_treatment_nll"

    CARD_KEYS: ClassVar[dict[str, str]] = {
        "threshold": "losses.confidence_threshold",
        "sharpening": "losses.sharpening",
        "stop_grad": "gradients.detached_targets",
    }

    def __post_init__(self) -> None:
        card_hyperparameters(self)
        _validate_pair(
            "SelfAdaptiveThresholdTreatmentNLL",
            name=self.name,
            port=self.port,
            target=self.target,
            prediction=self.prediction,
            rows=self.rows,
        )
        if not isinstance(self.threshold, SelfAdaptiveThreshold):
            raise LossError(
                f"SelfAdaptiveThresholdTreatmentNLL.threshold is the gate rule "
                f"of eqs. (5)-(7), a SelfAdaptiveThreshold, got "
                f"{type(self.threshold)}. A bare float is FixMatch's gate and "
                "`PseudoLabelTreatmentNLL` is the objective for it."
            )
        if self.sharpening != "hard":
            raise LossError(
                f"SelfAdaptiveThresholdTreatmentNLL.sharpening must be 'hard', "
                f"got {self.sharpening!r}. Eq. (8) charges `H(\\hat q_b, .)` "
                "against the arg max; SAT adjusts the threshold and never the "
                "target."
            )
        if self.stop_grad != "target":
            raise LossError(
                f"SelfAdaptiveThresholdTreatmentNLL.stop_grad must be 'target', "
                f"got {self.stop_grad!r}. The label is an arg max, the gate a "
                "step function and the thresholds a stored EMA, so no gradient "
                "exists on that side to keep."
            )
        if isinstance(self.num_treatments, bool) or not isinstance(
            self.num_treatments, int
        ):
            raise LossError(
                f"SelfAdaptiveThresholdTreatmentNLL.num_treatments must be an "
                f"int, got {type(self.num_treatments)}"
            )
        if self.num_treatments < 2:
            raise LossError(
                f"SelfAdaptiveThresholdTreatmentNLL.num_treatments is `C` of "
                f"eqs. (5)-(7), and must be at least 2, got "
                f"{self.num_treatments}"
            )

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
            "the three EMAs fold in this batch before this batch is gated",
            f"p~_t and h~_t are [{self.num_treatments}], checked against the schema",
            "denominator = every eligible row; rejected rows contribute 0",
        )

    @property
    def batch_coupled(self) -> bool:
        """Yes — and this is where FreeMatch parts from FlexMatch.

        Algorithm 1 folds this batch's confidences into `tau_t` and `p~_t`
        (lines 3-4) *before* computing `tau_t(c)` (line 7) and applying it
        (line 9), so splitting a batch in two changes the gate each half meets.
        `CurriculumPseudoLabelTreatmentNLL` reads its thresholds from marks laid
        down by earlier steps and answers false to the same question.
        """
        return True

    def initial_state(self, population: TrainingPopulation | None) -> object:
        """Eqs. (5), (6) and (10) at `t = 0`, over `K` alone.

        `population` is accepted and ignored. Every statistic is an average over
        a batch, so unlike FlexMatch's marks there is no `N` to count and no row
        identity to key — which is the property `flexmatch.md` §5.1 named this
        objective in advance to check the `TrainingPopulation | None` signature
        against.
        """
        del population
        return SelfAdaptiveThresholds(self.num_treatments, self.threshold)

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        thresholds_state = ctx.objective_state(self.name, SelfAdaptiveThresholds)
        target = treatment_distribution(
            state, self.port, self.target, objective=self.name
        ).probs.detach()
        prediction = treatment_distribution(
            state, self.port, self.prediction, objective=self.name
        )
        _check_classes(target, ctx, self.name, self.num_treatments)
        # Algorithm 1 lines 3-5, before the gate below: idempotent, so the
        # fairness term in the same stage may have laid it down already.
        thresholds_state.observe(ctx.global_step, target.index_select(0, rows))
        confidence, labels = target.max(dim=-1)
        thresholds = thresholds_state.thresholds().to(confidence.dtype)
        accepted = confidence > thresholds.index_select(0, labels)
        per_row = -prediction.log_prob(labels) * accepted.to(target.dtype)
        if per_row.shape[0] != batch.batch_size:
            raise LossError(
                f"SelfAdaptiveThresholdTreatmentNLL {self.name!r} got "
                f"{per_row.shape[0]} rows from its realisations for a batch of "
                f"{batch.batch_size}"
            )
        return reduce_rows(
            per_row,
            rows,
            diagnostics=_gate(confidence, accepted, thresholds, thresholds_state, rows),
        )


@dataclass(frozen=True)
class SelfAdaptiveFairness:
    """`H(SumNorm(p~/h~), SumNorm(p_bar/h_bar))` — eq. (11), sign per deviation 7.

    The batch quantities are eq. (9): the mean strong-view probability and the
    histogram of strong-view hard labels, both over the rows eq. (8) retained.
    The reference distribution is the pair of EMAs the threshold objective
    already maintains, which is why this term names that objective rather than
    carrying a second `lambda`.

    **The sign is `freematch.md` deviation 7.** Eq. (11) prints a leading minus,
    which minimised drives the batch marginal to a corner of the simplex — the
    opposite of the diverse predictions §1, §2, §4.2 and §6 of the paper all say
    the term is for. This computes the cross-entropy itself, whose minimum is at
    `B = A`. A recipe wanting the literal reading declares `weight=-w_f`.

    Attributes:
        port: The treatment-distribution port both sides read.
        target: The weak-view realisation. Read only through eq. (9)'s
            indicator and through the EMAs, so it carries no gradient.
        prediction: The strong-view realisation `p_bar` and `h_bar` are taken
            over.
        statistics: The `name` of the `SelfAdaptiveThresholdTreatmentNLL` in the
            same stage whose state this reads (`DESIGN.md` §4, a sibling state
            read). It has no default and no `REQUIRED` sentinel: the sentinel is
            for a *paper-governed* field (`DESIGN.md` §9.1) and this is not one
            — it is a wiring reference, and a field with no default at all is
            the stronger guard, since a recipe cannot construct the objective
            without it.
        rows: The population the term is entitled to. `all`, for eq. (8)'s
            reason.
        name: Keys the per-objective log (§6.2).
    """

    port: Port
    target: Realisation
    prediction: Realisation
    statistics: str
    rows: Rows = "all"
    name: str = "self_adaptive_fairness"

    def __post_init__(self) -> None:
        _validate_pair(
            "SelfAdaptiveFairness",
            name=self.name,
            port=self.port,
            target=self.target,
            prediction=self.prediction,
            rows=self.rows,
        )
        if not require_str(
            "SelfAdaptiveFairness.statistics", self.statistics, error=LossError
        ):
            raise LossError(
                "SelfAdaptiveFairness.statistics names the "
                "SelfAdaptiveThresholdTreatmentNLL in the same stage whose "
                "`tau_t`, `p~_t` and `h~_t` eq. (11) reads; it must be that "
                "objective's `name`."
            )

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.port, self.target), (self.port, self.prediction)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        """The weak view. Eq. (9)'s indicator and the EMAs are both constants."""
        return frozenset({(self.port, self.target)})

    def plan_details(self) -> tuple[str, ...]:
        """Arithmetic no port, row population or card key reveals."""
        return (
            f"reads tau_t, p~_t and h~_t from objective {self.statistics!r}",
            "p_bar = mean of the strong-view probabilities over retained rows",
            "h_bar = histogram of the strong-view arg max over retained rows",
            "A = SumNorm(p~_t / h~_t), B = SumNorm(p_bar / h_bar) (eq. 11)",
            "classes with an empty h_bar bin leave both SumNorms (card §7)",
            "loss = H(A, B), eq. (11) without its minus (card deviation 7)",
            "fewer than two surviving classes contributes 0",
        )

    @property
    def batch_coupled(self) -> bool:
        """Yes: `p_bar` and `h_bar` are aggregates over the whole batch."""
        return True

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        thresholds_state = _sibling_state(ctx, self.statistics, self.name)
        target = treatment_distribution(
            state, self.port, self.target, objective=self.name
        ).probs.detach()
        prediction = treatment_distribution(
            state, self.port, self.prediction, objective=self.name
        ).probs
        _check_classes(target, ctx, self.name)
        if prediction.shape[0] != batch.batch_size:
            raise LossError(
                f"SelfAdaptiveFairness {self.name!r} got {prediction.shape[0]} "
                f"rows from its realisations for a batch of {batch.batch_size}"
            )
        # Algorithm 1 lines 3-5. Idempotent, so it does not matter whether the
        # threshold objective in this stage has already laid this step down.
        thresholds_state.observe(ctx.global_step, target.index_select(0, rows))
        if rows.numel() == 0:
            return LossTerm.empty(like=prediction)
        confidence, labels = target.max(dim=-1)
        thresholds = thresholds_state.thresholds().to(confidence.dtype)
        # Eq. (9) writes `>=` and eq. (8) writes `>`; card §7 takes `>` in both
        # so the two terms are never given different rows in one step.
        retained = (confidence > thresholds.index_select(0, labels)).index_select(
            0, rows
        )
        strong = prediction.index_select(0, rows)
        return _fairness(strong, retained, thresholds_state, rows)


def _fairness(
    strong: Tensor,
    retained: Tensor,
    state: SelfAdaptiveThresholds,
    rows: RowIndex,
) -> LossTerm:
    """Eqs. (9) and (11) over the eligible rows, at deviation 7's sign.

    The `1/mu B` of eq. (9) is absent because eq. (11) takes the ratio of the
    two quantities it scales, so it cancels exactly. What does not cancel is a
    class no retained row predicts: `h_bar(c) = 0` makes the ratio undefined,
    and card §7 excludes such a class from both `SumNorm`s rather than reading
    the ratio as zero or as infinite.
    """
    classes = state.classes
    kept = retained.to(strong.dtype)
    probability = (strong * kept[:, None]).sum(dim=0)
    counts = torch.bincount(strong.argmax(dim=-1)[retained], minlength=classes)
    support = counts > 0
    surviving = int(support.sum())
    coverage = float(retained.to(torch.float64).mean())
    if surviving < 2:
        # One surviving class makes `SumNorm` the constant 1 and `H(A, B)`
        # identically zero; below two there is no distribution to be fair over,
        # and the support is logged so an inert step is visible as one.
        return LossTerm(
            value=torch.zeros((), dtype=strong.dtype, device=strong.device),
            n=int(rows.numel()),
            diagnostics={
                "coverage": coverage,
                "fairness_support": float(surviving),
                "marginal_entropy": 0.0,
            },
        )
    share = counts.to(strong.dtype)
    reference = (state.marginal / state.histogram).to(strong.dtype)
    a = _sum_norm(reference[support])
    b = _sum_norm(probability[support] / share[support])
    marginal = _sum_norm(probability[support])
    return LossTerm(
        value=-(a * torch.log(b.clamp_min(LOG_FLOOR))).sum(),
        n=int(rows.numel()),
        diagnostics={
            "coverage": coverage,
            "fairness_support": float(surviving),
            "marginal_entropy": float(
                -(marginal * torch.log(marginal.clamp_min(LOG_FLOOR))).sum().detach()
            ),
        },
    )


def _sum_norm(values: Tensor) -> Tensor:
    """`SumNorm` of eq. (11). The caller has already dropped the empty bins."""
    return values / values.sum()


def _sibling_state(
    ctx: TrainContext, owner: str, reader: str
) -> SelfAdaptiveThresholds:
    """The named sibling's state, with an error naming both objectives.

    `TrainContext.objective_state` already takes the name and checks the type;
    what it cannot know is that the *reader* is a different objective, so its
    message would report the owner asking for state it never declared.
    """
    try:
        return ctx.objective_state(owner, SelfAdaptiveThresholds)
    except Xty2Error as error:
        raise LossError(
            f"objective {reader!r} reads the self-adaptive threshold state of "
            f"{owner!r}, and the stage {ctx.stage or '<unnamed>'!r} does not "
            f"hold it: {error} A SelfAdaptiveFairness names the "
            "SelfAdaptiveThresholdTreatmentNLL it shares eqs. (5), (6) and "
            "(10) with, and the two belong to one stage (DESIGN.md §4)."
        ) from error


def _check_classes(
    probs: Tensor, ctx: TrainContext, objective: str, declared: int | None = None
) -> None:
    """The realisation's `K` against the schema's, at the read.

    `DESIGN.md` §3.1's argument: a term that took `K` from the head's own output
    would agree with a head that had the wrong `K`. `declared` is the
    threshold objective's `num_treatments` — the width its state was built at —
    so a recipe that sized the state from one schema and compiled against
    another fails here rather than broadcasting.
    """
    expected = ctx.schema.treatment_cardinality
    if probs.shape[-1] != expected:
        raise LossError(
            f"objective {objective!r} read a treatment distribution over "
            f"{probs.shape[-1]} classes where the schema declares {expected}"
        )
    if declared is not None and declared != expected:
        raise LossError(
            f"objective {objective!r} declares num_treatments = {declared} and "
            f"the schema declares {expected}. The self-adaptive state is `[C]` "
            "wide and is built before the first batch, so the two have to agree."
        )


def _validate_pair(
    label: str,
    *,
    name: str,
    port: Port,
    target: Realisation,
    prediction: Realisation,
    rows: Rows,
) -> None:
    """The port, the two realisations and the row population, once for both.

    Both objectives read one treatment-distribution port under a weak and a
    strong realisation over one row population, so the four checks are written
    once rather than twice with the class name substituted.
    """
    if not require_str(f"{label} name", name, error=LossError):
        raise LossError(f"{label}.name must be non-empty")
    if not isinstance(port, Port):
        raise LossError(f"{label}.port must be a Port, got {type(port)}")
    if port_spec(port).kind != "treatment_distribution":
        raise LossError(
            f"{label} reads a treatment distribution, but port {port!s} carries "
            f"{port_spec(port).kind}. Eqs. (6), (7) and (10) are per treatment "
            "level, so the mechanism has nothing to be per-class over otherwise."
        )
    candidates: tuple[object, object] = (target, prediction)
    if not all(isinstance(item, Realisation) for item in candidates):
        raise LossError(f"{label}.target and prediction must be Realisations")
    if target == prediction:
        raise LossError(
            f"{label} reads {target} on both sides. FreeMatch's terms are a "
            "weak-view label and a weak-view statistic charged against a "
            "strong-view prediction; one realisation on both sides is entropy "
            "minimisation, which is a different objective."
        )
    try:
        validate_population(rows)
    except Xty2Error as error:
        raise LossError(f"{label} {name!r}: {error}") from error


def _gate(
    confidence: Tensor,
    accepted: Tensor,
    thresholds: Tensor,
    state: SelfAdaptiveThresholds,
    rows: RowIndex,
) -> dict[str, float]:
    """The mask rate, plus the three numbers that say whether SAT is alive.

    `coverage` is FixMatch's eq. (6) over the eligible rows, so it is comparable
    to `PseudoLabelTreatmentNLL`'s line in a paired log. `tau_global` is eq. (5)
    on its own — a `tau_t` still at `1/K` late in a run is the failure §2's
    first limitation describes — and `threshold_min` / `threshold_max` are what
    `MaxNorm` does to it: a pair that never separates is eq. (7) with its local
    half dead.
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
        "tau_global": state.tau,
        "threshold_min": float(thresholds.min()),
        "threshold_max": float(thresholds.max()),
    }


__all__ = [
    "LOG_FLOOR",
    "SelfAdaptiveFairness",
    "SelfAdaptiveThreshold",
    "SelfAdaptiveThresholdTreatmentNLL",
    "SelfAdaptiveThresholds",
]
