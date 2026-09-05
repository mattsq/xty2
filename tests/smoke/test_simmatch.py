"""Tier 1 — SimMatch's predeclared eight-arm study on the card's DGP.

`docs/recipes/simmatch.md` §6.2 predeclares eight Tier 1 arms and this module
runs all of them, on one seed of the §6.1 fixture at a fifth of the §6 budget.

* **Arm 1** is the §6 pair itself: full SimMatch against the no-propagation
  ablation, holding seeds, initial parameters, batches, optimiser, schedule,
  views, bank and instance loss fixed.
* **Arms 2 and 3** are the paper's own one-directional ablations, `w/o hat p`
  and `w/o hat q`.
* **Arm 4** switches the instance loss off while leaving the propagation
  arithmetic in place.
* **Arm 5** empties the temporal bank of its memory.
* **Arm 6** permutes the memory labels — a wiring control, not a method arm.
* **Arm 7** turns distribution alignment off, on the balanced fixture and on a
  skewed one.
* **Arm 8** is the diagnostic report every arm carries.

What this module does **not** do is read an end-to-end direction out of arms 2
to 7. Card §6.2 predeclares expectations for arms 2, 3 and 4 from the paper's
table 5 and figure 5, and one seed at 600 of 3,000 steps cannot decide them:
§6's ten-replicate contract is where a direction may be read, and it covers the
pair only. So each arm asserts the arithmetic that defines it — `hat p = p^w`
exactly at `alpha = 1`, `hat q = q^w` exactly with unfolding off, a computed
but unweighted instance term at `lambda_in = 0`, a bank that keeps only its
last observation at momentum zero, and an aggregate that collapses to a
constant when the memory labels are rolled — and reports the rest through arm
8's table.

Arm 6 is the exception, and it is the one that pays for this module. Read in
its isolated form — one finished bank, one student, two slot-to-class maps — it
shows that `hat p`'s advantage over `p^w` on this fixture survives destroying
the map entirely, landing on a constant-prior shrinkage baseline. The
improvement §6's third tolerance measures is therefore not, by itself, evidence
that propagation propagated anything. Card §6.2 and §5 record it.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Constant,
    CosineDecay,
    Dataset,
    FeatureSpec,
    GaussianOutcome,
    OutcomeSpec,
    Port,
    Program,
    Recipe,
    Schema,
    ViewSpec,
    XTYBatch,
    compile,
)
from xty2.core.data import TrainingPopulation
from xty2.core.rows import resolve_rows
from xty2.objectives import (
    LabeledMemoryInstanceConsistency,
    LabeledSimilarityMemory,
    SimilarityMatchingSpec,
    SimilarityMatchingTreatmentNLL,
)
from xty2.recipes import simmatch
from xty2.recipes.simmatch import SIMILARITY_MATCHING
from xty2.training import StageResult, run_stage

FEATURES = 6
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048
STEPS = 600
CLUSTER_SIGNAL = 0.45
SEPARATED = 0.02
OBSERVED = 64
BALANCED = 0.5
SKEWED = (0.15 - SEPARATED) / (1.0 - 2.0 * SEPARATED)
"""The cluster prevalence that makes card §6.2 arm 7's `p(t = 1) = 0.15` exact.

The generator draws `t` at `0.02 + 0.96 c`, so a cluster prevalence of `c` gives
`p(t = 1) = 0.02 + 0.96 c`. Solving it for 0.15 rather than setting the cluster
prevalence to 0.15 keeps the arm's declared quantity the treatment prevalence,
which is what distribution alignment divides by.
"""
CLASSES = 2
MEMORY_TERM = "similarity_matching_treatment_nll"
INSTANCE_TERM = "labeled_memory_instance_consistency"
WARMUP = SIMILARITY_MATCHING.warmup_steps
VIEW_KEY = 90_004
"""The view draw the terminal target reading scores, fixed like every seed here."""


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _schema() -> Schema:
    return Schema(
        features=tuple(
            FeatureSpec(f"x{column}", "continuous") for column in range(FEATURES)
        ),
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


def _rows(
    rows: int, *, seed: int, row_offset: int, prevalence: float = BALANCED
) -> XTYBatch:
    """The card's §6.1 mechanism, fully observed; the recipe hides treatments.

    `prevalence` is the cluster probability, `0.5` in §6.1 and arm 7's skewed
    value otherwise. Every other draw is §6.1's, in §6.1's order.
    """
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < prevalence).float()
    sign = 2.0 * cluster - 1.0
    x = epsilon_x.clone()
    x[:, :4] = CLUSTER_SIGNAL * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = SEPARATED + (1.0 - 2.0 * SEPARATED) * cluster
    t = (u_t < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    y = baseline + t * (1.0 + 0.5 * torch.tanh(x[:, 2])) + 0.5 * epsilon_y
    return XTYBatch(
        x=x,
        t=t,
        y=y,
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(row_offset, row_offset + rows),
    )


def _dataset(train: XTYBatch) -> Dataset:
    return Dataset(
        schema=_schema(),
        rows=train,
        assignments={"train": torch.arange(train.batch_size)},
    )


def _with_steps(recipe: Recipe, steps: int) -> Recipe:
    """A shorter budget with the cosine schedule re-based on it (deviation 3)."""
    stage = recipe.program[0]
    schedule = stage.optimiser.lr_schedule
    assert isinstance(schedule, CosineDecay)
    optimiser = replace(stage.optimiser, lr_schedule=replace(schedule, steps=steps))
    return replace(
        recipe,
        program=Program((replace(stage, steps=steps, optimiser=optimiser),)),
    )


def _pair(
    recipe: Recipe,
) -> tuple[SimilarityMatchingTreatmentNLL, LabeledMemoryInstanceConsistency]:
    """The recipe's two SimMatch objectives, or a failure naming the drift."""
    stage = recipe.program[0]
    semantic = stage.objectives[2].objective
    instance = stage.objectives[3].objective
    assert isinstance(semantic, SimilarityMatchingTreatmentNLL)
    assert isinstance(instance, LabeledMemoryInstanceConsistency)
    return semantic, instance


def _with_spec(recipe: Recipe, **changes: object) -> Recipe:
    """Rebind both objectives to one amended `SimilarityMatchingSpec`.

    Both must carry the *same* spec object — the memory refuses a reader whose
    spec differs from its owner's — so every spec-level arm goes through here.
    """
    semantic, instance = _pair(recipe)
    spec = replace(SIMILARITY_MATCHING, **changes)  # type: ignore[arg-type]
    stage = recipe.program[0]
    objectives = (
        *stage.objectives[:2],
        replace(stage.objectives[2], objective=replace(semantic, spec=spec)),
        replace(stage.objectives[3], objective=replace(instance, spec=spec)),
        *stage.objectives[4:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _without_propagation(recipe: Recipe) -> Recipe:
    """The §6 ablation: `alpha = 1` and no unfolding. Nothing else moves."""
    return _with_spec(recipe, alpha=1.0, unfold=False)


def _without_instance_loss(recipe: Recipe) -> Recipe:
    """Card §6.2 arm 4: `lambda_in = 0`, propagation arithmetic unchanged.

    The term stays in the program at weight zero rather than being removed, so
    the arm plans the same forward passes, keeps the same projection head, and
    still writes and reads the bank; what it loses is eq. (5)'s gradient.
    """
    stage = recipe.program[0]
    objectives = (
        *stage.objectives[:3],
        replace(stage.objectives[3], weight=Constant(0.0)),
        *stage.objectives[4:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


class _PermutedLabels(LabeledSimilarityMemory):
    """Card §6.2 arm 6: `Q_l` rolled by one slot, `Q_f` and class counts kept.

    A cyclic roll is a permutation of the slot-to-label map, so every class
    keeps exactly the slots it had a count of, and the features are untouched.
    The card says "after warm-up"; nothing reads a label before the first
    propagated step, so permuting at construction is the same arm and does not
    need a step counter to arrange it.

    `_slots_for` is the one reader that must still see the *true* labels: it
    checks a support row's slot against that row's observed treatment, and a
    permuted bank would otherwise fail that check on the first step instead of
    running the control.
    """

    def __init__(
        self,
        *,
        classes: int,
        spec: SimilarityMatchingSpec,
        slot_ids: Tensor,
        labels: Tensor,
    ) -> None:
        super().__init__(classes=classes, spec=spec, slot_ids=slot_ids, labels=labels)
        self._true_labels = labels.detach().clone()
        permuted = torch.roll(self._true_labels, 1)
        assert not torch.equal(permuted, self._true_labels), (
            "arm 6 needs a non-identity permutation; a rolled label vector that "
            "equals itself means the fixture put every slot in one class"
        )
        self._labels = permuted

    def _slots_for(self, batch: XTYBatch, support_rows: Tensor) -> Tensor:
        permuted = self._labels
        self._labels = self._true_labels
        try:
            return super()._slots_for(batch, support_rows)
        finally:
            self._labels = permuted


class _Unaligned(LabeledSimilarityMemory):
    """Card §6.2 arm 7: distribution alignment off, everything else intact.

    A memory subclass rather than a spec field. Card §4 fixes what
    `SimilarityMatchingSpec` carries and what `plan_details()` prints, and DA is
    not among them; a switch added to the shipped object to run one Tier 1
    diagnostic would be a card amendment made by the test that needed it.
    """

    def _align(self, raw: Tensor) -> Tensor:
        return raw


def _with_memory(
    recipe: Recipe,
    factory: Callable[[LabeledSimilarityMemory], LabeledSimilarityMemory],
) -> Recipe:
    """Wrap the owner objective so the stage builds a diagnostic memory."""

    class _Overridden(SimilarityMatchingTreatmentNLL):
        def initial_state(self, population: TrainingPopulation | None) -> object:
            built = super().initial_state(population)
            assert isinstance(built, LabeledSimilarityMemory)
            return factory(built)

    semantic, _ = _pair(recipe)
    stage = recipe.program[0]
    overridden = _Overridden(
        spec=semantic.spec,
        target=semantic.target,
        weak_embedding=semantic.weak_embedding,
        prediction=semantic.prediction,
        num_treatments=semantic.num_treatments,
        sharpening=semantic.sharpening,
        stop_grad=semantic.stop_grad,
        support_rows=semantic.support_rows,
        rows=semantic.rows,
        name=semantic.name,
    )
    objectives = (
        *stage.objectives[:2],
        replace(stage.objectives[2], objective=overridden),
        *stage.objectives[3:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _permuted_labels(memory: LabeledSimilarityMemory) -> LabeledSimilarityMemory:
    return _PermutedLabels(
        classes=CLASSES,
        spec=memory.spec,
        slot_ids=memory.slot_ids,
        labels=memory.labels,
    )


def _unaligned(memory: LabeledSimilarityMemory) -> LabeledSimilarityMemory:
    return _Unaligned(
        classes=CLASSES,
        spec=memory.spec,
        slot_ids=memory.slot_ids,
        labels=memory.labels,
    )


def _term(result: StageResult, step: int, name: str) -> float:
    record = next(term for term in result.records[step].terms if term.name == name)
    return float(record.value)


def _diagnostic(result: StageResult, step: int, name: str, key: str) -> float:
    record = next(term for term in result.records[step].terms if term.name == name)
    return float(record.diagnostics[key])


def _mean(result: StageResult, name: str, steps: range) -> float:
    return sum(_term(result, step, name) for step in steps) / len(steps)


def _mean_diagnostic(result: StageResult, name: str, key: str, steps: range) -> float:
    return sum(_diagnostic(result, step, name, key) for step in steps) / len(steps)


def _weighted(result: StageResult, step: int, name: str) -> float:
    record = next(term for term in result.records[step].terms if term.name == name)
    return float(record.weighted)


def _view(run: CompiledRun, name: str) -> ViewSpec:
    return next(view for view in run.recipe.views if view.name == name)


@dataclass(frozen=True)
class _Targets:
    """The terminal reading of every target the two arrows produce.

    Scored on the fixture's hidden treatments, which card §6.2 arm 1 asks for
    "whether they improve or not". `aggregate_q_weak_nll` is `None` at
    `alpha = 1`: equation (10) is then `hat p = p^w` and the aggregate it would
    have mixed in is not recoverable from the arm's own arithmetic.
    """

    propagated_nll: float
    aligned_nll: float
    aggregate_hat_q_nll: float
    aggregate_q_weak_nll: float | None
    coverage: float
    accepted: float
    impurity: float
    predicted_marginal: float
    aligned_marginal: float
    true_marginal: float


@dataclass(frozen=True)
class _Metrics:
    """One arm: held-out fit, terminal targets, and the arm-8 diagnostics."""

    name: str
    run: CompiledRun
    result: StageResult
    treatment_nll: float
    ema_treatment_nll: float
    frequency_nll: float
    outcome_nll: float
    targets: _Targets
    slot_norm: float
    class_coverage: float
    gate_rate: float
    slot_agreement: float
    same_row_cosine: float
    cross_row_cosine: float
    early_same_row_cosine: float
    early_cross_row_cosine: float

    @property
    def alignment_margin(self) -> float:
        return self.same_row_cosine - self.cross_row_cosine


def _memory(result: StageResult) -> LabeledSimilarityMemory:
    """The bank the run itself finished with, not one rebuilt beside it."""
    memory = result.objective_states[MEMORY_TERM]
    assert isinstance(memory, LabeledSimilarityMemory)
    return memory


def _terminal_targets(run: CompiledRun, result: StageResult) -> _Targets:
    """Prepare one more step off the finished bank and score what comes out.

    Equations (7)-(10) read `p^w` and `z^w`, so this reads them under the weak
    view rather than off an untransformed batch, on the whole training
    population and with the parameters the run ended on. `aggregate(q^w)` is
    recovered from equation (10) rearranged — `(hat p - alpha p^w) /
    (1 - alpha)` — rather than transcribed a second time.
    """
    population = result.population
    assert population is not None
    schema = run.recipe.schema
    memory = _memory(result)
    batch = population.rows
    rows = resolve_rows(batch, "t_missing")
    hidden = batch.t.index_select(0, rows)
    weak = _view(run, "weak_x").apply(
        batch, schema, rng_key=VIEW_KEY, population=population
    )
    with torch.no_grad():
        values = run.graph.evaluate(weak, schema=schema, only=run.graph.names)
        propensity = values[Port.T_GIVEN_X]
        embedding = values[Port.X_PROJ]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(embedding, Tensor)
        targets = memory.prepare(
            step=STEPS,
            raw_probabilities=propensity.probs,
            weak_embeddings=embedding,
            batch=batch,
            eligible_rows=rows,
            support_rows=resolve_rows(batch, "t_observed"),
        )
        assert targets.propagated and targets.instance is not None
        labels = memory.labels
        aggregate_hat_q = torch.zeros_like(targets.aligned).index_add(
            1, labels, targets.instance
        )
        alpha = memory.spec.alpha
        aggregate_q_weak = (
            None
            if alpha >= 1.0
            else (targets.semantic - alpha * targets.aligned) / (1.0 - alpha)
        )
        accepted = targets.semantic.max(dim=-1).values >= memory.spec.threshold
        wrong = targets.semantic.argmax(dim=-1) != hidden
        return _Targets(
            propagated_nll=_nll(targets.semantic, hidden),
            aligned_nll=_nll(targets.aligned, hidden),
            aggregate_hat_q_nll=_nll(aggregate_hat_q, hidden),
            aggregate_q_weak_nll=(
                None if aggregate_q_weak is None else _nll(aggregate_q_weak, hidden)
            ),
            coverage=targets.coverage,
            accepted=float(accepted.to(torch.float32).mean()),
            impurity=(
                float(wrong[accepted].to(torch.float32).mean())
                if bool(accepted.any())
                else 0.0
            ),
            predicted_marginal=float(
                propensity.probs.index_select(0, rows)[:, 1].mean()
            ),
            aligned_marginal=float(targets.aligned[:, 1].mean()),
            true_marginal=float(hidden.to(torch.float32).mean()),
        )


def _nll(distribution: Tensor, hidden: Tensor) -> float:
    return float(F.nll_loss(distribution.clamp_min(1e-12).log(), hidden))


def _evaluate(
    run: CompiledRun, result: StageResult, test: XTYBatch, *, name: str
) -> _Metrics:
    schema = run.recipe.schema
    population = result.population
    assert population is not None and result.teacher is not None
    scaled = test.replace(
        y=(test.y - population.statistics["y_location"])
        / population.statistics["y_scale"]
    )
    early = range(WARMUP, WARMUP + 100)
    late = range(STEPS - 100, STEPS)
    memory = _memory(result)
    slots = memory.features
    with torch.no_grad():
        student = run.graph.evaluate(scaled, schema=schema, only=run.graph.names)
        teacher = result.teacher.graph.evaluate(
            scaled, schema=schema, only=run.graph.names
        )
        propensity = student[Port.T_GIVEN_X]
        ema_propensity = teacher[Port.T_GIVEN_X]
        outcome = teacher[Port.Y_GIVEN_XT]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(ema_propensity, CategoricalTreatment)
        assert isinstance(outcome, GaussianOutcome)
        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(observed, minlength=CLASSES).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        present = torch.bincount(memory.labels, minlength=CLASSES) > 0
        return _Metrics(
            name=name,
            run=run,
            result=result,
            treatment_nll=float(F.nll_loss(propensity.log_probs, scaled.t)),
            ema_treatment_nll=float(F.nll_loss(ema_propensity.log_probs, scaled.t)),
            frequency_nll=float(F.nll_loss(baseline, scaled.t)),
            outcome_nll=float(-outcome.log_prob(scaled.y, scaled.t).mean()),
            targets=_terminal_targets(run, result),
            slot_norm=float(slots.norm(dim=-1).mean()),
            class_coverage=float(present.to(torch.float32).mean()),
            gate_rate=_mean_diagnostic(result, MEMORY_TERM, "coverage", late),
            slot_agreement=_mean_diagnostic(
                result, INSTANCE_TERM, "nearest_slot_agreement", late
            ),
            same_row_cosine=_mean_diagnostic(
                result, INSTANCE_TERM, "same_row_cosine", late
            ),
            cross_row_cosine=_mean_diagnostic(
                result, INSTANCE_TERM, "cross_row_cosine", late
            ),
            early_same_row_cosine=_mean_diagnostic(
                result, INSTANCE_TERM, "same_row_cosine", early
            ),
            early_cross_row_cosine=_mean_diagnostic(
                result, INSTANCE_TERM, "cross_row_cosine", early
            ),
        )


def _run(recipe: Recipe, train: XTYBatch, test: XTYBatch, *, name: str) -> _Metrics:
    run = compile(_with_steps(recipe, STEPS))
    result = run_stage(run, "joint_fit", _dataset(train), seed=100_000)
    return _evaluate(run, result, test, name=name)


_ARMS: tuple[tuple[str, Callable[[Recipe], Recipe]], ...] = (
    ("full", lambda recipe: recipe),
    ("no_propagation", _without_propagation),
    ("alpha_one", lambda recipe: _with_spec(recipe, alpha=1.0)),
    ("no_unfolding", lambda recipe: _with_spec(recipe, unfold=False)),
    ("no_instance_loss", _without_instance_loss),
    ("no_temporal_memory", lambda recipe: _with_spec(recipe, memory_momentum=0.0)),
    ("permuted_labels", lambda recipe: _with_memory(recipe, _permuted_labels)),
    ("no_alignment", lambda recipe: _with_memory(recipe, _unaligned)),
)
"""Card §6.2's arms 1 to 7 on the balanced fixture, in the order §6.2 lists them."""

_SKEWED_ARMS: tuple[tuple[str, Callable[[Recipe], Recipe]], ...] = (
    ("skewed_full", lambda recipe: recipe),
    ("skewed_no_alignment", lambda recipe: _with_memory(recipe, _unaligned)),
)
"""Arm 7's second fixture. Distribution alignment divides by a running marginal,
so "is it doing anything" needs a world where the marginal is not already
uniform, and it needs the aligned arm beside the unaligned one to say so."""


@pytest.fixture(scope="module")
def study() -> dict[str, _Metrics]:
    """Every predeclared arm, one seed each, on identical initial parameters."""
    schema = _schema()
    balanced = (
        _rows(TRAIN_ROWS, seed=90_001, row_offset=0),
        _rows(TEST_ROWS, seed=90_002, row_offset=10_000),
    )
    skewed = (
        _rows(TRAIN_ROWS, seed=90_001, row_offset=0, prevalence=SKEWED),
        _rows(TEST_ROWS, seed=90_002, row_offset=10_000, prevalence=SKEWED),
    )
    reference: dict[str, Tensor] | None = None
    metrics: dict[str, _Metrics] = {}
    for arms, (train, test) in ((_ARMS, balanced), (_SKEWED_ARMS, skewed)):
        for name, build in arms:
            torch.manual_seed(90_006)
            recipe = build(simmatch(schema))
            state = recipe.system.state_dict()
            if reference is None:
                reference = {key: value.clone() for key, value in state.items()}
            else:
                for key, value in reference.items():
                    assert torch.equal(value, state[key]), (
                        f"arm {name!r} does not start from the same parameters "
                        f"as the pair; {key!r} differs"
                    )
            metrics[name] = _run(recipe, train, test, name=name)
    return metrics


@pytest.fixture(scope="module")
def paired_fit(study: dict[str, _Metrics]) -> tuple[_Metrics, _Metrics]:
    """The §6 pair: full SimMatch and its no-propagation arm, one seed."""
    return study["full"], study["no_propagation"]


def test_the_label_budget_and_the_bank_are_the_declared_ones(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """`K = 64` is a fact of the population, so it is asserted here (card §4)."""
    full, _ = paired_fit
    population = full.result.population
    assert population is not None
    assert int(population.rows.t_observed.sum()) == OBSERVED
    observed = population.rows.t[population.rows.t_observed]
    assert int(torch.bincount(observed, minlength=CLASSES).min()) > 0
    assert full.targets.coverage == 1.0
    assert _memory(full.result).size == OBSERVED


def test_the_instance_loss_is_silent_until_the_bank_is_filled(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 invariant 6, in a real run rather than on a hand-built bank."""
    for metrics in paired_fit:
        result = metrics.result
        for step in range(WARMUP):
            assert _term(result, step, INSTANCE_TERM) == 0.0
            assert _diagnostic(result, step, INSTANCE_TERM, "propagated") == 0.0
        assert _diagnostic(result, WARMUP, MEMORY_TERM, "bank_coverage") == 1.0
        assert _diagnostic(result, WARMUP, INSTANCE_TERM, "propagated") == 1.0
        assert _term(result, WARMUP, INSTANCE_TERM) > 0.0


def test_the_instance_loss_and_the_supervised_term_fall_after_the_warm_up(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1, first clause.

    Eq. (5) is ungated, so its level is readable directly; it starts below the
    `log K` a uniform slot distribution would charge and comes down from there.
    """
    full, _ = paired_fit
    early = range(WARMUP, WARMUP + 100)
    late = range(STEPS - 100, STEPS)
    uniform = float(torch.tensor(float(OBSERVED)).log())
    for name in (INSTANCE_TERM, "observed_treatment_nll"):
        assert _mean(full.result, name, late) < _mean(full.result, name, early), name
    assert _mean(full.result, INSTANCE_TERM, late) < uniform
    # `hat q` sharpens as the geometry forms; a flat target would make the
    # falling loss a statement about scale rather than about matching.
    entropy_early = _mean_diagnostic(
        full.result, INSTANCE_TERM, "target_entropy", early
    )
    entropy_late = _mean_diagnostic(full.result, INSTANCE_TERM, "target_entropy", late)
    assert entropy_late < entropy_early < uniform


def test_the_gate_and_not_the_level_is_how_equation_two_is_read(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1, second clause.

    The gated term *rises* while its gate opens, which is the reading
    `fixmatch.md` §6 records for that paper's eq. (4). What must be true is that
    the rise is the curriculum: more rows accepted, and accepted at a
    confidence at or above the declared threshold.
    """
    full, _ = paired_fit
    early = range(WARMUP, WARMUP + 100)
    late = range(STEPS - 100, STEPS)
    rate_early = _mean_diagnostic(full.result, MEMORY_TERM, "coverage", early)
    assert rate_early < 0.5 < full.gate_rate
    accepted = _mean_diagnostic(full.result, MEMORY_TERM, "accepted_confidence", late)
    assert accepted >= SIMILARITY_MATCHING.threshold


def test_the_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1: the fit is worth comparing at all."""
    full, _ = paired_fit
    assert full.treatment_nll < full.frequency_nll
    assert full.ema_treatment_nll < full.frequency_nll


def test_propagation_moves_the_run_it_is_supposed_to_move(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The pair is controlled, so any difference is equations (8) and (10)."""
    full, ablated = paired_fit
    assert full.result.trace[:WARMUP] == ablated.result.trace[:WARMUP]
    assert full.result.trace != ablated.result.trace
    assert full.targets.propagated_nll != full.targets.aligned_nll
    # The ablation is the card's `hat p = p^w`, exactly.
    assert ablated.targets.propagated_nll == ablated.targets.aligned_nll


def test_the_propagated_targets_are_scored_whether_they_improve_or_not(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Card §6.2 Tier 1, arm 1's third clause. Tier 2 owns the direction.

    One seed at a fifth of the declared budget cannot support a claim about
    the mechanism, so this asserts only that every target is live, finite and
    better than an uninformative one; the ten-seed §6 contract is where a
    direction is allowed to be read.
    """
    full, _ = paired_fit
    uninformative = float(torch.tensor(float(CLASSES)).log())
    aggregate_q_weak = full.targets.aggregate_q_weak_nll
    assert aggregate_q_weak is not None
    for measured in (
        full.targets.propagated_nll,
        full.targets.aligned_nll,
        full.targets.aggregate_hat_q_nll,
        aggregate_q_weak,
    ):
        assert 0.0 < measured < uninformative


def test_the_outcome_head_is_not_damaged_by_the_two_extra_terms(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    full, ablated = paired_fit
    assert full.outcome_nll < 1.05 * ablated.outcome_nll


def test_arm_two_switches_off_the_instance_to_semantic_arrow(
    study: dict[str, _Metrics],
) -> None:
    """Card §6.2 arm 2 — the paper's `w/o hat p`, at `alpha = 1`.

    Equation (10) collapses to `hat p = DA(p^w)` exactly, while equation (8)
    keeps calibrating `hat q` by that same semantic prediction. So the identity
    is asserted and the outcome is reported: table 5's direction is a
    ten-replicate question and this is one seed at a fifth of the budget.
    """
    arm = study["alpha_one"]
    assert arm.targets.propagated_nll == arm.targets.aligned_nll
    assert arm.targets.aggregate_q_weak_nll is None
    assert arm.result.trace != study["full"].result.trace
    # Equation (8) is retained, so the instance target is still the calibrated
    # one and still differs from what the pair's ablation would have used.
    assert arm.result.trace != study["no_propagation"].result.trace


def test_arm_three_switches_off_the_semantic_to_instance_arrow(
    study: dict[str, _Metrics],
) -> None:
    """Card §6.2 arm 3 — the paper's `w/o hat q`, with unfolding disabled.

    Equation (8) returns equation (7)'s `q^w` unchanged, so the two aggregates
    §6 compares become one number; equation (10) still runs, so `hat p` still
    moves off `p^w`.
    """
    arm = study["no_unfolding"]
    aggregate_q_weak = arm.targets.aggregate_q_weak_nll
    assert aggregate_q_weak is not None
    assert arm.targets.aggregate_hat_q_nll == pytest.approx(aggregate_q_weak, rel=1e-5)
    assert arm.targets.propagated_nll != arm.targets.aligned_nll
    assert arm.result.trace != study["full"].result.trace


def test_arm_four_keeps_the_propagation_arithmetic_and_drops_its_gradient(
    study: dict[str, _Metrics],
) -> None:
    """Card §6.2 arm 4 — `lambda_in = 0`.

    The distinction the arm exists to make: equation (5) is still *computed*,
    so the bank, the calibrated target and both propagation arrows are the full
    arm's, and only what enters the mixed total is zero. A projection space
    that is never trained can then be read against one that is.
    """
    arm = study["no_instance_loss"]
    for step in (WARMUP, STEPS - 1):
        assert _term(arm.result, step, INSTANCE_TERM) > 0.0
        assert _weighted(arm.result, step, INSTANCE_TERM) == 0.0
        assert _diagnostic(arm.result, step, INSTANCE_TERM, "propagated") == 1.0
    assert _weighted(study["full"].result, STEPS - 1, INSTANCE_TERM) > 0.0
    assert arm.result.trace != study["full"].result.trace


def test_arm_five_runs_the_bank_without_its_temporal_memory(
    study: dict[str, _Metrics],
) -> None:
    """Card §6.2 arm 5 — bank momentum zero. No direction is asserted.

    At momentum zero equation (12) keeps only the last observation of a slot,
    so the bank a target reads is one step of history rather than a decayed
    trace. The card asks for target NLL and weak/strong alignment and predeclares
    no direction, because the paper does not ablate `m`; `_report` carries both.
    """
    arm = study["no_temporal_memory"]
    assert _memory(arm.result).spec.memory_momentum == 0.0
    assert arm.result.trace != study["full"].result.trace
    assert arm.targets.coverage == 1.0
    assert arm.slot_norm == pytest.approx(1.0, abs=1e-5)


def test_arm_six_permuted_memory_labels_are_the_wiring_control(
    study: dict[str, _Metrics],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Card §6.2 arm 6, in both of its forms, and what the strict one found.

    The card predeclares that permuting `Q_l` should make "both target
    improvements disappear". Run two ways, it does not, and the reason is worth
    the card amendment it earns rather than an assertion tuned to hide it.

    * The **trained** arm — the card's literal wording — retrains with the
      permutation in place, so it changes the run as well as the map and cannot
      isolate either.
    * The **isolated** control re-reads the full arm's *own* finished bank
      through a rolled slot-to-class map. Same student, same view draw, same
      slot features, same `p^w`; only equations (8) and (9)'s label lookup
      moves.

    Isolated, the aggregate collapses exactly as a wiring control should: a
    random half of the slots carries almost exactly half the similarity mass for
    every row, so `aggregate(q^w)` goes from an informative distribution to a
    near-constant one. What does **not** disappear is `hat p`'s advantage over
    `p^w`, because equation (10) mixes in 10% of *something* and 10% of a
    constant is shrinkage. This test pins that: the rolled `hat p` lands on the
    constant-prior shrinkage baseline, which is what says the surviving gain is
    smoothing rather than propagation.
    """
    control = study["permuted_labels"]
    full = study["full"]
    trained = _memory(control.result)
    assert not torch.equal(trained.labels, _memory(full.result).labels)
    assert torch.equal(
        torch.bincount(trained.labels, minlength=CLASSES),
        torch.bincount(_memory(full.result).labels, minlength=CLASSES),
    )

    isolated = _label_control(full)
    with capsys.disabled():
        print(
            f"\narm 6 isolated on the full arm's own bank: p^w "
            f"{isolated.p_weak:.5f}; hat p true {isolated.true_hat_p:.5f}, "
            f"rolled {isolated.rolled_hat_p:.5f}, constant-prior shrinkage "
            f"{isolated.shrunk:.5f}; aggregate(q^w) true "
            f"{isolated.true_aggregate:.5f}, rolled "
            f"{isolated.rolled_aggregate:.5f} (spread "
            f"{isolated.rolled_spread:.4f} against {isolated.true_spread:.4f})"
        )
    # The map is load-bearing for the thing that reads it.
    assert isolated.rolled_aggregate > isolated.true_aggregate
    assert isolated.rolled_spread < 0.1 * isolated.true_spread
    # And what survives the permutation is shrinkage, to within a rounding of it.
    assert isolated.rolled_hat_p == pytest.approx(isolated.shrunk, abs=0.005)


@dataclass(frozen=True)
class _LabelControl:
    """Arm 6 isolated: one bank, one student, two slot-to-class maps."""

    p_weak: float
    true_hat_p: float
    rolled_hat_p: float
    true_aggregate: float
    rolled_aggregate: float
    true_spread: float
    rolled_spread: float
    shrunk: float


def _label_control(metrics: _Metrics) -> _LabelControl:
    """Re-read one finished bank through the true and a rolled label map.

    `prepare` is given no support rows, so it neither validates a batch against
    `Q_l` nor writes a slot: the call is a pure read of the bank as the run left
    it, which is what lets the permuted copy exist at all.
    """
    run, result = metrics.run, metrics.result
    population = result.population
    assert population is not None
    schema = run.recipe.schema
    batch = population.rows
    rows = resolve_rows(batch, "t_missing")
    hidden = batch.t.index_select(0, rows)
    weak = _view(run, "weak_x").apply(
        batch, schema, rng_key=VIEW_KEY, population=population
    )
    memory = _memory(result)
    with torch.no_grad():
        values = run.graph.evaluate(weak, schema=schema, only=run.graph.names)
        propensity = values[Port.T_GIVEN_X]
        embedding = values[Port.X_PROJ]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(embedding, Tensor)
        readings = {}
        for name, labels in (
            ("true", memory.labels),
            ("rolled", torch.roll(memory.labels, 1)),
        ):
            clone = copy.deepcopy(memory)
            clone._labels = labels
            targets = clone.prepare(
                step=STEPS + 1,
                raw_probabilities=propensity.probs,
                weak_embeddings=embedding,
                batch=batch,
                eligible_rows=rows,
                support_rows=torch.empty(0, dtype=torch.long),
            )
            assert targets.propagated
            alpha = clone.spec.alpha
            aggregate = (targets.semantic - alpha * targets.aligned) / (1.0 - alpha)
            readings[name] = (
                _nll(targets.semantic, hidden),
                _nll(aggregate, hidden),
                float(aggregate[:, 1].std()),
                targets.aligned,
            )
        aligned = readings["true"][3]
        frequency = torch.bincount(memory.labels, minlength=CLASSES).float()
        frequency /= frequency.sum()
        alpha = memory.spec.alpha
        shrunk = alpha * aligned + (1.0 - alpha) * frequency
        return _LabelControl(
            p_weak=_nll(aligned, hidden),
            true_hat_p=readings["true"][0],
            rolled_hat_p=readings["rolled"][0],
            true_aggregate=readings["true"][1],
            rolled_aggregate=readings["rolled"][1],
            true_spread=readings["true"][2],
            rolled_spread=readings["rolled"][2],
            shrunk=_nll(shrunk, hidden),
        )


def test_arm_seven_reports_what_distribution_alignment_divides_by(
    study: dict[str, _Metrics],
) -> None:
    """Card §6.2 arm 7 — alignment off, on both fixtures, marginals reported.

    The assertion is the switch, not a performance direction: with alignment
    off the target *is* the model's own weak prediction, so the two marginals
    coincide, and with it on they do not. Whether uniformising helps on a
    fixture whose true prevalence is 0.15 is exactly the misspecification
    `paws.md` §6.2 found for me-max, and this card refuses to assert a
    direction for it.
    """
    for name in ("no_alignment", "skewed_no_alignment"):
        targets = study[name].targets
        assert targets.aligned_marginal == pytest.approx(
            targets.predicted_marginal, abs=1e-6
        ), name
    for name in ("full", "skewed_full"):
        targets = study[name].targets
        assert targets.aligned_marginal != pytest.approx(
            targets.predicted_marginal, abs=1e-6
        ), name
    # The skewed fixture is the one the card declares, to the prevalence it
    # declares; a fixture that drifted would make the arm about a third world.
    assert study["skewed_full"].targets.true_marginal == pytest.approx(0.15, abs=0.02)
    assert study["full"].targets.true_marginal == pytest.approx(0.5, abs=0.05)


def test_arm_eight_reports_every_declared_diagnostic_for_every_arm(
    study: dict[str, _Metrics],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Card §6.2 arm 8 — the report, and the sanity bounds it must respect.

    Every quantity §6.2 arm 8 names is printed for every arm, so a reviewer
    reads one table rather than eight test bodies. What is *asserted* is only
    what cannot be true of a working run: an uncovered bank, an unnormalised
    slot, a class the memory never saw, a cosine outside [-1, 1], or a target
    the gate accepted that is worse than a coin.
    """
    header = (
        f"{'arm':<21}{'t-NLL':>8}{'ema':>8}{'gate':>7}{'impur':>7}"
        f"{'slotN':>7}{'agree':>7}{'same':>8}{'cross':>8}{'margin':>8}"
        f"{'hat p':>8}{'p^w':>8}{'ag q̂':>8}{'ag q^w':>8}{'marg':>7}{'true':>7}"
    )
    lines = [header]
    for name, metrics in study.items():
        targets = metrics.targets
        aggregate_q_weak = targets.aggregate_q_weak_nll
        lines.append(
            f"{name:<21}{metrics.treatment_nll:>8.4f}"
            f"{metrics.ema_treatment_nll:>8.4f}{metrics.gate_rate:>7.3f}"
            f"{targets.impurity:>7.3f}{metrics.slot_norm:>7.3f}"
            f"{metrics.slot_agreement:>7.3f}{metrics.same_row_cosine:>8.4f}"
            f"{metrics.cross_row_cosine:>8.4f}{metrics.alignment_margin:>8.4f}"
            f"{targets.propagated_nll:>8.4f}{targets.aligned_nll:>8.4f}"
            f"{targets.aggregate_hat_q_nll:>8.4f}"
            + (
                f"{aggregate_q_weak:>8.4f}"
                if aggregate_q_weak is not None
                else f"{'n/a':>8}"
            )
            + f"{targets.predicted_marginal:>7.3f}{targets.true_marginal:>7.3f}"
        )
        lines.append(
            f"{'  early cosines':<21}{'':>16}"
            f"same {metrics.early_same_row_cosine:.4f}  "
            f"cross {metrics.early_cross_row_cosine:.4f}  "
            f"accepted {targets.accepted:.3f}  coverage {targets.coverage:.3f}  "
            f"class coverage {metrics.class_coverage:.3f}"
        )
    with capsys.disabled():
        print("\n" + "\n".join(lines))

    for name, metrics in study.items():
        assert metrics.targets.coverage == 1.0, name
        assert metrics.class_coverage == 1.0, name
        assert metrics.slot_norm == pytest.approx(1.0, abs=1e-5), name
        for cosine in (metrics.same_row_cosine, metrics.cross_row_cosine):
            assert -1.0 <= cosine <= 1.0, name
        if metrics.targets.accepted > 0.0:
            assert metrics.targets.impurity < 0.5, name
    # Only the full arm is required to have separated the two views at all. The
    # margin is the mechanism under test, and an arm built to switch a piece of
    # it off is entitled to have none — `no_propagation` very nearly does.
    assert study["full"].alignment_margin > study["no_propagation"].alignment_margin


def test_the_fixture_is_measured_before_it_is_trained_on(
    study: dict[str, _Metrics],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Card §6.1's three "measure rather than assume" items, reported and bounded.

    1. The Bayes-optimal treatment label's flip rate under each view. On this
       DGP that label *is* the Bayes-optimal cluster label — the assignment puts
       0.98 on the cluster's own level — so it is closed form in the four signal
       columns and needs no training. Deviation 2 says a view that destroys the
       label makes the run evidence about the view; the number is printed for
       both views and only the weak one carries a bound, because `fixmatch`'s
       0.5 strong rate is a known, recorded departure (`flexmatch.md` §5.2) that
       this card inherits deliberately rather than a property to re-litigate.
    2. Observed support by treatment level and the labelled-memory size.
    3. The true treatment prevalence in the train and held-out populations,
       which is what distribution alignment's uniform target is compared with.
    """
    schema = _schema()
    train = _rows(TRAIN_ROWS, seed=90_001, row_offset=0)
    test = _rows(TEST_ROWS, seed=90_002, row_offset=10_000)
    recipe = simmatch(schema)
    label = _bayes_label(train.x)
    flips = {
        name: _bayes_flip_rate(recipe, name, train, label)
        for name in ("weak_x", "strong_x")
    }
    population = study["full"].result.population
    assert population is not None
    support = torch.bincount(
        population.rows.t[population.rows.t_observed], minlength=CLASSES
    )
    with capsys.disabled():
        print(
            f"\n§6.1 measurements: weak-view flip {flips['weak_x']:.4f}, "
            f"strong-view flip {flips['strong_x']:.4f}; support by level "
            f"{support.tolist()} over K = {_memory(study['full'].result).size}; "
            f"true p(t=1) train {float(train.t.float().mean()):.4f}, held out "
            f"{float(test.t.float().mean()):.4f}, skewed train "
            f"{study['skewed_full'].targets.true_marginal:.4f}"
        )
    assert flips["weak_x"] < 0.10
    assert flips["strong_x"] > flips["weak_x"]
    assert int(support.sum()) == OBSERVED and int(support.min()) > 0
    for prevalence in (float(train.t.float().mean()), float(test.t.float().mean())):
        assert abs(prevalence - 0.5) < 0.05


def _bayes_label(x: Tensor) -> Tensor:
    """`arg max_c p(c | x)` from every signal column — the unmasked Bayes rule."""
    return _cluster_posterior(x[:, :4], torch.ones_like(x[:, :4])) >= 0.5


def _cluster_posterior(signal: Tensor, visible: Tensor) -> Tensor:
    """`p(c = 1 | visible signal columns)` under the §6.1 mixture.

    `x_j | c ~ N(0.45 (2c - 1), 0.6^2)` independently, and `FeatureMask`
    replaces a column with a constant carrying no information about `c`, so the
    Bayes rule conditions on the visible columns alone (`flexmatch.md` §5.2).
    """
    scale = 0.6

    def loglik(mean: float) -> Tensor:
        return (-((signal - mean) ** 2) / (2.0 * scale**2) * visible).sum(dim=1)

    return torch.sigmoid(loglik(CLUSTER_SIGNAL) - loglik(-CLUSTER_SIGNAL))


def _bayes_flip_rate(
    recipe: Recipe, view: str, train: XTYBatch, label: Tensor
) -> float:
    """How often one view's draw moves the Bayes-optimal label off `label`."""
    realised = recipe.view(view).apply(train, recipe.schema, rng_key=VIEW_KEY)
    signal = realised.x[:, :4]
    visible = (signal != 0.0).to(signal.dtype)
    return float(((_cluster_posterior(signal, visible) >= 0.5) != label).float().mean())
