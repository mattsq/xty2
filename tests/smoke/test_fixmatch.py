"""Tier 1 — FixMatch wiring on the card's two-cluster treatment DGP.

The fixture is the one predeclared in `docs/recipes/fixmatch.md` §6.1, and it is
run twice: once on the near-deterministic assignment the card names, and once on
an *overlapping* one. The second run is not a control for the first — it is the
tension the card's §2 records, made measurable. A confidence gate at 0.95 and
the positivity a causal contrast needs pull in opposite directions, and what the
gate then does under overlap (open anyway, on labels that are wrong far more
often) is the fact worth asserting.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Constant,
    FeatureSpec,
    GaussianOutcome,
    OutcomeSpec,
    Port,
    Program,
    Recipe,
    Schema,
    XTYBatch,
    compile,
)
from xty2.objectives import PseudoLabelTreatmentNLL
from xty2.recipes import fixmatch
from xty2.training import StageResult, run_stage

FEATURES = 6
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048
BATCH_SIZE = 256
STEPS = 3_000
SHORT_STEPS = 600
OBSERVED_TREATMENTS = 40
"""Card §6: 40 of 1,024, the label-scarce regime FixMatch is aimed at."""

CLUSTER_SIGNAL = 0.45
SEPARATED = 0.02
"""`p(t=1 | cluster=0)`; the assignment is 0.02/0.98 and the gate can open."""

OVERLAPPING = 0.15
"""The same DGP with the overlap a causal contrast would need."""

THRESHOLD = 0.95
PSEUDO_LABEL = "pseudo_label_treatment_nll"


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    """Small dense layers are faster and deterministic without thread fan-out."""
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


@dataclass(frozen=True)
class _Population:
    batch: XTYBatch
    true_effect: torch.Tensor


def _population(
    rows: int,
    *,
    seed: int,
    observed_treatments: int,
    row_offset: int,
    low: float,
) -> _Population:
    """The card's §6.1 mechanism, with `low` the cluster-0 propensity."""
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < 0.5).float()
    sign = 2.0 * cluster - 1.0
    x = epsilon_x.clone()
    x[:, :4] = CLUSTER_SIGNAL * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = low + (1.0 - 2.0 * low) * cluster
    t = (u_t < propensity).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1.0)
    true_effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    y = baseline + t * true_effect + 0.5 * epsilon_y

    observed = torch.zeros(rows, dtype=torch.bool)
    if observed_treatments:
        missingness = torch.Generator().manual_seed(seed + 10_000)
        selected = torch.randperm(rows, generator=missingness)[:observed_treatments]
        observed[selected] = True
    return _Population(
        batch=XTYBatch(
            x=x,
            t=t,
            y=y,
            t_observed=observed,
            y_observed=torch.ones(rows, dtype=torch.bool),
            row_id=torch.arange(row_offset, row_offset + rows),
        ),
        true_effect=true_effect,
    )


def _take(batch: XTYBatch, rows: torch.Tensor) -> XTYBatch:
    return XTYBatch(
        x=batch.x.index_select(0, rows),
        t=batch.t.index_select(0, rows),
        y=batch.y.index_select(0, rows),
        t_observed=batch.t_observed.index_select(0, rows),
        y_observed=batch.y_observed.index_select(0, rows),
        row_id=batch.row_id.index_select(0, rows),
    )


@dataclass(frozen=True)
class _BatchStream:
    population: XTYBatch
    indices: torch.Tensor

    def __iter__(self) -> Iterator[XTYBatch]:
        for rows in self.indices:
            yield _take(self.population, rows)


def _batch_indices(rows: int, *, steps: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [torch.randperm(rows, generator=generator)[:BATCH_SIZE] for _ in range(steps)]
    )


def _with_steps(recipe: Recipe, steps: int) -> Recipe:
    stage = recipe.program[0]
    return replace(recipe, program=Program((replace(stage, steps=steps),)))


def _zero_weight(recipe: Recipe) -> Recipe:
    """The `lambda_u = 0` ablation: the same fit without eq. (4)."""
    stage = recipe.program[0]
    ablated = replace(stage.objectives[2], weight=Constant(0.0))
    objectives = (*stage.objectives[:2], ablated, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _with_threshold(recipe: Recipe, threshold: float) -> Recipe:
    stage = recipe.program[0]
    gated = stage.objectives[2].objective
    assert isinstance(gated, PseudoLabelTreatmentNLL)
    weighted = replace(
        stage.objectives[2], objective=replace(gated, threshold=threshold)
    )
    objectives = (*stage.objectives[:2], weighted, *stage.objectives[3:])
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def _term(result: StageResult, step: int, name: str) -> object:
    return next(term for term in result.records[step].terms if term.name == name)


def _coverage(result: StageResult, step: int) -> float:
    term = _term(result, step, PSEUDO_LABEL)
    return float(term.diagnostics["coverage"])  # type: ignore[attr-defined]


def _mean_coverage(result: StageResult, steps: range) -> float:
    return sum(_coverage(result, step) for step in steps) / len(steps)


def _raw(result: StageResult, step: int, name: str) -> float:
    return float(_term(result, step, name).value)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    treatment_nll: float
    frequency_nll: float
    outcome_nll: float
    mask_rate: float
    impurity: float


def _evaluate(
    run: CompiledRun, result: StageResult, test: _Population, train: XTYBatch
) -> _Metrics:
    """Two parameter sets, and the split between them is not cosmetic.

    Predictive metrics come from the EMA copy, because that is the model
    section 2.4 reports. The paper's mask rate (eq. 6) and impurity (eq. 5)
    come from the **trained** network, because they describe the labels the run
    actually trained on — eq. (4) reads the current parameters, so an EMA mask
    rate would be a statistic of a model that never gated anything. Measured
    off the EMA at decay 0.999 it reads 0.0 on the overlapping fixture, which
    is exactly the kind of true-but-unrelated number this split avoids.
    """
    schema = run.recipe.schema
    assert result.teacher is not None
    graph = result.teacher.graph
    with torch.no_grad():
        values = graph.evaluate(test.batch, schema=schema, only=run.graph.names)
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        assert isinstance(propensity, CategoricalTreatment)
        assert isinstance(outcome, GaussianOutcome)
        treatment_nll = float(F.nll_loss(propensity.log_probs, test.batch.t))
        outcome_nll = float(-outcome.log_prob(test.batch.y, test.batch.t).mean())
        frequencies = torch.bincount(train.t[train.t_observed], minlength=2).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(test.batch.batch_size, -1)
        frequency_nll = float(F.nll_loss(baseline, test.batch.t))

        # The paper's own mask rate (eq. 6) and impurity (eq. 5), on the
        # training rows whose treatment is missing. Impurity needs the true
        # `t`, which is why it is measured here and not by the objective.
        train_values = run.graph.evaluate(train, schema=schema, only=run.graph.names)
        train_propensity = train_values[Port.T_GIVEN_X]
        assert isinstance(train_propensity, CategoricalTreatment)
        confidence, labels = train_propensity.probs.max(dim=-1)
        missing = ~train.t_observed
        retained = (confidence >= THRESHOLD) & missing
        mask_rate = float(retained.sum() / missing.sum())
        impurity = (
            float((labels[retained] != train.t[retained]).float().mean())
            if int(retained.sum())
            else 0.0
        )
    return _Metrics(
        run=run,
        result=result,
        treatment_nll=treatment_nll,
        frequency_nll=frequency_nll,
        outcome_nll=outcome_nll,
        mask_rate=mask_rate,
        impurity=impurity,
    )


def _standardised(low: float, seed: int) -> tuple[XTYBatch, _Population]:
    train = _population(
        TRAIN_ROWS,
        seed=seed,
        observed_treatments=OBSERVED_TREATMENTS,
        row_offset=0,
        low=low,
    )
    test = _population(
        TEST_ROWS,
        seed=seed + 2,
        observed_treatments=TEST_ROWS,
        row_offset=10_000,
        low=low,
    )
    mean = train.batch.y.mean()
    scale = float(train.batch.y.std(unbiased=False))
    train_batch = train.batch.replace(y=(train.batch.y - mean) / scale)
    test = replace(test, batch=test.batch.replace(y=(test.batch.y - mean) / scale))
    return train_batch, test


def _run(recipe: Recipe, train: XTYBatch, test: _Population, steps: int) -> _Metrics:
    run = compile(_with_steps(recipe, steps))
    batches = _BatchStream(train, _batch_indices(TRAIN_ROWS, steps=steps, seed=90_005))
    result = run_stage(run, "joint_fit", batches, seed=90_010)
    return _evaluate(run, result, test, train)


@pytest.fixture(scope="module")
def paired_fit() -> tuple[_Metrics, _Metrics]:
    """FixMatch and its `lambda_u = 0` ablation, same seeds and same batches."""
    schema = _schema()
    train, test = _standardised(SEPARATED, seed=90_001)

    torch.manual_seed(90_006)
    scheduled = fixmatch(schema)
    torch.manual_seed(90_006)
    ablated = _zero_weight(fixmatch(schema))
    for name, value in scheduled.system.state_dict().items():
        assert torch.equal(value, ablated.system.state_dict()[name])

    return (
        _run(scheduled, train, test, STEPS),
        _run(ablated, train, test, STEPS),
    )


@pytest.fixture(scope="module")
def overlapping_fit() -> _Metrics:
    """The same recipe where positivity holds and the gate should not be trusted."""
    train, test = _standardised(OVERLAPPING, seed=91_001)
    torch.manual_seed(90_006)
    return _run(fixmatch(_schema()), train, test, SHORT_STEPS)


def test_the_evaluation_teacher_never_takes_a_gradient(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """It is reported from, and nothing else."""
    scheduled, _ = paired_fit
    teacher = scheduled.result.teacher
    assert teacher is not None
    assert teacher.spec.role == "evaluation"
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_the_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, _ = paired_fit
    assert scheduled.treatment_nll < scheduled.frequency_nll


def test_the_supervised_term_decreases(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The mixed total is the wrong thing to watch here, by construction.

    Eq. (4)'s contribution *grows* as the gate opens, so a falling total would
    be evidence about the curriculum rather than about the fit. The supervised
    cross-entropy of eq. (3) is the term that must come down.
    """
    scheduled, _ = paired_fit
    early = sum(_raw(scheduled.result, s, "observed_treatment_nll") for s in range(100))
    late = sum(
        _raw(scheduled.result, s, "observed_treatment_nll")
        for s in range(STEPS - 100, STEPS)
    )
    assert late < 0.5 * early


def test_the_gate_opens_as_the_model_sharpens(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Section 2.2: the threshold supplies a curriculum "for free".

    Nothing ramps eq. (4)'s weight, so if this is true at all it is true
    because `max(q_b)` crosses `tau` over training and not because a schedule
    made it.
    """
    scheduled, _ = paired_fit
    assert _coverage(scheduled.result, 0) == 0.0
    early = _mean_coverage(scheduled.result, range(100))
    late = _mean_coverage(scheduled.result, range(STEPS - 100, STEPS))
    assert early < 0.25 < late
    assert scheduled.mask_rate > 0.2


def test_the_retained_labels_are_mostly_correct(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, _ = paired_fit
    assert scheduled.impurity < 0.15


def test_pseudo_labelling_beats_the_zero_weight_ablation(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    scheduled, ablated = paired_fit
    assert scheduled.treatment_nll < ablated.treatment_nll


def test_the_outcome_stack_is_not_damaged(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """Eq. (4) trains `p(t | x)` only; the outcome head must not pay for it."""
    scheduled, ablated = paired_fit
    assert scheduled.outcome_nll < 1.05 * ablated.outcome_nll


def test_overlap_opens_the_gate_on_labels_it_should_not_trust(
    paired_fit: tuple[_Metrics, _Metrics], overlapping_fit: _Metrics
) -> None:
    """The card's §2 tension, measured rather than asserted.

    With `p(t | x)` in [0.15, 0.85] the Bayes-optimal propensity never reaches
    0.95, so a calibrated model would retain nothing here — and at the first
    step none is retained. The gate opens anyway as the model grows confident
    past what the data supports, and by the end it is keeping half the rows at
    an error rate several times the separable fit's. That is FixMatch's
    confirmation bias (appendix B.2) arriving exactly where a causal design
    would have put it: the gate is not a guard against the overlap regime, it
    is a measure of the model's own confidence in it.
    """
    scheduled, _ = paired_fit
    early = _mean_coverage(overlapping_fit.result, range(100))
    assert _coverage(overlapping_fit.result, 0) == 0.0
    # Stated as a ratio rather than an absolute: the claim is that the gate
    # starts near-shut and opens as the fit overreaches, and a threshold tuned
    # to one trajectory would be re-tuned every time the recipe changed.
    assert early < 0.2 * overlapping_fit.mask_rate
    assert overlapping_fit.mask_rate > 0.2
    assert overlapping_fit.impurity > 3.0 * scheduled.impurity
    assert overlapping_fit.impurity > 0.10


def _short_trace(threshold: float) -> StageResult:
    train, _ = _standardised(SEPARATED, seed=90_001)
    torch.manual_seed(90_006)
    recipe = _with_threshold(fixmatch(_schema()), threshold)
    run = compile(_with_steps(recipe, 200))
    batches = _BatchStream(train, _batch_indices(TRAIN_ROWS, steps=200, seed=90_005))
    return run_stage(run, "joint_fit", batches, seed=90_010)


def test_the_reviewed_threshold_is_the_one_that_ran() -> None:
    """A mutation test on `tau`, not a demonstration that a gate exists.

    At `tau = 1` nothing is ever retained and eq. (4) contributes exactly zero
    while still counting its rows; at `tau = 0` everything is retained. The
    reviewed 0.95 must sit strictly between the two, which no default and no
    silently-dropped threshold could satisfy.
    """
    closed = _short_trace(1.0)
    open_gate = _short_trace(0.0)
    reviewed = _short_trace(THRESHOLD)

    assert _mean_coverage(closed, range(200)) == 0.0
    assert all(_term(closed, step, PSEUDO_LABEL).weighted == 0.0 for step in range(200))  # type: ignore[attr-defined]
    assert all(_term(closed, step, PSEUDO_LABEL).n > 0 for step in range(200))  # type: ignore[attr-defined]
    assert _mean_coverage(open_gate, range(200)) == 1.0
    assert 0.0 < _mean_coverage(reviewed, range(200)) < 1.0
