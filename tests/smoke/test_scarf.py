"""Tier 1 — SCARF pretraining on the card's two-cluster treatment DGP.

The fixture is the one predeclared in `docs/recipes/scarf.md` §6.1 — the same
DGP as `fixmatch.md` §6.1, at the paper's batch size of 128 — and it is run as
a **pair**: the recipe's two-stage program, and the identical `joint_fit` stage
with no pretraining, from the same initial parameters and the same batch
stream. The only difference between the arms is whether the encoder was
contrastively pretrained.

This is a wiring test and not a fidelity claim (`FIDELITY.md` §3). What it
answers is whether the mechanism is connected: whether the corrupted view
reaches the loss, whether the loss shapes the representation in the way SCARF
describes rather than by collapsing it, and whether the pretrained encoder is
actually what the second stage starts from.

What it deliberately does **not** assert is the card's §6 headline — that
pretraining improves the scarce-label treatment fit. That was measured over
five seeds and four fine-tuning budgets and it is absent; `scarf.md` §6.2
records the numbers. A Tier 1 assertion on a quantity whose sign moves with the
seed would be a green light bought with a lottery ticket, which is the opposite
of what this tier is for. The frozen-encoder probe below is what survived: it
is the same comparison with the encoder held fixed, and it is where the
representation's content shows up.
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
from xty2.recipes import scarf
from xty2.training import STREAM_STRIDE, ProgramResult, run_program

FEATURES = 6
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048
BATCH_SIZE = 128
"""Card §6.1: SCARF's `N`, and a hyperparameter of its loss (deviation 6)."""

PRETRAIN_STEPS = 1_000
FIT_STEPS = 3_000
PROBE_STEPS = 1_000
"""The frozen-encoder probe's budget; see `_frozen`."""
OBSERVED_TREATMENTS = 40
CLUSTER_SIGNAL = 0.45
SEPARATED = 0.02
CONTRASTIVE = "info_nce_contrastive"


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
    rows: int, *, seed: int, observed_treatments: int, row_offset: int
) -> _Population:
    """The card's §6.1 mechanism, which is `fixmatch.md` §6.1's unchanged."""
    generator = torch.Generator().manual_seed(seed)
    u_c = torch.rand(rows, generator=generator)
    epsilon_x = torch.randn(rows, FEATURES, generator=generator)
    u_t = torch.rand(rows, generator=generator)
    epsilon_y = torch.randn(rows, generator=generator)

    cluster = (u_c < 0.5).float()
    sign = 2.0 * cluster - 1.0
    x = epsilon_x.clone()
    x[:, :4] = CLUSTER_SIGNAL * sign[:, None] + 0.6 * epsilon_x[:, :4]
    propensity = SEPARATED + (1.0 - 2.0 * SEPARATED) * cluster
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


def _unpretrained(recipe: Recipe) -> Recipe:
    """The ablation: the same fitting stage, from the recipe's initialisation.

    `initialise_from` is what the pretraining delivers, so dropping the stage
    without dropping the edge would leave a program that cannot run. Both are
    removed and nothing else is touched — same graph, same objectives, same
    optimiser, same budget.
    """
    fit = recipe.program[1]
    return replace(recipe, program=Program((replace(fit, initialise_from=None),)))


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: ProgramResult
    treatment_nll: float
    frequency_nll: float
    outcome_nll: float
    ate_error: float


def _evaluate(
    run: CompiledRun, result: ProgramResult, test: _Population, train: XTYBatch
) -> _Metrics:
    schema = run.recipe.schema
    with torch.no_grad():
        values = run.graph.evaluate(
            test.batch,
            schema=schema,
            only=("mlp_encoder", "tarnet_head", "categorical_propensity"),
        )
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

        candidates = torch.arange(2).expand(test.batch.batch_size, 2)
        means = outcome.mean(candidates)
        estimated = float((means[:, 1] - means[:, 0]).mean())
    return _Metrics(
        run=run,
        result=result,
        treatment_nll=treatment_nll,
        frequency_nll=frequency_nll,
        outcome_nll=outcome_nll,
        ate_error=abs(estimated - float(test.true_effect.mean())),
    )


def _standardised(seed: int) -> tuple[XTYBatch, _Population]:
    train = _population(
        TRAIN_ROWS, seed=seed, observed_treatments=OBSERVED_TREATMENTS, row_offset=0
    )
    test = _population(
        TEST_ROWS, seed=seed + 2, observed_treatments=TEST_ROWS, row_offset=10_000
    )
    mean = train.batch.y.mean()
    scale = float(train.batch.y.std(unbiased=False))
    train_batch = train.batch.replace(y=(train.batch.y - mean) / scale)
    test = replace(test, batch=test.batch.replace(y=(test.batch.y - mean) / scale))
    return train_batch, test


@pytest.fixture(scope="module")
def paired_fit() -> tuple[_Metrics, _Metrics]:
    """SCARF and its no-pretraining ablation, same start and same batches."""
    schema = _schema()
    train, test = _standardised(seed=90_001)

    torch.manual_seed(90_006)
    pretrained = scarf(schema)
    torch.manual_seed(90_006)
    ablated = _unpretrained(scarf(schema))
    for name, value in pretrained.system.state_dict().items():
        assert torch.equal(value, ablated.system.state_dict()[name])

    fitting = _BatchStream(
        train, _batch_indices(TRAIN_ROWS, steps=FIT_STEPS, seed=90_005)
    )
    contrastive = _BatchStream(
        train, _batch_indices(TRAIN_ROWS, steps=PRETRAIN_STEPS, seed=90_004)
    )

    with_pretraining = compile(pretrained)
    without = compile(ablated)
    full = run_program(
        with_pretraining,
        {"pretrain": contrastive, "joint_fit": fitting},
        seed=90_010,
    )
    first = _evaluate(with_pretraining, full, test, train)
    # The ablation's single stage is index 0, so its seed is offset by one
    # stride to give its fit the same stochastic stream as the paired arm's.
    bare = run_program(without, {"joint_fit": fitting}, seed=90_010 + STREAM_STRIDE)
    return first, _evaluate(without, bare, test, train)


def _diagnostic(result: ProgramResult, step: int, name: str) -> float:
    record = result.stage("pretrain").records[step]
    term = next(entry for entry in record.terms if entry.name == CONTRASTIVE)
    return float(term.diagnostics[name])


def _contrastive_value(result: ProgramResult, step: int) -> float:
    record = result.stage("pretrain").records[step]
    term = next(entry for entry in record.terms if entry.name == CONTRASTIVE)
    return float(term.value)


def _mean(values: Iterator[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected)


def test_the_contrastive_loss_decreases(paired_fit: tuple[_Metrics, _Metrics]) -> None:
    pretrained, _ = paired_fit
    early = _mean(_contrastive_value(pretrained.result, s) for s in range(50))
    late = _mean(
        _contrastive_value(pretrained.result, s)
        for s in range(PRETRAIN_STEPS - 50, PRETRAIN_STEPS)
    )
    assert late < early


def test_the_representation_separates_rather_than_collapses(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The assertion "the loss went down" cannot make (`FIDELITY.md` §4.1).

    A collapsed encoder — every embedding one direction — drives `L_cont` to
    exactly `0` and stays there, which is *below* the value an untrained
    network starts at. What separates the two is the gap between a row's
    similarity to its own corrupted copy and its similarity to the other rows
    of the batch, and the card's §6 tolerance is stated on that gap.
    """
    pretrained, _ = paired_fit
    start = _diagnostic(pretrained.result, 0, "alignment") - _diagnostic(
        pretrained.result, 0, "uniformity"
    )
    final = PRETRAIN_STEPS - 1
    gap = _diagnostic(pretrained.result, final, "alignment") - _diagnostic(
        pretrained.result, final, "uniformity"
    )
    # An untrained smooth map already puts a row near its own corrupted copy,
    # so the starting gap is well above zero and the claim has to be about
    # movement: the card's §6 tolerance is the terminal gap, and this adds that
    # it is not the gap the network was born with.
    assert gap > 0.2
    assert gap > 1.5 * start
    assert _diagnostic(pretrained.result, final, "alignment") > 0.4
    assert _diagnostic(pretrained.result, final, "uniformity") < 0.05


def test_the_fitting_stage_starts_from_the_pretrained_encoder(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """`initialise_from` as a fact about the run, not a field on a dataclass.

    The pretraining checkpoint carries `f` and `g` and nothing else — "after
    pre-training, `g` is discarded" is a statement about the *next* stage, and
    what makes it true here is that no `joint_fit` pass reads `X_PROJ`.
    """
    pretrained, _ = paired_fit
    checkpoint = pretrained.result.stage("pretrain").checkpoint
    assert checkpoint.components == ("mlp_encoder", "projection_head")
    # `ComponentGraph` qualifies its parameters with the module-registry
    # prefix; a checkpoint carries the component-relative names.
    initial = {
        name.removeprefix("_components."): value
        for name, value in pretrained.run.initial_parameters().items()
    }
    moved = [
        name
        for name, value in checkpoint.parameters.items()
        if not torch.equal(value, initial[name])
    ]
    assert any(name.startswith("mlp_encoder") for name in moved)
    assert any(name.startswith("projection_head") for name in moved)


def test_the_propensity_beats_the_frequency_baseline(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    pretrained, ablated = paired_fit
    assert pretrained.treatment_nll < pretrained.frequency_nll
    assert ablated.treatment_nll < ablated.frequency_nll


def test_the_pretrained_initialisation_reaches_the_fit(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """The two arms differ, which is the wiring claim `initialise_from` makes.

    Whether the difference is an *improvement* is the card's §6 question and
    §6.2's answer is "not at this budget". That it exists at all is this
    tier's: a program that silently dropped the checkpoint would produce two
    identical fits, and nothing else in this file would notice.
    """
    pretrained, ablated = paired_fit
    assert pretrained.treatment_nll != ablated.treatment_nll
    assert pretrained.outcome_nll != ablated.outcome_nll


def test_the_outcome_stack_is_not_damaged(
    paired_fit: tuple[_Metrics, _Metrics],
) -> None:
    """A representation tuned to `x` alone could cost the outcome head."""
    pretrained, ablated = paired_fit
    assert pretrained.outcome_nll < 1.05 * ablated.outcome_nll


def _frozen(recipe: Recipe) -> Recipe:
    """The recipe with the encoder held fixed during the fit.

    Not the recipe: SCARF fine-tunes `f` (card §3.1), and freezing it to make a
    number move would be the tuning this project exists to refuse. It is a
    *measurement* — the standard linear-probe question, asked with the
    framework's own vocabulary — and what it isolates is whether the pretrained
    representation carries anything about the treatment at all, separately from
    whether 1,000 steps of gradient on 40 labels then erases it.
    """
    stages = tuple(
        replace(
            stage,
            steps=PROBE_STEPS,
            trainable=("tarnet_head", "categorical_propensity"),
        )
        if stage.name == "joint_fit"
        else stage
        for stage in recipe.program
    )
    return replace(recipe, program=Program(stages))


@pytest.fixture(scope="module")
def frozen_probe() -> tuple[_Metrics, _Metrics]:
    """The same pair with the encoder frozen: a probe of the representation."""
    schema = _schema()
    train, test = _standardised(seed=90_001)

    torch.manual_seed(90_006)
    pretrained = _frozen(scarf(schema))
    torch.manual_seed(90_006)
    ablated = _frozen(_unpretrained(scarf(schema)))

    fitting = _BatchStream(
        train, _batch_indices(TRAIN_ROWS, steps=PROBE_STEPS, seed=90_005)
    )
    contrastive = _BatchStream(
        train, _batch_indices(TRAIN_ROWS, steps=PRETRAIN_STEPS, seed=90_004)
    )
    with_pretraining = compile(pretrained)
    without = compile(ablated)
    full = run_program(
        with_pretraining,
        {"pretrain": contrastive, "joint_fit": fitting},
        seed=90_010,
    )
    bare = run_program(without, {"joint_fit": fitting}, seed=90_010 + STREAM_STRIDE)
    return (
        _evaluate(with_pretraining, full, test, train),
        _evaluate(without, bare, test, train),
    )


def test_the_pretrained_representation_predicts_the_treatment(
    frozen_probe: tuple[_Metrics, _Metrics],
) -> None:
    """What SCARF's pretraining bought, measured where it is not overwritten.

    Card §6.2: over five seeds this margin is 0.87 in the mean, with four wins
    and one tie, so it is a real effect on this fixture and not a large one.
    The fixture seed is the comfortable end of that spread; a failure here
    should be read against those numbers rather than as a broken mechanism.
    """
    pretrained, ablated = frozen_probe
    assert pretrained.treatment_nll < ablated.treatment_nll
    assert pretrained.treatment_nll < pretrained.frequency_nll


def test_the_frozen_pretrained_representation_costs_the_outcome_head(
    frozen_probe: tuple[_Metrics, _Metrics],
) -> None:
    """The other half of the same measurement, and it is not a happy one.

    A representation shaped by the covariance of `x` alone is not shaped for
    `p(y | x, t)`, and with the encoder frozen the outcome head cannot reshape
    it. Card §2 says SCARF is not a causal method; this is what that costs when
    the representation is imposed rather than fine-tuned, and it is asserted so
    that it stays visible.
    """
    pretrained, ablated = frozen_probe
    assert pretrained.outcome_nll > ablated.outcome_nll
