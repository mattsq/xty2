"""Tier 1 — ReMixMatch wiring and the card's one-seed mechanism arms."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Dataset,
    Port,
    Program,
    Recipe,
    compile,
)
from xty2.evaluation.benchmarks import remixmatch as benchmark
from xty2.evaluation.benchmarks.common import (
    cluster_centres,
    cluster_population,
    continuous_schema,
    on_the_training_scale,
    training_dataset,
)
from xty2.objectives import AnchoredLabelGuess
from xty2.recipes import remixmatch
from xty2.recipes.remixmatch import GUESS_OWNER
from xty2.training import StageResult, run_stage

from tests.smoke.test_fixmatch import SEPARATED, _dataset, _populations, _schema

STUDY_STEPS = 300
BASE_SEED = 90_000
TRAIN_PRIOR = (0.55, 0.25, 0.13, 0.07)
EFFECTS = (0.0, 1.0, 0.4, 1.6)
CLASSES = 4


@pytest.mark.parametrize("index", [0, 1])
def test_corrected_benchmark_executes_all_arms_and_readonly_diagnostics(
    index: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise real fits at three steps; this is not Tier 2 evidence."""
    results: list[StageResult] = []

    def short_fit(
        run: CompiledRun, stage: str, data: Dataset, *, seed: int
    ) -> StageResult:
        recipe = run.recipe
        short_stage = replace(recipe.program[0], steps=3)
        short_run = compile(replace(recipe, program=Program((short_stage,))))
        result = run_stage(short_run, stage, data, seed=seed)
        results.append(result)
        return result

    monkeypatch.setattr(benchmark, "run_stage", short_fit)
    monkeypatch.setattr(benchmark, "_TERMINAL_STEPS", 2)
    metrics = benchmark._replicate(index)
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert len(results) == 4
    assert not results[1].objective_states
    assert {term.name for term in results[1].records[-1].terms} == {
        "observed_outcome_nll",
        "observed_treatment_nll",
        "missing_treatment_marginal_nll",
    }
    for name in (
        "estimated_labelled",
        "true_training",
        "true_unlabelled",
        "aligned_window",
        "unaligned_window",
    ):
        assert sum(
            metrics[f"{name}_class_{level}"] for level in range(4)
        ) == pytest.approx(1.0)

    truth = cluster_population(
        1024,
        seed=BASE_SEED + 100 * index + 1,
        row_offset=0,
        classes=4,
        prior=TRAIN_PRIOR,
        effects=EFFECTS,
    ).batch
    full, unaligned = results[0], results[2]
    guess = full.objective_states[GUESS_OWNER]
    assert isinstance(guess, AnchoredLabelGuess)
    previous = (
        guess.last_prepared_step,
        guess.prediction_marginal.clone(),
        guess.labelled_marginal.clone(),
    )
    expected = benchmark._marginal_diagnostics(full, unaligned, truth)
    # A shuffled truth table must select the same missing rows by ID.
    permutation = torch.arange(truth.batch_size - 1, -1, -1)
    shuffled = truth.replace(
        x=truth.x[permutation],
        t=truth.t[permutation],
        y=truth.y[permutation],
        row_id=truth.row_id[permutation],
        t_observed=truth.t_observed[permutation],
        y_observed=truth.y_observed[permutation],
    )
    assert benchmark._marginal_diagnostics(full, unaligned, shuffled) == expected
    assert guess.last_prepared_step == previous[0]
    assert torch.equal(guess.prediction_marginal, previous[1])
    assert torch.equal(guess.labelled_marginal, previous[2])
    population = full.population
    assert population is not None
    missing = torch.isin(
        truth.row_id, population.rows.row_id[population.rows.t_missing]
    )
    true_missing = (
        torch.bincount(truth.t[missing], minlength=4).double() / missing.sum()
    )
    for level in range(4):
        assert expected[f"true_unlabelled_class_{level}"] == float(true_missing[level])
    assert metrics["alignment_advantage"] == pytest.approx(
        benchmark._marginal_l1(unaligned, truth.t)
        - benchmark._marginal_l1(full, truth.t)
    )
    # The §6 guardrail: the same paired difference taken against `p(y)`, the
    # marginal alignment targets, which both arms estimate identically.
    labelled = guess.labelled_marginal
    assert torch.equal(labelled, benchmark._guess(unaligned).labelled_marginal)
    assert expected["alignment_labelled_marginal_L1_advantage"] == pytest.approx(
        float((benchmark._guess(unaligned).prediction_marginal - labelled).abs().sum())
        - float((guess.prediction_marginal - labelled).abs().sum())
    )


def test_all_terms_are_finite_and_the_state_advances() -> None:
    train, _ = _populations(SEPARATED, seed=90_001)
    recipe = remixmatch(_schema())
    stage = replace(recipe.program[0], steps=3)
    run = compile(replace(recipe, program=Program((stage,))))
    result = run_stage(run, "joint_fit", _dataset(train), seed=100_000)
    assert len(result.records) == 3
    assert all(torch.isfinite(torch.tensor(record.total)) for record in result.records)
    assert all(len(record.terms) == 6 for record in result.records)
    state = result.objective_states[GUESS_OWNER]
    assert isinstance(state, AnchoredLabelGuess)
    assert state.last_prepared_step == 2


def _short(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    return replace(recipe, program=Program((replace(stage, steps=STUDY_STEPS),)))


def _macro_nll(log_probs: torch.Tensor, labels: torch.Tensor) -> float:
    per_row = F.nll_loss(log_probs, labels, reduction="none")
    return float(
        torch.stack(
            [per_row[labels == level].mean() for level in range(CLASSES)]
        ).mean()
    )


def _mean_term(result: StageResult, name: str, steps: range) -> float:
    records = result.records
    values = [
        next(term.value for term in records[step].terms if term.name == name)
        for step in steps
    ]
    return sum(values) / len(values)


def test_the_predeclared_one_seed_mechanism_study_runs() -> None:
    """Exercise K=1, no-pretext, no-premixup, and mean-redux beside full."""
    torch.set_num_threads(1)
    schema = continuous_schema(6, treatments=CLASSES)
    train = cluster_population(
        1_024,
        seed=BASE_SEED + 1,
        row_offset=0,
        classes=CLASSES,
        prior=TRAIN_PRIOR,
        effects=EFFECTS,
    )
    test = cluster_population(
        2_048,
        seed=BASE_SEED + 2,
        row_offset=10_000,
        classes=CLASSES,
        effects=EFFECTS,
    )
    data = training_dataset(schema, train.batch)
    builders = {
        "full": {},
        "K1": {"strong_draws": 1},
        "no_pretext": {"pretext_weight": 0.0},
        "no_premixup": {"premixup_weight": 0.0},
        "mean_redux": {"redux": "mean"},
    }
    recipes = {}
    for name, arguments in builders.items():
        torch.manual_seed(BASE_SEED + 6)
        recipes[name] = _short(remixmatch(schema, **arguments))  # type: ignore[arg-type]
    reference = recipes["full"].system.state_dict()
    for name, recipe in recipes.items():
        for component, value in reference.items():
            assert torch.equal(value, recipe.system.state_dict()[component]), name

    runs = {name: compile(recipe) for name, recipe in recipes.items()}
    results = {
        name: run_stage(run, "joint_fit", data, seed=BASE_SEED + 10_000)
        for name, run in runs.items()
    }
    trained = results["full"].checkpoint.trained_on_row_ids
    for result in results.values():
        assert torch.equal(trained, result.checkpoint.trained_on_row_ids)
        assert all(
            torch.isfinite(torch.tensor(record.total)) for record in result.records
        )

    full = results["full"]
    early = range(0, 25)
    late = range(STUDY_STEPS - 25, STUDY_STEPS)
    for term in (
        "mixed_labelled_treatment_nll",
        "mixed_unlabelled_treatment_nll",
        GUESS_OWNER,
        "pretext_transform_nll",
    ):
        assert _mean_term(full, term, late) < _mean_term(full, term, early), term

    population = full.population
    assert population is not None
    scaled = on_the_training_scale(test.batch, population)
    with torch.no_grad():
        values = runs["full"].graph.evaluate(
            scaled, schema=schema, only=runs["full"].graph.names
        )
        propensity = values[Port.T_GIVEN_X]
        assert isinstance(propensity, CategoricalTreatment)
        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(observed, minlength=CLASSES).float()
        frequencies /= frequencies.sum()
        frequency_nll = _macro_nll(
            frequencies.log().expand(scaled.batch_size, -1), scaled.t
        )
        assert _macro_nll(propensity.log_probs, scaled.t) < frequency_nll

    terminal = range(STUDY_STEPS - 100, STUDY_STEPS)
    accuracy = sum(
        next(
            term.diagnostics["accuracy"]
            for term in full.records[step].terms
            if term.name == "pretext_transform_nll"
        )
        for step in terminal
    ) / len(terminal)
    assert accuracy > 0.25

    centres = cluster_centres(CLASSES)
    original = torch.cdist(train.batch.x[:, :4], centres).argmin(dim=-1)
    flip_rates = []
    for name in ("weak_x", "strong_x"):
        viewed = (
            recipes["full"]
            .view(name)
            .apply(train.batch, schema, rng_key=BASE_SEED + 10_000)
        )
        changed = torch.cdist(viewed.x[:, :4], centres).argmin(dim=-1)
        flip_rates.append(float((changed != original).float().mean()))
    assert 0.0 <= flip_rates[0] < flip_rates[1] < 0.5
