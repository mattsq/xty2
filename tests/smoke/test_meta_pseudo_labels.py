"""Tier 1 — one full-budget Meta Pseudo Labels paired run."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace

import pytest
import torch
from torch.nn import functional as F
from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    Dataset,
    FeatureSpec,
    Port,
    Program,
    Recipe,
    Schema,
    XTYBatch,
    compile,
)
from xty2.objectives import MetaFeedbackCoefficient
from xty2.recipes import INNER_ROLE, OUTER_ROLE, meta_pseudo_labels
from xty2.training import StageResult, run_meta_gradient

BASE_SEED = 94_000
TRAIN_ROWS = 1_024
TEST_ROWS = 2_048


@pytest.fixture(scope="module", autouse=True)
def _one_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _schema() -> Schema:
    return Schema(
        tuple(FeatureSpec(f"x{index}", "continuous") for index in range(6)),
        treatment_cardinality=2,
    )


def _population(rows: int, *, seed: int, row_offset: int) -> XTYBatch:
    generator = torch.Generator().manual_seed(seed)
    cluster = (torch.rand(rows, generator=generator) < 0.5).float()
    epsilon_x = torch.randn(rows, 6, generator=generator)
    x = epsilon_x.clone()
    x[:, :4] = 0.45 * (2.0 * cluster - 1.0)[:, None] + 0.6 * epsilon_x[:, :4]
    t = (torch.rand(rows, generator=generator) < (0.02 + 0.96 * cluster)).long()
    baseline = 0.5 * x[:, 0] - 0.3 * x[:, 1] + 0.2 * (x[:, 4].square() - 1)
    effect = 1.0 + 0.5 * torch.tanh(x[:, 2])
    y = baseline + t * effect + 0.5 * torch.randn(rows, generator=generator)
    return XTYBatch(
        x=x,
        t=t,
        y=y,
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(row_offset, row_offset + rows),
    )


def _zero_feedback(recipe: Recipe) -> Recipe:
    stage = recipe.program[0]
    meta = stage.meta_gradient
    assert meta is not None
    feedback = meta.feedback
    assert isinstance(feedback, MetaFeedbackCoefficient)
    return replace(
        recipe,
        program=Program(
            (
                replace(
                    stage,
                    meta_gradient=replace(
                        meta, feedback=replace(feedback, force_zero=True)
                    ),
                ),
            )
        ),
    )


@dataclass(frozen=True)
class _Metrics:
    run: CompiledRun
    result: StageResult
    student_nll: float
    teacher_nll: float
    frequency_nll: float
    class_mass_concentration: float


def _evaluate(run: CompiledRun, result: StageResult, test: XTYBatch) -> _Metrics:
    with torch.no_grad():
        student = result.role_graphs[INNER_ROLE].evaluate(
            test, schema=run.recipe.schema, only=run.graph.names
        )[Port.T_GIVEN_X]
        teacher = result.role_graphs[OUTER_ROLE].evaluate(
            test, schema=run.recipe.schema, only=run.graph.names
        )[Port.T_GIVEN_X]
    assert isinstance(student, CategoricalTreatment)
    assert isinstance(teacher, CategoricalTreatment)
    population = result.population
    assert population is not None
    observed = population.rows.t[population.rows.t_observed]
    frequency = torch.bincount(observed, minlength=2).float()
    frequency /= frequency.sum()
    baseline = frequency.log().expand(test.batch_size, -1)
    return _Metrics(
        run=run,
        result=result,
        student_nll=float(F.nll_loss(student.log_probs, test.t)),
        teacher_nll=float(F.nll_loss(teacher.log_probs, test.t)),
        frequency_nll=float(F.nll_loss(baseline, test.t)),
        class_mass_concentration=float(student.probs.mean(dim=0).max()),
    )


@pytest.fixture(scope="module")
def paired() -> Mapping[str, _Metrics]:
    train = _population(TRAIN_ROWS, seed=BASE_SEED + 1, row_offset=0)
    test = _population(TEST_ROWS, seed=BASE_SEED + 2, row_offset=10_000)
    data = Dataset(
        schema=_schema(),
        rows=train,
        assignments={"train": torch.arange(train.batch_size)},
    )
    full_recipe = meta_pseudo_labels(_schema())
    recipes = {"full": full_recipe, "zero": _zero_feedback(full_recipe)}
    results: dict[str, _Metrics] = {}
    for name, recipe in recipes.items():
        run = compile(recipe)
        result = run_meta_gradient(
            run,
            "meta_train",
            data,
            seed=BASE_SEED + 10_000,
            role_seeds={
                OUTER_ROLE: BASE_SEED + 6,
                INNER_ROLE: BASE_SEED + 7,
            },
            hard_label_seed=BASE_SEED + 20_000,
        )
        results[name] = _evaluate(run, result, test)
    return results


def test_both_paired_arms_run_the_full_budget_finitely(
    paired: Mapping[str, _Metrics],
) -> None:
    for metrics in paired.values():
        assert metrics.result.steps == 3_000
        for record in metrics.result.records:
            values = [
                record.lr,
                record.total,
                record.grad_norm,
                *record.role_lrs.values(),
                *record.role_grad_norms.values(),
            ]
            for term in record.terms:
                values.extend((term.value, term.weight, term.weighted))
                values.extend(term.diagnostics.values())
            assert all(math.isfinite(value) for value in values)
        for checkpoint in metrics.result.role_checkpoints.values():
            assert all(
                torch.isfinite(value).all() for value in checkpoint.parameters.values()
            )


def test_both_students_beat_frequency_and_avoid_one_class_collapse(
    paired: Mapping[str, _Metrics],
) -> None:
    for metrics in paired.values():
        assert metrics.student_nll < metrics.frequency_nll
        assert metrics.class_mass_concentration < 0.95


def test_feedback_and_both_parameter_roles_are_live(
    paired: Mapping[str, _Metrics],
) -> None:
    full = paired["full"].result
    h = [
        term.diagnostics["h"]
        for record in full.records
        for term in record.terms
        if term.name == "teacher_meta_score"
    ]
    assert any(value != 0.0 for value in h)
    assert any(record.role_grad_norms[INNER_ROLE] > 0.0 for record in full.records)
    assert any(record.role_grad_norms[OUTER_ROLE] > 0.0 for record in full.records)
    meta = next(
        term for term in full.records[-1].terms if term.name == "teacher_meta_score"
    )
    uda = next(
        term
        for term in full.records[-1].terms
        if term.name == "teacher_uda_consistency"
    )
    tsa = next(
        term for term in full.records[-1].terms if term.name == "teacher_tsa_nll"
    )
    assert 0.0 <= meta.diagnostics["sampled_label_accuracy"] <= 1.0
    assert meta.diagnostics["sampled_label_entropy"] >= 0.0
    assert 0.0 <= uda.diagnostics["coverage"] <= 1.0
    assert 0.0 <= tsa.diagnostics["retained_fraction"] <= 1.0


def test_view_flip_rates_are_reported_with_strong_above_weak() -> None:
    train = _population(TRAIN_ROWS, seed=BASE_SEED + 1, row_offset=0)
    recipe = meta_pseudo_labels(_schema())
    label = train.x[:, :4].sum(dim=-1) > 0
    rates = []
    for view in ("weak_x", "strong_x"):
        viewed = recipe.view(view).apply(
            train, recipe.schema, rng_key=BASE_SEED + 10_000
        )
        rates.append(float(((viewed.x[:, :4].sum(dim=-1) > 0) != label).float().mean()))
    assert 0.0 <= rates[0] < rates[1]
