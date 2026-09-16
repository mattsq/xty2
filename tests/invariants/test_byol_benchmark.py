"""Independent protocol, pairing and strict-bound regression oracles."""

import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from xty2.core import Recipe, compile
from xty2.evaluation.benchmarks import byol as benchmark
from xty2.evaluation.benchmarks.common import continuous_schema
from xty2.evaluation.byol_study import ARMS, arm_recipe, paired_metrics, require_equal
from xty2.evaluation.reporting import (
    MetricResult,
    assert_result_matches_card,
    load_reproduction_spec,
)
from xty2.evaluation.simsiam_study import snapshot
from xty2.evaluation.vicreg_views import OracleSymmetry
from xty2.recipes import byol

ROOT = Path(__file__).parents[2]
CARD = ROOT / "docs/recipes/byol.md"
RESULT = ROOT / "docs/experiments/results/byol-tier2/byol.json"


def test_tier2_seed_stream_and_budgets() -> None:
    with patch.object(benchmark, "study", return_value={}) as study:
        for index in range(10):
            benchmark.replicate(index)
            study.assert_called_with(
                620000 + 100 * index,
                train_rows=1024,
                test_rows=2048,
                pretrain_steps=1000,
                fit_steps=3000,
                warmup_steps=10,
                ramp_steps=1000,
                eval_batches=16,
            )


def test_full_budget_arms_preserve_recipe_schedules_and_initial_tensors() -> None:
    initial = None
    for arm in ARMS:
        torch.manual_seed(620006)
        recipe = byol(
            continuous_schema(6),
            first_transforms=(OracleSymmetry(),),
            second_transforms=(OracleSymmetry(),),
        )
        adapted = arm_recipe(
            recipe,
            arm,
            pretrain_steps=1000,
            fit_steps=3000,
            warmup_steps=10,
            ramp_steps=1000,
        )
        run = compile(adapted)
        if initial is None:
            initial = snapshot(run.graph)
        require_equal(snapshot(run.graph), initial)
        assert adapted.program[-1].steps == 3000
        assert adapted.program[-1].objectives == recipe.program[-1].objectives
        if arm == "no_pretrain":
            assert len(adapted.program) == 1
            assert adapted.program[0].initialise_from is None
        else:
            assert adapted.program[0].steps == 1000
            assert adapted.program[0].optimiser == recipe.program[0].optimiser
            if arm != "zero_decay":
                assert adapted.program[0].teacher == recipe.program[0].teacher


def test_paired_contrast_signs_use_within_seed_values() -> None:
    assert paired_metrics(
        {
            "full_outcome_nll": 2.0,
            "zero_decay_outcome_nll": 5.0,
            "no_pretrain_outcome_nll": 1.0,
            "no_predictor_outcome_nll": 7.0,
            "full_encoder_effective_rank": 4.0,
            "no_predictor_encoder_effective_rank": 1.5,
        }
    ) == {
        "ema_outcome_nll_gain": 3.0,
        "pretraining_outcome_nll_cost": 1.0,
        "encoder_effective_rank": 4.0,
        "predictor_outcome_nll_gain": 5.0,
        "predictor_encoder_rank_gain": 2.5,
    }


def test_strict_upper_bound_rejects_boundary_and_uncertain_pass() -> None:
    assert MetricResult("cost", (0.0, 0.0), "<", 0.05).passed
    assert not MetricResult("cost", (0.05, 0.05), "<", 0.05).passed
    assert not MetricResult("cost", (-0.04, 0.08), "<", 0.05).passed
    # Exact binary arithmetic: mean=0.25, SE=0.125, mean+SE=0.375.
    assert not MetricResult("cost", (0.125, 0.375), "<", 0.375).passed
    assert MetricResult("cost", (0.125, 0.375), "<=", 0.375).passed


def test_strict_upper_bound_oracle_kills_inclusive_mutant() -> None:
    with (
        patch.object(
            MetricResult,
            "passed",
            property(lambda metric: metric.margin >= metric.stderr),
        ),
        pytest.raises(AssertionError),
    ):
        test_strict_upper_bound_rejects_boundary_and_uncertain_pass()


def test_protocol_oracles_kill_seed_budget_and_contrast_mutants() -> None:
    original_replicate = benchmark.replicate

    def wrong_seed(index: int) -> dict[str, float]:
        return original_replicate(index + 1)

    with (
        patch.object(benchmark, "replicate", wrong_seed),
        pytest.raises(AssertionError),
    ):
        test_tier2_seed_stream_and_budgets()
    original_arm = arm_recipe

    def wrong_budget(recipe: Recipe, arm: str, **kwargs: object) -> Recipe:
        # Deliberately retaining smoke defaults must fail the full-budget check.
        return original_arm(recipe, arm)

    with (
        patch(f"{__name__}.arm_recipe", side_effect=wrong_budget),
        pytest.raises(AssertionError),
    ):
        test_full_budget_arms_preserve_recipe_schedules_and_initial_tensors()
    original_metrics = paired_metrics

    def wrong_sign(metrics: dict[str, float]) -> dict[str, float]:
        result = original_metrics(metrics)
        result["ema_outcome_nll_gain"] *= -1
        return result

    with (
        patch(f"{__name__}.paired_metrics", side_effect=wrong_sign),
        pytest.raises(AssertionError),
    ):
        test_paired_contrast_signs_use_within_seed_values()


def test_runner_keeps_all_three_gates_and_diagnostics() -> None:
    rows = tuple(
        {
            "ema_outcome_nll_gain": 0.02,
            "pretraining_outcome_nll_cost": 0.01,
            "encoder_effective_rank": 2.0,
            "no_predictor_effect_rmse": 99.0,
        }
        for _ in range(10)
    )
    with patch.object(benchmark, "parallel_replicates", return_value=rows) as run:
        result = benchmark.run(
            load_reproduction_spec(CARD, recipe="byol"),
            "test",
            "2026-09-16",
            3,
            ROOT / "runs",
        )
        run.assert_called_once_with(benchmark.replicate, 10, workers=3)
    assert result.status == "reproduced"
    assert [(m.name, m.relation, m.target) for m in result.metrics] == [
        ("ema_outcome_nll_gain", ">", 0.0),
        ("pretraining_outcome_nll_cost", "<", 0.05),
        ("encoder_effective_rank", ">", 1.1),
        ("no_predictor_effect_rmse", "info", None),
    ]
    for key, bad in (
        ("ema_outcome_nll_gain", 0.0),
        ("pretraining_outcome_nll_cost", 0.05),
        ("encoder_effective_rank", 1.1),
    ):
        with patch.object(
            benchmark,
            "parallel_replicates",
            return_value=tuple({**row, key: bad} for row in rows),
        ):
            failed = benchmark.run(
                load_reproduction_spec(CARD, recipe="byol"),
                "test",
                "2026-09-16",
                3,
                ROOT / "runs",
            )
        assert failed.status == "deviating"


@pytest.mark.parametrize(
    "key",
    [
        "dataset",
        "variant",
        "split",
        "metric",
        "published",
        "published_source",
        "tolerance",
        "seeds",
        "report",
    ],
)
def test_every_reproduction_scalar_is_bound(key: str) -> None:
    spec = load_reproduction_spec(CARD, recipe="byol")
    changed = replace(spec, values={**spec.values, key: "changed"})
    with pytest.raises(ValueError, match="reviewed value"):
        benchmark.run(changed, "unused", "2026-09-16", 1, ROOT / "runs")


def test_recorded_evidence_recomputes_from_all_ten_paired_replicates() -> None:
    saved: dict[str, Any] = json.loads(RESULT.read_text())
    manifest = json.loads((RESULT.parent / "environment.json").read_text())
    assert manifest["source_commit"].startswith(saved["commit"])
    metrics = {m["name"]: m for m in saved["metrics"]}
    assert saved["replicates"] == 10
    for metric in metrics.values():
        values = metric["values"]
        assert len(values) == 10 and all(math.isfinite(v) for v in values)
        mean = statistics.mean(values)
        stderr = statistics.stdev(values) / math.sqrt(10)
        assert metric["mean"] == pytest.approx(mean, abs=1e-12)
        assert metric["stderr"] == pytest.approx(stderr, abs=1e-12)
        if metric["relation"] == ">":
            assert metric["passed"] == (mean - stderr > metric["target"])
        elif metric["relation"] == "<":
            assert metric["passed"] == (mean + stderr < metric["target"])
        else:
            assert metric["relation"] == "info" and metric["passed"] is None
    rows = tuple(
        {name: metric["values"][i] for name, metric in metrics.items()}
        for i in range(10)
    )
    for row in rows:
        # Independent signs, using the saved arm observations directly.
        assert row["ema_outcome_nll_gain"] == pytest.approx(
            row["zero_decay_outcome_nll"] - row["full_outcome_nll"], abs=1e-12
        )
        assert row["pretraining_outcome_nll_cost"] == pytest.approx(
            row["full_outcome_nll"] - row["no_pretrain_outcome_nll"], abs=1e-12
        )
        assert row["encoder_effective_rank"] == row["full_encoder_effective_rank"]
        for arm in ARMS:
            for diagnostic in ("treatment_nll", "effect_rmse", "encoder_spread"):
                assert math.isfinite(row[f"{arm}_{diagnostic}"])
    with patch.object(benchmark, "parallel_replicates", return_value=rows):
        result = benchmark.run(
            load_reproduction_spec(CARD, recipe="byol"),
            saved["commit"],
            saved["date"],
            1,
            ROOT / "runs",
        )
    assert result.status == saved["status"]
    assert result.spec_digest == saved["spec_digest"]
    assert_result_matches_card(result, CARD)
