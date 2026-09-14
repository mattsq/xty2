"""Recorded SimSiam evidence is complete, paired and scored by the card."""

import json
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from xty2.evaluation.benchmarks import simsiam as benchmark
from xty2.evaluation.reporting import assert_result_matches_card, load_reproduction_spec

ROOT = Path(__file__).parents[2]
CARD = ROOT / "docs/recipes/simsiam.md"
RESULT = ROOT / "docs/experiments/results/simsiam-c14090a/simsiam.json"


def recorded() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(RESULT.read_text())
    return result


def test_saved_replicates_and_decisions_are_independently_recomputed() -> None:
    result = recorded()
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert result["replicates"] == 10
    assert result["commit"] == "c14090a43e2e"
    for metric in metrics.values():
        values = metric["values"]
        assert len(values) == 10 and all(math.isfinite(v) for v in values)
        mean = statistics.mean(values)
        stderr = statistics.stdev(values) / math.sqrt(10)
        assert metric["mean"] == pytest.approx(mean, abs=1e-12)
        assert metric["stderr"] == pytest.approx(stderr, abs=1e-12)
        if metric["relation"] == ">":
            assert metric["passed"] == (mean - stderr > metric["target"])
        elif metric["relation"] == ">=":
            assert metric["passed"] == (mean - stderr >= metric["target"])
        elif metric["relation"] == "<=":
            assert metric["passed"] == (mean + stderr <= metric["target"])
    for difference, full, control in (
        (
            "stop_gradient_spread_gap",
            "full_projection_spread",
            "no_stop_projection_spread",
        ),
        (
            "predictor_spread_gap",
            "full_projection_spread",
            "no_predictor_projection_spread",
        ),
        ("pretraining_outcome_nll_cost", "full_outcome_nll", "no_pretrain_outcome_nll"),
    ):
        for index in range(10):
            assert metrics[difference]["values"][index] == pytest.approx(
                metrics[full]["values"][index] - metrics[control]["values"][index],
                abs=1e-12,
            )
    assert result["status"] == "deviating"
    assert sum(metric["passed"] is True for metric in metrics.values()) == 2


def test_canonical_runner_rescores_the_saved_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = recorded()
    rows = tuple(
        {metric["name"]: metric["values"][index] for metric in saved["metrics"]}
        for index in range(10)
    )

    def replay(
        function: object, count: int, *, workers: int
    ) -> tuple[dict[str, float], ...]:
        assert function is benchmark.replicate
        assert count == 10 and workers == 1
        return rows

    monkeypatch.setattr(benchmark, "parallel_replicates", replay)
    result = benchmark.run(
        load_reproduction_spec(CARD, recipe="simsiam"),
        saved["commit"],
        saved["date"],
        1,
        ROOT / "runs",
    )
    assert result.as_json() == saved
    assert_result_matches_card(result, CARD)


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
    spec = load_reproduction_spec(CARD, recipe="simsiam")
    changed = replace(spec, values={**spec.values, key: "changed"})
    with pytest.raises(ValueError, match="reviewed value"):
        benchmark.run(changed, "unused", "2026-09-14", 1, ROOT / "runs")
