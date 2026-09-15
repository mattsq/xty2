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
RESULT = ROOT / "docs/experiments/results/simsiam-946601d/simsiam.json"


def recorded() -> dict[str, Any]:
    result: dict[str, Any] = json.loads(RESULT.read_text())
    return result


def test_saved_replicates_and_decisions_are_independently_recomputed() -> None:
    result = recorded()
    metrics = {metric["name"]: metric for metric in result["metrics"]}
    assert result["replicates"] == 10
    assert result["commit"] == "946601def994"
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
    for difference, left, right in (
        ("stop_gradient_alignment_gap", "no_stop_alignment", "full_alignment"),
        ("predictor_alignment_gap", "no_predictor_alignment", "full_alignment"),
        (
            "stop_gradient_rank_gap",
            "full_encoder_effective_rank",
            "no_stop_encoder_effective_rank",
        ),
        (
            "predictor_rank_gap",
            "full_encoder_effective_rank",
            "no_predictor_encoder_effective_rank",
        ),
        (
            "encoder_rank_retention",
            "full_encoder_effective_rank",
            "initial_encoder_effective_rank",
        ),
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
                metrics[left]["values"][index] - metrics[right]["values"][index],
                abs=1e-12,
            )
    assert result["status"] == "reproduced"
    assert sum(metric["passed"] is True for metric in metrics.values()) == 5
    # Card section 6.4's audit note, still true on a tree where the mechanism
    # does reproduce: both retired spread gaps are recorded, carry no bound, and
    # would have failed the `mean - stderr > 0` one they used to carry — on the
    # very run where all four replacement bounds pass.
    for retired in ("stop_gradient_spread_gap", "predictor_spread_gap"):
        assert metrics[retired]["criterion"] == "informational"
        values = metrics[retired]["values"]
        mean = statistics.mean(values)
        assert mean - statistics.stdev(values) / math.sqrt(10) <= 0.0


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
    actual = result.as_json()
    assert {key: value for key, value in actual.items() if key != "metrics"} == {
        key: value for key, value in saved.items() if key != "metrics"
    }
    for got, want in zip(actual["metrics"], saved["metrics"], strict=True):
        assert got.keys() == want.keys()
        for key in got:
            # Derived summaries can differ by roundoff across platforms.
            if key in {"mean", "stderr", "margin"} and want[key] is not None:
                assert got[key] == pytest.approx(want[key], rel=1e-12, abs=1e-12)
            else:
                assert got[key] == want[key]
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
