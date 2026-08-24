"""Tier 0 — cards drive replicate counts and result-ledger status."""

from __future__ import annotations

from pathlib import Path

import pytest
from xty2.evaluation import (
    BenchmarkResult,
    MetricResult,
    assert_result_matches_card,
    load_reproduction_spec,
    mean_and_stderr,
    update_card_text,
)

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("recipe", "seeds", "metric"),
    [
        ("tarnet", 10, "sqrt_PEHE_in_sample"),
        (
            "cnflow",
            10,
            "test conditional outcome NLL p(Y|X,T), explicitly not joint or "
            "missing-treatment marginal NLL",
        ),
        (
            "mean_teacher",
            10,
            "held-out masked-view student/teacher probability-MSE ratio; "
            "treatment NLL and sqrt_PEHE guardrails",
        ),
        ("cycle_dual", 10, "absolute_ATE_error"),
        ("ssdml", 20, "absolute_ATE_error"),
    ],
)
def test_every_recipe_card_supplies_a_parseable_tier2_spec(
    recipe: str, seeds: int, metric: str
) -> None:
    spec = load_reproduction_spec(ROOT / "docs" / "recipes" / f"{recipe}.md")
    assert spec.recipe == recipe
    assert spec.seed_count == seeds
    assert spec.primary_metric == metric
    assert len(spec.digest) == 64


def test_mean_and_stderr_uses_the_sample_standard_error() -> None:
    mean, stderr = mean_and_stderr([1.0, 2.0, 3.0, 4.0])
    assert mean == 2.5
    assert stderr == pytest.approx(0.6454972243679028)
    with pytest.raises(ValueError, match="at least two"):
        mean_and_stderr([1.0])


def _card(status: str = "smoke-passing") -> str:
    return f"""# Recipe spec card: example

**Status:** `{status}`

## 5. Deviations from the paper

None.

## 6. Reproduction target

### 6.1 Result ledger

| Date | Commit | Metric | Value +/- stderr | Within tolerance? |
|---|---|---|---|---|
| | | | | |

## 7. Unknowns
"""


def _result(*, passed: bool, protocol_deviation: str | None = None) -> BenchmarkResult:
    value = 0.1 if passed else 0.4
    return BenchmarkResult(
        recipe="example",
        commit="abc1234",
        date="2026-08-24",
        spec_digest="digest",
        metrics=(MetricResult.upper_bound("error", [value, value], 0.2),),
        interpretation="A fixed test interpretation.",
        protocol_deviation=protocol_deviation,
    )


def test_ledger_writeback_sets_reproduced_and_records_mean_stderr() -> None:
    updated = update_card_text(_card(), _result(passed=True))
    assert "**Status:** `reproduced`" in updated
    assert "| 2026-08-24 | `abc1234` | error | 0.1 +/- 0 | yes |" in updated
    assert "### Tier 2 outcome" not in updated


def test_deviation_writeback_adds_the_required_section5_explanation() -> None:
    result = _result(passed=False)
    updated = update_card_text(_card(), result)
    assert "**Status:** `deviating`" in updated
    assert "### Tier 2 outcome" in updated
    assert "Failed target(s): error was 0.4 +/- 0 against mean <= 0.2." in updated
    assert updated.index("### Tier 2 outcome") < updated.index(
        "## 6. Reproduction target"
    )


def test_protocol_deviation_forces_deviating_even_when_the_metric_matches() -> None:
    result = _result(
        passed=True,
        protocol_deviation="Only ten of the declared 1,000 realisations ran.",
    )
    assert result.target_passed
    assert result.status == "deviating"


def test_fresh_result_must_match_the_status_recorded_in_the_card(
    tmp_path: Path,
) -> None:
    result = _result(passed=False)
    card = tmp_path / "example.md"
    card.write_text(update_card_text(_card(), result), encoding="utf-8")
    assert_result_matches_card(result, card)

    card.write_text(_card("reproduced"), encoding="utf-8")
    with pytest.raises(AssertionError, match="fresh benchmark is 'deviating'"):
        assert_result_matches_card(result, card)
