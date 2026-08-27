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
from xty2.evaluation.benchmarks import RECIPES, benchmark_function, cycle_dual

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
    assert (
        "Failed target(s): error was 0.4 +/- 0 against mean <= 0.2, "
        "by at least one stderr."
    ) in updated
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


def test_the_template_ledger_header_spelling_is_also_accepted() -> None:
    # The template writes "Value ± stderr"; the cards written before P12 write
    # "Value +/- stderr". A card made from the template must not be rejected as
    # having no ledger.
    template = _card().replace("Value +/- stderr", "Value ± stderr")
    updated = update_card_text(template, _result(passed=True))
    assert "| 2026-08-24 | `abc1234` | error | 0.1 +/- 0 | yes |" in updated
    assert "Value ± stderr" in updated


def test_the_shipped_template_carries_a_writeable_result_ledger() -> None:
    template = (ROOT / "docs" / "recipes" / "_TEMPLATE.md").read_text(encoding="utf-8")
    updated = update_card_text(template, _result(passed=True))
    assert "| 2026-08-24 | `abc1234` | error | 0.1 +/- 0 | yes |" in updated


def test_a_one_seed_card_is_rejected_when_it_is_parsed(tmp_path: Path) -> None:
    # `mean_and_stderr` needs two replicates; the card is where that is said.
    source = (ROOT / "docs" / "recipes" / "cycle_dual.md").read_text(encoding="utf-8")
    card = tmp_path / "cycle_dual.md"
    card.write_text(source.replace("  seeds: 10", "  seeds: 1"), encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        load_reproduction_spec(card)


def _spec(tmp_path: Path, recipe: str = "cycle_dual") -> Path:
    source = (ROOT / "docs" / "recipes" / f"{recipe}.md").read_text(encoding="utf-8")
    card = tmp_path / f"{recipe}.md"
    card.write_text(source, encoding="utf-8")
    return card


def test_bind_accounts_for_every_declared_reproduction_scalar(
    tmp_path: Path,
) -> None:
    spec = load_reproduction_spec(_spec(tmp_path))
    implemented = {
        key: value for key, value in spec.values.items() if key != "published_source"
    }
    spec.bind(implemented, documentation=("published_source",))

    # A scalar the card declares but the benchmark never reads is what lets an
    # amended protocol run under this block's digest.
    partial = dict(implemented)
    del partial["split"]
    with pytest.raises(ValueError, match=r"declares \['split'\]"):
        spec.bind(partial, documentation=("published_source",))

    # A scalar the benchmark binds but the card no longer declares.
    with pytest.raises(ValueError, match=r"no longer declares \['folds'\]"):
        spec.bind({**implemented, "folds": "5"}, documentation=("published_source",))

    # A reviewed value the card changed without the benchmark changing.
    with pytest.raises(ValueError, match="Amend and review the card"):
        spec.bind(
            {**implemented, "variant": "ternary treatment; 70% treatment MCAR"},
            documentation=("published_source",),
        )

    with pytest.raises(ValueError, match="one or the other"):
        spec.bind(implemented, documentation=("split",))


def test_amending_a_protocol_scalar_stops_the_benchmark(tmp_path: Path) -> None:
    # The failure this guards: the digest recorded on the artifact is taken
    # over the whole block, so an amended `split` that no adapter reads would
    # otherwise be stamped onto evidence produced by the old protocol.
    source = (ROOT / "docs" / "recipes" / "cycle_dual.md").read_text(encoding="utf-8")
    card = tmp_path / "cycle_dual.md"
    card.write_text(
        source.replace(
            "  split: independent 2048 train / 1024 validation / 2048 test per "
            "replicate",
            "  split: independent 4096 train / 1024 validation / 2048 test per "
            "replicate",
        ),
        encoding="utf-8",
    )
    amended = load_reproduction_spec(card)
    assert amended.digest != load_reproduction_spec(_spec(tmp_path)).digest
    with pytest.raises(ValueError, match="Amend and review the card"):
        cycle_dual.run(
            amended, commit="abc1234", date="2026-08-25", workers=1, cache_root=tmp_path
        )


# ---------------------------------------------------------------------------
# The Tier 2 suite covers every recipe it claims to (PLAN.md P12)
# ---------------------------------------------------------------------------


def test_every_tier2_recipe_has_a_card_a_module_and_a_test() -> None:
    """The three files a nightly matrix entry needs, checked together.

    The nightly derives its matrix from `RECIPES`, so a name in that tuple is a
    job that will run `tests/benchmarks/test_<name>.py` against
    `docs/recipes/<name>.md`. A name with a missing piece fails at 3am in CI
    rather than here, which is the wrong end of the feedback loop — and a card
    that quietly has no Tier 2 job at all is the failure `scarf.md` §6.3 and
    `fixmatch.md` §6.3 each recorded about themselves.
    """
    root = Path(__file__).resolve().parents[2]
    for recipe in RECIPES:
        assert (root / "docs" / "recipes" / f"{recipe}.md").is_file(), recipe
        assert (
            root / "xty2" / "evaluation" / "benchmarks" / f"{recipe}.py"
        ).is_file(), recipe
        assert (root / "tests" / "benchmarks" / f"test_{recipe}.py").is_file(), recipe
        assert benchmark_function(recipe) is not None


def test_every_recipe_module_is_in_the_tier2_suite() -> None:
    """And the other direction: a benchmark module nothing runs is not a target."""
    root = Path(__file__).resolve().parents[2]
    modules = {
        path.stem
        for path in (root / "xty2" / "evaluation" / "benchmarks").glob("*.py")
        if not path.stem.startswith("_") and path.stem != "common"
    }
    assert modules == set(RECIPES)


# ---------------------------------------------------------------------------
# FIDELITY.md §3 — a match its own error bars swamp is not a match
# ---------------------------------------------------------------------------


def test_a_target_cleared_by_less_than_one_stderr_does_not_pass() -> None:
    """The rule §3 states in prose, checked rather than trusted.

    `scarf`'s first Tier 2 run is the worked example: a mean 0.00089 inside a
    tolerance whose standard error was 0.038, recorded as `reproduced` because
    `mean <= 1.0` was literally true. §3 had always said such a run "is a
    `deviating` outcome with an explanation, not a `reproduced` one"; nothing
    checked it.
    """
    # Mean 0.999, target 1.0, stderr well above the 0.001 margin.
    noisy = MetricResult.upper_bound("ratio", [0.9, 1.098], 1.0)
    assert noisy.mean == pytest.approx(0.999)
    margin = noisy.margin
    assert margin == pytest.approx(0.001)
    assert margin is not None and noisy.stderr > margin
    assert noisy.within_noise is True
    assert noisy.passed is False


def test_a_target_cleared_by_more_than_one_stderr_passes() -> None:
    clear = MetricResult.upper_bound("ratio", [0.80, 0.84], 1.0)
    margin = clear.margin
    assert margin == pytest.approx(0.18)
    assert margin is not None and clear.stderr < margin
    assert clear.within_noise is False
    assert clear.passed is True


def test_a_deterministic_guardrail_is_untouched_by_the_rule() -> None:
    """`1 +/- 0` is a check, not a statistic, and the rule must not bite it.

    Three of `cycle_dual`'s required metrics and three of `ssdml`'s are exactly
    this shape — out-of-fold provenance, batch immutability, a rejected unsafe
    recipe. A rule that failed them for having no spread would turn a proof
    into a coin toss.
    """
    guardrail = MetricResult.lower_bound("out_of_fold", [1.0, 1.0, 1.0], 1.0)
    assert guardrail.stderr == 0.0
    assert guardrail.margin == pytest.approx(0.0)
    assert guardrail.within_noise is False
    assert guardrail.passed is True


def test_a_miss_and_a_near_miss_read_differently_in_the_explanation() -> None:
    missed = BenchmarkResult(
        recipe="example",
        commit="abc1234",
        date="2026-08-24",
        spec_digest="digest",
        metrics=(MetricResult.upper_bound("error", [0.4, 0.4], 0.2),),
        interpretation="A fixed test interpretation.",
    )
    near = BenchmarkResult(
        recipe="example",
        commit="abc1234",
        date="2026-08-24",
        spec_digest="digest",
        metrics=(MetricResult.upper_bound("error", [0.1, 0.298], 0.2),),
        interpretation="A fixed test interpretation.",
    )
    assert missed.status == "deviating"
    assert near.status == "deviating"
    assert "Failed target(s)" in missed.explanation
    assert "Within noise of the target" not in missed.explanation
    assert "Within noise of the target" in near.explanation
    assert "Failed target(s)" not in near.explanation


def test_an_interval_measures_its_margin_to_the_nearer_edge() -> None:
    """A mean pressed against either edge is equally uninformative."""
    # Mean 0.80 sits 0.02 above the lower edge and 0.18 below the upper one,
    # so the nearer edge is what decides.
    tight = MetricResult.interval("value", [0.75, 0.85], 0.78, 0.98)
    assert tight.margin == pytest.approx(0.02)
    assert tight.stderr == pytest.approx(0.05)
    assert tight.within_noise is True
    assert tight.passed is False

    # The same mean and the same edges, measured precisely enough to clear.
    settled = MetricResult.interval("value", [0.795, 0.805], 0.78, 0.98)
    assert settled.margin == pytest.approx(0.02)
    assert settled.stderr == pytest.approx(0.005)
    assert settled.passed is True
