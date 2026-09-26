"""VIME's Tier 2 instruments: bound to the card, paired, and scored correctly."""

from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

import pytest
import torch
from xty2.core import (
    Constant,
    Dataset,
    DataSpec,
    MissingnessSpec,
    PreprocessSpec,
    SplitSpec,
    TrainingPopulation,
)
from xty2.evaluation.benchmarks import scarf as scarf_benchmark
from xty2.evaluation.benchmarks import vime as benchmark
from xty2.evaluation.benchmarks.common import (
    continuous_schema,
    take,
    two_cluster_population,
)
from xty2.evaluation.reporting import card_status, load_reproduction_spec
from xty2.recipes import vime
from xty2.recipes.vime import DATA_POLICY, MASK_PROBABILITY
from xty2.training import build_population
from xty2.views import BernoulliMarginalCorruption

ROOT = Path(__file__).parents[2]
CARD = ROOT / "docs/recipes/vime.md"
RESULT = ROOT / "docs/experiments/results/vime-46d3e4c/vime.json"


def test_the_benchmark_binds_every_section6_scalar_by_value() -> None:
    spec = load_reproduction_spec(CARD, recipe="vime")
    spec.bind(benchmark.PROTOCOL, documentation=benchmark.DOCUMENTATION)
    assert spec.seed_count == 10


def test_the_fixture_is_the_shared_generator_with_its_declared_changes() -> None:
    """Card §6.1: the fixmatch generator, noise 0.2, 16,384 rows, own stream."""
    imported = vars(benchmark)
    assert imported["two_cluster_population"] is two_cluster_population
    assert imported["held_out_nll"] is scarf_benchmark.held_out_nll
    assert imported["unpretrained"] is scarf_benchmark.unpretrained
    world = benchmark.fixture(3)
    assert world.base == 190_000 + 300
    assert world.initial_state_seed == world.base + 6
    assert world.run_seed == world.base + 10_000
    expected = two_cluster_population(
        16_384, seed=world.base + 1, row_offset=0, noise=0.2
    )
    assert torch.equal(world.train.batch.x, expected.batch.x)
    assert torch.equal(world.train.batch.t, expected.batch.t)
    held_out = two_cluster_population(
        2_048, seed=world.base + 2, row_offset=100_000, noise=0.2
    )
    assert torch.equal(world.test.batch.x, held_out.batch.x)
    assert int(world.train.batch.row_id.max()) < int(world.test.batch.row_id.min())
    # The noise shrinks only the signal columns' deviations from their centres:
    # the same seed at the default noise differs there and nowhere else.
    default = two_cluster_population(16_384, seed=world.base + 1, row_offset=0)
    assert not torch.equal(world.train.batch.x[:, :4], default.batch.x[:, :4])
    assert torch.equal(world.train.batch.x[:, 4:], default.batch.x[:, 4:])
    centred = world.train.batch.x[:, :4].abs() - 0.45
    assert float(centred.std()) == pytest.approx(0.2, rel=0.05)


def test_the_bayes_reference_sits_where_section_6_1_puts_it() -> None:
    """The dependent-block optimum near 0.49, the independent one near 1.49."""
    world = benchmark.fixture(0)
    population = build_population(world.data, DATA_POLICY, seed=0)
    corrupted, clean = benchmark._held_out_corruption(world, population)
    optimum = benchmark.bayes_pretext(corrupted, clean, population)
    assert 0.40 < optimum["dependent_ratio"] < 0.60
    assert 1.30 < optimum["independent_ratio"] < 1.70
    assert 0.65 < optimum["mask_auroc"] < 0.75


def test_auroc_matches_the_pairwise_definition_with_ties() -> None:
    scores = torch.tensor([0.1, 0.4, 0.4, 0.4, 0.8, 0.2, 0.8, 0.1])
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.bool)
    wins = 0.0
    pairs = 0
    for positive, negative in itertools.product(
        scores[labels].tolist(), scores[~labels].tolist()
    ):
        wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
        pairs += 1
    assert benchmark.auroc(scores, labels) == pytest.approx(wins / pairs, abs=1e-12)
    assert benchmark.auroc(-scores, labels) == pytest.approx(1 - wins / pairs)
    with pytest.raises(ValueError, match="positive and one negative"):
        benchmark.auroc(scores, torch.ones(8, dtype=torch.bool))


def test_block_ratios_read_only_the_cells_that_changed() -> None:
    """The copy predictor scores 2, the mean scores 1, and truth scores 0.

    The Bayes-optimal Eq. 6 predictor for an independent column,
    `(1 - p) x~ + p mu`, scores `1 + (1 - p)^2` on corrupted cells: the section
    6 audit's closed form, checked here by simulation.
    """
    generator = torch.Generator().manual_seed(0)
    rows, p = 200_000, MASK_PROBABILITY
    clean = torch.randn(rows, 6, generator=generator)
    donors = torch.randn(rows, 6, generator=generator)
    changed = torch.rand(rows, 6, generator=generator) < p
    corrupted = torch.where(changed, donors, clean)
    mean = torch.zeros(rows, 6)
    columns = benchmark.INDEPENDENT_BLOCK

    def ratio(prediction: torch.Tensor) -> float:
        return benchmark._block_ratio(prediction, mean, clean, changed, columns)

    assert ratio(clean) == 0.0
    assert ratio(mean) == pytest.approx(1.0)
    assert ratio(corrupted) == pytest.approx(2.0, rel=0.02)
    assert ratio((1 - p) * corrupted + p * mean) == pytest.approx(
        1 + (1 - p) ** 2, rel=0.02
    )
    with pytest.raises(RuntimeError, match="ratio is undefined"):
        benchmark._block_ratio(clean, mean, clean, torch.zeros_like(changed), columns)


def _population() -> tuple[Dataset, TrainingPopulation]:
    schema = continuous_schema(6)
    train = two_cluster_population(64, seed=5, row_offset=100)
    dataset = Dataset(
        schema=schema,
        rows=train.batch,
        assignments={"train": torch.arange(64)},
    )
    policy = DataSpec(
        split=SplitSpec(protocol="test", train="train"),
        preprocess=PreprocessSpec(features="minmax", outcome="zscore"),
        missingness=MissingnessSpec(mechanism="mcar", observed=10),
    )
    return dataset, build_population(dataset, policy, seed=1)


def test_the_fixed_draw_is_one_table_looked_up_by_row_id() -> None:
    dataset, population = _population()
    schema = dataset.schema
    transform = benchmark.FixedDrawMarginalCorruption(p=MASK_PROBABILITY, seed=11)
    rows = population.rows
    batch = take(rows, torch.tensor([7, 3, 40, 21, 0]))
    first = transform.apply(
        batch,
        schema,
        generator=torch.Generator().manual_seed(1),
        population=population,
    )
    second = transform.apply(
        batch,
        schema,
        generator=torch.Generator().manual_seed(2),
        population=population,
    )
    assert torch.equal(first.x, second.x)
    table = BernoulliMarginalCorruption(p=MASK_PROBABILITY).apply(
        rows, schema, generator=torch.Generator().manual_seed(11), population=population
    )
    assert torch.equal(first.x, table.x[torch.tensor([7, 3, 40, 21, 0])])
    assert not torch.equal(first.x, batch.x)
    stranger = batch.replace(row_id=batch.row_id + 10_000)
    with pytest.raises(ValueError, match="not exactly one"):
        transform.apply(
            stranger,
            schema,
            generator=torch.Generator().manual_seed(1),
            population=population,
        )


def test_an_ablation_silences_exactly_one_pretext_term() -> None:
    recipe = vime(continuous_schema(6))
    silenced = benchmark._reweighted(recipe, benchmark._RECONSTRUCTION_TERM)
    before = {term.objective.name: term for term in recipe.program[0].objectives}
    after = {term.objective.name: term for term in silenced.program[0].objectives}
    assert after[benchmark._MASK_TERM] == before[benchmark._MASK_TERM]
    assert after[benchmark._RECONSTRUCTION_TERM].weight == Constant(0.0)
    assert before[benchmark._RECONSTRUCTION_TERM].weight == Constant(2.0)
    assert silenced.program[1] == recipe.program[1]
    with pytest.raises(ValueError, match="no objective named"):
        benchmark._reweighted(recipe, "no_such_term")


def test_the_recorded_evidence_recomputes_and_still_describes_the_card() -> None:
    """The saved replicates give the ledger's decisions under today's card.

    The digest check is what stops a §6 amendment from inheriting this run:
    change any scalar of the reproduction block and the recorded evidence no
    longer describes the card.
    """
    result = json.loads(RESULT.read_text())
    spec = load_reproduction_spec(CARD, recipe="vime")
    assert result["spec_digest"] == spec.digest
    assert result["replicates"] == spec.seed_count == 10
    assert card_status(CARD.read_text()) == result["status"] == "reproduced"
    decided = []
    for metric in result["metrics"]:
        values = metric["values"]
        assert len(values) == 10 and all(math.isfinite(v) for v in values)
        mean = statistics.mean(values)
        stderr = statistics.stdev(values) / math.sqrt(10)
        assert metric["mean"] == pytest.approx(mean, abs=1e-12)
        assert metric["stderr"] == pytest.approx(stderr, abs=1e-12)
        if metric["relation"] == "<":
            decided.append(metric["passed"] == (mean + stderr < metric["target"]))
        elif metric["relation"] == "<=":
            decided.append(metric["passed"] == (mean + stderr <= metric["target"]))
        elif metric["relation"] == ">=":
            decided.append(metric["passed"] == (mean - stderr >= metric["target"]))
        else:
            assert metric["relation"] == "info" and metric["passed"] is None
    assert decided == [True, True, True]
    required = {
        m["name"]: m["passed"] for m in result["metrics"] if m["passed"] is not None
    }
    assert required == {
        "dependent_block_reconstruction_ratio": True,
        "independent_block_reconstruction_ratio": True,
        "held_out_outcome_NLL_ratio": True,
    }
