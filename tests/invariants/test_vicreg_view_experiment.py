"""Check the experimental symmetry against independent DGP expressions."""

from dataclasses import replace

import pytest
import torch
from experiments.vicreg_views import OracleSymmetry, fit_with_trace, policy_recipe
from xty2.core import DataSpec, Program, compile
from xty2.evaluation.benchmarks.common import (
    continuous_schema,
    on_the_training_scale,
    training_dataset,
    two_cluster_population,
)
from xty2.training.loading import build_population


@pytest.mark.parametrize("seed", [191, 293, 397])
def test_symmetry_preserves_all_dgp_targets_and_changes_inputs(seed: int) -> None:
    schema = continuous_schema(6)
    train = two_cluster_population(1024, seed=seed, row_offset=0)
    data = training_dataset(schema, train.batch)
    recipe = policy_recipe(schema, "oracle")
    assert isinstance(recipe.data, DataSpec)
    population = build_population(data, recipe.data, seed=seed)
    test = two_cluster_population(2048, seed=seed + 1, row_offset=10000)
    batch = on_the_training_scale(test.batch, population)
    snapshot = batch.x.clone()
    view = OracleSymmetry()
    first = view.apply(
        batch,
        schema,
        population=population,
        generator=torch.Generator().manual_seed(seed),
    )
    second = view.apply(
        batch,
        schema,
        population=population,
        generator=torch.Generator().manual_seed(seed + 1),
    )
    original = (
        batch.x * population.statistics["x_scale"] + population.statistics["x_location"]
    )
    transformed = (
        first.x * population.statistics["x_scale"] + population.statistics["x_location"]
    )
    # Independent sufficient statistics, not the transform's `targets` helper.
    torch.testing.assert_close(
        original[:, :4].sum(-1), transformed[:, :4].sum(-1), atol=3e-6, rtol=0
    )
    torch.testing.assert_close(
        0.5 * original[:, 0] - 0.3 * original[:, 1],
        0.5 * transformed[:, 0] - 0.3 * transformed[:, 1],
        atol=2e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        original[:, 4].square(), transformed[:, 4].square(), atol=5e-6, rtol=0
    )
    assert torch.equal(first.x[:, 2], batch.x[:, 2])
    assert torch.equal(batch.x, snapshot)
    assert torch.equal(first.y, batch.y) and torch.equal(first.t, batch.t)
    assert torch.isin(first.x[:, 5], population.rows.x[:, 5]).all()
    assert (first.x != second.x).any(-1).float().mean() > 0.99
    assert 2.8 < float((first.x != batch.x).float().sum(-1).mean()) < 3.2
    # The first reflection preserves distance to either mixture centre.
    for centre in (-0.45, 0.45):
        torch.testing.assert_close(
            (original[:, :4] - centre).square().sum(-1),
            (transformed[:, :4] - centre).square().sum(-1),
            atol=6e-6,
            rtol=1e-6,
        )


def test_actual_draw_trace_is_paired_and_counts_two_draws_per_step() -> None:
    torch.set_num_threads(1)
    schema = continuous_schema(6)
    train = two_cluster_population(1024, seed=191, row_offset=0)
    data = training_dataset(schema, train.batch)
    traces = []
    for policy in ("marginal", "oracle"):
        for coefficient in (25.0, 0.0):
            torch.manual_seed(197)
            recipe = policy_recipe(schema, policy)
            pretrain = recipe.program[0]
            pretrain = replace(
                pretrain,
                steps=2,
                objectives=tuple(
                    replace(term, weight=coefficient)
                    if term.objective.name == "embedding_variance"
                    else term
                    for term in pretrain.objectives
                ),
            )
            run = compile(replace(recipe, program=Program((pretrain,))))
            _, trace = fit_with_trace(run, {"pretrain": data}, 10191)
            assert trace["calls"] == "4"
            traces.append(trace)
    assert traces[0] == traces[1]
    assert traces[2] == traces[3]
    assert traces[0]["rows"] == traces[2]["rows"]
    assert traces[0]["views"] != traces[2]["views"]
