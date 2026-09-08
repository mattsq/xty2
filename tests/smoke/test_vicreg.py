"""The card's three-seed, four-arm Tier 1 study, without directional gates."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

import pytest
import torch
from torch import Tensor
from xty2.core import (
    CategoricalTreatment,
    ComponentGraph,
    GaussianOutcome,
    Port,
    Program,
    ViewSpec,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    SEPARATED,
    continuous_schema,
    on_the_training_scale,
    take,
    training_dataset,
    two_cluster_population,
)
from xty2.recipes import vicreg
from xty2.training import STREAM_STRIDE, executors, run_program
from xty2.views import FeatureCorruption


@pytest.fixture(scope="module", autouse=True)
def _one_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _snapshot(graph: ComponentGraph) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in graph.state_dict().items()}


@pytest.mark.parametrize("base", [191, 293, 397])
def test_paired_mechanism_study(base: int, monkeypatch: pytest.MonkeyPatch) -> None:
    schema = continuous_schema(6)
    train = two_cluster_population(1024, seed=base + 1, row_offset=0, low=SEPARATED)
    test = two_cluster_population(2048, seed=base + 2, row_offset=10000, low=SEPARATED)
    data = training_dataset(schema, train.batch)
    paired_rows: dict[str, list[Tensor]] = {}
    paired_views: list[Tensor] = []
    initial: dict[str, Tensor] | None = None
    report: dict[str, dict[str, float]] = {}
    rows: dict[str, list[Tensor]] = {}
    views: list[Tensor] = []
    transition: dict[str, dict[str, Tensor]] = {}
    optimisers: list[torch.optim.Optimizer] = []
    current_stage = ""
    original_step = executors._step
    original_view = ViewSpec.apply
    original_evaluate = ComponentGraph.evaluate

    for arm in ("full", "no_variance", "no_covariance", "no_pretrain"):
        torch.manual_seed(base + 6)
        recipe = vicreg(schema)
        pretrain, fit = recipe.program
        pretrain = replace(pretrain, steps=200)
        fit = replace(fit, steps=300)
        if arm in ("no_variance", "no_covariance"):
            target = "embedding_" + arm.removeprefix("no_")
            pretrain = replace(
                pretrain,
                objectives=tuple(
                    replace(term, weight=0.0) if term.objective.name == target else term
                    for term in pretrain.objectives
                ),
            )
        stages = (
            (replace(fit, initialise_from=None),)
            if arm == "no_pretrain"
            else (pretrain, fit)
        )
        run = compile(replace(recipe, program=Program(stages)))
        start = _snapshot(run.graph)
        if initial is None:
            initial = start
        assert all(torch.equal(value, initial[name]) for name, value in start.items())
        rows.clear()
        rows.update({stage.name: [] for stage in stages})
        views.clear()
        transition.clear()
        optimisers.clear()

        def traced_step(*args: Any, **kwargs: Any) -> Any:
            nonlocal current_stage
            active_run, compiled, batch, _, optimiser, _, step = args
            current_stage = compiled.name
            rows[current_stage].append(batch.row_id.clone())
            if step == 0:
                transition[current_stage] = _snapshot(active_run.graph)
                assert not optimiser.state
                assert all(optimiser is not previous for previous in optimisers)
                optimisers.append(optimiser)
            return original_step(*args, **kwargs)

        def traced_view(*args: Any, **kwargs: Any) -> Any:
            assert current_stage == "pretrain"
            population = kwargs["population"]
            assert population is not None
            assert torch.equal(population.rows.row_id, train.batch.row_id)
            result = original_view(*args, **kwargs)
            views.append(result.x.clone())
            return result

        def traced_evaluate(*args: Any, **kwargs: Any) -> Any:
            names = tuple(kwargs["only"])
            if current_stage == "pretrain":
                assert "tarnet_head" not in names
                assert "categorical_propensity" not in names
            else:
                assert "vicreg_expander" not in names
            return original_evaluate(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(executors, "_step", traced_step)
            patch.setattr(ViewSpec, "apply", traced_view)
            patch.setattr(ComponentGraph, "evaluate", traced_evaluate)
            result = run_program(
                run,
                {stage.name: data for stage in stages},
                seed=base + 10000 + (STREAM_STRIDE if arm == "no_pretrain" else 0),
            )

        for stage in result.stages:
            assert all(
                math.isfinite(r.total) and math.isfinite(r.grad_norm)
                for r in stage.records
            )
            assert all(
                math.isfinite(term.value) for r in stage.records for term in r.terms
            )
            assert stage.population is not None
            assert int(stage.population.rows.t_observed.sum()) == 40
            assert torch.equal(stage.population.rows.row_id, train.batch.row_id)
            torch.testing.assert_close(
                stage.population.statistics["x_location"], train.batch.x.mean(0)
            )
            assert set(stage.checkpoint.trained_on_row_ids.tolist()) <= set(range(1024))
            if arm == "full":
                paired_rows[stage.stage] = rows[stage.stage]
            else:
                assert len(rows[stage.stage]) == len(paired_rows[stage.stage])
                assert all(
                    torch.equal(a, b)
                    for a, b in zip(
                        rows[stage.stage], paired_rows[stage.stage], strict=True
                    )
                )

        assert len(views) == (0 if arm == "no_pretrain" else 400)
        if arm == "full":
            paired_views = list(views)
        elif arm != "no_pretrain":
            assert all(
                torch.equal(a, b) for a, b in zip(views, paired_views, strict=True)
            )
        fit_start = transition["joint_fit"]
        for name, value in start.items():
            if "tarnet_head." in name or "categorical_propensity." in name:
                assert torch.equal(value, fit_start[name])
        final = _snapshot(run.graph)
        assert any(
            not torch.equal(value, final[name])
            for name, value in fit_start.items()
            if "mlp_encoder." in name
        )
        for name, value in fit_start.items():
            if "vicreg_expander." in name:
                assert torch.equal(value, final[name])

        population = result.stage("joint_fit").population
        assert population is not None
        heldout = on_the_training_scale(test.batch, population)
        run.graph.eval()
        with torch.no_grad():
            predictions = run.graph.evaluate(
                heldout,
                schema=schema,
                only=("mlp_encoder", "tarnet_head", "categorical_propensity"),
            )
            propensity = predictions[Port.T_GIVEN_X]
            outcome = predictions[Port.Y_GIVEN_XT]
            assert isinstance(propensity, CategoricalTreatment)
            assert isinstance(outcome, GaussianOutcome)
            probabilities = propensity.log_probs.exp()
            assert bool(torch.isfinite(probabilities).all())
            torch.testing.assert_close(probabilities.sum(-1), torch.ones(2048))
            metrics = {
                "treatment_nll": float(-propensity.log_prob(heldout.t).mean()),
                "outcome_nll": float(-outcome.log_prob(heldout.y, heldout.t).mean()),
            }
            if arm != "no_pretrain":
                checkpoint = result.stage("pretrain").checkpoint
                for name, value in checkpoint.parameters.items():
                    assert torch.equal(value, fit_start["_components." + name])
                assert any(
                    not torch.equal(value, start["_components." + name])
                    for name, value in checkpoint.parameters.items()
                    if name.startswith("mlp_encoder.")
                )
                # Restore the immutable terminal pretraining state for evaluation.
                saved = {**checkpoint.parameters, **checkpoint.buffers}
                state = run.graph.state_dict()
                for name, value in saved.items():
                    state["_components." + name].copy_(value)
                before = _snapshot(run.graph)
                spread, redundancy = [], []
                for b in range(16):
                    batch = take(heldout, torch.arange(b * 128, (b + 1) * 128))
                    for branch in range(2):
                        view = FeatureCorruption(rate=0.6, columns=None).apply(
                            batch,
                            schema,
                            population=population,
                            generator=torch.Generator().manual_seed(
                                base + 20000 + 2 * b + branch
                            ),
                        )
                        z = run.graph.evaluate(
                            view, schema=schema, only=("mlp_encoder", "vicreg_expander")
                        )[Port.X_PROJ]
                        assert isinstance(z, Tensor)
                        centred = z - z.mean(0)
                        covariance = centred.T @ centred / 127
                        diagonal = covariance.diagonal().square().sum()
                        assert float(diagonal) > 0
                        off = (
                            covariance.square()
                            .masked_select(~torch.eye(z.shape[1], dtype=torch.bool))
                            .sum()
                        )
                        spread.append(
                            float((z.var(0, correction=1) + 1e-4).sqrt().mean())
                        )
                        redundancy.append(float(off / diagonal))
                metrics.update(spread=sum(spread) / 32, redundancy=sum(redundancy) / 32)
                assert all(
                    torch.equal(value, run.graph.state_dict()[name])
                    for name, value in before.items()
                )
            else:
                assert all(
                    torch.equal(value, fit_start[name]) for name, value in start.items()
                )
        assert all(math.isfinite(value) for value in metrics.values())
        report[arm] = metrics
    print(json.dumps({"base": base, "arms": report}, sort_keys=True))
