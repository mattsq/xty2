"""Independent equation, autograd, topology, pairing and plan oracles."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from torch import Tensor, nn
from xty2.core import (
    CompiledRun,
    GraphError,
    LossError,
    Port,
    State,
    TrainContext,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    SEPARATED,
    continuous_schema,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.reporting import MetricResult
from xty2.evaluation.simsiam_study import arm_recipe, embedding_metrics, require_equal
from xty2.evaluation.vicreg_views import OracleSymmetry
from xty2.objectives import CosineFeatureConsistency
from xty2.recipes import simsiam
from xty2.recipes.simsiam import CORRUPTED_A as A
from xty2.recipes.simsiam import CORRUPTED_B as B
from xty2.recipes.simsiam import DATA_POLICY
from xty2.training.loading import build_population
from xty2.training.loss_mixer import LossMixer

from tests.invariants import test_doublematch as card_parser


def recipe_run() -> CompiledRun:
    return compile(
        simsiam(
            continuous_schema(6),
            first_transforms=(OracleSymmetry(),),
            second_transforms=(OracleSymmetry(),),
        )
    )


def objective(
    reverse: bool = False, stop: Literal["target", "none"] = "target"
) -> CosineFeatureConsistency:
    return CosineFeatureConsistency(
        prediction_port=Port.X_PRED,
        target_port=Port.X_PROJ,
        prediction=B if reverse else A,
        target=A if reverse else B,
        stop_grad=stop,
    )


def tensors(scale: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    base = torch.arange(1, 22, dtype=torch.float64).reshape(7, 3)
    return tuple(
        (torch.sin(base * c) * scale).requires_grad_() for c in (0.4, 0.7, 1.1, 1.7)
    )  # type: ignore[return-value]


def scalar_cosine(p: Tensor, z: Tensor) -> Tensor:
    terms = []
    for row in range(7):
        pn = sum(p[row, j] ** 2 for j in range(3)).sqrt().clamp_min(1e-12)
        zn = sum(z[row, j] ** 2 for j in range(3)).sqrt().clamp_min(1e-12)
        terms.append(-sum(p[row, j] * z[row, j] for j in range(3)) / (pn * zn))
    return torch.stack(terms).mean()


@pytest.mark.parametrize("scale", [1.0, 1e-14])
@pytest.mark.parametrize("stop", ["target", "none"])
def test_directional_values_and_gradients(
    scale: float, stop: Literal["target", "none"]
) -> None:
    pa, za, pb, zb = tensors(scale)
    state = State(
        {A: {Port.X_PRED: pa, Port.X_PROJ: za}, B: {Port.X_PRED: pb, Port.X_PROJ: zb}}
    )
    batch = two_cluster_population(7, seed=1, row_offset=0, low=SEPARATED).batch
    ctx = TrainContext(global_step=0, schema=continuous_schema(6))
    loss = 0.5 * objective(stop=stop).compute(state, batch, torch.arange(7), ctx).value
    loss = (
        loss
        + 0.5 * objective(True, stop).compute(state, batch, torch.arange(7), ctx).value
    )
    expected = (
        scalar_cosine(pa, zb.detach() if stop == "target" else zb)
        + scalar_cosine(pb, za.detach() if stop == "target" else za)
    ) / 2
    torch.testing.assert_close(loss, expected)
    actual = torch.autograd.grad(loss, (pa, za, pb, zb), allow_unused=True)
    oracle = torch.autograd.grad(expected, (pa, za, pb, zb), allow_unused=True)
    for i, (got, want) in enumerate(zip(actual, oracle, strict=True)):
        if stop == "target" and i in (1, 3):
            assert got is want is None
        else:
            assert got is not None
            assert want is not None
            assert bool(got.ne(0).any())
            torch.testing.assert_close(got, want)
    assert bool(objective(stop=stop).detaches) == (stop == "target")


def test_no_predictor_half_gradient() -> None:
    a, b, _, _ = tensors(1.0)
    state = State({A: {Port.X_PROJ: a}, B: {Port.X_PROJ: b}})
    batch = two_cluster_population(7, seed=1, row_offset=0, low=SEPARATED).batch
    ctx = TrainContext(global_step=0, schema=continuous_schema(6))
    loss = sum(
        0.5
        * replace(objective(reverse), prediction_port=Port.X_PROJ)
        .compute(state, batch, torch.arange(7), ctx)
        .value
        for reverse in (False, True)
    )
    assert isinstance(loss, Tensor)
    gradients = torch.autograd.grad(loss, (a, b))
    reference = torch.autograd.grad(scalar_cosine(a, b), (a, b))
    for got, want in zip(gradients, reference, strict=True):
        torch.testing.assert_close(got, want / 2)


@pytest.mark.parametrize(
    "bad",
    [
        torch.ones(7),
        torch.ones(7, 0),
        torch.full((7, 3), float("nan")),
        torch.full((7, 3), float("inf")),
    ],
)
def test_bad_embeddings_rejected(bad: Tensor) -> None:
    state = State({A: {Port.X_PRED: bad}, B: {Port.X_PROJ: torch.ones(7, 3)}})
    with pytest.raises(LossError):
        objective().compute(
            state,
            two_cluster_population(7, seed=1, row_offset=0, low=SEPARATED).batch,
            torch.arange(7),
            TrainContext(global_step=0, schema=continuous_schema(6)),
        )


def test_source_topology_and_fixed_bias() -> None:
    run = recipe_run()
    projector = run.graph["simsiam_projector"].network
    predictor = run.graph["simsiam_predictor"].network
    assert [type(m) for m in projector] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Linear,
        nn.BatchNorm1d,
    ]
    assert [type(m) for m in predictor] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Linear,
    ]
    assert projector[0].bias is None and projector[3].bias is None
    assert not projector[6].bias.requires_grad
    assert bool(projector[6].bias.ne(0).any())
    assert "network.6.bias" in dict(run.graph["simsiam_projector"].named_buffers())
    assert not projector[7].affine
    assert predictor[0].bias is None and predictor[3].bias is not None
    assert [
        (m.in_features, m.out_features) for m in projector if isinstance(m, nn.Linear)
    ] == [(256, 256)] * 3
    assert [
        (m.in_features, m.out_features) for m in predictor if isinstance(m, nn.Linear)
    ] == [(256, 64), (64, 256)]
    for bn in [m for m in run.graph.modules() if isinstance(m, nn.BatchNorm1d)]:
        assert bn.eps == 1e-5 and bn.momentum == 0.1 and bn.track_running_stats


def test_two_cached_bn_updates_and_encoder_branch_gradients() -> None:
    run = recipe_run()
    schema = run.recipe.schema
    data = training_dataset(
        schema, two_cluster_population(128, seed=5, row_offset=0, low=SEPARATED).batch
    )
    population = build_population(data, DATA_POLICY, seed=6)
    projector = run.graph["simsiam_projector"].network
    run.graph.train()
    state = run.state(run.stages[0], population.rows, rng_key=8, population=population)
    for bn in [m for m in run.graph.modules() if isinstance(m, nn.BatchNorm1d)]:
        assert bn.num_batches_tracked is not None
        assert int(bn.num_batches_tracked) == 2
    expected_mean = torch.zeros(256)
    expected_var = torch.ones(256)
    for view in run.recipe.views:
        batch = view.apply(population.rows, schema, rng_key=8, population=population)
        features = run.graph.evaluate(batch, schema=schema, only=("mlp_encoder",))[
            Port.X_REPR
        ]
        assert isinstance(features, Tensor)
        hidden = projector[0](features)
        expected_mean = 0.9 * expected_mean + 0.1 * hidden.mean(0)
        expected_var = 0.9 * expected_var + 0.1 * hidden.var(0, correction=1)
    torch.testing.assert_close(projector[1].running_mean, expected_mean)
    torch.testing.assert_close(projector[1].running_var, expected_var)
    mixed = LossMixer.for_stage(run.stages[0]).mix(
        state, population.rows, TrainContext(global_step=0, schema=schema)
    )
    za, zb = state[A][Port.X_PROJ], state[B][Port.X_PROJ]
    assert isinstance(za, Tensor) and isinstance(zb, Tensor)
    grads = torch.autograd.grad(mixed.total, (za, zb), retain_graph=True)
    assert all(bool(g.ne(0).any()) for g in grads)
    parameters = tuple(run.graph["mlp_encoder"].parameters())
    assert any(
        bool(g.ne(0).any()) for g in torch.autograd.grad(mixed.total, parameters)
    )
    for bn in [m for m in run.graph.modules() if isinstance(m, nn.BatchNorm1d)]:
        assert bn.num_batches_tracked is not None
        assert int(bn.num_batches_tracked) == 2
    with pytest.raises(GraphError, match="at least two"):
        run.graph.evaluate(
            population.rows.replace(
                x=population.rows.x[:1],
                t=population.rows.t[:1],
                y=population.rows.y[:1],
                t_observed=population.rows.t_observed[:1],
                y_observed=population.rows.y_observed[:1],
                row_id=population.rows.row_id[:1],
            ),
            schema=schema,
            only=("mlp_encoder", "simsiam_projector"),
        )


def test_plan_and_ablation_contracts() -> None:
    run = recipe_run()
    plan = run.plan.hyperparameters
    assert plan["optimisation.total_steps_or_epochs"] == {
        "pretrain": 1000,
        "joint_fit": 3000,
    }
    assert plan["optimisation.batch_size"] == {"pretrain": 128, "joint_fit": 128}
    assert plan["losses.weights"]["pretrain.simsiam_a_to_b"] == 0.5
    assert plan["losses.weights"]["pretrain.simsiam_b_to_a"] == 0.5
    assert "epsilon=1e-12" in run.plan.render()
    assert not run.graph.port_depends_on_raw_outcome(Port.X_PRED)
    for arm in ("full", "no_stop", "no_predictor", "no_pretrain"):
        control = compile(arm_recipe(run.recipe, arm))
        if arm == "no_predictor":
            assert "simsiam_predictor" not in control.graph.names
            assert "simsiam_predictor" not in control.stages[0].trainable
        elif arm == "no_stop":
            assert all(
                not term.objective.detaches
                for term in control.recipe.program[0].objectives
            )
        elif arm == "no_pretrain":
            assert (
                len(control.stages) == 1 and control.stages[0].initialise_from is None
            )


def test_spread_rank_and_strict_boundary() -> None:
    constant = embedding_metrics(torch.ones(8, 4))
    assert constant["spread"] == 0 and constant["effective_rank"] == 0
    isotropic = embedding_metrics(torch.cat((torch.eye(4), -torch.eye(4))))
    assert isotropic["spread"] == pytest.approx(1.0)
    assert isotropic["effective_rank"] == pytest.approx(4.0)
    assert MetricResult("gap", (0.0, 0.0), ">", 0.0).passed is False
    assert MetricResult("gap", (0.0, 2.0), ">", 0.0).passed is False
    assert MetricResult("gap", (1.0, 1.0), ">", 0.0).passed is True
    assert math.isfinite(isotropic["raw_norm"])


def test_mask_swap_mutant_is_rejected() -> None:
    mask = torch.tensor([True, False, True, False])
    with pytest.raises(RuntimeError, match="mismatch"):
        require_equal({"t_observed": mask}, {"t_observed": mask.roll(1)})


def test_every_answered_card_value_matches_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        card_parser, "CARD", Path(__file__).parents[2] / "docs/recipes/simsiam.md"
    )
    answers = card_parser._card_section_four()
    plan = recipe_run().plan.hyperparameters
    checked = 0
    for key, answer in answers.items():
        assert key in plan, key
        actual = plan[key]
        scopes = answer if isinstance(answer, dict) else {"": answer}
        for scope, wanted in scopes.items():
            if key == "architecture.widths_depths":
                wanted = wanted.replace("X_REPR", "256").replace("K", "2")
            wanted = wanted.strip('"')
            values = (
                ([actual[scope]] if scope else list(actual.values()))
                if isinstance(actual, dict)
                else [actual]
            )
            for value in values:
                assert card_parser._rendered(value) == wanted, (
                    key,
                    scope,
                    value,
                    wanted,
                )
                checked += 1
    assert checked >= 70


@pytest.mark.parametrize(
    "mutant", ["remove_target_detach", "detach_prediction", "swap_target_view"]
)
def test_cosine_oracles_kill_mutants(
    mutant: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = CosineFeatureConsistency.compute

    def altered(self: CosineFeatureConsistency, state: State, *args: Any) -> Any:
        if mutant == "remove_target_detach":
            return original(replace(self, stop_grad="none"), state, *args)
        if mutant == "swap_target_view":
            return original(replace(self, target=self.prediction), state, *args)
        values = {
            view: {
                port: value.detach()
                if port == Port.X_PRED and isinstance(value, Tensor)
                else value
                for port, value in state[view].items()
            }
            for view in (A, B)
        }
        return original(self, State(values), *args)

    monkeypatch.setattr(CosineFeatureConsistency, "compute", altered)
    with pytest.raises((AssertionError, RuntimeError)):
        test_directional_values_and_gradients(1.0, "target")


@pytest.mark.parametrize("direction", [0, 1])
def test_plan_oracle_kills_missing_half_weight(
    direction: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = recipe_run()
    pretrain, fit = run.recipe.program
    terms = list(pretrain.objectives)
    terms[direction] = replace(terms[direction], weight=1.0)
    from xty2.core import Program

    altered = compile(
        replace(
            run.recipe,
            program=Program((replace(pretrain, objectives=tuple(terms)), fit)),
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "recipe_run", lambda: altered)
    with pytest.raises(AssertionError):
        test_plan_and_ablation_contracts()


def test_topology_oracle_kills_affine_output_bn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = recipe_run()
    run.graph["simsiam_projector"].network[7].affine = True
    monkeypatch.setattr(sys.modules[__name__], "recipe_run", lambda: run)
    with pytest.raises(AssertionError):
        test_source_topology_and_fixed_bias()


def test_ablation_oracle_kills_retained_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = arm_recipe
    monkeypatch.setattr(
        sys.modules[__name__],
        "arm_recipe",
        lambda recipe, arm: recipe if arm == "no_predictor" else original(recipe, arm),
    )
    with pytest.raises(AssertionError):
        test_plan_and_ablation_contracts()


def test_bn_oracle_kills_concatenated_views(monkeypatch: pytest.MonkeyPatch) -> None:
    def concatenated(
        self: CompiledRun, stage: object, batch: XTYBatch, **kwargs: Any
    ) -> State:
        del stage
        a, b = [
            view.apply(
                batch,
                self.recipe.schema,
                rng_key=kwargs["rng_key"],
                population=kwargs["population"],
            )
            for view in self.recipe.views
        ]
        joined = XTYBatch(
            x=torch.cat((a.x, b.x)),
            t=torch.cat((a.t, b.t)),
            y=torch.cat((a.y, b.y)),
            t_observed=torch.cat((a.t_observed, b.t_observed)),
            y_observed=torch.cat((a.y_observed, b.y_observed)),
            row_id=torch.arange(2 * batch.batch_size),
        )
        values = self.graph.evaluate(
            joined, schema=self.recipe.schema, only=self.stages[0].trainable
        )
        return State(
            {
                view: {
                    port: value[
                        index * batch.batch_size : (index + 1) * batch.batch_size
                    ]
                    for port, value in values.items()
                    if isinstance(value, Tensor)
                }
                for index, view in enumerate((A, B))
            }
        )

    monkeypatch.setattr(CompiledRun, "state", concatenated)
    with pytest.raises(AssertionError):
        test_two_cached_bn_updates_and_encoder_branch_gradients()
