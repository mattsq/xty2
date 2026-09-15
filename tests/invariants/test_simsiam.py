"""Independent equation, autograd, topology, pairing and plan oracles."""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import pytest
import torch
from torch import Tensor, nn
from xty2.components import MLPEncoder, SimSiamPredictor, SimSiamProjector
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    CompiledRun,
    CosineAnneal,
    GraphError,
    LossError,
    Port,
    Program,
    Realisation,
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
from xty2.evaluation.simsiam_study import arm_recipe, embedding_metrics, study
from xty2.evaluation.vicreg_views import OracleSymmetry
from xty2.objectives import CosineFeatureConsistency
from xty2.recipes import simsiam
from xty2.recipes.simsiam import CORRUPTED_A as A
from xty2.recipes.simsiam import CORRUPTED_B as B
from xty2.recipes.simsiam import DATA_POLICY, ENCODER_WIDTHS
from xty2.training import executors
from xty2.training.loading import build_population, iterate
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
        p_squared, z_squared, dot = (p.new_zeros(()) for _ in range(3))
        for j in range(3):
            p_squared = p_squared + p[row, j] ** 2
            z_squared = z_squared + z[row, j] ** 2
            dot = dot + p[row, j] * z[row, j]
        pn = p_squared.sqrt().clamp_min(1e-12)
        zn = z_squared.sqrt().clamp_min(1e-12)
        terms.append(-dot / (pn * zn))
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
    projector_component = run.graph["simsiam_projector"]
    predictor_component = run.graph["simsiam_predictor"]
    assert isinstance(projector_component, SimSiamProjector)
    assert isinstance(predictor_component, SimSiamPredictor)
    projector = projector_component.network
    predictor = predictor_component.network
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
    first, middle, final, output_bn = (
        projector[0],
        projector[3],
        projector[6],
        projector[7],
    )
    assert isinstance(first, nn.Linear) and isinstance(middle, nn.Linear)
    assert isinstance(final, nn.Linear) and isinstance(output_bn, nn.BatchNorm1d)
    assert first._parameters["bias"] is None and middle._parameters["bias"] is None
    assert final.bias is not None
    assert not final.bias.requires_grad
    assert bool(final.bias.ne(0).any())
    assert "network.6.bias" in dict(run.graph["simsiam_projector"].named_buffers())
    assert not output_bn.affine
    predictor_first, predictor_final = predictor[0], predictor[3]
    assert isinstance(predictor_first, nn.Linear)
    assert isinstance(predictor_final, nn.Linear)
    assert predictor_first._parameters["bias"] is None
    assert predictor_final.bias is not None
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
    component = run.graph["simsiam_projector"]
    assert isinstance(component, SimSiamProjector)
    projector = component.network
    first, hidden_bn = projector[0], projector[1]
    assert isinstance(first, nn.Linear) and isinstance(hidden_bn, nn.BatchNorm1d)
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
        hidden = first(features)
        expected_mean = 0.9 * expected_mean + 0.1 * hidden.mean(0)
        expected_var = 0.9 * expected_var + 0.1 * hidden.var(0, correction=1)
    torch.testing.assert_close(hidden_bn.running_mean, expected_mean)
    torch.testing.assert_close(hidden_bn.running_var, expected_var)
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


def test_pretraining_anneal_spans_exactly_the_pretraining_budget() -> None:
    """The source anneals over the training length, not over a fixed horizon.

    `adjust_learning_rate` reads `epoch / args.epochs`, so the rate reaches the
    bottom of the curve exactly as training ends. A schedule whose horizon
    outlived the stage would run a prefix of the curve and never get there,
    which is invisible in a diff and in a loss curve alike. Card section 4
    binds both numbers to one constant; `simsiam_study` re-bases the horizon
    whenever it shortens the budget, and this is the check on both.
    """
    pretrain = recipe_run().recipe.program[0]
    schedule = pretrain.optimiser.lr_schedule
    assert isinstance(schedule, CosineAnneal)
    assert schedule.steps == pretrain.steps
    assert schedule(0) == pytest.approx(1.0)
    assert schedule(pretrain.steps // 2) == pytest.approx(0.5)
    for budget in (16, 128):
        shortened = arm_recipe(
            replace(
                recipe_run().recipe,
                program=Program(
                    (
                        replace(
                            pretrain,
                            steps=budget,
                            optimiser=replace(
                                pretrain.optimiser,
                                lr_schedule=CosineAnneal(steps=budget),
                            ),
                        ),
                        recipe_run().recipe.program[1],
                    )
                ),
            ),
            "full",
        )
        rebased = shortened.program[0].optimiser.lr_schedule
        assert isinstance(rebased, CosineAnneal)
        assert rebased.steps == budget
        assert rebased(budget // 2) == pytest.approx(0.5)


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


def test_mixed_pretraining_loss_equals_the_reference_expression() -> None:
    """The stage's mixed total against `main_simsiam.train`'s own line.

    The reference writes
    `-(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5` with
    `criterion = nn.CosineSimilarity(dim=1)`. That is a different code path
    from this recipe's two half-weighted `Weighted(..., reduction="mean")`
    terms over separate `F.normalize` calls, so the comparison checks the
    whole objective-and-mixer chain rather than one function against itself:
    both half-weights, both directions, the row mean, and the absence of a
    second division by batch or width. Deviation 6 predicts that the epsilon
    difference is invisible at ordinary norms, and this is where that is
    checked; the near-zero behaviour is
    `test_directional_values_and_gradients` at scale 1e-14.
    """
    schema = continuous_schema(6)
    torch.manual_seed(310006)
    run = recipe_run()
    population = build_population(
        training_dataset(
            schema,
            two_cluster_population(
                1024, seed=310001, row_offset=0, low=SEPARATED
            ).batch,
        ),
        DATA_POLICY,
        seed=310002,
    )
    run.graph.train()
    state = run.state(run.stages[0], population.rows, rng_key=7, population=population)
    mixed = LossMixer.for_stage(run.stages[0]).mix(
        state, population.rows, TrainContext(global_step=0, schema=schema)
    )

    def embedding(realisation: Realisation, port: Port) -> Tensor:
        value = state[realisation][port]
        assert isinstance(value, Tensor)
        return value

    prediction_a = embedding(A, Port.X_PRED)
    projection_a = embedding(A, Port.X_PROJ)
    prediction_b = embedding(B, Port.X_PRED)
    projection_b = embedding(B, Port.X_PROJ)
    criterion = nn.CosineSimilarity(dim=1)
    reference = (
        -(
            criterion(prediction_a, projection_b.detach()).mean()
            + criterion(prediction_b, projection_a.detach()).mean()
        )
        * 0.5
    )
    torch.testing.assert_close(mixed.total, reference)
    # A degenerate agreement would be two zeros; the fixture is nondegenerate.
    assert abs(reference.detach().item()) > 1e-4


def projector_batchnorm_regimes(seed: int) -> list[float]:
    """`var / (var + eps)` at each projector BatchNorm, on one training batch.

    This is the fraction of the normalisation each declared layer actually
    performs: one when the batch variance dominates `eps`, and zero when the
    constant floor does. The pinned builder reads a ResNet-50 pooled feature,
    where every projector BatchNorm sits at one.
    """
    schema = continuous_schema(6)
    train = two_cluster_population(1024, seed=seed + 1, row_offset=0, low=SEPARATED)
    population = build_population(
        training_dataset(schema, train.batch), DATA_POLICY, seed=seed + 2
    )
    torch.manual_seed(seed + 6)
    run = recipe_run()
    component = run.graph["simsiam_projector"]
    assert isinstance(component, SimSiamProjector)
    rows = population.rows
    view = run.recipe.views[0].apply(
        rows, schema, rng_key=seed + 3, population=population
    )
    run.graph.train()
    regimes: list[float] = []
    with torch.no_grad():
        value = run.graph.evaluate(view, schema=schema, only=("mlp_encoder",))[
            Port.X_REPR
        ]
        assert isinstance(value, Tensor)
        for module in component.network:
            if isinstance(module, nn.BatchNorm1d):
                variance = value.var(0, correction=0)
                regimes.append(float((variance / (variance + 1e-5)).mean()))
            value = module(value)
    return regimes


def test_projector_batchnorm_normalising_regime() -> None:
    """Every projector BatchNorm normalises, as the source's does.

    The pinned builder's projector reads a ResNet-50 pooled feature of order
    one, so each `nn.BatchNorm1d` divides by a batch standard deviation that
    dominates `eps`. Card deviation 7 recorded a period in which the encoder's
    inherited initialiser put `X_REPR` four orders of magnitude below that and
    floored the two hidden layers at `eps`; the withdrawal of that row is what
    this asserts. Each threshold sits between two measured, well-separated
    regimes rather than beside one seed: over six seeds the two initialisers
    give `[0.888, 0.909]` against `[6.1e-6, 7.9e-6]` at the first hidden layer
    and `[0.99989, 0.99989]` against `[0.068, 0.079]` at the second. Paper
    Table 3 measures this configuration at 33 accuracy points, so it is a
    mechanic and not a scale detail.
    """
    for seed in (520000, 520400, 520900, 42):
        hidden_one, hidden_two, output = projector_batchnorm_regimes(seed)
        assert hidden_one > 0.5, seed
        assert hidden_two > 0.9, seed
        assert output > 0.99, seed


def test_encoder_representation_arrives_at_the_source_scale() -> None:
    """`X_REPR` reaches the projector at order one, as the pooled feature does.

    Card section 3.2's norm check reads `X_PROJ`, which the terminal non-affine
    BatchNorm pins near `sqrt(d) = 16` whatever the encoder does. This is the
    port that check did not read and the one the projector's own BatchNorm
    layers consume; deviation 7 is the row it cost. Measured across six seeds
    at `[0.567, 0.730]`, against `[2.9e-4, 3.3e-4]` under the withdrawn
    initialiser.
    """
    schema = continuous_schema(6)
    for seed in (520000, 520400, 520900, 42):
        torch.manual_seed(seed + 6)
        run = recipe_run()
        run.graph.train()
        with torch.no_grad():
            representation = run.graph.evaluate(
                two_cluster_population(
                    128, seed=seed + 1, row_offset=0, low=SEPARATED
                ).batch,
                schema=schema,
                only=("mlp_encoder",),
            )[Port.X_REPR]
        assert isinstance(representation, Tensor)
        assert 0.1 < float(representation.norm(dim=-1).mean()) < 10.0, seed


def scalar_spread(rows: Tensor) -> float:
    """Card section 6.4's S recomputed one index at a time."""
    count, width = rows.shape
    unit = []
    for i in range(count):
        squared = 0.0
        for j in range(width):
            squared += float(rows[i, j]) ** 2
        norm = max(math.sqrt(squared), 1e-12)
        unit.append([float(rows[i, j]) / norm for j in range(width)])
    total = 0.0
    for j in range(width):
        mean = sum(unit[i][j] for i in range(count)) / count
        variance = sum((unit[i][j] - mean) ** 2 for i in range(count)) / count
        total += math.sqrt(variance)
    return math.sqrt(width) * total / width


def test_projection_spread_cannot_see_directional_collapse() -> None:
    """Card section 6.4's S is a channel-balance statistic, not a rank test.

    S divides each row by its own norm, so it is invariant to the embedding's
    scale, and what remains is how evenly the channels share that direction.
    An embedding of exact rank one — every row a multiple of one vector, the
    strongest collapse there is — already scores above the `>= 0.5` bound on
    its own, and scores 0.99 once the projector's terminal non-affine
    BatchNorm equalises the channels. Paper section 4.1 reads the same
    quantity at zero, which requires the pre-BatchNorm output to be constant
    across rows; the measured pre-BatchNorm variance in every section 6.2 arm
    is 0.5 to 1.0 per channel, five orders of magnitude above `eps = 1e-5`.
    """
    torch.manual_seed(0)
    direction = torch.randn(256)
    collapsed = torch.randn(128, 1) * direction
    bare = embedding_metrics(collapsed)
    assert bare["effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert bare["spread"] == pytest.approx(scalar_spread(collapsed), abs=1e-6)
    assert bare["spread"] > 0.5
    normaliser = nn.BatchNorm1d(256, eps=1e-5, affine=False)
    normaliser.train()
    behind = embedding_metrics(normaliser(collapsed))
    assert behind["effective_rank"] == pytest.approx(1.0, abs=1e-6)
    assert behind["spread"] > 0.99
    # The discriminating comparison: an isotropic embedding scores what the
    # rank-one one scores, so at the values section 6.4 reads S carries no
    # rank information at all. A centred 128-row cross-product has rank at
    # most 127, so the ceiling below is the batch, not the width.
    isotropic = embedding_metrics(torch.randn(128, 256))
    assert isotropic["effective_rank"] > 90
    assert abs(isotropic["spread"] - behind["spread"]) < 0.02
    # S reaches zero only for a row-constant embedding, which the terminal
    # BatchNorm cannot produce while its batch variance dominates eps.
    assert embedding_metrics(torch.ones(8, 4))["spread"] == 0.0


def test_withdrawn_initialiser_oracle_kills_the_normalising_projector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deviation 7's withdrawal is what the two checks above are worth.

    Restoring the inherited `normal std=0.1/sqrt(fan_in)` encoder — the
    initialiser paper supplement A warns about, at `std = 0.00625` for
    `fan_in = 256` — floors the projector's two hidden BatchNorm layers at
    `eps` again and drops `X_REPR` by four orders of magnitude, so both fail.
    """
    original = recipe_run

    def withdrawn() -> CompiledRun:
        run = original()
        run.graph["mlp_encoder"].load_state_dict(
            MLPEncoder(
                input_dim=6,
                widths=ENCODER_WIDTHS,
                activation="relu",
                normalisation="none",
                dropout=0.0,
                initialisation=CFRNET_INITIALISATION,
            ).state_dict()
        )
        return run

    monkeypatch.setattr(sys.modules[__name__], "recipe_run", withdrawn)
    with pytest.raises(AssertionError):
        test_projector_batchnorm_normalising_regime()
    with pytest.raises(AssertionError):
        test_encoder_representation_arrives_at_the_source_scale()


def test_mask_swap_mutant_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    original = iterate
    calls = 0

    def changed_mask(*args: Any, **kwargs: Any) -> Iterator[XTYBatch]:
        nonlocal calls
        calls += 1
        alter = calls == 3  # First stage of the second arm.
        for batch in original(*args, **kwargs):
            if alter:
                mask = batch.t_observed.roll(1)
                assert mask.sum() == batch.t_observed.sum()
                assert not torch.equal(mask, batch.t_observed)
                batch = batch.replace(t_observed=mask)
            yield batch

    monkeypatch.setattr(executors, "iterate", changed_mask)
    with pytest.raises(RuntimeError, match="actual row/mask/value traces differ"):
        study(
            42,
            train_rows=128,
            test_rows=128,
            pretrain_steps=2,
            fit_steps=2,
            eval_batches=1,
        )


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
    component = run.graph["simsiam_projector"]
    assert isinstance(component, SimSiamProjector)
    output_bn = component.network[7]
    assert isinstance(output_bn, nn.BatchNorm1d)
    output_bn.affine = True
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
