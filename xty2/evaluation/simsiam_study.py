"""Paired SimSiam execution and diagnostics shared by Tiers 1 and 2."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import Tensor

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    CompiledStage,
    ComponentGraph,
    CosineAnneal,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    Schema,
    TrainingPopulation,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    SEPARATED,
    configure_worker,
    continuous_schema,
    on_the_training_scale,
    take,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.vicreg_views import (
    PRESERVATION_TOLERANCE,
    OracleSymmetry,
    fit_with_trace,
    targets,
)
from xty2.objectives import CosineFeatureConsistency
from xty2.recipes import simsiam
from xty2.recipes.simsiam import BATCH_SIZE, DATA_POLICY
from xty2.training import STREAM_STRIDE, executors
from xty2.training.executors import _Stepped
from xty2.training.loading import build_population
from xty2.training.loss_mixer import LossMixer
from xty2.training.teacher import EMATeacher

ARMS = ("full", "no_stop", "no_predictor", "no_pretrain")
PRETRAINED = ARMS[:3]


def arm_recipe(recipe: Recipe, arm: str) -> Recipe:
    """Construct all common initial tensors before removing ablated modules."""
    if arm not in ARMS:
        raise ValueError(f"unknown SimSiam arm {arm!r}")
    pretrain, fit = recipe.program
    if arm == "no_pretrain":
        return replace(recipe, program=Program((replace(fit, initialise_from=None),)))
    if arm == "full":
        return recipe
    objectives = []
    for term in pretrain.objectives:
        objective = term.objective
        assert isinstance(objective, CosineFeatureConsistency)
        objectives.append(
            replace(
                term,
                objective=replace(
                    objective,
                    stop_grad="none" if arm == "no_stop" else "target",
                    prediction_port=Port.X_PROJ
                    if arm == "no_predictor"
                    else Port.X_PRED,
                ),
            )
        )
    trainable = tuple(
        name
        for name in pretrain.trainable
        if arm != "no_predictor" or name != "simsiam_predictor"
    )
    graph = ComponentGraph(
        [
            component
            for component in recipe.system.components
            if arm != "no_predictor" or component.name != "simsiam_predictor"
        ]
    )
    return replace(
        recipe,
        system=graph,
        program=Program(
            (
                replace(pretrain, objectives=tuple(objectives), trainable=trainable),
                fit,
            )
        ),
    )


def snapshot(graph: ComponentGraph) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in graph.state_dict().items()}


def require_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> None:
    for name, value in left.items():
        if name not in right or not torch.equal(value, right[name]):
            raise RuntimeError(f"SimSiam pairing/transition mismatch: {name}")


def embedding_metrics(z: Tensor) -> dict[str, float]:
    """Card 6.4: row-normalised spread and centred covariance rank."""
    if z.ndim != 2 or z.shape[0] < 2 or not bool(torch.isfinite(z).all()):
        raise RuntimeError("invalid embedding diagnostics input")
    norms = z.norm(dim=1)
    u = z / norms.clamp_min(1e-12)[:, None]
    centred = z.double() - z.double().mean(0)
    eigenvalues = torch.linalg.svdvals(centred).square()
    total = float(eigenvalues.sum())
    probabilities = eigenvalues / total if total > 0 else eigenvalues
    positive = probabilities[probabilities > 0]
    return {
        "spread": float(math.sqrt(z.shape[1]) * u.std(0, correction=0).mean()),
        "raw_norm": float(norms.mean()),
        "zero_fraction": float((norms == 0).float().mean()),
        "near_zero_fraction": float((norms <= 1e-8).float().mean()),
        "concentration": float(u.mean(0).norm()),
        "effective_rank": float((-(positive * positive.log()).sum()).exp())
        if total > 0
        else 0.0,
        "covariance_top_share": float(probabilities.max()) if total > 0 else 0.0,
    }


def encoder_diagnostics(
    graph: ComponentGraph, views: Sequence[XTYBatch], schema: Schema
) -> dict[str, float]:
    """`X_REPR` diagnostics over the held-out views, in eval mode.

    The encoder is the port the downstream stage inherits and the only one no
    BatchNorm stands in front of, which is why section 6.4 reads its rank
    rather than the projection's.
    """
    was_training = graph.training
    graph.eval()
    observations: dict[str, list[float]] = {}
    try:
        with torch.no_grad():
            for view in views:
                value = graph.evaluate(view, schema=schema, only=("mlp_encoder",))[
                    Port.X_REPR
                ]
                assert isinstance(value, Tensor)
                for name, measurement in embedding_metrics(value).items():
                    observations.setdefault(name, []).append(measurement)
    finally:
        graph.train(was_training)
    return {
        name: math.fsum(values) / len(values) for name, values in observations.items()
    }


def study(
    base: int,
    *,
    train_rows: int,
    test_rows: int,
    pretrain_steps: int,
    fit_steps: int,
    eval_batches: int,
) -> dict[str, float]:
    """Run all four arms, rejecting broken pairing before returning any metric."""
    configure_worker()
    if test_rows != eval_batches * BATCH_SIZE:
        raise ValueError("held-out batches must partition the held-out population")
    schema = continuous_schema(6)
    train = two_cluster_population(
        train_rows, seed=base + 1, row_offset=0, low=SEPARATED
    )
    test = two_cluster_population(
        test_rows, seed=base + 2, row_offset=10000, low=SEPARATED
    )
    data = training_dataset(schema, train.batch)
    # One fitted object and one realised MCAR mask, shared by every stage/arm.
    population = build_population(data, DATA_POLICY, seed=base + 10000 + STREAM_STRIDE)
    heldout = on_the_training_scale(test.batch, population)
    views: list[XTYBatch] = []
    max_error = 0.0
    for index in range(eval_batches):
        batch = take(
            heldout, torch.arange(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
        )
        for branch in range(2):
            view = OracleSymmetry().apply(
                batch,
                schema,
                population=population,
                generator=torch.Generator().manual_seed(
                    base + 20000 + 2 * index + branch
                ),
            )
            scale, location = (
                population.statistics["x_scale"],
                population.statistics["x_location"],
            )
            max_error = max(
                max_error,
                float(
                    (
                        targets(view.x * scale + location)
                        - targets(batch.x * scale + location)
                    )
                    .abs()
                    .max()
                ),
            )
            views.append(view)
    if max_error > PRESERVATION_TOLERANCE:
        raise RuntimeError("held-out oracle views changed analytic targets")
    metrics: dict[str, float] = {"view_max_target_error": max_error}
    initial: dict[str, Tensor] | None = None
    paired_traces: dict[str, str] = {}
    paired_views: dict[str, str] | None = None
    original_step = executors._step
    transitions: dict[str, dict[str, Tensor]] = {}
    traces = {name: hashlib.sha256() for name in ("pretrain", "joint_fit")}
    optimisers: list[torch.optim.Optimizer] = []
    for arm in ARMS:
        torch.manual_seed(base + 6)
        recipe = simsiam(
            schema,
            first_transforms=(OracleSymmetry(),),
            second_transforms=(OracleSymmetry(),),
        )
        pretrain, fit = recipe.program
        recipe = arm_recipe(
            replace(
                recipe,
                program=Program(
                    (
                        replace(
                            pretrain,
                            steps=pretrain_steps,
                            # The source anneals over the training length, so a
                            # shortened budget re-bases the horizon with it
                            # rather than running a prefix of the long curve.
                            # A no-op at section 4's own 1000 steps.
                            optimiser=replace(
                                pretrain.optimiser,
                                lr_schedule=CosineAnneal(steps=pretrain_steps),
                            ),
                        ),
                        replace(fit, steps=fit_steps),
                    )
                ),
            ),
            arm,
        )
        run = compile(recipe)
        if recipe.data != DATA_POLICY:
            raise RuntimeError("SimSiam arm changed the shared preprocessing policy")
        start = snapshot(run.graph)
        if initial is None:
            initial = start
            # Section 6.4's noncollapse reference: the untrained encoder on the
            # same held-out views, under the shared initial tensors every arm
            # is about to be checked against. Pretraining that ends below this
            # has destroyed representation rank rather than built any.
            metrics.update(
                {
                    f"initial_encoder_{name}": value
                    for name, value in encoder_diagnostics(
                        run.graph, views, schema
                    ).items()
                }
            )
            require_equal(snapshot(run.graph), start)
        require_equal(start, initial)
        transitions.clear()
        traces.clear()
        traces.update({stage.name: hashlib.sha256() for stage in run.stages})
        optimisers.clear()

        def traced_step(
            active_run: CompiledRun,
            compiled: CompiledStage,
            batch: XTYBatch,
            mixer: LossMixer,
            optimiser: torch.optim.Optimizer,
            tensors: Sequence[Tensor],
            step: int,
            *,
            rng_key: int,
            teacher: EMATeacher | None,
            population: TrainingPopulation | None = None,
            objective_states: Mapping[str, object] | None = None,
        ) -> _Stepped:
            trace = traces[compiled.name]
            for value in (
                batch.row_id,
                batch.t_observed,
                batch.y_observed,
                batch.x,
                batch.y,
                batch.t,
            ):
                trace.update(value.numpy().tobytes())
            if step == 0:
                if optimiser.state or any(optimiser is old for old in optimisers):
                    raise RuntimeError("SimSiam stage did not start a fresh optimiser")
                optimisers.append(optimiser)
                transitions[compiled.name] = snapshot(active_run.graph)
            result = original_step(
                active_run,
                compiled,
                batch,
                mixer,
                optimiser,
                tensors,
                step,
                rng_key=rng_key,
                teacher=teacher,
                population=population,
                objective_states=objective_states,
            )
            if not bool(torch.isfinite(result.loss.total)):
                raise RuntimeError("non-finite SimSiam training loss")
            if step == 0:
                for name in compiled.trainable:
                    gradients = [
                        p.grad
                        for p in active_run.graph[name].parameters()
                        if p.grad is not None
                    ]
                    if (
                        not gradients
                        or not all(bool(torch.isfinite(g).all()) for g in gradients)
                        or not any(bool(g.ne(0).any()) for g in gradients)
                    ):
                        raise RuntimeError(
                            f"missing/non-finite intended gradient: {name}"
                        )
            return result

        with (
            patch.object(executors, "build_population", return_value=population),
            patch.object(executors, "_step", traced_step),
        ):
            result, view_trace = fit_with_trace(
                run,
                {stage.name: data for stage in run.stages},
                seed=base + 10000 + (STREAM_STRIDE if arm == "no_pretrain" else 0),
            )
        for stage in run.stages:
            digest = traces[stage.name].hexdigest()
            if stage.name in paired_traces and digest != paired_traces[stage.name]:
                raise RuntimeError("actual row/mask/value traces differ across arms")
            paired_traces[stage.name] = digest
            if result.stage(stage.name).population is not population:
                raise RuntimeError("stage replaced the shared fitted population")
        if result.stage("joint_fit").seed != base + 10000 + STREAM_STRIDE:
            raise RuntimeError("downstream execution stream mismatch")
        head_start = {
            k: v
            for k, v in start.items()
            if k.startswith(
                ("_components.tarnet_head.", "_components.categorical_propensity.")
            )
        }
        require_equal(head_start, transitions["joint_fit"])
        for value in run.graph.state_dict().values():
            if not bool(torch.isfinite(value).all()):
                raise RuntimeError("non-finite SimSiam parameter/buffer")
        run.graph.eval()
        with torch.no_grad():
            values = run.graph.evaluate(
                heldout,
                schema=schema,
                only=("mlp_encoder", "tarnet_head", "categorical_propensity"),
            )
            outcome = values[Port.Y_GIVEN_XT]
            assert isinstance(outcome, GaussianOutcome)
            propensity = values[Port.T_GIVEN_X]
            if not isinstance(propensity, CategoricalTreatment):
                raise TypeError("expected propensity distribution")
            metrics[f"{arm}_outcome_nll"] = float(
                -outcome.log_prob(heldout.y, heldout.t).mean()
            )
            metrics[f"{arm}_treatment_nll"] = float(
                -propensity.log_prob(heldout.t).mean()
            )
            effect = (
                outcome.mean(torch.ones(test_rows, dtype=torch.long))
                - outcome.mean(torch.zeros(test_rows, dtype=torch.long))
            ).reshape(-1)
            effect = effect * population.statistics["y_scale"]
            metrics[f"{arm}_effect_rmse"] = float(
                (effect - test.true_effect.reshape(-1)).square().mean().sqrt()
            )
        if arm == "no_pretrain":
            require_equal(start, transitions["joint_fit"])
            continue
        if paired_views is not None and paired_views != view_trace:
            raise RuntimeError("actual pretraining view draws differ across arms")
        paired_views = view_trace
        if int(view_trace["calls"]) != 2 * pretrain_steps:
            raise RuntimeError("expected exactly two cached views per pretraining step")
        checkpoint = result.stage("pretrain").checkpoint
        saved = {
            "_components." + k: v
            for k, v in {**checkpoint.parameters, **checkpoint.buffers}.items()
        }
        require_equal(saved, transitions["joint_fit"])
        if any(
            k.startswith(("tarnet_head.", "categorical_propensity."))
            for k in checkpoint.parameters
        ):
            raise RuntimeError("pretraining checkpoint contains downstream heads")
        for name in run.stages[0].trainable:
            changed = any(
                not torch.equal(v, start["_components." + k])
                for k, v in checkpoint.parameters.items()
                if k.startswith(name + ".")
            )
            if not changed:
                raise RuntimeError(f"pretraining did not update {name}")
        require_equal(
            {
                k: v
                for k, v in saved.items()
                if k.startswith(
                    ("_components.simsiam_projector.", "_components.simsiam_predictor.")
                )
            },
            run.graph.state_dict(),
        )
        with torch.no_grad():
            state = run.graph.state_dict()
            for name, value in saved.items():
                state[name].copy_(value)
            before = snapshot(run.graph)
            observations: dict[str, list[float]] = {}
            embeddings: list[dict[Port, Tensor]] = []
            for view in views:
                evaluated = run.graph.evaluate(
                    view, schema=schema, only=run.stages[0].trainable
                )
                ports = (
                    (Port.X_REPR, Port.X_PROJ)
                    if arm == "no_predictor"
                    else (Port.X_REPR, Port.X_PROJ, Port.X_PRED)
                )
                current: dict[Port, Tensor] = {}
                for port in ports:
                    z = evaluated[port]
                    assert isinstance(z, Tensor)
                    current[port] = z
                    label = {
                        Port.X_REPR: "encoder",
                        Port.X_PROJ: "projection",
                        Port.X_PRED: "predictor",
                    }[port]
                    for name, value in embedding_metrics(z).items():
                        observations.setdefault(f"{label}_{name}", []).append(value)
                embeddings.append(current)
            for index in range(0, len(embeddings), 2):
                first, second = embeddings[index : index + 2]
                prediction = Port.X_PROJ if arm == "no_predictor" else Port.X_PRED
                alignment = 0.5 * (
                    torch.nn.functional.cosine_similarity(
                        first[prediction], second[Port.X_PROJ], dim=1, eps=1e-12
                    ).mean()
                    + torch.nn.functional.cosine_similarity(
                        second[prediction], first[Port.X_PROJ], dim=1, eps=1e-12
                    ).mean()
                )
                observations.setdefault("alignment", []).append(float(alignment))
            for name, values_list in observations.items():
                metrics[f"{arm}_{name}"] = math.fsum(values_list) / len(values_list)
            require_equal(before, run.graph.state_dict())
    # Section 6.4's required contrasts. An ablation sitting closer to the
    # cosine's trivial optimum than the full arm is paper Figure 2 (left) in
    # this fixture's terms; the retention term is the absolute guard.
    metrics["stop_gradient_alignment_gap"] = (
        metrics["no_stop_alignment"] - metrics["full_alignment"]
    )
    metrics["predictor_alignment_gap"] = (
        metrics["no_predictor_alignment"] - metrics["full_alignment"]
    )
    for arm in ("no_stop", "no_predictor"):
        name = "stop_gradient" if arm == "no_stop" else "predictor"
        metrics[f"{name}_rank_gap"] = (
            metrics["full_encoder_effective_rank"]
            - metrics[f"{arm}_encoder_effective_rank"]
        )
    # Reported, never bounded: a random projection of six inputs already has an
    # effective rank near 19, so concentrating below it is what a working
    # representation does here (section 6.4).
    metrics["encoder_rank_retention"] = (
        metrics["full_encoder_effective_rank"]
        - metrics["initial_encoder_effective_rank"]
    )
    # Retained as informational only: the 2026-09-15 audit shows S is a
    # channel-balance statistic behind a non-affine BatchNorm, scoring 0.99 on
    # an embedding of exact rank one, so it cannot carry a bound here.
    metrics["stop_gradient_spread_gap"] = (
        metrics["full_projection_spread"] - metrics["no_stop_projection_spread"]
    )
    metrics["predictor_spread_gap"] = (
        metrics["full_projection_spread"] - metrics["no_predictor_projection_spread"]
    )
    metrics["pretraining_outcome_nll_cost"] = (
        metrics["full_outcome_nll"] - metrics["no_pretrain_outcome_nll"]
    )
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("non-finite SimSiam replicate; do not drop this seed")
    return metrics
