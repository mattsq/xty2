"""BYOL card §6: paired execution and pre-transfer diagnostics for both tiers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import Tensor, nn

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    CompiledStage,
    ComponentGraph,
    Constant,
    CosineEMADecay,
    GaussianOutcome,
    Port,
    Program,
    Ramp,
    Recipe,
    Schema,
    TrainingPopulation,
    WarmupCosine,
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
from xty2.evaluation.simsiam_study import embedding_metrics, snapshot
from xty2.evaluation.vicreg_views import (
    PRESERVATION_TOLERANCE,
    OracleSymmetry,
    fit_with_trace,
    targets,
)
from xty2.objectives import NormalizedSquaredFeatureConsistency
from xty2.objectives.feature_consistency import squared_norm_floor_normalize
from xty2.recipes import byol
from xty2.recipes.byol import BASE_TARGET_EMA, BATCH_SIZE, DATA_POLICY
from xty2.training import STREAM_STRIDE, executors
from xty2.training.executors import _Stepped
from xty2.training.loading import build_population
from xty2.training.loss_mixer import LossMixer
from xty2.training.teacher import EMATeacher

ARMS = ("full", "source_ema", "zero_decay", "no_predictor", "no_pretrain")
PRETRAIN_STEPS = 100
FIT_STEPS = 200
# `_EMA_PRESETS[1000]`, the row card §5 row 3 originally inherited. `source_ema`
# is the informational control that isolates deviation 7's re-derivation from
# everything else the 2026-09-17 protocol changed; it is not a scored arm.
INHERITED_EMA_BASE = 0.996
# Literal decays for the executed-update oracle. Written out rather than read
# from the recipe: an oracle that recomputes the recipe's own expression shares
# its call path and cannot catch it being wrong (CLAUDE.md, Evidence).
ORACLE_EMA_BASE = {
    "full": 0.68722,
    "source_ema": 0.996,
    "no_predictor": 0.68722,
    "no_pretrain": 0.68722,
    "zero_decay": 0.0,
}


def require_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> None:
    """Exact equality on the requested subset, including buffers."""
    if not left:
        raise RuntimeError("empty BYOL state comparison")
    for name, value in left.items():
        if name not in right or not torch.equal(value, right[name]):
            raise RuntimeError(f"BYOL pairing/transfer mismatch: {name}")


def training_batch(batch: XTYBatch) -> XTYBatch:
    """Remove oracle treatment payloads before either stage's forward pass.

    XTYBatch requires valid class indices even for missing treatments. Zero
    is an arbitrary payload here; the unchanged mask defines missingness.
    """
    return batch.replace(t=torch.where(batch.t_observed, batch.t, 0))


def arm_recipe(
    recipe: Recipe,
    arm: str,
    *,
    pretrain_steps: int = PRETRAIN_STEPS,
    fit_steps: int = FIT_STEPS,
    warmup_steps: int = 1,
    ramp_steps: int = 200,
) -> Recipe:
    """Construct common modules before ablation; bind each tier's schedules."""
    if arm not in ARMS:
        raise ValueError(f"unknown BYOL arm {arm!r}")
    pretrain, fit = recipe.program
    assert pretrain.teacher is not None
    pretrain = replace(
        pretrain,
        steps=pretrain_steps,
        teacher=replace(
            pretrain.teacher,
            decay=Constant(0.0)
            if arm == "zero_decay"
            else CosineEMADecay(
                base=INHERITED_EMA_BASE if arm == "source_ema" else BASE_TARGET_EMA,
                steps=pretrain_steps,
            ),
        ),
        optimiser=replace(
            pretrain.optimiser,
            lr_schedule=WarmupCosine(
                start=0.0, final=0.0, warmup=warmup_steps, steps=pretrain_steps
            ),
        ),
    )
    fit = replace(
        fit,
        steps=fit_steps,
        objectives=tuple(
            replace(term, weight=Ramp(0.0, 0.5, steps=ramp_steps))
            if term.objective.name == "missing_treatment_marginal_nll"
            else term
            for term in fit.objectives
        ),
    )
    graph = recipe.system
    if arm == "no_predictor":
        terms = []
        for term in pretrain.objectives:
            assert isinstance(term.objective, NormalizedSquaredFeatureConsistency)
            terms.append(
                replace(
                    term, objective=replace(term.objective, prediction_port=Port.X_PROJ)
                )
            )
        pretrain = replace(
            pretrain,
            objectives=tuple(terms),
            trainable=("mlp_encoder", "byol_projector"),
        )
        graph = ComponentGraph(
            [c for c in graph.components if c.name != "byol_predictor"]
        )
    program = (
        (replace(fit, initialise_from=None),)
        if arm == "no_pretrain"
        else (pretrain, fit)
    )
    return replace(recipe, system=graph, program=Program(program))


def encoder_effective_rank(z: Tensor) -> float:
    """Card §6.4: entropy rank of centred energy, with its absolute floor."""
    if z.ndim != 2 or min(z.shape) < 1 or z.shape[0] < 2:
        raise ValueError("rank requires a nonempty matrix with at least two rows")
    if not bool(torch.isfinite(z).all()):
        raise ValueError("rank requires finite embeddings")
    centred = z.double() - z.double().mean(0)
    energy = torch.linalg.svdvals(centred).square()
    total = float(energy.sum())
    if total <= 1e-12:
        return 0.0
    p = energy[energy > 0] / total
    return float((-(p * p.log()).sum()).exp())


@torch.no_grad()
def diagnostics(
    graph: ComponentGraph,
    teacher: EMATeacher | None,
    heldout: XTYBatch,
    views: Sequence[XTYBatch],
    schema: Schema,
    arm: str,
) -> dict[str, float]:
    """Clean encoder rank over all rows; directional residuals use fixed views."""
    before = snapshot(graph)
    target_before = snapshot(teacher.graph) if teacher is not None else None
    modes = [(module, module.training) for module in graph.modules()]
    if teacher is not None:
        modes.extend((module, module.training) for module in teacher.graph.modules())
    graph.eval()
    if teacher is not None:
        teacher.graph.eval()
    try:
        z = graph.evaluate(heldout, schema=schema, only=("mlp_encoder",))[Port.X_REPR]
        assert isinstance(z, Tensor)
        metrics = {f"encoder_{k}": v for k, v in embedding_metrics(z).items()}
        metrics["encoder_effective_rank"] = encoder_effective_rank(z)
        if teacher is not None:
            for side, network in (("online", graph), ("target", teacher.graph)):
                for name, module in network.named_modules():
                    if isinstance(module, nn.BatchNorm1d):
                        assert module.running_var is not None
                        metrics[f"{side}_{name}_bn_variance_over_epsilon"] = float(
                            module.running_var.mean() / module.eps
                        )
            online = dict(graph.named_parameters())
            metrics["target_online_lag"] = math.sqrt(
                sum(
                    float((p - online[name]).square().sum())
                    for name, p in teacher.graph.named_parameters()
                    if name.startswith(
                        ("_components.mlp_encoder.", "_components.byol_projector.")
                    )
                )
            )
            prediction = Port.X_PROJ if arm == "no_predictor" else Port.X_PRED
            components = ("mlp_encoder", "byol_projector") + (
                () if arm == "no_predictor" else ("byol_predictor",)
            )
            residuals = []
            alignments = []
            norms = []
            for i in range(0, len(views), 2):
                directional = []
                for a, b in ((views[i], views[i + 1]), (views[i + 1], views[i])):
                    p = graph.evaluate(a, schema=schema, only=components)[prediction]
                    t = teacher.graph.evaluate(
                        b, schema=schema, only=("mlp_encoder", "byol_projector")
                    )[Port.X_PROJ]
                    assert isinstance(p, Tensor) and isinstance(t, Tensor)
                    unit_p = squared_norm_floor_normalize(p, 1e-12)
                    unit_t = squared_norm_floor_normalize(t, 1e-12)
                    directional.append(float((unit_p - unit_t).square().sum(-1).mean()))
                    # Card §6.4's `A`. Read off the same two vectors as the
                    # distance, under the source's squared-norm floor rather
                    # than `cosine_similarity`'s floor on the norm, because
                    # §3.1 makes that floor the objective's own.
                    alignments.append(float((unit_p * unit_t).sum(-1).mean()))
                    norms.append(float(p.norm(dim=1).mean()))
                residuals.append(sum(directional))
            metrics["predictor_residual"] = math.fsum(residuals) / len(residuals)
            metrics["view_alignment"] = math.fsum(alignments) / len(alignments)
            metrics["prediction_norm"] = math.fsum(norms) / len(norms)
        require_equal(before, snapshot(graph))
        if teacher is not None and target_before is not None:
            require_equal(target_before, snapshot(teacher.graph))
        return metrics
    finally:
        for module, mode in modes:
            module.training = mode


def study(
    base: int,
    *,
    train_rows: int = 512,
    test_rows: int = 512,
    pretrain_steps: int = PRETRAIN_STEPS,
    fit_steps: int = FIT_STEPS,
    warmup_steps: int = 1,
    ramp_steps: int = 200,
    eval_batches: int = 4,
) -> dict[str, float]:
    """All four card-declared arms, with actual execution checks."""
    if eval_batches * BATCH_SIZE != test_rows:
        raise ValueError("diagnostic batches must cover the held-out population")
    configure_worker()
    schema = continuous_schema(6)
    train = two_cluster_population(
        train_rows, seed=base + 1, row_offset=0, low=SEPARATED
    )
    test = two_cluster_population(
        test_rows, seed=base + 2, row_offset=10000, low=SEPARATED
    )
    data = training_dataset(schema, train.batch)
    population = build_population(data, DATA_POLICY, seed=base + 10000 + STREAM_STRIDE)
    if int(population.rows.t_observed.sum()) != 40:
        raise RuntimeError("BYOL study requires exactly 40 observed treatments")
    heldout = on_the_training_scale(test.batch, population)
    views = []
    max_error = 0.0
    for i in range(eval_batches):
        batch = take(heldout, torch.arange(i * BATCH_SIZE, (i + 1) * BATCH_SIZE))
        for branch in range(2):
            view = OracleSymmetry().apply(
                batch,
                schema,
                population=population,
                generator=torch.Generator().manual_seed(base + 20000 + 2 * i + branch),
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
        raise RuntimeError("BYOL views changed analytic targets")
    metrics = {"view_max_target_error": max_error}
    initial: dict[str, Tensor] | None = None
    paired_traces: dict[str, str] = {}
    paired_views: dict[str, str] | None = None
    original_step = executors._step
    original_update = EMATeacher.update
    traces = {name: hashlib.sha256() for name in ("pretrain", "joint_fit")}
    transitions: dict[str, dict[str, Tensor]] = {}
    optimisers: list[torch.optim.Optimizer] = []
    updates = [0]
    for arm in ARMS:
        torch.manual_seed(base + 6)
        recipe = arm_recipe(
            byol(
                schema,
                first_transforms=(OracleSymmetry(),),
                second_transforms=(OracleSymmetry(),),
            ),
            arm,
            pretrain_steps=pretrain_steps,
            fit_steps=fit_steps,
            warmup_steps=warmup_steps,
            ramp_steps=ramp_steps,
        )
        run = compile(recipe)
        start = snapshot(run.graph)
        if initial is None:
            initial = start
        require_equal(start, initial)
        traces.clear()
        traces.update({stage.name: hashlib.sha256() for stage in run.stages})
        transitions.clear()
        optimisers.clear()
        updates[0] = 0
        moved = False
        last_teacher: EMATeacher | None = None

        def checked_update(
            teacher: EMATeacher, student: ComponentGraph, step: int, *, arm: str = arm
        ) -> None:
            nonlocal moved
            before = {
                n: p.detach().clone() for n, p in teacher.graph.named_parameters()
            }
            buffers = {n: b.clone() for n, b in teacher.graph.named_buffers()}
            tracked = buffers[
                "_components.byol_projector.network.1.num_batches_tracked"
            ]
            if int(tracked) != 2 * (step + 1):
                raise RuntimeError("BYOL target must forward each view exactly once")
            base = ORACLE_EMA_BASE[arm]
            tau = (
                0.0
                if arm == "zero_decay"
                else 1
                - (1 - base) * (1 + math.cos(math.pi * step / pretrain_steps)) / 2
            )
            original_update(teacher, student, step)
            online = dict(student.named_parameters())
            for name, parameter in teacher.graph.named_parameters():
                expected = before[name] + (1 - tau) * (
                    online[name].detach() - before[name]
                )
                torch.testing.assert_close(parameter, expected, rtol=1e-5, atol=1e-8)
                if parameter.requires_grad or parameter.grad is not None:
                    raise RuntimeError("BYOL target acquired gradients")
                if name.startswith("_components.mlp_encoder."):
                    moved |= not torch.equal(parameter, before[name])
            require_equal(buffers, dict(teacher.graph.named_buffers()))
            updates[0] += 1

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
            arm: str = arm,
        ) -> _Stepped:
            nonlocal last_teacher
            batch = training_batch(batch)
            for value in (
                batch.row_id,
                batch.x,
                batch.t,
                batch.y,
                batch.t_observed,
                batch.y_observed,
            ):
                traces[compiled.name].update(value.numpy().tobytes())
            if step == 0:
                if optimiser.state or any(optimiser is old for old in optimisers):
                    raise RuntimeError("BYOL stage requires a fresh optimiser")
                optimisers.append(optimiser)
                transitions[compiled.name] = snapshot(active_run.graph)
                if teacher is not None:
                    require_equal(snapshot(teacher.graph), snapshot(active_run.graph))
                if compiled.name == "joint_fit":
                    metrics.update(
                        {
                            f"{arm}_{k}": v
                            for k, v in diagnostics(
                                active_run.graph,
                                last_teacher,
                                heldout,
                                views,
                                schema,
                                arm,
                            ).items()
                        }
                    )
            before_updates = updates[0]
            # Independently check the executed LARS rule at an early, middle,
            # and final update, including momentum accumulated at zero LR.
            lars_before = []
            if compiled.name == "pretrain" and step in (
                warmup_steps,
                pretrain_steps // 2,
                pretrain_steps - 1,
            ):
                for group in optimiser.param_groups:
                    for parameter in group["params"]:
                        previous = optimiser.state[parameter].get("momentum_buffer")
                        lars_before.append(
                            (
                                parameter,
                                parameter.detach().clone(),
                                torch.zeros_like(parameter)
                                if previous is None
                                else previous.clone(),
                                float(group["weight_decay"]),
                            )
                        )
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
            with torch.no_grad():
                for parameter, old, momentum, decay in lars_before:
                    assert parameter.grad is not None
                    update = parameter.grad + decay * old
                    pnorm, unorm = float(old.norm()), float(update.norm())
                    ratio = (
                        0.001 * pnorm / unorm
                        if (old.ndim >= 2 and pnorm > 0 and unorm > 0)
                        else 1.0
                    )
                    expected = old - compiled.optimiser.lr_at(step) * (
                        0.9 * momentum + ratio * update
                    )
                    torch.testing.assert_close(
                        parameter, expected, rtol=1e-5, atol=1e-8
                    )
            if not bool(torch.isfinite(result.loss.total)) or not math.isfinite(
                result.grad_norm
            ):
                raise RuntimeError("non-finite BYOL training loss/gradient")
            for name in compiled.trainable:
                gradients = [
                    p.grad
                    for p in active_run.graph[name].parameters()
                    if p.grad is not None
                ]
                if not gradients or not all(
                    bool(torch.isfinite(g).all()) for g in gradients
                ):
                    raise RuntimeError(f"missing/non-finite BYOL gradient: {name}")
                if step == 0 and not any(bool(g.ne(0).any()) for g in gradients):
                    raise RuntimeError(f"zero intended BYOL gradient: {name}")
            if updates[0] - before_updates != int(teacher is not None):
                raise RuntimeError("BYOL target must update exactly once per step")
            if teacher is not None:
                last_teacher = teacher
            metrics[f"{arm}_{compiled.name}_final_loss"] = float(
                result.loss.total.detach()
            )
            metrics[f"{arm}_{compiled.name}_final_grad_norm"] = result.grad_norm
            return result

        with (
            patch.object(executors, "build_population", return_value=population),
            patch.object(executors, "_step", traced_step),
            patch.object(EMATeacher, "update", checked_update),
        ):
            result, view_trace = fit_with_trace(
                run,
                {s.name: data for s in run.stages},
                seed=base + 10000 + (STREAM_STRIDE if arm == "no_pretrain" else 0),
            )
        for stage in run.stages:
            digest = traces[stage.name].hexdigest()
            if stage.name in paired_traces and digest != paired_traces[stage.name]:
                raise RuntimeError("BYOL actual batch/mask traces differ across arms")
            paired_traces[stage.name] = digest
            if result.stage(stage.name).population is not population:
                raise RuntimeError("BYOL stage replaced shared fitted scales/masks")
        if result.stage("joint_fit").seed != base + 10000 + STREAM_STRIDE:
            raise RuntimeError("BYOL downstream execution stream mismatch")
        heads = {
            k: v
            for k, v in start.items()
            if k.startswith(
                (
                    "_components.tarnet_head.",
                    "_components.categorical_propensity.",
                )
            )
        }
        require_equal(heads, transitions["joint_fit"])
        if arm == "no_pretrain":
            require_equal(start, transitions["joint_fit"])
        else:
            if not moved or updates[0] != pretrain_steps:
                raise RuntimeError("BYOL target did not move through the full stage")
            if paired_views is not None and paired_views != view_trace:
                raise RuntimeError("BYOL actual view draws differ across arms")
            paired_views = view_trace
            if int(view_trace["calls"]) != 2 * pretrain_steps:
                raise RuntimeError("BYOL requires exactly two cached views per step")
            checkpoint = result.stage("pretrain").checkpoint
            saved = {
                "_components." + k: v
                for k, v in {
                    **checkpoint.parameters,
                    **checkpoint.buffers,
                }.items()
            }
            require_equal(saved, transitions["joint_fit"])
            for name in run.stages[0].trainable:
                if not any(
                    not torch.equal(v, start["_components." + k])
                    for k, v in checkpoint.parameters.items()
                    if k.startswith(name + ".")
                ):
                    raise RuntimeError(f"BYOL pretraining did not update {name}")
            require_equal(
                {
                    k: v
                    for k, v in saved.items()
                    if k.startswith(
                        (
                            "_components.byol_projector.",
                            "_components.byol_predictor.",
                        )
                    )
                },
                snapshot(run.graph),
            )
        # Deviation 4, as amended: `joint_fit` declares no encoder, so the
        # transferred backbone must be bit-identical after the stage. This is
        # the linear-evaluation posture, executed rather than declared.
        require_equal(
            {
                k: v
                for k, v in transitions["joint_fit"].items()
                if k.startswith("_components.mlp_encoder.")
            },
            snapshot(run.graph),
        )
        run.graph.eval()
        with torch.no_grad():
            values = run.graph.evaluate(
                heldout,
                schema=schema,
                only=("mlp_encoder", "tarnet_head", "categorical_propensity"),
            )
            outcome = values[Port.Y_GIVEN_XT]
            assert isinstance(outcome, GaussianOutcome)
            metrics[f"{arm}_outcome_nll"] = float(
                -outcome.log_prob(heldout.y, heldout.t).mean()
            )
            propensity = values[Port.T_GIVEN_X]
            assert isinstance(propensity, CategoricalTreatment)
            metrics[f"{arm}_treatment_nll"] = float(
                -propensity.log_prob(heldout.t).mean()
            )
            effect = (
                outcome.mean(torch.ones(test_rows, dtype=torch.long))
                - outcome.mean(torch.zeros(test_rows, dtype=torch.long))
            ).reshape(-1) * population.statistics["y_scale"]
            metrics[f"{arm}_effect_rmse"] = float(
                (effect - test.true_effect.reshape(-1)).square().mean().sqrt()
            )
    metrics.update(paired_metrics(metrics))
    if not all(math.isfinite(v) for v in metrics.values()):
        raise RuntimeError("non-finite BYOL diagnostics")
    return metrics


def paired_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    """Form contrasts within seed before the runner computes sampling error."""

    def alignment_gap(arm: str) -> float:
        return metrics[f"{arm}_view_alignment"] - metrics["full_view_alignment"]

    def rank_gap(arm: str) -> float:
        return (
            metrics["full_encoder_effective_rank"]
            - metrics[f"{arm}_encoder_effective_rank"]
        )

    return {
        # Card §6.4's two scored attribution contrasts. An arm sitting closer
        # to the objective's degenerate optimum than the full arm is the
        # paper's own mechanism statement, measured before transfer.
        "ema_alignment_gap": alignment_gap("zero_decay"),
        "ema_rank_gap": rank_gap("zero_decay"),
        # Retained as the §6.4 budget and the withdrawn superiority statistic.
        "pretraining_outcome_nll_cost": (
            metrics["full_outcome_nll"] - metrics["no_pretrain_outcome_nll"]
        ),
        "encoder_effective_rank": metrics["full_encoder_effective_rank"],
        "ema_outcome_nll_gain": (
            metrics["zero_decay_outcome_nll"] - metrics["full_outcome_nll"]
        ),
        # Predictor diagnostics, and deviation 7's own control.
        "predictor_alignment_gap": alignment_gap("no_predictor"),
        "predictor_outcome_nll_gain": (
            metrics["no_predictor_outcome_nll"] - metrics["full_outcome_nll"]
        ),
        "predictor_encoder_rank_gain": rank_gap("no_predictor"),
        "source_ema_alignment_gap": alignment_gap("source_ema"),
        "source_ema_rank_gap": rank_gap("source_ema"),
        "source_ema_outcome_nll_gain": (
            metrics["source_ema_outcome_nll"] - metrics["full_outcome_nll"]
        ),
    }
