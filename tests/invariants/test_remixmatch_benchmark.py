"""Tier 0: the ReMixMatch baseline has no pseudo-target path or relaxed gate."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xty2.core import TrainContext, compile
from xty2.evaluation.benchmarks import remixmatch as benchmark
from xty2.evaluation.reporting import load_reproduction_spec
from xty2.recipes import remixmatch

from tests.invariants.test_remixmatch import _batch, _schema

ROOT = Path(__file__).parents[2]


def test_no_remixmatch_retains_the_causal_stack_and_labelled_view() -> None:
    full = remixmatch(_schema())
    baseline = benchmark._no_remixmatch(full)
    run = compile(baseline)
    stage = run.stage("joint_fit")
    assert {item.name for item in stage.objectives} == {
        "observed_outcome_nll",
        "observed_treatment_nll",
        "missing_treatment_marginal_nll",
    }
    assert {item.realisation.view for item in stage.passes} == {"identity", "strong_x"}
    assert not baseline.mixes
    assert baseline.system is full.system
    assert baseline.data == full.data
    assert baseline.views == (replace(full.view("strong_x"), draws=1),)
    batch = _batch()
    assert torch.equal(
        baseline.view("strong_x").apply(batch, full.schema, rng_key=42).x,
        full.view("strong_x").apply(batch, full.schema, rng_key=42).x,
    )
    for field in ("sampler", "teacher", "steps", "rows"):
        assert getattr(baseline.program[0], field) == getattr(full.program[0], field)
    assert baseline.program[0].trainable == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )
    assert baseline.program[0].optimiser.weight_decay.components == (
        "mlp_encoder",
        "categorical_propensity",
    )
    assert (
        replace(
            baseline.program[0].optimiser,
            weight_decay=full.program[0].optimiser.weight_decay,
        )
        == full.program[0].optimiser
    )
    assert (
        replace(
            baseline.program[0].optimiser.weight_decay,
            components=full.program[0].optimiser.weight_decay.components,
        )
        == full.program[0].optimiser.weight_decay
    )
    for name in ("observed_outcome_nll", "missing_treatment_marginal_nll"):
        assert next(
            x for x in baseline.program[0].objectives if x.name == name
        ) == next(x for x in full.program[0].objectives if x.name == name)
    labelled = next(
        x for x in baseline.program[0].objectives if x.name == "observed_treatment_nll"
    )
    original = next(
        x
        for x in full.program[0].objectives
        if x.name == "mixed_labelled_treatment_nll"
    )
    assert labelled.weight == original.weight
    assert labelled.reduction == original.reduction == "mean"
    # Constructing the ablation must not remove mechanics from the full arm.
    assert len(full.mixes) == 1 and len(full.program[0].objectives) == 6


def test_baseline_labelled_loss_cannot_train_from_unlabelled_partners() -> None:
    schema = _schema()
    run = compile(benchmark._no_remixmatch(remixmatch(schema)))
    stage = run.stage("joint_fit")
    objective = next(
        x.objective for x in stage.objectives if x.name == "observed_treatment_nll"
    )
    batch = _batch()
    changed_x = batch.x.clone()
    changed_x[batch.t_missing] = 100 * torch.randn_like(changed_x[batch.t_missing])
    poisoned = batch.replace(
        x=changed_x,
        t=torch.where(batch.t_observed, batch.t, (batch.t + 1) % 4),
    )
    values = []
    for candidate in (batch, poisoned):
        inputs = candidate.x.clone().requires_grad_()
        candidate = candidate.replace(x=inputs)
        state = run.state(stage, candidate, rng_key=13)
        rows = torch.nonzero(candidate.t_observed, as_tuple=False).flatten()
        # No anchor state is supplied: a residual pseudo-target read must fail.
        loss = objective.compute(
            state, candidate, rows, TrainContext(global_step=0, schema=schema)
        ).value
        (gradient,) = torch.autograd.grad(loss, inputs)
        assert torch.count_nonzero(gradient[candidate.t_missing]) == 0
        assert torch.count_nonzero(gradient[candidate.t_observed]) > 0
        values.append(float(loss.detach()))
    assert values[0] == values[1]


def test_new_protocol_preserves_the_noisy_true_marginal_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Successful target diagnostics must not rescue the original guardrail."""
    rows = []
    for index in range(10):
        row = dict.fromkeys(benchmark._MARGINAL_DIAGNOSTICS, 0.0)
        row.update(
            {
                "full_no_remixmatch_student_ratio": 0.8,
                "full_no_remixmatch_ema_ratio": 0.8,
                "full_no_alignment_student_ratio": 0.8,
                "full_no_alignment_ema_ratio": 0.8,
                "full_no_mixup_student_ratio": 0.9,
                "full_no_mixup_ema_ratio": 0.9,
                "outcome_ratio": 1.0,
                "alignment_advantage": 0.0103444 + (-0.0915 if index < 5 else 0.0915),
                "pretext_accuracy": 0.4,
                "lambda_min": 0.5,
                "lambda_max": 1.0,
                "target_copy_agreement": 0.7,
                "anchor_entropy": 0.5,
                "lambda_mean": 0.8,
                "aligned_marginal_l1": 0.1,
                "unaligned_marginal_l1": 0.1,
                "alignment_labelled_marginal_L1_advantage": 0.5,
                "alignment_true_unlabelled_marginal_L1_advantage": 0.5,
            }
        )
        rows.append(row)

    def replicated(
        function: object, count: int, *, workers: int
    ) -> tuple[dict[str, float], ...]:
        assert count == 10
        return tuple(rows)

    monkeypatch.setattr(benchmark, "parallel_replicates", replicated)
    spec = load_reproduction_spec(ROOT / "docs/recipes/remixmatch.md")
    result = benchmark.run(spec, "test-only", "2026-09-06", 1, tmp_path)
    required = [m for m in result.metrics if m.relation != "info"]
    assert len(required) == 9
    assert sum(m.passed is True for m in required) == 8
    gate = result.metric("alignment_marginal_L1_advantage")
    assert gate.mean == pytest.approx(0.0103444)
    assert gate.stderr == pytest.approx(0.0305)
    assert gate.passed is False
    assert result.status == "deviating"
    assert all(
        result.metric(name).passed is None for name in benchmark._MARGINAL_DIAGNOSTICS
    )
    assert "supervised" not in " ".join(m.name for m in result.metrics)
    assert result.metric("full_vs_no_remixmatch_outcome_NLL_ratio").passed is True
