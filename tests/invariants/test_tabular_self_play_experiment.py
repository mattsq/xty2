"""Guard the external pilot's leakage and gradient-probe boundaries."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch
from experiments import tabular_self_play_tuning as tuning
from experiments.tabular_self_play import (
    CONFIRMATION_SEEDS,
    POLICIES,
    TASKS,
    Config,
    Predictor,
    Task,
    alignment,
    analyse,
    condense,
    fit_logistic_probe,
    fixture,
    interval,
    policy_weights,
    readout,
    run_arm,
    task_loss,
    trace_summary,
    view,
)

SMALL = Config(
    steps=9,
    batch_size=8,
    train_rows=64,
    validation_rows=16,
    test_rows=16,
    labeled_rows=16,
)


def test_target_is_inaccessible_for_every_context() -> None:
    x = torch.arange(60, dtype=torch.float32).reshape(10, 6)
    changed = x.clone()
    changed[:, 2] += 1000
    for context in ("full", "sparse", "jitter", "replace"):
        task = Task(2, context)
        original_view = view(x, x, task, seed=42)
        changed_view = view(changed, changed, task, seed=42)
        # The donor pool changes in the target column too; no path may leak it.
        assert all(
            torch.equal(a, b) for a, b in zip(original_view, changed_view, strict=True)
        )
        assert bool((original_view[1][:, 2] == 0).all())
        assert bool((original_view[1].sum(dim=1) >= 1).all())


def test_probes_leave_parameters_and_adam_moments_unchanged() -> None:
    torch.manual_seed(42)
    x = torch.randn(20, 6)
    model = Predictor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    initial = tuple(p.detach().clone() for p in model.parameters())
    task_loss(model, x[:8], x, Task(1, "full"), 1).backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    history = [initial, tuple(p.detach().clone() for p in model.parameters())]
    before = tuple(p.detach().clone() for p in model.parameters())
    moments = tuple(
        optimizer.state[p]["exp_avg_sq"].clone() for p in model.parameters()
    )
    gradients = torch.autograd.grad(
        task_loss(model, x[8:], x, Task(2, "jitter"), 3), tuple(model.parameters())
    )
    score = alignment(model, optimizer, history, 1, gradients)
    assert torch.isfinite(torch.tensor(score))
    assert all(
        torch.equal(p, saved)
        for p, saved in zip(model.parameters(), before, strict=True)
    )
    assert all(
        torch.equal(optimizer.state[p]["exp_avg_sq"], saved)
        for p, saved in zip(model.parameters(), moments, strict=True)
    )


def test_binary_probe_fits_the_reported_likelihood() -> None:
    x = torch.linspace(-2, 2, 40)
    z = torch.stack((x, torch.ones_like(x)), dim=1)
    labels = (x > 0).float()
    weights = fit_logistic_probe(z, labels)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ weights, labels)
    assert float(loss) < 0.25


def test_shuffled_trace_records_reward_assignment() -> None:
    result = run_arm("dependent", 310_000, "shuffled", SMALL)
    trace = result["trace"]
    assert isinstance(trace, list)
    assert len(trace) == 1
    entry = trace[0]
    assert isinstance(entry, dict)
    raw, assigned, permutation = (
        entry["rewards"],
        entry["assigned_rewards"],
        entry["reward_permutation"],
    )
    assert isinstance(raw, list)
    assert isinstance(assigned, list)
    assert isinstance(permutation, list)
    assert permutation != list(range(24))
    assert torch.allclose(
        torch.tensor(assigned), torch.tensor([raw[i] for i in permutation])
    )


def test_splits_are_disjoint_and_scaled_from_training_rows() -> None:
    split = fixture("dependent", 1, SMALL)
    assert (len(split.x), len(split.validation_x), len(split.test_x)) == (64, 16, 16)
    assert torch.allclose(split.x.mean(dim=0), torch.zeros(6), atol=1e-5)
    rows = {tuple(r.tolist()) for r in split.x}
    assert not rows & {tuple(r.tolist()) for r in split.validation_x}
    assert not rows & {tuple(r.tolist()) for r in split.test_x}


def test_validation_endpoint_never_reads_test_rows() -> None:
    split = fixture("interaction", 3, SMALL)
    torch.manual_seed(0)
    model = Predictor()
    before = readout(model, split, "validation", SMALL)
    poisoned = replace(split, test_x=split.test_x + 100, test_y=1 - split.test_y)
    assert readout(model, poisoned, "validation", SMALL) == before
    assert readout(model, poisoned, "test", SMALL) != readout(
        model, split, "test", SMALL
    )


@pytest.mark.parametrize("arm", ["uniform", "fixed_dependent"])
def test_skipping_discarded_probes_leaves_fixed_training_identical(arm: str) -> None:
    probed = run_arm("dependent", 310_000, arm, SMALL)
    skipped = run_arm("dependent", 310_000, arm, replace(SMALL, fixed_probes=False))
    assert probed["probe_gradients"] == len(TASKS)
    assert skipped["probe_gradients"] == 0
    for key in ("bce", "masked_mse", "counts", "feature_std"):
        assert probed[key] == skipped[key]


def test_no_pretraining_reference_reads_out_the_paired_initial_encoder() -> None:
    result = run_arm("independent", 310_000, "none", SMALL)
    assert result["update_gradients"] == 0
    assert result["probe_gradients"] == 0
    torch.manual_seed(310_002)
    initial = readout(
        Predictor(), fixture("independent", 310_001, SMALL), "validation", SMALL
    )
    assert result["bce"] == initial["bce"]
    trained = run_arm("independent", 310_000, "uniform", SMALL)
    assert trained["bce"] != result["bce"]


def test_policy_grid_is_normalised_with_declared_support() -> None:
    assert len(set(POLICIES)) == len(POLICIES)
    for name in POLICIES:
        weights = policy_weights(name)
        assert torch.isclose(weights.sum(), torch.tensor(1.0))
    context = policy_weights("context:jitter")
    support = [t.context == "jitter" for t in TASKS]
    assert (context > 0).tolist() == support
    target = policy_weights("target:4")
    assert (target > 0).tolist() == [t.target == 4 for t in TASKS]


def test_current_loss_trace_is_a_pure_loss_controller() -> None:
    result = run_arm("dependent", 310_000, "current_loss", SMALL)
    summary = trace_summary(result)
    assert summary["reward_vs_loss"] == pytest.approx(1.0)
    assert sum(cast(list[float], summary["target_frequency"])) == pytest.approx(1.0)
    (round_,) = condense(cast(list[dict[str, list[float]]], result["trace"]))
    assert round_["reward_vs_loss"] == pytest.approx(1.0)
    assert sum(cast(list[float], round_["target_probability"])) == pytest.approx(1.0)
    assert sum(cast(list[float], round_["context_probability"])) == pytest.approx(1.0)
    alignment_summary = trace_summary(run_arm("dependent", 310_000, "alignment", SMALL))
    assert alignment_summary["reward_vs_loss"] != pytest.approx(1.0)


def test_tuning_selects_on_validation_only(tmp_path: Path) -> None:
    seen: list[Config] = []

    def fake(
        jobs: list[tuple[str, int, str, Config, str | None]], workers: int
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for kind, seed, arm, cfg, policy in jobs:
            seen.append(cfg)
            # A unique minimum per fixture: target:1 on dependent only.
            bce = 0.5 - 0.01 * cfg.steps / 1024
            if policy == "target:1" and kind == "dependent":
                bce -= 0.1
            rows.append(
                {"fixture": kind, "seed": seed, "arm": arm, "policy": policy}
                | {"bce": bce, "trace": []}
            )
        return rows

    output = tmp_path / "tuning"
    with (
        patch.object(tuning, "run_jobs", fake),
        patch.object(tuning, "provenance", lambda workers: {}),
        patch("subprocess.check_output", lambda *a, **k: ""),
        patch("sys.argv", ["tuning", "--output", str(output)]),
    ):
        tuning.main()
    assert seen
    assert {cfg.endpoint for cfg in seen} == {"validation"}
    selection = json.loads((output / "selection.json").read_text())
    assert selection["config"]["steps"] == 1024
    assert selection["config"]["endpoint"] == "validation"
    assert selection["selected"] == {
        "dependent": "target:1",
        "interaction": "uniform",
        "independent": "uniform",
    }


def _rows(offsets: dict[str, float]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in ("dependent", "interaction", "independent"):
        for i, seed in enumerate(CONFIRMATION_SEEDS):
            # Seed-level noise shared by every arm: pairing must cancel it.
            common = 0.05 * ((-1) ** i) + 0.001 * i
            for arm in ("none", "uniform", "fixed_dependent", "tuned"):
                bce = 0.4 + common + offsets.get(arm, 0.0)
                rows.append({"fixture": kind, "seed": seed, "arm": arm, "bce": bce})
            for arm in ("alignment", "current_loss", "shuffled"):
                bce = 0.4 + common + offsets.get(arm, 0.0) + 0.002 * (i % 3)
                rows.append({"fixture": kind, "seed": seed, "arm": arm, "bce": bce})
    return rows


def test_analysis_pairs_by_seed_and_requires_both_fixed_controls() -> None:
    better = analyse(_rows({"alignment": -0.02}))
    contrast = better["contrasts"]
    assert isinstance(contrast, dict)
    interval = contrast["dependent"]["alignment-uniform"]
    # Unpaired, the +-0.05 seed noise would swamp a 0.02 gain.
    assert interval["high"] < 0 < interval["low"] + 0.03
    assert better["adaptive_supported"] is True
    # Beating uniform alone is not enough when the tuned control is as good.
    tied = analyse(_rows({"alignment": -0.02, "tuned": -0.02}))
    assert tied["adaptive_supported"] is False


def test_interval_is_the_declared_ten_seed_t_interval() -> None:
    # Alternating +-1: mean 0, sample sd sqrt(10/9), so the half-width is
    # 2.2622 * sqrt(10/9) / sqrt(10) = 2.2622 / 3.
    result = interval([(-1.0) ** i for i in range(10)])
    assert result["mean"] == pytest.approx(0.0)
    assert result["high"] == pytest.approx(2.2622 / 3)
    assert result["low"] == pytest.approx(-2.2622 / 3)
    with pytest.raises(ValueError):
        interval([0.0] * 9)
