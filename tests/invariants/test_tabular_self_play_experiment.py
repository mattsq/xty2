"""Guard the external pilot's leakage and gradient-probe boundaries."""

from __future__ import annotations

import torch
from experiments.tabular_self_play import (
    Config,
    Predictor,
    Task,
    alignment,
    fit_logistic_probe,
    run_arm,
    task_loss,
    view,
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
    cfg = Config(steps=9, batch_size=8, train_rows=64, test_rows=16, labeled_rows=16)
    result = run_arm("dependent", 310_000, "shuffled", cfg)
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
