"""Guard the external pilot's leakage and gradient-probe boundaries."""

from __future__ import annotations

import torch
from experiments.tabular_self_play import Predictor, Task, alignment, task_loss, view


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
