"""Tier 1 — a short end-to-end ReMixMatch wiring fit."""

from __future__ import annotations

from dataclasses import replace

import torch
from xty2.core import Program, compile
from xty2.objectives import AnchoredLabelGuess
from xty2.recipes import remixmatch
from xty2.recipes.remixmatch import GUESS_OWNER
from xty2.training import run_stage

from tests.smoke.test_fixmatch import SEPARATED, _dataset, _populations, _schema


def test_all_terms_are_finite_and_the_state_advances() -> None:
    train, _ = _populations(SEPARATED, seed=90_001)
    recipe = remixmatch(_schema())
    stage = replace(recipe.program[0], steps=3)
    run = compile(replace(recipe, program=Program((stage,))))
    result = run_stage(run, "joint_fit", _dataset(train), seed=100_000)
    assert len(result.records) == 3
    assert all(torch.isfinite(torch.tensor(record.total)) for record in result.records)
    assert all(len(record.terms) == 6 for record in result.records)
    state = result.objective_states[GUESS_OWNER]
    assert isinstance(state, AnchoredLabelGuess)
    assert state.last_prepared_step == 2
