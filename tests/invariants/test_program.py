"""Tier 0 — ordered program execution and stage transitions (P8)."""

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch
from xty2.core import (
    CompileError,
    ComponentGraph,
    Program,
    Recipe,
    Stage,
    TrainingError,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.objectives import ObservedOutcomeNLL
from xty2.training import RunDirectory, is_read_only, run_program, run_stage

from tests.invariants.conftest import (
    HIDDEN,
    ToyEncoder,
    ToyOutcomeHead,
    make_batch,
    make_schema,
    optimiser,
    stage,
)


def _fit(name: str, *, initialise_from: str | None = None) -> Stage:
    return stage(
        name=name,
        objectives=(Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="mean"),),
        trainable=("encoder", "outcome_head"),
        initialise_from=initialise_from,
        steps=1,
    )


def _recipe() -> Recipe:
    return Recipe(
        name="staged",
        schema=make_schema(),
        system=ComponentGraph([ToyEncoder(width=HIDDEN), ToyOutcomeHead()]),
        program=Program(
            (
                _fit("base"),
                _fit("branch", initialise_from="base"),
                _fit("replay", initialise_from="base"),
            )
        ),
        card="docs/recipes/staged.md",
    )


def _sources() -> dict[str, list[XTYBatch]]:
    batch = make_batch()
    return {name: [batch] for name in ("base", "branch", "replay")}


def test_recipe_coerces_the_legacy_tuple_surface_to_a_program() -> None:
    first = _fit("first")
    recipe = Recipe(
        name="coerced",
        schema=make_schema(),
        system=ComponentGraph([ToyEncoder(width=HIDDEN), ToyOutcomeHead()]),
        program=(first,),
        card="docs/recipes/coerced.md",
    )
    assert isinstance(recipe.program, Program)
    assert tuple(recipe.program) == (first,)
    assert recipe.program[0] is first


def test_program_is_an_immutable_ordered_sequence() -> None:
    program = _recipe().program
    assert [entry.name for entry in program] == ["base", "branch", "replay"]
    assert program.stage("branch") is program[1]
    with pytest.raises(FrozenInstanceError):
        program.stages = ()  # type: ignore[misc]
    with pytest.raises(TypeError):
        program.stages[0] = program.stages[1]  # type: ignore[index]


def test_initialise_from_may_only_name_an_earlier_stage() -> None:
    with pytest.raises(CompileError, match="unknown stage"):
        Program((_fit("first"), _fit("second", initialise_from="missing")))
    with pytest.raises(CompileError, match="not an earlier stage"):
        Program((_fit("first", initialise_from="second"), _fit("second")))
    with pytest.raises(CompileError, match="not an earlier stage"):
        Program((_fit("first", initialise_from="first"),))


def test_duplicate_stage_names_are_rejected_by_the_program() -> None:
    entry = _fit("same")
    with pytest.raises(CompileError, match="more than one stage called"):
        Program((entry, replace(entry)))


def test_the_plan_prints_checkpoint_transitions() -> None:
    plan = compile(_recipe()).plan.render()
    assert "stage base\n  rows: all\n  executor: gradient\n  steps: 1" in plan
    assert (
        "stage branch\n  rows: all\n  executor: gradient\n"
        "  initialise from: base\n  steps: 1"
    ) in plan
    assert (
        "stage replay\n  rows: all\n  executor: gradient\n"
        "  initialise from: base\n  steps: 1"
    ) in plan


def test_multistage_card_bindings_are_keyed_by_stage() -> None:
    first = replace(_fit("first"), optimiser=optimiser(lr=0.01), steps=2)
    second = replace(_fit("second"), optimiser=optimiser(lr=0.02), steps=3)
    recipe = replace(_recipe(), program=Program((first, second)))

    hyperparameters = compile(recipe).plan.hyperparameters
    assert hyperparameters["optimisation.lr"] == {"first": 0.01, "second": 0.02}
    assert hyperparameters["optimisation.total_steps_or_epochs"] == {
        "first": 2,
        "second": 3,
    }


def test_branches_start_from_the_named_checkpoint_not_the_previous_stage() -> None:
    torch.manual_seed(19)
    run = compile(_recipe())
    result = run_program(run, _sources(), seed=7)

    branch = result.stage("branch").checkpoint
    replay = result.stage("replay").checkpoint
    assert result.seed == 7
    assert [entry.seed for entry in result.stages] == [7, 8, 9]
    for name, value in branch.parameters.items():
        assert torch.equal(value, replay.parameters[name])


def test_repeated_program_runs_reset_to_the_state_captured_by_compile() -> None:
    torch.manual_seed(29)
    run = compile(_recipe())
    sources = _sources()
    first = run_program(run, sources, seed=11)
    second = run_program(run, sources, seed=11)

    assert [stage.trace for stage in first.stages] == [
        stage.trace for stage in second.stages
    ]
    for first_stage, second_stage in zip(first.stages, second.stages, strict=True):
        for name, value in first_stage.checkpoint.parameters.items():
            assert torch.equal(value, second_stage.checkpoint.parameters[name])


def test_repeated_standalone_stage_runs_reset_to_compile_time_state() -> None:
    torch.manual_seed(31)
    run = compile(_recipe())
    batch = make_batch()
    first = run_stage(run, "base", [batch], seed=13)
    second = run_stage(run, "base", [batch], seed=13)

    assert first.trace == second.trace
    for name, value in first.checkpoint.parameters.items():
        assert torch.equal(value, second.checkpoint.parameters[name])


def test_a_later_stage_cannot_mutate_an_earlier_checkpoint() -> None:
    torch.manual_seed(23)
    result = run_program(compile(_recipe()), _sources(), seed=3)
    base = result.stage("base").checkpoint
    before = {name: value.clone() for name, value in base.parameters.items()}
    for value in result.stage("replay").checkpoint.parameters.values():
        value.zero_()
    for name, value in before.items():
        assert torch.equal(base.parameters[name], value)


def test_a_stage_with_a_transition_cannot_be_run_out_of_context() -> None:
    run = compile(_recipe())
    with pytest.raises(TrainingError, match="through run_program"):
        run_stage(run, "branch", [make_batch()], seed=0)


def test_program_requires_exactly_one_batch_source_per_stage() -> None:
    run = compile(_recipe())
    before = {
        name: value.detach().clone() for name, value in run.graph.named_parameters()
    }
    with pytest.raises(TrainingError, match=r"missing .*replay"):
        run_program(run, {"base": [], "branch": []}, seed=0)
    with pytest.raises(TrainingError, match=r"unexpected .*typo"):
        run_program(run, _sources() | {"typo": []}, seed=0)
    assert all(
        torch.equal(value, dict(run.graph.named_parameters())[name])
        for name, value in before.items()
    )


def test_a_program_writes_one_immutable_checkpoint_per_stage(tmp_path: Path) -> None:
    directory = RunDirectory.create(tmp_path / "run")
    result = run_program(compile(_recipe()), _sources(), seed=0, run_dir=directory)
    assert set(result.checkpoints) == {"base", "branch", "replay"}
    for name in result.checkpoints:
        path = directory.root / "stages" / name / "checkpoint.pt"
        assert path.exists()
        assert is_read_only(path)
