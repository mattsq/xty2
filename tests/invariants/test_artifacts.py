"""Tier 0 — artifacts and the run directory (`DESIGN.md` §7.1).

§7.1 makes two claims that are worth nothing unless something enforces them,
and each test here is one half of one of them.

**Provenance is derived, not declared.** A `Checkpoint` cannot be constructed
at all except by the executor factory, which computes `trained_on_row_ids` from
the batches the loop stepped on and `plan_digest` from the plan it stepped
under. The earlier draft of §7.1 wrote these as ordinary keyword arguments,
which would have let a producer assert `out_of_fold` without ever being
checked — "a guardrail that any caller can talk its way past is not a
guardrail".

**Artifacts are immutable.** Not only frozen: `frozen=True` stops field
rebinding and does nothing about an in-place write to a tensor leaf, and a
file on disk is mutable until something says otherwise. So tensors are cloned
in both directions and the written file carries no write bit.
"""

from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import Tensor
from xty2.core import (
    ArtifactError,
    CompiledRun,
    ComponentGraph,
    Recipe,
    Weighted,
    compile,
)
from xty2.objectives import ObservedOutcomeNLL
from xty2.training import Checkpoint, RunDirectory, emit_checkpoint, is_read_only

from tests.invariants.conftest import (
    HIDDEN,
    ToyEncoder,
    ToyOutcomeHead,
    make_schema,
    stage,
)


def _run() -> CompiledRun:
    recipe = Recipe(
        name="two_head",
        schema=make_schema(),
        system=ComponentGraph([ToyEncoder(width=HIDDEN), ToyOutcomeHead()]),
        program=(
            stage(
                objectives=(
                    Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="mean"),
                ),
                trainable=("encoder", "outcome_head"),
            ),
        ),
        card="docs/recipes/two_head.md",
    )
    return compile(recipe)


def _checkpoint(**overrides: object) -> Checkpoint:
    run = _run()
    parameters = list(run.graph["outcome_head"].named_parameters())
    defaults: dict[str, object] = {
        "row_ids": torch.tensor([3, 1, 1, 2], dtype=torch.long),
        "steps": 2,
        "seed": 0,
    }
    resolved = defaults | overrides
    return emit_checkpoint(
        run,
        run.stage("fit"),
        parameters,
        cast("Tensor", resolved.pop("row_ids")),
        steps=cast("int", resolved.pop("steps")),
        seed=cast("int", resolved.pop("seed")),
        **cast("dict[str, Any]", resolved),
    )


# ---------------------------------------------------------------------------
# Provenance is derived, not declared
# ---------------------------------------------------------------------------


def test_a_checkpoint_cannot_be_constructed_directly() -> None:
    with pytest.raises(ArtifactError, match="executor factory"):
        Checkpoint(
            recipe="r",
            stage="fit",
            fold=None,
            trained_on_row_ids=torch.zeros(1, dtype=torch.long),
            parameters={},
            components=(),
            steps=1,
            seed=0,
            plan_digest="",
        )


def test_a_forged_token_is_still_rejected() -> None:
    with pytest.raises(ArtifactError, match="executor factory"):
        Checkpoint(
            recipe="r",
            stage="fit",
            fold=None,
            trained_on_row_ids=torch.zeros(1, dtype=torch.long),
            parameters={},
            components=(),
            steps=1,
            seed=0,
            plan_digest="",
            issued_by=object(),
        )


def test_the_factory_sorts_and_deduplicates_the_row_ids() -> None:
    # The leakage check of §7.2 asks "is this row in the set this fit saw?", so
    # the set is what the field holds — not the sequence of batches, in which
    # a row appears once per epoch.
    checkpoint = _checkpoint()
    assert checkpoint.trained_on_row_ids.tolist() == [1, 2, 3]


def test_the_factory_rejects_row_ids_that_are_not_row_ids() -> None:
    with pytest.raises(Exception, match="long tensor"):
        _checkpoint(row_ids=torch.tensor([1.0, 2.0]))


def test_the_digest_changes_when_the_plan_does() -> None:
    run = _run()
    other = compile(
        Recipe(
            name="two_head",
            schema=make_schema(),
            system=ComponentGraph([ToyEncoder(width=HIDDEN), ToyOutcomeHead()]),
            program=(
                stage(
                    objectives=(
                        Weighted(ObservedOutcomeNLL(), weight=0.5, reduction="mean"),
                    ),
                    trainable=("encoder", "outcome_head"),
                ),
            ),
            card="docs/recipes/two_head.md",
        )
    )
    assert run.plan.digest != other.plan.digest
    assert run.plan.digest == compile(run.recipe).plan.digest


# ---------------------------------------------------------------------------
# Immutability, in memory
# ---------------------------------------------------------------------------


def test_writing_into_the_tensor_the_factory_was_given_does_not_reach_it() -> None:
    rows = torch.tensor([5, 6], dtype=torch.long)
    checkpoint = _checkpoint(row_ids=rows)
    rows[0] = 99
    assert checkpoint.trained_on_row_ids.tolist() == [5, 6]


def test_writing_into_what_a_reader_gets_back_does_not_reach_it() -> None:
    checkpoint = _checkpoint()
    borrowed = checkpoint.row_ids
    borrowed[0] = 99
    assert checkpoint.trained_on_row_ids.tolist() == [1, 2, 3]
    name = next(iter(checkpoint.parameters))
    parameter = checkpoint.parameter(name)
    parameter.zero_()
    assert not torch.equal(parameter, checkpoint.parameters[name])


def test_a_saved_parameter_is_detached_from_the_live_module() -> None:
    # A checkpoint that held the live parameters would silently follow the next
    # stage's training, and "restore from the checkpoint" would restore nothing.
    run = _run()
    checkpoint = _checkpoint()
    with torch.no_grad():
        for parameter in run.graph["outcome_head"].parameters():
            parameter.add_(1.0)
    saved = next(iter(checkpoint.parameters.values()))
    live = next(iter(run.graph["outcome_head"].parameters()))
    assert not torch.equal(saved, live.detach())


def test_an_unknown_parameter_says_what_is_there() -> None:
    checkpoint = _checkpoint()
    with pytest.raises(ArtifactError, match=r"loc\.weight"):
        checkpoint.parameter("nothing.like.this")


# ---------------------------------------------------------------------------
# Immutability, on disk
# ---------------------------------------------------------------------------


def test_a_written_checkpoint_round_trips(tmp_path: Path) -> None:
    checkpoint = _checkpoint()
    path = checkpoint.save(tmp_path / "checkpoint.pt")
    reloaded = Checkpoint.load(path)
    assert reloaded.recipe == checkpoint.recipe
    assert reloaded.stage == checkpoint.stage
    assert reloaded.fold is None
    assert reloaded.components == checkpoint.components
    assert reloaded.steps == checkpoint.steps
    assert reloaded.seed == checkpoint.seed
    assert reloaded.plan_digest == checkpoint.plan_digest
    assert torch.equal(reloaded.trained_on_row_ids, checkpoint.trained_on_row_ids)
    assert set(reloaded.parameters) == set(checkpoint.parameters)
    for name, parameter in checkpoint.parameters.items():
        assert torch.equal(reloaded.parameters[name], parameter)


def test_a_written_checkpoint_is_read_only(tmp_path: Path) -> None:
    path = _checkpoint().save(tmp_path / "checkpoint.pt")
    assert is_read_only(path)


def test_writing_over_a_checkpoint_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    _checkpoint().save(path)
    with pytest.raises(ArtifactError, match="immutable"):
        _checkpoint().save(path)


def test_loading_something_that_is_not_a_checkpoint_is_refused(
    tmp_path: Path,
) -> None:
    path = tmp_path / "not.pt"
    torch.save({"format": 99}, path)
    with pytest.raises(ArtifactError, match="artifact format"):
        Checkpoint.load(path)


def test_loading_a_checkpoint_that_is_not_there_says_so(tmp_path: Path) -> None:
    with pytest.raises(ArtifactError, match="no checkpoint at"):
        Checkpoint.load(tmp_path / "absent.pt")


# ---------------------------------------------------------------------------
# The run directory
# ---------------------------------------------------------------------------


def test_a_run_directory_refuses_to_reuse_one_that_holds_a_run(
    tmp_path: Path,
) -> None:
    directory = RunDirectory.create(tmp_path / "run")
    directory.write_checkpoint(_checkpoint())
    with pytest.raises(ArtifactError, match="already holds a run"):
        RunDirectory.create(tmp_path / "run")


def test_a_run_directory_refuses_a_file(tmp_path: Path) -> None:
    (tmp_path / "run").write_text("not a directory")
    with pytest.raises(ArtifactError, match="not a directory"):
        RunDirectory.create(tmp_path / "run")


def test_the_plan_may_be_written_twice_only_if_it_is_the_same_plan(
    tmp_path: Path,
) -> None:
    # A program's stages share one plan, so writing it per stage has to be
    # allowed; two different plans in one directory is the ambiguity the
    # directory exists to prevent.
    run = _run()
    directory = RunDirectory.create(tmp_path / "run")
    first = directory.write_plan(run.plan)
    assert directory.write_plan(run.plan) == first
    assert is_read_only(first)


def test_a_second_plan_in_one_run_directory_is_refused(tmp_path: Path) -> None:
    run = _run()
    other = compile(
        Recipe(
            name="renamed",
            schema=make_schema(),
            system=ComponentGraph([ToyEncoder(width=HIDDEN), ToyOutcomeHead()]),
            program=(
                stage(
                    objectives=(
                        Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="mean"),
                    ),
                    trainable=("encoder", "outcome_head"),
                ),
            ),
            card="docs/recipes/renamed.md",
        )
    )
    directory = RunDirectory.create(tmp_path / "run")
    directory.write_plan(run.plan)
    with pytest.raises(ArtifactError, match="different execution plan"):
        directory.write_plan(other.plan)


def test_a_log_is_written_once(tmp_path: Path) -> None:
    directory = RunDirectory.create(tmp_path / "run")
    path = directory.write_log("fit", [{"step": 0, "total": 1.0}])
    assert is_read_only(path)
    assert directory.read_log("fit") == ({"step": 0, "total": 1.0},)
    with pytest.raises(ArtifactError, match="written once"):
        directory.write_log("fit", [{"step": 0, "total": 1.0}])


def test_reading_a_log_that_is_not_there_says_so(tmp_path: Path) -> None:
    directory = RunDirectory.create(tmp_path / "run")
    with pytest.raises(ArtifactError, match="no log at"):
        directory.read_log("fit")
