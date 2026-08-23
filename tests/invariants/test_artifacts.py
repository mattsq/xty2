"""Tier 0 — artifacts and the run directory (`DESIGN.md` §7.1).

§7.1 makes two claims that are worth nothing unless something enforces them,
and each test here is one half of one of them.

**Provenance is derived, not declared.** A `Checkpoint` cannot be constructed
directly, and the factory that can is private to the executor: `run_stage` is
the only public thing that produces one, and what it records is what the loop
did. The earlier draft of §7.1 wrote these fields as ordinary keyword
arguments, which would have let a producer assert `out_of_fold` without ever
being checked — "a guardrail that any caller can talk its way past is not a
guardrail". A *public* factory taking the row ids would have been the same hole
one level up, which is why the tests below build checkpoints by running a
stage rather than by calling a factory.

**Artifacts are immutable.** Not only frozen: `frozen=True` stops field
rebinding and does nothing about an in-place write to a tensor leaf or about a
plain dict behind a `Mapping` annotation, and a file on disk is mutable until
something says otherwise. So the stored values are private, every accessor
returns a copy, and the written file carries no write bit.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from xty2.core import (
    ArtifactError,
    CompiledRun,
    ComponentGraph,
    Recipe,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.objectives import ObservedOutcomeNLL
from xty2.training import Checkpoint, RunDirectory, is_read_only, run_stage

from tests.invariants.conftest import (
    BATCH_SIZE,
    HIDDEN,
    STEPS,
    ToyEncoder,
    ToyOutcomeHead,
    make_batch,
    make_schema,
    stage,
)


def _run(*, weight: float = 1.0, name: str = "two_head") -> CompiledRun:
    return compile(
        Recipe(
            name=name,
            schema=make_schema(),
            system=ComponentGraph([ToyEncoder(width=HIDDEN), ToyOutcomeHead()]),
            program=(
                stage(
                    objectives=(
                        Weighted(ObservedOutcomeNLL(), weight=weight, reduction="mean"),
                    ),
                    trainable=("encoder", "outcome_head"),
                ),
            ),
            card=f"docs/recipes/{name}.md",
        )
    )


def _batches(row_ids: list[list[int]] | None = None) -> list[XTYBatch]:
    if row_ids is None:
        return [make_batch() for _ in range(STEPS)]
    return [make_batch(row_id=torch.tensor(ids, dtype=torch.long)) for ids in row_ids]


def _checkpoint(run: CompiledRun | None = None, **kwargs: object) -> Checkpoint:
    """A checkpoint the only way one can be had: by running a stage."""
    resolved = run if run is not None else _run()
    return run_stage(resolved, "fit", _batches(), seed=0, **kwargs).checkpoint  # type: ignore[arg-type]


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


def test_no_public_factory_hands_out_a_checkpoint() -> None:
    # The construction guard is only worth as much as the narrowest public way
    # to get past it. A factory taking `row_ids` would be that way, so the
    # package exports no such thing: `run_stage` is the only producer.
    import xty2.training as training

    assert "emit_checkpoint" not in training.__all__
    assert not hasattr(training, "emit_checkpoint")


def test_the_row_ids_are_the_union_of_the_batches_stepped_on() -> None:
    # The leakage check of §7.2 asks "is this row in the set this fit saw?", so
    # the set is what the field holds — not the sequence of batches, in which a
    # row appears once per epoch.
    batches = _batches(
        [
            [3, 1, 2, 4, 5, 6, 7],
            [5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 9, 10, 11, 12],
        ]
    )
    result = run_stage(_run(), "fit", batches, seed=0)
    assert result.checkpoint.trained_on_row_ids.tolist() == list(range(1, 13))


def test_a_row_id_buffer_the_source_reuses_does_not_rewrite_provenance() -> None:
    # A loader that refills one preallocated tensor per step is ordinary. If
    # the loop recorded a view of it rather than a snapshot, every step already
    # taken would end up claiming the rows of the last one — and the artifact
    # would name a training set the fit never saw.
    buffer = torch.zeros(BATCH_SIZE, dtype=torch.long)

    def refilling() -> Iterator[XTYBatch]:
        for step in range(STEPS):
            start = 10 + step * BATCH_SIZE
            buffer.copy_(torch.arange(start, start + BATCH_SIZE))
            yield make_batch(row_id=buffer)

    result = run_stage(_run(), "fit", refilling(), seed=0)
    assert result.checkpoint.trained_on_row_ids.tolist() == list(
        range(10, 10 + STEPS * BATCH_SIZE)
    )


def test_the_artifact_rejects_row_ids_that_are_not_row_ids() -> None:
    # Reached through the private constructor on purpose: no public path can
    # supply these, and the check is what keeps that true for the producers
    # P8 and P10 add.
    with pytest.raises(ArtifactError, match="long tensor"):
        Checkpoint._issue(
            recipe="r",
            stage="fit",
            fold=None,
            trained_on_row_ids=torch.tensor([1.0, 2.0]),
            parameters={},
            components=(),
            steps=1,
            seed=0,
            plan_digest="",
        )


def test_the_digest_changes_when_the_plan_does() -> None:
    run = _run()
    assert run.plan.digest != _run(weight=0.5).plan.digest
    assert run.plan.digest == compile(run.recipe).plan.digest


# ---------------------------------------------------------------------------
# Immutability, in memory
# ---------------------------------------------------------------------------


def test_writing_into_what_a_reader_gets_back_does_not_reach_it() -> None:
    checkpoint = _checkpoint()
    borrowed = checkpoint.trained_on_row_ids
    before = borrowed.tolist()
    borrowed[0] = 99
    assert checkpoint.trained_on_row_ids.tolist() == before

    name = next(iter(checkpoint.parameters))
    parameter = checkpoint.parameter(name)
    parameter.zero_()
    assert not torch.equal(parameter, checkpoint.parameters[name])
    checkpoint.parameters[name].zero_()
    assert not bool((checkpoint.parameters[name] == 0).all())


def test_the_parameter_mapping_cannot_be_added_to_or_deleted_from() -> None:
    # `Mapping` is an annotation, not an enforcement: a plain dict behind it
    # would let a consumer add a key that no component ever trained and have
    # `save()` persist it.
    checkpoint = _checkpoint()
    with pytest.raises(TypeError):
        checkpoint.parameters["invented"] = torch.zeros(1)  # type: ignore[index]
    with pytest.raises(TypeError):
        del checkpoint.parameters[next(iter(checkpoint.parameters))]  # type: ignore[attr-defined]


def test_a_saved_parameter_is_detached_from_the_live_module() -> None:
    # A checkpoint that held the live parameters would silently follow the next
    # stage's training, and "restore from the checkpoint" would restore nothing.
    run = _run()
    checkpoint = _checkpoint(run)
    saved = checkpoint.parameter("outcome_head.loc.weight")
    with torch.no_grad():
        for parameter in run.graph["outcome_head"].parameters():
            parameter.add_(1.0)
    assert torch.equal(saved, checkpoint.parameter("outcome_head.loc.weight"))


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
# The run directory holds one compiled recipe
# ---------------------------------------------------------------------------


def test_a_run_directory_refuses_to_reuse_one_that_holds_a_run(
    tmp_path: Path,
) -> None:
    run = _run()
    directory = RunDirectory.create(tmp_path / "run")
    directory.write_plan(run.plan)
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
    directory = RunDirectory.create(tmp_path / "run")
    directory.write_plan(_run().plan)
    with pytest.raises(ArtifactError, match="different execution plan"):
        directory.write_plan(_run(name="renamed").plan)


def test_a_checkpoint_from_another_plan_is_refused(tmp_path: Path) -> None:
    # The checkpoint records the plan it ran under, so a mismatch here is
    # detectable — and detecting it is not the same as refusing it. A
    # directory whose plan and checkpoint describe different recipes cannot be
    # read back, and each half looks valid on its own.
    directory = RunDirectory.create(tmp_path / "run")
    directory.write_plan(_run().plan)
    foreign = _checkpoint(_run(name="renamed"))
    with pytest.raises(ArtifactError, match="one compiled recipe"):
        directory.write_checkpoint(foreign)


def test_a_checkpoint_needs_a_plan_to_be_written_against(tmp_path: Path) -> None:
    directory = RunDirectory.create(tmp_path / "run")
    with pytest.raises(ArtifactError, match="holds no execution plan"):
        directory.write_checkpoint(_checkpoint())


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
