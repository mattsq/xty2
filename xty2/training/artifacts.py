"""Immutable artifacts and the run directory (`DESIGN.md` §7.1).

A stage emits artifacts; **no stage mutates the source dataset**. What makes
that worth anything is that the artifact's provenance is *verifiable rather
than declarative* — a label saying `out_of_fold` is worth nothing if nothing
can check it — so two rules from §7.1 are structural here rather than
documented.

**Artifacts are constructed only by executor factories.** `Checkpoint(...)`
raises. The factory is `xty2.training.executors.emit_checkpoint`, which takes
the compiled run and the rows the fit actually saw, so `trained_on_row_ids` is
what the loop stepped on and `plan_digest` is the plan it stepped under.
Neither can be asserted by a caller who would like them to be true. Python
cannot make this a type error, so it is a construction-time one with a message
naming the factory.

**Artifacts are immutable, on both sides of the boundary.** `frozen=True`
stops field rebinding but not an in-place write to a tensor leaf — the same
hole `DESIGN.md` §1.1 closes for batches — so every tensor is cloned on the
way in and cloned on the way out. On disk the file is written once and made
read-only, and a second write to the same path raises instead of replacing
what a later stage may already have loaded.

`PseudoLabels` — the other §7.1 artifact — is absent rather than stubbed. It
carries `predicted_by_fold` and a fold-disjointness check, neither of which
exists until `cross_fit` does (P10), and an artifact whose provenance nothing
computes is precisely what this module exists to prevent.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import torch
from torch import Tensor

from xty2.core.compile import ExecutionPlan
from xty2.core.errors import ArtifactError

ARTIFACT_FORMAT: Final = 1
"""The on-disk payload version. A load of anything else is a loud failure."""

CHECKPOINT_FILE: Final = "checkpoint.pt"
LOG_FILE: Final = "log.jsonl"
PLAN_FILE: Final = "plan.txt"
STAGES_DIR: Final = "stages"

_FACTORY_TOKEN: Final = object()
"""Held by this module and passed by the executor factories, and by nothing
else. It is what makes "constructed only by executor factories" (§7.1) a
property of the code rather than a sentence in a docstring."""


def issue_token() -> object:
    """The token an executor factory passes to construct an artifact.

    Public so `xty2.training.executors` can reach it and private in effect:
    calling it does not help a caller who wants to *assert* provenance,
    because the factories are what compute the fields.
    """
    return _FACTORY_TOKEN


@dataclass(frozen=True)
class Checkpoint:
    """One fit's parameters and the provenance of the fit (`DESIGN.md` §7.1).

    Attributes:
        recipe: The recipe this came from.
        stage: The stage that produced it.
        fold: The fold it was fit on, or `None` outside cross-fitting.
        trained_on_row_ids: `[M]` long, sorted and unique — exactly the rows
            this fit saw. This is the field the §7.2 leakage check is run
            against, which is why it is accumulated from the batches the loop
            actually stepped on rather than from the dataset it was pointed at.
        parameters: `{qualified name: tensor}` for the components the stage
            trained, and only those: a checkpoint that quietly carried frozen
            components would restore them over a later stage's work.
        components: The trained component names, in the stage's order.
        steps: Optimiser steps taken.
        seed: The seed the run was executed under.
        plan_digest: `sha256` of the execution plan (`ExecutionPlan.digest`).
    """

    recipe: str
    stage: str
    fold: int | None
    trained_on_row_ids: Tensor
    parameters: Mapping[str, Tensor]
    components: tuple[str, ...]
    steps: int
    seed: int
    plan_digest: str
    issued_by: object = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.issued_by is not _FACTORY_TOKEN:
            raise ArtifactError(
                "a Checkpoint is constructed by an executor factory — "
                "xty2.training.executors.emit_checkpoint — and not directly. "
                "Its provenance fields are computed from the run (the rows the "
                "loop stepped on, the plan it stepped under), so a direct call "
                "would let a caller assert them instead (DESIGN.md §7.1)."
            )
        rows = self.trained_on_row_ids
        if rows.ndim != 1 or rows.dtype != torch.long:
            raise ArtifactError(
                f"trained_on_row_ids must be a [M] long tensor, got shape "
                f"{tuple(rows.shape)} of {rows.dtype}"
            )
        # Cloned on the way in: `frozen=True` does not stop the caller writing
        # into the tensor it handed over (DESIGN.md §1.1).
        object.__setattr__(self, "trained_on_row_ids", rows.detach().clone())
        object.__setattr__(
            self,
            "parameters",
            {
                name: value.detach().clone()
                for name, value in sorted(self.parameters.items())
            },
        )
        object.__setattr__(self, "components", tuple(self.components))

    # -- reading -----------------------------------------------------------

    @property
    def row_ids(self) -> Tensor:
        """A copy of `trained_on_row_ids`, so a reader cannot write into it."""
        return self.trained_on_row_ids.clone()

    def parameter(self, name: str) -> Tensor:
        """A copy of one saved parameter."""
        try:
            return self.parameters[name].clone()
        except KeyError:
            raise ArtifactError(
                f"this checkpoint holds no parameter {name!r}; it holds "
                f"{sorted(self.parameters)!r}"
            ) from None

    def state_dict(self) -> dict[str, Tensor]:
        """A copy of every saved parameter, keyed as `<component>.<parameter>`."""
        return {name: value.clone() for name, value in self.parameters.items()}

    def describe(self) -> str:
        """One line for a log or a PR body."""
        return (
            f"checkpoint {self.recipe}/{self.stage}: {len(self.components)} "
            f"components, {int(self.trained_on_row_ids.numel())} rows, "
            f"{self.steps} steps, seed {self.seed}, plan {self.plan_digest[:12]}"
        )

    # -- disk --------------------------------------------------------------

    def save(self, path: Path) -> Path:
        """Write this checkpoint to `path`, once.

        Raises:
            ArtifactError: if `path` already exists. Artifacts are immutable,
                and a stage that overwrote one would invalidate whatever a
                later stage had already loaded from it.
        """
        path = Path(path)
        if path.exists():
            raise ArtifactError(
                f"{path} already exists. Artifacts are immutable (DESIGN.md "
                "§7.1): write a new run directory rather than replacing a "
                "checkpoint a later stage may already have loaded."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "recipe": self.recipe,
            "stage": self.stage,
            "fold": self.fold,
            "trained_on_row_ids": self.trained_on_row_ids,
            "parameters": dict(self.parameters),
            "components": list(self.components),
            "steps": self.steps,
            "seed": self.seed,
            "plan_digest": self.plan_digest,
        }
        torch.save(payload, path)
        _make_read_only(path)
        return path

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """Read a checkpoint written by `save`."""
        path = Path(path)
        if not path.exists():
            raise ArtifactError(f"no checkpoint at {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ArtifactError(f"{path} does not hold a checkpoint payload")
        version = payload.get("format")
        if version != ARTIFACT_FORMAT:
            raise ArtifactError(
                f"{path} was written in artifact format {version!r}; this build "
                f"reads format {ARTIFACT_FORMAT}"
            )
        return cls(
            recipe=str(payload["recipe"]),
            stage=str(payload["stage"]),
            fold=payload["fold"],
            trained_on_row_ids=payload["trained_on_row_ids"],
            parameters=payload["parameters"],
            components=tuple(payload["components"]),
            steps=int(payload["steps"]),
            seed=int(payload["seed"]),
            plan_digest=str(payload["plan_digest"]),
            issued_by=_FACTORY_TOKEN,
        )


@dataclass(frozen=True)
class RunDirectory:
    """Where one run's artifacts go (`DESIGN.md` §7.1).

    A run directory is written once and read afterwards:

    ```
    <root>/plan.txt                     the execution plan the run compiled to
    <root>/stages/<stage>/checkpoint.pt the stage's parameters and provenance
    <root>/stages/<stage>/log.jsonl     the §6.2 per-step log
    ```

    `create` refuses a directory that already holds a run, for the same reason
    `Checkpoint.save` refuses an existing path: the alternative is a directory
    whose plan describes one recipe and whose checkpoint came from another.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @classmethod
    def create(cls, root: Path | str) -> RunDirectory:
        """Make `root`, or reject it if a run is already there."""
        path = Path(root)
        if path.exists():
            if not path.is_dir():
                raise ArtifactError(f"{path} exists and is not a directory")
            if any(path.iterdir()):
                raise ArtifactError(
                    f"{path} already holds a run. A run directory is written "
                    "once (DESIGN.md §7.1); reusing one leaves a plan and a "
                    "checkpoint that need not describe the same recipe."
                )
        path.mkdir(parents=True, exist_ok=True)
        return cls(root=path)

    def stage_dir(self, stage: str) -> Path:
        """`<root>/stages/<stage>`, created on demand."""
        path = self.root / STAGES_DIR / stage
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_plan(self, plan: ExecutionPlan) -> Path:
        """Write the execution plan, or accept that it is already there.

        Idempotent for the *same* plan and a rejection for a different one: a
        program's stages share one plan, so writing it per stage has to be
        allowed, and a second plan in one run directory is the ambiguity this
        directory exists to prevent.
        """
        path = self.root / PLAN_FILE
        rendered = plan.render()
        if path.exists():
            if path.read_text(encoding="utf-8") != rendered:
                raise ArtifactError(
                    f"{path} holds a different execution plan. One run "
                    "directory holds one compiled recipe (DESIGN.md §7.1)."
                )
            return path
        path.write_text(rendered, encoding="utf-8")
        _make_read_only(path)
        return path

    def write_checkpoint(self, checkpoint: Checkpoint) -> Path:
        """Write `checkpoint` under its own stage."""
        return checkpoint.save(self.stage_dir(checkpoint.stage) / CHECKPOINT_FILE)

    def read_checkpoint(self, stage: str) -> Checkpoint:
        """Read the checkpoint a stage wrote."""
        return Checkpoint.load(self.root / STAGES_DIR / stage / CHECKPOINT_FILE)

    def write_log(self, stage: str, records: Iterable[Mapping[str, Any]]) -> Path:
        """Write the §6.2 per-step log as JSON lines."""
        path = self.stage_dir(stage) / LOG_FILE
        if path.exists():
            raise ArtifactError(
                f"{path} already exists; a stage's log is written once (DESIGN.md §7.1)"
            )
        lines = [json.dumps(record, sort_keys=True) for record in records]
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        _make_read_only(path)
        return path

    def read_log(self, stage: str) -> tuple[dict[str, Any], ...]:
        """Read back what `write_log` wrote."""
        path = self.root / STAGES_DIR / stage / LOG_FILE
        if not path.exists():
            raise ArtifactError(f"no log at {path}")
        text = path.read_text(encoding="utf-8")
        parsed: Sequence[Any] = [
            json.loads(line) for line in text.splitlines() if line.strip()
        ]
        return tuple(record for record in parsed)


def _make_read_only(path: Path) -> None:
    """Drop every write bit, so the file system carries the immutability too.

    Not a security boundary — a determined caller can chmod it back — but it
    turns an accidental overwrite from something that silently succeeds into
    something that raises at the point of the mistake.
    """
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def is_read_only(path: Path) -> bool:
    """Does `path` carry no write bit at all?

    The mode is what is asserted, deliberately, rather than `os.access`: root
    may write a mode-444 file, so an access check would report an artifact as
    mutable purely because of who is running the tests.
    """
    writable = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    return not bool(path.stat().st_mode & writable)


__all__ = [
    "ARTIFACT_FORMAT",
    "CHECKPOINT_FILE",
    "LOG_FILE",
    "PLAN_FILE",
    "STAGES_DIR",
    "Checkpoint",
    "RunDirectory",
    "is_read_only",
    "issue_token",
]
