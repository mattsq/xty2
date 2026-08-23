"""Immutable artifacts and the run directory (`DESIGN.md` §7.1).

A stage emits artifacts; **no stage mutates the source dataset**. What makes
that worth anything is that the artifact's provenance is *verifiable rather
than declarative* — a label saying `out_of_fold` is worth nothing if nothing
can check it — so two rules from §7.1 are structural here rather than
documented.

**Artifacts are constructed only by executor factories.** `Checkpoint(...)`
raises, and the factory that can construct one is module-private to the
executor — `run_stage` is the only public thing that returns a checkpoint, and
it passes the rows the loop stepped on and the plan it stepped under. A public
factory would have been the same hole one level up: a caller could not write
the provenance directly, but could hand it to something that would.

This is a legibility guard, not a security boundary. Python cannot make the
constructor a type error, and a caller determined to reach `Checkpoint._issue`
can. What it buys is that no *ordinary* path produces an artifact whose
provenance was asserted rather than observed, and that a path which does is a
deliberate, reviewable line in a diff.

**Artifacts are immutable, on both sides of the boundary.** `frozen=True`
stops field rebinding and does nothing about `checkpoint.row_ids.zero_()` or
about a plain dict behind a `Mapping` annotation — the same hole `DESIGN.md`
§1.1 closes for batches. So the stored values are private and every accessor
returns a copy. On disk the file is written once and made read-only, a second
write to the same path raises instead of replacing what a later stage may
already have loaded, and a checkpoint is only written into a run directory
whose plan it was actually produced under.

`PseudoLabels` — the other §7.1 artifact — is absent rather than stubbed. It
carries `predicted_by_fold` and a fold-disjointness check, neither of which
exists until `cross_fit` does (P10), and an artifact whose provenance nothing
computes is precisely what this module exists to prevent.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import torch
from torch import Tensor

from xty2.core.compile import ExecutionPlan, plan_digest_of
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


class Checkpoint:
    """One fit's parameters and the provenance of the fit (`DESIGN.md` §7.1).

    Not a dataclass, and not for style: a frozen dataclass stops a field being
    rebound and does nothing about `checkpoint.trained_on_row_ids.zero_()`, or
    about a caller adding, replacing or deleting an entry of a `parameters`
    dict that is a plain dict behind a `Mapping` annotation. The stored values
    are therefore private, and every accessor hands back a copy — the same
    reasoning `DESIGN.md` §1.1 gives for batches, applied to the artifact whose
    entire purpose is to still say later what it said when it was written.

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

    __slots__ = (
        "_components",
        "_fold",
        "_parameters",
        "_plan_digest",
        "_recipe",
        "_row_ids",
        "_seed",
        "_stage",
        "_steps",
    )

    def __init__(
        self,
        *,
        recipe: str,
        stage: str,
        fold: int | None,
        trained_on_row_ids: Tensor,
        parameters: Mapping[str, Tensor],
        components: tuple[str, ...],
        steps: int,
        seed: int,
        plan_digest: str,
        issued_by: object = None,
    ) -> None:
        if issued_by is not _FACTORY_TOKEN:
            raise ArtifactError(
                "a Checkpoint is constructed by an executor factory — it is "
                "what `xty2.training.run_stage` returns — and not directly. "
                "Its provenance fields are computed from the run (the rows the "
                "loop stepped on, the plan it stepped under), so a direct call "
                "would let a caller assert them instead (DESIGN.md §7.1)."
            )
        if trained_on_row_ids.ndim != 1 or trained_on_row_ids.dtype != torch.long:
            raise ArtifactError(
                f"trained_on_row_ids must be a [M] long tensor, got shape "
                f"{tuple(trained_on_row_ids.shape)} of {trained_on_row_ids.dtype}"
            )
        self._recipe = recipe
        self._stage = stage
        self._fold = fold
        self._row_ids = trained_on_row_ids.detach().clone()
        self._parameters: Mapping[str, Tensor] = MappingProxyType(
            {name: value.detach().clone() for name, value in sorted(parameters.items())}
        )
        self._components = tuple(components)
        self._steps = steps
        self._seed = seed
        self._plan_digest = plan_digest

    @classmethod
    def _issue(
        cls,
        *,
        recipe: str,
        stage: str,
        fold: int | None,
        trained_on_row_ids: Tensor,
        parameters: Mapping[str, Tensor],
        components: tuple[str, ...],
        steps: int,
        seed: int,
        plan_digest: str,
    ) -> Checkpoint:
        """Construct one. Package-private: the executor's factory calls it.

        A determined caller can reach this, and reaching it is the point at
        which the bypass becomes deliberate and visible in a diff rather than
        something an ordinary call does by accident.
        """
        return cls(
            recipe=recipe,
            stage=stage,
            fold=fold,
            trained_on_row_ids=trained_on_row_ids,
            parameters=parameters,
            components=components,
            steps=steps,
            seed=seed,
            plan_digest=plan_digest,
            issued_by=_FACTORY_TOKEN,
        )

    # -- reading -----------------------------------------------------------

    @property
    def recipe(self) -> str:
        return self._recipe

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def fold(self) -> int | None:
        return self._fold

    @property
    def components(self) -> tuple[str, ...]:
        return self._components

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def plan_digest(self) -> str:
        return self._plan_digest

    @property
    def trained_on_row_ids(self) -> Tensor:
        """A copy of the rows this fit saw. A copy, so a reader cannot edit it."""
        return self._row_ids.clone()

    @property
    def parameters(self) -> Mapping[str, Tensor]:
        """A read-only mapping of copies: neither the keys nor the values bite back."""
        return MappingProxyType(
            {name: value.clone() for name, value in self._parameters.items()}
        )

    def parameter(self, name: str) -> Tensor:
        """A copy of one saved parameter."""
        try:
            return self._parameters[name].clone()
        except KeyError:
            raise ArtifactError(
                f"this checkpoint holds no parameter {name!r}; it holds "
                f"{sorted(self._parameters)!r}"
            ) from None

    def state_dict(self) -> dict[str, Tensor]:
        """A copy of every saved parameter, keyed as `<component>.<parameter>`."""
        return {name: value.clone() for name, value in self._parameters.items()}

    def __repr__(self) -> str:
        return (
            f"Checkpoint(recipe={self._recipe!r}, stage={self._stage!r}, "
            f"fold={self._fold!r}, steps={self._steps!r}, seed={self._seed!r})"
        )

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
            "trained_on_row_ids": self._row_ids,
            "parameters": dict(self._parameters),
            "components": list(self._components),
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
        return cls._issue(
            recipe=str(payload["recipe"]),
            stage=str(payload["stage"]),
            fold=payload["fold"],
            trained_on_row_ids=payload["trained_on_row_ids"],
            parameters=payload["parameters"],
            components=tuple(payload["components"]),
            steps=int(payload["steps"]),
            seed=int(payload["seed"]),
            plan_digest=str(payload["plan_digest"]),
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

    def plan_digest(self) -> str | None:
        """The digest of the plan this directory holds, or `None` if it holds none."""
        path = self.root / PLAN_FILE
        if not path.exists():
            return None
        return plan_digest_of(path.read_text(encoding="utf-8"))

    def write_checkpoint(self, checkpoint: Checkpoint) -> Path:
        """Write `checkpoint` under its own stage, against this run's plan.

        The checkpoint's `plan_digest` has to match the plan already in the
        directory. Rejecting the mismatch rather than merely recording it is
        what `write_plan` already does for a second plan, and for the same
        reason: a directory whose plan describes one recipe and whose
        checkpoint came from another is a run that cannot be read back, and
        both halves would look individually valid.
        """
        digest = self.plan_digest()
        if digest is None:
            raise ArtifactError(
                f"{self.root} holds no execution plan, so there is nothing to "
                "write this checkpoint against. Write the plan first — it is "
                "what the checkpoint's plan digest identifies (DESIGN.md §7.1)."
            )
        if digest != checkpoint.plan_digest:
            raise ArtifactError(
                f"checkpoint {checkpoint.stage!r} was produced under plan "
                f"{checkpoint.plan_digest[:12]}, and {self.root} holds plan "
                f"{digest[:12]}. One run directory holds one compiled recipe "
                "(DESIGN.md §7.1)."
            )
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
]
