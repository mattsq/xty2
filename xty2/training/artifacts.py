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

P8 adds component buffers to the checkpoint beside parameters. A later stage
cannot be initialised faithfully from a BatchNorm-bearing component if its
running statistics vanished at the stage boundary. They remain a separate
mapping because teacher buffer EMA is independently card-driven and because an
optimiser never owns them.

P10 adds ``PseudoLabels`` under the same rules. ``used_y`` is derived from the
compiled producing port; ``prediction_mode`` is computed from checkpoint row
sets and ``predicted_by_fold``. A cross-fit artifact is accepted only when the
loader reruns that disjointness decision against the actual saved fields.
"""

from __future__ import annotations

import json
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.compile import CompiledRun, ExecutionPlan, plan_digest_of
from xty2.core.errors import ArtifactError, CompileError
from xty2.core.recipe import PseudoLabelAction

ARTIFACT_FORMAT: Final = 1
"""The on-disk payload version. A load of anything else is a loud failure."""

CHECKPOINT_FILE: Final = "checkpoint.pt"
PSEUDO_LABELS_FILE: Final = "pseudo_labels.pt"
FOLDS_DIR: Final = "folds"
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
            trained, and only those. For `array_fit`, this is the action's
            complete named tensor state under `<action>.<name>`; array
            checkpoints cannot be used as component-graph initialisers.
        buffers: `{qualified name: tensor}` for buffers owned by trained graph
            components; empty for an array-fit checkpoint. They are separate
            from parameters because an optimiser never owns them.
        components: The trained component names in stage order, or the single
            array action name.
        steps: Optimiser steps taken, or one functional array-fit call.
        seed: The seed the run was executed under.
        plan_digest: `sha256` of the execution plan (`ExecutionPlan.digest`).
    """

    __slots__ = (
        "_buffers",
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
        buffers: Mapping[str, Tensor] | None = None,
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
        self._buffers: Mapping[str, Tensor] = MappingProxyType(
            {
                name: value.detach().clone()
                for name, value in sorted((buffers or {}).items())
            }
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
        buffers: Mapping[str, Tensor] | None = None,
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
            buffers=buffers,
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

    @property
    def buffers(self) -> Mapping[str, Tensor]:
        """A read-only mapping of copies of the trained components' buffers."""
        return MappingProxyType(
            {name: value.clone() for name, value in self._buffers.items()}
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

    def buffer(self, name: str) -> Tensor:
        """A copy of one saved component buffer."""
        try:
            return self._buffers[name].clone()
        except KeyError:
            raise ArtifactError(
                f"this checkpoint holds no buffer {name!r}; it holds "
                f"{sorted(self._buffers)!r}"
            ) from None

    def state_dict(self) -> dict[str, Tensor]:
        """A copy of every saved parameter, keyed as `<component>.<parameter>`."""
        return {name: value.clone() for name, value in self._parameters.items()}

    def buffer_dict(self) -> dict[str, Tensor]:
        """A copy of every saved buffer, keyed as `<component>.<buffer>`."""
        return {name: value.clone() for name, value in self._buffers.items()}

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
            "buffers": dict(self._buffers),
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
            buffers=payload.get("buffers", {}),
            components=tuple(payload["components"]),
            steps=int(payload["steps"]),
            seed=int(payload["seed"]),
            plan_digest=str(payload["plan_digest"]),
        )


PredictionMode = Literal["in_sample", "out_of_fold"]


class PseudoLabels:
    """Immutable hard treatment labels with falsifiable provenance (§7.1).

    ``used_y`` and ``prediction_mode`` are properties, not constructor
    arguments on the public surface. The executor derives the former from the
    compiled source port and the latter is recomputed from the saved row sets
    every time it is asked. Loading against a ``cross_fit`` stage calls
    :meth:`assert_out_of_fold`, so a forged or corrupted fold assignment fails
    where a consuming stage crosses the artifact boundary.
    """

    __slots__ = (
        "_checkpoints",
        "_labels",
        "_predicted_by_fold",
        "_row_id",
        "_source_stage",
        "_used_y",
    )

    def __init__(
        self,
        *,
        source_stage: str,
        source_checkpoints: Mapping[int, Checkpoint],
        predicted_by_fold: Tensor,
        row_id: Tensor,
        labels: Tensor,
        used_y: bool,
        issued_by: object = None,
    ) -> None:
        if issued_by is not _FACTORY_TOKEN:
            raise ArtifactError(
                "PseudoLabels are constructed only by the stage executors. "
                "used_y comes from graph reachability and prediction_mode from "
                "actual checkpoint row sets, so a direct constructor would "
                "let a caller assert the provenance the loader must verify "
                "(DESIGN.md §7.1)."
            )
        checkpoints = dict(source_checkpoints)
        if not checkpoints:
            raise ArtifactError("PseudoLabels need at least one source checkpoint")
        if any(type(fold) is not int or fold < 0 for fold in checkpoints):
            raise ArtifactError(
                "PseudoLabels source checkpoint keys must be non-negative fold ids"
            )
        if row_id.ndim != 1 or row_id.dtype != torch.long:
            raise ArtifactError(
                f"PseudoLabels.row_id must be a [N] long tensor, got "
                f"shape {tuple(row_id.shape)} of {row_id.dtype}"
            )
        if len(set(row_id.tolist())) != int(row_id.numel()):
            raise ArtifactError(
                "PseudoLabels.row_id must be unique; the side table is keyed by it"
            )
        if (
            predicted_by_fold.shape != row_id.shape
            or predicted_by_fold.dtype != torch.long
        ):
            raise ArtifactError(
                "PseudoLabels.predicted_by_fold must be a [N] long tensor "
                f"aligned with row_id, got shape {tuple(predicted_by_fold.shape)} "
                f"of {predicted_by_fold.dtype}"
            )
        if labels.shape != row_id.shape or labels.dtype != torch.long:
            raise ArtifactError(
                "P10 PseudoLabels.labels must be hard [N] long treatment labels, "
                f"got shape {tuple(labels.shape)} of {labels.dtype}"
            )
        unknown_folds = sorted(set(predicted_by_fold.tolist()) - set(checkpoints))
        if unknown_folds:
            raise ArtifactError(
                f"predicted_by_fold names folds {unknown_folds!r} with no source "
                f"checkpoint; available folds are {sorted(checkpoints)!r}"
            )
        for fold, checkpoint in checkpoints.items():
            if checkpoint.fold not in (None, fold):
                raise ArtifactError(
                    f"source checkpoint key {fold} holds checkpoint.fold="
                    f"{checkpoint.fold!r}; cross-fit checkpoint keys and their "
                    "recorded fold must agree"
                )
        if type(used_y) is not bool:
            raise ArtifactError(f"derived used_y must be bool, got {used_y!r}")

        # The side table is keyed by row id, so store it in canonical sorted
        # order once. ``apply_to`` can then use ``searchsorted`` on every
        # consuming batch without rebuilding a Python lookup table.
        row_values = row_id.detach().cpu().clone()
        fold_values = predicted_by_fold.detach().cpu().clone()
        label_values = labels.detach().cpu().clone()
        order = torch.argsort(row_values)
        self._source_stage = source_stage
        self._checkpoints: Mapping[int, Checkpoint] = MappingProxyType(
            dict(sorted(checkpoints.items()))
        )
        self._predicted_by_fold = fold_values.index_select(0, order)
        self._row_id = row_values.index_select(0, order)
        self._labels = label_values.index_select(0, order)
        self._used_y = used_y

    @classmethod
    def _issue(
        cls,
        *,
        source_stage: str,
        source_checkpoints: Mapping[int, Checkpoint],
        predicted_by_fold: Tensor,
        row_id: Tensor,
        labels: Tensor,
        used_y: bool,
    ) -> PseudoLabels:
        """Package-private construction point used by executors and loading."""
        return cls(
            source_stage=source_stage,
            source_checkpoints=source_checkpoints,
            predicted_by_fold=predicted_by_fold,
            row_id=row_id,
            labels=labels,
            used_y=used_y,
            issued_by=_FACTORY_TOKEN,
        )

    @property
    def source_stage(self) -> str:
        return self._source_stage

    @property
    def source_checkpoints(self) -> Mapping[int, Checkpoint]:
        return MappingProxyType(dict(self._checkpoints))

    @property
    def predicted_by_fold(self) -> Tensor:
        return self._predicted_by_fold.clone()

    @property
    def row_id(self) -> Tensor:
        return self._row_id.clone()

    @property
    def labels(self) -> Tensor:
        return self._labels.clone()

    @property
    def used_y(self) -> bool:
        """Whether the producing port transitively reads ``Y_RAW``."""
        return self._used_y

    @property
    def prediction_mode(self) -> PredictionMode:
        """Derived from actual prediction/checkpoint row disjointness."""
        return "in_sample" if self._first_overlap() is not None else "out_of_fold"

    @property
    def plan_digest(self) -> str:
        digests = {checkpoint.plan_digest for checkpoint in self._checkpoints.values()}
        if len(digests) != 1:
            raise ArtifactError(
                "PseudoLabels source checkpoints come from different execution "
                f"plans: {sorted(digests)!r}"
            )
        return next(iter(digests))

    def _first_overlap(self) -> tuple[int, int] | None:
        trained = {
            fold: set(checkpoint.trained_on_row_ids.tolist())
            for fold, checkpoint in self._checkpoints.items()
        }
        for row, fold in zip(
            self._row_id.tolist(), self._predicted_by_fold.tolist(), strict=True
        ):
            if row in trained[fold]:
                return int(row), int(fold)
        return None

    def assert_out_of_fold(self) -> None:
        """Execute the §7.1 decision procedure, raising on the first overlap."""
        overlap = self._first_overlap()
        if overlap is None:
            return
        row, fold = overlap
        raise ArtifactError(
            f"pseudo-label row {row} was predicted by fold {fold}, whose "
            "checkpoint trained on that same row. A cross_fit artifact must be "
            "out of fold for every row (DESIGN.md §7.1); the saved fold "
            "assignment is overlapping."
        )

    def validate_for(self, run: CompiledRun) -> None:
        """Validate this artifact against the compiled program consuming it."""
        try:
            source = run.stage(self.source_stage)
        except CompileError as error:
            raise ArtifactError(
                f"PseudoLabels name unknown source stage {self.source_stage!r}"
            ) from error
        action = source.action
        if not isinstance(action, PseudoLabelAction):
            raise ArtifactError(
                f"stage {self.source_stage!r} does not declare a "
                "PseudoLabelAction in this plan"
            )
        expected_used_y = run.graph.port_depends_on_raw_outcome(action.port)
        if self.used_y != expected_used_y:
            raise ArtifactError(
                f"PseudoLabels used_y={self.used_y} disagrees with graph-derived "
                f"used_y={expected_used_y} for {action.port}"
            )
        if source.executor == "cross_fit":
            expected_checkpoint_stage = source.name
        elif source.objectives:
            expected_checkpoint_stage = source.name
            if set(self._checkpoints) != {0}:
                raise ArtifactError(
                    f"in-sample PseudoLabels from stage {source.name!r} need "
                    "exactly source checkpoint key 0"
                )
        else:
            if source.initialise_from is None:  # compile() prevents this plan
                raise ArtifactError(
                    f"action-only pseudo-label stage {source.name!r} has no "
                    "producing checkpoint"
                )
            expected_checkpoint_stage = source.initialise_from
            if set(self._checkpoints) != {0}:
                raise ArtifactError(
                    f"action-only PseudoLabels from stage {source.name!r} need "
                    "exactly source checkpoint key 0"
                )
        expected_checkpoint = run.stage(expected_checkpoint_stage)
        for fold, checkpoint in self._checkpoints.items():
            if checkpoint.recipe != run.recipe.name:
                raise ArtifactError(
                    f"source checkpoint fold {fold} belongs to recipe "
                    f"{checkpoint.recipe!r}, not {run.recipe.name!r}"
                )
            if checkpoint.plan_digest != run.plan.digest:
                raise ArtifactError(
                    f"source checkpoint fold {fold} was produced under plan "
                    f"{checkpoint.plan_digest[:12]}, not {run.plan.digest[:12]}"
                )
            if checkpoint.stage != expected_checkpoint_stage:
                raise ArtifactError(
                    f"source checkpoint fold {fold} belongs to stage "
                    f"{checkpoint.stage!r}, but pseudo-label stage "
                    f"{source.name!r} expects {expected_checkpoint_stage!r}"
                )
            if checkpoint.components != expected_checkpoint.trainable:
                raise ArtifactError(
                    f"source checkpoint fold {fold} carries components "
                    f"{checkpoint.components!r}, but stage "
                    f"{expected_checkpoint_stage!r} trains "
                    f"{expected_checkpoint.trainable!r}"
                )
            if checkpoint.steps != expected_checkpoint.steps:
                raise ArtifactError(
                    f"source checkpoint fold {fold} records "
                    f"{checkpoint.steps} steps, but stage "
                    f"{expected_checkpoint_stage!r} declares "
                    f"{expected_checkpoint.steps}"
                )
            if source.executor == "cross_fit" and checkpoint.fold != fold:
                raise ArtifactError(
                    f"cross-fit source checkpoint key {fold} records fold "
                    f"{checkpoint.fold!r}; the producing fold must agree"
                )
            if source.executor != "cross_fit" and checkpoint.fold is not None:
                raise ArtifactError(
                    f"non-cross-fit source checkpoint records fold {checkpoint.fold!r}"
                )
        if self._labels.numel() and (
            int(self._labels.min()) < 0
            or int(self._labels.max()) >= run.recipe.schema.treatment_cardinality
        ):
            raise ArtifactError(
                "PseudoLabels contain a treatment outside the recipe schema's "
                "cardinality"
            )
        if source.executor == "cross_fit":
            self.assert_out_of_fold()

    def apply_to(self, batch: XTYBatch, *, treatment_cardinality: int) -> XTYBatch:
        """Functionally join the hard-label side table by ``row_id``."""
        treatment = batch.t.clone()
        observed = batch.t_observed.clone()
        if self._row_id.numel() == 0 or batch.batch_size == 0:
            return batch.replace(t=treatment, t_observed=observed)

        source_rows = self._row_id.to(device=batch.device)
        source_labels = self._labels.to(device=batch.device)
        positions = torch.searchsorted(source_rows, batch.row_id)
        safe_positions = positions.clamp(max=source_rows.numel() - 1)
        joined_labels = source_labels.index_select(0, safe_positions)
        matched = (positions < source_rows.numel()) & (
            source_rows.index_select(0, safe_positions) == batch.row_id
        )

        invalid = matched & (
            (joined_labels < 0) | (joined_labels >= treatment_cardinality)
        )
        if bool(invalid.any()):
            index = int(torch.nonzero(invalid, as_tuple=False)[0, 0].item())
            raise ArtifactError(
                f"pseudo-label {int(joined_labels[index])} for row "
                f"{int(batch.row_id[index])} is outside [0, "
                f"{treatment_cardinality})"
            )

        conflicts = matched & observed & (treatment != joined_labels)
        if bool(conflicts.any()):
            index = int(torch.nonzero(conflicts, as_tuple=False)[0, 0].item())
            raise ArtifactError(
                f"pseudo-label {int(joined_labels[index])} for row "
                f"{int(batch.row_id[index])} conflicts with its observed "
                f"treatment {int(treatment[index])}; artifact inputs never "
                "overwrite observed data"
            )

        write = matched & ~observed
        treatment[write] = joined_labels[write]
        observed[write] = True
        return batch.replace(t=treatment, t_observed=observed)

    def describe(self) -> str:
        return (
            f"pseudo labels from {self.source_stage}: {self._row_id.numel()} "
            f"rows, mode {self.prediction_mode}, used_y={self.used_y}"
        )

    def save(self, path: Path) -> Path:
        path = Path(path)
        if path.exists():
            raise ArtifactError(
                f"{path} already exists. Artifacts are immutable (DESIGN.md §7.1)."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "format": ARTIFACT_FORMAT,
            "kind": "pseudo_labels",
            "source_stage": self.source_stage,
            "source_checkpoints": {
                fold: _checkpoint_payload(checkpoint)
                for fold, checkpoint in self._checkpoints.items()
            },
            "predicted_by_fold": self._predicted_by_fold,
            "row_id": self._row_id,
            "labels": self._labels,
        }
        torch.save(payload, path)
        _make_read_only(path)
        return path

    @classmethod
    def load(cls, path: Path, run: CompiledRun) -> PseudoLabels:
        """Load, re-derive lineage, and verify cross-fit disjointness."""
        path = Path(path)
        if not path.exists():
            raise ArtifactError(f"no pseudo labels at {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("kind") != "pseudo_labels":
            raise ArtifactError(f"{path} does not hold a pseudo-label payload")
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ArtifactError(
                f"{path} was written in artifact format "
                f"{payload.get('format')!r}; this build reads "
                f"format {ARTIFACT_FORMAT}"
            )
        source_stage = str(payload["source_stage"])
        try:
            source = run.stage(source_stage)
        except CompileError as error:
            raise ArtifactError(
                f"PseudoLabels name unknown source stage {source_stage!r}"
            ) from error
        action = source.action
        if not isinstance(action, PseudoLabelAction):
            raise ArtifactError(
                f"stage {source_stage!r} does not emit pseudo labels in this plan"
            )
        raw_checkpoints = payload["source_checkpoints"]
        if not isinstance(raw_checkpoints, dict):
            raise ArtifactError("source_checkpoints must be a mapping")
        artifact = cls._issue(
            source_stage=source_stage,
            source_checkpoints={
                int(fold): _checkpoint_from_payload(checkpoint)
                for fold, checkpoint in raw_checkpoints.items()
            },
            predicted_by_fold=payload["predicted_by_fold"],
            row_id=payload["row_id"],
            labels=payload["labels"],
            used_y=run.graph.port_depends_on_raw_outcome(action.port),
        )
        artifact.validate_for(run)
        return artifact


def _checkpoint_payload(checkpoint: Checkpoint) -> dict[str, Any]:
    """Plain payload used when a pseudo-label table embeds its producers."""
    return {
        "recipe": checkpoint.recipe,
        "stage": checkpoint.stage,
        "fold": checkpoint.fold,
        "trained_on_row_ids": checkpoint.trained_on_row_ids,
        "parameters": dict(checkpoint.parameters),
        "buffers": dict(checkpoint.buffers),
        "components": list(checkpoint.components),
        "steps": checkpoint.steps,
        "seed": checkpoint.seed,
        "plan_digest": checkpoint.plan_digest,
    }


def _checkpoint_from_payload(payload: object) -> Checkpoint:
    if not isinstance(payload, dict):
        raise ArtifactError("embedded source checkpoint is not a mapping")
    return Checkpoint._issue(
        recipe=str(payload["recipe"]),
        stage=str(payload["stage"]),
        fold=payload["fold"],
        trained_on_row_ids=payload["trained_on_row_ids"],
        parameters=payload["parameters"],
        buffers=payload.get("buffers", {}),
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

    def fold_dir(self, stage: str, fold: int) -> Path:
        """The immutable artifact directory for one cross-fit fold."""
        path = self.stage_dir(stage) / FOLDS_DIR / str(fold)
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
        directory = (
            self.stage_dir(checkpoint.stage)
            if checkpoint.fold is None
            else self.fold_dir(checkpoint.stage, checkpoint.fold)
        )
        return checkpoint.save(directory / CHECKPOINT_FILE)

    def read_checkpoint(self, stage: str, *, fold: int | None = None) -> Checkpoint:
        """Read the checkpoint a stage wrote."""
        directory = self.root / STAGES_DIR / stage
        if fold is not None:
            directory = directory / FOLDS_DIR / str(fold)
        return Checkpoint.load(directory / CHECKPOINT_FILE)

    def write_pseudo_labels(self, labels: PseudoLabels) -> Path:
        """Write one stage's pseudo-label side table against this run's plan."""
        digest = self.plan_digest()
        if digest is None:
            raise ArtifactError(
                f"{self.root} holds no execution plan, so pseudo labels cannot "
                "be checked against the run that produced them"
            )
        if labels.plan_digest != digest:
            raise ArtifactError(
                f"pseudo labels from {labels.source_stage!r} were produced under "
                f"plan {labels.plan_digest[:12]}, and this directory holds "
                f"{digest[:12]}"
            )
        return labels.save(self.stage_dir(labels.source_stage) / PSEUDO_LABELS_FILE)

    def read_pseudo_labels(self, stage: str, run: CompiledRun) -> PseudoLabels:
        """Load a side table and execute its plan/fold provenance checks."""
        return PseudoLabels.load(
            self.root / STAGES_DIR / stage / PSEUDO_LABELS_FILE,
            run,
        )

    def write_log(
        self,
        stage: str,
        records: Iterable[Mapping[str, Any]],
        *,
        fold: int | None = None,
    ) -> Path:
        """Write the §6.2 per-step log as JSON lines."""
        directory = (
            self.stage_dir(stage) if fold is None else self.fold_dir(stage, fold)
        )
        path = directory / LOG_FILE
        if path.exists():
            raise ArtifactError(
                f"{path} already exists; a stage's log is written once (DESIGN.md §7.1)"
            )
        lines = [json.dumps(record, sort_keys=True) for record in records]
        path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
        _make_read_only(path)
        return path

    def read_log(
        self, stage: str, *, fold: int | None = None
    ) -> tuple[dict[str, Any], ...]:
        """Read back what `write_log` wrote."""
        path = self.root / STAGES_DIR / stage
        if fold is not None:
            path = path / FOLDS_DIR / str(fold)
        path = path / LOG_FILE
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
    "FOLDS_DIR",
    "LOG_FILE",
    "PLAN_FILE",
    "PSEUDO_LABELS_FILE",
    "STAGES_DIR",
    "Checkpoint",
    "PredictionMode",
    "PseudoLabels",
    "RunDirectory",
    "is_read_only",
]
