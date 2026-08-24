"""Tier 0 — explicit executors, pseudo-label provenance and §7.2 (P10)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import Tensor
from xty2.core import (
    DEFAULT,
    ArtifactError,
    CompileError,
    ComponentGraph,
    Executor,
    LossTerm,
    Port,
    Program,
    PseudoLabelAction,
    Purpose,
    Realisation,
    Recipe,
    RowIndex,
    Rows,
    Stage,
    State,
    TrainContext,
    TrainingError,
    Weighted,
    XTYBatch,
    compile,
    reduce_rows,
    treatment_at,
    treatment_distribution,
)
from xty2.objectives import (
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)
from xty2.training import (
    PseudoLabels,
    RunDirectory,
    run_array_fit,
    run_cross_fit,
    run_program,
    run_stage,
)

from tests.invariants.conftest import (
    BATCH_SIZE,
    HIDDEN,
    ToyEncoder,
    ToyOutcomeHead,
    ToyPosterior,
    ToyPropensity,
    make_batch,
    make_schema,
    stage,
)


@dataclass(frozen=True)
class _PosteriorTreatmentNLL:
    """Supervise the toy ``q(t|x,y)`` on rows where treatment is observed."""

    name: str = "posterior_treatment_nll"
    rows: Rows = "t_observed"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(Port.T_GIVEN_XY, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    def compute(
        self,
        state: State,
        batch: XTYBatch,
        rows: RowIndex,
        ctx: TrainContext,
    ) -> LossTerm:
        del ctx
        posterior = treatment_distribution(
            state,
            Port.T_GIVEN_XY,
            DEFAULT,
            objective=self.name,
        )
        return reduce_rows(-posterior.log_prob(treatment_at(batch, rows)), rows)


@dataclass(frozen=True)
class _MeanArrayFit:
    """Minimal non-autograd estimator used to exercise the executor seam."""

    mutate: bool = False

    @property
    def name(self) -> str:
        return "feature_mean"

    def fit(
        self,
        batch: XTYBatch,
        rows: RowIndex,
        *,
        seed: int,
    ) -> Mapping[str, Tensor]:
        if self.mutate:
            batch.x.zero_()
        return {
            "value": batch.x.index_select(0, rows).mean(dim=0),
            "seed": torch.tensor(seed, dtype=torch.long),
        }


def _posterior_objective() -> Weighted:
    return Weighted(_PosteriorTreatmentNLL(), weight=1.0, reduction="mean")


def _p10_recipe(
    *,
    executor: Executor,
    purpose: Purpose = "causal",
    allow_leakage: bool = False,
    source_port: Port = Port.T_GIVEN_XY,
) -> Recipe:
    source_trainable: tuple[str, ...]
    if source_port == Port.T_GIVEN_XY:
        source_objective = _posterior_objective()
        source_trainable = ("posterior",)
    else:
        source_objective = Weighted(
            ObservedTreatmentNLL(), weight=1.0, reduction="mean"
        )
        source_trainable = ("encoder", "propensity")

    return Recipe(
        name="p10_fixture",
        schema=make_schema(),
        system=ComponentGraph(
            [
                ToyEncoder(width=HIDDEN),
                ToyOutcomeHead(),
                ToyPropensity(),
                ToyPosterior(),
            ]
        ),
        program=Program(
            (
                stage(
                    name="labels",
                    objectives=(source_objective,),
                    trainable=source_trainable,
                    action=PseudoLabelAction(port=source_port, rows="t_missing"),
                    executor=executor,
                    steps=1,
                ),
                stage(
                    name="outcome",
                    objectives=(
                        Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="mean"),
                    ),
                    trainable=("outcome_head",),
                    inputs=("labels",),
                    allow_leakage=allow_leakage,
                    steps=1,
                ),
            )
        ),
        card="docs/recipes/p10_fixture.md",
        purpose=purpose,
    )


def _fold_batch() -> XTYBatch:
    return make_batch(fold_id=torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long))


def _array_recipe(action: _MeanArrayFit | None = None) -> Recipe:
    return Recipe(
        name="array_fixture",
        schema=make_schema(),
        system=ComponentGraph([ToyEncoder(width=HIDDEN)]),
        program=(
            Stage(
                name="array",
                rows="t_observed",
                action=action if action is not None else _MeanArrayFit(),
                executor="array_fit",
            ),
        ),
        card="docs/recipes/array_fixture.md",
    )


# ---------------------------------------------------------------------------
# Static leakage rule
# ---------------------------------------------------------------------------


def test_in_sample_outcome_dependent_labels_are_rejected_for_causal_fit() -> None:
    with pytest.raises(CompileError, match=r"circular q\(t\|x,y\)"):
        compile(_p10_recipe(executor="gradient"))


def test_predictive_fit_must_put_the_opt_out_on_the_consuming_stage() -> None:
    with pytest.raises(CompileError, match="allow_leakage=True"):
        compile(_p10_recipe(executor="gradient", purpose="predictive"))

    run = compile(
        _p10_recipe(
            executor="gradient",
            purpose="predictive",
            allow_leakage=True,
        )
    )
    labels = run.stage("labels")
    assert labels.action_uses_y is True
    assert "action uses raw y: true" in run.plan.render()
    assert "allow leakage: true (predictive only)" in run.plan.render()


def test_a_causal_recipe_cannot_claim_the_predictive_opt_out() -> None:
    with pytest.raises(CompileError, match="causal recipe"):
        compile(
            _p10_recipe(
                executor="gradient",
                purpose="causal",
                allow_leakage=True,
            )
        )


def test_propensity_labels_do_not_trigger_the_outcome_dependent_guard() -> None:
    run = compile(
        _p10_recipe(
            executor="gradient",
            source_port=Port.T_GIVEN_X,
        )
    )
    assert run.stage("labels").action_uses_y is False


def test_cross_fit_defers_actual_disjointness_to_artifact_loading() -> None:
    run = compile(_p10_recipe(executor="cross_fit"))
    assert run.stage("labels").executor == "cross_fit"
    assert "executor: cross_fit" in run.plan.render()


def test_action_only_labels_name_their_producing_checkpoint() -> None:
    with pytest.raises(CompileError, match="needs initialise_from"):
        Stage(
            name="labels",
            action=PseudoLabelAction(port=Port.T_GIVEN_XY),
        )


def test_the_leakage_opt_out_cannot_be_placed_on_a_non_consumer() -> None:
    with pytest.raises(CompileError, match="consumes no artifact inputs"):
        Stage(name="fit", allow_leakage=True)


def test_an_outcome_term_that_excludes_joined_rows_is_not_rejected() -> None:
    base = _p10_recipe(executor="gradient")
    safe_consumer = stage(
        name="outcome",
        objectives=(
            Weighted(
                MissingTreatmentMarginalNLL(grad_path="outcome"),
                weight=1.0,
                reduction="mean",
            ),
        ),
        trainable=("outcome_head",),
        inputs=("labels",),
        steps=1,
    )
    recipe = Recipe(
        name=base.name,
        schema=base.schema,
        system=base.system,
        program=(base.program[0], safe_consumer),
        card=base.card,
        purpose=base.purpose,
    )
    assert compile(recipe).stage("outcome").objectives[0].rows == ("t_missing",)


# ---------------------------------------------------------------------------
# Explicit array fitting
# ---------------------------------------------------------------------------


def test_array_fit_runs_once_on_resolved_rows_and_emits_tensor_state() -> None:
    run = compile(_array_recipe())
    batch = make_batch()
    before = batch.clone()

    result = run_array_fit(run, "array", [batch], seed=17)

    assert batch.equal_to(before)
    assert result.checkpoint.trained_on_row_ids.tolist() == [100, 101, 102, 103]
    assert result.checkpoint.components == ("feature_mean",)
    assert torch.equal(
        result.checkpoint.parameter("feature_mean.value"),
        batch.x[:4].mean(dim=0),
    )
    assert result.checkpoint.parameter("feature_mean.seed").item() == 17
    assert "losses.eligible_rows" not in run.plan.render()


def test_array_fit_is_never_inferred_from_the_presence_of_fit() -> None:
    with pytest.raises(CompileError, match="only under executor='array_fit'"):
        Stage(name="array", action=_MeanArrayFit(), executor="gradient")

    run = compile(_array_recipe())
    with pytest.raises(TrainingError, match="run_array_fit"):
        run_stage(run, "array", [make_batch()], seed=0)


def test_array_fit_detects_source_batch_mutation() -> None:
    run = compile(_array_recipe(_MeanArrayFit(mutate=True)))
    with pytest.raises(TrainingError, match="mutated its input batch"):
        run_array_fit(run, "array", [make_batch()], seed=0)


# ---------------------------------------------------------------------------
# Cross-fit artifacts and functional joins
# ---------------------------------------------------------------------------


def test_cross_fit_earns_out_of_fold_provenance_from_actual_rows() -> None:
    run = compile(_p10_recipe(executor="cross_fit"))
    batch = _fold_batch()
    before = batch.clone()

    result = run_cross_fit(run, "labels", [batch], seed=23)
    labels = result.pseudo_labels
    assert labels is not None

    assert batch.equal_to(before)
    assert set(result.fold_checkpoints) == {0, 1}
    assert labels.row_id.tolist() == [104, 105, 106]
    assert labels.used_y is True
    assert labels.prediction_mode == "out_of_fold"
    for row, fold in zip(
        labels.row_id.tolist(), labels.predicted_by_fold.tolist(), strict=True
    ):
        assert row not in result.fold_checkpoints[fold].trained_on_row_ids.tolist()


def test_a_program_joins_labels_without_mutating_the_source_dataset() -> None:
    run = compile(_p10_recipe(executor="cross_fit"))
    batch = _fold_batch()
    before = batch.clone()

    result = run_program(
        run,
        {"labels": [batch], "outcome": [batch]},
        seed=29,
    )

    assert batch.equal_to(before)
    assert set(result.pseudo_labels) == {"labels"}
    # Four treatments were observed in the source batch. The three immutable
    # pseudo labels are joined only for the consuming stage, so its observed
    # outcome objective sees all seven rows without rewriting ``batch``.
    assert result.stage("outcome").records[0].terms[0].n == BATCH_SIZE


def test_pseudo_labels_cannot_be_constructed_with_asserted_provenance() -> None:
    result = run_cross_fit(
        compile(_p10_recipe(executor="cross_fit")),
        "labels",
        [_fold_batch()],
        seed=31,
    )
    labels = result.pseudo_labels
    assert labels is not None
    with pytest.raises(ArtifactError, match="constructed only by"):
        PseudoLabels(
            source_stage="labels",
            source_checkpoints=result.fold_checkpoints,
            predicted_by_fold=labels.predicted_by_fold,
            row_id=labels.row_id,
            labels=labels.labels,
            used_y=False,
        )


def test_loading_rechecks_fold_disjointness_and_rejects_overlap(
    tmp_path: Path,
) -> None:
    run = compile(_p10_recipe(executor="cross_fit"))
    directory = RunDirectory.create(tmp_path / "run")
    result = run_cross_fit(
        run,
        "labels",
        [_fold_batch()],
        seed=37,
        run_dir=directory,
    )
    labels = result.pseudo_labels
    assert labels is not None

    path = Path(result.paths["pseudo_labels"])
    loaded = directory.read_pseudo_labels("labels", run)
    assert loaded.prediction_mode == "out_of_fold"
    assert torch.equal(loaded.labels, labels.labels)

    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(payload, dict)
    forged_folds = labels.predicted_by_fold
    first_row = int(labels.row_id[0])
    overlapping_fold = next(
        fold
        for fold, checkpoint in result.fold_checkpoints.items()
        if first_row in checkpoint.trained_on_row_ids.tolist()
    )
    forged_folds[0] = overlapping_fold
    payload["predicted_by_fold"] = forged_folds
    forged = tmp_path / "overlapping.pt"
    torch.save(payload, forged)

    with pytest.raises(ArtifactError, match="trained on that same row"):
        PseudoLabels.load(forged, run)
