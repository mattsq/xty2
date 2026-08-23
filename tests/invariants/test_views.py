"""Tier 0 — schema-aware, deterministic data views (`DESIGN.md` §1.2, §5)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest
import torch
from xty2.core import (
    CompileError,
    FeatureSpec,
    Port,
    Realisation,
    Recipe,
    Schema,
    ViewError,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.objectives import ConsistencyLoss
from xty2.views import BoundedJitter, FeatureMask, RecomputeRule, ViewSpec

from tests.invariants.conftest import (
    BATCH_SIZE,
    make_batch,
    objective,
    stage,
    two_head_recipe,
)


def _schema(*, derived: bool = False) -> Schema:
    return Schema(
        features=(
            FeatureSpec(
                "mass", "continuous", bounds=(1.0, 10.0), perturbation_scale=20.0
            ),
            FeatureSpec(
                "speed", "continuous", bounds=(-2.0, 2.0), perturbation_scale=20.0
            ),
            FeatureSpec(
                "momentum",
                "ordinal",
                bounds=(0.0, 50.0),
                perturbation_scale=20.0,
                derived_from=("mass", "speed") if derived else (),
            ),
            FeatureSpec("site", "categorical", bounds=(0.0, 3.0), mutable=False),
        ),
        treatment_cardinality=3,
    )


def _batch() -> XTYBatch:
    x = torch.tensor(
        [[5.0, 1.0, 5.0, 2.0]] * BATCH_SIZE,
        dtype=torch.float32,
    )
    return make_batch(x=x)


def _view(
    name: str,
    *transforms: object,
    recompute_rules: tuple[RecomputeRule, ...] = (),
) -> ViewSpec:
    return ViewSpec(
        name=name,
        transforms=transforms,  # type: ignore[arg-type]
        preserves=frozenset({"t", "y", "t_observed", "y_observed", "row_id"}),
        recompute_rules=recompute_rules,
    )


def test_a_view_is_deterministic_given_its_rng_key() -> None:
    schema = _schema()
    view = _view("strong_x", BoundedJitter(("mass", "speed", "momentum")))
    first = view.apply(_batch(), schema, rng_key=17)
    second = view.apply(_batch(), schema, rng_key=17)
    different = view.apply(_batch(), schema, rng_key=18)
    assert torch.equal(first.x, second.x)
    assert not torch.equal(first.x, different.x)


def test_the_view_name_is_part_of_the_rng_key() -> None:
    schema = _schema()
    transform = BoundedJitter(("mass", "speed"))
    left = _view("left_x", transform).apply(_batch(), schema, rng_key=4)
    right = _view("right_x", transform).apply(_batch(), schema, rng_key=4)
    assert not torch.equal(left.x, right.x)


def test_views_do_not_mutate_the_source_batch() -> None:
    schema = _schema()
    batch = _batch()
    before = batch.clone()
    _view("masked_x", FeatureMask(0.75)).apply(batch, schema, rng_key=3)
    assert batch.equal_to(before)


def test_feature_mask_respects_mutability_and_bounds() -> None:
    schema = _schema()
    result = _view("masked_x", FeatureMask(1.0)).apply(_batch(), schema, rng_key=0)
    # Zero is outside mass's declared range, so its schema-valid mask value is
    # the lower bound. The immutable site column is bit-identical.
    assert torch.equal(result.x[:, 0], torch.ones(BATCH_SIZE))
    assert torch.equal(result.x[:, 3], torch.full((BATCH_SIZE,), 2.0))


def test_bounded_jitter_clips_and_rounds_using_the_schema() -> None:
    schema = _schema()
    result = _view(
        "jittered_x", BoundedJitter(("mass", "speed", "momentum"), scale=100.0)
    ).apply(_batch(), schema, rng_key=2)
    assert bool(((result.x[:, 0] >= 1.0) & (result.x[:, 0] <= 10.0)).all())
    assert bool(((result.x[:, 1] >= -2.0) & (result.x[:, 1] <= 2.0)).all())
    assert bool(((result.x[:, 2] >= 0.0) & (result.x[:, 2] <= 50.0)).all())
    assert torch.equal(result.x[:, 2], result.x[:, 2].round())


def test_bounded_jitter_leaves_an_explicitly_selected_immutable_column_alone() -> None:
    schema = _schema()
    result = _view("jittered_x", BoundedJitter(("mass", "site"), scale=2.0)).apply(
        _batch(), schema, rng_key=9
    )
    assert torch.equal(result.x[:, 3], torch.full((BATCH_SIZE,), 2.0))


def test_the_derived_column_rule_rejects_a_stale_view_at_compile_time() -> None:
    recipe = two_head_recipe(
        schema=_schema(derived=True),
        views=(_view("strong_x", BoundedJitter(("mass",))),),
    )
    with pytest.raises(CompileError, match=r"derived column.*momentum.*stale"):
        compile(recipe)


def _momentum(values: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return values["mass"] * values["speed"]


def test_a_registered_rule_recomputes_an_affected_derived_column() -> None:
    schema = _schema(derived=True)
    rule = RecomputeRule("momentum", _momentum)
    view = _view(
        "strong_x",
        BoundedJitter(("mass",)),
        recompute_rules=(rule,),
    )
    recipe = two_head_recipe(schema=schema, views=(view,))
    compile(recipe)
    result = view.apply(_batch(), schema, rng_key=5)
    assert torch.equal(result.x[:, 2], result.x[:, 0] * result.x[:, 1])


@dataclass(frozen=True)
class _CorruptTreatment:
    def validate(self, schema: Schema) -> None:
        del schema

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        del schema
        return frozenset()

    def apply(
        self, batch: XTYBatch, schema: Schema, *, generator: torch.Generator
    ) -> XTYBatch:
        del generator
        return batch.replace(t=(batch.t + 1) % schema.treatment_cardinality)

    def describe(self) -> str:
        return "corrupt treatment"


def test_preserves_is_enforced_instead_of_trusted() -> None:
    with pytest.raises(ViewError, match=r"declares preserves=.*changed.*t"):
        _view("bad_x", _CorruptTreatment()).apply(_batch(), _schema(), rng_key=0)


class _InPlaceTransform:
    def validate(self, schema: Schema) -> None:
        del schema

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        del schema
        return frozenset({"mass"})

    def apply(
        self, batch: XTYBatch, schema: Schema, *, generator: torch.Generator
    ) -> XTYBatch:
        del schema, generator
        batch.x.add_(1.0)
        return batch

    def describe(self) -> str:
        return "in-place write"


def test_a_transform_that_writes_into_its_input_is_rejected() -> None:
    batch = _batch()
    before = batch.clone()
    with pytest.raises(ViewError, match="wrote into its input batch"):
        _view("bad_x", _InPlaceTransform()).apply(batch, _schema(), rng_key=0)
    assert batch.equal_to(before)


def _consistency_recipe() -> Recipe:
    weak = _view("weak_x", FeatureMask(0.25, columns=("mass",)))
    strong = _view("strong_x", FeatureMask(0.75, columns=("mass",)))
    left = Realisation(view="weak_x")
    right = Realisation(view="strong_x")
    consistency = Weighted(
        ConsistencyLoss(
            port=Port.T_GIVEN_X,
            left=left,
            right=right,
            divergence="kl",
            stop_grad="left",
            rows="all",
        ),
        weight=1.0,
        reduction="mean",
    )
    return two_head_recipe(
        schema=_schema(),
        views=(weak, strong),
        program=(
            stage(
                objectives=(
                    consistency,
                    objective(
                        "strong_aux",
                        Port.T_GIVEN_X,
                        realisation=right,
                    ),
                ),
                trainable=("encoder", "propensity"),
            ),
        ),
    )


def test_the_compiler_plans_one_pass_per_demanded_view_and_no_identity_pass() -> None:
    passes = compile(_consistency_recipe()).stage("fit").passes
    assert [forward.realisation.view for forward in passes] == ["strong_x", "weak_x"]
    assert all(forward.components == ("encoder", "propensity") for forward in passes)


def test_view_keyed_state_uses_the_transformed_batch() -> None:
    run = compile(_consistency_recipe())
    batch = _batch()
    state = run.state("fit", batch, rng_key=12)
    expected = run.recipe.view("weak_x").apply(batch, run.recipe.schema, rng_key=12)
    actual = state[Realisation(view="weak_x")][Port.X_RAW]
    assert isinstance(actual, torch.Tensor)
    assert torch.equal(actual, expected.x)
    assert Realisation() not in state


def test_the_plan_prints_validated_view_mechanics() -> None:
    rendered = compile(_consistency_recipe()).plan.render()
    assert "views\n  weak_x" in rendered
    assert "transform  FeatureMask" in rendered
    assert "preserves  row_id, t, t_observed, y, y_observed" in rendered


def test_an_undeclared_view_is_still_an_unsatisfied_realisation() -> None:
    recipe = _consistency_recipe()
    bad = ConsistencyLoss(
        port=Port.T_GIVEN_X,
        left=Realisation(view="missing_x"),
        right=Realisation(view="strong_x"),
        divergence="mse",
        stop_grad="left",
    )
    replaced = Recipe(
        name=recipe.name,
        schema=recipe.schema,
        system=recipe.system,
        program=(
            stage(
                objectives=(Weighted(bad, weight=1.0, reduction="mean"),),
                trainable=("encoder", "propensity"),
            ),
        ),
        card=recipe.card,
        views=recipe.views,
    )
    with pytest.raises(CompileError, match="cannot produce"):
        compile(replaced)
