"""Tier 0 — `Schema` validates its own feature graph (`DESIGN.md` §1.2).

`derived_from` has to be **complete** (every named parent exists) and
**acyclic**, because P6's derived-column rule walks it: a view that perturbs a
parent must be able to find the dependants it staled, and a walk over a cyclic
or dangling graph does not terminate or does not resolve.
"""

import pytest
import torch
from xty2.core import FeatureSpec, OutcomeSpec, Schema, SchemaError, XTYBatch

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_FEATURES,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)

# -- FeatureSpec ------------------------------------------------------------


def test_a_feature_spec_carries_its_augmentation_metadata() -> None:
    spec = FeatureSpec(
        "mass", "continuous", bounds=(0.0, 10.0), perturbation_scale=0.5, mutable=False
    )
    assert spec.bounds == (0.0, 10.0)
    assert spec.perturbation_scale == 0.5
    assert not spec.mutable
    assert spec.derived_from == ()


def test_derived_from_is_normalised_to_a_tuple() -> None:
    spec = FeatureSpec("m", "continuous", derived_from=["a", "b"])  # type: ignore[arg-type]
    assert spec.derived_from == ("a", "b")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": ""}, "non-empty string"),
        ({"bounds": (10.0, 0.0)}, "low < high"),
        ({"bounds": (1.0, 1.0)}, "low < high"),
        ({"perturbation_scale": 0.0}, "positive value"),
        ({"perturbation_scale": -1.0}, "positive value"),
        ({"derived_from": ("mass",)}, "derived from itself"),
    ],
)
def test_feature_spec_rejects_incoherent_metadata(
    kwargs: dict[str, object], message: str
) -> None:
    defaults: dict[str, object] = {"name": "mass", "kind": "continuous"}
    with pytest.raises(SchemaError, match=message):
        FeatureSpec(**(defaults | kwargs))  # type: ignore[arg-type]


def test_a_categorical_feature_cannot_be_jittered() -> None:
    with pytest.raises(SchemaError, match="jitter on a category"):
        FeatureSpec("site", "categorical", perturbation_scale=1.0)


# -- the feature graph ------------------------------------------------------


def test_a_valid_schema_resolves_its_lookups(schema: Schema) -> None:
    assert schema.num_features == NUM_FEATURES
    assert schema.feature_names == ("mass", "speed", "momentum", "site")
    assert schema.index_of("momentum") == 2
    assert schema.feature("site").mutable is False
    with pytest.raises(SchemaError, match="unknown feature"):
        schema.index_of("nope")
    with pytest.raises(SchemaError, match="unknown feature"):
        schema.feature("nope")


def test_duplicate_feature_names_are_rejected() -> None:
    with pytest.raises(SchemaError, match="duplicate feature name 'a'"):
        Schema(
            features=(FeatureSpec("a", "continuous"), FeatureSpec("a", "continuous")),
            treatment_cardinality=NUM_TREATMENTS,
        )


def test_derived_from_must_be_complete() -> None:
    with pytest.raises(SchemaError, match="unknown column"):
        Schema(
            features=(FeatureSpec("m", "continuous", derived_from=("absent",)),),
            treatment_cardinality=NUM_TREATMENTS,
        )


@pytest.mark.parametrize(
    "cycle",
    [
        (("a", ("b",)), ("b", ("a",))),
        (("a", ("b",)), ("b", ("c",)), ("c", ("a",))),
    ],
    ids=["two-cycle", "three-cycle"],
)
def test_derived_from_must_be_acyclic(
    cycle: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    features = tuple(
        FeatureSpec(name, "continuous", derived_from=parents) for name, parents in cycle
    )
    with pytest.raises(SchemaError, match="acyclic"):
        Schema(features=features, treatment_cardinality=NUM_TREATMENTS)


def test_a_deep_acyclic_graph_is_accepted() -> None:
    schema = Schema(
        features=(
            FeatureSpec("a", "continuous"),
            FeatureSpec("b", "continuous", derived_from=("a",)),
            FeatureSpec("c", "continuous", derived_from=("a", "b")),
            FeatureSpec("d", "continuous", derived_from=("c",)),
        ),
        treatment_cardinality=NUM_TREATMENTS,
    )
    assert schema.num_features == 4


def test_a_schema_needs_features_and_at_least_two_treatments() -> None:
    with pytest.raises(SchemaError, match="at least one feature"):
        Schema(features=(), treatment_cardinality=NUM_TREATMENTS)
    with pytest.raises(SchemaError, match="at least 2"):
        Schema(features=(FeatureSpec("a", "continuous"),), treatment_cardinality=1)


# -- OutcomeSpec ------------------------------------------------------------


def test_outcome_defaults_to_a_scalar_continuous_outcome() -> None:
    outcome = OutcomeSpec()
    assert outcome.shape == ()
    assert outcome.is_continuous


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "categorical"}, "num_classes >= 2"),
        ({"kind": "categorical", "num_classes": 1}, "num_classes >= 2"),
        ({"kind": "categorical", "num_classes": 3, "shape": (2,)}, "shape must be"),
        ({"num_classes": 3}, "meaningless for a continuous outcome"),
        ({"shape": (0,)}, "non-positive axis"),
    ],
)
def test_outcome_spec_rejects_incoherent_metadata(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(SchemaError, match=message):
        OutcomeSpec(**kwargs)  # type: ignore[arg-type]


# -- agreement with a batch -------------------------------------------------


def test_a_conforming_batch_validates(schema: Schema, batch: XTYBatch) -> None:
    schema.validate_batch(batch)


def test_feature_width_must_match(schema: Schema) -> None:
    with pytest.raises(SchemaError, match="columns but the schema declares"):
        schema.validate_batch(make_batch(x=torch.randn(BATCH_SIZE, NUM_FEATURES + 1)))


def test_treatment_values_must_lie_below_k(schema: Schema) -> None:
    over = torch.full((BATCH_SIZE,), NUM_TREATMENTS, dtype=torch.long)
    with pytest.raises(SchemaError, match="observed or not"):
        schema.validate_batch(make_batch(t=over))


def test_treatment_range_is_checked_on_unobserved_rows_too(schema: Schema) -> None:
    """The mask says the value is meaningless; it does not say it may be junk."""
    t = torch.zeros(BATCH_SIZE, dtype=torch.long)
    t[-1] = NUM_TREATMENTS + 5  # a row with t_observed False
    with pytest.raises(SchemaError, match="observed or not"):
        schema.validate_batch(make_batch(t=t))


def test_outcome_shape_and_dtype_must_match(schema: Schema) -> None:
    with pytest.raises(SchemaError, match="trailing shape"):
        schema.validate_batch(make_batch(y=torch.randn(BATCH_SIZE, 2)))
    with pytest.raises(SchemaError, match="continuous outcome needs a float y"):
        schema.validate_batch(make_batch(y=torch.zeros(BATCH_SIZE, dtype=torch.long)))


def test_a_categorical_outcome_is_checked_as_class_indices() -> None:
    schema = make_schema(outcome=OutcomeSpec(kind="categorical", num_classes=2))
    schema.validate_batch(make_batch(y=torch.zeros(BATCH_SIZE, dtype=torch.long)))
    with pytest.raises(SchemaError, match="needs a long y"):
        schema.validate_batch(make_batch(y=torch.zeros(BATCH_SIZE)))
    with pytest.raises(SchemaError, match="class index outside"):
        schema.validate_batch(
            make_batch(y=torch.full((BATCH_SIZE,), 2, dtype=torch.long))
        )


def test_a_vector_outcome_is_accepted_when_declared() -> None:
    schema = make_schema(outcome=OutcomeSpec(shape=(2,)))
    schema.validate_batch(make_batch(y=torch.randn(BATCH_SIZE, 2)))
