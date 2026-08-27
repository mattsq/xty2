"""Tier 0 — every port's shape contract (`DESIGN.md` §2).

Ports are the framework's vocabulary, and a port with no `PortSpec` is a port
the compiler cannot check — so the first test here is that the two lists cannot
drift apart. The rest exercise each spec against a conforming value and against
the way that value is usually wrong.
"""

import pytest
import torch
from xty2.core import (
    PORT_SPECS,
    Axis,
    OutcomeSpec,
    Port,
    PortContractError,
    Schema,
    port_spec,
    resolve_shape,
)
from xty2.core.distributions import CategoricalTreatment, GaussianOutcome

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_FEATURES,
    NUM_TREATMENTS,
    make_schema,
)

REPR_WIDTH = 5


def conforming_value(port: Port) -> object:
    """A value that satisfies `port` for the fixture schema."""
    if port is Port.X_RAW or port is Port.RECONSTRUCTION:
        return torch.randn(BATCH_SIZE, NUM_FEATURES)
    if port is Port.Y_RAW:
        return torch.randn(BATCH_SIZE)
    if port is Port.X_REPR or port is Port.X_PROJ:
        return torch.randn(BATCH_SIZE, REPR_WIDTH)
    if port is Port.JOINT_ENERGY:
        return torch.randn(BATCH_SIZE, NUM_TREATMENTS)
    if port in (Port.T_GIVEN_X, Port.T_GIVEN_XY):
        return CategoricalTreatment(logits=torch.randn(BATCH_SIZE, NUM_TREATMENTS))
    shape = (BATCH_SIZE, NUM_TREATMENTS)
    return GaussianOutcome(loc=torch.randn(shape), scale=torch.ones(shape))


def test_the_port_vocabulary_is_exactly_the_design_document_s() -> None:
    assert {port.value for port in Port} == {
        "x",
        "y",
        "x_repr",
        "x_proj",
        "p(y|x,t)",
        "p(t|x)",
        "q(t|x,y)",
        "energy(x,t,y)",
        "reconstruction",
    }


def test_there_is_no_raw_treatment_port() -> None:
    """`t` is only valid where `t_observed`, and a bare tensor cannot say so."""
    assert not any(port.name.startswith("T_RAW") for port in Port)


@pytest.mark.parametrize("port", list(Port), ids=lambda port: port.name)
def test_every_port_has_a_spec(port: Port) -> None:
    spec = port_spec(port)
    assert spec is PORT_SPECS[port]
    assert spec.port is port
    assert spec.description


@pytest.mark.parametrize("port", list(Port), ids=lambda port: port.name)
def test_every_spec_accepts_a_conforming_value(port: Port, schema: Schema) -> None:
    port_spec(port).check(conforming_value(port), batch_size=BATCH_SIZE, schema=schema)


@pytest.mark.parametrize("port", list(Port), ids=lambda port: port.name)
def test_every_spec_rejects_the_wrong_type(port: Port, schema: Schema) -> None:
    with pytest.raises(PortContractError, match="expected"):
        port_spec(port).check("not a tensor", batch_size=BATCH_SIZE, schema=schema)


@pytest.mark.parametrize(
    ("port", "value", "message"),
    [
        (Port.X_RAW, torch.randn(BATCH_SIZE, NUM_FEATURES + 1), "axis 1 must be 4"),
        (Port.X_RAW, torch.randn(BATCH_SIZE + 1, NUM_FEATURES), "axis 0 must be 7"),
        (Port.X_RAW, torch.randn(BATCH_SIZE), "must have 2 axes"),
        (Port.X_RAW, torch.zeros(BATCH_SIZE, NUM_FEATURES, dtype=torch.long), "float"),
        (Port.X_REPR, torch.randn(BATCH_SIZE, 0), "must be positive"),
        (Port.X_PROJ, torch.randn(BATCH_SIZE, 0), "must be positive"),
        (
            Port.JOINT_ENERGY,
            torch.randn(BATCH_SIZE, NUM_TREATMENTS + 1),
            "axis 1 must be 3",
        ),
        (
            Port.RECONSTRUCTION,
            torch.randn(BATCH_SIZE, NUM_FEATURES + 2),
            "axis 1 must be 4",
        ),
        (Port.Y_RAW, torch.randn(BATCH_SIZE, 2), "must have 1 axes"),
    ],
)
def test_shape_violations_are_rejected(
    port: Port, value: torch.Tensor, message: str, schema: Schema
) -> None:
    with pytest.raises(PortContractError, match=message):
        port_spec(port).check(value, batch_size=BATCH_SIZE, schema=schema)


def test_a_free_axis_accepts_any_positive_width(schema: Schema) -> None:
    for width in (1, 3, 512):
        port_spec(Port.X_REPR).check(
            torch.randn(BATCH_SIZE, width), batch_size=BATCH_SIZE, schema=schema
        )


def test_the_outcome_axis_follows_the_schema() -> None:
    vector = make_schema(outcome=OutcomeSpec(shape=(2,)))
    port_spec(Port.Y_RAW).check(
        torch.randn(BATCH_SIZE, 2), batch_size=BATCH_SIZE, schema=vector
    )
    with pytest.raises(PortContractError, match="must have 2 axes"):
        port_spec(Port.Y_RAW).check(
            torch.randn(BATCH_SIZE), batch_size=BATCH_SIZE, schema=vector
        )


def test_a_categorical_outcome_makes_y_raw_a_long_tensor() -> None:
    categorical = make_schema(outcome=OutcomeSpec(kind="categorical", num_classes=3))
    port_spec(Port.Y_RAW).check(
        torch.zeros(BATCH_SIZE, dtype=torch.long),
        batch_size=BATCH_SIZE,
        schema=categorical,
    )
    with pytest.raises(PortContractError, match="long tensor"):
        port_spec(Port.Y_RAW).check(
            torch.randn(BATCH_SIZE), batch_size=BATCH_SIZE, schema=categorical
        )


@pytest.mark.parametrize("port", [Port.T_GIVEN_X, Port.T_GIVEN_XY])
def test_treatment_ports_require_normalised_probabilities(
    port: Port, schema: Schema
) -> None:
    class Unnormalised:
        @property
        def probs(self) -> torch.Tensor:
            return torch.ones(BATCH_SIZE, NUM_TREATMENTS)

        def log_prob(self, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros(t.shape)

    with pytest.raises(PortContractError, match="rows must sum to 1"):
        port_spec(port).check(Unnormalised(), batch_size=BATCH_SIZE, schema=schema)


@pytest.mark.parametrize("port", [Port.T_GIVEN_X, Port.T_GIVEN_XY])
def test_treatment_ports_require_k_columns(port: Port, schema: Schema) -> None:
    wrong = CategoricalTreatment(logits=torch.randn(BATCH_SIZE, NUM_TREATMENTS + 1))
    with pytest.raises(PortContractError, match="probs axis 1 must be 3"):
        port_spec(port).check(wrong, batch_size=BATCH_SIZE, schema=schema)


def test_the_outcome_port_requires_the_full_distribution_protocol(
    schema: Schema,
) -> None:
    class OnlyLogProb:
        def log_prob(self, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros(y.shape[0])

    with pytest.raises(PortContractError, match="log_prob, mean, sample"):
        port_spec(Port.Y_GIVEN_XT).check(
            OnlyLogProb(), batch_size=BATCH_SIZE, schema=schema
        )


def test_symbolic_axes_resolve_against_the_schema(schema: Schema) -> None:
    resolved = resolve_shape(
        (Axis.BATCH, Axis.FEATURES, Axis.TREATMENTS, Axis.FREE),
        batch_size=BATCH_SIZE,
        schema=schema,
    )
    assert resolved == (BATCH_SIZE, NUM_FEATURES, NUM_TREATMENTS, None)


def test_the_outcome_axis_is_variadic() -> None:
    scalar = make_schema()
    vector = make_schema(outcome=OutcomeSpec(shape=(2, 3)))
    assert resolve_shape((Axis.BATCH, Axis.OUTCOME), batch_size=2, schema=scalar) == (
        2,
    )
    assert resolve_shape((Axis.BATCH, Axis.OUTCOME), batch_size=2, schema=vector) == (
        2,
        2,
        3,
    )
