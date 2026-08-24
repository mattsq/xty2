"""Tier 0 — the three components the first recipe is built from (`PLAN.md` P5).

Two of these tests are the ones that matter, and both are contracts the rest
of the framework is entitled to assume rather than properties of TARNet:

* **the candidate-treatment conformance check** (`DESIGN.md` §3.1) is run
  against `tarnet_head`'s output, not re-written for it. It is the same
  function `GaussianOutcome` is checked with in `test_distributions.py`, and
  it will be the same function P7's conditional flow is checked with. A head
  that passed a bespoke check would prove nothing about the objective that
  consumes it;
* **the port shape contracts** are checked by running the graph, because that
  is where `PortSpec.check` runs. A component whose output is the wrong shape
  fails at the port and names the port.

Everything else here is about the shared `MLPArchitecture`: it exists so that
one card line can describe three components, and the thing that would quietly
break is two components disagreeing about it.
"""

from dataclasses import replace
from typing import Any

import pytest
import torch
from xty2.components import (
    ACTIVATIONS,
    INITIALISATIONS,
    NORMALISATIONS,
    CategoricalPropensity,
    MLPArchitecture,
    MLPEncoder,
    TarnetHead,
    build_mlp,
)
from xty2.core import (
    REQUIRED,
    CompileError,
    ComponentGraph,
    GraphError,
    OutcomeDistribution,
    OutcomeSpec,
    Port,
    Schema,
    XTYBatch,
    check_outcome_distribution_contract,
    check_treatment_distribution_contract,
    compile,
)
from xty2.recipes import tarnet

from tests.invariants.conftest import (
    BATCH_SIZE,
    NUM_TREATMENTS,
    make_batch,
    make_schema,
)


def architecture(**overrides: Any) -> MLPArchitecture:
    """A small but complete architecture. Tier 0 is seconds, not a fit."""
    defaults: dict[str, Any] = {
        "representation": (6, 5),
        "head": (4,),
        "activation": "elu",
        "normalisation": "none",
        "dropout": 0.0,
        "initialisation": "xavier_normal",
    }
    return MLPArchitecture(**(defaults | overrides))


def graph_of(schema: Schema, spec: MLPArchitecture) -> ComponentGraph:
    return ComponentGraph(
        [
            MLPEncoder(schema, architecture=spec),
            TarnetHead(schema, architecture=spec),
            CategoricalPropensity(schema, architecture=spec),
        ]
    )


# ---------------------------------------------------------------------------
# The contracts
# ---------------------------------------------------------------------------


def test_the_outcome_head_satisfies_the_candidate_treatment_contract(
    schema: Schema, batch: XTYBatch
) -> None:
    # B = 7, K = 3. A run with B == K passes under accidental broadcasting and
    # proves nothing (FIDELITY.md §3); the conformance function rejects it.
    assert BATCH_SIZE != NUM_TREATMENTS
    ports = graph_of(schema, architecture()).evaluate(batch, schema=schema)
    check_outcome_distribution_contract(
        ports[Port.Y_GIVEN_XT],  # type: ignore[arg-type]
        y=batch.y,
        num_treatments=NUM_TREATMENTS,
    )


def test_the_propensity_head_satisfies_the_treatment_contract(
    schema: Schema, batch: XTYBatch
) -> None:
    ports = graph_of(schema, architecture()).evaluate(batch, schema=schema)
    check_treatment_distribution_contract(
        ports[Port.T_GIVEN_X],  # type: ignore[arg-type]
        num_treatments=NUM_TREATMENTS,
    )


def test_every_port_matches_its_spec(schema: Schema, batch: XTYBatch) -> None:
    # `evaluate` runs PortSpec.check on every produced value, so reaching the
    # end is the assertion; the shapes below say what was checked.
    ports = graph_of(schema, architecture()).evaluate(batch, schema=schema)
    assert set(ports) == {
        Port.X_RAW,
        Port.Y_RAW,
        Port.X_REPR,
        Port.Y_GIVEN_XT,
        Port.T_GIVEN_X,
    }
    representation = ports[Port.X_REPR]
    assert isinstance(representation, torch.Tensor)
    assert tuple(representation.shape) == (BATCH_SIZE, 5)


def test_the_arms_disagree_so_the_head_is_treatment_sensitive(
    schema: Schema, batch: XTYBatch
) -> None:
    # The failure this rules out is a head that ignores its treatment
    # argument: it satisfies every shape assertion, passes the column-
    # agreement check trivially, and makes every CATE zero.
    ports = graph_of(schema, architecture()).evaluate(batch, schema=schema)
    head = ports[Port.Y_GIVEN_XT]
    assert isinstance(head, OutcomeDistribution)
    candidates = torch.arange(NUM_TREATMENTS).expand(BATCH_SIZE, NUM_TREATMENTS)
    means = head.mean(candidates)
    assert not torch.allclose(means[:, 0], means[:, 1])


def test_the_outcome_head_is_a_unit_scale_gaussian(
    schema: Schema, batch: XTYBatch
) -> None:
    # Card §4 `architecture.output_parameterisation`, as arithmetic: the NLL
    # is the paper's squared error halved, plus a constant. If that stops
    # holding, deviation 4 in the card has stopped being a no-op.
    ports = graph_of(schema, architecture()).evaluate(batch, schema=schema)
    head = ports[Port.Y_GIVEN_XT]
    assert isinstance(head, OutcomeDistribution)
    observed = batch.t
    residual = batch.y - head.mean(observed)
    expected = -0.5 * residual**2 - 0.5 * torch.log(torch.tensor(2 * torch.pi))
    assert torch.allclose(head.log_prob(batch.y, observed), expected, atol=1e-5)


# ---------------------------------------------------------------------------
# The shared architecture
# ---------------------------------------------------------------------------


def test_one_architecture_binds_one_value_of_every_card_key(schema: Schema) -> None:
    spec = architecture()
    for component in graph_of(schema, spec).components:
        resolved = component.hyperparameters()
        assert resolved["architecture.widths_depths"] == spec.widths_description
        assert resolved["architecture.activation"] == "elu"
        assert resolved["architecture.dropout"] == 0.0


def test_two_architectures_in_one_recipe_are_rejected_at_the_card_key(
    schema: Schema,
) -> None:
    # The whole reason MLPArchitecture is one shared object: the card states
    # the architecture in one line, and two components describing different
    # networks would make that line describe neither.
    graph = ComponentGraph(
        [
            MLPEncoder(schema, architecture=architecture()),
            TarnetHead(schema, architecture=architecture(head=(9,))),
            CategoricalPropensity(schema, architecture=architecture()),
        ]
    )
    recipe = tarnet(schema)
    with pytest.raises(CompileError, match=r"architecture\.widths_depths"):
        compile(replace(recipe, system=graph))


def test_the_widths_description_names_the_whole_stack() -> None:
    assert (
        architecture(
            representation=(200, 200, 200), head=(100, 100, 100)
        ).widths_description
    ) == "representation 3x200, heads 3x100"
    assert architecture(representation=(6, 5), head=(4,)).widths_description == (
        "representation [6, 5], heads 1x4"
    )


@pytest.mark.parametrize("field", ["representation", "head", "activation", "dropout"])
def test_an_unset_architecture_field_has_no_usable_default(field: str) -> None:
    with pytest.raises(CompileError, match="no usable default"):
        architecture(**{field: REQUIRED})


@pytest.mark.parametrize("widths", [(), (0,), (3, -1)])
def test_widths_are_positive_integers(widths: tuple[int, ...]) -> None:
    with pytest.raises(CompileError):
        architecture(representation=widths)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5, float("nan")])
def test_dropout_outside_the_unit_interval_is_rejected(dropout: float) -> None:
    with pytest.raises(CompileError, match=r"dropout must be in \[0, 1\)"):
        architecture(dropout=dropout)


def test_an_unknown_activation_says_what_exists() -> None:
    with pytest.raises(CompileError, match="expected one of"):
        architecture(activation="gelu")


def test_a_component_given_something_other_than_an_architecture_is_rejected(
    schema: Schema,
) -> None:
    with pytest.raises(GraphError, match="takes an MLPArchitecture"):
        MLPEncoder(schema, architecture="3x200")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# What the architecture builds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("activation", ACTIVATIONS)
@pytest.mark.parametrize("normalisation", NORMALISATIONS)
@pytest.mark.parametrize("initialisation", INITIALISATIONS)
def test_every_declared_option_builds_and_runs(
    schema: Schema,
    batch: XTYBatch,
    activation: str,
    normalisation: str,
    initialisation: str,
) -> None:
    # Each of these is a value a card can state, so each has to be a network
    # that exists. An option nothing can build is a card line that lies.
    spec = architecture(
        activation=activation,
        normalisation=normalisation,
        initialisation=initialisation,
    )
    graph_of(schema, spec).evaluate(batch, schema=schema)


def test_the_representation_is_an_activated_layer(
    schema: Schema, batch: XTYBatch
) -> None:
    # Phi(x) is the output of the last activated layer, not a bare projection
    # hanging off one. With ReLU that is visible: a projection would produce
    # negative entries.
    encoder = MLPEncoder(schema, architecture=architecture(activation="relu"))
    ports = ComponentGraph([encoder]).evaluate(batch, schema=schema)
    representation = ports[Port.X_REPR]
    assert isinstance(representation, torch.Tensor)
    assert bool((representation >= 0).all())


def test_dropout_is_built_only_when_it_is_asked_for() -> None:
    spec = architecture(dropout=0.0)
    assert not any(
        isinstance(layer, torch.nn.Dropout)
        for layer in build_mlp(3, spec.head, 2, spec).modules()
    )
    wet = architecture(dropout=0.25)
    assert any(
        isinstance(layer, torch.nn.Dropout)
        for layer in build_mlp(3, wet.head, 2, wet).modules()
    )


def test_a_trunk_ends_at_its_last_hidden_width() -> None:
    spec = architecture()
    trunk = build_mlp(3, spec.representation, None, spec)
    assert tuple(trunk(torch.randn(2, 3)).shape) == (2, spec.width)


def test_xavier_initialisation_actually_reaches_the_weights() -> None:
    # torch_default is a real branch, not a no-op by omission, so the two must
    # differ; an initialisation that silently did nothing would make the card
    # key unfalsifiable.
    torch.manual_seed(0)
    default = build_mlp(8, (8,), 8, architecture(initialisation="torch_default"))
    torch.manual_seed(0)
    xavier = build_mlp(8, (8,), 8, architecture(initialisation="xavier_normal"))
    first_default = next(default.parameters())
    first_xavier = next(xavier.parameters())
    assert not torch.equal(first_default, first_xavier)


# ---------------------------------------------------------------------------
# What the head refuses
# ---------------------------------------------------------------------------


def test_the_outcome_head_refuses_a_categorical_outcome() -> None:
    schema = make_schema(
        outcome=OutcomeSpec(kind="categorical", num_classes=4),
    )
    with pytest.raises(GraphError, match="categorical outcome"):
        TarnetHead(schema, architecture=architecture())


def test_a_vector_outcome_keeps_the_candidate_axis_after_the_batch_axis() -> None:
    # Dy = (2,) exercises the reshape the head does per arm. The contract puts
    # the candidate axis immediately after the batch axis, before Dy.
    schema = make_schema(outcome=OutcomeSpec(shape=(2,)))
    batch = make_batch(y=torch.randn(BATCH_SIZE, 2))
    ports = graph_of(schema, architecture()).evaluate(batch, schema=schema)
    check_outcome_distribution_contract(
        ports[Port.Y_GIVEN_XT],  # type: ignore[arg-type]
        y=batch.y,
        num_treatments=NUM_TREATMENTS,
    )
