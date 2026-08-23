"""Tier 0 — the component contract and the graph (`DESIGN.md` §2.1, §2.2, §3).

The rules under test are the ones the design states as guarantees rather than
conventions: a component reads only what it declares, is handed no batch to
reach around its ports with, writes exactly what it declares, and cannot be
constructed without the hyperparameters its card governs.
"""

from typing import Any, ClassVar

import pytest
import torch
from xty2.core import (
    DEFAULT,
    REQUIRED,
    CardKeyError,
    CategoricalTreatment,
    Component,
    ComponentGraph,
    GraphError,
    Port,
    PortContractError,
    PortValue,
    PortView,
    Realisation,
    Schema,
    State,
    XTYBatch,
)

from tests.invariants.conftest import (
    HIDDEN,
    NUM_FEATURES,
    NUM_TREATMENTS,
    ToyEncoder,
    ToyOutcomeHead,
    ToyPosterior,
    ToyPropensity,
    two_head_graph,
)

# ---------------------------------------------------------------------------
# Realisations and state
# ---------------------------------------------------------------------------


def test_the_default_realisation_is_the_batch_under_student_parameters() -> None:
    assert Realisation(view="identity", params="student") == DEFAULT


def test_realisations_sort_deterministically() -> None:
    # The compiler sorts them so a plan printed twice is byte-identical.
    unsorted = [
        Realisation(view="strong_x"),
        Realisation(view="identity", params="teacher"),
        DEFAULT,
    ]
    assert sorted(unsorted) == [
        DEFAULT,
        Realisation(view="identity", params="teacher"),
        Realisation(view="strong_x"),
    ]


def test_realisation_rejects_a_third_parameter_set() -> None:
    with pytest.raises(GraphError, match="'student' or 'teacher'"):
        Realisation(params="ema")  # type: ignore[arg-type]


def test_realisation_rejects_an_empty_view_name() -> None:
    with pytest.raises(GraphError, match="non-empty ViewSpec name"):
        Realisation(view="")


def test_state_is_keyed_by_realisation_and_defaults_to_the_batch() -> None:
    ports: dict[Port, PortValue] = {Port.X_RAW: torch.zeros(2, 2)}
    state = State({DEFAULT: ports})
    assert state.default is state[DEFAULT]
    assert state.realisations == (DEFAULT,)
    assert DEFAULT in state


def test_state_rejects_a_realisation_the_stage_does_not_compute() -> None:
    state = State({DEFAULT: {}})
    with pytest.raises(GraphError, match="state holds no view=strong_x"):
        state[Realisation(view="strong_x")]


def test_state_does_not_hand_out_a_writable_port_map() -> None:
    state = State({DEFAULT: {Port.X_RAW: torch.zeros(2, 2)}})
    with pytest.raises(TypeError):
        state.default[Port.X_REPR] = torch.zeros(2, 2)  # type: ignore[index]


# ---------------------------------------------------------------------------
# PortView: a component reads only what it declares
# ---------------------------------------------------------------------------


def _view(declared: frozenset[Port]) -> PortView:
    values: dict[Port, PortValue] = {
        Port.X_RAW: torch.zeros(2, NUM_FEATURES),
        Port.Y_RAW: torch.zeros(2),
    }
    return PortView(values, declared=declared, component="toy")


def test_port_view_raises_on_an_undeclared_read() -> None:
    view = _view(frozenset({Port.X_RAW}))
    with pytest.raises(GraphError, match="read port 'y' without declaring it"):
        view[Port.Y_RAW]


def test_port_view_iterates_over_declared_ports_only() -> None:
    # Undeclared ports are not merely awkward to reach — they are invisible.
    view = _view(frozenset({Port.X_RAW}))
    assert list(view) == [Port.X_RAW]
    assert len(view) == 1
    assert Port.Y_RAW not in view


def test_port_view_raises_when_a_declared_port_was_never_produced() -> None:
    view = _view(frozenset({Port.X_REPR}))
    with pytest.raises(GraphError, match="nothing has produced"):
        view[Port.X_REPR]


def test_port_view_tensor_rejects_a_distribution_port() -> None:
    head = ToyOutcomeHead()
    produced = head(
        PortView(
            {Port.X_REPR: torch.zeros(2, HIDDEN)},
            declared=frozenset({Port.X_REPR}),
            component="head",
        )
    )
    view = PortView(produced, declared=frozenset({Port.Y_GIVEN_XT}), component="toy")
    with pytest.raises(GraphError, match="as a tensor"):
        view.tensor(Port.Y_GIVEN_XT)


# ---------------------------------------------------------------------------
# Component construction
# ---------------------------------------------------------------------------


class _TakesABatch(Component):
    """The mistake §2.2 exists to make impossible."""

    def __init__(self) -> None:
        super().__init__("sneaky", requires={Port.X_RAW}, provides={Port.X_REPR})

    def forward(self, ports: PortView, batch: XTYBatch) -> dict[Port, PortValue]:  # type: ignore[override]
        return {Port.X_REPR: batch.x}


class _TakesKwargs(Component):
    def __init__(self) -> None:
        super().__init__("sneaky", requires={Port.X_RAW}, provides={Port.X_REPR})

    def forward(self, **ports: Any) -> dict[Port, PortValue]:  # type: ignore[override]
        return {}


def test_a_component_cannot_take_a_batch_argument() -> None:
    with pytest.raises(GraphError, match="no XTYBatch argument"):
        _TakesABatch()


def test_a_component_cannot_take_open_keyword_arguments() -> None:
    with pytest.raises(GraphError, match="forward\\(self, ports: PortView\\)"):
        _TakesKwargs()


def test_a_paper_governed_field_has_no_usable_default() -> None:
    # The REQUIRED sentinel removes the "explicit or inherited?" question by
    # making inheritance impossible (DESIGN.md §9.1).
    with pytest.raises(CardKeyError, match="widths_depths"):
        ToyEncoder()


def test_a_set_hyperparameter_reaches_the_card_binding() -> None:
    assert ToyEncoder(width=HIDDEN).hyperparameters() == {
        "architecture.widths_depths": HIDDEN
    }


class _UnknownKey(Component):
    CARD_KEYS: ClassVar[dict[str, str]] = {"width": "architecture.width"}

    def __init__(self) -> None:
        super().__init__("odd", requires={Port.X_RAW}, provides={Port.X_REPR})
        self.width = 3

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_REPR: ports.tensor(Port.X_RAW)}


def test_a_card_key_outside_the_closed_vocabulary_is_rejected() -> None:
    with pytest.raises(CardKeyError, match="checklist vocabulary"):
        _UnknownKey()


class _ProvidesRawX(Component):
    def __init__(self) -> None:
        super().__init__("shadow", requires={Port.X_REPR}, provides={Port.X_RAW})

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_RAW: ports.tensor(Port.X_REPR)}


def test_a_component_cannot_provide_a_raw_input() -> None:
    with pytest.raises(GraphError, match="virtual source node"):
        _ProvidesRawX()


class _NamedLikeTheSource(Component):
    def __init__(self) -> None:
        super().__init__("source", requires={Port.X_RAW}, provides={Port.X_REPR})

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_REPR: ports.tensor(Port.X_RAW)}


def test_a_component_cannot_be_called_source() -> None:
    with pytest.raises(GraphError, match="virtual source node"):
        _NamedLikeTheSource()


def test_a_component_name_must_be_an_identifier() -> None:
    with pytest.raises(GraphError, match="Python identifier"):
        ToyEncoder("outcome head", width=HIDDEN)


class _SelfDependent(Component):
    def __init__(self) -> None:
        super().__init__("loop", requires={Port.X_REPR}, provides={Port.X_REPR})

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_REPR: ports.tensor(Port.X_REPR)}


def test_a_component_cannot_require_what_it_provides() -> None:
    with pytest.raises(GraphError, match="its own dependency"):
        _SelfDependent()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def test_two_producers_of_one_port_are_rejected() -> None:
    with pytest.raises(GraphError, match="both provide port 'x_repr'"):
        ComponentGraph([ToyEncoder(width=HIDDEN), ToyEncoder("other", width=HIDDEN)])


def test_a_requirement_no_component_provides_is_rejected() -> None:
    with pytest.raises(GraphError, match="requires \\['x_repr'\\]"):
        ComponentGraph([ToyOutcomeHead()])


class _Reconstructor(Component):
    """`X_REPR -> RECONSTRUCTION`, for the cycle below."""

    def __init__(self) -> None:
        super().__init__(
            "decoder", requires={Port.X_REPR}, provides={Port.RECONSTRUCTION}
        )
        self.net = torch.nn.Linear(HIDDEN, NUM_FEATURES)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.RECONSTRUCTION: self.net(ports.tensor(Port.X_REPR))}


class _ReEncoder(Component):
    """`RECONSTRUCTION -> X_REPR`: closes the loop with `_Reconstructor`."""

    def __init__(self) -> None:
        super().__init__(
            "re_encoder", requires={Port.RECONSTRUCTION}, provides={Port.X_REPR}
        )
        self.net = torch.nn.Linear(NUM_FEATURES, HIDDEN)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_REPR: self.net(ports.tensor(Port.RECONSTRUCTION))}


def test_a_cycle_is_rejected() -> None:
    with pytest.raises(GraphError, match="cycle"):
        ComponentGraph([_Reconstructor(), _ReEncoder()])


def test_a_duplicate_component_name_is_rejected() -> None:
    with pytest.raises(GraphError, match="duplicate component name"):
        ComponentGraph([ToyPropensity(), ToyPropensity()])


def test_an_empty_graph_is_rejected() -> None:
    with pytest.raises(GraphError, match="at least one component"):
        ComponentGraph([])


def test_a_bare_module_is_not_a_component() -> None:
    with pytest.raises(GraphError, match="no ports"):
        ComponentGraph([torch.nn.Linear(1, 1)])  # type: ignore[list-item]


def test_topological_order_puts_producers_before_consumers() -> None:
    graph = ComponentGraph(
        [ToyOutcomeHead(), ToyPropensity(), ToyEncoder(width=HIDDEN)]
    )
    assert graph.names.index("encoder") == 0
    # Ties break by declaration order, so the same recipe orders the same way.
    assert graph.names == ("encoder", "outcome_head", "propensity")


def test_the_graph_registers_component_parameters() -> None:
    graph = two_head_graph()
    assert {name.split(".")[0] for name, _ in graph.named_parameters()} == {
        "_components"
    }
    assert all(parameter.requires_grad for parameter in graph.parameters())


# ---------------------------------------------------------------------------
# Lineage: `used_y` is derived from the graph, never declared (§2.2, §7.2)
# ---------------------------------------------------------------------------


def test_raw_outcome_reachability_is_a_property_of_the_graph() -> None:
    graph = ComponentGraph([ToyEncoder(width=HIDDEN), ToyPropensity(), ToyPosterior()])
    assert graph.reads_raw_outcome() == ("posterior",)
    assert graph.outcome_dependent() == ("posterior",)
    assert graph.depends_on_raw_outcome("posterior")
    assert not graph.depends_on_raw_outcome("propensity")


class _RefinesThePosterior(Component):
    """`T_GIVEN_XY -> T_GIVEN_X`: outcome-dependent without reading `Y_RAW`."""

    def __init__(self) -> None:
        super().__init__(
            "refiner", requires={Port.T_GIVEN_XY}, provides={Port.T_GIVEN_X}
        )

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.T_GIVEN_X: ports[Port.T_GIVEN_XY]}


def test_outcome_dependence_is_transitive() -> None:
    graph = ComponentGraph([ToyPosterior(), _RefinesThePosterior()])
    assert graph.reads_raw_outcome() == ("posterior",)
    assert graph.outcome_dependent() == ("posterior", "refiner")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def test_evaluate_returns_every_computed_port_including_the_raw_inputs(
    batch: XTYBatch, schema: Schema
) -> None:
    values = two_head_graph().evaluate(batch, schema=schema)
    assert set(values) == {
        Port.X_RAW,
        Port.Y_RAW,
        Port.X_REPR,
        Port.Y_GIVEN_XT,
        Port.T_GIVEN_X,
    }
    propensity = values[Port.T_GIVEN_X]
    assert isinstance(propensity, CategoricalTreatment)
    assert propensity.probs.shape == (batch.batch_size, NUM_TREATMENTS)


def test_a_planned_selection_runs_only_the_components_it_names(
    batch: XTYBatch, schema: Schema
) -> None:
    graph = two_head_graph()
    values = graph.evaluate(batch, schema=schema, only=("encoder", "propensity"))
    assert Port.Y_GIVEN_XT not in values


def test_subgraph_for_prunes_everything_nothing_depends_on() -> None:
    graph = two_head_graph()
    assert graph.subgraph_for([Port.T_GIVEN_X]) == ("encoder", "propensity")
    assert graph.subgraph_for([Port.X_REPR]) == ("encoder",)


def test_a_selection_missing_a_dependency_is_rejected(
    batch: XTYBatch, schema: Schema
) -> None:
    graph = two_head_graph()
    with pytest.raises(GraphError, match="without its dependencies"):
        graph.evaluate(batch, schema=schema, only=("propensity",))


def test_an_unknown_component_in_a_selection_is_rejected(
    batch: XTYBatch, schema: Schema
) -> None:
    with pytest.raises(GraphError, match="unknown component"):
        two_head_graph().evaluate(batch, schema=schema, only=("missing",))


class _WritesAnUndeclaredPort(Component):
    def __init__(self) -> None:
        super().__init__("leaky", requires={Port.X_RAW}, provides={Port.X_REPR})
        self.net = torch.nn.Linear(NUM_FEATURES, HIDDEN)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        x = ports.tensor(Port.X_RAW)
        return {Port.X_REPR: self.net(x), Port.RECONSTRUCTION: x}


def test_a_component_that_writes_an_undeclared_port_is_rejected(
    batch: XTYBatch, schema: Schema
) -> None:
    graph = ComponentGraph([_WritesAnUndeclaredPort()])
    with pytest.raises(GraphError, match="wrote \\[reconstruction\\]"):
        graph.evaluate(batch, schema=schema)


class _WritesNothing(Component):
    def __init__(self) -> None:
        super().__init__("silent", requires={Port.X_RAW}, provides={Port.X_REPR})

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {}


def test_a_component_that_skips_a_declared_port_is_rejected(
    batch: XTYBatch, schema: Schema
) -> None:
    graph = ComponentGraph([_WritesNothing()])
    with pytest.raises(GraphError, match="did not produce"):
        graph.evaluate(batch, schema=schema)


class _WrongShape(Component):
    def __init__(self) -> None:
        super().__init__("wrong", requires={Port.X_RAW}, provides={Port.X_REPR})

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_REPR: ports.tensor(Port.X_RAW)[:, 0]}


def test_every_produced_port_is_checked_against_its_spec(
    batch: XTYBatch, schema: Schema
) -> None:
    graph = ComponentGraph([_WrongShape()])
    with pytest.raises(PortContractError, match="port x_repr"):
        graph.evaluate(batch, schema=schema)


class _ReachesForAnUndeclaredPort(Component):
    def __init__(self) -> None:
        super().__init__("greedy", requires={Port.X_RAW}, provides={Port.X_REPR})
        self.net = torch.nn.Linear(NUM_FEATURES + 1, HIDDEN)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        x = ports.tensor(Port.X_RAW)
        y = ports.tensor(Port.Y_RAW)  # never declared
        return {Port.X_REPR: self.net(torch.cat([x, y[:, None]], dim=1))}


def test_the_undeclared_read_rule_is_enforced_during_a_forward_pass(
    batch: XTYBatch, schema: Schema
) -> None:
    graph = ComponentGraph([_ReachesForAnUndeclaredPort()])
    with pytest.raises(GraphError, match="without declaring it"):
        graph.evaluate(batch, schema=schema)


def test_the_required_sentinel_has_no_value() -> None:
    assert repr(REQUIRED) == "REQUIRED"
    with pytest.raises(CardKeyError, match="has no value"):
        bool(REQUIRED)
