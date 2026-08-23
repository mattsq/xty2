"""Shared fixtures for the Tier 0 invariants.

`BATCH_SIZE != NUM_TREATMENTS` everywhere on purpose: a test with `B == K`
passes under accidental broadcasting and proves nothing (`FIDELITY.md` §3).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
import torch
from torch import Tensor
from xty2.core import (
    DEFAULT,
    REQUIRED,
    CategoricalTreatment,
    Component,
    ComponentGraph,
    FeatureSpec,
    GaussianOutcome,
    LossTerm,
    Objective,
    OutcomeSpec,
    Port,
    PortValue,
    PortView,
    Realisation,
    Recipe,
    Reduction,
    RowIndex,
    Rows,
    Schedule,
    Schema,
    Stage,
    State,
    TrainContext,
    Weighted,
    XTYBatch,
    reduce_rows,
)

BATCH_SIZE = 7
NUM_TREATMENTS = 3
NUM_FEATURES = 4


@pytest.fixture(autouse=True)
def _deterministic_rng() -> None:
    torch.manual_seed(20260822)


def make_schema(**overrides: Any) -> Schema:
    """A four-column schema exercising bounds, mutability and derivation."""
    defaults: dict[str, Any] = {
        "features": (
            FeatureSpec(
                "mass", "continuous", bounds=(0.0, 100.0), perturbation_scale=0.5
            ),
            FeatureSpec("speed", "continuous", perturbation_scale=1.0),
            FeatureSpec("momentum", "continuous", derived_from=("mass", "speed")),
            FeatureSpec("site", "categorical", mutable=False),
        ),
        "treatment_cardinality": NUM_TREATMENTS,
        "outcome": OutcomeSpec(),
    }
    return Schema(**(defaults | overrides))


def make_batch(**overrides: Any) -> XTYBatch:
    """A valid batch: rows 0-3 have an observed treatment, rows 4-6 do not."""
    t_observed = torch.zeros(BATCH_SIZE, dtype=torch.bool)
    t_observed[:4] = True
    defaults: dict[str, Tensor | None] = {
        "x": torch.randn(BATCH_SIZE, NUM_FEATURES),
        "t": torch.arange(BATCH_SIZE, dtype=torch.long) % NUM_TREATMENTS,
        "y": torch.randn(BATCH_SIZE),
        "t_observed": t_observed,
        "y_observed": torch.ones(BATCH_SIZE, dtype=torch.bool),
        "row_id": torch.arange(100, 100 + BATCH_SIZE, dtype=torch.long),
    }
    return XTYBatch(**(defaults | overrides))


@pytest.fixture
def schema() -> Schema:
    return make_schema()


@pytest.fixture
def batch() -> XTYBatch:
    return make_batch()


# ---------------------------------------------------------------------------
# Toy components and objectives for the graph and compiler invariants (P2)
# ---------------------------------------------------------------------------
#
# These are the smallest things that satisfy the contracts, not previews of
# real components: `mlp_encoder`, `tarnet_head` and `categorical_propensity`
# arrive with the first recipe and its card (P5). Everything here exists to
# give the graph, the port checks and the compiler something to be wrong
# about.

HIDDEN = 5


class ToyEncoder(Component):
    """`X_RAW -> X_REPR`. Carries a card key, so hyperparameters are exercised."""

    CARD_KEYS: ClassVar[Mapping[str, str]] = {"width": "architecture.widths_depths"}

    def __init__(self, name: str = "encoder", *, width: int = REQUIRED) -> None:
        super().__init__(name, requires={Port.X_RAW}, provides={Port.X_REPR})
        self.width = width
        self.net = torch.nn.Linear(NUM_FEATURES, width if width is not REQUIRED else 1)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {Port.X_REPR: self.net(ports.tensor(Port.X_RAW))}


class ToyOutcomeHead(Component):
    """`X_REPR -> Y_GIVEN_XT`, as the reference Gaussian head."""

    def __init__(self, name: str = "outcome_head", *, width: int = HIDDEN) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={Port.Y_GIVEN_XT})
        self.loc = torch.nn.Linear(width, NUM_TREATMENTS)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        loc = self.loc(ports.tensor(Port.X_REPR))
        return {Port.Y_GIVEN_XT: GaussianOutcome(loc=loc, scale=torch.ones_like(loc))}


class ToyPropensity(Component):
    """`X_REPR -> T_GIVEN_X`."""

    def __init__(self, name: str = "propensity", *, width: int = HIDDEN) -> None:
        super().__init__(name, requires={Port.X_REPR}, provides={Port.T_GIVEN_X})
        self.logits = torch.nn.Linear(width, NUM_TREATMENTS)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        return {
            Port.T_GIVEN_X: CategoricalTreatment(self.logits(ports.tensor(Port.X_REPR)))
        }


class ToyPosterior(Component):
    """`X_RAW, Y_RAW -> T_GIVEN_XY` — the one toy that reads the raw outcome."""

    def __init__(self, name: str = "posterior") -> None:
        super().__init__(
            name, requires={Port.X_RAW, Port.Y_RAW}, provides={Port.T_GIVEN_XY}
        )
        self.logits = torch.nn.Linear(NUM_FEATURES + 1, NUM_TREATMENTS)

    def forward(self, ports: PortView) -> dict[Port, PortValue]:
        x = ports.tensor(Port.X_RAW)
        y = ports.tensor(Port.Y_RAW)
        joined = torch.cat([x, y.reshape(x.shape[0], -1)], dim=1)
        return {Port.T_GIVEN_XY: CategoricalTreatment(self.logits(joined))}


@dataclass(frozen=True)
class ToyObjective:
    """The smallest thing that satisfies the `Objective` protocol (§4).

    `compute` returns a genuine zero term through `reduce_rows`, so the row
    bookkeeping every compiler and mixer test relies on is real even though the
    loss is not. The three shipped objectives are tested against brute force in
    `test_objectives.py`; this one exists to be wired up, not to be right.
    """

    name: str
    requires: frozenset[tuple[Port, Realisation]]
    rows: Rows = "all"
    CARD_KEYS: ClassVar[Mapping[str, str]] = {}

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del state, ctx
        return reduce_rows(torch.zeros(batch.batch_size), rows)


def weighted(
    objective: Objective,
    *,
    weight: float | Schedule = 1.0,
    reduction: Reduction = "mean",
) -> Weighted:
    """Wrap an objective for a stage. Both fields are explicit, as recipes must be."""
    return Weighted(objective, weight=weight, reduction=reduction)


def objective(
    name: str,
    *ports: Port,
    rows: Rows = "all",
    realisation: Realisation = DEFAULT,
    weight: float | Schedule = 1.0,
    reduction: Reduction = "mean",
) -> Weighted:
    """A weighted `ToyObjective` requiring `ports` under one realisation."""
    return weighted(
        ToyObjective(
            name=name,
            requires=frozenset((port, realisation) for port in ports),
            rows=rows,
        ),
        weight=weight,
        reduction=reduction,
    )


def two_head_graph() -> ComponentGraph:
    """Encoder -> {outcome head, propensity}: the shape every recipe starts from."""
    return ComponentGraph(
        [
            ToyEncoder(width=HIDDEN),
            ToyOutcomeHead(),
            ToyPropensity(),
        ]
    )


def two_head_recipe(**overrides: Any) -> Recipe:
    """A compilable recipe over `two_head_graph`."""
    defaults: dict[str, Any] = {
        "name": "two_head",
        "schema": make_schema(),
        "system": two_head_graph(),
        "program": (
            Stage(
                name="fit",
                objectives=(
                    objective("outcome_nll", Port.Y_GIVEN_XT, rows="y_observed"),
                    objective("treatment_nll", Port.T_GIVEN_X, rows="t_observed"),
                ),
                trainable=("encoder", "outcome_head", "propensity"),
            ),
        ),
        "card": "docs/recipes/two_head.md",
    }
    return Recipe(**(defaults | overrides))


def backward(value: Tensor) -> None:
    """`value.backward()`, with torch's untyped signature confined to one place."""
    value.backward()  # type: ignore[no-untyped-call]
