"""Recipe 1 — `tarnet` (`PLAN.md` P5, card `docs/recipes/tarnet.md`).

TARNet's shared representation and per-arm outcome heads, plus the two things
the framework needs and the paper does not have: a propensity head and exact
marginalisation over a missing treatment. Both are deviations, both are
written into card §5 with their expected effect, and both are visible in the
printed plan.

**There is no logic here.** Every value below is a module constant naming the
card line it comes from, and the function that assembles them contains no
branch, no loop and no conditional (`DESIGN.md` §9). The constants are
defaults on the function rather than literals inside it so that a sweep or a
Tier 1 fit can override one without copying the recipe — overriding a card
value is a *deviation from the card* and shows up in the plan the override
compiles to, which is the property that makes the card cross-check worth
running.
"""

from __future__ import annotations

from xty2.components import (
    CategoricalPropensity,
    MLPArchitecture,
    MLPEncoder,
    TarnetHead,
)
from xty2.core.graph import ComponentGraph
from xty2.core.loss import Reduction
from xty2.core.optimisation import GradientClipping, OptimiserSpec, WeightDecay
from xty2.core.recipe import Recipe, Stage, Weighted
from xty2.core.schedules import Constant, Ramp, Schedule
from xty2.core.schema import Schema
from xty2.objectives import (
    GradPath,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)

CARD = "docs/recipes/tarnet.md"

ARCHITECTURE = MLPArchitecture(
    representation=(200, 200, 200),
    head=(100, 100, 100),
    activation="elu",
    normalisation="none",
    dropout=0.0,
    initialisation="xavier_normal",
)
"""Card §4 `architecture`. Widths and activation are the paper's; the other
three are our choices and are recorded in card §7."""

OPTIMISER = OptimiserSpec(
    name="adam",
    lr=1e-3,
    weight_decay=WeightDecay(value=1e-4, on_norm_and_bias=False),
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
)
"""Card §4 `optimisation` and `gradients.gradient_clipping`. The paper
grid-searches `lr` and its `λ`, so neither number is *the* paper's; card §7
records what we chose instead."""

STEPS = 3000
"""Card §4 `optimisation.total_steps_or_epochs`. Optimiser steps, never epochs
(`FIDELITY.md` §2)."""

OUTCOME_WEIGHT: float | Schedule = 1.0
"""Card §4 `losses.weights` for the factual term — the paper's own loss."""

TREATMENT_WEIGHT: float | Schedule = 1.0
"""Card §4 `losses.weights` for the propensity term. Our extension (§5)."""

MARGINAL_WEIGHT: float | Schedule = Ramp(start=0.0, end=1.0, steps=1000)
"""Card §4 `losses.schedules`. Our extension (§5): the term is meaningless
while both heads are at initialisation, so it ramps in over the first 1000
steps rather than starting at full weight."""

MARGINAL_GRAD_PATH: GradPath = "both"
"""Card §4 `gradients.marginal_nll_grad_path`: the plain likelihood gradient,
reaching the propensity head and the outcome head alike."""

REDUCTION: Reduction = "mean"
"""Card §4 `losses.reduction`, for all three terms. `mean` over `t_observed`
is exactly the paper's reduction in the paper's setting, where every row has a
treatment; card §4 records what it stops being once they go missing."""

TRAINABLE = ("mlp_encoder", "tarnet_head", "categorical_propensity")
"""Everything. One stage, one optimiser, no freezing."""


def tarnet(
    schema: Schema,
    *,
    architecture: MLPArchitecture = ARCHITECTURE,
    optimiser: OptimiserSpec = OPTIMISER,
    steps: int = STEPS,
    outcome_weight: float | Schedule = OUTCOME_WEIGHT,
    treatment_weight: float | Schedule = TREATMENT_WEIGHT,
    marginal_weight: float | Schedule = MARGINAL_WEIGHT,
    marginal_grad_path: GradPath = MARGINAL_GRAD_PATH,
    reduction: Reduction = REDUCTION,
) -> Recipe:
    """Assemble the `tarnet` recipe against `schema`.

    Args:
        schema: Resolved once; supplies `D`, `K` and `Dy` to the components
            and to every port check.
        architecture: The one `architecture.*` card block, shared by all three
            components so that they cannot disagree about it.
        optimiser: The `optimisation` card block and the clipping mode.
        steps: Optimiser steps for the single stage.
        outcome_weight: Weight on `ObservedOutcomeNLL`.
        treatment_weight: Weight on `ObservedTreatmentNLL`.
        marginal_weight: Weight on `MissingTreatmentMarginalNLL`. A `Schedule`
            by default, because the term ramps in.
        marginal_grad_path: Which head the marginalisation term trains.
        reduction: How every term's mean-over-rows value enters the total.

    Returns:
        The recipe. Pass it to `compile()` for the checked, printable plan the
        card is diffed against.
    """
    encoder = MLPEncoder(schema, architecture=architecture)
    head = TarnetHead(schema, architecture=architecture)
    propensity = CategoricalPropensity(schema, architecture=architecture)
    return Recipe(
        name="tarnet",
        schema=schema,
        system=ComponentGraph([encoder, head, propensity]),
        program=(
            Stage(
                name="fit",
                objectives=(
                    Weighted(
                        ObservedOutcomeNLL(),
                        weight=outcome_weight,
                        reduction=reduction,
                    ),
                    Weighted(
                        ObservedTreatmentNLL(),
                        weight=treatment_weight,
                        reduction=reduction,
                    ),
                    Weighted(
                        MissingTreatmentMarginalNLL(grad_path=marginal_grad_path),
                        weight=marginal_weight,
                        reduction=reduction,
                    ),
                ),
                trainable=TRAINABLE,
                rows="all",
                optimiser=optimiser,
                steps=steps,
            ),
        ),
        card=CARD,
        purpose="causal",
    )


__all__ = [
    "ARCHITECTURE",
    "CARD",
    "MARGINAL_GRAD_PATH",
    "MARGINAL_WEIGHT",
    "OPTIMISER",
    "OUTCOME_WEIGHT",
    "REDUCTION",
    "STEPS",
    "TRAINABLE",
    "TREATMENT_WEIGHT",
    "tarnet",
]
