"""The reviewed TARNet P5 assembly — declarations only."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    ExponentialDecay,
    GradientClipping,
    OptimiserSpec,
    Ramp,
    Recipe,
    Schema,
    Stage,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)

ENCODER_WIDTHS = (200, 200, 200)
OUTCOME_WIDTHS = (100, 100, 100)


def tarnet(schema: Schema) -> Recipe:
    """Build the single-stage P5 recipe exactly as `docs/recipes/tarnet.md`."""
    return Recipe(
        name="tarnet",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=ENCODER_WIDTHS,
                    activation="elu",
                    normalisation="row_l2",
                    dropout=0.0,
                    initialisation=CFRNET_INITIALISATION,
                ),
                TARNetHead(
                    representation_dim=ENCODER_WIDTHS[-1],
                    num_treatments=schema.treatment_cardinality,
                    outcome=schema.outcome,
                    widths=OUTCOME_WIDTHS,
                    activation="elu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=CFRNET_INITIALISATION,
                    output_parameterisation="K means; fixed Gaussian scale=1.0",
                ),
                CategoricalPropensity(
                    representation_dim=ENCODER_WIDTHS[-1],
                    num_treatments=schema.treatment_cardinality,
                    activation="linear logits",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=CFRNET_INITIALISATION,
                    output_parameterisation="K softmax logits",
                ),
            ]
        ),
        program=(
            Stage(
                name="joint_fit",
                objectives=(
                    Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="population"),
                    Weighted(
                        ObservedTreatmentNLL(), weight=1.0, reduction="population"
                    ),
                    Weighted(
                        MissingTreatmentMarginalNLL(grad_path="both"),
                        weight=Ramp(0.0, 0.5, steps=1_000),
                        reduction="population",
                    ),
                ),
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                    "categorical_propensity",
                ),
                rows="all",
                optimiser=OptimiserSpec(
                    name="adam",
                    lr=1e-3,
                    weight_decay=WeightDecay(
                        value=1e-4,
                        on_norm_and_bias=False,
                        components=("tarnet_head",),
                    ),
                    lr_schedule=ExponentialDecay(gamma=0.97, every=100),
                    clipping=GradientClipping.none(),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                ),
                steps=3_000,
            ),
        ),
        card="docs/recipes/tarnet.md",
        purpose="causal",
    )


__all__ = ["ENCODER_WIDTHS", "OUTCOME_WIDTHS", "tarnet"]
