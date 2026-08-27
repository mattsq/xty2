"""The reviewed P11 semi-supervised DML assembly."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder
from xty2.components._nn import (
    CFRNET_INITIALISATION,
    TORCH_LINEAR_INITIALISATION,
)
from xty2.core import (
    ComponentGraph,
    Constant,
    ExternalBatches,
    GradientClipping,
    OptimiserSpec,
    Port,
    PseudoLabelAction,
    Recipe,
    Schema,
    Stage,
    WeightDecay,
    Weighted,
)
from xty2.estimators import SSDMLATEAction
from xty2.objectives import ObservedTreatmentNLL

SSDML_ENCODER_WIDTHS = (64, 64)


def ssdml(schema: Schema) -> Recipe:
    """Build the reviewed P11 recipe in `docs/recipes/ssdml.md`."""
    return Recipe(
        name="ssdml",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=SSDML_ENCODER_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                ),
                CategoricalPropensity(
                    representation_dim=SSDML_ENCODER_WIDTHS[-1],
                    num_treatments=schema.treatment_cardinality,
                    activation="linear logits",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=CFRNET_INITIALISATION,
                    output_parameterisation="two softmax logits",
                ),
            ]
        ),
        program=(
            Stage(
                name="propensity_labels",
                objectives=(
                    Weighted(
                        ObservedTreatmentNLL(),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                trainable=("mlp_encoder", "categorical_propensity"),
                rows="all",
                action=PseudoLabelAction(
                    port=Port.T_GIVEN_X,
                    rows="t_missing",
                ),
                executor="cross_fit",
                optimiser=OptimiserSpec(
                    name="adam",
                    lr=1e-3,
                    weight_decay=WeightDecay.none(),
                    lr_schedule=Constant(1.0),
                    clipping=GradientClipping.none(),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                ),
                steps=500,
                sampler=ExternalBatches(),
            ),
            Stage(
                name="dml_ate",
                rows="all",
                action=SSDMLATEAction(
                    num_treatments=schema.treatment_cardinality,
                    outcome=schema.outcome,
                    ridge_penalty=0.001,
                    propensity_clip=(0.025, 0.975),
                    folds=5,
                    max_irls_iterations=100,
                    irls_relative_tolerance=1e-8,
                ),
                inputs=("propensity_labels",),
                executor="array_fit",
                sampler=ExternalBatches(),
            ),
        ),
        card="docs/recipes/ssdml.md",
        purpose="causal",
    )


__all__ = ["SSDML_ENCODER_WIDTHS", "ssdml"]
