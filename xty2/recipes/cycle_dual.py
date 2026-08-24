"""The reviewed P11 cycle-dual staged posterior/outcome assembly."""

from __future__ import annotations

from xty2.components import CategoricalPosterior, MLPEncoder, TARNetHead
from xty2.components._nn import TORCH_LINEAR_INITIALISATION
from xty2.core import (
    ComponentGraph,
    Constant,
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
from xty2.objectives import ObservedOutcomeNLL, ObservedTreatmentNLL

CYCLE_DUAL_POSTERIOR_WIDTHS = (128, 128)
CYCLE_DUAL_ENCODER_WIDTHS = (128, 128)
CYCLE_DUAL_OUTCOME_WIDTHS = (128, 128)


def cycle_dual(schema: Schema) -> Recipe:
    """Build the reviewed P11 recipe in `docs/recipes/cycle_dual.md`."""
    return Recipe(
        name="cycle_dual",
        schema=schema,
        system=ComponentGraph(
            [
                CategoricalPosterior(
                    input_dim=schema.num_features,
                    num_treatments=schema.treatment_cardinality,
                    outcome=schema.outcome,
                    widths=CYCLE_DUAL_POSTERIOR_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                    output_parameterisation="K softmax logits",
                    standardisation=(
                        "posterior_labels: z-score X and Y with "
                        "training-population statistics; outcome_fit: z-score X "
                        "and Y with the same frozen training-population statistics"
                    ),
                    outcome_scaling=(
                        "inverse-transform each candidate-treatment mean with the "
                        "frozen training Y mean and standard deviation before "
                        "scoring"
                    ),
                    treatment_encoding=(
                        "categorical integers 0..K-1 with t_observed mask; no sentinel"
                    ),
                ),
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=CYCLE_DUAL_ENCODER_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                ),
                TARNetHead(
                    representation_dim=CYCLE_DUAL_ENCODER_WIDTHS[-1],
                    num_treatments=schema.treatment_cardinality,
                    outcome=schema.outcome,
                    widths=CYCLE_DUAL_OUTCOME_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                    output_parameterisation="K means; fixed Gaussian scale=1.0",
                ),
            ]
        ),
        program=(
            Stage(
                name="posterior_labels",
                objectives=(
                    Weighted(
                        ObservedTreatmentNLL(
                            name="observed_posterior_nll",
                            port=Port.T_GIVEN_XY,
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                trainable=("categorical_posterior",),
                rows="all",
                action=PseudoLabelAction(
                    port=Port.T_GIVEN_XY,
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
            ),
            Stage(
                name="outcome_fit",
                objectives=(
                    Weighted(
                        ObservedOutcomeNLL(),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                trainable=("mlp_encoder", "tarnet_head"),
                rows="all",
                inputs=("posterior_labels",),
                executor="gradient",
                optimiser=OptimiserSpec(
                    name="adam",
                    lr=1e-3,
                    weight_decay=WeightDecay.none(),
                    lr_schedule=Constant(1.0),
                    clipping=GradientClipping.none(),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                ),
                steps=1_000,
            ),
        ),
        card="docs/recipes/cycle_dual.md",
        purpose="causal",
    )


__all__ = [
    "CYCLE_DUAL_ENCODER_WIDTHS",
    "CYCLE_DUAL_OUTCOME_WIDTHS",
    "CYCLE_DUAL_POSTERIOR_WIDTHS",
    "cycle_dual",
]
