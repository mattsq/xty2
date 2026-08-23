"""The reviewed conditional-flow P7 assembly — declarations only."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, ConditionalFlow, MLPEncoder
from xty2.components._nn import (
    CFRNET_INITIALISATION,
    TORCH_LINEAR_INITIALISATION,
)
from xty2.components.density import (
    NFLOWS_INITIALISATION,
    RANDOM_PERMUTATION,
    STANDARD_NORMAL,
)
from xty2.core import (
    ComponentGraph,
    Constant,
    GradientClipping,
    OptimiserSpec,
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

CNFLOW_ENCODER_WIDTHS = (128, 128)


def cnflow(schema: Schema) -> Recipe:
    """Build the single-stage P7 recipe exactly as `docs/recipes/cnflow.md`."""
    return Recipe(
        name="cnflow",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=CNFLOW_ENCODER_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                ),
                ConditionalFlow(
                    representation_dim=CNFLOW_ENCODER_WIDTHS[-1],
                    num_treatments=schema.treatment_cardinality,
                    outcome=schema.outcome,
                    num_transforms=5,
                    hidden_features=128,
                    num_blocks=2,
                    use_residual_blocks=True,
                    num_bins=8,
                    tails="linear",
                    tail_bound=3.0,
                    permutation=RANDOM_PERMUTATION,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=NFLOWS_INITIALISATION,
                    base_distribution=STANDARD_NORMAL,
                    mean_samples=100,
                ),
                CategoricalPropensity(
                    representation_dim=CNFLOW_ENCODER_WIDTHS[-1],
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
                        weight=1.0,
                        reduction="population",
                    ),
                ),
                trainable=(
                    "mlp_encoder",
                    "conditional_flow",
                    "categorical_propensity",
                ),
                rows="all",
                optimiser=OptimiserSpec(
                    name="adam",
                    lr=1e-3,
                    weight_decay=WeightDecay.none(),
                    lr_schedule=Constant(1.0),
                    clipping=GradientClipping.none(),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                ),
                steps=3_000,
            ),
        ),
        card="docs/recipes/cnflow.md",
        purpose="causal",
    )


__all__ = ["CNFLOW_ENCODER_WIDTHS", "cnflow"]
