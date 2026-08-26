"""The SCARF assembly of `docs/recipes/scarf.md`, declarations only."""

from __future__ import annotations

from xty2.components import (
    CategoricalPropensity,
    MLPEncoder,
    ProjectionHead,
    TARNetHead,
)
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    DEFAULT,
    ComponentGraph,
    Constant,
    GradientClipping,
    OptimiserSpec,
    Port,
    PreservedField,
    Ramp,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    Stage,
    ViewSpec,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    InfoNCEContrastive,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureCorruption

CORRUPTED_X = Realisation(view="corrupted_x")
"""`x~`: the row with `floor(c M)` of its features redrawn from the marginals."""

ANCHOR_X = DEFAULT
"""`x`: SCARF embeds the *uncorrupted* row as the anchor, unlike SimCLR."""

CORRUPTION_RATE = 0.6
"""`c`, card §4. The paper's ablation: stable over 50-80%, recommended 60%."""

TEMPERATURE = 1.0
"""`tau`, card §4. The paper's ablation finds the untempered softmax best."""

PROJECTION_WIDTHS = (256, 256)
"""`g`: "2 layers", "hidden dimension 256" (card §4)."""

PRETRAIN_STEPS = 1_000
"""Card §4 and deviation 4: a fixed budget where the paper early-stops."""

JOINT_FIT_STEPS = 3_000
"""Card §4: the project-local fitting budget every other recipe uses."""

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)

ADAM = OptimiserSpec(
    name="adam",
    lr=1e-3,
    weight_decay=WeightDecay.none(),
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
    betas=(0.9, 0.999),
    eps=1e-8,
)
"""Card §4: "the Adam optimizer using the default learning rate of 0.001".

One value object for both stages, because the paper names the same optimiser
for pretraining and for fine-tuning and a second literal would be two places to
edit one reviewed decision. `OptimiserSpec` is frozen, so sharing it shares no
state — the executor builds a fresh optimiser per stage (card §7).
"""


def scarf(schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()) -> Recipe:
    """Build the two-stage recipe from `docs/recipes/scarf.md`."""
    return Recipe(
        name="scarf",
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
                ProjectionHead(
                    representation_dim=ENCODER_WIDTHS[-1],
                    widths=PROJECTION_WIDTHS,
                    activation="relu",
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
                name="pretrain",
                objectives=(
                    # `L_cont`: the anchor is the uncorrupted row, the contrast
                    # its corrupted copy, and the mean is over the batch — the
                    # term's own rows, which is `mean` (card §3.2).
                    Weighted(
                        InfoNCEContrastive(
                            port=Port.X_PROJ,
                            anchor=ANCHOR_X,
                            contrast=CORRUPTED_X,
                            temperature=TEMPERATURE,
                            rows="all",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                # `f` and `g`, and nothing else: the outcome and propensity
                # heads read no port this stage computes, so naming them here
                # would be the dead-weight stage the compiler rejects.
                trainable=("mlp_encoder", "projection_head"),
                rows="all",
                optimiser=ADAM,
                steps=PRETRAIN_STEPS,
            ),
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
                # "After pre-training, g is discarded": `projection_head` is in
                # no forward pass of this stage and in no trainable list, and
                # "both f and h are subsequently fine-tuned" is why the encoder
                # is in both.
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                    "categorical_propensity",
                ),
                rows="all",
                initialise_from="pretrain",
                optimiser=ADAM,
                steps=JOINT_FIT_STEPS,
            ),
        ),
        card="docs/recipes/scarf.md",
        purpose="causal",
        views=(
            ViewSpec(
                name="corrupted_x",
                transforms=(FeatureCorruption(rate=CORRUPTION_RATE, columns=None),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = [
    "ADAM",
    "ANCHOR_X",
    "CORRUPTED_X",
    "CORRUPTION_RATE",
    "JOINT_FIT_STEPS",
    "PRETRAIN_STEPS",
    "PROJECTION_WIDTHS",
    "TEMPERATURE",
    "scarf",
]
