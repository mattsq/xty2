"""The FixMatch assembly of `docs/recipes/fixmatch.md`, declarations only."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    CosineDecay,
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
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    PseudoLabelTreatmentNLL,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

WEAK_X = Realisation(view="weak_x")
"""`alpha(.)`: the view the artificial label and the supervised term read."""

STRONG_X = Realisation(view="strong_x")
"""`A(.)`: the view the pseudo-label is charged against."""

WEAK_MASK_RATE = 0.1
"""Card §7: the weak view reuses Mean Teacher's reviewed masking strength."""

STRONG_MASK_RATE = 0.5
"""Card §7: a deliberate step in strength, not a tuned value."""

FIXMATCH_STEPS = 3_000
"""Card §4. Also the `K` of the cosine rate schedule (card §5, deviation 3)."""

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def fixmatch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single-stage recipe from `docs/recipes/fixmatch.md`."""
    return Recipe(
        name="fixmatch",
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
                    # Eq. (3): the labelled cross-entropy is taken on the weak
                    # view, and divides by the labelled batch — `mean` over the
                    # term's own rows (card §3.2).
                    Weighted(
                        ObservedTreatmentNLL(realisation=WEAK_X),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (4), with lambda_u = 1 and no ramp: §2.2 states that
                    # the gate supplies the curriculum a ramp would otherwise
                    # provide.
                    Weighted(
                        PseudoLabelTreatmentNLL(
                            port=Port.T_GIVEN_X,
                            target=WEAK_X,
                            prediction=STRONG_X,
                            threshold=0.95,
                            sharpening="hard",
                            stop_grad="target",
                            rows="all",
                        ),
                        weight=1.0,
                        reduction="mean",
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
                    name="sgd",
                    lr=0.03,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=WeightDecay(
                        value=5e-4,
                        on_norm_and_bias=True,
                        components=None,
                    ),
                    lr_schedule=CosineDecay(steps=FIXMATCH_STEPS, phase=7 / 16),
                    clipping=GradientClipping.none(),
                ),
                steps=FIXMATCH_STEPS,
            ),
        ),
        card="docs/recipes/fixmatch.md",
        purpose="causal",
        views=(
            ViewSpec(
                name="weak_x",
                transforms=(FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="strong_x",
                transforms=(FeatureMask(p=STRONG_MASK_RATE, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = [
    "FIXMATCH_STEPS",
    "STRONG_MASK_RATE",
    "STRONG_X",
    "WEAK_MASK_RATE",
    "WEAK_X",
    "fixmatch",
]
