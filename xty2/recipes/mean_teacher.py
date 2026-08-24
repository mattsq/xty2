"""The reviewed Mean Teacher P9 assembly, declarations only."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    ExponentialDecay,
    GradientClipping,
    OptimiserSpec,
    Port,
    PreservedField,
    Ramp,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    SigmoidRamp,
    Stage,
    TeacherSpec,
    ViewSpec,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    ConsistencyLoss,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

STUDENT_X = Realisation(view="student_x")
TEACHER_X = Realisation(view="teacher_x", params="teacher")
PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def mean_teacher(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single-stage P9 recipe from `docs/recipes/mean_teacher.md`."""
    return Recipe(
        name="mean_teacher",
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
                        ObservedTreatmentNLL(realisation=STUDENT_X),
                        weight=1.0,
                        reduction="population",
                    ),
                    Weighted(
                        MissingTreatmentMarginalNLL(grad_path="both"),
                        weight=Ramp(0.0, 0.5, steps=1_000),
                        reduction="population",
                    ),
                    Weighted(
                        ConsistencyLoss(
                            port=Port.T_GIVEN_X,
                            left=STUDENT_X,
                            right=TEACHER_X,
                            divergence="mse",
                            stop_grad="right",
                            rows="all",
                        ),
                        weight=SigmoidRamp(
                            end=float(schema.treatment_cardinality), steps=40
                        ),
                        reduction="population",
                    ),
                ),
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                    "categorical_propensity",
                ),
                rows="all",
                teacher=TeacherSpec(
                    decay=0.99,
                    applies_to_buffers=False,
                    train_mode=True,
                    requires_grad=False,
                ),
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
        card="docs/recipes/mean_teacher.md",
        purpose="causal",
        views=(
            ViewSpec(
                name="student_x",
                transforms=(FeatureMask(p=0.1, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="teacher_x",
                transforms=(FeatureMask(p=0.1, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = ["mean_teacher"]
