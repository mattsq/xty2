"""The declarative Meta Pseudo Labels assembly from its reviewed card."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    META_GRADIENT_ORDER,
    ComponentGraph,
    CosineDecay,
    DataSpec,
    GradientClipping,
    MetaGradientSpec,
    MissingnessSpec,
    OptimiserSpec,
    ParameterRole,
    Port,
    PreprocessSpec,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    SplitSpec,
    Stage,
    ViewSpec,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    ConfidenceMaskedConsistencyLoss,
    MetaFeedbackCoefficient,
    MetaPseudoLabelScore,
    ObservedTreatmentNLL,
    SampledTeacherTreatmentNLL,
    TrainingSignalAnnealedTreatmentNLL,
)
from xty2.recipes.fixmatch import (
    FIXMATCH_SAMPLER,
    OBSERVED_TREATMENTS,
    PRESERVED_FIELDS,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS
from xty2.recipes.uda import (
    TARGET_TEMPERATURE,
    UDA_STEPS,
    UDA_STRONG_MASK_RATE,
    UDA_THRESHOLDS,
)
from xty2.views import FeatureMask

OUTER_ROLE = "outer_teacher"
INNER_ROLE = "inner_student"

OUTER_WEAK = Realisation(view="weak_x", role=OUTER_ROLE)
OUTER_STRONG = Realisation(view="strong_x", role=OUTER_ROLE)
INNER_STRONG = Realisation(view="strong_x", role=INNER_ROLE)
INNER_WEAK_POST = Realisation(view="weak_x", role=INNER_ROLE, state="post_update")

META_FEEDBACK = MetaFeedbackCoefficient(
    kind="cosine_similarity",
    baseline_decay=0.99,
    baseline_initial=0.0,
    baseline_order="update_then_subtract",
)

OPTIMISER = OptimiserSpec(
    name="sgd",
    lr=0.03,
    momentum=0.9,
    nesterov=True,
    weight_decay=WeightDecay(
        value=5e-4,
        on_norm_and_bias=True,
        components=None,
    ),
    lr_schedule=CosineDecay(steps=UDA_STEPS, phase=7 / 16),
    clipping=GradientClipping.none(),
)

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6.1 "
            "fixture; no image protocol applies (deviation 1)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="none"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)


def meta_pseudo_labels(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single atomic meta-gradient stage; recipes contain no loop."""
    system = ComponentGraph(
        [
            MLPEncoder(
                input_dim=schema.num_features,
                widths=ENCODER_WIDTHS,
                activation="elu",
                normalisation="row_l2",
                dropout=0.0,
                initialisation=CFRNET_INITIALISATION,
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
    )
    trainable = ("mlp_encoder", "categorical_propensity")
    return Recipe(
        name="meta_pseudo_labels",
        schema=schema,
        system=system,
        program=(
            Stage(
                name="meta_train",
                executor="meta_gradient",
                roles=(
                    ParameterRole(
                        name=OUTER_ROLE,
                        trainable=trainable,
                        optimiser=OPTIMISER,
                    ),
                    ParameterRole(
                        name=INNER_ROLE,
                        trainable=trainable,
                        optimiser=OPTIMISER,
                    ),
                ),
                meta_gradient=MetaGradientSpec(
                    inner_role=INNER_ROLE,
                    outer_role=OUTER_ROLE,
                    inner_objective="student_pseudo_label_nll",
                    feedback_objective="student_labelled_feedback_nll",
                    meta_objective="teacher_meta_score",
                    outer_objectives=(
                        "teacher_tsa_nll",
                        "teacher_uda_consistency",
                    ),
                    feedback=META_FEEDBACK,
                    update_order=META_GRADIENT_ORDER,
                    inner_steps=1,
                    outer_steps=1,
                ),
                objectives=(
                    Weighted(
                        SampledTeacherTreatmentNLL(
                            port=Port.T_GIVEN_X,
                            teacher=OUTER_STRONG,
                            student=INNER_STRONG,
                            temperature=1.0,
                            sharpening="none",
                            confidence_threshold="none",
                            detached_target=(
                                "hard categorical sample and outer_teacher graph"
                            ),
                            treatment_encoding=(
                                "integer classes 0..K-1; one categorical draw per "
                                "t_missing row"
                            ),
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        ObservedTreatmentNLL(
                            name="student_labelled_feedback_nll",
                            realisation=INNER_WEAK_POST,
                        ),
                        weight=0.0,
                        reduction="mean",
                    ),
                    Weighted(
                        MetaPseudoLabelScore(
                            port=Port.T_GIVEN_X,
                            teacher=OUTER_STRONG,
                            detached_target=(
                                "hard categorical sample and centred feedback "
                                "coefficient"
                            ),
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        TrainingSignalAnnealedTreatmentNLL(
                            name="teacher_tsa_nll",
                            port=Port.T_GIVEN_X,
                            realisation=OUTER_WEAK,
                            thresholds=UDA_THRESHOLDS,
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        ConfidenceMaskedConsistencyLoss(
                            name="teacher_uda_consistency",
                            port=Port.T_GIVEN_X,
                            target=OUTER_WEAK,
                            prediction=OUTER_STRONG,
                            thresholds=UDA_THRESHOLDS,
                            target_temperature=TARGET_TEMPERATURE,
                            sharpening="softmax_temperature",
                            stop_grad="target",
                            divergence="kl",
                            rows="t_missing",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                rows="all",
                steps=UDA_STEPS,
                sampler=FIXMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/meta_pseudo_labels.md",
        purpose="predictive",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="weak_x",
                transforms=(FeatureMask(p=0.1, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="strong_x",
                transforms=(
                    FeatureMask(p=0.1, columns=None, value=0.0),
                    FeatureMask(p=UDA_STRONG_MASK_RATE, columns=None, value=0.0),
                ),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = [
    "DATA_POLICY",
    "INNER_ROLE",
    "INNER_STRONG",
    "INNER_WEAK_POST",
    "META_FEEDBACK",
    "OPTIMISER",
    "OUTER_ROLE",
    "OUTER_STRONG",
    "OUTER_WEAK",
    "meta_pseudo_labels",
]
