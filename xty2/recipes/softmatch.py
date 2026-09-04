"""The SoftMatch assembly of ``docs/recipes/softmatch.md``, declarations only."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    CosineDecay,
    DataSpec,
    GradientClipping,
    MissingnessSpec,
    OptimiserSpec,
    Port,
    PreprocessSpec,
    Ramp,
    Recipe,
    RecomputeRule,
    Schema,
    SplitSpec,
    Stage,
    TeacherSpec,
    ViewSpec,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    SoftWeightedTreatmentNLL,
    TruncatedGaussianWeighting,
)
from xty2.recipes.fixmatch import (
    FIXMATCH_SAMPLER,
    FIXMATCH_STEPS,
    OBSERVED_TREATMENTS,
    PRESERVED_FIELDS,
    STRONG_X,
    WEAK_MASK_RATE,
    WEAK_X,
    WEAK_X_LABELLED,
)
from xty2.recipes.flexmatch import STRONG_MASK_RATE
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

SOFTMATCH_WEIGHTING = TruncatedGaussianWeighting(
    decay=0.999, n_sigma=2, alignment="uniform"
)
SOFTMATCH_STEPS = FIXMATCH_STEPS
UNSUPERVISED_WEIGHT = 1.0
SOFTMATCH_TERM = "soft_weighted_treatment_nll"

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6 "
            "fixture; no CIFAR/SVHN/STL protocol applies (deviation 1)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)


def softmatch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single-stage recipe reviewed in ``softmatch.md``."""
    return Recipe(
        name="softmatch",
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
                        ObservedTreatmentNLL(realisation=WEAK_X_LABELLED),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        SoftWeightedTreatmentNLL(
                            port=Port.T_GIVEN_X,
                            target=WEAK_X,
                            prediction=STRONG_X,
                            num_treatments=schema.treatment_cardinality,
                            weighting=SOFTMATCH_WEIGHTING,
                            sharpening="hard",
                            stop_grad="target",
                            rows="all",
                            name=SOFTMATCH_TERM,
                        ),
                        weight=UNSUPERVISED_WEIGHT,
                        reduction="mean",
                    ),
                    Weighted(
                        MissingTreatmentMarginalNLL(grad_path="both"),
                        weight=Ramp(0.0, 0.5, steps=1_000),
                        reduction="population",
                    ),
                ),
                trainable=("mlp_encoder", "tarnet_head", "categorical_propensity"),
                rows="all",
                teacher=TeacherSpec(
                    decay=0.999,
                    applies_to_buffers=False,
                    train_mode=False,
                    requires_grad=False,
                    role="evaluation",
                ),
                optimiser=OptimiserSpec(
                    name="sgd",
                    lr=0.03,
                    momentum=0.9,
                    nesterov=True,
                    weight_decay=WeightDecay(
                        value=5e-4, on_norm_and_bias=False, components=None
                    ),
                    lr_schedule=CosineDecay(steps=SOFTMATCH_STEPS, phase=7 / 16),
                    clipping=GradientClipping.none(),
                ),
                steps=SOFTMATCH_STEPS,
                sampler=FIXMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/softmatch.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="weak_x",
                transforms=(FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
                draws=2,
            ),
            ViewSpec(
                name="strong_x",
                transforms=(
                    FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),
                    FeatureMask(p=STRONG_MASK_RATE, columns=None, value=0.0),
                ),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = [
    "DATA_POLICY",
    "SOFTMATCH_STEPS",
    "SOFTMATCH_TERM",
    "SOFTMATCH_WEIGHTING",
    "UNSUPERVISED_WEIGHT",
    "softmatch",
]
