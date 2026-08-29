"""The PAWS assembly of ``docs/recipes/paws.md``, declarations only."""

from __future__ import annotations

from xty2.components import (
    CategoricalPropensity,
    MLPEncoder,
    ProjectionHead,
    TARNetHead,
)
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    Constant,
    DataSpec,
    GradientClipping,
    MissingnessSpec,
    OptimiserSpec,
    PreprocessSpec,
    PreservedField,
    Quota,
    QuotaSampler,
    Ramp,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    SplitSpec,
    Stage,
    ViewSpec,
    WarmupCosine,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    MeanEntropyMaximisation,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    SupportSetClassifier,
    SupportSetPseudoLabelConsistency,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureCorruption

LARGE_X = (
    Realisation(view="paws_large_x", draw=0),
    Realisation(view="paws_large_x", draw=1),
)
"""The two large views, both prediction branches and swapped target sources."""

SMALL_X = tuple(Realisation(view="paws_small_x", draw=draw) for draw in range(6))
"""The six small views trained against the mean of the two large targets."""

TAU = 0.1
SHARPENING = 0.25
LABEL_SMOOTHING = 0.1
TARGET_FLOOR = 1e-4
LARGE_CORRUPTION_RATE = 0.25
SMALL_CORRUPTION_RATE = 0.5
PROJECTION_WIDTHS = (128, 128, 128)
SUPPORT_PER_TREATMENT = 16
MISSING_ANCHORS = 128
PRETRAIN_STEPS = 1_000
JOINT_FIT_STEPS = 3_000
OBSERVED_TREATMENTS = 64

SUPPORT_CLASSIFIER = SupportSetClassifier(
    temperature=TAU,
    label_smoothing=LABEL_SMOOTHING,
    support_rows="t_observed",
)
"""One value object shared by both pretraining terms."""

PAWS_SAMPLER = QuotaSampler(
    quotas=(
        Quota(rows="t_observed", size=SUPPORT_PER_TREATMENT, stratify="t"),
        Quota(rows="t_missing", size=MISSING_ANCHORS),
    )
)

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6 "
            "fixture; no CIFAR-10 or ImageNet protocol applies (deviation 7)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)

WEIGHT_DECAY = WeightDecay(
    value=1e-6,
    on_norm_and_bias=False,
    components=None,
)

PRETRAIN_OPTIMISER = OptimiserSpec(
    name="adam",
    lr=1e-3,
    weight_decay=WEIGHT_DECAY,
    lr_schedule=WarmupCosine(
        start=0.25,
        final=0.01,
        warmup=17,
        steps=PRETRAIN_STEPS,
    ),
    clipping=GradientClipping.none(),
    betas=(0.9, 0.999),
    eps=1e-8,
)

JOINT_FIT_OPTIMISER = OptimiserSpec(
    name="adam",
    lr=1e-3,
    weight_decay=WEIGHT_DECAY,
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
    betas=(0.9, 0.999),
    eps=1e-8,
)

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def paws(schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()) -> Recipe:
    """Build the two-stage recipe from ``docs/recipes/paws.md``."""
    return Recipe(
        name="paws",
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
                    Weighted(
                        SupportSetPseudoLabelConsistency(
                            classifier=SUPPORT_CLASSIFIER,
                            large=LARGE_X,
                            small=SMALL_X,
                            sharpening=SHARPENING,
                            stop_grad="target",
                            target_floor=TARGET_FLOOR,
                            rows="t_missing",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        MeanEntropyMaximisation(
                            classifier=SUPPORT_CLASSIFIER,
                            views=(*LARGE_X, *SMALL_X),
                            support_views=LARGE_X,
                            sharpening=SHARPENING,
                            rows="t_missing",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                trainable=("mlp_encoder", "projection_head"),
                rows="all",
                optimiser=PRETRAIN_OPTIMISER,
                steps=PRETRAIN_STEPS,
                sampler=PAWS_SAMPLER,
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
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                    "categorical_propensity",
                ),
                rows="all",
                initialise_from="pretrain",
                optimiser=JOINT_FIT_OPTIMISER,
                steps=JOINT_FIT_STEPS,
                sampler=PAWS_SAMPLER,
            ),
        ),
        card="docs/recipes/paws.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="paws_large_x",
                transforms=(
                    FeatureCorruption(rate=LARGE_CORRUPTION_RATE, columns=None),
                ),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
                draws=2,
            ),
            ViewSpec(
                name="paws_small_x",
                transforms=(
                    FeatureCorruption(rate=SMALL_CORRUPTION_RATE, columns=None),
                ),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
                draws=6,
            ),
        ),
    )


__all__ = [
    "DATA_POLICY",
    "JOINT_FIT_OPTIMISER",
    "JOINT_FIT_STEPS",
    "LABEL_SMOOTHING",
    "LARGE_CORRUPTION_RATE",
    "LARGE_X",
    "MISSING_ANCHORS",
    "OBSERVED_TREATMENTS",
    "PAWS_SAMPLER",
    "PRETRAIN_OPTIMISER",
    "PRETRAIN_STEPS",
    "PROJECTION_WIDTHS",
    "SHARPENING",
    "SMALL_CORRUPTION_RATE",
    "SMALL_X",
    "SUPPORT_CLASSIFIER",
    "SUPPORT_PER_TREATMENT",
    "TARGET_FLOOR",
    "TAU",
    "WEIGHT_DECAY",
    "paws",
]
