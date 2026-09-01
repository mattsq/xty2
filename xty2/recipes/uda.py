"""The UDA assembly of ``docs/recipes/uda.md``, declarations only."""

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
    Realisation,
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
    ConfidenceMaskedConsistencyLoss,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    TrainingSignalAnnealedTreatmentNLL,
    UDAConfidenceThresholds,
)
from xty2.recipes.fixmatch import (
    FIXMATCH_SAMPLER,
    OBSERVED_TREATMENTS,
    PRESERVED_FIELDS,
    WEAK_MASK_RATE,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

WEAK_X = Realisation(view="weak_x")
"""The current student's ordinary weak-view distribution."""

STRONG_X = Realisation(view="strong_x")
"""The stronger view charged against the detached weak target."""

UDA_STEPS = 3_000
"""Card §4's project-local horizon and every step-based schedule's ``T``."""

UDA_THRESHOLDS = UDAConfidenceThresholds(
    unsupervised=0.8,
    tsa_schedule="exp_schedule",
    scale=5.0,
    total_steps=UDA_STEPS,
)
"""The fixed consistency gate and appendix A.1's exponential TSA policy."""

TARGET_TEMPERATURE = 0.4
"""The CIFAR-10/SVHN/ImageNet weak-target temperature from section 2.4."""

UDA_STRONG_MASK_RATE = 0.1
"""The second strong-view mask, limited by card §6's label-flip guardrail."""

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6.1 "
            "fixture; no CIFAR/SVHN protocol applies (deviation 1)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)
"""The four ``data.*`` card keys."""


def uda(schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()) -> Recipe:
    """Build the single-stage recipe from ``docs/recipes/uda.md``."""
    return Recipe(
        name="uda",
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
                        TrainingSignalAnnealedTreatmentNLL(
                            port=Port.T_GIVEN_X,
                            realisation=WEAK_X,
                            thresholds=UDA_THRESHOLDS,
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        ConfidenceMaskedConsistencyLoss(
                            port=Port.T_GIVEN_X,
                            target=WEAK_X,
                            prediction=STRONG_X,
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
                teacher=TeacherSpec(
                    decay=0.9999,
                    applies_to_buffers=True,
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
                        value=5e-4,
                        on_norm_and_bias=True,
                        components=None,
                    ),
                    lr_schedule=CosineDecay(steps=UDA_STEPS, phase=7 / 16),
                    clipping=GradientClipping.none(),
                ),
                steps=UDA_STEPS,
                sampler=FIXMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/uda.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="weak_x",
                transforms=(FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="strong_x",
                transforms=(
                    FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),
                    FeatureMask(p=UDA_STRONG_MASK_RATE, columns=None, value=0.0),
                ),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = [
    "DATA_POLICY",
    "STRONG_X",
    "TARGET_TEMPERATURE",
    "UDA_STEPS",
    "UDA_STRONG_MASK_RATE",
    "UDA_THRESHOLDS",
    "WEAK_X",
    "uda",
]
