"""Declarative tabular SimSiam assembly; card sections 3-5 govern departures."""

from xty2.components import CategoricalPropensity, MLPEncoder, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION, TORCH_LINEAR_INITIALISATION
from xty2.components.simsiam import (
    ACTIVATION,
    PREDICTOR_INITIALISATION,
    PREDICTOR_NORMALISATION,
    PROJECTOR_INITIALISATION,
    PROJECTOR_NORMALISATION,
    SimSiamPredictor,
    SimSiamProjector,
)
from xty2.core import (
    ComponentGraph,
    Constant,
    CosineAnneal,
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
    UniformSampler,
    ViewSpec,
    ViewTransform,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    CosineFeatureConsistency,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)

ENCODER_WIDTHS = (256, 256, 256, 256)
PROJECTOR_WIDTHS = (256, 256, 256)
PREDICTOR_WIDTHS = (64, 256)
PRETRAIN_STEPS = 1000
JOINT_FIT_STEPS = 3000
BATCH_SIZE = 128
OBSERVED_TREATMENTS = 40
CORRUPTED_A = Realisation(view="corrupted_a")
CORRUPTED_B = Realisation(view="corrupted_b")
# `main_simsiam.main_worker`: init_lr = args.lr * args.batch_size / 256 with
# base lr 0.05 (paper section 4 "Baseline settings", linear scaling rule).
PRETRAIN_LR = 0.05 * BATCH_SIZE / 256
SGD = OptimiserSpec(
    name="sgd",
    lr=PRETRAIN_LR,
    momentum=0.9,
    # Supplement A: "a weight decay of 0.0001 for all parameter layers,
    # including the BN scales and biases".
    weight_decay=WeightDecay(value=1e-4, on_norm_and_bias=True, components=None),
    lr_schedule=CosineAnneal(steps=PRETRAIN_STEPS),
    clipping=GradientClipping.none(),
)
# Deviation 4: the downstream stage fits this repository's own heads, which the
# source has no counterpart for, so it keeps the local protocol.
ADAM = OptimiserSpec(
    name="adam",
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=WeightDecay.none(),
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
)
DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "fixed two-cluster DGP; disjoint train and held-out populations; "
            "no test-based selection"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="zscore", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)


def simsiam(
    schema: Schema,
    *,
    first_transforms: tuple[ViewTransform, ...],
    second_transforms: tuple[ViewTransform, ...],
    recompute_rules: tuple[RecomputeRule, ...] = (),
) -> Recipe:
    """SimSiam eq. (4), followed by the card's local XTY transfer protocol."""
    return Recipe(
        name="simsiam",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=ENCODER_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    # Paper supplement A: the source's fc layers take the torch
                    # defaults, and it warns that a fixed small std may not
                    # converge. Card deviation 7, withdrawn.
                    initialisation=TORCH_LINEAR_INITIALISATION,
                ),
                SimSiamProjector(
                    "simsiam_projector",
                    representation_dim=256,
                    widths=PROJECTOR_WIDTHS,
                    activation=ACTIVATION,
                    normalisation=PROJECTOR_NORMALISATION,
                    dropout=0.0,
                    initialisation=PROJECTOR_INITIALISATION,
                ),
                SimSiamPredictor(
                    "simsiam_predictor",
                    representation_dim=256,
                    widths=PREDICTOR_WIDTHS,
                    activation=ACTIVATION,
                    normalisation=PREDICTOR_NORMALISATION,
                    dropout=0.0,
                    initialisation=PREDICTOR_INITIALISATION,
                ),
                TARNetHead(
                    representation_dim=256,
                    num_treatments=schema.treatment_cardinality,
                    outcome=schema.outcome,
                    widths=(100, 100, 100),
                    activation="elu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=CFRNET_INITIALISATION,
                    output_parameterisation="K means; fixed Gaussian scale=1.0",
                ),
                CategoricalPropensity(
                    representation_dim=256,
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
                        CosineFeatureConsistency(
                            prediction_port=Port.X_PRED,
                            target_port=Port.X_PROJ,
                            prediction=CORRUPTED_A,
                            target=CORRUPTED_B,
                            stop_grad="target",
                            epsilon=1e-12,
                            rows="all",
                            name="simsiam_a_to_b",
                        ),
                        weight=0.5,
                        reduction="mean",
                    ),
                    Weighted(
                        CosineFeatureConsistency(
                            prediction_port=Port.X_PRED,
                            target_port=Port.X_PROJ,
                            prediction=CORRUPTED_B,
                            target=CORRUPTED_A,
                            stop_grad="target",
                            epsilon=1e-12,
                            rows="all",
                            name="simsiam_b_to_a",
                        ),
                        weight=0.5,
                        reduction="mean",
                    ),
                ),
                trainable=("mlp_encoder", "simsiam_projector", "simsiam_predictor"),
                rows="all",
                optimiser=SGD,
                steps=PRETRAIN_STEPS,
                sampler=UniformSampler(batch_size=BATCH_SIZE),
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
                        weight=Ramp(0.0, 0.5, steps=1000),
                        reduction="population",
                    ),
                ),
                trainable=("mlp_encoder", "tarnet_head", "categorical_propensity"),
                rows="all",
                initialise_from="pretrain",
                optimiser=ADAM,
                steps=JOINT_FIT_STEPS,
                sampler=UniformSampler(batch_size=BATCH_SIZE),
            ),
        ),
        card="docs/recipes/simsiam.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="corrupted_a",
                transforms=first_transforms,
                preserves=frozenset(
                    {
                        "t",
                        "y",
                        "t_observed",
                        "y_observed",
                        "row_id",
                        "fold_id",
                        "weight",
                    }
                ),
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="corrupted_b",
                transforms=second_transforms,
                preserves=frozenset(
                    {
                        "t",
                        "y",
                        "t_observed",
                        "y_observed",
                        "row_id",
                        "fold_id",
                        "weight",
                    }
                ),
                recompute_rules=recompute_rules,
            ),
        ),
    )
