"""Declarative tabular BYOL assembly; card sections 3-5 govern departures."""

from xty2.components import (
    BYOLPredictor,
    BYOLProjector,
    CategoricalPropensity,
    MLPEncoder,
    TARNetHead,
)
from xty2.components._nn import CFRNET_INITIALISATION, TORCH_LINEAR_INITIALISATION
from xty2.components.byol import ACTIVATION, INITIALISATION, NORMALISATION
from xty2.core import (
    ComponentGraph,
    Constant,
    CosineEMADecay,
    DataSpec,
    GradientClipping,
    MissingnessSpec,
    OptimiserSpec,
    Port,
    PreprocessSpec,
    PreservedField,
    Ramp,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    SplitSpec,
    Stage,
    TeacherSpec,
    UniformSampler,
    ViewSpec,
    ViewTransform,
    WarmupCosine,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    MissingTreatmentMarginalNLL,
    NormalizedSquaredFeatureConsistency,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)

ENCODER_WIDTHS = (256, 256, 256, 256)
# `configs/byol.py`: projector_hidden_size=4096, projector_output_size=256,
# predictor_hidden_size=4096, and the predictor reuses the projector's output
# size. Both heads are one `networks.MLP`, so both are (hidden, output).
PROJECTOR_WIDTHS = (4096, 256)
PREDICTOR_WIDTHS = (4096, 256)
REPRESENTATION_DIM = ENCODER_WIDTHS[-1]
PROJECTION_DIM = PROJECTOR_WIDTHS[-1]
PRETRAIN_STEPS = 1000
JOINT_FIT_STEPS = 3000
BATCH_SIZE = 128
OBSERVED_TREATMENTS = 40
# Deviation 3: ten warm-up epochs of a thousand become ten steps of a thousand,
# which is the same 1% of the horizon. `learning_schedule` warms linearly from
# zero and then runs a half cosine over the remaining steps, all the way down —
# `_cosine_decay(..., total_steps - warmup_steps, scaled_lr)` reaches zero at
# the horizon, so the final multiplier is 0 rather than a floor the card would
# otherwise have to invent.
WARMUP_STEPS = 10
PRETRAIN_FINAL_LR_MULTIPLIER = 0.0
# `_LR_PRESETS[1000] = 0.2`, scaled by `batch_size / 256` in `learning_schedule`.
PRETRAIN_LR = 0.2 * BATCH_SIZE / 256
# `_WD_PRESETS[1000]`, and `optimizer_config` eta and momentum.
PRETRAIN_WEIGHT_DECAY = 1.5e-6
LARS_ETA = 1e-3
LARS_MOMENTUM = 0.9
# Deviation 7. `configs/byol.py` keys every preset to the epoch budget and hands
# `target_ema` a horizon of `num_epochs * train_images_per_epoch // batch_size`,
# so `base_target_ema` is a function of run length rather than a constant: the
# source itself uses 0.97 at 40 epochs and 0.996 at 1000. Deviation 3 shortens
# the horizon to PRETRAIN_STEPS, which is 125 epochs of this fixture, so the row
# that budget reaches is `_EMA_PRESETS[100]` and its base is translated to the
# local horizon by the one invariant the curve has: the EMA time constant as a
# fraction of the pretraining horizon,
#
#     1 - base_local = (1 - base_source) * steps_source / steps_local
#
# Inheriting `_EMA_PRESETS[1000] = 0.996` at this horizon is not a smaller
# version of the source's regime; it is a different one. Under the rule above
# the selected 1000-epoch row translates to a base of -0.251, outside the
# `[0, 1)` every EMA update requires, so that row is unreachable here at any
# base — which is the §5 row 7 finding rather than a licence to clamp it.
SOURCE_IMAGES_PER_EPOCH = 1281167
SOURCE_BATCH_SIZE = 4096
SOURCE_EMA_EPOCHS = 100
SOURCE_EMA_BASE = 0.99
SOURCE_EMA_STEPS = SOURCE_EMA_EPOCHS * SOURCE_IMAGES_PER_EPOCH // SOURCE_BATCH_SIZE
BASE_TARGET_EMA = 1.0 - (1.0 - SOURCE_EMA_BASE) * SOURCE_EMA_STEPS / PRETRAIN_STEPS

ONLINE_A = Realisation(view="byol_a")
ONLINE_B = Realisation(view="byol_b")
TARGET_A = Realisation(view="byol_a", params="teacher")
TARGET_B = Realisation(view="byol_b", params="teacher")

LARS = OptimiserSpec(
    name="lars",
    lr=PRETRAIN_LR,
    momentum=LARS_MOMENTUM,
    eta=LARS_ETA,
    # `optimizers.exclude_bias_and_norm` filters the decay as well as the trust
    # ratio, so biases and BN scales are exempt from both. The adaptation half
    # of that is the optimiser's; this declares the decay half.
    weight_decay=WeightDecay(
        value=PRETRAIN_WEIGHT_DECAY, on_norm_and_bias=False, components=None
    ),
    lr_schedule=WarmupCosine(
        start=0.0,
        final=PRETRAIN_FINAL_LR_MULTIPLIER,
        warmup=WARMUP_STEPS,
        steps=PRETRAIN_STEPS,
    ),
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
PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def byol(
    schema: Schema,
    *,
    first_transforms: tuple[ViewTransform, ...],
    second_transforms: tuple[ViewTransform, ...],
    recompute_rules: tuple[RecomputeRule, ...] = (),
) -> Recipe:
    """BYOL eqs. (1)-(3), followed by the card's local XTY transfer protocol."""
    return Recipe(
        name="byol",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=ENCODER_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    # Deviation 6: the source's Haiku defaults are a truncated
                    # normal this backend has no equivalent of, so the card
                    # names torch's own and claims no numerical equivalence.
                    initialisation=TORCH_LINEAR_INITIALISATION,
                ),
                BYOLProjector(
                    "byol_projector",
                    input_dim=REPRESENTATION_DIM,
                    widths=PROJECTOR_WIDTHS,
                    activation=ACTIVATION,
                    normalisation=NORMALISATION,
                    dropout=0.0,
                    initialisation=INITIALISATION,
                ),
                BYOLPredictor(
                    "byol_predictor",
                    input_dim=PROJECTION_DIM,
                    widths=PREDICTOR_WIDTHS,
                    activation=ACTIVATION,
                    normalisation=NORMALISATION,
                    dropout=0.0,
                    initialisation=INITIALISATION,
                ),
                TARNetHead(
                    representation_dim=REPRESENTATION_DIM,
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
                    representation_dim=REPRESENTATION_DIM,
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
                    # `loss_fn`: prediction_view1 against projection_view2, then
                    # prediction_view2 against projection_view1, summed and
                    # meaned once. Two terms of weight 1 under `mean` is that
                    # sum; a half-weight each would be SimSiam's convention.
                    Weighted(
                        NormalizedSquaredFeatureConsistency(
                            prediction_port=Port.X_PRED,
                            target_port=Port.X_PROJ,
                            prediction=ONLINE_A,
                            target=TARGET_B,
                            stop_grad="target",
                            epsilon=1e-12,
                            rows="all",
                            name="byol_a_to_b",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        NormalizedSquaredFeatureConsistency(
                            prediction_port=Port.X_PRED,
                            target_port=Port.X_PROJ,
                            prediction=ONLINE_B,
                            target=TARGET_A,
                            stop_grad="target",
                            epsilon=1e-12,
                            rows="all",
                            name="byol_b_to_a",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                ),
                trainable=("mlp_encoder", "byol_projector", "byol_predictor"),
                rows="all",
                teacher=TeacherSpec(
                    # `schedules.target_ema` over the stage's own horizon: the
                    # executor updates at global steps 0..PRETRAIN_STEPS-1, and
                    # the curve's endpoint at PRETRAIN_STEPS is never applied.
                    decay=CosineEMADecay(base=BASE_TARGET_EMA, steps=PRETRAIN_STEPS),
                    # `_update_fn` returns `net_states['target_state']` — the
                    # target's BN statistics come from its own forwards, not
                    # from an EMA of the online ones.
                    applies_to_buffers=False,
                    # `loss_fn` applies both networks with `is_training=True`.
                    train_mode=True,
                    requires_grad=False,
                    role="consistency_target",
                ),
                optimiser=LARS,
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
                # Deviation 4, as amended: the source evaluates a *frozen*
                # backbone (paper §3.3, "linear evaluation"), so the encoder
                # transfers and is held fixed. Only the XTY heads this
                # repository adds are fitted here.
                trainable=("tarnet_head", "categorical_propensity"),
                rows="all",
                initialise_from="pretrain",
                optimiser=ADAM,
                steps=JOINT_FIT_STEPS,
                sampler=UniformSampler(batch_size=BATCH_SIZE),
            ),
        ),
        card="docs/recipes/byol.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="byol_a",
                transforms=first_transforms,
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="byol_b",
                transforms=second_transforms,
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = ["byol"]
