"""The CoMatch assembly of ``docs/recipes/comatch.md``, declarations only."""

from __future__ import annotations

from xty2.components import (
    CategoricalPropensity,
    MLPEncoder,
    ProjectionHead,
    TARNetHead,
)
from xty2.components._nn import CFRNET_INITIALISATION, TORCH_LINEAR_INITIALISATION
from xty2.core import (
    ComponentGraph,
    CosineDecay,
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
    TeacherSpec,
    ViewSpec,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    CoMatchConfidenceThresholds,
    MemorySmoothedLabelGraph,
    MemorySmoothedPseudoLabelTreatmentNLL,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    PseudoLabelGraphContrastive,
)
from xty2.recipes.fixmatch import STRONG_MASK_RATE, WEAK_MASK_RATE
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

WEAK_X = Realisation(view="weak_x")
STRONG_X = (
    Realisation(view="strong_x", draw=0),
    Realisation(view="strong_x", draw=1),
)

COMATCH_STEPS = 3_000
LABELLED_BATCH = 64
MU = 7
OBSERVED_TREATMENTS = LABELLED_BATCH
PROJECTION_WIDTHS = (ENCODER_WIDTHS[-1], 64)

CONFIDENCE_THRESHOLDS = CoMatchConfidenceThresholds(
    pseudo_label=0.95,
    edge=0.8,
)

LABEL_GRAPH = MemorySmoothedLabelGraph(
    temperature=0.2,
    alpha=0.9,
    capacity=2_560,
    thresholds=CONFIDENCE_THRESHOLDS,
    alignment_window=32,
    # The source condition is zero-based `it > queue_batch`, queue_batch=5:
    # iterations 0 through 5 are unsmoothed and iteration 6 is the first that
    # applies eq. (8).
    unsmoothed_steps=6,
)

PSEUDO_LABEL_TERM = "memory_smoothed_pseudo_label_treatment_nll"

COMATCH_SAMPLER = QuotaSampler(
    quotas=(
        Quota(rows="t_observed", size=LABELLED_BATCH),
        Quota(rows="t_missing", size=MU * LABELLED_BATCH),
    )
)

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6.1 "
            "fixture; no CIFAR-10, STL-10 or ImageNet protocol applies "
            "(deviation 9)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)

WEIGHT_DECAY = WeightDecay(
    value=5e-4,
    # The source exempts only parameter names containing `bn`.  The adapted
    # graph has no BatchNorm, so its matrices and ordinary biases all decay.
    on_norm_and_bias=True,
    components=None,
)

OPTIMISER = OptimiserSpec(
    name="sgd",
    lr=0.03,
    momentum=0.9,
    nesterov=True,
    weight_decay=WEIGHT_DECAY,
    lr_schedule=CosineDecay(steps=COMATCH_STEPS, phase=7 / 16),
    clipping=GradientClipping.none(),
)

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def comatch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the one-stage recipe from ``docs/recipes/comatch.md``."""
    return Recipe(
        name="comatch",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=ENCODER_WIDTHS,
                    activation="elu",
                    normalisation="row_l2",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                ),
                ProjectionHead(
                    representation_dim=ENCODER_WIDTHS[-1],
                    widths=PROJECTION_WIDTHS,
                    activation="leaky_relu:0.1",
                    normalisation="row_l2",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
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
                    Weighted(
                        ObservedOutcomeNLL(),
                        weight=1.0,
                        reduction="population",
                    ),
                    Weighted(
                        ObservedTreatmentNLL(realisation=WEAK_X),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        MemorySmoothedPseudoLabelTreatmentNLL(
                            graph=LABEL_GRAPH,
                            target=WEAK_X,
                            weak_embedding=WEAK_X,
                            prediction=STRONG_X[0],
                            num_treatments=schema.treatment_cardinality,
                            sharpening="none",
                            stop_grad="target",
                            support_rows="t_observed",
                            rows="t_missing",
                            name=PSEUDO_LABEL_TERM,
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        PseudoLabelGraphContrastive(
                            graph=LABEL_GRAPH,
                            labels=PSEUDO_LABEL_TERM,
                            target=WEAK_X,
                            weak_embedding=WEAK_X,
                            anchor=STRONG_X[0],
                            contrast=STRONG_X[1],
                            num_treatments=schema.treatment_cardinality,
                            support_rows="t_observed",
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
                    "projection_head",
                    "tarnet_head",
                    "categorical_propensity",
                ),
                rows="all",
                teacher=TeacherSpec(
                    decay=0.999,
                    applies_to_buffers=False,
                    train_mode=False,
                    requires_grad=False,
                    role="evaluation",
                ),
                optimiser=OPTIMISER,
                steps=COMATCH_STEPS,
                sampler=COMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/comatch.md",
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
                    FeatureMask(p=STRONG_MASK_RATE, columns=None, value=0.0),
                ),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
                draws=2,
            ),
        ),
    )


__all__ = [
    "COMATCH_SAMPLER",
    "COMATCH_STEPS",
    "CONFIDENCE_THRESHOLDS",
    "DATA_POLICY",
    "LABELLED_BATCH",
    "LABEL_GRAPH",
    "MU",
    "OBSERVED_TREATMENTS",
    "OPTIMISER",
    "PROJECTION_WIDTHS",
    "PSEUDO_LABEL_TERM",
    "STRONG_X",
    "WEAK_X",
    "WEIGHT_DECAY",
    "comatch",
]
