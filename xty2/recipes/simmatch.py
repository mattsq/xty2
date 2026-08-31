"""The SimMatch assembly of ``docs/recipes/simmatch.md``, declarations only."""

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
    LabeledMemoryInstanceConsistency,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    SimilarityMatchingSpec,
    SimilarityMatchingTemperatures,
    SimilarityMatchingTreatmentNLL,
)
from xty2.recipes.fixmatch import STRONG_MASK_RATE, WEAK_MASK_RATE
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

WEAK_X = Realisation(view="weak_x")
STRONG_X = Realisation(view="strong_x")

SIMMATCH_STEPS = 3_000
LABELLED_BATCH = 64
MU = 7
OBSERVED_TREATMENTS = LABELLED_BATCH
PROJECTION_WIDTHS = (ENCODER_WIDTHS[-1], 128)
"""Card §7: the authors' head is `trunk_width -> trunk_width -> dim`, dim=128."""

INSTANCE_TEMPERATURES = SimilarityMatchingTemperatures(
    # Eqs. (3) and (4). The source exposes `tt` and `st` separately; the
    # published command sets both to 0.1, which is what this pair states.
    instance_weak=0.1,
    instance_strong=0.1,
)

SIMILARITY_MATCHING = SimilarityMatchingSpec(
    temperatures=INSTANCE_TEMPERATURES,
    alpha=0.9,
    memory_momentum=0.7,
    alignment_window=32,
    # Card §7: both ports disable propagation and the instance loss for epoch
    # 0. On the §6.1 fixture's 960 missing rows at 448 per step that epoch is
    # `floor(960/448) = 2` optimiser steps, printed rather than hidden behind
    # the word "epoch".
    warmup_steps=2,
    threshold=0.95,
    unfold=True,
)

MEMORY_TERM = "similarity_matching_treatment_nll"

SIMMATCH_SAMPLER = QuotaSampler(
    quotas=(
        Quota(rows="t_observed", size=LABELLED_BATCH),
        Quota(rows="t_missing", size=MU * LABELLED_BATCH),
    )
)

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "fixed project-local DGP in §6.1; no CIFAR or ImageNet "
            "split applies (deviation 10)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)

WEIGHT_DECAY = WeightDecay(
    value=5e-4,
    # Deviation 4: the controlled arm is `fixmatch`, so its exemptions are the
    # ones held fixed. The source's own CIFAR script is not pinned in Git and
    # its ImageNet code decays everything at 1e-4 instead (card §7).
    on_norm_and_bias=False,
    components=None,
)

OPTIMISER = OptimiserSpec(
    name="sgd",
    lr=0.03,
    momentum=0.9,
    nesterov=True,
    weight_decay=WEIGHT_DECAY,
    lr_schedule=CosineDecay(steps=SIMMATCH_STEPS, phase=7 / 16),
    clipping=GradientClipping.none(),
)

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def simmatch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the one-stage recipe from ``docs/recipes/simmatch.md``."""
    return Recipe(
        name="simmatch",
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
                    # Eq. (1), on the weak view, dividing by `B`.
                    Weighted(
                        ObservedTreatmentNLL(realisation=WEAK_X),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (2) with eq. (10)'s soft `hat p`, gated on that same
                    # propagated target.
                    Weighted(
                        SimilarityMatchingTreatmentNLL(
                            spec=SIMILARITY_MATCHING,
                            target=WEAK_X,
                            weak_embedding=WEAK_X,
                            prediction=STRONG_X,
                            num_treatments=schema.treatment_cardinality,
                            sharpening="none",
                            stop_grad="target",
                            support_rows="t_observed",
                            rows="t_missing",
                            name=MEMORY_TERM,
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (5) against eq. (8)'s calibrated `hat q`.
                    Weighted(
                        LabeledMemoryInstanceConsistency(
                            spec=SIMILARITY_MATCHING,
                            owner=MEMORY_TERM,
                            target=WEAK_X,
                            weak_embedding=WEAK_X,
                            prediction=STRONG_X,
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
                steps=SIMMATCH_STEPS,
                sampler=SIMMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/simmatch.md",
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
            ),
        ),
    )


__all__ = [
    "DATA_POLICY",
    "INSTANCE_TEMPERATURES",
    "LABELLED_BATCH",
    "MEMORY_TERM",
    "MU",
    "OBSERVED_TREATMENTS",
    "OPTIMISER",
    "PROJECTION_WIDTHS",
    "SIMILARITY_MATCHING",
    "SIMMATCH_SAMPLER",
    "SIMMATCH_STEPS",
    "STRONG_X",
    "WEAK_X",
    "WEIGHT_DECAY",
    "simmatch",
]
