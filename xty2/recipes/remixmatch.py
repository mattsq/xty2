"""The reviewed ReMixMatch assembly; declarations only."""

from __future__ import annotations

from typing import Literal

from xty2.components import CategoricalPropensity, MLPEncoder, PretextHead, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    Constant,
    DataSpec,
    GradientClipping,
    MissingnessSpec,
    MixMember,
    MixSpec,
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
    AnchoredTargetTreatmentNLL,
    MissingTreatmentMarginalNLL,
    MixedTargetTreatmentNLL,
    ObservedOutcomeNLL,
    PretextTransformNLL,
)
from xty2.recipes.fixmatch import STRONG_MASK_RATE, WEAK_MASK_RATE
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import ColumnRoll, FeatureMask

REMIXMATCH_STEPS = 3_000
LABELLED_BATCH = 64
UNLABELLED_BATCH = 64
STRONG_DRAWS = 8
ANCHOR = Realisation(view="weak_x")
STRONG = tuple(
    Realisation(view="strong_x", draw=index) for index in range(STRONG_DRAWS)
)
PRETEXT = Realisation(view="pretext_x")
GUESS_OWNER = "premixup_treatment_nll"
PRESERVED: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by section 6.1; "
            "no CIFAR-10, SVHN or STL-10 protocol applies (deviation 7)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=64),
)
SAMPLER = QuotaSampler(
    quotas=(
        Quota(rows="t_observed", size=LABELLED_BATCH),
        Quota(rows="t_missing", size=UNLABELLED_BATCH),
    )
)


def remixmatch(
    schema: Schema,
    *,
    recompute_rules: tuple[RecomputeRule, ...] = (),
    use_alignment: bool = True,
    mix_rule: str = "max",
    strong_draws: int = STRONG_DRAWS,
    pretext_weight: float = 0.5,
    premixup_weight: float = 0.5,
    unsupervised_weight: float = 1.5,
    redux: Literal["1st", "mean"] = "1st",
) -> Recipe:
    """Build the one-stage recipe and its declared mechanism ablations."""
    if not 1 <= strong_draws <= STRONG_DRAWS:
        raise ValueError(f"strong_draws must lie in [1, {STRONG_DRAWS}]")
    strong = STRONG[:strong_draws]
    target_copies = strong if redux == "mean" else ()
    members = (
        MixMember(strong[0], "t_observed"),
        *(MixMember(value, "t_missing") for value in strong),
        MixMember(ANCHOR, "t_missing"),
    )
    mixing = MixSpec(name="mixed", members=members, alpha=0.75, rule=mix_rule)
    return Recipe(
        name="remixmatch",
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
                PretextHead(
                    representation_dim=ENCODER_WIDTHS[-1],
                    num_transforms=4,
                    activation="linear logits",
                    normalisation="none",
                    dropout=0.0,
                    initialisation="glorot_normal, bias=0",
                    output_parameterisation="4 softmax logits",
                ),
            ]
        ),
        program=(
            Stage(
                name="joint_fit",
                objectives=(
                    Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="population"),
                    Weighted(
                        MixedTargetTreatmentNLL(
                            owner=GUESS_OWNER,
                            anchor=ANCHOR,
                            predictions=(mixing.output(0),),
                            num_treatments=schema.treatment_cardinality,
                            first_target="observed",
                            anchor_copies=target_copies,
                            redux=redux,
                            rows="t_observed",
                            name="mixed_labelled_treatment_nll",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    Weighted(
                        MixedTargetTreatmentNLL(
                            owner=GUESS_OWNER,
                            anchor=ANCHOR,
                            predictions=tuple(
                                mixing.output(index) for index in range(1, len(members))
                            ),
                            num_treatments=schema.treatment_cardinality,
                            first_target="anchor",
                            anchor_copies=target_copies,
                            redux=redux,
                            rows="t_missing",
                        ),
                        weight=Ramp(0.0, unsupervised_weight, steps=47),
                        reduction="mean",
                    ),
                    Weighted(
                        AnchoredTargetTreatmentNLL(
                            target=ANCHOR,
                            prediction=strong[0],
                            num_treatments=schema.treatment_cardinality,
                            target_copies=target_copies,
                            temperature=0.5,
                            sharpening="probability_power_temperature",
                            stop_grad="target",
                            confidence_threshold="n/a",
                            use_alignment=use_alignment,
                            augmentation_count=strong_draws,
                            redux=redux,
                            name=GUESS_OWNER,
                        ),
                        weight=Ramp(0.0, premixup_weight, steps=47),
                        reduction="mean",
                    ),
                    Weighted(
                        PretextTransformNLL(realisation=PRETEXT),
                        weight=pretext_weight,
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
                    "pretext_head",
                ),
                rows="all",
                teacher=TeacherSpec(
                    decay=0.999,
                    applies_to_buffers=False,
                    train_mode=False,
                    requires_grad=False,
                    role="evaluation",
                ),
                optimiser=OptimiserSpec(
                    name="adamw",
                    lr=0.002,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=WeightDecay(
                        value=0.02,
                        on_norm_and_bias=False,
                        components=(
                            "mlp_encoder",
                            "categorical_propensity",
                            "pretext_head",
                        ),
                    ),
                    lr_schedule=Constant(1.0),
                    clipping=GradientClipping.none(),
                ),
                steps=REMIXMATCH_STEPS,
                sampler=SAMPLER,
            ),
        ),
        card="docs/recipes/remixmatch.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="weak_x",
                transforms=(FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),),
                preserves=PRESERVED,
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="strong_x",
                transforms=(
                    FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),
                    FeatureMask(p=STRONG_MASK_RATE, columns=None, value=0.0),
                ),
                preserves=PRESERVED,
                recompute_rules=recompute_rules,
                draws=strong_draws,
            ),
            ViewSpec(
                name="pretext_x",
                transforms=(ColumnRoll(),),
                preserves=PRESERVED,
                recompute_rules=recompute_rules,
                source=strong[0],
            ),
        ),
        mixes=(mixing,),
    )


__all__ = ["ANCHOR", "GUESS_OWNER", "REMIXMATCH_STEPS", "remixmatch"]
