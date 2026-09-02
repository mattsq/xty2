"""Published TARNet and the separately named xty2 missing-treatment extension."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    DataSpec,
    ExponentialDecay,
    GradientClipping,
    MissingnessSpec,
    OptimiserSpec,
    PreprocessSpec,
    Ramp,
    Recipe,
    Schema,
    SplitSpec,
    Stage,
    UniformSampler,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    MissingTreatmentMarginalNLL,
    ObservedOutcomeMSE,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)

ENCODER_WIDTHS = (200, 200, 200)
OUTCOME_WIDTHS = (100, 100, 100)

BATCH_SIZE = 100
"""`optimisation.batch_size`, card §4: the pinned reference implementation's.

Card §4 used to read `n/a  # external BatchSource; ref impl uses 100` — the
number was known, and there was nowhere to put it. Deviation 5 is what that
cost, and this is where it is repaid.
"""

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "the archive's own realisation, fit/validation split 70/30 by "
            "seeded permutation as the reference loader does"
        ),
        train="fit",
    ),
    # "the IHDP archive x is passed through unchanged by the reference loader",
    # card §4. Declaring `none` is not a no-op: it is what makes the plan say
    # so, and what the fitted-on check has to agree with.
    preprocess=PreprocessSpec(features="none", outcome="none"),
    # TARNet's data arrives fully observed. The 50% MCAR of Tier 1 belongs to
    # the fixture that generates it and is card §5 deviation 4's business, not
    # a property of the method — so the recipe declares that it consumes
    # whatever mask the data carries.
    missingness=MissingnessSpec(mechanism="observed"),
)
"""The four `data.*` card keys, in the one place a plan can print them."""


def tarnet(schema: Schema) -> Recipe:
    """Build Shalit et al.'s outcome-only TARNet baseline."""
    return Recipe(
        name="tarnet",
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
            ]
        ),
        program=(
            Stage(
                name="joint_fit",
                objectives=(
                    Weighted(ObservedOutcomeMSE(), weight=1.0, reduction="population"),
                ),
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                ),
                rows="all",
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
                sampler=UniformSampler(batch_size=BATCH_SIZE),
            ),
        ),
        card="docs/recipes/tarnet.md",
        purpose="causal",
        data=DATA_POLICY,
    )


def tarnet_extension(schema: Schema) -> Recipe:
    """Build the xty2 propensity/marginal-likelihood TARNet extension."""
    return Recipe(
        name="tarnet_extension",
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
                sampler=UniformSampler(batch_size=BATCH_SIZE),
            ),
        ),
        card="docs/recipes/tarnet_extension.md",
        purpose="causal",
        data=DATA_POLICY,
    )


__all__ = [
    "BATCH_SIZE",
    "DATA_POLICY",
    "ENCODER_WIDTHS",
    "OUTCOME_WIDTHS",
    "tarnet",
    "tarnet_extension",
]
