"""The reviewed variational-treatment assembly — declarations only."""

from __future__ import annotations

from xty2.components import (
    CategoricalPosterior,
    CategoricalPropensity,
    MLPEncoder,
    TARNetHead,
)
from xty2.components._nn import CFRNET_INITIALISATION, TORCH_LINEAR_INITIALISATION
from xty2.core import (
    ComponentGraph,
    Constant,
    DataSpec,
    GradientClipping,
    MissingnessSpec,
    OptimiserSpec,
    Port,
    PreprocessSpec,
    Quota,
    QuotaSampler,
    Recipe,
    Schema,
    SplitSpec,
    Stage,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    VariationalTreatmentELBO,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS

POSTERIOR_WIDTHS = (300,)
OBSERVED_TREATMENTS = 64
OBSERVED_PER_BATCH = 8
MISSING_PER_BATCH = 120
VARIATIONAL_TREATMENT_STEPS = 3_000
POSTERIOR_SUPERVISION_WEIGHT = 0.1
TREATMENT_ENCODING = "categorical integers 0..K-1 with t_observed mask; no sentinel"

VARIATIONAL_TREATMENT_SAMPLER = QuotaSampler(
    quotas=(
        Quota(rows="t_observed", size=OBSERVED_PER_BATCH),
        Quota(rows="t_missing", size=MISSING_PER_BATCH),
    )
)

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6 "
            "fixture; no MNIST/SVHN/NORB protocol applies (deviation 3)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)

WEIGHT_DECAY = WeightDecay(
    value=1.0 / 1024.0,
    on_norm_and_bias=True,
    components=None,
)

OPTIMISER = OptimiserSpec(
    name="adam",
    lr=3e-4,
    weight_decay=WEIGHT_DECAY,
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
    betas=(0.9, 0.999),
    eps=1e-8,
)


def variational_treatment(schema: Schema) -> Recipe:
    """Build the reviewed recipe in ``docs/recipes/variational_treatment.md``."""
    return Recipe(
        name="variational_treatment",
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
                CategoricalPosterior(
                    input_dim=schema.num_features,
                    num_treatments=schema.treatment_cardinality,
                    outcome=schema.outcome,
                    widths=POSTERIOR_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                    output_parameterisation="K softmax logits",
                    standardisation=DATA_POLICY.standardisation,
                    outcome_scaling=DATA_POLICY.outcome_scaling,
                    treatment_encoding=TREATMENT_ENCODING,
                ),
            ]
        ),
        program=(
            Stage(
                name="elbo_fit",
                objectives=(
                    Weighted(
                        ObservedOutcomeNLL(),
                        weight=1.0,
                        reduction="population",
                    ),
                    Weighted(
                        ObservedTreatmentNLL(),
                        weight=1.0,
                        reduction="population",
                    ),
                    Weighted(
                        VariationalTreatmentELBO(),
                        weight=1.0,
                        reduction="population",
                    ),
                    Weighted(
                        ObservedTreatmentNLL(
                            name="posterior_treatment_nll",
                            port=Port.T_GIVEN_XY,
                        ),
                        weight=POSTERIOR_SUPERVISION_WEIGHT,
                        reduction="mean",
                    ),
                ),
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                    "categorical_propensity",
                    "categorical_posterior",
                ),
                rows="all",
                optimiser=OPTIMISER,
                steps=VARIATIONAL_TREATMENT_STEPS,
                sampler=VARIATIONAL_TREATMENT_SAMPLER,
            ),
        ),
        card="docs/recipes/variational_treatment.md",
        purpose="causal",
        data=DATA_POLICY,
    )


__all__ = [
    "DATA_POLICY",
    "MISSING_PER_BATCH",
    "OBSERVED_PER_BATCH",
    "OBSERVED_TREATMENTS",
    "OPTIMISER",
    "POSTERIOR_SUPERVISION_WEIGHT",
    "POSTERIOR_WIDTHS",
    "TREATMENT_ENCODING",
    "VARIATIONAL_TREATMENT_SAMPLER",
    "VARIATIONAL_TREATMENT_STEPS",
    "WEIGHT_DECAY",
    "variational_treatment",
]
