"""The VIME-self assembly of `docs/recipes/vime.md`, declarations only."""

from __future__ import annotations

from xty2.components import (
    CategoricalPropensity,
    FeatureEstimatorHead,
    MaskEstimatorHead,
    MLPEncoder,
    TARNetHead,
)
from xty2.components._nn import CFRNET_INITIALISATION, GLOROT_UNIFORM_INITIALISATION
from xty2.core import (
    DEFAULT,
    ComponentGraph,
    Constant,
    DataSpec,
    GradientClipping,
    MissingnessSpec,
    OptimiserSpec,
    PreprocessSpec,
    PreservedField,
    Ramp,
    Realisation,
    Recipe,
    RecomputeRule,
    Schema,
    SplitSpec,
    Stage,
    UniformSampler,
    ViewSpec,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    FeatureReconstruction,
    MaskEstimationBCE,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)
from xty2.recipes.scarf import ADAM as JOINT_FIT_ADAM
from xty2.recipes.scarf import BATCH_SIZE as JOINT_FIT_BATCH_SIZE
from xty2.recipes.scarf import JOINT_FIT_STEPS as VIME_JOINT_FIT_STEPS
from xty2.recipes.scarf import OBSERVED_TREATMENTS as VIME_OBSERVED_TREATMENTS
from xty2.recipes.tarnet import OUTCOME_WIDTHS
from xty2.views import BernoulliMarginalCorruption

VIME_CORRUPTED = Realisation(view="vime_corrupted")
"""`x~ = g_m(x, m)`, Eq. 3: the only input the encoder sees in `pretrain`."""

CLEAN_X = DEFAULT
"""`x`: the clean row, a target of both losses and never an encoder input."""

MASK_PROBABILITY = 0.3
"""`p_m`, card §7: `main_vime.py` argparse default `--p_m 0.3`."""

ALPHA = 2.0
"""`alpha`, Eq. 4 and card §4: `main_vime.py` argparse default `--alpha 2.0`."""

VIME_PRETRAIN_BATCH_SIZE = 128
"""Card §4: `main_vime.py` `vime_self_parameters['batch_size'] = 128`."""

VIME_PRETRAIN_STEPS = 80
"""Card §4 and deviation 3: 10 epochs (`main_vime.py`) x 1,024 rows / 128."""

RMSPROP = OptimiserSpec(
    name="rmsprop",
    lr=1e-3,
    weight_decay=WeightDecay.none(),
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
    rho=0.9,
    eps=1e-7,
    momentum=0.0,
    centered=False,
)
"""Card §4: `vime_self.py` compiles with `optimizer='rmsprop'`.

Every constant is the Keras 2.3.1 default, bound explicitly because torch's
defaults differ (`alpha=0.99`, `eps=1e-8`; card §5.1).
"""

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol="scarf.md section 6.1 fixture unchanged",
        train="train",
    ),
    # Section 5: "We use Min-max scaler to normalize the data between 0 and 1",
    # which is also what makes `s_r`'s sigmoid able to reach its targets. The
    # outcome scaling is project-local and SCARF's (card §4).
    preprocess=PreprocessSpec(features="minmax", outcome="zscore"),
    missingness=MissingnessSpec(mechanism="mcar", observed=VIME_OBSERVED_TREATMENTS),
)
"""The four `data.*` card keys."""

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def vime(schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()) -> Recipe:
    """Build the two-stage recipe from `docs/recipes/vime.md`."""
    width = schema.num_features
    return Recipe(
        name="vime",
        schema=schema,
        system=ComponentGraph(
            [
                # `e`: `Dense(int(dim), activation='relu')` in `vime_self.py`.
                MLPEncoder(
                    input_dim=width,
                    widths=(width,),
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=GLOROT_UNIFORM_INITIALISATION,
                ),
                # `s_m`: `Dense(dim, activation='sigmoid', name='mask')`, with
                # the sigmoid folded into the BCE (card §7).
                MaskEstimatorHead(
                    representation_dim=width,
                    num_features=width,
                    activation="linear logits",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=GLOROT_UNIFORM_INITIALISATION,
                    output_parameterisation=f"{width} Bernoulli logits",
                ),
                # `s_r`: `Dense(dim, activation='sigmoid', name='feature')`.
                FeatureEstimatorHead(
                    representation_dim=width,
                    num_features=width,
                    activation="sigmoid",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=GLOROT_UNIFORM_INITIALISATION,
                    output_parameterisation=f"{width} values in (0, 1)",
                ),
                TARNetHead(
                    representation_dim=width,
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
                    representation_dim=width,
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
                    # Eq. 5, `l_m`: `loss_weights={'mask': 1, ...}`.
                    Weighted(
                        MaskEstimationBCE(
                            clean=CLEAN_X, corrupted=VIME_CORRUPTED, rows="all"
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. 6, `l_r`, weighted by `alpha` (Eq. 4).
                    Weighted(
                        FeatureReconstruction(
                            clean=CLEAN_X,
                            corrupted=VIME_CORRUPTED,
                            cells="all",
                            rows="all",
                        ),
                        weight=ALPHA,
                        reduction="mean",
                    ),
                ),
                # Eq. 4 minimises over `e`, `s_m` and `s_r` jointly.
                trainable=("mlp_encoder", "mask_estimator", "feature_estimator"),
                rows="all",
                optimiser=RMSPROP,
                steps=VIME_PRETRAIN_STEPS,
                sampler=UniformSampler(batch_size=VIME_PRETRAIN_BATCH_SIZE),
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
                # "Only `e` survives", and `main_vime.py` fits downstream on
                # `vime_self_encoder.predict(x)`: both estimators are in no
                # forward pass here, and the encoder runs but is not trained.
                trainable=("tarnet_head", "categorical_propensity"),
                rows="all",
                initialise_from="pretrain",
                optimiser=JOINT_FIT_ADAM,
                steps=VIME_JOINT_FIT_STEPS,
                sampler=UniformSampler(batch_size=JOINT_FIT_BATCH_SIZE),
            ),
        ),
        card="docs/recipes/vime.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            ViewSpec(
                name="vime_corrupted",
                transforms=(
                    BernoulliMarginalCorruption(p=MASK_PROBABILITY, columns=None),
                ),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = [
    "ALPHA",
    "CLEAN_X",
    "DATA_POLICY",
    "JOINT_FIT_ADAM",
    "JOINT_FIT_BATCH_SIZE",
    "MASK_PROBABILITY",
    "RMSPROP",
    "VIME_CORRUPTED",
    "VIME_JOINT_FIT_STEPS",
    "VIME_OBSERVED_TREATMENTS",
    "VIME_PRETRAIN_BATCH_SIZE",
    "VIME_PRETRAIN_STEPS",
    "vime",
]
