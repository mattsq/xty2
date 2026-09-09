"""The VICReg assembly of `docs/recipes/vicreg.md`, declarations only."""

from __future__ import annotations

from xty2.components import (
    CategoricalPropensity,
    MLPEncoder,
    TARNetHead,
    VICRegExpander,
)
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.components.vicreg import (
    VICREG_EXPANDER_ACTIVATION,
    VICREG_EXPANDER_INITIALISATION,
    VICREG_EXPANDER_NORMALISATION,
)
from xty2.core import (
    ComponentGraph,
    Constant,
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
    UniformSampler,
    ViewSpec,
    ViewTransform,
    WeightDecay,
    Weighted,
)
from xty2.objectives import (
    EmbeddingCovariance,
    EmbeddingInvariance,
    EmbeddingVariance,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)
from xty2.recipes.tarnet import OUTCOME_WIDTHS

CORRUPTED_A = Realisation(view="corrupted_a")
"""`t(x)`: one of the two transformations of eq. (7)'s inner sum."""

CORRUPTED_B = Realisation(view="corrupted_b")
"""`t'(x)`: the other. VICReg has **no** uncorrupted anchor — unlike SCARF,
where the anchor is the clean row — so `DEFAULT` appears in no pretraining
pass."""

VICREG_CORRUPTION_RATE = 0.6
"""Historical marginal-corruption rate, withdrawn in card deviation 3.

Not inherited from `scarf`: the paper's transformations are image crops, blurs
and solarisation, so there is no VICReg rate to reproduce. 0.6 is this card's
declared starting choice for the tabular substitute, re-derived against this
fixture rather than adopted because a sibling uses it, and the card records
that it is neither a paper constant nor guaranteed to preserve treatment
signal.
"""

VICREG_ENCODER_WIDTHS = (256, 256, 256, 256)
"""`f_theta`, card §4 and deviation 2: the tabular encoder that replaces
ResNet-50, four ReLU layers of width 256."""

EXPANDER_WIDTHS = (512, 512, 512)
"""`h_phi`, card §4 and deviation 2: the author's three-layer expander at a
tabular width. Still *expanding* — 512 > 256 — which is the property §4.2 of
the paper attributes the gain to, at 1/16 of the published 8192."""

INVARIANCE_WEIGHT = 25.0
"""`lambda`, card §3.1. The author code's coefficient on the **elementwise**
MSE; the printed equation's equivalent is `25/d` (deviation 1)."""

VARIANCE_WEIGHT = 25.0
"""`mu`, card §3.1. Charged on the *mean* of the two branch penalties, so the
printed equation's equivalent is 12.5 (deviation 1)."""

COVARIANCE_WEIGHT = 1.0
"""`nu`, card §3.1. Charged on the sum of the two branch penalties, which is
what both the code and eq. (6) do."""

VARIANCE_TARGET = 1.0
"""`gamma`: "we fix gamma to 1 in our experiments" (paper §4.1)."""

VARIANCE_EPSILON = 1e-4
"""`epsilon` of eq. (2), from the author code's `Var(x) + 1e-4` inside the
square root."""

SAMPLE_CORRECTION = 1
"""The `Var` and covariance denominators' correction: `n - 1`, the author
convention for both (card §7)."""

VICREG_PRETRAIN_STEPS = 1_000
"""Card §4 and deviation 4: a fixed local budget where §4.2 trains for epochs
on ImageNet."""

VICREG_JOINT_FIT_STEPS = 3_000
"""Card §4: the project-local fitting budget every other recipe uses."""

VICREG_OBSERVED_TREATMENTS = 40
"""Card §6: 40 labelled treatments of 1,024, the label-scarce regime."""

VICREG_BATCH_SIZE = 128
"""`n`, card §4 — and arithmetic rather than plumbing, for a second reason
SCARF's card did not have.

`v` and `c` are statistics over the batch axis: the variance hinge is computed
from `n` rows and the covariance denominator *is* `n - 1`, so the same
declaration at a different batch size charges a different penalty. Both terms
declare `batch_coupled = True`, so the compiler refuses to let this stage hand
batch construction back to the caller (deviation 4 fixes the value at 128
against §4.2's 2,048).
"""

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "fixed two-cluster DGP; disjoint train and held-out populations; "
            "no test-based selection"
        ),
        train="train",
    ),
    # Deviation 5: a project policy rather than a paper constant. VICReg
    # standardises images by channel statistics and says nothing about tabular
    # columns; declaring both here is what makes the plan name the split each
    # is fitted on, which is the leakage the card's §6.2 protocol turns on.
    preprocess=PreprocessSpec(features="zscore", outcome="zscore"),
    # Card §6: pretraining reads no treatment at all, so this governs the
    # fine-tuning stage's populations only.
    missingness=MissingnessSpec(mechanism="mcar", observed=VICREG_OBSERVED_TREATMENTS),
)
"""The four `data.*` card keys."""

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)

VICREG_ADAM = OptimiserSpec(
    name="adam",
    lr=1e-3,
    weight_decay=WeightDecay.none(),
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
    betas=(0.9, 0.999),
    eps=1e-8,
)
"""Card §4, deviation 4.

Deliberately **not** the paper's optimiser and deliberately not imported from
`scarf`: §4.2 trains with LARS, weight decay 1e-6 and a cosine schedule with
warmup at batch 2,048, none of which this fixture's 1,024 rows can exercise.
The card states a matched local protocol so that the four §6 arms differ by one
VICReg coefficient and nothing else, and re-derives it here rather than
inheriting a sibling's constant. One frozen value object for both stages; the
executor builds a fresh optimiser per stage, so no moment crosses the
transition.
"""


def vicreg(
    schema: Schema,
    *,
    first_transforms: tuple[ViewTransform, ...],
    second_transforms: tuple[ViewTransform, ...],
    recompute_rules: tuple[RecomputeRule, ...] = (),
) -> Recipe:
    """Build the two-stage recipe from `docs/recipes/vicreg.md`."""
    return Recipe(
        name="vicreg",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=VICREG_ENCODER_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=CFRNET_INITIALISATION,
                ),
                VICRegExpander(
                    representation_dim=VICREG_ENCODER_WIDTHS[-1],
                    widths=EXPANDER_WIDTHS,
                    activation=VICREG_EXPANDER_ACTIVATION,
                    normalisation=VICREG_EXPANDER_NORMALISATION,
                    dropout=0.0,
                    initialisation=VICREG_EXPANDER_INITIALISATION,
                ),
                TARNetHead(
                    representation_dim=VICREG_ENCODER_WIDTHS[-1],
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
                    representation_dim=VICREG_ENCODER_WIDTHS[-1],
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
                    # `s(Z,Z')/d`, `(v+v')/2` and `c+c'` as three declarations
                    # rather than one: eq. (6)'s coefficients are what §6's
                    # study sets to zero, one at a time. Each term is its own
                    # batch mean, so `mean` is the reduction that adds no
                    # second division (card §3.2).
                    Weighted(
                        EmbeddingInvariance(
                            port=Port.X_PROJ,
                            first=CORRUPTED_A,
                            second=CORRUPTED_B,
                            rows="all",
                        ),
                        weight=INVARIANCE_WEIGHT,
                        reduction="mean",
                    ),
                    Weighted(
                        EmbeddingVariance(
                            port=Port.X_PROJ,
                            first=CORRUPTED_A,
                            second=CORRUPTED_B,
                            gamma=VARIANCE_TARGET,
                            epsilon=VARIANCE_EPSILON,
                            correction=SAMPLE_CORRECTION,
                            rows="all",
                        ),
                        weight=VARIANCE_WEIGHT,
                        reduction="mean",
                    ),
                    Weighted(
                        EmbeddingCovariance(
                            port=Port.X_PROJ,
                            first=CORRUPTED_A,
                            second=CORRUPTED_B,
                            correction=SAMPLE_CORRECTION,
                            rows="all",
                        ),
                        weight=COVARIANCE_WEIGHT,
                        reduction="mean",
                    ),
                ),
                # `f_theta` and `h_phi`, and nothing else: the outcome and
                # propensity heads read no port this stage computes.
                trainable=("mlp_encoder", "vicreg_expander"),
                rows="all",
                optimiser=VICREG_ADAM,
                steps=VICREG_PRETRAIN_STEPS,
                sampler=UniformSampler(batch_size=VICREG_BATCH_SIZE),
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
                # "the expander is discarded" (paper §3): `vicreg_expander` is
                # in no forward pass and in no trainable list of this stage,
                # and `mlp_encoder` is in both because the encoder is what
                # pretraining was for.
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                    "categorical_propensity",
                ),
                rows="all",
                initialise_from="pretrain",
                optimiser=VICREG_ADAM,
                steps=VICREG_JOINT_FIT_STEPS,
                sampler=UniformSampler(batch_size=VICREG_BATCH_SIZE),
            ),
        ),
        card="docs/recipes/vicreg.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            # Two independent draws of one transform, not two draws of one
            # view: `ViewSpec.apply` keys its generator by the view's *name*
            # (`core/views.py`), so two names is what makes `t` and `t'`
            # independent, and the three objectives then read one cached pair.
            ViewSpec(
                name="corrupted_a",
                transforms=first_transforms,
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
            ViewSpec(
                name="corrupted_b",
                transforms=second_transforms,
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
            ),
        ),
    )


__all__ = [
    "CORRUPTED_A",
    "CORRUPTED_B",
    "COVARIANCE_WEIGHT",
    "DATA_POLICY",
    "EXPANDER_WIDTHS",
    "INVARIANCE_WEIGHT",
    "SAMPLE_CORRECTION",
    "VARIANCE_EPSILON",
    "VARIANCE_TARGET",
    "VARIANCE_WEIGHT",
    "VICREG_ADAM",
    "VICREG_BATCH_SIZE",
    "VICREG_CORRUPTION_RATE",
    "VICREG_ENCODER_WIDTHS",
    "VICREG_JOINT_FIT_STEPS",
    "VICREG_OBSERVED_TREATMENTS",
    "VICREG_PRETRAIN_STEPS",
    "vicreg",
]
