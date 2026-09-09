"""The Barlow Twins assembly of `docs/recipes/barlow_twins.md`, declarations only."""

from __future__ import annotations

from xty2.components import (
    BarlowTwinsProjector,
    CategoricalPropensity,
    MLPEncoder,
    TARNetHead,
)
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.components.barlow_twins import (
    BARLOW_TWINS_PROJECTOR_ACTIVATION,
    BARLOW_TWINS_PROJECTOR_INITIALISATION,
    BARLOW_TWINS_PROJECTOR_NORMALISATION,
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
    CrossCorrelationDiagonal,
    CrossCorrelationOffDiagonal,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
)
from xty2.recipes.tarnet import OUTCOME_WIDTHS

CORRUPTED_A = Realisation(view="corrupted_a")
"""`Y^A`: one of the two distorted versions of the batch (paper §2.1)."""

CORRUPTED_B = Realisation(view="corrupted_b")
"""`Y^B`: the other. Barlow Twins has **no** uncorrupted anchor — unlike SCARF,
where the anchor is the clean row — so `DEFAULT` appears in no pretraining
pass."""

BARLOW_TWINS_ENCODER_WIDTHS = (256, 256, 256, 256)
"""`f_theta`'s encoder, card §4 and deviation 2: the tabular encoder that
replaces ResNet-50, four ReLU layers of width 256. Re-derived against this
fixture rather than inherited: the card's decisive comparison is against this
recipe's own zero-lambda arm, and matching `vicreg`'s capacity is what makes
the contextual VICReg arm of §6.2 a comparison of objectives rather than of
network sizes."""

PROJECTOR_WIDTHS = (512, 512, 512)
"""`f_theta`'s projector, card §4 and deviation 2: the author's three-layer
head at a tabular width. The published one is 8192-8192-8192, and §4.3 of the
paper attributes real accuracy to that width; 512 is what this fixture's 1,024
training rows and `B = 128` can support, and §6.4 states the consequence — at
`B = 128` a centred cross-product has rank at most 127, so `C = I` is
unreachable and zero loss is not the target."""

DIAGONAL_WEIGHT = 1.0
"""The implicit coefficient on `sum_i (1 - C_ii)^2` in eq. (1)."""

OFF_DIAGONAL_WEIGHT = 0.0051
"""`lambda`, card §3.1 and deviation 1. The **parser's** value, not §2.2's
printed 0.005: `main.py` writes `--lambd, default=0.0051`, and that is the
number the published runs executed. Retained unchanged at `d = 512` against the
published `d = 8192`; the card says explicitly that this is not a claim it is
optimal there (deviation 4)."""

NORMALISATION_EPSILON = 1e-5
"""The `eps` of the author's output `nn.BatchNorm1d`, **inside** each branch's
square root. A torch default in the source, and therefore a value this card
states rather than inherits (§7)."""

POPULATION_CORRECTION = 0
"""The variance denominator's correction. Training-mode `BatchNorm1d`
normalises by the biased estimate, so `0` — not torch's `Tensor.var` default of
`1`, and not the unspecified `std` of Algorithm 1 (card §3.1, deviation 1)."""

BARLOW_TWINS_PRETRAIN_STEPS = 1_000
"""Card §4 and deviation 4: a fixed local budget where the paper trains 1000
epochs on ImageNet with LARS."""

BARLOW_TWINS_JOINT_FIT_STEPS = 3_000
"""Card §4: the project-local fitting budget every other recipe uses."""

BARLOW_TWINS_OBSERVED_TREATMENTS = 40
"""Card §6: 40 labelled treatments of 1,024, the label-scarce regime."""

BARLOW_TWINS_BATCH_SIZE = 128
"""`B`, card §4 — arithmetic rather than plumbing.

`C` divides by `B` and is a second moment over the batch axis, so the same
declaration at a different batch size charges a different penalty and admits a
different attainable rank. Both objectives declare `batch_coupled = True`, so
the compiler refuses to let this stage hand batch construction back to the
caller (deviation 4 fixes the value at 128 against the published 2,048)."""

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "fixed two-cluster DGP; disjoint train and held-out populations; "
            "no test-based selection"
        ),
        train="train",
    ),
    # Deviation 5: a project policy rather than a paper constant. The paper
    # standardises images by channel statistics and says nothing about tabular
    # columns; declaring both here is what makes the plan name the split each
    # is fitted on, which is the leakage the card's §6.2 protocol turns on.
    preprocess=PreprocessSpec(features="zscore", outcome="zscore"),
    # Card §6: pretraining reads no treatment at all, so this governs the
    # fine-tuning stage's populations only.
    missingness=MissingnessSpec(
        mechanism="mcar", observed=BARLOW_TWINS_OBSERVED_TREATMENTS
    ),
)
"""The four `data.*` card keys."""

PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)

BARLOW_TWINS_ADAM = OptimiserSpec(
    name="adam",
    lr=1e-3,
    weight_decay=WeightDecay.none(),
    lr_schedule=Constant(1.0),
    clipping=GradientClipping.none(),
    betas=(0.9, 0.999),
    eps=1e-8,
)
"""Card §4, deviation 4.

Deliberately **not** the paper's optimiser: §2.3 trains with LARS, a
learning-rate warmup and cosine decay, weight decay 1.5e-6 and separate
parameter groups for biases and BatchNorm, at batch 2,048 across 32 V100s —
none of which this fixture's 1,024 rows can exercise. The card states a matched
local protocol so that the §6 arms differ by one Barlow Twins coefficient and
nothing else, and re-derives it against this fixture rather than inheriting
`vicreg`'s constant. One frozen value object for both stages; the executor
builds a fresh optimiser per stage, so no moment crosses the transition."""


def barlow_twins(
    schema: Schema,
    *,
    first_transforms: tuple[ViewTransform, ...],
    second_transforms: tuple[ViewTransform, ...],
    recompute_rules: tuple[RecomputeRule, ...] = (),
) -> Recipe:
    """Build the two-stage recipe from `docs/recipes/barlow_twins.md`."""
    return Recipe(
        name="barlow_twins",
        schema=schema,
        system=ComponentGraph(
            [
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=BARLOW_TWINS_ENCODER_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=CFRNET_INITIALISATION,
                ),
                BarlowTwinsProjector(
                    representation_dim=BARLOW_TWINS_ENCODER_WIDTHS[-1],
                    widths=PROJECTOR_WIDTHS,
                    activation=BARLOW_TWINS_PROJECTOR_ACTIVATION,
                    normalisation=BARLOW_TWINS_PROJECTOR_NORMALISATION,
                    dropout=0.0,
                    initialisation=BARLOW_TWINS_PROJECTOR_INITIALISATION,
                ),
                TARNetHead(
                    representation_dim=BARLOW_TWINS_ENCODER_WIDTHS[-1],
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
                    representation_dim=BARLOW_TWINS_ENCODER_WIDTHS[-1],
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
                    # `D` and `O` as two declarations rather than one: eq. (1)'s
                    # `lambda` is what §6's study sets to zero, and the
                    # zero-weight arm keeps the objective declared so the two
                    # arms differ by a number and not by the shape of the loss.
                    # Each term is one batch statistic, so `mean` is the
                    # reduction that adds no second division (card §3.2).
                    Weighted(
                        CrossCorrelationDiagonal(
                            port=Port.X_PROJ,
                            first=CORRUPTED_A,
                            second=CORRUPTED_B,
                            epsilon=NORMALISATION_EPSILON,
                            correction=POPULATION_CORRECTION,
                            rows="all",
                        ),
                        weight=DIAGONAL_WEIGHT,
                        reduction="mean",
                    ),
                    Weighted(
                        CrossCorrelationOffDiagonal(
                            port=Port.X_PROJ,
                            first=CORRUPTED_A,
                            second=CORRUPTED_B,
                            epsilon=NORMALISATION_EPSILON,
                            correction=POPULATION_CORRECTION,
                            rows="all",
                        ),
                        weight=OFF_DIAGONAL_WEIGHT,
                        reduction="mean",
                    ),
                ),
                # `f_theta`, encoder and projector, and nothing else: the
                # outcome and propensity heads read no port this stage computes.
                trainable=("mlp_encoder", "barlow_twins_projector"),
                rows="all",
                optimiser=BARLOW_TWINS_ADAM,
                steps=BARLOW_TWINS_PRETRAIN_STEPS,
                sampler=UniformSampler(batch_size=BARLOW_TWINS_BATCH_SIZE),
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
                # The projector is not transferred: it is in no forward pass and
                # in no trainable list of this stage, and `mlp_encoder` is in
                # both because the encoder is what pretraining was for.
                trainable=(
                    "mlp_encoder",
                    "tarnet_head",
                    "categorical_propensity",
                ),
                rows="all",
                initialise_from="pretrain",
                optimiser=BARLOW_TWINS_ADAM,
                steps=BARLOW_TWINS_JOINT_FIT_STEPS,
                sampler=UniformSampler(batch_size=BARLOW_TWINS_BATCH_SIZE),
            ),
        ),
        card="docs/recipes/barlow_twins.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            # Two independent draws of one transform, not two draws of one
            # view: `ViewSpec.apply` keys its generator by the view's *name*
            # (`core/views.py`), so two names is what makes `Y^A` and `Y^B`
            # independent, and both objectives then read one cached pair.
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
    "BARLOW_TWINS_ADAM",
    "BARLOW_TWINS_BATCH_SIZE",
    "BARLOW_TWINS_ENCODER_WIDTHS",
    "BARLOW_TWINS_JOINT_FIT_STEPS",
    "BARLOW_TWINS_OBSERVED_TREATMENTS",
    "BARLOW_TWINS_PRETRAIN_STEPS",
    "CORRUPTED_A",
    "CORRUPTED_B",
    "DATA_POLICY",
    "DIAGONAL_WEIGHT",
    "NORMALISATION_EPSILON",
    "OFF_DIAGONAL_WEIGHT",
    "POPULATION_CORRECTION",
    "PROJECTOR_WIDTHS",
    "barlow_twins",
]
