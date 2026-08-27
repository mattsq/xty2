"""The DoubleMatch assembly of `docs/recipes/doublematch.md`, declarations only."""

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
    Port,
    PreprocessSpec,
    Ramp,
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
    CosineFeatureConsistency,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    PseudoLabelTreatmentNLL,
)
from xty2.recipes.fixmatch import (
    FIXMATCH_SAMPLER,
    FIXMATCH_STEPS,
    OBSERVED_TREATMENTS,
    PRESERVED_FIELDS,
    STRONG_MASK_RATE,
    STRONG_X,
    WEAK_MASK_RATE,
    WEAK_X,
    WEAK_X_LABELLED,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

SELF_SUPERVISED_WEIGHT = 0.5
"""`w_s`, eq. (4). Card §7: the paper's value at the label count nearest ours.

§IV-D gives eleven values across four datasets and one rule for reading them —
a well-tuned `w_s` "will be largely correlated with the number of labeled
training data" — so there is no value to transcribe for a tabular fixture. The
reference's own README pairs CIFAR-10's 40/250/4,000 labels with 0.5/1/5, and
this fixture's 64 observed treatments sit at the bottom of that range.
"""

PROJECTION_WIDTHS = (ENCODER_WIDTHS[-1],)
"""`h`: "a single dimension-preserving linear layer" (§III, Fig. 2).

One width, equal to the encoder's output, is the whole declaration: a head of
`n` layers is affine in its last one, so at `n = 1` there is no activation and
no hidden width to name. `normalisation="none"` below is the other half —
eq. (3) normalises both sides itself, so a head that emitted unit vectors would
be normalising twice and hiding the scale the bias sees.
"""

DOUBLEMATCH_STEPS = FIXMATCH_STEPS
"""Card §4, and the `K` of eq. (5)'s cosine schedule.

The paper runs 352,000 steps (22,000 kimg at `B = 64`); deviation 3 fixes the
shared project-local budget instead, which is what makes a difference between
this recipe and `fixmatch` attributable to eq. (3) rather than to the budget.
"""

COSINE_PHASE = 7 / 16
"""`gamma / 2`, eq. (5): `eta_0 cos(gamma pi k / 2K)` at the paper's `gamma = 7/8`.

`CosineDecay(phase=p)` is `cos(pi p k / K)`, so `p = gamma / 2`. At 7/8 this is
FixMatch's own fixed schedule, which is why §6's pair differs in eq. (3) alone
(card §7 records that `gamma` is tuned per dataset in §III-B and that three of
the paper's four datasets take this value).
"""

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6 "
            "fixture; no CIFAR/SVHN/STL protocol applies (deviation 1)"
        ),
        train="train",
    ),
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    # Card §6 reuses `fixmatch`'s fixture unchanged, budget included: the
    # `w_s = 0` arm of the paired comparison is that recipe, and a different
    # label budget would make it a different one.
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)
"""The four `data.*` card keys."""


def doublematch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single-stage recipe from `docs/recipes/doublematch.md`."""
    return Recipe(
        name="doublematch",
        schema=schema,
        system=ComponentGraph(
            [
                # `f`. The one departure from the shared P5 backbone, and it
                # is the *initialisation*, not the geometry: eq. (3)'s gradient
                # carries `1/||.||`, and CFRNet's `0.1/sqrt(fan_in)` leaves this
                # encoder's pre-normalisation activations at a norm of 0.011,
                # about a hundredth of what a batch-normalised backbone hands
                # the term in the paper. `row_l2` then passes that factor
                # upstream and the representation collapses. Measured across
                # three encoder configurations in card §6.2; deviation 9 is
                # the row, and an earlier version of this recipe changed the
                # normalisation instead, which does not fix the scale and only
                # lets the run climb back out of the basin it still enters.
                MLPEncoder(
                    input_dim=schema.num_features,
                    widths=ENCODER_WIDTHS,
                    activation="elu",
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
                # `h`. The initialisation is the one place this head departs
                # from its three neighbours, and card §7 is where the reason
                # is: the reference initialises it Glorot-normal, which for a
                # `d -> d` layer is `std = 1/sqrt(d)`. Torch's default linear
                # initialisation is `0.577/sqrt(d)` and CFRNet's is
                # `0.1/sqrt(d)`, so the project convention would be the
                # further of the two by an order of magnitude.
                ProjectionHead(
                    representation_dim=ENCODER_WIDTHS[-1],
                    widths=PROJECTION_WIDTHS,
                    activation="relu",
                    normalisation="none",
                    dropout=0.0,
                    initialisation=TORCH_LINEAR_INITIALISATION,
                ),
            ]
        ),
        program=(
            Stage(
                name="joint_fit",
                objectives=(
                    Weighted(ObservedOutcomeNLL(), weight=1.0, reduction="population"),
                    # Eq. (1), on its own draw of the weak view: FixMatch's
                    # footnote 2 means a labelled row is weakly augmented
                    # twice, and DoubleMatch inherits the framework whole.
                    Weighted(
                        ObservedTreatmentNLL(realisation=WEAK_X_LABELLED),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (2), unchanged from `fixmatch`: the gate, the hard
                    # label and the detached weak-view target.
                    Weighted(
                        PseudoLabelTreatmentNLL(
                            port=Port.T_GIVEN_X,
                            target=WEAK_X,
                            prediction=STRONG_X,
                            threshold=0.95,
                            sharpening="hard",
                            stop_grad="target",
                            rows="all",
                        ),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (3) — the method. `h(v_i)` against `z_i`, on every
                    # row rather than the ones the gate above retained, and
                    # reading the *same* weak pass that produced the label
                    # (Alg. 1 lines 8 and 14), which is what makes the paper's
                    # "minimal computational overhead" true here too.
                    Weighted(
                        CosineFeatureConsistency(
                            prediction_port=Port.X_PROJ,
                            target_port=Port.X_REPR,
                            prediction=STRONG_X,
                            target=WEAK_X,
                            stop_grad="target",
                            rows="all",
                        ),
                        weight=SELF_SUPERVISED_WEIGHT,
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
                    "projection_head",
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
                    name="sgd",
                    lr=0.03,
                    momentum=0.9,
                    nesterov=True,
                    # Eq. (6) sums over `f`, `g` **and** `h`, and the reference
                    # decays the `kernel` variables of the scope all three live
                    # in — so every component, biases exempt.
                    weight_decay=WeightDecay(
                        value=5e-4,
                        on_norm_and_bias=False,
                        components=None,
                    ),
                    lr_schedule=CosineDecay(
                        steps=DOUBLEMATCH_STEPS, phase=COSINE_PHASE
                    ),
                    clipping=GradientClipping.none(),
                ),
                steps=DOUBLEMATCH_STEPS,
                sampler=FIXMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/doublematch.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            # §III-A: "We follow one of the augmentation schemes used in
            # FixMatch". The two rates and the preserved-field set are imported
            # so that those have one home; the two `ViewSpec` constructions are
            # restated, which is a second place to edit one reviewed decision
            # (card §4). Tier 0's plan comparison against `fixmatch` is what
            # catches a divergence between them.
            ViewSpec(
                name="weak_x",
                transforms=(FeatureMask(p=WEAK_MASK_RATE, columns=None, value=0.0),),
                preserves=PRESERVED_FIELDS,
                recompute_rules=recompute_rules,
                draws=2,
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
    "COSINE_PHASE",
    "DATA_POLICY",
    "DOUBLEMATCH_STEPS",
    "PROJECTION_WIDTHS",
    "SELF_SUPERVISED_WEIGHT",
    "doublematch",
]
