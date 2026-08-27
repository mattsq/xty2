"""The FixMatch assembly of `docs/recipes/fixmatch.md`, declarations only."""

from __future__ import annotations

from xty2.components import CategoricalPropensity, MLPEncoder, TARNetHead
from xty2.components._nn import CFRNET_INITIALISATION
from xty2.core import (
    ComponentGraph,
    CosineDecay,
    DataSpec,
    GradientClipping,
    MissingnessSpec,
    OptimiserSpec,
    Port,
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
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    PseudoLabelTreatmentNLL,
)
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

WEAK_X = Realisation(view="weak_x")
"""`alpha(u_b)`: the draw of the weak view eq. (4)'s artificial label reads."""

WEAK_X_LABELLED = Realisation(view="weak_x", draw=1)
"""`alpha(x_b)`: a *second, independent* draw of the same weak view.

The reference draws a labelled batch and an unlabelled batch separately, and
footnote 2 puts every labelled row into the unlabelled one as well — so a
labelled row is weakly augmented twice, once for eq. (3) and once as its own
eq. (4) target, under independently sampled `alpha`. One batch and one view
here, two draws of it, which is the same thing.
"""

STRONG_X = Realisation(view="strong_x")
"""`A(u_b)`: the view the pseudo-label is charged against."""

WEAK_MASK_RATE = 0.1
"""Card §7: the weak view reuses Mean Teacher's reviewed masking strength."""

STRONG_MASK_RATE = 0.5
"""Card §7: the extra corruption the strong view adds *on top of* the weak one.

The reference implementation does not replace the weak transform on the strong
branch — it samples the ordinary augmentation independently a second time and
then layers CTAugment and Cutout onto that. `strong_x` says the same thing by
listing both transforms, so the recipe reads the way the method works. For a
constant-fill mask the two collapse to one of rate `1 - 0.9 * 0.5 = 0.55`
(card §7); the layering is legible rather than load-bearing.
"""

FIXMATCH_STEPS = 3_000
"""Card §4. Also the `K` of the cosine rate schedule (card §5, deviation 3)."""

LABELLED_BATCH = 64
"""`B`, section 2.4 and appendix B.3: labelled examples per step."""

OBSERVED_TREATMENTS = LABELLED_BATCH
"""Card section 6's label budget: the scarcest xty2 can run this recipe at.

Section 4's smallest CIFAR-10 regime is 40 labels *in total*, against `B = 64`
drawn per step — the reference iterates the labelled set as an endlessly
repeating shuffle, so a step sees some labelled rows twice. `XTYBatch.row_id`
must be unique (`DESIGN.md` section 7.1), so a repeated row cannot be put in a
batch and the smallest budget expressible here is `B` itself. Card section 5
carries that as a framework limitation against ledger key
`batch-row-repetition`; it is a deviation from the *fixture's* label regime,
not from `B` or `mu`, both of which are the paper's.
"""

MU = 7
"""`mu`, eq. (5): the unlabelled batch is `mu B` rows.

This is the number deviation 4 was written about. It is a batch *composition*,
so a per-term weight cannot stand in for it: `lambda_u` sets the two terms'
relative gradient, and `mu` sets how many rows eq. (4) averages over, which is
what makes its estimate of the unlabelled loss as precise as the paper's.
`QuotaSampler` derives both card keys from the quotas below, so the plan prints
the ratio the sampler runs rather than one the recipe asserts.
"""

FIXMATCH_SAMPLER = QuotaSampler(
    quotas=(
        Quota(rows="t_observed", size=LABELLED_BATCH),
        Quota(rows="t_missing", size=MU * LABELLED_BATCH),
    )
)
"""`optimisation.batch_size = 512` and `labelled_unlabelled_ratio = 7`, derived."""

DATA_POLICY = DataSpec(
    split=SplitSpec(
        protocol=(
            "one fixed project-local DGP, split train/test by the section 6 "
            "fixture; no CIFAR/SVHN protocol applies (deviation 1)"
        ),
        train="train",
    ),
    # Section 6 standardises the outcome on the training rows. Declared here
    # rather than done in the fixture, so the plan names the split it is fitted
    # on and the run checks that claim rather than carrying it.
    preprocess=PreprocessSpec(features="none", outcome="zscore"),
    # The scarce label here is the *treatment*, and the quota above is stated
    # over `t_observed` and `t_missing`, so the budget that produces those two
    # populations is part of the same declaration. A count rather than a rate,
    # because that is how section 4 states it: CIFAR-10 at 40, 250 and 4,000
    # labels.
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)
"""The four `data.*` card keys."""


PRESERVED_FIELDS: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)


def fixmatch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single-stage recipe from `docs/recipes/fixmatch.md`."""
    return Recipe(
        name="fixmatch",
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
                    # Eq. (3): the labelled cross-entropy is taken on its own
                    # draw of the weak view — footnote 2 means a labelled row
                    # is weakly augmented twice — and divides by the labelled
                    # batch, which is `mean` over the term's rows (card §3.2).
                    Weighted(
                        ObservedTreatmentNLL(realisation=WEAK_X_LABELLED),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (4), with lambda_u = 1 and no ramp: §2.2 states that
                    # the gate supplies the curriculum a ramp would otherwise
                    # provide.
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
                # Section 2.4 reports final performance from an EMA of the
                # parameters. Nothing reads it during training — eq. (4)'s
                # label comes from the current network — so it is declared for
                # what it is, and the compiler checks that no objective takes
                # it as a target.
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
                    weight_decay=WeightDecay(
                        value=5e-4,
                        # The reference implementation sums `l2_loss` over the
                        # variables whose name carries `kernel`, so biases and
                        # any norm parameters are exempt. `tf.nn.l2_loss`
                        # carries the 1/2, which makes this coupled L2 exactly
                        # torch's SGD `weight_decay` (card §7).
                        on_norm_and_bias=False,
                        components=None,
                    ),
                    lr_schedule=CosineDecay(steps=FIXMATCH_STEPS, phase=7 / 16),
                    clipping=GradientClipping.none(),
                ),
                steps=FIXMATCH_STEPS,
                sampler=FIXMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/fixmatch.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
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
    "DATA_POLICY",
    "FIXMATCH_SAMPLER",
    "FIXMATCH_STEPS",
    "LABELLED_BATCH",
    "MU",
    "OBSERVED_TREATMENTS",
    "STRONG_MASK_RATE",
    "STRONG_X",
    "WEAK_MASK_RATE",
    "WEAK_X",
    "WEAK_X_LABELLED",
    "fixmatch",
]
