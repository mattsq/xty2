"""The FreeMatch assembly of `docs/recipes/freematch.md`, declarations only."""

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
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
    SelfAdaptiveFairness,
    SelfAdaptiveThreshold,
    SelfAdaptiveThresholdTreatmentNLL,
)
from xty2.recipes.fixmatch import (
    FIXMATCH_SAMPLER,
    FIXMATCH_STEPS,
    OBSERVED_TREATMENTS,
    PRESERVED_FIELDS,
    STRONG_X,
    WEAK_MASK_RATE,
    WEAK_X,
    WEAK_X_LABELLED,
)
from xty2.recipes.flexmatch import STRONG_MASK_RATE
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

EMA_DECAY = 0.999
"""`lambda`, table 5: "Thresholding EMA decay for all experiments".

The momentum of all three statistics — eq. (5)'s global confidence, eq. (6)'s
per-class probability and eq. (10)'s label histogram. Numerically equal to the
*parameter* EMA of §5.1 and unrelated to it: this one is a statistic of the
predictions, that one a copy of the weights, and the card states both.
"""

SAT = SelfAdaptiveThreshold(decay=EMA_DECAY)
"""The whole gate rule, as card §4's `losses.confidence_threshold`.

FreeMatch's gate contains no threshold: `tau_t(c)` is a function of the training
history and `lambda` is the only number a recipe sets. Card §4 says why the key
holds a policy object rather than a float, and `flexmatch.md` §4 read the same
key the same way for a different rule.
"""

UNSUPERVISED_WEIGHT = 1.0
"""`w_u`, §5.1 and table 5: "we set `w_u = 1` for all experiments"."""

FAIRNESS_WEIGHT = 0.05
"""`w_f`, table 5's "Loss weight `w_f` for others".

The 0.01 of the other row is the barely-supervised end of each dataset —
CIFAR-10 at 10 labels, CIFAR-100 at 400, STL-10 at 40. At the 64 labels over
`K = 2` deviation 9 fixes, this fixture is not in that regime; card §7 records
the choice and §5.9 records that the fixture cannot reach the alternative.

**Negating this is eq. (11) exactly as printed** (deviation 7 and card §6.1's
`literal` arm), which is why the sign lives in the weight and not in a field on
the objective.
"""

FREEMATCH_STEPS = FIXMATCH_STEPS
"""Card §4, and the `K` of the cosine rate schedule.

The paper trains for `2^20` steps; deviation 3 fixes the shared project-local
budget instead, which is what makes a difference between this recipe and its
constant-gate arm attributable to the gate rather than to the budget.
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
    # Card §6.1 reuses `fixmatch`'s fixture unchanged, budget included: the
    # comparison arm is that recipe's gate, and a different label budget would
    # make it a different comparison.
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)
"""The four `data.*` card keys."""

SAT_TERM = "self_adaptive_threshold_treatment_nll"
"""The name eq. (8)'s term is logged under, and what eq. (11)'s term reads.

Named once rather than twice: `SelfAdaptiveFairness.statistics` has to be the
`name` of the `SelfAdaptiveThresholdTreatmentNLL` in the same stage, because
`tau_t`, `p~_t` and `h~_t` are one set of statistics that both terms read
(card §3.2). Two spellings of one string is the failure that would produce.
"""


def freematch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single-stage recipe from `docs/recipes/freematch.md`."""
    return Recipe(
        name="freematch",
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
                    # Eq. (3), on its own draw of the weak view: FixMatch's
                    # footnote 2 means a labelled row is weakly augmented
                    # twice, and FreeMatch inherits that framework whole.
                    Weighted(
                        ObservedTreatmentNLL(realisation=WEAK_X_LABELLED),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (8) — the method. The same two realisations, the same
                    # hard label and the same detached weak-view target as
                    # `fixmatch`'s eq. (4); the gate is `tau_t(arg max q_b)`
                    # instead of `tau`, and eqs. (5)-(7) are what moves it.
                    Weighted(
                        SelfAdaptiveThresholdTreatmentNLL(
                            port=Port.T_GIVEN_X,
                            target=WEAK_X,
                            prediction=STRONG_X,
                            num_treatments=schema.treatment_cardinality,
                            threshold=SAT,
                            sharpening="hard",
                            stop_grad="target",
                            rows="all",
                            name=SAT_TERM,
                        ),
                        weight=UNSUPERVISED_WEIGHT,
                        reduction="mean",
                    ),
                    # Eq. (11), reading the same `tau_t`, `p~_t` and `h~_t` the
                    # term above maintains. Its own weight because eq. (12)
                    # gives it one; `mean` because it is one scalar per batch
                    # and that is the mode under which a value enters the total
                    # unscaled (card §4).
                    Weighted(
                        SelfAdaptiveFairness(
                            port=Port.T_GIVEN_X,
                            target=WEAK_X,
                            prediction=STRONG_X,
                            statistics=SAT_TERM,
                            rows="all",
                        ),
                        weight=FAIRNESS_WEIGHT,
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
                # §5.1 evaluates from an EMA at 0.999. Nothing reads it during
                # training — eq. (8)'s label and eqs. (5), (6) and (10)'s
                # statistics all come from the current network — so it is
                # declared for what it is, and the compiler checks that no
                # objective takes it as a target.
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
                        on_norm_and_bias=False,
                        components=None,
                    ),
                    lr_schedule=CosineDecay(steps=FREEMATCH_STEPS, phase=7 / 16),
                    clipping=GradientClipping.none(),
                ),
                steps=FREEMATCH_STEPS,
                sampler=FIXMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/freematch.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            # §3 restates FixMatch's and UDA's framework unchanged and §5.1
            # adopts their settings, so the weak view is `fixmatch`'s exactly.
            # The strong one is `flexmatch`'s — deviation 2 — because
            # `tau_0(c) = 1/K` leaves eq. (8) ungated on the whole batch at
            # `K = 2`, which is the configuration `flexmatch.md` §5.2 measured
            # and rejected. Both rates are imported so the reviewed values have
            # one home; the two `ViewSpec` constructions are restated, and
            # Tier 0 compares them against the compiled plan.
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
    "EMA_DECAY",
    "FAIRNESS_WEIGHT",
    "FREEMATCH_STEPS",
    "SAT",
    "SAT_TERM",
    "UNSUPERVISED_WEIGHT",
    "freematch",
]
