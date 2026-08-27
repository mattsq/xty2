"""The FlexMatch assembly of `docs/recipes/flexmatch.md`, declarations only."""

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
    CurriculumPseudoLabelTreatmentNLL,
    CurriculumThreshold,
    MissingTreatmentMarginalNLL,
    ObservedOutcomeNLL,
    ObservedTreatmentNLL,
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
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS
from xty2.views import FeatureMask

STRONG_MASK_RATE = 0.2
"""The extra corruption the strong view adds on top of the weak one.

**Not `fixmatch`'s 0.5, and the difference is deviation 2's** — the one place
this recipe departs from that one outside the gate itself. FixMatch §2.3 asks a
strong augmentation to be severe *and label-preserving*; RandAugment and
CTAugment are chosen so that a rotated, sheared, colour-jittered image is still
unambiguously its class. Layering a 50% mask on a 10% one gives an effective
rate of `1 - 0.9 * 0.5 = 0.55` over six columns of which four carry the signal,
and card §5.2 measures what that does to the *Bayes-optimal* classifier —
training-free, on the §6.1 DGP: it flips the label on 16.8% of rows, and on the
rows the gate retains it leaves eq. (8)'s one-hot target attainable on 38%.

Card §5.2's criterion picks this value: keep the Bayes-optimal label on at least
90% of rows, be at least twice the weak view's corruption, and be as severe as
possible subject to both. At `0.2` the effective rate is 0.28, the flip rate
0.074 and the gap 2.8x. `fixmatch` keeps 0.5 and is not changed here; its
constant gate holds eq. (4) inert until the model is already confident, which is
what made the difference invisible on that card and visible on this one.
"""

TAU = 0.95
"""`tau`, §4. FixMatch's value, which §4 says FlexMatch adopts along with the rest.

It is no longer the gate: eq. (8) gates on `T_t(c)` and `tau` is what that rises
*towards* (eq. 12) and the fixed threshold the marks are set at (Alg. 1 line 14).
"""

CURRICULUM = CurriculumThreshold(tau=TAU, warm_up=True, mapping="convex")
"""The whole gate rule, as card §4's `losses.confidence_threshold`.

`warm_up` is Algorithm 1 lines 6-9 and eq. (11); `convex` is the `x / (2 - x)`
of eq. (12) that §3.3 chooses "for our experiments" and §4.4's ablation reports
best. Card §7 records that Algorithm 1 line 11 cites eq. (7), the identity case,
and why the convex form is read as the one the results were produced under.
"""

FLEXMATCH_STEPS = FIXMATCH_STEPS
"""Card §4, and the `K` of the cosine rate schedule.

The paper trains for `2^20` steps; deviation 3 fixes the shared project-local
budget instead, which is what makes a difference between this recipe and
`fixmatch` attributable to the gate rather than to the budget.
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
    # comparison arm *is* that recipe, and a different label budget would make
    # it a different one.
    missingness=MissingnessSpec(mechanism="mcar", observed=OBSERVED_TREATMENTS),
)
"""The four `data.*` card keys."""


def flexmatch(
    schema: Schema, *, recompute_rules: tuple[RecomputeRule, ...] = ()
) -> Recipe:
    """Build the single-stage recipe from `docs/recipes/flexmatch.md`."""
    return Recipe(
        name="flexmatch",
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
                    # Eq. (10), on its own draw of the weak view: FixMatch's
                    # footnote 2 means a labelled row is weakly augmented
                    # twice, and FlexMatch inherits the framework whole.
                    Weighted(
                        ObservedTreatmentNLL(realisation=WEAK_X_LABELLED),
                        weight=1.0,
                        reduction="mean",
                    ),
                    # Eq. (8) — the method. The same two realisations, the same
                    # hard label and the same detached weak-view target as
                    # `fixmatch`'s eq. (4); the gate is `T_t(arg max q_b)`
                    # instead of `tau`, and the marks Algorithm 1 lines 13-17
                    # lay down are what moves it.
                    Weighted(
                        CurriculumPseudoLabelTreatmentNLL(
                            port=Port.T_GIVEN_X,
                            target=WEAK_X,
                            prediction=STRONG_X,
                            threshold=CURRICULUM,
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
                # §4 reports from an EMA at 0.999. Nothing reads it during
                # training — eq. (8)'s label and Algorithm 1's marks both come
                # from the current network — so it is declared for what it is,
                # and the compiler checks that no objective takes it as a
                # target.
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
                    lr_schedule=CosineDecay(steps=FLEXMATCH_STEPS, phase=7 / 16),
                    clipping=GradientClipping.none(),
                ),
                steps=FLEXMATCH_STEPS,
                sampler=FIXMATCH_SAMPLER,
            ),
        ),
        card="docs/recipes/flexmatch.md",
        purpose="causal",
        data=DATA_POLICY,
        views=(
            # §2 restates FixMatch's framework unchanged and §4 adopts its
            # settings, so the weak view is `fixmatch`'s exactly and the strong
            # one is its shape at a corruption this card measured rather than
            # inherited (deviation 2, `STRONG_MASK_RATE` above). The weak rate
            # and the preserved-field set are imported so that those have one
            # home; the two `ViewSpec` constructions are restated, which is a
            # second place to edit one reviewed decision (card §4). Tier 0's
            # plan comparison against `fixmatch` enumerates every line the two
            # recipes differ in, so a divergence beyond these cannot appear.
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
    "CURRICULUM",
    "DATA_POLICY",
    "FLEXMATCH_STEPS",
    "STRONG_MASK_RATE",
    "TAU",
    "flexmatch",
]
