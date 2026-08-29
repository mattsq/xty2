"""Losses as independent objects (`DESIGN.md` §4).

An objective is a function of data and predictions and nothing else: it reads
declared `(port, realisation)` pairs out of `State`, gathers by the eligible
row set the compiler resolved for it, and returns an **unweighted** `LossTerm`.
It never weights itself, never calls `.backward()`, never mutates `State`, the
batch or parameters — weighting is the mixer's job (§6) and stepping is the
executor's (§7). The one exception is state of its *own*: an objective may
declare `initial_state` and read the result back through
`TrainContext.objective_states`, which the executor builds once per stage
execution (§4, `StatefulObjective`). `CurriculumPseudoLabelTreatmentNLL` and
`SelfAdaptiveThresholdTreatmentNLL` are the two objectives that do, and the
second is also the first whose state a *sibling* objective in the same stage
reads: `SelfAdaptiveFairness` names it, because FreeMatch's eq. (8) and eq. (11)
are two weighted terms over one set of statistics.

Twelve objectives exist so far: the three likelihood terms the first recipe
needs (P3, P5), the view-keyed ``ConsistencyLoss`` (P6), the confidence-gated
``PseudoLabelTreatmentNLL`` that FixMatch is assembled from, the
``InfoNCEContrastive`` that SCARF is, the ``CosineFeatureConsistency``
DoubleMatch adds to FixMatch, the ``CurriculumPseudoLabelTreatmentNLL``
FlexMatch replaces FixMatch's gate with, and the
``SelfAdaptiveThresholdTreatmentNLL`` / ``SelfAdaptiveFairness`` pair FreeMatch
is, and PAWS's support-set consistency and me-max terms. The rest arrive with
the recipes that consume them; migration is lazy by
design (§11).
"""

from xty2.objectives.adaptive_threshold import (
    LOG_FLOOR,
    SelfAdaptiveFairness,
    SelfAdaptiveThreshold,
    SelfAdaptiveThresholds,
    SelfAdaptiveThresholdTreatmentNLL,
)
from xty2.objectives.consistency import (
    CONSISTENCY_DIVERGENCES,
    STOP_GRADIENTS,
    ConsistencyDivergence,
    ConsistencyLoss,
    StopGrad,
)
from xty2.objectives.contrastive import InfoNCEContrastive
from xty2.objectives.curriculum import (
    UNUSED,
    CurriculumMapping,
    CurriculumPseudoLabelTreatmentNLL,
    CurriculumStatus,
    CurriculumThreshold,
)
from xty2.objectives.feature_consistency import (
    CosineFeatureConsistency,
    FeatureStopGrad,
)
from xty2.objectives.marginal import (
    GRAD_PATHS,
    GradPath,
    MissingTreatmentMarginalNLL,
)
from xty2.objectives.pseudo_label import (
    PseudoLabelStopGrad,
    PseudoLabelTreatmentNLL,
    Sharpening,
)
from xty2.objectives.supervised import ObservedOutcomeNLL, ObservedTreatmentNLL
from xty2.objectives.support_set import (
    MeanEntropyMaximisation,
    SupportSetClassifier,
    SupportSetPseudoLabelConsistency,
    TargetRole,
)

__all__ = [
    "CONSISTENCY_DIVERGENCES",
    "GRAD_PATHS",
    "LOG_FLOOR",
    "STOP_GRADIENTS",
    "UNUSED",
    "ConsistencyDivergence",
    "ConsistencyLoss",
    "CosineFeatureConsistency",
    "CurriculumMapping",
    "CurriculumPseudoLabelTreatmentNLL",
    "CurriculumStatus",
    "CurriculumThreshold",
    "FeatureStopGrad",
    "GradPath",
    "InfoNCEContrastive",
    "MeanEntropyMaximisation",
    "MissingTreatmentMarginalNLL",
    "ObservedOutcomeNLL",
    "ObservedTreatmentNLL",
    "PseudoLabelStopGrad",
    "PseudoLabelTreatmentNLL",
    "SelfAdaptiveFairness",
    "SelfAdaptiveThreshold",
    "SelfAdaptiveThresholdTreatmentNLL",
    "SelfAdaptiveThresholds",
    "Sharpening",
    "StopGrad",
    "SupportSetClassifier",
    "SupportSetPseudoLabelConsistency",
    "TargetRole",
]
