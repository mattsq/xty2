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

The objective catalogue grows lazily with reviewed recipes. The variational
 treatment objective is the first term that keeps a predicted categorical
 distribution soft while differentiating through its weights over candidate
 treatments; exact marginalisation and pseudo-label objectives remain separate
 objects because they express different source equations.
"""

from xty2.objectives.adaptive_threshold import (
    LOG_FLOOR,
    SelfAdaptiveFairness,
    SelfAdaptiveThreshold,
    SelfAdaptiveThresholds,
    SelfAdaptiveThresholdTreatmentNLL,
)
from xty2.objectives.comatch import (
    CoMatchConfidenceThresholds,
    MemorySmoothedLabelGraph,
    MemorySmoothedLabels,
    MemorySmoothedPseudoLabelTreatmentNLL,
    PseudoLabelGraphContrastive,
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
from xty2.objectives.simmatch import (
    LabeledMemoryInstanceConsistency,
    LabeledSimilarityMemory,
    PropagatedTargets,
    SimilarityMatchingSpec,
    SimilarityMatchingTemperatures,
    SimilarityMatchingTreatmentNLL,
)
from xty2.objectives.supervised import ObservedOutcomeNLL, ObservedTreatmentNLL
from xty2.objectives.support_set import (
    MeanEntropyMaximisation,
    SupportSetClassifier,
    SupportSetPseudoLabelConsistency,
    TargetRole,
)
from xty2.objectives.uda import (
    ConfidenceMaskedConsistencyLoss,
    TrainingSignalAnnealedTreatmentNLL,
    UDAConfidenceThresholds,
    UDADivergence,
    UDASchedule,
    UDASharpening,
    UDAStopGrad,
)
from xty2.objectives.variational import VariationalTreatmentELBO

__all__ = [
    "CONSISTENCY_DIVERGENCES",
    "GRAD_PATHS",
    "LOG_FLOOR",
    "STOP_GRADIENTS",
    "UNUSED",
    "CoMatchConfidenceThresholds",
    "ConfidenceMaskedConsistencyLoss",
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
    "LabeledMemoryInstanceConsistency",
    "LabeledSimilarityMemory",
    "MeanEntropyMaximisation",
    "MemorySmoothedLabelGraph",
    "MemorySmoothedLabels",
    "MemorySmoothedPseudoLabelTreatmentNLL",
    "MissingTreatmentMarginalNLL",
    "ObservedOutcomeNLL",
    "ObservedTreatmentNLL",
    "PropagatedTargets",
    "PseudoLabelGraphContrastive",
    "PseudoLabelStopGrad",
    "PseudoLabelTreatmentNLL",
    "SelfAdaptiveFairness",
    "SelfAdaptiveThreshold",
    "SelfAdaptiveThresholdTreatmentNLL",
    "SelfAdaptiveThresholds",
    "Sharpening",
    "SimilarityMatchingSpec",
    "SimilarityMatchingTemperatures",
    "SimilarityMatchingTreatmentNLL",
    "StopGrad",
    "SupportSetClassifier",
    "SupportSetPseudoLabelConsistency",
    "TargetRole",
    "TrainingSignalAnnealedTreatmentNLL",
    "UDAConfidenceThresholds",
    "UDADivergence",
    "UDASchedule",
    "UDASharpening",
    "UDAStopGrad",
    "VariationalTreatmentELBO",
]
