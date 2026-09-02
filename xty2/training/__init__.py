"""Sequencing, mixing and execution (`DESIGN.md` §6, §7).

`training/` imports `core/` and never the reverse, which is why the pieces a
recipe *declares* — `Stage`, `Weighted`, the schedules, the `OptimiserSpec` —
live in the leaf layer and only the machinery that *runs* them lives here. So
far that is the loss mixer and its gradient probe (P3), the `gradient`
executor and immutable artifacts (P4), plus the ordered program runner and EMA
teacher parameter set (P8). P10 adds functional `array_fit`, fold-aware
`cross_fit`, and pseudo-label artifacts whose lineage and disjointness are
derived rather than asserted.
"""

from xty2.training.artifacts import (
    ARTIFACT_FORMAT,
    Checkpoint,
    PredictionMode,
    PseudoLabels,
    RunDirectory,
    is_read_only,
)
from xty2.training.executors import (
    MAX_STAGE_STEPS,
    STREAM_STRIDE,
    BatchSource,
    BatchSources,
    ProgramResult,
    StageResult,
    StepRecord,
    run_array_fit,
    run_cross_fit,
    run_program,
    run_stage,
    trainable_only,
)
from xty2.training.loading import build_population, check_fitted_on, iterate
from xty2.training.loss_mixer import (
    PERIODIC_STEPS,
    GradientProbe,
    GradientReport,
    LossMixer,
    MixedLoss,
    ObjectiveLog,
    gradient_report,
)
from xty2.training.selection import MinimumValidationSelection, SelectionResult
from xty2.training.teacher import EMATeacher

__all__ = [
    "ARTIFACT_FORMAT",
    "MAX_STAGE_STEPS",
    "PERIODIC_STEPS",
    "STREAM_STRIDE",
    "BatchSource",
    "BatchSources",
    "Checkpoint",
    "EMATeacher",
    "GradientProbe",
    "GradientReport",
    "LossMixer",
    "MinimumValidationSelection",
    "MixedLoss",
    "ObjectiveLog",
    "PredictionMode",
    "ProgramResult",
    "PseudoLabels",
    "RunDirectory",
    "SelectionResult",
    "StageResult",
    "StepRecord",
    "build_population",
    "check_fitted_on",
    "gradient_report",
    "is_read_only",
    "iterate",
    "run_array_fit",
    "run_cross_fit",
    "run_program",
    "run_stage",
    "trainable_only",
]
