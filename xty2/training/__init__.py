"""Sequencing, mixing and execution (`DESIGN.md` §6, §7).

`training/` imports `core/` and never the reverse, which is why the pieces a
recipe *declares* — `Stage`, `Weighted`, the schedules, the `OptimiserSpec` —
live in the leaf layer and only the machinery that *runs* them lives here. So
far that is the loss mixer and its gradient probe (P3), the single-stage
`gradient` executor, and the immutable artifacts and run directory it writes
(P4). `Program`, the `array_fit` and `cross_fit` executors and the pseudo-label
artifact arrive with the packets that need them (P8, P10).
"""

from xty2.training.artifacts import (
    ARTIFACT_FORMAT,
    Checkpoint,
    RunDirectory,
    is_read_only,
)
from xty2.training.executors import (
    BatchSource,
    StageResult,
    StepRecord,
    run_stage,
    trainable_only,
)
from xty2.training.loss_mixer import (
    PERIODIC_STEPS,
    GradientProbe,
    GradientReport,
    LossMixer,
    MixedLoss,
    ObjectiveLog,
    gradient_report,
)

__all__ = [
    "ARTIFACT_FORMAT",
    "PERIODIC_STEPS",
    "BatchSource",
    "Checkpoint",
    "GradientProbe",
    "GradientReport",
    "LossMixer",
    "MixedLoss",
    "ObjectiveLog",
    "RunDirectory",
    "StageResult",
    "StepRecord",
    "gradient_report",
    "is_read_only",
    "run_stage",
    "trainable_only",
]
