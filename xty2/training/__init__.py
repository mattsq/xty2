"""Sequencing, mixing and execution (`DESIGN.md` §6, §7).

`training/` imports `core/` and never the reverse, which is why the pieces a
recipe *declares* — `Stage`, `Weighted`, the schedules — live in the leaf layer
and only the machinery that *runs* them lives here. So far that is the loss
mixer and its gradient probe (P3). `Program`, the executors and the artifacts
arrive with the packets that need them (P4, P8, P10).
"""

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
    "PERIODIC_STEPS",
    "GradientProbe",
    "GradientReport",
    "LossMixer",
    "MixedLoss",
    "ObjectiveLog",
    "gradient_report",
]
