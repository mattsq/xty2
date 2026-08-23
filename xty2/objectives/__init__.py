"""Losses as independent objects (`DESIGN.md` §4).

An objective is a function of data and predictions and nothing else: it reads
declared `(port, realisation)` pairs out of `State`, gathers by the eligible
row set the compiler resolved for it, and returns an **unweighted** `LossTerm`.
It never weights itself, never calls `.backward()`, never mutates state and
never touches parameters — weighting is the mixer's job (§6) and stepping is
the executor's (§7).

Four of the ten objectives in §4.2 exist so far: the three likelihood terms
the first recipe needs (P3, P5), plus the view-keyed ``ConsistencyLoss`` (P6).
The rest arrive with the recipes that consume them; migration is lazy by
design (§11).
"""

from xty2.objectives.consistency import (
    CONSISTENCY_DIVERGENCES,
    STOP_GRADIENTS,
    ConsistencyDivergence,
    ConsistencyLoss,
    StopGrad,
)
from xty2.objectives.marginal import (
    GRAD_PATHS,
    GradPath,
    MissingTreatmentMarginalNLL,
)
from xty2.objectives.supervised import ObservedOutcomeNLL, ObservedTreatmentNLL

__all__ = [
    "CONSISTENCY_DIVERGENCES",
    "GRAD_PATHS",
    "STOP_GRADIENTS",
    "ConsistencyDivergence",
    "ConsistencyLoss",
    "GradPath",
    "MissingTreatmentMarginalNLL",
    "ObservedOutcomeNLL",
    "ObservedTreatmentNLL",
    "StopGrad",
]
