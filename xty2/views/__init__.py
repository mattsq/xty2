"""Data views and schema-aware transforms (`DESIGN.md` §5)."""

from xty2.core.views import (
    FeatureValues,
    PreservedField,
    RecomputeFunction,
    RecomputeRule,
    ViewSpec,
    ViewTransform,
)
from xty2.views.corruption import FeatureCorruption
from xty2.views.masking import FeatureMask
from xty2.views.perturbations import BoundedJitter
from xty2.views.pretext import ColumnRoll

__all__ = [
    "BoundedJitter",
    "ColumnRoll",
    "FeatureCorruption",
    "FeatureMask",
    "FeatureValues",
    "PreservedField",
    "RecomputeFunction",
    "RecomputeRule",
    "ViewSpec",
    "ViewTransform",
]
