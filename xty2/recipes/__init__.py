"""Declarative assemblies of registered pieces; no logic (`DESIGN.md` §9)."""

from xty2.recipes.cnflow import CNFLOW_ENCODER_WIDTHS, cnflow
from xty2.recipes.cycle_dual import (
    CYCLE_DUAL_ENCODER_WIDTHS,
    CYCLE_DUAL_OUTCOME_WIDTHS,
    CYCLE_DUAL_POSTERIOR_WIDTHS,
    cycle_dual,
)
from xty2.recipes.doublematch import (
    COSINE_PHASE,
    DOUBLEMATCH_STEPS,
    SELF_SUPERVISED_WEIGHT,
    doublematch,
)
from xty2.recipes.fixmatch import (
    FIXMATCH_STEPS,
    STRONG_MASK_RATE,
    WEAK_MASK_RATE,
    fixmatch,
)
from xty2.recipes.mean_teacher import mean_teacher
from xty2.recipes.scarf import (
    CORRUPTION_RATE,
    JOINT_FIT_STEPS,
    PRETRAIN_STEPS,
    PROJECTION_WIDTHS,
    TEMPERATURE,
    scarf,
)
from xty2.recipes.ssdml import SSDML_ENCODER_WIDTHS, ssdml
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS, tarnet

__all__ = [
    "CNFLOW_ENCODER_WIDTHS",
    "CORRUPTION_RATE",
    "COSINE_PHASE",
    "CYCLE_DUAL_ENCODER_WIDTHS",
    "CYCLE_DUAL_OUTCOME_WIDTHS",
    "CYCLE_DUAL_POSTERIOR_WIDTHS",
    "DOUBLEMATCH_STEPS",
    "ENCODER_WIDTHS",
    "FIXMATCH_STEPS",
    "JOINT_FIT_STEPS",
    "OUTCOME_WIDTHS",
    "PRETRAIN_STEPS",
    "PROJECTION_WIDTHS",
    "SELF_SUPERVISED_WEIGHT",
    "SSDML_ENCODER_WIDTHS",
    "STRONG_MASK_RATE",
    "TEMPERATURE",
    "WEAK_MASK_RATE",
    "cnflow",
    "cycle_dual",
    "doublematch",
    "fixmatch",
    "mean_teacher",
    "scarf",
    "ssdml",
    "tarnet",
]
