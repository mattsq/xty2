"""Declarative assemblies of registered pieces; no logic (`DESIGN.md` §9)."""

from xty2.recipes.cnflow import CNFLOW_ENCODER_WIDTHS, cnflow
from xty2.recipes.cycle_dual import (
    CYCLE_DUAL_ENCODER_WIDTHS,
    CYCLE_DUAL_OUTCOME_WIDTHS,
    CYCLE_DUAL_POSTERIOR_WIDTHS,
    cycle_dual,
)
from xty2.recipes.mean_teacher import mean_teacher
from xty2.recipes.ssdml import SSDML_ENCODER_WIDTHS, ssdml
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS, tarnet

__all__ = [
    "CNFLOW_ENCODER_WIDTHS",
    "CYCLE_DUAL_ENCODER_WIDTHS",
    "CYCLE_DUAL_OUTCOME_WIDTHS",
    "CYCLE_DUAL_POSTERIOR_WIDTHS",
    "ENCODER_WIDTHS",
    "OUTCOME_WIDTHS",
    "SSDML_ENCODER_WIDTHS",
    "cnflow",
    "cycle_dual",
    "mean_teacher",
    "ssdml",
    "tarnet",
]
