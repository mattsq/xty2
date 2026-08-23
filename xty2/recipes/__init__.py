"""Declarative assemblies of registered pieces; no logic (`DESIGN.md` §9)."""

from xty2.recipes.cnflow import CNFLOW_ENCODER_WIDTHS, cnflow
from xty2.recipes.tarnet import ENCODER_WIDTHS, OUTCOME_WIDTHS, tarnet

__all__ = [
    "CNFLOW_ENCODER_WIDTHS",
    "ENCODER_WIDTHS",
    "OUTCOME_WIDTHS",
    "cnflow",
    "tarnet",
]
