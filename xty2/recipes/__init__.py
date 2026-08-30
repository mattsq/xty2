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
from xty2.recipes.flexmatch import CURRICULUM, FLEXMATCH_STEPS, TAU, flexmatch
from xty2.recipes.freematch import (
    EMA_DECAY,
    FAIRNESS_WEIGHT,
    FREEMATCH_STEPS,
    SAT,
    UNSUPERVISED_WEIGHT,
    freematch,
)
from xty2.recipes.mean_teacher import mean_teacher
from xty2.recipes.paws import (
    LARGE_CORRUPTION_RATE,
    MISSING_ANCHORS,
    PAWS_SAMPLER,
    SHARPENING,
    SMALL_CORRUPTION_RATE,
    SUPPORT_PER_TREATMENT,
    paws,
)
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
from xty2.recipes.variational_treatment import (
    POSTERIOR_WIDTHS,
    VARIATIONAL_TREATMENT_STEPS,
    variational_treatment,
)

__all__ = [
    "CNFLOW_ENCODER_WIDTHS",
    "CORRUPTION_RATE",
    "COSINE_PHASE",
    "CURRICULUM",
    "CYCLE_DUAL_ENCODER_WIDTHS",
    "CYCLE_DUAL_OUTCOME_WIDTHS",
    "CYCLE_DUAL_POSTERIOR_WIDTHS",
    "DOUBLEMATCH_STEPS",
    "EMA_DECAY",
    "ENCODER_WIDTHS",
    "FAIRNESS_WEIGHT",
    "FIXMATCH_STEPS",
    "FLEXMATCH_STEPS",
    "FREEMATCH_STEPS",
    "JOINT_FIT_STEPS",
    "LARGE_CORRUPTION_RATE",
    "MISSING_ANCHORS",
    "OUTCOME_WIDTHS",
    "PAWS_SAMPLER",
    "POSTERIOR_WIDTHS",
    "PRETRAIN_STEPS",
    "PROJECTION_WIDTHS",
    "SAT",
    "SELF_SUPERVISED_WEIGHT",
    "SHARPENING",
    "SMALL_CORRUPTION_RATE",
    "SSDML_ENCODER_WIDTHS",
    "STRONG_MASK_RATE",
    "SUPPORT_PER_TREATMENT",
    "TAU",
    "TEMPERATURE",
    "UNSUPERVISED_WEIGHT",
    "VARIATIONAL_TREATMENT_STEPS",
    "WEAK_MASK_RATE",
    "cnflow",
    "cycle_dual",
    "doublematch",
    "fixmatch",
    "flexmatch",
    "freematch",
    "mean_teacher",
    "paws",
    "scarf",
    "ssdml",
    "tarnet",
    "variational_treatment",
]
