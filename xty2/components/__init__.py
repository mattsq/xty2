"""Parameterisations, and nothing else (`DESIGN.md` §3).

A component reads declared ports, writes declared ports and computes no loss.
Three of them exist so far, and they are the three the first recipe needs
(P5): the shared trunk, the per-arm outcome head TARNet is named for, and a
categorical propensity head. `conditional_flow` (P7), the posterior
`q(t|x,y)` (P11) and the rest of `DESIGN.md` §10's subpackages arrive with the
recipes that consume them; migration is lazy by design (§11).

`architecture` sits at the root because all three subpackages read it: the
`architecture.*` card block is one value the whole recipe shares, not a field
each component sets for itself.
"""

from xty2.components.architecture import (
    ACTIVATIONS,
    INITIALISATIONS,
    NORMALISATIONS,
    ActivationName,
    InitialisationName,
    MLPArchitecture,
    MLPComponent,
    NormalisationName,
    build_mlp,
)
from xty2.components.encoders.mlp import MLPEncoder
from xty2.components.outcome.tarnet import OUTPUT_PARAMETERISATION, TarnetHead
from xty2.components.treatment.categorical import CategoricalPropensity

__all__ = [
    "ACTIVATIONS",
    "INITIALISATIONS",
    "NORMALISATIONS",
    "OUTPUT_PARAMETERISATION",
    "ActivationName",
    "CategoricalPropensity",
    "InitialisationName",
    "MLPArchitecture",
    "MLPComponent",
    "MLPEncoder",
    "NormalisationName",
    "TarnetHead",
    "build_mlp",
]
