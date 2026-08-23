"""Core data types: batch, schema, ports, distributions, row populations (P1).

The graph and the compiler (`graph.py`, `compile.py`) land in P2.
"""

from xty2.core.batch import XTYBatch, assert_unchanged_by
from xty2.core.conformance import (
    check_outcome_distribution_contract,
    check_treatment_distribution_contract,
)
from xty2.core.distributions import (
    CategoricalTreatment,
    GaussianOutcome,
    OutcomeDistribution,
    TreatmentDistribution,
    TreatmentMode,
    treatment_mode,
)
from xty2.core.errors import (
    BatchContractError,
    ContractError,
    PortContractError,
    SchemaError,
    Xty2Error,
)
from xty2.core.ports import PORT_SPECS, Axis, Port, PortSpec, port_spec, resolve_shape
from xty2.core.rows import (
    ROW_POPULATIONS,
    RowIndex,
    Rows,
    populations_are_disjoint,
    resolve_rows,
    row_mask,
    validate_population,
)
from xty2.core.schema import FeatureSpec, OutcomeSpec, Schema

__all__ = [
    "PORT_SPECS",
    "ROW_POPULATIONS",
    "Axis",
    "BatchContractError",
    "CategoricalTreatment",
    "ContractError",
    "FeatureSpec",
    "GaussianOutcome",
    "OutcomeDistribution",
    "OutcomeSpec",
    "Port",
    "PortContractError",
    "PortSpec",
    "RowIndex",
    "Rows",
    "Schema",
    "SchemaError",
    "TreatmentDistribution",
    "TreatmentMode",
    "XTYBatch",
    "Xty2Error",
    "assert_unchanged_by",
    "check_outcome_distribution_contract",
    "check_treatment_distribution_contract",
    "populations_are_disjoint",
    "port_spec",
    "resolve_rows",
    "resolve_shape",
    "row_mask",
    "treatment_mode",
    "validate_population",
]
