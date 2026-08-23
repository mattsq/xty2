"""Exception types for the core contracts.

One base class so a caller can catch everything xty2 raises deliberately, and
one subclass per contract so a test can assert *which* rule fired. Tests that
assert on ``Exception`` pass when the wrong thing goes wrong.
"""


class Xty2Error(Exception):
    """Base class for every error xty2 raises on purpose."""


class SchemaError(Xty2Error):
    """A `Schema` or `FeatureSpec` is internally inconsistent (`DESIGN.md` §1.2)."""


class BatchContractError(Xty2Error):
    """An `XTYBatch` violates the §1.1 contract (shape, dtype, mask, row ids)."""


class PortContractError(Xty2Error):
    """A value offered for a port does not match its `PortSpec` (`DESIGN.md` §2)."""


class ContractError(Xty2Error):
    """A distribution fails the conformance checks of `DESIGN.md` §3.1."""
