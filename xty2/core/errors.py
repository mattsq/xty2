"""Exception types for the core contracts.

One base class so a caller can catch everything xty2 raises deliberately, and
one subclass per contract so a test can assert *which* rule fired. Tests that
assert on ``Exception`` pass when the wrong thing goes wrong.

`require_str` lives here too, because it is the guard that turns a wrongly
typed name into the right one of those exceptions.
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


class GraphError(Xty2Error):
    """A component or a `ComponentGraph` violates `DESIGN.md` §2.2 / §3.

    Undeclared reads and writes, a port two components claim to provide, a
    cycle, an unsatisfiable `requires` — everything that is wrong about the
    *graph* independently of any program that runs it.
    """


class CardKeyError(Xty2Error):
    """A card-key binding is unusable (`DESIGN.md` §9.1).

    Either the canonical key is outside the closed vocabulary of
    `FIDELITY.md` §2, or a paper-governed field still holds `REQUIRED`.
    """


class CompileError(Xty2Error):
    """`compile()` rejects a recipe (`DESIGN.md` §8).

    Distinct from `GraphError`: the graph can be perfectly well formed and the
    *program* still ask it for something it cannot supply.
    """


def require_str(label: str, value: object, *, error: type[Xty2Error]) -> str:
    """Runtime type check for a field the annotations already promise is a `str`.

    That promise binds anything mypy sees, and nothing else. A recipe assembled
    by hand or loaded from a dict is not obliged to have been type-checked, and
    the fields this guards — component, stage and recipe names — key module
    dicts, `trainable` lists and the printed plan, so a wrong type surfaces much
    later as something unrecognisable.
    """
    if not isinstance(value, str):
        raise error(f"{label} must be a string, got {type(value)}")
    return value
