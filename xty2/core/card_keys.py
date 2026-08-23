"""The closed card-key vocabulary and the `REQUIRED` sentinel (`DESIGN.md` §9.1).

`FIDELITY.md` §1.2 asks CI to prove a recipe sets every hyperparameter its card
names. A recipe is arbitrary Python, so that claim needs something to compare
*against*, and §9.1 supplies it in three parts. Two of them live here:

* **The vocabulary is closed.** The keys in `FIDELITY.md` §2 are the whole
  namespace, so cards cannot drift apart from each other or from the code.
  `CARD_KEY_VOCABULARY` is that list as data, and a Tier 0 test reads the
  document and asserts the two agree — a vocabulary nothing checks would rot in
  exactly the way it exists to stop the cards rotting.
* **Paper-governed fields have no usable default.** A field bound to a card key
  holds `REQUIRED` until a recipe passes a value, and constructing its owner
  without one raises. That removes the "explicit or inherited?" ambiguity by
  making inheritance impossible rather than by tracking provenance.

The third part — `plan.hyperparameters`, the flat `{canonical_key: value}` dict
the card cross-check compares against — belongs to the compiler and lives in
`xty2.core.compile`.

Nothing here checks *correctness*. It can prove a recipe sets
`teacher.ema_decay`; it cannot prove 0.999 is the number in the paper. Only the
card review establishes that (`FIDELITY.md` §1.2).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, final

from xty2.core.errors import CardKeyError

CARD_KEY_SECTIONS: Final[Mapping[str, tuple[str, ...]]] = {
    "gradients": (
        "stop_gradients",
        "detached_targets",
        "gradient_clipping",
        "marginal_nll_grad_path",
    ),
    "teacher": (
        "ema_decay",
        "ema_applies_to_buffers",
        "teacher_in_train_mode",
        "teacher_requires_grad",
    ),
    "losses": (
        "reduction",
        "eligible_rows",
        "weights",
        "schedules",
        "temperature",
        "sharpening",
        "confidence_threshold",
    ),
    "optimisation": (
        "optimiser",
        "lr",
        "lr_schedule",
        "weight_decay",
        "batch_size",
        "labelled_unlabelled_ratio",
        "total_steps_or_epochs",
    ),
    "architecture": (
        "widths_depths",
        "activation",
        "normalisation",
        "dropout",
        "initialisation",
        "output_parameterisation",
    ),
    "data": (
        "standardisation",
        "outcome_scaling",
        "treatment_encoding",
        "split_protocol",
        "missingness_mechanism",
    ),
}
"""The mechanics checklist of `FIDELITY.md` §2, section by section."""

CARD_KEY_VOCABULARY: Final[frozenset[str]] = frozenset(
    f"{section}.{key}" for section, keys in CARD_KEY_SECTIONS.items() for key in keys
)
"""Every canonical key, flattened. Adding one is a framework change (§9.1)."""


@final
class _Required:
    """The sentinel a paper-governed field holds until a recipe sets it.

    A singleton, so `value is REQUIRED` is the whole test, and deliberately
    inert: it has no arithmetic, no truth value and no length, so a field that
    escaped the construction-time check would fail loudly at first use rather
    than quietly standing in for a number.
    """

    __slots__ = ()
    _instance: _Required | None = None

    def __new__(cls) -> _Required:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "REQUIRED"

    def __bool__(self) -> bool:
        raise CardKeyError(
            "REQUIRED is a sentinel for a hyperparameter the card governs and "
            "the recipe has not set; it has no value (DESIGN.md §9.1)."
        )


REQUIRED: Any = _Required()
"""Sentinel default for a paper-governed field. Typed `Any` so it can stand in
for a `float` or a `str` field without a lie in the annotation."""


def is_required(value: object) -> bool:
    """Is `value` the unset sentinel?"""
    return value is REQUIRED


def validate_card_key(key: str, *, owner: str) -> None:
    """Raise unless `key` is in the closed vocabulary of `FIDELITY.md` §2."""
    if key in CARD_KEY_VOCABULARY:
        return
    section = key.split(".", 1)[0]
    known = CARD_KEY_SECTIONS.get(section)
    hint = (
        f" Section {section!r} holds {list(known)!r}."
        if known is not None
        else f" Known sections: {sorted(CARD_KEY_SECTIONS)!r}."
    )
    raise CardKeyError(
        f"{owner} binds a field to card key {key!r}, which is not in the "
        f"checklist vocabulary of FIDELITY.md §2.{hint} The vocabulary is "
        "closed: adding a key is a framework change, not a per-card decision "
        "(DESIGN.md §9.1)."
    )


def card_hyperparameters(owner: object) -> dict[str, Any]:
    """The `{canonical_key: value}` bindings `owner` declares via `CARD_KEYS`.

    An owner with no `CARD_KEYS` contributes nothing — that is the ordinary
    case for anything the paper does not govern.

    Raises:
        CardKeyError: if two fields bind to one key, if a key is outside the
            vocabulary, if a bound field does not exist on `owner`, or if a
            bound field still holds `REQUIRED`.
    """
    bindings: Mapping[str, str] = getattr(owner, "CARD_KEYS", {})
    label = _label(owner)
    _reject_shared_keys(bindings, label)
    resolved: dict[str, Any] = {}
    for field, key in bindings.items():
        validate_card_key(key, owner=label)
        if not hasattr(owner, field):
            raise CardKeyError(
                f"{label} binds card key {key!r} to attribute {field!r}, which "
                "it does not have. CARD_KEYS names fields of the object that "
                "declares it (DESIGN.md §9.1)."
            )
        value = getattr(owner, field)
        if is_required(value):
            raise CardKeyError(
                f"{label} was constructed without {field!r}, the field bound to "
                f"card key {key!r}. Paper-governed hyperparameters have no "
                "usable default — the recipe sets them explicitly (DESIGN.md "
                "§9.1, CLAUDE.md standing rules)."
            )
        resolved[key] = value
    return resolved


def _reject_shared_keys(bindings: Mapping[str, str], label: str) -> None:
    """Reject two fields of one owner bound to the same canonical key.

    `CARD_KEYS` maps field to key, so nothing about its *type* stops two fields
    naming one key — and resolving them writes both into one slot, where the
    last one wins by dictionary order. That is the worst available outcome: the
    cross-check goes green because the key is present, and the plan reports
    whichever value happened to be written last.

    Rejected whether or not the two values currently agree. A canonical key
    names one number, so which field the card governs has to be decidable from
    the binding rather than from the values two fields happen to hold on the
    day. A paper that states several numbers under one key binds one field
    holding a tuple.
    """
    fields_by_key: dict[str, list[str]] = {}
    for field, key in bindings.items():
        fields_by_key.setdefault(key, []).append(field)
    shared = sorted(
        (key, fields) for key, fields in fields_by_key.items() if len(fields) > 1
    )
    if shared:
        key, fields = shared[0]
        raise CardKeyError(
            f"{label} binds fields {sorted(fields)!r} to the same card key "
            f"{key!r}. A canonical key names one value, so which field the card "
            "governs would depend on dictionary order, and the cross-check "
            "would pass while the plan reported the other one. Bind one field, "
            "holding a tuple if the paper states several numbers together "
            "(DESIGN.md §9.1)."
        )


def _label(owner: object) -> str:
    name = getattr(owner, "name", None)
    return (
        f"{type(owner).__name__} {name!r}"
        if isinstance(name, str)
        else type(owner).__name__
    )
