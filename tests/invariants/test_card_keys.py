"""Tier 0 — the card-key vocabulary is closed, and matches the document.

`DESIGN.md` §9.1 makes the checklist in `FIDELITY.md` §2 the whole namespace
for card keys, so that cards cannot drift apart from each other or from the
code. That claim needs the same treatment as every other claim in this repo:
the vocabulary in the code and the vocabulary in the document are read here and
compared, because a closed vocabulary nobody checks rots in exactly the way it
exists to stop the cards rotting.
"""

import re
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import pytest
from xty2.core import (
    CARD_KEY_SECTIONS,
    CARD_KEY_VOCABULARY,
    REQUIRED,
    CardKeyError,
    card_hyperparameters,
    is_required,
    validate_card_key,
)

FIDELITY = Path(__file__).resolve().parents[2] / "docs" / "FIDELITY.md"


def _checklist_from_the_document() -> set[str]:
    """Parse the YAML block under `FIDELITY.md` §2 into canonical keys.

    Deliberately a hand-rolled parser over a two-level block rather than a YAML
    dependency: the block's *shape* — `section:` then indented `key:` — is what
    is being asserted, and a real parser would happily accept a third level
    that the vocabulary has no way to express.
    """
    text = FIDELITY.read_text(encoding="utf-8")
    section = text.split("## 2. The mechanics checklist", 1)[1]
    block = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert block is not None, "FIDELITY.md §2 no longer contains a yaml block"
    keys: set[str] = set()
    current: str | None = None
    for line in block.group(1).splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        name = stripped.strip().rstrip(":")
        if not line.startswith(" "):
            current = name
        else:
            assert current is not None, f"key {name!r} appears before any section"
            keys.add(f"{current}.{name}")
    return keys


def test_the_vocabulary_in_the_code_is_the_one_in_the_document() -> None:
    assert _checklist_from_the_document() == CARD_KEY_VOCABULARY


def test_every_section_contributes_at_least_one_key() -> None:
    assert all(CARD_KEY_SECTIONS.values())
    assert len(CARD_KEY_VOCABULARY) == sum(
        len(keys) for keys in CARD_KEY_SECTIONS.values()
    )


def test_a_key_outside_the_vocabulary_is_rejected_with_the_section_it_missed() -> None:
    with pytest.raises(CardKeyError, match="Section 'teacher' holds"):
        validate_card_key("teacher.ema_rate", owner="a component")


def test_an_unknown_section_is_rejected_with_the_sections_that_exist() -> None:
    with pytest.raises(CardKeyError, match="Known sections"):
        validate_card_key("training.lr", owner="a component")


def test_a_key_in_the_vocabulary_is_accepted() -> None:
    validate_card_key("teacher.ema_decay", owner="a component")


class _Bound:
    CARD_KEYS: ClassVar[Mapping[str, str]] = {"decay": "teacher.ema_decay"}

    def __init__(self, decay: float = REQUIRED) -> None:
        self.decay = decay


class _BindsAMissingField:
    CARD_KEYS: ClassVar[Mapping[str, str]] = {"decay": "teacher.ema_decay"}


def test_an_unset_paper_governed_field_is_caught_at_the_binding() -> None:
    with pytest.raises(CardKeyError, match="no usable default"):
        card_hyperparameters(_Bound())


def test_a_set_field_resolves_to_its_canonical_key() -> None:
    assert card_hyperparameters(_Bound(decay=0.999)) == {"teacher.ema_decay": 0.999}


def test_a_binding_to_a_field_that_does_not_exist_is_rejected() -> None:
    with pytest.raises(CardKeyError, match="which it does not have"):
        card_hyperparameters(_BindsAMissingField())


def test_an_owner_with_no_card_keys_contributes_nothing() -> None:
    assert card_hyperparameters(object()) == {}


def test_the_sentinel_is_a_singleton() -> None:
    assert is_required(REQUIRED)
    assert not is_required(0.999)
    assert type(REQUIRED)() is REQUIRED
