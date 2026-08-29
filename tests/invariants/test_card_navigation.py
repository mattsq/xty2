"""Tier 0 — compact cards keep their routes and local section contracts live."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CARDS = tuple(
    sorted(
        path
        for path in (ROOT / "docs" / "recipes").glob("*.md")
        if path.stem != "_TEMPLATE"
    )
)


def test_every_card_routes_implementation_through_sections_two_to_five() -> None:
    route = "**Agent route:** read " + chr(0xA7) + "2" + chr(0x2013) + chr(0xA7) + "5"
    for card in CARDS:
        text = card.read_text(encoding="utf-8")
        assert route in text, card.name
        assert "read §2, §3.2, and §4" not in text, card.name


def test_every_card_keeps_the_normative_framework_additions_heading() -> None:
    for card in CARDS:
        text = card.read_text(encoding="utf-8")
        assert "### 5.1 Framework additions made for this card" in text, card.name


def test_local_reproduction_references_have_live_headings() -> None:
    """Catch plain §6 references that an ordinary Markdown link check misses."""
    signals = {
        "6.1": ("§6.1", "section 6.1", "sections 6.1", "specified in 6.1"),
        "6.2": ("§6.2", "section 6.2", "see 6.2", "6.2 below"),
    }
    for card in CARDS:
        text = card.read_text(encoding="utf-8")
        for number, references in signals.items():
            if any(reference in text for reference in references):
                assert re.search(rf"^### {re.escape(number)}\b", text, re.MULTILINE), (
                    f"{card.name} refers to §{number} but has no matching heading"
                )
