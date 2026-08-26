"""Tier 0 — card deviations and the YAGNI ledger are reconciled both ways.

`DESIGN.md` §11.3 makes a `framework-limitation` deviation a debt with a named
creditor: the card cites a ledger key, and the ledger row cites the cards. Two
documents agreeing is not something either can enforce alone, so it is enforced
here, for the same reason `test_card_keys.py` reads `FIDELITY.md` §2 rather
than trusting it — a register nobody reconciles rots exactly like the
documentation it exists to protect.

The assertion that does the work is `test_every_debt_names_a_live_ledger_key`.
Discharging a ledger entry means deleting its row, and deleting a row fails
this test on every card still citing it. That is the collection step: the PR
that builds the capability cannot go green until each card that was paying for
its absence has been revisited (`FIDELITY.md` §5.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs"
DESIGN = DOCS / "DESIGN.md"
CARDS = DOCS / "recipes"

KINDS = frozenset({"judgement", "framework-limitation", "withdrawn"})
EMPTY = frozenset({"", "—", "-", "n/a"})


@dataclass(frozen=True)
class Deviation:
    """One row of a card's §5 table."""

    card: str
    number: str
    kind: str
    blocked_on: str

    @property
    def citation(self) -> str:
        """How the ledger's **Who is paying** column names this row."""
        return f"`{self.card}` §5.{self.number}"


def _cells(line: str) -> list[str]:
    """The cells of one markdown table row, unescaped and stripped."""
    return [cell.replace(r"\|", "|").strip() for cell in line.strip()[1:-1].split("|")]


def _unquote(cell: str) -> str:
    return cell.strip().strip("`").strip()


def _first_table(text: str) -> list[list[str]]:
    """The rows of the first markdown table in `text`, header and rule dropped."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if rows:
                break
            continue
        rows.append(_cells(line))
    assert rows, "expected a markdown table"
    return rows[2:]


def _card_paths() -> list[Path]:
    # `_TEMPLATE.md` is the blank form, not a card: its one empty row has no
    # kind to check and no ledger key to cite.
    return sorted(path for path in CARDS.glob("*.md") if path.stem != "_TEMPLATE")


def _deviations() -> list[Deviation]:
    found: list[Deviation] = []
    for path in _card_paths():
        section = path.read_text(encoding="utf-8").split("\n## 5.", 1)
        assert len(section) == 2, f"{path.name} has no §5 deviations section"
        for row in _first_table(section[1]):
            assert len(row) >= 3, f"{path.name} §5 row {row!r} is missing columns"
            found.append(
                Deviation(
                    card=path.stem,
                    number=row[0].strip(),
                    kind=_unquote(row[1]),
                    blocked_on=_unquote(row[2]),
                )
            )
    assert found, "no deviations parsed — the §5 table shape has changed"
    return found


def _ledger() -> dict[str, list[str]]:
    """`DESIGN.md` §11.4's `{key: [citation, ...]}`, from its **Who is paying**."""
    body = DESIGN.read_text(encoding="utf-8").split("### 11.4 The ledger", 1)
    assert len(body) == 2, "DESIGN.md §11.4 is no longer titled 'The ledger'"
    rows = _first_table(body[1])
    ledger: dict[str, list[str]] = {}
    for row in rows:
        key = _unquote(row[0])
        paying = row[-1].strip()
        cited = [part.strip() for part in paying.split(";")]
        citations = [] if paying in EMPTY else cited
        assert key not in ledger, f"duplicate ledger key {key!r}"
        ledger[key] = citations
    assert ledger, "no ledger rows parsed — the §11.4 table shape has changed"
    return ledger


DEVIATIONS = _deviations()
LEDGER = _ledger()

CITATION = re.compile(r"^`(?P<card>[a-z0-9_]+)` §5\.(?P<number>\d+)\b")


@pytest.mark.parametrize("deviation", DEVIATIONS, ids=lambda d: f"{d.card}-{d.number}")
def test_every_deviation_declares_one_of_the_three_kinds(deviation: Deviation) -> None:
    """`FIDELITY.md` §5: a deviation is a judgement, a debt, or withdrawn.

    An untyped row is the failure the typing exists to stop — a framework
    limitation and a modelling decision rendered in the same typeface, so the
    debt reads to the next reviewer as a decision somebody already made.
    """
    assert deviation.kind in KINDS, (
        f"{deviation.card} §5.{deviation.number} has kind {deviation.kind!r}; "
        f"expected one of {sorted(KINDS)}"
    )


@pytest.mark.parametrize("deviation", DEVIATIONS, ids=lambda d: f"{d.card}-{d.number}")
def test_every_debt_names_a_live_ledger_key(deviation: Deviation) -> None:
    """The collection step (`DESIGN.md` §11.3).

    A `framework-limitation` cites a key that must exist in §11.4. Discharging
    that entry deletes its row and fails this test here, on the card that was
    paying for the absence — which is the point: the capability's own PR is
    what forces the earlier card to be revisited, at the moment the agent
    holding the context is cheapest to ask.
    """
    if deviation.kind != "framework-limitation":
        assert deviation.blocked_on in EMPTY, (
            f"{deviation.card} §5.{deviation.number} is a {deviation.kind} and "
            f"cites {deviation.blocked_on!r}. Only a framework limitation is "
            "blocked on anything; a judgement we would make again is blocked "
            "on nothing."
        )
        return
    assert deviation.blocked_on in LEDGER, (
        f"{deviation.card} §5.{deviation.number} is blocked on "
        f"{deviation.blocked_on!r}, which is not a key in DESIGN.md §11.4. "
        "Either the ledger entry was discharged — in which case revisit this "
        "deviation and withdraw it or restate it as a judgement "
        "(FIDELITY.md §5.2) — or the ledger is missing a row and this pass "
        "writes it."
    )


def test_the_ledger_and_the_cards_name_each_other() -> None:
    """Neither direction may drift.

    A ledger row whose **Who is paying** has gone stale understates the cost of
    an omission, and a card citing a key the ledger does not credit it for
    hides the same cost from the gate that reads the ledger (`PLAN.md`, Gate 2).
    """
    from_cards: dict[str, set[str]] = {}
    for deviation in DEVIATIONS:
        if deviation.kind == "framework-limitation":
            from_cards.setdefault(deviation.blocked_on, set()).add(deviation.citation)

    from_ledger: dict[str, set[str]] = {}
    for key, citations in LEDGER.items():
        for citation in citations:
            match = CITATION.match(citation)
            assert match is not None, (
                f"DESIGN.md §11.4 row {key!r} cites {citation!r}, which is not "
                "of the form `<card>` §5.<n>. Entries are separated by ';' and "
                "each names one card row, so that this test can check it."
            )
            from_ledger.setdefault(key, set()).add(match.group(0))

    assert from_ledger == from_cards, (
        "DESIGN.md §11.4 and the cards disagree about who is paying for what.\n"
        f"  ledger says: { {k: sorted(v) for k, v in sorted(from_ledger.items())} }\n"
        f"  cards say:   { {k: sorted(v) for k, v in sorted(from_cards.items())} }"
    )


def test_a_reproduced_card_carrying_a_debt_says_so_in_its_reproduction_target() -> None:
    """`FIDELITY.md` §5.2 (4).

    `reproduced` on a card with an open framework limitation is a claim that
    the omitted mechanic did not matter for the published number. That may well
    be true — but it is a claim, and a claim belongs in writing next to the
    result rather than in the gap between two sections.
    """
    debts: dict[str, list[str]] = {}
    for deviation in DEVIATIONS:
        if deviation.kind == "framework-limitation":
            debts.setdefault(deviation.card, []).append(deviation.number)

    for card, numbers in sorted(debts.items()):
        text = (CARDS / f"{card}.md").read_text(encoding="utf-8")
        if "**Status:** `reproduced`" not in text:
            continue
        target = text.split("\n## 6.", 1)
        assert len(target) == 2, f"{card}.md has no §6 reproduction target"
        cited = [number for number in numbers if f"§5.{number}" in target[1]]
        assert cited == numbers, (
            f"{card}.md is `reproduced` and carries open framework limitations "
            f"{numbers}, but §6 only refers to {cited}. A published number "
            "measured without a mechanic the paper states needs that said out "
            "loud beside the number."
        )
