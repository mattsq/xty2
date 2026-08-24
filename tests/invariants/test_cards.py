"""Tier 0 — the card §4 checklist against `plan.hyperparameters` (§1.2, §9.1).

This is the mechanism that stops a card rotting away from the recipe it
describes. `FIDELITY.md` §1.2 states it in one sentence — *every card key not
marked `n/a` is present in `plan.hyperparameters` with a non-null value* — and
three things have to be true for that sentence to be worth running.

**It has to be seen to fail.** A cross-check that has never failed is not
known to work, so the mutations below are the point of this module and not an
afterthought: a key deleted from the *recipe*, a key present but null, and a
key the card calls `n/a` that the recipe sets anyway.

**It runs in both directions.** The card is asserted to name the whole closed
vocabulary, so a card that quietly deletes a line fails rather than passing
with less to check, and a key marked `n/a` must genuinely be absent from the
plan.

**It checks presence, never correctness.** It can prove the recipe sets
`optimisation.lr`; it cannot prove `1e-3` is the paper's. Card review is the
only thing that establishes the latter, and a green run here must not be read
as fidelity.
"""

import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest
from xty2.components import MLPArchitecture, TarnetHead
from xty2.core import CARD_KEY_VOCABULARY, ComponentGraph, ExecutionPlan, compile
from xty2.recipes import tarnet
from xty2.recipes.tarnet import CARD

from tests.invariants.conftest import make_schema

REPOSITORY = Path(__file__).resolve().parents[2]

NOT_APPLICABLE = "n/a"
"""The one value that excuses a key from the cross-check (`FIDELITY.md` §2)."""


# ---------------------------------------------------------------------------
# Reading a card
# ---------------------------------------------------------------------------


def card_checklist(card: str) -> dict[str, str]:
    """Parse a card's §4 YAML block into `{canonical_key: value}`.

    A hand-rolled two-level parser rather than a YAML dependency, for the same
    reason `test_card_keys.py` parses `FIDELITY.md` by hand: the block's
    *shape* — `section:` then indented `key: value` — is part of what is being
    asserted, and a real parser would happily accept a third level the closed
    vocabulary has no way to express.
    """
    text = (REPOSITORY / card).read_text(encoding="utf-8")
    section_text = text.split("## 4. Mechanics checklist", 1)[1]
    block = re.search(r"```yaml\n(.*?)```", section_text, re.DOTALL)
    assert block is not None, f"{card} §4 no longer contains a yaml block"
    checklist: dict[str, str] = {}
    section: str | None = None
    for line in block.group(1).splitlines():
        stripped = line.split("#", 1)[0].rstrip()
        if not stripped:
            continue
        if not line.startswith(" "):
            section = stripped.rstrip(":")
            continue
        assert section is not None, f"{card}: {stripped!r} appears before any section"
        name, _, value = stripped.strip().partition(":")
        checklist[f"{section}.{name}"] = value.strip()
    return checklist


def cross_check(checklist: Mapping[str, str], resolved: Mapping[str, Any]) -> list[str]:
    """Every way a card and a compiled plan can disagree, as a list of reasons.

    Returned rather than raised so the mutation tests can assert on *which*
    disagreement fired. An empty list is a passing cross-check.
    """
    problems: list[str] = []
    unknown = sorted(set(checklist) - CARD_KEY_VOCABULARY)
    if unknown:
        problems.append(f"card names key(s) outside the vocabulary: {unknown}")
    uncovered = sorted(CARD_KEY_VOCABULARY - set(checklist))
    if uncovered:
        problems.append(f"card leaves key(s) unanswered: {uncovered}")
    for key, value in sorted(checklist.items()):
        applicable = value != NOT_APPLICABLE
        present = key in resolved
        if applicable and not present:
            problems.append(f"{key}: card says {value!r}, the recipe sets nothing")
        elif applicable and resolved[key] is None:
            problems.append(f"{key}: card says {value!r}, the recipe sets None")
        elif not applicable and present:
            problems.append(f"{key}: card says n/a, the recipe sets {resolved[key]!r}")
    return problems


def tarnet_plan() -> ExecutionPlan:
    return compile(tarnet(make_schema())).plan


# ---------------------------------------------------------------------------
# The cross-check itself
# ---------------------------------------------------------------------------


def test_the_tarnet_card_and_the_tarnet_plan_agree() -> None:
    assert cross_check(card_checklist(CARD), tarnet_plan().hyperparameters) == []


def test_the_card_answers_every_key_in_the_closed_vocabulary() -> None:
    assert set(card_checklist(CARD)) == CARD_KEY_VOCABULARY


def test_the_card_the_recipe_names_is_the_card_that_exists() -> None:
    assert (REPOSITORY / CARD).is_file()
    assert tarnet(make_schema()).card == CARD


def test_the_card_status_is_one_of_the_declared_statuses() -> None:
    # FIDELITY.md §1.1: the status is the recipe's real state, so a typo in it
    # is a recipe claiming a state that does not exist.
    text = (REPOSITORY / CARD).read_text(encoding="utf-8")
    status = re.search(r"\*\*Status:\*\* `([a-z-]+)`", text)
    assert status is not None
    assert status.group(1) in {
        "draft",
        "reviewed",
        "implemented",
        "smoke-passing",
        "reproduced",
        "deviating",
    }


# ---------------------------------------------------------------------------
# The mutations — the cross-check has to be seen to fail (PLAN.md P5)
# ---------------------------------------------------------------------------


class _ForgetfulHead(TarnetHead):
    """`tarnet_head` with one card key dropped from its bindings.

    The realistic failure, not a synthetic one: a component stops binding a
    field the card names, the recipe still constructs, the plan still prints,
    and nothing but the cross-check notices. `output_parameterisation` is the
    key to drop because this head is its only owner — a key three components
    bind would survive one of them forgetting it.
    """

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        field: key
        for field, key in TarnetHead.CARD_KEYS.items()
        if key != "architecture.output_parameterisation"
    }


def _recipe_with_a_forgetful_head() -> ExecutionPlan:
    schema = make_schema()
    recipe = tarnet(schema)
    architecture = recipe.system["tarnet_head"].architecture
    assert isinstance(architecture, MLPArchitecture)
    rewired = ComponentGraph(
        [
            recipe.system["mlp_encoder"],
            _ForgetfulHead(schema, architecture=architecture),
            recipe.system["categorical_propensity"],
        ]
    )
    return compile(replace(recipe, system=rewired)).plan


def test_the_cross_check_fails_when_the_recipe_stops_setting_a_card_key() -> None:
    problems = cross_check(
        card_checklist(CARD), _recipe_with_a_forgetful_head().hyperparameters
    )
    assert problems == [
        "architecture.output_parameterisation: card says 'gaussian per-arm "
        "mean, unit scale', the recipe sets nothing"
    ]


def test_the_cross_check_fails_on_a_key_present_but_null() -> None:
    # Distinguished from absence deliberately: `FIDELITY.md` §1.2 asks for a
    # non-null value, and a binding that resolved to None would otherwise
    # satisfy a presence test while naming no number.
    resolved = dict(tarnet_plan().hyperparameters)
    resolved["optimisation.lr"] = None
    problems = cross_check(card_checklist(CARD), resolved)
    assert problems == ["optimisation.lr: card says '0.001', the recipe sets None"]


def test_the_cross_check_fails_when_the_card_says_n_a_and_the_recipe_does_not() -> None:
    checklist = dict(card_checklist(CARD))
    checklist["architecture.dropout"] = NOT_APPLICABLE
    problems = cross_check(checklist, tarnet_plan().hyperparameters)
    assert problems == ["architecture.dropout: card says n/a, the recipe sets 0.0"]


def test_the_cross_check_fails_when_the_card_drops_a_key_entirely() -> None:
    checklist = dict(card_checklist(CARD))
    del checklist["teacher.ema_decay"]
    problems = cross_check(checklist, tarnet_plan().hyperparameters)
    assert problems == ["card leaves key(s) unanswered: ['teacher.ema_decay']"]


def test_the_cross_check_fails_when_the_card_invents_a_key() -> None:
    checklist = dict(card_checklist(CARD))
    checklist["optimisation.warmup"] = "500 steps"
    problems = cross_check(checklist, tarnet_plan().hyperparameters)
    assert problems[0] == (
        "card names key(s) outside the vocabulary: ['optimisation.warmup']"
    )


@pytest.mark.parametrize("card", [CARD])
def test_no_card_key_is_answered_with_an_empty_value(card: str) -> None:
    # An empty value passes the presence check against the plan while telling
    # a reader nothing; `FIDELITY.md` §2 wants a value, `n/a` or `unspecified`.
    empty = sorted(key for key, value in card_checklist(card).items() if not value)
    assert empty == []
