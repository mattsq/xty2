"""Tier 0 — the printed execution plan (`DESIGN.md` §8, `FIDELITY.md` §1.2).

The plan is a review artifact: a reviewer diffs it against the card's §3
mapping table and §4 checklist, and that diff *is* the review. Two properties
have to hold for that to mean anything, and both are tested here.

**It is complete.** Everything the card names has to be visible in it —
components and their wiring, which of them touch the raw outcome, the forward
passes, the active objectives with their eligible rows, the trainable groups,
and the resolved hyperparameters.

**It is stable.** The same recipe prints the same bytes, every time and in any
order the components were declared in. A plan that reorders itself run to run
produces a diff full of noise, which is a diff nobody reads. That is why this
is a snapshot test rather than a set of substring assertions: a change to the
format is a change to the review surface and should show up in a diff as one.
"""

from xty2.core import ComponentGraph, Port, Recipe, Stage, compile

from tests.invariants.conftest import (
    HIDDEN,
    ToyEncoder,
    ToyOutcomeHead,
    ToyPosterior,
    ToyPropensity,
    make_schema,
    objective,
    two_head_recipe,
)

TWO_HEAD_PLAN = """\
recipe: two_head
purpose: causal
card: docs/recipes/two_head.md
schema: D = 4, K = 3

components (topological order)
  encoder       [x] -> [x_repr]
  outcome_head  [x_repr] -> [p(y|x,t)]
  propensity    [x_repr] -> [p(t|x)]

data lineage
  encoder       x <- source
  outcome_head  x_repr <- encoder
  propensity    x_repr <- encoder
  reads raw y directly:  none
  depends on raw y:      none

stage fit
  rows: all
  forward passes (1)
    view=identity params=student: encoder -> outcome_head -> propensity
  objectives
    outcome_nll    rows y_observed  requires p(y|x,t) @ view=identity params=student
    treatment_nll  rows t_observed  requires p(t|x) @ view=identity params=student
  trainable
    encoder, outcome_head, propensity

hyperparameters
  architecture.widths_depths = 5
"""

POSTERIOR_PLAN = """\
recipe: with_posterior
purpose: predictive
card: docs/recipes/with_posterior.md
schema: D = 4, K = 3

components (topological order)
  encoder     [x] -> [x_repr]
  propensity  [x_repr] -> [p(t|x)]
  posterior   [x, y] -> [q(t|x,y)]

data lineage
  encoder     x <- source
  propensity  x_repr <- encoder
  posterior   x <- source, y <- source
  reads raw y directly:  posterior
  depends on raw y:      posterior

stage infer
  rows: t_observed
  forward passes (1)
    view=identity params=student: encoder -> propensity -> posterior
  objectives
    posterior_kl   rows t_observed  requires q(t|x,y) @ view=identity params=student
    treatment_nll  rows t_observed  requires p(t|x) @ view=identity params=student
  trainable
    encoder, propensity, posterior

hyperparameters
  architecture.widths_depths = 5
"""


def _posterior_recipe() -> Recipe:
    return Recipe(
        name="with_posterior",
        schema=make_schema(),
        system=ComponentGraph(
            [ToyEncoder(width=HIDDEN), ToyPropensity(), ToyPosterior()]
        ),
        program=(
            Stage(
                name="infer",
                objectives=(
                    objective("posterior_kl", Port.T_GIVEN_XY),
                    objective("treatment_nll", Port.T_GIVEN_X, rows="t_observed"),
                ),
                trainable=("encoder", "propensity", "posterior"),
                rows="t_observed",
            ),
        ),
        card="docs/recipes/with_posterior.md",
        purpose="predictive",
    )


def test_the_plan_prints_the_expected_bytes() -> None:
    assert compile(two_head_recipe()).plan.render() == TWO_HEAD_PLAN


def test_the_plan_shows_which_components_reach_the_raw_outcome() -> None:
    # `used_y` is derived from the graph rather than declared (DESIGN.md §2.2),
    # and the plan is where a reviewer sees it before P10 enforces it.
    assert compile(_posterior_recipe()).plan.render() == POSTERIOR_PLAN


def test_str_and_render_agree() -> None:
    plan = compile(two_head_recipe()).plan
    assert str(plan) == plan.render()


def test_the_plan_is_stable_across_repeated_compilation() -> None:
    assert len({compile(two_head_recipe()).plan.render() for _ in range(3)}) == 1


def test_tie_breaking_follows_declaration_order() -> None:
    # Ties in the topological order are broken by declaration order rather than
    # by set iteration, which is what makes the snapshot above reproducible at
    # all: two independent heads have no order of their own.
    swapped = two_head_recipe(
        system=ComponentGraph(
            [ToyPropensity(), ToyOutcomeHead(), ToyEncoder(width=HIDDEN)]
        )
    )
    plan = compile(swapped).plan
    assert [component.name for component in plan.components] == [
        "encoder",
        "propensity",
        "outcome_head",
    ]
