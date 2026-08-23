"""Tier 0 — the compiler's rejections and its plan (`DESIGN.md` §7.0, §8, §9.1).

Every test here is a program that is wrong *by construction*: not wrong on some
batch, wrong before any data exists. That is the division the design draws —
the compiler rejects programs that cannot be right, the loader and the runtime
catch executions that turn out wrong in fact — and it is why each of these is a
compile error rather than a term that quietly returns `n = 0` forever.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
import torch
from xty2.core import (
    DEFAULT,
    REQUIRED,
    CompileError,
    ComponentGraph,
    LossTerm,
    Port,
    Ramp,
    Realisation,
    Recipe,
    RowIndex,
    Rows,
    Stage,
    State,
    TrainContext,
    Weighted,
    XTYBatch,
    compile,
    reduce_rows,
)

from tests.invariants.conftest import (
    HIDDEN,
    STEPS,
    ToyEncoder,
    ToyPosterior,
    ToyPropensity,
    make_schema,
    objective,
    stage,
    two_head_recipe,
    weighted,
)


def _stage(*objectives: Weighted, **overrides: Any) -> Stage:
    return stage(objectives=objectives, **overrides)


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_well_formed_recipe_compiles() -> None:
    run = compile(two_head_recipe())
    assert run.stage("fit").name == "fit"
    assert [o.name for o in run.stage("fit").objectives] == [
        "outcome_nll",
        "treatment_nll",
    ]


def test_the_compiler_plans_one_pass_per_demanded_realisation() -> None:
    run = compile(two_head_recipe())
    passes = run.stage("fit").passes
    assert [forward.realisation for forward in passes] == [DEFAULT]


def test_a_pass_runs_only_the_components_its_objectives_depend_on() -> None:
    # The propensity head is in the graph but nothing active needs it, so it is
    # not executed: "the minimum number of forward passes" (DESIGN.md §8.3)
    # prunes within a pass as well as across them.
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("outcome_nll", Port.Y_GIVEN_XT),
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    assert compile(recipe).stage("fit").passes[0].components == (
        "encoder",
        "outcome_head",
    )


def test_the_eligible_set_is_the_stage_scope_intersected_with_the_objective() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("outcome_nll", Port.Y_GIVEN_XT, rows="y_observed"),
                rows="t_observed",
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    compiled = compile(recipe).stage("fit").objectives[0]
    assert compiled.rows == ("t_observed", "y_observed")


def test_an_all_rows_declaration_drops_out_of_the_intersection() -> None:
    compiled = compile(two_head_recipe()).stage("fit").objectives[0]
    assert compiled.rows == ("y_observed",)


def test_state_holds_the_planned_realisation(batch: XTYBatch) -> None:
    run = compile(two_head_recipe())
    state = run.state("fit", batch)
    assert state.realisations == (DEFAULT,)
    assert Port.Y_GIVEN_XT in state.default
    assert Port.T_GIVEN_X in state.default


# ---------------------------------------------------------------------------
# Compile-time rejections
# ---------------------------------------------------------------------------


def test_an_objective_requiring_a_port_nothing_provides_is_rejected() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("posterior_kl", Port.T_GIVEN_XY),
                trainable=("encoder",),
            ),
        )
    )
    with pytest.raises(CompileError, match="which no component provides"):
        compile(recipe)


def test_an_objective_requiring_an_unrealisable_realisation_is_rejected() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                objective(
                    "consistency",
                    Port.T_GIVEN_X,
                    realisation=Realisation(view="strong_x"),
                ),
                trainable=("encoder",),
            ),
        )
    )
    with pytest.raises(CompileError, match="cannot produce"):
        compile(recipe)


def test_a_teacher_realisation_is_rejected_until_a_stage_can_build_one() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                objective(
                    "distillation",
                    Port.T_GIVEN_X,
                    realisation=Realisation(params="teacher"),
                ),
                trainable=("encoder",),
            ),
        )
    )
    with pytest.raises(CompileError, match="teacher parameter set"):
        compile(recipe)


def test_an_unknown_trainable_name_is_rejected() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(objective("outcome_nll", Port.Y_GIVEN_XT), trainable=("decoder",)),
        )
    )
    with pytest.raises(CompileError, match="not a component of this graph"):
        compile(recipe)


def test_a_trainable_component_no_objective_depends_on_is_rejected() -> None:
    # The dead-weight stage: `propensity` would receive no gradient, and the
    # recipe would not do what it appears to.
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("outcome_nll", Port.Y_GIVEN_XT),
                trainable=("encoder", "outcome_head", "propensity"),
            ),
        )
    )
    with pytest.raises(CompileError, match="no active objective depends on"):
        compile(recipe)


def test_a_stage_that_trains_nothing_at_all_is_rejected() -> None:
    # The other extreme of the dead-trainable rule: with an empty `trainable`
    # the optimiser gets no parameter group, so every step is a no-op. Only
    # gradient stages exist, so this is knowable before any data.
    recipe = two_head_recipe(
        program=(_stage(objective("outcome_nll", Port.Y_GIVEN_XT)),)
    )
    with pytest.raises(CompileError, match="empty `trainable`"):
        compile(recipe)


def test_a_row_pairing_empty_by_construction_is_rejected() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("marginal_nll", Port.Y_GIVEN_XT, rows="t_missing"),
                rows="t_observed",
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    with pytest.raises(CompileError, match="empty by construction"):
        compile(recipe)


def test_a_stage_with_no_objectives_is_rejected() -> None:
    with pytest.raises(CompileError, match="would train nothing"):
        compile(two_head_recipe(program=(Stage(name="fit"),)))


# ---------------------------------------------------------------------------
# How a stage descends (DESIGN.md §7, §9.1 — P4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["optimiser", "steps"])
def test_a_stage_that_descends_says_how(field: str) -> None:
    # Both bind card keys (`FIDELITY.md` §2, the `optimisation` block), so
    # neither may fall through to a framework default: a stage that ran 1,000
    # steps at whatever lr the framework liked would print a plan a reviewer
    # could not diff against a paper.
    with pytest.raises(CompileError, match="no usable default"):
        Stage(
            name="fit",
            objectives=(objective("outcome_nll", Port.Y_GIVEN_XT),),
            trainable=("encoder", "outcome_head"),
            **{field: REQUIRED},
        )


def test_a_stage_with_no_objectives_needs_neither() -> None:
    # It is already rejected for having no objectives, and replacing that
    # message with one about an optimiser would be a worse error, not an
    # earlier one.
    assert Stage(name="fit").steps is REQUIRED


@pytest.mark.parametrize("steps", [0, -1, 2.5])
def test_a_stage_runs_for_a_positive_number_of_steps(steps: object) -> None:
    with pytest.raises(CompileError, match="optimiser steps"):
        _stage(
            objective("outcome_nll", Port.Y_GIVEN_XT),
            trainable=("encoder", "outcome_head"),
            steps=steps,
        )


def test_a_stage_holds_an_optimiser_spec() -> None:
    with pytest.raises(CompileError, match="OptimiserSpec"):
        _stage(
            objective("outcome_nll", Port.Y_GIVEN_XT),
            trainable=("encoder", "outcome_head"),
            optimiser="adam",
        )


def test_the_optimiser_reaches_the_compiled_stage() -> None:
    run = compile(two_head_recipe())
    assert run.stage("fit").optimiser is run.recipe.program[0].optimiser
    assert run.stage("fit").steps == STEPS


def test_two_objectives_sharing_a_name_are_rejected() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("nll", Port.Y_GIVEN_XT),
                objective("nll", Port.T_GIVEN_X),
                trainable=("encoder", "outcome_head", "propensity"),
            ),
        )
    )
    with pytest.raises(CompileError, match="more than one objective called"):
        compile(recipe)


# ---------------------------------------------------------------------------
# The declarative surface validates itself
# ---------------------------------------------------------------------------


def test_a_recipe_needs_a_card() -> None:
    with pytest.raises(CompileError, match="names no card"):
        two_head_recipe(card="")


def test_a_recipe_needs_a_program() -> None:
    with pytest.raises(CompileError, match="empty program"):
        two_head_recipe(program=())


def test_two_stages_sharing_a_name_are_rejected() -> None:
    stage = _stage(objective("outcome_nll", Port.Y_GIVEN_XT), trainable=("encoder",))
    with pytest.raises(CompileError, match="more than one stage called"):
        two_head_recipe(program=(stage, stage))


def test_a_purpose_outside_the_two_is_rejected() -> None:
    with pytest.raises(CompileError, match="'causal' or 'predictive'"):
        two_head_recipe(purpose="exploratory")


def test_a_stage_rejects_an_unknown_row_population() -> None:
    with pytest.raises(CompileError, match="stage 'fit': unknown row population"):
        Stage(name="fit", rows="t_known")  # type: ignore[arg-type]


def test_an_objective_with_an_unknown_row_population_names_where_it_is() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("outcome_nll", Port.Y_GIVEN_XT, rows="t_known"),  # type: ignore[arg-type]
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    with pytest.raises(CompileError, match="objective 'outcome_nll' in stage 'fit'"):
        compile(recipe)


def test_looking_up_a_stage_that_does_not_exist_says_what_does() -> None:
    run = compile(two_head_recipe())
    with pytest.raises(CompileError, match=r"\['fit'\]"):
        run.stage("refit")


# ---------------------------------------------------------------------------
# plan.hyperparameters — the flat dict the card cross-check compares against
# ---------------------------------------------------------------------------


def test_the_plan_carries_the_resolved_hyperparameters() -> None:
    plan = compile(two_head_recipe()).plan
    assert plan.hyperparameters == {
        "architecture.widths_depths": HIDDEN,
        # The optimisation block comes from the stage and its optimiser: five
        # of the six card keys are rendered rather than structured, because
        # this dict is what a reviewer diffs against a card's §4 YAML.
        "gradients.gradient_clipping": "none",
        "optimisation.lr": 0.05,
        "optimisation.lr_schedule": "constant 1.0",
        "optimisation.optimiser": "sgd(momentum=0.0, nesterov=False)",
        "optimisation.total_steps_or_epochs": STEPS,
        "optimisation.weight_decay": "none",
        "losses.eligible_rows": {
            "fit.outcome_nll": "y_observed",
            "fit.treatment_nll": "t_observed",
        },
        "losses.reduction": {"fit.outcome_nll": "mean", "fit.treatment_nll": "mean"},
        "losses.schedules": {
            "fit.outcome_nll": "constant 1.0",
            "fit.treatment_nll": "constant 1.0",
        },
        "losses.weights": {"fit.outcome_nll": 1.0, "fit.treatment_nll": 1.0},
    }


@dataclass(frozen=True)
class _WidthBindingObjective:
    """An objective that binds the same card key the encoder does."""

    name: str
    requires: frozenset[tuple[Port, Realisation]]
    width: int
    rows: Rows = "all"
    CARD_KEYS: ClassVar[Mapping[str, str]] = {"width": "architecture.widths_depths"}

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset()

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del state, ctx
        return reduce_rows(torch.zeros(batch.batch_size), rows)


def test_a_hyperparameter_conflict_between_two_owners_is_rejected() -> None:
    # A canonical key names one number. Two owners disagreeing about it would
    # make the card cross-check meaningless in the only case where it matters.
    recipe = two_head_recipe(
        program=(
            _stage(
                weighted(
                    _WidthBindingObjective(
                        name="outcome_nll",
                        requires=frozenset({(Port.Y_GIVEN_XT, DEFAULT)}),
                        width=HIDDEN + 3,
                    )
                ),
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    with pytest.raises(CompileError, match="bound twice with different values"):
        compile(recipe)


def test_two_owners_agreeing_on_a_card_key_is_not_a_conflict() -> None:
    recipe = two_head_recipe(
        program=(
            _stage(
                weighted(
                    _WidthBindingObjective(
                        name="outcome_nll",
                        requires=frozenset({(Port.Y_GIVEN_XT, DEFAULT)}),
                        width=HIDDEN,
                    )
                ),
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    assert compile(recipe).plan.hyperparameters["architecture.widths_depths"] == HIDDEN


# ---------------------------------------------------------------------------
# Lineage reaches the plan (DESIGN.md §2.2, §7.2)
# ---------------------------------------------------------------------------


def test_the_plan_records_which_components_reach_the_raw_outcome() -> None:
    graph = ComponentGraph([ToyEncoder(width=HIDDEN), ToyPropensity(), ToyPosterior()])
    recipe = Recipe(
        name="with_posterior",
        schema=make_schema(),
        system=graph,
        program=(
            _stage(
                objective("posterior_kl", Port.T_GIVEN_XY),
                objective("treatment_nll", Port.T_GIVEN_X),
                trainable=("encoder", "propensity", "posterior"),
            ),
        ),
        card="docs/recipes/with_posterior.md",
    )
    planned = {
        component.name: component for component in compile(recipe).plan.components
    }
    assert planned["posterior"].reads_raw_outcome
    assert planned["posterior"].outcome_dependent
    assert not planned["propensity"].outcome_dependent


# ---------------------------------------------------------------------------
# The dead-trainable rule sees through a stop-gradient (DESIGN.md §4, §8.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _DetachingObjective:
    """Requires two ports and backpropagates through only one of them."""

    name: str
    trains: Port
    detached: Port
    rows: Rows = "all"

    @property
    def requires(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.trains, DEFAULT), (self.detached, DEFAULT)})

    @property
    def detaches(self) -> frozenset[tuple[Port, Realisation]]:
        return frozenset({(self.detached, DEFAULT)})

    def compute(
        self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
    ) -> LossTerm:
        del state, ctx
        return reduce_rows(torch.zeros(batch.batch_size), rows)


def _detaching_recipe(*, trainable: tuple[str, ...]) -> Recipe:
    return two_head_recipe(
        program=(
            _stage(
                weighted(
                    _DetachingObjective(
                        name="marginal_nll",
                        trains=Port.T_GIVEN_X,
                        detached=Port.Y_GIVEN_XT,
                    )
                ),
                trainable=trainable,
            ),
        )
    )


def test_a_trainable_reached_only_through_a_stop_gradient_is_rejected() -> None:
    # The hole a stop-gradient opens in the §8.4 check: `outcome_head` runs in
    # the forward pass, so it passes the "no active objective depends on it"
    # test, and still receives no gradient. Without this the stage compiles,
    # trains, and every optimiser step is a no-op.
    with pytest.raises(CompileError, match="through a stop-gradient"):
        compile(_detaching_recipe(trainable=("encoder", "outcome_head")))


def test_a_trainable_the_objective_does_backpropagate_into_is_accepted() -> None:
    run = compile(_detaching_recipe(trainable=("encoder", "propensity")))
    assert run.stage("fit").trainable == ("encoder", "propensity")


def test_the_detached_side_still_runs_in_the_forward_pass() -> None:
    # It is read, just not trained: the pass has to compute it or the term
    # cannot be evaluated at all.
    passes = compile(_detaching_recipe(trainable=("encoder", "propensity")))
    assert "outcome_head" in passes.stage("fit").passes[0].components


def test_an_objective_that_detaches_everything_it_requires_is_rejected() -> None:
    # A term with no gradient path contributes a constant to the total.
    recipe = two_head_recipe(
        program=(
            _stage(
                weighted(
                    _DetachingObjective(
                        name="constant_term",
                        trains=Port.Y_GIVEN_XT,
                        detached=Port.Y_GIVEN_XT,
                    )
                ),
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    with pytest.raises(CompileError, match="detaches every port it requires"):
        compile(recipe)


def test_detaching_a_port_the_objective_does_not_require_is_rejected() -> None:
    @dataclass(frozen=True)
    class _Stray:
        name: str = "stray"
        rows: Rows = "all"

        @property
        def requires(self) -> frozenset[tuple[Port, Realisation]]:
            return frozenset({(Port.Y_GIVEN_XT, DEFAULT)})

        @property
        def detaches(self) -> frozenset[tuple[Port, Realisation]]:
            return frozenset({(Port.T_GIVEN_X, DEFAULT)})

        def compute(
            self, state: State, batch: XTYBatch, rows: RowIndex, ctx: TrainContext
        ) -> LossTerm:
            del state, ctx
            return reduce_rows(torch.zeros(batch.batch_size), rows)

    recipe = two_head_recipe(
        program=(_stage(weighted(_Stray()), trainable=("encoder", "outcome_head")),)
    )
    with pytest.raises(CompileError, match="which it does not require"):
        compile(recipe)


def test_the_plan_prints_the_stop_gradient() -> None:
    # Which side a paper detaches is a card §4 field, so the review artifact
    # has to say it rather than leave it implied by `requires`.
    plan = compile(_detaching_recipe(trainable=("encoder", "propensity"))).plan
    assert "      detaches  p(y|x,t) @ view=identity params=student" in plan.render()


def test_an_objective_that_detaches_nothing_prints_no_detaches_line() -> None:
    assert "detaches" not in compile(two_head_recipe()).plan.render()


# ---------------------------------------------------------------------------
# The four per-objective `losses.*` keys (FIDELITY.md §2, DESIGN.md §9.1)
# ---------------------------------------------------------------------------


def test_the_weighted_settings_reach_plan_hyperparameters() -> None:
    # The card cross-check reads `plan.hyperparameters`, not the stage text, so
    # a recipe that supplies a weight and a reduction has to appear there or
    # the check reports its own mechanism rather than the recipe.
    recipe = two_head_recipe(
        program=(
            _stage(
                objective(
                    "outcome_nll",
                    Port.Y_GIVEN_XT,
                    rows="y_observed",
                    weight=Ramp(0.0, 0.5, steps=5_000),
                    reduction="sum",
                ),
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    hyperparameters = compile(recipe).plan.hyperparameters
    assert hyperparameters["losses.weights"] == {"fit.outcome_nll": 0.5}
    assert hyperparameters["losses.schedules"] == {
        "fit.outcome_nll": "ramp 0.0 -> 0.5 over 5000 steps"
    }
    assert hyperparameters["losses.reduction"] == {"fit.outcome_nll": "sum"}
    assert hyperparameters["losses.eligible_rows"] == {"fit.outcome_nll": "y_observed"}


def test_the_weight_key_carries_the_number_the_paper_states() -> None:
    # A paper states "λ = 0.5, ramped over the first 5,000 steps" as two facts.
    # `losses.weights` is the λ; `losses.schedules` is the ramp.
    recipe = two_head_recipe(
        program=(
            _stage(
                objective(
                    "outcome_nll",
                    Port.Y_GIVEN_XT,
                    weight=Ramp(0.0, 0.5, steps=5_000),
                ),
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    assert compile(recipe).plan.hyperparameters["losses.weights"] == {
        "fit.outcome_nll": 0.5
    }


def test_the_eligible_rows_key_reports_the_intersected_set() -> None:
    # What the term is actually given (DESIGN.md §7.0), not the objective's
    # half of it: a card describing only the objective's population would not
    # describe what runs.
    recipe = two_head_recipe(
        program=(
            _stage(
                objective("outcome_nll", Port.Y_GIVEN_XT, rows="y_observed"),
                rows="t_observed",
                trainable=("encoder", "outcome_head"),
            ),
        )
    )
    assert compile(recipe).plan.hyperparameters["losses.eligible_rows"] == {
        "fit.outcome_nll": "t_observed & y_observed"
    }
