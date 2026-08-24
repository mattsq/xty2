"""Tier 0 — the `tarnet` recipe compiles to the plan its card describes.

`PLAN.md` P5 makes this the packet where the card-to-plan diff is exercised
for the first time, so the plan is snapshotted rather than probed with
substring assertions: it is the review surface (`FIDELITY.md` §1.2), and a
change to it should show up in a diff as a change to the review surface.

Read the snapshot below against `docs/recipes/tarnet.md` §3.2 and §4. Every
component and objective in the mapping table is in the plan, and every §4 key
not marked `n/a` is in the hyperparameter block — which is what
`test_cards.py` asserts mechanically.
"""

import pytest
import torch
from xty2.components import MLPArchitecture
from xty2.core import REQUIRED, CompileError, Port, Ramp, compile
from xty2.recipes import tarnet
from xty2.recipes.tarnet import ARCHITECTURE, MARGINAL_WEIGHT, STEPS
from xty2.training import run_stage

from tests.invariants.conftest import make_batch, make_schema

TARNET_PLAN = """\
recipe: tarnet
purpose: causal
card: docs/recipes/tarnet.md
schema: D = 4, K = 3

components (topological order)
  mlp_encoder             [x] -> [x_repr]
  tarnet_head             [x_repr] -> [p(y|x,t)]
  categorical_propensity  [x_repr] -> [p(t|x)]

data lineage
  mlp_encoder             x <- source
  tarnet_head             x_repr <- mlp_encoder
  categorical_propensity  x_repr <- mlp_encoder
  reads raw y directly:  none
  depends on raw y:      none

stage fit
  rows: all
  steps: 3000
  optimisation
    optimiser     adam(betas=(0.9, 0.999), eps=1e-08)
    lr            0.001
    lr schedule   constant 1.0
    weight decay  0.0001 (norm and bias exempt)
    clipping      none
  forward passes (1)
    view=identity params=student: mlp_encoder -> tarnet_head -> categorical_propensity
  objectives
    observed_outcome_nll            rows t_observed  reduction mean
      weight    constant 1.0
      requires  p(y|x,t) @ view=identity params=student
    observed_treatment_nll          rows t_observed  reduction mean
      weight    constant 1.0
      requires  p(t|x) @ view=identity params=student
    missing_treatment_marginal_nll  rows t_missing   reduction mean
      weight    ramp 0.0 -> 1.0 over 1000 steps
      requires  p(t|x) @ view=identity params=student, p(y|x,t) @ view=identity params=student
  trainable
    mlp_encoder, tarnet_head, categorical_propensity

hyperparameters
  architecture.activation              = 'elu'
  architecture.dropout                 = 0.0
  architecture.initialisation          = 'xavier_normal'
  architecture.normalisation           = 'none'
  architecture.output_parameterisation = 'gaussian per-arm mean, unit scale'
  architecture.widths_depths           = 'representation 3x200, heads 3x100'
  gradients.gradient_clipping          = 'none'
  gradients.marginal_nll_grad_path     = 'both'
  gradients.stop_gradients
    fit.missing_treatment_marginal_nll = 'none'
    fit.observed_outcome_nll           = 'none'
    fit.observed_treatment_nll         = 'none'
  losses.eligible_rows
    fit.missing_treatment_marginal_nll = 't_missing'
    fit.observed_outcome_nll           = 't_observed'
    fit.observed_treatment_nll         = 't_observed'
  losses.reduction
    fit.missing_treatment_marginal_nll = 'mean'
    fit.observed_outcome_nll           = 'mean'
    fit.observed_treatment_nll         = 'mean'
  losses.schedules
    fit.missing_treatment_marginal_nll = 'ramp 0.0 -> 1.0 over 1000 steps'
    fit.observed_outcome_nll           = 'constant 1.0'
    fit.observed_treatment_nll         = 'constant 1.0'
  losses.weights
    fit.missing_treatment_marginal_nll = 1.0
    fit.observed_outcome_nll           = 1.0
    fit.observed_treatment_nll         = 1.0
  optimisation.lr                      = 0.001
  optimisation.lr_schedule             = 'constant 1.0'
  optimisation.optimiser               = 'adam(betas=(0.9, 0.999), eps=1e-08)'
  optimisation.total_steps_or_epochs   = 3000
  optimisation.weight_decay            = '0.0001 (norm and bias exempt)'
"""


def test_the_recipe_compiles_to_the_plan_the_card_is_diffed_against() -> None:
    assert compile(tarnet(make_schema())).plan.render() == TARNET_PLAN


def test_the_plan_is_stable_across_compilations() -> None:
    schema = make_schema()
    first = compile(tarnet(schema)).plan
    second = compile(tarnet(schema)).plan
    assert first.render() == second.render()
    assert first.digest == second.digest


def test_one_stage_one_forward_pass() -> None:
    # Three objectives over two ports, and the compiler plans a single
    # evaluation of the graph for all of them (DESIGN.md §8.3). If this ever
    # became three, the recipe would be paying for the realisation machinery
    # without using it.
    stage = compile(tarnet(make_schema())).stage("fit")
    assert len(stage.passes) == 1
    assert stage.passes[0].components == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )


def test_every_component_the_card_names_is_trainable() -> None:
    stage = compile(tarnet(make_schema())).stage("fit")
    assert set(stage.trainable) == set(compile(tarnet(make_schema())).graph.names)


def test_the_marginal_term_ramps_rather_than_starting_at_full_weight() -> None:
    # Card §7: the term is meaningless while both heads are at initialisation.
    # A constant weight here would be a silent departure from the card, and
    # the plan line is the only place a reviewer would see it.
    assert isinstance(MARGINAL_WEIGHT, Ramp)
    assert MARGINAL_WEIGHT(0) == 0.0
    assert MARGINAL_WEIGHT(MARGINAL_WEIGHT.steps) == 1.0


def test_nothing_in_the_recipe_reads_the_raw_outcome() -> None:
    # The leakage rule (DESIGN.md §7.2) reasons about Y_RAW reachability, and
    # for this recipe the answer is "nothing" — the outcome enters only
    # through the objectives, which are handed the batch.
    graph = compile(tarnet(make_schema())).graph
    assert graph.outcome_dependent() == ()
    assert all(Port.Y_RAW not in graph[name].requires for name in graph.names)


def test_the_recipe_runs_a_step_and_moves_every_component() -> None:
    # Not a fit — that is Tier 1. This is the wiring assertion Tier 0 can
    # afford: three objectives, one optimiser, and no component left behind by
    # the gradient. A head that received none would be a dead-weight stage the
    # compiler happened not to catch.
    schema = make_schema()
    run = compile(tarnet(schema, steps=2, architecture=_small()))
    before = {
        f"{component}.{name}": parameter.detach().clone()
        for component in run.graph.names
        for name, parameter in run.graph[component].named_parameters()
    }
    result = run_stage(run, "fit", (make_batch() for _ in range(2)), seed=0)
    assert result.steps == 2
    moved = {
        component
        for component in run.graph.names
        for name, parameter in run.graph[component].named_parameters()
        if not torch.equal(parameter, before[f"{component}.{name}"])
    }
    assert moved == set(run.graph.names)


def test_the_same_seed_gives_the_same_trace() -> None:
    schema = make_schema()
    traces = []
    for _ in range(2):
        # Seeded before the recipe is built, not only before the loop: the
        # initialisation draws from the same global RNG the run does, so two
        # graphs built from one seed are the same graph.
        torch.manual_seed(4)
        run = compile(tarnet(schema, steps=3, architecture=_small()))
        traces.append(
            run_stage(run, "fit", (make_batch() for _ in range(3)), seed=11).trace
        )
    assert traces[0] == traces[1]


def test_a_recipe_that_unsets_a_card_bound_field_cannot_be_constructed() -> None:
    # `steps` is REQUIRED on the stage and binds
    # `optimisation.total_steps_or_epochs`, so a caller who clears it gets a
    # construction error rather than a framework default. Asserted from the
    # recipe's side, because that is the side a sweep overrides from.
    with pytest.raises(CompileError, match="no usable default"):
        tarnet(make_schema(), steps=REQUIRED)


def _small() -> MLPArchitecture:
    """The card architecture at Tier 0 sizes. Tier 0 is seconds, not a fit."""
    return MLPArchitecture(
        representation=(6, 5),
        head=(4,),
        activation=ARCHITECTURE.activation,
        normalisation=ARCHITECTURE.normalisation,
        dropout=ARCHITECTURE.dropout,
        initialisation=ARCHITECTURE.initialisation,
    )


def test_the_default_step_count_is_the_card_value() -> None:
    assert STEPS == 3000
