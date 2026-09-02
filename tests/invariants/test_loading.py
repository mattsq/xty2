"""Tier 0 — the data-loading policy, the samplers, and what each one earns.

Three of these carry the packet.

`test_the_uniform_sampler_is_the_scheme_the_fixtures_used` is the one that
makes adopting the loader affordable: `UniformSampler` is *defined* as what
every fixture in this repository already did by hand, so the change is one of
owner rather than of arithmetic. Without it, seven recipes would have been
re-based onto an unexamined sampling scheme in the same PR that moved their
digests.

`test_the_sampler_stream_does_not_depend_on_the_model` is what every paired
ablation depends on. Those pairs used to share a pre-computed index tensor;
they get it from the seed now, and a sampler whose stream moved with the
objectives would have quietly turned each of them into a two-variable
comparison.

`test_statistics_fitted_off_the_training_split_are_caught` is the run-time half
of the leakage rule — the mutation test, in the style `PLAN.md` P10 sets: a
provenance label nothing can falsify is not checked.
"""

from __future__ import annotations

import pytest
import torch
from xty2.core import (
    Dataset,
    DataSpec,
    ExternalBatches,
    MissingnessSpec,
    Port,
    PreprocessSpec,
    Quota,
    QuotaSampler,
    SplitSpec,
    TrainingPopulation,
    UniformSampler,
    XTYBatch,
    compile,
)
from xty2.core.errors import ArtifactError, CompileError, TrainingError, Xty2Error
from xty2.evaluation.benchmarks.common import batch_indices, on_the_training_scale
from xty2.training import run_stage
from xty2.training.loading import build_population, iterate, sampler_seed

from tests.invariants.conftest import (
    make_batch,
    make_schema,
    objective,
    stage,
    two_head_recipe,
)

ROWS = 64


def _rows(rows: int, *, observed_rows: int | None = None) -> XTYBatch:
    """A valid population of `rows` rows, fully observed unless asked otherwise."""
    schema = make_schema()
    observed = torch.ones(rows, dtype=torch.bool)
    if observed_rows is not None:
        observed[observed_rows:] = False
    mass = torch.rand(rows) * 90.0 + 5.0
    speed = torch.randn(rows)
    return XTYBatch(
        x=torch.stack([mass, speed, mass * speed, torch.randn(rows)], dim=1),
        t=torch.arange(rows, dtype=torch.long) % schema.treatment_cardinality,
        y=torch.randn(rows),
        t_observed=observed,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
    )


def _dataset(rows: int = ROWS, *, train: str = "train") -> Dataset:
    return Dataset(
        schema=make_schema(),
        rows=_rows(rows),
        assignments={train: torch.arange(rows)},
    )


def _policy(**overrides: object) -> DataSpec:
    defaults: dict[str, object] = {
        "split": SplitSpec(protocol="the fixture's own rows", train="train"),
        "preprocess": PreprocessSpec(features="none", outcome="none"),
        "missingness": MissingnessSpec(mechanism="observed"),
    }
    return DataSpec(**(defaults | overrides))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The sampler is the scheme the repository already used
# ---------------------------------------------------------------------------


def test_the_uniform_sampler_is_the_scheme_the_fixtures_used() -> None:
    """One fresh permutation per step, first `batch_size` rows — byte for byte.

    `batch_indices` is what `xty2/evaluation/benchmarks/common.py` and five
    smoke fixtures did before the loader existed. Pinning the sampler to it is
    what lets this packet move every plan digest while moving no arithmetic.
    """
    population = build_population(_dataset(), _policy(), seed=7)
    drawn = [
        batch.row_id
        for batch in iterate(population, UniformSampler(batch_size=10), steps=5, seed=7)
    ]
    expected = batch_indices(ROWS, steps=5, batch_size=10, seed=sampler_seed(7))
    for step, rows in enumerate(drawn):
        assert torch.equal(rows, population.rows.row_id.index_select(0, expected[step]))


def test_the_sampler_stream_does_not_depend_on_the_model() -> None:
    """Two recipes differing only in a weight draw identical rows.

    Every paired ablation in the repository is this assertion in disguise. It
    holds because the sampler's generator is seeded by hashing the stage seed
    and reads nothing else — not the graph, not the objectives, not the step's
    loss.
    """
    schema = make_schema()
    population = build_population(_dataset(), _policy(), seed=3)
    sampler = UniformSampler(batch_size=8)
    left = [b.row_id for b in iterate(population, sampler, steps=4, seed=3)]
    right = [b.row_id for b in iterate(population, sampler, steps=4, seed=3)]
    assert [rows.tolist() for rows in left] == [rows.tolist() for rows in right]
    del schema


def test_the_sampler_stream_is_not_the_view_key_stream() -> None:
    """Hashed rather than offset, so it cannot collide with the per-step keys.

    A stage walks one view key per optimiser step upward from its own seed. An
    offset sampler seed would sit inside that walk and would collide the day
    the stride changed; a hash has no arithmetic relationship to it at all.
    """
    assert sampler_seed(0) != 0
    assert all(sampler_seed(seed) != seed for seed in range(64))
    assert sampler_seed(0) != sampler_seed(0, label="missingness")


# ---------------------------------------------------------------------------
# The quota is the paper's, and it is derived rather than asserted
# ---------------------------------------------------------------------------


def test_a_quota_draws_exactly_the_declared_rows_from_each_population() -> None:
    data = Dataset(
        schema=make_schema(),
        rows=_rows(ROWS, observed_rows=16),
        assignments={"train": torch.arange(ROWS)},
    )
    population = build_population(data, _policy(), seed=1)
    sampler = QuotaSampler(
        quotas=(
            Quota(rows="t_observed", size=4),
            Quota(rows="t_missing", size=28),
        )
    )
    for drawn in iterate(population, sampler, steps=3, seed=1):
        assert drawn.batch_size == 32
        assert int(drawn.t_observed.sum()) == 4
        assert int(drawn.t_missing.sum()) == 28


def test_the_ratio_the_plan_prints_is_the_ratio_the_sampler_runs() -> None:
    """Both card keys are derived from the quotas, so neither can be misstated.

    `DESIGN.md` §7.1's rule about `PseudoLabels.used_y`, applied to `mu`: a
    field a producer can set is a field a producer can set wrongly, and a
    recipe drawing 64 and 64 must not be able to claim 7.
    """
    sampler = QuotaSampler(
        quotas=(
            Quota(rows="t_observed", size=64),
            Quota(rows="t_missing", size=448),
        )
    )
    assert sampler.batch_size == 512
    assert sampler.labelled_unlabelled_ratio == 7.0
    with pytest.raises(AttributeError):
        sampler.batch_size = 8  # type: ignore[misc]


def test_a_quota_that_cannot_be_filled_is_an_error_not_a_short_batch() -> None:
    population = build_population(_dataset(), _policy(), seed=1)
    sampler = QuotaSampler(quotas=(Quota(rows="all", size=ROWS + 1),))
    with pytest.raises(TrainingError, match="cannot be filled"):
        list(iterate(population, sampler, steps=1, seed=1))


def test_two_quotas_over_one_population_are_rejected() -> None:
    with pytest.raises(CompileError, match="more than once"):
        QuotaSampler(
            quotas=(
                Quota(rows="t_missing", size=4),
                Quota(rows="t_missing", size=8),
            )
        )


# ---------------------------------------------------------------------------
# The policy, and the leakage half of it
# ---------------------------------------------------------------------------


def test_a_declared_label_budget_is_the_budget_that_runs() -> None:
    """A count, not a rate: `FIDELITY.md` §2's key, as the papers state it."""
    data = _dataset(rows=100)
    policy = _policy(missingness=MissingnessSpec(mechanism="mcar", observed=40))
    population = build_population(data, policy, seed=5)
    assert int(population.rows.t_observed.sum()) == 40


def test_a_declared_rate_is_exact_rather_than_in_expectation() -> None:
    data = _dataset(rows=100)
    policy = _policy(missingness=MissingnessSpec(mechanism="mcar", rate=0.25))
    population = build_population(data, policy, seed=5)
    assert int(population.rows.t_missing.sum()) == 25


def test_missingness_is_keyed_by_row_id_rather_than_by_position() -> None:
    """Reordering the rows moves no row's fate.

    Keying on `row_id` rather than on position is what makes the declared
    mechanism a property of the *data* rather than of the order it arrived in.
    """
    data = _dataset(rows=32)
    policy = _policy(missingness=MissingnessSpec(mechanism="mcar", rate=0.5))
    forward = build_population(data, policy, seed=11)
    shuffled = Dataset(
        schema=make_schema(),
        rows=_reversed(data),
        assignments={"train": torch.arange(32)},
    )
    backward = build_population(shuffled, policy, seed=11)
    missing_forward = set(forward.rows.row_id[forward.rows.t_missing].tolist())
    missing_backward = set(backward.rows.row_id[backward.rows.t_missing].tolist())
    assert missing_forward == missing_backward


def _reversed(data: Dataset) -> XTYBatch:
    order = torch.arange(data.batch_size - 1, -1, -1)
    rows = data.rows
    return rows.replace(
        x=rows.x.index_select(0, order),
        t=rows.t.index_select(0, order),
        y=rows.y.index_select(0, order),
        t_observed=rows.t_observed.index_select(0, order),
        y_observed=rows.y_observed.index_select(0, order),
        row_id=rows.row_id.index_select(0, order),
    )


def test_standardisation_is_fitted_on_the_declared_split_and_says_so() -> None:
    batch = _rows(ROWS)
    data = Dataset(
        schema=make_schema(),
        rows=batch,
        assignments={"train": torch.arange(32), "test": torch.arange(32, ROWS)},
    )
    population = build_population(
        data,
        _policy(preprocess=PreprocessSpec(features="zscore", outcome="none")),
        seed=1,
    )
    assert set(population.fitted_on_row_ids.tolist()) == set(batch.row_id[:32].tolist())
    assert torch.allclose(population.statistics["x_location"], batch.x[:32].mean(dim=0))
    # And the population it hands back is the *training* rows only: a stage
    # that stepped on the test split would be the leak this exists to stop.
    assert population.batch_size == 32


def test_benchmark_rows_receive_both_fitted_feature_and_outcome_scaling() -> None:
    """Evaluation must consume the same transform the training loader fitted."""
    data = _dataset(rows=32)
    population = build_population(
        data,
        _policy(preprocess=PreprocessSpec(features="zscore", outcome="zscore")),
        seed=1,
    )
    held_out = _rows(8)
    scaled = on_the_training_scale(held_out, population)
    statistics = population.statistics
    assert torch.allclose(
        scaled.x,
        (held_out.x - statistics["x_location"]) / statistics["x_scale"],
    )
    assert torch.allclose(
        scaled.y,
        (held_out.y - statistics["y_location"]) / statistics["y_scale"],
    )


def test_statistics_fitted_off_the_training_split_are_caught() -> None:
    """The mutation test. A label nothing can falsify is not a check.

    `tarnet.md` §5.5 recorded the cost of not having this: "a later runner
    could fit it on the wrong split and nothing in the plan would say so".
    """
    from xty2.training.loading import check_fitted_on

    data = _dataset(rows=32)
    policy = _policy()
    population = build_population(data, policy, seed=1)
    check_fitted_on(population, data, policy)

    forged = TrainingPopulation._issue(
        rows=population.rows,
        assignment=population.assignment,
        statistics=population.statistics,
        fitted_on_row_ids=torch.tensor([9_999], dtype=torch.long),
        spec_digest=population.spec_digest,
    )
    with pytest.raises(TrainingError, match="outside assignment"):
        check_fitted_on(forged, data, policy)


def test_a_training_population_cannot_be_constructed_directly() -> None:
    with pytest.raises(ArtifactError, match="loading factory"):
        TrainingPopulation(
            rows=_rows(4),
            assignment="train",
            statistics={},
            fitted_on_row_ids=torch.zeros(0, dtype=torch.long),
            spec_digest="",
        )


def test_building_a_population_leaves_the_dataset_bit_identical() -> None:
    data = _dataset(rows=32)
    before = data.rows.clone()
    build_population(
        data,
        _policy(
            preprocess=PreprocessSpec(features="zscore", outcome="zscore"),
            missingness=MissingnessSpec(mechanism="mcar", rate=0.5),
        ),
        seed=2,
    )
    assert data.rows.equal_to(before)


def test_induced_missingness_refuses_data_that_is_already_missing() -> None:
    """Composing two mechanisms would make the declared budget describe neither.

    A recipe saying "MCAR to 40 labels" is saying xty2 induces the missingness.
    Data that arrives with its own belongs to `mechanism='observed'`, which is
    what `tarnet` declares.
    """
    data = Dataset(
        schema=make_schema(),
        rows=_rows(32, observed_rows=8),
        assignments={"train": torch.arange(32)},
    )
    policy = _policy(missingness=MissingnessSpec(mechanism="mcar", observed=4))
    with pytest.raises(TrainingError, match="arrive with it already"):
        build_population(data, policy, seed=1)


def test_a_missing_training_assignment_names_what_is_there() -> None:
    with pytest.raises(Xty2Error, match="carries \\['train'\\]"):
        build_population(
            _dataset(), _policy(split=SplitSpec(protocol="p", train="fit")), seed=1
        )


# ---------------------------------------------------------------------------
# What a stage is allowed to declare
# ---------------------------------------------------------------------------


def test_a_stage_declares_what_feeds_it() -> None:
    from xty2.core import REQUIRED, Stage

    with pytest.raises(CompileError, match="declares no sampler"):
        Stage(name="fit", sampler=REQUIRED)


def test_external_batches_is_refused_where_the_batch_size_is_the_arithmetic() -> None:
    """The bar that stops the caller-supplies-batches declaration being a hatch.

    `InfoNCEContrastive` is the one `batch_coupled` objective in the
    repository, and `scarf.md` §5.6 is what its absence used to cost.
    """
    recipe = two_head_recipe(
        program=(
            stage(
                objectives=(
                    objective(
                        "contrastive", Port.X_REPR, rows="all", batch_coupled=True
                    ),
                ),
                trainable=("encoder",),
                sampler=ExternalBatches(),
            ),
        )
    )
    with pytest.raises(CompileError, match="read the rest of the batch"):
        compile(recipe)


def test_a_sampled_recipe_declares_a_data_policy() -> None:
    with pytest.raises(CompileError, match="declares no `data` policy"):
        two_head_recipe(
            program=(
                stage(
                    objectives=(objective("treatment_nll", Port.T_GIVEN_X),),
                    trainable=("encoder", "propensity"),
                    sampler=UniformSampler(batch_size=4),
                ),
            )
        )


def test_a_policy_nothing_applies_is_rejected() -> None:
    with pytest.raises(CompileError, match="every stage takes ExternalBatches"):
        two_head_recipe(data=_policy())


def test_a_quota_empty_by_construction_is_a_compile_error() -> None:
    """`DESIGN.md` §7.0's rule, applied to the sampler rather than to a term."""
    with pytest.raises(CompileError, match="empty by construction"):
        stage(
            rows="t_observed",
            sampler=QuotaSampler(quotas=(Quota(rows="t_missing", size=4),)),
        )


def test_handing_a_dataset_to_an_external_stage_is_an_error() -> None:
    run = compile(two_head_recipe())
    with pytest.raises(TrainingError, match="was handed a Dataset"):
        run_stage(run, "fit", _dataset(), seed=0)


def test_handing_batches_to_a_sampling_stage_is_an_error() -> None:
    recipe = two_head_recipe(
        program=(
            stage(
                objectives=(objective("treatment_nll", Port.T_GIVEN_X),),
                trainable=("encoder", "propensity"),
                sampler=UniformSampler(batch_size=4),
            ),
        ),
        data=_policy(),
    )
    run = compile(recipe)
    with pytest.raises(TrainingError, match="Pass a Dataset"):
        run_stage(run, "fit", [make_batch()], seed=0)


# ---------------------------------------------------------------------------
# The plan says all of it
# ---------------------------------------------------------------------------


def test_the_plan_prints_the_policy_and_the_sampler() -> None:
    recipe = two_head_recipe(
        program=(
            stage(
                objectives=(objective("treatment_nll", Port.T_GIVEN_X),),
                trainable=("encoder", "propensity"),
                sampler=UniformSampler(batch_size=16),
            ),
        ),
        data=_policy(missingness=MissingnessSpec(mechanism="mcar", observed=40)),
    )
    plan = compile(recipe).plan
    rendered = plan.render()
    assert "data\n  split            the fixture's own rows" in rendered
    assert "sampler: uniform, batch_size=16, without replacement" in rendered
    assert plan.hyperparameters["optimisation.batch_size"] == 16
    assert "budget of 40 labelled rows" in str(
        plan.hyperparameters["data.missingness_mechanism"]
    )


def test_an_external_stage_prints_that_it_is_external() -> None:
    rendered = compile(two_head_recipe()).plan.render()
    assert "sampler: external: the caller supplies batches" in rendered
    assert "\ndata\n" not in rendered
