"""Tier 0 — the Barlow Twins recipe, its projector and its two correlation terms.

The load-bearing tests are `test_each_term_is_the_author_forward` and
`test_each_terms_input_gradients_are_the_author_forwards`: like the
exact-marginalisation invariant (`FIDELITY.md` §3), they compare the vectorised
implementation against a loop transcription of
`docs/recipes/barlow_twins.md` §3.1 in float64, on the §6.3 fixture — unequal
column variances, nonzero column means, an asymmetric cross-correlation, and
one near-constant column whose variance sits below the pinned epsilon. Each of
those conditions removes one way a wrong reduction can cancel: a symmetric `C`
hides a summed triangle, centred columns hide a missing centring, and a
fixture whose every column has healthy variance never notices where epsilon
went.

Both halves are needed. The value alone does not distinguish `1/B` from
`1/(B-1)` at large `B`, and a term whose value is right and whose gradient is
scaled wrong trains a different model at the same logged number.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn
from xty2.components import BarlowTwinsProjector
from xty2.components.barlow_twins import (
    BARLOW_TWINS_PROJECTOR_ACTIVATION,
    BARLOW_TWINS_PROJECTOR_INITIALISATION,
    BARLOW_TWINS_PROJECTOR_NORMALISATION,
)
from xty2.core import (
    DEFAULT,
    CategoricalTreatment,
    CompiledRun,
    CompileError,
    Dataset,
    DataSpec,
    ExternalBatches,
    FeatureSpec,
    GraphError,
    LossError,
    MissingnessSpec,
    OutcomeSpec,
    Port,
    PortContractError,
    PortView,
    PreprocessSpec,
    PreservedField,
    Program,
    Schema,
    SplitSpec,
    State,
    TrainContext,
    TrainingPopulation,
    XTYBatch,
    compile,
)
from xty2.objectives import (
    CrossCorrelationDiagonal,
    CrossCorrelationOffDiagonal,
    cross_correlation,
)
from xty2.objectives.barlow_twins import MINIMUM_ROWS
from xty2.recipes import barlow_twins as build_barlow_twins
from xty2.recipes.barlow_twins import (
    BARLOW_TWINS_ENCODER_WIDTHS,
    CORRUPTED_A,
    CORRUPTED_B,
    NORMALISATION_EPSILON,
    POPULATION_CORRECTION,
    PROJECTOR_WIDTHS,
)
from xty2.training import executors, run_program
from xty2.training.loading import build_population
from xty2.views import FeatureCorruption

from tests.invariants.conftest import backward

barlow_twins = partial(
    build_barlow_twins,
    first_transforms=(FeatureCorruption(rate=0.6, columns=None),),
    second_transforms=(FeatureCorruption(rate=0.6, columns=None),),
)

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "barlow_twins.md"
_BENCHMARK = (
    ROOT / "xty2" / "evaluation" / "benchmarks" / "barlow_twins.py"
).read_text(encoding="utf-8")
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "barlow_twins.py"
PRESERVED: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)
ROWS = 5
WIDTH = 3
"""`B = 5`, `d = 3`, card §6.3. Neither equals the other or `K`, so a
transposed statistic or a broadcast over the wrong axis cannot pass."""


def _schema(*, derived: bool = False) -> Schema:
    return Schema(
        features=(
            FeatureSpec("mass", "continuous"),
            FeatureSpec("speed", "continuous"),
            FeatureSpec(
                "momentum",
                "continuous",
                derived_from=("mass", "speed") if derived else (),
            ),
            FeatureSpec("site", "categorical", mutable=False),
        ),
        treatment_cardinality=3,
        outcome=OutcomeSpec(),
    )


def _batch(rows: int = ROWS, offset: int = 0) -> XTYBatch:
    x = torch.arange(float(rows * 4)).reshape(rows, 4) + float(offset * 4)
    observed = torch.arange(rows) % 3 == 0
    return XTYBatch(
        x=x,
        t=torch.arange(rows) % 3,
        y=torch.linspace(-1.0, 1.0, rows),
        t_observed=observed,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(offset, offset + rows),
    )


def _branches() -> tuple[Tensor, Tensor]:
    """Two float64 embeddings meeting every §6.3 fixture condition.

    Built by hand rather than sampled. Column variances differ within and
    between the branches, no column is centred, the resulting `C` is not
    symmetric — so a summed upper triangle is a different number from the
    summed off-diagonal — and the **third column of the first branch is
    near-constant**, with a population variance of `5.76e-07` against the
    pinned `epsilon = 1e-5`. That last column is what makes the position of
    epsilon observable: inside the square root it dominates the denominator and
    attenuates the third row of `C`, and outside it barely moves.
    """
    first = torch.tensor(
        [
            [0.5, 1.25, 0.5010],
            [1.0, 2.0, 0.4995],
            [-0.75, -1.0, 0.5008],
            [0.25, 0.625, 0.4990],
            [1.5, 2.75, 0.5002],
        ],
        dtype=torch.float64,
    )
    second = torch.tensor(
        [
            [0.125, -0.5, 1.0],
            [-0.375, 0.25, 1.25],
            [0.75, 1.5, -0.5],
            [1.125, 0.375, 0.25],
            [-1.0, 0.75, 2.0],
        ],
        dtype=torch.float64,
    )
    return first, second


def _state(first: Tensor, second: Tensor) -> State:
    return State(
        {CORRUPTED_A: {Port.X_PROJ: first}, CORRUPTED_B: {Port.X_PROJ: second}}
    )


def _ctx() -> TrainContext:
    return TrainContext(global_step=0, schema=_schema())


def _all_rows(rows: int = ROWS) -> Tensor:
    return torch.arange(rows)


def _diagonal(**overrides: object) -> CrossCorrelationDiagonal:
    defaults: dict[str, object] = {
        "port": Port.X_PROJ,
        "first": CORRUPTED_A,
        "second": CORRUPTED_B,
        "epsilon": NORMALISATION_EPSILON,
        "correction": POPULATION_CORRECTION,
    }
    return CrossCorrelationDiagonal(**(defaults | overrides))  # type: ignore[arg-type]


def _off_diagonal(**overrides: object) -> CrossCorrelationOffDiagonal:
    defaults: dict[str, object] = {
        "port": Port.X_PROJ,
        "first": CORRUPTED_A,
        "second": CORRUPTED_B,
        "epsilon": NORMALISATION_EPSILON,
        "correction": POPULATION_CORRECTION,
    }
    return CrossCorrelationOffDiagonal(
        **(defaults | overrides)  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# The loop oracle: §3.1, one scalar at a time
# ---------------------------------------------------------------------------


def _oracle_correlation(
    first: Tensor, second: Tensor, epsilon: float, correction: int
) -> list[list[Tensor]]:
    """`C_ij` of §3.1, entry by entry, with no matrix operation anywhere."""
    rows, width = first.shape
    zero = first.new_zeros(())

    def normalised(branch: Tensor) -> list[list[Tensor]]:
        columns: list[list[Tensor]] = []
        for j in range(width):
            column = [branch[b, j] for b in range(rows)]
            mean = sum(column, zero) / rows
            squared = sum(((value - mean) ** 2 for value in column), zero)
            deviation = torch.sqrt(squared / (rows - correction) + epsilon)
            columns.append([(value - mean) / deviation for value in column])
        return columns

    left = normalised(first)
    right = normalised(second)
    return [
        [
            sum((left[i][b] * right[j][b] for b in range(rows)), zero) / rows
            for j in range(width)
        ]
        for i in range(width)
    ]


def _oracle_terms(
    first: Tensor,
    second: Tensor,
    epsilon: float = NORMALISATION_EPSILON,
    correction: int = POPULATION_CORRECTION,
) -> tuple[Tensor, Tensor]:
    """`D` and `O` under the pinned author reductions."""
    correlation = _oracle_correlation(first, second, epsilon, correction)
    width = len(correlation)
    zero = first.new_zeros(())
    diagonal = sum(((1.0 - correlation[i][i]) ** 2 for i in range(width)), zero)
    off = sum(
        (correlation[i][j] ** 2 for i in range(width) for j in range(width) if i != j),
        zero,
    )
    return diagonal, off


def _computed(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor]:
    state = _state(first, second)
    batch = _batch()
    rows = _all_rows()
    return tuple(  # type: ignore[return-value]
        objective.compute(state, batch, rows, _ctx()).value
        for objective in (_diagonal(), _off_diagonal())
    )


# ---------------------------------------------------------------------------
# The two terms against the oracle
# ---------------------------------------------------------------------------


def test_the_fixture_meets_every_section_six_three_condition() -> None:
    """A fixture that quietly degenerates makes every assertion below vacuous."""
    first, second = _branches()
    for branch in (first, second):
        variances = branch.var(dim=0, correction=POPULATION_CORRECTION)
        assert len(set(variances.tolist())) == WIDTH  # unequal
        assert bool((branch.mean(dim=0).abs() > 1e-3).all())  # nonzero means
    # One near-constant column, and only in one branch, so exchanging the two
    # branches is not the identity on the fixture either.
    near_constant = first.var(dim=0, correction=POPULATION_CORRECTION)[2]
    assert float(near_constant) < NORMALISATION_EPSILON / 10.0
    assert float(second.var(dim=0, correction=POPULATION_CORRECTION).min()) > 0.1
    correlation = cross_correlation(
        first, second, epsilon=NORMALISATION_EPSILON, correction=POPULATION_CORRECTION
    )
    assert float((correlation - correlation.transpose(0, 1)).abs().max()) > 0.5


def test_each_term_is_the_author_forward() -> None:
    first, second = _branches()
    for computed, expected in zip(
        _computed(first, second), _oracle_terms(first, second), strict=True
    ):
        assert float(computed) == pytest.approx(float(expected), abs=1e-12)


def test_each_terms_input_gradients_are_the_author_forwards() -> None:
    """A value can be right for a reason the gradient is not (`FIDELITY.md` §3)."""
    for index in range(2):
        first, second = _branches()
        first.requires_grad_(True)
        second.requires_grad_(True)
        backward(_computed(first, second)[index])
        computed = (first.grad, second.grad)

        oracle_first, oracle_second = _branches()
        oracle_first.requires_grad_(True)
        oracle_second.requires_grad_(True)
        backward(_oracle_terms(oracle_first, oracle_second)[index])
        expected = (oracle_first.grad, oracle_second.grad)

        for actual, wanted in zip(computed, expected, strict=True):
            assert actual is not None and wanted is not None
            assert torch.allclose(actual, wanted, atol=1e-10)
            assert float(actual.abs().sum()) > 0.0


def test_both_terms_sum_over_coordinates_without_dividing_by_d() -> None:
    """Eq. (1) sums; §6.4's diagnostics average, and they are not the loss.

    A per-coordinate mean is the same declaration `d` times quieter, which at
    the card's `d = 512` moves both terms by nearly three orders of magnitude
    against the pinned `lambda = 0.0051`.
    """
    first, second = _branches()
    diagonal, off = _computed(first, second)
    oracle_diagonal, oracle_off = _oracle_terms(first, second)
    assert float(diagonal) == pytest.approx(float(oracle_diagonal), abs=1e-12)
    assert float(diagonal) != pytest.approx(float(oracle_diagonal) / WIDTH, abs=1e-6)
    assert float(off) == pytest.approx(float(oracle_off), abs=1e-12)
    assert float(off) != pytest.approx(float(oracle_off) / WIDTH, abs=1e-6)
    assert float(off) != pytest.approx(
        float(oracle_off) / (WIDTH * (WIDTH - 1)), abs=1e-6
    )


def test_the_off_diagonal_charges_every_ordered_pair() -> None:
    """`C` is a cross-view matrix and is not symmetric, so a triangle is half.

    The author's `off_diagonal` helper flattens all but the diagonal, keeping
    `C_ij` and `C_ji` as two separate correlations (card §3.1).
    """
    first, second = _branches()
    off = _computed(first, second)[1]
    correlation = cross_correlation(
        first, second, epsilon=NORMALISATION_EPSILON, correction=POPULATION_CORRECTION
    )
    upper = float(torch.triu(correlation, diagonal=1).pow(2).sum())
    lower = float(torch.tril(correlation, diagonal=-1).pow(2).sum())
    assert float(off) == pytest.approx(upper + lower, abs=1e-12)
    assert float(off) != pytest.approx(upper, abs=1e-6)
    assert upper != pytest.approx(lower, abs=1e-6)


def test_the_diagonal_is_excluded_from_the_redundancy_term() -> None:
    """`C_ii` is what the alignment term is *raising*; charging it here would
    make the two objectives pull against each other."""
    first, second = _branches()
    off = _computed(first, second)[1]
    correlation = cross_correlation(
        first, second, epsilon=NORMALISATION_EPSILON, correction=POPULATION_CORRECTION
    )
    with_diagonal = float(correlation.pow(2).sum())
    assert float(off) < with_diagonal
    assert float(off) != pytest.approx(with_diagonal, abs=1e-6)


def test_both_statistics_use_the_population_correction_and_divide_by_b() -> None:
    """`correction = 0` and denominator `B`: training-mode `BatchNorm1d` and
    `c.div_(self.args.batch_size)` (card §3.1, §7)."""
    first, second = _branches()
    diagonal, off = _computed(first, second)
    sample = _oracle_terms(first, second, correction=1)
    assert float(diagonal) != pytest.approx(float(sample[0]), abs=1e-6)
    assert float(off) != pytest.approx(float(sample[1]), abs=1e-6)
    # And the objectives compute the sample convention when asked for it, so
    # the difference above is the correction and not an unrelated mismatch.
    state, batch, rows = _state(first, second), _batch(), _all_rows()
    assert float(
        _diagonal(correction=1).compute(state, batch, rows, _ctx()).value
    ) == pytest.approx(float(sample[0]), abs=1e-12)
    assert float(
        _off_diagonal(correction=1).compute(state, batch, rows, _ctx()).value
    ) == pytest.approx(float(sample[1]), abs=1e-12)

    # The `/ B` is the author's, not `/ (B - 1)`: scaling every entry of `C` by
    # `B / (B - 1)` changes both reductions.
    correlation = cross_correlation(
        first, second, epsilon=NORMALISATION_EPSILON, correction=POPULATION_CORRECTION
    )
    rescaled = correlation * ROWS / (ROWS - 1)
    assert float(diagonal) != pytest.approx(
        float((1.0 - rescaled.diagonal()).pow(2).sum()), abs=1e-6
    )


def test_epsilon_sits_inside_the_square_root() -> None:
    """`sqrt(var + eps)`, not `sqrt(var) + eps`.

    Observable only because the fixture carries a column whose variance is
    below epsilon: there the correct denominator is `3.32e-3` and the mutant's
    is `1.01e-3`, so the third row of `C` is attenuated by more than three in
    the pinned arithmetic and barely at all in the mutant.
    """
    first, second = _branches()

    def outside(branch: Tensor) -> Tensor:
        variance = branch.var(dim=0, correction=POPULATION_CORRECTION)
        deviation = torch.sqrt(variance) + NORMALISATION_EPSILON
        return (branch - branch.mean(dim=0, keepdim=True)) / deviation

    mutant = outside(first).transpose(0, 1) @ outside(second) / ROWS
    diagonal, off = _computed(first, second)
    assert float(diagonal) != pytest.approx(
        float((1.0 - mutant.diagonal()).pow(2).sum()), abs=1e-6
    )
    assert float(off) != pytest.approx(
        float(mutant.masked_select(~torch.eye(WIDTH, dtype=torch.bool)).pow(2).sum()),
        abs=1e-6,
    )


def test_scaling_one_branch_is_not_free_because_epsilon_is_not_zero() -> None:
    """The card forbids asserting exact scale invariance, so assert the reverse.

    An implementation that normalised by a column norm with no epsilon — eq.
    (2) as printed — would be exactly scale invariant and would pass a test
    written the other way round.
    """
    first, second = _branches()
    plain = _computed(first, second)
    scaled = _computed(first * 2.0, second)
    assert float(scaled[0]) != pytest.approx(float(plain[0]), abs=1e-6)
    assert float(scaled[1]) != pytest.approx(float(plain[1]), abs=1e-6)


def test_exchanging_the_two_branches_transposes_c_and_changes_neither_term() -> None:
    """`D` reads the diagonal, which a transpose fixes pointwise, and `O` sums
    both triangles, which a transpose swaps."""
    first, second = _branches()
    swapped = tuple(
        objective.compute(_state(second, first), _batch(), _all_rows(), _ctx()).value
        for objective in (_diagonal(), _off_diagonal())
    )
    for computed, exchanged in zip(_computed(first, second), swapped, strict=True):
        assert float(computed) == pytest.approx(float(exchanged), abs=1e-12)


def test_a_common_row_permutation_leaves_both_terms_alone() -> None:
    """`C` is a sum over rows: reordering the batch cannot change it."""
    first, second = _branches()
    order = torch.tensor([3, 0, 4, 1, 2])
    permuted = _computed(first.index_select(0, order), second.index_select(0, order))
    for computed, reordered in zip(_computed(first, second), permuted, strict=True):
        assert float(computed) == pytest.approx(float(reordered), abs=1e-12)


def test_repairing_one_branchs_row_pairing_changes_both_terms() -> None:
    """The pairing is the whole signal: `C` correlates row `b` of one branch
    with row `b` of the other, and permuting one side alone is a different
    statistic. Without this, the invariance above is satisfied by any function
    of the two branches' column statistics."""
    first, second = _branches()
    order = torch.tensor([3, 0, 4, 1, 2])
    mismatched = _computed(first, second.index_select(0, order))
    for computed, broken in zip(_computed(first, second), mismatched, strict=True):
        assert float(computed) != pytest.approx(float(broken), abs=1e-6)


def test_independent_column_shifts_leave_both_terms_alone() -> None:
    """`U` centres each branch's columns, so a per-coordinate constant — the
    bias the author's projector deliberately does not carry — cannot be seen."""
    first, second = _branches()
    plain = _computed(first, second)
    shifted = _computed(
        first + torch.tensor([2.0, -3.5, 7.25], dtype=torch.float64),
        second + torch.tensor([-1.5, 0.75, 4.0], dtype=torch.float64),
    )
    for computed, moved in zip(plain, shifted, strict=True):
        assert float(computed) == pytest.approx(float(moved), abs=1e-10)


def test_a_constant_batch_is_finite_at_d_and_zero_without_a_gradient() -> None:
    """Card §3.1: exact collapse has finite loss `D = d`, `O = 0`, and may have
    zero gradient. Asserting escape from it would assert a property the
    arithmetic does not have."""
    constant = torch.full((ROWS, WIDTH), 0.75, dtype=torch.float64, requires_grad=True)
    diagonal = _diagonal().compute(
        _state(constant, constant + 0.0), _batch(), _all_rows(), _ctx()
    )
    off = _off_diagonal().compute(
        _state(constant, constant + 0.0), _batch(), _all_rows(), _ctx()
    )
    assert float(diagonal.value.detach()) == pytest.approx(float(WIDTH), abs=1e-12)
    assert float(off.value.detach()) == pytest.approx(0.0, abs=1e-12)
    backward(diagonal.value)
    assert torch.isfinite(diagonal.value) and torch.isfinite(off.value)
    assert constant.grad is not None and torch.isfinite(constant.grad).all()


def test_a_batch_below_two_rows_is_refused_by_both_terms() -> None:
    first, second = _branches()
    state, batch = _state(first, second), _batch()
    single = torch.tensor([2], dtype=torch.long)
    assert MINIMUM_ROWS == 2
    for objective in (_diagonal(), _off_diagonal()):
        with pytest.raises(LossError, match="eligible row"):
            objective.compute(state, batch, single, _ctx())
    # Two rows is enough for the statistic to be defined, and reporting `D = 0`
    # there would be the same false perfection one row reports as `D = d`.
    pair = torch.tensor([1, 3], dtype=torch.long)
    for objective in (_diagonal(), _off_diagonal()):
        assert torch.isfinite(objective.compute(state, batch, pair, _ctx()).value)


def test_no_eligible_rows_returns_the_zero_term() -> None:
    """The zero-eligible-row rule (`DESIGN.md` §1.3) reaches batch-coupled terms
    too — an empty batch is not the same failure as a one-row batch."""
    first, second = _branches()
    empty = torch.zeros(0, dtype=torch.long)
    for objective in (_diagonal(), _off_diagonal()):
        term = objective.compute(_state(first, second), _batch(), empty, _ctx())
        assert term.n == 0
        assert float(term.value) == 0.0


def test_the_statistics_read_the_eligible_rows_and_only_those() -> None:
    """A batch-coupled term's population is its own, not the whole batch."""
    first, second = _branches()
    rows = torch.tensor([0, 1, 3], dtype=torch.long)
    selected = _diagonal().compute(_state(first, second), _batch(), rows, _ctx())
    expected, _ = _oracle_terms(
        first.index_select(0, rows), second.index_select(0, rows)
    )
    assert float(selected.value) == pytest.approx(float(expected), abs=1e-12)
    assert selected.n == int(rows.numel())


def test_every_term_descends_both_branches() -> None:
    """No detach, no predictor, no teacher: `BarlowTwins.forward` (card §3.1)."""
    for objective in (_diagonal(), _off_diagonal()):
        assert objective.detaches == frozenset()
        assert objective.requires == frozenset(
            {(Port.X_PROJ, CORRUPTED_A), (Port.X_PROJ, CORRUPTED_B)}
        )
        assert objective.batch_coupled
        first, second = _branches()
        first.requires_grad_(True)
        second.requires_grad_(True)
        term = objective.compute(_state(first, second), _batch(), _all_rows(), _ctx())
        backward(term.value)
        assert first.grad is not None and float(first.grad.abs().sum()) > 0.0
        assert second.grad is not None and float(second.grad.abs().sum()) > 0.0


def test_the_diagnostics_separate_collapse_from_decorrelation() -> None:
    """`O -> 0` happens both ways; only the raw variance says which."""
    first, second = _branches()
    state, batch, rows = _state(first, second), _batch(), _all_rows()
    diagonal = _diagonal().compute(state, batch, rows, _ctx())
    off = _off_diagonal().compute(state, batch, rows, _ctx())
    assert set(diagonal.diagnostics) == {
        "mean_diagonal",
        "diagonal_error",
        "raw_variance_first",
        "raw_variance_second",
    }
    assert set(off.diagnostics) == {
        "redundancy",
        "raw_variance_first",
        "raw_variance_second",
    }
    correlation = cross_correlation(
        first, second, epsilon=NORMALISATION_EPSILON, correction=POPULATION_CORRECTION
    )
    assert diagonal.diagnostics["mean_diagonal"] == pytest.approx(
        float(correlation.diagonal().mean()), abs=1e-9
    )
    assert diagonal.diagnostics["diagonal_error"] == pytest.approx(
        float(diagonal.value) / WIDTH, abs=1e-9
    )
    assert off.diagnostics["redundancy"] == pytest.approx(
        float(off.value) / (WIDTH * (WIDTH - 1)), abs=1e-9
    )
    # The near-constant third column is what the §6.4 activity guard is for,
    # and it is visible here rather than only in the benchmark.
    assert off.diagnostics["raw_variance_first"] == pytest.approx(
        float(first.var(dim=0, correction=POPULATION_CORRECTION).mean()), abs=1e-9
    )
    assert off.diagnostics["raw_variance_first"] != pytest.approx(
        off.diagnostics["raw_variance_second"], abs=1e-6
    )


def test_the_objectives_reject_what_they_cannot_mean() -> None:
    with pytest.raises(LossError, match=r"reads .* twice"):
        _diagonal(second=CORRUPTED_A)
    with pytest.raises(LossError, match="carries treatment_distribution"):
        _off_diagonal(port=Port.T_GIVEN_X)
    with pytest.raises(LossError, match="correction must be 0 or 1"):
        _diagonal(correction=2)
    with pytest.raises(LossError, match="finite and positive"):
        _off_diagonal(epsilon=0.0)
    with pytest.raises(LossError, match="without `epsilon`"):
        CrossCorrelationDiagonal(
            port=Port.X_PROJ,
            first=CORRUPTED_A,
            second=CORRUPTED_B,
            correction=POPULATION_CORRECTION,
        )
    with pytest.raises(LossError, match="without `correction`"):
        CrossCorrelationOffDiagonal(
            port=Port.X_PROJ,
            first=CORRUPTED_A,
            second=CORRUPTED_B,
            epsilon=NORMALISATION_EPSILON,
        )
    with pytest.raises(PortContractError, match="embedding tensor"):
        _diagonal().compute(
            State(
                {
                    CORRUPTED_A: {
                        Port.X_PROJ: CategoricalTreatment(torch.randn(ROWS, 3))
                    },
                    CORRUPTED_B: {Port.X_PROJ: torch.randn(ROWS, WIDTH)},
                }
            ),
            _batch(),
            _all_rows(),
            _ctx(),
        )
    with pytest.raises(LossError, match=r"\[B, d\] embedding"):
        _off_diagonal().compute(
            State(
                {
                    CORRUPTED_A: {Port.X_PROJ: torch.randn(ROWS)},
                    CORRUPTED_B: {Port.X_PROJ: torch.randn(ROWS)},
                }
            ),
            _batch(),
            _all_rows(),
            _ctx(),
        )


@pytest.mark.parametrize("shape", [(), (ROWS, 0), (ROWS, WIDTH + 1)])
def test_malformed_branch_shapes_are_rejected(shape: tuple[int, ...]) -> None:
    first, second = _branches()
    malformed = torch.ones(shape, dtype=torch.float64)
    for left, right in ((malformed, second), (first, malformed)):
        for objective in (_diagonal(), _off_diagonal()):
            with pytest.raises(LossError, match="embedding"):
                objective.compute(_state(left, right), _batch(), _all_rows(), _ctx())


def test_terms_and_gradients_match_training_batchnorm() -> None:
    """Independent executable reference, including a sub-epsilon column."""
    for objective, diagonal in ((_diagonal(), True), (_off_diagonal(), False)):
        first, second = (z.requires_grad_() for z in _branches())
        actual = objective.compute(
            _state(first, second), _batch(), _all_rows(), _ctx()
        ).value
        normalise = nn.BatchNorm1d(WIDTH, affine=False, eps=1e-5).double().train()
        correlation = normalise(first).T @ normalise(second) / ROWS
        expected = (
            (correlation.diagonal() - 1).square().sum()
            if diagonal
            else correlation.flatten()[:-1]
            .view(WIDTH - 1, WIDTH + 1)[:, 1:]
            .flatten()
            .square()
            .sum()
        )
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
        actual_gradients = torch.autograd.grad(actual, (first, second))
        expected_gradients = torch.autograd.grad(expected, (first, second))
        for actual_gradient, expected_gradient in zip(
            actual_gradients, expected_gradients, strict=True
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1e-9, atol=1e-9
            )


def test_the_plan_shows_the_arithmetic_no_other_field_reveals() -> None:
    stage = compile(barlow_twins(_schema())).stage("pretrain")
    details = {objective.name: objective.plan_details for objective in stage.objectives}
    for name in ("cross_correlation_diagonal", "cross_correlation_off_diagonal"):
        assert any("correction = 0" in line for line in details[name])
        assert any("sqrt(var + 1e-05)" in line for line in details[name])
        assert any(
            "epsilon is inside the square root" in line for line in details[name]
        )
    assert any(
        "no division by d: eq. (1) sums the diagonal" in line
        for line in details["cross_correlation_diagonal"]
    )
    assert any(
        "both triangles are charged" in line
        for line in details["cross_correlation_off_diagonal"]
    )


def test_the_two_normalisation_constants_reach_the_provenance_digest() -> None:
    """`epsilon` and `correction` are not card keys, so `plan_details` is the
    only place a changed one can show up — and it must change the digest."""
    baseline = compile(barlow_twins(_schema())).plan
    recipe = barlow_twins(_schema())
    pretrain = recipe.program[0]
    altered = replace(
        recipe,
        program=Program(
            (
                replace(
                    pretrain,
                    objectives=(
                        replace(
                            pretrain.objectives[0],
                            objective=_diagonal(correction=1),
                        ),
                        pretrain.objectives[1],
                    ),
                ),
                recipe.program[1],
            )
        ),
    )
    assert compile(altered).plan.digest != baseline.digest


# ---------------------------------------------------------------------------
# BarlowTwinsProjector
# ---------------------------------------------------------------------------


def _projector(**overrides: object) -> BarlowTwinsProjector:
    defaults: dict[str, object] = {
        "representation_dim": 6,
        "widths": (8, 8, 4),
        "activation": BARLOW_TWINS_PROJECTOR_ACTIVATION,
        "normalisation": BARLOW_TWINS_PROJECTOR_NORMALISATION,
        "dropout": 0.0,
        "initialisation": BARLOW_TWINS_PROJECTOR_INITIALISATION,
    }
    return BarlowTwinsProjector(**(defaults | overrides))  # type: ignore[arg-type]


def test_the_projector_is_the_author_stack_layer_for_layer() -> None:
    """`[Linear(bias=False) -> BN -> ReLU] * (n-1)` then `Linear(bias=False)`.

    The `[False, False, False]` below is the whole difference from
    `VICRegExpander`, whose hidden linears keep their biases.
    """
    layers = list(_projector().network)
    kinds = [type(layer).__name__ for layer in layers]
    assert kinds == [
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Linear",
        "BatchNorm1d",
        "ReLU",
        "Linear",
    ]
    affine = [layer for layer in layers if isinstance(layer, nn.Linear)]
    assert [layer.bias is None for layer in affine] == [True, True, True]
    norms = [layer for layer in layers if isinstance(layer, nn.BatchNorm1d)]
    assert len(norms) == 2
    for norm in norms:
        assert norm.eps == 1e-5
        assert norm.momentum == 0.1
        assert norm.affine and norm.track_running_stats


def test_the_projector_output_is_unconstrained_and_signed() -> None:
    """No row-`l2` and no output normalisation: the author's fourth
    `BatchNorm1d` is loss arithmetic, reproduced statelessly inside the two
    objectives (deviation 6)."""
    projector = _projector()
    assert projector.requires == frozenset({Port.X_REPR})
    assert projector.provides == frozenset({Port.X_PROJ})
    torch.manual_seed(11)
    # Through `forward`, not through `network`: the claim is about what the
    # *port* carries, and a normalisation added in `forward` is invisible to a
    # test that calls the module stack directly.
    ports = PortView(
        {Port.X_REPR: torch.randn(ROWS + 3, 6)},
        declared=frozenset({Port.X_REPR}),
        component="barlow_twins_projector",
    )
    output = projector.forward(ports)[Port.X_PROJ]
    assert isinstance(output, Tensor)
    assert bool((output < 0.0).any())
    norms = output.detach().norm(dim=-1)
    assert float((norms - 1.0).abs().max()) > 1e-3
    # And not already unit-variance per coordinate: normalising here as well as
    # in the loss would make the objectives' epsilon unreachable.
    variance = output.detach().var(dim=0, correction=POPULATION_CORRECTION)
    assert float((variance - 1.0).abs().max()) > 1e-3


def test_the_projector_refuses_a_topology_the_card_does_not_pin() -> None:
    for field, value in (
        ("activation", "relu"),
        ("normalisation", "row_l2"),
        (
            "initialisation",
            BARLOW_TWINS_PROJECTOR_INITIALISATION.replace("false", "true"),
        ),
    ):
        with pytest.raises(GraphError, match="supports"):
            _projector(**{field: value})
    with pytest.raises(GraphError, match=r"dropout must be 0\.0"):
        _projector(dropout=0.1)


# ---------------------------------------------------------------------------
# The recipe and its plan
# ---------------------------------------------------------------------------


def test_the_recipe_plans_two_stages_and_two_distorted_passes() -> None:
    run = compile(barlow_twins(_schema()))
    assert run.graph.names == (
        "mlp_encoder",
        "barlow_twins_projector",
        "tarnet_head",
        "categorical_propensity",
    )
    assert [stage.name for stage in run.stages] == ["pretrain", "joint_fit"]

    pretrain = run.stage("pretrain")
    assert pretrain.steps == 1_000
    assert pretrain.trainable == ("mlp_encoder", "barlow_twins_projector")
    # Both branches are distorted: Barlow Twins has no clean anchor, so
    # `DEFAULT` appears in no pretraining pass. SCARF's does.
    assert sorted(str(forward.realisation) for forward in pretrain.passes) == sorted(
        str(realisation) for realisation in (CORRUPTED_A, CORRUPTED_B)
    )
    assert str(DEFAULT) not in {str(forward.realisation) for forward in pretrain.passes}
    for forward in pretrain.passes:
        assert forward.components == ("mlp_encoder", "barlow_twins_projector")

    fit = run.stage("joint_fit")
    assert fit.steps == 3_000
    assert fit.initialise_from == "pretrain"
    assert [str(forward.realisation) for forward in fit.passes] == [str(DEFAULT)]
    assert fit.passes[0].components == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )


def test_the_projector_is_not_executed_downstream() -> None:
    fit = compile(barlow_twins(_schema())).stage("joint_fit")
    assert "barlow_twins_projector" not in fit.trainable
    assert not any(
        "barlow_twins_projector" in forward.components for forward in fit.passes
    )


def test_a_projector_parameter_leaking_into_fine_tuning_is_a_compile_error() -> None:
    recipe = barlow_twins(_schema())
    fit = recipe.program[1]
    broken = replace(
        recipe,
        program=Program(
            (
                recipe.program[0],
                replace(fit, trainable=(*fit.trainable, "barlow_twins_projector")),
            ),
        ),
    )
    with pytest.raises(CompileError, match="dead weight"):
        compile(broken)


def test_the_downstream_heads_are_absent_from_pretraining() -> None:
    pretrain = compile(barlow_twins(_schema())).stage("pretrain")
    for head in ("tarnet_head", "categorical_propensity"):
        assert head not in pretrain.trainable
        assert not any(head in forward.components for forward in pretrain.passes)


def test_the_pretraining_stage_reads_no_label_of_any_kind() -> None:
    plan = compile(barlow_twins(_schema())).plan
    planned = {component.name: component for component in plan.components}
    for name in ("mlp_encoder", "barlow_twins_projector"):
        assert not planned[name].reads_raw_outcome
        assert not planned[name].outcome_dependent


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_each_stage_has_exactly_the_reviewed_objectives() -> None:
    run = compile(barlow_twins(_schema()))
    pretrain = run.stage("pretrain")
    assert [objective.name for objective in pretrain.objectives] == [
        "cross_correlation_diagonal",
        "cross_correlation_off_diagonal",
    ]
    assert [objective.rows for objective in pretrain.objectives] == [("all",)] * 2
    assert [objective.reduction for objective in pretrain.objectives] == ["mean"] * 2

    fit = run.stage("joint_fit")
    assert [objective.name for objective in fit.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        "missing_treatment_marginal_nll",
    ]
    assert [objective.rows for objective in fit.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("t_missing",),
    ]


def test_the_recipe_declares_two_independent_view_names() -> None:
    plan = compile(barlow_twins(_schema())).plan
    assert [view.name for view in plan.views] == ["corrupted_a", "corrupted_b"]
    assert [view.transforms for view in plan.views] == [
        ("FeatureCorruption(rate=0.6, columns=all)",)
    ] * 2
    assert [view.draws for view in plan.views] == [1, 1]
    for view in plan.views:
        assert set(view.preserves) == PRESERVED


def test_a_stage_holding_the_batch_statistics_cannot_take_external_batches() -> None:
    """`optimisation.batch_size` is arithmetic here — `C` divides by `B` — so it
    may not be the caller's to choose (`core/compile.py`, `scarf.md` §5.6)."""
    recipe = barlow_twins(_schema())
    broken = replace(
        recipe,
        program=Program(
            (
                replace(recipe.program[0], sampler=ExternalBatches()),
                recipe.program[1],
            )
        ),
    )
    with pytest.raises(CompileError, match="ExternalBatches"):
        compile(broken)


def test_a_schema_with_a_stale_derived_column_is_rejected() -> None:
    with pytest.raises(CompileError, match="derived column"):
        compile(barlow_twins(_schema(derived=True)))


def test_the_recipe_is_a_function_of_the_schema() -> None:
    wide = compile(
        barlow_twins(
            Schema(
                features=tuple(
                    FeatureSpec(f"f{index}", "continuous") for index in range(9)
                ),
                treatment_cardinality=4,
                outcome=OutcomeSpec(),
            )
        )
    ).plan
    assert wide.hyperparameters["architecture.widths_depths"]["mlp_encoder"] == (
        BARLOW_TWINS_ENCODER_WIDTHS
    )
    assert wide.hyperparameters["architecture.widths_depths"][
        "barlow_twins_projector"
    ] == (PROJECTOR_WIDTHS)
    assert (
        wide.hyperparameters["architecture.widths_depths"]["categorical_propensity"]
        == "linear 256 -> 4"
    )


def test_caller_supplies_both_view_policies_without_recipe_substitution() -> None:
    recipe = build_barlow_twins(
        _schema(),
        first_transforms=(FeatureCorruption(rate=0.2, columns=None),),
        second_transforms=(FeatureCorruption(rate=0.8, columns=None),),
    )
    assert [view.transforms for view in compile(recipe).plan.views] == [
        ("FeatureCorruption(rate=0.2, columns=all)",),
        ("FeatureCorruption(rate=0.8, columns=all)",),
    ]


# ---------------------------------------------------------------------------
# Views, provenance and the cached pair of draws
# ---------------------------------------------------------------------------


def _population(batch: XTYBatch, schema: Schema) -> TrainingPopulation:
    return build_population(
        Dataset(
            schema=schema,
            rows=batch,
            assignments={"train": torch.arange(batch.batch_size)},
        ),
        DataSpec(
            split=SplitSpec(protocol="the batch itself", train="train"),
            preprocess=PreprocessSpec(features="none", outcome="none"),
            missingness=MissingnessSpec(mechanism="observed"),
        ),
        seed=0,
    )


def test_the_two_views_are_independent_draws_of_one_transform() -> None:
    """`ViewSpec.apply` keys its generator by the view's *name*, so two names is
    what makes `Y^A` and `Y^B` independent. One name and two draws would tie
    both branches to one realisation, and `C` would be a within-view
    correlation matrix, removing the cross-view alignment problem."""
    schema = _schema()
    batch = _batch(rows=16)
    population = _population(batch, schema)
    recipe = barlow_twins(schema)
    first = recipe.view("corrupted_a").apply(
        batch, schema, rng_key=7, population=population
    )
    second = recipe.view("corrupted_b").apply(
        batch, schema, rng_key=7, population=population
    )
    assert not bool(torch.equal(first.x, second.x))
    # Deterministic in the key, so the difference above is the name and not an
    # unseeded generator.
    again = recipe.view("corrupted_a").apply(
        batch, schema, rng_key=7, population=population
    )
    assert bool(torch.equal(first.x, again.x))


def test_the_two_objectives_read_one_cached_pair_of_draws() -> None:
    """Two planned passes, two objectives, and no term draws a view of its own.

    A per-objective resample would make the redundancy term's `C` a different
    matrix from the alignment term's, so `lambda` would no longer be a weight
    on one pair of embeddings and the §6 zero-lambda arm would not be an
    ablation of anything.
    """
    schema = _schema()
    batch = _batch(rows=16)
    run = compile(barlow_twins(schema))
    pretrain = run.stage("pretrain")
    state = run.state(pretrain, batch, rng_key=3, population=_population(batch, schema))
    assert len(pretrain.passes) == 2
    first = state[CORRUPTED_A][Port.X_PROJ]
    second = state[CORRUPTED_B][Port.X_PROJ]
    assert isinstance(first, Tensor) and isinstance(second, Tensor)
    assert not bool(torch.equal(first, second))
    for compiled in pretrain.objectives:
        requires = compiled.objective.requires
        assert {realisation for _, realisation in requires} == {
            CORRUPTED_A,
            CORRUPTED_B,
        }
        for port, realisation in requires:
            wanted = first if realisation == CORRUPTED_A else second
            assert state[realisation][port] is wanted


def test_the_donor_pool_is_the_training_population_for_both_views() -> None:
    """Card §6.2: the view's donor rows and the scaler are training-only.

    Both views are checked, not one: a second `ViewSpec` that fell back to the
    batch would be a leak the first view's test could not see.
    """
    schema = _schema()
    batch = _batch(rows=8)
    training = _batch(rows=64, offset=100)
    marked = training.x.clone()
    marked[:, 0] = 999.0
    population = build_population(
        Dataset(
            schema=schema,
            rows=training.replace(x=marked),
            assignments={"train": torch.arange(training.batch_size)},
        ),
        DataSpec(
            split=SplitSpec(protocol="a population the batch is not", train="train"),
            preprocess=PreprocessSpec(features="none", outcome="none"),
            missingness=MissingnessSpec(mechanism="observed"),
        ),
        seed=0,
    )
    recipe = barlow_twins(schema)
    for name in ("corrupted_a", "corrupted_b"):
        distorted = recipe.view(name).apply(
            batch, schema, rng_key=5, population=population, draw=0
        )
        assert not bool((batch.x[:, 0] == 999.0).any())
        assert bool((distorted.x[:, 0] == 999.0).any())


# ---------------------------------------------------------------------------
# The stage transition
# ---------------------------------------------------------------------------


def _continuous_schema() -> Schema:
    return Schema(
        features=tuple(FeatureSpec(f"x{index}", "continuous") for index in range(4)),
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


def _dataset(schema: Schema) -> Dataset:
    """256 deterministic rows: more than the card's `B = 128`, and enough for
    the declared 40-observed-treatment budget."""
    generator = torch.Generator().manual_seed(2_026)
    rows = 256
    x = torch.randn(rows, 4, generator=generator)
    batch = XTYBatch(
        x=x,
        t=(torch.rand(rows, generator=generator) < 0.5).long(),
        y=x[:, 0] + torch.randn(rows, generator=generator) * 0.1,
        t_observed=torch.ones(rows, dtype=torch.bool),
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
    )
    return Dataset(schema=schema, rows=batch, assignments={"train": torch.arange(rows)})


def _two_step_run(schema: Schema) -> CompiledRun:
    """The reviewed program at two steps a stage.

    Only `steps` moves. The sampler, the objectives, the trainable lists and
    `initialise_from` are the reviewed ones, because they are what these two
    tests are about; the 1,000/3,000-step budget is Tier 1's and Tier 2's.
    """
    recipe = barlow_twins(schema)
    pretrain, fit = recipe.program
    return compile(
        replace(
            recipe,
            program=Program((replace(pretrain, steps=2), replace(fit, steps=2))),
        )
    )


def _snapshot(run: CompiledRun) -> dict[str, Tensor]:
    return {
        name: value.detach().clone() for name, value in run.graph.state_dict().items()
    }


def test_the_transition_carries_the_encoder_and_leaves_the_heads_alone() -> None:
    """Card §3.2: transfer the encoder's parameters and buffers, retain the
    heads' initial tensors, and never execute the projector downstream."""
    schema = _continuous_schema()
    data = _dataset(schema)
    run = _two_step_run(schema)
    start = _snapshot(run)
    heads = tuple(
        name
        for name in start
        if "tarnet_head." in name or "categorical_propensity." in name
    )
    assert heads, "the fixture found no head tensors to hold fixed"

    fit_start: dict[str, Tensor] = {}
    original = executors._step

    def traced(*args: Any, **kwargs: Any) -> Any:
        active_run, compiled, _batch_arg, _, _optimiser, _, step = args
        if compiled.name == "joint_fit" and step == 0:
            fit_start.update(_snapshot(active_run))
        return original(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as patch:
        patch.setattr(executors, "_step", traced)
        result = run_program(run, {"pretrain": data, "joint_fit": data}, seed=1_234)

    assert all(
        math.isfinite(record.total)
        for stage in result.stages
        for record in stage.records
    )
    checkpoint = result.stage("pretrain").checkpoint
    assert checkpoint.components == ("mlp_encoder", "barlow_twins_projector")
    # Pretraining moved the encoder, so "the fitting stage starts from the
    # pretrained encoder" is a claim with content.
    assert any(
        not torch.equal(value, start["_components." + name])
        for name, value in checkpoint.parameters.items()
        if name.startswith("mlp_encoder.")
    )
    assert fit_start, "the fitting stage never took a step"
    for name, value in checkpoint.parameters.items():
        assert torch.equal(value, fit_start["_components." + name])
    for name in heads:
        assert torch.equal(start[name], fit_start[name])
    # And the encoder then moves on, which is what fine-tuning it means, while
    # the projector — restored but never run — still holds pretrained tensors.
    final = _snapshot(run)
    assert any(
        not torch.equal(value, final[name])
        for name, value in fit_start.items()
        if "mlp_encoder." in name
    )
    for name, value in fit_start.items():
        if "barlow_twins_projector." in name:
            assert torch.equal(value, final[name])


def test_the_fitting_stage_starts_from_a_fresh_optimiser() -> None:
    """Card §3.2: "reset optimiser moments". One `OptimiserSpec` value object
    serves both stages, so the guarantee is the executor's rather than the
    declaration's, and it is asserted rather than assumed."""
    schema = _continuous_schema()
    data = _dataset(schema)
    run = _two_step_run(schema)
    seen: list[object] = []
    original = executors._step

    def traced(*args: Any, **kwargs: Any) -> Any:
        optimiser, step = args[4], args[6]
        if step == 0:
            assert not optimiser.state, "a stage began with inherited moments"
            assert all(optimiser is not previous for previous in seen)
            seen.append(optimiser)
        return original(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    with monkeypatch.context() as patch:
        patch.setattr(executors, "_step", traced)
        run_program(run, {"pretrain": data, "joint_fit": data}, seed=99)
    assert len(seen) == 2


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def _card_section_four() -> dict[str, str | dict[str, str]]:
    """Card §4 as data — the same two-level parse `test_vicreg.py` uses."""
    text = CARD.read_text(encoding="utf-8")
    section = text.split("## 4. Mechanics checklist", 1)[1].split(
        "## 5. Deviations from the paper", 1
    )[0]
    match = re.search(r"```yaml\n(.*?)```", section, re.DOTALL)
    assert match is not None
    answered: dict[str, str | dict[str, str]] = {}
    current = ""
    key = ""
    for line in match.group(1).splitlines():
        statement = line.split("#", 1)[0].rstrip()
        if not statement:
            continue
        indent = len(statement) - len(statement.lstrip())
        name, _, value = statement.strip().partition(":")
        if indent == 0:
            current = name
        elif indent == 2:
            key = f"{current}.{name}"
            if value.strip() == "n/a":
                key = ""
                continue
            answered[key] = value.strip()
        elif indent == 4 and key:
            nested = answered.get(key)
            if not isinstance(nested, dict):
                nested = {}
                answered[key] = nested
            nested[name] = value.strip()
    return answered


def _rendered(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def test_every_answered_card_key_reaches_the_plan() -> None:
    plan = compile(barlow_twins(_schema())).plan
    answered = set(_card_section_four())
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    # The one weight §6's study ablates, and the constant the parser sets
    # against §2.2's printed 0.005, spelled out so that a card which quietly
    # rounded it would fail here rather than pass a subset.
    assert plan.hyperparameters["losses.weights"] == {
        "pretrain.cross_correlation_diagonal": 1.0,
        "pretrain.cross_correlation_off_diagonal": 0.0051,
        "joint_fit.observed_outcome_nll": 1.0,
        "joint_fit.observed_treatment_nll": 1.0,
        "joint_fit.missing_treatment_marginal_nll": 0.5,
    }


def test_every_card_value_the_plan_also_carries_agrees_with_it() -> None:
    """Key presence is not the cross-check; the values are (`FIDELITY.md` §1.2)."""
    hyperparameters = compile(barlow_twins(_schema())).plan.hyperparameters
    mismatched: list[str] = []
    symbolic = {"architecture.widths_depths": {"K": "3", "X_REPR": "256"}}
    checked = 0
    for key, stated in _card_section_four().items():
        planned = hyperparameters.get(key)
        if planned is None:
            mismatched.append(f"{key}: absent from the plan")
            continue
        if isinstance(stated, str):
            if not isinstance(planned, dict) and _rendered(planned) != stated:
                mismatched.append(f"{key}: card {stated!r} vs plan {planned!r}")
            checked += 1
            continue
        assert isinstance(planned, Mapping), f"{key} is scoped in the card only"
        for scope, value in stated.items():
            if scope not in planned:
                mismatched.append(f"{key}[{scope}]: absent from the plan")
                continue
            resolved = value
            for symbol, concrete in symbolic.get(key, {}).items():
                resolved = resolved.replace(symbol, concrete)
            if _rendered(planned[scope]) != resolved:
                mismatched.append(
                    f"{key}[{scope}]: card {resolved!r} vs plan {planned[scope]!r}"
                )
            checked += 1
    assert not mismatched, "card and plan disagree:\n  " + "\n  ".join(mismatched)
    assert checked >= 66


def test_the_card_records_the_implementation_it_now_has() -> None:
    """The status ladder is card content, and `reproduced` is a claim about
    three files: this one, `tests/smoke/test_barlow_twins.py`, and the Tier 2
    module and test whose ten-seed result §6.1 records (`FIDELITY.md` §1.1)."""
    text = CARD.read_text(encoding="utf-8")
    root = CARD.parents[2]
    assert "**Status:** `reproduced`" in text
    assert "| Recipe implemented, Tier 0 passing (status → `implemented`) |" in text
    assert (
        "| Tier 1 study run on bases 419/523/631 (status → `smoke-passing`) |" in text
    )
    tier2_row = "| Tier 2 benchmark registered and ten-seed study run "
    assert tier2_row + "(status → `reproduced`) |" in text
    assert (root / "tests/smoke/test_barlow_twins.py").is_file()
    assert (root / "tests/benchmarks/test_barlow_twins.py").is_file()
    assert (root / "xty2/evaluation/benchmarks/barlow_twins.py").is_file()
    assert "0.0051" in text
    assert "390000+100*i" in text
    # The §6.1 row the run wrote, not an empty placeholder: the commit it was
    # produced on, and the four §6.4 bounds it scored, by the names the
    # benchmark gives them.
    assert "| | | | | |" not in text
    assert "`eb47d8bcf810`" in text
    for metric in (
        "full_arm_diagonal_alignment",
        "full_arm_active_fraction",
        "off_diagonal_ablation_redundancy_gap",
        "pretraining_outcome_NLL_cost",
    ):
        assert metric in text, metric
        assert metric in _BENCHMARK, metric
    assert "| [`barlow_twins.md`](recipes/barlow_twins.md) | `barlow_twins` |" in (
        CARD.parents[1] / "RECIPES.md"
    ).read_text(encoding="utf-8")
