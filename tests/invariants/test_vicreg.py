"""Tier 0 — the VICReg recipe, its expander and its three embedding terms.

The load-bearing tests are `test_each_term_is_the_author_forward` and
`test_each_terms_input_gradients_are_the_author_forwards`: like the
exact-marginalisation invariant (`FIDELITY.md` §3), they compare the vectorised
implementation against a loop transcription of `docs/recipes/vicreg.md` §3.1 in
float64, on a fixture with unequal branch variances, nonzero column means and
nonzero off-diagonal covariance — so that a reduction, a denominator or a
centring that is wrong in the way a reimplementation usually gets it wrong
cannot cancel.

Both halves are needed. The value alone does not distinguish `(1/(n-1))` from
`(1/n)` at large `n`, and a term whose value is right and whose gradient is
scaled wrong trains a different model at the same logged number.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn
from xty2.components import VICRegExpander
from xty2.components.vicreg import (
    VICREG_EXPANDER_ACTIVATION,
    VICREG_EXPANDER_INITIALISATION,
    VICREG_EXPANDER_NORMALISATION,
)
from xty2.core import (
    DEFAULT,
    CategoricalTreatment,
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
    EmbeddingCovariance,
    EmbeddingInvariance,
    EmbeddingVariance,
)
from xty2.objectives.vicreg import MINIMUM_ROWS
from xty2.recipes import vicreg as build_vicreg
from xty2.recipes.vicreg import (
    CORRUPTED_A,
    CORRUPTED_B,
    EXPANDER_WIDTHS,
    SAMPLE_CORRECTION,
    VARIANCE_EPSILON,
    VARIANCE_TARGET,
    VICREG_ENCODER_WIDTHS,
)
from xty2.training.loading import build_population
from xty2.views import FeatureCorruption

from tests.invariants.conftest import backward

vicreg = partial(
    build_vicreg,
    first_transforms=(FeatureCorruption(rate=0.6, columns=None),),
    second_transforms=(FeatureCorruption(rate=0.6, columns=None),),
)

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "vicreg.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "vicreg.py"
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
    """Two float64 embeddings the §6.3 fixture conditions describe.

    Built by hand rather than sampled: the column variances differ between the
    branches, no column is centred, and the columns are correlated within each
    branch, so `(n-1)` vs `n`, a missing centring and an included diagonal all
    change the number. The scale is chosen so that the hinge of eq. (1) is
    **active on some columns and slack on others** — one column of the first
    branch stands above `gamma` and every other column below it — because a
    fixture whose standard deviations all clear `gamma` gives the variance term
    the value zero and the gradient zero, which every wrong reduction also
    produces.
    """
    first = torch.tensor(
        [
            [0.5, 1.25, -0.25],
            [1.0, 2.0, 0.125],
            [-0.75, -1.0, 1.5],
            [0.25, 0.625, 0.5],
            [1.5, 2.75, -1.0],
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


def _invariance(**overrides: object) -> EmbeddingInvariance:
    defaults: dict[str, object] = {
        "port": Port.X_PROJ,
        "first": CORRUPTED_A,
        "second": CORRUPTED_B,
    }
    return EmbeddingInvariance(**(defaults | overrides))  # type: ignore[arg-type]


def _variance(**overrides: object) -> EmbeddingVariance:
    defaults: dict[str, object] = {
        "port": Port.X_PROJ,
        "first": CORRUPTED_A,
        "second": CORRUPTED_B,
        "gamma": VARIANCE_TARGET,
        "epsilon": VARIANCE_EPSILON,
        "correction": SAMPLE_CORRECTION,
    }
    return EmbeddingVariance(**(defaults | overrides))  # type: ignore[arg-type]


def _covariance(**overrides: object) -> EmbeddingCovariance:
    defaults: dict[str, object] = {
        "port": Port.X_PROJ,
        "first": CORRUPTED_A,
        "second": CORRUPTED_B,
        "correction": SAMPLE_CORRECTION,
    }
    return EmbeddingCovariance(**(defaults | overrides))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The loop oracle: §3.1, one scalar at a time
# ---------------------------------------------------------------------------


def _oracle_invariance(first: Tensor, second: Tensor) -> Tensor:
    """`mean_{i,j} (Z_ij - Z'_ij)^2` — the author's elementwise `F.mse_loss`."""
    rows, width = first.shape
    total = first.new_zeros(())
    for i in range(rows):
        for j in range(width):
            total = total + (first[i, j] - second[i, j]) ** 2
    return total / (rows * width)


def _oracle_v(
    embedding: Tensor, gamma: float, epsilon: float, correction: int
) -> Tensor:
    """`v(Z)` of eq. (1)-(2), column by column."""
    rows, width = embedding.shape
    total = embedding.new_zeros(())
    for j in range(width):
        column = [embedding[i, j] for i in range(rows)]
        mean = sum(column, embedding.new_zeros(())) / rows
        squared = sum(
            ((value - mean) ** 2 for value in column), embedding.new_zeros(())
        )
        deviation = torch.sqrt(squared / (rows - correction) + epsilon)
        total = total + torch.clamp(gamma - deviation, min=0.0)
    return total / width


def _oracle_c(embedding: Tensor, correction: int) -> Tensor:
    """`c(Z)` of eq. (3)-(4), entry by entry, diagonal excluded."""
    rows, width = embedding.shape
    means = [
        sum((embedding[i, j] for i in range(rows)), embedding.new_zeros(())) / rows
        for j in range(width)
    ]
    total = embedding.new_zeros(())
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            entry = sum(
                (
                    (embedding[k, i] - means[i]) * (embedding[k, j] - means[j])
                    for k in range(rows)
                ),
                embedding.new_zeros(()),
            ) / (rows - correction)
            total = total + entry**2
    return total / width


def _oracle_terms(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The three published terms under the pinned author reductions."""
    return (
        _oracle_invariance(first, second),
        (
            _oracle_v(first, VARIANCE_TARGET, VARIANCE_EPSILON, SAMPLE_CORRECTION)
            + _oracle_v(second, VARIANCE_TARGET, VARIANCE_EPSILON, SAMPLE_CORRECTION)
        )
        / 2.0,
        _oracle_c(first, SAMPLE_CORRECTION) + _oracle_c(second, SAMPLE_CORRECTION),
    )


def _computed(first: Tensor, second: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    state = _state(first, second)
    batch = _batch()
    rows = _all_rows()
    return tuple(  # type: ignore[return-value]
        objective.compute(state, batch, rows, _ctx()).value
        for objective in (_invariance(), _variance(), _covariance())
    )


# ---------------------------------------------------------------------------
# The three terms against the oracle
# ---------------------------------------------------------------------------


def test_each_term_is_the_author_forward() -> None:
    first, second = _branches()
    for computed, expected in zip(
        _computed(first, second), _oracle_terms(first, second), strict=True
    ):
        assert float(computed) == pytest.approx(float(expected), abs=1e-12)


def test_each_terms_input_gradients_are_the_author_forwards() -> None:
    """A value can be right for a reason the gradient is not (`FIDELITY.md` §3)."""
    for index in range(3):
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
            assert torch.allclose(actual, wanted, atol=1e-12)
            assert float(actual.abs().sum()) > 0.0


def test_the_invariance_term_divides_by_the_embedding_dimension() -> None:
    """Deviation 1: the author's `F.mse_loss`, not eq. (5)'s summed norm.

    Reading eq. (6)'s `lambda = 25` onto eq. (5)'s reduction is the same
    declaration `d` times louder, which at the card's `d = 512` is a
    three-orders-of-magnitude error in the term holding the branches together.
    """
    first, second = _branches()
    published = (first - second).pow(2).sum(dim=-1).mean()
    computed = _computed(first, second)[0]
    assert float(computed) == pytest.approx(float(published) / WIDTH, abs=1e-12)
    assert float(computed) != pytest.approx(float(published), abs=1e-6)


def test_the_variance_term_averages_the_branches_and_the_covariance_sums_them() -> None:
    """`VICReg.forward` is asymmetric between its two regularisers; eq. (6) is not."""
    first, second = _branches()
    _, variance, covariance = _computed(first, second)
    left_v = _oracle_v(first, VARIANCE_TARGET, VARIANCE_EPSILON, SAMPLE_CORRECTION)
    right_v = _oracle_v(second, VARIANCE_TARGET, VARIANCE_EPSILON, SAMPLE_CORRECTION)
    left_c = _oracle_c(first, SAMPLE_CORRECTION)
    right_c = _oracle_c(second, SAMPLE_CORRECTION)
    assert float(variance) == pytest.approx(float(left_v + right_v) / 2.0, abs=1e-12)
    assert float(variance) != pytest.approx(float(left_v + right_v), abs=1e-6)
    assert float(covariance) == pytest.approx(float(left_c + right_c), abs=1e-12)
    assert float(covariance) != pytest.approx(float(left_c + right_c) / 2.0, abs=1e-6)


def test_both_statistics_use_the_sample_correction() -> None:
    """`n - 1`, the author convention for `Var` and for `C` alike (card §7)."""
    first, second = _branches()
    _, variance, covariance = _computed(first, second)
    population_v = (
        _oracle_v(first, VARIANCE_TARGET, VARIANCE_EPSILON, 0)
        + _oracle_v(second, VARIANCE_TARGET, VARIANCE_EPSILON, 0)
    ) / 2.0
    population_c = _oracle_c(first, 0) + _oracle_c(second, 0)
    assert float(variance) != pytest.approx(float(population_v), abs=1e-6)
    assert float(covariance) != pytest.approx(float(population_c), abs=1e-6)
    # And the objective computes the population convention when asked for it,
    # so the difference above is the correction and not an unrelated mismatch.
    state, batch, rows = _state(first, second), _batch(), _all_rows()
    assert float(
        _variance(correction=0).compute(state, batch, rows, _ctx()).value
    ) == pytest.approx(float(population_v), abs=1e-12)
    assert float(
        _covariance(correction=0).compute(state, batch, rows, _ctx()).value
    ) == pytest.approx(float(population_c), abs=1e-12)


def test_the_covariance_excludes_its_own_diagonal() -> None:
    """The diagonal is what the variance term is *raising*; charging it here
    would make the two objectives pull against each other."""
    first, second = _branches()
    covariance = _computed(first, second)[2]
    with_diagonal = sum(
        float(
            (
                (branch - branch.mean(dim=0, keepdim=True)).transpose(0, 1)
                @ (branch - branch.mean(dim=0, keepdim=True))
                / (ROWS - SAMPLE_CORRECTION)
            )
            .pow(2)
            .sum()
            / WIDTH
        )
        for branch in (first, second)
    )
    assert float(covariance) < with_diagonal
    assert float(covariance) != pytest.approx(with_diagonal, abs=1e-6)


def test_epsilon_sits_inside_the_square_root() -> None:
    """A collapsed dimension is charged `gamma - sqrt(eps)`, not `gamma`.

    Outside the root, `epsilon` would be a floor on the hinge and a fully
    collapsed embedding would be charged the full `gamma`; inside it, the
    penalty is finite, the gradient is smooth, and the difference is `1e-2` per
    collapsed dimension — small, and exactly the kind of constant a
    reimplementation moves.
    """
    collapsed = torch.zeros(ROWS, WIDTH, dtype=torch.float64)
    term = _variance().compute(
        _state(collapsed, collapsed.clone()), _batch(), _all_rows(), _ctx()
    )
    assert float(term.value) == pytest.approx(
        VARIANCE_TARGET - VARIANCE_EPSILON**0.5, abs=1e-12
    )
    assert float(term.value) != pytest.approx(VARIANCE_TARGET, abs=1e-6)
    assert torch.isfinite(term.value)


def test_exchanging_the_two_branches_is_the_same_number() -> None:
    """All three terms are symmetric, unlike SCARF's anchor and contrast."""
    first, second = _branches()
    swapped = tuple(
        objective.compute(_state(second, first), _batch(), _all_rows(), _ctx()).value
        for objective in (_invariance(), _variance(), _covariance())
    )
    for computed, exchanged in zip(_computed(first, second), swapped, strict=True):
        assert float(computed) == pytest.approx(float(exchanged), abs=1e-12)


def test_the_two_regularisers_are_shift_invariant() -> None:
    """`v` and `c` are centred statistics; only `s` sees a constant offset."""
    first, second = _branches()
    shift = torch.tensor([2.0, -3.5, 7.25], dtype=torch.float64)
    plain = _computed(first, second)
    shifted = _computed(first + shift, second + shift)
    assert float(shifted[1]) == pytest.approx(float(plain[1]), abs=1e-12)
    assert float(shifted[2]) == pytest.approx(float(plain[2]), abs=1e-10)
    # The invariance term is *not* shift-invariant when only one branch moves,
    # which is what makes the pair above a property of `v` and `c` rather than
    # of the fixture.
    moved = _computed(first + shift, second)
    assert float(moved[0]) != pytest.approx(float(plain[0]), abs=1e-6)


def test_a_collapsed_batch_is_finite_without_claiming_a_gradient() -> None:
    """At exact collapse `d(sqrt(Var + eps))/dz` is zero, and that is honest.

    The value is what must be finite; asserting a nonzero gradient here would
    be asserting a property the arithmetic does not have, and the mechanism the
    variance term relies on is the gradient *near* collapse, which the second
    half checks.
    """
    collapsed = torch.zeros(ROWS, WIDTH, dtype=torch.float64, requires_grad=True)
    term = _variance().compute(
        _state(collapsed, collapsed + 0.0), _batch(), _all_rows(), _ctx()
    )
    backward(term.value)
    assert torch.isfinite(term.value)
    assert collapsed.grad is not None and torch.isfinite(collapsed.grad).all()

    nearly = torch.zeros(ROWS, WIDTH, dtype=torch.float64)
    nearly[0] = 1e-3
    nearly.requires_grad_(True)
    near_term = _variance().compute(
        _state(nearly, nearly + 0.0), _batch(), _all_rows(), _ctx()
    )
    backward(near_term.value)
    assert nearly.grad is not None and float(nearly.grad.abs().sum()) > 0.0


def test_a_batch_below_two_rows_is_refused_by_both_statistics() -> None:
    first, second = _branches()
    state, batch = _state(first, second), _batch()
    single = torch.tensor([2], dtype=torch.long)
    assert MINIMUM_ROWS == 2
    for objective in (_variance(), _covariance()):
        with pytest.raises(LossError, match="eligible row"):
            objective.compute(state, batch, single, _ctx())
    # One row *is* enough for the invariance term: it reads no other row.
    assert torch.isfinite(_invariance().compute(state, batch, single, _ctx()).value)


def test_no_eligible_rows_returns_the_zero_term() -> None:
    """The zero-eligible-row rule (`DESIGN.md` §1.3) reaches the batch-coupled
    terms too — an empty batch is not the same failure as a one-row batch."""
    first, second = _branches()
    empty = torch.zeros(0, dtype=torch.long)
    for objective in (_invariance(), _variance(), _covariance()):
        term = objective.compute(_state(first, second), _batch(), empty, _ctx())
        assert term.n == 0
        assert float(term.value) == 0.0


def test_the_statistics_read_the_eligible_rows_and_only_those() -> None:
    """A batch-coupled term's population is its own, not the whole batch."""
    first, second = _branches()
    rows = torch.tensor([0, 1, 3], dtype=torch.long)
    selected = _variance().compute(_state(first, second), _batch(), rows, _ctx())
    expected = (
        _oracle_v(
            first.index_select(0, rows),
            VARIANCE_TARGET,
            VARIANCE_EPSILON,
            SAMPLE_CORRECTION,
        )
        + _oracle_v(
            second.index_select(0, rows),
            VARIANCE_TARGET,
            VARIANCE_EPSILON,
            SAMPLE_CORRECTION,
        )
    ) / 2.0
    assert float(selected.value) == pytest.approx(float(expected), abs=1e-12)
    assert selected.n == int(rows.numel())


def test_every_term_descends_both_branches() -> None:
    """No detach, no predictor, no teacher: `VICReg.forward` (card §3.1)."""
    for objective in (_invariance(), _variance(), _covariance()):
        assert objective.detaches == frozenset()
        assert objective.requires == frozenset(
            {(Port.X_PROJ, CORRUPTED_A), (Port.X_PROJ, CORRUPTED_B)}
        )
        first, second = _branches()
        first.requires_grad_(True)
        second.requires_grad_(True)
        term = objective.compute(_state(first, second), _batch(), _all_rows(), _ctx())
        backward(term.value)
        assert first.grad is not None and float(first.grad.abs().sum()) > 0.0
        assert second.grad is not None and float(second.grad.abs().sum()) > 0.0


def test_only_the_batch_statistics_declare_themselves_batch_coupled() -> None:
    assert not _invariance().batch_coupled
    assert _variance().batch_coupled
    assert _covariance().batch_coupled


def test_the_diagnostics_separate_collapse_from_decorrelation() -> None:
    """`c(Z) -> 0` happens both ways; only the diagonal says which."""
    first, second = _branches()
    variance = _variance().compute(_state(first, second), _batch(), _all_rows(), _ctx())
    covariance = _covariance().compute(
        _state(first, second), _batch(), _all_rows(), _ctx()
    )
    assert set(variance.diagnostics) == {"spread_first", "spread_second"}
    assert variance.diagnostics["spread_first"] == pytest.approx(
        float(
            torch.sqrt(
                first.var(dim=0, correction=SAMPLE_CORRECTION) + VARIANCE_EPSILON
            ).mean()
        ),
        abs=1e-9,
    )
    assert set(covariance.diagnostics) == {
        "off_diagonal_first",
        "off_diagonal_second",
        "diagonal_first",
        "diagonal_second",
    }
    assert covariance.diagnostics["diagonal_first"] > 0.0


def test_the_objectives_reject_what_they_cannot_mean() -> None:
    with pytest.raises(LossError, match=r"reads .* twice"):
        _invariance(second=CORRUPTED_A)
    with pytest.raises(LossError, match="carries treatment_distribution"):
        _variance(port=Port.T_GIVEN_X)
    with pytest.raises(LossError, match="correction must be 0 or 1"):
        _covariance(correction=2)
    with pytest.raises(LossError, match="finite and positive"):
        _variance(gamma=0.0)
    with pytest.raises(LossError, match="finite and positive"):
        _variance(epsilon=-1e-4)
    with pytest.raises(TypeError):
        EmbeddingVariance(  # type: ignore[call-arg]
            port=Port.X_PROJ, first=CORRUPTED_A, second=CORRUPTED_B
        )
    with pytest.raises(PortContractError, match="embedding tensor"):
        _invariance().compute(
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
        _covariance().compute(
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


def test_the_plan_shows_the_arithmetic_no_other_field_reveals() -> None:
    stage = compile(vicreg(_schema())).stage("pretrain")
    details = {objective.name: objective.plan_details for objective in stage.objectives}
    assert any(
        "eq. (5) divided by d" in line for line in details["embedding_invariance"]
    )
    assert any(
        "epsilon is inside the square root" in line
        for line in details["embedding_variance"]
    )
    assert any("correction = 1" in line for line in details["embedding_variance"])
    assert any(
        "the diagonal is excluded" in line for line in details["embedding_covariance"]
    )
    assert any("denominator n - 1" in line for line in details["embedding_covariance"])


def test_two_variance_constants_do_not_share_a_provenance_identity() -> None:
    """`gamma` and `epsilon` are in the plan, so a changed one changes the digest."""
    baseline = compile(vicreg(_schema())).plan
    recipe = vicreg(_schema())
    pretrain = recipe.program[0]
    weighted = pretrain.objectives[1]
    altered = replace(
        recipe,
        program=Program(
            (
                replace(
                    pretrain,
                    objectives=(
                        pretrain.objectives[0],
                        replace(weighted, objective=_variance(gamma=2.0)),
                        pretrain.objectives[2],
                    ),
                ),
                recipe.program[1],
            )
        ),
    )
    assert compile(altered).plan.digest != baseline.digest


# ---------------------------------------------------------------------------
# VICRegExpander
# ---------------------------------------------------------------------------


def _expander(**overrides: object) -> VICRegExpander:
    defaults: dict[str, object] = {
        "representation_dim": 6,
        "widths": (8, 8, 4),
        "activation": VICREG_EXPANDER_ACTIVATION,
        "normalisation": VICREG_EXPANDER_NORMALISATION,
        "dropout": 0.0,
        "initialisation": VICREG_EXPANDER_INITIALISATION,
    }
    return VICRegExpander(**(defaults | overrides))  # type: ignore[arg-type]


def test_the_expander_is_the_authors_projector_layer_for_layer() -> None:
    """`[Linear -> BN -> ReLU] * (n-1)` then a bias-free `Linear`."""
    layers = list(_expander().network)
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
    assert [layer.bias is None for layer in affine] == [False, False, True]
    norms = [layer for layer in layers if isinstance(layer, nn.BatchNorm1d)]
    assert len(norms) == 2
    for norm in norms:
        assert norm.eps == 1e-5
        assert norm.momentum == 0.1
        assert norm.affine and norm.track_running_stats


def test_the_expander_output_is_unconstrained() -> None:
    """No row-`l2`: `v` and `c` are statistics of an unnormalised embedding.

    A unit-norm output would bound `Var(z^j)` above by construction and hand
    the variance hinge a ceiling the paper does not put there.
    """
    expander = _expander()
    assert expander.requires == frozenset({Port.X_REPR})
    assert expander.provides == frozenset({Port.X_PROJ})
    torch.manual_seed(11)
    # Through `forward`, not through `network`: the claim is about what the
    # *port* carries, and a normalisation added in `forward` is invisible to a
    # test that calls the module stack directly.
    ports = PortView(
        {Port.X_REPR: torch.randn(ROWS + 3, 6)},
        declared=frozenset({Port.X_REPR}),
        component="vicreg_expander",
    )
    output = expander.forward(ports)[Port.X_PROJ]
    assert isinstance(output, Tensor)
    assert bool((output < 0.0).any())
    norms = output.detach().norm(dim=-1)
    assert float((norms - 1.0).abs().max()) > 1e-3


def test_the_expander_refuses_a_topology_the_card_does_not_pin() -> None:
    for field, value in (
        ("activation", "relu"),
        ("normalisation", "row_l2"),
        ("initialisation", "normal std=0.1/sqrt(fan_in), bias=0"),
    ):
        with pytest.raises(GraphError, match="supports"):
            _expander(**{field: value})
    with pytest.raises(GraphError, match=r"dropout must be 0\.0"):
        _expander(dropout=0.1)


# ---------------------------------------------------------------------------
# The recipe and its plan
# ---------------------------------------------------------------------------


def test_the_recipe_plans_two_stages_and_two_corrupted_passes() -> None:
    run = compile(vicreg(_schema()))
    assert run.graph.names == (
        "mlp_encoder",
        "vicreg_expander",
        "tarnet_head",
        "categorical_propensity",
    )
    assert [stage.name for stage in run.stages] == ["pretrain", "joint_fit"]

    pretrain = run.stage("pretrain")
    assert pretrain.steps == 1_000
    assert pretrain.trainable == ("mlp_encoder", "vicreg_expander")
    # Both branches are corrupted: VICReg has no clean anchor, so `DEFAULT`
    # appears in no pretraining pass. SCARF's does.
    assert sorted(str(forward.realisation) for forward in pretrain.passes) == sorted(
        str(realisation) for realisation in (CORRUPTED_A, CORRUPTED_B)
    )
    assert str(DEFAULT) not in {str(forward.realisation) for forward in pretrain.passes}
    for forward in pretrain.passes:
        assert forward.components == ("mlp_encoder", "vicreg_expander")

    fit = run.stage("joint_fit")
    assert fit.steps == 3_000
    assert fit.initialise_from == "pretrain"
    assert [str(forward.realisation) for forward in fit.passes] == [str(DEFAULT)]
    assert fit.passes[0].components == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )


def test_the_expander_is_discarded_by_the_fitting_stage() -> None:
    fit = compile(vicreg(_schema())).stage("joint_fit")
    assert "vicreg_expander" not in fit.trainable
    assert not any("vicreg_expander" in forward.components for forward in fit.passes)


def test_an_expander_parameter_leaking_into_fine_tuning_is_a_compile_error() -> None:
    recipe = vicreg(_schema())
    fit = recipe.program[1]
    broken = replace(
        recipe,
        program=Program(
            (
                recipe.program[0],
                replace(fit, trainable=(*fit.trainable, "vicreg_expander")),
            ),
        ),
    )
    with pytest.raises(CompileError, match="dead weight"):
        compile(broken)


def test_the_downstream_heads_are_absent_from_pretraining() -> None:
    pretrain = compile(vicreg(_schema())).stage("pretrain")
    for head in ("tarnet_head", "categorical_propensity"):
        assert head not in pretrain.trainable
        assert not any(head in forward.components for forward in pretrain.passes)


def test_the_pretraining_stage_reads_no_label_of_any_kind() -> None:
    plan = compile(vicreg(_schema())).plan
    planned = {component.name: component for component in plan.components}
    for name in ("mlp_encoder", "vicreg_expander"):
        assert not planned[name].reads_raw_outcome
        assert not planned[name].outcome_dependent


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_each_stage_has_exactly_the_reviewed_objectives() -> None:
    run = compile(vicreg(_schema()))
    pretrain = run.stage("pretrain")
    assert [objective.name for objective in pretrain.objectives] == [
        "embedding_invariance",
        "embedding_variance",
        "embedding_covariance",
    ]
    assert [objective.rows for objective in pretrain.objectives] == [("all",)] * 3
    assert [objective.reduction for objective in pretrain.objectives] == ["mean"] * 3

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


def test_the_recipe_declares_two_independent_corruption_views() -> None:
    plan = compile(vicreg(_schema())).plan
    assert [view.name for view in plan.views] == ["corrupted_a", "corrupted_b"]
    assert [view.transforms for view in plan.views] == [
        ("FeatureCorruption(rate=0.6, columns=all)",)
    ] * 2
    assert [view.draws for view in plan.views] == [1, 1]
    for view in plan.views:
        assert set(view.preserves) == PRESERVED


def test_a_stage_holding_the_batch_statistics_cannot_take_external_batches() -> None:
    """`optimisation.batch_size` is arithmetic here, so it may not be the
    caller's to choose (`core/compile.py`, `scarf.md` §5.6)."""
    recipe = vicreg(_schema())
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
        compile(vicreg(_schema(derived=True)))


def test_the_recipe_is_a_function_of_the_schema() -> None:
    wide = compile(
        vicreg(
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
        VICREG_ENCODER_WIDTHS
    )
    assert wide.hyperparameters["architecture.widths_depths"]["vicreg_expander"] == (
        EXPANDER_WIDTHS
    )
    assert (
        wide.hyperparameters["architecture.widths_depths"]["categorical_propensity"]
        == "linear 256 -> 4"
    )


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
    """`ViewSpec.apply` keys its generator by the view's *name*, so two names
    is what makes `t` and `t'` independent. One name and two draws would tie
    both branches to one realisation and make the invariance term zero."""
    schema = _schema()
    batch = _batch(rows=16)
    population = _population(batch, schema)
    recipe = vicreg(schema)
    first = recipe.view("corrupted_a").apply(
        batch, schema, rng_key=7, population=population
    )
    second = recipe.view("corrupted_b").apply(
        batch, schema, rng_key=7, population=population
    )
    assert not bool(torch.equal(first.x, second.x))
    # Deterministic in the key, so the difference above is the name and not
    # an unseeded generator.
    again = recipe.view("corrupted_a").apply(
        batch, schema, rng_key=7, population=population
    )
    assert bool(torch.equal(first.x, again.x))


def test_the_three_objectives_read_one_cached_pair_of_draws() -> None:
    """Two planned passes, three objectives, and no term draws a view of its own.

    A per-objective resample would make the covariance term's `Z` a different
    corruption from the invariance term's, so the three coefficients of eq. (6)
    would no longer be weights on one pair of embeddings.
    """
    schema = _schema()
    batch = _batch(rows=16)
    run = compile(vicreg(schema))
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
    """Card §6.2: the corruption marginal and the scaler are training-only.

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
    recipe = vicreg(schema)
    for name in ("corrupted_a", "corrupted_b"):
        corrupted = recipe.view(name).apply(
            batch, schema, rng_key=5, population=population, draw=0
        )
        assert not bool((batch.x[:, 0] == 999.0).any())
        assert bool((corrupted.x[:, 0] == 999.0).any())


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def _card_section_four() -> dict[str, str | dict[str, str]]:
    """Card §4 as data — the same two-level parse `test_scarf.py` uses."""
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
    plan = compile(vicreg(_schema())).plan
    answered = set(_card_section_four())
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    # The three weights §6's study ablates, spelled out so that a card which
    # quietly dropped one would fail here rather than pass a subset.
    assert plan.hyperparameters["losses.weights"] == {
        "pretrain.embedding_invariance": 25.0,
        "pretrain.embedding_variance": 25.0,
        "pretrain.embedding_covariance": 1.0,
        "joint_fit.observed_outcome_nll": 1.0,
        "joint_fit.observed_treatment_nll": 1.0,
        "joint_fit.missing_treatment_marginal_nll": 0.5,
    }


def test_every_card_value_the_plan_also_carries_agrees_with_it() -> None:
    """Key presence is not the cross-check; the values are (`FIDELITY.md` §1.2)."""
    hyperparameters = compile(vicreg(_schema())).plan.hyperparameters
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
    assert checked >= 45


def test_small_off_diagonal_energy_survives_large_diagonal_energy() -> None:
    embedding = torch.tensor(
        [
            [10000.0, 10001.0],
            [-10000.0, 9999.0],
            [10000.0, -9999.0],
            [-10000.0, -10001.0],
        ],
        requires_grad=True,
    )
    actual = _covariance()._off_diagonal(embedding)
    oracle = _oracle_c(embedding.double(), correction=1)
    assert float(actual.detach()) > 0.0
    torch.testing.assert_close(actual.double(), oracle, rtol=1e-6, atol=0.0)
    actual_gradient = torch.autograd.grad(actual, embedding, retain_graph=True)[0]
    oracle_gradient = torch.autograd.grad(oracle, embedding)[0]
    torch.testing.assert_close(actual_gradient, oracle_gradient, rtol=1e-5, atol=0.0)


def test_the_card_records_the_implementation_it_now_has() -> None:
    """Keep historical failure and approved scope legible through promotion."""
    text = CARD.read_text(encoding="utf-8")
    assert "`a9fec5cc9623`" in text
    assert "target-preserving" in text
    assert "290000+100*i" in text
    assert "full_arm_embedding_spread" in text
    assert "| Card reviewed (status → `reviewed`) | Claude |" in text
    assert "| [`vicreg.md`](recipes/vicreg.md) | `vicreg` |" in (
        CARD.parents[1] / "RECIPES.md"
    ).read_text(encoding="utf-8")


def test_caller_supplies_both_view_policies_without_recipe_substitution() -> None:
    recipe = build_vicreg(
        _schema(),
        first_transforms=(FeatureCorruption(rate=0.2, columns=None),),
        second_transforms=(FeatureCorruption(rate=0.8, columns=None),),
    )
    assert [view.transforms for view in compile(recipe).plan.views] == [
        ("FeatureCorruption(rate=0.2, columns=all)",),
        ("FeatureCorruption(rate=0.8, columns=all)",),
    ]
