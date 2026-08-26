"""Tier 0 — the SCARF recipe, its corruption view and its contrastive loss.

The load-bearing test here is `test_the_loss_is_the_papers_expression`: like
the exact-marginalisation invariant (`FIDELITY.md` §3), it compares the vector
implementation against a brute-force transcription of the published formula,
including the two places SCARF differs from the SimCLR form everyone
half-remembers — the normaliser contains the positive pair, and it carries a
`1/N`.
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xty2.components import ProjectionHead
from xty2.core import (
    DEFAULT,
    CardKeyError,
    CategoricalTreatment,
    CompileError,
    FeatureSpec,
    LossError,
    OutcomeSpec,
    Port,
    PortContractError,
    PreservedField,
    Program,
    Recipe,
    Schema,
    State,
    TrainContext,
    ViewError,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.objectives import InfoNCEContrastive
from xty2.recipes import scarf
from xty2.recipes.scarf import CORRUPTED_X, CORRUPTION_RATE, TEMPERATURE
from xty2.views import FeatureCorruption, ViewSpec

from tests.invariants.conftest import backward

ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "scarf.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "scarf.py"
PRESERVED: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)
ROWS = 12
WIDTH = 5


def _schema(*, derived: bool = False, immutable: bool = True) -> Schema:
    return Schema(
        features=(
            FeatureSpec("mass", "continuous"),
            FeatureSpec("speed", "continuous"),
            FeatureSpec(
                "momentum",
                "continuous",
                derived_from=("mass", "speed") if derived else (),
            ),
            FeatureSpec("site", "categorical", mutable=not immutable),
        ),
        treatment_cardinality=3,
        outcome=OutcomeSpec(),
    )


def _batch(rows: int = ROWS) -> XTYBatch:
    # Every cell distinct, so a value that moved can be traced to the column it
    # was drawn from and a corrupted cell is only ever equal to its original
    # when the donor happened to be its own row.
    x = torch.arange(float(rows * 4)).reshape(rows, 4)
    observed = torch.arange(rows) % 3 == 0
    return XTYBatch(
        x=x,
        t=torch.arange(rows) % 3,
        y=torch.linspace(-1.0, 1.0, rows),
        t_observed=observed,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(rows),
    )


# ---------------------------------------------------------------------------
# The recipe and its plan
# ---------------------------------------------------------------------------


def test_the_recipe_plans_two_stages_and_three_forward_passes() -> None:
    run = compile(scarf(_schema()))
    assert run.graph.names == (
        "mlp_encoder",
        "projection_head",
        "tarnet_head",
        "categorical_propensity",
    )
    assert [stage.name for stage in run.stages] == ["pretrain", "joint_fit"]

    pretrain = run.stage("pretrain")
    assert pretrain.steps == 1_000
    assert pretrain.trainable == ("mlp_encoder", "projection_head")
    assert sorted(str(forward.realisation) for forward in pretrain.passes) == sorted(
        str(realisation) for realisation in (DEFAULT, CORRUPTED_X)
    )
    for forward in pretrain.passes:
        assert forward.components == ("mlp_encoder", "projection_head")

    fit = run.stage("joint_fit")
    assert fit.steps == 3_000
    assert fit.initialise_from == "pretrain"
    assert [str(forward.realisation) for forward in fit.passes] == [str(DEFAULT)]
    assert fit.passes[0].components == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )


def test_the_pretrain_head_is_discarded_by_the_fitting_stage() -> None:
    """ "After pre-training, `g` is discarded" — as a property, not a comment.

    The paper's sentence is only honoured if `projection_head` is absent from
    both halves of the second stage: it must not be trained, and it must not
    even run. Either half alone would be satisfiable by a recipe that still
    carried it.
    """
    fit = compile(scarf(_schema())).stage("joint_fit")
    assert "projection_head" not in fit.trainable
    assert not any("projection_head" in forward.components for forward in fit.passes)


def test_training_the_discarded_head_downstream_is_a_compile_error() -> None:
    """The claim above is enforced by the compiler, not by our restraint."""
    recipe = scarf(_schema())
    fit = recipe.program[1]
    broken = replace(
        recipe,
        program=Program(
            (
                recipe.program[0],
                replace(fit, trainable=(*fit.trainable, "projection_head")),
            )
        ),
    )
    with pytest.raises(CompileError, match="dead weight"):
        compile(broken)


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_each_stage_has_exactly_the_reviewed_objectives() -> None:
    run = compile(scarf(_schema()))
    pretrain = run.stage("pretrain")
    assert [objective.name for objective in pretrain.objectives] == [
        "info_nce_contrastive"
    ]
    assert [objective.rows for objective in pretrain.objectives] == [("all",)]
    # `L_cont` is (1/N) over the batch — the term's own rows, which is `mean`.
    assert [objective.reduction for objective in pretrain.objectives] == ["mean"]

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
    assert [objective.reduction for objective in fit.objectives] == [
        "population",
        "population",
        "population",
    ]


def test_the_pretraining_stage_reads_no_label_of_any_kind() -> None:
    """Section 2: `DESIGN.md` §7.2 is not engaged, as a graph property.

    Pretraining that touched `Y_RAW` would be a treatment representation
    learned from the outcome, which is the shape of the leakage the causal
    guardrail exists for. Here the subgraph simply does not contain it.
    """
    plan = compile(scarf(_schema())).plan
    planned = {component.name: component for component in plan.components}
    for name in ("mlp_encoder", "projection_head"):
        assert not planned[name].reads_raw_outcome
        assert not planned[name].outcome_dependent


def test_the_recipe_declares_one_view_of_one_corruption() -> None:
    plan = compile(scarf(_schema())).plan
    assert [view.name for view in plan.views] == ["corrupted_x"]
    assert [view.transforms for view in plan.views] == [
        ("FeatureCorruption(rate=0.6, columns=all)",)
    ]
    assert plan.views[0].draws == 1
    assert set(plan.views[0].preserves) == PRESERVED


def test_a_schema_with_a_stale_derived_column_is_rejected() -> None:
    with pytest.raises(CompileError, match="derived column"):
        compile(scarf(_schema(derived=True)))


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def _card_section_four() -> dict[str, str | dict[str, str]]:
    """Card §4 as data — the same two-level parse `test_fixmatch.py` uses."""
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
    """The plan's value as §4 writes it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


def test_every_answered_card_key_reaches_the_plan() -> None:
    plan = compile(scarf(_schema())).plan
    answered = set(_card_section_four())
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    # The key this recipe is the first to answer, spelled out so that a card
    # that quietly dropped it would fail here rather than pass a subset.
    assert "losses.temperature" in answered
    assert plan.hyperparameters["losses.temperature"] == {
        "pretrain.info_nce_contrastive": 1.0
    }


def test_every_card_value_the_plan_also_carries_agrees_with_it() -> None:
    """Key presence is not the cross-check; the values are.

    Both halves of a two-stage program are compared, which is where this card
    can drift in a way `fixmatch.md` could not: a learning rate or a step count
    edited for one stage and read as if it applied to both.
    """
    hyperparameters = compile(scarf(_schema())).plan.hyperparameters
    mismatched: list[str] = []
    symbolic = {"architecture.widths_depths": {"K": "3", "X_REPR": "200"}}
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
    assert checked >= 30


# ---------------------------------------------------------------------------
# FeatureCorruption
# ---------------------------------------------------------------------------


def _corrupt(
    batch: XTYBatch, schema: Schema, *, rate: float = CORRUPTION_RATE, key: int = 7
) -> XTYBatch:
    view = ViewSpec(
        name="corrupted_x",
        transforms=(FeatureCorruption(rate=rate, columns=None),),
        preserves=PRESERVED,
    )
    return view.apply(batch, schema, rng_key=key)


def test_the_count_of_corrupted_columns_is_floor_rate_times_mutable() -> None:
    """`q` is a count, not a per-cell rate, and immutable columns are not in M."""
    schema = _schema()
    transform = FeatureCorruption(rate=CORRUPTION_RATE, columns=None)
    # Three mutable columns of four: floor(0.6 * 3) = 1.
    assert transform.corrupted_per_row(schema) == 1
    assert FeatureCorruption(rate=1.0).corrupted_per_row(schema) == 3
    assert FeatureCorruption(rate=0.3).corrupted_per_row(schema) == 0
    assert (
        FeatureCorruption(rate=CORRUPTION_RATE).corrupted_per_row(
            _schema(immutable=False)
        )
        == 2
    )


def test_no_row_has_more_corrupted_cells_than_q() -> None:
    schema = _schema(immutable=False)
    batch = _batch()
    result = _corrupt(batch, schema)
    changed = (result.x != batch.x).sum(dim=1)
    assert int(changed.max()) <= 2
    # A donor is drawn per cell and may be the row itself, so `q` is an upper
    # bound rather than an equality — but with 12 rows it is nearly attained.
    assert float(changed.float().mean()) > 1.5


def _corrupted_columns(before: XTYBatch, after: XTYBatch) -> list[frozenset[int]]:
    """Which columns actually moved, per row."""
    moved = before.x != after.x
    return [frozenset(torch.nonzero(row).flatten().tolist()) for row in moved]


def _donor_rows(after: XTYBatch, columns: int = 4) -> list[frozenset[int]]:
    """Which rows each corrupted cell was drawn from, per row.

    Only decodable because `_batch` lays `x` out as `arange`, so a cell's value
    names the row it came from: `value = 4 * donor + column`.
    """
    donors: list[frozenset[int]] = []
    for index, row in enumerate(after.x):
        sources = {
            int((value - column) // columns)
            for column, value in enumerate(row.tolist())
        }
        donors.append(frozenset(sources - {index}))
    return donors


def test_the_corrupted_columns_are_drawn_per_row_and_not_per_batch() -> None:
    """Card §7: "independently per row [...] would be invisible in a diff".

    It is invisible in a diff, so it is asserted here instead. A per-batch mask
    — one column choice reused for every row, the cheaper implementation —
    satisfies every other test in this file, including the `q` bound: it
    corrupts exactly `q` columns per row and stays inside the column's support.
    What it cannot do is corrupt *different* columns in different rows.
    """
    schema = _schema(immutable=False)
    batch = _batch(rows=64)
    result = _corrupt(batch, schema, rate=0.3)
    per_row = _corrupted_columns(batch, result)
    assert all(len(columns) <= 1 for columns in per_row)  # q = floor(0.3*4) = 1
    # Every column is chosen by some row, which one shared mask cannot manage,
    # and the rows do not all agree, which is the same fact stated twice
    # because each half fails a different wrong sampler.
    assert len({column for columns in per_row for column in columns}) == 4
    assert len(set(per_row)) > 1


def test_each_corrupted_cell_draws_its_own_donor_row() -> None:
    """Card §7: one donor row per row "would preserve that row's cross-feature
    dependence, which is exactly what the corruption is meant to destroy".

    The cheaper implementation draws one donor per row and broadcasts it. It
    passes every other test here — the values are still real values of their
    columns, still exactly `q` of them — and it copies a *slice of another row*
    rather than a draw per feature, which is a different augmentation.
    """
    schema = _schema(immutable=False)
    batch = _batch(rows=64)
    result = _corrupt(batch, schema, rate=1.0, key=3)
    multi_donor = [row for row in _donor_rows(result) if len(row) > 1]
    # With four columns corrupted per row and 64 candidate donors, a per-row
    # donor gives every row exactly one source; independent draws give almost
    # every row four.
    assert len(multi_donor) > 32


def test_every_corrupted_value_is_one_the_column_actually_took() -> None:
    """The property that makes this transform safe on physical data.

    No clamp, no fill value, no bound arithmetic: a corrupted cell holds a
    value that column holds somewhere else in the same batch, so bounds, kinds
    and any implicit support constraint hold by construction.
    """
    schema = _schema(immutable=False)
    batch = _batch()
    result = _corrupt(batch, schema)
    for column in range(4):
        support = set(batch.x[:, column].tolist())
        assert set(result.x[:, column].tolist()) <= support


def test_an_immutable_column_is_never_corrupted() -> None:
    schema = _schema()
    batch = _batch()
    result = _corrupt(batch, schema, rate=1.0)
    site = schema.index_of("site")
    assert torch.equal(result.x[:, site], batch.x[:, site])
    assert FeatureCorruption(rate=1.0).affected_columns(schema) == frozenset(
        {"mass", "speed", "momentum"}
    )


def test_a_rate_that_floors_to_zero_columns_changes_nothing() -> None:
    schema = _schema()
    batch = _batch()
    transform = FeatureCorruption(rate=0.3, columns=None)
    assert transform.affected_columns(schema) == frozenset()
    result = transform.apply(batch, schema, generator=torch.Generator().manual_seed(0))
    assert torch.equal(result.x, batch.x)


def test_the_view_is_deterministic_in_its_key_and_functional() -> None:
    schema = _schema(immutable=False)
    batch = _batch()
    once = _corrupt(batch, schema, key=11)
    again = _corrupt(batch, schema, key=11)
    other = _corrupt(batch, schema, key=12)
    assert torch.equal(once.x, again.x)
    assert not torch.equal(once.x, other.x)
    assert torch.equal(batch.x, _batch().x)
    assert torch.equal(once.t, batch.t)
    assert torch.equal(once.y, batch.y)


def test_a_single_row_batch_can_only_donate_to_itself() -> None:
    """Not a curiosity: it is the degenerate end of the batch-marginal debt.

    Card §5 deviation 2 takes the marginal from the batch. At `B = 1` that
    marginal is one point, so the corruption is the identity — which is the
    honest behaviour, and it is asserted rather than left to be discovered as a
    silently inert augmentation.
    """
    schema = _schema(immutable=False)
    batch = _batch(rows=1)
    assert torch.equal(_corrupt(batch, schema).x, batch.x)


def test_the_transform_rejects_settings_that_are_not_meaningful() -> None:
    with pytest.raises(ViewError, match="must be in"):
        FeatureCorruption(rate=1.5)
    with pytest.raises(ViewError, match="must be a number"):
        FeatureCorruption(rate="0.6")  # type: ignore[arg-type]
    with pytest.raises(ViewError, match="cannot be empty"):
        FeatureCorruption(rate=0.6, columns=())
    with pytest.raises(ViewError, match="duplicates"):
        FeatureCorruption(rate=0.6, columns=("mass", "mass"))
    with pytest.raises(ViewError, match="unknown column"):
        FeatureCorruption(rate=0.6, columns=("weight",)).validate(_schema())


# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------


def test_the_projection_head_emits_unit_vectors_on_x_proj() -> None:
    head = ProjectionHead(
        representation_dim=WIDTH,
        widths=(8, 6),
        activation="relu",
        normalisation="row_l2",
        dropout=0.0,
        initialisation="torch Linear default Kaiming-uniform",
    )
    assert head.requires == frozenset({Port.X_REPR})
    assert head.provides == frozenset({Port.X_PROJ})
    from xty2.core import PortView

    ports = PortView(
        {Port.X_REPR: torch.randn(ROWS, WIDTH)},
        declared=frozenset({Port.X_REPR}),
        component="projection_head",
    )
    embedding = head.forward(ports)[Port.X_PROJ]
    assert isinstance(embedding, torch.Tensor)
    norms = embedding.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


# ---------------------------------------------------------------------------
# InfoNCEContrastive
# ---------------------------------------------------------------------------


def _objective(**overrides: object) -> InfoNCEContrastive:
    defaults: dict[str, object] = {
        "port": Port.X_PROJ,
        "anchor": DEFAULT,
        "contrast": CORRUPTED_X,
        "temperature": TEMPERATURE,
    }
    return InfoNCEContrastive(**(defaults | overrides))  # type: ignore[arg-type]


def _state(anchor: torch.Tensor, contrast: torch.Tensor) -> State:
    return State(
        {
            DEFAULT: {Port.X_PROJ: anchor},
            CORRUPTED_X: {Port.X_PROJ: contrast},
        }
    )


def _reference(
    anchor: torch.Tensor, contrast: torch.Tensor, *, temperature: float
) -> float:
    """Algorithm 1's expression, transcribed as a loop over `i` and `k`."""
    rows = anchor.shape[0]
    total = 0.0
    for i in range(rows):
        similarities = [
            float(
                torch.dot(anchor[i], contrast[k])
                / (anchor[i].norm() * contrast[k].norm())
            )
            for k in range(rows)
        ]
        denominator = sum(math.exp(s / temperature) for s in similarities) / rows
        total += -math.log(math.exp(similarities[i] / temperature) / denominator)
    return total / rows


def test_the_loss_is_the_papers_expression() -> None:
    torch.manual_seed(3)
    anchor = torch.randn(ROWS, WIDTH)
    contrast = torch.randn(ROWS, WIDTH)
    batch = _batch()
    rows = torch.arange(ROWS)
    for temperature in (0.5, 1.0, 2.0):
        term = _objective(temperature=temperature).compute(
            _state(anchor, contrast),
            batch,
            rows,
            TrainContext(global_step=0, schema=_schema()),
        )
        expected = _reference(anchor, contrast, temperature=temperature)
        assert term.n == ROWS
        assert float(term.value) == pytest.approx(expected, abs=1e-5)


def test_the_normaliser_carries_the_papers_one_over_n() -> None:
    """The `1/N` SimCLR does not have, worth exactly `log n` on the value.

    It contributes no gradient, which is why it is the kind of detail a
    reimplementation drops. Dropping it would make every logged contrastive
    value incomparable with the paper's, and this is what would notice.
    """
    torch.manual_seed(5)
    anchor = torch.randn(ROWS, WIDTH)
    contrast = torch.randn(ROWS, WIDTH)
    term = _objective().compute(
        _state(anchor, contrast),
        _batch(),
        torch.arange(ROWS),
        TrainContext(global_step=0, schema=_schema()),
    )
    normalised = torch.nn.functional.normalize(anchor, dim=-1) @ (
        torch.nn.functional.normalize(contrast, dim=-1).transpose(0, 1)
    )
    without = (
        normalised.logsumexp(dim=-1) - normalised.diagonal()
    ).mean() / TEMPERATURE
    assert float(term.value) == pytest.approx(float(without) - math.log(ROWS), abs=1e-5)


def test_a_collapsed_representation_sits_exactly_at_the_papers_zero() -> None:
    """Every embedding in one direction gives `s_ij = 1` and a loss of 0.

    A single loss number cannot distinguish that from progress, which is why
    the objective also logs alignment and uniformity — here they are equal, and
    the smoke test is what watches them separate.
    """
    collapsed = torch.ones(ROWS, WIDTH)
    term = _objective().compute(
        _state(collapsed, collapsed.clone()),
        _batch(),
        torch.arange(ROWS),
        TrainContext(global_step=0, schema=_schema()),
    )
    assert float(term.value) == pytest.approx(0.0, abs=1e-6)
    assert term.diagnostics["alignment"] == pytest.approx(1.0, abs=1e-6)
    assert term.diagnostics["uniformity"] == pytest.approx(1.0, abs=1e-6)


def test_discriminating_rows_scores_below_the_collapsed_floor() -> None:
    identity = torch.eye(ROWS, WIDTH * 3)
    term = _objective().compute(
        _state(identity, identity.clone()),
        _batch(),
        torch.arange(ROWS),
        TrainContext(global_step=0, schema=_schema()),
    )
    assert float(term.value) < 0.0
    assert term.diagnostics["alignment"] == pytest.approx(1.0, abs=1e-6)
    assert term.diagnostics["uniformity"] == pytest.approx(0.0, abs=1e-6)


def test_the_negatives_are_the_eligible_rows_and_only_those() -> None:
    """§3.2: the row set is both the anchors and the candidates.

    Changing the embeddings of rows this term is not entitled to must not move
    its value at all — otherwise a `t_missing` contrastive term would be
    contrasting against rows outside its declared population.
    """
    torch.manual_seed(9)
    anchor = torch.randn(ROWS, WIDTH)
    contrast = torch.randn(ROWS, WIDTH)
    rows = torch.arange(0, ROWS, 2)
    context = TrainContext(global_step=0, schema=_schema())
    objective = _objective(rows="t_observed")
    before = objective.compute(_state(anchor, contrast), _batch(), rows, context)

    disturbed_anchor = anchor.clone()
    disturbed_contrast = contrast.clone()
    disturbed_anchor[1::2] = 100.0
    disturbed_contrast[1::2] = -100.0
    after = objective.compute(
        _state(disturbed_anchor, disturbed_contrast), _batch(), rows, context
    )
    assert float(before.value) == pytest.approx(float(after.value), abs=1e-6)
    assert before.n == after.n == rows.numel()
    # And it is the *paper's* loss over those rows alone, `N` being their count.
    expected = _reference(
        anchor.index_select(0, rows),
        contrast.index_select(0, rows),
        temperature=TEMPERATURE,
    )
    assert float(before.value) == pytest.approx(expected, abs=1e-5)


def test_no_eligible_rows_returns_the_zero_term() -> None:
    term = _objective().compute(
        _state(torch.randn(ROWS, WIDTH), torch.randn(ROWS, WIDTH)),
        _batch(),
        torch.zeros(0, dtype=torch.long),
        TrainContext(global_step=0, schema=_schema()),
    )
    assert term.n == 0
    assert float(term.value) == 0.0
    assert not term.diagnostics


def test_both_branches_receive_a_gradient() -> None:
    """SCARF is not BYOL: `detaches` is empty and both sides are descended."""
    objective = _objective()
    assert objective.detaches == frozenset()
    assert objective.requires == frozenset(
        {(Port.X_PROJ, DEFAULT), (Port.X_PROJ, CORRUPTED_X)}
    )
    anchor = torch.randn(ROWS, WIDTH, requires_grad=True)
    contrast = torch.randn(ROWS, WIDTH, requires_grad=True)
    term = objective.compute(
        _state(anchor, contrast),
        _batch(),
        torch.arange(ROWS),
        TrainContext(global_step=0, schema=_schema()),
    )
    backward(term.value)
    assert anchor.grad is not None and float(anchor.grad.abs().sum()) > 0.0
    assert contrast.grad is not None and float(contrast.grad.abs().sum()) > 0.0


def test_the_objective_rejects_what_it_cannot_mean() -> None:
    with pytest.raises(CardKeyError, match="no usable default"):
        InfoNCEContrastive(port=Port.X_PROJ, anchor=DEFAULT, contrast=CORRUPTED_X)
    with pytest.raises(LossError, match="with itself"):
        _objective(contrast=DEFAULT)
    with pytest.raises(LossError, match="must be positive"):
        _objective(temperature=0.0)
    with pytest.raises(LossError, match="carries treatment_distribution"):
        _objective(port=Port.T_GIVEN_X)
    with pytest.raises(PortContractError, match="embedding tensor"):
        _objective().compute(
            State(
                {
                    DEFAULT: {Port.X_PROJ: CategoricalTreatment(torch.randn(ROWS, 3))},
                    CORRUPTED_X: {Port.X_PROJ: torch.randn(ROWS, WIDTH)},
                }
            ),
            _batch(),
            torch.arange(ROWS),
            TrainContext(global_step=0, schema=_schema()),
        )


def test_the_plan_shows_the_arithmetic_no_other_field_reveals() -> None:
    stage = compile(scarf(_schema())).stage("pretrain")
    details = stage.objectives[0].plan_details
    assert "similarity = cosine(anchor row, contrast row)" in details
    assert any("negatives = the other eligible rows" in line for line in details)
    assert any("(1/n)" in line for line in details)
    assert f"anchor rows (s_ij row index) = {DEFAULT}" in details
    assert f"contrast columns (s_ij column index) = {CORRUPTED_X}" in details


def test_swapping_the_anchor_and_the_contrast_changes_the_plan() -> None:
    """`s_ij` is not symmetric, so its two sides are not interchangeable.

    `requires` is a set and the plan renders it in a canonical order, so before
    the two role lines existed, a recipe anchored on the *corrupted* row — a
    different loss, worth about 4% on the Tier 0 fixture — printed a
    byte-identical plan and hashed to the same digest. That is the provenance
    collision `DESIGN.md` §4 makes `plan_details` responsible for, and this is
    the test that would notice it coming back.
    """
    recipe = scarf(_schema())
    stage = recipe.program[0]
    objective = stage.objectives[0].objective
    assert isinstance(objective, InfoNCEContrastive)
    swapped = Weighted(
        replace(objective, anchor=CORRUPTED_X, contrast=DEFAULT),
        weight=1.0,
        reduction="mean",
    )
    other = replace(
        recipe,
        program=Program((replace(stage, objectives=(swapped,)), recipe.program[1])),
    )
    assert compile(other).plan.render() != compile(recipe).plan.render()
    assert compile(other).plan.digest != compile(recipe).plan.digest

    # And the two really are different arithmetic, which is what makes the
    # digest difference worth having.
    torch.manual_seed(4)
    anchor = torch.randn(ROWS, WIDTH)
    contrast = torch.randn(ROWS, WIDTH)
    context = TrainContext(global_step=0, schema=_schema())
    forward = _objective().compute(
        _state(anchor, contrast), _batch(), torch.arange(ROWS), context
    )
    backward = _objective(anchor=CORRUPTED_X, contrast=DEFAULT).compute(
        _state(anchor, contrast), _batch(), torch.arange(ROWS), context
    )
    assert float(forward.value) != pytest.approx(float(backward.value), abs=1e-4)


def test_a_single_eligible_row_reports_no_uniformity() -> None:
    """There is no off-diagonal to average, so the key is omitted.

    Filling it with the diagonal would log `alignment == uniformity`, which is
    the signature the smoke test reads as collapse, on a batch that has said
    nothing at all.
    """
    term = _objective().compute(
        _state(torch.randn(ROWS, WIDTH), torch.randn(ROWS, WIDTH)),
        _batch(),
        torch.zeros(1, dtype=torch.long),
        TrainContext(global_step=0, schema=_schema()),
    )
    assert term.n == 1
    assert set(term.diagnostics) == {"alignment"}
    # One row against itself: the positive is the whole normaliser, so the
    # paper's expression is exactly 0 and no NaN reaches the mixer.
    assert float(term.value) == pytest.approx(0.0, abs=1e-6)


def test_two_temperatures_do_not_share_a_provenance_identity() -> None:
    """A card key changes the plan, and therefore the digest."""
    recipe = scarf(_schema())
    stage = recipe.program[0]
    objective = stage.objectives[0].objective
    assert isinstance(objective, InfoNCEContrastive)
    warmer = Weighted(replace(objective, temperature=0.5), weight=1.0, reduction="mean")
    other = replace(
        recipe,
        program=Program((replace(stage, objectives=(warmer,)), recipe.program[1])),
    )
    assert compile(recipe).plan.digest != compile(other).plan.digest


def _recipe_with(schema: Schema) -> Recipe:
    return scarf(schema)


def test_the_recipe_is_a_function_of_the_schema() -> None:
    plan = compile(_recipe_with(_schema(immutable=False))).plan
    assert plan.views[0].affected_columns == ("mass", "momentum", "site", "speed")
