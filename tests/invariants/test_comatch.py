"""Tier 0 — CoMatch memory labels, graph loss, and declarative assembly."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.nn import functional as F
from xty2.components import ProjectionHead
from xty2.components._nn import TORCH_LINEAR_INITIALISATION
from xty2.core import (
    CategoricalTreatment,
    ExternalBatches,
    FeatureSpec,
    OutcomeSpec,
    Port,
    Program,
    PseudoLabelAction,
    Schema,
    State,
    TrainContext,
    XTYBatch,
    compile,
)
from xty2.core.errors import CompileError, LossError
from xty2.core.rows import resolve_rows
from xty2.objectives import (
    CoMatchConfidenceThresholds,
    InfoNCEContrastive,
    MemorySmoothedLabelGraph,
    MemorySmoothedLabels,
    MemorySmoothedPseudoLabelTreatmentNLL,
    PseudoLabelGraphContrastive,
)
from xty2.recipes import comatch
from xty2.recipes.comatch import (
    COMATCH_STEPS,
    CONFIDENCE_THRESHOLDS,
    LABEL_GRAPH,
    PSEUDO_LABEL_TERM,
    STRONG_X,
    WEAK_X,
    WEIGHT_DECAY,
)

ROOT = Path(__file__).resolve().parents[2]
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "comatch.py"
CARD = ROOT / "docs" / "recipes" / "comatch.md"
CLASSES = 2
FEATURES = 6
EMBEDDING = 4


def _schema() -> Schema:
    return Schema(
        features=tuple(
            FeatureSpec(f"x{column}", "continuous") for column in range(FEATURES)
        ),
        treatment_cardinality=CLASSES,
        outcome=OutcomeSpec(),
    )


def _batch(*, hidden_support: bool = False) -> XTYBatch:
    observed = torch.tensor([True, True, False, False, False])
    if hidden_support:
        observed = torch.tensor([True, True, False, False, False, False])
    rows = int(observed.numel())
    return XTYBatch(
        x=torch.randn(rows, FEATURES),
        t=torch.tensor(([0, 1, 0, 1, 0, 1])[:rows]),
        y=torch.randn(rows),
        t_observed=observed,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(100, 100 + rows),
    )


def _graph(**changes: object) -> MemorySmoothedLabelGraph:
    defaults: dict[str, object] = {
        "temperature": 0.2,
        "alpha": 0.9,
        "capacity": 20,
        "thresholds": CoMatchConfidenceThresholds(0.0, 0.8),
        "alignment_window": 3,
        "unsmoothed_steps": 0,
    }
    return MemorySmoothedLabelGraph(**(defaults | changes))  # type: ignore[arg-type]


def _probabilities() -> Tensor:
    return torch.tensor(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.7, 0.3],
        ],
        dtype=torch.float64,
    )


def _embeddings(rows: int = 5) -> Tensor:
    return F.normalize(
        torch.arange(1, rows * EMBEDDING + 1, dtype=torch.float64).reshape(
            rows, EMBEDDING
        ),
        dim=-1,
    )


def _state(
    *,
    weak_logits: Tensor | None = None,
    strong_logits: Tensor | None = None,
    weak_embedding: Tensor | None = None,
    strong_0: Tensor | None = None,
    strong_1: Tensor | None = None,
) -> State:
    probabilities = _probabilities()
    weak_logits = probabilities.log() if weak_logits is None else weak_logits
    strong_logits = (
        torch.tensor(
            [
                [0.3, -0.2],
                [-0.4, 0.5],
                [0.7, -0.1],
                [-0.3, 0.8],
                [0.2, -0.6],
            ],
            dtype=torch.float64,
        )
        if strong_logits is None
        else strong_logits
    )
    weak_embedding = _embeddings() if weak_embedding is None else weak_embedding
    strong_0 = _embeddings().flip(1) if strong_0 is None else strong_0
    strong_1 = _embeddings().roll(1, dims=1) if strong_1 is None else strong_1
    return State(
        {
            WEAK_X: {
                Port.T_GIVEN_X: CategoricalTreatment(weak_logits),
                Port.X_PROJ: weak_embedding,
            },
            STRONG_X[0]: {
                Port.T_GIVEN_X: CategoricalTreatment(strong_logits),
                Port.X_PROJ: strong_0,
            },
            STRONG_X[1]: {Port.X_PROJ: strong_1},
        }
    )


def _objectives(
    graph: MemorySmoothedLabelGraph,
) -> tuple[MemorySmoothedPseudoLabelTreatmentNLL, PseudoLabelGraphContrastive]:
    labels = MemorySmoothedPseudoLabelTreatmentNLL(
        graph=graph,
        target=WEAK_X,
        weak_embedding=WEAK_X,
        prediction=STRONG_X[0],
        num_treatments=CLASSES,
        sharpening="none",
        stop_grad="target",
        name=PSEUDO_LABEL_TERM,
    )
    contrastive = PseudoLabelGraphContrastive(
        graph=graph,
        labels=PSEUDO_LABEL_TERM,
        target=WEAK_X,
        weak_embedding=WEAK_X,
        anchor=STRONG_X[0],
        contrast=STRONG_X[1],
        num_treatments=CLASSES,
    )
    return labels, contrastive


def _context(step: int, memory: MemorySmoothedLabels) -> TrainContext:
    return TrainContext(
        global_step=step,
        schema=_schema(),
        stage="joint_fit",
        objective_states={PSEUDO_LABEL_TERM: memory},
    )


def _card_section_four() -> dict[str, str | dict[str, str]]:
    text = CARD.read_text(encoding="utf-8")
    match = re.search(r"## 4\..*?```yaml\n(.*?)\n```", text, re.DOTALL)
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


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, (ast.If, ast.IfExp, ast.Match)) for node in ast.walk(tree)
    )


def test_the_stage_has_the_five_reviewed_objectives_and_populations() -> None:
    stage = compile(comatch(_schema())).stage("joint_fit")
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        PSEUDO_LABEL_TERM,
        "pseudo_label_graph_contrastive",
        "missing_treatment_marginal_nll",
    ]
    assert [objective.rows for objective in stage.objectives] == [
        ("t_observed",),
        ("t_observed",),
        ("t_missing",),
        ("t_missing",),
        ("t_missing",),
    ]
    assert [objective.reduction for objective in stage.objectives] == [
        "population",
        "mean",
        "mean",
        "mean",
        "population",
    ]


def test_the_shared_policy_and_plan_hold_every_reviewed_scalar() -> None:
    plan = compile(comatch(_schema())).plan
    assert plan.hyperparameters["losses.confidence_threshold"] == (
        CONFIDENCE_THRESHOLDS
    )
    assert plan.hyperparameters["losses.temperature"] == 0.2
    assert plan.hyperparameters["losses.sharpening"] == "none"
    assert plan.hyperparameters["optimisation.batch_size"] == 512
    assert plan.hyperparameters["optimisation.labelled_unlabelled_ratio"] == 7.0
    assert plan.hyperparameters["optimisation.total_steps_or_epochs"] == COMATCH_STEPS
    assert LABEL_GRAPH.unsmoothed_steps == 6
    rendered = plan.render()
    for statement in (
        "raw weak predictions before distribution alignment",
        "FIFO capacity 2560; read before write",
        "steps 0..5 are unsmoothed",
        "both strong embedding realisations train",
    ):
        assert statement in rendered


def test_the_card_and_plan_agree_on_every_value_section_four_states() -> None:
    hyperparameters = compile(comatch(_schema())).plan.hyperparameters
    symbolic = {"architecture.widths_depths": {"K": "2", "X_REPR": "200"}}
    mismatched: list[str] = []
    checked = 0
    for key, stated in _card_section_four().items():
        planned = hyperparameters.get(key)
        if planned is None:
            mismatched.append(f"{key}: absent from plan")
            continue
        if isinstance(stated, str):
            if not isinstance(planned, Mapping) and _rendered(planned) != stated:
                mismatched.append(f"{key}: card {stated!r} vs plan {planned!r}")
            checked += 1
            continue
        assert isinstance(planned, Mapping), f"{key} is scoped in the card only"
        for scope, value in stated.items():
            if scope not in planned:
                mismatched.append(f"{key}[{scope}]: absent from plan")
                continue
            resolved = value
            for symbol, concrete in symbolic.get(key, {}).items():
                resolved = resolved.replace(symbol, concrete)
            if _rendered(planned[scope]) != resolved:
                mismatched.append(
                    f"{key}[{scope}]: card {resolved!r} vs plan {planned[scope]!r}"
                )
            checked += 1
    assert not mismatched, "card and plan disagree: " + "; ".join(mismatched)
    assert checked >= 35


def test_both_terms_can_prepare_q_from_the_inputs_they_declare() -> None:
    labels, graph = _objectives(_graph())
    required = {(Port.T_GIVEN_X, WEAK_X), (Port.X_PROJ, WEAK_X)}
    assert required <= labels.requires
    assert required <= graph.requires
    assert labels.detaches == graph.detaches == frozenset(required)
    assert labels.batch_coupled and graph.batch_coupled


def test_projection_head_uses_the_reference_slope_and_torch_initialisation() -> None:
    recipe = comatch(_schema())
    head = recipe.system["projection_head"]
    assert isinstance(head, ProjectionHead)
    assert head.activation == "leaky_relu:0.1"
    assert head.initialisation == TORCH_LINEAR_INITIALISATION
    activation = next(
        module for module in head.network if isinstance(module, torch.nn.LeakyReLU)
    )
    assert activation.negative_slope == 0.1
    assert WEIGHT_DECAY.on_norm_and_bias is True


def test_distribution_alignment_uses_the_current_batch_but_bank_stores_raw_probs() -> (
    None
):
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    support = resolve_rows(batch, "t_observed")
    state = MemorySmoothedLabels(CLASSES, _graph(unsmoothed_steps=9))
    raw = _probabilities()
    q = state.prepare(
        step=0,
        raw_probabilities=raw,
        weak_embeddings=_embeddings(),
        batch=batch,
        eligible_rows=rows,
        support_rows=support,
    )
    selected = raw.index_select(0, rows)
    expected = selected / selected.mean(dim=0)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(q, expected)
    expected_bank = torch.cat(
        (selected, F.one_hot(batch.t[:2], num_classes=CLASSES).to(raw)), dim=0
    )
    torch.testing.assert_close(state.probabilities, expected_bank)


def test_all_six_zero_based_reference_iterations_are_unsmoothed() -> None:
    graph = _graph(unsmoothed_steps=6, alpha=0.0, capacity=5)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    support = resolve_rows(batch, "t_observed")
    memory = MemorySmoothedLabels(CLASSES, graph)
    raw = _probabilities()
    weak = _embeddings()
    results = []
    for step in range(7):
        results.append(
            memory.prepare(
                step=step,
                raw_probabilities=raw,
                weak_embeddings=weak,
                batch=batch,
                eligible_rows=rows,
                support_rows=support,
            )
        )
    aligned = raw.index_select(0, rows) / raw.index_select(0, rows).mean(dim=0)
    aligned = aligned / aligned.sum(dim=-1, keepdim=True)
    for result in results[:6]:
        torch.testing.assert_close(result, aligned)
    assert not torch.allclose(results[6], aligned)


def test_eq8_matches_a_hand_computed_previous_bank_affinity() -> None:
    graph = _graph(alpha=0.25, capacity=20)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    support = resolve_rows(batch, "t_observed")
    memory = MemorySmoothedLabels(CLASSES, graph)
    raw = _probabilities()
    weak = _embeddings()
    memory.prepare(
        step=0,
        raw_probabilities=raw,
        weak_embeddings=weak,
        batch=batch,
        eligible_rows=rows,
        support_rows=support,
    )
    prior_probabilities = memory.probabilities
    prior_embeddings = memory.embeddings
    q = memory.prepare(
        step=1,
        raw_probabilities=raw,
        weak_embeddings=weak,
        batch=batch,
        eligible_rows=rows,
        support_rows=support,
    )
    selected = raw.index_select(0, rows)
    aligned = selected / selected.mean(dim=0)
    aligned = aligned / aligned.sum(dim=-1, keepdim=True)
    affinity = torch.softmax(
        weak.index_select(0, rows) @ prior_embeddings.T / graph.temperature, dim=-1
    )
    expected = graph.alpha * aligned + (1.0 - graph.alpha) * (
        affinity @ prior_probabilities
    )
    torch.testing.assert_close(q, expected)


def test_fifo_is_bounded_and_preparation_is_idempotent_within_a_step() -> None:
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    support = resolve_rows(batch, "t_observed")
    memory = MemorySmoothedLabels(CLASSES, _graph(capacity=6))
    q0 = memory.prepare(
        step=0,
        raw_probabilities=_probabilities(),
        weak_embeddings=_embeddings(),
        batch=batch,
        eligible_rows=rows,
        support_rows=support,
    )
    bank = memory.probabilities
    q_again = memory.prepare(
        step=0,
        raw_probabilities=_probabilities(),
        weak_embeddings=_embeddings(),
        batch=batch,
        eligible_rows=rows,
        support_rows=support,
    )
    torch.testing.assert_close(q_again, q0)
    torch.testing.assert_close(memory.probabilities, bank)
    memory.prepare(
        step=1,
        raw_probabilities=_probabilities().flip(0),
        weak_embeddings=_embeddings().flip(0),
        batch=batch,
        eligible_rows=rows,
        support_rows=support,
    )
    assert memory.size == 6
    assert memory.row_ids.tolist() == [101, 102, 103, 104, 100, 101]


def test_memory_refuses_a_support_population_containing_hidden_treatments() -> None:
    batch = _batch(hidden_support=True)
    memory = MemorySmoothedLabels(CLASSES, _graph())
    with pytest.raises(LossError, match="support rows must all have observed"):
        memory.prepare(
            step=0,
            raw_probabilities=torch.softmax(torch.randn(6, 2), dim=-1),
            weak_embeddings=_embeddings(6),
            batch=batch,
            eligible_rows=torch.tensor([2, 3]),
            support_rows=torch.tensor([4, 5]),
        )


def test_soft_pseudo_label_loss_matches_eq4_and_detaches_both_weak_inputs() -> None:
    graph = _graph(unsmoothed_steps=9)
    objective, _ = _objectives(graph)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    weak_logits = _probabilities().log().clone().requires_grad_()
    strong_logits = torch.randn(5, 2, dtype=torch.float64, requires_grad=True)
    weak_embedding = _embeddings().clone().requires_grad_()
    memory = MemorySmoothedLabels(CLASSES, graph)
    term = objective.compute(
        _state(
            weak_logits=weak_logits,
            strong_logits=strong_logits,
            weak_embedding=weak_embedding,
        ),
        batch,
        rows,
        _context(0, memory),
    )
    selected = weak_logits.softmax(dim=-1).index_select(0, rows)
    q = selected / selected.mean(dim=0)
    q = (q / q.sum(dim=-1, keepdim=True)).detach()
    expected = (
        -(q * strong_logits.log_softmax(dim=-1).index_select(0, rows))
        .sum(dim=-1)
        .mean()
    )
    torch.testing.assert_close(term.value, expected)
    term.value.backward()  # type: ignore[no-untyped-call]
    assert weak_logits.grad is None
    assert weak_embedding.grad is None
    assert strong_logits.grad is not None and float(strong_logits.grad.abs().sum()) > 0


def test_graph_cross_entropy_matches_eqs_9_to_11_and_trains_both_views() -> None:
    graph = _graph(unsmoothed_steps=9)
    _, objective = _objectives(graph)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    first = _embeddings().flip(1).detach().clone().requires_grad_()
    second = _embeddings().roll(1, dims=1).detach().clone().requires_grad_()
    memory = MemorySmoothedLabels(CLASSES, graph)
    term = objective.compute(
        _state(strong_0=first, strong_1=second),
        batch,
        rows,
        _context(0, memory),
    )
    raw = _probabilities().index_select(0, rows)
    q = raw / raw.mean(dim=0)
    q = q / q.sum(dim=-1, keepdim=True)
    q_graph = q @ q.T
    q_graph = torch.where(q_graph >= graph.thresholds.edge, q_graph, 0.0)
    q_graph.fill_diagonal_(1.0)
    q_graph = q_graph / q_graph.sum(dim=-1, keepdim=True)
    similarities = torch.exp(
        F.normalize(first.index_select(0, rows), dim=-1)
        @ F.normalize(second.index_select(0, rows), dim=-1).T
        / graph.temperature
    )
    z_graph = similarities / similarities.sum(dim=-1, keepdim=True)
    expected = -(q_graph * torch.log(z_graph + 1e-7)).sum(dim=-1).mean()
    torch.testing.assert_close(term.value, expected)
    term.value.backward()  # type: ignore[no-untyped-call]
    assert first.grad is not None and float(first.grad.abs().sum()) > 0
    assert second.grad is not None and float(second.grad.abs().sum()) > 0


def test_at_t_one_the_graph_loss_is_infonce_plus_log_n() -> None:
    thresholds = CoMatchConfidenceThresholds(0.0, 1.0)
    graph = _graph(thresholds=thresholds, unsmoothed_steps=9)
    _, objective = _objectives(graph)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    state = _state()
    actual = objective.compute(
        state,
        batch,
        rows,
        _context(0, MemorySmoothedLabels(CLASSES, graph)),
    ).value
    reference = (
        InfoNCEContrastive(
            port=Port.X_PROJ,
            anchor=STRONG_X[0],
            contrast=STRONG_X[1],
            temperature=graph.temperature,
            rows="t_missing",
        )
        .compute(state, batch, rows, TrainContext(0, _schema()))
        .value
    )
    assert actual == pytest.approx(
        float(reference + math.log(int(rows.numel()))), rel=1e-5
    )


def test_at_alpha_one_q_is_exactly_the_aligned_weak_prediction() -> None:
    graph = _graph(alpha=1.0)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    support = resolve_rows(batch, "t_observed")
    memory = MemorySmoothedLabels(CLASSES, graph)
    for step in range(2):
        q = memory.prepare(
            step=step,
            raw_probabilities=_probabilities(),
            weak_embeddings=_embeddings(),
            batch=batch,
            eligible_rows=rows,
            support_rows=support,
        )
    raw = _probabilities().index_select(0, rows)
    expected = raw / raw.mean(dim=0)
    expected = expected / expected.sum(dim=-1, keepdim=True)
    torch.testing.assert_close(q, expected)


def test_the_two_objectives_are_value_identical_in_either_declaration_order() -> None:
    graph = _graph(unsmoothed_steps=9)
    labels, contrastive = _objectives(graph)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    state = _state()
    traces = []
    for order in ((labels, contrastive), (contrastive, labels)):
        memory = MemorySmoothedLabels(CLASSES, graph)
        context = _context(0, memory)
        values = {
            objective.name: objective.compute(state, batch, rows, context).value
            for objective in order
        }
        traces.append(values)
    for name in (labels.name, contrastive.name):
        torch.testing.assert_close(traces[0][name], traces[1][name], rtol=0, atol=0)


def test_batch_coupling_and_state_refuse_unreviewed_execution_paths() -> None:
    recipe = comatch(_schema())
    stage = recipe.program[0]
    external = replace(
        recipe,
        program=Program((replace(stage, sampler=ExternalBatches()),)),
        data=None,
    )
    with pytest.raises(CompileError, match="batch"):
        compile(external)
    with pytest.raises(CompileError, match="uses cross_fit and holds the stateful"):
        replace(
            stage,
            executor="cross_fit",
            action=PseudoLabelAction(port=Port.T_GIVEN_X),
        )
