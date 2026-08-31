"""Tier 0 — SimMatch propagation, the labelled bank, and the declarative assembly.

The twelve assertions `docs/recipes/simmatch.md` §6.2 predeclares, in its order.
"""

from __future__ import annotations

import ast
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
    Dataset,
    DataSpec,
    ExternalBatches,
    FeatureSpec,
    MissingnessSpec,
    OutcomeSpec,
    Port,
    PreprocessSpec,
    Program,
    PseudoLabelAction,
    Recipe,
    Schema,
    SplitSpec,
    State,
    TrainContext,
    XTYBatch,
    compile,
)
from xty2.core.data import TrainingPopulation
from xty2.core.errors import CompileError, LossError
from xty2.core.rows import resolve_rows
from xty2.objectives import (
    LabeledMemoryInstanceConsistency,
    LabeledSimilarityMemory,
    PropagatedTargets,
    SimilarityMatchingSpec,
    SimilarityMatchingTemperatures,
    SimilarityMatchingTreatmentNLL,
)
from xty2.recipes import simmatch
from xty2.recipes.simmatch import (
    INSTANCE_TEMPERATURES,
    MEMORY_TERM,
    SIMILARITY_MATCHING,
    SIMMATCH_STEPS,
    STRONG_X,
    WEAK_X,
    WEIGHT_DECAY,
)
from xty2.training.loading import build_population

ROOT = Path(__file__).resolve().parents[2]
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "simmatch.py"
CARD = ROOT / "docs" / "recipes" / "simmatch.md"
CLASSES = 2
FEATURES = 6
EMBEDDING = 4
INSTANCE_TERM = "labeled_memory_instance_consistency"
SLOT_IDS = torch.tensor([100, 101, 102, 103])
SLOT_LABELS = torch.tensor([0, 1, 0, 1])


def _schema() -> Schema:
    return Schema(
        features=tuple(
            FeatureSpec(f"x{column}", "continuous") for column in range(FEATURES)
        ),
        treatment_cardinality=CLASSES,
        outcome=OutcomeSpec(),
    )


def _batch(*, hidden_support: bool = False, order: Tensor | None = None) -> XTYBatch:
    """Four observed support rows and four rows with a hidden treatment."""
    observed = torch.tensor([True] * 4 + [False] * 4)
    if hidden_support:
        observed = torch.tensor([True] * 3 + [False] * 5)
    batch = XTYBatch(
        x=torch.arange(8 * FEATURES, dtype=torch.float64).reshape(8, FEATURES) / 48.0,
        t=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        y=torch.linspace(-1.0, 1.0, 8, dtype=torch.float64),
        t_observed=observed,
        y_observed=torch.ones(8, dtype=torch.bool),
        row_id=torch.arange(100, 108),
    )
    if order is None:
        return batch
    return XTYBatch(
        x=batch.x.index_select(0, order),
        t=batch.t.index_select(0, order),
        y=batch.y.index_select(0, order),
        t_observed=batch.t_observed.index_select(0, order),
        y_observed=batch.y_observed.index_select(0, order),
        row_id=batch.row_id.index_select(0, order),
    )


def _spec(**changes: object) -> SimilarityMatchingSpec:
    defaults: dict[str, object] = {
        "temperatures": SimilarityMatchingTemperatures(
            instance_weak=0.5, instance_strong=0.25
        ),
        "alpha": 0.75,
        "memory_momentum": 0.5,
        "alignment_window": 3,
        "warmup_steps": 0,
        "threshold": 0.0,
        "unfold": True,
    }
    return SimilarityMatchingSpec(**(defaults | changes))  # type: ignore[arg-type]


def _memory(spec: SimilarityMatchingSpec) -> LabeledSimilarityMemory:
    return LabeledSimilarityMemory(
        classes=CLASSES, spec=spec, slot_ids=SLOT_IDS, labels=SLOT_LABELS
    )


def _probabilities() -> Tensor:
    return torch.tensor(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.9, 0.1],
            [0.4, 0.6],
            [0.7, 0.3],
            [0.25, 0.75],
            [0.55, 0.45],
            [0.1, 0.9],
        ],
        dtype=torch.float64,
    )


def _embeddings(rows: int = 8, *, shift: int = 0) -> Tensor:
    values = torch.arange(
        1 + shift, rows * EMBEDDING + 1 + shift, dtype=torch.float64
    ).reshape(rows, EMBEDDING)
    return F.normalize(values.sin(), dim=-1)


def _aligned(raws: list[Tensor], window: int) -> Tensor:
    """`DA(p^w)` for the last entry of `raws`, over a `window`-batch history."""
    marginals = [raw.mean(dim=0) for raw in raws][-window:]
    marginal = torch.stack(marginals).mean(dim=0)
    aligned = raws[-1] / marginal
    return aligned / aligned.sum(dim=-1, keepdim=True)


def _state(
    *,
    weak_logits: Tensor | None = None,
    strong_logits: Tensor | None = None,
    weak_embedding: Tensor | None = None,
    strong_embedding: Tensor | None = None,
) -> State:
    probabilities = _probabilities()
    weak_logits = probabilities.log() if weak_logits is None else weak_logits
    strong_logits = (
        (probabilities.roll(1, dims=0)).log()
        if strong_logits is None
        else strong_logits
    )
    weak_embedding = _embeddings() if weak_embedding is None else weak_embedding
    strong_embedding = (
        _embeddings(shift=7) if strong_embedding is None else strong_embedding
    )
    return State(
        {
            WEAK_X: {
                Port.T_GIVEN_X: CategoricalTreatment(weak_logits),
                Port.X_PROJ: weak_embedding,
            },
            STRONG_X: {
                Port.T_GIVEN_X: CategoricalTreatment(strong_logits),
                Port.X_PROJ: strong_embedding,
            },
        }
    )


def _objectives(
    spec: SimilarityMatchingSpec,
) -> tuple[SimilarityMatchingTreatmentNLL, LabeledMemoryInstanceConsistency]:
    semantic = SimilarityMatchingTreatmentNLL(
        spec=spec,
        target=WEAK_X,
        weak_embedding=WEAK_X,
        prediction=STRONG_X,
        num_treatments=CLASSES,
        sharpening="none",
        stop_grad="target",
        name=MEMORY_TERM,
    )
    instance = LabeledMemoryInstanceConsistency(
        spec=spec,
        owner=MEMORY_TERM,
        target=WEAK_X,
        weak_embedding=WEAK_X,
        prediction=STRONG_X,
        num_treatments=CLASSES,
    )
    return semantic, instance


def _context(step: int, memory: LabeledSimilarityMemory) -> TrainContext:
    return TrainContext(
        global_step=step,
        schema=_schema(),
        stage="joint_fit",
        objective_states={MEMORY_TERM: memory},
    )


def _prepare(
    memory: LabeledSimilarityMemory,
    *,
    step: int,
    batch: XTYBatch | None = None,
    probabilities: Tensor | None = None,
    embeddings: Tensor | None = None,
) -> PropagatedTargets:
    batch = _batch() if batch is None else batch
    return memory.prepare(
        step=step,
        raw_probabilities=_probabilities() if probabilities is None else probabilities,
        weak_embeddings=_embeddings() if embeddings is None else embeddings,
        batch=batch,
        eligible_rows=resolve_rows(batch, "t_missing"),
        support_rows=resolve_rows(batch, "t_observed"),
    )


def _population(batch: XTYBatch | None = None) -> TrainingPopulation:
    rows = _batch() if batch is None else batch
    return build_population(
        Dataset(
            schema=_schema(),
            rows=rows,
            assignments={"train": torch.arange(rows.batch_size)},
        ),
        DataSpec(
            split=SplitSpec(protocol="tier 0 fixture", train="train"),
            preprocess=PreprocessSpec(features="none", outcome="none"),
            missingness=MissingnessSpec(mechanism="observed"),
        ),
        seed=0,
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
            answered[key] = value.strip().strip('"')
        elif indent == 4 and key:
            nested = answered.get(key)
            if not isinstance(nested, dict):
                nested = {}
                answered[key] = nested
            nested[name] = value.strip().strip('"')
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
    stage = compile(simmatch(_schema())).stage("joint_fit")
    assert [objective.name for objective in stage.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        MEMORY_TERM,
        INSTANCE_TERM,
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


def test_the_card_and_plan_agree_on_every_value_section_four_states() -> None:
    """§6.2 invariant 12, first half: every non-`n/a` key reaches the plan."""
    hyperparameters = compile(simmatch(_schema())).plan.hyperparameters
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
    assert checked >= 60


def test_the_plan_prints_the_five_values_no_card_key_names() -> None:
    """§6.2 invariant 12, second half. `K` is a population fact, not a plan one."""
    plan = compile(simmatch(_schema())).plan
    assert plan.hyperparameters["losses.temperature"] == INSTANCE_TEMPERATURES
    assert plan.hyperparameters["losses.confidence_threshold"] == 0.95
    assert plan.hyperparameters["optimisation.total_steps_or_epochs"] == SIMMATCH_STEPS
    rendered = plan.render()
    for statement in (
        "hat p = 0.9 DA(p^w) + 0.1 aggregate(q^w) (eq. 10)",
        "hat q = normalize(q^w * unfold(DA(p^w))) (eqs. 7, 8); unfolding on",
        "memory update = normalize(0.7 * previous slot + 0.3 * current detached "
        "weak embedding), replacing outright on a slot's first observation",
        "instance temperatures = simmatch(instance_weak=0.1, instance_strong=0.1)",
        "distribution alignment = unweighted mean of last 32 current-inclusive",
        "propagation and eq. (5) begin at step 2, and not before every slot is filled",
        "memory = one slot per observed training row_id",
        f"memory owner = {MEMORY_TERM}",
    ):
        assert statement in rendered


def test_the_projection_head_is_the_sources_two_layer_normalised_head() -> None:
    head = simmatch(_schema()).system["projection_head"]
    assert isinstance(head, ProjectionHead)
    assert head.widths == (200, 128)
    assert head.activation == "relu"
    assert head.normalisation == "row_l2"
    assert head.initialisation == TORCH_LINEAR_INITIALISATION
    assert WEIGHT_DECAY.on_norm_and_bias is False


def test_equations_seven_to_ten_match_hand_computed_tensors() -> None:
    """§6.2 invariant 1, on a hand-built two-class, four-slot bank."""
    spec = _spec()
    memory = _memory(spec)
    batch = _batch()
    raw = _probabilities()
    weak = _embeddings()
    _prepare(memory, step=0)  # fills every slot; nothing propagates yet
    bank = memory.features
    targets = _prepare(memory, step=1)

    rows = resolve_rows(batch, "t_missing")
    eligible = raw.index_select(0, rows)
    aligned = _aligned([eligible, eligible], spec.alignment_window)
    similarity = torch.softmax(
        weak.index_select(0, rows) @ bank.T / spec.temperatures.instance_weak, dim=-1
    )
    unfolded = aligned.index_select(1, SLOT_LABELS)
    expected_q = similarity * unfolded
    expected_q = expected_q / expected_q.sum(dim=-1, keepdim=True)
    expected_agg = torch.zeros_like(aligned).index_add(1, SLOT_LABELS, similarity)
    expected_p = spec.alpha * aligned + (1.0 - spec.alpha) * expected_agg

    assert targets.propagated and targets.coverage == 1.0
    instance = targets.instance
    assert instance is not None
    torch.testing.assert_close(instance, expected_q)
    torch.testing.assert_close(targets.semantic, expected_p)
    torch.testing.assert_close(
        instance.sum(dim=-1), torch.ones(int(rows.numel()), dtype=torch.float64)
    )
    torch.testing.assert_close(
        targets.semantic.sum(dim=-1), torch.ones(int(rows.numel()), dtype=torch.float64)
    )


def test_equation_nine_aggregates_the_original_weak_similarity() -> None:
    """§6.2 invariant 2: `hat q`'s calibration never reaches `q^agg`."""
    unfolded = _memory(_spec(unfold=True))
    plain = _memory(_spec(unfold=False))
    semantics: list[Tensor] = []
    instances: list[Tensor] = []
    for memory in (unfolded, plain):
        _prepare(memory, step=0)
        targets = _prepare(memory, step=1)
        assert targets.instance is not None
        semantics.append(targets.semantic)
        instances.append(targets.instance)
    torch.testing.assert_close(semantics[0], semantics[1], rtol=0, atol=0)
    assert not torch.allclose(instances[0], instances[1])


def test_slots_are_keyed_by_row_identity_and_never_by_batch_order() -> None:
    """§6.2 invariant 3."""
    straight = _memory(_spec())
    shuffled = _memory(_spec())
    order = torch.tensor([3, 1, 0, 2, 7, 5, 4, 6])
    permuted = _batch(order=order)
    _prepare(straight, step=0)
    _prepare(
        shuffled,
        step=0,
        batch=permuted,
        probabilities=_probabilities().index_select(0, order),
        embeddings=_embeddings().index_select(0, order),
    )
    torch.testing.assert_close(straight.features, shuffled.features, rtol=0, atol=0)


def test_a_hidden_treatment_or_a_stranger_row_never_reaches_a_slot() -> None:
    """§6.2 invariant 3, continued."""
    memory = _memory(_spec())
    hidden = _batch(hidden_support=True)
    with pytest.raises(LossError, match="must all have observed"):
        memory.prepare(
            step=0,
            raw_probabilities=_probabilities(),
            weak_embeddings=_embeddings(),
            batch=hidden,
            eligible_rows=resolve_rows(hidden, "t_missing"),
            # Row 3 is `t_missing` here; naming it a support row is the leak.
            support_rows=torch.tensor([0, 1, 2, 3]),
        )
    stranger = _batch()
    with pytest.raises(LossError, match="have no slot"):
        LabeledSimilarityMemory(
            classes=CLASSES,
            spec=_spec(),
            slot_ids=torch.tensor([100, 101]),
            labels=torch.tensor([0, 1]),
        ).prepare(
            step=0,
            raw_probabilities=_probabilities(),
            weak_embeddings=_embeddings(),
            batch=stranger,
            eligible_rows=resolve_rows(stranger, "t_missing"),
            support_rows=resolve_rows(stranger, "t_observed"),
        )


def test_the_slot_key_space_itself_must_be_sorted_and_distinct() -> None:
    """§6.2 invariant 3: one row, one slot, decided by identity alone."""
    with pytest.raises(LossError, match="sorted, distinct"):
        LabeledSimilarityMemory(
            classes=CLASSES,
            spec=_spec(),
            slot_ids=torch.tensor([101, 100, 102, 103]),
            labels=SLOT_LABELS,
        )


def test_a_repeated_support_identity_is_rejected_rather_than_written_twice() -> None:
    """§6.2 invariant 3, continued: one slot, one write per step."""
    memory = _memory(_spec())
    batch = _batch()
    with pytest.raises(LossError, match="repeat a row_id"):
        memory.prepare(
            step=0,
            raw_probabilities=_probabilities(),
            weak_embeddings=_embeddings(),
            batch=batch,
            eligible_rows=resolve_rows(batch, "t_missing"),
            support_rows=torch.tensor([0, 0, 1, 2]),
        )


def test_the_bank_is_read_before_it_is_written_and_mixes_only_after_filling() -> None:
    """§6.2 invariant 4, and deviation 11's fill-then-mix update."""
    spec = _spec()
    memory = _memory(spec)
    weak = _embeddings()
    support = resolve_rows(_batch(), "t_observed")
    first = _prepare(memory, step=0)
    # Nothing was in the bank when step 0's targets were built.
    assert first.coverage == 0.0 and not first.propagated
    torch.testing.assert_close(
        memory.features, F.normalize(weak.index_select(0, support), dim=-1)
    )
    later = _embeddings(shift=3)
    read = _prepare(memory, step=1, embeddings=later)
    torch.testing.assert_close(
        read.bank, F.normalize(weak.index_select(0, support), dim=-1)
    )
    expected = F.normalize(
        spec.memory_momentum * F.normalize(weak.index_select(0, support), dim=-1)
        + (1.0 - spec.memory_momentum)
        * F.normalize(later.index_select(0, support), dim=-1),
        dim=-1,
    )
    torch.testing.assert_close(memory.features, expected)
    assert not memory.features.requires_grad


def test_preparation_is_idempotent_within_a_step_in_either_order() -> None:
    """§6.2 invariant 5."""
    spec = _spec()
    memory = _memory(spec)
    first = _prepare(memory, step=0)
    bank = memory.features
    again = _prepare(memory, step=0)
    assert again is first
    torch.testing.assert_close(memory.features, bank, rtol=0, atol=0)

    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    state = _state()
    semantic, instance = _objectives(spec)
    traces = []
    for order in ((semantic, instance), (instance, semantic)):
        fresh = _memory(spec)
        _prepare(fresh, step=0)
        context = _context(1, fresh)
        traces.append(
            {
                objective.name: objective.compute(state, batch, rows, context).value
                for objective in order
            }
        )
        assert fresh.last_prepared_step == 1
    for name in (MEMORY_TERM, INSTANCE_TERM):
        torch.testing.assert_close(traces[0][name], traces[1][name], rtol=0, atol=0)


def test_warm_up_and_coverage_both_hold_the_propagation_back() -> None:
    """§6.2 invariant 6."""
    spec = _spec(warmup_steps=2)
    memory = _memory(spec)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    state = _state()
    semantic, instance = _objectives(spec)
    raw = _probabilities().index_select(0, rows)
    for step in range(3):
        context = _context(step, memory)
        targets = semantic.compute(state, batch, rows, context)
        charged = instance.compute(state, batch, rows, context)
        prepared = memory.prepare(
            step=step,
            raw_probabilities=_probabilities(),
            weak_embeddings=_embeddings(),
            batch=batch,
            eligible_rows=rows,
            support_rows=resolve_rows(batch, "t_observed"),
        )
        if step < 2:
            assert not prepared.propagated
            assert float(charged.value) == 0.0
            assert charged.n == int(rows.numel())
            history = [raw] * (step + 1)
            torch.testing.assert_close(
                prepared.semantic, _aligned(history, spec.alignment_window)
            )
        else:
            assert prepared.propagated and prepared.coverage == 1.0
            assert float(charged.value) > 0.0
        assert targets.n == int(rows.numel())

    # A slot nobody has written holds propagation back past the warm-up.
    sparse = _memory(_spec(warmup_steps=0))
    for step in range(3):
        prepared = sparse.prepare(
            step=step,
            raw_probabilities=_probabilities(),
            weak_embeddings=_embeddings(),
            batch=batch,
            eligible_rows=rows,
            support_rows=torch.tensor([0, 1]),
        )
        # `coverage` is what the bank held when the targets were read, so the
        # first step sees nothing and every later one sees the two slots this
        # quota can reach. Neither is 1.0, so nothing ever propagates.
        assert not prepared.propagated
        assert prepared.coverage == (0.0 if step == 0 else 0.5)


def test_the_targets_and_the_bank_carry_no_gradient_and_the_strong_views_do() -> None:
    """§6.2 invariant 7."""
    spec = _spec()
    memory = _memory(spec)
    _prepare(memory, step=0)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    weak_logits = _probabilities().log().clone().requires_grad_()
    weak_embedding = _embeddings().clone().requires_grad_()
    strong_logits = _probabilities().roll(1, dims=0).log().clone().requires_grad_()
    strong_embedding = _embeddings(shift=7).clone().requires_grad_()
    state = _state(
        weak_logits=weak_logits,
        weak_embedding=weak_embedding,
        strong_logits=strong_logits,
        strong_embedding=strong_embedding,
    )
    semantic, instance = _objectives(spec)
    context = _context(1, memory)
    total = (
        semantic.compute(state, batch, rows, context).value
        + instance.compute(state, batch, rows, context).value
    )
    total.backward()  # type: ignore[no-untyped-call]
    assert weak_logits.grad is None
    assert weak_embedding.grad is None
    assert strong_logits.grad is not None and float(strong_logits.grad.abs().sum()) > 0
    assert (
        strong_embedding.grad is not None
        and float(strong_embedding.grad.abs().sum()) > 0
    )
    assert not memory.features.requires_grad


def test_rejected_rows_stay_in_the_denominator_of_equation_two() -> None:
    """§6.2 invariant 8."""
    spec = _spec(threshold=1.0)
    memory = _memory(spec)
    _prepare(memory, step=0)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    semantic, instance = _objectives(spec)
    context = _context(1, memory)
    gated = semantic.compute(_state(), batch, rows, context)
    assert float(gated.value) == 0.0
    assert gated.n == int(rows.numel())
    assert gated.diagnostics["coverage"] == 0.0
    charged = instance.compute(_state(), batch, rows, context)
    assert charged.n == int(rows.numel())
    assert float(charged.value) > 0.0


def test_alpha_one_and_unfolding_off_recover_the_no_propagation_arm() -> None:
    """§6.2 invariant 9."""
    spec = _spec(alpha=1.0, unfold=False)
    memory = _memory(spec)
    _prepare(memory, step=0)
    targets = _prepare(memory, step=1)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    raw = _probabilities().index_select(0, rows)
    aligned = _aligned([raw, raw], spec.alignment_window)
    torch.testing.assert_close(targets.semantic, aligned, rtol=0, atol=0)
    similarity = torch.softmax(
        _embeddings().index_select(0, rows)
        @ targets.bank.T
        / spec.temperatures.instance_weak,
        dim=-1,
    )
    torch.testing.assert_close(targets.instance, similarity)

    # Both switches are the objects' own; nothing else in the plan moves.
    full = compile(simmatch(_schema())).plan
    ablated = compile(_ablated(_schema())).plan
    moved = {
        key
        for key, value in full.hyperparameters.items()
        if ablated.hyperparameters[key] != value
    }
    assert moved == set()


def _ablated(schema: Schema) -> Recipe:
    """The §6 no-propagation arm: `alpha = 1` and unfolding off, nothing else."""
    recipe = simmatch(schema)
    stage = recipe.program[0]
    spec = replace(SIMILARITY_MATCHING, alpha=1.0, unfold=False)
    semantic = stage.objectives[2].objective
    instance = stage.objectives[3].objective
    assert isinstance(semantic, SimilarityMatchingTreatmentNLL)
    assert isinstance(instance, LabeledMemoryInstanceConsistency)
    objectives = (
        *stage.objectives[:2],
        replace(stage.objectives[2], objective=replace(semantic, spec=spec)),
        replace(stage.objectives[3], objective=replace(instance, spec=spec)),
        *stage.objectives[4:],
    )
    return replace(recipe, program=Program((replace(stage, objectives=objectives),)))


def test_distribution_alignment_is_stationary_and_bounded_by_its_window() -> None:
    """§6.2 invariant 10."""
    spec = _spec(alignment_window=3)
    memory = _memory(spec)
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    uniform = torch.full((8, CLASSES), 0.5, dtype=torch.float64)
    for step in range(4):
        targets = memory.prepare(
            step=step,
            raw_probabilities=uniform,
            weak_embeddings=_embeddings(),
            batch=batch,
            eligible_rows=rows,
            support_rows=resolve_rows(batch, "t_observed"),
        )
        torch.testing.assert_close(
            targets.aligned, uniform.index_select(0, rows), rtol=0, atol=0
        )

    windowed = _memory(spec)
    skewed = [
        torch.tensor([[0.9, 0.1]], dtype=torch.float64).repeat(8, 1),
        torch.tensor([[0.5, 0.5]], dtype=torch.float64).repeat(8, 1),
        torch.tensor([[0.2, 0.8]], dtype=torch.float64).repeat(8, 1),
        torch.tensor([[0.4, 0.6]], dtype=torch.float64).repeat(8, 1),
    ]
    for step, raw in enumerate(skewed):
        targets = windowed.prepare(
            step=step,
            raw_probabilities=raw,
            weak_embeddings=_embeddings(),
            batch=batch,
            eligible_rows=rows,
            support_rows=resolve_rows(batch, "t_observed"),
        )
    history = [raw.index_select(0, rows) for raw in skewed]
    torch.testing.assert_close(targets.aligned, _aligned(history, 3))


def test_the_state_is_built_from_the_observed_rows_of_the_training_population() -> None:
    """§6.2 invariant 11, and where `Q_l` comes from."""
    semantic, _ = _objectives(_spec())
    memory = semantic.initial_state(_population())
    assert isinstance(memory, LabeledSimilarityMemory)
    assert memory.size == 4
    assert memory.slot_ids.tolist() == [100, 101, 102, 103]
    assert memory.labels.tolist() == [0, 1, 0, 1]
    assert memory.coverage == 0.0
    with pytest.raises(LossError, match="needs the stage's training population"):
        semantic.initial_state(None)


def test_batch_coupling_and_state_refuse_the_unreviewed_execution_paths() -> None:
    """§6.2 invariant 11, continued."""
    recipe = simmatch(_schema())
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


def test_the_two_objectives_must_carry_the_same_spec() -> None:
    """One bank, one arithmetic: a mismatched arm is refused, not averaged."""
    memory = _memory(_spec())
    _, instance = _objectives(_spec(alpha=0.5))
    batch = _batch()
    rows = resolve_rows(batch, "t_missing")
    with pytest.raises(LossError, match="different SimilarityMatchingSpec"):
        instance.compute(_state(), batch, rows, _context(0, memory))
