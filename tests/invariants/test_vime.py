"""Tier 0 — the VIME recipe, its corruption view, its heads and its losses.

The load-bearing tests compare the vector implementations against loop
transcriptions of Eqs. 5 and 6, and check the three properties card §3.1 calls
easy to lose: the mask is Bernoulli per cell rather than a fixed count, Eq. 6
averages over every feature and not only the masked ones, and only the encoder
survives into the downstream stage, frozen.
"""

from __future__ import annotations

import ast
import importlib
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xty2.components import FeatureEstimatorHead, MaskEstimatorHead, MLPEncoder
from xty2.core import (
    DEFAULT,
    CompileError,
    Dataset,
    DataSpec,
    FeatureSpec,
    GradientClipping,
    LossError,
    MissingnessSpec,
    OptimiserSpec,
    OutcomeSpec,
    Port,
    PortView,
    PreprocessSpec,
    PreservedField,
    Program,
    Schema,
    SplitSpec,
    State,
    TrainContext,
    TrainingPopulation,
    ViewError,
    WeightDecay,
    XTYBatch,
    compile,
)
from xty2.core.errors import GraphError, TrainingError
from xty2.evaluation.benchmarks.common import on_the_training_scale
from xty2.objectives import FeatureReconstruction, MaskEstimationBCE
from xty2.recipes import vime
from xty2.recipes.vime import (
    ALPHA,
    CLEAN_X,
    MASK_PROBABILITY,
    RMSPROP,
    VIME_CORRUPTED,
)
from xty2.training.loading import build_population
from xty2.views import BernoulliMarginalCorruption, ViewSpec

from tests.invariants.conftest import backward

vime_module = importlib.import_module("xty2.recipes.vime")
scarf_recipe = importlib.import_module("xty2.recipes.scarf")
ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "recipes" / "vime.md"
RECIPE_SOURCE = ROOT / "xty2" / "recipes" / "vime.py"
PRESERVED: frozenset[PreservedField] = frozenset(
    {"t", "y", "t_observed", "y_observed", "row_id", "fold_id", "weight"}
)
FEATURES = 6
"""`d` on the card's section 6 fixture."""


def _schema(
    *, features: int = FEATURES, immutable: int | None = None, kind: str = ""
) -> Schema:
    return Schema(
        features=tuple(
            FeatureSpec(
                f"x{column}",
                "categorical" if column == 0 and kind else "continuous",
                mutable=column != immutable,
            )
            for column in range(features)
        ),
        treatment_cardinality=2,
        outcome=OutcomeSpec(),
    )


def _batch(rows: int, *, features: int = FEATURES, offset: int = 0) -> XTYBatch:
    # Every cell distinct, and disjoint across offsets, so a value that moved
    # can be traced to the column and the population it was drawn from.
    x = (
        torch.arange(float(rows * features)).reshape(rows, features)
        + float(offset * features)
    ) / float(rows * features)
    return XTYBatch(
        x=x,
        t=torch.arange(rows) % 2,
        y=torch.linspace(-1.0, 1.0, rows),
        t_observed=torch.arange(rows) % 3 == 0,
        y_observed=torch.ones(rows, dtype=torch.bool),
        row_id=torch.arange(offset, offset + rows),
    )


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


def _corrupt(
    batch: XTYBatch,
    schema: Schema,
    *,
    p: float = MASK_PROBABILITY,
    key: int = 7,
    population: TrainingPopulation | None = None,
) -> XTYBatch:
    view = ViewSpec(
        name="vime_corrupted",
        transforms=(BernoulliMarginalCorruption(p=p, columns=None),),
        preserves=PRESERVED,
    )
    return view.apply(
        batch,
        schema,
        rng_key=key,
        population=population if population is not None else _population(batch, schema),
    )


# ---------------------------------------------------------------------------
# The recipe and its plan
# ---------------------------------------------------------------------------


def test_the_recipe_plans_two_stages_and_the_passes_each_needs() -> None:
    run = compile(vime(_schema()))
    assert run.graph.names == (
        "mlp_encoder",
        "mask_estimator",
        "feature_estimator",
        "tarnet_head",
        "categorical_propensity",
    )
    assert [stage.name for stage in run.stages] == ["pretrain", "joint_fit"]

    pretrain = run.stage("pretrain")
    assert pretrain.steps == 80
    assert pretrain.trainable == ("mlp_encoder", "mask_estimator", "feature_estimator")
    passes = {
        str(forward.realisation): forward.components for forward in pretrain.passes
    }
    # The clean row is a target only: its pass runs no component at all.
    assert passes == {
        str(DEFAULT): (),
        str(VIME_CORRUPTED): ("mlp_encoder", "mask_estimator", "feature_estimator"),
    }

    fit = run.stage("joint_fit")
    assert fit.steps == 3_000
    assert fit.initialise_from == "pretrain"
    assert [str(forward.realisation) for forward in fit.passes] == [str(DEFAULT)]
    assert fit.passes[0].components == (
        "mlp_encoder",
        "tarnet_head",
        "categorical_propensity",
    )


def test_only_the_encoder_survives_and_it_is_frozen_downstream() -> None:
    """ "It is the only part we will utilize" — and `main_vime.py` freezes it."""
    fit = compile(vime(_schema())).stage("joint_fit")
    assert fit.trainable == ("tarnet_head", "categorical_propensity")
    for head in ("mask_estimator", "feature_estimator"):
        assert head not in fit.trainable
        assert not any(head in forward.components for forward in fit.passes)


@pytest.mark.parametrize("head", ["mask_estimator", "feature_estimator"])
def test_training_a_discarded_head_downstream_is_a_compile_error(head: str) -> None:
    recipe = vime(_schema())
    fit = recipe.program[1]
    broken = replace(
        recipe,
        program=Program(
            (recipe.program[0], replace(fit, trainable=(*fit.trainable, head)))
        ),
    )
    with pytest.raises(CompileError, match="dead weight"):
        compile(broken)


def test_the_recipe_file_contains_declarations_and_no_conditionals() -> None:
    tree = ast.parse(RECIPE_SOURCE.read_text(encoding="utf-8"))
    conditionals = (ast.If, ast.IfExp, ast.Match)
    assert not any(isinstance(node, conditionals) for node in ast.walk(tree))


def test_each_stage_has_exactly_the_reviewed_objectives() -> None:
    run = compile(vime(_schema()))
    pretrain = run.stage("pretrain")
    assert [objective.name for objective in pretrain.objectives] == [
        "mask_estimation_bce",
        "feature_reconstruction",
    ]
    assert [objective.rows for objective in pretrain.objectives] == [
        ("all",),
        ("all",),
    ]
    assert [objective.reduction for objective in pretrain.objectives] == [
        "mean",
        "mean",
    ]
    fit = run.stage("joint_fit")
    assert [objective.name for objective in fit.objectives] == [
        "observed_outcome_nll",
        "observed_treatment_nll",
        "missing_treatment_marginal_nll",
    ]


def test_eq_4_weights_the_reconstruction_by_alpha() -> None:
    weights = compile(vime(_schema())).plan.hyperparameters["losses.weights"]
    assert isinstance(weights, Mapping)
    assert weights["pretrain.mask_estimation_bce"] == 1.0
    assert weights["pretrain.feature_reconstruction"] == ALPHA == 2.0


def test_the_pretraining_stage_reads_no_label_of_any_kind() -> None:
    plan = compile(vime(_schema())).plan
    planned = {component.name: component for component in plan.components}
    for name in ("mlp_encoder", "mask_estimator", "feature_estimator"):
        assert not planned[name].reads_raw_outcome
        assert not planned[name].outcome_dependent


def test_the_recipe_declares_one_view_of_the_bernoulli_corruption() -> None:
    plan = compile(vime(_schema())).plan
    assert [view.name for view in plan.views] == ["vime_corrupted"]
    assert [view.transforms for view in plan.views] == [
        ("BernoulliMarginalCorruption(p=0.3, columns=all)",)
    ]
    assert set(plan.views[0].preserves) == PRESERVED


def test_the_encoder_and_heads_are_as_wide_as_the_schema() -> None:
    """`Dense(int(dim))`: `d` is the schema's, not a constant."""
    run = compile(vime(_schema(features=4)))
    hyperparameters = run.plan.hyperparameters["architecture.widths_depths"]
    assert isinstance(hyperparameters, Mapping)
    assert hyperparameters["mlp_encoder"] == (4,)
    assert hyperparameters["mask_estimator"] == "linear 4 -> 4"
    assert hyperparameters["feature_estimator"] == "linear 4 -> 4"


def test_the_downstream_stage_is_scarfs_imported_not_retyped() -> None:
    """Deviation 4 adopts SCARF's stage unchanged, so it imports it."""
    assert vime_module.JOINT_FIT_ADAM is scarf_recipe.ADAM
    assert vime_module.JOINT_FIT_BATCH_SIZE is scarf_recipe.BATCH_SIZE
    assert vime_module.VIME_JOINT_FIT_STEPS is scarf_recipe.JOINT_FIT_STEPS
    assert vime_module.VIME_OBSERVED_TREATMENTS is scarf_recipe.OBSERVED_TREATMENTS


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def _card_section_four() -> dict[str, str | dict[str, str]]:
    """Card §4 as data, skipping every `n/a` at either level."""
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
        value = value.strip()
        if indent == 0:
            current = name
        elif indent == 2:
            key = f"{current}.{name}"
            if value:
                answered[key] = value
        elif indent == 4 and value != "n/a":
            nested = answered.setdefault(key, {})
            assert isinstance(nested, dict)
            nested[name] = value
    return {key: value for key, value in answered.items() if value != "n/a"}


def _rendered(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "[" + ", ".join(str(item) for item in value) + "]"
    return str(value)


SYMBOLS = {
    "architecture.widths_depths": {"d": "6", "K": "2", "X_REPR": "6"},
    "architecture.output_parameterisation": {"d": "6"},
}


def test_every_answered_card_key_reaches_the_plan() -> None:
    plan = compile(vime(_schema())).plan
    answered = set(_card_section_four())
    missing = sorted(answered - set(plan.hyperparameters))
    assert not missing, "card keys missing from plan: " + ", ".join(missing)
    assert "optimisation.optimiser" in answered


def test_every_card_value_the_plan_also_carries_agrees_with_it() -> None:
    hyperparameters = compile(vime(_schema())).plan.hyperparameters
    mismatched: list[str] = []
    checked = 0
    for key, stated in _card_section_four().items():
        planned = hyperparameters.get(key)
        if planned is None:
            mismatched.append(f"{key}: absent from the plan")
            continue
        if isinstance(stated, str):
            if isinstance(planned, Mapping) or _rendered(planned) != stated:
                mismatched.append(f"{key}: card {stated!r} vs plan {planned!r}")
            checked += 1
            continue
        assert isinstance(planned, Mapping), f"{key} is scoped in the card only"
        extra = sorted(set(planned) - set(stated))
        if extra:
            mismatched.append(f"{key}: plan scopes {extra!r} the card omits")
        for scope, value in stated.items():
            if scope not in planned:
                mismatched.append(f"{key}[{scope}]: absent from the plan")
                continue
            resolved = value
            for symbol, concrete in SYMBOLS.get(key, {}).items():
                resolved = re.sub(rf"\b{symbol}\b", concrete, resolved)
            if _rendered(planned[scope]) != resolved:
                mismatched.append(
                    f"{key}[{scope}]: card {resolved!r} vs plan {planned[scope]!r}"
                )
            checked += 1
    assert not mismatched, "card and plan disagree:\n  " + "\n  ".join(mismatched)
    assert checked >= 60


# ---------------------------------------------------------------------------
# BernoulliMarginalCorruption (Eq. 3)
# ---------------------------------------------------------------------------


def test_the_mask_is_bernoulli_per_cell_not_a_fixed_count() -> None:
    """Card §3.1: a row corrupts Binomial(`d`, `p_m`) cells, not `floor(p d)`.

    The mean alone cannot tell the two apart; the per-row spread can. A
    fixed count has variance zero.
    """
    rows = 4_000
    batch = _batch(rows)
    schema = _schema()
    corrupted = _corrupt(batch, schema)
    changed = corrupted.x != batch.x
    rate = float(changed.float().mean())
    assert rate == pytest.approx(MASK_PROBABILITY, abs=0.01)
    per_row = changed.sum(dim=1).float()
    expected = FEATURES * MASK_PROBABILITY * (1.0 - MASK_PROBABILITY)
    assert float(per_row.var()) == pytest.approx(expected, rel=0.1)
    assert set(per_row.long().tolist()) >= {0, 1, 2, 3, 4}


def test_each_column_is_masked_independently_at_the_same_rate() -> None:
    rows = 4_000
    batch = _batch(rows)
    changed = (_corrupt(batch, _schema()).x != batch.x).float()
    for column in range(FEATURES):
        assert float(changed[:, column].mean()) == pytest.approx(
            MASK_PROBABILITY, abs=0.03
        )
    # Independent columns: the correlation of two columns' masks is near zero.
    correlation = torch.corrcoef(changed.transpose(0, 1))
    off_diagonal = correlation[~torch.eye(FEATURES, dtype=torch.bool)]
    assert float(off_diagonal.abs().max()) < 0.06


def test_the_donor_pool_is_the_training_population_not_the_batch() -> None:
    """Eq. 3's `p_hat_{X_j}` is over the unlabelled set, not the batch in hand."""
    schema = _schema()
    batch = _batch(64)
    population = _population(_batch(512, offset=1_000), schema)
    corrupted = _corrupt(batch, schema, population=population)
    changed = corrupted.x != batch.x
    assert bool(changed.any())
    for column in range(FEATURES):
        values = corrupted.x[changed[:, column], column]
        pool = population.rows.x[:, column]
        assert bool(torch.isin(values, pool).all())
        assert not bool(torch.isin(values, batch.x[:, column]).any())


def test_each_replacement_comes_from_its_own_column() -> None:
    schema = _schema()
    batch = _batch(256)
    corrupted = _corrupt(batch, schema)
    changed = corrupted.x != batch.x
    for column in range(FEATURES):
        values = corrupted.x[changed[:, column], column]
        assert bool(torch.isin(values, batch.x[:, column]).all())


def test_each_masked_cell_draws_its_own_donor_row() -> None:
    """One donor per row would keep a row's cross-feature dependence."""
    schema = _schema()
    batch = _batch(512)
    corrupted = _corrupt(batch, schema, p=1.0)
    donors = torch.stack(
        [
            torch.searchsorted(
                batch.x[:, column].contiguous(), corrupted.x[:, column].contiguous()
            )
            for column in range(FEATURES)
        ],
        dim=1,
    )
    same_donor = (donors == donors[:, :1]).all(dim=1)
    assert float(same_donor.float().mean()) < 0.01


def test_an_immutable_column_is_never_corrupted() -> None:
    """Deviation 6."""
    schema = _schema(immutable=2)
    batch = _batch(1_000)
    corrupted = _corrupt(batch, schema, p=1.0)
    assert torch.equal(corrupted.x[:, 2], batch.x[:, 2])
    others = [column for column in range(FEATURES) if column != 2]
    assert bool((corrupted.x[:, others] != batch.x[:, others]).float().mean() > 0.99)


def test_the_endpoints_of_p_mean_nothing_and_everything() -> None:
    schema = _schema()
    batch = _batch(64)
    population = _population(_batch(256, offset=500), schema)
    assert torch.equal(_corrupt(batch, schema, p=0.0).x, batch.x)
    every = _corrupt(batch, schema, p=1.0, population=population)
    assert bool((every.x != batch.x).all())


def test_the_view_is_deterministic_in_its_key_and_functional() -> None:
    schema = _schema()
    batch = _batch(64)
    original = batch.x.clone()
    first = _corrupt(batch, schema, key=3)
    assert torch.equal(first.x, _corrupt(batch, schema, key=3).x)
    assert not torch.equal(first.x, _corrupt(batch, schema, key=4).x)
    assert torch.equal(batch.x, original)
    assert torch.equal(first.t, batch.t)
    assert torch.equal(first.row_id, batch.row_id)


def test_a_transform_without_a_population_says_so() -> None:
    transform = BernoulliMarginalCorruption(p=0.3)
    with pytest.raises(ViewError, match="training population"):
        transform.apply(_batch(4), _schema(), generator=torch.Generator())


@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan"), True, "0.3"])
def test_the_transform_rejects_a_p_that_is_not_a_probability(p: object) -> None:
    with pytest.raises(ViewError):
        BernoulliMarginalCorruption(p=p)  # type: ignore[arg-type]


def test_the_transform_rejects_unknown_empty_or_duplicate_columns() -> None:
    with pytest.raises(ViewError):
        BernoulliMarginalCorruption(p=0.3, columns=())
    with pytest.raises(ViewError):
        BernoulliMarginalCorruption(p=0.3, columns=("x0", "x0"))
    with pytest.raises(ViewError, match="unknown column"):
        BernoulliMarginalCorruption(p=0.3, columns=("nope",)).validate(_schema())


# ---------------------------------------------------------------------------
# The heads and the encoder's Keras initialisation
# ---------------------------------------------------------------------------


def _head_kwargs(width: int, **overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "representation_dim": width,
        "num_features": width,
        "normalisation": "none",
        "dropout": 0.0,
        "initialisation": "glorot_uniform, bias=0",
    }
    return defaults | overrides


def _ports(value: torch.Tensor) -> PortView:
    return PortView(
        {Port.X_REPR: value}, declared=frozenset({Port.X_REPR}), component="head"
    )


def test_the_mask_estimator_emits_logits_and_the_feature_estimator_unit_values() -> (
    None
):
    mask = MaskEstimatorHead(
        **_head_kwargs(  # type: ignore[arg-type]
            FEATURES,
            activation="linear logits",
            output_parameterisation=f"{FEATURES} Bernoulli logits",
        )
    )
    feature = FeatureEstimatorHead(
        **_head_kwargs(  # type: ignore[arg-type]
            FEATURES,
            activation="sigmoid",
            output_parameterisation=f"{FEATURES} values in (0, 1)",
        )
    )
    assert mask.provides == frozenset({Port.FEATURE_MASK_LOGITS})
    assert feature.provides == frozenset({Port.RECONSTRUCTION})
    representation = 50.0 * torch.randn(32, FEATURES)
    with torch.no_grad():
        logits = mask.forward(_ports(representation))[Port.FEATURE_MASK_LOGITS]
        values = feature.forward(_ports(representation))[Port.RECONSTRUCTION]
        direct = torch.sigmoid(feature.layer(representation))
    assert isinstance(logits, torch.Tensor) and isinstance(values, torch.Tensor)
    assert logits.shape == values.shape == (32, FEATURES)
    # Logits leave the unit interval; the reconstruction never does.
    assert float(logits.abs().max()) > 1.0
    assert float(values.min()) >= 0.0 and float(values.max()) <= 1.0
    assert torch.allclose(values, direct)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"activation": "sigmoid"}, "activation"),
        ({"initialisation": "torch Linear default Kaiming-uniform"}, "glorot"),
        ({"dropout": 0.1}, "affine"),
        ({"output_parameterisation": "6 values in (0, 1)"}, "output"),
    ],
)
def test_the_mask_estimator_rejects_what_the_reference_does_not_build(
    overrides: dict[str, object], match: str
) -> None:
    arguments = _head_kwargs(
        FEATURES,
        activation="linear logits",
        output_parameterisation=f"{FEATURES} Bernoulli logits",
    )
    with pytest.raises(GraphError, match=match):
        MaskEstimatorHead(**(arguments | overrides))  # type: ignore[arg-type]


def test_glorot_uniform_is_keras_dense_default_not_torch_linear() -> None:
    """Bound `sqrt(6 / (fan_in + fan_out))`, zero bias; torch's is `1/sqrt(fan_in)`."""
    torch.manual_seed(0)
    width = 256
    encoder = MLPEncoder(
        input_dim=width,
        widths=(width,),
        activation="relu",
        normalisation="none",
        dropout=0.0,
        initialisation="glorot_uniform, bias=0",
    )
    layer = encoder.network[0]
    assert isinstance(layer, torch.nn.Linear)
    bound = math.sqrt(6.0 / (width + width))
    weight = layer.weight.detach()
    assert float(weight.abs().max()) <= bound
    assert float(weight.std()) == pytest.approx(bound / math.sqrt(3.0), rel=0.02)
    assert float(layer.bias.detach().abs().max()) == 0.0


# ---------------------------------------------------------------------------
# MaskEstimationBCE (Eq. 5) and FeatureReconstruction (Eq. 6)
# ---------------------------------------------------------------------------


ROWS = 10


def _pretext_state(
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    logits: torch.Tensor,
    reconstruction: torch.Tensor,
) -> State:
    return State(
        {
            CLEAN_X: {Port.X_RAW: clean},
            VIME_CORRUPTED: {
                Port.X_RAW: corrupted,
                Port.FEATURE_MASK_LOGITS: logits,
                Port.RECONSTRUCTION: reconstruction,
            },
        }
    )


def _pretext_inputs(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    clean = torch.rand(ROWS, FEATURES, generator=generator)
    mask = torch.rand(ROWS, FEATURES, generator=generator) < 0.3
    donors = torch.rand(ROWS, FEATURES, generator=generator)
    corrupted = torch.where(mask, donors, clean)
    logits = 3.0 * torch.randn(ROWS, FEATURES, generator=generator)
    reconstruction = torch.rand(ROWS, FEATURES, generator=generator)
    return clean, corrupted, logits, reconstruction


def _context(schema: Schema | None = None) -> TrainContext:
    return TrainContext(global_step=0, schema=schema or _schema())


def _bce() -> MaskEstimationBCE:
    return MaskEstimationBCE(clean=CLEAN_X, corrupted=VIME_CORRUPTED)


def _mse(**overrides: object) -> FeatureReconstruction:
    arguments: dict[str, object] = {"clean": CLEAN_X, "corrupted": VIME_CORRUPTED}
    return FeatureReconstruction(**(arguments | overrides))  # type: ignore[arg-type]


def _eq_5(
    clean: torch.Tensor, corrupted: torch.Tensor, logits: torch.Tensor, rows: list[int]
) -> float:
    """Eq. 5 as a loop, with `m = 1[x != x~]` (`pretext_generator`'s `m_new`)."""
    total = 0.0
    for i in rows:
        row = 0.0
        for j in range(FEATURES):
            m = 1.0 if float(clean[i, j]) != float(corrupted[i, j]) else 0.0
            s = 1.0 / (1.0 + math.exp(-float(logits[i, j])))
            row -= m * math.log(s) + (1.0 - m) * math.log(1.0 - s)
        total += row / FEATURES
    return total / len(rows)


def _eq_6(clean: torch.Tensor, reconstruction: torch.Tensor, rows: list[int]) -> float:
    """Eq. 6 as a loop, over all `d` features whether masked or not."""
    total = 0.0
    for i in rows:
        row = sum(
            (float(clean[i, j]) - float(reconstruction[i, j])) ** 2
            for j in range(FEATURES)
        )
        total += row / FEATURES
    return total / len(rows)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_mask_estimation_is_eq_5(seed: int) -> None:
    clean, corrupted, logits, reconstruction = _pretext_inputs(seed)
    state = _pretext_state(clean, corrupted, logits, reconstruction)
    rows = [0, 2, 3, 7, 9]
    term = _bce().compute(state, _batch(ROWS), torch.tensor(rows), _context())
    assert term.n == len(rows)
    assert float(term.value) == pytest.approx(_eq_5(clean, corrupted, logits, rows))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_feature_reconstruction_is_eq_6_over_every_feature(seed: int) -> None:
    clean, corrupted, logits, reconstruction = _pretext_inputs(seed)
    state = _pretext_state(clean, corrupted, logits, reconstruction)
    rows = [1, 4, 5, 8]
    term = _mse().compute(state, _batch(ROWS), torch.tensor(rows), _context())
    assert term.n == len(rows)
    assert float(term.value) == pytest.approx(_eq_6(clean, reconstruction, rows))


def test_the_reconstruction_charges_the_unmasked_cells_too() -> None:
    """Card §3.1: part of Eq. 6 is an identity copy, and it is kept."""
    clean = torch.rand(ROWS, FEATURES)
    corrupted = clean.clone()
    corrupted[:, 0] = corrupted[:, 0] + 1.0
    reconstruction = clean.clone()
    reconstruction[:, 1] = reconstruction[:, 1] + 0.5  # an unmasked column
    state = _pretext_state(clean, corrupted, torch.zeros_like(clean), reconstruction)
    term = _mse().compute(state, _batch(ROWS), torch.arange(ROWS), _context())
    assert float(term.value) == pytest.approx(0.25 / FEATURES)
    assert term.diagnostics["masked_cell_mse"] == pytest.approx(0.0)


def test_the_mask_label_is_the_cells_that_changed() -> None:
    """With nothing changed, every label is 0 and the loss is `softplus(logit)`."""
    clean = torch.rand(ROWS, FEATURES)
    logits = torch.randn(ROWS, FEATURES)
    state = _pretext_state(clean, clean.clone(), logits, clean)
    term = _bce().compute(state, _batch(ROWS), torch.arange(ROWS), _context())
    expected = float(torch.nn.functional.softplus(logits).mean())
    assert float(term.value) == pytest.approx(expected)
    assert term.diagnostics["mask_rate"] == 0.0


def test_no_eligible_rows_returns_the_zero_term() -> None:
    clean, corrupted, logits, reconstruction = _pretext_inputs(0)
    state = _pretext_state(clean, corrupted, logits, reconstruction)
    empty = torch.zeros(0, dtype=torch.long)
    for objective in (_bce(), _mse()):
        term = objective.compute(state, _batch(ROWS), empty, _context())
        assert term.n == 0 and float(term.value) == 0.0


def test_both_estimators_receive_a_gradient_and_nothing_is_detached() -> None:
    clean, corrupted, logits, reconstruction = _pretext_inputs(0)
    logits.requires_grad_(True)
    reconstruction.requires_grad_(True)
    state = _pretext_state(clean, corrupted, logits, reconstruction)
    for objective in (_bce(), _mse()):
        assert objective.detaches == frozenset()
        assert not objective.batch_coupled
        term = objective.compute(state, _batch(ROWS), torch.arange(ROWS), _context())
        backward(term.value)
    assert logits.grad is not None and float(logits.grad.abs().sum()) > 0.0
    assert reconstruction.grad is not None
    assert float(reconstruction.grad.abs().sum()) > 0.0


def test_the_losses_read_the_clean_row_as_a_target_under_its_own_realisation() -> None:
    for objective, port in (
        (_bce(), Port.FEATURE_MASK_LOGITS),
        (_mse(), Port.RECONSTRUCTION),
    ):
        assert objective.requires == frozenset(
            {
                (port, VIME_CORRUPTED),
                (Port.X_RAW, VIME_CORRUPTED),
                (Port.X_RAW, CLEAN_X),
            }
        )


def test_swapping_the_clean_and_corrupted_rows_changes_the_plan() -> None:
    """`requires` is a set and cannot show which `X_RAW` is the target."""
    swapped = _mse(clean=VIME_CORRUPTED, corrupted=CLEAN_X)
    assert swapped.requires != _mse().requires
    assert _mse(clean=VIME_CORRUPTED, corrupted=CLEAN_X).plan_details() != (
        _mse().plan_details()
    )
    assert "target = x @ view=identity params=student" in _mse().plan_details()


def test_the_losses_refuse_a_categorical_feature() -> None:
    """Deviation 5: no squared error on class codes."""
    clean, corrupted, logits, reconstruction = _pretext_inputs(0)
    state = _pretext_state(clean, corrupted, logits, reconstruction)
    for objective in (_bce(), _mse()):
        with pytest.raises(LossError, match="deviation 5"):
            objective.compute(
                state,
                _batch(ROWS),
                torch.arange(ROWS),
                _context(_schema(kind="categorical")),
            )


def test_the_losses_reject_what_they_cannot_mean() -> None:
    with pytest.raises(LossError, match="identity"):
        MaskEstimationBCE(clean=CLEAN_X, corrupted=CLEAN_X)
    with pytest.raises(LossError, match="proposal"):
        _mse(cells="masked")
    with pytest.raises(LossError):
        _mse(cells="some")


# ---------------------------------------------------------------------------
# RMSprop with Keras's constants
# ---------------------------------------------------------------------------


def test_rmsprop_is_the_keras_update_with_the_bound_constants() -> None:
    """`v <- rho v + (1 - rho) g^2`, `p <- p - lr g / (sqrt(v) + eps)`."""
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimiser = RMSPROP.build([("mlp_encoder.weight", parameter)])
    assert isinstance(optimiser, torch.optim.RMSprop)
    expected = [1.0, -2.0]
    accumulator = [0.0, 0.0]
    for gradient in ([0.5, -1.0], [0.25, 3.0]):
        optimiser.zero_grad()
        parameter.grad = torch.tensor(gradient)
        optimiser.step()
        for index, g in enumerate(gradient):
            accumulator[index] = 0.9 * accumulator[index] + 0.1 * g * g
            expected[index] -= 1e-3 * g / (math.sqrt(accumulator[index]) + 1e-7)
    assert parameter.detach().tolist() == pytest.approx(expected, rel=1e-6)


def test_rmsprop_knobs_on_another_optimiser_are_rejected() -> None:
    base: dict[str, object] = {
        "lr": 1e-3,
        "weight_decay": WeightDecay.none(),
        "lr_schedule": 1.0,
        "clipping": GradientClipping.none(),
    }
    with pytest.raises(CompileError, match="rho"):
        OptimiserSpec(name="adam", rho=0.9, **base)  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="betas"):
        OptimiserSpec(name="rmsprop", betas=(0.8, 0.9), **base)  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="rho"):
        OptimiserSpec(name="rmsprop", rho=1.0, **base)  # type: ignore[arg-type]
    with pytest.raises(CompileError, match="Nesterov"):
        OptimiserSpec(name="rmsprop", nesterov=True, **base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Min-max scaling (section 5)
# ---------------------------------------------------------------------------


def _minmax_population(batch: XTYBatch) -> TrainingPopulation:
    return build_population(
        Dataset(
            schema=_schema(),
            rows=batch,
            assignments={"train": torch.arange(batch.batch_size)},
        ),
        replace(
            vime_module.DATA_POLICY,
            missingness=MissingnessSpec(mechanism="observed"),
        ),
        seed=0,
    )


def test_minmax_maps_each_training_column_onto_the_unit_interval() -> None:
    generator = torch.Generator().manual_seed(0)
    x = 3.0 * torch.randn(200, FEATURES, generator=generator) + 5.0
    batch = replace(_batch(200), x=x)
    population = _minmax_population(batch)
    assert torch.equal(population.statistics["x_location"], x.amin(dim=0))
    assert torch.equal(population.statistics["x_scale"], x.amax(dim=0) - x.amin(dim=0))
    scaled = population.rows.x
    assert torch.allclose(scaled.amin(dim=0), torch.zeros(FEATURES))
    assert torch.allclose(scaled.amax(dim=0), torch.ones(FEATURES))


def test_held_out_rows_take_the_fitted_map_and_are_not_refitted() -> None:
    generator = torch.Generator().manual_seed(1)
    train = replace(_batch(100), x=torch.randn(100, FEATURES, generator=generator))
    population = _minmax_population(train)
    held_out = replace(
        _batch(50, offset=100), x=4.0 * torch.randn(50, FEATURES, generator=generator)
    )
    scaled = on_the_training_scale(held_out, population)
    expected = (held_out.x - train.x.amin(dim=0)) / (
        train.x.amax(dim=0) - train.x.amin(dim=0)
    )
    assert torch.allclose(scaled.x, expected)
    assert float(scaled.x.max()) > 1.0 or float(scaled.x.min()) < 0.0


def test_a_constant_column_cannot_be_min_max_scaled() -> None:
    batch = _batch(20)
    constant = batch.x.clone()
    constant[:, 3] = 1.0
    with pytest.raises(TrainingError, match="minmax"):
        _minmax_population(replace(batch, x=constant))
