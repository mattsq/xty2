"""VIME-self's paired mechanism benchmark from card section 6.

Card section 6.1 adopts SCARF's fixture and seed streams unchanged, so this
module imports them (`xty2.evaluation.benchmarks.scarf`) rather than retyping
them: the populations, the initial-state seed and the program seed are SCARF's
for the same replicate index. Min-max feature scaling is the recipe's own
`DataSpec` declaration, applied by the loader and carried to held-out rows
through the fitted `TrainingPopulation`, never refitted here.

Two kinds of measurement, and the card keeps them apart:

* **Pretext diagnostics** read the fitted estimator heads immediately after
  `pretrain`, before any downstream stage is constructed. Each pretext arm runs
  a pretrain-only program, so its live graph *is* its pretraining checkpoint.
  The held-out rows take one fixed corruption draw per replicate, shared by
  every pretext arm, with donors from the training population only.
* **The downstream guardrail** pairs the full two-stage program against the
  same `joint_fit` stage from the recipe's untrained initialisation: same
  initial parameters, same populations, same batches.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import Tensor

from xty2.core import (
    CompiledRun,
    Port,
    Program,
    Recipe,
    Schema,
    TrainingPopulation,
    Weighted,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    SIGNAL_COLUMNS,
    column,
    configure_worker,
    on_the_training_scale,
    parallel_replicates,
)
from xty2.evaluation.benchmarks.scarf import (
    Fixture,
    fixture,
    held_out_nll,
    unpretrained,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.recipes import vime
from xty2.recipes.vime import MASK_PROBABILITY
from xty2.training import STREAM_STRIDE, run_program
from xty2.views import BernoulliMarginalCorruption

EVALUATION_CORRUPTION_OFFSET = 7
"""Replicate `base + 7` seeds the one held-out corruption draw (card §6.1)."""

FIXED_DRAW_OFFSET = 8
"""Replicate `base + 8` seeds the fixed-draw ablation's single training draw."""

DEPENDENT_BLOCK = tuple(range(SIGNAL_COLUMNS))
"""`x0..x3`: they share the cluster indicator, so a masked cell is predictable."""

INDEPENDENT_BLOCK = tuple(range(SIGNAL_COLUMNS, 6))
"""`x4, x5`: independent noise, so no function of the visible cells beats the mean."""

PROTOCOL: dict[str, str] = {
    "dataset": (
        "scarf.md section 6.1 fixture (fixmatch.md 6.1 generator, 6 "
        "features, K=2), imported unchanged"
    ),
    "variant": (
        "paired VIME pretraining against an untrained encoder with the "
        "same initialisation, both frozen under the identical joint_fit "
        "stage, same seeds and same batches"
    ),
    "split": (
        "1024 train rows with 40 observed treatments, 2048 held-out "
        "rows with every treatment observed"
    ),
    "metric": (
        "held-out reconstruction MSE on corrupted cells of the "
        "dependent block x0..x3, as a ratio to imputing the training "
        "column mean; the same ratio on the independent block x4..x5 as "
        "a leakage canary; held-out outcome NLL ratio as an adaptation "
        "guardrail; mask-estimation AUROC and treatment NLL ratio are "
        "informational"
    ),
    "published": "none - no published number applies to this adaptation",
    "tolerance": (
        "dependent-block ratio < 0.95 in mean; independent-block ratio "
        ">= 0.98 in mean; held-out outcome NLL within 1.05x of the "
        "untrained-encoder arm"
    ),
    "seeds": "10",
    "report": "mean_and_stderr",
}
"""Card §6's reproduction block, bound by value."""

DOCUMENTATION = ("published_source",)
"""Card §6 scalars no code reads."""

_MASK_TERM = "mask_estimation_bce"
_RECONSTRUCTION_TERM = "feature_reconstruction"
_PRETEXT_COMPONENTS = ("mlp_encoder", "mask_estimator", "feature_estimator")


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run ten paired replicates of card section 6."""
    del cache_root
    spec.bind(PROTOCOL, documentation=DOCUMENTATION)
    if spec.seed_count != 10:
        raise ValueError(f"vime card reviewed ten replicates, got {spec.seed_count}")
    rows = parallel_replicates(replicate, spec.seed_count, workers=workers)
    required = (
        MetricResult(
            "dependent_block_reconstruction_ratio",
            column(rows, "dependent_ratio"),
            "<",
            0.95,
        ),
        MetricResult.lower_bound(
            "independent_block_reconstruction_ratio",
            column(rows, "independent_ratio"),
            0.98,
        ),
        MetricResult.upper_bound(
            "held_out_outcome_NLL_ratio",
            column(rows, "outcome_ratio"),
            1.05,
        ),
    )
    informational = (
        ("mask_estimation_AUROC", "mask_auroc", ""),
        ("held_out_treatment_NLL_ratio", "treatment_ratio", ""),
        ("pretrained_treatment_NLL", "pretrained_treatment_nll", "nat/row"),
        ("untrained_treatment_NLL", "untrained_treatment_nll", "nat/row"),
        ("pretrained_outcome_NLL", "pretrained_outcome_nll", "nat/row"),
        ("untrained_outcome_NLL", "untrained_outcome_nll", "nat/row"),
        ("held_out_corrupted_cell_rate", "corrupted_rate", ""),
        ("terminal_mask_estimation_BCE", "terminal_mask_bce", ""),
        ("terminal_feature_reconstruction", "terminal_reconstruction", ""),
        ("fixed_draw_dependent_block_ratio", "fixed_draw_dependent_ratio", ""),
        ("fixed_draw_independent_block_ratio", "fixed_draw_independent_ratio", ""),
        ("fixed_draw_mask_AUROC", "fixed_draw_mask_auroc", ""),
        ("mask_only_mask_AUROC", "mask_only_mask_auroc", ""),
        (
            "reconstruction_only_dependent_block_ratio",
            "reconstruction_only_dependent_ratio",
            "",
        ),
        (
            "reconstruction_only_independent_block_ratio",
            "reconstruction_only_independent_ratio",
            "",
        ),
    )
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            *required,
            *(
                MetricResult.information(name, column(rows, key), unit=unit)
                for name, key, unit in informational
            ),
        ),
        interpretation=(
            "This project-local mechanism target asks whether VIME-self's "
            "published pretext task (Eqs. 3-6, the reference script's p_m, "
            "alpha, widths, optimiser and ten epochs) learns conditional "
            "structure where the fixture has some and none where it has none, "
            "and whether the frozen pretrained encoder leaves the causal "
            "outcome fit no worse than a frozen untrained one. Pretext ratios "
            "read the diagnostic heads immediately after pretraining. The "
            "fixed-draw, mask-only and reconstruction-only arms are ablations, "
            "reported and not gated. No number from Yoon et al. is reproduced."
        ),
    )


@dataclass(frozen=True)
class FixedDrawMarginalCorruption:
    """Card §6.1's fixed single-draw ablation of deviation 2.

    `vime_self.py` calls `mask_generator` and `pretext_generator` once, before
    `model.fit`, so each training row keeps one corruption for every epoch.
    This transform reproduces that schedule with the recipe's own pretext
    generator: `BernoulliMarginalCorruption(p)` is applied once to the whole
    training population under a generator seeded by `seed`, and a batch looks
    its rows up by `row_id`. The executor's per-step generator is not read, so
    the draw is the same at every step. An evaluation-only object: no recipe
    declares it.
    """

    p: float
    seed: int

    def validate(self, schema: Schema) -> None:
        BernoulliMarginalCorruption(p=self.p).validate(schema)

    def affected_columns(self, schema: Schema) -> frozenset[str]:
        return BernoulliMarginalCorruption(p=self.p).affected_columns(schema)

    def apply(
        self,
        batch: XTYBatch,
        schema: Schema,
        *,
        generator: torch.Generator,
        population: TrainingPopulation | None = None,
    ) -> XTYBatch:
        del generator
        if population is None:
            raise ValueError(
                "the fixed-draw ablation corrupts the training population once "
                "and needs it; this stage supplied none"
            )
        table = BernoulliMarginalCorruption(p=self.p).apply(
            population.rows,
            schema,
            generator=torch.Generator().manual_seed(self.seed),
            population=population,
        )
        matches = batch.row_id[:, None] == population.rows.row_id[None, :]
        if not bool((matches.sum(dim=1) == 1).all()):
            raise ValueError(
                "the fixed-draw ablation found a batch row that is not exactly "
                "one training-population row"
            )
        return batch.replace(x=table.x.index_select(0, matches.long().argmax(dim=1)))

    def describe(self) -> str:
        return f"FixedDrawMarginalCorruption(p={float(self.p)!r}, seed={self.seed})"


def replicate(index: int) -> dict[str, float]:
    """One paired replicate: four pretext arms and two downstream arms."""
    configure_worker()
    world = fixture(index)

    vime_pretext, population = _pretext(world, _vime(world))
    corrupted_test, clean_test = _held_out_corruption(world, population)
    scored = _score(vime_pretext, corrupted_test, clean_test, population)
    fixed = _score(
        _pretext_checked(world, _fixed_draw(_vime(world), world), population),
        corrupted_test,
        clean_test,
        population,
    )
    mask_only = _score(
        _pretext_checked(
            world, _reweighted(_vime(world), _RECONSTRUCTION_TERM), population
        ),
        corrupted_test,
        clean_test,
        population,
    )
    reconstruction_only = _score(
        _pretext_checked(world, _reweighted(_vime(world), _MASK_TERM), population),
        corrupted_test,
        clean_test,
        population,
    )

    # The downstream pair. Both arms start from bit-identical parameters; the
    # untrained arm's single stage is index 0, so its seed is offset by one
    # stride to give its `joint_fit` the same stream as the paired arm's.
    pretrained_recipe = _vime(world)
    untrained_recipe = unpretrained(_vime(world))
    for name, value in pretrained_recipe.system.state_dict().items():
        if not torch.equal(value, untrained_recipe.system.state_dict()[name]):
            raise RuntimeError(f"vime paired initial state differs at {name!r}")
    pretrained_run = compile(pretrained_recipe)
    untrained_run = compile(untrained_recipe)
    full = run_program(
        pretrained_run,
        {"pretrain": world.data, "joint_fit": world.data},
        seed=world.run_seed,
    )
    # The scored heads and the transferred encoder are one fit: the full
    # program's pretraining checkpoint must equal the pretrain-only graph.
    checkpoint = full.stage("pretrain").checkpoint.parameters
    for name, value in vime_pretext.graph.named_parameters():
        key = name.removeprefix("_components.")
        if key.split(".", 1)[0] in _PRETEXT_COMPONENTS and not torch.equal(
            value, checkpoint[key]
        ):
            raise RuntimeError(f"vime scored pretext state differs at {key!r}")
    bare = run_program(
        untrained_run,
        {"joint_fit": world.data},
        seed=world.run_seed + STREAM_STRIDE,
    )
    pretrained = held_out_nll(pretrained_run, full, world.test)
    untrained = held_out_nll(untrained_run, bare, world.test)
    if untrained["treatment_nll"] <= 0.0 or untrained["outcome_nll"] <= 0.0:
        raise RuntimeError(
            "the untrained-encoder arm produced a non-positive NLL, so the "
            "paired ratio the card declares is undefined"
        )
    terminal = full.stage("pretrain").records[-1]
    terms = {term.name: float(term.value) for term in terminal.terms}
    return {
        "dependent_ratio": scored["dependent_ratio"],
        "independent_ratio": scored["independent_ratio"],
        "mask_auroc": scored["mask_auroc"],
        "corrupted_rate": scored["corrupted_rate"],
        "outcome_ratio": pretrained["outcome_nll"] / untrained["outcome_nll"],
        "treatment_ratio": pretrained["treatment_nll"] / untrained["treatment_nll"],
        "pretrained_treatment_nll": pretrained["treatment_nll"],
        "untrained_treatment_nll": untrained["treatment_nll"],
        "pretrained_outcome_nll": pretrained["outcome_nll"],
        "untrained_outcome_nll": untrained["outcome_nll"],
        "terminal_mask_bce": terms[_MASK_TERM],
        "terminal_reconstruction": terms[_RECONSTRUCTION_TERM],
        "fixed_draw_dependent_ratio": fixed["dependent_ratio"],
        "fixed_draw_independent_ratio": fixed["independent_ratio"],
        "fixed_draw_mask_auroc": fixed["mask_auroc"],
        "mask_only_mask_auroc": mask_only["mask_auroc"],
        "reconstruction_only_dependent_ratio": reconstruction_only["dependent_ratio"],
        "reconstruction_only_independent_ratio": reconstruction_only[
            "independent_ratio"
        ],
    }


def _vime(world: Fixture) -> Recipe:
    """The recipe from SCARF's initial-state seed, so every arm starts equal."""
    torch.manual_seed(world.initial_state_seed)
    return vime(world.schema)


def _pretext(world: Fixture, recipe: Recipe) -> tuple[CompiledRun, TrainingPopulation]:
    """Run `pretrain` alone, from the full program's seed, and stop there.

    Stage index 0 in both programs, so the pretrain-only run draws the same
    batches and corruptions as the full program's first stage.
    """
    pretrain = recipe.program[0]
    compiled = compile(replace(recipe, program=Program((pretrain,))))
    result = run_program(compiled, {"pretrain": world.data}, seed=world.run_seed)
    population = result.stage("pretrain").population
    if population is None:
        raise RuntimeError(
            "pretrain declares a sampler, so it must report a population"
        )
    return compiled, population


def _pretext_checked(
    world: Fixture, recipe: Recipe, population: TrainingPopulation
) -> CompiledRun:
    """An ablation arm's pretext run, on the same fitted population."""
    compiled, own = _pretext(world, recipe)
    if not torch.equal(own.rows.x, population.rows.x) or not torch.equal(
        own.rows.row_id, population.rows.row_id
    ):
        raise RuntimeError("a vime pretext arm saw a different training population")
    return compiled


def _fixed_draw(recipe: Recipe, world: Fixture) -> Recipe:
    """Deviation 2's ablation: one corruption per training row for all steps."""
    (view,) = recipe.views
    fixed = FixedDrawMarginalCorruption(
        p=MASK_PROBABILITY, seed=world.base + FIXED_DRAW_OFFSET
    )
    return replace(recipe, views=(replace(view, transforms=(fixed,)),))


def _reweighted(recipe: Recipe, silenced: str) -> Recipe:
    """The recipe with one pretext term at weight 0 and nothing else changed."""
    pretrain = recipe.program[0]
    objectives = tuple(
        Weighted(term.objective, weight=0.0, reduction=term.reduction)
        if term.objective.name == silenced
        else term
        for term in pretrain.objectives
    )
    if objectives == pretrain.objectives:
        raise ValueError(f"vime pretrain has no objective named {silenced!r}")
    program = (replace(pretrain, objectives=objectives), *recipe.program[1:])
    return replace(recipe, program=Program(program))


def _held_out_corruption(
    world: Fixture, population: TrainingPopulation
) -> tuple[XTYBatch, XTYBatch]:
    """One fixed Eq. 3 draw over the held-out rows, donors from training only."""
    clean = on_the_training_scale(world.test.batch, population)
    corrupted = BernoulliMarginalCorruption(p=MASK_PROBABILITY).apply(
        clean,
        world.schema,
        generator=torch.Generator().manual_seed(
            world.base + EVALUATION_CORRUPTION_OFFSET
        ),
        population=population,
    )
    return corrupted, clean


def _score(
    run: CompiledRun,
    corrupted: XTYBatch,
    clean: XTYBatch,
    population: TrainingPopulation,
) -> dict[str, float]:
    """Held-out pretext metrics from the fitted heads, before they are dropped."""
    with torch.no_grad():
        values = run.graph.evaluate(
            corrupted, schema=run.recipe.schema, only=_PRETEXT_COMPONENTS
        )
    logits = values[Port.FEATURE_MASK_LOGITS]
    reconstruction = values[Port.RECONSTRUCTION]
    if not isinstance(logits, Tensor) or not isinstance(reconstruction, Tensor):
        raise TypeError("vime benchmark expected tensor pretext outputs")
    # The label `pretext_generator` returns and card §7 adopts: cells that
    # changed, not cells the mask selected.
    changed = clean.x != corrupted.x
    column_mean = population.rows.x.mean(dim=0).expand_as(clean.x)
    return {
        "dependent_ratio": _block_ratio(
            reconstruction, column_mean, clean.x, changed, DEPENDENT_BLOCK
        ),
        "independent_ratio": _block_ratio(
            reconstruction, column_mean, clean.x, changed, INDEPENDENT_BLOCK
        ),
        "mask_auroc": auroc(logits.flatten(), changed.flatten()),
        "corrupted_rate": float(changed.float().mean()),
    }


def _block_ratio(
    prediction: Tensor,
    baseline: Tensor,
    target: Tensor,
    changed: Tensor,
    columns: tuple[int, ...],
) -> float:
    """Corrupted-cell MSE over one column block, relative to the column mean."""
    selected = torch.tensor(columns, dtype=torch.long)
    cells = changed.index_select(1, selected)
    truth = target.index_select(1, selected)[cells]
    model = (prediction.index_select(1, selected)[cells] - truth).square().mean()
    mean = (baseline.index_select(1, selected)[cells] - truth).square().mean()
    if not bool(cells.any()) or float(mean) <= 0.0:
        raise RuntimeError(
            f"columns {columns!r} have no corrupted held-out cell with a "
            "positive baseline error, so the card's ratio is undefined"
        )
    return float(model / mean)


def auroc(scores: Tensor, labels: Tensor) -> float:
    """The Mann-Whitney AUROC, with tied scores sharing their average rank."""
    labels = labels.bool()
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC needs at least one positive and one negative")
    order = torch.argsort(scores.double())
    ordered = scores.double()[order]
    _, counts = torch.unique_consecutive(ordered, return_counts=True)
    ends = counts.cumsum(0).double()
    average = ends - (counts.double() - 1.0) / 2.0
    ranks = torch.empty_like(ordered)
    ranks[order] = torch.repeat_interleave(average, counts)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
