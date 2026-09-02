"""PAWS's paired pretrained/unpretrained benchmark from card section 6.

The measurement is the pair: one `joint_fit` stage, the same seeds, the same
batch stream and bit-identical initial parameters, run once from the
PAWS-pretrained encoder and once from the recipe's initialisation. Nothing here
compares against Assran et al., whose evidence is ImageNet and CIFAR-10 top-1
accuracy on data carrying no treatment.

Three guardrails travel with the pair, and each exists so that a null downstream
result can be attributed to the representation rather than to the fine-tuning
stage:

* **paws-nn** — the paper's own test-time readout, `arg max_k [pi_d(z, z_S)]_k`,
  scored before the `arg max` as an NLL over held-out rows with the full
  labelled training pool as support. It is measured on the *pretraining*
  checkpoint, which is the representation the pair is about; `joint_fit` trains
  the encoder onward and drops the projection head.
* **Terminal `H(p_bar)`** — the paper's §4 non-collapse condition, read from the
  me-max term at the last pretraining step rather than recomputed here.
* **Positive-view alignment** — a row's similarity to its own second large view
  against its similarity to the other anchors of the batch. A collapsed encoder
  clears the entropy condition and fails this one.

The recipe declares the label budget, the outcome standardisation and the
quotas, so this module masks nothing, standardises nothing and sizes no batch:
all three are `DataSpec` and `QuotaSampler` declarations applied by the loader
and printed in the plan.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from xty2.core import (
    CategoricalTreatment,
    CompiledRun,
    GaussianOutcome,
    Port,
    Program,
    Recipe,
    Schema,
    TrainingPopulation,
    XTYBatch,
    compile,
)
from xty2.evaluation.benchmarks.common import (
    ClusterPopulation,
    column,
    configure_worker,
    continuous_schema,
    on_the_training_scale,
    parallel_replicates,
    take,
    training_dataset,
    two_cluster_population,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.recipes import paws
from xty2.recipes.paws import (
    MISSING_ANCHORS,
    PRETRAIN_STEPS,
    SUPPORT_CLASSIFIER,
    SUPPORT_PER_TREATMENT,
)
from xty2.training import STREAM_STRIDE, Checkpoint, ProgramResult, run_program

_TRAIN_ROWS = 1_024
_TEST_ROWS = 2_048
_BASE_SEED = 90_000
_ME_MAX = "mean_entropy_maximisation"
_REPRESENTATION = ("mlp_encoder", "projection_head")
"""What pretraining trains, and so what the guardrails may read."""


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the ten paired fits predeclared by ``paws.md`` section 6."""
    del cache_root
    spec.bind(
        {
            "dataset": (
                "project-local seed-locked two-cluster XTY DGP (6 features, "
                "K=2), specified in 6.1"
            ),
            "variant": (
                "paired fit against the identical joint_fit stage with no "
                "pretraining, same seeds and same batches"
            ),
            "split": (
                "1024 train rows with 64 observed treatments, 2048 held-out "
                "rows with every treatment observed"
            ),
            "metric": (
                "held-out p(t|x) NLL ratio, pretrained over unpretrained; "
                "paws-nn held-out treatment NLL over the full labelled pool as "
                "support, and terminal H(p_bar), as mechanism guardrails"
            ),
            "published": "none - no published number applies to this adaptation",
            "tolerance": (
                "NLL ratio < 1.0 in mean by at least one standard error; "
                "held-out outcome NLL within 1.05x of the unpretrained arm; "
                "paws-nn held-out treatment NLL below the fixture's "
                "marginal-prior NLL; terminal H(p_bar) at least 0.95 * log(K); "
                "mean cosine similarity of a row to its own second large view "
                "at least 0.2 above its mean similarity to the other anchors "
                "of the batch"
            ),
            "seeds": "10",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 10:
        raise ValueError(f"paws card reviewed ten replicates, got {spec.seed_count}")
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            # The card writes "< 1.0" and this is "<= 1.0", the same single
            # point of a continuous statistic `scarf`'s benchmark records: a
            # strict relation added to the reporting vocabulary for one card
            # would be the convenience quadrant of `DESIGN.md` §11.2 with
            # nothing riding on it.
            MetricResult.upper_bound(
                "held_out_treatment_NLL_ratio",
                column(rows, "treatment_ratio"),
                1.0,
            ),
            MetricResult.upper_bound(
                "held_out_outcome_NLL_ratio",
                column(rows, "outcome_ratio"),
                1.05,
            ),
            # "below the fixture's marginal-prior NLL" is a per-replicate
            # comparison, so it is carried as a ratio rather than as two means
            # differenced after the fact: the pairing is what removes the
            # replicate-to-replicate variation in how hard the draw is.
            MetricResult.upper_bound(
                "paws_nn_over_marginal_prior_NLL",
                column(rows, "paws_nn_ratio"),
                1.0,
            ),
            MetricResult.lower_bound(
                "terminal_marginal_entropy",
                column(rows, "marginal_entropy"),
                0.95 * math.log(2.0),
                unit="nat",
            ),
            MetricResult.lower_bound(
                "positive_view_alignment_gap",
                column(rows, "alignment_gap"),
                0.2,
            ),
            MetricResult.information(
                "pretrained_treatment_NLL",
                column(rows, "pretrained_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "unpretrained_treatment_NLL",
                column(rows, "unpretrained_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "marginal_frequency_NLL",
                column(rows, "frequency_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "paws_nn_treatment_NLL",
                column(rows, "paws_nn_nll"),
                unit="nat/row",
            ),
            MetricResult.information(
                "paws_nn_accuracy", column(rows, "paws_nn_accuracy")
            ),
            MetricResult.information(
                "pretrained_outcome_NLL",
                column(rows, "pretrained_outcome_nll"),
                unit="nat/row",
            ),
        ),
        interpretation=(
            "This is the predeclared project-local PAWS mechanism target: does "
            "an encoder pretrained by predicting view assignments against a "
            "class-stratified support set of observed treatments help a "
            "scarce-label treatment fit. It is not a reproduction of Assran et "
            "al., whose evidence is ImageNet and CIFAR-10 top-1 accuracy under "
            "label-fraction splits and whose downstream task carries no "
            "treatment."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = _BASE_SEED + 100 * index
    schema = continuous_schema(6)
    train = two_cluster_population(_TRAIN_ROWS, seed=base + 1, row_offset=0)
    test = two_cluster_population(_TEST_ROWS, seed=base + 2, row_offset=10_000)
    data = training_dataset(schema, train.batch)

    # Both arms start from bit-identical parameters: the pairing is the whole
    # measurement, so a difference in initialisation would be a second variable.
    torch.manual_seed(base + 6)
    pretrained_recipe = paws(schema)
    torch.manual_seed(base + 6)
    ablated_recipe = _unpretrained(paws(schema))
    for name, value in pretrained_recipe.system.state_dict().items():
        if not torch.equal(value, ablated_recipe.system.state_dict()[name]):
            raise RuntimeError(f"paws paired initial state differs at {name!r}")

    pretrained_run = compile(pretrained_recipe)
    ablated_run = compile(ablated_recipe)
    full = run_program(
        pretrained_run,
        {"pretrain": data, "joint_fit": data},
        seed=base + 10_000,
    )
    # The ablation's single stage is index 0, so its seed is offset by one
    # stride to give its fit the same stochastic stream as the paired arm's.
    bare = run_program(
        ablated_run, {"joint_fit": data}, seed=base + 10_000 + STREAM_STRIDE
    )
    if not torch.equal(
        full.stage("joint_fit").checkpoint.trained_on_row_ids,
        bare.stage("joint_fit").checkpoint.trained_on_row_ids,
    ):
        raise RuntimeError("paws paired arms fitted on different training rows")

    pretrained = _evaluate(pretrained_run, full, test)
    unpretrained = _evaluate(ablated_run, bare, test)
    if unpretrained["treatment_nll"] <= 0.0 or unpretrained["outcome_nll"] <= 0.0:
        raise RuntimeError(
            "the unpretrained arm produced a non-positive NLL, so the paired "
            "ratio the card declares is undefined"
        )
    guardrails = _pretraining_guardrails(schema, full, test, seed=base + 20_000)
    return {
        "treatment_ratio": (
            pretrained["treatment_nll"] / unpretrained["treatment_nll"]
        ),
        "outcome_ratio": pretrained["outcome_nll"] / unpretrained["outcome_nll"],
        "paws_nn_ratio": guardrails["paws_nn_nll"] / pretrained["frequency_nll"],
        "marginal_entropy": _terminal_entropy(full),
        "alignment_gap": guardrails["alignment_gap"],
        "pretrained_nll": pretrained["treatment_nll"],
        "unpretrained_nll": unpretrained["treatment_nll"],
        "frequency_nll": pretrained["frequency_nll"],
        "paws_nn_nll": guardrails["paws_nn_nll"],
        "paws_nn_accuracy": guardrails["paws_nn_accuracy"],
        "pretrained_outcome_nll": pretrained["outcome_nll"],
    }


def _unpretrained(recipe: Recipe) -> Recipe:
    """The ablation: the same fitting stage, from the recipe's initialisation.

    `initialise_from` is what the pretraining delivers, so dropping the stage
    without dropping the edge would leave a program that cannot run. The two
    view specifications go with it: only `pretrain` realises the eight draws,
    and `DESIGN.md` §2.1's compiler rule refuses a recipe advertising draws
    nothing reads. That removal cannot reach the fit — a stage's batches are
    drawn from `sampler_seed(stage seed)` over the declared quotas
    (`xty2/training/loading.py`), and `joint_fit`'s objectives realise no view
    — and `_replicate` checks the two arms' fitted row ids agree.

    Nothing else is touched: same graph, same objectives, same optimiser, same
    sampler, same quotas, same budget. The support quota stays too, because it
    draws the rows `observed_treatment_nll` and `observed_outcome_nll` need.
    """
    fit = recipe.program[1]
    return replace(
        recipe,
        program=Program((replace(fit, initialise_from=None),)),
        views=(),
    )


def _evaluate(
    run: CompiledRun, result: ProgramResult, test: ClusterPopulation
) -> dict[str, float]:
    """Held-out NLLs after `joint_fit`, on the outcome scale the run fitted."""
    population = _population(result)
    scaled = on_the_training_scale(test.batch, population)
    schema = run.recipe.schema
    with torch.no_grad():
        values = run.graph.evaluate(
            scaled,
            schema=schema,
            only=("mlp_encoder", "tarnet_head", "categorical_propensity"),
        )
        propensity = values[Port.T_GIVEN_X]
        outcome = values[Port.Y_GIVEN_XT]
        if not isinstance(propensity, CategoricalTreatment) or not isinstance(
            outcome, GaussianOutcome
        ):
            raise TypeError("paws benchmark expected its reviewed P5 heads")
        treatment_nll = float(F.nll_loss(propensity.log_probs, scaled.t))
        outcome_nll = float(-outcome.log_prob(scaled.y, scaled.t).mean())
        # The baseline is the *labelled* training rows' marginal, which is what
        # a model with no covariate information would predict. It reads the
        # policy's mask rather than the source data's, because that is the
        # supervision the fit actually had.
        observed = population.rows.t[population.rows.t_observed]
        frequencies = torch.bincount(
            observed, minlength=schema.treatment_cardinality
        ).float()
        frequencies /= frequencies.sum()
        baseline = frequencies.log().expand(scaled.batch_size, -1)
        frequency_nll = float(F.nll_loss(baseline, scaled.t))
    return {
        "treatment_nll": treatment_nll,
        "outcome_nll": outcome_nll,
        "frequency_nll": frequency_nll,
    }


def _pretraining_guardrails(
    schema: Schema,
    result: ProgramResult,
    test: ClusterPopulation,
    *,
    seed: int,
) -> dict[str, float]:
    """paws-nn and positive-view alignment on the *pretraining* checkpoint.

    `joint_fit` trains the encoder onward and leaves the projection head where
    pretraining put it, so the live graph after the program is neither arm of
    the question these two guardrails ask. The checkpoint is, and it carries
    exactly the two components pretraining trained.
    """
    stage = result.stage("pretrain")
    population = _population(result)
    representation = _restored(schema, stage.checkpoint)
    scaled = on_the_training_scale(test.batch, population)
    support = _support(population)
    paws_nn_nll, paws_nn_accuracy = _paws_nn(
        representation, population, support, scaled
    )
    return {
        "paws_nn_nll": paws_nn_nll,
        "paws_nn_accuracy": paws_nn_accuracy,
        "alignment_gap": _alignment_gap(representation, population, seed=seed),
    }


def _restored(schema: Schema, checkpoint: Checkpoint) -> CompiledRun:
    """A compiled run holding the pretraining checkpoint's parameters.

    Restoring into a *fresh* compilation rather than into the run that produced
    it keeps the measurement independent of when it is taken: the live graph
    still holds whatever `joint_fit` last wrote, and the downstream numbers are
    read from it.
    """
    if checkpoint.components != _REPRESENTATION:
        raise RuntimeError(
            f"the paws pretraining checkpoint carries {checkpoint.components!r}; "
            f"these guardrails read {_REPRESENTATION!r} and would otherwise "
            "score an untrained component"
        )
    torch.manual_seed(0)
    run = compile(paws(schema))
    state = dict(checkpoint.parameters) | dict(checkpoint.buffers)
    for component in checkpoint.components:
        prefix = f"{component}."
        run.graph[component].load_state_dict(
            {
                name[len(prefix) :]: value
                for name, value in state.items()
                if name.startswith(prefix)
            },
            strict=True,
        )
    return run


def _population(result: ProgramResult) -> TrainingPopulation:
    """The declared policy's training population, shared by every stage."""
    population = result.stage("joint_fit").population
    if population is None:
        raise RuntimeError(
            "the fitting stage reported no training population; the recipe "
            "declares a sampler, so one is expected"
        )
    return population


def _support(population: TrainingPopulation) -> Tensor:
    """The full labelled pool, and card §6.1's fillable-quota assertion.

    A replicate whose scarcer level cannot fill the 16-row stratified quota is
    a card amendment — a smaller quota or a larger budget — and never a
    re-seed, so it stops the run rather than being worked around here.
    """
    rows = population.rows.t_observed.nonzero(as_tuple=False).flatten()
    counts = torch.bincount(population.rows.t.index_select(0, rows), minlength=2)
    if int(counts.min()) < SUPPORT_PER_TREATMENT:
        raise RuntimeError(
            f"this replicate's labelled pool holds {counts.tolist()!r} rows per "
            f"treatment and the card's support quota draws {SUPPORT_PER_TREATMENT} "
            "of each; §6.1 makes that a card amendment, not a re-seed"
        )
    return rows


def _paws_nn(
    run: CompiledRun,
    population: TrainingPopulation,
    support_rows: Tensor,
    test: XTYBatch,
) -> tuple[float, float]:
    """`pi_d(z, z_S)` over the full labelled pool, scored before the arg max.

    Temperature and label smoothing come from the recipe's own
    `SupportSetClassifier`, so the readout cannot drift from the objective that
    trained the representation it reads.
    """
    schema = run.recipe.schema
    with torch.no_grad():
        support = _projection(run, population.rows).index_select(0, support_rows)
        query = _projection(run, test)
        labels = population.rows.t.index_select(0, support_rows)
        smoothed = SUPPORT_CLASSIFIER.smoothed_labels(
            labels, classes=schema.treatment_cardinality
        ).to(dtype=query.dtype)
        weights = torch.softmax(
            F.normalize(query, dim=-1)
            @ F.normalize(support, dim=-1).transpose(0, 1)
            / SUPPORT_CLASSIFIER.temperature,
            dim=-1,
        )
        probabilities = weights @ smoothed
        return (
            float(F.nll_loss(probabilities.log(), test.t)),
            float((probabilities.argmax(dim=-1) == test.t).float().mean()),
        )


def _alignment_gap(
    run: CompiledRun, population: TrainingPopulation, *, seed: int
) -> float:
    """A row's own second large view against the batch's other anchors.

    The card's non-collapse pair: terminal `H(p_bar)` says the *marginal* is
    not degenerate, and this says the per-row embeddings are not all the same
    point. One batch of the declared anchor quota, under the two declared large
    draws, on a view key the training run never walked.
    """
    anchors = population.rows.t_missing.nonzero(as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(anchors.numel(), generator=generator)
    batch = take(population.rows, anchors.index_select(0, order[:MISSING_ANCHORS]))
    view = run.recipe.view("paws_large_x")
    with torch.no_grad():
        embeddings = [
            F.normalize(
                _projection(
                    run,
                    view.apply(
                        batch,
                        run.recipe.schema,
                        rng_key=seed,
                        draw=draw,
                        population=population,
                    ),
                ),
                dim=-1,
            )
            for draw in (0, 1)
        ]
        similarity = embeddings[0] @ embeddings[1].transpose(0, 1)
        rows = similarity.shape[0]
        own = similarity.diagonal().sum()
        others = (similarity.sum() - own) / (rows * (rows - 1))
        return float(own / rows - others)


def _projection(run: CompiledRun, batch: XTYBatch) -> Tensor:
    value = run.graph.evaluate(batch, schema=run.recipe.schema, only=_REPRESENTATION)[
        Port.X_PROJ
    ]
    if not isinstance(value, Tensor):
        raise TypeError(f"paws benchmark expected an embedding, got {type(value)}")
    return value


def _terminal_entropy(result: ProgramResult) -> float:
    """`H(p_bar)` at the last pretraining step, as the me-max term saw it."""
    record = result.stage("pretrain").records[PRETRAIN_STEPS - 1]
    term = next(entry for entry in record.terms if entry.name == _ME_MAX)
    return float(term.diagnostics["marginal_entropy"])


__all__ = ["run"]
