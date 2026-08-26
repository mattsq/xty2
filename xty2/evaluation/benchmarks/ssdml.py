"""SSDML's staged-imputation DML2 benchmark from card section 6."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor

from xty2.core import XTYBatch, compile, resolve_rows
from xty2.estimators import SSDMLATEAction
from xty2.evaluation.benchmarks.common import (
    bool_float,
    column,
    configure_worker,
    continuous_schema,
    parallel_replicates,
)
from xty2.evaluation.causal import absolute_ate_error
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
)
from xty2.recipes import ssdml
from xty2.training import run_program

_ROWS = 4_000


def run(
    spec: ReproductionSpec,
    commit: str,
    date: str,
    workers: int,
    cache_root: Path,
) -> BenchmarkResult:
    """Run the twenty-replicate staged and oracle-treatment DML fits."""
    del cache_root
    spec.bind(
        {
            "dataset": "fixed project-local staged-imputation IRM DGP",
            "variant": (
                "binary treatment; 50% treatment MCAR; staged hard imputation; "
                "five-fold DML2"
            ),
            "split": (
                "one independent 4000-row estimation population per replicate; "
                "five held-out folds"
            ),
            "metric": "absolute_ATE_error",
            "published": "n/a",
            "tolerance": "0.30 from analytic ATE 1.0",
            "seeds": "20",
            "report": "mean_and_stderr",
        },
        documentation=("published_source",),
    )
    if spec.seed_count != 20:
        raise ValueError(
            f"SSDML card reviewed twenty replicates, got {spec.seed_count}"
        )
    rows = parallel_replicates(_replicate, spec.seed_count, workers=workers)
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            MetricResult.upper_bound(
                "staged_absolute_ATE_error",
                column(rows, "staged_error"),
                0.30,
            ),
            MetricResult.upper_bound(
                "oracle_treatment_absolute_ATE_error",
                column(rows, "oracle_error"),
                0.10,
            ),
            MetricResult.lower_bound(
                "out_of_fold_without_y",
                column(rows, "provenance"),
                1.0,
            ),
            MetricResult.lower_bound(
                "finite_complete_clipped_state",
                column(rows, "state_valid"),
                1.0,
            ),
            MetricResult.lower_bound(
                "deterministic_array_state",
                column(rows, "repeatable"),
                1.0,
            ),
            MetricResult.lower_bound(
                "source_batch_unchanged",
                column(rows, "source_unchanged"),
                1.0,
            ),
            MetricResult.information(
                "staged_minus_oracle_ATE",
                column(rows, "staged_minus_oracle"),
            ),
            MetricResult.information(
                "hidden_treatment_accuracy",
                column(rows, "hidden_accuracy"),
            ),
        ),
        interpretation=(
            "This matches the predeclared project-local staged-imputation IRM "
            "target. It validates deterministic executor/artifact plumbing and "
            "does not transfer the DML paper's inference claim to hard labels."
        ),
    )


def _replicate(index: int) -> dict[str, float]:
    configure_worker()
    base = 120_000 + 100 * index
    batch = _population(seed=base)
    before = batch.clone()
    torch.manual_seed(base + 10)
    run = compile(ssdml(continuous_schema(6)))
    result = run_program(
        run,
        {
            "propensity_labels": (batch,),
            "dml_ate": (batch,),
        },
        seed=base + 10,
    )
    labels = result.stage("propensity_labels").pseudo_labels
    if labels is None:
        raise RuntimeError("SSDML emitted no propensity labels")
    joined = labels.apply_to(batch, treatment_cardinality=2)
    hidden_accuracy = float(
        (joined.t[batch.t_missing] == batch.t[batch.t_missing]).float().mean()
    )
    staged = _unprefix(result.stage("dml_ate").checkpoint.parameters)
    action = run.stage("dml_ate").action
    if not isinstance(action, SSDMLATEAction):
        raise TypeError("SSDML benchmark expected SSDMLATEAction")
    repeated = action.fit(joined, resolve_rows(joined, "all"), seed=base + 99)
    oracle = batch.replace(t_observed=torch.ones(_ROWS, dtype=torch.bool))
    oracle_state = action.fit(
        oracle,
        resolve_rows(oracle, "all"),
        seed=base + 99,
    )
    staged_ate = float(staged["ate"])
    oracle_ate = float(oracle_state["ate"])
    state_valid = _valid_state(staged)
    repeatable = set(staged) == set(repeated) and all(
        torch.equal(staged[name], repeated[name]) for name in staged
    )
    provenance = labels.prediction_mode == "out_of_fold" and labels.used_y is False
    return {
        "staged_error": absolute_ate_error(staged_ate, 1.0),
        "oracle_error": absolute_ate_error(oracle_ate, 1.0),
        "staged_minus_oracle": staged_ate - oracle_ate,
        "hidden_accuracy": hidden_accuracy,
        "provenance": bool_float(provenance),
        "state_valid": bool_float(state_valid),
        "repeatable": bool_float(repeatable),
        "source_unchanged": bool_float(batch.equal_to(before)),
    }


def _population(*, seed: int) -> XTYBatch:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(_ROWS, 6, generator=generator)
    treatment_uniform = torch.rand(_ROWS, generator=generator)
    outcome_noise = torch.randn(_ROWS, generator=generator)
    missingness_uniform = torch.rand(_ROWS, generator=generator)
    propensity = 0.05 + 0.90 * torch.sigmoid(
        1.25 * x[:, 0] - 0.75 * x[:, 1] + 0.5 * x[:, 2]
    )
    treatment = (treatment_uniform < propensity).long()
    baseline = x[:, 0] + 0.5 * x[:, 1] - 0.25 * x[:, 2]
    effect = 1.0 + 0.25 * x[:, 3]
    outcome = baseline + treatment * effect + outcome_noise
    row_id = torch.arange(_ROWS)
    return XTYBatch(
        x=x,
        t=treatment,
        y=outcome,
        t_observed=missingness_uniform < 0.50,
        y_observed=torch.ones(_ROWS, dtype=torch.bool),
        row_id=row_id,
        fold_id=row_id % 5,
    )


def _unprefix(parameters: Mapping[str, Tensor]) -> dict[str, Tensor]:
    prefix = "ssdml_ate."
    if any(not name.startswith(prefix) for name in parameters):
        raise RuntimeError(
            f"SSDML array checkpoint has unexpected keys {sorted(parameters)!r}"
        )
    return {name.removeprefix(prefix): value for name, value in parameters.items()}


def _valid_state(state: Mapping[str, Tensor]) -> bool:
    expected = {
        "ate",
        "diagnostic_standard_error",
        "influence_score",
        "g0_hat",
        "g1_hat",
        "m_hat",
        "row_id",
        "fold_id",
    }
    if set(state) != expected:
        return False
    if any(not bool(torch.isfinite(value).all()) for value in state.values()):
        return False
    propensity = state["m_hat"]
    return bool((propensity >= 0.025).all() and (propensity <= 0.975).all())


__all__ = ["run"]
