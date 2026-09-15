"""Ten-seed Tier 2 study, bound by value to SimSiam card section 6."""

from pathlib import Path

from xty2.evaluation.benchmarks.common import column, parallel_replicates
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec
from xty2.evaluation.simsiam_study import study


def replicate(index: int) -> dict[str, float]:
    return study(
        520000 + 100 * index,
        train_rows=1024,
        test_rows=2048,
        pretrain_steps=1000,
        fit_steps=3000,
        eval_batches=16,
    )


def run(
    spec: ReproductionSpec, commit: str, date: str, workers: int, cache_root: Path
) -> BenchmarkResult:
    del cache_root
    spec.bind(
        {
            "dataset": (
                "shared two_cluster_population DGP; six features; K=2; "
                "OracleSymmetry views"
            ),
            "variant": "full versus no-stop-gradient, no-predictor and no-pretraining",
            "split": (
                "1024 train with 40 observed treatments; "
                "2048 fully observed held-out rows"
            ),
            "metric": (
                "two paired view-alignment gaps; two paired encoder "
                "effective-rank gaps; factual outcome NLL cost"
            ),
            "published": "none - project-local tabular adaptation",
            "published_source": "n/a",
            "tolerance": "all five one-standard-error bounds in section 6.4",
            "seeds": "10",
            "report": "mean_and_stderr",
        }
    )
    rows = parallel_replicates(replicate, spec.seed_count, workers=workers)
    required = (
        *(
            MetricResult(name, tuple(column(rows, name)), ">", 0.0)
            for name in (
                "stop_gradient_alignment_gap",
                "predictor_alignment_gap",
                "stop_gradient_rank_gap",
                "predictor_rank_gap",
            )
        ),
        MetricResult.upper_bound(
            "pretraining_outcome_nll_cost",
            column(rows, "pretraining_outcome_nll_cost"),
            0.05,
            unit="nat/row",
        ),
    )
    scored = {metric.name for metric in required}
    return BenchmarkResult(
        recipe=spec.recipe,
        commit=commit,
        date=date,
        spec_digest=spec.digest,
        metrics=(
            *required,
            *(
                MetricResult.information(name, column(rows, name))
                for name in sorted(rows[0])
                if name not in scored
            ),
        ),
        interpretation=(
            "Project-local SimSiam mechanism study with privileged DGP views, "
            "tabular widths and a reduced batch and step budget. Every arm shares "
            "actual initial common tensors, fitted scales, treatment masks, "
            "batches and view draws; stage transitions and optimiser reset are "
            "checked during execution. Differences are paired within seed. "
            "Both attribution axes are paired, so uniform collapse fails them "
            "rather than passing. This is not an ImageNet or "
            "causal-identification claim."
        ),
    )
