"""Ten-seed BYOL study bound to the prospective card section 6 protocol."""

from pathlib import Path

from xty2.evaluation.benchmarks.common import column, parallel_replicates
from xty2.evaluation.byol_study import study
from xty2.evaluation.reporting import BenchmarkResult, MetricResult, ReproductionSpec


# Card §6.2: a stream disjoint from 620000-620900, which the 2026-09-16 run and
# the 2026-09-17 audit both saw. §6.4's statistics were chosen on that audit;
# their thresholds are each statistic's own zero and were not read off it.
def replicate(index: int) -> dict[str, float]:
    return study(
        630000 + 100 * index,
        train_rows=1024,
        test_rows=2048,
        pretrain_steps=1000,
        fit_steps=3000,
        warmup_steps=10,
        ramp_steps=1000,
        eval_batches=16,
    )


def run(
    spec: ReproductionSpec, commit: str, date: str, workers: int, cache_root: Path
) -> BenchmarkResult:
    del cache_root
    spec.bind(
        {
            "dataset": (
                "common.two_cluster_population with low=SEPARATED; six features; "
                "K=2; OracleSymmetry views"
            ),
            "variant": (
                "scheduled EMA versus zero-decay target, the inherited "
                "1000-epoch EMA base, no predictor, and no pretraining"
            ),
            "split": (
                "1024 train, 40 observed treatments; 2048 fully observed "
                "held-out rows; frozen-encoder transfer"
            ),
            "metric": (
                "ema_alignment_gap; ema_rank_gap; "
                "pretraining_outcome_nll_cost; encoder_effective_rank"
            ),
            "published": "none - project-local tabular adaptation",
            "published_source": "n/a",
            "tolerance": "all four one-standard-error bounds in section 6.4",
            "seeds": "10",
            "report": "mean_and_stderr",
        }
    )
    rows = parallel_replicates(replicate, spec.seed_count, workers=workers)
    required = (
        MetricResult(
            "ema_alignment_gap",
            tuple(column(rows, "ema_alignment_gap")),
            ">",
            0.0,
        ),
        MetricResult(
            "ema_rank_gap",
            tuple(column(rows, "ema_rank_gap")),
            ">",
            0.0,
        ),
        MetricResult(
            "pretraining_outcome_nll_cost",
            tuple(column(rows, "pretraining_outcome_nll_cost")),
            "<",
            0.05,
            unit="nat/row",
        ),
        MetricResult(
            "encoder_effective_rank",
            tuple(column(rows, "encoder_effective_rank")),
            ">",
            1.1,
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
            "Project-local BYOL mechanism study, not ImageNet reproduction. "
            "All five arms share actual common initial tensors, fitted scales, "
            "masks and batch streams; pretraining arms share actual view draws. "
            "EMA changes only target parameter decay; target BN owns its state. "
            "Contrasts are formed within each of the ten predeclared seeds. "
            "Attribution is carried by alignment and rank read from the terminal "
            "pretraining checkpoint; downstream NLL is a non-inferiority budget "
            "and no longer a superiority gate (section 6.4, amendment 2026-09-17). "
            "The encoder transfers frozen, so downstream numbers are a probe of "
            "the pretrained representation rather than of a fine-tuned one. "
            "Rank is measured on all 2048 clean held-out rows before transfer, "
            "without BN recalibration. OracleSymmetry uses privileged DGP "
            "knowledge; no general tabular or causal-identification claim follows."
        ),
    )
