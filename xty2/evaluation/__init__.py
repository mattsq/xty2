"""Evaluation metrics and the card-driven Tier 2 benchmark surface (P12)."""

from xty2.evaluation.calibration import (
    classification_accuracy,
    expected_calibration_error,
    multiclass_brier_score,
)
from xty2.evaluation.causal import (
    absolute_ate_error,
    average_treatment_effect,
    candidate_treatment_means,
    sqrt_pehe,
    treatment_contrast,
)
from xty2.evaluation.predictive import (
    conditional_outcome_nll,
    root_mean_squared_error,
    treatment_nll,
)
from xty2.evaluation.reporting import (
    BenchmarkResult,
    MetricResult,
    ReproductionSpec,
    assert_result_matches_card,
    load_reproduction_spec,
    mean_and_stderr,
    update_card,
    update_card_text,
)

__all__ = [
    "BenchmarkResult",
    "MetricResult",
    "ReproductionSpec",
    "absolute_ate_error",
    "assert_result_matches_card",
    "average_treatment_effect",
    "candidate_treatment_means",
    "classification_accuracy",
    "conditional_outcome_nll",
    "expected_calibration_error",
    "load_reproduction_spec",
    "mean_and_stderr",
    "multiclass_brier_score",
    "root_mean_squared_error",
    "sqrt_pehe",
    "treatment_contrast",
    "treatment_nll",
    "update_card",
    "update_card_text",
]
