"""Deterministic array fitting for the P11 semi-supervised DML recipe."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from xty2.core.batch import XTYBatch
from xty2.core.card_keys import card_hyperparameters
from xty2.core.errors import CompileError, TrainingError
from xty2.core.rows import RowIndex
from xty2.core.schema import OutcomeSpec

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class SSDMLATEAction:
    """Five-fold ridge-nuisance AIPW fit returning portable tensor state."""

    num_treatments: int
    outcome: OutcomeSpec
    ridge_penalty: float
    propensity_clip: tuple[float, float]
    folds: int
    max_irls_iterations: int
    irls_relative_tolerance: float

    CARD_KEYS: ClassVar[Mapping[str, str]] = {
        "optimiser_description": "optimisation.optimiser",
        "weight_decay_description": "optimisation.weight_decay",
        "steps_description": "optimisation.total_steps_or_epochs",
        "widths_description": "architecture.widths_depths",
        "activation_description": "architecture.activation",
        "normalisation_description": "architecture.normalisation",
        "initialisation_description": "architecture.initialisation",
        "output_parameterisation": "architecture.output_parameterisation",
        "standardisation_description": "data.standardisation",
        "outcome_scaling": "data.outcome_scaling",
        "treatment_encoding_description": "data.treatment_encoding",
        "split_protocol_description": "data.split_protocol",
    }

    def __post_init__(self) -> None:
        if self.num_treatments != 2:
            raise CompileError(
                "SSDMLATEAction supports binary treatment only; "
                f"schema K must equal 2, got {self.num_treatments}"
            )
        if not isinstance(self.outcome, OutcomeSpec):
            raise CompileError(
                "SSDMLATEAction.outcome must be an OutcomeSpec, got "
                f"{type(self.outcome)}"
            )
        if not self.outcome.is_continuous or self.outcome.shape != ():
            raise CompileError(
                "SSDMLATEAction supports one scalar continuous outcome, got "
                f"kind={self.outcome.kind!r}, shape={self.outcome.shape!r}"
            )
        ridge_penalty: object = self.ridge_penalty
        if (
            isinstance(ridge_penalty, bool)
            or not isinstance(ridge_penalty, int | float)
            or not math.isfinite(float(ridge_penalty))
            or float(ridge_penalty) <= 0.0
        ):
            raise CompileError(
                "SSDMLATEAction.ridge_penalty must be finite and positive, got "
                f"{self.ridge_penalty!r}"
            )
        object.__setattr__(self, "ridge_penalty", float(ridge_penalty))
        propensity_clip: object = self.propensity_clip
        if not isinstance(propensity_clip, tuple | list) or len(propensity_clip) != 2:
            raise CompileError(
                "SSDMLATEAction.propensity_clip must be a (low, high) pair"
            )
        low, high = propensity_clip
        if any(
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            for value in (low, high)
        ):
            raise CompileError(
                "SSDMLATEAction.propensity_clip bounds must be finite numbers, "
                f"got {self.propensity_clip!r}"
            )
        resolved_clip = (float(low), float(high))
        if not 0.0 < resolved_clip[0] < resolved_clip[1] < 1.0:
            raise CompileError(
                "SSDMLATEAction.propensity_clip must satisfy 0 < low < high < 1, "
                f"got {resolved_clip!r}"
            )
        object.__setattr__(self, "propensity_clip", resolved_clip)
        if type(self.folds) is not int or self.folds != 5:
            raise CompileError(
                f"SSDMLATEAction requires exactly five folds, got {self.folds!r}"
            )
        if type(self.max_irls_iterations) is not int or self.max_irls_iterations < 1:
            raise CompileError(
                "SSDMLATEAction.max_irls_iterations must be a positive integer, "
                f"got {self.max_irls_iterations!r}"
            )
        irls_tolerance: object = self.irls_relative_tolerance
        if (
            isinstance(irls_tolerance, bool)
            or not isinstance(irls_tolerance, int | float)
            or not math.isfinite(float(irls_tolerance))
            or float(irls_tolerance) <= 0.0
        ):
            raise CompileError(
                "SSDMLATEAction.irls_relative_tolerance must be finite and "
                f"positive, got {self.irls_relative_tolerance!r}"
            )
        object.__setattr__(
            self,
            "irls_relative_tolerance",
            float(irls_tolerance),
        )
        card_hyperparameters(self)

    @property
    def name(self) -> str:
        return "ssdml_ate"

    @property
    def optimiser_description(self) -> str:
        low, high = self.propensity_clip
        return (
            "closed-form ridge solves plus Newton/IRLS logistic ridge; "
            f"zero start; held-out propensity clip [{low:g}, {high:g}]"
        )

    @property
    def weight_decay_description(self) -> str:
        return (
            f"ridge penalty {self.ridge_penalty:g} on nuisance slopes; intercept exempt"
        )

    @property
    def steps_description(self) -> str:
        return (
            f"{self.folds} held-out fold evaluations; IRLS maximum "
            f"{self.max_irls_iterations} iterations at relative parameter "
            f"tolerance {self.irls_relative_tolerance:g}"
        )

    @property
    def widths_description(self) -> str:
        return (
            "two linear ridge outcome nuisances and one logistic ridge "
            "propensity per fold"
        )

    @property
    def activation_description(self) -> str:
        return "identity outcome links; logistic propensity link"

    @property
    def normalisation_description(self) -> str:
        return "none inside action"

    @property
    def initialisation_description(self) -> str:
        return (
            "deterministic zero-start IRLS; ridge solves have no iterative "
            "initialisation"
        )

    @property
    def output_parameterisation(self) -> str:
        return "scalar ATE plus held-out nuisance and influence-score tensors"

    @property
    def standardisation_description(self) -> str:
        return (
            "propensity_labels: none; dml_ate: z-score X from each fold "
            "complement and freeze for its held-out fold"
        )

    @property
    def outcome_scaling(self) -> str:
        return "none"

    @property
    def treatment_encoding_description(self) -> str:
        return "binary integer 0/1 with t_observed mask; K must equal 2"

    @property
    def split_protocol_description(self) -> str:
        return (
            "supplied fold_id with exactly five non-empty folds; every nuisance "
            "prediction held out"
        )

    def fit(
        self,
        batch: XTYBatch,
        rows: RowIndex,
        *,
        seed: int,
    ) -> Mapping[str, Tensor]:
        """Fit fold-complement nuisances and aggregate held-out AIPW scores."""
        del seed  # The reviewed action is deterministic and has no random branch.
        if rows.dtype != torch.long or rows.ndim != 1:
            raise TrainingError(
                "SSDMLATEAction rows must be a one-dimensional long index"
            )
        if rows.numel() < 2:
            raise TrainingError("SSDMLATEAction needs at least two eligible rows")
        if batch.fold_id is None:
            raise TrainingError("SSDMLATEAction requires batch.fold_id")
        if batch.weight is not None:
            raise TrainingError(
                "SSDMLATEAction card defines no sample-weighted nuisance or score "
                "fit; pass an unweighted batch"
            )
        if not bool(batch.t_observed[rows].all()):
            raise TrainingError(
                "SSDMLATEAction received unresolved missing treatments; consume "
                "the propensity-label artifact before array fitting"
            )
        if not bool(batch.y_observed[rows].all()):
            raise TrainingError(
                "SSDMLATEAction requires the scalar outcome on every fitted row"
            )

        x = _float_array(batch.x[rows])
        y = _float_array(batch.y[rows]).reshape(-1)
        treatment = _int_array(batch.t[rows]).reshape(-1)
        fold_id = _int_array(batch.fold_id[rows]).reshape(-1)
        row_id = _int_array(batch.row_id[rows]).reshape(-1)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise TrainingError("SSDMLATEAction requires finite X and Y")
        if np.any((treatment < 0) | (treatment > 1)):
            raise TrainingError("SSDMLATEAction treatment values must be 0 or 1")

        fold_values = np.unique(fold_id)
        if fold_values.size != self.folds:
            raise TrainingError(
                "SSDMLATEAction requires exactly five non-empty fold ids, got "
                f"{fold_values.tolist()!r}"
            )

        n_rows = x.shape[0]
        g0_hat = np.empty(n_rows, dtype=np.float64)
        g1_hat = np.empty(n_rows, dtype=np.float64)
        m_hat = np.empty(n_rows, dtype=np.float64)
        penalty = _slope_penalty(x.shape[1])
        for fold in fold_values:
            held_out = fold_id == fold
            training = ~held_out
            x_train = x[training]
            x_held_out = x[held_out]
            mean = np.mean(x_train, axis=0)
            scale = np.std(x_train, axis=0, ddof=0)
            if not np.isfinite(mean).all() or not np.isfinite(scale).all():
                raise TrainingError(
                    f"SSDMLATEAction fold {int(fold)} produced non-finite "
                    "standardisation statistics"
                )
            if np.any(scale <= 0.0):
                raise TrainingError(
                    f"SSDMLATEAction fold {int(fold)} has a constant feature "
                    "on its training complement"
                )
            z_train = _with_intercept((x_train - mean) / scale)
            z_held_out = _with_intercept((x_held_out - mean) / scale)
            t_train = treatment[training]
            y_train = y[training]
            arm0 = t_train == 0
            arm1 = t_train == 1
            if not bool(arm0.any()) or not bool(arm1.any()):
                raise TrainingError(
                    f"SSDMLATEAction fold {int(fold)} training complement needs "
                    "both treatment arms"
                )
            beta0 = _ridge_fit(
                z_train[arm0],
                y_train[arm0],
                penalty,
                ridge=self.ridge_penalty,
                label=f"fold {int(fold)} outcome arm 0",
            )
            beta1 = _ridge_fit(
                z_train[arm1],
                y_train[arm1],
                penalty,
                ridge=self.ridge_penalty,
                label=f"fold {int(fold)} outcome arm 1",
            )
            propensity_beta = _logistic_ridge_fit(
                z_train,
                t_train.astype(np.float64),
                penalty,
                ridge=self.ridge_penalty,
                max_iterations=self.max_irls_iterations,
                tolerance=self.irls_relative_tolerance,
                label=f"fold {int(fold)} propensity",
            )
            g0_hat[held_out] = z_held_out @ beta0
            g1_hat[held_out] = z_held_out @ beta1
            held_out_propensity = _sigmoid(z_held_out @ propensity_beta)
            m_hat[held_out] = np.clip(
                held_out_propensity,
                self.propensity_clip[0],
                self.propensity_clip[1],
            )

        if not all(np.isfinite(value).all() for value in (g0_hat, g1_hat, m_hat)):
            raise TrainingError(
                "SSDMLATEAction produced a non-finite held-out nuisance prediction"
            )
        treatment_float = treatment.astype(np.float64)
        score = (
            g1_hat
            - g0_hat
            + treatment_float * (y - g1_hat) / m_hat
            - (1.0 - treatment_float) * (y - g0_hat) / (1.0 - m_hat)
        )
        ate = float(np.mean(score))
        influence = score - ate
        standard_error = float(np.std(influence, ddof=1) / math.sqrt(n_rows))
        if not math.isfinite(ate) or not math.isfinite(standard_error):
            raise TrainingError("SSDMLATEAction produced a non-finite ATE state")

        dtype = batch.y.dtype
        return {
            "ate": torch.tensor(ate, dtype=dtype),
            "diagnostic_standard_error": torch.tensor(standard_error, dtype=dtype),
            "influence_score": _tensor(influence, dtype=dtype),
            "g0_hat": _tensor(g0_hat, dtype=dtype),
            "g1_hat": _tensor(g1_hat, dtype=dtype),
            "m_hat": _tensor(m_hat, dtype=dtype),
            "row_id": torch.from_numpy(row_id.copy()),
            "fold_id": torch.from_numpy(fold_id.copy()),
        }


def _float_array(value: Tensor) -> FloatArray:
    return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)


def _int_array(value: Tensor) -> IntArray:
    return np.asarray(value.detach().cpu().numpy(), dtype=np.int64)


def _with_intercept(x: FloatArray) -> FloatArray:
    return np.column_stack((np.ones(x.shape[0], dtype=np.float64), x))


def _slope_penalty(num_features: int) -> FloatArray:
    penalty = np.eye(num_features + 1, dtype=np.float64)
    penalty[0, 0] = 0.0
    return penalty


def _ridge_fit(
    design: FloatArray,
    target: FloatArray,
    penalty: FloatArray,
    *,
    ridge: float,
    label: str,
) -> FloatArray:
    system = design.T @ design + ridge * penalty
    right = design.T @ target
    try:
        solution = np.linalg.solve(system, right)
    except np.linalg.LinAlgError as error:
        raise TrainingError(f"SSDMLATEAction {label} ridge solve failed") from error
    resolved = np.asarray(solution, dtype=np.float64)
    if not np.isfinite(resolved).all():
        raise TrainingError(
            f"SSDMLATEAction {label} ridge solve returned non-finite coefficients"
        )
    return resolved


def _logistic_ridge_fit(
    design: FloatArray,
    target: FloatArray,
    penalty: FloatArray,
    *,
    ridge: float,
    max_iterations: int,
    tolerance: float,
    label: str,
) -> FloatArray:
    beta = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(max_iterations):
        probability = _sigmoid(design @ beta)
        gradient = design.T @ (probability - target) + ridge * (penalty @ beta)
        variance = probability * (1.0 - probability)
        hessian = design.T @ (variance[:, None] * design) + ridge * penalty
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise TrainingError(
                f"SSDMLATEAction {label} Newton/IRLS solve failed"
            ) from error
        candidate = beta - np.asarray(step, dtype=np.float64)
        if not np.isfinite(candidate).all():
            raise TrainingError(
                f"SSDMLATEAction {label} Newton/IRLS returned non-finite coefficients"
            )
        relative_change = float(
            np.linalg.norm(candidate - beta)
            / max(1.0, float(np.linalg.norm(candidate)))
        )
        beta = candidate.reshape(-1)
        if relative_change <= tolerance:
            return beta
    raise TrainingError(
        f"SSDMLATEAction {label} Newton/IRLS did not converge within "
        f"{max_iterations} iterations at relative tolerance {tolerance:g}"
    )


def _sigmoid(value: FloatArray) -> FloatArray:
    result = np.empty_like(value)
    nonnegative = value >= 0.0
    result[nonnegative] = 1.0 / (1.0 + np.exp(-value[nonnegative]))
    exponential = np.exp(value[~nonnegative])
    result[~nonnegative] = exponential / (1.0 + exponential)
    return result


def _tensor(value: FloatArray, *, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(value.copy(), dtype=dtype)


__all__ = ["SSDMLATEAction"]
