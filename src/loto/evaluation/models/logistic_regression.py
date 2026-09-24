"""Pre-registered per-number regularized logistic inclusion model.

The family consists of 49 binary L2 logistic models, one for each marginal number
inclusion.  Their raw sigmoid outputs are not an exact joint law for a draw of
five distinct numbers without replacement.  Bounded-simplex projection imposes
the required marginal contract (49 values in [0, 1] summing to five) without
constructing that joint law.

Each supervised row is chronological: its lag, rolling-frequency, and cumulative
features use only observations before the labelled draw; only then is inclusion
in that draw attached as the label.  Prediction similarly builds one current row
from visible history and uses only frozen scaler parameters and coefficients.
The features, regularization, and solver are pre-registered rather than selected
on evaluation data.  This implementation makes no real-world performance claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
import math
from numbers import Real
from typing import Any
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from .._walk_forward_metrics import NUMBER_COUNT, validate_probabilities
from ..walk_forward import DrawObservation, History, WalkForwardCallbacks
from . import _logistic_features
from ._logistic_features import SupervisedRow
from ._model_provenance import (
    ObservationFingerprint,
    observation_fingerprint as _observation_fingerprint,
    validate_training_provenance as _validate_training_provenance,
    validate_visible_training_overlap as _validate_visible_training_overlap,
)
from .dirichlet_multinomial import project_bounded_simplex

DEFAULT_LAGS = (1, 2, 3)
DEFAULT_WINDOWS = (5, 10, 25, 50)
FEATURE_NAMES = (
    "lag_inclusion_1",
    "lag_inclusion_2",
    "lag_inclusion_3",
    "rolling_frequency_5",
    "rolling_frequency_10",
    "rolling_frequency_25",
    "rolling_frequency_50",
    "cumulative_frequency",
)
PREREGISTERED_METRICS = (
    "log_loss",
    "brier",
    "mean_matches",
    "calibration_error",
    "mean_true_number_rank",
    "regret_vs_uniform",
)
TRIAL_BUDGET = 1
ALLOWED_PARAMETERS = (
    "lags",
    "windows",
    "C",
    "solver",
    "penalty",
    "max_iter",
    "tol",
    "minimum_history",
    "standardize",
    "random_state",
)
PRIMARY_BASELINES = (
    "uniform",
    "cumulative_frequency_alpha=1.0",
    "dirichlet_multinomial_marginal_concentration=1.0_minimum_history=1_"
    "prior=uniform_5_over_49",
)
PROMOTION_RULE = (
    "Promote only after out-of-sample improvement over the primary baselines is "
    "statistically and economically significant on at least three temporal periods, "
    "then confirmed on an untouched final holdout."
)
REJECTION_RULE = (
    "Reject unless out-of-sample improvement over the primary baselines is "
    "statistically and economically significant on at least three temporal periods "
    "and is then confirmed on an untouched final holdout."
)
PREREGISTRATION_NOTE = (
    "Exactly one fixed L2 experiment; no grid, search, evaluation-window tuning, "
    "or alternative feature set is permitted."
)
INSUFFICIENT_DATA_BEHAVIOR = (
    "reject deterministically before fitting when len(history) < minimum_history; "
    "no silent fallback"
)
DEGENERATE_CLASS_POLICY = (
    "reject the complete fit if either class is absent for any number; no fallback"
)
NON_CONVERGENCE_POLICY = (
    "reject the complete fit on any ConvergenceWarning or failed convergence audit"
)


def _finite_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer at least {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _ordered_positive_integers(values: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple of positive integers")
    if not values:
        raise ValueError(f"{name} must not be empty")
    for value in values:
        _positive_integer(value, name=name)
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be strictly ordered")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    return values


def _non_negative_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_history(history: History, *, minimum_size: int) -> History:
    if not isinstance(history, tuple):
        raise TypeError("history must be an immutable tuple")
    if len(history) < minimum_size:
        raise ValueError(f"history must contain at least {minimum_size} observations")
    if any(not isinstance(observation, DrawObservation) for observation in history):
        raise TypeError("history must contain only DrawObservation values")
    if any(
        left.draw_date >= right.draw_date
        for left, right in zip(history, history[1:])
    ):
        raise ValueError("draw dates must be strictly increasing")
    original_indices = [observation.original_index for observation in history]
    if len(original_indices) != len(set(original_indices)):
        raise ValueError("original_index values must be unique")
    return history


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")


def _feature_names(lags: Sequence[int], windows: Sequence[int]) -> tuple[str, ...]:
    return (
        *(f"lag_inclusion_{lag}" for lag in lags),
        *(f"rolling_frequency_{window}" for window in windows),
        "cumulative_frequency",
    )


@dataclass(frozen=True, slots=True)
class LogisticRegressionConfig:
    """Immutable registration of one fixed regularized logistic experiment."""

    lags: tuple[int, ...] = DEFAULT_LAGS
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    C: float = 1.0
    solver: str = "liblinear"
    penalty: str = "l2"
    max_iter: int = 1000
    tol: float = 1e-4
    minimum_history: int = 60
    standardize: bool = True
    random_state: int = 0
    identifier: str = field(init=False)
    family: str = field(
        default="per_number_binary_logistic_regression_l2", init=False
    )
    feature_names: tuple[str, ...] = field(init=False)
    warmup: int = field(init=False)
    class_weight: None = field(default=None, init=False)
    trial_budget: int = field(default=TRIAL_BUDGET, init=False)
    allowed_parameters: tuple[str, ...] = field(
        default=ALLOWED_PARAMETERS, init=False
    )
    primary_baselines: tuple[str, ...] = field(
        default=PRIMARY_BASELINES, init=False
    )
    metrics: tuple[str, ...] = field(default=PREREGISTERED_METRICS, init=False)
    promotion_rule: str = field(default=PROMOTION_RULE, init=False)
    rejection_rule: str = field(default=REJECTION_RULE, init=False)
    preregistration_note: str = field(default=PREREGISTRATION_NOTE, init=False)
    insufficient_data_behavior: str = field(
        default=INSUFFICIENT_DATA_BEHAVIOR, init=False
    )
    degenerate_class_policy: str = field(
        default=DEGENERATE_CLASS_POLICY, init=False
    )
    non_convergence_policy: str = field(
        default=NON_CONVERGENCE_POLICY, init=False
    )

    def __post_init__(self) -> None:
        lags = _ordered_positive_integers(self.lags, name="lags")
        windows = _ordered_positive_integers(self.windows, name="windows")
        c_value = _finite_positive_real(self.C, name="C")
        tolerance = _finite_positive_real(self.tol, name="tol")
        max_iter = _positive_integer(self.max_iter, name="max_iter")
        warmup = max(windows)
        if max(lags) > warmup:
            raise ValueError("lags must not exceed the warmup derived from windows")
        minimum_history = _positive_integer(
            self.minimum_history, name="minimum_history", minimum=warmup + 2
        )
        if self.solver != "liblinear":
            raise ValueError("solver must be the pre-registered value 'liblinear'")
        if self.penalty != "l2":
            raise ValueError("penalty must be the pre-registered value 'l2'")
        if not isinstance(self.standardize, bool):
            raise TypeError("standardize must be a boolean")
        random_state = _non_negative_integer(self.random_state, name="random_state")

        names = _feature_names(lags, windows)
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "C", c_value)
        object.__setattr__(self, "tol", tolerance)
        object.__setattr__(self, "max_iter", max_iter)
        object.__setattr__(self, "minimum_history", minimum_history)
        object.__setattr__(self, "random_state", random_state)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "warmup", warmup)
        object.__setattr__(
            self,
            "identifier",
            "logistic_regression_per_number_l2_"
            f"lags={','.join(map(str, lags))}_"
            f"windows={','.join(map(str, windows))}_"
            f"C={c_value!r}_penalty=l2_solver=liblinear_"
            f"max_iter={max_iter}_tol={tolerance!r}_"
            f"minimum_history={minimum_history}_"
            f"standardize={str(self.standardize).lower()}_"
            f"random_state={random_state}",
        )


def build_supervised_rows(
    history: History, config: LogisticRegressionConfig
) -> tuple[SupervisedRow, ...]:
    """Build target-time rows from prefixes, attaching each label afterwards."""

    if not isinstance(config, LogisticRegressionConfig):
        raise TypeError("config must be a LogisticRegressionConfig")
    checked = _validate_history(history, minimum_size=config.warmup + 1)
    return _logistic_features.build_audit_rows(
        checked,
        lags=config.lags,
        windows=config.windows,
        warmup=config.warmup,
    )


def _float_matrix(
    values: object, *, name: str, rows: int, columns: int, positive: bool = False
) -> tuple[tuple[float, ...], ...]:
    try:
        matrix = tuple(tuple(row) for row in values)  # type: ignore[union-attr]
    except TypeError as exc:
        raise TypeError(f"state {name} must be a matrix") from exc
    if len(matrix) != rows or any(len(row) != columns for row in matrix):
        raise ValueError(f"state {name} has invalid dimensions")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for row in matrix
        for value in row
    ):
        raise TypeError(f"state {name} must contain only real numbers")
    result = tuple(tuple(float(value) for value in row) for row in matrix)
    if not all(math.isfinite(value) for row in result for value in row):
        raise ValueError(f"state {name} must contain only finite values")
    if positive and any(value <= 0.0 for row in result for value in row):
        raise ValueError(f"state {name} must contain only positive values")
    return result


def _float_vector(values: object, *, name: str) -> tuple[float, ...]:
    try:
        vector = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"state {name} must be a sequence") from exc
    if len(vector) != NUMBER_COUNT:
        raise ValueError(f"state {name} must contain exactly 49 values")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in vector
    ):
        raise TypeError(f"state {name} must contain only real numbers")
    result = tuple(float(value) for value in vector)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"state {name} must contain only finite values")
    return result


def _training_transform(
    values: np.ndarray, *, standardize: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a deterministic scaler on training rows and transform those rows."""

    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("training features must be a non-empty matrix")
    if not np.isfinite(values).all():
        raise ValueError("training features must all be finite")
    if not standardize:
        mean = np.zeros(values.shape[1], dtype=float)
        scale = np.ones(values.shape[1], dtype=float)
        return values.copy(), mean, scale
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    # Match StandardScaler's deterministic treatment of constant columns.
    scale[scale == 0.0] = 1.0
    return (values - mean) / scale, mean, scale


@dataclass(frozen=True, slots=True)
class LogisticRegressionState:
    config_identifier: str
    feature_names: tuple[str, ...]
    lags: tuple[int, ...]
    windows: tuple[int, ...]
    scaler_means: tuple[tuple[float, ...], ...]
    scaler_scales: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    classes_observed: tuple[tuple[int, int], ...]
    rows_per_model: tuple[int, ...]
    training_row_count: int
    fitted_through_original_index: int
    fitted_through_date: date
    training_provenance: tuple[ObservationFingerprint, ...]
    n_iter: tuple[int, ...]
    converged: tuple[bool, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_names", tuple(self.feature_names))
        lags = _ordered_positive_integers(tuple(self.lags), name="state lags")
        windows = _ordered_positive_integers(tuple(self.windows), name="state windows")
        if max(lags) > max(windows):
            raise ValueError("state lags must not exceed the warmup derived from windows")
        object.__setattr__(self, "lags", lags)
        object.__setattr__(self, "windows", windows)
        feature_count = len(self.feature_names)
        if feature_count < 1 or len(set(self.feature_names)) != feature_count:
            raise ValueError("state feature_names must be non-empty and unique")
        if self.feature_names != _feature_names(self.lags, self.windows):
            raise ValueError("state feature_names are inconsistent with lags and windows")
        object.__setattr__(
            self,
            "scaler_means",
            _float_matrix(
                self.scaler_means,
                name="scaler_means",
                rows=NUMBER_COUNT,
                columns=feature_count,
            ),
        )
        object.__setattr__(
            self,
            "scaler_scales",
            _float_matrix(
                self.scaler_scales,
                name="scaler_scales",
                rows=NUMBER_COUNT,
                columns=feature_count,
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "coefficients",
            _float_matrix(
                self.coefficients,
                name="coefficients",
                rows=NUMBER_COUNT,
                columns=feature_count,
            ),
        )
        object.__setattr__(self, "intercepts", _float_vector(self.intercepts, name="intercepts"))
        classes = tuple(tuple(values) for values in self.classes_observed)
        if (
            classes != ((0, 1),) * NUMBER_COUNT
            or any(type(value) is not int for pair in classes for value in pair)
        ):
            raise ValueError("state classes_observed must be (0, 1) for every number")
        object.__setattr__(self, "classes_observed", classes)
        rows = tuple(self.rows_per_model)
        if len(rows) != NUMBER_COUNT or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in rows
        ):
            raise ValueError("state rows_per_model must contain 49 positive integers")
        if len(set(rows)) != 1:
            raise ValueError("state rows_per_model must contain one common row count")
        object.__setattr__(self, "rows_per_model", rows)
        if (
            not isinstance(self.training_row_count, int)
            or isinstance(self.training_row_count, bool)
            or self.training_row_count != sum(rows)
        ):
            raise ValueError("state training_row_count must equal rows_per_model total")
        if not isinstance(self.fitted_through_original_index, int) or isinstance(
            self.fitted_through_original_index, bool
        ):
            raise TypeError("state fitted_through_original_index must be an integer")
        if type(self.fitted_through_date) is not date:
            raise TypeError("state fitted_through_date must be a datetime.date")
        object.__setattr__(
            self,
            "training_provenance",
            _validate_training_provenance(
                self.training_provenance,
                expected_size=max(self.windows) + rows[0],
                fitted_through_date=self.fitted_through_date,
                fitted_through_original_index=self.fitted_through_original_index,
            ),
        )
        iterations = tuple(self.n_iter)
        if len(iterations) != NUMBER_COUNT or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in iterations
        ):
            raise ValueError("state n_iter must contain 49 positive integers")
        object.__setattr__(self, "n_iter", iterations)
        convergence = tuple(self.converged)
        if convergence != (True,) * NUMBER_COUNT or any(
            type(value) is not bool for value in convergence
        ):
            raise ValueError("state converged must confirm all 49 fits")
        object.__setattr__(self, "converged", convergence)
        if not isinstance(self.config_identifier, str) or not self.config_identifier:
            raise ValueError("state config_identifier must be non-empty")


@dataclass(frozen=True, slots=True)
class LogisticPrediction:
    raw_probabilities: tuple[float, ...]
    probabilities: tuple[float, ...]
    projection_applied: bool

    def __post_init__(self) -> None:
        raw = _float_vector(self.raw_probabilities, name="raw_probabilities")
        if any(value < 0.0 or value > 1.0 for value in raw):
            raise ValueError("state raw_probabilities must lie in [0, 1]")
        try:
            probabilities = validate_probabilities(self.probabilities)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"probabilities are invalid: {exc}") from exc
        if not isinstance(self.projection_applied, bool):
            raise TypeError("projection_applied must be a boolean")
        if self.projection_applied is not (probabilities != raw):
            raise ValueError("projection_applied is inconsistent")
        object.__setattr__(self, "raw_probabilities", raw)
        object.__setattr__(self, "probabilities", probabilities)


def _validate_state(state: LogisticRegressionState) -> None:
    # Reconstructing exercises all immutable-state invariants even after object-level forgery.
    LogisticRegressionState(
        config_identifier=state.config_identifier,
        feature_names=state.feature_names,
        lags=state.lags,
        windows=state.windows,
        scaler_means=state.scaler_means,
        scaler_scales=state.scaler_scales,
        coefficients=state.coefficients,
        intercepts=state.intercepts,
        classes_observed=state.classes_observed,
        rows_per_model=state.rows_per_model,
        training_row_count=state.training_row_count,
        fitted_through_original_index=state.fitted_through_original_index,
        fitted_through_date=state.fitted_through_date,
        training_provenance=state.training_provenance,
        n_iter=state.n_iter,
        converged=state.converged,
    )


def _sigmoid(score: float) -> float:
    if score >= 0.0:
        inverse = math.exp(-score)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(score)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True, slots=True)
class LogisticRegressionModel:
    config: LogisticRegressionConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, LogisticRegressionConfig):
            raise TypeError("config must be a LogisticRegressionConfig")

    @property
    def callbacks(self) -> WalkForwardCallbacks:
        return WalkForwardCallbacks(fit=self.fit, predict=self.predict)

    def fit(
        self,
        history: History,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> LogisticRegressionState:
        checked = _validate_history(history, minimum_size=self.config.minimum_history)
        _validate_rng(rng)
        feature_matrix, label_matrix = _logistic_features.build_training_matrices(
            checked,
            lags=self.config.lags,
            windows=self.config.windows,
            warmup=self.config.warmup,
        )
        rows_per_model = len(checked) - self.config.warmup
        means = []
        scales = []
        coefficients = []
        intercepts = []
        classes_observed = []
        iterations = []

        for number in range(1, NUMBER_COUNT + 1):
            x_values = feature_matrix[:, number - 1, :]
            labels = label_matrix[:, number - 1]
            classes = tuple(int(value) for value in np.unique(labels))
            if classes != (0, 1):
                raise ValueError(f"degenerate class for number {number}: observed {classes}")
            transformed, mean, scale = _training_transform(
                x_values, standardize=self.config.standardize
            )
            estimator = LogisticRegression(
                penalty=self.config.penalty,
                C=self.config.C,
                class_weight=self.config.class_weight,
                solver=self.config.solver,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
                random_state=self.config.random_state,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                estimator.fit(transformed, labels)
            if any(issubclass(item.category, ConvergenceWarning) for item in caught):
                raise RuntimeError(f"logistic regression did not converge for number {number}")
            iteration_count = int(estimator.n_iter_[0])
            if iteration_count > self.config.max_iter:
                raise RuntimeError(f"logistic regression did not converge for number {number}")
            means.append(tuple(float(value) for value in mean))
            scales.append(tuple(float(value) for value in scale))
            coefficients.append(tuple(float(value) for value in estimator.coef_[0]))
            intercepts.append(float(estimator.intercept_[0]))
            classes_observed.append(classes)
            iterations.append(iteration_count)

        return LogisticRegressionState(
            config_identifier=self.config.identifier,
            feature_names=self.config.feature_names,
            lags=self.config.lags,
            windows=self.config.windows,
            scaler_means=tuple(means),
            scaler_scales=tuple(scales),
            coefficients=tuple(coefficients),
            intercepts=tuple(intercepts),
            classes_observed=tuple(classes_observed),  # type: ignore[arg-type]
            rows_per_model=(rows_per_model,) * NUMBER_COUNT,
            training_row_count=rows_per_model * NUMBER_COUNT,
            fitted_through_original_index=checked[-1].original_index,
            fitted_through_date=checked[-1].draw_date,
            training_provenance=tuple(
                _observation_fingerprint(observation) for observation in checked
            ),
            n_iter=tuple(iterations),
            converged=(True,) * NUMBER_COUNT,
        )

    def predict_details(
        self,
        history: History,
        state: LogisticRegressionState,
        *,
        rng: np.random.Generator,
    ) -> LogisticPrediction:
        checked = _validate_history(history, minimum_size=self.config.minimum_history)
        _validate_rng(rng)
        if not isinstance(state, LogisticRegressionState):
            raise TypeError("state must be a LogisticRegressionState")
        _validate_state(state)
        if state.config_identifier != self.config.identifier:
            raise ValueError("state does not belong to this model configuration")
        _validate_visible_training_overlap(
            checked,
            training_provenance=state.training_provenance,
            fitted_through_date=state.fitted_through_date,
        )

        feature_matrix = _logistic_features.build_prediction_matrix(
            checked,
            lags=self.config.lags,
            windows=self.config.windows,
        )
        raw = []
        for number in range(1, NUMBER_COUNT + 1):
            index = number - 1
            standardized = tuple(
                (value - mean) / scale
                for value, mean, scale in zip(
                    feature_matrix[index],
                    state.scaler_means[index],
                    state.scaler_scales[index],
                    strict=True,
                )
            )
            score = state.intercepts[index] + math.fsum(
                coefficient * value
                for coefficient, value in zip(
                    state.coefficients[index], standardized, strict=True
                )
            )
            if not math.isfinite(score):
                raise ValueError("prediction scores must all be finite")
            raw.append(_sigmoid(score))
        raw_probabilities = tuple(raw)
        probabilities = project_bounded_simplex(raw_probabilities)
        return LogisticPrediction(
            raw_probabilities=raw_probabilities,
            probabilities=probabilities,
            projection_applied=probabilities != raw_probabilities,
        )

    def predict(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        if not isinstance(state, LogisticRegressionState):
            raise TypeError("state must be a LogisticRegressionState")
        return self.predict_details(history, state, rng=rng).probabilities


__all__ = [
    "ALLOWED_PARAMETERS",
    "DEGENERATE_CLASS_POLICY",
    "FEATURE_NAMES",
    "INSUFFICIENT_DATA_BEHAVIOR",
    "LogisticPrediction",
    "LogisticRegressionConfig",
    "LogisticRegressionModel",
    "LogisticRegressionState",
    "NON_CONVERGENCE_POLICY",
    "PREREGISTERED_METRICS",
    "PRIMARY_BASELINES",
    "PREREGISTRATION_NOTE",
    "PROMOTION_RULE",
    "REJECTION_RULE",
    "SupervisedRow",
    "TRIAL_BUDGET",
    "build_supervised_rows",
]
