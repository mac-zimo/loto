"""Leak-resistant chronological splits and walk-forward evaluation.

Every callback receives an immutable history ending strictly before its target.
Probability vectors contain the 49 marginal inclusion probabilities for a 5/49
selection. Their sum must equal five within ``PROBABILITY_SUM_TOLERANCE``. A
predictor may additionally report the separately realized five-number grid.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import Enum
import hashlib
from typing import Any, Protocol

import numpy as np

from ._walk_forward_metrics import (
    DRAW_SIZE,
    NUMBER_COUNT,
    PROBABILITY_SUM_TOLERANCE,
    EvaluationMetrics,
    evaluate_predictions,
    validate_selected_numbers,
    validate_probabilities,
)

History = tuple["DrawObservation", ...]


class FeatureBuilder(Protocol):
    def __call__(
        self,
        history: History,
        *,
        rng: np.random.Generator,
    ) -> Any: ...


class FitCallback(Protocol):
    def __call__(
        self, history: History, feature_data: Any, *, rng: np.random.Generator
    ) -> Any: ...


class CalibrationCallback(Protocol):
    def __call__(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> Any: ...


class PredictCallback(Protocol):
    def __call__(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> Sequence[float] | PredictionResult: ...


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Ex-ante marginal probabilities and an optional realized selection."""

    probabilities: tuple[float, ...]
    selected_numbers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "probabilities", tuple(self.probabilities))
        if self.selected_numbers is not None:
            object.__setattr__(self, "selected_numbers", tuple(self.selected_numbers))


@dataclass(frozen=True, slots=True)
class DrawObservation:
    """One chronologically ordered draw and its source-row identity."""

    draw_date: date
    original_index: int
    numbers: tuple[int, int, int, int, int]

    def __post_init__(self) -> None:
        if type(self.draw_date) is not date:
            raise TypeError("draw_date must be a datetime.date")
        if not isinstance(self.original_index, int) or isinstance(self.original_index, bool):
            raise TypeError("original_index must be an integer")
        numbers = tuple(self.numbers)
        if len(numbers) != DRAW_SIZE:
            raise ValueError("numbers must contain exactly five values")
        if any(
            not isinstance(number, int)
            or isinstance(number, bool)
            or not 1 <= number <= NUMBER_COUNT
            for number in numbers
        ):
            raise ValueError("numbers must be integers between 1 and 49")
        if len(set(numbers)) != DRAW_SIZE:
            raise ValueError("numbers must be unique")
        object.__setattr__(self, "numbers", numbers)


@dataclass(frozen=True, slots=True)
class EvaluationWindow:
    """Half-open positional slices: ``train`` then immediately ``test``.

    ``max_train_size`` is part of the window contract so a multi-draw test
    window can keep every callback history bounded as targets advance.
    """

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    max_train_size: int | None = None

    def __post_init__(self) -> None:
        values = (self.train_start, self.train_end, self.test_start, self.test_end)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise TypeError("window boundaries must be integers")
        if not (0 <= self.train_start < self.train_end <= self.test_start < self.test_end):
            raise ValueError("window must contain chronological, non-overlapping slices")
        if self.train_end != self.test_start:
            raise ValueError("training and evaluation slices must be contiguous")
        if self.max_train_size is not None:
            if (
                not isinstance(self.max_train_size, int)
                or isinstance(self.max_train_size, bool)
                or self.max_train_size < 1
            ):
                raise ValueError("max_train_size must be a positive integer")
            if self.train_end - self.train_start > self.max_train_size:
                raise ValueError("training slice exceeds max_train_size")


class RetrainPolicy(str, Enum):
    EACH_DRAW = "each_draw"
    FIXED_FREQUENCY = "fixed_frequency"
    NONE_DURING_WINDOW = "none_during_window"


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    initial_train_size: int
    retrain_policy: RetrainPolicy = RetrainPolicy.EACH_DRAW
    retrain_frequency: int | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.initial_train_size, int)
            or isinstance(self.initial_train_size, bool)
            or self.initial_train_size < 1
        ):
            raise ValueError("initial_train_size must be a positive integer")
        if not isinstance(self.retrain_policy, RetrainPolicy):
            raise ValueError("retrain_policy must be a RetrainPolicy")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.retrain_policy is RetrainPolicy.FIXED_FREQUENCY:
            if (
                not isinstance(self.retrain_frequency, int)
                or isinstance(self.retrain_frequency, bool)
                or self.retrain_frequency < 1
            ):
                raise ValueError(
                    "retrain_frequency must be a positive integer for fixed frequency"
                )
        elif self.retrain_frequency is not None:
            raise ValueError(
                "retrain_frequency is only valid with the fixed-frequency policy"
            )


@dataclass(frozen=True, slots=True)
class WalkForwardCallbacks:
    """Minimal extension points for later models, features, and calibration."""

    fit: FitCallback
    predict: PredictCallback
    build_features: FeatureBuilder | None = None
    calibrate: CalibrationCallback | None = None

    def __post_init__(self) -> None:
        for name in ("fit", "predict"):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        for name in ("build_features", "calibrate"):
            callback = getattr(self, name)
            if callback is not None and not callable(callback):
                raise TypeError(f"{name} must be callable or None")


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    target_date: date
    target_original_index: int
    target_numbers: tuple[int, int, int, int, int]
    probabilities: tuple[float, ...]
    history_dates: tuple[date, ...]
    history_original_indices: tuple[int, ...]
    fitted_through_date: date
    fitted_through_original_index: int
    window_index: int
    retrained: bool
    selected_numbers: tuple[int, int, int, int, int] | None = None


def _validated_draws(draws: Sequence[DrawObservation]) -> tuple[DrawObservation, ...]:
    observations = tuple(draws)
    if not observations:
        raise ValueError("draws must not be empty")
    if any(not isinstance(draw, DrawObservation) for draw in observations):
        raise TypeError("every draw must be a DrawObservation")
    if any(
        left.draw_date >= right.draw_date
        for left, right in zip(observations, observations[1:])
    ):
        raise ValueError("draw dates must be strictly increasing")
    indices = [draw.original_index for draw in observations]
    if len(indices) != len(set(indices)):
        raise ValueError("original_index values must be unique")
    return observations


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def rolling_origin_windows(
    draws: Sequence[DrawObservation],
    *,
    initial_train_size: int,
    evaluation_size: int = 1,
    step_size: int | None = None,
    max_train_size: int | None = None,
) -> tuple[EvaluationWindow, ...]:
    """Build expanding or bounded rolling-origin windows without test overlap."""

    observations = _validated_draws(draws)
    _positive_integer(initial_train_size, "initial_train_size")
    _positive_integer(evaluation_size, "evaluation_size")
    step = evaluation_size if step_size is None else step_size
    _positive_integer(step, "step_size")
    if step < evaluation_size:
        raise ValueError("step_size must prevent overlap between evaluation windows")
    if max_train_size is not None:
        _positive_integer(max_train_size, "max_train_size")
        if max_train_size < initial_train_size:
            raise ValueError("max_train_size cannot be smaller than initial_train_size")
    if initial_train_size >= len(observations):
        raise ValueError("insufficient draws for the requested initial training window")

    windows = []
    for test_start in range(initial_train_size, len(observations), step):
        test_end = min(test_start + evaluation_size, len(observations))
        train_start = 0 if max_train_size is None else max(0, test_start - max_train_size)
        windows.append(
            EvaluationWindow(
                train_start,
                test_start,
                test_start,
                test_end,
                max_train_size=max_train_size,
            )
        )
    return tuple(windows)


def annual_windows(
    draws: Sequence[DrawObservation],
    *,
    initial_train_size: int,
    evaluation_start_year: int | None = None,
    evaluation_end_year: int | None = None,
    allow_partial_year: bool = False,
) -> tuple[EvaluationWindow, ...]:
    """Build annual evaluation windows with explicit boundary semantics.

    By default, callers must provide the inclusive first and last calendar years
    they have independently established as complete. Completeness cannot be
    inferred from draw dates because the engine does not know the draw calendar.
    With ``allow_partial_year=True`` and no bounds, every eligible observed year
    is accepted explicitly, including boundary segments.
    """

    observations = _validated_draws(draws)
    _positive_integer(initial_train_size, "initial_train_size")
    if initial_train_size >= len(observations):
        raise ValueError("insufficient draws for the requested initial training window")
    if not isinstance(allow_partial_year, bool):
        raise TypeError("allow_partial_year must be a boolean")

    bounds = (evaluation_start_year, evaluation_end_year)
    if not allow_partial_year and any(bound is None for bound in bounds):
        raise ValueError(
            "evaluation_start_year and evaluation_end_year are required unless "
            "allow_partial_year=True"
        )
    if any(bound is not None for bound in bounds):
        if any(
            not isinstance(bound, int) or isinstance(bound, bool)
            for bound in bounds
        ):
            raise ValueError(
                "evaluation_start_year and evaluation_end_year must both be integers"
            )
        assert evaluation_start_year is not None and evaluation_end_year is not None
        if evaluation_start_year > evaluation_end_year:
            raise ValueError("evaluation_start_year cannot exceed evaluation_end_year")
        years = list(range(evaluation_start_year, evaluation_end_year + 1))
    else:
        years = sorted({draw.draw_date.year for draw in observations})

    positions_by_year: dict[int, list[int]] = {}
    for index, draw in enumerate(observations):
        positions_by_year.setdefault(draw.draw_date.year, []).append(index)

    windows = []
    for year in years:
        positions = positions_by_year.get(year)
        if not positions:
            raise ValueError(f"no observations for declared evaluation year {year}")
        test_start, test_end = positions[0], positions[-1] + 1
        if test_start < initial_train_size:
            if evaluation_start_year is not None:
                raise ValueError(
                    f"insufficient training observations before evaluation year {year}"
                )
            continue
        windows.append(EvaluationWindow(0, test_start, test_start, test_end))
    if not windows:
        raise ValueError("insufficient draws for an annual evaluation window")
    return tuple(windows)


def _rng(
    seed: int,
    stage: str,
    history_original_indices: tuple[int, ...],
) -> np.random.Generator:
    """Derive a stream solely from the seed, stage, and callback-visible history."""

    digest = hashlib.sha256()
    components = (
        str(seed),
        stage,
        *(str(index) for index in history_original_indices),
    )
    for component in components:
        encoded = component.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    derived_seed = int.from_bytes(digest.digest()[:16], "big")
    return np.random.default_rng(derived_seed)


def _validate_windows(
    windows: Sequence[EvaluationWindow], draw_count: int, minimum_train_size: int
) -> tuple[EvaluationWindow, ...]:
    result = tuple(windows)
    if not result:
        raise ValueError("at least one evaluation window is required")
    previous_test_end = 0
    for window in result:
        if not isinstance(window, EvaluationWindow):
            raise TypeError("every window must be an EvaluationWindow")
        if window.test_end > draw_count:
            raise ValueError("evaluation window exceeds available draws")
        if window.train_end - window.train_start < minimum_train_size:
            raise ValueError("evaluation window has insufficient training observations")
        if window.test_start < previous_test_end:
            raise ValueError("evaluation windows must not overlap")
        previous_test_end = window.test_end
    return result


def _should_retrain(config: WalkForwardConfig, offset: int) -> bool:
    if offset == 0 or config.retrain_policy is RetrainPolicy.EACH_DRAW:
        return True
    if config.retrain_policy is RetrainPolicy.FIXED_FREQUENCY:
        assert config.retrain_frequency is not None
        return offset % config.retrain_frequency == 0
    return False


def run_walk_forward(
    draws: Sequence[DrawObservation],
    windows: Sequence[EvaluationWindow],
    callbacks: WalkForwardCallbacks,
    config: WalkForwardConfig,
    *,
    retain_history: bool = True,
) -> tuple[PredictionRecord, ...]:
    """Evaluate predictions in order while controlling every callback history."""

    observations = _validated_draws(draws)
    if not isinstance(callbacks, WalkForwardCallbacks):
        raise TypeError("callbacks must be WalkForwardCallbacks")
    if not isinstance(config, WalkForwardConfig):
        raise TypeError("config must be WalkForwardConfig")
    if not isinstance(retain_history, bool):
        raise TypeError("retain_history must be a boolean")
    checked_windows = _validate_windows(
        windows, len(observations), config.initial_train_size
    )
    observation_dates = tuple(draw.draw_date for draw in observations)
    observation_indices = tuple(draw.original_index for draw in observations)

    records = []
    for window_index, window in enumerate(checked_windows):
        state: Any = None
        fitted_through_date: date | None = None
        fitted_through: int | None = None
        for offset, target_position in enumerate(
            range(window.test_start, window.test_end)
        ):
            history_start = window.train_start
            if window.max_train_size is not None:
                history_start = max(
                    history_start, target_position - window.max_train_size
                )
            history: History = observations[history_start:target_position]
            history_original_indices = observation_indices[
                history_start:target_position
            ]
            retrained = _should_retrain(config, offset)
            if retrained:
                fit_features = None
                if callbacks.build_features is not None:
                    fit_features = callbacks.build_features(
                        history,
                        rng=_rng(
                            config.seed,
                            "fit-features",
                            history_original_indices,
                        ),
                    )
                state = callbacks.fit(
                    history,
                    fit_features,
                    rng=_rng(config.seed, "fit", history_original_indices),
                )
                if callbacks.calibrate is not None:
                    state = callbacks.calibrate(
                        history,
                        state,
                        fit_features,
                        rng=_rng(config.seed, "calibrate", history_original_indices),
                    )
                fitted_through_date = history[-1].draw_date
                fitted_through = history[-1].original_index

            prediction_features = None
            if callbacks.build_features is not None:
                prediction_features = callbacks.build_features(
                    history,
                    rng=_rng(
                        config.seed,
                        "predict-features",
                        history_original_indices,
                    ),
                )
            prediction = callbacks.predict(
                history,
                state,
                prediction_features,
                rng=_rng(config.seed, "predict", history_original_indices),
            )
            if isinstance(prediction, PredictionResult):
                probabilities = validate_probabilities(prediction.probabilities)
                selected_numbers = (
                    None
                    if prediction.selected_numbers is None
                    else validate_selected_numbers(prediction.selected_numbers)
                )
            else:
                probabilities = validate_probabilities(prediction)
                selected_numbers = None
            assert fitted_through_date is not None and fitted_through is not None
            target = observations[target_position]
            records.append(
                PredictionRecord(
                    target_date=target.draw_date,
                    target_original_index=target.original_index,
                    target_numbers=target.numbers,
                    probabilities=probabilities,
                    selected_numbers=selected_numbers,
                    history_dates=(
                        observation_dates[history_start:target_position]
                        if retain_history
                        else ()
                    ),
                    history_original_indices=(
                        history_original_indices if retain_history else ()
                    ),
                    fitted_through_date=fitted_through_date,
                    fitted_through_original_index=fitted_through,
                    window_index=window_index,
                    retrained=retrained,
                )
            )
    return tuple(records)


__all__ = [
    "PROBABILITY_SUM_TOLERANCE",
    "CalibrationCallback",
    "DrawObservation",
    "EvaluationMetrics",
    "EvaluationWindow",
    "FeatureBuilder",
    "FitCallback",
    "PredictCallback",
    "PredictionRecord",
    "PredictionResult",
    "RetrainPolicy",
    "WalkForwardCallbacks",
    "WalkForwardConfig",
    "annual_windows",
    "evaluate_predictions",
    "rolling_origin_windows",
    "run_walk_forward",
    "validate_probabilities",
]
