"""Mandatory leak-free baselines for walk-forward evaluation.

Frequency baselines use symmetric Bernoulli pseudo-count smoothing. For a
history of ``n`` draws and smoothing strength ``alpha``, number ``i`` receives
``(count_i + alpha * (5 / 49)) / (n + alpha)``. ``alpha`` is fixed when the
baseline is created and is never fitted on an evaluation window. Because each
draw contains five numbers, the 49 resulting marginals sum to exactly five up
to floating-point rounding.

The constrained-random baseline deliberately distinguishes ex-ante prediction
from realization: its callback returns uniform 5/49 marginals together with one
grid of five distinct numbers drawn from the generator injected by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

import numpy as np

from ._walk_forward_metrics import (
    DRAW_SIZE,
    NUMBER_COUNT,
    UNIFORM_PROBABILITIES,
    UNIFORM_PROBABILITY,
    validate_probabilities,
)
from .walk_forward import (
    DrawObservation,
    History,
    PredictionResult,
    WalkForwardCallbacks,
)


def _fit_without_training(
    history: History, feature_data: Any, *, rng: np.random.Generator
) -> None:
    """Satisfy the engine fit contract without learning baseline parameters."""

    return None


def _validate_history(history: History, *, minimum_size: int) -> History:
    if not isinstance(history, tuple):
        raise TypeError("history must be an immutable tuple")
    if len(history) < minimum_size:
        required = (
            "one observation"
            if minimum_size == 1
            else f"{minimum_size} observations"
        )
        raise ValueError(f"history must contain at least {required}")
    if any(not isinstance(observation, DrawObservation) for observation in history):
        raise TypeError("history must contain only DrawObservation values")
    return history


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a finite positive real number")
    value = float(alpha)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("alpha must be a finite positive real number")
    return value


def _smoothed_frequencies(history: History, alpha: float) -> tuple[float, ...]:
    counts = [0] * NUMBER_COUNT
    for observation in history:
        for number in observation.numbers:
            counts[number - 1] += 1
    denominator = len(history) + alpha
    probabilities = tuple(
        (count + alpha * UNIFORM_PROBABILITY) / denominator for count in counts
    )
    return validate_probabilities(probabilities)


def _draw_selected_numbers(
    rng: np.random.Generator,
) -> tuple[int, int, int, int, int]:
    selected = tuple(
        sorted(
            int(value)
            for value in rng.choice(NUMBER_COUNT, DRAW_SIZE, replace=False) + 1
        )
    )
    return selected  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class UniformBaseline:
    """Constant marginal forecast; it performs no fitting or random draw."""

    identifier: str = "uniform"

    @property
    def callbacks(self) -> WalkForwardCallbacks:
        return WalkForwardCallbacks(fit=_fit_without_training, predict=self.predict)

    def predict(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        _validate_history(history, minimum_size=0)
        _validate_rng(rng)
        return validate_probabilities(UNIFORM_PROBABILITIES)


@dataclass(frozen=True, slots=True)
class CumulativeFrequencyBaseline:
    """Smoothed inclusion frequencies over every callback-visible draw."""

    alpha: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))

    @property
    def identifier(self) -> str:
        return f"cumulative_frequency_alpha={self.alpha:g}"

    @property
    def callbacks(self) -> WalkForwardCallbacks:
        return WalkForwardCallbacks(fit=_fit_without_training, predict=self.predict)

    def predict(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        checked_history = _validate_history(history, minimum_size=1)
        _validate_rng(rng)
        return _smoothed_frequencies(checked_history, self.alpha)


@dataclass(frozen=True, slots=True)
class RollingFrequencyBaseline:
    """Smoothed inclusion frequencies over exactly the last ``window_size`` draws.

    Histories shorter than the configured window are rejected rather than
    silently changing the effective baseline.
    """

    window_size: int
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.window_size, int)
            or isinstance(self.window_size, (bool, np.bool_))
            or self.window_size <= 0
        ):
            raise ValueError("window_size must be a positive integer")
        object.__setattr__(self, "alpha", _validate_alpha(self.alpha))

    @property
    def identifier(self) -> str:
        return f"rolling_frequency_window={self.window_size}_alpha={self.alpha:g}"

    @property
    def callbacks(self) -> WalkForwardCallbacks:
        return WalkForwardCallbacks(fit=_fit_without_training, predict=self.predict)

    def predict(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        checked_history = _validate_history(history, minimum_size=self.window_size)
        _validate_rng(rng)
        return _smoothed_frequencies(checked_history[-self.window_size :], self.alpha)


@dataclass(frozen=True, slots=True)
class ConstrainedRandomGrid:
    """One realized grid and its uniform ex-ante marginal forecast."""

    selected_numbers: tuple[int, int, int, int, int]
    marginal_probabilities: tuple[float, ...] = UNIFORM_PROBABILITIES

    def __post_init__(self) -> None:
        numbers = tuple(self.selected_numbers)
        if len(numbers) != DRAW_SIZE:
            raise ValueError("selected_numbers must contain exactly five values")
        if any(
            not isinstance(number, int)
            or isinstance(number, (bool, np.bool_))
            or not 1 <= number <= NUMBER_COUNT
            for number in numbers
        ):
            raise ValueError("selected_numbers must be integers between 1 and 49")
        if len(set(numbers)) != DRAW_SIZE:
            raise ValueError("selected_numbers must be unique")
        marginals = validate_probabilities(self.marginal_probabilities)
        if marginals != UNIFORM_PROBABILITIES:
            raise ValueError("marginal_probabilities must be the uniform ex-ante forecast")
        object.__setattr__(self, "selected_numbers", numbers)
        object.__setattr__(self, "marginal_probabilities", marginals)


@dataclass(frozen=True, slots=True)
class ConstrainedRandomBaseline:
    """Uniform ex-ante forecast plus a separately auditable random grid."""

    identifier: str = "constrained_random"

    @property
    def callbacks(self) -> WalkForwardCallbacks:
        return WalkForwardCallbacks(fit=_fit_without_training, predict=self.predict)

    def predict(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> PredictionResult:
        _validate_history(history, minimum_size=0)
        _validate_rng(rng)
        return PredictionResult(
            probabilities=validate_probabilities(UNIFORM_PROBABILITIES),
            selected_numbers=_draw_selected_numbers(rng),
        )

    def materialize(
        self, history: History, *, rng: np.random.Generator
    ) -> ConstrainedRandomGrid:
        _validate_history(history, minimum_size=0)
        _validate_rng(rng)
        return ConstrainedRandomGrid(selected_numbers=_draw_selected_numbers(rng))


def uniform_baseline() -> UniformBaseline:
    return UniformBaseline()


def cumulative_frequency_baseline(
    *, alpha: float = 1.0
) -> CumulativeFrequencyBaseline:
    return CumulativeFrequencyBaseline(alpha=alpha)


def rolling_frequency_baseline(
    *, window_size: int, alpha: float = 1.0
) -> RollingFrequencyBaseline:
    return RollingFrequencyBaseline(window_size=window_size, alpha=alpha)


def constrained_random_baseline() -> ConstrainedRandomBaseline:
    return ConstrainedRandomBaseline()


__all__ = [
    "ConstrainedRandomBaseline",
    "ConstrainedRandomGrid",
    "CumulativeFrequencyBaseline",
    "RollingFrequencyBaseline",
    "UniformBaseline",
    "constrained_random_baseline",
    "cumulative_frequency_baseline",
    "rolling_frequency_baseline",
    "uniform_baseline",
]
