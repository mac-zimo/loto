"""Vectorized, leak-free feature construction for the logistic model."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from .._walk_forward_metrics import NUMBER_COUNT
from ..walk_forward import History


@dataclass(frozen=True, slots=True)
class SupervisedRow:
    """Auditable representation of one number's features at one target time."""

    number: int
    target_position: int
    target_date: date
    target_original_index: int
    features: tuple[float, ...]
    label: int


def _inclusion_matrix(history: History) -> np.ndarray:
    """Build the draw-by-number inclusion matrix exactly once."""

    inclusions = np.zeros((len(history), NUMBER_COUNT), dtype=float)
    for position, observation in enumerate(history):
        inclusions[position, np.asarray(observation.numbers, dtype=int) - 1] = 1.0
    return inclusions


def _feature_tensor(
    inclusions: np.ndarray,
    target_positions: np.ndarray,
    *,
    lags: Sequence[int],
    windows: Sequence[int],
) -> np.ndarray:
    """Return target-by-number-by-feature values using prefix sums only."""

    prefix_sums = np.vstack(
        (np.zeros((1, NUMBER_COUNT), dtype=float), inclusions.cumsum(axis=0))
    )
    lag_values = np.stack(
        tuple(inclusions[target_positions - lag] for lag in lags), axis=2
    )
    rolling_values = np.stack(
        tuple(
            (prefix_sums[target_positions] - prefix_sums[target_positions - window])
            / window
            for window in windows
        ),
        axis=2,
    )
    cumulative_values = (
        prefix_sums[target_positions] / target_positions[:, np.newaxis]
    )[:, :, np.newaxis]
    result = np.concatenate((lag_values, rolling_values, cumulative_values), axis=2)
    if not np.isfinite(result).all():
        raise ValueError("constructed features are invalid")
    return result


def build_training_matrices(
    history: History,
    *,
    lags: Sequence[int],
    windows: Sequence[int],
    warmup: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build all model matrices in O(n * 49 * feature_count)."""

    inclusions = _inclusion_matrix(history)
    target_positions = np.arange(warmup, len(history), dtype=int)
    features = _feature_tensor(
        inclusions, target_positions, lags=lags, windows=windows
    )
    return features, inclusions[target_positions].astype(int, copy=False)


def build_prediction_matrix(
    history: History,
    *,
    lags: Sequence[int],
    windows: Sequence[int],
) -> np.ndarray:
    """Build the 49 current feature rows in one vectorized operation."""

    inclusions = _inclusion_matrix(history)
    target_positions = np.asarray((len(history),), dtype=int)
    return _feature_tensor(
        inclusions, target_positions, lags=lags, windows=windows
    )[0]


def build_audit_rows(
    history: History,
    *,
    lags: Sequence[int],
    windows: Sequence[int],
    warmup: int,
) -> tuple[SupervisedRow, ...]:
    """Materialize the public audit rows from the same vectorized matrices."""

    features, labels = build_training_matrices(
        history, lags=lags, windows=windows, warmup=warmup
    )
    rows = []
    for row_index, target_position in enumerate(range(warmup, len(history))):
        target = history[target_position]
        for number_index in range(NUMBER_COUNT):
            rows.append(
                SupervisedRow(
                    number=number_index + 1,
                    target_position=target_position,
                    target_date=target.draw_date,
                    target_original_index=target.original_index,
                    features=tuple(
                        float(value) for value in features[row_index, number_index]
                    ),
                    label=int(labels[row_index, number_index]),
                )
            )
    return tuple(rows)


__all__ = [
    "SupervisedRow",
    "build_audit_rows",
    "build_prediction_matrix",
    "build_training_matrices",
]
