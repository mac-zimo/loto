"""Prediction validation and metrics for walk-forward evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Real
from typing import Protocol

import numpy as np


NUMBER_COUNT = 49
DRAW_SIZE = 5
UNIFORM_PROBABILITY = DRAW_SIZE / NUMBER_COUNT
UNIFORM_PROBABILITIES = (UNIFORM_PROBABILITY,) * NUMBER_COUNT
# Float32 normalization can accumulate slightly over 1e-6 across 49 marginals;
# 1e-5 leaves a rounding margin while still rejecting material sum errors.
PROBABILITY_SUM_TOLERANCE = 1e-5


class PredictionLike(Protocol):
    probabilities: Sequence[float]
    target_numbers: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    log_loss: float
    brier: float
    mean_matches: float
    calibration_error: float
    mean_true_number_rank: float
    regret_vs_uniform: float


def _positive_integer(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_probabilities(values: Sequence[float]) -> tuple[float, ...]:
    """Validate 49 finite marginals in [0, 1] whose sum is five (atol 1e-5)."""

    try:
        raw_probabilities = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be numeric") from exc
    if raw_probabilities.shape != (NUMBER_COUNT,):
        raise ValueError("probabilities must contain exactly 49 values")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw_probabilities
    ):
        raise ValueError("probabilities must contain only real numbers")
    probabilities = np.asarray(raw_probabilities, dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must all be finite")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    if not math.isclose(
        float(probabilities.sum()),
        float(DRAW_SIZE),
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError("probabilities must sum to 5")
    return tuple(float(value) for value in probabilities)


def _validate_numbers(
    values: object, *, name: str
) -> tuple[int, int, int, int, int]:
    try:
        numbers = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{name} must contain exactly five integers") from exc
    if len(numbers) != DRAW_SIZE:
        raise ValueError(f"{name} must contain exactly five values")
    if any(
        not isinstance(number, int) or isinstance(number, bool)
        for number in numbers
    ):
        raise TypeError(f"{name} must contain only integers")
    if any(not 1 <= number <= NUMBER_COUNT for number in numbers):
        raise ValueError(f"{name} must be between 1 and 49")
    if len(set(numbers)) != DRAW_SIZE:
        raise ValueError(f"{name} must be unique")
    return numbers  # type: ignore[return-value]


def _validate_target_numbers(values: object) -> tuple[int, int, int, int, int]:
    return _validate_numbers(values, name="target_numbers")


def validate_selected_numbers(values: object) -> tuple[int, int, int, int, int]:
    """Validate an explicitly realized five-number grid."""

    return _validate_numbers(values, name="selected_numbers")


def _validated_predictions(
    records: Sequence[PredictionLike],
) -> tuple[
    tuple[
        tuple[float, ...],
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int] | None,
    ],
    ...,
]:
    validated = []
    for record in records:
        if not all(
            hasattr(record, attribute)
            for attribute in ("probabilities", "target_numbers")
        ):
            raise TypeError(
                "every prediction record must provide probabilities and target_numbers"
            )
        target_numbers = _validate_target_numbers(record.target_numbers)
        probabilities = validate_probabilities(record.probabilities)
        selected = getattr(record, "selected_numbers", None)
        selected_numbers = (
            None if selected is None else validate_selected_numbers(selected)
        )
        validated.append((probabilities, target_numbers, selected_numbers))
    return tuple(validated)


def _binary_log_loss(probability: float, included: bool) -> float:
    if included:
        return math.inf if probability == 0.0 else -math.log(probability)
    return math.inf if probability == 1.0 else -math.log1p(-probability)


def _average_descending_ranks(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Return one-based average ranks, assigning equal values the same rank."""

    values = np.asarray(probabilities)
    return tuple(
        1.0
        + float(np.count_nonzero(values > probability))
        + (float(np.count_nonzero(values == probability)) - 1.0) / 2.0
        for probability in values
    )


def evaluate_predictions(
    records: Sequence[PredictionLike], *, calibration_bins: int = 10
) -> EvaluationMetrics:
    """Aggregate out-of-sample metrics; regret is excess log-loss.

    Probabilistic metrics always use ex-ante marginals. ``mean_matches`` uses an
    explicitly realized grid when supplied, and otherwise the probability
    vector's deterministic top-five selection.
    """

    predictions = tuple(records)
    if not predictions:
        raise ValueError("at least one prediction record is required")
    _positive_integer(calibration_bins, "calibration_bins")

    validated_predictions = _validated_predictions(predictions)

    all_probabilities = []
    all_outcomes = []
    losses = []
    squared_errors = []
    matches = []
    true_ranks = []
    for probabilities, target_numbers, selected_numbers in validated_predictions:
        truth = set(target_numbers)
        outcomes = tuple(number in truth for number in range(1, NUMBER_COUNT + 1))
        losses.extend(
            _binary_log_loss(probability, included)
            for probability, included in zip(probabilities, outcomes)
        )
        squared_errors.extend(
            (probability - float(included)) ** 2
            for probability, included in zip(probabilities, outcomes)
        )
        ranked_numbers = sorted(
            range(1, NUMBER_COUNT + 1),
            key=lambda number: (-probabilities[number - 1], number),
        )
        evaluated_selection = (
            ranked_numbers[:DRAW_SIZE]
            if selected_numbers is None
            else selected_numbers
        )
        matches.append(len(set(evaluated_selection) & truth))
        average_ranks = _average_descending_ranks(probabilities)
        true_ranks.extend(average_ranks[number - 1] for number in truth)
        all_probabilities.extend(probabilities)
        all_outcomes.extend(float(included) for included in outcomes)

    probabilities_array = np.asarray(all_probabilities)
    outcomes_array = np.asarray(all_outcomes)
    bin_indices = np.minimum(
        (probabilities_array * calibration_bins).astype(int), calibration_bins - 1
    )
    calibration_error = 0.0
    for bin_index in range(calibration_bins):
        mask = bin_indices == bin_index
        if mask.any():
            calibration_error += float(mask.mean()) * abs(
                float(probabilities_array[mask].mean())
                - float(outcomes_array[mask].mean())
            )

    log_loss = float(np.mean(losses))
    uniform_loss = -(
        DRAW_SIZE * math.log(UNIFORM_PROBABILITY)
        + (NUMBER_COUNT - DRAW_SIZE) * math.log1p(-UNIFORM_PROBABILITY)
    ) / NUMBER_COUNT
    return EvaluationMetrics(
        log_loss=log_loss,
        brier=float(np.mean(squared_errors)),
        mean_matches=float(np.mean(matches)),
        calibration_error=calibration_error,
        mean_true_number_rank=float(np.mean(true_ranks)),
        regret_vs_uniform=log_loss - uniform_loss,
    )
