"""Finite Monte-Carlo and simultaneous binomial confidence procedures."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta

from loto.audit.power_models import (
    CHANCE_BASELINE,
    MAIN_BASELINE,
    Scenario,
)


def marginal_statistics(
    main: NDArray[np.int64], chance: NDArray[np.int64]
) -> dict[Scenario, float]:
    """Return maximum absolute marginal deviations for both games."""
    main_counts = np.bincount(main.ravel(), minlength=50)[1:]
    chance_counts = np.bincount(chance, minlength=11)[1:]
    return {
        "main_marginal": float(
            np.max(np.abs(main_counts / len(main) - MAIN_BASELINE))
        ),
        "chance_marginal": float(
            np.max(np.abs(chance_counts / len(chance) - CHANCE_BASELINE))
        ),
    }


def _probability(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0 < normalized < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return normalized


def minimum_null_repetitions(alpha: float, *, number_of_tests: int) -> int:
    """Minimum null simulations able to attain Bonferroni-adjusted ``alpha``.

    A finite Monte-Carlo p-value with the required +1 correction cannot be
    smaller than ``1 / (B + 1)`` for ``B`` null repetitions.
    """
    normalized_alpha = _probability(alpha, "alpha")
    if (
        isinstance(number_of_tests, (bool, np.bool_))
        or not isinstance(number_of_tests, Integral)
        or number_of_tests < 1
    ):
        raise ValueError("number_of_tests must be a positive integer")
    per_test_alpha = normalized_alpha / int(number_of_tests)
    if per_test_alpha == 0.0:
        raise ValueError("alpha / number_of_tests must be representable above zero")

    numerator, denominator = per_test_alpha.as_integer_ratio()
    repetitions = max(1, (denominator + numerator - 1) // numerator - 1)
    while 1 / (repetitions + 1) > per_test_alpha:
        repetitions += 1
    while repetitions > 1 and 1 / repetitions <= per_test_alpha:
        repetitions -= 1
    return repetitions


def _monte_carlo_p_value(
    observed: float, null_statistics: NDArray[np.float64]
) -> float:
    """Return the finite-sample valid upper-tail Monte-Carlo p-value."""
    return float(
        (1 + np.count_nonzero(null_statistics >= observed))
        / (len(null_statistics) + 1)
    )


def _critical_value(
    null_statistics: NDArray[np.float64], per_test_alpha: float
) -> float:
    """Threshold equivalent to the +1 p-value rule, including ties safely."""
    denominator = len(null_statistics) + 1
    lower = -1
    upper = len(null_statistics)
    while upper - lower > 1:
        candidate = (lower + upper) // 2
        if (candidate + 1) / denominator <= per_test_alpha:
            lower = candidate
        else:
            upper = candidate
    if lower < 0:
        return float("inf")
    descending = np.sort(null_statistics)[::-1]
    return float(descending[lower])


def simultaneous_max_deviation_upper_bound(
    counts: ArrayLike,
    *,
    n_draws: int,
    expected_probability: float,
    alpha: float,
) -> float:
    """Upper confidence bound for the largest absolute marginal deviation.

    Each marginal count is binomial because its per-draw inclusion indicator is
    Bernoulli. Two-sided Clopper-Pearson intervals use ``alpha / (2 * K)`` in
    each tail for ``K`` marginals. The union bound therefore gives simultaneous
    coverage of at least ``1 - alpha`` without requiring independence between
    indicators from the same draw (which are dependent for 5-of-49 sampling).
    """
    if (
        isinstance(n_draws, (bool, np.bool_))
        or not isinstance(n_draws, Integral)
        or n_draws < 1
    ):
        raise ValueError("n_draws must be a positive integer")
    count = int(n_draws)
    expected = _probability(expected_probability, "expected_probability")
    family_alpha = _probability(alpha, "alpha")
    values = np.asarray(counts)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("counts must be a non-empty one-dimensional array")
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError("counts must contain integers")
    if np.any(values < 0) or np.any(values > count):
        raise ValueError("counts must be between zero and n_draws")

    tail_alpha = family_alpha / (2 * values.size)
    lower = np.where(
        values == 0,
        0.0,
        beta.ppf(tail_alpha, values, count - values + 1),
    )
    upper = np.where(
        values == count,
        1.0,
        beta.ppf(1 - tail_alpha, values + 1, count - values),
    )
    return float(np.max(np.maximum(expected - lower, upper - expected)))
