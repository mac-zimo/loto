"""Biased-draw simulation and deterministic marginal power estimation."""

from __future__ import annotations

from collections.abc import Sequence
from math import sqrt
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

from loto.audit.power_models import (
    CHANCE_BASELINE,
    CHANCE_MAX_EFFECT,
    DEFAULT_ALPHA,
    DEFAULT_CONFIDENCE,
    DEFAULT_EFFECT_SIZES,
    DEFAULT_EQUIVALENCE_MARGIN,
    DEFAULT_N_DRAWS,
    DEFAULT_NULL_REPETITIONS,
    DEFAULT_POWER_REPETITIONS,
    DEFAULT_POWER_TARGET,
    DEFAULT_SEED,
    MAIN_BASELINE,
    MAIN_MAX_EFFECT,
    SCENARIOS,
    PowerCurveReport,
    PowerPoint,
    Scenario,
)
from loto.audit.power_statistics import (
    _critical_value,
    _monte_carlo_p_value,
    marginal_statistics,
    minimum_null_repetitions,
)
from loto.audit.power_validation import effect, positive_integer, unit_interval
from loto.audit.uniformity import _require_generator


def _sample_without_target(
    n_draws: int, sizes: NDArray[np.int64], *, rng: np.random.Generator
) -> NDArray[np.int64]:
    """Sample rows from 2..49 without replacement, with row-specific sizes."""
    result = np.zeros((n_draws, 5), dtype=np.int64)
    selected = np.zeros((n_draws, 48), dtype=bool)
    for column in range(5):
        pending = np.flatnonzero(sizes > column)
        while pending.size:
            candidates = rng.integers(0, 48, size=pending.size)
            accepted = ~selected[pending, candidates]
            rows = pending[accepted]
            values = candidates[accepted]
            result[rows, column] = values + 2
            selected[rows, values] = True
            pending = pending[~accepted]
    return result


def simulate_biased_draws(
    n_draws: int,
    *,
    main_effect: float = 0.0,
    chance_effect: float = 0.0,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Generate exact 5/49 and 1/10 draws with known marginal biases.

    ``main_effect`` and ``chance_effect`` are probability-point increases in
    the marginal probability of number 1. Remaining values stay exchangeable.
    Main-number sampling is always without replacement. At zero effects this
    conditional construction is exactly the fair game distribution.
    """
    _require_generator(rng)
    count = positive_integer(n_draws, "n_draws")
    main_delta = effect(main_effect, "main_effect", maximum=MAIN_MAX_EFFECT)
    chance_delta = effect(chance_effect, "chance_effect", maximum=CHANCE_MAX_EFFECT)

    includes_target = rng.random(count) < MAIN_BASELINE + main_delta
    sizes = np.where(includes_target, 4, 5).astype(np.int64)
    main = _sample_without_target(count, sizes, rng=rng)
    main[includes_target, 4] = 1
    main.sort(axis=1)

    target_chance = rng.random(count) < CHANCE_BASELINE + chance_delta
    chance = rng.integers(2, 11, size=count, dtype=np.int64)
    chance[target_chance] = 1
    return main, chance


def _wilson_interval(
    detections: int, repetitions: int, confidence: float
) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    estimate = detections / repetitions
    denominator = 1 + z * z / repetitions
    center = (estimate + z * z / (2 * repetitions)) / denominator
    radius = (
        z
        * sqrt(
            estimate * (1 - estimate) / repetitions
            + z * z / (4 * repetitions * repetitions)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _validate_curve_options(
    n_draws: object,
    effect_sizes: Sequence[float],
    null_repetitions: object,
    power_repetitions: object,
    seed: object,
    alpha: object,
    confidence: object,
    power_target: object,
) -> tuple[int, tuple[float, ...], int, int, int, float, float, float]:
    count = positive_integer(n_draws, "n_draws", minimum=2)
    null_count = positive_integer(null_repetitions, "null_repetitions")
    power_count = positive_integer(power_repetitions, "power_repetitions")
    normalized_seed = positive_integer(seed, "seed", minimum=0)
    normalized_alpha = unit_interval(alpha, "alpha")
    normalized_confidence = unit_interval(confidence, "confidence")
    normalized_target = unit_interval(power_target, "power_target")
    if isinstance(effect_sizes, (str, bytes)):
        raise TypeError("effect_sizes must be a sequence of real numbers")
    try:
        effects = tuple(
            effect(value, "effect_sizes", maximum=MAIN_MAX_EFFECT)
            for value in effect_sizes
        )
    except TypeError as error:
        if "not iterable" in str(error):
            raise TypeError("effect_sizes must be a sequence of real numbers") from error
        raise
    if not effects:
        raise ValueError("effect_sizes must not be empty")
    if any(first >= second for first, second in zip(effects, effects[1:])):
        raise ValueError("effect_sizes must be strictly increasing and unique")
    minimum = minimum_null_repetitions(
        normalized_alpha, number_of_tests=len(SCENARIOS)
    )
    if null_count < minimum:
        raise ValueError(
            "null_repetitions must be at least "
            f"{minimum} so the finite Monte-Carlo p-value can attain "
            "the Bonferroni-adjusted alpha"
        )
    return (
        count,
        effects,
        null_count,
        power_count,
        normalized_seed,
        normalized_alpha,
        normalized_confidence,
        normalized_target,
    )


def estimate_power_curves(
    *,
    n_draws: int = DEFAULT_N_DRAWS,
    effect_sizes: Sequence[float] = DEFAULT_EFFECT_SIZES,
    null_repetitions: int = DEFAULT_NULL_REPETITIONS,
    power_repetitions: int = DEFAULT_POWER_REPETITIONS,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    confidence: float = DEFAULT_CONFIDENCE,
    power_target: float = DEFAULT_POWER_TARGET,
) -> PowerCurveReport:
    """Calibrate under the null, then estimate power on independent repeats.

    The alpha level, seed, effect grid, target power and repetition counts are
    inputs recorded in the returned report rather than selected after results
    are observed. Production defaults target the 2,811-draw history; callers
    may use smaller repetition counts for tests.
    """
    (
        count,
        effects,
        null_count,
        power_count,
        normalized_seed,
        normalized_alpha,
        normalized_confidence,
        normalized_target,
    ) = _validate_curve_options(
        n_draws,
        effect_sizes,
        null_repetitions,
        power_repetitions,
        seed,
        alpha,
        confidence,
        power_target,
    )
    streams = np.random.SeedSequence(normalized_seed).spawn(
        1 + len(SCENARIOS) * len(effects)
    )
    calibration_rng = np.random.default_rng(streams[0])
    null_statistics: dict[Scenario, NDArray[np.float64]] = {
        scenario: np.empty(null_count) for scenario in SCENARIOS
    }
    for index in range(null_count):
        main, chance = simulate_biased_draws(count, rng=calibration_rng)
        statistics = marginal_statistics(main, chance)
        for scenario in SCENARIOS:
            null_statistics[scenario][index] = statistics[scenario]

    per_test_alpha = normalized_alpha / len(SCENARIOS)
    critical_values: dict[Scenario, float] = {
        scenario: _critical_value(values, per_test_alpha)
        for scenario, values in null_statistics.items()
    }
    curves: dict[Scenario, tuple[PowerPoint, ...]] = {}
    stream_index = 1
    for scenario in SCENARIOS:
        points = []
        for effect_size in effects:
            rng = np.random.default_rng(streams[stream_index])
            stream_index += 1
            detections = 0
            for _ in range(power_count):
                main, chance = simulate_biased_draws(
                    count,
                    main_effect=effect_size if scenario == "main_marginal" else 0.0,
                    chance_effect=effect_size if scenario == "chance_marginal" else 0.0,
                    rng=rng,
                )
                observed = marginal_statistics(main, chance)[scenario]
                detections += (
                    _monte_carlo_p_value(observed, null_statistics[scenario])
                    <= per_test_alpha
                )
            estimated_power = detections / power_count
            low, high = _wilson_interval(
                detections, power_count, normalized_confidence
            )
            points.append(
                PowerPoint(
                    effect_size=effect_size,
                    detections=int(detections),
                    repetitions=power_count,
                    power=estimated_power,
                    interval_low=low,
                    interval_high=high,
                    detectability=(
                        "detectable"
                        if estimated_power >= normalized_target
                        else "insufficient_power"
                    ),
                )
            )
        curves[scenario] = tuple(points)

    return PowerCurveReport(
        n_draws=count,
        seed=normalized_seed,
        alpha=normalized_alpha,
        per_test_alpha=per_test_alpha,
        confidence=normalized_confidence,
        power_target=normalized_target,
        null_repetitions=null_count,
        power_repetitions=power_count,
        multiplicity_correction="Bonferroni across two marginal tests",
        statistic="maximum absolute marginal inclusion-rate deviation",
        bias_definition=(
            "probability-point increase for inclusion of number 1; remaining "
            "values exchangeable; main draws remain 5-of-49 without replacement"
        ),
        monte_carlo_method=(
            "finite upper-tail p=(1 + count(null >= observed))/(B + 1)"
        ),
        equivalence_margin=DEFAULT_EQUIVALENCE_MARGIN,
        equivalence_method=(
            "two-sided exact Clopper-Pearson intervals for all 49 main and 10 "
            "Chance Bernoulli marginals; Bonferroni within each test and across "
            "the two tests; bias_absent only when the simultaneous maximum-"
            "deviation upper bound is strictly below the margin"
        ),
        critical_values=critical_values,
        curves=curves,
    )
