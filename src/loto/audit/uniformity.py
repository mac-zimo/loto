"""Monte-Carlo conformity audit for the current 5-from-49 Loto game."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from math import comb

import numpy as np
from numpy.typing import ArrayLike, NDArray

_POOL = 49
_DRAW_SIZE = 5
_CHANCE_POOL = 10
_TOTAL_COMBINATIONS = comb(_POOL, _DRAW_SIZE)


@dataclass(frozen=True)
class TestResult:
    """One test result, retaining raw and Holm FWER-adjusted p-values."""

    statistic: float
    # Raw Monte-Carlo p-value; never replace it with the adjusted value.
    p_value: float
    adjusted_p_value: float
    effect_size: float
    effect_name: str
    null_interval: tuple[float, float]


@dataclass(frozen=True)
class UniformityReport:
    """Global audit; its seven p-values form one Holm-corrected family."""

    n_draws: int
    simulations: int
    confidence: float
    p_value_adjustment: str
    results: Mapping[str, TestResult]


def _require_generator(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an explicit numpy.random.Generator")


def simulate_draws(
    n_draws: int, *, rng: np.random.Generator
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Generate independent exact uniform 5/49 and 1/10 draws."""
    _require_generator(rng)
    if isinstance(n_draws, (bool, np.bool_)) or not isinstance(
        n_draws, (int, np.integer)
    ):
        raise TypeError("n_draws must be a positive integer")
    if n_draws < 1:
        raise ValueError("n_draws must be a positive integer")

    main = np.zeros((int(n_draws), _DRAW_SIZE), dtype=np.int64)
    selected = np.zeros((int(n_draws), _POOL), dtype=bool)
    for column in range(_DRAW_SIZE):
        pending = np.arange(int(n_draws))
        while pending.size:
            candidates = rng.integers(1, _POOL + 1, size=pending.size)
            accepted = ~selected[pending, candidates - 1]
            rows = pending[accepted]
            values = candidates[accepted]
            main[rows, column] = values
            selected[rows, values - 1] = True
            pending = pending[~accepted]
    main.sort(axis=1)
    chance = rng.integers(1, _CHANCE_POOL + 1, size=int(n_draws), dtype=np.int64)
    return main, chance


def _integer_array(values: ArrayLike, *, name: str, ndim: int) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim != ndim:
        raise ValueError(f"{name} has invalid shape")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} values must be integers")
    return array.astype(np.int64, copy=False)


def _validate_draws(
    main_numbers: ArrayLike, chance: ArrayLike
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    main = _integer_array(main_numbers, name="main_numbers", ndim=2)
    chance_array = _integer_array(chance, name="chance", ndim=1)
    if main.shape[1:] != (_DRAW_SIZE,):
        raise ValueError("main_numbers must have shape (n, 5)")
    if main.shape[0] < 2:
        raise ValueError("at least two draws are required")
    if chance_array.shape != (main.shape[0],):
        raise ValueError("chance must have shape (n,)")
    if np.any((main < 1) | (main > _POOL)):
        raise ValueError("main_numbers must be between 1 and 49")
    if np.any(np.diff(np.sort(main, axis=1), axis=1) == 0):
        raise ValueError("main numbers within a draw must be distinct")
    if np.any((chance_array < 1) | (chance_array > _CHANCE_POOL)):
        raise ValueError("chance must be between 1 and 10")
    return np.sort(main, axis=1), chance_array


def _pearson(counts: NDArray[np.float64], expected: NDArray[np.float64]) -> float:
    mask = expected > 0
    return float(np.sum((counts[mask] - expected[mask]) ** 2 / expected[mask]))


def _total_variation(
    counts: NDArray[np.float64], probabilities: NDArray[np.float64]
) -> float:
    return float(0.5 * np.abs(counts / counts.sum() - probabilities).sum())


@lru_cache(maxsize=1)
def _ordered_probabilities() -> NDArray[np.float64]:
    probabilities = np.zeros((_DRAW_SIZE, _POOL), dtype=float)
    for order in range(_DRAW_SIZE):
        for value in range(1, _POOL + 1):
            below = comb(value - 1, order) if value - 1 >= order else 0
            above_needed = _DRAW_SIZE - order - 1
            above_available = _POOL - value
            above = comb(above_available, above_needed) if above_available >= above_needed else 0  # fmt: skip
            probabilities[order, value - 1] = below * above / _TOTAL_COMBINATIONS
    return probabilities


@lru_cache(maxsize=1)
def _sum_probabilities() -> NDArray[np.float64]:
    maximum = sum(range(_POOL - _DRAW_SIZE + 1, _POOL + 1))
    ways = np.zeros((_DRAW_SIZE + 1, maximum + 1), dtype=np.int64)
    ways[0, 0] = 1
    for value in range(1, _POOL + 1):
        for count in range(_DRAW_SIZE, 0, -1):
            ways[count, value:] += ways[count - 1, :-value]
    return ways[_DRAW_SIZE].astype(float) / _TOTAL_COMBINATIONS


@lru_cache(maxsize=1)
def _distance_probabilities() -> NDArray[np.float64]:
    probabilities = np.zeros(_POOL, dtype=float)
    for gap in range(1, _POOL):
        ways = 0
        for order in range(_DRAW_SIZE - 1):
            for lower in range(1, _POOL - gap + 1):
                left = comb(lower - 1, order) if lower - 1 >= order else 0
                right_needed = _DRAW_SIZE - order - 2
                right_available = _POOL - lower - gap
                right = comb(right_available, right_needed) if right_available >= right_needed else 0  # fmt: skip
                ways += left * right
        probabilities[gap] = ways / ((_DRAW_SIZE - 1) * _TOTAL_COMBINATIONS)
    return probabilities


@lru_cache(maxsize=1)
def _overlap_probabilities() -> NDArray[np.float64]:
    values = [comb(_DRAW_SIZE, overlap) * comb(_POOL - _DRAW_SIZE, _DRAW_SIZE - overlap) / _TOTAL_COMBINATIONS for overlap in range(_DRAW_SIZE + 1)]  # fmt: skip
    return np.asarray(values)


def _marginal(main: NDArray[np.int64]) -> tuple[float, float]:
    counts = np.bincount(main.ravel(), minlength=_POOL + 1)[1:].astype(float)
    expected = main.shape[0] * _DRAW_SIZE / _POOL
    deviation = counts - expected
    covariance_eigenvalue = (
        main.shape[0] * _DRAW_SIZE * (_POOL - _DRAW_SIZE) / (_POOL * (_POOL - 1))
    )
    statistic = float(np.sum(deviation**2) / covariance_eigenvalue)
    effect = float(np.max(np.abs(counts / main.shape[0] - _DRAW_SIZE / _POOL)))
    return statistic, effect


def _ordered(main: NDArray[np.int64]) -> tuple[float, float]:
    probabilities = _ordered_probabilities()
    counts = np.stack([np.bincount(main[:, order], minlength=_POOL + 1)[1:] for order in range(_DRAW_SIZE)]).astype(float)  # fmt: skip
    statistic = _pearson(counts, main.shape[0] * probabilities)
    effect = float(np.mean(0.5 * np.abs(counts / main.shape[0] - probabilities).sum(axis=1)))  # fmt: skip
    return statistic, effect


def _distribution_result(
    values: NDArray[np.int64], probabilities: NDArray[np.float64]
) -> tuple[float, float]:
    counts = np.bincount(values, minlength=probabilities.size)[: probabilities.size].astype(float)  # fmt: skip
    return _pearson(counts, values.size * probabilities), _total_variation(counts, probabilities)  # fmt: skip


def _metrics(
    main: NDArray[np.int64], chance: NDArray[np.int64]
) -> dict[str, tuple[float, float]]:
    parity_probabilities = np.asarray([comb(24, even) * comb(25, _DRAW_SIZE - even) / _TOTAL_COMBINATIONS for even in range(6)])  # fmt: skip
    overlaps = (main[:-1, :, None] == main[1:, None, :]).sum(axis=(1, 2))
    chance_probabilities = np.full(_CHANCE_POOL, 1 / _CHANCE_POOL)
    return {
        "marginal_frequencies": _marginal(main),
        "ordered_numbers": _ordered(main),
        "sum": _distribution_result(main.sum(axis=1), _sum_probabilities()),
        "parity": _distribution_result(
            (main % 2 == 0).sum(axis=1), parity_probabilities
        ),
        "distances": _distribution_result(
            np.diff(main, axis=1).ravel(), _distance_probabilities()
        ),
        "consecutive_overlap": _distribution_result(overlaps, _overlap_probabilities()),
        "chance": _distribution_result(chance - 1, chance_probabilities),
    }


_EFFECT_NAMES = {
    "marginal_frequencies": "maximum absolute inclusion-rate deviation",
    "ordered_numbers": "mean total-variation distance by order statistic",
    "sum": "total-variation distance",
    "parity": "total-variation distance",
    "distances": "total-variation distance",
    "consecutive_overlap": "total-variation distance",
    "chance": "total-variation distance",
}


def _holm_adjust(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Control family-wise error while preserving the original result order."""
    order = np.argsort(p_values, kind="stable")
    ranked = p_values[order]
    adjusted_ranked = np.maximum.accumulate(
        ranked * np.arange(len(ranked), 0, -1)
    ).clip(max=1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def audit_uniformity(
    main_numbers: ArrayLike,
    chance: ArrayLike,
    *,
    simulations: int,
    rng: np.random.Generator,
    confidence: float = 0.95,
) -> UniformityReport:
    """Audit draws against exact game mechanics using Monte-Carlo p-values."""
    _require_generator(rng)
    if (
        isinstance(simulations, (bool, np.bool_))
        or not isinstance(simulations, (int, np.integer))
        or simulations < 1
    ):
        raise ValueError("simulations must be a positive integer")
    if (
        isinstance(confidence, (bool, np.bool_))
        or not isinstance(confidence, (int, float, np.integer, np.floating))
        or not np.isfinite(confidence)
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be strictly between 0 and 1")
    main, chance_array = _validate_draws(main_numbers, chance)
    observed = _metrics(main, chance_array)
    null_statistics = {name: np.empty(simulations) for name in observed}
    null_effects = {name: np.empty(simulations) for name in observed}
    for simulation in range(simulations):
        null_main, null_chance = simulate_draws(len(main), rng=rng)
        for name, (statistic, effect) in _metrics(null_main, null_chance).items():
            null_statistics[name][simulation] = statistic
            null_effects[name][simulation] = effect

    tail = (1.0 - confidence) / 2.0
    raw_p_values = np.asarray(
        [
            (1 + np.count_nonzero(null_statistics[name] >= statistic))
            / (simulations + 1)
            for name, (statistic, _) in observed.items()
        ]
    )
    adjusted_p_values = _holm_adjust(raw_p_values)
    results = {}
    for index, (name, (statistic, effect)) in enumerate(observed.items()):
        interval = np.quantile(null_effects[name], [tail, 1.0 - tail])
        results[name] = TestResult(
            statistic=statistic,
            p_value=float(raw_p_values[index]),
            adjusted_p_value=float(adjusted_p_values[index]),
            effect_size=effect,
            effect_name=_EFFECT_NAMES[name],
            null_interval=(float(interval[0]), float(interval[1])),
        )
    return UniformityReport(
        n_draws=len(main),
        simulations=int(simulations),
        confidence=float(confidence),
        p_value_adjustment="Holm FWER across all seven tests",
        results=results,
    )
