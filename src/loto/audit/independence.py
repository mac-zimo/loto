"""Conditional permutation tests for temporal independence of Loto draws."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from loto.audit.uniformity import _require_generator, _validate_draws

_DEFAULT_MINIMUM_EFFECTS = {"overlap": 0.05, "waiting_time": 0.03, "runs": 0.03}


@dataclass(frozen=True)
class TemporalTestResult:
    statistic: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    effect_name: str
    replication_p_values: tuple[float, float]
    replication_adjusted_p_values: tuple[float, float]
    replication_effect_sizes: tuple[float, float]
    replicated: bool
    exceeds_minimum_effect: bool
    signal: bool
    signed_effect_size: float | None = None
    replication_signed_effect_sizes: tuple[float | None, float | None] = (None, None)
    direction_consistent: bool = True


@dataclass(frozen=True)
class IndependenceReport:
    n_draws: int
    lags: tuple[int, ...]
    simulations: int
    alpha: float
    p_value_adjustment: str
    replication: str
    minimum_effects: Mapping[str, float]
    results: Mapping[str, TemporalTestResult]


def _fdr_bh(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Benjamini-Hochberg adjustment, restored to input order."""
    values = np.asarray(p_values)
    if (
        values.ndim != 1
        or values.dtype.kind not in "iuf"
        or not np.all(np.isfinite(values))
        or np.any((values < 0) | (values > 1))
    ):
        raise ValueError("p_values must be a one-dimensional array of finite values in [0, 1]")
    values = values.astype(float, copy=False)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1].clip(max=1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def _validate_options(
    lags: Sequence[int],
    simulations: int,
    alpha: float,
    minimum_effects: Mapping[str, Real] | None,
) -> tuple[tuple[int, ...], dict[str, float]]:
    if not lags:
        raise ValueError("lags must be a non-empty sequence")
    if any(
        isinstance(lag, (bool, np.bool_))
        or not isinstance(lag, (int, np.integer))
        or lag < 1
        for lag in lags
    ):
        raise TypeError("lags must contain positive integers")
    normalized_lags = tuple(int(lag) for lag in lags)
    if len(set(normalized_lags)) != len(normalized_lags):
        raise ValueError("lags must be unique")
    if (
        isinstance(simulations, (bool, np.bool_))
        or not isinstance(simulations, (int, np.integer))
        or simulations < 1
    ):
        raise ValueError("simulations must be a positive integer")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    thresholds = dict(_DEFAULT_MINIMUM_EFFECTS if minimum_effects is None else minimum_effects)
    if set(thresholds) != set(_DEFAULT_MINIMUM_EFFECTS):
        raise ValueError("minimum_effects must define overlap, waiting_time, and runs")
    for value in thresholds.values():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError("minimum_effects must be finite and non-negative")
        if not np.isfinite(float(value)) or value < 0:
            raise ValueError("minimum_effects must be finite and non-negative")
    return normalized_lags, {name: float(value) for name, value in thresholds.items()}


def _overlap(main: NDArray[np.int64], lag: int) -> tuple[float, float, float]:
    values = (main[:-lag, :, None] == main[lag:, None, :]).sum(axis=(1, 2))
    signed_effect = (float(values.mean()) - 25 / 49) / 5
    return abs(signed_effect), abs(signed_effect), signed_effect


def _repeat(chance: NDArray[np.int64], lag: int) -> tuple[float, float, float]:
    signed_effect = float(np.mean(chance[:-lag] == chance[lag:])) - 0.1
    return abs(signed_effect), abs(signed_effect), signed_effect


def _waiting(values: NDArray[np.bool_], probability: float) -> tuple[float, float, None]:
    gaps: list[int] = []
    for column in range(values.shape[1]):
        hits = np.flatnonzero(values[:, column])
        gaps.extend(np.diff(hits).tolist())
    if not gaps:
        return 0.0, 0.0, None
    clipped = np.minimum(np.asarray(gaps), 11)
    counts = np.bincount(clipped, minlength=12)[1:12].astype(float)
    expected = np.r_[probability * (1 - probability) ** np.arange(10), (1 - probability) ** 10]
    effect = float(0.5 * np.abs(counts / counts.sum() - expected).sum())
    return effect, effect, None


def _runs(values: NDArray[np.bool_]) -> tuple[float, float, float]:
    maximum_z = 0.0
    maximum_effect = 0.0
    signed_effect = 0.0
    n = len(values)
    for column in range(values.shape[1]):
        series = values[:, column]
        n1 = int(series.sum())
        n0 = n - n1
        if not n0 or not n1:
            continue
        runs = 1 + int(np.count_nonzero(series[1:] != series[:-1]))
        expected = 1 + 2 * n0 * n1 / n
        variance = 2 * n0 * n1 * (2 * n0 * n1 - n) / (n * n * (n - 1))
        if variance > 0:
            maximum_z = max(maximum_z, abs(runs - expected) / np.sqrt(variance))
        candidate = (runs - expected) / n
        if abs(candidate) > maximum_effect:
            maximum_effect = abs(candidate)
            signed_effect = candidate
    return float(maximum_z), float(maximum_effect), float(signed_effect)


def _metrics(
    main: NDArray[np.int64], chance: NDArray[np.int64], lags: tuple[int, ...]
) -> dict[str, tuple[float, float, float | None, str, str]]:
    main_hits = (main[:, :, None] == np.arange(1, 50)).any(axis=1)
    chance_hits = chance[:, None] == np.arange(1, 11)
    metrics: dict[str, tuple[float, float, float | None, str, str]] = {}
    for lag in lags:
        if lag >= len(main):
            raise ValueError("lags must be smaller than each replication period")
        metrics[f"main_overlap_lag_{lag}"] = (*_overlap(main, lag), "overlap", "absolute normalized mean-overlap deviation")
        metrics[f"chance_repeat_lag_{lag}"] = (*_repeat(chance, lag), "overlap", "absolute repeat-rate deviation")
    metrics["main_waiting_time"] = (*_waiting(main_hits, 5 / 49), "waiting_time", "total-variation distance from geometric waiting times")
    metrics["chance_waiting_time"] = (*_waiting(chance_hits, 0.1), "waiting_time", "total-variation distance from geometric waiting times")
    metrics["main_runs"] = (*_runs(main_hits), "runs", "maximum normalized runs deviation")
    metrics["chance_runs"] = (*_runs(chance_hits), "runs", "maximum normalized runs deviation")
    return metrics


def _permutation_results(
    main: NDArray[np.int64],
    chance: NDArray[np.int64],
    lags: tuple[int, ...],
    simulations: int,
    rng: np.random.Generator,
) -> tuple[dict[str, tuple[float, float, float | None, str, str]], NDArray[np.float64]]:
    observed = _metrics(main, chance, lags)
    null = np.empty((simulations, len(observed)))
    for index in range(simulations):
        order = rng.permutation(len(main))
        shuffled = _metrics(main[order], chance[order], lags)
        null[index] = [value[0] for value in shuffled.values()]
    p_values = np.asarray(
        [(1 + np.count_nonzero(null[:, i] >= value[0])) / (simulations + 1) for i, value in enumerate(observed.values())]
    )
    return observed, p_values


def _has_consistent_direction(
    signed_effect: float | None,
    replication_signed_effects: tuple[float | None, float | None],
) -> bool:
    first, second = replication_signed_effects
    if signed_effect is None:
        return first is None and second is None
    if first is None or second is None:
        return False
    directions = tuple(np.sign(value) for value in (signed_effect, first, second))
    return directions[0] != 0 and len(set(directions)) == 1


def audit_independence(
    main_numbers: ArrayLike,
    chance: ArrayLike,
    *,
    lags: Sequence[int] = (1, 2, 5, 10),
    simulations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    minimum_effects: Mapping[str, Real] | None = None,
) -> IndependenceReport:
    """Test temporal structure conditionally on observed marginal frequencies."""
    _require_generator(rng)
    normalized_lags, thresholds = _validate_options(lags, simulations, alpha, minimum_effects)
    main, chance_array = _validate_draws(main_numbers, chance)
    midpoint = len(main) // 2
    if midpoint < 2 or len(main) - midpoint < 2:
        raise ValueError("at least four draws are required for replication")
    periods = [(main, chance_array), (main[:midpoint], chance_array[:midpoint]), (main[midpoint:], chance_array[midpoint:])]
    analyses = [_permutation_results(period_main, period_chance, normalized_lags, int(simulations), rng) for period_main, period_chance in periods]
    adjusted = [_fdr_bh(p_values) for _, p_values in analyses]
    results = {}
    for index, (name, (statistic, effect, signed_effect, family, effect_name)) in enumerate(analyses[0][0].items()):
        period_effects = (
            float(analyses[1][0][name][1]),
            float(analyses[2][0][name][1]),
        )
        period_signed_effects = (
            analyses[1][0][name][2],
            analyses[2][0][name][2],
        )
        period_p = (
            float(analyses[1][1][index]),
            float(analyses[2][1][index]),
        )
        period_adjusted = (
            float(adjusted[1][index]),
            float(adjusted[2][index]),
        )
        replicated = all(value <= alpha for value in period_adjusted)
        exceeds = effect >= thresholds[family] and all(value >= thresholds[family] for value in period_effects)
        direction_consistent = _has_consistent_direction(signed_effect, period_signed_effects)
        full_adjusted = float(adjusted[0][index])
        results[name] = TemporalTestResult(
            statistic=float(statistic), p_value=float(analyses[0][1][index]), adjusted_p_value=full_adjusted,
            effect_size=float(effect), effect_name=effect_name, replication_p_values=period_p,
            replication_adjusted_p_values=period_adjusted, replication_effect_sizes=period_effects,
            replicated=replicated, exceeds_minimum_effect=exceeds,
            signal=full_adjusted <= alpha and replicated and exceeds and direction_consistent,
            signed_effect_size=signed_effect,
            replication_signed_effect_sizes=period_signed_effects,
            direction_consistent=direction_consistent,
        )
    return IndependenceReport(
        n_draws=len(main), lags=normalized_lags, simulations=int(simulations), alpha=float(alpha),
        p_value_adjustment="Benjamini-Hochberg FDR", replication="chronological halves",
        minimum_effects=thresholds, results=results,
    )
