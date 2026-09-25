"""Permutation audit for year and available draw-mechanism regimes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np
from numpy.typing import ArrayLike, NDArray

from loto.audit.independence import _fdr_bh
from loto.audit.uniformity import _require_generator, _validate_draws


@dataclass(frozen=True)
class RegimeTestResult:
    groups: int
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


@dataclass(frozen=True)
class ChangePointReport:
    n_draws: int
    simulations: int
    alpha: float
    minimum_effect: float
    p_value_adjustment: str
    replication: str
    results: Mapping[str, RegimeTestResult]
    skipped: Mapping[str, str]


def _validate_options(simulations: int, alpha: float, minimum_effect: float) -> None:
    if (
        isinstance(simulations, (bool, np.bool_))
        or not isinstance(simulations, (int, np.integer))
        or simulations < 1
    ):
        raise ValueError("simulations must be a positive integer")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between zero and one")
    if isinstance(minimum_effect, (bool, np.bool_)) or not isinstance(minimum_effect, (int, float)) or not np.isfinite(minimum_effect) or minimum_effect < 0:
        raise ValueError("minimum_effect must be finite and non-negative")


def _years(values: Sequence[date | datetime], n_draws: int) -> NDArray[np.int64]:
    if len(values) != n_draws:
        raise ValueError("dates must have one value per draw")
    if any(not isinstance(value, (date, datetime)) for value in values):
        raise TypeError("dates must contain date or datetime values")
    aware = [isinstance(value, datetime) and value.utcoffset() is not None for value in values]
    if any(aware) and not all(aware):
        raise ValueError(
            "dates cannot mix timezone-aware datetimes with naive datetimes or date values"
        )
    if all(aware):
        chronological = [value.astimezone(timezone.utc) for value in values if isinstance(value, datetime)]
        if any(chronological[index] > chronological[index + 1] for index in range(n_draws - 1)):
            raise ValueError("dates must be in chronological order")
        years = [value.year for value in chronological]
    else:
        chronological = [
            (value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond, value.fold)
            if isinstance(value, datetime)
            else (value.year, value.month, value.day, 0, 0, 0, 0, 0)
            for value in values
        ]
        if any(chronological[index] > chronological[index + 1] for index in range(n_draws - 1)):
            raise ValueError("dates must be in chronological order")
        years = [value.year for value in values]
    return np.asarray(years, dtype=np.int64)


def _is_scalar_label(value: object) -> bool:
    if isinstance(value, np.ndarray):
        return False
    try:
        if np.asarray(value, dtype=object).ndim != 0:
            return False
        hash(value)
    except (TypeError, ValueError):
        return False
    return True


def _is_missing_label(value: object) -> bool:
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    try:
        unequal = value != value
        return bool(unequal)
    except (TypeError, ValueError):
        # Nullable scalar sentinels such as pandas.NA have no truth value.
        return True


def _encode_labels(values: NDArray[np.object_]) -> NDArray[np.int64]:
    groups: dict[object, int] = {}
    encoded = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        if value not in groups:
            groups[value] = len(groups)
        encoded[index] = groups[value]
    return encoded


def _labels(
    years: NDArray[np.int64],
    metadata: Mapping[str, ArrayLike | None] | None,
) -> tuple[dict[str, NDArray[np.int64]], dict[str, str]]:
    factors = {"year": years}
    skipped: dict[str, str] = {}
    for name, values in (metadata or {}).items():
        if name == "year":
            raise ValueError("metadata key 'year' is reserved")
        if values is None:
            skipped[name] = "metadata unavailable"
            continue
        try:
            array = np.asarray(values, dtype=object)
        except (TypeError, ValueError):
            skipped[name] = "metadata must be one-dimensional scalar labels"
            continue
        if array.ndim != 1 or any(not _is_scalar_label(value) for value in array):
            skipped[name] = "metadata must be one-dimensional scalar labels"
            continue
        if len(array) != len(years):
            raise ValueError(f"metadata field {name} must have one value per draw")
        if any(_is_missing_label(value) for value in array):
            skipped[name] = "metadata incomplete"
            continue
        factors[name] = _encode_labels(array)
    return factors, skipped


def _profiles(
    main: NDArray[np.int64], chance: NDArray[np.int64], labels: NDArray[np.int64]
) -> list[NDArray[np.float64]]:
    profiles = []
    for group in np.unique(labels):
        selected = labels == group
        main_counts = np.bincount(main[selected].ravel(), minlength=50)[1:].astype(float)
        main_distribution = main_counts / main_counts.sum()
        chance_counts = np.bincount(chance[selected], minlength=11)[1:].astype(float)
        chance_distribution = chance_counts / chance_counts.sum()
        profiles.append(np.r_[main_distribution, chance_distribution])
    return profiles


def _effect(
    main: NDArray[np.int64], chance: NDArray[np.int64], labels: NDArray[np.int64]
) -> float:
    profiles = _profiles(main, chance, labels)
    distances = []
    for first in range(len(profiles)):
        for second in range(first + 1, len(profiles)):
            main_tv = 0.5 * np.abs(profiles[first][:49] - profiles[second][:49]).sum()
            chance_tv = 0.5 * np.abs(profiles[first][49:] - profiles[second][49:]).sum()
            distances.append(max(main_tv, chance_tv))
    return float(max(distances, default=0.0))


def _replication_indices(labels: NDArray[np.int64]) -> tuple[NDArray[np.int64], NDArray[np.int64]] | None:
    early: list[int] = []
    late: list[int] = []
    for group in np.unique(labels):
        positions = np.flatnonzero(labels == group)
        if len(positions) < 4:
            return None
        midpoint = len(positions) // 2
        early.extend(positions[:midpoint])
        late.extend(positions[midpoint:])
    return np.asarray(sorted(early)), np.asarray(sorted(late))


def _permutation_p_value(
    main: NDArray[np.int64],
    chance: NDArray[np.int64],
    labels: NDArray[np.int64],
    simulations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = _effect(main, chance, labels)
    exceedances = 0
    for _ in range(simulations):
        exceedances += _effect(main, chance, labels[rng.permutation(len(labels))]) >= observed
    return observed, (1 + exceedances) / (simulations + 1)


def audit_regime_changes(
    main_numbers: ArrayLike,
    chance: ArrayLike,
    dates: Sequence[date | datetime],
    *,
    metadata: Mapping[str, ArrayLike | None] | None = None,
    simulations: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
    minimum_effect: float = 0.10,
) -> ChangePointReport:
    """Compare years and supplied machine/rule regimes with replicated tests.

    Naive datetimes and dates share local wall-clock ordering (dates are midnight).
    Aware datetimes are ordered and assigned to years in UTC, and cannot be mixed
    with naive datetimes or dates because that ordering would be ambiguous.
    """
    _require_generator(rng)
    _validate_options(simulations, alpha, minimum_effect)
    main, chance_array = _validate_draws(main_numbers, chance)
    years = _years(dates, len(main))
    factors, skipped = _labels(years, metadata)
    eligible: dict[str, tuple[NDArray[np.int64], tuple[NDArray[np.int64], NDArray[np.int64]]]] = {}
    for name, labels in factors.items():
        if len(np.unique(labels)) < 2:
            skipped[name] = "fewer than two regimes"
            continue
        period_indices = _replication_indices(labels)
        if period_indices is None:
            skipped[name] = "a regime has fewer than four draws"
            continue
        eligible[name] = (labels, period_indices)

    raw = np.empty((len(eligible), 3))
    effects = np.empty((len(eligible), 3))
    for row, (labels, period_indices) in enumerate(eligible.values()):
        effects[row, 0], raw[row, 0] = _permutation_p_value(main, chance_array, labels, int(simulations), rng)
        for period, indices in enumerate(period_indices, 1):
            effects[row, period], raw[row, period] = _permutation_p_value(
                main[indices], chance_array[indices], labels[indices], int(simulations), rng
            )
    adjusted = np.column_stack([_fdr_bh(raw[:, column]) for column in range(3)]) if eligible else raw
    results = {}
    for row, (name, (labels, _)) in enumerate(eligible.items()):
        replication_p = (float(raw[row, 1]), float(raw[row, 2]))
        replication_adjusted = (float(adjusted[row, 1]), float(adjusted[row, 2]))
        replication_effects = (float(effects[row, 1]), float(effects[row, 2]))
        replicated = all(value <= alpha for value in replication_adjusted)
        exceeds = bool(np.all(effects[row] >= minimum_effect))
        results[name] = RegimeTestResult(
            groups=len(np.unique(labels)), statistic=float(effects[row, 0]),
            p_value=float(raw[row, 0]), adjusted_p_value=float(adjusted[row, 0]),
            effect_size=float(effects[row, 0]), effect_name="maximum pairwise total-variation distance",
            replication_p_values=replication_p,
            replication_adjusted_p_values=replication_adjusted,
            replication_effect_sizes=replication_effects, replicated=replicated,
            exceeds_minimum_effect=exceeds,
            signal=adjusted[row, 0] <= alpha and replicated and exceeds,
        )
    return ChangePointReport(
        n_draws=len(main), simulations=int(simulations), alpha=float(alpha),
        minimum_effect=float(minimum_effect), p_value_adjustment="Benjamini-Hochberg FDR",
        replication="early/late halves within each regime", results=results, skipped=skipped,
    )
