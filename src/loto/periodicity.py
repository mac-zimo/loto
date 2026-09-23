"""Analyse de périodicité — détection de cycles (ACF + FFT)."""

import sqlite3
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal, stats
from src.loto.config import MAX_BALL


def _binary_series(df: pd.DataFrame) -> dict[int, np.ndarray]:
    """Pour chaque numéro (1..49), un tableau binaire 0/1 par tirage."""
    series: dict[int, np.ndarray] = {}
    n_draws = len(df)
    for num in range(1, MAX_BALL + 1):
        arr = np.zeros(n_draws, dtype=float)
        for idx, row in enumerate(df.iloc[:n_draws].itertuples(index=False)):
            if num in (row.boule_1, row.boule_2, row.boule_3, row.boule_4, row.boule_5):
                arr[idx] = 1.0
        series[num] = arr
    return series


def _acf(x: np.ndarray, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Autocorrélation normalisée. Returns (lags, acf_vals), acf[0]==1.0."""
    n = len(x)
    if max_lag is None:
        max_lag = min(n // 2, 50)
    mean = np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return np.arange(max_lag), np.zeros(max_lag)
    vals = [1.0] + [float(np.mean((x[:n - l] - mean) * (x[l:] - mean)) / var) for l in range(1, max_lag)]
    return np.arange(max_lag), np.array(vals)


def _fft_periods(x: np.ndarray) -> dict[str, Any]:
    """FFT → périodes dominantes. Returns dominant_periods threshold total_power."""
    n = len(x)
    fft_vals = np.fft.rfft(x - np.mean(x))
    mags = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n)
    mags[0] = 0.0  # ignorer DC
    thresh = float(np.mean(mags[1:]) + 2 * np.std(mags[1:])) if n > 4 else 0
    peaks_idx = np.where(mags > thresh)[0]
    periods = []
    for idx in peaks_idx:
        if freqs[idx] < 1e-6:
            continue
        period = 1.0 / freqs[idx]
        if 5 <= period <= n / 2:
            periods.append((int(round(period)), float(mags[idx])))
    periods.sort(key=lambda x: x[1], reverse=True)
    return {"dominant_periods": periods[:5], "threshold": thresh, "total_power": float(np.sum(mags[1:]**2))}


def _benjamini_hochberg(p_values: list[float]) -> np.ndarray:
    """Ajuste une famille de p-values par contrôle FDR de Benjamini-Hochberg."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate(
        (ranked * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def analyze_periodicity(db_path: str | None = None) -> dict[str, Any]:
    """Analyse exploratoire ACF/FFT avec correction globale des tests multiples."""
    from src.loto.analysis import get_dataframe

    df = get_dataframe(db_path)
    n_draws = len(df)
    if n_draws < 20:
        return {"error": "Pas assez de tirages (< 20) pour une analyse de périodicité"}

    series = _binary_series(df)
    max_lag = min(n_draws // 2, 50)
    raw: dict[int, dict[str, Any]] = {}
    tests: list[tuple[int, int, float]] = []

    for num in range(1, MAX_BALL + 1):
        arr = series[num]
        _, acf_vals = _acf(arr, max_lag)
        for lag in range(1, len(acf_vals)):
            z_score = abs(acf_vals[lag]) * np.sqrt(n_draws)
            p_value = float(2 * stats.norm.sf(z_score))
            tests.append((num, lag, p_value))
        raw[num] = {
            "occurrence_rate": float(np.mean(arr)),
            "acf": acf_vals,
            "fft": _fft_periods(arr),
        }

    q_values = _benjamini_hochberg([test[2] for test in tests])
    corrected_lags: dict[int, list[int]] = {num: [] for num in range(1, MAX_BALL + 1)}
    for (num, lag, _), q_value in zip(tests, q_values):
        if q_value < 0.05:
            corrected_lags[num].append(lag)

    per_number: dict[int, dict[str, Any]] = {}
    cyclic_numbers = []
    for num in range(1, MAX_BALL + 1):
        periods = raw[num]["fft"]["dominant_periods"][:3]
        sig_lags = corrected_lags[num]
        matched = [
            period for period, _ in periods
            if any(abs(period - lag) <= 1 for lag in sig_lags)
        ]
        per_number[num] = {
            "occurrence_rate": round(raw[num]["occurrence_rate"], 4),
            "acf_fdr_significant_lags": sig_lags[:10],
            "fft_candidate_periods": periods,
            "acf_fft_matches": matched,
        }
        if matched:
            cyclic_numbers.append({
                "number": num,
                "matched_periods": matched,
                "acf_fdr_significant_lags": sig_lags[:10],
            })

    summary = {
        "total_draws": n_draws,
        "tests_corrected": len(tests),
        "cyclic_numbers_count": len(cyclic_numbers),
        "cyclic_numbers": cyclic_numbers,
        "interpretation": (
            "Candidats exploratoires après correction FDR; toute réplication doit être "
            "confirmée sur une période hors échantillon et par permutation."
            if cyclic_numbers
            else "Aucun cycle robuste après correction globale des tests multiples."
        ),
    }
    return {"per_number": per_number, "summary": summary}


