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
        for idx, (_, row) in enumerate(df.iloc[:n_draws].itertuples()):
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


def analyze_periodicity(db_path: str | None = None) -> dict[str, Any]:
    """Lance l'analyse de périodicité complète. Returns per_number + summary."""
    from src.loto.analysis import get_dataframe

    df = get_dataframe(db_path)
    n_draws = len(df)
    if n_draws < 20:
        return {"error": "Pas assez de tirages (< 20) pour une analyse de périodicité"}

    series = _binary_series(df)
    max_lag = min(n_draws // 2, 50)
    per_number: dict[int, dict[str, Any]] = {}
    cyclic_numbers = []

    for num in range(1, MAX_BALL + 1):
        arr = series[num]
        rate = float(np.mean(arr))
        lags, acf_vals = _acf(arr, max_lag)
        # Significativité ACF: seuil ≈ ±1.96/√n (IC 95%)
        ci = 1.96 / np.sqrt(n_draws)
        sig_lags = [lag for lag in range(1, len(acf_vals)) if abs(acf_vals[lag]) > ci]

        fft_info = _fft_periods(arr)

        per_number[num] = {
            "occurrence_rate": round(rate, 4),
            "acf_significant_lags": sig_lags[:10],
            "fft_dominant_periods": fft_info["dominant_periods"][:3],
        }
        if fft_info["dominant_periods"] and sig_lags:
            cyclic_numbers.append({
                "number": num, "rate": round(rate, 4),
                "periods": fft_info["dominant_periods"], "sig_lags": sig_lags[:5],
            })

    cyclic_numbers.sort(key=lambda x: len(x["periods"]) + len(x["sig_lags"]), reverse=True)

    summary = {
        "total_draws": n_draws,
        "cyclic_numbers_count": len(cyclic_numbers),
        "cyclic_numbers": cyclic_numbers[:20],
        "interpretation": (
            f"{len(cyclic_numbers)} numéro(s) cyclique(s) détecté(s) sur {n_draws} tirages."
            if cyclic_numbers
            else "Aucun cycle significatif — compatible avec du bruit blanc."
        ),
    }
    return {"per_number": per_number, "summary": summary}


