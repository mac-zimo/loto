"""Temporal and frequency-summary helpers for descriptive analysis."""

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd

from loto.config import MAX_BALL, NUM_BALLS


def temporal_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """Analyze draw days and delays between number appearances."""
    result = {
        "by_day": dict(Counter(df["jour_tirage"].dropna().str.strip().values).most_common()),
        "avg_delays": {},
        "longest_absences": {},
    }
    draw_indices = defaultdict(list)
    for index, row in df.iterrows():
        for position in range(1, NUM_BALLS + 1):
            value = row[f"boule_{position}"]
            if pd.notna(value):
                draw_indices[int(value)].append(index)

    delays = {}
    longest_absences = {}
    for number, indices in draw_indices.items():
        if len(indices) < 2:
            continue
        gaps = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        delays[number] = float(np.mean(gaps))
        longest_absences[number] = len(df) - max(indices)

    result["avg_delays"] = delays
    result["longest_absences"] = longest_absences
    return result


def hot_cold_numbers(freq_data: dict, window: int | None = None) -> tuple[list, list]:
    """Identify numbers above and below their expected global frequency."""
    if window:
        hot = []
        cold = []
        for counter in freq_data["by_ball"].values():
            top_n = min(window, len(counter))
            hot.extend(counter.most_common(top_n)[:3])
            cold.extend(counter.most_common()[-3:])
        return hot, cold

    overall = freq_data["overall"]
    expected_frequency = sum(overall.values()) / MAX_BALL
    hot = [
        (number, count)
        for number, count in overall.most_common()
        if count > expected_frequency * 1.1
    ]
    cold = [
        (number, count)
        for number, count in overall.most_common()
        if count < expected_frequency * 0.9
    ]
    hot.sort(key=lambda item: item[1], reverse=True)
    cold.sort(key=lambda item: item[1])
    return hot[:10], cold[:10]
