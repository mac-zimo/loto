"""Significance helpers for legacy exploratory models."""

import pandas as pd
from scipy.stats import chi2_contingency

from loto.config import MAX_BALL, NUM_BALLS


def correlation_test_significance(df: pd.DataFrame, top_n: int = 50) -> list[dict]:
    """Apply the legacy chi-square test to frequent pairs."""
    from loto.analysis import pair_cooccurrence

    pair_matrix = pair_cooccurrence(df)
    total_draws = len(df)
    pair_counts = [
        ((i + 1, j + 1), int(pair_matrix.iloc[i, j]))
        for i in range(MAX_BALL)
        for j in range(i + 1, MAX_BALL)
        if pair_matrix.iloc[i, j] > 0
    ]
    pair_counts.sort(key=lambda item: item[1], reverse=True)
    expected = total_draws * (NUM_BALLS / MAX_BALL) ** 2 * 2
    results = []
    for (first, second), observed in pair_counts[:top_n]:
        chi2, p_value, _, _ = chi2_contingency(
            [[observed, total_draws - observed], [expected, total_draws - expected]]
        )
        if p_value < 0.05:
            results.append(
                {
                    "pair": (first, second),
                    "observed": observed,
                    "expected": round(expected, 2),
                    "chi2": round(chi2, 4),
                    "p_value": round(p_value, 6),
                    "significant": p_value < 0.01,
                }
            )
    return results[:20]
