"""
Analyse statistique des tirages du Loto.

Fonctionnalités:
- Fréquences individuelles et par période
- Analyses de paires, triplets (co-occurrences)
- Distribution des sommes, pairs/impairs, ranges
- Correlations entre boules
- Analyse des numéros de chance
- Détection de patterns temporels (retours à l'identique, delays)
"""

import sqlite3
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from src.loto.config import NUM_BALLS, MAX_BALL, NUM_CHANCE_MAX


def get_dataframe(db_path: str | None = None) -> pd.DataFrame:
    """Charge tous les tirages dans un DataFrame pandas."""
    from src.loto.database import get_connection

    if db_path is None:
        from src.loto.config import DB_PATH
        db_path = str(DB_PATH)

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("""
        SELECT annee_numero, date_tirage, jour_tirage,
               boule_1, boule_2, boule_3, boule_4, boule_5,
               numero_chance, combinaison_croissante
        FROM tirages
        ORDER BY date_tirage
    """, conn)
    conn.close()

    # Normaliser les jours de tirage (enlever espaces multiples)
    df["jour_tirage"] = df["jour_tirage"].str.strip()

    return df


def frequency_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyse des fréquences d'apparition de chaque numéro.

    Returns un dict avec:
      - overall: Counter global
      - by_ball: Counter par position (boule_1..5)
      - recency: dict avec les 10 derniers retours pour chaque numéro
    """
    # Récupérer tous les numéros tirés
    all_balls = []
    for _, row in df.iterrows():
        for i in range(1, NUM_BALLS + 1):
            val = row[f"boule_{i}"]
            if pd.notna(val):
                all_balls.append(int(val))

    overall = Counter(all_balls)

    # Fréquence par position
    by_ball = {}
    for i in range(1, NUM_BALLS + 1):
        positions = []
        for _, row in df.iterrows():
            val = row[f"boule_{i}"]
            if pd.notna(val):
                positions.append(int(val))
        by_ball[f"boule_{i}"] = Counter(positions)

    # Analyse de fraîcheur (derniers retours)
    recency = {}
    for num in range(1, MAX_BALL + 1):
        last_draws = []
        draw_count = len(df)
        for j, (_, row) in enumerate(df.iterrows()):
            balls = [int(row[f"boule_{k}"]) for k in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{k}"])]
            if num in balls:
                last_draws.append(draw_count - j)
        recency[num] = sorted(last_draws)[:10] if last_draws else [draw_count]

    return {
        "overall": overall,
        "by_ball": by_ball,
        "recency": recency,
    }


def pair_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la co-occurrence de toutes les paires de numéros.

    Returns un DataFrame carré (49x49) avec les comptages.
    """
    pairs = Counter()

    for _, row in df.iterrows():
        balls = sorted([int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])])
        for i in range(NUM_BALLS):
            for j in range(i + 1, NUM_BALLS):
                pairs[(balls[i], balls[j])] += 1

    # Construire le DataFrame
    matrix = np.zeros((MAX_BALL, MAX_BALL), dtype=int)
    for (a, b), count in pairs.items():
        matrix[a - 1][b - 1] = count
        matrix[b - 1][a - 1] = count

    df_matrix = pd.DataFrame(matrix, index=range(1, MAX_BALL + 1), columns=range(1, MAX_BALL + 1))
    return df_matrix


def triplet_frequencies(df: pd.DataFrame, top_n: int = 50) -> list[tuple]:
    """
    Trouve les triplets (combinaisons de 3 numéros) les plus fréquents.
    """
    triplets = Counter()

    for _, row in df.iterrows():
        balls = sorted([int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])])
        from itertools import combinations
        for combo in combinations(balls, 3):
            triplets[combo] += 1

    return triplets.most_common(top_n)


def ball_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Matrice de corrélation entre les positions des boules.
    Utilise le coefficient de Spearman (non-paramétrique).
    """
    ball_cols = [f"boule_{i}" for i in range(1, NUM_BALLS + 1)]

    # Filtrer les lignes complètes
    valid_df = df[ball_cols].dropna()
    if len(valid_df) < 10:
        return pd.DataFrame()

    # Spearman correlation
    corr_matrix = valid_df.corr(method="spearman")
    return corr_matrix


def distribution_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyses de distributions sur les tirages:
    - Somme des 5 boules
    - Ratio pairs/impairs
    - Répartition par ranges (1-10, 11-20, etc.)
    - Écart-type, min/max
    """
    all_sums = []
    all_even_ratios = []
    all_ranges = []

    for _, row in df.iterrows():
        balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
        if len(balls) != NUM_BALLS:
            continue

        # Somme
        all_sums.append(sum(balls))

        # Ratio pairs/impairs
        even = sum(1 for b in balls if b % 2 == 0)
        all_even_ratios.append(even / NUM_BALLS)

        # Ranges par tranches de 10
        range_counts = [0] * 5
        for b in balls:
            idx = min((b - 1) // 10, 4)
            range_counts[idx] += 1
        all_ranges.append(range_counts)

    stats = {}
    sums_arr = np.array(all_sums)
    stats["sum"] = {
        "mean": float(np.mean(sums_arr)),
        "std": float(np.std(sums_arr)),
        "min": int(np.min(sums_arr)),
        "max": int(np.max(sums_arr)),
        "median": float(np.median(sums_arr)),
    }

    # Distribution ranges (tranches 1-10, 11-20, etc.)
    range_df = pd.DataFrame(all_ranges, columns=["1-10", "11-20", "21-30", "31-40", "41-49"])
    stats["range_dist"] = {
        col: {
            "mean": float(range_df[col].mean()),
            "std": float(range_df[col].std()),
        }
        for col in range_df.columns
    }

    # Pair/odd ratio stats
    even_arr = np.array(all_even_ratios)
    stats["even_odd"] = {
        "mean_ratio": float(np.mean(even_arr)),
        "std_ratio": float(np.std(even_arr)),
    }

    return stats


def chance_number_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyse du numéro de chance: fréquences, distributions.
    Le numéro de chance (1-10) est indépendant mais peut avoir des patterns.
    """
    chance_values = [int(row["numero_chance"]) for _, row in df.iterrows() if pd.notna(row["numero_chance"])]

    if not chance_values:
        return {}

    counter = Counter(chance_values)

    # Distribution par jour de tirage
    chance_by_day = defaultdict(lambda: Counter())
    for _, row in df.iterrows():
        if pd.notna(row["numero_chance"]):
            day = row.get("jour_tirage", "unknown")
            chance_by_day[day][int(row["numero_chance"])] += 1

    return {
        "overall": dict(counter.most_common()),
        "by_day": {k: dict(v.most_common()) for k, v in chance_by_day.items()},
        "total_draws": len(chance_values),
    }


def temporal_patterns(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyse des patterns temporels:
    - Tirages par jour de la semaine
    - Délais entre apparitions successives d'un numéro
    - Séries (consecutive draws containing a number)
    """
    result = {
        "by_day": {},
        "avg_delays": {},
        "longest_absences": {},
    }

    # Par jour de tirage
    day_counts = Counter(df["jour_tirage"].dropna().str.strip().values)
    result["by_day"] = dict(day_counts.most_common())

    # Analyse des délais pour chaque numéro
    draw_indices = defaultdict(list)
    for idx, row in df.iterrows():
        for i in range(1, NUM_BALLS + 1):
            val = row[f"boule_{i}"]
            if pd.notna(val):
                draw_indices[int(val)].append(idx)

    delays = defaultdict(list)
    longest_absences = {}

    for num, indices in draw_indices.items():
        if len(indices) < 2:
            continue
        gap_list = [indices[i + 1] - indices[i] for i in range(len(indices) - 1)]
        delays[num].append(float(np.mean(gap_list)))

        # Plus longue absence (depuis la dernière apparition)
        last_appearance = max(indices)
        longest_absences[num] = len(df) - last_appearance

    result["avg_delays"] = {k: v[0] for k, v in delays.items()}
    result["longest_absences"] = longest_absences

    return result


def hot_cold_numbers(freq_data: dict, window: int | None = None) -> tuple[list, list]:
    """
    Identifie les numéros chauds (fréquents) et froids (rares).

    Args:
        freq_data: Résultat de frequency_analysis()
        window: Si spécifié, analyse sur la fenêtre récente (nb de tirages)

    Returns:
        (hot_numbers, cold_numbers) chacun trié par fréquence décroissante/croissante
    """
    if window:
        # Analyse sur une fenêtre glissante (recent draws only)
        hot = []
        cold = []
        for ball_pos, counter in freq_data["by_ball"].items():
            top_n = min(window, len(counter))
            hot.extend(counter.most_common(top_n)[:3])
            cold.extend(counter.most_common()[-3:])
        return hot, cold

    # Analyse globale
    overall = freq_data["overall"]
    expected_freq = sum(overall.values()) / MAX_BALL  # fréquence attendue si uniforme

    hot = [(num, count) for num, count in overall.most_common() if count > expected_freq * 1.1]
    cold = [(num, count) for num, count in overall.most_common() if count < expected_freq * 0.9]

    # Trier et limiter aux 10 principaux
    hot.sort(key=lambda x: x[1], reverse=True)
    cold.sort(key=lambda x: x[1])
    hot = hot[:10]
    cold = cold[:10]

    return hot, cold


def run_full_analysis(db_path: str | None = None) -> dict[str, Any]:
    """
    Lance toutes les analyses et retourne les résultats regroupés.
    """
    df = get_dataframe(db_path)

    analysis = {}
    analysis["frequency"] = frequency_analysis(df)
    analysis["pairs"] = pair_cooccurrence(df)
    analysis["triplets"] = triplet_frequencies(df, top_n=30)
    analysis["correlation"] = ball_correlation(df)
    analysis["distribution"] = distribution_analysis(df)
    analysis["chance"] = chance_number_analysis(df)
    analysis["temporal"] = temporal_patterns(df)

    # Hot/cold numbers
    freq_data = analysis["frequency"]
    hot, cold = hot_cold_numbers(freq_data)
    analysis["hot_numbers"] = hot
    analysis["cold_numbers"] = cold

    return analysis


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    result = run_full_analysis()
    print(f"Tirages analysés: {len(result['frequency']['overall'])} numéros tirés au total")
    print(f"Numéros les plus fréquents (global): {result['hot_numbers'][:5]}")
