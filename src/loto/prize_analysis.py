"""
Analyse des gains réels par rang du Loto.

Exploite les colonnes gagnants_rang* / rapport_rang* pour calculer
le ROI réel, comparer avec les odds théoriques.
"""

import sqlite3
from math import comb as math_comb

import numpy as np
import pandas as pd


RANK_LABELS = {
    1: "Jackpot (5+chance)", 2: "5 bons", 3: "4 bons",
    4: "3 bons", 5: "2 bons + chance", 6: "2 bons",
    7: "1 bon + chance", 8: "1 bon", 9: "Match nul (5num, chance libre)",
}
RANG_COLS = list(range(1, 10))


def get_prize_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """Charge les colonnes de gains depuis la BD (renvoie DF vide si table absente)."""
    cols = ["date_tirage", "annee_numero"]
    for r in RANG_COLS:
        cols += [f"gagnants_rang{r}", f"rapport_rang{r}"]
    query = f"SELECT {', '.join(cols)} FROM tirages ORDER BY date_tirage"
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame(columns=cols)


def _theoretical_odds() -> dict:
    """Probabilités théoriques pour chaque rang (5/49 + chance 1/10)."""
    total = math_comb(49, 5)  # 1 533 939
    return {
        1: {"matches": "5+chance", "prob": 1.0 / (total * 10)},
        2: {"matches": "5", "prob": 9.0 / (total * 10)},
        3: {"matches": "4", "prob": (5 * 44) / total},
        4: {"matches": "3", "prob": (10 * 44) / total * 9 / 10},
        5: {"matches": "2+chance", "prob": (45 * 6) / total / 10},
        6: {"matches": "2", "prob": (45 * 6) / total * 9 / 10},
        7: {"matches": "1+chance", "prob": (44 * 6) / total / 10},
        8: {"matches": "1", "prob": (44 * 6) / total * 9 / 10},
        9: {"matches": "0,5num", "prob": 374 / total / 10},
    }


def roi_by_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le ROI moyen par rang.

    Compare le montant distribué à l'enjeu estimé (joueurs_estimes * 2€).
    Returns un DataFrame avec roi_moyen, nb_tirages_gagnants, etc.
    """
    results = []
    for rang in RANG_COLS:
        wins_col = f"gagnants_rang{rang}"
        prize_col = f"rapport_rang{rang}"
        valid = df[(df[wins_col] > 0) & (df[prize_col] > 0)].copy()
        if len(valid) == 0:
            results.append({"rang": rang, "label": RANK_LABELS[rang],
                            "nb_tirages_gagnants": 0, "total_distribue": 0.0,
                            "joueurs_estimes": 0, "roi_moyen": None})
            continue

        total_prize = float(valid[prize_col].sum())
        p_theo = _theoretical_odds()[rang]["prob"]
        joueurs_par_tirage = valid[wins_col] / max(p_theo, 1e-15)
        total_joueurs_estimes = int(joueurs_par_tirage.sum())
        mise_totale = total_joueurs_estimes * 2.0
        roi_moyen = ((total_prize - mise_totale) / mise_totale * 100) if mise_totale > 0 else None

        results.append({"rang": rang, "label": RANK_LABELS[rang],
                        "nb_tirages_gagnants": len(valid),
                        "total_distribue": round(total_prize, 2),
                        "joueurs_estimes": total_joueurs_estimes,
                        "roi_moyen": round(roi_moyen, 2) if roi_moyen is not None else None})
    return pd.DataFrame(results)


def frequency_by_rank(df: pd.DataFrame) -> dict:
    """Fréquence réelle d'attribution des prix par rang."""
    stats = {}
    for rang in RANG_COLS:
        col = f"gagnants_rang{rang}"
        wins = df[col] > 0
        nb_wins = int(wins.sum())
        total = len(df)
        stats[f"rang{rang}"] = {
            "label": RANK_LABELS[rang],
            "tirages_payants": nb_wins,
            "proportion": round(nb_wins / max(total, 1), 6),
            "gagnants_moyen_gagnant": round(float(df[wins][col].mean()), 2) if nb_wins > 0 else None,
            "gagnants_moyen_global": round(float(df[col].mean()), 4),
        }
    return stats


def compare_odds_theory_vs_reel(df: pd.DataFrame) -> pd.DataFrame:
    """Compare probabilités théoriques avec fréquence observée."""
    rows = []
    total_draws = len(df)
    for rang in RANG_COLS:
        wins_count = int((df[f"gagnants_rang{rang}"] > 0).sum())
        observed_freq = wins_count / max(total_draws, 1)
        theo_prob = _theoretical_odds()[rang]["prob"]
        abs_diff = observed_freq - theo_prob
        rel_diff = (abs_diff / theo_prob * 100) if theo_prob > 0 else None

        rows.append({"rang": rang, "label": RANK_LABELS[rang],
                     "od_theorique": round(theo_prob, 8),
                     "frequence_observee": round(observed_freq, 6),
                     "cartes_absolu": round(abs_diff, 8),
                     "cartes_relif_pct": round(rel_diff, 2) if rel_diff is not None else None})
    return pd.DataFrame(rows)


def summarize_all(db_path: str | None = None) -> dict:
    """Désactivé: l'ancien calcul de ROI/rangs n'est pas méthodologiquement valide."""
    raise RuntimeError(
        "prize_analysis legacy désactivé: reconstruire les règles FDJ versionnées "
        "et utiliser les rapports réels avant toute analyse de ROI."
    )


if __name__ == "__main__":
    res = summarize_all()
    print(f"=== Analyse des gains — {res['total_draws']} tirages ===\n")
    print("--- ROI moyen par rang ---")
    for _, row in res["roi_by_rank"].iterrows():
        roi_s = f"{row['roi_moyen']}%" if row['roi_moyen'] is not None else "N/A"
        print(f"  {row['label']:28s} | payants: {row['nb_tirages_gagnants']:>3d} | "
              f"distribué: {row['total_distribue']:>14,.0f}€ | ROI: {roi_s}")

    print(f"\n--- Comparaison odds théoriques vs réel ---")
    for _, row in res["odds_comparison"].iterrows():
        print(f"  {row['label']:28s} | théorie: {row['od_theorique']:.6f} | "
              f"observé: {row['frequence_observee']:.6f} | écart: {row['cartes_relif_pct']}%")
