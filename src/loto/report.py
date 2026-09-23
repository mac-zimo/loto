"""
Génération de rapports et visualisations.

Crée des graphiques (matplotlib/seaborn) et un rapport textuel résumé.
Les outputs sont sauvegardés dans le dossier output/.
"""

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from math import comb as math_comb

from src.loto.config import NUM_BALLS, MAX_BALL, OUTPUT_DIR


def ensure_output_dir():
    """Assure que le dossier de sortie existe."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_frequency_heatmap(freq_data: dict, output_path: Path | None = None):
    """
    Graphique heatmap des fréquences par position (boule_1 à boule_5).
    """
    ensure_output_dir()
    path = output_path or OUTPUT_DIR / "frequency_heatmap.png"

    fig, axes = plt.subplots(1, NUM_BALLS, figsize=(20, 4), sharey=True)
    if NUM_BALLS == 1:
        axes = [axes]

    colors = sns.color_palette("viridis", as_cmap=True)

    for ax_idx, ball_pos in enumerate(["boule_1", "boule_2", "boule_3", "boule_4", "boule_5"]):
        counter = freq_data["by_ball"][ball_pos]
        freqs = np.zeros(MAX_BALL)

        for num in range(1, MAX_BALL + 1):
            if num in counter:
                freqs[num - 1] = counter[num]

        im = ax_idx.plot(freqs, "o-", color="steelblue", linewidth=1.5, markersize=4)
        ax_idx.set_title(ball_pos.replace("boule_", "Boule "), fontsize=12)
        ax_idx.set_xlim(-0.5, MAX_BALL - 0.5)
        ax_idx.set_xticks(range(1, MAX_BALL + 1))
        ax_idx.set_ylabel("Fréq.")

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap sauvegardé: {path}")



def plot_correlation_matrix(corr_matrix: pd.DataFrame, output_path: Path | None = None):
    """
    Graphique de la matrice de corrélation Spearman entre positions.
    """
    ensure_output_dir()
    path = output_path or OUTPUT_DIR / "correlation_matrix.png"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")

    ax.set_xticks(range(NUM_BALLS))
    ax.set_yticks(range(NUM_BALLS))
    ax.set_xticklabels([f"Boule {i+1}" for i in range(NUM_BALLS)])
    ax.set_yticklabels([f"Boule {i+1}" for i in range(NUM_BALLS)])

    # Afficher les valeurs sur la heatmap
    for i in range(NUM_BALLS):
        for j in range(NUM_BALLS):
            text = ax.text(j, i, f"{corr_matrix.values[i, j]:.3f}",
                          ha="center", va="center", color="black" if abs(corr_matrix.values[i, j]) < 0.5 else "white")

    plt.colorbar(im, ax=ax, label="Coefficient Spearman")
    plt.title("Corrélation entre Positions de Boules (Spearman)")
    plt.tight_layout()

    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Corrélation sauvegardé: {path}")


def plot_top_pairs(pair_matrix: pd.DataFrame, top_n: int = 20, output_path: Path | None = None):
    """
    Graphique des paires les plus fréquentes.
    """
    ensure_output_dir()
    path = output_path or OUTPUT_DIR / "top_pairs.png"

    pair_counts = []
    for i in range(MAX_BALL):
        for j in range(i + 1, MAX_BALL):
            count = int(pair_matrix.iloc[i][j])
            if count > 0:
                pair_counts.append(((i + 1, j + 1), count))

    pair_counts.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pair_counts[:top_n]

    labels = [f"{a}-{b}" for (a, b), _ in top_pairs]
    counts = [c for _, c in top_pairs]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(counts)), counts, color="steelblue")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Co-occurrences")
    ax.set_title(f"Top {top_n} Paires les Plus Fréquentes")

    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(count), va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Top pairs sauvegardé: {path}")


def plot_strategy_results(strategy_results: list[dict], output_path: Path | None = None):
    """
    Compare les résultats des stratégies via un graphique.
    """
    ensure_output_dir()
    path = output_path or OUTPUT_DIR / "strategy_comparison.png"

    names = [r["strategy_name"] for r in strategy_results]
    rois = [r["roi_pct"] for r in strategy_results]
    profits = [r["net_profit"] for r in strategy_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ROI comparison
    colors_roi = ["green" if r >= 0 else "red" for r in rois]
    bars = ax1.barh(names, rois, color=colors_roi)
    ax1.set_xlabel("ROI (%)")
    ax1.set_title("ROI par Stratégie")
    ax1.axvline(x=0, color="black", linewidth=0.5)

    for bar, roi in zip(bars, rois):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{roi:.1f}%", va="center", fontsize=9)

    # Net profit comparison
    colors_profit = ["green" if p >= 0 else "red" for p in profits]
    bars = ax2.barh(names, profits, color=colors_profit)
    ax2.set_xlabel("Profit net (€)")
    ax2.set_title("Profit Net par Stratégie")
    ax2.axvline(x=0, color="black", linewidth=0.5)

    for bar, profit in zip(bars, profits):
        ax2.text(bar.get_width() + 50, bar.get_y() + bar.get_height()/2,
                f"{profit:.0f}€", va="center", fontsize=9)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stratégies comparées sauvegardé: {path}")


def plot_chance_results(chance_results: list[dict], output_path: Path | None = None):
    """Compare les résultats des stratégies numéro de chance."""
    ensure_output_dir()
    path = output_path or OUTPUT_DIR / "chance_strategy_comparison.png"

    names = [r["method"] for r in chance_results]
    improvements = [r["improvement"] for r in chance_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if v >= 0 else "red" for v in improvements]
    bars = ax.barh(names, improvements, color=colors)
    ax.set_xlabel("Amélioration ROI vs Random (pp)")
    ax.set_title("Stratégie Numéro de Chance — Impact sur le ROI")
    ax.axvline(x=0, color="black", linewidth=0.5)

    for bar, val in zip(bars, improvements):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f"{val:+.1f}pp", va="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chance strategy comparison sauvegardé: {path}")


def generate_text_report(analysis: dict, modeling: dict, strategies: list[dict], chance_results: list[dict] | None = None, output_path: Path | None = None):
    """
    Génère un rapport textuel complet.
    """
    ensure_output_dir()
    path = output_path or OUTPUT_DIR / "report.txt"

    lines = []
    lines.append("=" * 70)
    lines.append("RAPPORT D'ANALYSE - LOTTO FRANÇAIS")
    lines.append(f"Généré le: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Section 1: Fréquences
    freq_data = analysis.get("frequency", {})
    overall = freq_data.get("overall", Counter())

    lines.append("📊 FRÉQUENCES GLOBALES (Tous les tirages)")
    lines.append("-" * 40)
    if overall:
        sorted_freq = sorted(overall.items(), key=lambda x: x[1], reverse=True)
        lines.append("Numéros les plus fréquents:")
        for num, count in sorted_freq[:10]:
            lines.append(f"  #{num}: {count} apparitions")
        lines.append("")
        lines.append("Numéros les moins fréquents:")
        for num, count in sorted_freq[-5:]:
            lines.append(f"  #{num}: {count} apparitions")

    total_draws = sum(overall.values()) // NUM_BALLS
    lines.append(f"\nTotal tirages analysés: {total_draws}")
    lines.append("")

    # Section 2: Distribution
    dist = analysis.get("distribution", {})
    if "sum" in dist:
        s = dist["sum"]
        lines.append("📈 DISTRIBUTION DE LA SOMME")
        lines.append("-" * 40)
        lines.append(f"  Moyenne: {s['mean']:.1f}")
        lines.append(f"  Écart-type: {s['std']:.1f}")
        lines.append(f"  Min: {s['min']} | Max: {s['max']}")
        lines.append(f"  Médiane: {s['median']:.1f}")
        lines.append("")

    # Section 3: Paires fréquentes
    triplets = analysis.get("triplets", [])
    if triplets:
        lines.append("🔗 TOP TRIPLETS (co-occurrences)")
        lines.append("-" * 40)
        for combo, count in triplets[:10]:
            lines.append(f"  {combo}: {count} fois")
        lines.append("")

    # Section 4: Hot/Cold numbers
    hot = analysis.get("hot_numbers", [])
    cold = analysis.get("cold_numbers", [])
    if hot:
        lines.append("🔥 NUMÉROS CHAUDS (tendance haussière)")
        lines.append("-" * 40)
        for num in hot[:5]:
            lines.append(f"  #{num}")

    if cold:
        lines.append("\n❄️ NUMÉROS FROIDS (tendance baissière)")
        lines.append("-" * 40)
        for num in cold[:5]:
            lines.append(f"  #{num}")
    lines.append("")

    # Section 5: Markov predictions
    markov_pred = modeling.get("markov_predictions", [])
    if markov_pred:
        lines.append("🎯 PRÉDICTIONS MARKOV (prochain tirage)")
        lines.append("-" * 40)
        for num, score in markov_pred[:10]:
            lines.append(f"  #{num}: probabilité relative {score:.4f}")
    lines.append("")

    # Section 6: Anomalies
    anomalies_count = modeling.get("anomalies_count", 0)
    lines.append(f"🔍 ANOMALIES DÉTECTÉES: {anomalies_count}")
    if anomalies_count > 0:
        lines.append("  (Tirages statistiquement atypiques)")
    lines.append("")

    # Section 7: Stratégies
    if strategies:
        lines.append("🎮 RÉSULTATS STRATÉGIES (backtest historique)")
        lines.append("-" * 40)
        for s in strategies:
            lines.append(f"\n  {s['strategy_name']}:")
            lines.append(f"    Investi: {s['total_invested']:.2f}€ | Gagné: {s['total_won']:.2f}€")
            lines.append(f"    ROI: {s['roi_pct']}% | Profit net: {s['net_profit']}€")
            lines.append(f"    Gain max: {s['max_win']}€")
            if s.get('win_dates'):
                lines.append(f"    Gains sur: {', '.join(s['win_dates'][:5])}")

    # Section 7b: Numéro de chance
    if chance_results:
        lines.append("")
        lines.append("🍀 STRATÉGIE NUMÉRO DE CHANCE")
        lines.append("-" * 40)
        for r in chance_results:
            strat = r["strategy"]
            baseline = r["random_baseline"]
            lines.append(f"\n  Méthode '{r['method']}':")
            lines.append(f"    Stratégie ROI: {strat['roi_pct']}% | Random ROI: {baseline['roi_pct']}%")
            lines.append(f"    Amélioration: {r['improvement']:+.2f}pp")
            lines.append(f"    Investi: {strat['total_invested']:.2f}€ | Gagné: {strat['total_won']:.2f}€")

    # Section 8: Odds théoriques
    total_combos = math_comb(49, 5)
    lines.append(f"\n📐 PROBABILITÉS THÉORIQUES")
    lines.append("-" * 40)
    lines.append(f"  Total combinaisons possibles: {total_combos:,}")
    lines.append(f"  Jackpot (5+chance): 1/{total_combos * 10:,}")
    lines.append(f"  5 bons sans chance:     1/{total_combos:,}")
    lines.append(f"  ~4 bons:                1~{total_combos // 450:,}")
    lines.append("")

    lines.append("=" * 70)
    lines.append("FIN DU RAPPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    with open(str(path), "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"  Rapport sauvegardé: {path}")
    return report_text


# NUM_BALLS, MAX_BALL déjà importés plus haut avec OUTPUT_DIR


def plot_distribution_stats(df: pd.DataFrame, dist_data: dict, output_path: Path | None = None):
    """Version corrigée qui prend le DataFrame en paramètre."""
    ensure_output_dir()
    path = output_path or OUTPUT_DIR / "distribution_stats.png"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution des sommes
    sum_stats = dist_data["sum"]
    x = np.linspace(sum_stats["min"], sum_stats["max"], 200)
    y = (1 / (sum_stats["std"] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - sum_stats["mean"]) / sum_stats["std"]) ** 2)

    axes[0].axvline(sum_stats["mean"], color="red", linestyle="--", label=f"Moyenne: {sum_stats['mean']:.1f}")
    axes[0].fill_between(
        [sum_stats["mean"] - sum_stats["std"], sum_stats["mean"] + sum_stats["std"]],
        0, y.max(), alpha=0.3, color="orange", label=f"±1σ ({sum_stats['std']:.1f})"
    )
    axes[0].plot(x, y * len(df) * 0.1, "k--", alpha=0.3, label="Normale théorique")
    axes[0].set_title("Distribution de la Somme des Boules")
    axes[0].set_xlabel("Somme")
    axes[0].set_ylabel("Fréquence (rel.)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ratio pairs/impairs
    eo_stats = dist_data["even_odd"]
    even_values = []
    np.random.seed(42)
    for _ in range(10000):
        drawn = np.random.choice(range(1, MAX_BALL + 1), size=NUM_BALLS, replace=False)
        even_ratio = sum(1 for b in drawn if b % 2 == 0) / NUM_BALLS
        even_values.append(even_ratio)

    axes[1].hist(even_values, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].axvline(eo_stats["mean_ratio"], color="red", linestyle="--", label=f"Moyenne: {eo_stats['mean_ratio']:.2f}")
    axes[1].set_title("Distribution du Ratio Pairs/Impairs")
    axes[1].set_xlabel("Ratio pairs")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distribution stats sauvegardé: {path}")


def generate_full_report(analysis: dict, modeling: dict, strategies: list[dict], df: pd.DataFrame, chance_results: list[dict] | None = None):
    """
    Génère tous les rapports et visualisations.

    Args:
        analysis: Résultats de analysis.run_full_analysis()
        modeling: Résultats de models.run_modeling()
        strategies: Résultats de strategy.run_strategy_backtest()
        df: DataFrame des tirages
        chance_results: Résultats optionnels de run_chance_evaluation()

    Returns:
        Path vers le rapport textuel
    """
    print("\n=== Génération des visualisations ===")

    # Fréquence heatmap
    try:
        plot_frequency_heatmap(analysis.get("frequency", {}))
    except Exception as e:
        print(f"  Erreur heatmap fréquences: {e}")

    # Distribution stats
    try:
        dist_data = analysis.get("distribution", {})
        if dist_data and "sum" in dist_data:
            plot_distribution_stats(df, dist_data)
        else:
            print("  Skipped distribution stats (données insuffisantes)")
    except Exception as e:
        print(f"  Erreur distribution stats: {e}")

    # Correlation matrix
    corr = analysis.get("correlation")
    if isinstance(corr, pd.DataFrame) and not corr.empty:
        try:
            plot_correlation_matrix(corr)
        except Exception as e:
            print(f"  Erreur corrélation: {e}")

    # Top pairs
    pairs = analysis.get("pairs")
    if isinstance(pairs, pd.DataFrame):
        try:
            plot_top_pairs(pairs)
        except Exception as e:
            print(f"  Erreur top pairs: {e}")

    # Strategy comparison
    if strategies:
        try:
            plot_strategy_results(strategies)
        except Exception as e:
            print(f"  Erreur stratégie comparison: {e}")

    # Chance strategy comparison
    if chance_results:
        try:
            plot_chance_results(chance_results)
        except Exception as e:
            print(f"  Erreur chance strategy comparison: {e}")

    print("\n=== Génération du rapport textuel ===")
    report_path = generate_text_report(analysis, modeling, strategies, chance_results)

    # Sauvegarder aussi les résultats complets en JSON
    json_path = OUTPUT_DIR / "analysis_results.json"
    serializable_analysis = {
        "hot_numbers": analysis.get("hot_numbers", []),
        "cold_numbers": analysis.get("cold_numbers", []),
        "distribution": analysis.get("distribution", {}),
        "markov_predictions": modeling.get("markov_predictions", [])[:10],
        "anomalies_count": modeling.get("anomalies_count", 0),
        "significant_pairs_count": len(modeling.get("significant_pairs", [])),
        "strategies": strategies,
    }

    with open(str(json_path), "w", encoding="utf-8") as f:
        json.dump(serializable_analysis, f, indent=2, default=str)
    print(f"  Résultats JSON sauvegardé: {json_path}")

    return report_path


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from src.loto.analysis import get_dataframe
    df = get_dataframe()
