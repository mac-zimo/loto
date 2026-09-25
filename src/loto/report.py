"""
Génération de rapports et visualisations.

Crée des graphiques (matplotlib/seaborn) et un rapport textuel résumé.
Les outputs sont sauvegardés dans le dossier output/.
"""

from pathlib import Path
from datetime import datetime

from collections import Counter
from math import comb as math_comb

from loto.config import NUM_BALLS, OUTPUT_DIR


def ensure_output_dir(output_dir: str | Path | None = None) -> Path:
    """Create and return the output directory only when explicitly called."""
    path = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


def _output_path(
    filename: str,
    output_dir: Path = OUTPUT_DIR,
    output_path: Path | None = None,
) -> Path:
    """Resolve an output path and create only its parent directory."""
    path = output_path or output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def generate_text_report(
    analysis: dict,
    modeling: dict,
    strategies: list[dict],
    chance_results: list[dict] | None = None,
    output_path: Path | None = None,
    output_dir: Path = OUTPUT_DIR,
):
    """
    Génère un rapport textuel complet.
    """
    path = _output_path("report.txt", output_dir, output_path)

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


def generate_full_report(*args, **kwargs):
    """Lazily load plotting support so importing this module has no side effects."""
    from loto.report_plots import generate_full_report as generate

    return generate(*args, **kwargs)


def plot_frequency_heatmap(freq_data: dict, output_path: Path | None = None):
    """Compatibility wrapper that imports plotting support only when called."""
    from loto.report_plots_basic import plot_frequency_heatmap as plot

    return plot(freq_data, output_path=output_path)


def plot_correlation_matrix(corr_matrix, output_path: Path | None = None):
    """Compatibility wrapper that imports plotting support only when called."""
    from loto.report_plots_basic import plot_correlation_matrix as plot

    return plot(corr_matrix, output_path=output_path)


def plot_top_pairs(pair_matrix, top_n: int = 20, output_path: Path | None = None):
    """Compatibility wrapper that imports plotting support only when called."""
    from loto.report_plots_basic import plot_top_pairs as plot

    return plot(pair_matrix, top_n=top_n, output_path=output_path)


def plot_strategy_results(strategy_results: list[dict], output_path: Path | None = None):
    """Compatibility wrapper that imports plotting support only when called."""
    from loto.report_plots_basic import plot_strategy_results as plot

    return plot(strategy_results, output_path=output_path)


def plot_chance_results(chance_results: list[dict], output_path: Path | None = None):
    """Compatibility wrapper that imports plotting support only when called."""
    from loto.report_plots_basic import plot_chance_results as plot

    return plot(chance_results, output_path=output_path)


def plot_distribution_stats(df, dist_data: dict, output_path: Path | None = None):
    """Compatibility wrapper that imports plotting support only when called."""
    from loto.report_plots import plot_distribution_stats as plot

    return plot(df, dist_data, output_path=output_path)
