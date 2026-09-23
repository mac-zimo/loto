"""Plotting orchestration for complete Loto reports."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loto.config import MAX_BALL, NUM_BALLS, OUTPUT_DIR
from loto.report import generate_text_report
from loto.report_plots_basic import (
    plot_chance_results,
    plot_correlation_matrix,
    plot_frequency_heatmap,
    plot_strategy_results,
    plot_top_pairs,
)


def plot_distribution_stats(
    df: pd.DataFrame,
    dist_data: dict,
    output_dir: Path = OUTPUT_DIR,
    output_path: Path | None = None,
) -> None:
    """Plot draw-sum and even-number ratio distributions."""
    path = output_path or output_dir / "distribution_stats.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sum_stats = dist_data["sum"]
    x = np.linspace(sum_stats["min"], sum_stats["max"], 200)
    y = (1 / (sum_stats["std"] * np.sqrt(2 * np.pi))) * np.exp(
        -0.5 * ((x - sum_stats["mean"]) / sum_stats["std"]) ** 2
    )
    axes[0].axvline(
        sum_stats["mean"],
        color="red",
        linestyle="--",
        label=f"Moyenne: {sum_stats['mean']:.1f}",
    )
    axes[0].fill_between(
        [sum_stats["mean"] - sum_stats["std"], sum_stats["mean"] + sum_stats["std"]],
        0,
        y.max(),
        alpha=0.3,
        color="orange",
        label=f"±1σ ({sum_stats['std']:.1f})",
    )
    axes[0].plot(x, y * len(df) * 0.1, "k--", alpha=0.3, label="Normale théorique")
    axes[0].set_title("Distribution de la Somme des Boules")
    axes[0].set_xlabel("Somme")
    axes[0].set_ylabel("Fréquence (rel.)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    rng = np.random.default_rng(42)
    even_values = [
        sum(value % 2 == 0 for value in rng.choice(range(1, MAX_BALL + 1), NUM_BALLS, replace=False))
        / NUM_BALLS
        for _ in range(10_000)
    ]
    even_stats = dist_data["even_odd"]
    axes[1].hist(even_values, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].axvline(
        even_stats["mean_ratio"],
        color="red",
        linestyle="--",
        label=f"Moyenne: {even_stats['mean_ratio']:.2f}",
    )
    axes[1].set_title("Distribution du Ratio Pairs/Impairs")
    axes[1].set_xlabel("Ratio pairs")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Distribution stats sauvegardé: {path}")


def _try_plot(label: str, plotter, *args) -> None:
    try:
        plotter(*args)
    except Exception as error:
        print(f"  Erreur {label}: {error}")


def generate_full_report(
    analysis: dict,
    modeling: dict,
    strategies: list[dict],
    df: pd.DataFrame,
    chance_results: list[dict] | None = None,
    output_dir: Path = OUTPUT_DIR,
):
    """Generate all plots, the text report, and the JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n=== Génération des visualisations ===")
    _try_plot("heatmap fréquences", plot_frequency_heatmap, analysis.get("frequency", {}), output_dir)

    distribution = analysis.get("distribution", {})
    if distribution and "sum" in distribution:
        _try_plot("distribution stats", plot_distribution_stats, df, distribution, output_dir)
    else:
        print("  Skipped distribution stats (données insuffisantes)")

    correlation = analysis.get("correlation")
    if isinstance(correlation, pd.DataFrame) and not correlation.empty:
        _try_plot("corrélation", plot_correlation_matrix, correlation, output_dir)
    pairs = analysis.get("pairs")
    if isinstance(pairs, pd.DataFrame):
        _try_plot("top pairs", plot_top_pairs, pairs, output_dir)
    if strategies:
        _try_plot("stratégie comparison", plot_strategy_results, strategies, output_dir)
    if chance_results:
        _try_plot("chance strategy comparison", plot_chance_results, chance_results, output_dir)

    print("\n=== Génération du rapport textuel ===")
    report = generate_text_report(
        analysis,
        modeling,
        strategies,
        chance_results,
        output_dir=output_dir,
    )
    summary = {
        "hot_numbers": analysis.get("hot_numbers", []),
        "cold_numbers": analysis.get("cold_numbers", []),
        "distribution": distribution,
        "markov_predictions": modeling.get("markov_predictions", [])[:10],
        "anomalies_count": modeling.get("anomalies_count", 0),
        "significant_pairs_count": len(modeling.get("significant_pairs", [])),
        "strategies": strategies,
    }
    json_path = output_dir / "analysis_results.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"  Résultats JSON sauvegardé: {json_path}")
    return report
