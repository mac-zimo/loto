"""Basic Matplotlib plots used by the report pipeline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from loto.config import MAX_BALL, NUM_BALLS, OUTPUT_DIR


def _plot_path(filename: str, output_dir: Path, output_path: Path | None) -> Path:
    path = output_path or output_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_frequency_heatmap(
    freq_data: dict,
    output_dir: Path = OUTPUT_DIR,
    output_path: Path | None = None,
) -> None:
    """Plot frequencies for each ordered ball position."""
    path = _plot_path("frequency_heatmap.png", output_dir, output_path)
    fig, axes = plt.subplots(1, NUM_BALLS, figsize=(20, 4), sharey=True)
    if NUM_BALLS == 1:
        axes = [axes]

    for index, ball_pos in enumerate(f"boule_{i}" for i in range(1, NUM_BALLS + 1)):
        counter = freq_data["by_ball"][ball_pos]
        freqs = np.array([counter.get(num, 0) for num in range(1, MAX_BALL + 1)])
        ax = axes[index]
        ax.plot(
            range(1, MAX_BALL + 1),
            freqs,
            "o-",
            color="steelblue",
            linewidth=1.5,
            markersize=4,
        )
        ax.set_title(ball_pos.replace("boule_", "Boule "), fontsize=12)
        ax.set_xlim(0.5, MAX_BALL + 0.5)
        ax.set_xticks(range(1, MAX_BALL + 1, 8))
        ax.set_ylabel("Fréq.")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap sauvegardé: {path}")


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    output_path: Path | None = None,
) -> None:
    """Plot the Spearman correlation matrix between ball positions."""
    path = _plot_path("correlation_matrix.png", output_dir, output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(NUM_BALLS))
    ax.set_yticks(range(NUM_BALLS))
    ax.set_xticklabels([f"Boule {i + 1}" for i in range(NUM_BALLS)])
    ax.set_yticklabels([f"Boule {i + 1}" for i in range(NUM_BALLS)])
    for i in range(NUM_BALLS):
        for j in range(NUM_BALLS):
            value = corr_matrix.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="black" if abs(value) < 0.5 else "white",
            )
    fig.colorbar(im, ax=ax, label="Coefficient Spearman")
    ax.set_title("Corrélation entre Positions de Boules (Spearman)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Corrélation sauvegardé: {path}")


def plot_top_pairs(
    pair_matrix: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    top_n: int = 20,
    output_path: Path | None = None,
) -> None:
    """Plot the most frequent number pairs."""
    path = _plot_path("top_pairs.png", output_dir, output_path)
    pair_counts = [
        ((i + 1, j + 1), int(pair_matrix.iloc[i, j]))
        for i in range(MAX_BALL)
        for j in range(i + 1, MAX_BALL)
        if pair_matrix.iloc[i, j] > 0
    ]
    pair_counts.sort(key=lambda item: item[1], reverse=True)
    top_pairs = pair_counts[:top_n]
    labels = [f"{a}-{b}" for (a, b), _ in top_pairs]
    counts = [count for _, count in top_pairs]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(counts)), counts, color="steelblue")
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Co-occurrences")
    ax.set_title(f"Top {top_n} Paires les Plus Fréquentes")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Top pairs sauvegardé: {path}")


def plot_strategy_results(
    strategy_results: list[dict],
    output_dir: Path = OUTPUT_DIR,
    output_path: Path | None = None,
) -> None:
    """Compare strategy ROI and net profit."""
    path = _plot_path("strategy_comparison.png", output_dir, output_path)
    names = [result["strategy_name"] for result in strategy_results]
    rois = [result["roi_pct"] for result in strategy_results]
    profits = [result["net_profit"] for result in strategy_results]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    bars = ax1.barh(names, rois, color=["green" if roi >= 0 else "red" for roi in rois])
    ax1.set_xlabel("ROI (%)")
    ax1.set_title("ROI par Stratégie")
    ax1.axvline(x=0, color="black", linewidth=0.5)
    for bar, roi in zip(bars, rois):
        ax1.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{roi:.1f}%",
            va="center",
            fontsize=9,
        )

    bars = ax2.barh(
        names,
        profits,
        color=["green" if profit >= 0 else "red" for profit in profits],
    )
    ax2.set_xlabel("Profit net (€)")
    ax2.set_title("Profit Net par Stratégie")
    ax2.axvline(x=0, color="black", linewidth=0.5)
    for bar, profit in zip(bars, profits):
        ax2.text(
            bar.get_width() + 50,
            bar.get_y() + bar.get_height() / 2,
            f"{profit:.0f}€",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Stratégies comparées sauvegardé: {path}")


def plot_chance_results(
    chance_results: list[dict],
    output_dir: Path = OUTPUT_DIR,
    output_path: Path | None = None,
) -> None:
    """Compare chance-number strategy results."""
    path = _plot_path("chance_strategy_comparison.png", output_dir, output_path)
    names = [result["method"] for result in chance_results]
    improvements = [result["improvement"] for result in chance_results]
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if value >= 0 else "red" for value in improvements]
    bars = ax.barh(names, improvements, color=colors)
    ax.set_xlabel("Amélioration ROI vs Random (pp)")
    ax.set_title("Stratégie Numéro de Chance — Impact sur le ROI")
    ax.axvline(x=0, color="black", linewidth=0.5)
    for bar, value in zip(bars, improvements):
        ax.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.1f}pp",
            va="center",
            fontsize=10,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Chance strategy comparison sauvegardé: {path}")
