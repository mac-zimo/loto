"""Deterministic data and figure exports for marginal power reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, fields
from pathlib import Path

from loto.audit.power_models import PowerCurveReport, PowerPoint


def _serializable(report: PowerCurveReport) -> dict[str, object]:
    payload = asdict(report)
    payload["curves"] = {
        scenario: [asdict(point) for point in points]
        for scenario, points in report.curves.items()
    }
    return payload


def _require_power_points(report: PowerCurveReport) -> None:
    if not any(report.curves.values()):
        raise ValueError("report must contain at least one power point")


def export_power_data(report: PowerCurveReport, path: str | Path) -> None:
    """Export deterministic curve data as ``.csv`` or a full JSON manifest."""
    if not isinstance(report, PowerCurveReport):
        raise TypeError("report must be a PowerCurveReport")
    _require_power_points(report)
    destination = Path(path)
    if destination.suffix.lower() == ".json":
        destination.write_text(
            json.dumps(_serializable(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if destination.suffix.lower() != ".csv":
        raise ValueError("power data path must end in .csv or .json")
    fieldnames = ["scenario", *(field.name for field in fields(PowerPoint))]
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for scenario, points in report.curves.items():
            for point in points:
                writer.writerow({"scenario": scenario, **asdict(point)})


def plot_power_curves(report: PowerCurveReport, path: str | Path) -> None:
    """Render a byte-reproducible PNG or SVG power curve."""
    if not isinstance(report, PowerCurveReport):
        raise TypeError("report must be a PowerCurveReport")
    _require_power_points(report)
    destination = Path(path)
    suffix = destination.suffix.lower()
    if suffix not in {".png", ".svg"}:
        raise ValueError("power plot path must end in .png or .svg")

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    with mpl.rc_context(rc=mpl.rcParamsDefault):
        mpl.rcParams["axes.linewidth"] = 0.8
        mpl.rcParams["figure.dpi"] = 100.0
        mpl.rcParams["font.family"] = ["DejaVu Sans"]
        mpl.rcParams["font.size"] = 10.0
        mpl.rcParams["lines.linewidth"] = 1.5
        mpl.rcParams["savefig.dpi"] = 150.0
        mpl.rcParams["svg.hashsalt"] = "loto-analyze-power-v1"
        figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
        for scenario, points in report.curves.items():
            effects = [point.effect_size for point in points]
            powers = [point.power for point in points]
            lows = [point.interval_low for point in points]
            highs = [point.interval_high for point in points]
            axis.plot(effects, powers, marker="o", label=scenario)
            axis.fill_between(effects, lows, highs, alpha=0.18)
        axis.axhline(
            report.power_target,
            color="black",
            linestyle="--",
            label="power target",
        )
        axis.set(
            xlabel="Injected probability-point bias",
            ylabel="Estimated power",
            ylim=(0, 1.02),
        )
        axis.legend()
        metadata = (
            {"Software": "loto-analyze"}
            if suffix == ".png"
            else {"Date": None, "Creator": "loto-analyze"}
        )
        figure.savefig(destination, dpi=150, metadata=metadata)
        plt.close(figure)
