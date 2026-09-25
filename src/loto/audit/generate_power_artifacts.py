"""Reproducible command for the pre-registered task 1.3 artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from loto.audit.power import estimate_power_curves, export_power_data, plot_power_curves

DEFAULT_ARTIFACT_DIR = Path("artifacts/task-1.3-power")


def generate_default_artifacts(output_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> None:
    """Generate the CSV data, full JSON manifest, and PNG curve."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report = estimate_power_curves()
    export_power_data(report, destination / "power_curves_2811.csv")
    export_power_data(report, destination / "power_manifest_2811.json")
    plot_power_curves(report, destination / "power_curves_2811.png")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate deterministic task 1.3 power-analysis artifacts."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help=f"destination directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    arguments = parser.parse_args(argv)
    generate_default_artifacts(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
