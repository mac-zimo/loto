import csv
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

import loto.cli
from loto.cli import parse_args


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_parse_args_accepts_pipeline_flags():
    args = parse_args([
        "--skip-plots",
        "--reset",
        "--data-dir",
        "csv",
        "--db-path",
        "state/loto.db",
        "--output-dir",
        "reports",
    ])

    assert args.skip_plots is True
    assert args.reset is True
    assert args.data_dir == "csv"
    assert args.db_path == "state/loto.db"
    assert args.output_dir == "reports"
    assert args.include_legacy_models is False
    assert args.include_legacy_backtests is False


def test_root_wrapper_delegates_to_cli_main(monkeypatch):
    calls = []

    def fake_main():
        calls.append(())
        return 73

    monkeypatch.setattr(loto.cli, "main", fake_main)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(PROJECT_ROOT / "run_pipeline.py"), run_name="__main__")

    assert exc_info.value.code == 73
    assert calls == [()]


def test_package_module_delegates_to_cli_main(monkeypatch):
    calls = []

    def fake_main():
        calls.append(())
        return 74

    monkeypatch.setattr(loto.cli, "main", fake_main)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("loto.__main__", run_name="__main__")

    assert exc_info.value.code == 74
    assert calls == [()]


def test_importing_cli_has_no_filesystem_or_matplotlib_side_effects(tmp_path):
    code = (
        "import sys; import loto.cli; "
        "assert 'matplotlib' not in sys.modules; "
        "assert not __import__('pathlib').Path('data').exists(); "
        "assert not __import__('pathlib').Path('output').exists()"
    )

    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_report_preserves_lazy_plotting_exports(tmp_path):
    code = (
        "import sys; import loto.report as report; "
        "assert 'matplotlib.pyplot' not in sys.modules; "
        "assert all(callable(getattr(report, name)) for name in ("
        "'ensure_output_dir', "
        "'plot_frequency_heatmap', 'plot_correlation_matrix', 'plot_top_pairs', "
        "'plot_strategy_results', 'plot_chance_results', 'plot_distribution_stats')); "
        "path = report.ensure_output_dir(__import__('pathlib').Path('reports')); "
        "assert path.is_dir(); "
        "assert 'matplotlib.pyplot' not in sys.modules"
    )

    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_cli_runs_with_explicit_isolated_paths(tmp_path):
    data_dir = tmp_path / "csv"
    db_path = tmp_path / "state" / "loto.db"
    output_dir = tmp_path / "reports"
    data_dir.mkdir()
    headers = [
        "annee_numero_de_tirage",
        "jour_de_tirage",
        "date_de_tirage",
        "date_de_forclusion",
        "boule_1",
        "boule_2",
        "boule_3",
        "boule_4",
        "boule_5",
        "numero_chance",
        "combinaison_gagnante_en_ordre_croissant",
    ]
    for index, filename in enumerate(
        ["nouveau_loto.csv", "loto2017.csv", "loto_201902.csv", "loto_201911.csv"],
        start=1,
    ):
        with (data_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter=";")
            writer.writerow(headers)
            writer.writerow([
                f"2026{index:03}",
                "lundi",
                f"0{index}/01/2026",
                "01/03/2026",
                "1",
                "2",
                "3",
                "4",
                "5",
                str(index),
                "1-2-3-4-5",
            ])

    assert loto.cli.main([
        "--skip-plots",
        "--data-dir",
        str(data_dir),
        "--db-path",
        str(db_path),
        "--output-dir",
        str(output_dir),
    ]) == 0

    assert db_path.is_file()
    assert (output_dir / "report.txt").is_file()