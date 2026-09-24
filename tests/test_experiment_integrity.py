import csv
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

from loto.experiments import (
    REGISTRY_COLUMNS,
    ExperimentConfig,
    ExperimentOutcome,
    build_data_manifest,
    current_commit,
    run_experiment,
)


def _config(identifier: str = "EXP-INTEGRITY") -> ExperimentConfig:
    return ExperimentConfig(
        id=identifier,
        hypothesis="Hypothèse testable",
        split="train/test",
        features="fréquences",
        model="modèle",
        baseline="uniforme",
        metric="Brier",
        threshold="delta > 0",
        seed=7,
    )


def _head(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(tmp_path, clean_git_repo, command, **kwargs):
    data = tmp_path / "draws.csv"
    data.write_text("input\n", encoding="utf-8")
    registry = kwargs.pop("registry_path", tmp_path / "registry.csv")
    return run_experiment(
        _config(),
        command,
        data_paths=[data],
        registry_path=registry,
        repo_path=clean_git_repo,
        **kwargs,
    )


def test_git_root_resolution_does_not_depend_on_cwd(tmp_path, monkeypatch):
    expected = current_commit()
    monkeypatch.chdir(tmp_path)

    assert current_commit() == expected


def test_git_provenance_rejects_absent_repository(tmp_path):
    with pytest.raises(RuntimeError, match="Provenance Git indisponible"):
        current_commit(tmp_path)


@pytest.mark.parametrize("commit", ["abc", "g" * 40, "A" * 40, "0" * 40])
def test_invalid_commit_override_is_rejected_before_command(
    tmp_path, clean_git_repo, commit
):
    started = False

    def command():
        nonlocal started
        started = True
        return ExperimentOutcome("résultat", "retenue")

    expected = ValueError
    with pytest.raises(expected, match="commit"):
        _run(tmp_path, clean_git_repo, command, commit=commit)

    assert not started


@pytest.mark.parametrize("dirty_kind", ["tracked", "untracked"])
def test_dirty_or_untracked_repository_is_rejected_before_command(
    tmp_path, clean_git_repo, dirty_kind
):
    target = clean_git_repo / ("tracked.txt" if dirty_kind == "tracked" else "new.py")
    target.write_text("dirty\n", encoding="utf-8")
    started = False

    def command():
        nonlocal started
        started = True
        return ExperimentOutcome("résultat", "retenue")

    with pytest.raises(RuntimeError, match="modifications ou fichiers non suivis"):
        _run(tmp_path, clean_git_repo, command)

    assert not started


def test_registry_itself_is_exempt_from_cleanliness_check(tmp_path, clean_git_repo):
    registry = clean_git_repo / "registry.csv"

    outcome = _run(
        tmp_path,
        clean_git_repo,
        lambda: ExperimentOutcome("résultat", "retenue"),
        registry_path=registry,
    )

    assert outcome.result == "résultat"
    assert registry.is_file()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[]], "colonnes"),
        ([["id"] * (len(REGISTRY_COLUMNS) + 1)], "colonnes"),
        ([[""] + ["x"] * (len(REGISTRY_COLUMNS) - 1)], "Identifiant vide"),
        (
            [
                ["EXP-DUP"] + ["x"] * (len(REGISTRY_COLUMNS) - 1),
                ["EXP-DUP"] + ["y"] * (len(REGISTRY_COLUMNS) - 1),
            ],
            "dupliqué",
        ),
    ],
)
def test_malformed_existing_rows_are_rejected_before_command(
    tmp_path, clean_git_repo, rows, message
):
    registry = tmp_path / "registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(REGISTRY_COLUMNS)
        writer.writerows(rows)
    started = False

    def command():
        nonlocal started
        started = True
        return ExperimentOutcome("résultat", "retenue")

    with pytest.raises(ValueError, match=message):
        _run(
            tmp_path,
            clean_git_repo,
            command,
            registry_path=registry,
        )

    assert not started


@pytest.mark.parametrize(
    ("outcome", "error", "message"),
    [
        (cast(Any, {"result": "ok"}), TypeError, "ExperimentOutcome"),
        (ExperimentOutcome("", "retenue"), ValueError, "résultat"),
        (ExperimentOutcome("ok", "  "), ValueError, "décision"),
        (ExperimentOutcome("ok", "retenue", cast(Any, [])), TypeError, "artefacts"),
        (
            ExperimentOutcome("ok", "retenue", cast(Any, ("ok", 2))),
            TypeError,
            "artefacts",
        ),
    ],
)
def test_invalid_outcome_is_recorded_as_error_then_raised(
    tmp_path, clean_git_repo, outcome, error, message
):
    registry = tmp_path / "registry.csv"

    with pytest.raises(error, match=message):
        _run(tmp_path, clean_git_repo, lambda: outcome, registry_path=registry)

    with registry.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["décision"] == "erreur"
    assert message in row["résultat"]


def test_deleted_input_is_recorded_as_changed_data(tmp_path, clean_git_repo):
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    registry = tmp_path / "registry.csv"

    def command():
        data.unlink()
        return ExperimentOutcome("ok", "retenue")

    with pytest.raises(RuntimeError, match="données ont changé"):
        run_experiment(
            _config(),
            command,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    with registry.open(encoding="utf-8", newline="") as handle:
        assert next(csv.DictReader(handle))["décision"] == "erreur"


def test_resolved_duplicate_data_paths_are_rejected(tmp_path):
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")

    with pytest.raises(ValueError, match="uniques"):
        build_data_manifest([data, data.parent / "." / data.name])


def test_valid_commit_override_is_the_existing_full_hash(tmp_path, clean_git_repo):
    outcome = _run(
        tmp_path,
        clean_git_repo,
        lambda: ExperimentOutcome("ok", "retenue"),
        commit=_head(clean_git_repo),
    )

    assert outcome.result == "ok"


def test_run_rejects_an_existing_commit_other_than_head(tmp_path, clean_git_repo):
    previous = _head(clean_git_repo)
    (clean_git_repo / "tracked.txt").write_text("second\n", encoding="utf-8")
    subprocess.run(["git", "-C", clean_git_repo, "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            clean_git_repo,
            "-c",
            "user.name=Loto Tests",
            "-c",
            "user.email=loto-tests@example.invalid",
            "commit",
            "-q",
            "-m",
            "second",
        ],
        check=True,
    )

    with pytest.raises(ValueError, match="HEAD exécuté"):
        _run(
            tmp_path,
            clean_git_repo,
            lambda: ExperimentOutcome("ok", "retenue"),
            commit=previous,
        )
