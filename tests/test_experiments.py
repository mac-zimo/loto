import csv
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import loto.experiments as experiments
from loto.experiments import (
    REGISTRY_COLUMNS,
    ExperimentConfig,
    ExperimentOutcome,
    append_experiment,
    build_data_manifest,
    run_experiment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def config(identifier: str = "EXP-001") -> ExperimentConfig:
    return ExperimentConfig(
        id=identifier,
        hypothesis="Les fréquences battent l'uniforme.",
        split="train:2020-2023/test:2024",
        features="fréquences cumulées",
        model="fréquence lissée",
        baseline="uniforme",
        metric="Brier",
        threshold="amélioration >= 0.01",
        seed=42,
    )


def test_versioned_registry_has_the_exact_header_and_no_entries():
    with (PROJECT_ROOT / "experiments/registry.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.reader(handle))

    assert rows == [list(REGISTRY_COLUMNS)]


def test_data_manifest_contains_each_file_hash_and_a_stable_dataset_hash(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    first.write_bytes(b"a\n1\n")
    second.write_bytes(b"b\n2\n")

    manifest = json.loads(build_data_manifest([second, first]))

    assert [item["path"] for item in manifest["files"]] == ["a.csv", "b.csv"]
    assert manifest["files"][0]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    assert len(manifest["sha256"]) == 64
    assert manifest == json.loads(build_data_manifest([first, second]))


def test_append_experiment_writes_exact_columns_and_rejects_duplicate_id(
    tmp_path, clean_git_repo
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("draw\n1\n", encoding="utf-8")
    outcome = ExperimentOutcome("0.19", "rejetée", ("reports/EXP-001.json",))

    append_experiment(
        config(),
        outcome,
        data_paths=[data],
        registry_path=registry,
        repo_path=clean_git_repo,
    )

    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == REGISTRY_COLUMNS
    assert len(rows) == 1
    assert rows[0]["id"] == "EXP-001"
    assert len(rows[0]["commit"]) == 40
    assert json.loads(rows[0]["données"])["files"][0]["sha256"] == hashlib.sha256(
        data.read_bytes()
    ).hexdigest()
    assert json.loads(rows[0]["artefacts"]) == ["reports/EXP-001.json"]

    with pytest.raises(ValueError, match="EXP-001"):
        append_experiment(
            config(),
            outcome,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    with registry.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def _reservation_path(registry: Path, identifier: str) -> Path:
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return registry.with_name(f"{registry.name}.reservations") / f"{digest}.json"


def test_write_failure_after_command_leaves_reservation_and_prevents_replay(
    tmp_path, clean_git_repo, monkeypatch
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    published = tmp_path / "published"
    commands = 0

    def command():
        nonlocal commands
        commands += 1
        published.mkdir()
        return ExperimentOutcome("result", "retenue")

    monkeypatch.setattr(
        experiments,
        "_write_row",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected write failure")),
    )

    with pytest.raises(OSError, match="injected write failure"):
        run_experiment(
            config(), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )

    reservation = _reservation_path(registry, "EXP-001")
    assert reservation.is_file()
    payload = json.loads(reservation.read_text(encoding="utf-8"))
    assert payload["id"] == "EXP-001"
    assert payload["commit"]
    assert payload["manifest"] == json.loads(build_data_manifest([data]))

    with pytest.raises(ValueError, match="EXP-001.*réserv"):
        run_experiment(
            config(), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )
    assert commands == 1


def test_partial_registry_write_leaves_reservation_and_prevents_replay(
    tmp_path, clean_git_repo, monkeypatch
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    commands = 0

    def command():
        nonlocal commands
        commands += 1
        return ExperimentOutcome("result", "retenue")

    def partial_write(handle, row):
        handle.seek(0, os.SEEK_END)
        handle.write("EXP-001,partial")
        handle.flush()
        os.fsync(handle.fileno())
        raise OSError("injected partial write")

    monkeypatch.setattr(experiments, "_write_row", partial_write)
    with pytest.raises(OSError, match="partial write"):
        run_experiment(
            config(), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )

    assert _reservation_path(registry, "EXP-001").is_file()
    with pytest.raises(ValueError):
        run_experiment(
            config(), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )
    assert commands == 1


def test_registry_failure_does_not_mask_primary_command_error(
    tmp_path, clean_git_repo, monkeypatch
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    monkeypatch.setattr(
        experiments,
        "_write_row",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("registry unavailable")),
    )

    with pytest.raises(RuntimeError, match="primary failure") as raised:
        run_experiment(
            config(),
            lambda: (_ for _ in ()).throw(RuntimeError("primary failure")),
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    assert _reservation_path(registry, "EXP-001").is_file()
    assert any(
        "Aucune ligne d'erreur durable n'a pu être confirmée" in note
        and "réservation durable" in note
        for note in raised.value.__notes__
    )


@pytest.mark.parametrize("fails", [False, True])
def test_durable_row_removes_reservation_and_csv_consumes_id(
    tmp_path, clean_git_repo, fails
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")

    def command():
        if fails:
            raise RuntimeError("boom")
        return ExperimentOutcome("result", "retenue")

    if fails:
        with pytest.raises(RuntimeError, match="boom"):
            run_experiment(
                config(), command, data_paths=[data], registry_path=registry,
                repo_path=clean_git_repo,
            )
    else:
        run_experiment(
            config(), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )

    assert not _reservation_path(registry, "EXP-001").exists()
    with registry.open(encoding="utf-8", newline="") as handle:
        assert [row["id"] for row in csv.DictReader(handle)] == ["EXP-001"]


def test_success_cleanup_failure_is_non_fatal_and_leaves_inspectable_reservation(
    tmp_path, clean_git_repo, monkeypatch
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    expected = ExperimentOutcome("result", "retenue")
    commands = 0

    def command():
        nonlocal commands
        commands += 1
        return expected

    def fail_cleanup(reservation):
        raise OSError(f"injected cleanup failure for {reservation}")

    monkeypatch.setattr(experiments, "_remove_reservation", fail_cleanup)

    with pytest.warns(
        RuntimeWarning,
        match="success row durable; residual reservation requires inspection",
    ) as captured:
        outcome = run_experiment(
            config(), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )

    reservation = _reservation_path(registry, "EXP-001")
    assert outcome is expected
    assert reservation.is_file()
    assert str(reservation) in str(captured[0].message)
    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["id"] == "EXP-001"
    assert rows[0]["décision"] == "retenue"

    with pytest.raises(ValueError, match="EXP-001.*existe déjà"):
        run_experiment(
            config(), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )
    assert commands == 1


def test_success_cleanup_normally_removes_reservation_without_warning(
    tmp_path, clean_git_repo, recwarn
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")

    outcome = run_experiment(
        config(),
        lambda: ExperimentOutcome("result", "retenue"),
        data_paths=[data],
        registry_path=registry,
        repo_path=clean_git_repo,
    )

    assert outcome.result == "result"
    assert not recwarn
    assert not _reservation_path(registry, "EXP-001").exists()


def test_reservations_are_per_id_and_invalid_content_fails_closed(
    tmp_path, clean_git_repo
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    first = _reservation_path(registry, "EXP-A")
    first.parent.mkdir()
    first.write_text("not-json", encoding="utf-8")
    started = False

    def command():
        nonlocal started
        started = True
        return ExperimentOutcome("result", "retenue")

    with pytest.raises(RuntimeError, match="réservation.*invalide"):
        run_experiment(
            config("EXP-A"), command, data_paths=[data], registry_path=registry,
            repo_path=clean_git_repo,
        )
    assert not started

    outcome = run_experiment(
        config("EXP-B"), command, data_paths=[data], registry_path=registry,
        repo_path=clean_git_repo,
    )
    assert outcome.result == "result"
    assert first.read_text(encoding="utf-8") == "not-json"


def test_validate_before_record_controls_success_commit(
    tmp_path, clean_git_repo
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")

    with pytest.raises(RuntimeError, match="publication substituted"):
        run_experiment(
            config(),
            lambda: ExperimentOutcome("result", "retenue"),
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
            validate_before_record=lambda outcome: (_ for _ in ()).throw(
                RuntimeError("publication substituted")
            ),
        )

    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["décision"] == "erreur"
    assert "publication substituted" in rows[0]["résultat"]
    assert not _reservation_path(registry, "EXP-001").exists()


def test_no_fallible_validation_occurs_after_success_row(
    tmp_path, clean_git_repo, monkeypatch
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    events = []
    substituted = False
    real_write_row = experiments._write_row

    def validate(outcome):
        if substituted:
            raise RuntimeError("post-record validation must not run")
        events.append("validate")

    def write_row(handle, row):
        nonlocal substituted
        real_write_row(handle, row)
        substituted = True
        events.append("success-row")

    monkeypatch.setattr(experiments, "_write_row", write_row)
    outcome = run_experiment(
        config(),
        lambda: ExperimentOutcome("result", "retenue"),
        data_paths=[data],
        registry_path=registry,
        repo_path=clean_git_repo,
        validate_before_record=validate,
    )

    assert outcome.result == "result"
    assert substituted
    assert events == ["validate", "success-row"]


def test_append_experiment_separates_row_after_missing_final_newline(
    tmp_path, clean_git_repo
):
    registry = tmp_path / "registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(REGISTRY_COLUMNS)
        writer.writerow(["EXP-000", *("existing" for _ in REGISTRY_COLUMNS[1:])])
    contents = registry.read_bytes()
    assert contents.endswith(b"\n")
    registry.write_bytes(contents[:-1])
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")

    append_experiment(
        config(),
        ExperimentOutcome("result", "retenue"),
        data_paths=[data],
        registry_path=registry,
        repo_path=clean_git_repo,
    )

    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == ["EXP-000", "EXP-001"]


def test_run_experiment_records_error_if_inputs_change(tmp_path, clean_git_repo):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("before", encoding="utf-8")
    def experiment() -> ExperimentOutcome:
        data.write_text("after", encoding="utf-8")
        return ExperimentOutcome("gain=0.02", "retenue")

    with pytest.raises(RuntimeError, match="données ont changé"):
        run_experiment(
            config(),
            experiment,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    with registry.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert "données ont changé" in row["résultat"]
    assert row["décision"] == "erreur"


def test_run_experiment_records_failure_then_reraises(tmp_path, clean_git_repo):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")

    def failing_experiment() -> ExperimentOutcome:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_experiment(
            config(),
            failing_experiment,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    with registry.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["résultat"] == "RuntimeError: boom"
    assert row["décision"] == "erreur"


def test_run_experiment_does_not_start_command_for_duplicate_id(
    tmp_path, clean_git_repo
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    append_experiment(
        config(),
        ExperimentOutcome("initial", "retenue"),
        data_paths=[data],
        registry_path=registry,
        repo_path=clean_git_repo,
    )
    started = False

    def experiment() -> ExperimentOutcome:
        nonlocal started
        started = True
        return ExperimentOutcome("duplicate", "retenue")

    with pytest.raises(ValueError, match="EXP-001"):
        run_experiment(
            config(),
            experiment,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    assert not started


def test_run_experiment_does_not_start_command_for_invalid_header(
    tmp_path, clean_git_repo
):
    registry = tmp_path / "registry.csv"
    registry.write_text("wrong,header\n", encoding="utf-8")
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    started = False

    def experiment() -> ExperimentOutcome:
        nonlocal started
        started = True
        return ExperimentOutcome("result", "retenue")

    with pytest.raises(ValueError, match="En-tête invalide"):
        run_experiment(
            config(),
            experiment,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    assert not started


def test_run_experiment_records_system_exit_then_reraises(tmp_path, clean_git_repo):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")

    def exiting_experiment() -> ExperimentOutcome:
        raise SystemExit("stop")

    with pytest.raises(SystemExit, match="stop"):
        run_experiment(
            config(),
            exiting_experiment,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    with registry.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["résultat"] == "SystemExit: stop"
    assert row["décision"] == "erreur"


def test_concurrent_runs_with_same_id_start_only_one_command(
    tmp_path, clean_git_repo
):
    registry = tmp_path / "registry.csv"
    data = tmp_path / "draws.csv"
    data.write_text("input", encoding="utf-8")
    ready = threading.Barrier(2)
    started = 0
    counter_lock = threading.Lock()

    def run() -> ExperimentOutcome:
        nonlocal started
        ready.wait()

        def experiment() -> ExperimentOutcome:
            nonlocal started
            with counter_lock:
                started += 1
            return ExperimentOutcome("result", "retenue")

        return run_experiment(
            config(),
            experiment,
            data_paths=[data],
            registry_path=registry,
            repo_path=clean_git_repo,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run) for _ in range(2)]
        outcomes = []
        errors = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except ValueError as error:
                errors.append(error)

    assert started == 1
    assert len(outcomes) == 1
    assert len(errors) == 1
    assert "EXP-001" in str(errors[0])
    with registry.open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1
