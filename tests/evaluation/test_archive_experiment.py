from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import sqlite3
import struct
import subprocess

import pytest

import loto.evaluation._archive_artifact_publication as publication
import loto.evaluation.archive_experiment as archive
import loto.experiments as experiments
from loto.data_schema import SourceSchema, schema_contract
from loto.evaluation.walk_forward import DrawObservation


def _draws(count: int = 64) -> tuple[DrawObservation, ...]:
    return tuple(
        DrawObservation(
            draw_date=(
                date(2013, 1, 1) + timedelta(days=index)
                if index < 60
                else date(2014, 1, 1) + timedelta(days=index - 60)
            ),
            original_index=index,
            numbers=tuple(((index * 5 + offset) % 49) + 1 for offset in range(5)),
        )
        for index in range(count)
    )


def _source_row(schema: SourceSchema, draw_id: str, draw_date: date, offset: int):
    row = {column: "" for column in schema_contract(schema).expected_columns}
    numbers = tuple(((offset * 5 + item) % 49) + 1 for item in range(5))
    row.update(
        {
            "annee_numero_de_tirage": draw_id,
            "jour_de_tirage": "TEST",
            "date_de_tirage": draw_date.strftime("%d/%m/%Y"),
            "date_de_forclusion": (draw_date + timedelta(days=60)).strftime("%d/%m/%Y"),
            **{f"boule_{index}": str(number) for index, number in enumerate(numbers, 1)},
            "numero_chance": "1",
            "combinaison_gagnante_en_ordre_croissant": "-".join(map(str, numbers)) + "+1",
            "devise": "eur",
        }
    )
    return row


def _write_source(path: Path, schema: SourceSchema, rows: list[dict[str, str]]) -> None:
    columns = schema_contract(schema).expected_columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_archives(root: Path) -> tuple[Path, ...]:
    cases = (
        (archive.ARCHIVE_FILENAMES[0], SourceSchema.OCT_2008_MAR_2017, date(2013, 1, 2)),
        (archive.ARCHIVE_FILENAMES[1], SourceSchema.MAR_2017_FEB_2019, date(2017, 3, 6)),
        (archive.ARCHIVE_FILENAMES[2], SourceSchema.FEB_2019_NOV_2019, date(2019, 2, 27)),
        (archive.ARCHIVE_FILENAMES[3], SourceSchema.NOV_2019_ONWARD, date(2020, 1, 1)),
    )
    paths = []
    for index, (filename, schema, draw_date) in enumerate(cases):
        path = root / filename
        _write_source(path, schema, [_source_row(schema, f"D{index}", draw_date, index)])
        paths.append(path)
    return tuple(paths)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(root: Path) -> str:
    _git(root, "init", "-q")
    _git(root, "add", ".")
    subprocess.run(
        ["git", "-c", "user.name=Loto Tests", "-c", "user.email=loto@example.invalid", "commit", "-qm", "fixture"],
        cwd=root,
        check=True,
    )
    return _git(root, "rev-parse", "HEAD")


def test_preregistration_is_immutable_exact_and_uses_model_identifiers():
    spec = archive.PREREGISTRATION
    candidates = archive.preregistered_candidates()

    assert spec.experiment_id == "EXP-2026-09-25-TASK-2.3A-ARCHIVE-WF-001"
    assert spec.seed == 20260925
    assert spec.archive_filenames == (
        "nouveau_loto.csv",
        "loto2017.csv",
        "loto_201902.csv",
        "loto_201911.csv",
    )
    assert (spec.evaluation_start_year, spec.evaluation_end_year) == (2014, 2025)
    assert spec.cutoff == date(2014, 1, 1)
    assert spec.excluded_year == 2026
    assert spec.dataset_contract == archive.DatasetContract(
        2811, date(2008, 10, 6), date(2026, 9, 21)
    )
    assert spec.max_train_size is None
    assert len(candidates) == 10
    assert len({candidate.identifier for candidate in candidates}) == 10
    assert tuple(candidate.identifier for candidate in candidates) == spec.candidate_ids
    assert [candidate.kind for candidate in candidates] == [
        "baseline", "baseline", "baseline", "baseline", "baseline", "baseline",
        "baseline", "family", "family", "family",
    ]
    parameters = {candidate.label: dict(candidate.parameters) for candidate in candidates}
    assert parameters["uniform"] == {}
    assert parameters["cumulative"] == {"alpha": 1.0}
    assert [parameters[f"rolling_{window}"] for window in (5, 10, 25, 50)] == [
        {"window_size": window, "alpha": 1.0} for window in (5, 10, 25, 50)
    ]
    assert parameters["constrained_random"] == {}
    assert parameters["dirichlet"]["concentration"] == 1.0
    assert parameters["dirichlet"]["minimum_history"] == 1
    assert parameters["logistic"] == archive._configured_parameters(
        archive.LogisticRegressionConfig()
    )
    assert parameters["hazard"] == archive._configured_parameters(
        archive.DiscreteTimeHazardConfig()
    )
    with pytest.raises(FrozenInstanceError):
        spec.seed = 1


def test_experiment_config_is_exact_and_derived_from_preregistration():
    config = archive.experiment_config()
    assert config.id == archive.EXPERIMENT_ID
    assert config.seed == archive.SEED
    assert config.split == "train:<2014-01-01; test:calendar-years-2014..2025; expanding; 2026-excluded"
    assert config.metric == ",".join(archive.METRIC_NAMES)
    assert config.model == ",".join(archive.PREREGISTRATION.candidate_ids)
    assert "at least 3 complete calendar years" in config.threshold
    assert "holdout 2026 untouched" in config.threshold


def test_synthetic_canonical_loading_is_strict_sorted_and_has_stable_indices(tmp_path, monkeypatch):
    paths = _synthetic_archives(tmp_path)
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SQLite accessed")))
    contract = archive.DatasetContract(4, date(2013, 1, 2), date(2020, 1, 1))

    loaded = archive.load_archive_draws(tuple(reversed(paths)), contract=contract)
    repeated = archive.load_archive_draws(paths, contract=contract)

    assert loaded == repeated
    assert [draw.draw_date for draw in loaded] == sorted(draw.draw_date for draw in loaded)
    assert [draw.original_index for draw in loaded] == list(range(4))
    assert len({draw.original_index for draw in loaded}) == 4

    duplicate = tmp_path / archive.ARCHIVE_FILENAMES[3]
    schema = SourceSchema.NOV_2019_ONWARD
    _write_source(duplicate, schema, [_source_row(schema, "D0", date(2020, 1, 1), 3)])
    with pytest.raises(Exception, match="duplicate canonical draw ID"):
        archive.load_archive_draws(paths, contract=contract)


def test_dataset_contract_rejects_count_and_date_divergence(tmp_path):
    paths = _synthetic_archives(tmp_path)
    with pytest.raises(ValueError, match="dataset contract"):
        archive.load_archive_draws(paths, contract=archive.OFFICIAL_DATASET_CONTRACT)

def test_full_contract_counts_2026_without_parsing_or_exposing_its_numbers(tmp_path, monkeypatch):
    paths = _synthetic_archives(tmp_path)
    early_schema = SourceSchema.OCT_2008_MAR_2017
    _write_source(paths[0], early_schema, [
        _source_row(early_schema, "D0", date(2013, 1, 2), 0),
        _source_row(early_schema, "D2014", date(2014, 1, 2), 5),
    ])
    schema = SourceSchema.NOV_2019_ONWARD
    withheld = _source_row(schema, "WITHHELD", date(2026, 9, 21), 4)
    withheld["boule_1"] = "not a number"
    _write_source(paths[3], schema, [
        _source_row(schema, "D3", date(2020, 1, 1), 3), withheld,
    ])
    import loto.data_source as data_source
    real_parse = data_source.parse_canonical_draw
    parsed_dates = []

    def checked_parse(row, **kwargs):
        parsed_dates.append(row["date_de_tirage"])
        assert row["annee_numero_de_tirage"] != "WITHHELD"
        return real_parse(row, **kwargs)

    monkeypatch.setattr(data_source, "parse_canonical_draw", checked_parse)
    contract = archive.DatasetContract(6, date(2013, 1, 2), date(2026, 9, 21))
    draws = archive.load_archive_draws(paths, contract=contract)
    assert len(draws) == 5
    assert len(parsed_dates) == 5
    assert all(draw.draw_date.year < 2026 for draw in draws)
    assert [draw.original_index for draw in draws] == list(range(5))
    protocol = archive.build_protocol(
        draws, evaluation_start_year=2014, evaluation_end_year=2014
    )
    assert protocol.windows[0].test_end < len(draws)
    assert all(draw.draw_date.year == 2014 for draw in archive.target_draws(draws, protocol.windows))
    with pytest.raises(ValueError, match="dataset contract"):
        archive.load_archive_draws(paths, contract=replace(contract, draw_count=5))


def test_protocol_has_explicit_annual_windows_and_strictly_excludes_2026():
    draws = tuple(
        DrawObservation(date(year, 1, 2), year, (1, 2, 3, 4, 5))
        for year in range(2012, 2026)
    )
    protocol = archive.build_protocol(draws)

    assert protocol.walk_forward.initial_train_size == 2
    assert protocol.walk_forward.retrain_policy.value == "each_draw"
    assert protocol.walk_forward.seed == 20260925
    assert protocol.max_train_size is None
    assert len(protocol.windows) == 12
    targets = archive.target_draws(draws, protocol.windows)
    assert {draw.draw_date.year for draw in targets} == set(range(2014, 2026))
    assert max(draw.draw_date for draw in targets) <= date(2025, 12, 31)
    assert all(draw.draw_date.year != 2026 for draw in targets)


def test_protocol_rejects_any_2026_target_even_if_window_is_forged():
    draws = tuple(
        DrawObservation(date(year, 1, 2), year, (1, 2, 3, 4, 5))
        for year in range(2012, 2027)
    )
    protocol = archive.build_protocol(draws[:-1])
    forged = replace(protocol, windows=(*protocol.windows, archive.EvaluationWindow(0, 14, 14, 15)))
    with pytest.raises(ValueError, match="2026"):
        archive.validate_target_period(draws, forged.windows)


def test_all_candidates_run_on_synthetic_data_with_stable_aggregate_and_year_order():
    draws = _draws()
    protocol = archive.build_protocol(draws, evaluation_start_year=2014, evaluation_end_year=2014)

    run = archive.evaluate_candidates(draws, protocol)

    assert tuple(run.prediction_summaries) == archive.PREREGISTRATION.candidate_ids
    assert len(run.metric_rows) == 20
    assert [row.period for row in run.metric_rows[::2]] == ["aggregate"] * 10
    assert [row.period for row in run.metric_rows[1::2]] == ["2014"] * 10
    assert all(row.prediction_count > 0 for row in run.metric_rows)
    assert all(row.first_target_date.year == 2014 for row in run.metric_rows)
    assert all(row.last_target_date.year == 2014 for row in run.metric_rows)


def test_required_scientific_equivalences_are_exact_on_predictions_and_metrics():
    draws = _draws()
    protocol = archive.build_protocol(draws, evaluation_start_year=2014, evaluation_end_year=2014)
    run = archive.evaluate_candidates(draws, protocol)
    ids = {candidate.label: candidate.identifier for candidate in archive.preregistered_candidates()}

    cumulative = run.prediction_summaries[ids["cumulative"]]
    dirichlet = run.prediction_summaries[ids["dirichlet"]]
    uniform = run.prediction_summaries[ids["uniform"]]
    constrained = run.prediction_summaries[ids["constrained_random"]]
    assert cumulative.probability_digest == dirichlet.probability_digest
    assert [
        replace(row, candidate_id="same")
        for row in archive.rows_for(run, ids["cumulative"])
    ] == [
        replace(row, candidate_id="same")
        for row in archive.rows_for(run, ids["dirichlet"])
    ]
    assert uniform.probability_digest == constrained.probability_digest
    for uniform_row, constrained_row in zip(
        archive.rows_for(run, ids["uniform"]),
        archive.rows_for(run, ids["constrained_random"]),
        strict=True,
    ):
        for metric in archive.PROBABILISTIC_METRIC_NAMES:
            assert getattr(uniform_row, metric) == getattr(constrained_row, metric)


def _representative_records(
    draws, protocol, probabilities=None
) -> tuple[archive.PredictionRecord, ...]:
    marginal = probabilities or (5.0 / 49.0,) * 49
    first = draws[0]
    return tuple(
        archive.PredictionRecord(
            target_date=target.draw_date,
            target_original_index=target.original_index,
            target_numbers=target.numbers,
            probabilities=marginal,
            history_dates=(first.draw_date,),
            history_original_indices=(first.original_index,),
            fitted_through_date=first.draw_date,
            fitted_through_original_index=first.original_index,
            window_index=window_index,
            retrained=True,
            selected_numbers=(1, 2, 3, 4, 5),
        )
        for window_index, window in enumerate(protocol.windows)
        for target in draws[window.test_start : window.test_end]
    )


def test_prediction_series_summary_is_exact_and_uses_canonical_binary_digest():
    draws = _draws()
    protocol = archive.build_protocol(
        draws, evaluation_start_year=2014, evaluation_end_year=2014
    )
    records = _representative_records(draws, protocol)

    summary = archive._prediction_series_summary("candidate", records)

    probability_hash = hashlib.sha256(b"loto-probability-series-v1\0")
    selected_hash = hashlib.sha256(b"loto-selected-number-series-v1\0")
    for order, record in enumerate(records):
        identity = struct.pack(
            ">QHBBq", order, record.target_date.year, record.target_date.month,
            record.target_date.day, record.target_original_index,
        )
        probability_hash.update(identity)
        probability_hash.update(struct.pack(">H", len(record.probabilities)))
        for probability in record.probabilities:
            probability_hash.update(struct.pack(">d", probability))
        selected_hash.update(identity)
        assert record.selected_numbers is not None
        selected_hash.update(b"\x01")
        selected_hash.update(struct.pack(">H", len(record.selected_numbers)))
        for number in record.selected_numbers:
            selected_hash.update(struct.pack(">H", number))
    probability_hash.update(struct.pack(">Q", len(records)))
    selected_hash.update(struct.pack(">Q", len(records)))

    assert summary == archive.PredictionSeriesSummary(
        candidate_id="candidate",
        prediction_count=len(records),
        first_target_date=records[0].target_date,
        last_target_date=records[-1].target_date,
        targets_in_2026=0,
        probability_digest=probability_hash.hexdigest(),
        selected_number_digest=selected_hash.hexdigest(),
    )
    changed = list(records[0].probabilities)
    changed[0] += 1e-12
    changed[1] -= 1e-12
    changed_records = (replace(records[0], probabilities=tuple(changed)), *records[1:])
    assert (
        archive._prediction_series_summary("candidate", changed_records).probability_digest
        != summary.probability_digest
    )


def test_probability_equivalence_detects_one_changed_probability(monkeypatch):
    draws = _draws()
    protocol = archive.build_protocol(
        draws, evaluation_start_year=2014, evaluation_end_year=2014
    )
    calls = 0
    equal = (5.0 / 49.0,) * 49
    changed = list(equal)
    changed[0] += 1e-12
    changed[1] -= 1e-12

    def records_for_next_candidate(*args, **kwargs):
        nonlocal calls
        calls += 1
        probabilities = tuple(changed) if calls == 8 else equal
        return _representative_records(draws, protocol, probabilities)

    monkeypatch.setattr(archive, "run_walk_forward", records_for_next_candidate)

    with pytest.raises(
        AssertionError,
        match="cumulative and uniform-prior Dirichlet predictions differ",
    ):
        archive.evaluate_candidates(draws, protocol)

    assert calls == 10


def test_candidate_metric_rows_are_unchanged_from_walk_forward_metrics():
    draws = _draws()
    protocol = archive.build_protocol(
        draws, evaluation_start_year=2014, evaluation_end_year=2014
    )
    candidate = archive.preregistered_candidates()[0]
    records = archive.run_walk_forward(
        draws, protocol.windows, candidate.callbacks, protocol.walk_forward
    )

    summary, rows = archive._evaluate_one_candidate(draws, protocol, candidate)
    aggregate = archive.evaluate_predictions(records)

    assert summary.prediction_count == len(records)
    assert rows[0] == archive.MetricRow.from_metrics(
        candidate.identifier, "aggregate", records, aggregate
    )
    assert rows[1] == archive.MetricRow.from_metrics(
        candidate.identifier, "2014", records, aggregate
    )


def _metric_row(candidate_id: str, period: str, log_loss: float, brier: float):
    return archive.MetricRow(
        candidate_id=candidate_id,
        period=period,
        prediction_count=1,
        first_target_date=date(2014, 1, 1),
        last_target_date=date(2014, 1, 1),
        log_loss=log_loss,
        brier=brier,
        mean_matches=0.0,
        calibration_error=0.0,
        mean_true_number_rank=25.0,
        regret_vs_uniform=0.0,
    )


def test_screening_rule_is_strict_joint_and_never_promotes():
    candidate = "family"
    uniform = "uniform"
    cumulative = "cumulative"
    rows = [
        _metric_row(uniform, "aggregate", 1.0, 1.0),
        _metric_row(cumulative, "aggregate", 0.9, 0.9),
        _metric_row(candidate, "aggregate", 0.8, 0.8),
    ]
    for year in range(2014, 2017):
        rows.extend(
            (
                _metric_row(uniform, str(year), 1.0, 1.0),
                _metric_row(cumulative, str(year), 0.9, 0.9),
                _metric_row(candidate, str(year), 0.8, 0.8),
            )
        )
    result = archive.screen_candidate(
        tuple(rows), candidate, cumulative_id=cumulative, uniform_id=uniform
    )
    assert result.winning_period_count == 3
    assert result.screening_pass is True
    assert result.decision == "non_promu_significativite_et_holdout_requis"

    tied = tuple(replace(row, log_loss=0.9) if row.candidate_id == candidate else row for row in rows)
    rejected = archive.screen_candidate(tied, candidate, cumulative_id=cumulative, uniform_id=uniform)
    assert rejected.screening_pass is False
    assert rejected.decision == "rejete_au_screening_descriptif"


def test_artifact_serialization_is_deterministic_atomic_and_refuses_overwrite(tmp_path):
    draws = _draws()
    protocol = archive.build_protocol(draws, evaluation_start_year=2014, evaluation_end_year=2014)
    run = archive.evaluate_candidates(draws, protocol)
    repeated_run = archive.evaluate_candidates(draws, protocol)
    first = tmp_path / "first"
    second = tmp_path / "second"

    archive.write_artifacts(first, run, protocol, commit="a" * 40, dataset_manifest={"sha256": "x", "files": []})
    archive.write_artifacts(second, repeated_run, protocol, commit="a" * 40, dataset_manifest={"sha256": "x", "files": []})

    assert run.prediction_summaries == repeated_run.prediction_summaries
    for filename in archive.ARTIFACT_FILENAMES:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    result = json.loads((first / "result.json").read_bytes())
    assert result["integrity"]["prediction_series_by_candidate"] == {
        candidate_id: {
            "count": run.prediction_summaries[candidate_id].prediction_count,
            "probability_digest": run.prediction_summaries[
                candidate_id
            ].probability_digest,
            "selected_number_digest": run.prediction_summaries[
                candidate_id
            ].selected_number_digest,
        }
        for candidate_id in archive.PREREGISTRATION.candidate_ids
    }
    assert len(result["integrity"]["prediction_series_by_candidate"]) == 10
    assert set(path.name for path in first.iterdir()) == set(archive.ARTIFACT_FILENAMES)
    assert not list(tmp_path.glob(".*.staging"))
    with pytest.raises(FileExistsError):
        archive.write_artifacts(first, run, protocol, commit="a" * 40, dataset_manifest={"sha256": "x", "files": []})


def test_artifact_group_publication_cleans_staging_when_second_write_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    monkeypatch.setattr(archive, "_artifact_documents", lambda *args, **kwargs: documents)
    original_open = Path.open
    writes = 0

    def fail_second_write(path, mode="r", *args, **kwargs):
        nonlocal writes
        if mode == "wb" and path.parent.name.startswith(".official."):
            writes += 1
            if writes == 2:
                raise OSError("injected second-file failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_second_write)

    with pytest.raises(OSError, match="second-file"):
        archive.write_artifacts(
            output, None, None, commit="a" * 40, dataset_manifest={}
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".official.*.staging"))


def test_artifact_group_publication_cleans_staging_when_rename_fails(
    tmp_path, monkeypatch
):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    monkeypatch.setattr(archive, "_artifact_documents", lambda *args, **kwargs: documents)
    monkeypatch.setattr(
        publication,
        "_rename_noreplace",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected rename failure")),
    )

    with pytest.raises(OSError, match="rename"):
        archive.write_artifacts(
            output, None, None, commit="a" * 40, dataset_manifest={}
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".official.*.staging"))


@pytest.mark.parametrize("failure_point", ["parent_fsync", "verification"])
def test_post_rename_failure_exposes_authentic_receipt_and_retains_publication(
    tmp_path, monkeypatch, failure_point
):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    real_fsync_directory = publication.fsync_directory
    real_verify_publication = publication.verify_publication

    if failure_point == "parent_fsync":
        def injected_fsync(directory):
            if Path(directory) == output.parent and output.exists():
                raise OSError("injected parent fsync failure")
            return real_fsync_directory(directory)

        monkeypatch.setattr(publication, "fsync_directory", injected_fsync)
    else:
        monkeypatch.setattr(
            publication,
            "verify_publication",
            lambda receipt: (_ for _ in ()).throw(
                RuntimeError("injected publication verification failure")
            ),
        )

    with pytest.raises(publication.PublishedArtifactError) as raised:
        publication.publish_artifact_directory(
            output, documents, archive.ARTIFACT_FILENAMES
        )

    error = raised.value
    assert error.receipt.directory == output
    assert error.__cause__ is not None
    assert "recovery" in str(error).lower()
    assert output.is_dir()
    assert set(path.name for path in output.iterdir()) == set(archive.ARTIFACT_FILENAMES)
    assert not list(tmp_path.glob(".official.*.staging"))
    monkeypatch.setattr(publication, "fsync_directory", real_fsync_directory)
    monkeypatch.setattr(publication, "verify_publication", real_verify_publication)
    assert publication.verify_publication(error.receipt) == error.receipt.paths
    with pytest.raises((AttributeError, FrozenInstanceError)):
        error.receipt = error.receipt


@pytest.mark.parametrize("destination_kind", ["directory", "symlink"])
def test_atomic_publication_never_replaces_destination_created_at_rename(
    tmp_path, monkeypatch, destination_kind
):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    real_rename = publication._rename_noreplace

    def race(staging, destination):
        if destination_kind == "directory":
            destination.mkdir()
            (destination / "sentinel").write_bytes(b"foreign")
        else:
            target = tmp_path / "foreign-target"
            target.write_bytes(b"foreign")
            destination.symlink_to(target)
        return real_rename(staging, destination)

    monkeypatch.setattr(publication, "_rename_noreplace", race)

    with pytest.raises(FileExistsError):
        publication.publish_artifact_directory(
            output, documents, archive.ARTIFACT_FILENAMES
        )

    before = output.lstat()
    assert not list(tmp_path.glob(".official.*.staging"))
    if destination_kind == "directory":
        assert (output / "sentinel").read_bytes() == b"foreign"
    else:
        assert output.is_symlink()
        assert output.read_bytes() == b"foreign"
    after = output.lstat()
    assert (after.st_dev, after.st_ino) == (before.st_dev, before.st_ino)


def test_receipt_cannot_authenticate_destination_substituted_after_rename(
    tmp_path, monkeypatch
):
    output = tmp_path / "official"
    moved = tmp_path / "real-publication"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    real_rename = publication._rename_noreplace

    def substitute_after_rename(staging, destination):
        real_rename(staging, destination)
        os.rename(destination, moved)
        destination.mkdir()
        for filename, contents in documents.items():
            (destination / filename).write_bytes(contents)

    monkeypatch.setattr(publication, "_rename_noreplace", substitute_after_rename)

    with pytest.raises(RuntimeError, match="replaced|identity"):
        publication.publish_artifact_directory(
            output, documents, archive.ARTIFACT_FILENAMES
        )

    assert output.is_dir()
    assert moved.is_dir()
    assert set(path.name for path in output.iterdir()) == set(archive.ARTIFACT_FILENAMES)
    assert set(path.name for path in moved.iterdir()) == set(archive.ARTIFACT_FILENAMES)


def test_publication_fails_closed_when_renameat2_is_unavailable(tmp_path, monkeypatch):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    monkeypatch.setattr(publication.ctypes, "CDLL", lambda *args, **kwargs: object())

    with pytest.raises(RuntimeError, match="RENAME_NOREPLACE.*unavailable"):
        publication.publish_artifact_directory(
            output, documents, archive.ARTIFACT_FILENAMES
        )

    assert not os.path.lexists(output)
    assert not list(tmp_path.glob(".official.*.staging"))


def test_receipt_detects_artifact_content_changes_without_removing_them(tmp_path):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    receipt = publication.publish_artifact_directory(
        output, documents, archive.ARTIFACT_FILENAMES
    )
    (output / archive.ARTIFACT_FILENAMES[0]).write_bytes(b"foreign change")

    with pytest.raises(RuntimeError, match="contents differ"):
        publication.verify_publication(receipt)

    assert output.is_dir()


def test_receipt_rejects_named_file_replaced_after_both_descriptor_hash_passes(
    tmp_path, monkeypatch
):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    receipt = publication.publish_artifact_directory(
        output, documents, archive.ARTIFACT_FILENAMES
    )
    name = archive.ARTIFACT_FILENAMES[0]
    moved = tmp_path / "original-file"
    foreign = tmp_path / "same-bytes-different-inode"
    foreign.write_bytes(documents[name])
    real_hash = publication._descriptor_sha256
    hashes = 0

    def substitute_after_second_pass(descriptor):
        nonlocal hashes
        digest = real_hash(descriptor)
        hashes += 1
        if hashes == 2 * len(archive.ARTIFACT_FILENAMES):
            os.rename(output / name, moved)
            os.rename(foreign, output / name)
        return digest

    monkeypatch.setattr(publication, "_descriptor_sha256", substitute_after_second_pass)
    with pytest.raises(RuntimeError, match="contents differ"):
        with publication.lock_parent_directory(output):
            publication.verify_publication(receipt)
    assert hashes >= 2 * len(archive.ARTIFACT_FILENAMES)
    assert moved.read_bytes() == documents[name]
    assert (output / name).read_bytes() == documents[name]
    assert moved.stat().st_ino != (output / name).stat().st_ino


def test_receipt_rejects_first_file_mutated_during_last_named_hash(tmp_path, monkeypatch):
    output = tmp_path / "official"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    receipt = publication.publish_artifact_directory(
        output, documents, archive.ARTIFACT_FILENAMES
    )
    first_name = archive.ARTIFACT_FILENAMES[0]
    real_hash = publication._descriptor_sha256
    hashes = 0
    mutated = False

    def mutate_after_last_named_hash(descriptor):
        nonlocal hashes, mutated
        digest = real_hash(descriptor)
        hashes += 1
        if hashes == 3 * len(archive.ARTIFACT_FILENAMES):
            with (output / first_name).open("r+b") as handle:
                handle.write(b"X" * len(documents[first_name]))
            mutated = True
        return digest

    monkeypatch.setattr(publication, "_descriptor_sha256", mutate_after_last_named_hash)
    with pytest.raises(RuntimeError, match="contents differ"):
        with publication.lock_parent_directory(output):
            publication.verify_publication(receipt)
    assert mutated
    assert (output / first_name).read_bytes() != documents[first_name]


def test_receipt_rejects_destination_substituted_after_directory_fstat(
    tmp_path, monkeypatch
):
    output = tmp_path / "official"
    moved = tmp_path / "authenticated-publication"
    documents = {name: name.encode() for name in archive.ARTIFACT_FILENAMES}
    receipt = publication.publish_artifact_directory(
        output, documents, archive.ARTIFACT_FILENAMES
    )
    real_fstat = publication.os.fstat
    substituted = False

    def substitute_after_directory_fstat(descriptor):
        nonlocal substituted
        result = real_fstat(descriptor)
        if not substituted and (result.st_dev, result.st_ino) == (
            receipt.st_dev,
            receipt.st_ino,
        ):
            substituted = True
            os.rename(output, moved)
            output.mkdir()
            for filename, contents in documents.items():
                (output / filename).write_bytes(contents)
        return result

    monkeypatch.setattr(publication.os, "fstat", substitute_after_directory_fstat)

    with pytest.raises(RuntimeError, match="replaced|incompatible"):
        publication.verify_publication(receipt)

    assert substituted
    assert set(path.name for path in output.iterdir()) == set(archive.ARTIFACT_FILENAMES)
    assert set(path.name for path in moved.iterdir()) == set(archive.ARTIFACT_FILENAMES)


def test_official_entry_point_has_locked_signature_and_exact_constants(monkeypatch):
    assert tuple(inspect.signature(archive.run_registered_experiment).parameters) == (
        "expected_commit",
    )
    captured = {}
    expected = archive.ExperimentOutcome("result", "decision")

    def spy(**kwargs):
        captured.update(kwargs)
        return expected

    monkeypatch.setattr(archive, "_run_registered_experiment_with_overrides", spy)

    assert archive.run_registered_experiment(expected_commit="a" * 40) is expected
    assert captured == {
        "expected_commit": "a" * 40,
        "repo_root": archive.PROJECT_ROOT,
        "data_paths": tuple(
            archive.PROJECT_ROOT / filename for filename in archive.ARCHIVE_FILENAMES
        ),
        "registry_path": archive.DEFAULT_REGISTRY_PATH,
        "artifact_dir": archive.PROJECT_ROOT / archive.ARTIFACT_DIRECTORY,
        "dataset_contract": archive.OFFICIAL_DATASET_CONTRACT,
        "evaluation_start_year": 2014,
        "evaluation_end_year": 2025,
    }


def test_public_api_cannot_register_official_id_with_synthetic_contract(tmp_path):
    registry = tmp_path / "registry.csv"
    with pytest.raises(TypeError):
        archive.run_registered_experiment(
            expected_commit="a" * 40,
            dataset_contract=archive.DatasetContract(1, date(2020, 1, 1), date(2020, 1, 1)),
            registry_path=registry,
        )
    assert not registry.exists()


def test_registered_command_refuses_2026_before_protocol_or_artifacts(tmp_path, monkeypatch):
    paths = _synthetic_archives(tmp_path)
    draws = (*_draws(), DrawObservation(date(2026, 9, 21), 64, (1, 2, 3, 4, 5)))
    output = tmp_path / "artifacts" / "official"
    monkeypatch.setattr(archive, "load_archive_draws", lambda *args, **kwargs: draws)
    monkeypatch.setattr(archive, "build_data_manifest", lambda paths: "{}")
    monkeypatch.setattr(
        archive, "build_protocol", lambda *args, **kwargs: pytest.fail("protocol saw 2026")
    )
    monkeypatch.setattr(
        archive, "run_experiment", lambda config, command, **kwargs: command()
    )
    with pytest.raises(ValueError, match="2026"):
        archive._run_registered_experiment_with_overrides(
            expected_commit="a" * 40,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=tmp_path / "registry.csv",
            artifact_dir=output,
            dataset_contract=archive.DatasetContract(65, draws[0].draw_date, draws[-1].draw_date),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )
    assert not output.exists()


def test_registered_path_calls_run_experiment_and_appends_once(tmp_path, monkeypatch):
    paths = _synthetic_archives(tmp_path)
    commit = _init_repo(tmp_path)
    registry = tmp_path / "experiments" / "registry.csv"
    output = tmp_path / "artifacts" / "task-2.3a-archive-walk-forward"
    called = []
    real = archive.run_experiment
    real_build_manifest = archive.build_data_manifest
    manifests = []

    def spy(*args, **kwargs):
        called.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(archive, "run_experiment", spy)
    def recording_manifest(paths):
        manifest = real_build_manifest(paths)
        manifests.append(manifest)
        return manifest

    monkeypatch.setattr(archive, "build_data_manifest", recording_manifest)
    monkeypatch.setattr(experiments, "build_data_manifest", recording_manifest)
    synthetic = _draws()
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)
    outcome = archive._run_registered_experiment_with_overrides(
        expected_commit=commit,
        repo_root=tmp_path,
        data_paths=paths,
        registry_path=registry,
        artifact_dir=output,
        dataset_contract=archive.DatasetContract(len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date),
        evaluation_start_year=2014,
        evaluation_end_year=2014,
    )

    assert len(called) == 1
    assert len(manifests) == 3
    assert manifests[0] == manifests[1] == manifests[2]
    assert called[0][0][0] == archive.experiment_config()
    assert tuple(inspect.signature(archive._run_registered_experiment_with_overrides).parameters) == (
        "expected_commit",
        "repo_root",
        "data_paths",
        "registry_path",
        "artifact_dir",
        "dataset_contract",
        "evaluation_start_year",
        "evaluation_end_year",
    )
    assert outcome.artifacts == tuple(str(path.relative_to(tmp_path)) for path in (output / "preregistration.json", output / "metrics.csv", output / "result.json"))
    assert {path.name for path in output.iterdir()} == set(archive.ARTIFACT_FILENAMES)
    assert not list(output.parent.glob(f".{output.name}.*.staging"))
    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["id"] for row in rows] == [archive.EXPERIMENT_ID]
    assert callable(called[0][1]["validate_before_record"])
    with pytest.raises(ValueError, match=archive.EXPERIMENT_ID):
        archive._run_registered_experiment_with_overrides(
            expected_commit=commit,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=registry,
            artifact_dir=tmp_path / "unused",
            dataset_contract=archive.DatasetContract(len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )


def test_registered_path_rejects_dirty_tree_wrong_commit_and_existing_artifact(tmp_path, monkeypatch):
    paths = _synthetic_archives(tmp_path)
    commit = _init_repo(tmp_path)
    registry = tmp_path / "registry.csv"
    output = tmp_path / "artifacts"
    synthetic = _draws()
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)
    real_build_manifest = archive.build_data_manifest
    manifest_accesses = 0

    def recording_manifest(paths):
        nonlocal manifest_accesses
        manifest_accesses += 1
        return real_build_manifest(paths)

    monkeypatch.setattr(archive, "build_data_manifest", recording_manifest)
    kwargs = dict(
        expected_commit=commit,
        repo_root=tmp_path,
        data_paths=paths,
        registry_path=registry,
        artifact_dir=output,
        dataset_contract=archive.DatasetContract(len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date),
        evaluation_start_year=2014,
        evaluation_end_year=2014,
    )

    (tmp_path / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(RuntimeError, match="modifications"):
        archive._run_registered_experiment_with_overrides(**kwargs)
    assert manifest_accesses == 0
    (tmp_path / "dirty.txt").unlink()

    with pytest.raises(ValueError, match="commit"):
        archive._run_registered_experiment_with_overrides(
            **{**kwargs, "expected_commit": "0" * 40}
        )
    assert manifest_accesses == 0

    output.mkdir()
    (output / "metrics.csv").write_text("existing", encoding="utf-8")
    _git(tmp_path, "add", "artifacts")
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Loto Tests",
            "-c",
            "user.email=loto@example.invalid",
            "commit",
            "-qm",
            "existing artifact fixture",
        ],
        cwd=tmp_path,
        check=True,
    )
    kwargs["expected_commit"] = _git(tmp_path, "rev-parse", "HEAD")
    with pytest.raises(FileExistsError):
        archive._run_registered_experiment_with_overrides(**kwargs)
    assert (output / "metrics.csv").read_text(encoding="utf-8") == "existing"


def test_registered_path_retains_its_publication_and_error_trace(
    tmp_path, monkeypatch
):
    paths = _synthetic_archives(tmp_path)
    registry = tmp_path / "registry.csv"
    output = tmp_path / "artifacts" / "official"
    synthetic = _draws()
    monkeypatch.setattr(
        archive, "build_data_manifest", lambda paths: '{"files":[],"sha256":"x"}'
    )
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)
    real_write_artifacts = archive.write_artifacts
    receipts = []

    def recording_write_artifacts(*args, **kwargs):
        receipt = real_write_artifacts(*args, **kwargs)
        receipts.append(receipt)
        return receipt

    monkeypatch.setattr(archive, "write_artifacts", recording_write_artifacts)

    error_trace = b"append-only-error-trace\n"

    def fail_after_publication(config, command, **kwargs):
        command()
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        registry.write_bytes(error_trace)
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(archive, "run_experiment", fail_after_publication)

    with pytest.raises(RuntimeError, match="registry failure") as raised:
        archive._run_registered_experiment_with_overrides(
            expected_commit="a" * 40,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=registry,
            artifact_dir=output,
            dataset_contract=archive.DatasetContract(
                len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date
            ),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )

    assert set(path.name for path in output.iterdir()) == set(archive.ARTIFACT_FILENAMES)
    assert all((output / name).is_file() for name in archive.ARTIFACT_FILENAMES)
    assert publication.verify_publication(receipts[0]) == receipts[0].paths
    assert registry.read_bytes() == error_trace
    assert not list(output.parent.glob(".official.*.staging"))
    assert any(
        "deliberately retained for inspection" in note
        and "durable registry row or durable reservation" in note
        for note in raised.value.__notes__
    )


@pytest.mark.parametrize("failure_point", ["parent_fsync", "verification"])
def test_registered_post_rename_failure_records_error_and_recovery_receipt(
    tmp_path, monkeypatch, failure_point
):
    paths = _synthetic_archives(tmp_path)
    commit = _init_repo(tmp_path)
    registry = tmp_path / "experiments" / "registry.csv"
    output = tmp_path / "artifacts" / "official"
    synthetic = _draws()
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)
    real_fsync_directory = publication.fsync_directory
    real_verify_publication = publication.verify_publication

    if failure_point == "parent_fsync":
        def injected_fsync(directory):
            if Path(directory) == output.parent and output.exists():
                raise OSError("injected parent fsync failure")
            return real_fsync_directory(directory)

        monkeypatch.setattr(publication, "fsync_directory", injected_fsync)
    else:
        monkeypatch.setattr(
            publication,
            "verify_publication",
            lambda receipt: (_ for _ in ()).throw(
                RuntimeError("injected publication verification failure")
            ),
        )

    with pytest.raises(publication.PublishedArtifactError) as raised:
        archive._run_registered_experiment_with_overrides(
            expected_commit=commit,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=registry,
            artifact_dir=output,
            dataset_contract=archive.DatasetContract(
                len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date
            ),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )

    receipt = raised.value.receipt
    assert receipt.directory == output
    assert output.is_dir()
    assert not list(output.parent.glob(".official.*.staging"))
    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["décision"] == "erreur"
    assert "PublishedArtifactError" in rows[0]["résultat"]
    assert json.loads(rows[0]["artefacts"]) == []
    reservation_directory = registry.with_name(f"{registry.name}.reservations")
    assert not reservation_directory.exists() or not list(reservation_directory.iterdir())
    assert any(
        f"directory={output}" in note and f"receipt={receipt!r}" in note
        for note in raised.value.__notes__
    )
    monkeypatch.setattr(publication, "fsync_directory", real_fsync_directory)
    monkeypatch.setattr(publication, "verify_publication", real_verify_publication)
    assert publication.verify_publication(receipt) == receipt.paths


def test_registered_path_rejects_substitution_in_pre_append_hook(
    tmp_path, monkeypatch
):
    paths = _synthetic_archives(tmp_path)
    commit = _init_repo(tmp_path)
    registry = tmp_path / "experiments" / "registry.csv"
    output = tmp_path / "artifacts" / "official"
    moved = tmp_path / "artifacts" / "real-publication"
    synthetic = _draws()
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)
    real_listdir = publication.os.listdir
    enumerations = 0

    def substitute_during_pre_append_verification(directory_descriptor):
        nonlocal enumerations
        enumerations += 1
        if enumerations == 2:
            documents = {
                filename: (output / filename).read_bytes()
                for filename in archive.ARTIFACT_FILENAMES
            }
            os.rename(output, moved)
            output.mkdir()
            for filename, contents in documents.items():
                (output / filename).write_bytes(contents)
        return real_listdir(directory_descriptor)

    monkeypatch.setattr(
        publication.os, "listdir", substitute_during_pre_append_verification
    )

    with pytest.raises(RuntimeError, match="replaced by a foreign object"):
        archive._run_registered_experiment_with_overrides(
            expected_commit=commit,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=registry,
            artifact_dir=output,
            dataset_contract=archive.DatasetContract(
                len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date
            ),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )

    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["décision"] == "erreur"
    assert "replaced by a foreign object" in rows[0]["résultat"]
    assert enumerations == 2
    assert set(path.name for path in output.iterdir()) == set(archive.ARTIFACT_FILENAMES)
    assert moved.is_dir()


def test_registered_path_rejects_first_artifact_mutated_while_second_is_hashed(
    tmp_path, monkeypatch
):
    paths = _synthetic_archives(tmp_path)
    commit = _init_repo(tmp_path)
    registry = tmp_path / "experiments" / "registry.csv"
    output = tmp_path / "artifacts" / "official"
    synthetic = _draws()
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)
    real_open = publication.os.open
    real_read = publication.os.read
    second_filename = archive.ARTIFACT_FILENAMES[1]
    second_opens = 0
    armed_descriptor = None
    mutated = False

    def arm_on_second_artifact_open(path, flags, *args, **kwargs):
        nonlocal armed_descriptor, second_opens
        descriptor = real_open(path, flags, *args, **kwargs)
        if path == second_filename and kwargs.get("dir_fd") is not None:
            second_opens += 1
            if second_opens == 2:
                armed_descriptor = descriptor
        return descriptor

    def mutate_first_artifact_while_hashing_second(descriptor, size):
        nonlocal mutated
        if descriptor == armed_descriptor and not mutated:
            mutated = True
            (output / archive.ARTIFACT_FILENAMES[0]).write_bytes(b"concurrent mutation")
        return real_read(descriptor, size)

    monkeypatch.setattr(publication.os, "open", arm_on_second_artifact_open)
    monkeypatch.setattr(publication.os, "read", mutate_first_artifact_while_hashing_second)

    with pytest.raises(RuntimeError, match="contents differ"):
        archive._run_registered_experiment_with_overrides(
            expected_commit=commit,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=registry,
            artifact_dir=output,
            dataset_contract=archive.DatasetContract(
                len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date
            ),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )

    assert mutated
    with registry.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["décision"] == "erreur"
    assert "contents differ" in rows[0]["résultat"]


def test_registry_failure_never_removes_a_foreign_replacement(tmp_path, monkeypatch):
    paths = _synthetic_archives(tmp_path)
    output = tmp_path / "artifacts" / "official"
    moved = output.parent / "publication-moved-by-foreign-process"
    synthetic = _draws()
    monkeypatch.setattr(
        archive, "build_data_manifest", lambda paths: '{"files":[],"sha256":"x"}'
    )
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)

    def fail_after_substitution(config, command, **kwargs):
        command()
        os.rename(output, moved)
        output.mkdir()
        (output / "foreign").write_bytes(b"must survive")
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(archive, "run_experiment", fail_after_substitution)

    with pytest.raises(RuntimeError, match="registry failure") as raised:
        archive._run_registered_experiment_with_overrides(
            expected_commit="a" * 40,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=tmp_path / "registry.csv",
            artifact_dir=output,
            dataset_contract=archive.DatasetContract(
                len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date
            ),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )

    assert (output / "foreign").read_bytes() == b"must survive"
    assert moved.is_dir()
    assert set(path.name for path in moved.iterdir()) == set(archive.ARTIFACT_FILENAMES)
    assert any(
        "deliberately retained for inspection" in note
        for note in raised.value.__notes__
    )


def test_registry_failure_has_no_verified_then_path_removal_toctou(
    tmp_path, monkeypatch
):
    paths = _synthetic_archives(tmp_path)
    output = tmp_path / "artifacts" / "official"
    moved = output.parent / "verified-publication"
    foreign = output.parent / "foreign-directory"
    foreign.mkdir(parents=True)
    sentinel = foreign / "must-survive"
    sentinel.write_bytes(b"foreign")
    synthetic = _draws()
    monkeypatch.setattr(
        archive, "build_data_manifest", lambda paths: '{"files":[],"sha256":"x"}'
    )
    monkeypatch.setattr(archive, "load_archive_draws", lambda paths, contract: synthetic)

    real_rmtree = publication.shutil.rmtree

    def substitute_after_verification(path, *args, **kwargs):
        if Path(path) == output:
            os.rename(output, moved)
            os.rename(foreign, output)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(publication.shutil, "rmtree", substitute_after_verification)

    def fail_after_publication(config, command, **kwargs):
        command()
        raise RuntimeError("injected registry failure")

    monkeypatch.setattr(archive, "run_experiment", fail_after_publication)

    with pytest.raises(RuntimeError, match="registry failure"):
        archive._run_registered_experiment_with_overrides(
            expected_commit="a" * 40,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=tmp_path / "registry.csv",
            artifact_dir=output,
            dataset_contract=archive.DatasetContract(
                len(synthetic), synthetic[0].draw_date, synthetic[-1].draw_date
            ),
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )

    assert sentinel.read_bytes() == b"foreign"
    assert set(path.name for path in output.iterdir()) == set(archive.ARTIFACT_FILENAMES)


def test_registered_path_refuses_residual_staging_for_recovery(tmp_path, monkeypatch):
    paths = _synthetic_archives(tmp_path)
    output = tmp_path / "artifacts" / "official"
    output.parent.mkdir()
    residual = output.parent / ".official.interrupted.staging"
    residual.mkdir()
    sentinel = residual / "sentinel"
    sentinel.write_text("inspect me", encoding="utf-8")
    monkeypatch.setattr(
        archive, "run_experiment", lambda config, command, **kwargs: command()
    )

    with pytest.raises(RuntimeError, match="staging"):
        archive._run_registered_experiment_with_overrides(
            expected_commit="a" * 40,
            repo_root=tmp_path,
            data_paths=paths,
            registry_path=tmp_path / "registry.csv",
            artifact_dir=output,
            dataset_contract=archive.OFFICIAL_DATASET_CONTRACT,
            evaluation_start_year=2014,
            evaluation_end_year=2014,
        )

    assert sentinel.read_text(encoding="utf-8") == "inspect me"


def test_json_result_contains_integrity_controls_and_no_2026_targets(tmp_path):
    draws = _draws()
    protocol = archive.build_protocol(draws, evaluation_start_year=2014, evaluation_end_year=2014)
    run = archive.evaluate_candidates(draws, protocol)
    output = tmp_path / "artifacts"
    archive.write_artifacts(output, run, protocol, commit="b" * 40, dataset_manifest={"sha256": "x", "files": []})

    preregistration = json.loads((output / "preregistration.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert preregistration["exclusion_2026"]["targets"] == 0
    assert result["integrity"]["maximum_target_date"] <= "2025-12-31"
    assert result["integrity"]["targets_in_2026"] == 0
    assert result["equivalence_checks"]["cumulative_dirichlet_probabilities_exact"] is True
    assert result["equivalence_checks"]["cumulative_dirichlet_metrics_exact"] is True
    assert result["equivalence_checks"]["uniform_constrained_probabilistic_metrics_exact"] is True
    assert "duration" not in json.dumps(result).lower()
    assert "timestamp" not in json.dumps(result).lower()
