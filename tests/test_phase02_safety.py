import csv
import hashlib
import sqlite3

import pytest

from loto.config import CSV_FILES
from loto.data_schema import SourceSchema, StructuredValidationError, schema_contract
from loto.data_source import DuplicateDrawIdError
from loto.database import init_db
from loto.loader import RowImportError, load_all_db, load_csv_file


SCHEMAS_AND_DATES = (
    (SourceSchema.OCT_2008_MAR_2017, "04/03/2017"),
    (SourceSchema.MAR_2017_FEB_2019, "06/03/2017"),
    (SourceSchema.FEB_2019_NOV_2019, "27/02/2019"),
    (SourceSchema.NOV_2019_ONWARD, "06/11/2019"),
)


def _row(schema, draw_id, draw_date):
    row = {column: "" for column in schema_contract(schema).expected_columns}
    row.update(
        {
            "annee_numero_de_tirage": draw_id,
            "jour_de_tirage": "LUNDI",
            "date_de_tirage": draw_date,
            "date_de_forclusion": "06/05/2019",
            "boule_1": "1",
            "boule_2": "2",
            "boule_3": "3",
            "boule_4": "4",
            "boule_5": "5",
            "numero_chance": "6",
            "combinaison_gagnante_en_ordre_croissant": "1-2-3-4-5+6",
            "devise": "eur",
        }
    )
    return row


def _write(path, schema, rows):
    header = schema_contract(schema).expected_columns
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[column] for column in header])


def _write_complete_set(data_dir):
    for index, (filename, (schema, draw_date)) in enumerate(
        zip(CSV_FILES, SCHEMAS_AND_DATES), start=1
    ):
        _write(data_dir / filename, schema, [_row(schema, f"draw-{index}", draw_date)])


def _insert_sentinel(database):
    conn = init_db(database)
    conn.execute(
        """INSERT INTO tirages (
               annee_numero, jour_tirage, date_tirage, boule_1, boule_2,
               boule_3, boule_4, boule_5, numero_chance
           ) VALUES ('sentinel', 'LUNDI', '2020-01-01', 1, 2, 3, 4, 5, 6)"""
    )
    conn.commit()
    conn.close()


def _draw_ids(database):
    conn = sqlite3.connect(database)
    result = [row[0] for row in conn.execute("SELECT annee_numero FROM tirages")]
    conn.close()
    return result


def _create_legacy_database(database):
    columns = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "annee_numero TEXT UNIQUE NOT NULL",
        "jour_tirage TEXT NOT NULL",
        "date_tirage TEXT NOT NULL",
        "date_forclusion TEXT",
        *(f"boule_{index} INTEGER NOT NULL" for index in range(1, 6)),
        "numero_chance INTEGER NOT NULL",
        "combinaison_croissante TEXT",
        *(column for rank in range(1, 10) for column in (
            f"gagnants_rang{rank} INTEGER DEFAULT 0",
            f"rapport_rang{rank} REAL DEFAULT 0",
        )),
        "nombre_codes INTEGER DEFAULT 0",
        "rapport_codes REAL DEFAULT 0",
        "codes_gagnants TEXT",
        "numero_jokerplus TEXT",
        "devise TEXT DEFAULT 'EUR'",
    ]
    conn = sqlite3.connect(database)
    conn.execute(f"CREATE TABLE tirages ({', '.join(columns)})")
    conn.execute(
        """INSERT INTO tirages (
               annee_numero, jour_tirage, date_tirage, boule_1, boule_2,
               boule_3, boule_4, boule_5, numero_chance
           ) VALUES ('legacy', 'LUNDI', '2017-01-01', 1, 2, 3, 4, 5, 6)"""
    )
    conn.commit()
    conn.close()


def _database_snapshot(database):
    conn = sqlite3.connect(database)
    schema = conn.execute(
        "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
    ).fetchall()
    data = conn.execute("SELECT * FROM tirages ORDER BY id").fetchall()
    conn.close()
    return schema, data


def test_reset_header_only_unique_layout_preserves_sentinel(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    _write(data_dir / CSV_FILES[0], SourceSchema.OCT_2008_MAR_2017, [])
    _insert_sentinel(database)

    with pytest.raises(StructuredValidationError, match="zero data rows"):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert _draw_ids(database) == ["sentinel"]


def test_reset_duplicate_and_missing_schema_preserves_sentinel(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    schema, draw_date = SCHEMAS_AND_DATES[0]
    _write(data_dir / CSV_FILES[1], schema, [_row(schema, "duplicate-period", draw_date)])
    _insert_sentinel(database)

    with pytest.raises(StructuredValidationError, match="exactly one source"):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert _draw_ids(database) == ["sentinel"]


def test_reset_zero_accepted_rows_preserves_sentinel(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    schema, draw_date = SCHEMAS_AND_DATES[0]
    invalid = _row(schema, "invalid", draw_date)
    invalid["boule_1"] = "invalid"
    _write(data_dir / CSV_FILES[0], schema, [invalid])
    _insert_sentinel(database)

    with pytest.raises(RowImportError):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert _draw_ids(database) == ["sentinel"]


def test_shared_header_malformed_date_is_counted_in_normal_mode(tmp_path, caplog):
    source = tmp_path / "shared.csv"
    database = tmp_path / "loto.db"
    schema, draw_date = SCHEMAS_AND_DATES[1]
    valid = _row(schema, "valid", draw_date)
    malformed = _row(schema, "malformed", "not-a-date")
    _write(source, schema, [valid, malformed])

    conn = init_db(database)
    assert load_csv_file(source, conn) == (2, 1)
    counts = conn.execute(
        "SELECT rows_read, rows_accepted, rows_rejected FROM source_files"
    ).fetchone()
    conn.close()

    assert counts == (2, 1, 1)
    assert _draw_ids(database) == ["valid"]
    warning = next(record for record in caplog.records if "Validation CSV" in record.message)
    assert warning.args["field"] == "date_de_tirage"
    assert warning.args["row"] == 2


def test_shared_header_malformed_date_rolls_back_strict_reset(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    schema, draw_date = SCHEMAS_AND_DATES[1]
    rows = [_row(schema, "valid", draw_date), _row(schema, "malformed", "bad")]
    _write(data_dir / CSV_FILES[1], schema, rows)
    _insert_sentinel(database)

    with pytest.raises(RowImportError):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert _draw_ids(database) == ["sentinel"]


def test_reset_duplicate_draw_id_is_rejected_before_existing_db_changes(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    first_schema, first_date = SCHEMAS_AND_DATES[0]
    second_schema, second_date = SCHEMAS_AND_DATES[1]
    _write(data_dir / CSV_FILES[0], first_schema, [_row(first_schema, "same", first_date)])
    _write(data_dir / CSV_FILES[1], second_schema, [_row(second_schema, "same", second_date)])
    _insert_sentinel(database)
    before = hashlib.sha256(database.read_bytes()).hexdigest()

    with pytest.raises(DuplicateDrawIdError) as raised:
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert raised.value.as_dict() == {
        "file": CSV_FILES[1],
        "row": 1,
        "field": "annee_numero_de_tirage",
        "message": (
            f"duplicate canonical draw ID 'same'; first occurrence "
            f"{CSV_FILES[0]}:data-row 1"
        ),
        "draw_id": "same",
        "first_file": CSV_FILES[0],
        "first_row": 1,
    }
    assert hashlib.sha256(database.read_bytes()).hexdigest() == before
    assert _draw_ids(database) == ["sentinel"]


def test_duplicate_preflight_does_not_create_database_or_parent(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "state" / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    first_schema, first_date = SCHEMAS_AND_DATES[0]
    second_schema, second_date = SCHEMAS_AND_DATES[1]
    _write(data_dir / CSV_FILES[0], first_schema, [_row(first_schema, "same", first_date)])
    _write(data_dir / CSV_FILES[1], second_schema, [_row(second_schema, "same", second_date)])

    with pytest.raises(DuplicateDrawIdError):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert not database.exists()
    assert not database.parent.exists()


def test_malformed_preflight_does_not_create_database_or_parent(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "state" / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    schema, draw_date = SCHEMAS_AND_DATES[2]
    malformed = _row(schema, "malformed", draw_date)
    malformed["boule_1"] = "bad"
    _write(data_dir / CSV_FILES[2], schema, [malformed])

    with pytest.raises(RowImportError):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert not database.exists()
    assert not database.parent.exists()


def test_malformed_preflight_does_not_migrate_legacy_database(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "legacy.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    schema, draw_date = SCHEMAS_AND_DATES[2]
    malformed = _row(schema, "malformed", draw_date)
    malformed["boule_1"] = "bad"
    _write(data_dir / CSV_FILES[2], schema, [malformed])
    raw = sqlite3.connect(database)
    raw.execute("CREATE TABLE tirages (annee_numero TEXT PRIMARY KEY, note TEXT)")
    raw.execute("INSERT INTO tirages VALUES ('legacy', 'untouched')")
    raw.commit()
    schema_before = raw.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall()
    data_before = raw.execute("SELECT * FROM tirages").fetchall()
    raw.close()
    bytes_before = database.read_bytes()

    with pytest.raises(RowImportError):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert database.read_bytes() == bytes_before
    raw = sqlite3.connect(database)
    assert raw.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' ORDER BY name"
    ).fetchall() == schema_before
    assert raw.execute("SELECT * FROM tirages").fetchall() == data_before
    raw.close()


def test_failed_reset_rolls_back_schema_migration_and_data(tmp_path, monkeypatch):
    data_dir = tmp_path / "sources"
    database = tmp_path / "legacy.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    _create_legacy_database(database)
    before = _database_snapshot(database)

    def fail_invariants(*args, **kwargs):
        raise RuntimeError("forced post-migration invariant failure")

    monkeypatch.setattr("loto.loader.assert_reset_invariants", fail_invariants)

    with pytest.raises(RuntimeError, match="post-migration invariant failure"):
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert _database_snapshot(database) == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("nombre_de_gagnant_au_rang1", "9" * 100),
        ("rapport_du_rang1", "1e999999"),
    ],
)
def test_strict_reset_oversized_numeric_preserves_existing_database(
    tmp_path, field, value
):
    data_dir = tmp_path / "sources"
    database = tmp_path / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)
    schema, draw_date = SCHEMAS_AND_DATES[1]
    invalid = _row(schema, "oversized", draw_date)
    invalid[field] = value
    _write(data_dir / CSV_FILES[1], schema, [invalid])
    _insert_sentinel(database)

    with pytest.raises(RowImportError) as exc_info:
        load_all_db(reset=True, data_dir=data_dir, db_path=database)

    assert exc_info.value.as_dict()["field"] == field
    assert _draw_ids(database) == ["sentinel"]


def test_successful_reset_enforces_counts_provenance_and_run_timestamp(tmp_path):
    data_dir = tmp_path / "sources"
    database = tmp_path / "loto.db"
    data_dir.mkdir()
    _write_complete_set(data_dir)

    assert load_all_db(reset=True, data_dir=data_dir, db_path=database) == (4, 4)
    conn = sqlite3.connect(database)
    counts = conn.execute(
        """SELECT (SELECT COUNT(*) FROM tirages),
                  (SELECT COUNT(*) FROM tirages WHERE source_file_id IS NOT NULL
                                                AND source_row IS NOT NULL),
                  (SELECT SUM(rows_accepted) FROM source_files),
                  (SELECT COUNT(DISTINCT imported_at) FROM source_files)"""
    ).fetchone()
    timestamps = [row[0] for row in conn.execute("SELECT imported_at FROM source_files")]
    conn.close()

    assert counts == (4, 4, 4, 1)
    assert all(timestamp.endswith("+00:00") for timestamp in timestamps)