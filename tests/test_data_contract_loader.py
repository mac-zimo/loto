import csv
import hashlib
import sqlite3
from pathlib import Path

import pytest

from loto.config import CSV_FILES
from loto.data_quality import data_quality_summary
from loto.data_schema import SourceSchema, StructuredValidationError, schema_contract
from loto.database import init_db
from loto.loader import load_all_db, load_csv_file


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = SourceSchema.OCT_2008_MAR_2017
HEADERS = schema_contract(SCHEMA).expected_columns


def write_source(
    path, *, draw_id="2017027", ball_1="1", rank_1="", report_1=""
):
    row = {column: "" for column in HEADERS}
    row.update({
        "annee_numero_de_tirage": draw_id,
        "jour_de_tirage": "SAMEDI",
        "date_de_tirage": "04/03/2017",
        "date_de_forclusion": "04/05/2017",
        "boule_1": ball_1,
        "boule_2": "2",
        "boule_3": "3",
        "boule_4": "4",
        "boule_5": "5",
        "numero_chance": "6",
        "combinaison_gagnante_en_ordre_croissant": "1-2-3-4-5+6",
        "nombre_de_gagnant_au_rang1": rank_1,
        "rapport_du_rang1": report_1,
        "devise": "eur",
    })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(HEADERS)
        writer.writerow([row[column] for column in HEADERS])


def test_hash_provenance_one_based_row_and_nulls_are_persisted(tmp_path):
    source = tmp_path / "archive.csv"
    database = tmp_path / "loto.db"
    write_source(source)
    expected_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    conn = init_db(database)
    assert load_csv_file(source, conn) == (1, 1)
    stored = conn.execute(
        """SELECT t.source_row, s.filename, s.sha256, s.size_bytes,
                  s.source_schema, s.imported_at, t.gagnants_rang1,
                  t.gagnants_rang7, t.nombre_codes,
                  t.boule_1_second_tirage, t.numero_7
           FROM tirages t JOIN source_files s ON s.id = t.source_file_id"""
    ).fetchone()
    conn.close()

    assert stored[:5] == (1, "archive.csv", expected_hash, source.stat().st_size, SCHEMA.value)
    assert stored[5].endswith("+00:00")
    assert stored[6:] == (None, None, None, None, None)


def test_non_strict_zero_accepted_rows_rejects_without_metadata(tmp_path, caplog):
    source = tmp_path / "invalid.csv"
    database = tmp_path / "loto.db"
    write_source(source, draw_id="bad", ball_1="not-an-integer")

    conn = init_db(database)
    with pytest.raises(StructuredValidationError, match="zero accepted rows"):
        load_csv_file(source, conn)
    source_count = conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
    conn.close()

    assert source_count == 0
    record = next(record for record in caplog.records if "Validation CSV" in record.message)
    assert record.args == {
        "file": "invalid.csv",
        "row": 1,
        "field": "boule_1",
        "message": "value must be an integer",
    }


@pytest.mark.parametrize(
    "kwargs,field,message",
    [
        (
            {"rank_1": "9" * 100},
            "nombre_de_gagnant_au_rang1",
            "winner count must fit SQLite INTEGER",
        ),
        (
            {"report_1": "1e999999"},
            "rapport_du_rang1",
            "monetary report must fit finite SQLite REAL",
        ),
    ],
)
def test_non_strict_oversized_numeric_is_structured_and_does_not_mutate(
    tmp_path, caplog, kwargs, field, message
):
    source = tmp_path / "oversized.csv"
    database = tmp_path / "loto.db"
    write_source(source, **kwargs)
    conn = init_db(database)

    with pytest.raises(StructuredValidationError, match="zero accepted rows"):
        load_csv_file(source, conn)

    assert conn.execute("SELECT COUNT(*) FROM tirages").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM source_files").fetchone()[0] == 0
    conn.close()
    record = next(record for record in caplog.records if "Validation CSV" in record.message)
    assert record.args == {
        "file": "oversized.csv",
        "row": 1,
        "field": field,
        "message": message,
    }


def test_existing_legacy_draw_gets_provenance_without_becoming_new(tmp_path):
    source = tmp_path / "archive.csv"
    database = tmp_path / "loto.db"
    write_source(source)
    conn = init_db(database)
    conn.execute(
        """INSERT INTO tirages (
               annee_numero, jour_tirage, date_tirage, date_forclusion,
               boule_1, boule_2, boule_3, boule_4, boule_5, numero_chance
           ) VALUES ('2017027', 'SAMEDI', '2017-03-04', '2017-05-04',
                     1, 2, 3, 4, 5, 6)"""
    )
    conn.commit()

    assert load_csv_file(source, conn) == (1, 0)
    provenance = conn.execute(
        """SELECT s.filename, t.source_row
           FROM tirages t JOIN source_files s ON s.id = t.source_file_id"""
    ).fetchone()
    conn.close()

    assert provenance == ("archive.csv", 1)


def test_database_migration_is_idempotent_and_enables_foreign_keys(tmp_path):
    database = tmp_path / "old.db"
    old_columns = [
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
    raw = sqlite3.connect(database)
    raw.execute(f"CREATE TABLE tirages ({', '.join(old_columns)})")
    raw.commit()
    raw.close()

    first = init_db(database)
    first.close()
    second = init_db(database)
    columns = {row[1] for row in second.execute("PRAGMA table_info(tirages)")}
    foreign_keys = second.execute("PRAGMA foreign_keys").fetchone()[0]
    second.close()

    assert {"source_file_id", "source_row", "boule_1_second_tirage", "numero_7"} <= columns
    assert foreign_keys == 1


def test_quality_summary_is_deterministic_and_counts_duplicates(tmp_path):
    source = tmp_path / "archive.csv"
    database = tmp_path / "loto.db"
    write_source(source)
    conn = init_db(database)
    load_csv_file(source, conn)
    load_csv_file(source, conn)
    conn.close()

    summary = data_quality_summary(database)
    assert summary.rows_read == 1
    assert summary.rows_accepted == 1
    assert summary.rows_rejected == 0
    assert summary.duplicates == 1
    assert summary.schema_counts == {SCHEMA.value: 1}
    assert (summary.min_date, summary.max_date) == ("2017-03-04", "2017-03-04")
    assert summary.nullable_field_counts["rapport_rang1"] == 1


@pytest.mark.integration
def test_real_archives_validate_to_expected_unique_contract(tmp_path):
    assert all((ROOT / filename).is_file() for filename in CSV_FILES)
    database = tmp_path / "real.db"

    assert load_all_db(reset=True, data_dir=ROOT, db_path=database) == (2811, 2811)
    conn = sqlite3.connect(database)
    result = conn.execute(
        """SELECT COUNT(*), COUNT(DISTINCT annee_numero),
                  MIN(date_tirage), MAX(date_tirage)
           FROM tirages"""
    ).fetchone()
    source_counts = conn.execute(
        "SELECT SUM(rows_rejected), COUNT(*) FROM source_files"
    ).fetchone()
    conn.close()

    assert result == (2811, 2811, "2008-10-06", "2026-09-21")
    assert source_counts == (0, 4)
