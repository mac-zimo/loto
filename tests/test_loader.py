import csv

import pytest

from loto.database import init_db
from loto import loader
from loto.config import CSV_FILES
from loto.loader import load_all_db, load_csv_file, normalize_date


CSV_HEADERS = [
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
    "rapport_du_rang1",
]


def write_draw_csv(path, draw_id, chance=6, boule_1="1", rapport_rang1="0"):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(CSV_HEADERS)
        writer.writerow([
            draw_id,
            "lundi",
            "05/01/2026",
            "07/03/2026",
            boule_1,
            "2",
            "3",
            "4",
            "5",
            str(chance),
            "1-2-3-4-5",
            rapport_rang1,
        ])


def write_complete_source_set(data_dir):
    for index, filename in enumerate(CSV_FILES, start=1):
        write_draw_csv(data_dir / filename, f"replacement-{index}", chance=index)


def insert_sentinel(db_path):
    conn = init_db(db_path)
    conn.execute(
        """INSERT INTO tirages (
            annee_numero, jour_tirage, date_tirage, boule_1, boule_2,
            boule_3, boule_4, boule_5, numero_chance
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("sentinel", "lundi", "2025-01-01", 10, 20, 30, 40, 49, 1),
    )
    conn.commit()
    conn.close()


def stored_draw_ids(db_path):
    conn = init_db(db_path)
    values = [row[0] for row in conn.execute(
        "SELECT annee_numero FROM tirages ORDER BY annee_numero"
    )]
    conn.close()
    return values


def test_normalize_date():
    assert normalize_date("21/09/2026") == "2026-09-21"
    assert normalize_date("2026-09-21") == "2026-09-21"


def test_load_csv_row_is_normalized_and_idempotent(tmp_path):
    csv_path = tmp_path / "draw.csv"
    db_path = tmp_path / "draw.db"
    write_draw_csv(csv_path, "2026001")

    conn = init_db(str(db_path))
    assert load_csv_file(csv_path, conn) == (1, 1)
    assert load_csv_file(csv_path, conn) == (1, 0)
    stored = conn.execute(
        "SELECT date_tirage, date_forclusion, boule_1, boule_5, numero_chance FROM tirages"
    ).fetchall()
    conn.close()

    assert stored == [("2026-01-05", "2026-03-07", 1, 5, 6)]


def test_init_db_creates_parent_directory_at_use_time(tmp_path):
    db_path = tmp_path / "nested" / "state" / "draw.db"

    assert not db_path.parent.exists()
    connection = init_db(db_path)
    connection.close()

    assert db_path.is_file()


def test_reset_with_incomplete_source_set_preserves_existing_database(tmp_path, caplog):
    db_path = tmp_path / "draw.db"
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    insert_sentinel(db_path)
    write_draw_csv(data_dir / CSV_FILES[0], "replacement-1")

    assert load_all_db(reset=True, data_dir=data_dir, db_path=db_path) == (0, 0)

    assert stored_draw_ids(db_path) == ["sentinel"]
    message = next(record.message for record in caplog.records if "Reset annulé" in record.message)
    assert str(data_dir / CSV_FILES[0]) not in message
    for filename in CSV_FILES[1:]:
        assert str(data_dir / filename) in message


def test_reset_failure_in_later_source_rolls_back_all_changes(tmp_path, monkeypatch):
    db_path = tmp_path / "draw.db"
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    insert_sentinel(db_path)
    write_complete_source_set(data_dir)

    real_load_csv_file = loader.load_csv_file

    def fail_on_second_file(filepath, conn, *, commit=True, strict=False):
        if filepath.name == "loto2017.csv":
            raise RuntimeError("forced later-file failure")
        return real_load_csv_file(filepath, conn, commit=commit, strict=strict)

    monkeypatch.setattr(loader, "load_csv_file", fail_on_second_file)

    with pytest.raises(RuntimeError, match="forced later-file failure"):
        load_all_db(reset=True, data_dir=data_dir, db_path=db_path)

    assert stored_draw_ids(db_path) == ["sentinel"]


def test_successful_reset_atomically_replaces_existing_data(tmp_path):
    db_path = tmp_path / "draw.db"
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    insert_sentinel(db_path)
    write_complete_source_set(data_dir)

    assert load_all_db(reset=True, data_dir=data_dir, db_path=db_path) == (4, 4)

    assert stored_draw_ids(db_path) == [
        "replacement-1",
        "replacement-2",
        "replacement-3",
        "replacement-4",
    ]


def test_reset_integrity_failure_rolls_back_and_preserves_existing_data(tmp_path):
    db_path = tmp_path / "draw.db"
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    insert_sentinel(db_path)
    write_complete_source_set(data_dir)
    write_draw_csv(data_dir / CSV_FILES[2], "invalid-draw", chance=11)

    with pytest.raises(loader.RowImportError, match=r"invalid-draw.*CHECK constraint failed"):
        load_all_db(reset=True, data_dir=data_dir, db_path=db_path)

    assert stored_draw_ids(db_path) == ["sentinel"]


def test_reset_invalid_numeric_text_rolls_back_and_preserves_existing_data(tmp_path):
    db_path = tmp_path / "draw.db"
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    insert_sentinel(db_path)
    write_complete_source_set(data_dir)
    write_draw_csv(
        data_dir / CSV_FILES[2],
        "invalid-numeric-draw",
        boule_1="not-an-int",
    )

    with pytest.raises(
        loader.RowImportError,
        match=r"invalid-numeric-draw.*Valeur entière invalide: 'not-an-int'",
    ):
        load_all_db(reset=True, data_dir=data_dir, db_path=db_path)

    assert stored_draw_ids(db_path) == ["sentinel"]


def test_reset_missing_draw_identifier_rolls_back_and_preserves_existing_data(tmp_path):
    db_path = tmp_path / "draw.db"
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    insert_sentinel(db_path)
    write_complete_source_set(data_dir)
    write_draw_csv(data_dir / CSV_FILES[2], "   ")

    with pytest.raises(
        loader.RowImportError,
        match=r"identifiant manquant.*annee_numero_de_tirage est obligatoire",
    ):
        load_all_db(reset=True, data_dir=data_dir, db_path=db_path)

    assert stored_draw_ids(db_path) == ["sentinel"]


def test_strict_loader_rejects_non_empty_invalid_float_text(tmp_path):
    csv_path = tmp_path / "invalid-float.csv"
    db_path = tmp_path / "draw.db"
    write_draw_csv(csv_path, "invalid-float-draw", rapport_rang1="not-a-float")

    conn = init_db(db_path)
    with pytest.raises(
        loader.RowImportError,
        match=r"invalid-float-draw.*Valeur décimale invalide: 'not-a-float'",
    ):
        load_csv_file(csv_path, conn, strict=True)
    conn.close()


def test_safe_parsers_keep_defaults_for_direct_callers():
    assert loader.parse_int_safe("not-an-int") == 0
    assert loader.parse_int_safe("", default=7) == 7
    assert loader.parse_float_safe("not-a-float") == 0.0
    assert loader.parse_float_safe("   ", default=1.5) == 1.5
    assert loader.parse_int_safe("   ", default=7, strict=True) == 7
    assert loader.parse_float_safe("", default=1.5, strict=True) == 1.5


def test_direct_loader_keeps_warning_and_skip_compatibility(tmp_path, caplog):
    csv_path = tmp_path / "invalid.csv"
    db_path = tmp_path / "draw.db"
    write_draw_csv(csv_path, "invalid-draw", chance=11)

    conn = init_db(db_path)
    assert load_csv_file(csv_path, conn) == (1, 0)
    conn.close()

    assert stored_draw_ids(db_path) == []
    assert any("Erreur ligne invalid-draw" in record.message for record in caplog.records)


def test_direct_loader_keeps_invalid_numeric_warning_and_skip_compatibility(tmp_path, caplog):
    csv_path = tmp_path / "invalid-numeric.csv"
    db_path = tmp_path / "draw.db"
    write_draw_csv(csv_path, "invalid-numeric-draw", boule_1="not-an-int")

    conn = init_db(db_path)
    assert load_csv_file(csv_path, conn) == (1, 0)
    conn.close()

    assert stored_draw_ids(db_path) == []
    assert any(
        "Erreur ligne invalid-numeric-draw" in record.message
        for record in caplog.records
    )
