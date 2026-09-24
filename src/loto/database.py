"""
Module de gestion de la base de données SQLite.
Schema unique pour les tirages du Loto français.
"""

import sqlite3
from os import PathLike
from pathlib import Path
from loto.config import DB_PATH


DRAW_MIGRATION_COLUMNS = {
    "source_file_id": "INTEGER REFERENCES source_files(id)",
    "source_row": "INTEGER CHECK(source_row > 0)",
    **{
        f"boule_{index}_second_tirage":
            f"INTEGER CHECK(boule_{index}_second_tirage BETWEEN 1 AND 49)"
        for index in range(1, 6)
    },
    "promotion_second_tirage": "TEXT",
    "combinaison_second_tirage": "TEXT",
    **{
        column: definition
        for rank in range(1, 5)
        for column, definition in (
            (f"gagnants_rang{rank}_second_tirage", "INTEGER CHECK(" +
             f"gagnants_rang{rank}_second_tirage >= 0)"),
            (f"rapport_rang{rank}_second_tirage", "REAL CHECK(" +
             f"rapport_rang{rank}_second_tirage >= 0)"),
        )
    },
    "numero_7": "TEXT",
}


def get_connection(db_path: str | PathLike[str] | None = None) -> sqlite3.Connection:
    """Retourne une connexion SQLite."""
    path = Path(db_path or DB_PATH).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


SOURCE_FILES_SCHEMA = """
    CREATE TABLE IF NOT EXISTS source_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        sha256 TEXT NOT NULL CHECK(length(sha256) = 64),
        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
        source_schema TEXT NOT NULL,
        imported_at TEXT NOT NULL,
        rows_read INTEGER NOT NULL CHECK(rows_read >= 0),
        rows_accepted INTEGER NOT NULL CHECK(rows_accepted >= 0),
        rows_rejected INTEGER NOT NULL CHECK(rows_rejected >= 0),
        rows_duplicates INTEGER NOT NULL CHECK(rows_duplicates >= 0),
        UNIQUE(filename, sha256)
    )
"""

DRAW_SCHEMA = """
    CREATE TABLE IF NOT EXISTS tirages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        annee_numero TEXT UNIQUE NOT NULL,
        jour_tirage TEXT NOT NULL,
        date_tirage TEXT NOT NULL,
        date_forclusion TEXT,
        boule_1 INTEGER NOT NULL CHECK(boule_1 BETWEEN 1 AND 49),
        boule_2 INTEGER NOT NULL CHECK(boule_2 BETWEEN 1 AND 49),
        boule_3 INTEGER NOT NULL CHECK(boule_3 BETWEEN 1 AND 49),
        boule_4 INTEGER NOT NULL CHECK(boule_4 BETWEEN 1 AND 49),
        boule_5 INTEGER NOT NULL CHECK(boule_5 BETWEEN 1 AND 49),
        numero_chance INTEGER NOT NULL CHECK(numero_chance BETWEEN 1 AND 10),
        combinaison_croissante TEXT,
        gagnants_rang1 INTEGER, rapport_rang1 REAL,
        gagnants_rang2 INTEGER, rapport_rang2 REAL,
        gagnants_rang3 INTEGER, rapport_rang3 REAL,
        gagnants_rang4 INTEGER, rapport_rang4 REAL,
        gagnants_rang5 INTEGER, rapport_rang5 REAL,
        gagnants_rang6 INTEGER, rapport_rang6 REAL,
        gagnants_rang7 INTEGER, rapport_rang7 REAL,
        gagnants_rang8 INTEGER, rapport_rang8 REAL,
        gagnants_rang9 INTEGER, rapport_rang9 REAL,
        nombre_codes INTEGER, rapport_codes REAL, codes_gagnants TEXT,
        numero_jokerplus TEXT, devise TEXT,
        source_file_id INTEGER REFERENCES source_files(id),
        source_row INTEGER CHECK(source_row > 0),
        boule_1_second_tirage INTEGER, boule_2_second_tirage INTEGER,
        boule_3_second_tirage INTEGER, boule_4_second_tirage INTEGER,
        boule_5_second_tirage INTEGER,
        promotion_second_tirage TEXT, combinaison_second_tirage TEXT,
        gagnants_rang1_second_tirage INTEGER, rapport_rang1_second_tirage REAL,
        gagnants_rang2_second_tirage INTEGER, rapport_rang2_second_tirage REAL,
        gagnants_rang3_second_tirage INTEGER, rapport_rang3_second_tirage REAL,
        gagnants_rang4_second_tirage INTEGER, rapport_rang4_second_tirage REAL,
        numero_7 TEXT
    )
"""


def initialize_schema(conn: sqlite3.Connection, *, commit: bool = True) -> None:
    """Initialize or migrate schema, optionally within the caller's transaction."""
    cursor = conn.cursor()
    cursor.execute(SOURCE_FILES_SCHEMA)
    cursor.execute(DRAW_SCHEMA)
    existing = {
        row[1] for row in cursor.execute("PRAGMA table_info(tirages)").fetchall()
    }
    for column, definition in DRAW_MIGRATION_COLUMNS.items():
        if column not in existing:
            cursor.execute(f"ALTER TABLE tirages ADD COLUMN {column} {definition}")

    source_columns = {
        row[1] for row in cursor.execute("PRAGMA table_info(source_files)").fetchall()
    }
    if "rows_duplicates" not in source_columns:
        cursor.execute(
            "ALTER TABLE source_files ADD COLUMN rows_duplicates INTEGER NOT NULL DEFAULT 0"
        )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_tirage_source ON tirages(source_file_id, source_row)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tirage_date ON tirages(date_tirage)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tirage_annee ON tirages(annee_numero)")
    if commit:
        conn.commit()


def init_db(db_path: str | PathLike[str] | None = None) -> sqlite3.Connection:
    """Initialize and commit the database schema for normal callers."""
    conn = get_connection(db_path)
    initialize_schema(conn)
    return conn


def clear_db(conn: sqlite3.Connection, *, commit: bool = True) -> None:
    """Supprime toutes les données existantes."""
    conn.execute("DELETE FROM tirages")
    conn.execute("DELETE FROM source_files")
    if commit:
        conn.commit()
