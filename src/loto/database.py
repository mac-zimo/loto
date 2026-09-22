"""
Module de gestion de la base de données SQLite.
Schema unique pour les tirages du Loto français.
"""

import sqlite3
from pathlib import Path
from src.loto.config import DB_PATH


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Retourne une connexion SQLite."""
    path = db_path or str(DB_PATH)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialise la base de données avec le schema."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
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

            -- Rang 1
            gagnants_rang1 INTEGER DEFAULT 0,
            rapport_rang1 REAL DEFAULT 0,

            -- Rang 2-5 (plus tard)
            gagnants_rang2 INTEGER DEFAULT 0,
            rapport_rang2 REAL DEFAULT 0,
            gagnants_rang3 INTEGER DEFAULT 0,
            rapport_rang3 REAL DEFAULT 0,
            gagnants_rang4 INTEGER DEFAULT 0,
            rapport_rang4 REAL DEFAULT 0,
            gagnants_rang5 INTEGER DEFAULT 0,
            rapport_rang5 REAL DEFAULT 0,

            -- Rang 6-9 (nouveau loto)
            gagnants_rang6 INTEGER DEFAULT 0,
            rapport_rang6 REAL DEFAULT 0,
            gagnants_rang7 INTEGER DEFAULT 0,
            rapport_rang7 REAL DEFAULT 0,
            gagnants_rang8 INTEGER DEFAULT 0,
            rapport_rang8 REAL DEFAULT 0,
            gagnants_rang9 INTEGER DEFAULT 0,
            rapport_rang9 REAL DEFAULT 0,

            -- Codes gagnants
            nombre_codes INTEGER DEFAULT 0,
            rapport_codes REAL DEFAULT 0,
            codes_gagnants TEXT,

            numero_jokerplus TEXT,
            devise TEXT DEFAULT 'EUR'
        );

        CREATE INDEX IF NOT EXISTS idx_tirage_date ON tirages(date_tirage);
        CREATE INDEX IF NOT EXISTS idx_tirage_annee ON tirages(annee_numero);
    """)

    conn.commit()
    return conn


def clear_db(conn: sqlite3.Connection) -> None:
    """Supprime toutes les données existantes."""
    conn.execute("DELETE FROM tirages")
    conn.commit()
