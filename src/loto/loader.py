"""
Charge les fichiers CSV du Loto dans la base SQLite.
Gère les différents formats (ancien/nouveau loto).
"""

import csv
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from src.loto.config import CSV_FILES, PROJECT_ROOT, DATA_DIR
from src.loto.database import get_connection, init_db, clear_db

logger = logging.getLogger(__name__)


def parse_int_safe(val: str, default=0) -> int:
    """Parse une chaîne en entier, retourne default si échec."""
    try:
        return int(float(val.replace(',', '.')))
    except (ValueError, TypeError):
        return default


def parse_float_safe(val: str, default=0.0) -> float:
    """Parse une chaîne en float, retourne default si échec."""
    try:
        return float(val.replace(',', '.'))
    except (ValueError, TypeError):
        return default


def normalize_date(val: str) -> str:
    """Normalise une date FDJ en ISO (YYYY-MM-DD) pour un tri fiable."""
    value = (val or "").strip()
    if not value:
        return ""
    for fmt in ("%d/%m/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"Format de date non reconnu: {value!r}")


def load_csv_file(filepath: Path, conn: sqlite3.Connection) -> tuple[int, int]:
    """
    Charge un fichier CSV dans la base de données.

    Returns: (total_lus, insertes)
    """
    inserted = 0
    skipped = 0

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read().rstrip()
        reader = csv.DictReader(raw.splitlines(), delimiter=";")

        for row in reader:
            # Supprimer les clés vides créées par un ';' final dans le header
            row = {k: v for k, v in row.items() if k is not None and k.strip()}
            try:
                annee_numero = row.get("annee_numero_de_tirage", "").strip()
                if not annee_numero:
                    skipped += 1
                    continue

                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO tirages (
                        annee_numero, jour_tirage, date_tirage, date_forclusion,
                        boule_1, boule_2, boule_3, boule_4, boule_5, numero_chance,
                        combinaison_croissante,
                        gagnants_rang1, rapport_rang1,
                        gagnants_rang2, rapport_rang2,
                        gagnants_rang3, rapport_rang3,
                        gagnants_rang4, rapport_rang4,
                        gagnants_rang5, rapport_rang5,
                        gagnants_rang6, rapport_rang6,
                        gagnants_rang7, rapport_rang7,
                        gagnants_rang8, rapport_rang8,
                        gagnants_rang9, rapport_rang9,
                        nombre_codes, rapport_codes, codes_gagnants,
                        numero_jokerplus, devise
                    ) VALUES (
                        ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?,
                        ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?, ?, ?
                    )
                """, (
                    annee_numero,
                    row.get("jour_de_tirage", "").strip(),
                    normalize_date(row.get("date_de_tirage", "")),
                    normalize_date(row.get("date_de_forclusion", "")),
                    parse_int_safe(row.get("boule_1", "0")),
                    parse_int_safe(row.get("boule_2", "0")),
                    parse_int_safe(row.get("boule_3", "0")),
                    parse_int_safe(row.get("boule_4", "0")),
                    parse_int_safe(row.get("boule_5", "0")),
                    parse_int_safe(row.get("numero_chance", "0")),
                    row.get("combinaison_gagnante_en_ordre_croissant", ""),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang1", "0")),
                    parse_float_safe(row.get("rapport_du_rang1", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang2", "0")),
                    parse_float_safe(row.get("rapport_du_rang2", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang3", "0")),
                    parse_float_safe(row.get("rapport_du_rang3", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang4", "0")),
                    parse_float_safe(row.get("rapport_du_rang4", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang5", "0")),
                    parse_float_safe(row.get("rapport_du_rang5", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang6", "0")),
                    parse_float_safe(row.get("rapport_du_rang6", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang7", "0")),
                    parse_float_safe(row.get("rapport_du_rang7", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang8", "0")),
                    parse_float_safe(row.get("rapport_du_rang8", "0")),
                    parse_int_safe(row.get("nombre_de_gagnant_au_rang9", "0")),
                    parse_float_safe(row.get("rapport_du_rang9", "0")),
                    parse_int_safe(row.get("nombre_de_codes_gagnants", "0") or row.get("nombre_codes_gagnants", "0")),
                    parse_float_safe(row.get("rapport_codes_gagnants", "0")),
                    row.get("codes_gagnants", ""),
                    row.get("numero_jokerplus", ""),
                    row.get("devise", "EUR"),
                ))

                if cursor.rowcount > 0:
                    inserted += 1
                else:
                    skipped += 1  # déjà présent (INSERT OR IGNORE)

            except Exception as e:
                logger.warning(f"Erreur ligne {annee_numero}: {e}")
                skipped += 1

    conn.commit()
    total = inserted + skipped
    return total, inserted


def load_all_db(reset: bool = False) -> tuple[int, int]:
    """
    Charge tous les CSV dans la base de données.

    Args:
        reset: Si True, supprime les données existantes avant rechargement.

    Returns:
        (total_lus, total_inseres)
    """
    conn = init_db()

    if reset:
        clear_db(conn)
        logger.info("Base de données vidée.")

    total_loaded = 0
    total_inserted = 0

    csv_dir = PROJECT_ROOT
    for csv_file in CSV_FILES:
        filepath = csv_dir / csv_file
        if not filepath.exists():
            logger.warning(f"Fichier non trouvé: {filepath}")
            continue

        logger.info(f"Chargement de {csv_file}...")
        loaded, inserted = load_csv_file(filepath, conn)
        total_loaded += loaded
        total_inserted += inserted
        logger.info(
            f"  -> {loaded} lignes lues, {inserted} insérées "
            f"({loaded - inserted} doublons ou ignorées)"
        )

    # Afficher le comptage final
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tirages")
    count = cursor.fetchone()[0]
    logger.info(f"Total dans la BD: {count} tirages")

    conn.close()
    return total_loaded, total_inserted


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    load_all_db(reset=True)
