"""
Charge les fichiers CSV du Loto dans la base SQLite.
Gère les différents formats (ancien/nouveau loto).
"""

import logging
import sqlite3
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from loto.config import CSV_FILES, DATA_DIR, DB_PATH
from loto.database import clear_db, get_connection, init_db, initialize_schema
from loto.data_persistence import (
    assert_reset_invariants,
    persist_draw,
    persist_source_file,
    update_source_counts,
)
from loto.data_source import (
    DuplicateDrawIdError,
    PreflightSource,
    SourceInspection,
    inspect_source,
    preflight_source_set,
)
from loto.data_schema import (
    StructuredValidationError,
    parse_canonical_draw,
)

logger = logging.getLogger(__name__)


class RowImportError(RuntimeError):
    """Raised when strict CSV loading cannot import a row."""

    def __init__(self, error: StructuredValidationError, draw_id: str = "") -> None:
        self.validation_error = error
        identifier = draw_id or "identifiant manquant"
        super().__init__(f"{identifier}: {error}")

    def as_dict(self) -> dict[str, str | int | None]:
        return self.validation_error.as_dict()


def parse_int_safe(val: str, default=0, *, strict: bool = False) -> int:
    """Parse une chaîne en entier, retourne default si échec hors mode strict."""
    try:
        return int(float(val.replace(',', '.')))
    except (ValueError, TypeError) as exc:
        if strict and isinstance(val, str) and val.strip():
            raise ValueError(f"Valeur entière invalide: {val!r}") from exc
        return default


def parse_float_safe(val: str, default=0.0, *, strict: bool = False) -> float:
    """Parse une chaîne en float, retourne default si échec hors mode strict."""
    try:
        return float(val.replace(',', '.'))
    except (ValueError, TypeError) as exc:
        if strict and isinstance(val, str) and val.strip():
            raise ValueError(f"Valeur décimale invalide: {val!r}") from exc
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


def load_csv_file(
    filepath: Path,
    conn: sqlite3.Connection,
    *,
    commit: bool = True,
    strict: bool = False,
    _inspection: SourceInspection | PreflightSource | None = None,
) -> tuple[int, int]:
    """
    Charge un fichier CSV dans la base de données.

    En mode strict, toute erreur de ligne interrompt le chargement afin que
    l'appelant puisse annuler sa transaction. Le mode par défaut signale puis
    ignore ces lignes pour rester compatible avec les imports historiques.

    Returns: (total_lus, insertes)
    """
    prepared = _inspection if isinstance(_inspection, PreflightSource) else None
    if prepared:
        inspection = prepared.inspection
    else:
        inspection = (
            _inspection
            if isinstance(_inspection, SourceInspection)
            else inspect_source(filepath)
        )
    imported_at = prepared.imported_at if prepared else datetime.now(timezone.utc)
    accepted = list(prepared.canonical_draws) if prepared else []
    rejected = len(prepared.row_errors) if prepared else 0
    if not prepared:
        for inspected_row in inspection.rows:
            if inspected_row.error is not None:
                if strict:
                    raise RowImportError(inspected_row.error) from inspected_row.error
                logger.warning("Validation CSV: %s", inspected_row.error.as_dict())
                rejected += 1
                continue
            row = inspected_row.values
            assert row is not None
            try:
                accepted.append(parse_canonical_draw(
                    row,
                    schema=inspection.schema,
                    source_filename=filepath.name,
                    source_sha256=inspection.sha256,
                    source_row=inspected_row.source_row,
                    imported_at=imported_at,
                ))
            except StructuredValidationError as error:
                if strict:
                    raise RowImportError(
                        error, row.get("annee_numero_de_tirage", "").strip()
                    ) from error
                logger.warning("Validation CSV: %s", error.as_dict())
                rejected += 1

    if not accepted:
        raise StructuredValidationError(
            "source contains zero accepted rows",
            filename=filepath.name,
            source_row=None,
            field="rows",
        )

    source_file_id = persist_source_file(
        conn,
        filename=filepath.name,
        sha256=inspection.sha256,
        size_bytes=inspection.size_bytes,
        schema=inspection.schema,
        imported_at=imported_at.isoformat(),
        rows_read=len(inspection.rows),
    )
    inserted = sum(
        persist_draw(conn, draw, source_file_id, reject_conflict=strict)
        for draw in accepted
    )
    update_source_counts(
        conn,
        source_file_id,
        accepted=len(accepted),
        rejected=rejected,
        duplicates=len(accepted) - inserted,
    )

    if commit:
        conn.commit()
    return len(inspection.rows), inserted


def load_all_db(
    reset: bool = False,
    data_dir: str | PathLike[str] | None = None,
    db_path: str | PathLike[str] | None = None,
) -> tuple[int, int]:
    """
    Charge tous les CSV dans la base de données.

    Args:
        reset: Si True, supprime les données existantes avant rechargement.
        data_dir: Répertoire contenant les fichiers CSV historiques.
        db_path: Chemin de la base SQLite.

    Returns:
        (total_lus, total_inseres)
    """
    csv_dir = Path(data_dir or DATA_DIR)
    source_files = []
    missing_files = []
    for csv_file in CSV_FILES:
        filepath = csv_dir / csv_file
        if not filepath.is_file():
            missing_files.append(filepath)
        else:
            source_files.append(filepath)

    # Validate the source set before opening SQLite: a bad --data-dir must not
    # create a database or destroy data in an existing one.
    if reset and missing_files:
        logger.error(
            "Reset annulé; fichiers source manquants: %s",
            ", ".join(str(path) for path in missing_files),
        )
        return 0, 0
    for filepath in missing_files:
        logger.warning(f"Fichier non trouvé: {filepath}")
    if not source_files:
        return 0, 0

    prepared_sources = None
    if reset:
        inspections = [inspect_source(filepath) for filepath in source_files]
        try:
            prepared_sources = preflight_source_set(inspections)
        except DuplicateDrawIdError:
            raise
        except StructuredValidationError as error:
            if error.source_row is None:
                raise
            draw_id = ""
            for inspection in inspections:
                if inspection.path.name != error.filename:
                    continue
                row = next(
                    (item for item in inspection.rows if item.source_row == error.source_row),
                    None,
                )
                if row and row.values:
                    draw_id = row.values.get("annee_numero_de_tirage", "").strip()
                break
            raise RowImportError(error, draw_id) from error

    conn = get_connection(db_path or DB_PATH) if reset else init_db(db_path or DB_PATH)
    total_loaded = 0
    total_inserted = 0

    try:
        if reset:
            conn.execute("BEGIN")
            initialize_schema(conn, commit=False)
            clear_db(conn, commit=False)
            logger.info("Base de données vidée (transaction en cours).")

        for index, filepath in enumerate(source_files):
            logger.info(f"Chargement de {filepath.name}...")
            loaded, inserted = load_csv_file(
                filepath,
                conn,
                commit=not reset,
                strict=reset,
                _inspection=prepared_sources[index] if prepared_sources else None,
            )
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

        if reset:
            accepted = sum(
                len(source.canonical_draws) for source in prepared_sources or ()
            )
            assert_reset_invariants(
                conn,
                loaded=total_loaded,
                accepted=accepted,
                inserted=total_inserted,
            )
            conn.commit()
    except Exception:
        if reset:
            conn.rollback()
        raise
    finally:
        conn.close()

    return total_loaded, total_inserted


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    load_all_db(reset=True)
