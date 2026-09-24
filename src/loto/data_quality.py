"""Deterministic data-quality reporting for canonical imports."""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass
from os import PathLike


@dataclass(frozen=True)
class FileQuality:
    filename: str
    sha256: str
    size_bytes: int
    source_schema: str
    imported_at: str
    rows_read: int
    rows_accepted: int
    rows_rejected: int
    duplicates: int


@dataclass(frozen=True)
class DataQualitySummary:
    files: tuple[FileQuality, ...]
    rows_read: int
    rows_accepted: int
    rows_rejected: int
    duplicates: int
    schema_counts: dict[str, int]
    min_date: str | None
    max_date: str | None
    nullable_field_counts: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["files"] = [asdict(file) for file in self.files]
        return result


NULLABLE_COLUMNS = (
    "date_forclusion",
    *(column for rank in range(1, 10) for column in (
        f"gagnants_rang{rank}", f"rapport_rang{rank}"
    )),
    "nombre_codes",
    "rapport_codes",
    "codes_gagnants",
    "numero_jokerplus",
    *(f"boule_{index}_second_tirage" for index in range(1, 6)),
    "promotion_second_tirage",
    "combinaison_second_tirage",
    *(column for rank in range(1, 5) for column in (
        f"gagnants_rang{rank}_second_tirage",
        f"rapport_rang{rank}_second_tirage",
    )),
    "numero_7",
)


def data_quality_summary(
    db_path: str | PathLike[str] | None = None,
    *,
    connection: sqlite3.Connection | None = None,
) -> DataQualitySummary:
    """Read a stable summary; callers may supply an in-transaction connection."""
    if connection is None and db_path is None:
        raise ValueError("db_path or connection is required")
    owned = connection is None
    if connection is None:
        assert db_path is not None
        conn = sqlite3.connect(db_path)
    else:
        conn = connection
    try:
        source_rows = conn.execute(
            """SELECT filename, sha256, size_bytes, source_schema, imported_at,
                      rows_read, rows_accepted, rows_rejected, rows_duplicates
               FROM source_files ORDER BY filename, sha256"""
        ).fetchall()
        files = tuple(FileQuality(*row) for row in source_rows)
        date_range = conn.execute(
            "SELECT MIN(date_tirage), MAX(date_tirage) FROM tirages"
        ).fetchone()
        nullable = {
            column: conn.execute(
                f"SELECT COUNT(*) FROM tirages WHERE {column} IS NULL"
            ).fetchone()[0]
            for column in NULLABLE_COLUMNS
        }
        schema_counts = {
            schema: count
            for schema, count in conn.execute(
                """SELECT source_schema, SUM(rows_accepted)
                   FROM source_files GROUP BY source_schema ORDER BY source_schema"""
            )
        }
        return DataQualitySummary(
            files=files,
            rows_read=sum(file.rows_read for file in files),
            rows_accepted=sum(file.rows_accepted for file in files),
            rows_rejected=sum(file.rows_rejected for file in files),
            duplicates=sum(file.duplicates for file in files),
            schema_counts=schema_counts,
            min_date=date_range[0],
            max_date=date_range[1],
            nullable_field_counts=nullable,
        )
    finally:
        if owned:
            conn.close()
