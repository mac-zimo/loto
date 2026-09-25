"""SQLite persistence for validated canonical Loto records."""

from __future__ import annotations

import sqlite3
from decimal import Decimal

from loto.data_schema import CanonicalDraw, SourceSchema


class ResetInvariantError(RuntimeError):
    """Raised when a reset transaction does not match its preflight contract."""


MAIN_COLUMNS = [
    "annee_numero", "jour_tirage", "date_tirage", "date_forclusion",
    "boule_1", "boule_2", "boule_3", "boule_4", "boule_5", "numero_chance",
    "combinaison_croissante",
]
RANK_COLUMNS = [
    column
    for rank in range(1, 10)
    for column in (f"gagnants_rang{rank}", f"rapport_rang{rank}")
]
CODE_COLUMNS = ["nombre_codes", "rapport_codes", "codes_gagnants"]
SECOND_COLUMNS = [
    *(f"boule_{index}_second_tirage" for index in range(1, 6)),
    "promotion_second_tirage", "combinaison_second_tirage",
    *(
        column
        for rank in range(1, 5)
        for column in (
            f"gagnants_rang{rank}_second_tirage",
            f"rapport_rang{rank}_second_tirage",
        )
    ),
]
OTHER_COLUMNS = [
    "numero_jokerplus", "numero_7", "devise", "source_file_id", "source_row"
]
DRAW_COLUMNS = MAIN_COLUMNS + RANK_COLUMNS + CODE_COLUMNS + SECOND_COLUMNS + OTHER_COLUMNS


def _number(value: Decimal | None) -> float | None:
    return float(value) if value is not None else None


def persist_source_file(
    conn: sqlite3.Connection,
    *,
    filename: str,
    sha256: str,
    size_bytes: int,
    schema: SourceSchema,
    imported_at: str,
    rows_read: int,
) -> int:
    conn.execute(
        """INSERT INTO source_files (
               filename, sha256, size_bytes, source_schema, imported_at,
               rows_read, rows_accepted, rows_rejected, rows_duplicates
           ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, 0)
           ON CONFLICT(filename, sha256) DO UPDATE SET
               size_bytes=excluded.size_bytes,
               source_schema=excluded.source_schema,
               rows_read=excluded.rows_read""",
        (filename, sha256, size_bytes, schema.value, imported_at, rows_read),
    )
    row = conn.execute(
        "SELECT id FROM source_files WHERE filename = ? AND sha256 = ?",
        (filename, sha256),
    ).fetchone()
    assert row is not None
    return int(row[0])


def update_source_counts(
    conn: sqlite3.Connection,
    source_file_id: int,
    *,
    accepted: int,
    rejected: int,
    duplicates: int,
) -> None:
    conn.execute(
        """UPDATE source_files
           SET rows_accepted = ?, rows_rejected = ?, rows_duplicates = ?
           WHERE id = ?""",
        (accepted, rejected, duplicates, source_file_id),
    )


def _draw_values(draw: CanonicalDraw, source_file_id: int) -> list[object]:
    values: list[object] = [
        draw.draw_id,
        draw.draw_day,
        draw.draw_date.isoformat(),
        draw.claim_deadline.isoformat(),
        *draw.main_numbers,
        draw.chance,
        draw.winning_combination,
    ]
    ranks = {rank.rank: rank for rank in draw.prize_ranks}
    for rank_number in range(1, 10):
        rank = ranks.get(rank_number)
        values.extend(
            [
                rank.winner_count if rank else None,
                _number(rank.report) if rank else None,
            ]
        )
    if draw.code_loto:
        values.extend(
            [
                draw.code_loto.winner_count,
                _number(draw.code_loto.report),
                draw.code_loto.winning_codes,
            ]
        )
    else:
        values.extend([None, None, None])
    if draw.second_draw:
        values.extend(draw.second_draw.numbers)
        values.extend([draw.second_draw.promotion, draw.second_draw.winning_combination])
        for rank in draw.second_draw.prize_ranks:
            values.extend([rank.winner_count, _number(rank.report)])
    else:
        values.extend([None] * len(SECOND_COLUMNS))
    values.extend(
        [
            draw.joker_plus,
            draw.numero_7,
            draw.currency,
            source_file_id,
            draw.provenance.source_row,
        ]
    )
    return values


def persist_draw(
    conn: sqlite3.Connection,
    draw: CanonicalDraw,
    source_file_id: int,
    *,
    reject_conflict: bool = False,
) -> bool:
    placeholders = ", ".join("?" for _ in DRAW_COLUMNS)
    conflict_clause = "" if reject_conflict else " ON CONFLICT(annee_numero) DO NOTHING"
    cursor = conn.execute(
        f"INSERT INTO tirages ({', '.join(DRAW_COLUMNS)}) "
        f"VALUES ({placeholders}){conflict_clause}",
        _draw_values(draw, source_file_id),
    )
    if cursor.rowcount > 0:
        return True
    # A pre-contract database can already contain the draw but no provenance.
    # Backfill only that missing link; never replace provenance recorded earlier.
    conn.execute(
        """UPDATE tirages SET source_file_id = ?, source_row = ?
           WHERE annee_numero = ? AND source_file_id IS NULL""",
        (source_file_id, draw.provenance.source_row, draw.draw_id),
    )
    return False


def assert_reset_invariants(
    conn: sqlite3.Connection,
    *,
    loaded: int,
    accepted: int,
    inserted: int,
) -> None:
    """Validate reset cardinality and provenance before transaction commit."""
    draw_count, missing_provenance, duplicate_ids = conn.execute(
        """SELECT COUNT(*),
                  SUM(CASE WHEN source_file_id IS NULL OR source_row IS NULL
                           THEN 1 ELSE 0 END),
                  COUNT(*) - COUNT(DISTINCT annee_numero)
           FROM tirages"""
    ).fetchone()
    source_accepted, source_duplicates = conn.execute(
        """SELECT COALESCE(SUM(rows_accepted), 0),
                  COALESCE(SUM(rows_duplicates), 0)
           FROM source_files"""
    ).fetchone()
    actual = {
        "loaded": loaded,
        "accepted": accepted,
        "inserted": inserted,
        "draw_count": draw_count,
        "missing_provenance": missing_provenance,
        "source_accepted": source_accepted,
        "source_duplicates": source_duplicates,
        "duplicate_ids": duplicate_ids,
    }
    if not (
        loaded == accepted == inserted == draw_count == source_accepted
        and missing_provenance == 0
        and source_duplicates == 0
        and duplicate_ids == 0
    ):
        raise ResetInvariantError(f"reset persistence invariants failed: {actual}")
