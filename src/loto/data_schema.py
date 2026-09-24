"""Canonical, version-aware contract for official FDJ Loto CSV archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Mapping, Sequence


class SourceSchema(str, Enum):
    """Known FDJ archive layouts; boundaries follow archive publication periods."""

    OCT_2008_MAR_2017 = "oct-2008_mar-2017"
    MAR_2017_FEB_2019 = "mar-2017_feb-2019"
    FEB_2019_NOV_2019 = "feb-2019_nov-2019"
    NOV_2019_ONWARD = "nov-2019_onward"


@dataclass(frozen=True)
class SourceContract:
    identifier: SourceSchema
    expected_columns: tuple[str, ...]
    first_draw: date
    last_draw: date | None


@dataclass(frozen=True)
class Provenance:
    source_filename: str
    source_sha256: str
    source_row: int
    source_schema: SourceSchema
    imported_at: datetime


@dataclass(frozen=True)
class PrizeRank:
    rank: int
    winner_count: int | None
    report: Decimal | None
    currency: str | None


@dataclass(frozen=True)
class CodeLoto:
    winner_count: int | None
    report: Decimal | None
    winning_codes: str | None
    currency: str | None


@dataclass(frozen=True)
class SecondDraw:
    numbers: tuple[int, int, int, int, int]
    winning_combination: str | None
    promotion: str | None
    prize_ranks: tuple[PrizeRank, ...]


@dataclass(frozen=True)
class CanonicalDraw:
    draw_id: str
    draw_date: date
    draw_day: str
    claim_deadline: date
    main_numbers: tuple[int, int, int, int, int]
    chance: int
    winning_combination: str
    second_draw: SecondDraw | None
    joker_plus: str | None
    code_loto: CodeLoto | None
    numero_7: str | None
    prize_ranks: tuple[PrizeRank, ...]
    currency: str | None
    provenance: Provenance


class StructuredValidationError(ValueError):
    """A machine-readable source validation failure."""

    def __init__(
        self,
        message: str,
        *,
        filename: str,
        source_row: int | None,
        field: str,
    ) -> None:
        self.message = message
        self.filename = filename
        self.source_row = source_row
        self.field = field
        location = f"{filename}:data-row {source_row}" if source_row else filename
        super().__init__(f"{location}:{field}: {message}")

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "file": self.filename,
            "row": self.source_row,
            "field": self.field,
            "message": self.message,
        }


CORE_COLUMNS = (
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
)


def _rank_columns(first: int, last: int) -> tuple[str, ...]:
    return tuple(
        column
        for rank in range(first, last + 1)
        for column in (f"nombre_de_gagnant_au_rang{rank}", f"rapport_du_rang{rank}")
    )


OLD_COLUMNS = CORE_COLUMNS + _rank_columns(1, 6) + ("numero_jokerplus", "devise")
CODE_COLUMNS = CORE_COLUMNS + _rank_columns(1, 9) + (
    "nombre_de_codes_gagnants",
    "rapport_codes_gagnants",
    "codes_gagnants",
    "numero_jokerplus",
    "devise",
)
SECOND_DRAW_COLUMNS = CODE_COLUMNS[:-2] + (
    "boule_1_second_tirage",
    "boule_2_second_tirage",
    "boule_3_second_tirage",
    "boule_4_second_tirage",
    "boule_5_second_tirage",
    "promotion_second_tirage",
    "combinaison_gagnant_second_tirage_en_ordre_croissant",
    "nombre_de_gagnant_au_rang_1_second_tirage",
    "rapport_du_rang1_second_tirage",
    "nombre_de_gagnant_au_rang_2_second_tirage",
    "rapport_du_rang2_second_tirage",
    "nombre_de_gagnant_au_rang_3_second_tirage",
    "rapport_du_rang3_second_tirage",
    "nombre_de_gagnant_au_rang_4_second_tirage",
    "rapport_du_rang4_second_tirage",
    "numero_7",
    "devise",
)

_CONTRACTS = {
    SourceSchema.OCT_2008_MAR_2017: SourceContract(
        SourceSchema.OCT_2008_MAR_2017, OLD_COLUMNS, date(2008, 10, 6), date(2017, 3, 4)
    ),
    SourceSchema.MAR_2017_FEB_2019: SourceContract(
        SourceSchema.MAR_2017_FEB_2019, CODE_COLUMNS, date(2017, 3, 6), date(2019, 2, 25)
    ),
    SourceSchema.FEB_2019_NOV_2019: SourceContract(
        SourceSchema.FEB_2019_NOV_2019, CODE_COLUMNS, date(2019, 2, 27), date(2019, 11, 2)
    ),
    SourceSchema.NOV_2019_ONWARD: SourceContract(
        SourceSchema.NOV_2019_ONWARD, SECOND_DRAW_COLUMNS, date(2019, 11, 6), None
    ),
}


def schema_contract(schema: SourceSchema) -> SourceContract:
    return _CONTRACTS[schema]


def normalize_header(header: Sequence[str]) -> tuple[str, ...]:
    """Strip BOM/space and only discard harmless trailing empty columns."""
    columns = [value.strip() for value in header]
    if columns:
        columns[0] = columns[0].lstrip("\ufeff")
    while columns and not columns[-1]:
        columns.pop()
    return tuple(columns)


def _header_error(message: str, filename: str) -> StructuredValidationError:
    return StructuredValidationError(
        message, filename=filename, source_row=None, field="header"
    )


def _source_date(value: str, row_number: int, filename: str) -> date:
    try:
        return datetime.strptime((value or "").strip(), "%d/%m/%Y").date()
    except (TypeError, ValueError) as exc:
        raise StructuredValidationError(
            "required date must use DD/MM/YYYY",
            filename=filename,
            source_row=row_number,
            field="date_de_tirage",
        ) from exc


def detect_source_schema(
    header: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    *,
    filename: str = "<header>",
) -> SourceSchema:
    """Identify a layout by exact header and, where shared, its draw-date period."""
    columns = normalize_header(header)
    candidates = [c for c in _CONTRACTS.values() if c.expected_columns == columns]
    if not candidates:
        raise _header_error("unknown source schema", filename)
    if len(candidates) == 1:
        return candidates[0].identifier
    if not rows:
        raise _header_error(
            "ambiguous source schema: shared header has no content", filename
        )

    dates = []
    for index, row in enumerate(rows, 1):
        try:
            dates.append(_source_date(row.get("date_de_tirage", ""), index, filename))
        except StructuredValidationError:
            # Detection identifies the period from parseable rows. Canonical
            # row validation remains responsible for reporting malformed dates.
            continue
    if not dates:
        raise _header_error(
            "ambiguous source schema: no parseable draw dates", filename
        )
    matching = [
        contract
        for contract in candidates
        if all(
            draw_date >= contract.first_draw
            and (contract.last_draw is None or draw_date <= contract.last_draw)
            for draw_date in dates
        )
    ]
    if len(matching) != 1:
        raise _header_error(
            "ambiguous source schema: rows cross or miss known periods", filename
        )
    return matching[0].identifier


def parse_canonical_draw(
    row: Mapping[str, str],
    *,
    schema: SourceSchema,
    source_filename: str,
    source_sha256: str,
    source_row: int,
    imported_at: datetime,
) -> CanonicalDraw:
    """Lazily preserve the historical parser export without an import cycle."""
    from loto.data_parsing import parse_canonical_draw as parse

    return parse(
        row,
        schema=schema,
        source_filename=source_filename,
        source_sha256=source_sha256,
        source_row=source_row,
        imported_at=imported_at,
    )


__all__ = [
    "CanonicalDraw", "CodeLoto", "PrizeRank", "Provenance", "SecondDraw",
    "SourceContract", "SourceSchema", "StructuredValidationError",
    "detect_source_schema", "normalize_header", "parse_canonical_draw", "schema_contract",
]
