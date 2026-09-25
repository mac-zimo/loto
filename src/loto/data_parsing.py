"""Strict conversion of an FDJ source row into the canonical contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping

from loto.data_numeric import bounded_integer as _bounded_integer
from loto.data_numeric import monetary_decimal as _decimal
from loto.data_numeric import winner_count
from loto.data_schema import (
    CanonicalDraw,
    CodeLoto,
    PrizeRank,
    Provenance,
    SecondDraw,
    SourceSchema,
    StructuredValidationError,
    schema_contract,
)


def _fail(
    message: str, *, filename: str, source_row: int, field: str
) -> StructuredValidationError:
    return StructuredValidationError(
        message, filename=filename, source_row=source_row, field=field
    )


def _required_text(
    row: Mapping[str, str], field: str, filename: str, source_row: int
) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise _fail(
            "required value is blank",
            filename=filename,
            source_row=source_row,
            field=field,
        )
    return value


def _optional_text(row: Mapping[str, str], field: str) -> str | None:
    value = (row.get(field) or "").strip()
    return value or None


def _date(
    row: Mapping[str, str], field: str, filename: str, source_row: int
):
    value = _required_text(row, field, filename, source_row)
    try:
        return datetime.strptime(value, "%d/%m/%Y").date()
    except ValueError as exc:
        raise _fail(
            "required date must use DD/MM/YYYY",
            filename=filename,
            source_row=source_row,
            field=field,
        ) from exc


def _rank(
    row: Mapping[str, str],
    rank: int,
    filename: str,
    source_row: int,
    currency: str | None,
    *,
    second: bool = False,
) -> PrizeRank:
    if second:
        winners_field = f"nombre_de_gagnant_au_rang_{rank}_second_tirage"
        report_field = f"rapport_du_rang{rank}_second_tirage"
    else:
        winners_field = f"nombre_de_gagnant_au_rang{rank}"
        report_field = f"rapport_du_rang{rank}"
    winners = winner_count(row, winners_field, filename, source_row)
    report = _decimal(row, report_field, filename, source_row)
    return PrizeRank(rank, winners, report, currency if report is not None else None)


def _validate_numbers(
    numbers: list[int], filename: str, source_row: int, field: str
) -> tuple[int, int, int, int, int]:
    if len(set(numbers)) != 5:
        raise _fail(
            "five numbers must be distinct",
            filename=filename,
            source_row=source_row,
            field=field,
        )
    ordered = tuple(sorted(numbers))
    return ordered  # type: ignore[return-value]


def _second_draw(
    row: Mapping[str, str], filename: str, source_row: int, currency: str | None
) -> SecondDraw | None:
    ball_fields = [f"boule_{index}_second_tirage" for index in range(1, 6)]
    associated_fields = [
        "promotion_second_tirage",
        "combinaison_gagnant_second_tirage_en_ordre_croissant",
        *(
            field
            for rank in range(1, 5)
            for field in (
                f"nombre_de_gagnant_au_rang_{rank}_second_tirage",
                f"rapport_du_rang{rank}_second_tirage",
            )
        ),
    ]
    ball_values = [(row.get(field) or "").strip() for field in ball_fields]
    group_values = ball_values + [
        (row.get(field) or "").strip() for field in associated_fields
    ]
    if not any(group_values):
        return None
    if not all(ball_values):
        raise _fail(
            "second draw must have all five balls or be fully absent",
            filename=filename,
            source_row=source_row,
            field="second_draw",
        )
    numbers = [
        _bounded_integer(row, field, filename, source_row, 1, 49)
        for field in ball_fields
    ]
    return SecondDraw(
        numbers=_validate_numbers(numbers, filename, source_row, "second_draw"),
        winning_combination=_optional_text(
            row, "combinaison_gagnant_second_tirage_en_ordre_croissant"
        ),
        promotion=_optional_text(row, "promotion_second_tirage"),
        prize_ranks=tuple(
            _rank(row, rank, filename, source_row, currency, second=True)
            for rank in range(1, 5)
        ),
    )


def parse_canonical_draw(
    row: Mapping[str, str],
    *,
    schema: SourceSchema,
    source_filename: str,
    source_sha256: str,
    source_row: int,
    imported_at: datetime,
) -> CanonicalDraw:
    """Validate and canonicalize one source data row (header excluded)."""
    if source_row < 1:
        raise _fail(
            "source row must be one-based with the header excluded",
            filename=source_filename,
            source_row=source_row,
            field="source_row",
        )
    if len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise _fail(
            "SHA-256 must be 64 lowercase hexadecimal characters",
            filename=source_filename,
            source_row=source_row,
            field="source_sha256",
        )
    if imported_at.tzinfo is None or imported_at.utcoffset() != timezone.utc.utcoffset(imported_at):
        raise _fail(
            "imported_at must be UTC-aware",
            filename=source_filename,
            source_row=source_row,
            field="imported_at",
        )

    draw_date = _date(row, "date_de_tirage", source_filename, source_row)
    contract = schema_contract(schema)
    if draw_date < contract.first_draw or (
        contract.last_draw is not None and draw_date > contract.last_draw
    ):
        raise _fail(
            f"date is outside source schema period {schema.value}",
            filename=source_filename,
            source_row=source_row,
            field="date_de_tirage",
        )
    currency = _optional_text(row, "devise")
    numbers = [
        _bounded_integer(row, f"boule_{index}", source_filename, source_row, 1, 49)
        for index in range(1, 6)
    ]
    rank_count = 6 if schema is SourceSchema.OCT_2008_MAR_2017 else 9
    code_loto = None
    if schema is not SourceSchema.OCT_2008_MAR_2017:
        code_loto = CodeLoto(
            winner_count(row, "nombre_de_codes_gagnants", source_filename, source_row),
            _decimal(row, "rapport_codes_gagnants", source_filename, source_row),
            _optional_text(row, "codes_gagnants"),
            currency if _optional_text(row, "rapport_codes_gagnants") else None,
        )

    return CanonicalDraw(
        draw_id=_required_text(row, "annee_numero_de_tirage", source_filename, source_row),
        draw_date=draw_date,
        draw_day=_required_text(row, "jour_de_tirage", source_filename, source_row),
        claim_deadline=_date(row, "date_de_forclusion", source_filename, source_row),
        main_numbers=_validate_numbers(numbers, source_filename, source_row, "main_numbers"),
        chance=_bounded_integer(row, "numero_chance", source_filename, source_row, 1, 10),
        winning_combination=_required_text(
            row, "combinaison_gagnante_en_ordre_croissant", source_filename, source_row
        ),
        second_draw=(
            _second_draw(row, source_filename, source_row, currency)
            if schema is SourceSchema.NOV_2019_ONWARD
            else None
        ),
        joker_plus=_optional_text(row, "numero_jokerplus"),
        code_loto=code_loto,
        numero_7=_optional_text(row, "numero_7"),
        prize_ranks=tuple(
            _rank(row, rank, source_filename, source_row, currency)
            for rank in range(1, rank_count + 1)
        ),
        currency=currency,
        provenance=Provenance(
            source_filename, source_sha256, source_row, schema, imported_at
        ),
    )
