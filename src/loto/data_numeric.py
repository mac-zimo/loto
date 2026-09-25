"""Storage-safe numeric parsing for canonical FDJ source fields."""

import math
from decimal import Decimal, InvalidOperation
from typing import Mapping

from loto.data_schema import StructuredValidationError


SQLITE_INTEGER_MAX = 9_223_372_036_854_775_807


def _error(
    message: str, *, filename: str, source_row: int, field: str
) -> StructuredValidationError:
    return StructuredValidationError(
        message, filename=filename, source_row=source_row, field=field
    )


def parse_integer(
    row: Mapping[str, str],
    field: str,
    filename: str,
    source_row: int,
    *,
    optional: bool = False,
) -> int | None:
    value = (row.get(field) or "").strip()
    if optional and not value:
        return None
    if not value or not value.isascii() or not value.isdecimal():
        raise _error(
            "value must be an integer",
            filename=filename,
            source_row=source_row,
            field=field,
        )
    try:
        return int(value)
    except ValueError as exc:
        raise _error(
            "value must be an integer",
            filename=filename,
            source_row=source_row,
            field=field,
        ) from exc


def bounded_integer(
    row: Mapping[str, str],
    field: str,
    filename: str,
    source_row: int,
    minimum: int,
    maximum: int,
) -> int:
    value = parse_integer(row, field, filename, source_row)
    assert value is not None
    if not minimum <= value <= maximum:
        raise _error(
            f"value must be between {minimum} and {maximum}",
            filename=filename,
            source_row=source_row,
            field=field,
        )
    return value


def winner_count(
    row: Mapping[str, str], field: str, filename: str, source_row: int
) -> int | None:
    value = parse_integer(row, field, filename, source_row, optional=True)
    if value is not None and value > SQLITE_INTEGER_MAX:
        raise _error(
            "winner count must fit SQLite INTEGER",
            filename=filename,
            source_row=source_row,
            field=field,
        )
    return value


def monetary_decimal(
    row: Mapping[str, str], field: str, filename: str, source_row: int
) -> Decimal | None:
    value = (row.get(field) or "").strip()
    if not value:
        return None
    try:
        result = Decimal(value.replace(" ", "").replace(",", "."))
    except InvalidOperation as exc:
        raise _error(
            "value must be monetary numeric text",
            filename=filename,
            source_row=source_row,
            field=field,
        ) from exc
    if not result.is_finite() or result < 0:
        raise _error(
            "monetary report must be finite and non-negative",
            filename=filename,
            source_row=source_row,
            field=field,
        )
    try:
        sqlite_value = float(result)
    except (OverflowError, ValueError) as exc:
        raise _error(
            "monetary report must fit finite SQLite REAL",
            filename=filename,
            source_row=source_row,
            field=field,
        ) from exc
    if not math.isfinite(sqlite_value):
        raise _error(
            "monetary report must fit finite SQLite REAL",
            filename=filename,
            source_row=source_row,
            field=field,
        )
    return result