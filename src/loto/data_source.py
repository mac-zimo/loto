"""Single-pass inspection of FDJ CSV sources before database mutation."""

from __future__ import annotations

import csv
import hashlib
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from loto.data_schema import (
    CanonicalDraw,
    SourceSchema,
    StructuredValidationError,
    detect_source_schema,
    normalize_header,
)
from loto.data_parsing import parse_canonical_draw


@dataclass(frozen=True)
class InspectedRow:
    source_row: int
    values: Mapping[str, str] | None
    error: StructuredValidationError | None


@dataclass(frozen=True)
class SourceInspection:
    path: Path
    sha256: str
    size_bytes: int
    schema: SourceSchema
    rows: tuple[InspectedRow, ...]


@dataclass(frozen=True)
class PreflightSource:
    """One fully inspected and canonicalized reset source."""

    inspection: SourceInspection
    canonical_draws: tuple[CanonicalDraw, ...]
    row_errors: tuple[StructuredValidationError, ...]
    imported_at: datetime


class DuplicateDrawIdError(StructuredValidationError):
    """A canonical draw ID repeated within the reset source set."""

    def __init__(self, draw: CanonicalDraw, first: CanonicalDraw) -> None:
        self.draw_id = draw.draw_id
        self.first_filename = first.provenance.source_filename
        self.first_source_row = first.provenance.source_row
        super().__init__(
            f"duplicate canonical draw ID {draw.draw_id!r}; first occurrence "
            f"{self.first_filename}:data-row {self.first_source_row}",
            filename=draw.provenance.source_filename,
            source_row=draw.provenance.source_row,
            field="annee_numero_de_tirage",
        )

    def as_dict(self) -> dict[str, str | int | None]:
        result = super().as_dict()
        result.update({
            "draw_id": self.draw_id,
            "first_file": self.first_filename,
            "first_row": self.first_source_row,
        })
        return result


def inspect_source(path: Path) -> SourceInspection:
    """Read, hash, and identify one source without touching SQLite."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")), delimiter=";")
    try:
        raw_header = next(reader)
    except StopIteration as exc:
        raise StructuredValidationError(
            "unknown source schema: empty file",
            filename=path.name,
            source_row=None,
            field="header",
        ) from exc

    header = normalize_header(raw_header)
    rows: list[InspectedRow] = []
    for source_row, values in enumerate(reader, 1):
        while len(values) > len(header) and not values[-1].strip():
            values.pop()
        if len(values) != len(header):
            error = StructuredValidationError(
                "row column count does not match header",
                filename=path.name,
                source_row=source_row,
                field="row",
            )
            rows.append(InspectedRow(source_row, None, error))
        else:
            rows.append(InspectedRow(source_row, dict(zip(header, values)), None))

    if not rows:
        raise StructuredValidationError(
            "source contains zero data rows",
            filename=path.name,
            source_row=None,
            field="rows",
        )
    valid_mappings = [row.values for row in rows if row.values is not None]
    schema = detect_source_schema(raw_header, valid_mappings, filename=path.name)
    return SourceInspection(path, digest, len(raw), schema, tuple(rows))


def validate_complete_source_set(inspections: list[SourceInspection]) -> None:
    """Require one content-detected source for every supported schema epoch."""
    detected = [inspection.schema for inspection in inspections]
    expected = set(SourceSchema)
    if len(detected) == len(expected) and set(detected) == expected:
        return
    duplicates = sorted(
        schema.value for schema in expected if detected.count(schema) > 1
    )
    missing = sorted(schema.value for schema in expected if schema not in detected)
    details = f"duplicates={duplicates}; missing={missing}"
    raise StructuredValidationError(
        f"reset requires exactly one source for each schema epoch ({details})",
        filename="<source-set>",
        source_row=None,
        field="source_schema",
    )


def preflight_source_set(
    inspections: list[SourceInspection], *, imported_at: datetime | None = None
) -> tuple[PreflightSource, ...]:
    """Canonicalize every reset row and validate the complete source set.

    One UTC timestamp is assigned to the entire reset run. Parsing is completed
    for every physically readable row before any collected row error is raised.
    """
    timestamp = imported_at or datetime.now(timezone.utc)
    preflight: list[PreflightSource] = []
    for inspection in inspections:
        draws: list[CanonicalDraw] = []
        errors: list[StructuredValidationError] = []
        for inspected_row in inspection.rows:
            if inspected_row.error is not None:
                errors.append(inspected_row.error)
                continue
            assert inspected_row.values is not None
            try:
                draws.append(parse_canonical_draw(
                    inspected_row.values,
                    schema=inspection.schema,
                    source_filename=inspection.path.name,
                    source_sha256=inspection.sha256,
                    source_row=inspected_row.source_row,
                    imported_at=timestamp,
                ))
            except StructuredValidationError as error:
                errors.append(error)
        preflight.append(PreflightSource(
            inspection, tuple(draws), tuple(errors), timestamp
        ))

    validate_complete_source_set(inspections)
    for source in preflight:
        if source.row_errors:
            raise source.row_errors[0]
        if not source.canonical_draws:
            raise StructuredValidationError(
                "source contains zero accepted rows",
                filename=source.inspection.path.name,
                source_row=None,
                field="rows",
            )

    first_by_id: dict[str, CanonicalDraw] = {}
    for source in preflight:
        for draw in source.canonical_draws:
            first = first_by_id.setdefault(draw.draw_id, draw)
            if first is not draw:
                raise DuplicateDrawIdError(draw, first)
    return tuple(preflight)
