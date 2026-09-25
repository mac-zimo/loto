import csv
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from loto.data_schema import (
    SourceSchema,
    StructuredValidationError,
    detect_source_schema,
    normalize_header,
    parse_canonical_draw,
    schema_contract,
)


ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "nouveau_loto.csv": SourceSchema.OCT_2008_MAR_2017,
    "loto2017.csv": SourceSchema.MAR_2017_FEB_2019,
    "loto_201902.csv": SourceSchema.FEB_2019_NOV_2019,
    "loto_201911.csv": SourceSchema.NOV_2019_ONWARD,
}


def source_row(schema: SourceSchema) -> dict[str, str]:
    row = {column: "" for column in schema_contract(schema).expected_columns}
    row.update({
        "annee_numero_de_tirage": "2018001",
        "jour_de_tirage": "LUNDI",
        "date_de_tirage": "08/01/2018",
        "date_de_forclusion": "08/03/2018",
        "boule_1": "9",
        "boule_2": "1",
        "boule_3": "49",
        "boule_4": "12",
        "boule_5": "5",
        "numero_chance": "10",
        "combinaison_gagnante_en_ordre_croissant": "1-5-9-12-49+10",
        "nombre_de_gagnant_au_rang1": "",
        "rapport_du_rang1": "",
        "devise": "eur",
    })
    if schema is SourceSchema.OCT_2008_MAR_2017:
        row["date_de_tirage"] = "04/03/2017"
    elif schema is SourceSchema.FEB_2019_NOV_2019:
        row["date_de_tirage"] = "27/02/2019"
    elif schema is SourceSchema.NOV_2019_ONWARD:
        row["date_de_tirage"] = "06/11/2019"
    return row


def parse(row, schema=SourceSchema.MAR_2017_FEB_2019):
    return parse_canonical_draw(
        row,
        schema=schema,
        source_filename="fixture.csv",
        source_sha256="a" * 64,
        source_row=1,
        imported_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.parametrize("filename, expected", FILES.items())
def test_actual_headers_are_detected(filename, expected):
    with (ROOT / filename).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        header = next(reader)
        rows = [dict(zip(normalize_header(header), values)) for values in reader]

    assert detect_source_schema(header, rows) is expected
    assert normalize_header(header) == schema_contract(expected).expected_columns


def test_unknown_and_ambiguous_layouts_are_rejected():
    with pytest.raises(StructuredValidationError, match="unknown source schema"):
        detect_source_schema(["unexpected"], [])

    contract = schema_contract(SourceSchema.MAR_2017_FEB_2019)
    rows = [source_row(SourceSchema.MAR_2017_FEB_2019)]
    rows.append(dict(rows[0], date_de_tirage="27/02/2019"))
    with pytest.raises(StructuredValidationError, match="ambiguous source schema"):
        detect_source_schema(contract.expected_columns, rows)


def test_optional_blanks_are_none_and_main_numbers_are_sorted():
    draw = parse(source_row(SourceSchema.MAR_2017_FEB_2019))

    assert draw.main_numbers == (1, 5, 9, 12, 49)
    assert draw.prize_ranks[0].winner_count is None
    assert draw.prize_ranks[0].report is None
    assert draw.second_draw is None
    assert draw.joker_plus is None
    assert draw.code_loto is not None
    assert draw.numero_7 is None
    assert draw.provenance.source_row == 1


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("boule_1", "word", "integer"),
        ("boule_1", "²", "integer"),
        ("boule_1", "50", "between 1 and 49"),
        ("numero_chance", "0", "between 1 and 10"),
        ("date_de_tirage", "2026/01/01", "date"),
        ("rapport_du_rang1", "-1", "non-negative"),
        ("nombre_de_gagnant_au_rang1", "1.2", "integer"),
    ],
)
def test_malformed_fields_raise_structured_errors(field, value, message):
    row = source_row(SourceSchema.MAR_2017_FEB_2019)
    row[field] = value

    with pytest.raises(StructuredValidationError) as exc_info:
        parse(row)

    error = exc_info.value
    assert error.filename == "fixture.csv"
    assert error.source_row == 1
    assert error.field == field
    assert message in error.message


@pytest.mark.parametrize(
    "schema,field",
    [
        (SourceSchema.MAR_2017_FEB_2019, "nombre_de_gagnant_au_rang1"),
        (SourceSchema.MAR_2017_FEB_2019, "nombre_de_codes_gagnants"),
        (
            SourceSchema.NOV_2019_ONWARD,
            "nombre_de_gagnant_au_rang_1_second_tirage",
        ),
    ],
)
def test_winner_counts_must_fit_sqlite_integer(schema, field):
    row = source_row(schema)
    row[field] = "9" * 100
    if schema is SourceSchema.NOV_2019_ONWARD:
        for index in range(1, 6):
            row[f"boule_{index}_second_tirage"] = str(index)

    with pytest.raises(StructuredValidationError) as exc_info:
        parse(row, schema)

    assert exc_info.value.as_dict() == {
        "file": "fixture.csv",
        "row": 1,
        "field": field,
        "message": "winner count must fit SQLite INTEGER",
    }


@pytest.mark.parametrize(
    "schema,field",
    [
        (SourceSchema.MAR_2017_FEB_2019, "rapport_du_rang1"),
        (SourceSchema.MAR_2017_FEB_2019, "rapport_codes_gagnants"),
        (SourceSchema.NOV_2019_ONWARD, "rapport_du_rang1_second_tirage"),
    ],
)
def test_monetary_reports_must_fit_finite_sqlite_real(schema, field):
    row = source_row(schema)
    row[field] = "1e999999"
    if schema is SourceSchema.NOV_2019_ONWARD:
        for index in range(1, 6):
            row[f"boule_{index}_second_tirage"] = str(index)

    with pytest.raises(StructuredValidationError) as exc_info:
        parse(row, schema)

    assert exc_info.value.as_dict() == {
        "file": "fixture.csv",
        "row": 1,
        "field": field,
        "message": "monetary report must fit finite SQLite REAL",
    }


def test_duplicate_main_numbers_are_rejected():
    row = source_row(SourceSchema.MAR_2017_FEB_2019)
    row["boule_2"] = row["boule_1"]

    with pytest.raises(StructuredValidationError, match="distinct"):
        parse(row)


def test_second_draw_is_all_or_none_and_sorted():
    schema = SourceSchema.NOV_2019_ONWARD
    row = source_row(schema)
    row.update({
        "boule_1_second_tirage": "30",
        "boule_2_second_tirage": "2",
        "boule_3_second_tirage": "45",
        "boule_4_second_tirage": "11",
        "boule_5_second_tirage": "7",
    })
    draw = parse(row, schema)
    assert draw.second_draw is not None
    assert draw.second_draw.numbers == (2, 7, 11, 30, 45)

    row["boule_3_second_tirage"] = ""
    with pytest.raises(StructuredValidationError, match="fully absent") as exc_info:
        parse(row, schema)
    assert exc_info.value.field == "second_draw"


@pytest.mark.parametrize(
    "field,value",
    [
        ("promotion_second_tirage", "PROMO"),
        ("combinaison_gagnant_second_tirage_en_ordre_croissant", "1-2-3-4-5"),
        ("nombre_de_gagnant_au_rang_1_second_tirage", "1"),
        ("rapport_du_rang1_second_tirage", "10,50"),
    ],
)
def test_second_draw_metadata_without_balls_is_rejected(field, value):
    schema = SourceSchema.NOV_2019_ONWARD
    row = source_row(schema)
    row[field] = value

    with pytest.raises(StructuredValidationError) as exc_info:
        parse(row, schema)

    assert exc_info.value.field == "second_draw"


def test_fully_blank_second_draw_group_is_absent():
    schema = SourceSchema.NOV_2019_ONWARD

    assert parse(source_row(schema), schema).second_draw is None


def test_provenance_requires_utc_and_positive_data_row():
    row = source_row(SourceSchema.MAR_2017_FEB_2019)
    with pytest.raises(StructuredValidationError, match="one-based"):
        parse_canonical_draw(
            row,
            schema=SourceSchema.MAR_2017_FEB_2019,
            source_filename="fixture.csv",
            source_sha256="a" * 64,
            source_row=0,
            imported_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize(
    "imports",
    [
        "import loto.data_parsing; import loto.data_schema",
        "import loto.data_schema; import loto.data_parsing",
    ],
)
def test_schema_and_parsing_import_cleanly_in_either_order(imports):
    subprocess.run([sys.executable, "-c", imports], check=True)
