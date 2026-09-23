import csv
import tempfile
import unittest
from pathlib import Path

from src.loto.database import init_db
from src.loto.loader import load_csv_file, normalize_date


class LoaderTests(unittest.TestCase):
    def test_normalize_date(self):
        self.assertEqual(normalize_date("21/09/2026"), "2026-09-21")
        self.assertEqual(normalize_date("2026-09-21"), "2026-09-21")

    def test_load_csv_row_inserts_all_columns(self):
        headers = [
            "annee_numero_de_tirage", "jour_de_tirage", "date_de_tirage",
            "date_de_forclusion", "boule_1", "boule_2", "boule_3", "boule_4",
            "boule_5", "numero_chance", "combinaison_gagnante_en_ordre_croissant",
        ]
        row = [
            "2026001", "lundi", "05/01/2026", "07/03/2026",
            "1", "2", "3", "4", "5", "6", "1-2-3-4-5",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "draw.csv"
            db_path = Path(tmp) / "draw.db"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, delimiter=";")
                writer.writerow(headers)
                writer.writerow(row)

            conn = init_db(str(db_path))
            total, inserted = load_csv_file(csv_path, conn)
            stored = conn.execute(
                "SELECT date_tirage, date_forclusion, boule_1, boule_5, numero_chance FROM tirages"
            ).fetchone()
            conn.close()

        self.assertEqual((total, inserted), (1, 1))
        self.assertEqual(stored, ("2026-01-05", "2026-03-07", 1, 5, 6))


if __name__ == "__main__":
    unittest.main()
