"""Validation des donnees : doublons, boules nulles, dates manquantes,
combinaison non triee. Resume console + JSON."""

import json
from pathlib import Path

from loto.config import CSV_FILES, PROJECT_ROOT, DB_PATH, OUTPUT_DIR
from loto.database import get_connection


def _count_csv_lines() -> dict[str, int]:
    counts = {}
    for f in CSV_FILES:
        p = PROJECT_ROOT / f
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as fh:
            counts[f] = sum(1 for _ in fh) - 1
    return counts


def validate_db() -> list[str]:
    conn = get_connection()
    cur = conn.cursor()
    errors = []
    cur.execute("SELECT COUNT(*) FROM tirages")
    if cur.fetchone()[0] == 0:
        errors.append("Aucun tirage en base — chargez les CSV d'abord.")
        conn.close()
        return errors
    # Doublons
    cur.execute(
        "SELECT annee_numero, COUNT(*) c FROM tirages GROUP BY annee_numero HAVING c > 1"
    )
    for row in cur.fetchall():
        errors.append(f"Doublon: {row[0]}")
    # Boules nulles
    cur.execute(
        "SELECT COUNT(*) FROM tirages WHERE boule_1 = 0 OR boule_5 = 0"
    )
    if cur.fetchone()[0] > 0:
        errors.append("Boules a zero detectees")
    # Dates manquantes
    cur.execute(
        "SELECT COUNT(*) FROM tirages WHERE date_tirage = '' OR date_tirage IS NULL"
    )
    if cur.fetchone()[0] > 0:
        errors.append("Dates manquantes detectees")
    # Combinaison non triee
    cur.execute(
        "SELECT combinaison_croissante FROM tirages "
        "WHERE combinaison_croissante IS NOT NULL AND combinaison_croissante != ''"
    )
    bad = 0
    for row in cur.fetchall():
        parts = [int(x) for x in str(row[0]).split("-") if x.strip().isdigit()]
        if parts and parts != sorted(parts):
            bad += 1
    if bad:
        errors.append(f"{bad} combinaison(s) non triee(s)")
    conn.close()
    return errors


def run_validation() -> str:
    csv_counts = _count_csv_lines()
    total = sum(csv_counts.values())
    print("=" * 50)
    print("  VALIDATION DONNEES LOTO")
    print("=" * 50)
    print(f"\n[CSV] {total} lignes sources")
    for name, n in csv_counts.items():
        print(f"  {name}: {n}")
    errors = []
    if DB_PATH.exists():
        errors = validate_db()
        icon = "OK" if not errors else "!!"
        print(f"\n[BD] [{icon}] {len(errors)} anomalie(s)")
        for e in errors[:10]:
            print(f"  ! {e}")
    else:
        errors.append("Base introuvable — lancez loader.py d'abord.")
        print("\n[BD] Aucun fichier base")
    status = "SUCCES" if not errors else "ECHEC"
    icon = "+" if status == "SUCCES" else "X"
    print(f"\n{'='*50}\n  [{icon}] {status}: {total} lignes, {len(errors)} anomalie(s)\n{'='*50}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"status": status, "total_csv_lines": total,
              "errors_count": len(errors), "errors": errors}
    path = OUTPUT_DIR / "validation_report.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n  Rapport: {path}")
    return status


if __name__ == "__main__":
    status = run_validation()
    exit(0 if status == "SUCCES" else 1)
