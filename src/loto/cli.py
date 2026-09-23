"""Command-line interface for the Loto analysis pipeline."""

import argparse
import logging
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse pipeline command-line arguments."""
    parser = argparse.ArgumentParser(description="Pipeline d'analyse Loto français")
    parser.add_argument("--reset", action="store_true", help="Vider la BD avant chargement")
    parser.add_argument("--skip-plots", action="store_true", help="Omettre les visualisations")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Répertoire contenant les quatre CSV historiques (défaut: répertoire courant)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Chemin de la base SQLite (défaut: data/loto_analyze.db)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Répertoire de sortie (défaut: output/)",
    )
    parser.add_argument(
        "--include-legacy-models",
        action="store_true",
        help="Exécuter les anciens modèles exploratoires (Markov/KMeans, non prédictifs)",
    )
    parser.add_argument(
        "--include-legacy-backtests",
        action="store_true",
        help="Exécuter les anciens backtests non fiables (gains estimés, exploration uniquement)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete analysis pipeline."""
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    from loto.config import DATA_DIR, DB_PATH, OUTPUT_DIR

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    db_path = Path(args.db_path) if args.db_path else DB_PATH
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    logger.info("[1/5] Chargement des CSV dans la base de données ...")
    from loto.loader import load_all_db

    total_loaded, total_inserted = load_all_db(
        reset=args.reset,
        data_dir=data_dir,
        db_path=db_path,
    )
    if total_loaded == 0:
        logger.error("Aucun tirage source lu. Vérifiez les fichiers CSV.")
        return 1
    logger.info(
        "  %s nouveaux tirages insérés sur %s lignes lues.",
        total_inserted,
        total_loaded,
    )

    logger.info("[2/5] Analyse statistique ...")
    from loto.analysis import get_dataframe, run_full_analysis

    analysis = run_full_analysis(db_path=db_path)
    df = get_dataframe(db_path=db_path)

    modeling = {}
    if args.include_legacy_models:
        logger.warning(
            "[3/5] Modèles legacy activés: résultats exploratoires, "
            "sans valeur prédictive démontrée."
        )
        from loto.models import run_modeling

        modeling = run_modeling(df)
    else:
        logger.info("[3/5] Modèles legacy ignorés (validation hors échantillon absente).")

    strategies = []
    chance_results = []
    if args.include_legacy_backtests:
        logger.warning(
            "[4/5] Backtests legacy activés: gains estimés et ROI non exploitables."
        )
        from loto.strategy import run_chance_evaluation, run_strategy_backtest

        strategies = run_strategy_backtest(df)
        chance_results = run_chance_evaluation(df)
    else:
        logger.info(
            "[4/5] Backtests legacy ignorés (méthodologie non fiable, refonte planifiée)."
        )

    if args.skip_plots:
        logger.info("[5/5] Génération du rapport textuel (sans plots) ...")
        from loto.report import generate_text_report

        generate_text_report(
            analysis,
            modeling,
            strategies,
            chance_results,
            output_dir=output_dir,
        )
    else:
        logger.info("[5/5] Génération des rapports et visualisations ...")
        import matplotlib

        matplotlib.use("Agg")
        from loto.report import generate_full_report

        generate_full_report(
            analysis,
            modeling,
            strategies,
            df,
            chance_results,
            output_dir=output_dir,
        )

    logger.info("\n=== Pipeline terminé avec succès ===")
    return 0
