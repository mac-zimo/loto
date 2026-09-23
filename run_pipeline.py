#!/usr/bin/env python3
"""
Point d'entrée CLI du pipeline Loto Analyze.

Enchaîne : loader → analysis → models → strategy → report.

Usage :
    python run_pipeline.py                 # exécution normale
    python run_pipeline.py --reset         # reconduit depuis zéro
    python run_pipeline.py --skip-plots    # sans visualisations
    python run_pipeline.py --output-dir out/   # répertoire de sortie personnalisé
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Force headless matplotlib backend before any pyplot import
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline d'analyse Loto français")
    p.add_argument("--reset", action="store_true", help="Vider la BD avant chargement")
    p.add_argument("--skip-plots", action="store_true", help="Omettre les visualisations")
    p.add_argument("--output-dir", type=str, default=None, help="Répertoire de sortie (default: output/)")
    p.add_argument(
        "--include-legacy-models",
        action="store_true",
        help="Exécuter les anciens modèles exploratoires (Markov/KMeans, non prédictifs)",
    )
    p.add_argument(
        "--include-legacy-backtests",
        action="store_true",
        help="Exécuter les anciens backtests non fiables (gains estimés, exploration uniquement)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Override OUTPUT_DIR si spécifié
    if args.output_dir:
        from src.loto import config
        config.OUTPUT_DIR = Path(args.output_dir)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Étape 1 : Loader ---
    logger.info("[1/5] Chargement des CSV dans la base de données ...")
    from src.loto.loader import load_all_db
    total_loaded, total_inserted = load_all_db(reset=args.reset)
    if total_loaded == 0:
        logger.error("Aucun tirage source lu. Vérifiez les fichiers CSV.")
        sys.exit(1)
    logger.info(f"  {total_inserted} nouveaux tirages insérés sur {total_loaded} lignes lues.")

    # --- Étape 2 : Analyse ---
    logger.info("[2/5] Analyse statistique ...")
    from src.loto.analysis import run_full_analysis, get_dataframe
    analysis = run_full_analysis()
    df = get_dataframe()

    # --- Étape 3 : Modélisation exploratoire ---
    modeling = {}
    if args.include_legacy_models:
        logger.warning("[3/5] Modèles legacy activés: résultats exploratoires, sans valeur prédictive démontrée.")
        from src.loto.models import run_modeling
        modeling = run_modeling(df)
    else:
        logger.info("[3/5] Modèles legacy ignorés (validation hors échantillon absente).")

    # --- Étape 4 : Stratégies & Backtest ---
    strategies = []
    chance_results = []
    if args.include_legacy_backtests:
        logger.warning(
            "[4/5] Backtests legacy activés: gains estimés et ROI non exploitables."
        )
        from src.loto.strategy import run_strategy_backtest, run_chance_evaluation
        strategies = run_strategy_backtest(df)
        chance_results = run_chance_evaluation(df)
    else:
        logger.info("[4/5] Backtests legacy ignorés (méthodologie non fiable, refonte planifiée).")

    # --- Étape 5 : Rapport ---
    if args.skip_plots:
        logger.info("[5/5] Génération du rapport textuel (sans plots) ...")
        from src.loto.report import generate_text_report
        generate_text_report(analysis, modeling, strategies, chance_results)
    else:
        logger.info("[5/5] Génération des rapports et visualisations ...")
        from src.loto.report import generate_full_report
        generate_full_report(analysis, modeling, strategies, df, chance_results)

    logger.info("\n=== Pipeline terminé avec succès ===")


if __name__ == "__main__":
    main()
