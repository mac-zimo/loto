# PLAN — Projet Loto Analyze

## État actuel du projet

Le squelette est **assez complet** : 5 modules Python (config, database, loader, analysis, models, strategy, report), requirements.txt, .gitignore, 4 fichiers CSV + leurs archives zip. Le code est fonctionnel en théorie mais aucun `main.py` ou script d'exécution n'existe pour lancer le pipeline de bout en bout.

---

## Tâches

### Infrastructure & exécution

- [x] **Initialiser la structure du projet** — `src/loto/` package, `__init__.py`, `.gitignore`, `requirements.txt`
- [x] **config.py** — constantes (règles du jeu, chemins, seuils)
- [x] **database.py** — schéma SQLite, connexions, CREATE TABLE avec CHECK constraints
- [x] **loader.py** — lecture CSV multi-format → SQLite (INSERT OR IGNORE)
- [x] **analysis.py** — fréquences, paires, triplets, corrélations Spearman, distributions, analyse chance, patterns temporels, hot/cold
- [x] **models.py** — Markov chains, KMeans clustering, détection anomalies, test Chi-deux des paires
- [x] **strategy.py** — 4 stratégies (hot, cold, paired, weighted random) + backtest + odds théoriques partiels
- [x] **report.py** — visualisations matplotlib/seaborn + rapport textuel + export JSON
- [x] **Créer un script `run_pipeline.py` principal** — point d'entrée CLI qui enchaîne : loader → analysis → models → strategy → report. Accepte des flags (`--reset`, `--skip-plots`, `--output-dir`). Un seul fichier, <100 lignes.

### Qualité & fiabilité

- [x] **Valider le chargement CSV** — ajouter un check dans `loader.py` ou un script autonome qui vérifie la cohérence des données après import (lignes nulles, boules hors 1-49, doublons, dates non triées). Générer un résumé de qualité. <100 lignes.
- [x] **Fixer le bug dans `report.py:343`** — `from src.loto.config import NUM_BALLS, MAX_BALL, df = None` mélange import et affectation. Supprimé la première définition orpheline de `plot_distribution_stats` (lignes 59-108), corrigé l'import en deux instructions séparées avec `Counter` et `math_comb` remontés en haut du fichier. <50 lignes.
- [x] **Configurer matplotlib backend** — `matplotlib.use("Agg")` déjà présent dans `run_pipeline.py` (lignes 21-22). Aucune modification nécessaire. <10 lignes.
- [x] **Fixer `hot_cold_numbers()` dans `analysis.py:310-311`** — le slice `[:20]` après `sort()` retournait None. Corrigé : tri en place puis truncage explicite avec `hot = hot[:10]`. La fonction retournait aussi `overall.most_common()` au lieu des listes `hot`/`cold`. Correction appliquée. <20 lignes.

### Analyses manquantes

- [x] **Analyse des gains réels par rang** — créé `src/loto/prize_analysis.py` (149 l.). Fonctions : `roi_by_rank()` (ROI moyen par rang), `frequency_by_rank()` (fréquence réelle d'attribution), `compare_odds_theory_vs_reel()` (comparaison théorie vs réel). `summarize_all()` orchestre tout. Gère BD vide gracefully.
- [x] **Analyse de périodicité (Fourier/ACF)** — créé `src/loto/periodicity.py` (~95 l.). Fonctions: `_binary_series()` (série binaire 0/1), `_acf()` (autocorrélation normalisée avec seuil IC 95%), `_fft_periods()` (FFT + détection de pics > moyenne+2σ), `analyze_periodicity()` (orchestre et identifie les numéros cycliques). Interprétation: cycles significatifs ou bruit blanc.
- [ ] **Stratégie "numéro de chance"** — ajouter au moins une stratégie qui exploite spécifiquement le numéro de chance (fréquences, corrélation avec les boules, combinaison optimale). <80 lignes dans `strategy.py`.

### Améliorations backtest

- [ ] **Backtest avec intervalles de confiance** — dans `evaluate_strategy()`, ajouter un calcul de IC 95% sur le ROI via bootstrap (1000 rééchantillonnages). <80 lignes.
- [ ] **Stratégie combinée optimisée** — ajouter une stratégie qui combine les insights : paires fréquentes + hot numbers récents + profil cluster dominant. Évaluer sa performance. <100 lignes.

### Livrables finaux

- [ ] **Jupyter notebook exploratoire** — `analysis.ipynb` ou `exploration.ipynb` permettant de naviguer dans les résultats interactivement (graphiques, tableaux). Réutilise les fonctions existantes. <200 lignes.
- [ ] **README.md** — description du projet, installation (`pip install -r requirements.txt`), utilisation (`python run_pipeline.py`), explication des sorties. <80 lignes.
- [ ] **`__main__.py` ou entry point** — permettre `python -m src.loto` pour lancer le pipeline par défaut. <20 lignes.

---

## Résumé

| Catégorie | Fait | À faire |
|-----------|------|---------|
| Infrastructure & config | ✅ 7/7 modules existants | `run_pipeline.py`, `__main__.py`, README |
| Données (CSV→BD) | ✅ Chargement fonctionnel | Validation qualité, vérification cohérence |
| Analyses statistiques | ✅ Fréquences, corrélations, temporel, gains réels par rang, périodicité (Fourier/ACF) | — |
| Modèles prédictifs | ✅ Markov, KMeans, anomalies | — |
| Stratégies & backtest | ✅ 4 stratégies + backtest | IC bootstrap, stratégie combinée, numéro chance |
| Rapports & visuels | ✅ Heatmaps, charts, texte, JSON | Notebook exploratoire |
| Qualité code | — | Fix report.py, fix hot_cold(), matplotlib backend |

**Priorité immédiate :** `run_pipeline.py` (lancement end-to-end) + fixes bugs (`report.py`, `hot_cold_numbers()`) → ça rend le projet utilisable immédiatement.
