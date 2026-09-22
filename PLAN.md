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
- [ ] **Fixer le bug dans `report.py:343`** — `from src.loto.config import NUM_BALLS, MAX_BALL, df = None` mélange import et affectation, ça plantera si le module est rechargé. Corriger en deux instructions séparées et retirer la redéfinition de `plot_distribution_stats`. <50 lignes.
- [ ] **Configurer matplotlib backend** — ajouter `matplotlib.use('Agg')` au début du pipeline pour éviter les erreurs X11/WSLdisplay sur les environnements headless. <10 lignes (dans `run_pipeline.py`).
- [ ] **Fixer `hot_cold_numbers()` dans `analysis.py:310-311`** — le slice `[:20]` après `sort()` ne modifie pas la liste en place (le retour de `sort()` est None). Les hot/cold globaux sont donc toujours vides. <20 lignes.

### Analyses manquantes

- [ ] **Analyse des gains réels par rang** — exploiter les colonnes `gagnants_rang*` / `rapport_rang*` (déjà stockées en BD mais jamais utilisées). Calculer le ROI réel moyen par rang, la fréquence réelle de chaque prix, et comparer avec les odds théoriques. <150 lignes dans un nouveau fichier `src/loto/prize_analysis.py`.
- [ ] **Analyse de périodicité (Fourier/ACF)** — détecter si des cycles existent dans les séries temporelles de fréquence des numéros (autocorrélation, transformée de Fourier simplifiée). <100 lignes.
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
| Analyses statistiques | ✅ Fréquences, corrélations, temporel | Gains réels par rang, Fourier/ACF |
| Modèles prédictifs | ✅ Markov, KMeans, anomalies | — |
| Stratégies & backtest | ✅ 4 stratégies + backtest | IC bootstrap, stratégie combinée, numéro chance |
| Rapports & visuels | ✅ Heatmaps, charts, texte, JSON | Notebook exploratoire |
| Qualité code | — | Fix report.py, fix hot_cold(), matplotlib backend |

**Priorité immédiate :** `run_pipeline.py` (lancement end-to-end) + fixes bugs (`report.py`, `hot_cold_numbers()`) → ça rend le projet utilisable immédiatement.
