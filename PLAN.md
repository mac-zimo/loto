# Plan expérimental — Loto Analyze

> **Pour Hermes :** exécuter ce plan tâche par tâche avec une validation indépendante des résultats.

**But :** déterminer honnêtement si les tirages présentent un signal reproductible et, séparément, si une sélection moins populaire peut réduire le partage d'un éventuel gain.

**Architecture :** pipeline reproductible `ingestion -> audit de hasard -> modèles hors échantillon -> simulation de gains réels -> registre d'expériences`. Les analyses descriptives restent disponibles, mais aucune sortie n'est appelée « prédiction » sans battre des baselines préenregistrées sur des périodes futures non utilisées pendant le développement.

**Stack :** Python, SQLite, pandas, NumPy, SciPy, scikit-learn, matplotlib, unittest/pytest, notebooks uniquement comme interface d'exploration.

---

## Verdict sur l'état initial

La base de code est utile comme prototype exploratoire, mais la direction initiale n'était pas suffisamment rigoureuse pour conclure à une stratégie rentable.

### À conserver

- ingestion CSV vers SQLite et contrôles de qualité ;
- analyses descriptives : fréquences, distributions, sommes, parité, cooccurrences ;
- visualisations et exports ;
- principe du backtest chronologique ;
- ACF/FFT uniquement comme tests exploratoires corrigés pour comparaisons multiples.

### À quarantainer jusqu'à refonte

- `models.py` : Markov et KMeans décrivent les données mais ne démontrent aucune prédiction ;
- `strategy.py` et `chance_strategy.py` : gains forfaitaires inventés, erreurs de comparaison et stochasticité insuffisamment contrôlée ;
- `prize_analysis.py` : rangs et probabilités incorrects, ROI reconstruit à partir du nombre de gagnants non identifiable ;
- corrélations entre positions triées : elles sont mécaniques et ne constituent pas un signal ;
- recherche de « hot/cold numbers » : hypothèses exploratoires, pas stratégie par défaut.

### Corrections immédiates déjà appliquées

- import des 2 811 tirages réparé ;
- dates normalisées en ISO avant tri chronologique ;
- exécution répétée du pipeline rendue idempotente ;
- bugs d'itération pandas et de graphique réparés ;
- seuil hot/cold corrigé ;
- faux backtests et modèles legacy désactivés par défaut ;
- périodicité corrigée sur 2 401 tests ACF avec Benjamini-Hochberg : aucun cycle robuste retenu.

---

## Règles scientifiques du projet

1. Séparer trois objectifs : **audit du hasard**, **prédiction hors échantillon**, **économie du partage des gains**.
2. Figé avant chaque expérience : hypothèse, variables, baseline, période d'apprentissage, période de test, métrique, seuil de succès et règle d'arrêt.
3. Ne jamais sélectionner un modèle sur le même échantillon qui sert à annoncer sa performance.
4. Comparer toute méthode à plusieurs baselines : uniforme, Flash simulé, fréquences historiques simples et modèle constant.
5. Corriger les tests multiples et publier aussi les résultats négatifs.
6. Employer des permutations/Monte-Carlo adaptés au tirage sans remise ; ne pas supposer une distribution indépendante incorrecte.
7. Utiliser les rapports réels par tirage et le prix officiel de la grille ; aucun gain forfaitaire inventé.
8. Une martingale ne peut pas modifier l'espérance de la grille. Les progressions de mise ne seront étudiées que comme profils de risque de ruine, jamais comme source d'avantage.
9. Un modèle n'est promu que si son avantage est reproduit sur plusieurs fenêtres temporelles et reste positif après coûts et correction du data snooping.
10. Budget réel nul pendant la recherche. Une éventuelle phase de jeu se limite à un protocole prospectif plafonné et préenregistré.

---

## Phase 0 — Fondations reproductibles

### Tâche 0.1 — Packaging, environnement et tests

**Fichiers :** créer `pyproject.toml`, `README.md`, `tests/`; modifier les imports `src` si nécessaire.

**Actions :**
- verrouiller les dépendances avec `uv` ;
- ajouter `pytest`, couverture et lint minimal ;
- permettre `uv run python run_pipeline.py` et `python -m loto` ;
- tester ingestion, dates, contraintes et idempotence.

**Validation :** suite verte depuis une base temporaire ; aucun fichier de production modifié par les tests.

### Tâche 0.2 — Contrat de données canonique

**Fichiers :** créer `src/loto/data_schema.py`, `tests/test_data_schema.py`.

**Actions :**
- documenter chaque époque/règle du jeu ;
- distinguer tirage principal, second tirage, Joker/Code Loto et rapports ;
- détecter les colonnes absentes par époque au lieu de les convertir silencieusement en zéro ;
- rejeter et journaliser les lignes dont une valeur obligatoire ne peut pas être parsée, sans substitution silencieuse ;
- conserver provenance, hash du fichier, date d'import et schéma source.

**Validation :** 2 811 lignes principales uniques, plage 2008-10-06 à 2026-09-21, rapports manquants représentés par `NULL`.

### Tâche 0.3 — Registre d'expériences

**Fichiers :** créer `experiments/registry.csv`, `src/loto/experiments.py`, `docs/EXPERIMENT_PROTOCOL.md`.

**Colonnes :** id, hypothèse, commit, données, split, features, modèle, baseline, métrique, seuil, seed, résultat, décision, artefacts.

**Validation :** toute commande d'expérience écrit une ligne immuable avec configuration et hash des données.

---

## Phase 1 — Audit du mécanisme de tirage

### Tâche 1.1 — Tests globaux de conformité

**Fichiers :** créer `src/loto/audit/uniformity.py`, `tests/audit/test_uniformity.py`.

**Tests :**
- fréquences marginales avec statistique adaptée au tirage sans remise ;
- distribution des numéros ordonnés ;
- somme, parité, distances et chevauchement entre tirages ;
- numéro Chance séparé.

**Méthode :** p-values Monte-Carlo sous un générateur exact 5 parmi 49 + Chance 1 parmi 10.

**Succès :** les tests sont calibrés sur données simulées ; rapport avec tailles d'effet et intervalles, pas seulement p-values.

### Tâche 1.2 — Indépendance temporelle et ruptures

**Fichiers :** créer `src/loto/audit/independence.py`, `src/loto/audit/change_points.py`.

**Tests :**
- répétition/chevauchement à plusieurs retards ;
- temps d'attente géométriques ;
- runs tests ;
- changements de régime par année, machine/règle si métadonnées disponibles ;
- correction FDR et réplication par sous-période.

**Règle de décision :** aucun « signal » si l'effet ne se réplique pas sur une seconde période et n'excède pas une taille minimale définie avant test.

### Tâche 1.3 — Permutations et puissance statistique

**Fichiers :** créer `src/loto/audit/power.py`.

**Actions :** injecter des biais simulés connus pour mesurer ce que 2 811 tirages permettent réellement de détecter.

**Livrable :** courbes de puissance indiquant les biais trop faibles pour être identifiables. Cela évite de confondre « non détecté » avec « inexistant ».

---

## Phase 2 — Banc d'essai prédictif honnête

### Tâche 2.1 — Split chronologique et moteur walk-forward

**Fichiers :** créer `src/loto/evaluation/walk_forward.py`, `tests/evaluation/test_walk_forward.py`.

**Protocole :**
- entraînement initial ;
- prédiction du tirage suivant sans accès au futur ;
- réentraînement éventuel ;
- fenêtres d'évaluation annuelles et rolling-origin ;
- seeds déterministes.

**Métriques :** log-loss/Brier sur inclusion de chaque numéro, nombre moyen de correspondances, calibration, rang du vrai numéro et regret versus uniforme.

### Tâche 2.2 — Baselines obligatoires

**Modèles :**
- uniforme exact ;
- fréquence cumulative lissée ;
- fréquences glissantes préfixées ;
- tirage aléatoire contraint.

**Règle :** aucune méthode complexe n'est testée avant que ces baselines et leurs intervalles soient corrects.

### Tâche 2.3 — Familles de modèles, une à la fois

Chaque famille reçoit un identifiant et une limite d'essais pour réduire le data snooping :

1. modèle bayésien Dirichlet-multinomial avec shrinkage ;
2. régression logistique régularisée par numéro ;
3. modèles de survie/hazard pour temps depuis dernière sortie ;
4. modèles de changement de régime ;
5. arbres boosting avec contraintes fortes ;
6. modèles séquentiels/transformers uniquement si les modèles simples montrent déjà un signal reproductible.

**Seuil de promotion :** amélioration hors échantillon statistiquement et économiquement significative sur au moins trois périodes, puis confirmation finale sur un holdout jamais consulté.

### Tâche 2.4 — Contrôles négatifs

- labels permutés ;
- colonnes temporelles décalées ;
- séries simulées équitables ;
- features absurdes.

Un pipeline qui « trouve » un avantage sur ces contrôles est rejeté comme surajusté.

---

## Phase 3 — Gains, espérance et partage

### Tâche 3.1 — Moteur officiel des rangs

**Fichiers :** créer `src/loto/payouts/rules.py`, `src/loto/payouts/engine.py`.

**Actions :** versionner les règles FDJ par date ; calculer exactement le rang d'une grille ; intégrer prix de mise, rapports réels, Code Loto et options seulement quand les données existent.

**Validation :** tests combinatoires exhaustifs pour chaque rang et comparaison à des exemples officiels.

### Tâche 3.2 — Backtest économique réel

**Métriques :** retour brut, ROI net, probabilité de perte, drawdown, CVaR, risque de ruine, distribution bootstrap par blocs temporels.

**Interdit :** extrapoler le nombre de joueurs depuis les gagnants d'un rang ou attribuer 150 €/1 M€/5 M€ arbitrairement.

### Tâche 3.3 — Modèle de popularité des grilles

C'est la piste rationnelle la plus plausible : elle ne change pas la probabilité de sortie, mais peut réduire le nombre de co-gagnants.

**Données possibles :**
- rapports et nombres de gagnants historiques ;
- biais documentés : dates 1–31, 7, suites, motifs visuels, combinaisons « équilibrées » ;
- si accessible, données de popularité agrégées ou proxy issu du nombre de gagnants.

**Sortie :** score de popularité d'une combinaison, validé sur des tirages où le nombre de gagnants est informatif.

**Stratégie candidate :** échantillonner uniformément parmi les combinaisons les moins populaires, sans prétendre qu'elles sortent davantage.

### Tâche 3.4 — Analyse jackpot/volume/Code Loto

Modéliser l'espérance conditionnelle au jackpot et au volume estimé de grilles. Tester séparément l'affirmation officielle selon laquelle la probabilité du Code Loto varie avec le nombre de prises de jeu.

**Décision :** seulement si l'espérance totale calculée avec règles et partage approche ou dépasse la mise dans des conditions historiques vérifiables.

---

## Phase 4 — Allocation et martingales

### Tâche 4.1 — Démonstration et simulateur de risque

**Fichiers :** créer `src/loto/bankroll.py`, `tests/test_bankroll.py`.

Comparer mise fixe, progression après pertes, progression après gains et budget périodique sous la même espérance par grille.

**Sortie attendue :** la progression modifie variance et risque de ruine, pas l'espérance. Toute martingale est rejetée comme stratégie de rentabilité.

### Tâche 4.2 — Politique de budget

Si une anomalie ou opportunité économique est un jour validée : utiliser un Kelly fractionnel plafonné à partir d'un avantage estimé conservateur et d'un intervalle inférieur. Sans avantage positif robuste : mise optimale financière = 0 €.

---

## Phase 5 — Recherche externe ciblée

Créer `docs/research/` avec fiches standardisées : question, source primaire, hypothèses, méthode, données, résultat, limites, expérience locale proposée.

Priorités :
- tests d'équité pour loteries k/N ;
- sélection consciente et partage de jackpot ;
- syndicates/buy-the-pot et seuils de jackpot ;
- validation walk-forward et correction du backtest overfitting ;
- détection de biais physiques et change points.

Un article ou dépôt n'est intégré que si son protocole peut être reproduit sur données simulées puis locales.

---

## Phase 6 — Livrables et arrêt

### Livrables

- rapport d'audit reproductible ;
- leaderboard hors échantillon avec intervalles ;
- registre complet des expériences, y compris échecs ;
- notebook de lecture, jamais source de logique ;
- générateur éventuel de grilles anti-partage clairement étiqueté ;
- rapport économique utilisant les gains réels.

### Règles d'arrêt

Arrêter la branche « prédiction des boules » si, après les baselines et trois familles de modèles préenregistrées, aucune amélioration ne se reproduit hors échantillon. Continuer alors uniquement : audit de hasard, popularité des choix, partage des gains et pédagogie du risque.

Ne jamais conclure « rentable » sur quelques jackpots historiques, sur une recherche hyperparamétrique non corrigée, ou sur un intervalle de confiance qui inclut la baseline.

---

## Ordre d'exécution immédiat

1. Tâche 0.1 — packaging et tests.
2. Tâche 0.2 — schéma canonique et données manquantes en `NULL`.
3. Tâche 0.3 — registre d'expériences.
4. Tâches 1.1 à 1.3 — audit et puissance.
5. Tâches 2.1 et 2.2 — walk-forward et baselines.
6. Décision go/no-go avant tout modèle IA complexe.
7. Tâches 3.1 à 3.3 — moteur de gains et anti-partage.
