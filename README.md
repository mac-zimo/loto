# Loto Analyze

Loto Analyze charge les historiques du Loto français dans SQLite et produit des
analyses descriptives reproductibles. Le projet utilise Python 3.11 ou plus,
`uv` et une disposition de paquet `src` importable sous le nom `loto`.

## Installation

Depuis la racine du dépôt :

    uv sync

`uv sync` installe le paquet, ses dépendances d'exécution et le groupe de
développement contenant `pytest` et `pytest-cov`. `uv.lock` fixe les versions
résolues. `requirements.txt` reste uniquement un point d'entrée de compatibilité
et délègue à `pyproject.toml`.

## Tests

    uv run pytest -q

Les tests d'ingestion utilisent des fichiers CSV et des bases SQLite temporaires ;
ils ne modifient pas les données de production.
Le contrat canonique, ses règles de validation et la provenance sont décrits dans
[`docs/DATA_CONTRACT.md`](docs/DATA_CONTRACT.md).

Une couverture peut être calculée avec :

    uv run pytest --cov=loto --cov-report=term-missing

## Pipeline

Les fichiers CSV historiques attendus sont à la racine du dépôt. Pour lancer le
pipeline sans générer les visualisations :

    uv run python -m loto --skip-plots

Le point d'entrée historique reste compatible et délègue à la même CLI :

    uv run python run_pipeline.py --skip-plots

Le script installé peut également être appelé avec `uv run loto-analyze`.
Depuis un autre répertoire (notamment après installation d'une wheel), indiquer
explicitement les emplacements d'entrée et de sortie :

    loto-analyze --skip-plots \
        --data-dir /chemin/vers/les/csv \
        --db-path /chemin/vers/loto.db \
        --output-dir /chemin/vers/output

Utiliser `--help` pour voir les options `--reset` et les options legacy. Les
notebooks sont optionnels et s'installent avec `uv sync --extra notebook`.

Par sécurité, `--reset` n'ouvre ni ne modifie la base tant que les quatre fichiers
CSV configurés ne sont pas présents. Une erreur sur une ligne annule également le
rechargement complet. Sans `--reset`, les sources disponibles sont chargées et les
lignes invalides restent signalées puis ignorées pour préserver le comportement
historique.

Les modèles exploratoires legacy et les anciens backtests sont désactivés par
défaut. Ils ne sont exécutés que si `--include-legacy-models` ou
`--include-legacy-backtests` est fourni explicitement ; leurs résultats ne
doivent pas être interprétés comme des performances prédictives validées.

## Puissance de l'audit marginal

`loto.audit.power.estimate_power_curves()` simule par défaut les 2 811 tirages
historiques avec la seed préenregistrée `20260924`, alpha `0,05` (alpha par
test `0,025`), une correction de Bonferroni sur les tests marginaux principal et
Chance, et une cible de puissance de `0,80`. Les biais injectés sont des
augmentations, en points de probabilité, de l'inclusion du numéro 1 ; les autres
valeurs restent échangeables et le tirage principal reste strictement sans
remise (5 parmi 49).

La calibration utilise la p-value Monte-Carlo finie
`(1 + nombre(statistique_nulle >= statistique_observée)) / (B + 1)`. Le nombre
de répétitions nul doit donc permettre d'atteindre l'alpha par test (au moins 39
pour alpha global `0,05`). Les répétitions de puissance utilisent des flux
aléatoires indépendants de la calibration.

`audit_detectability()` ne déduit jamais l'équivalence de la puissance d'un test
de différence. Pour chacun des 49 indicateurs d'inclusion principaux et des 10
indicateurs Chance, il construit un intervalle binomial exact de
Clopper-Pearson. Bonferroni est appliqué aux indicateurs puis aux deux familles :
la couverture simultanée reste valide malgré la dépendance des cinq indicateurs
présents dans un même tirage. `bias_absent` exige que la borne supérieure
simultanée de la déviation marginale maximale soit strictement inférieure à la
marge préenregistrée `0,02`; sinon le résultat est `insufficient_power`.

`export_power_data(report, path)` écrit les points en CSV ou le manifeste complet
en JSON. `plot_power_curves(report, path)` produit un PNG ou SVG reproductible
octet par octet dans une même version et un même environnement logiciel,
indépendamment du `matplotlibrc` utilisateur. Cette garantie ne couvre pas les
différences entre versions de Matplotlib ou jeux de fontes installés. Pour
régénérer les livrables versionnés avec tous les paramètres par défaut :

    uv run --frozen --no-sync python -m loto.audit.generate_power_artifacts

La commande écrit dans `artifacts/task-1.3-power/` le CSV des courbes, le JSON
qui l'accompagne (seed, alphas, seuils, répétitions, correction, cible,
définitions statistiques) et la courbe PNG pour 2 811 tirages.

## Première comparaison réelle préenregistrée (tâche 2.3a)

Le runner officiel est exclusivement Linux/POSIX. Il exige `fcntl`,
`O_DIRECTORY`, `O_NOFOLLOW` et un système de fichiers prenant en charge
`renameat2(RENAME_NOREPLACE)`. Toute capacité absente provoque un échec fermé
avant une décision succès ; aucun fallback non sûr n'est utilisé.

Après revue et commit des sources, depuis un arbre Git propre dont le registre
ne contient pas encore l'ID préenregistré :

    uv run --frozen --no-sync python -m loto.evaluation.archive_experiment \
        --commit "$(git rev-parse HEAD)"

Cette commande est l'unique exécution officielle prévue. Elle lit directement,
sans SQLite, les quatre CSV canoniques de la racine. L'apprentissage initial
utilise les tirages antérieurs au 2014-01-01 ; le walk-forward expansif évalue
chaque tirage des années calendaires 2014 à 2025 avec réentraînement à chaque
tirage. L'année partielle 2026 est explicitement exclue et demeure un holdout
futur intact.

La commande refuse un arbre sale, un commit différent de `HEAD`, un ID déjà
enregistré, des données divergentes ou un artefact existant. Elle écrit sans
écrasement `preregistration.json`, `metrics.csv` et `result.json` sous
`artifacts/task-2.3a-archive-walk-forward/`, puis référence ces chemins relatifs
dans `experiments/registry.csv` via `run_experiment`. Les résultats constituent
uniquement un screening descriptif : aucun succès de screening ne promeut un
modèle sans significativité sur plusieurs périodes et confirmation ultérieure
sur le holdout resté intact.

Les trois fichiers sont préparés dans un répertoire caché voisin, puis publiés
ensemble par renommage atomique. Un staging résiduel après arrêt brutal bloque
explicitement une nouvelle exécution afin de permettre son inspection. Si
`run_experiment` échoue après publication, le répertoire publié par cette
tentative est conservé. Avant l'exécution, l'ID est durablement réservé sous
`experiments/registry.csv.reservations/`; la réservation n'est supprimée qu'après
le `fsync` d'une ligne succès ou erreur. Une réservation résiduelle consomme donc
l'ID et impose une inspection manuelle, sans relance ni nettoyage automatique.
L'identité et les hashes du staging sont authentifiés sous les verrous registre
et parent immédiatement avant l'écriture d'une décision succès.

Avant cette exécution irréversible, le preflight de capacité synthétique et non
scientifique exécute les dix candidats sur 2 811 observations déterministes,
sans archive FDJ, SQLite, registre, réseau ni écriture d'artefact :

    uv run --frozen --no-sync python -m loto.evaluation.archive_capacity_preflight

Il journalise sur stderr, avant et après chaque candidat, son ID, sa durée, son
nombre de prédictions et le `ru_maxrss` Linux, puis écrit le rapport JSON final
sur stdout. Les plafonds fixés avant mesure restent 20 minutes et 1,5 GiB de
`ru_maxrss` pour le processus complet. `tracemalloc` est désactivé par défaut :
son overhead important, notamment pendant les fits scikit-learn, rend la durée
non représentative de l'expérience officielle. Le mode diagnostique explicite
`--trace-python-memory` ajoute le plafond de 512 MiB au verdict, mais n'est ni
requis ni approprié pour la décision temporelle complète à 2 811 tirages. Un
candidat figé peut être isolé avec `--candidate ID` (option répétable). Le test
long associé, exclu par défaut, se lance explicitement par :

    LOTO_RUN_CAPACITY=1 uv run --frozen --no-sync pytest -q \
        tests/evaluation/test_archive_capacity_preflight.py \
        -m integration_capacity
