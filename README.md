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
