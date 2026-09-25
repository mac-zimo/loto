# Protocole du registre d’expériences

Le registre `experiments/registry.csv` est le journal append-only des expériences.
Une exécution, positive, négative ou en erreur, produit exactement une nouvelle
ligne. Une ligne existante n’est jamais corrigée ni remplacée : une nouvelle
exécution reçoit un nouvel identifiant.

## Configuration obligatoire

Avant d’exécuter le calcul, figer un `ExperimentConfig` avec :

- `id` : identifiant unique et stable, par exemple `EXP-2026-001` ;
- `hypothesis` : hypothèse réfutable ;
- `split` : périodes d’apprentissage, validation et test ;
- `features` et `model` : variables et méthode exactes ;
- `baseline` : comparateur préenregistré ;
- `metric` et `threshold` : métrique et seuil de décision ;
- `seed` : graine déterministe.

Les colonnes CSV sont strictement, dans cet ordre : `id`, `hypothèse`, `commit`,
`données`, `split`, `features`, `modèle`, `baseline`, `métrique`, `seuil`, `seed`,
`résultat`, `décision`, `artefacts`.

## Exécution obligatoire

Toute commande d’expérience doit passer par `loto.experiments.run_experiment`.
Elle fournit tous les fichiers d’entrée via `data_paths` et retourne un
`ExperimentOutcome`. Le wrapper calcule les empreintes avant le calcul, relève
le commit Git, exécute la commande, recalcule les empreintes puis ajoute la ligne
au registre. Une modification des données pendant le calcul est enregistrée
comme erreur et l’exception est relancée, sans publier de faux succès.

Le dépôt Git Loto est résolu indépendamment du répertoire courant. Son commit
doit exister et être fourni sous forme de hash hexadécimal complet. Une commande
ne démarre pas si le dépôt contient une modification suivie ou un fichier non
suivi ; seul le fichier exact du registre de sortie est exempté.

Le verrou du registre utilise explicitement `fcntl.flock` et exige donc un
système POSIX. Le CSV est un artefact machine : il doit être lu par un parseur
CSV et n’est pas destiné à être ouvert comme feuille de calcul.

Exemple minimal :

```python
from loto.experiments import ExperimentConfig, ExperimentOutcome, run_experiment

config = ExperimentConfig(
    id="EXP-2026-001",
    hypothesis="La méthode réduit le Brier hors échantillon.",
    split="train:2008-2023/test:2024",
    features="fréquences cumulées",
    model="fréquence lissée",
    baseline="uniforme exact",
    metric="Brier",
    threshold="amélioration >= 0.01",
    seed=2026,
)

run_experiment(
    config,
    lambda: ExperimentOutcome("delta=0.012", "retenue", ("output/EXP-2026-001.json",)),
    data_paths=["nouveau_loto.csv", "loto2017.csv"],
)
```

`données` contient un manifeste JSON canonique : le chemin relatif et le SHA-256
de chaque fichier, ainsi qu’un SHA-256 global du manifeste. `artefacts` est une
liste JSON de chemins. Le résultat doit inclure la valeur de la métrique, pas
seulement une appréciation textuelle.

## Règles d’intégrité

- Ne jamais éditer ni supprimer une ligne enregistrée.
- Ne jamais réutiliser un `id`; l’API rejette les doublons.
- Chaque ligne existante doit avoir exactement quatorze colonnes, un `id` non
  vide et distinct de tous les autres identifiants.
- `result` et `decision` sont des chaînes non vides ; `artifacts` est un tuple
  sérialisable de chemins sous forme de chaînes.
- Publier les résultats négatifs et les erreurs techniques.
- Ne pas appeler directement `append_experiment` depuis une commande : cette
  fonction est réservée à l’import contrôlé de résultats déjà exécutés.
- Conserver les artefacts référencés et ne pas les écraser.
- Un changement de configuration, de données ou de code impose un nouvel `id`.
