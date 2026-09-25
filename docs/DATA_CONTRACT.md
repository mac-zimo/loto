# Contrat de données FDJ Loto

Ce document décrit le contrat canonique de la phase 0.2. Il concerne uniquement
les quatre archives CSV officielles configurées. Il ne définit ni moteur de gains,
ni règle de classement d'une grille, ni époque de règlement non documentée.

## Sources et statut des faits

Sources officielles consultables :

- archives et périodes publiées par FDJ :
  https://www.fdj.fr/jeux-de-tirage/loto/historique
- jeu actuel (5 numéros de 1 à 49 et Chance de 1 à 10 ; second tirage avec
  cinq numéros et sans Chance) :
  https://www.fdj.fr/jeux-de-tirage/loto/comment-jouer
- règlement historique communiqué par FDJ :
  https://media.fdj.fr/static/contrib/files/pdf/reglement_loto-ancien_0.pdf

Les quatre périodes « octobre 2008–mars 2017 », « mars 2017–février 2019 »,
« février–novembre 2019 » et « novembre 2019 à aujourd'hui » sont annoncées sur
la page d'archives FDJ. Les bornes inclusives exactes ci-dessous sont **inférées
du contenu des CSV et de leurs ruptures** ; elles ne sont pas présentées comme
de nouvelles époques réglementaires :

| Identifiant de schéma | Bornes observées inclusives | Colonnes normalisées |
|---|---:|---:|
| `oct-2008_mar-2017` | 2008-10-06 – 2017-03-04 | 25 |
| `mar-2017_feb-2019` | 2017-03-06 – 2019-02-25 | 34 |
| `feb-2019_nov-2019` | 2019-02-27 – 2019-11-02 | 34 |
| `nov-2019_onward` | 2019-11-06 – ouvert | 49 |

Les deux schémas intermédiaires ont le même en-tête. Ils sont donc distingués
par les dates du contenu. Un fichier vide, un fichier qui traverse ces deux
périodes ou une combinaison en-tête/période inconnue est ambiguë et rejetée.
Le nom du fichier ne décide jamais du schéma. Un ou plusieurs champs d'en-tête
vides à la fin, produits par les points-virgules terminaux FDJ, sont supprimés ;
un champ vide au milieu ou toute autre différence est rejeté.

## Modèle canonique

`CanonicalDraw` dans `src/loto/data_schema.py` contient :

- `draw_id` : identifiant opaque FDJ, obligatoire et unique en base ;
- `draw_date`, `draw_day`, `claim_deadline` : date du tirage, libellé du jour et
  date de forclusion obligatoires ;
- `main_numbers` : exactement cinq entiers distincts dans `[1, 49]`, triés lors
  de la canonicalisation ;
- `chance` : entier dans `[1, 10]` ;
- `winning_combination` : texte source obligatoire ;
- `second_draw` : absent avant le schéma de novembre 2019, sinon cinq numéros
  distincts dans `[1, 49]` ou aucun numéro, avec texte de combinaison,
  promotion et quatre rangs propres facultatifs ;
- `joker_plus`, `code_loto` et `numero_7` : trois concepts séparés. Aucun n'est
  dérivé d'un autre ;
- `prize_ranks` : rangs 1 à 6 dans la première archive, 1 à 9 ensuite ; chaque
  rang possède un nombre de gagnants nullable, un rapport monétaire nullable
  (`Decimal` pendant la validation) et la devise source lorsque le rapport est
  présent ;
- `currency` : texte de devise source, conservé sans substitution ;
- `provenance` : origine exacte décrite ci-dessous.

Les anciens noms SQL nécessaires aux analyses (`annee_numero`, `date_tirage`,
`boule_1` à `boule_5`, `numero_chance`, etc.) restent disponibles. Les colonnes
du second tirage sont distinctes et ne sont jamais sélectionnées par les
analyses du tirage principal.

## Sémantique de NULL

Une cellule vide facultative, un champ qui n'existe pas dans une archive
ancienne et un concept non applicable sont stockés en `NULL`/`None`, jamais en
zéro ou en chaîne inventée. Un zéro explicitement fourni reste un zéro valide
pour un nombre de gagnants ou un rapport. En particulier :

- rangs 7 à 9 des archives 2008–2017 : `NULL` ;
- Code Loto absent de cette archive : `NULL` ;
- second tirage avant novembre 2019 : `NULL` ;
- Joker+ absent du schéma postérieur : `NULL` ;
- tout rapport ou nombre de gagnants vide : `NULL`.

Les lignes déjà présentes dans une ancienne base ne sont pas réécrites pendant
la migration. L'import normal complète toutefois leur lien de provenance s'il
est encore NULL, sans remplacer une provenance existante. Un `--reset`
canonique recharge les quatre sources et applique les nouvelles sémantiques
NULL.

## Validation

La conversion refuse notamment :

- identifiant, jour, date, date de forclusion ou combinaison obligatoire vide ;
- date source autre que `JJ/MM/AAAA` ou hors période du schéma détecté ;
- entier mal formé (les décimaux ne sont pas tronqués) ;
- numéro principal ou second hors `[1, 49]`, doublon, ou second tirage partiel ;
- Chance hors `[1, 10]` ;
- gagnants négatifs, rapports négatifs, non finis ou mal formés ;
- nombre de colonnes différent de l'en-tête.

`StructuredValidationError.as_dict()` fournit toujours `file`, `row`, `field`
et `message`. `row` est **le numéro un-based de la ligne de données, en excluant
l'en-tête** : la première ligne après l'en-tête vaut donc 1. En mode normal, les
lignes invalides sont journalisées sous cette forme et ignorées. Lors d'un
`--reset`, les quatre fichiers sont entièrement canonicalisés avant
l'ouverture de SQLite. Toute ligne invalide lève `RowImportError` avec la même
erreur structurée ; aucun schéma, octet ou répertoire de base n'est alors créé ou
modifié.

## Provenance et qualité

Le SHA-256 est calculé une seule fois sur les octets exacts de chaque fichier.
`source_files` conserve : nom de fichier, SHA-256, taille en octets, schéma,
horodatage UTC d'import, lignes lues, acceptées, rejetées et doublons. Chaque
tirage référence cette ligne et conserve son numéro de ligne source. Une
jointure `tirages.source_file_id = source_files.id` permet donc de retrouver le
nom, le hash, le schéma et l'instant exact de son import.

`loto.data_quality.data_quality_summary(db_path)` retourne une synthèse stable :
fichiers triés, totaux lus/acceptés/rejetés/doublons, comptes par schéma, bornes
de dates et comptes NULL par champ nullable.

## Migration SQLite et idempotence

`init_db` active systématiquement `PRAGMA foreign_keys=ON`, crée
`source_files`, puis ajoute seulement les colonnes canoniques absentes avec
`ALTER TABLE`. Il ne reconstruit ni ne supprime la table historique. Réexécuter
la migration est sans effet. Les tirages restent uniques par `annee_numero` et
les sources par `(filename, sha256)`, ce qui rend les imports répétés
idempotents.

`--reset` vérifie d'abord la présence des quatre fichiers, exige exactement une
source non vide et sans erreur par époque, puis canonicalise toutes les lignes
avec un horodatage UTC unique pour l'exécution. Les identifiants de tirage doivent
être globalement uniques ; un doublon indique sa provenance et celle de sa
première occurrence. Ces contrôles ont lieu avant `init_db`, donc une source
invalide ne crée pas la base et ne migre pas une base historique.

Les objets canoniques du précontrôle sont réutilisés lors de l'écriture, sans
seconde analyse. Après insertion et avant commit, le chargeur vérifie que lignes
lues, acceptées et insérées concordent, que le nombre final de tirages et la somme
des acceptations source concordent, que chaque tirage possède une provenance et
qu'aucun doublon n'a été toléré. Toute divergence provoque le rollback de la
transaction unique ; une base antérieure reste alors intacte.
