# Audit initial — 23 septembre 2026

## Conclusion

Le projet avait une bonne base d'exploration, mais n'était pas dans une direction permettant d'affirmer une stratégie rentable. Le principal problème n'était pas le choix des algorithmes : c'était l'absence de protocole expérimental strict, plusieurs erreurs d'implémentation et des backtests économiquement invalides.

La direction est désormais rectifiée : audit statistique du hasard, validation walk-forward hors échantillon, gains réels, contrôle des tests multiples et piste distincte de réduction du partage des jackpots.

## État vérifié

- 4 CSV, 2 811 tirages uniques.
- Période : 6 octobre 2008 au 21 septembre 2026.
- Base SQLite : 2 811 lignes, aucune date dupliquée, aucune répétition interne de boule.
- Pipeline complet : exécution réussie avec rapports et graphiques.
- Installation locale : environnement `.venv` créé avec `uv`.
- Tests ajoutés : chargement d'une ligne et normalisation de date.

## Corrections apportées

1. Requête d'insertion SQLite : 36 paramètres pour 34 colonnes, donc aucune ligne n'était importée.
2. Dates : stockage `DD/MM/YYYY`, donc tri historique faux ; conversion en ISO `YYYY-MM-DD`.
3. Pipeline : une seconde exécution échouait lorsque toutes les lignes existaient déjà ; comportement rendu idempotent.
4. `temporal_patterns()` et `periodicity._binary_series()` : unpacking pandas invalide.
5. Graphique des fréquences : l'indice entier de boucle était utilisé comme axe matplotlib.
6. Hot/cold : fréquence attendue calculée avec le nombre de numéros distincts au lieu du nombre total d'occurrences.
7. Backtests legacy : désactivés par défaut car ils utilisaient des gains arbitraires et contenaient des comparaisons incorrectes.
8. Modèles Markov/KMeans : désactivés par défaut tant qu'ils ne sont pas validés hors échantillon.
9. Périodicité : 2 401 tests ACF corrigés par Benjamini-Hochberg ; aucun cycle robuste détecté.

## Diagnostic statistique rapide

Ces contrôles sont exploratoires et ne remplacent pas l'audit Monte-Carlo planifié :

- fréquences des 49 boules : comptes de 259 à 327, contre 286,84 attendus ; test global p = 0,861 ;
- numéro Chance : test global p = 0,416 ;
- chevauchement entre deux tirages consécutifs : compatible avec l'hypergéométrique, p = 0,144 ;
- dépendance au retard 1 par numéro : 1 test nominal sur 49 sous 0,05, aucun après correction Benjamini-Hochberg ;
- ACF/FFT : aucun cycle retenu après correction globale.

À ce stade, les données ressemblent donc à un tirage équitable et sans mémoire exploitable. Cela ne prouve pas mathématiquement le hasard ; cela signifie qu'aucun signal robuste n'a encore été trouvé.

L'association brute entre jour de tirage et numéro Chance a produit p = 0,009. Elle doit être traitée comme une hypothèse exploratoire : vérifier les changements de régime, faire une correction multi-tests, mesurer la taille d'effet et reproduire hors échantillon avant toute exploitation.

## Éléments incorrects ou trompeurs dans le code legacy

- `strategy.py` attribuait forfaitairement 150 €, 1 M€ ou 5 M€ au lieu des rapports réels.
- Les tickets legacy ne comportaient généralement pas de numéro Chance ; le jackpot ne pouvait donc pas être évalué correctement.
- `chance_strategy.py` comparait parfois la baseline avec la dernière combinaison générée au lieu de chaque ticket correspondant.
- `prize_analysis.py` utilisait `C(49,5) = 1 533 939`, alors que la valeur exacte est 1 906 884.
- Les probabilités des rangs y sont incomplètes ou fausses, et le ROI ne peut pas être déduit proprement du nombre de gagnants d'un seul rang.
- La corrélation entre `boule_1` à `boule_5` mesure surtout l'effet mécanique de leur ordre croissant.
- Le clustering de sommes/parités décrit des formes normales de tirages ; il ne prédit pas le prochain tirage.

## Revue indépendante

Une seconde revue méthodologique, réalisée séparément après les corrections, confirme le diagnostic : conserver le socle de données et l'EDA, quarantainer les ROI legacy, remplacer les tests de paires par des simulations globales, imposer une évaluation préquentielle et traiter l'anti-partage comme un problème distinct de la prédiction.

## Décision

- Continuer le projet : oui.
- Continuer l'ancien plan tel quel : non.
- Chercher directement un modèle IA complexe : non, tant que les baselines walk-forward ne montrent aucun signal.
- Explorer une stratégie rationnelle : oui, surtout la réduction du partage par évitement des choix populaires, qui ne change pas la probabilité de sortie mais peut améliorer le gain conditionnel en cas de victoire.
- Martingale : à simuler uniquement pour démontrer le risque de ruine ; elle ne transforme pas une espérance négative en espérance positive.

Le plan détaillé est dans `PLAN.md`.
