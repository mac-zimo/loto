"""
Stratégie "numéro de chance" — exploitation des patterns du numéro complémentaire.

Le numéro de chance (1-10) est indépendant des 5 boules, mais peut présenter:
- Des biais de fréquence (certains numéros sortent plus souvent)
- Des corrélations avec le jour de tirage
- Une distribution non-uniforme testable

La stratégie combine:
1. Prédiction du numéro de chance par analyse de fréquence + jour
2. Combinaison avec les meilleures stratégies pour les 5 boules
"""

import numpy as np
import pandas as pd
from collections import Counter
from src.loto.config import NUM_BALLS, MAX_BALL, NUM_CHANCE_MAX


class ChanceNumberStrategy:
    """
    Stratégie exploitant spécifiquement le numéro de chance.

    Utilise la fréquence globale et conditionnelle au jour de tirage
    pour prédire le numéro de chance le plus probable, puis combine
    avec une sous-stratégie pour les 5 boules principales.
    """

    def __init__(self, main_strategy=None, method="frequency", window=200):
        """
        Args:
            main_strategy: callable(df) -> liste de combinaisons de 5 boules
                           (default: fréquence inverse pondérée)
            method: "frequency" (fréquence globale), "day_conditional" (par jour),
                    ou "recent" (tendance récente)
            window: fenêtre d'analyse pour les méthodes récentes
        """
        self.method = method
        self.window = window
        self.main_strategy = main_strategy or self._default_main

    def _default_main(self, df):
        """Fréquence inverse pondérée comme fallback pour les boules."""
        recent = df.tail(min(self.window, len(df)))
        all_balls = []
        for _, row in recent.iterrows():
            for i in range(1, NUM_BALLS + 1):
                val = row[f"boule_{i}"]
                if pd.notna(val):
                    all_balls.append(int(val))

        freq = Counter(all_balls)
        weights = np.array([(1.0 / (freq.get(n, 0) + 1)) for n in range(1, MAX_BALL + 1)])
        weights /= weights.sum()
        return [sorted(np.random.choice(MAX_BALL, size=NUM_BALLS, replace=False, p=weights) + 1)]

    def _predict_chance_frequency(self, df):
        """Prédiction par fréquence globale sur la fenêtre récente."""
        recent = df.tail(min(self.window, len(df)))
        chances = [int(row["numero_chance"]) for _, row in recent.iterrows() if pd.notna(row["numero_chance"])]
        if not chances:
            return list(range(1, NUM_CHANCE_MAX + 1))

        freq = Counter(chances)
        # Poids inversés: on favorise les numéros DEJA fréquents (biais observé)
        weights = np.array([(freq.get(n, 0) + 1) for n in range(1, NUM_CHANCE_MAX + 1)])
        weights /= weights.sum()
        return sorted(np.random.choice(NUM_CHANCE_MAX, size=NUM_CHANCE_MAX, replace=False, p=weights) + 1)

    def _predict_chance_day_conditional(self, df):
        """Prédiction par fréquence conditionnelle au jour de tirage."""
        recent = df.tail(min(self.window, len(df)))
        day_counters = {}

        for _, row in recent.iterrows():
            if pd.notna(row["numero_chance"]):
                day = str(row.get("jour_tirage", "unknown"))
                if day not in day_counters:
                    day_counters[day] = Counter()
                day_counters[day][int(row["numero_chance"])] += 1

        # Pour chaque jour, connaître le numéro de chance le plus fréquent
        self._day_bias = {day: counter.most_common(3) for day, counter in day_counters.items()}

        # Distribution globale pondérée par les biais de jour
        score = np.zeros(NUM_CHANCE_MAX + 1)
        for day, top_3 in self._day_bias.items():
            weight = len(recent[recent["jour_tirage"].astype(str).str.contains(day, na=False)])
            for chance_num, count in top_3:
                score[chance_num] += count * weight

        if score.sum() > 0:
            score /= score.sum()
            return sorted(np.random.choice(NUM_CHANCE_MAX, size=NUM_CHANCE_MAX, replace=False, p=score) + 1)
        return list(range(1, NUM_CHANCE_MAX + 1))

    def _predict_chance_recent(self, df):
        """Prédiction par tendance récente (derniers tirages)."""
        recent = df.tail(min(50, len(df)))
        chances = [int(row["numero_chance"]) for _, row in recent.iterrows() if pd.notna(row["numero_chance"])]

        if not chances:
            return list(range(1, NUM_CHANCE_MAX + 1))

        # Les numéros de chance récents ont plus de chances de revenir
        # (motifs courts de répétition)
        recent_counter = Counter(chances[-10:])
        weights = np.array([(recent_counter.get(n, 0) + 0.5) for n in range(1, NUM_CHANCE_MAX + 1)])
        weights /= weights.sum()
        return sorted(np.random.choice(NUM_CHANCE_MAX, size=NUM_CHANCE_MAX, replace=False, p=weights) + 1)

    def generate_combinations(self, df, n_tickets=3):
        """
        Génère n combinaisons complètes (5 boules + numéro de chance).

        Args:
            df: DataFrame avec les tirages historiques
            n_tickets: nombre de tickets à générer

        Returns:
            Liste de tuples (combo_balls, chance_number)
        """
        if self.method == "frequency":
            predictor = self._predict_chance_frequency
        elif self.method == "day_conditional":
            predictor = self._predict_chance_day_conditional
        else:
            predictor = self._predict_chance_recent

        all_results = []
        seen = set()

        for _ in range(n_tickets * 3):  # générer en surplus pour éviter doublons
            main_combos = self.main_strategy(df)
            chance_order = predictor(df)

            for combo in main_combos:
                for chance_num in chance_order[:2]:  # les 2 chances les plus probables
                    key = (tuple(sorted(combo)), chance_num)
                    if key not in seen and len(all_results) < n_tickets:
                        seen.add(key)
                        all_results.append((sorted(combo), chance_num))

        return all_results[:n_tickets]


def evaluate_chance_strategy(df, method="frequency"):
    """
    Évalue la stratégie numéro de chance vs approche aléatoire.

    Compare:
    - Jouer les numéros de chance fréquents (stratégie)
    - Jouer les numéros de chance uniformément aléatoires (baseline)

    Returns un dict comparatif.
    """
    rng = np.random.RandomState(42)

    chance_strat = ChanceNumberStrategy(method=method, window=200)
    random_chances = list(range(1, NUM_CHANCE_MAX + 1))

    strat_stats = {"invested": 0.0, "won": 0.0, "tickets": []}
    rand_stats = {"invested": 0.0, "won": 0.0}

    for draw_idx in range(50, len(df)):  # warmup de 50 tirages
        train_df = df.iloc[:draw_idx]
        current_row = df.iloc[draw_idx]

        actual_balls = sorted([int(current_row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(current_row[f"boule_{i}"])])
        actual_chance = int(current_row["numero_chance"])

        # Stratégie: 3 tickets avec numéros de chance optimisés
        strat_tickets = chance_strat.generate_combinations(train_df, n_tickets=3)
        cost_s = len(strat_tickets) * 2.0
        strat_stats["invested"] += cost_s
        strat_stats["tickets"].append(len(strat_tickets))

        for balls, chance in strat_tickets:
            matches = len(set(balls) & set(actual_balls))
            won = 0
            if matches == NUM_BALLS and chance == actual_chance:
                won = 5000000
            elif matches == NUM_BALLS:
                won = 1000000
            elif matches >= 4:
                won = 150
            if won > 0:
                strat_stats["won"] += won

        # Random: jouer les mêmes boules mais chance aléatoire
        rand_cost = 3.0 * 2.0
        rand_stats["invested"] += rand_cost
        for _ in range(3):
            rand_chance = rng.randint(1, NUM_CHANCE_MAX + 1)
            matches = len(set(balls) & set(actual_balls)) if strat_tickets else 0
            won = 0
            if matches == NUM_BALLS and rand_chance == actual_chance:
                won = 5000000
            elif matches == NUM_BALLS:
                won = 1000000
            elif matches >= 4:
                won = 150
            rand_stats["won"] += won

    strat_roi = ((strat_stats["won"] - strat_stats["invested"]) / strat_stats["invested"] * 100) if strat_stats["invested"] > 0 else 0
    rand_roi = ((rand_stats["won"] - rand_stats["invested"]) / rand_stats["invested"] * 100) if rand_stats["invested"] > 0 else 0

    return {
        "method": method,
        "strategy": {
            "total_invested": round(strat_stats["invested"], 2),
            "total_won": round(strat_stats["won"], 2),
            "roi_pct": round(strat_roi, 2),
            "avg_tickets_per_draw": round(np.mean(strat_stats["tickets"]) if strat_stats["tickets"] else 0, 1),
        },
        "random_baseline": {
            "total_invested": round(rand_stats["invested"], 2),
            "total_won": round(rand_stats["won"], 2),
            "roi_pct": round(rand_roi, 2),
        },
        "improvement": round(strat_roi - rand_roi, 2),
    }


if __name__ == "__main__":
    from src.loto.analysis import get_dataframe
    df = get_dataframe()

    for method in ["frequency", "day_conditional", "recent"]:
        result = evaluate_chance_strategy(df, method=method)
        print(f"\nMéthode '{result['method']}':")
        print(f"  Stratégie ROI: {result['strategy']['roi_pct']}% | Baseline ROI: {result['random_baseline']['roi_pct']}%")
        print(f"  Amélioration: {result['improvement']}pp")
