"""
Simulation de stratégies de jeu et évaluation de rentabilité.

Stratégies implémentées:
- Jouer les numéros chauds (suivi des tendances)
- Jouer les numéros froids (loi des grands nombres)
- Jeu combiné basé sur les paires fréquentes
- Jeu randomisé pondéré par fréquence inverse
- Stratégie de tiercé (optimiser le rapport risque/rentabilité)

Évaluation:
- ROI par stratégie sur données historiques
- Espérance de gain
- Analyse du risque (drawdown, variance)
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from src.loto.config import (
    NUM_BALLS, MAX_BALL, NUM_CHANCE_MAX,
    SIMULATION_ROUNDS, SIMULATION_SEED,
)


class BaseStrategy:
    """Classe de base pour une stratégie de jeu."""

    def __init__(self, name: str):
        self.name = name

    def select_numbers(self, df: pd.DataFrame, n_draws_to_use: int = 200) -> list[list[int]]:
        """
        Retourne les combinaisons à jouer basées sur la stratégie.

        Args:
            df: DataFrame avec tous les tirages
            n_draws_to_use: Nombre de tirages récents à considérer

        Returns:
            Liste de combinaisons (chaque combinaison = liste de 5 numéros)
        """
        raise NotImplementedError


class HotNumbersStrategy(BaseStrategy):
    """Joue les numéros les plus fréquents sur la période récente."""

    def __init__(self, window: int = 100):
        super().__init__("Hot Numbers")
        self.window = window

    def select_numbers(self, df: pd.DataFrame, n_draws_to_use: int | None = None) -> list[list[int]]:
        n = n_draws_to_use or self.window
        recent = df.tail(n) if n < len(df) else df

        all_balls = []
        for _, row in recent.iterrows():
            balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
            all_balls.extend(balls)

        freq = Counter(all_balls)
        top_numbers = [num for num, _ in freq.most_common(NUM_BALLS * 2)]
        top_numbers = top_numbers[:NUM_BALLS * 2]  # limiter à 10 numéros max par combinaison

        return [sorted(top_numbers[:NUM_BALLS])]


class ColdNumbersStrategy(BaseStrategy):
    """Joue les numéros les moins fréquents (théorie du rattrapage)."""

    def __init__(self, window: int = 200):
        super().__init__("Cold Numbers")
        self.window = window

    def select_numbers(self, df: pd.DataFrame, n_draws_to_use: int | None = None) -> list[list[int]]:
        n = n_draws_to_use or self.window
        recent = df.tail(n) if n < len(df) else df

        all_balls = []
        for _, row in recent.iterrows():
            balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
            all_balls.extend(balls)

        freq = Counter(all_balls)

        # Assigner des poids inversés aux fréquences
        weights = []
        for num in range(1, MAX_BALL + 1):
            count = freq.get(num, 0)
            weight = 1.0 / (count + 1)  # inverse fréquence + 1 pour éviter div by zero
            weights.append(weight)

        weights = np.array(weights) / sum(weights)

        # Sélectionner NUM_BALLS numéros selon les poids
        selected = sorted(np.random.choice(MAX_BALL, size=NUM_BALLS, replace=False, p=weights) + 1)
        return [selected]


class PairedNumbersStrategy(BaseStrategy):
    """Joue en se basant sur les paires les plus fréquentes."""

    def __init__(self, n_combinations: int = 5, window: int = 200):
        super().__init__("Paired Numbers")
        self.n_combinations = n_combinations
        self.window = window

    def select_numbers(self, df: pd.DataFrame, n_draws_to_use: int | None = None) -> list[list[int]]:
        n = n_draws_to_use or self.window
        recent = df.tail(n) if n < len(df) else df

        # Compter les paires fréquentes
        pair_freq = Counter()
        for _, row in recent.iterrows():
            balls = sorted([int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])])
            from itertools import combinations as it_combs
            for pair in it_combs(balls, 2):
                pair_freq[pair] += 1

        # Récupérer les numéros les plus liés aux paires fréquentes
        top_pairs = [pair for pair, _ in pair_freq.most_common(30)]
        partner_freq = Counter()
        for pair, count in top_pairs:
            partner_freq[pair[0]] += count
            partner_freq[pair[1]] += count

        # Choisir les numéros les plus fréquemment partenaires
        sorted_partners = [num for num, _ in partner_freq.most_common(NUM_BALLS * 2)]

        results = []
        seen = set()

        for _ in range(self.n_combinations):
            if len(sorted_partners) < NUM_BALLS:
                break
            subset = sorted(np.random.choice(sorted_partners, size=NUM_BALLS, replace=False))
            key = tuple(subset)
            if key not in seen:
                seen.add(key)
                results.append(subset)

        return results


class WeightedRandomStrategy(BaseStrategy):
    """
    Pondération inverse des fréquences + randomisation.
    Donne un avantage aux numéros froids tout en gardant de l'aléatoire.
    """

    def __init__(self, n_combinations: int = 5, window: int = 200, bias: float = 2.0):
        super().__init__("Weighted Random")
        self.n_combinations = n_combinations
        self.window = window
        self.bias = bias  # paramètre de biais (plus = plus favorise les froids)

    def select_numbers(self, df: pd.DataFrame, n_draws_to_use: int | None = None) -> list[list[int]]:
        n = n_draws_to_use or self.window
        recent = df.tail(n) if n < len(df) else df

        all_balls = []
        for _, row in recent.iterrows():
            balls = [int(row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(row[f"boule_{i}"])]
            all_balls.extend(balls)

        freq = Counter(all_balls)

        # Poids: P(numéro) ∝ 1/(freq+1)^bias
        weights = []
        for num in range(1, MAX_BALL + 1):
            f = freq.get(num, 0)
            w = (1.0 / (f + 1)) ** self.bias
            weights.append(w)

        weights = np.array(weights) / sum(weights)
        results = []
        seen = set()

        for _ in range(self.n_combinations):
            subset = sorted(np.random.choice(MAX_BALL, size=NUM_BALLS, replace=False, p=weights) + 1)
            key = tuple(subset)
            if key not in seen:
                seen.add(key)
                results.append(subset)

        return results


class AllCombinationBacktestStrategy(BaseStrategy):
    """Simule un jeu systématique sur toutes les combinaisons (pour calculer les odds)."""

    def __init__(self):
        super().__init__("All Combinations (Theoretical)")

    def theoretical_odds(self) -> dict:
        """Calcule les probabilités théoriques pour chaque rang."""
        total_combos = self._n_combinations(MAX_BALL, NUM_BALLS)
        odds = {}

        # Rang 1: 5 bons + chance = 1/combinaison_totale * 1/chance
        odds["rang1"] = {
            "matches": "5 + chance",
            "probability": 1.0 / (total_combos * NUM_CHANCE_MAX),
            "odds_ratio": f"1/{total_combos * NUM_CHANCE_MAX}",
        }

        # Rang 2: 5 bons sans chance
        odds["rang2"] = {
            "matches": "5",
            "probability": (NUM_CHANCE_MAX - 1) / (total_combos * NUM_CHANCE_MAX),
            "odds_ratio": f"1/{total_combos}",
        }

        # Rang 3: 4 bons
        odds["rang3"] = {
            "matches": "4",
            "probability": (NUM_BALLS * (MAX_BALL - NUM_BALLS)) / total_combos * (NUM_CHANCE_MAX - 1) / NUM_CHANCE_MAX,
            "odds_ratio": f"1~{total_combos // (NUM_BALLS * (MAX_BALL - NUM_BALLS))}",
        }

        # Rang 4: 3 bons
        odds["rang4"] = {
            "matches": "3",
        }

        # Rang 5: 2 bons + chance
        odds["rang5"] = {
            "matches": "2 + chance",
        }

        return odds

    def _n_combinations(self, n: int, k: int) -> int:
        from math import comb
        return comb(n, k)


def evaluate_strategy(strategy: BaseStrategy, df: pd.DataFrame, budget_per_draw: float = 2.0) -> dict:
    """
    Évalue la rentabilité d'une stratégie sur les données historiques.

    Simule: à chaque tirage, on joue les numéros choisis par la stratégie,
    puis on compare avec le vrai tirage.

    Returns un dict avec les métriques de performance.
    """
    total_invested = 0
    total_won = 0
    wins_by_rank = Counter()
    max_win = 0
    win_dates = []
    running_balance = []

    n_draws = len(df)

    for draw_idx in range(20, n_draws):  # skip first 20 as warmup
        # Utiliser les tirages jusqu'ici pour choisir les numéros
        train_df = df.iloc[:draw_idx]
        combos = strategy.select_numbers(train_df)

        # Coût du jeu (une combinaison = 2€)
        cost = len(combos) * budget_per_draw
        total_invested += cost

        # Tirage réel de ce jour
        current_row = df.iloc[draw_idx]
        actual_balls = sorted([int(current_row[f"boule_{i}"]) for i in range(1, NUM_BALLS + 1) if pd.notna(current_row[f"boule_{i}"])])
        actual_chance = int(current_row["numero_chance"])

        # Vérifier chaque combinaison jouée
        draw_won = 0
        for combo in combos:
            sorted_combo = sorted(combo[:NUM_BALLS])
            matches = len(set(sorted_combo) & set(actual_balls))

            if matches == NUM_BALLS and actual_chance == (combo[NUM_BALLS] if len(combo) > NUM_BALLS else None):
                # Jackpot!
                draw_won += 5000000  # estimation
            elif matches == NUM_BALLS:
                draw_won += 1000000  # Rang 2 estimé
            elif matches >= 4:
                draw_won += 150  # Rang 3+ estimé

        wins_by_rank[str(matches)] += len(combos) if draw_won > 0 else 0
        total_won += draw_won
        max_win = max(max_win, draw_won)
        if draw_won > 0:
            win_dates.append(str(current_row.get("date_tirage", "")))

        running_balance.append(total_won - total_invested)

    roi = ((total_won - total_invested) / total_invested * 100) if total_invested > 0 else 0
    net_profit = total_won - total_invested

    return {
        "strategy_name": strategy.name,
        "total_invested": round(total_invested, 2),
        "total_won": round(total_won, 2),
        "net_profit": round(net_profit, 2),
        "roi_pct": round(roi, 2),
        "max_win": max_win,
        "win_dates": win_dates[:10],
        "final_balance": running_balance[-1] if running_balance else 0,
        "draws_evaluated": n_draws - 20,
    }


def run_chance_evaluation(df: pd.DataFrame) -> list[dict]:
    """Évalue les stratégies numéro de chance (méthodes alternatives)."""
    from src.loto.chance_strategy import evaluate_chance_strategy

    results = []
    for method in ["frequency", "day_conditional", "recent"]:
        print(f"Évaluation chance '{method}'...")
        result = evaluate_chance_strategy(df, method=method)
        results.append(result)
        print(f"  Stratégie ROI: {result['strategy']['roi_pct']}% | Baseline: {result['random_baseline']['roi_pct']}% | Amélioration: {result['improvement']}pp")

    return results


def run_strategy_backtest(df: pd.DataFrame) -> list[dict]:
    """Évalue toutes les stratégies sur les données historiques."""
    strategies = [
        HotNumbersStrategy(window=100),
        ColdNumbersStrategy(window=200),
        PairedNumbersStrategy(n_combinations=3, window=200),
        WeightedRandomStrategy(n_combinations=3, window=200, bias=1.5),
        WeightedRandomStrategy(n_combinations=3, window=200, bias=3.0),
    ]

    results = []
    for strategy in strategies:
        print(f"Évaluation de: {strategy.name}...")
        result = evaluate_strategy(strategy, df)
        results.append(result)
        print(f"  ROI: {result['roi_pct']}%, Profit net: {result['net_profit']:.2f}€")

    return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from src.loto.analysis import get_dataframe
    df = get_dataframe()
    results = run_strategy_backtest(df)
    for r in results:
        print(f"\n{r['strategy_name']}:")
        print(f"  ROI: {r['roi_pct']}% | Net: {r['net_profit']}€ | Max win: {r['max_win']}€")
