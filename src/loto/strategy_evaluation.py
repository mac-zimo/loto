"""Legacy strategy evaluation routines, disabled by default in the CLI."""

from collections import Counter

import pandas as pd

from loto.config import NUM_BALLS
from loto.strategy import (
    BaseStrategy,
    ColdNumbersStrategy,
    HotNumbersStrategy,
    PairedNumbersStrategy,
    WeightedRandomStrategy,
)


def evaluate_strategy(
    strategy: BaseStrategy,
    df: pd.DataFrame,
    budget_per_draw: float = 2.0,
) -> dict:
    """Evaluate a legacy strategy against historical draws."""
    total_invested = 0.0
    total_won = 0.0
    wins_by_rank = Counter()
    max_win = 0.0
    win_dates = []
    running_balance = []
    n_draws = len(df)

    for draw_index in range(20, n_draws):
        combinations = strategy.select_numbers(df.iloc[:draw_index])
        total_invested += len(combinations) * budget_per_draw
        current = df.iloc[draw_index]
        actual_balls = sorted(
            int(current[f"boule_{position}"])
            for position in range(1, NUM_BALLS + 1)
            if pd.notna(current[f"boule_{position}"])
        )
        actual_chance = int(current["numero_chance"])
        draw_won = 0.0
        matches = 0
        for combination in combinations:
            selected = sorted(combination[:NUM_BALLS])
            matches = len(set(selected) & set(actual_balls))
            selected_chance = combination[NUM_BALLS] if len(combination) > NUM_BALLS else None
            if matches == NUM_BALLS and actual_chance == selected_chance:
                draw_won += 5_000_000
            elif matches == NUM_BALLS:
                draw_won += 1_000_000
            elif matches >= 4:
                draw_won += 150
        wins_by_rank[str(matches)] += len(combinations) if draw_won > 0 else 0
        total_won += draw_won
        max_win = max(max_win, draw_won)
        if draw_won > 0:
            win_dates.append(str(current.get("date_tirage", "")))
        running_balance.append(total_won - total_invested)

    net_profit = total_won - total_invested
    roi = net_profit / total_invested * 100 if total_invested else 0
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
    """Evaluate the legacy chance-number strategies."""
    from loto.chance_strategy import evaluate_chance_strategy

    results = []
    for method in ["frequency", "day_conditional", "recent"]:
        print(f"Évaluation chance '{method}'...")
        result = evaluate_chance_strategy(df, method=method)
        results.append(result)
        print(
            f"  Stratégie ROI: {result['strategy']['roi_pct']}% | "
            f"Baseline: {result['random_baseline']['roi_pct']}% | "
            f"Amélioration: {result['improvement']}pp"
        )
    return results


def run_strategy_backtest(df: pd.DataFrame) -> list[dict]:
    """Evaluate every legacy number-selection strategy."""
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
