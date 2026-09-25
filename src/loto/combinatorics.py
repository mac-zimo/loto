"""Exact combinatorics for the current 5/49 plus Chance game."""

from math import comb

from loto.config import MAX_BALL, NUM_BALLS, NUM_CHANCE_MAX


def number_combination_count(
    pool_size: int = MAX_BALL,
    numbers_drawn: int = NUM_BALLS,
) -> int:
    """Return the number of unordered main-number combinations."""
    return comb(pool_size, numbers_drawn)


def jackpot_outcome_space(chance_numbers: int = NUM_CHANCE_MAX) -> int:
    """Return the number of main-number plus Chance jackpot outcomes."""
    return number_combination_count() * chance_numbers
