"""Input validation shared by power simulation and observed classification."""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def positive_integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return normalized


def unit_interval(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0 < normalized < 1:
        raise ValueError(f"{name} must be strictly between zero and one")
    return normalized


def effect(value: object, name: str, *, maximum: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must contain real numbers")
    normalized = float(value)
    if not np.isfinite(normalized) or not 0 <= normalized <= maximum:
        raise ValueError(
            f"{name} must be finite and between zero and {float(maximum)}"
        )
    return normalized
