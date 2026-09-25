"""Shared models and pre-registered constants for marginal power analysis."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

DEFAULT_ALPHA = 0.05
DEFAULT_POWER_TARGET = 0.80
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 20260924
DEFAULT_EFFECT_SIZES = (0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.10)
DEFAULT_EQUIVALENCE_MARGIN = 0.02
DEFAULT_N_DRAWS = 2_811
DEFAULT_NULL_REPETITIONS = 4_999
DEFAULT_POWER_REPETITIONS = 1_000

Scenario = Literal["main_marginal", "chance_marginal"]
Detectability = Literal["detectable", "insufficient_power"]
AuditStatus = Literal["bias_detected", "bias_absent", "insufficient_power"]
SCENARIOS: tuple[Scenario, ...] = ("main_marginal", "chance_marginal")
MAIN_BASELINE = 5 / 49
CHANCE_BASELINE = 0.1
MAIN_MAX_EFFECT = 1 - MAIN_BASELINE
CHANCE_MAX_EFFECT = 1 - CHANCE_BASELINE


@dataclass(frozen=True)
class PowerPoint:
    """Estimated rejection probability and Wilson binomial interval."""

    effect_size: float
    detections: int
    repetitions: int
    power: float
    interval_low: float
    interval_high: float
    detectability: Detectability


@dataclass(frozen=True)
class PowerCurveReport:
    """Pre-registered calibration and power curves for both marginal tests."""

    n_draws: int
    seed: int
    alpha: float
    per_test_alpha: float
    confidence: float
    power_target: float
    null_repetitions: int
    power_repetitions: int
    multiplicity_correction: str
    statistic: str
    bias_definition: str
    monte_carlo_method: str
    equivalence_margin: float
    equivalence_method: str
    critical_values: Mapping[Scenario, float]
    curves: Mapping[Scenario, tuple[PowerPoint, ...]]


@dataclass(frozen=True)
class DetectabilityResult:
    """Classification of one observed marginal audit."""

    statistic: float
    critical_value: float
    detected: bool
    equivalence_margin: float
    equivalence_upper_bound: float
    simultaneous_confidence: float
    status: AuditStatus


@dataclass(frozen=True)
class DetectabilityReport:
    """Observed classifications; non-detection is never treated as absence alone."""

    n_draws: int
    alpha: float
    equivalence_margin: float
    interpretation: str
    results: Mapping[Scenario, DetectabilityResult]
