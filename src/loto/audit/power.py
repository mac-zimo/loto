"""Backward-compatible façade for task 1.3 marginal power analysis.

Models, simulation, observed classification, statistics, and exports live in
cohesive modules. Existing imports from :mod:`loto.audit.power` remain valid.
"""

from loto.audit.power_classification import audit_detectability
from loto.audit.power_exports import export_power_data, plot_power_curves
from loto.audit.power_models import (
    DEFAULT_ALPHA,
    DEFAULT_CONFIDENCE,
    DEFAULT_EFFECT_SIZES,
    DEFAULT_EQUIVALENCE_MARGIN,
    DEFAULT_N_DRAWS,
    DEFAULT_NULL_REPETITIONS,
    DEFAULT_POWER_REPETITIONS,
    DEFAULT_POWER_TARGET,
    DEFAULT_SEED,
    AuditStatus,
    Detectability,
    DetectabilityReport,
    DetectabilityResult,
    PowerCurveReport,
    PowerPoint,
    Scenario,
)
from loto.audit.power_simulation import estimate_power_curves, simulate_biased_draws
from loto.audit.power_statistics import (
    _critical_value,
    _monte_carlo_p_value,
    minimum_null_repetitions,
    simultaneous_max_deviation_upper_bound,
)

__all__ = [
    "DEFAULT_ALPHA",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_EFFECT_SIZES",
    "DEFAULT_EQUIVALENCE_MARGIN",
    "DEFAULT_N_DRAWS",
    "DEFAULT_NULL_REPETITIONS",
    "DEFAULT_POWER_REPETITIONS",
    "DEFAULT_POWER_TARGET",
    "DEFAULT_SEED",
    "AuditStatus",
    "Detectability",
    "DetectabilityReport",
    "DetectabilityResult",
    "PowerCurveReport",
    "PowerPoint",
    "Scenario",
    "audit_detectability",
    "estimate_power_curves",
    "export_power_data",
    "minimum_null_repetitions",
    "plot_power_curves",
    "simulate_biased_draws",
    "simultaneous_max_deviation_upper_bound",
]
