"""Classification of observed draws using detection and equivalence evidence."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from loto.audit.power_models import (
    AuditStatus,
    CHANCE_BASELINE,
    DEFAULT_EQUIVALENCE_MARGIN,
    MAIN_BASELINE,
    MAIN_MAX_EFFECT,
    SCENARIOS,
    DetectabilityReport,
    DetectabilityResult,
    PowerCurveReport,
    Scenario,
)
from loto.audit.power_statistics import (
    marginal_statistics,
    simultaneous_max_deviation_upper_bound,
)
from loto.audit.power_validation import effect
from loto.audit.uniformity import _validate_draws


def audit_detectability(
    main_numbers: ArrayLike,
    chance: ArrayLike,
    *,
    power_report: PowerCurveReport,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
) -> DetectabilityReport:
    """Classify using detection and a separate simultaneous equivalence test.

    For each game, exact binomial intervals cover every marginal simultaneously
    at its Bonferroni-allocated confidence level. ``bias_absent`` requires the
    resulting upper bound on the maximum deviation to be strictly below the
    pre-registered margin; power of a difference test is not used as evidence
    of equivalence.
    """
    if not isinstance(power_report, PowerCurveReport):
        raise TypeError("power_report must be a PowerCurveReport")
    margin = effect(
        equivalence_margin,
        "equivalence_margin",
        maximum=MAIN_MAX_EFFECT,
    )
    main, chance_array = _validate_draws(main_numbers, chance)
    if len(main) != power_report.n_draws:
        raise ValueError("observed draws must match power_report.n_draws")
    statistics = marginal_statistics(main, chance_array)
    marginal_counts: dict[Scenario, NDArray[np.int64]] = {
        "main_marginal": np.bincount(main.ravel(), minlength=50)[1:],
        "chance_marginal": np.bincount(chance_array, minlength=11)[1:],
    }
    baselines = {
        "main_marginal": MAIN_BASELINE,
        "chance_marginal": CHANCE_BASELINE,
    }
    results: dict[Scenario, DetectabilityResult] = {}
    for scenario in SCENARIOS:
        statistic = statistics[scenario]
        critical = power_report.critical_values[scenario]
        detected = statistic > critical
        equivalence_bound = simultaneous_max_deviation_upper_bound(
            marginal_counts[scenario],
            n_draws=len(main),
            expected_probability=baselines[scenario],
            alpha=power_report.per_test_alpha,
        )
        if detected:
            status: AuditStatus = "bias_detected"
        elif equivalence_bound < margin:
            status = "bias_absent"
        else:
            status = "insufficient_power"
        results[scenario] = DetectabilityResult(
            statistic=statistic,
            critical_value=critical,
            detected=detected,
            equivalence_margin=margin,
            equivalence_upper_bound=equivalence_bound,
            simultaneous_confidence=1 - power_report.per_test_alpha,
            status=status,
        )
    return DetectabilityReport(
        n_draws=len(main),
        alpha=power_report.alpha,
        equivalence_margin=margin,
        interpretation=(
            "A non-detection is bias_absent only when an exact simultaneous "
            "Bonferroni confidence bound for every Bernoulli marginal is strictly "
            "inside the pre-specified equivalence margin; otherwise it is "
            "insufficient_power. Dependence within a draw does not invalidate "
            "the union-bound coverage."
        ),
        results=results,
    )
