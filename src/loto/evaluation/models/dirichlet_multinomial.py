"""Auditable Dirichlet shrinkage for marginal 5/49 inclusion counts.

This model deliberately concerns only the 49 marginal inclusion counts.  It uses
one categorical Dirichlet parameter per number for the ``5 * n`` observed
inclusions; an ordinary multinomial is not the exact joint law of a draw of five
distinct numbers without replacement.

The configured ``concentration`` is measured in draws (``alpha``), not in
categorical events.  Thus ``prior_parameters_i = alpha * prior_mean_i`` and the
Dirichlet prior mass is ``5 * alpha``.  The marginal posterior mean is
``(count_i + alpha * prior_mean_i) / (n + alpha)``.  With the uniform prior
``prior_mean_i = 5/49``, this is algebraically identical to
``CumulativeFrequencyBaseline(alpha)`` and is not a distinct predictive family.
The added value here is the immutable, validated posterior state and its
marginal posterior uncertainty, not a claim of a new point forecast.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import math
from numbers import Real
import struct
from typing import Any

import numpy as np

from .._walk_forward_metrics import (
    DRAW_SIZE,
    NUMBER_COUNT,
    PROBABILITY_SUM_TOLERANCE,
    UNIFORM_PROBABILITY,
    validate_probabilities,
)
from ..walk_forward import DrawObservation, History, WalkForwardCallbacks

UNIFORM_PRIOR_MEAN = (UNIFORM_PROBABILITY,) * NUMBER_COUNT
PREREGISTERED_METRICS = (
    "log_loss",
    "brier",
    "mean_matches",
    "calibration_error",
    "mean_true_number_rank",
    "regret_vs_uniform",
)
TRIAL_BUDGET = 1
ALLOWED_PARAMETERS = ("concentration", "prior_mean", "minimum_history")
PROMOTION_RULE = (
    "Promote only after out-of-sample improvement over the primary baseline is "
    "statistically and economically significant on at least three temporal periods, "
    "then confirmed on an untouched final holdout."
)
REJECTION_RULE = (
    "Reject unless out-of-sample improvement over the primary baseline is "
    "statistically and economically significant on at least three temporal periods "
    "and is then confirmed on an untouched final holdout."
)
PREREGISTRATION_NOTE = (
    "One trial with concentration, prior mean, minimum history, baseline, metrics, "
    "and decision rules fixed before evaluation; no evaluation-window tuning is "
    "permitted."
)
INSUFFICIENT_DATA_BEHAVIOR = (
    "reject deterministically before fitting when len(history) < minimum_history; "
    "no silent fallback"
)
UNIFORM_PRIOR_EQUIVALENCE = (
    "For prior_mean_i = 5/49, p_i = (count_i + alpha * 5/49) / (n + alpha), "
    "which is algebraically identical to CumulativeFrequencyBaseline(alpha) and "
    "is not a distinct predictive family."
)


def _finite_real(value: object, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        qualifier = "positive " if positive else ""
        raise TypeError(f"{name} must be a finite {qualifier}real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}real number")
    return result


def _validate_dirichlet_total_mass(concentration: float, *, name: str) -> None:
    if not math.isfinite(DRAW_SIZE * concentration):
        raise ValueError(
            f"{name} produces a Dirichlet total mass that must be finite"
        )


def _real_vector(
    values: Sequence[float],
    *,
    name: str,
    length: int = NUMBER_COUNT,
) -> tuple[float, ...]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain exactly {length} real numbers") from exc
    if len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw
    ):
        raise TypeError(f"{name} must contain only real numbers")
    result = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} values must all be finite")
    return result


def _validate_prior_mean(values: Sequence[float]) -> tuple[float, ...]:
    prior_mean = _real_vector(values, name="prior_mean")
    if any(value < 0.0 or value > 1.0 for value in prior_mean):
        raise ValueError("prior_mean values must lie in [0, 1]")
    prior_mass = math.fsum(prior_mean)
    if not math.isclose(
        prior_mass,
        float(DRAW_SIZE),
        rel_tol=0.0,
        abs_tol=PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError("prior_mean values must sum to 5")
    if any(value == 0.0 for value in prior_mean):
        raise ValueError("prior_mean values must be strictly positive")
    if prior_mean == UNIFORM_PRIOR_MEAN:
        return UNIFORM_PRIOR_MEAN

    scale = DRAW_SIZE / prior_mass
    normalized = [value * scale for value in prior_mean]
    residual = DRAW_SIZE - math.fsum(normalized)
    if residual != 0.0:
        for index, value in enumerate(normalized):
            adjusted = value + residual
            if 0.0 < adjusted <= 1.0:
                normalized[index] = adjusted
                break
        else:
            raise ValueError("prior_mean cannot be normalized within (0, 1]")

    normalized_prior = tuple(normalized)
    if any(value <= 0.0 or value > 1.0 for value in normalized_prior):
        raise ValueError("normalized prior_mean values must lie in (0, 1]")
    if math.fsum(normalized_prior) != float(DRAW_SIZE):
        raise ValueError("prior_mean cannot be normalized to sum exactly to 5")
    return normalized_prior


def _derive_prior_parameters(
    concentration: float, prior_mean: Sequence[float]
) -> tuple[float, ...]:
    _validate_dirichlet_total_mass(concentration, name="concentration")
    prior_parameters = tuple(concentration * value for value in prior_mean)
    if any(
        not math.isfinite(parameter) or parameter <= 0.0
        for parameter in prior_parameters
    ):
        raise ValueError(
            "concentration produces prior parameters that must all be finite and "
            "strictly positive"
        )
    return prior_parameters


def _posterior_variances(
    posterior_parameters: Sequence[float], total_mass: float
) -> tuple[float, ...]:
    return tuple(
        DRAW_SIZE**2
        * (parameter / total_mass)
        * (1.0 - parameter / total_mass)
        / (total_mass + 1.0)
        for parameter in posterior_parameters
    )


def _masses_are_consistent(actual: float, theoretical: float) -> bool:
    rounding_bound = NUMBER_COUNT * max(math.ulp(actual), math.ulp(theoretical))
    return math.isclose(actual, theoretical, rel_tol=0.0, abs_tol=rounding_bound)


def _prior_identifier(prior_mean: tuple[float, ...]) -> str:
    if prior_mean == UNIFORM_PRIOR_MEAN:
        return "uniform_5_over_49"
    digest = hashlib.sha256(struct.pack(f"!{NUMBER_COUNT}d", *prior_mean)).hexdigest()
    return f"sha256:{digest}"


def _validate_history(history: History, *, minimum_size: int) -> History:
    if not isinstance(history, tuple):
        raise TypeError("history must be an immutable tuple")
    if len(history) < minimum_size:
        required = (
            "one observation"
            if minimum_size == 1
            else f"{minimum_size} observations"
        )
        raise ValueError(f"history must contain at least {required}")
    if any(not isinstance(observation, DrawObservation) for observation in history):
        raise TypeError("history must contain only DrawObservation values")
    return history


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")


def project_bounded_simplex(values: Sequence[float]) -> tuple[float, ...]:
    """Euclidean projection onto ``{p in [0,1]^49: sum(p)=5}``.

    The unique projection has ``p_i = clip(values_i - lambda, 0, 1)``.  A fixed
    iteration bisection makes the threshold search deterministic.  Already valid
    vectors are returned unchanged to preserve auditable formula equality.  A
    finite input range whose reference translation overflows to a non-finite
    float is rejected because that projection cannot be computed in float
    arithmetic.
    """

    vector = _real_vector(values, name="values")
    if (
        all(0.0 <= value <= 1.0 for value in vector)
        and abs(math.fsum(vector) - DRAW_SIZE) <= 1e-12
    ):
        return vector

    reference = max(vector)
    translated = tuple(value - reference for value in vector)
    if not all(math.isfinite(value) for value in translated):
        raise ValueError("values translation must remain finite")
    lower = min(translated) - 1.0
    upper = max(translated)
    for _ in range(256):
        threshold = lower * 0.5 + upper * 0.5
        projected_sum = math.fsum(
            min(1.0, max(0.0, value - threshold)) for value in translated
        )
        if projected_sum > DRAW_SIZE:
            lower = threshold
        else:
            upper = threshold

    threshold = lower * 0.5 + upper * 0.5
    projected = [
        min(1.0, max(0.0, value - threshold)) for value in translated
    ]
    residual = DRAW_SIZE - math.fsum(projected)
    if residual != 0.0:
        if residual > 0.0:
            adjustable = [
                index for index, value in enumerate(projected) if 0.0 < value < 1.0
            ] + [index for index, value in enumerate(projected) if value == 0.0]
        else:
            adjustable = [
                index for index, value in enumerate(projected) if 0.0 < value < 1.0
            ] + [index for index, value in enumerate(projected) if value == 1.0]
        for position, index in enumerate(adjustable):
            if abs(residual) <= 1e-14:
                break
            remaining = len(adjustable) - position
            if residual > 0.0:
                adjustment = min(residual / remaining, 1.0 - projected[index])
            else:
                adjustment = max(residual / remaining, -projected[index])
            projected[index] += adjustment
            residual -= adjustment

    return validate_probabilities(projected)


@dataclass(frozen=True, slots=True)
class DirichletMultinomialConfig:
    """Immutable pre-registration for one fixed marginal shrinkage experiment."""

    concentration: float = 1.0
    prior_mean: tuple[float, ...] = UNIFORM_PRIOR_MEAN
    minimum_history: int = 1
    identifier: str = field(init=False)
    primary_baseline: str = field(init=False)
    trial_budget: int = field(default=TRIAL_BUDGET, init=False)
    allowed_parameters: tuple[str, ...] = field(
        default=ALLOWED_PARAMETERS, init=False
    )
    metrics: tuple[str, ...] = field(default=PREREGISTERED_METRICS, init=False)
    promotion_rule: str = field(default=PROMOTION_RULE, init=False)
    rejection_rule: str = field(default=REJECTION_RULE, init=False)
    preregistration_note: str = field(default=PREREGISTRATION_NOTE, init=False)
    insufficient_data_behavior: str = field(
        default=INSUFFICIENT_DATA_BEHAVIOR, init=False
    )

    def __post_init__(self) -> None:
        concentration = _finite_real(
            self.concentration, name="concentration", positive=True
        )
        if (
            not isinstance(self.minimum_history, int)
            or isinstance(self.minimum_history, (bool, np.bool_))
        ):
            raise TypeError("minimum_history must be a positive integer")
        if self.minimum_history < 1:
            raise ValueError("minimum_history must be a positive integer")
        prior_mean = _validate_prior_mean(self.prior_mean)
        _derive_prior_parameters(concentration, prior_mean)

        object.__setattr__(self, "concentration", concentration)
        object.__setattr__(self, "prior_mean", prior_mean)
        object.__setattr__(
            self,
            "identifier",
            f"dirichlet_multinomial_marginal_concentration={concentration!r}_"
            f"minimum_history={self.minimum_history}_"
            f"prior={_prior_identifier(prior_mean)}",
        )
        baseline_identifier = f"cumulative_frequency_alpha={concentration!r}"
        if prior_mean != UNIFORM_PRIOR_MEAN:
            baseline_identifier += (
                " (fixed uniform-prior comparator; not equivalent to the "
                "non-uniform-prior model)"
            )
        object.__setattr__(
            self,
            "primary_baseline",
            baseline_identifier,
        )


@dataclass(frozen=True, slots=True)
class DirichletMultinomialState:
    """Immutable sufficient statistics and posterior marginal summaries.

    ``posterior_variances`` belong to the working Dirichlet-multinomial model of
    categorical inclusions. They do not correctly incorporate the within-draw
    dependence of the exact scheme without replacement and are not uncertainty
    estimates for the true joint law.
    """

    sample_size: int
    concentration: float
    counts: tuple[int, ...]
    prior_parameters: tuple[float, ...]
    posterior_parameters: tuple[float, ...]
    raw_probabilities: tuple[float, ...]
    probabilities: tuple[float, ...]
    posterior_variances: tuple[float, ...]
    projection_applied: bool

    def __post_init__(self) -> None:
        for name in (
            "counts",
            "prior_parameters",
            "posterior_parameters",
            "raw_probabilities",
            "probabilities",
            "posterior_variances",
        ):
            try:
                object.__setattr__(self, name, tuple(getattr(self, name)))
            except TypeError as exc:
                raise TypeError(f"state {name} must be a sequence") from exc
        _validate_state(self)


def _validate_state(state: DirichletMultinomialState) -> None:
    if not isinstance(state.sample_size, int) or isinstance(
        state.sample_size, (bool, np.bool_)
    ):
        raise TypeError("state sample_size must be a positive integer")
    if state.sample_size < 1:
        raise ValueError("state sample_size must be a positive integer")
    concentration = _finite_real(
        state.concentration, name="state concentration", positive=True
    )
    _validate_dirichlet_total_mass(concentration, name="state concentration")

    if len(state.counts) != NUMBER_COUNT:
        raise ValueError("state counts must contain exactly 49 values")
    if any(
        not isinstance(count, int)
        or isinstance(count, (bool, np.bool_))
        for count in state.counts
    ):
        raise TypeError("state counts must contain only integers")
    if any(count < 0 or count > state.sample_size for count in state.counts):
        raise ValueError("state counts must lie between zero and sample_size")
    if sum(state.counts) != DRAW_SIZE * state.sample_size:
        raise ValueError("state counts must sum to five times sample_size")

    prior = _real_vector(state.prior_parameters, name="state prior_parameters")
    posterior = _real_vector(
        state.posterior_parameters, name="state posterior_parameters"
    )
    raw = _real_vector(state.raw_probabilities, name="state raw_probabilities")
    variances = _real_vector(
        state.posterior_variances, name="state posterior_variances"
    )
    if any(value <= 0.0 for value in prior):
        raise ValueError("state prior_parameters must be strictly positive")
    if any(value > concentration for value in prior):
        raise ValueError(
            "state prior_parameters must each be at most state concentration"
        )
    if not _masses_are_consistent(
        math.fsum(prior), DRAW_SIZE * concentration
    ):
        raise ValueError("state prior_parameters have the wrong total mass")

    expected_posterior = tuple(
        prior_value + count for prior_value, count in zip(prior, state.counts, strict=True)
    )
    if posterior != expected_posterior:
        raise ValueError("state posterior_parameters do not equal prior plus counts")

    denominator = state.sample_size + concentration
    expected_raw = tuple(parameter / denominator for parameter in posterior)
    if raw != expected_raw:
        raise ValueError("state raw_probabilities do not equal the posterior mean")
    if any(value < 0.0 or value > 1.0 for value in raw):
        raise ValueError("state raw_probabilities must lie in [0, 1]")

    expected_probabilities = project_bounded_simplex(raw)
    try:
        probabilities = validate_probabilities(state.probabilities)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"state probabilities are invalid: {exc}") from exc
    if probabilities != expected_probabilities:
        raise ValueError("state probabilities do not equal the bounded-simplex projection")

    if not isinstance(state.projection_applied, bool):
        raise TypeError("state projection_applied must be a boolean")
    expected_projection_applied = probabilities != raw
    if state.projection_applied is not expected_projection_applied:
        raise ValueError("state projection_applied is inconsistent")

    total_mass = math.fsum(posterior)
    theoretical_mass = DRAW_SIZE * denominator
    if not _masses_are_consistent(total_mass, theoretical_mass):
        raise ValueError("state posterior_parameters have the wrong total mass")
    expected_variances = _posterior_variances(posterior, total_mass)
    if any(value < 0.0 for value in variances):
        raise ValueError("state posterior_variances must be non-negative")
    if variances != expected_variances:
        raise ValueError("state posterior_variances are inconsistent")


@dataclass(frozen=True, slots=True)
class DirichletMultinomialModel:
    """Marginal inclusion model, not the exact joint law without replacement.

    The point forecast with a uniform prior is intentionally the existing
    cumulative-frequency forecast.  ``fit`` alone reads observations and creates
    posterior state; ``predict`` only validates and returns that frozen state.
    """

    config: DirichletMultinomialConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, DirichletMultinomialConfig):
            raise TypeError("config must be a DirichletMultinomialConfig")

    @property
    def callbacks(self) -> WalkForwardCallbacks:
        return WalkForwardCallbacks(fit=self.fit, predict=self.predict)

    def fit(
        self,
        history: History,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> DirichletMultinomialState:
        checked_history = _validate_history(
            history, minimum_size=self.config.minimum_history
        )
        _validate_rng(rng)

        counts = [0] * NUMBER_COUNT
        for observation in checked_history:
            for number in observation.numbers:
                counts[number - 1] += 1
        prior_parameters = _derive_prior_parameters(
            self.config.concentration, self.config.prior_mean
        )
        posterior_parameters = tuple(
            prior + count
            for prior, count in zip(prior_parameters, counts, strict=True)
        )
        denominator = len(checked_history) + self.config.concentration
        raw_probabilities = tuple(
            parameter / denominator for parameter in posterior_parameters
        )
        probabilities = project_bounded_simplex(raw_probabilities)
        total_mass = math.fsum(posterior_parameters)
        theoretical_mass = DRAW_SIZE * denominator
        if not _masses_are_consistent(total_mass, theoretical_mass):
            raise ValueError("posterior parameters have the wrong total mass")
        posterior_variances = _posterior_variances(
            posterior_parameters, total_mass
        )
        return DirichletMultinomialState(
            sample_size=len(checked_history),
            concentration=self.config.concentration,
            counts=tuple(counts),
            prior_parameters=prior_parameters,
            posterior_parameters=posterior_parameters,
            raw_probabilities=raw_probabilities,
            probabilities=probabilities,
            posterior_variances=posterior_variances,
            projection_applied=probabilities != raw_probabilities,
        )

    def predict(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        _validate_history(history, minimum_size=self.config.minimum_history)
        _validate_rng(rng)
        if not isinstance(state, DirichletMultinomialState):
            raise TypeError("state must be a DirichletMultinomialState")
        _validate_state(state)
        if state.sample_size < self.config.minimum_history:
            raise ValueError(
                "state sample_size must be at least config minimum_history"
            )
        expected_prior = _derive_prior_parameters(
            self.config.concentration, self.config.prior_mean
        )
        if (
            state.concentration != self.config.concentration
            or state.prior_parameters != expected_prior
        ):
            raise ValueError("state does not belong to this model configuration")
        return validate_probabilities(state.probabilities)


__all__ = [
    "ALLOWED_PARAMETERS",
    "INSUFFICIENT_DATA_BEHAVIOR",
    "PREREGISTERED_METRICS",
    "PREREGISTRATION_NOTE",
    "PROMOTION_RULE",
    "REJECTION_RULE",
    "TRIAL_BUDGET",
    "UNIFORM_PRIOR_EQUIVALENCE",
    "UNIFORM_PRIOR_MEAN",
    "DirichletMultinomialConfig",
    "DirichletMultinomialModel",
    "DirichletMultinomialState",
    "project_bounded_simplex",
]
