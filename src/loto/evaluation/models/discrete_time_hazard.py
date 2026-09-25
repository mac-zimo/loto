"""Pre-registered discrete-time hazards for 49 marginal recurrent processes.

For each number, age 0 means that it appeared at draw t-1, age 1 that its last
appearance was t-2, and so on. Observations before that number's first appearance
are left-censored and excluded. Every later target is an observed risk exposure;
an appearance is an event and resets age. A final right-censored spell contributes
its observed exposures but no invented future event.

The fixed age bins are estimated independently with a Beta-Bernoulli posterior
mean. The Beta smoothing only prevents empty cells from being non-estimable.
These 49 marginal hazards are not an exact joint law for a five-number draw
without replacement. Bounded-simplex projection merely enforces the marginal
5/49 contract. Bins, prior, and minimum history are pre-registered, with one
trial and no evaluation-period tuning. This model makes no real-world performance
claim.
"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from datetime import date
import math
from numbers import Real
from typing import Any

import numpy as np

from .._walk_forward_metrics import NUMBER_COUNT, validate_probabilities
from ..walk_forward import DrawObservation, History, WalkForwardCallbacks
from ._model_provenance import (
    ObservationFingerprint,
    observation_fingerprint as _observation_fingerprint,
    validate_training_provenance as _validate_training_provenance,
    validate_visible_training_overlap as _validate_visible_training_overlap,
)
from .dirichlet_multinomial import project_bounded_simplex

AGE_BIN_LOWER_BOUNDS = (0, 1, 2, 3, 4, 8, 16, 32)
PRIOR_MEAN = 5 / 49
PRIOR_STRENGTH = 10.0
MINIMUM_HISTORY = 60
TRIAL_BUDGET = 1
PREREGISTERED_METRICS = (
    "log_loss",
    "brier",
    "mean_matches",
    "calibration_error",
    "mean_true_number_rank",
    "regret_vs_uniform",
)
ALLOWED_PARAMETERS = (
    "age_bin_lower_bounds",
    "prior_mean",
    "prior_strength",
    "minimum_history",
)
PRIMARY_BASELINES = (
    "uniform",
    "cumulative_frequency_alpha=1.0",
    "dirichlet_multinomial_marginal_concentration=1.0_minimum_history=1_"
    "prior=uniform_5_over_49",
    "logistic_regression_per_number_l2_lags=1,2,3_windows=5,10,25,50_"
    "C=1.0_penalty=l2_solver=liblinear_max_iter=1000_tol=0.0001_"
    "minimum_history=60_standardize=true_random_state=0",
)
PROMOTION_RULE = (
    "Promote only after out-of-sample improvement over the primary baselines is "
    "statistically and economically significant on at least three temporal periods, "
    "then confirmed on an untouched final holdout."
)
REJECTION_RULE = (
    "Reject unless out-of-sample improvement over the primary baselines is "
    "statistically and economically significant on at least three temporal periods "
    "and is then confirmed on an untouched final holdout."
)
PREREGISTRATION_NOTE = (
    "Exactly one fixed experiment; no grid, search, tuning of bins, prior strength, "
    "or minimum history on evaluation metrics is permitted."
)
INSUFFICIENT_DATA_BEHAVIOR = (
    "reject deterministically before fitting when len(history) < minimum_history; "
    "no silent fallback"
)
AGE_DEFINITION = (
    "age 0 iff present at t-1; otherwise elapsed draws since last appearance minus one"
)
LEFT_CENSOR_POLICY = (
    "left-censored before first observed appearance: exclude all such targets"
)
RIGHT_CENSOR_POLICY = (
    "right-censored final spell: count observed exposures and invent no future event"
)
NEVER_OBSERVED_POLICY = "reject fit if any number was never observed; no fallback"
EMPTY_BIN_POLICY = "use the Beta prior alone when a class has no exposure"
_FIT_VALIDATION_TOKEN = object()


def _finite_probability(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number in (0, 1)")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be a finite real number in (0, 1)")
    return result


def _finite_positive_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite positive real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_age_bins(values: object) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise TypeError("age_bin_lower_bounds must be an immutable tuple of integers")
    if not values:
        raise ValueError("age_bin_lower_bounds must not be empty")
    if any(
        not isinstance(value, int) or isinstance(value, (bool, np.bool_))
        for value in values
    ):
        raise TypeError("age_bin_lower_bounds must contain only integers")
    if values[0] != 0:
        raise ValueError("age_bin_lower_bounds must start at 0")
    if any(value < 0 for value in values):
        raise ValueError("age_bin_lower_bounds must be non-negative")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise ValueError(
            "age_bin_lower_bounds must be strictly increasing without duplicates"
        )
    return values


def _validate_history(history: History, *, minimum_size: int) -> History:
    if not isinstance(history, tuple):
        raise TypeError("history must be an immutable tuple")
    if len(history) < minimum_size:
        raise ValueError(f"history must contain at least {minimum_size} observations")
    if any(not isinstance(observation, DrawObservation) for observation in history):
        raise TypeError("history must contain only DrawObservation values")
    if any(
        left.draw_date >= right.draw_date
        for left, right in zip(history, history[1:])
    ):
        raise ValueError("draw dates must be strictly increasing")
    indices = tuple(observation.original_index for observation in history)
    if len(indices) != len(set(indices)):
        raise ValueError("original_index values must be unique")
    return history


def _validate_rng(rng: np.random.Generator) -> None:
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")


def _age_bin_index(age: int, bounds: tuple[int, ...]) -> int:
    for index in range(len(bounds) - 1, -1, -1):
        if age >= bounds[index]:
            return index
    raise ValueError("age must be non-negative")


@dataclass(frozen=True, slots=True)
class HazardRow:
    """One auditable at-risk exposure, created only after first observation."""

    number: int
    target_position: int
    target_date: date
    target_original_index: int
    age: int
    age_bin_index: int
    label: int


@dataclass(frozen=True, slots=True)
class DiscreteTimeHazardConfig:
    """Immutable registration of the one fixed hazard experiment."""

    age_bin_lower_bounds: tuple[int, ...] = AGE_BIN_LOWER_BOUNDS
    prior_mean: float = PRIOR_MEAN
    prior_strength: float = PRIOR_STRENGTH
    minimum_history: int = MINIMUM_HISTORY
    identifier: str = field(init=False)
    family: str = field(
        default="per_number_discrete_time_recurrent_event_hazard", init=False
    )
    age_definition: str = field(default=AGE_DEFINITION, init=False)
    left_censor_policy: str = field(default=LEFT_CENSOR_POLICY, init=False)
    right_censor_policy: str = field(default=RIGHT_CENSOR_POLICY, init=False)
    never_observed_policy: str = field(default=NEVER_OBSERVED_POLICY, init=False)
    empty_bin_policy: str = field(default=EMPTY_BIN_POLICY, init=False)
    trial_budget: int = field(default=TRIAL_BUDGET, init=False)
    allowed_parameters: tuple[str, ...] = field(
        default=ALLOWED_PARAMETERS, init=False
    )
    primary_baselines: tuple[str, ...] = field(default=PRIMARY_BASELINES, init=False)
    metrics: tuple[str, ...] = field(default=PREREGISTERED_METRICS, init=False)
    promotion_rule: str = field(default=PROMOTION_RULE, init=False)
    rejection_rule: str = field(default=REJECTION_RULE, init=False)
    preregistration_note: str = field(default=PREREGISTRATION_NOTE, init=False)
    insufficient_data_behavior: str = field(
        default=INSUFFICIENT_DATA_BEHAVIOR, init=False
    )

    def __post_init__(self) -> None:
        bounds = _validate_age_bins(self.age_bin_lower_bounds)
        prior_mean = _finite_probability(self.prior_mean, name="prior_mean")
        prior_strength = _finite_positive_real(
            self.prior_strength, name="prior_strength"
        )
        minimum_history = _positive_integer(
            self.minimum_history, name="minimum_history"
        )
        beta_pseudocounts = (
            prior_strength * prior_mean,
            prior_strength * (1.0 - prior_mean),
        )
        if any(
            not math.isfinite(pseudocount) or pseudocount <= 0.0
            for pseudocount in beta_pseudocounts
        ):
            raise ValueError(
                "Beta prior pseudocounts must be finite and strictly positive"
            )

        object.__setattr__(self, "age_bin_lower_bounds", bounds)
        object.__setattr__(self, "prior_mean", prior_mean)
        object.__setattr__(self, "prior_strength", prior_strength)
        object.__setattr__(self, "minimum_history", minimum_history)
        object.__setattr__(
            self,
            "identifier",
            "discrete_time_hazard_per_number_recurrent_"
            f"bins={','.join(map(str, bounds))}_"
            f"prior_mean={prior_mean!r}_prior_strength={prior_strength!r}_"
            f"minimum_history={minimum_history}_"
            "age=elapsed_since_last_minus_one_left=exclude_before_first_"
            "right=count_observed_exposures_only_never_observed=reject_"
            "empty_bin=beta_prior_only",
        )


def _hazard_observation_counts(
    history: History, bounds: tuple[int, ...]
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    """Count all cells in one chronological O(n * 49) state-machine pass."""

    exposures = [[0] * len(bounds) for _ in range(NUMBER_COUNT)]
    events = [[0] * len(bounds) for _ in range(NUMBER_COUNT)]
    last_seen: list[int | None] = [None] * NUMBER_COUNT
    first_positions: list[int | None] = [None] * NUMBER_COUNT

    for position, observation in enumerate(history):
        present = set(observation.numbers)
        for number_index in range(NUMBER_COUNT):
            previous = last_seen[number_index]
            if previous is not None:
                age = position - previous - 1
                age_bin = _age_bin_index(age, bounds)
                exposures[number_index][age_bin] += 1
                if number_index + 1 in present:
                    events[number_index][age_bin] += 1
            if number_index + 1 in present:
                if first_positions[number_index] is None:
                    first_positions[number_index] = position
                last_seen[number_index] = position

    missing = next(
        (index + 1 for index, position in enumerate(first_positions) if position is None),
        None,
    )
    if missing is not None:
        raise ValueError(f"number {missing} was never observed in training history")
    return (
        tuple(tuple(row) for row in exposures),
        tuple(tuple(row) for row in events),
        tuple(int(position) for position in first_positions if position is not None),
    )


def _count_hazard_observations(
    history: History, bounds: tuple[int, ...]
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
]:
    """Fit-path entry point for the shared chronological state machine."""

    return _hazard_observation_counts(history, bounds)


def build_hazard_rows(
    history: History, config: DiscreteTimeHazardConfig
) -> tuple[HazardRow, ...]:
    """Materialize chronological audit rows; the fit path does not call this API."""

    if not isinstance(config, DiscreteTimeHazardConfig):
        raise TypeError("config must be a DiscreteTimeHazardConfig")
    checked = _validate_history(history, minimum_size=1)
    rows: list[HazardRow] = []
    last_seen: list[int | None] = [None] * NUMBER_COUNT
    for position, observation in enumerate(checked):
        present = set(observation.numbers)
        for number_index in range(NUMBER_COUNT):
            previous = last_seen[number_index]
            if previous is not None:
                age = position - previous - 1
                rows.append(
                    HazardRow(
                        number=number_index + 1,
                        target_position=position,
                        target_date=observation.draw_date,
                        target_original_index=observation.original_index,
                        age=age,
                        age_bin_index=_age_bin_index(
                            age, config.age_bin_lower_bounds
                        ),
                        label=int(number_index + 1 in present),
                    )
                )
            if number_index + 1 in present:
                last_seen[number_index] = position
    return tuple(rows)


def _integer_matrix(
    values: object, *, name: str, columns: int
) -> tuple[tuple[int, ...], ...]:
    if type(values) is not tuple or any(type(row) is not tuple for row in values):
        raise TypeError(f"state {name} must be an immutable tuple matrix")
    if len(values) != NUMBER_COUNT or any(len(row) != columns for row in values):
        raise ValueError(f"state {name} has invalid dimensions")
    if any(
        not isinstance(value, int) or isinstance(value, (bool, np.bool_))
        for row in values
        for value in row
    ):
        raise TypeError(f"state {name} must contain only integers")
    if any(value < 0 for row in values for value in row):
        raise ValueError(f"state {name} must contain only non-negative values")
    return values


def _real_matrix(
    values: object, *, name: str, columns: int
) -> tuple[tuple[float, ...], ...]:
    if type(values) is not tuple or any(type(row) is not tuple for row in values):
        raise TypeError(f"state {name} must be an immutable tuple matrix")
    if len(values) != NUMBER_COUNT or any(len(row) != columns for row in values):
        raise ValueError(f"state {name} has invalid dimensions")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for row in values
        for value in row
    ):
        raise TypeError(f"state {name} must contain only real numbers")
    result = tuple(tuple(float(value) for value in row) for row in values)
    if not all(math.isfinite(value) for row in result for value in row):
        raise ValueError(f"state {name} must contain only finite values")
    return result


def _integer_vector(
    values: object, *, name: str, minimum: int = 0
) -> tuple[int, ...]:
    if type(values) is not tuple:
        raise TypeError(f"state {name} must be an immutable tuple")
    if len(values) != NUMBER_COUNT:
        raise ValueError(f"state {name} must contain exactly 49 values")
    if any(
        not isinstance(value, int) or isinstance(value, (bool, np.bool_))
        for value in values
    ):
        raise TypeError(f"state {name} must contain only integers")
    if any(value < minimum for value in values):
        raise ValueError(f"state {name} values must be at least {minimum}")
    return values


@dataclass(frozen=True, slots=True)
class DiscreteTimeHazardState:
    """Frozen sufficient statistics, estimates, provenance, and audit metadata."""

    config_identifier: str
    age_bin_lower_bounds: tuple[int, ...]
    prior_mean: float
    prior_strength: float
    age_definition: str
    left_censor_policy: str
    right_censor_policy: str
    never_observed_policy: str
    empty_bin_policy: str
    exposure_counts: tuple[tuple[int, ...], ...]
    event_counts: tuple[tuple[int, ...], ...]
    hazards: tuple[tuple[float, ...], ...]
    rows_per_number: tuple[int, ...]
    total_risk_rows: int
    events_per_number: tuple[int, ...]
    first_observation_positions: tuple[int, ...]
    first_observation_fingerprints: tuple[ObservationFingerprint, ...]
    training_observation_count: int
    fitted_through_original_index: int
    fitted_through_date: date
    training_provenance: tuple[ObservationFingerprint, ...]
    _fit_validation_token: InitVar[object | None] = None

    def __post_init__(self, _fit_validation_token: object | None) -> None:
        _validate_state(
            self,
            recount_training_history=_fit_validation_token is not _FIT_VALIDATION_TOKEN,
        )


def _validate_state(
    state: DiscreteTimeHazardState, *, recount_training_history: bool = True
) -> None:
    if not isinstance(state.config_identifier, str) or not state.config_identifier:
        raise ValueError("state config_identifier must be non-empty")
    bounds = _validate_age_bins(state.age_bin_lower_bounds)
    prior_mean = _finite_probability(state.prior_mean, name="state prior_mean")
    prior_strength = _finite_positive_real(
        state.prior_strength, name="state prior_strength"
    )
    for name in (
        "age_definition",
        "left_censor_policy",
        "right_censor_policy",
        "never_observed_policy",
        "empty_bin_policy",
    ):
        if not isinstance(getattr(state, name), str) or not getattr(state, name):
            raise ValueError(f"state {name} must be non-empty")

    exposures = _integer_matrix(
        state.exposure_counts, name="exposure_counts", columns=len(bounds)
    )
    events = _integer_matrix(
        state.event_counts, name="event_counts", columns=len(bounds)
    )
    if any(
        event > exposure
        for exposure_row, event_row in zip(exposures, events, strict=True)
        for exposure, event in zip(exposure_row, event_row, strict=True)
    ):
        raise ValueError("state event_counts cannot exceed exposure_counts")
    hazards = _real_matrix(state.hazards, name="hazards", columns=len(bounds))
    expected_hazards = tuple(
        tuple(
            (event + prior_strength * prior_mean) / (exposure + prior_strength)
            for exposure, event in zip(exposure_row, event_row, strict=True)
        )
        for exposure_row, event_row in zip(exposures, events, strict=True)
    )
    if hazards != expected_hazards:
        raise ValueError("state hazards are inconsistent with counts and Beta formula")
    if any(not 0.0 < value < 1.0 for row in hazards for value in row):
        raise ValueError("state hazards must lie strictly inside (0, 1)")

    rows = _integer_vector(state.rows_per_number, name="rows_per_number")
    event_totals = _integer_vector(
        state.events_per_number, name="events_per_number"
    )
    if rows != tuple(sum(row) for row in exposures):
        raise ValueError("state rows_per_number is inconsistent with exposure_counts")
    if event_totals != tuple(sum(row) for row in events):
        raise ValueError("state events_per_number is inconsistent with event_counts")
    if (
        not isinstance(state.total_risk_rows, int)
        or isinstance(state.total_risk_rows, (bool, np.bool_))
        or state.total_risk_rows != sum(rows)
    ):
        raise ValueError("state total_risk_rows must equal rows_per_number total")
    training_count = _positive_integer(
        state.training_observation_count, name="state training_observation_count"
    )
    first_positions = _integer_vector(
        state.first_observation_positions, name="first_observation_positions"
    )
    if any(position >= training_count for position in first_positions):
        raise ValueError("state first_observation_positions exceed training history")
    expected_rows = tuple(training_count - position - 1 for position in first_positions)
    if rows != expected_rows:
        raise ValueError(
            "state exposure mass is inconsistent with first observation positions"
        )
    if sum(event_totals) != 5 * training_count - NUMBER_COUNT:
        raise ValueError("state event mass is inconsistent with recurrent first events")

    if not isinstance(state.fitted_through_original_index, int) or isinstance(
        state.fitted_through_original_index, (bool, np.bool_)
    ):
        raise TypeError("state fitted_through_original_index must be an integer")
    if type(state.fitted_through_date) is not date:
        raise TypeError("state fitted_through_date must be a datetime.date")
    provenance = _validate_training_provenance(
        state.training_provenance,
        expected_size=training_count,
        fitted_through_date=state.fitted_through_date,
        fitted_through_original_index=state.fitted_through_original_index,
    )
    if recount_training_history:
        training_history: History = tuple(
            DrawObservation(draw_date, original_index, numbers)
            for draw_date, original_index, numbers in provenance
        )
        expected_exposures, expected_events, expected_first_positions = (
            _hazard_observation_counts(training_history, bounds)
        )
        if exposures != expected_exposures:
            raise ValueError(
                "state exposure_counts are inconsistent with training provenance"
            )
        if events != expected_events:
            raise ValueError(
                "state event_counts are inconsistent with training provenance"
            )
        if first_positions != expected_first_positions:
            raise ValueError(
                "state first_observation_positions are inconsistent with training provenance"
            )
    if type(state.first_observation_fingerprints) is not tuple:
        raise TypeError(
            "state first_observation_fingerprints must be an immutable tuple"
        )
    if len(state.first_observation_fingerprints) != NUMBER_COUNT:
        raise ValueError(
            "state first_observation_fingerprints must contain exactly 49 values"
        )
    expected_first = tuple(provenance[position] for position in first_positions)
    if state.first_observation_fingerprints != expected_first:
        raise ValueError(
            "state first_observation_fingerprints are inconsistent with provenance"
        )
    for number_index, fingerprint in enumerate(
        state.first_observation_fingerprints
    ):
        if number_index + 1 not in fingerprint[2]:
            raise ValueError(
                "state first_observation_fingerprints omit their modelled number"
            )


@dataclass(frozen=True, slots=True)
class HazardPrediction:
    """Auditable hazards with exact ages distinguished from visible lower bounds.

    Each ``current_age_or_lower_bounds`` value is an exact age exactly when the
    corresponding ``age_is_exact`` flag is true. Otherwise it is only the lower
    bound proved by the visible history.
    """

    current_age_or_lower_bounds: tuple[int, ...]
    age_is_exact: tuple[bool, ...]
    age_bin_indices: tuple[int, ...]
    raw_hazards: tuple[float, ...]
    probabilities: tuple[float, ...]
    projection_applied: bool

    def __post_init__(self) -> None:
        ages = _integer_vector(
            self.current_age_or_lower_bounds, name="current_age_or_lower_bounds"
        )
        if type(self.age_is_exact) is not tuple:
            raise TypeError("age_is_exact must be an immutable tuple")
        if len(self.age_is_exact) != NUMBER_COUNT:
            raise ValueError("age_is_exact must contain exactly 49 values")
        if any(type(value) is not bool for value in self.age_is_exact):
            raise TypeError("age_is_exact must contain only booleans")
        bins = _integer_vector(self.age_bin_indices, name="age_bin_indices")
        if type(self.raw_hazards) is not tuple or len(self.raw_hazards) != NUMBER_COUNT:
            raise ValueError("state raw_hazards must contain exactly 49 values")
        if any(
            isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            for value in self.raw_hazards
        ):
            raise TypeError("state raw_hazards must contain only real numbers")
        raw = tuple(float(value) for value in self.raw_hazards)
        if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in raw):
            raise ValueError("state raw_hazards must be finite values in (0, 1)")
        try:
            probabilities = validate_probabilities(self.probabilities)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"probabilities are invalid: {exc}") from exc
        if not isinstance(self.projection_applied, bool):
            raise TypeError("projection_applied must be a boolean")
        if self.projection_applied is not (probabilities != raw):
            raise ValueError("projection_applied is inconsistent")
        object.__setattr__(self, "current_age_or_lower_bounds", ages)
        object.__setattr__(self, "age_bin_indices", bins)
        object.__setattr__(self, "raw_hazards", raw)
        object.__setattr__(self, "probabilities", probabilities)


def _current_age_information(
    history: History, bounds: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[int, ...]]:
    """Return exact ages or sufficient visible lower bounds and their bins."""

    last_seen: list[int | None] = [None] * NUMBER_COUNT
    for position, observation in enumerate(history):
        for number in observation.numbers:
            last_seen[number - 1] = position

    values: list[int] = []
    exactness: list[bool] = []
    bin_indices: list[int] = []
    for number_index, position in enumerate(last_seen):
        if position is not None:
            age = len(history) - position - 1
            values.append(age)
            exactness.append(True)
            bin_indices.append(_age_bin_index(age, bounds))
            continue

        lower_bound = len(history)
        if lower_bound < bounds[-1]:
            raise ValueError(
                f"number {number_index + 1} was not observed in visible history; "
                f"lower age bound {lower_bound} leaves its exact age bin unknown"
            )
        values.append(lower_bound)
        exactness.append(False)
        bin_indices.append(len(bounds) - 1)
    return tuple(values), tuple(exactness), tuple(bin_indices)


@dataclass(frozen=True, slots=True)
class DiscreteTimeHazardModel:
    config: DiscreteTimeHazardConfig

    def __post_init__(self) -> None:
        if not isinstance(self.config, DiscreteTimeHazardConfig):
            raise TypeError("config must be a DiscreteTimeHazardConfig")

    @property
    def callbacks(self) -> WalkForwardCallbacks:
        return WalkForwardCallbacks(fit=self.fit, predict=self.predict)

    def fit(
        self,
        history: History,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> DiscreteTimeHazardState:
        checked = _validate_history(
            history, minimum_size=self.config.minimum_history
        )
        _validate_rng(rng)
        exposures, events, first_positions = _count_hazard_observations(
            checked, self.config.age_bin_lower_bounds
        )
        hazards = tuple(
            tuple(
                (event + self.config.prior_strength * self.config.prior_mean)
                / (exposure + self.config.prior_strength)
                for exposure, event in zip(exposure_row, event_row, strict=True)
            )
            for exposure_row, event_row in zip(exposures, events, strict=True)
        )
        provenance = tuple(
            _observation_fingerprint(observation) for observation in checked
        )
        rows = tuple(sum(row) for row in exposures)
        event_totals = tuple(sum(row) for row in events)
        return DiscreteTimeHazardState(
            config_identifier=self.config.identifier,
            age_bin_lower_bounds=self.config.age_bin_lower_bounds,
            prior_mean=self.config.prior_mean,
            prior_strength=self.config.prior_strength,
            age_definition=self.config.age_definition,
            left_censor_policy=self.config.left_censor_policy,
            right_censor_policy=self.config.right_censor_policy,
            never_observed_policy=self.config.never_observed_policy,
            empty_bin_policy=self.config.empty_bin_policy,
            exposure_counts=exposures,
            event_counts=events,
            hazards=hazards,
            rows_per_number=rows,
            total_risk_rows=sum(rows),
            events_per_number=event_totals,
            first_observation_positions=first_positions,
            first_observation_fingerprints=tuple(
                provenance[position] for position in first_positions
            ),
            training_observation_count=len(checked),
            fitted_through_original_index=checked[-1].original_index,
            fitted_through_date=checked[-1].draw_date,
            training_provenance=provenance,
            _fit_validation_token=_FIT_VALIDATION_TOKEN,
        )

    def predict_details(
        self,
        history: History,
        state: DiscreteTimeHazardState,
        *,
        rng: np.random.Generator,
    ) -> HazardPrediction:
        checked = _validate_history(
            history, minimum_size=self.config.minimum_history
        )
        _validate_rng(rng)
        if not isinstance(state, DiscreteTimeHazardState):
            raise TypeError("state must be a DiscreteTimeHazardState")
        _validate_state(state)
        if (
            state.config_identifier != self.config.identifier
            or state.age_bin_lower_bounds != self.config.age_bin_lower_bounds
            or state.prior_mean != self.config.prior_mean
            or state.prior_strength != self.config.prior_strength
            or state.age_definition != self.config.age_definition
            or state.left_censor_policy != self.config.left_censor_policy
            or state.right_censor_policy != self.config.right_censor_policy
            or state.never_observed_policy != self.config.never_observed_policy
            or state.empty_bin_policy != self.config.empty_bin_policy
            or state.training_observation_count < self.config.minimum_history
        ):
            raise ValueError("state does not belong to this model configuration")
        _validate_visible_training_overlap(
            checked,
            training_provenance=state.training_provenance,
            fitted_through_date=state.fitted_through_date,
        )

        age_values, age_is_exact, bin_indices = _current_age_information(
            checked, self.config.age_bin_lower_bounds
        )
        raw = tuple(
            state.hazards[number_index][bin_indices[number_index]]
            for number_index in range(NUMBER_COUNT)
        )
        probabilities = project_bounded_simplex(raw)
        return HazardPrediction(
            current_age_or_lower_bounds=age_values,
            age_is_exact=age_is_exact,
            age_bin_indices=bin_indices,
            raw_hazards=raw,
            probabilities=probabilities,
            projection_applied=probabilities != raw,
        )

    def predict(
        self,
        history: History,
        state: Any,
        feature_data: Any,
        *,
        rng: np.random.Generator,
    ) -> tuple[float, ...]:
        if not isinstance(state, DiscreteTimeHazardState):
            raise TypeError("state must be a DiscreteTimeHazardState")
        return self.predict_details(history, state, rng=rng).probabilities


__all__ = [
    "AGE_BIN_LOWER_BOUNDS",
    "ALLOWED_PARAMETERS",
    "EMPTY_BIN_POLICY",
    "HazardPrediction",
    "HazardRow",
    "INSUFFICIENT_DATA_BEHAVIOR",
    "LEFT_CENSOR_POLICY",
    "NEVER_OBSERVED_POLICY",
    "PREREGISTERED_METRICS",
    "PRIMARY_BASELINES",
    "PRIOR_MEAN",
    "PRIOR_STRENGTH",
    "PROMOTION_RULE",
    "REJECTION_RULE",
    "RIGHT_CENSOR_POLICY",
    "TRIAL_BUDGET",
    "DiscreteTimeHazardConfig",
    "DiscreteTimeHazardModel",
    "DiscreteTimeHazardState",
    "build_hazard_rows",
]
