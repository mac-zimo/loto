from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
import inspect
import math
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from loto.evaluation.baselines import cumulative_frequency_baseline, uniform_baseline
from loto.evaluation.models.dirichlet_multinomial import (
    DirichletMultinomialConfig,
    DirichletMultinomialModel,
    project_bounded_simplex,
)
from loto.evaluation.models.discrete_time_hazard import (
    AGE_BIN_LOWER_BOUNDS,
    PREREGISTERED_METRICS,
    DiscreteTimeHazardConfig,
    DiscreteTimeHazardModel,
    DiscreteTimeHazardState,
    build_hazard_rows,
)
from loto.evaluation.models.logistic_regression import (
    LogisticRegressionConfig,
    LogisticRegressionModel,
)
from loto.evaluation.walk_forward import (
    DrawObservation,
    EvaluationWindow,
    RetrainPolicy,
    WalkForwardCallbacks,
    WalkForwardConfig,
    evaluate_predictions,
    run_walk_forward,
)


def _draws(count: int) -> tuple[DrawObservation, ...]:
    start = date(2020, 1, 1)
    return tuple(
        DrawObservation(
            draw_date=start + timedelta(days=position),
            original_index=1000 + position,
            numbers=tuple(
                ((position * 5 + offset) % 49) + 1 for offset in range(5)
            ),
        )
        for position in range(count)
    )


def _fit(model: DiscreteTimeHazardModel, history):
    return model.fit(history, None, rng=np.random.default_rng(10))


def _predict(model: DiscreteTimeHazardModel, history, state):
    return model.predict(history, state, None, rng=np.random.default_rng(11))


def _run(
    draws,
    model_or_baseline,
    *,
    policy=RetrainPolicy.EACH_DRAW,
    frequency=None,
    seed=3,
    max_train_size=None,
):
    return run_walk_forward(
        draws,
        (
            EvaluationWindow(
                0,
                60,
                60,
                len(draws),
                max_train_size=max_train_size,
            ),
        ),
        model_or_baseline.callbacks,
        WalkForwardConfig(
            initial_train_size=60,
            retrain_policy=policy,
            retrain_frequency=frequency,
            seed=seed,
        ),
    )


def _manual_age(history, number: int) -> int:
    for distance, observation in enumerate(reversed(history)):
        if number in observation.numbers:
            return distance
    raise ValueError("number has no known age")


def _without_number(draw: DrawObservation, number: int) -> DrawObservation:
    if number not in draw.numbers:
        return draw
    replacement = next(
        candidate
        for candidate in range(1, 50)
        if candidate != number and candidate not in draw.numbers
    )
    return replace(
        draw,
        numbers=tuple(
            replacement if value == number else value for value in draw.numbers
        ),
    )


def test_hazard_rows_manual_chronology_left_and_right_censoring():
    history = (
        DrawObservation(date(2020, 1, 1), 10, (1, 2, 3, 4, 5)),
        DrawObservation(date(2020, 1, 2), 11, (2, 3, 4, 5, 6)),
        DrawObservation(date(2020, 1, 3), 12, (1, 3, 4, 5, 6)),
        DrawObservation(date(2020, 1, 4), 13, (3, 4, 5, 6, 7)),
        DrawObservation(date(2020, 1, 5), 14, (3, 4, 5, 6, 7)),
    )
    rows = build_hazard_rows(history, DiscreteTimeHazardConfig())
    number_one = tuple(row for row in rows if row.number == 1)
    number_six = tuple(row for row in rows if row.number == 6)

    assert [
        (row.target_position, row.age, row.age_bin_index, row.label)
        for row in number_one
    ] == [(1, 0, 0, 0), (2, 1, 1, 1), (3, 0, 0, 0), (4, 1, 1, 0)]
    assert [row.target_position for row in number_six] == [2, 3, 4]
    assert [row.age for row in number_six] == [0, 0, 0]
    assert [row.label for row in number_six] == [1, 1, 1]
    assert not any(row.number == 7 and row.target_position < 4 for row in rows)
    # The final open spell contributes observed zero-labelled exposures only.
    assert number_one[-1].label == 0


def test_age_bins_cover_every_preregistered_boundary():
    history = list(_draws(60))
    # Number 49 appears at position 9 and never again, yielding ages through 49.
    history = tuple(
        _without_number(draw, 49) if position > 9 else draw
        for position, draw in enumerate(history)
    )
    rows = build_hazard_rows(history, DiscreteTimeHazardConfig())
    by_age = {row.age: row.age_bin_index for row in rows if row.number == 49}

    assert AGE_BIN_LOWER_BOUNDS == (0, 1, 2, 3, 4, 8, 16, 32)
    assert {age: by_age[age] for age in (0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 32)} == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        7: 4,
        8: 5,
        15: 5,
        16: 6,
        31: 6,
        32: 7,
    }


def test_future_change_does_not_change_past_audit_rows():
    history = _draws(62)
    changed = (*history[:61], replace(history[61], numbers=(40, 41, 42, 43, 44)))
    original = build_hazard_rows(history, DiscreteTimeHazardConfig())
    modified = build_hazard_rows(changed, DiscreteTimeHazardConfig())

    assert tuple(row for row in original if row.target_position < 61) == tuple(
        row for row in modified if row.target_position < 61
    )


def test_fit_counts_hazards_dimensions_and_prediction_are_manually_auditable():
    history = _draws(60)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, history)
    details = model.predict_details(history, state, rng=np.random.default_rng(12))
    rows = build_hazard_rows(history, model.config)

    assert len(state.exposure_counts) == len(state.event_counts) == len(state.hazards) == 49
    assert all(len(row) == 8 for row in state.exposure_counts)
    assert all(len(row) == 8 for row in state.event_counts)
    assert all(len(row) == 8 for row in state.hazards)
    manual_exposure = sum(row.number == 1 and row.age_bin_index == 0 for row in rows)
    manual_events = sum(
        row.number == 1 and row.age_bin_index == 0 and row.label == 1
        for row in rows
    )
    assert state.exposure_counts[0][0] == manual_exposure
    assert state.event_counts[0][0] == manual_events
    assert state.hazards[0][0] == (
        manual_events + 10.0 * (5 / 49)
    ) / (manual_exposure + 10.0)
    assert state.rows_per_number == tuple(sum(row) for row in state.exposure_counts)
    assert state.total_risk_rows == sum(state.rows_per_number)
    assert state.events_per_number == tuple(sum(row) for row in state.event_counts)
    assert all(0.0 < value < 1.0 for row in state.hazards for value in row)

    expected_ages = tuple(_manual_age(history, number) for number in range(1, 50))
    assert details.current_age_or_lower_bounds == expected_ages
    assert details.age_is_exact == (True,) * 49
    assert details.age_bin_indices == tuple(
        max(index for index, bound in enumerate(AGE_BIN_LOWER_BOUNDS) if bound <= age)
        for age in expected_ages
    )
    expected_raw = tuple(
        state.hazards[number][details.age_bin_indices[number]]
        for number in range(49)
    )
    assert details.raw_hazards == expected_raw
    assert details.probabilities == project_bounded_simplex(expected_raw)
    assert details.probabilities == _predict(model, history, state)
    assert details.projection_applied is (details.probabilities != details.raw_hazards)
    assert len(details.probabilities) == 49
    assert all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in details.probabilities)
    assert math.fsum(details.probabilities) == pytest.approx(5.0, abs=1e-12)


def test_empty_cells_use_beta_prior_only():
    state = _fit(DiscreteTimeHazardModel(DiscreteTimeHazardConfig()), _draws(60))
    empty = next(
        (number, age_bin)
        for number in range(49)
        for age_bin in range(8)
        if state.exposure_counts[number][age_bin] == 0
    )
    assert state.event_counts[empty[0]][empty[1]] == 0
    assert state.hazards[empty[0]][empty[1]] == 5 / 49


def test_config_is_frozen_complete_single_trial_preregistration():
    config = DiscreteTimeHazardConfig()

    assert config.family == "per_number_discrete_time_recurrent_event_hazard"
    assert config.age_definition == "age 0 iff present at t-1; otherwise elapsed draws since last appearance minus one"
    assert config.age_bin_lower_bounds == AGE_BIN_LOWER_BOUNDS
    assert config.prior_mean == 5 / 49
    assert config.prior_strength == 10.0
    assert config.minimum_history == 60
    assert "left-censored" in config.left_censor_policy
    assert "right-censored" in config.right_censor_policy
    assert "reject" in config.never_observed_policy
    assert "Beta prior alone" in config.empty_bin_policy
    assert config.trial_budget == 1
    assert config.allowed_parameters == (
        "age_bin_lower_bounds",
        "prior_mean",
        "prior_strength",
        "minimum_history",
    )
    assert config.metrics == PREREGISTERED_METRICS
    assert config.primary_baselines == (
        "uniform",
        "cumulative_frequency_alpha=1.0",
        "dirichlet_multinomial_marginal_concentration=1.0_minimum_history=1_prior=uniform_5_over_49",
        LogisticRegressionConfig().identifier,
    )
    assert "no grid" in config.preregistration_note.lower()
    assert "untouched final holdout" in config.promotion_rule
    assert "untouched final holdout" in config.rejection_rule
    assert "no silent fallback" in config.insufficient_data_behavior
    assert not hasattr(config, "prior_strength_grid")
    with pytest.raises(FrozenInstanceError):
        config.prior_strength = 20.0
    with pytest.raises(TypeError):
        DiscreteTimeHazardConfig(trial_budget=2)


def test_identifier_encodes_every_predictive_parameter():
    base = DiscreteTimeHazardConfig()
    variants = (
        replace(base, age_bin_lower_bounds=(0, 1, 3)),
        replace(base, prior_mean=0.2),
        replace(base, prior_strength=11.0),
        replace(base, minimum_history=61),
    )
    assert len({base.identifier, *(config.identifier for config in variants)}) == 5
    for token in (
        "bins=0,1,2,3,4,8,16,32",
        "prior_mean=0.10204081632653061",
        "prior_strength=10.0",
        "minimum_history=60",
        "age=elapsed_since_last_minus_one",
        "left=exclude_before_first",
        "right=count_observed_exposures_only",
        "never_observed=reject",
        "empty_bin=beta_prior_only",
    ):
        assert token in base.identifier


@pytest.mark.parametrize("invalid", [0, -1, math.nan, math.inf, -math.inf, True, "10", None])
def test_config_rejects_invalid_prior_strength(invalid):
    with pytest.raises((TypeError, ValueError), match="prior_strength"):
        DiscreteTimeHazardConfig(prior_strength=invalid)


@pytest.mark.parametrize("invalid", [0, 1, -0.1, 1.1, math.nan, math.inf, True, "x", None])
def test_config_rejects_invalid_prior_mean(invalid):
    with pytest.raises((TypeError, ValueError), match="prior_mean"):
        DiscreteTimeHazardConfig(prior_mean=invalid)


@pytest.mark.parametrize(
    ("prior_mean", "prior_strength"),
    [(5e-324, 0.5), (0.5, 5e-324)],
)
def test_config_rejects_beta_prior_pseudocount_underflow(
    prior_mean, prior_strength
):
    with pytest.raises(ValueError, match="Beta prior pseudocounts"):
        DiscreteTimeHazardConfig(
            prior_mean=prior_mean, prior_strength=prior_strength
        )


@pytest.mark.parametrize(
    "invalid",
    [(), (-1, 0), (1, 2), (0, 1, 1), (0, 2, 1), (0, True), (0, 1.5), [0, 1]],
)
def test_config_rejects_invalid_bins(invalid):
    with pytest.raises((TypeError, ValueError), match="age_bin_lower_bounds"):
        DiscreteTimeHazardConfig(age_bin_lower_bounds=invalid)


@pytest.mark.parametrize("invalid", [0, -1, 1.5, True, "60", None])
def test_config_rejects_invalid_minimum_history(invalid):
    with pytest.raises((TypeError, ValueError), match="minimum_history"):
        DiscreteTimeHazardConfig(minimum_history=invalid)


def test_fit_rejects_bad_history_rng_and_never_observed_number():
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    with pytest.raises(ValueError, match="at least 60"):
        _fit(model, ())
    with pytest.raises(ValueError, match="at least 60"):
        _fit(model, _draws(59))
    with pytest.raises(TypeError, match="immutable tuple"):
        model.fit(list(_draws(60)), None, rng=np.random.default_rng(1))
    with pytest.raises(TypeError, match="rng"):
        model.fit(_draws(60), None, rng=object())

    history = tuple(_without_number(draw, 49) for draw in _draws(60))
    with pytest.raises(ValueError, match="number 49.*never observed"):
        _fit(model, history)


def test_direct_apis_reject_unordered_dates_and_duplicate_indices():
    history = _draws(60)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    invalid_histories = (
        ((*history[:58], history[59], history[58]), "strictly increasing"),
        ((*history[:-1], replace(history[-1], original_index=history[-2].original_index)), "unique"),
    )
    for invalid, message in invalid_histories:
        with pytest.raises(ValueError, match=message):
            build_hazard_rows(invalid, model.config)
        with pytest.raises(ValueError, match=message):
            _fit(model, invalid)


def test_state_rejects_bad_dimensions_types_counts_hazards_and_configuration():
    history = _draws(60)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, history)
    bad_exposure = ((True,) + state.exposure_counts[0][1:],) + state.exposure_counts[1:]
    negative_events = ((-1,) + state.event_counts[0][1:],) + state.event_counts[1:]
    too_many_events = (
        (state.exposure_counts[0][0] + 1,) + state.event_counts[0][1:],
    ) + state.event_counts[1:]
    wrong_hazard = ((state.hazards[0][0] + 0.01,) + state.hazards[0][1:],) + state.hazards[1:]
    invalid_replacements = (
        {"exposure_counts": state.exposure_counts[:-1]},
        {"event_counts": tuple(row[:-1] for row in state.event_counts)},
        {"hazards": state.hazards[:-1]},
        {"exposure_counts": bad_exposure},
        {"event_counts": negative_events},
        {"event_counts": too_many_events},
        {"hazards": ((math.nan,) + state.hazards[0][1:],) + state.hazards[1:]},
        {"hazards": wrong_hazard},
        {"rows_per_number": (0,) + state.rows_per_number[1:]},
        {"events_per_number": (0,) + state.events_per_number[1:]},
        {"total_risk_rows": state.total_risk_rows + 1},
        {"first_observation_positions": (True,) + state.first_observation_positions[1:]},
    )
    for values in invalid_replacements:
        with pytest.raises((TypeError, ValueError)):
            replace(state, **values)

    other = DiscreteTimeHazardModel(replace(model.config, prior_strength=11.0))
    with pytest.raises(ValueError, match="configuration"):
        _predict(other, history, state)
    with pytest.raises(TypeError, match="state"):
        model.predict(history, object(), None, rng=np.random.default_rng(1))
    with pytest.raises(FrozenInstanceError):
        state.hazards = state.hazards
    assert not hasattr(state, "__dict__")


def test_predict_revalidates_forged_state_and_projection(monkeypatch):
    import loto.evaluation.models.discrete_time_hazard as module

    history = _draws(60)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, history)
    forged = deepcopy(state)
    object.__setattr__(forged, "hazards", ((math.inf,) + state.hazards[0][1:],) + state.hazards[1:])
    with pytest.raises(ValueError, match="hazards"):
        _predict(model, history, forged)

    monkeypatch.setattr(module, "project_bounded_simplex", lambda values: (math.nan,) * 49)
    with pytest.raises(ValueError, match="probabilities"):
        _predict(model, history, state)


def test_predict_rejects_counts_moved_between_bins_despite_coherent_derived_fields():
    history = _draws(60)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, history)
    number_index, source_bin = next(
        (number_index, age_bin)
        for number_index, (exposure_row, event_row) in enumerate(
            zip(state.exposure_counts, state.event_counts, strict=True)
        )
        for age_bin, (exposure, event) in enumerate(
            zip(exposure_row, event_row, strict=True)
        )
        if exposure > event
    )
    destination_bin = (source_bin + 1) % len(AGE_BIN_LOWER_BOUNDS)
    exposure_rows = [list(row) for row in state.exposure_counts]
    exposure_rows[number_index][source_bin] -= 1
    exposure_rows[number_index][destination_bin] += 1
    forged_exposures = tuple(tuple(row) for row in exposure_rows)
    forged_hazards = tuple(
        tuple(
            (event + state.prior_strength * state.prior_mean)
            / (exposure + state.prior_strength)
            for exposure, event in zip(exposure_row, event_row, strict=True)
        )
        for exposure_row, event_row in zip(
            forged_exposures, state.event_counts, strict=True
        )
    )
    forged = deepcopy(state)
    object.__setattr__(forged, "exposure_counts", forged_exposures)
    object.__setattr__(forged, "hazards", forged_hazards)

    assert replace(state) == state
    with pytest.raises(ValueError, match="provenance"):
        _predict(model, history, forged)


def test_prediction_provenance_future_overlap_eviction_and_reused_identity():
    draws = _draws(122)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, draws[:60])

    future_state = _fit(model, draws[:61])
    with pytest.raises(ValueError, match="later than visible history"):
        _predict(model, draws[:60], future_state)

    changed_overlap = (*draws[:10], replace(draws[10], numbers=(40, 41, 42, 43, 44)), *draws[11:60])
    with pytest.raises(ValueError, match="training provenance.*visible history"):
        _predict(model, changed_overlap, state)

    evicted_visible = draws[61:121]
    assert state.fitted_through_date < evicted_visible[0].draw_date
    assert _predict(model, evicted_visible, state)
    reused = (
        replace(evicted_visible[0], original_index=draws[10].original_index, numbers=(40, 41, 42, 43, 44)),
        *evicted_visible[1:],
    )
    with pytest.raises(ValueError, match="training provenance.*visible history"):
        _predict(model, reused, state)


def test_fitted_and_provenance_fields_are_coherent_and_revalidated():
    history = _draws(60)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, history)
    for values in (
        {"fitted_through_original_index": 999_999},
        {"fitted_through_date": date(2099, 1, 1)},
        {"training_provenance": state.training_provenance[:-1]},
        {"training_provenance": (*state.training_provenance[:-1], state.training_provenance[-2])},
    ):
        with pytest.raises((TypeError, ValueError), match="provenance"):
            replace(state, **values)

    forged = deepcopy(state)
    object.__setattr__(forged, "training_provenance", state.training_provenance[:-1])
    with pytest.raises((TypeError, ValueError), match="provenance"):
        _predict(model, history, forged)


def test_fit_is_linear_state_machine_and_does_not_materialize_audit_rows(monkeypatch):
    import loto.evaluation.models.discrete_time_hazard as module

    calls = 0
    original = module._count_hazard_observations

    def counted(history, bounds):
        nonlocal calls
        calls += 1
        return original(history, bounds)

    def forbidden(*args, **kwargs):
        raise AssertionError("fit materialized HazardRow values")

    monkeypatch.setattr(module, "_count_hazard_observations", counted)
    monkeypatch.setattr(module, "build_hazard_rows", forbidden)
    state = _fit(DiscreteTimeHazardModel(DiscreteTimeHazardConfig()), _draws(65))

    assert calls == 1
    assert state.total_risk_rows == sum(state.rows_per_number)


@pytest.mark.parametrize(
    ("policy", "frequency", "expected_retrained", "expected_fitted"),
    [
        (RetrainPolicy.EACH_DRAW, None, [True, True, True, True], [1059, 1060, 1061, 1062]),
        (RetrainPolicy.FIXED_FREQUENCY, 2, [True, False, True, False], [1059, 1059, 1061, 1061]),
        (RetrainPolicy.NONE_DURING_WINDOW, None, [True, False, False, False], [1059] * 4),
    ],
)
def test_real_walk_forward_all_policies_audit_and_no_target(
    policy, frequency, expected_retrained, expected_fitted
):
    draws = _draws(64)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    records = _run(draws, model, policy=policy, frequency=frequency)

    assert [record.retrained for record in records] == expected_retrained
    assert [record.fitted_through_original_index for record in records] == expected_fitted
    assert [record.target_original_index for record in records] == [1060, 1061, 1062, 1063]
    for record in records:
        assert record.target_original_index not in record.history_original_indices
        assert max(record.history_original_indices) < record.target_original_index
        assert record.fitted_through_date < record.target_date
    assert "target" not in inspect.signature(model.fit).parameters
    assert "target" not in inspect.signature(model.predict).parameters


def test_bounded_walk_forward_accepts_fully_evicted_frozen_state():
    records = _run(
        _draws(122),
        DiscreteTimeHazardModel(DiscreteTimeHazardConfig()),
        policy=RetrainPolicy.NONE_DURING_WINDOW,
        max_train_size=60,
    )
    assert len(records) == 62
    assert records[-1].history_original_indices == tuple(range(1061, 1121))
    assert records[-1].fitted_through_original_index == 1059
    assert [record.retrained for record in records] == [True] + [False] * 61


def test_bounded_walk_forward_uses_last_bin_for_number_absent_after_eviction():
    draws = tuple(
        _without_number(draw, 49) if position >= 60 else draw
        for position, draw in enumerate(_draws(120))
    )
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    details = []

    def predict(history, state, feature_data, *, rng):
        prediction = model.predict_details(history, state, rng=rng)
        details.append(prediction)
        return prediction.probabilities

    records = run_walk_forward(
        draws,
        (EvaluationWindow(0, 60, 60, 120, max_train_size=60),),
        WalkForwardCallbacks(fit=model.fit, predict=predict),
        WalkForwardConfig(
            initial_train_size=60,
            retrain_policy=RetrainPolicy.NONE_DURING_WINDOW,
            seed=3,
        ),
    )

    assert len(records) == 60
    assert details[-1].current_age_or_lower_bounds[48] == 60
    assert details[-1].age_is_exact[48] is False
    assert details[-1].age_bin_indices[48] == 7


def test_absent_number_records_visible_length_as_inexact_lower_bound_and_last_bin():
    draws = _draws(120)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, draws[:60])
    visible = tuple(_without_number(draw, 49) for draw in draws[60:120])

    details = model.predict_details(visible, state, rng=np.random.default_rng(12))

    assert details.current_age_or_lower_bounds[48] == len(visible)
    assert details.age_is_exact[48] is False
    assert details.age_bin_indices[48] == len(AGE_BIN_LOWER_BOUNDS) - 1


def test_absent_number_is_rejected_when_lower_bound_does_not_identify_age_bin():
    draws = _draws(120)
    config = DiscreteTimeHazardConfig(
        age_bin_lower_bounds=(*AGE_BIN_LOWER_BOUNDS, 64), minimum_history=60
    )
    model = DiscreteTimeHazardModel(config)
    state = _fit(model, draws[:60])
    visible = tuple(_without_number(draw, 49) for draw in draws[60:120])

    with pytest.raises(ValueError, match="number 49.*age bin.*unknown"):
        model.predict_details(visible, state, rng=np.random.default_rng(12))


def test_predict_uses_frozen_hazards_but_current_ages_without_refit(monkeypatch):
    history = _draws(60)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    state = _fit(model, history)
    before = deepcopy(state)
    changed_visible = (*history, replace(_draws(61)[-1], numbers=(40, 41, 42, 43, 44)))

    def forbidden(*args, **kwargs):
        raise AssertionError("predict attempted to recount hazards")

    monkeypatch.setattr(
        "loto.evaluation.models.discrete_time_hazard._count_hazard_observations",
        forbidden,
    )
    first = model.predict_details(history, state, rng=np.random.default_rng(1))
    second = model.predict_details(changed_visible, state, rng=np.random.default_rng(2))
    assert first.current_age_or_lower_bounds != second.current_age_or_lower_bounds
    assert first.raw_hazards != second.raw_hazards
    assert state == before


def test_future_changes_do_not_modify_earlier_walk_forward_predictions():
    draws = _draws(64)
    changed = (*draws[:62], replace(draws[62], numbers=(40, 41, 42, 43, 44)), draws[63])
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    original = _run(draws, model)
    modified = _run(changed, model)

    assert [record.probabilities for record in original[:3]] == [
        record.probabilities for record in modified[:3]
    ]


def test_determinism_rng_unconsumed_no_global_random_and_no_mutation():
    history = _draws(60)
    before = deepcopy(history)
    global_before = deepcopy(np.random.get_state())
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())

    fit_rng = np.random.default_rng(91)
    fit_rng_before = deepcopy(fit_rng.bit_generator.state)
    first = model.fit(history, None, rng=fit_rng)
    assert fit_rng.bit_generator.state == fit_rng_before
    second = model.fit(history, None, rng=np.random.default_rng(92))
    assert first == second

    predict_rng = np.random.default_rng(93)
    predict_rng_before = deepcopy(predict_rng.bit_generator.state)
    prediction = model.predict(history, first, None, rng=predict_rng)
    assert predict_rng.bit_generator.state == predict_rng_before
    assert prediction == model.predict(history, second, None, rng=np.random.default_rng(94))
    assert history == before
    assert all(
        np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
        for left, right in zip(np.random.get_state(), global_before, strict=True)
    )
    assert [record.probabilities for record in _run(_draws(62), model, seed=1)] == [
        record.probabilities for record in _run(_draws(62), model, seed=999)
    ]


def test_model_performs_no_file_database_or_production_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("hazard model attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    model = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    history = _draws(60)
    assert _predict(model, history, _fit(model, history))


def test_same_engine_comparison_all_families_has_finite_unchanged_metrics():
    draws = _draws(62)
    cumulative = cumulative_frequency_baseline(alpha=1.0)
    dirichlet = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=1.0))
    logistic = LogisticRegressionModel(LogisticRegressionConfig())
    hazard = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    candidates = (uniform_baseline(), cumulative, dirichlet, logistic, hazard)
    records = tuple(_run(draws, candidate) for candidate in candidates)
    metrics = tuple(evaluate_predictions(values) for values in records)

    assert [record.probabilities for record in records[1]] == pytest.approx(
        [record.probabilities for record in records[2]], abs=1e-15
    )
    assert len(
        {
            candidates[0].identifier,
            candidates[1].identifier,
            candidates[2].config.identifier,
            candidates[3].config.identifier,
            candidates[4].config.identifier,
        }
    ) == 5
    assert all(
        math.isfinite(getattr(result, metric))
        for result in metrics
        for metric in PREREGISTERED_METRICS
    )


def test_scientific_documentation_states_scope_estimator_and_limits():
    import loto.evaluation.models.discrete_time_hazard as module

    documentation = " ".join((inspect.getdoc(module) or "").lower().split())
    for phrase in (
        "49 marginal",
        "recurrent",
        "age 0",
        "left-censored",
        "right-censored",
        "beta",
        "not an exact joint law",
        "without replacement",
        "projection",
        "pre-registered",
        "no real-world performance",
    ):
        assert phrase in documentation
