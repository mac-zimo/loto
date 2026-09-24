from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
import inspect
import math
from pathlib import Path
import sqlite3
import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from loto.evaluation.baselines import cumulative_frequency_baseline, uniform_baseline
from loto.evaluation.models.dirichlet_multinomial import (
    DirichletMultinomialConfig,
    DirichletMultinomialModel,
)
from loto.evaluation.models.logistic_regression import (
    FEATURE_NAMES,
    PREREGISTERED_METRICS,
    LogisticRegressionConfig,
    LogisticRegressionModel,
    LogisticRegressionState,
    build_supervised_rows,
)
from loto.evaluation.walk_forward import (
    DrawObservation,
    EvaluationWindow,
    RetrainPolicy,
    WalkForwardConfig,
    evaluate_predictions,
    run_walk_forward,
)


def _draws(count: int) -> tuple[DrawObservation, ...]:
    start = date(2020, 1, 1)
    return tuple(
        DrawObservation(
            draw_date=start + timedelta(days=index),
            original_index=1000 + index,
            numbers=tuple(((index * 5 + offset) % 49) + 1 for offset in range(5)),
        )
        for index in range(count)
    )


def _fit(model: LogisticRegressionModel, history):
    return model.fit(history, None, rng=np.random.default_rng(10))


def _predict(model: LogisticRegressionModel, history, state):
    return model.predict(history, state, None, rng=np.random.default_rng(11))


def _run(draws, model, *, policy=RetrainPolicy.EACH_DRAW, frequency=None, seed=3):
    return run_walk_forward(
        draws,
        (EvaluationWindow(0, 60, 60, len(draws)),),
        model.callbacks,
        WalkForwardConfig(
            initial_train_size=60,
            retrain_policy=policy,
            retrain_frequency=frequency,
            seed=seed,
        ),
    )


def _manual_features(history, target_position: int, number: int):
    prefix = history[:target_position]
    return (
        *(float(number in history[target_position - lag].numbers) for lag in (1, 2, 3)),
        *(sum(number in draw.numbers for draw in prefix[-window:]) / window for window in (5, 10, 25, 50)),
        sum(number in draw.numbers for draw in prefix) / target_position,
    )


def test_supervised_rows_have_manual_features_labels_order_and_warmup_without_leakage():
    history = _draws(53)
    config = LogisticRegressionConfig()

    rows = build_supervised_rows(history, config)

    assert config.warmup == 50
    assert FEATURE_NAMES == (
        "lag_inclusion_1",
        "lag_inclusion_2",
        "lag_inclusion_3",
        "rolling_frequency_5",
        "rolling_frequency_10",
        "rolling_frequency_25",
        "rolling_frequency_50",
        "cumulative_frequency",
    )
    assert len(rows) == 3 * 49
    assert [(row.target_position, row.number) for row in rows[:50]] == [
        *((50, number) for number in range(1, 50)),
        (51, 1),
    ]
    for target_position, number in ((50, 1), (50, 49), (51, 8), (52, 30)):
        row = next(
            row
            for row in rows
            if row.target_position == target_position and row.number == number
        )
        assert row.features == pytest.approx(
            _manual_features(history, target_position, number), abs=1e-15
        )
        assert row.label == int(number in history[target_position].numbers)
        assert row.target_original_index == history[target_position].original_index
        assert row.target_date == history[target_position].draw_date

    changed_future = (*history[:52], replace(history[52], numbers=(40, 41, 42, 43, 44)))
    changed_rows = build_supervised_rows(changed_future, config)
    assert rows[: 2 * 49] == changed_rows[: 2 * 49]
    assert all(row.target_position >= config.warmup for row in rows)


def test_fit_state_dimensions_scaler_and_manual_prediction_are_auditable():
    history = _draws(65)
    model = LogisticRegressionModel(LogisticRegressionConfig())

    state = _fit(model, history)
    details = model.predict_details(history, state, rng=np.random.default_rng(12))

    assert state.feature_names == FEATURE_NAMES
    assert state.lags == (1, 2, 3)
    assert state.windows == (5, 10, 25, 50)
    assert state.training_row_count == (len(history) - 50) * 49
    assert state.rows_per_model == (len(history) - 50,) * 49
    assert state.classes_observed == ((0, 1),) * 49
    assert len(state.scaler_means) == len(state.scaler_scales) == 49
    assert len(state.coefficients) == 49
    assert all(len(values) == len(FEATURE_NAMES) for values in state.scaler_means)
    assert all(len(values) == len(FEATURE_NAMES) for values in state.scaler_scales)
    assert all(len(values) == len(FEATURE_NAMES) for values in state.coefficients)
    assert len(state.intercepts) == len(state.n_iter) == 49
    assert all(math.isfinite(value) for matrix in state.scaler_means for value in matrix)
    assert all(math.isfinite(value) and value > 0 for matrix in state.scaler_scales for value in matrix)
    assert all(math.isfinite(value) for matrix in state.coefficients for value in matrix)
    assert all(math.isfinite(value) for value in state.intercepts)
    assert state.fitted_through_original_index == history[-1].original_index
    assert state.fitted_through_date == history[-1].draw_date
    assert state.converged == (True,) * 49

    features = _manual_features(history, len(history), 1)
    scaled = tuple(
        (value - mean) / scale
        for value, mean, scale in zip(
            features, state.scaler_means[0], state.scaler_scales[0], strict=True
        )
    )
    score = state.intercepts[0] + math.fsum(
        coefficient * value
        for coefficient, value in zip(state.coefficients[0], scaled, strict=True)
    )
    expected_raw = 1.0 / (1.0 + math.exp(-score))
    assert details.raw_probabilities[0] == pytest.approx(expected_raw, abs=1e-15)
    assert details.probabilities == _predict(model, history, state)
    assert details.projection_applied is (
        details.probabilities != details.raw_probabilities
    )
    assert len(details.raw_probabilities) == len(details.probabilities) == 49
    assert all(0.0 <= value <= 1.0 and math.isfinite(value) for value in details.probabilities)
    assert math.fsum(details.probabilities) == pytest.approx(5.0, abs=1e-12)


def test_config_is_immutable_fixed_l2_preregistration_without_search_grid():
    config = LogisticRegressionConfig()

    assert config.family == "per_number_binary_logistic_regression_l2"
    assert config.lags == (1, 2, 3)
    assert config.windows == (5, 10, 25, 50)
    assert config.feature_names == FEATURE_NAMES
    assert config.C == 1.0
    assert config.penalty == "l2"
    assert config.class_weight is None
    assert config.solver == "liblinear"
    assert config.max_iter == 1000
    assert config.tol == 1e-4
    assert config.minimum_history == 60
    assert config.standardize is True
    assert config.random_state == 0
    assert config.trial_budget == 1
    assert config.allowed_parameters == (
        "lags",
        "windows",
        "C",
        "solver",
        "penalty",
        "max_iter",
        "tol",
        "minimum_history",
        "standardize",
        "random_state",
    )
    assert config.metrics == PREREGISTERED_METRICS
    assert config.primary_baselines == (
        "uniform",
        "cumulative_frequency_alpha=1.0",
        "dirichlet_multinomial_marginal_concentration=1.0_minimum_history=1_prior=uniform_5_over_49",
    )
    assert "reject" in config.degenerate_class_policy
    assert "reject" in config.non_convergence_policy
    assert "no grid" in config.preregistration_note.lower()
    assert "untouched final holdout" in config.promotion_rule
    assert "untouched final holdout" in config.rejection_rule
    assert "len(history) < minimum_history" in config.insufficient_data_behavior
    assert not hasattr(config, "C_grid")
    assert "grid" not in {name.lower() for name in config.allowed_parameters}
    with pytest.raises(FrozenInstanceError):
        config.C = 2.0
    with pytest.raises(TypeError):
        LogisticRegressionConfig(trial_budget=2)


def test_identifier_encodes_every_prediction_changing_parameter():
    base = LogisticRegressionConfig()
    variants = (
        replace(base, lags=(1, 2)),
        replace(base, windows=(5, 10, 25, 49)),
        replace(base, C=2.0),
        replace(base, solver="liblinear"),
        replace(base, max_iter=999),
        replace(base, tol=2e-4),
        replace(base, minimum_history=61),
        replace(base, standardize=False),
        replace(base, random_state=1),
    )
    assert len({base.identifier, *(variant.identifier for variant in variants)}) == 9
    for token in (
        "lags=1,2,3",
        "windows=5,10,25,50",
        "C=1.0",
        "penalty=l2",
        "solver=liblinear",
        "max_iter=1000",
        "tol=0.0001",
        "minimum_history=60",
        "standardize=true",
        "random_state=0",
    ):
        assert token in base.identifier


@pytest.mark.parametrize("invalid", [0, -1, math.nan, math.inf, -math.inf, True, "1", None])
def test_config_rejects_invalid_c(invalid):
    with pytest.raises((TypeError, ValueError), match="C"):
        LogisticRegressionConfig(C=invalid)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("lags", ()),
        ("lags", (1, 1, 3)),
        ("lags", (2, 1)),
        ("lags", (0, 1)),
        ("lags", (1, True)),
        ("lags", (1, 51)),
        ("windows", ()),
        ("windows", (5, 5)),
        ("windows", (10, 5)),
        ("windows", (0, 5)),
        ("windows", (5, 2.0)),
        ("solver", "lbfgs"),
        ("penalty", "l1"),
        ("tol", 0),
        ("tol", math.nan),
        ("tol", True),
        ("max_iter", 0),
        ("max_iter", 1.5),
        ("minimum_history", 51),
        ("minimum_history", True),
        ("standardize", 1),
        ("random_state", -1),
        ("random_state", True),
    ],
)
def test_config_rejects_invalid_parameters(field, invalid):
    with pytest.raises((TypeError, ValueError), match=field):
        LogisticRegressionConfig(**{field: invalid})


def test_fit_rejects_short_bad_history_degenerate_classes_and_bad_rng():
    model = LogisticRegressionModel(LogisticRegressionConfig())
    with pytest.raises(ValueError, match="at least 60"):
        _fit(model, ())
    with pytest.raises(ValueError, match="at least 60"):
        _fit(model, _draws(59))
    with pytest.raises(TypeError, match="immutable tuple"):
        model.fit(list(_draws(60)), None, rng=np.random.default_rng(1))
    with pytest.raises(TypeError, match="rng"):
        model.fit(_draws(60), None, rng=object())

    degenerate = tuple(
        DrawObservation(
            draw_date=date(2020, 1, 1) + timedelta(days=index),
            original_index=index,
            numbers=(1, 2, 3, 4, 5),
        )
        for index in range(60)
    )
    with pytest.raises(ValueError, match="degenerate class.*number 1"):
        _fit(model, degenerate)


def test_direct_training_apis_reject_unordered_dates_and_duplicate_indices():
    history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    unordered = (*history[:58], history[59], history[58])
    duplicate_index = (
        *history[:-1],
        replace(history[-1], original_index=history[-2].original_index),
    )

    for invalid_history, message in (
        (unordered, "draw dates must be strictly increasing"),
        (duplicate_index, "original_index values must be unique"),
    ):
        with pytest.raises(ValueError, match=message):
            build_supervised_rows(invalid_history, model.config)
        with pytest.raises(ValueError, match=message):
            _fit(model, invalid_history)


def test_fit_turns_any_convergence_warning_into_rejection(monkeypatch):
    import loto.evaluation.models.logistic_regression as module

    original_fit = module.LogisticRegression.fit

    def warning_fit(estimator, X, y):
        result = original_fit(estimator, X, y)
        warnings.warn("forced", ConvergenceWarning)
        return result

    monkeypatch.setattr(module.LogisticRegression, "fit", warning_fit)
    with pytest.raises(RuntimeError, match="did not converge.*number"):
        _fit(LogisticRegressionModel(LogisticRegressionConfig()), _draws(60))


def test_state_rejects_bad_dimensions_nonfinite_values_and_incompatibility():
    history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, history)

    invalid_replacements = (
        {"feature_names": state.feature_names[:-1]},
        {"scaler_means": state.scaler_means[:-1]},
        {"scaler_scales": ((0.0,) + state.scaler_scales[0][1:],) + state.scaler_scales[1:]},
        {"scaler_means": ((math.nan,) + state.scaler_means[0][1:],) + state.scaler_means[1:]},
        {"coefficients": state.coefficients[:-1]},
        {"coefficients": ((math.inf,) + state.coefficients[0][1:],) + state.coefficients[1:]},
        {"intercepts": (math.nan,) + state.intercepts[1:]},
        {"classes_observed": ((0,),) + state.classes_observed[1:]},
        {"classes_observed": ((False, True),) + state.classes_observed[1:]},
        {"rows_per_model": (0,) + state.rows_per_model[1:]},
        {"n_iter": (0,) + state.n_iter[1:]},
        {"converged": (False,) + state.converged[1:]},
        {"converged": (1,) + state.converged[1:]},
    )
    for values in invalid_replacements:
        with pytest.raises((TypeError, ValueError)):
            replace(state, **values)

    other = LogisticRegressionModel(replace(model.config, C=2.0))
    with pytest.raises(ValueError, match="configuration"):
        _predict(other, history, state)
    with pytest.raises(TypeError, match="state"):
        model.predict(history, object(), None, rng=np.random.default_rng(1))
    with pytest.raises(TypeError, match="rng"):
        model.predict(history, state, None, rng=object())
    with pytest.raises(FrozenInstanceError):
        setattr(state, "intercepts", (0.0,) * 49)
    assert not hasattr(state, "__dict__")


def test_predict_revalidates_forged_state_and_rejects_invalid_projection(monkeypatch):
    import loto.evaluation.models.logistic_regression as module

    history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, history)
    forged = deepcopy(state)
    object.__setattr__(forged, "intercepts", (math.nan,) + state.intercepts[1:])
    with pytest.raises(ValueError, match="intercepts"):
        _predict(model, history, forged)

    monkeypatch.setattr(module, "project_bounded_simplex", lambda values: (math.nan,) * 49)
    with pytest.raises(ValueError, match="probabilities"):
        _predict(model, history, state)


def test_prediction_apis_reject_future_and_absent_fitted_observations():
    visible_history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    future_state = _fit(model, _draws(61))
    current_state = _fit(model, visible_history)
    with pytest.raises(ValueError, match="provenance.*fitted_through"):
        replace(
            current_state,
            fitted_through_original_index=visible_history[-1].original_index + 10_000,
        )
    with pytest.raises(ValueError, match="provenance.*fitted_through"):
        replace(
            current_state,
            fitted_through_date=visible_history[-1].draw_date + timedelta(days=10_000),
        )

    with pytest.raises(ValueError, match="later than visible history"):
        model.predict_details(
            visible_history, future_state, rng=np.random.default_rng(12)
        )
    with pytest.raises(ValueError, match="later than visible history"):
        _predict(model, visible_history, future_state)


def test_prediction_apis_accept_older_fitted_observation_in_visible_prefix():
    training_history = _draws(60)
    visible_history = _draws(62)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, training_history)

    details = model.predict_details(
        visible_history, state, rng=np.random.default_rng(12)
    )

    assert details.probabilities == _predict(model, visible_history, state)


def test_frozen_state_remains_valid_after_fitted_draw_is_evicted_from_bounded_history():
    draws = _draws(122)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    records = run_walk_forward(
        draws,
        (EvaluationWindow(0, 60, 60, 122, max_train_size=60),),
        model.callbacks,
        WalkForwardConfig(
            initial_train_size=60,
            retrain_policy=RetrainPolicy.NONE_DURING_WINDOW,
            seed=3,
        ),
    )

    assert len(records) == 62
    assert records[-1].history_original_indices == tuple(range(1061, 1121))
    assert records[-1].fitted_through_original_index == 1059
    assert [record.retrained for record in records] == [True] + [False] * 61


def test_prediction_apis_reject_reused_evicted_training_index_with_new_fingerprint():
    draws = _draws(121)
    training_history = draws[:60]
    visible_history = (
        replace(
            draws[61],
            original_index=training_history[10].original_index,
            numbers=(40, 41, 42, 43, 44),
        ),
        *draws[62:],
    )
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, training_history)

    assert state.fitted_through_date < visible_history[0].draw_date
    with pytest.raises(ValueError, match="training provenance.*visible history"):
        model.predict_details(visible_history, state, rng=np.random.default_rng(12))
    with pytest.raises(ValueError, match="training provenance.*visible history"):
        _predict(model, visible_history, state)


def test_prediction_rejects_changed_numbers_in_visible_training_overlap():
    training_history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, training_history)
    changed_history = (
        *training_history[:10],
        replace(training_history[10], numbers=(40, 41, 42, 43, 44)),
        *training_history[11:],
    )

    with pytest.raises(ValueError, match="training provenance.*visible history"):
        _predict(model, changed_history, state)


def test_prediction_rejects_divergent_visible_training_prefix_with_same_terminal():
    training_history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, training_history)
    divergent_history = (
        replace(training_history[0], original_index=99_999),
        *training_history[1:],
    )

    assert divergent_history[-1].draw_date == state.fitted_through_date
    assert divergent_history[-1].original_index == state.fitted_through_original_index
    with pytest.raises(ValueError, match="training provenance.*visible history"):
        _predict(model, divergent_history, state)


def test_prediction_revalidates_forged_malformed_and_incoherent_provenance():
    history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, history)
    malformed = deepcopy(state)
    object.__setattr__(malformed, "training_provenance", state.training_provenance[:-1])
    incoherent = deepcopy(state)
    object.__setattr__(
        incoherent,
        "training_provenance",
        (*state.training_provenance[:-1], state.training_provenance[-2]),
    )

    for forged in (malformed, incoherent):
        with pytest.raises((TypeError, ValueError), match="provenance"):
            _predict(model, history, forged)


def test_fit_uses_single_vectorized_inclusion_matrix_without_supervised_rows(monkeypatch):
    import loto.evaluation.models._logistic_features as feature_module
    import loto.evaluation.models.logistic_regression as model_module

    calls = 0
    original = feature_module._inclusion_matrix

    def counted_inclusion_matrix(history):
        nonlocal calls
        calls += 1
        return original(history)

    def forbidden_rows(*args, **kwargs):
        raise AssertionError("fit materialized SupervisedRow values")

    monkeypatch.setattr(feature_module, "_inclusion_matrix", counted_inclusion_matrix)
    monkeypatch.setattr(model_module, "build_supervised_rows", forbidden_rows)
    monkeypatch.setattr(feature_module, "build_audit_rows", forbidden_rows)

    state = _fit(
        LogisticRegressionModel(LogisticRegressionConfig()),
        _draws(65),
    )

    assert calls == 1
    assert state.training_row_count == 15 * 49


def test_constant_features_receive_unit_scale_deterministically():
    import loto.evaluation.models.logistic_regression as module

    values = np.asarray(((1.0, 2.0), (1.0, 4.0), (1.0, 6.0)))
    transformed, means, scales = module._training_transform(
        values, standardize=True
    )

    assert means == pytest.approx((1.0, 4.0))
    assert scales[0] == 1.0
    assert transformed[:, 0] == pytest.approx((0.0, 0.0, 0.0))
    repeated = module._training_transform(values, standardize=True)
    assert all(
        np.array_equal(left, right)
        for left, right in zip((transformed, means, scales), repeated, strict=True)
    )


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
    model = LogisticRegressionModel(LogisticRegressionConfig())

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


def test_predict_uses_frozen_parameters_but_current_visible_features_without_refit(monkeypatch):
    history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, history)
    state_before = deepcopy(state)
    changed_visible = (*history, replace(_draws(61)[-1], numbers=(40, 41, 42, 43, 44)))

    def forbidden_fit(*args, **kwargs):
        raise AssertionError("predict attempted to fit")

    monkeypatch.setattr("sklearn.linear_model.LogisticRegression.fit", forbidden_fit)
    first = _predict(model, history, state)
    second = _predict(model, changed_visible, state)
    assert first != second
    assert state == state_before


def test_future_changes_do_not_modify_earlier_predictions():
    draws = _draws(64)
    changed = (*draws[:62], replace(draws[62], numbers=(40, 41, 42, 43, 44)), draws[63])
    model = LogisticRegressionModel(LogisticRegressionConfig())

    original = _run(draws, model)
    modified = _run(changed, model)

    assert [record.probabilities for record in original[:3]] == [
        record.probabilities for record in modified[:3]
    ]


def test_deterministic_seed_independent_rng_unconsumed_no_global_random_and_no_mutation():
    history = _draws(60)
    before = deepcopy(history)
    global_before = deepcopy(np.random.get_state())
    model = LogisticRegressionModel(LogisticRegressionConfig())

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

    seed_one = _run(_draws(62), model, seed=1)
    seed_two = _run(_draws(62), model, seed=999)
    assert [record.probabilities for record in seed_one] == [
        record.probabilities for record in seed_two
    ]


def test_model_performs_no_file_database_or_production_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("logistic model attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    history = _draws(60)
    model = LogisticRegressionModel(LogisticRegressionConfig())
    state = _fit(model, history)
    assert _predict(model, history, state)


def test_synthetic_same_engine_comparison_has_finite_metrics_and_baseline_equivalence():
    draws = _draws(62)
    logistic = LogisticRegressionModel(LogisticRegressionConfig())
    cumulative = cumulative_frequency_baseline(alpha=1.0)
    dirichlet = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=1.0))
    candidates = (uniform_baseline(), cumulative, dirichlet, logistic)

    records = tuple(_run(draws, candidate) for candidate in candidates)
    evaluations = tuple(evaluate_predictions(candidate_records) for candidate_records in records)

    assert [record.probabilities for record in records[1]] == pytest.approx(
        [record.probabilities for record in records[2]], abs=1e-15
    )
    assert len(
        {
            candidates[0].identifier,
            candidates[1].identifier,
            candidates[2].config.identifier,
            candidates[3].config.identifier,
        }
    ) == 4
    assert all(
        math.isfinite(getattr(metrics, name))
        for metrics in evaluations
        for name in PREREGISTERED_METRICS
    )


def test_module_scientific_documentation_states_limits_and_chronology():
    import loto.evaluation.models.logistic_regression as module

    documentation = " ".join((inspect.getdoc(module) or "").lower().split())
    for phrase in (
        "49 binary",
        "not an exact joint",
        "without replacement",
        "projection",
        "before the labelled draw",
        "pre-registered",
        "no real-world performance",
    ):
        assert phrase in documentation
