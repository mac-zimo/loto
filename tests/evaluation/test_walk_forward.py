from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import replace
from datetime import date
import inspect
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

import loto.evaluation.walk_forward as walk_forward
from loto.evaluation.walk_forward import (
    CalibrationCallback,
    FeatureBuilder,
    FitCallback,
    PROBABILITY_SUM_TOLERANCE,
    DrawObservation,
    EvaluationWindow,
    PredictionRecord,
    PredictionResult,
    PredictCallback,
    RetrainPolicy,
    WalkForwardCallbacks,
    WalkForwardConfig,
    annual_windows,
    evaluate_predictions,
    rolling_origin_windows,
    run_walk_forward,
    validate_probabilities,
)


def _draws(count: int, *, year: int = 2020) -> tuple[DrawObservation, ...]:
    return tuple(
        DrawObservation(
            draw_date=date(year + index // 12, index % 12 + 1, 1),
            original_index=100 + index,
            numbers=tuple(((index + offset) % 49) + 1 for offset in range(5)),
        )
        for index in range(count)
    )


def _probabilities(history, *, rng) -> np.ndarray:
    weights = rng.random(49) + len(history) + sum(draw.numbers[0] for draw in history)
    return weights * (5.0 / weights.sum())


def _callbacks(log=None) -> WalkForwardCallbacks:
    events = [] if log is None else log

    def features(history, *, rng):
        events.append(("features", (history,), {"rng": rng}))
        return len(history), float(rng.random())

    def fit(history, feature_data, *, rng):
        events.append(("fit", (history, feature_data), {"rng": rng}))
        return feature_data, float(rng.random())

    def calibrate(history, state, feature_data, *, rng):
        events.append(("calibrate", (history, state, feature_data), {"rng": rng}))
        return state, feature_data, float(rng.random())

    def predict(history, state, feature_data, *, rng):
        events.append(("predict", (history, state, feature_data), {"rng": rng}))
        return _probabilities(history, rng=rng)

    return WalkForwardCallbacks(
        fit=fit,
        predict=predict,
        build_features=features,
        calibrate=calibrate,
    )


def _run(
    draws,
    *,
    policy=RetrainPolicy.EACH_DRAW,
    frequency=None,
    seed=123,
    log=None,
):
    windows = rolling_origin_windows(
        draws, initial_train_size=4, evaluation_size=3
    )
    config = WalkForwardConfig(
        initial_train_size=4,
        retrain_policy=policy,
        retrain_frequency=frequency,
        seed=seed,
    )
    return run_walk_forward(draws, windows, _callbacks(log), config)


def test_future_observation_cannot_change_earlier_predictions():
    draws = _draws(10)
    changed_index = 8
    changed_future = (
        *draws[:changed_index],
        replace(draws[changed_index], numbers=(45, 46, 47, 48, 49)),
        *draws[changed_index + 1 :],
    )

    original = _run(draws)
    changed = _run(changed_future)
    unaffected_original = [
        record
        for record in original
        if draws[changed_index].original_index not in record.history_original_indices
    ]
    unaffected_changed = [
        record
        for record in changed
        if draws[changed_index].original_index not in record.history_original_indices
    ]

    assert [
        (record.probabilities, record.selected_numbers) for record in unaffected_original
    ] == [
        (record.probabilities, record.selected_numbers) for record in unaffected_changed
    ]
    assert unaffected_original[-1].target_original_index == draws[changed_index].original_index


def test_bounded_multi_draw_callbacks_receive_no_target_context_or_position():
    draws = _draws(12)
    windows = rolling_origin_windows(
        draws, initial_train_size=4, evaluation_size=4, max_train_size=5
    )
    journals = {stage: [] for stage in ("features", "fit", "calibrate", "predict")}
    captured_draws = draws
    captured_windows = windows

    def features(history, *, rng):
        assert captured_draws is draws and captured_windows is windows
        journals["features"].append(((history,), {"rng": rng}))
        return len(history)

    def fit(history, feature_data, *, rng):
        assert captured_draws is draws and captured_windows is windows
        journals["fit"].append(((history, feature_data), {"rng": rng}))
        return feature_data

    def calibrate(history, state, feature_data, *, rng):
        assert captured_draws is draws and captured_windows is windows
        journals["calibrate"].append(
            ((history, state, feature_data), {"rng": rng})
        )
        return state

    def predict(history, state, feature_data, *, rng):
        assert captured_draws is draws and captured_windows is windows
        journals["predict"].append(
            ((history, state, feature_data), {"rng": rng})
        )
        return _probabilities(history, rng=rng)

    callbacks = WalkForwardCallbacks(
        fit=fit,
        predict=predict,
        build_features=features,
        calibrate=calibrate,
    )
    records = run_walk_forward(
        draws,
        windows,
        callbacks,
        WalkForwardConfig(initial_train_size=4),
    )

    assert "PredictionContext" not in walk_forward.__all__
    assert not hasattr(walk_forward, "PredictionContext")
    expected_protocol_parameters = {
        FeatureBuilder: ("self", "history", "rng"),
        FitCallback: ("self", "history", "feature_data", "rng"),
        CalibrationCallback: ("self", "history", "state", "feature_data", "rng"),
        PredictCallback: ("self", "history", "state", "feature_data", "rng"),
    }
    for protocol, expected in expected_protocol_parameters.items():
        parameters = inspect.signature(protocol.__call__).parameters
        assert tuple(parameters) == expected
        assert parameters["rng"].kind is inspect.Parameter.KEYWORD_ONLY

    targets = tuple(
        draws[position]
        for window in windows
        for position in range(window.test_start, window.test_end)
    )
    expected_targets_by_stage = {
        "features": tuple(target for target in targets for _ in range(2)),
        "fit": targets,
        "calibrate": targets,
        "predict": targets,
    }
    expected_arg_counts = {"features": 1, "fit": 2, "calibrate": 3, "predict": 3}
    for stage, calls in journals.items():
        assert len(calls) == len(expected_targets_by_stage[stage])
        for (args, kwargs), target in zip(
            calls, expected_targets_by_stage[stage], strict=True
        ):
            history = args[0]
            assert len(args) == expected_arg_counts[stage]
            assert set(kwargs) == {"rng"}
            assert isinstance(kwargs["rng"], np.random.Generator)
            assert 1 <= len(history) <= 5
            assert target not in history
            assert all(item.draw_date < target.draw_date for item in history)
            assert all(item.original_index < target.original_index for item in history)

    assert tuple(record.target_original_index for record in records) == tuple(
        target.original_index for target in targets
    )
    for record, target in zip(records, targets, strict=True):
        assert record.target_date == target.draw_date
        assert record.target_original_index not in record.history_original_indices
        assert max(record.history_original_indices) < record.target_original_index
        assert max(record.history_dates) < record.target_date
        assert record.fitted_through_date < record.target_date
        assert record.fitted_through_original_index < record.target_original_index


def test_rolling_origin_and_annual_windows_are_explicit_and_non_overlapping():
    draws = _draws(30)

    rolling = rolling_origin_windows(
        draws,
        initial_train_size=6,
        evaluation_size=4,
        step_size=4,
        max_train_size=10,
    )
    assert rolling == (
        EvaluationWindow(0, 6, 6, 10, max_train_size=10),
        EvaluationWindow(0, 10, 10, 14, max_train_size=10),
        EvaluationWindow(4, 14, 14, 18, max_train_size=10),
        EvaluationWindow(8, 18, 18, 22, max_train_size=10),
        EvaluationWindow(12, 22, 22, 26, max_train_size=10),
        EvaluationWindow(16, 26, 26, 30, max_train_size=10),
    )
    assert all(left.test_end <= right.test_start for left, right in zip(rolling, rolling[1:]))

    yearly = annual_windows(
        draws,
        initial_train_size=12,
        evaluation_start_year=2021,
        evaluation_end_year=2021,
    )
    assert yearly == (EvaluationWindow(0, 12, 12, 24),)
    assert all(
        len({draw.draw_date.year for draw in draws[window.test_start : window.test_end]}) == 1
        for window in yearly
    )


def test_bounded_training_history_is_enforced_for_every_hook_and_target():
    draws = _draws(12)
    events = []
    windows = rolling_origin_windows(
        draws,
        initial_train_size=4,
        evaluation_size=4,
        max_train_size=5,
    )

    records = run_walk_forward(
        draws,
        windows,
        _callbacks(events),
        WalkForwardConfig(initial_train_size=4),
    )

    assert any(window.test_end - window.test_start > 1 for window in windows)
    assert events
    assert all(len(args[0]) <= 5 for _, args, _ in events)
    assert all(len(record.history_dates) <= 5 for record in records)


def test_annual_windows_require_explicit_complete_year_bounds_by_default():
    draws = _draws(30)

    with pytest.raises(ValueError, match="evaluation_start_year.*evaluation_end_year"):
        annual_windows(draws, initial_train_size=12)

    assert annual_windows(
        draws,
        initial_train_size=12,
        evaluation_start_year=2021,
        evaluation_end_year=2021,
    ) == (EvaluationWindow(0, 12, 12, 24),)
    assert annual_windows(
        draws, initial_train_size=12, allow_partial_year=True
    ) == (
        EvaluationWindow(0, 12, 12, 24),
        EvaluationWindow(0, 24, 24, 30),
    )


def test_retraining_policies_are_respected():
    draws = _draws(10)

    each_log = []
    each = _run(draws, policy=RetrainPolicy.EACH_DRAW, log=each_log)
    fixed_log = []
    fixed = _run(
        draws,
        policy=RetrainPolicy.FIXED_FREQUENCY,
        frequency=2,
        log=fixed_log,
    )
    frozen_log = []
    frozen = _run(
        draws,
        policy=RetrainPolicy.NONE_DURING_WINDOW,
        log=frozen_log,
    )

    assert sum(event[0] == "fit" for event in each_log) == len(each)
    assert sum(event[0] == "fit" for event in fixed_log) == 4
    assert sum(event[0] == "fit" for event in frozen_log) == 2
    assert [record.retrained for record in fixed] == [True, False, True] * 2
    assert [record.retrained for record in frozen] == [True, False, False] * 2


def test_same_inputs_and_seed_are_exactly_deterministic():
    draws = _draws(10)

    first = _run(draws, seed=987)
    second = _run(draws, seed=987)
    different = _run(draws, seed=988)

    assert first == second
    assert [record.probabilities for record in first] != [
        record.probabilities for record in different
    ]


def test_history_independent_stochastic_predictor_varies_by_target_reproducibly():
    draws = _draws(10)
    windows = rolling_origin_windows(
        draws, initial_train_size=4, evaluation_size=3
    )

    def fit(history, feature_data, *, rng):
        return None

    def predict(history, state, feature_data, *, rng):
        weights = rng.random(49)
        return weights * (5.0 / weights.sum())

    callbacks = WalkForwardCallbacks(fit=fit, predict=predict)
    config = WalkForwardConfig(initial_train_size=4, seed=987)

    first = run_walk_forward(draws, windows, callbacks, config)
    second = run_walk_forward(draws, windows, callbacks, config)

    assert first == second
    assert len({record.probabilities for record in first}) > 1


def test_rng_is_derived_only_from_seed_stage_and_visible_history(monkeypatch):
    draws = _draws(12)
    windows = rolling_origin_windows(
        draws,
        initial_train_size=4,
        evaluation_size=3,
        max_train_size=5,
    )
    captures = {
        stage: []
        for stage in (
            "fit-features",
            "fit",
            "calibrate",
            "predict-features",
            "predict",
        )
    }
    rng_derivations = []
    feature_call_count = 0
    original_rng = walk_forward._rng

    def rng_probe(seed, stage, history_original_indices):
        rng_derivations.append((seed, stage, history_original_indices))
        return original_rng(seed, stage, history_original_indices)

    monkeypatch.setattr(walk_forward, "_rng", rng_probe)

    def capture(stage, rng):
        initial_state = deepcopy(rng.bit_generator.state)
        initial_output = tuple(rng.random(4))
        captures[stage].append((initial_state, initial_output))

    def features(history, *, rng):
        nonlocal feature_call_count
        stage = "fit-features" if feature_call_count % 2 == 0 else "predict-features"
        feature_call_count += 1
        capture(stage, rng)
        return len(history)

    def fit(history, feature_data, *, rng):
        capture("fit", rng)
        return feature_data

    def calibrate(history, state, feature_data, *, rng):
        capture("calibrate", rng)
        return state

    def predict(history, state, feature_data, *, rng):
        capture("predict", rng)
        return [5 / 49] * 49

    records = run_walk_forward(
        draws,
        windows,
        WalkForwardCallbacks(
            fit=fit,
            predict=predict,
            build_features=features,
            calibrate=calibrate,
        ),
        WalkForwardConfig(initial_train_size=4, seed=321),
    )

    assert len(windows) > 1
    assert any(window.test_end - window.test_start > 1 for window in windows)
    assert all(len(stage_captures) == len(records) for stage_captures in captures.values())
    assert tuple(inspect.signature(original_rng).parameters) == (
        "seed",
        "stage",
        "history_original_indices",
    )
    expected_histories = [record.history_original_indices for record in records]
    for stage, stage_captures in captures.items():
        assert len({repr(state) for state, _ in stage_captures}) == len(
            expected_histories
        )
        assert len({output for _, output in stage_captures}) == len(expected_histories)
        assert [
            history_indices
            for seed, derived_stage, history_indices in rng_derivations
            if derived_stage == stage
        ] == expected_histories
    for event_index in range(len(records)):
        states = {
            repr(captures[stage][event_index][0]) for stage in captures
        }
        assert len(states) == len(captures)
    assert all(seed == 321 for seed, _, _ in rng_derivations)
    assert all(
        history_indices in expected_histories
        for _, _, history_indices in rng_derivations
    )


def test_rng_states_are_exactly_repeatable_for_the_same_visible_history():
    history_indices = (100, 104, 109)

    first = walk_forward._rng(321, "predict", history_indices)
    second = walk_forward._rng(321, "predict", history_indices)

    assert first.bit_generator.state == second.bit_generator.state
    assert tuple(first.random(8)) == tuple(second.random(8))


def test_metrics_are_correct_for_a_calculable_perfect_example():
    base_draws = _draws(5)
    draws = base_draws[:-1] + (replace(base_draws[-1], numbers=(1, 2, 3, 4, 5)),)
    probabilities = tuple(1.0 if number <= 5 else 0.0 for number in range(1, 50))

    def fit(history, feature_data, *, rng):
        return None

    def predict(history, state, feature_data, *, rng):
        return probabilities

    records = run_walk_forward(
        draws,
        (EvaluationWindow(0, 4, 4, 5),),
        WalkForwardCallbacks(fit=fit, predict=predict),
        WalkForwardConfig(initial_train_size=4, seed=1),
    )
    metrics = evaluate_predictions(records, calibration_bins=10)
    uniform = 5.0 / 49.0
    uniform_loss = -(5 * math.log(uniform) + 44 * math.log1p(-uniform)) / 49

    assert metrics.log_loss == pytest.approx(0.0)
    assert metrics.brier == pytest.approx(0.0)
    assert metrics.mean_matches == pytest.approx(5.0)
    assert metrics.calibration_error == pytest.approx(0.0)
    assert metrics.mean_true_number_rank == pytest.approx(3.0)
    assert metrics.regret_vs_uniform == pytest.approx(-uniform_loss)


def test_prediction_record_preserves_the_original_positional_signature():
    record = PredictionRecord(
        date(2020, 2, 1),
        101,
        (1, 2, 3, 4, 5),
        (5 / 49,) * 49,
        (date(2020, 1, 1),),
        (100,),
        date(2020, 1, 1),
        100,
        0,
        True,
    )

    assert record.selected_numbers is None


def test_true_number_rank_uses_label_neutral_average_ranks_for_ties():
    draws = _draws(5)

    def fit(history, feature_data, *, rng):
        return None

    def predict(history, state, feature_data, *, rng):
        return [5 / 49] * 49

    uniform_record = run_walk_forward(
        draws,
        (EvaluationWindow(0, 4, 4, 5),),
        WalkForwardCallbacks(fit=fit, predict=predict),
        WalkForwardConfig(initial_train_size=4),
    )[0]
    assert evaluate_predictions(
        (uniform_record,)
    ).mean_true_number_rank == pytest.approx(25.0)

    tied_probabilities = (
        0.5,
        0.5,
        *(0.4 for _ in range(5)),
        *(2 / 42 for _ in range(42)),
    )
    tied_record = replace(
        uniform_record,
        target_numbers=(1, 2, 3, 4, 8),
        probabilities=tied_probabilities,
    )
    assert evaluate_predictions((tied_record,)).mean_true_number_rank == pytest.approx(
        8.3
    )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([5 / 49] * 48, "exactly 49"),
        ([5 / 49] * 48 + [math.nan], "finite"),
        ([5 / 49] * 48 + [-0.1], r"\[0, 1\]"),
        ([5 / 49] * 48 + [1.1], r"\[0, 1\]"),
        ([0.0] * 49, "sum to 5"),
    ],
)
def test_invalid_probability_vectors_are_rejected_explicitly(values, message):
    with pytest.raises(ValueError, match=message):
        validate_probabilities(values)


@pytest.mark.parametrize(
    "values",
    [
        [True] * 5 + [False] * 44,
        [str(5 / 49)] * 49,
        [complex(5 / 49, 0.0)] * 49,
        [True] + [4 / 48] * 48,
        [str(1.0)] + [4 / 48] * 48,
        [complex(1.0, 0.0)] + [4 / 48] * 48,
        [np.bool_(True)] + [4 / 48] * 48,
    ],
)
def test_probability_vectors_reject_non_real_or_boolean_values_before_coercion(values):
    with pytest.raises(ValueError, match="probabil.*real numbers"):
        validate_probabilities(values)


@pytest.mark.parametrize("number_type", [int, float, np.int64, np.float64])
def test_probability_vectors_accept_python_and_numpy_real_numbers(number_type):
    values = [number_type(1)] * 5 + [number_type(0)] * 44

    assert validate_probabilities(values) == (1.0,) * 5 + (0.0,) * 44


def test_probability_vectors_accept_float32_rounding_within_tolerance():
    uniform = np.full(49, np.float32(5 / 49), dtype=np.float32)
    weights = np.arange(1, 50, dtype=np.float32)
    non_trivial = weights * (np.float32(5.0) / weights.sum(dtype=np.float32))

    assert validate_probabilities(uniform) == tuple(float(value) for value in uniform)
    assert validate_probabilities(non_trivial) == tuple(
        float(value) for value in non_trivial
    )


def test_probability_vectors_accept_observed_float32_normalization_noise():
    # Captured from row 67,270 of a fixed-seed 200,000-vector normalization probe.
    normalized = np.array(
        [
            0.21511727571487427,
            0.016381721943616867,
            0.06748960167169571,
            0.07918750494718552,
            0.13278521597385406,
            0.30092668533325195,
            0.06052325665950775,
            0.18568886816501617,
            0.004082827363163233,
            0.04077644273638725,
            0.15200430154800415,
            0.05701049417257309,
            0.022628862410783768,
            0.0828801691532135,
            0.03859647363424301,
            0.028109950944781303,
            0.06910575181245804,
            0.14132288098335266,
            0.10308432579040527,
            0.16218003630638123,
            0.0643506795167923,
            0.203959122300148,
            0.155479297041893,
            0.14525824785232544,
            0.1351783573627472,
            0.07145275175571442,
            0.3490775227546692,
            0.2911154329776764,
            0.06560469418764114,
            0.021793484687805176,
            0.11287591606378555,
            0.00007182955596363172,
            0.0386914387345314,
            0.09940523654222488,
            0.11837134510278702,
            0.02807212807238102,
            0.0642739087343216,
            0.18141616880893707,
            0.038993820548057556,
            0.00861414149403572,
            0.06944429129362106,
            0.20784291625022888,
            0.154146209359169,
            0.11240362375974655,
            0.01997240073978901,
            0.0157980564981699,
            0.1114949882030487,
            0.08956015855073929,
            0.06539813429117203,
        ],
        dtype=np.float32,
    )
    sum_error = abs(float(normalized.astype(float).sum()) - 5.0)

    assert 1e-6 < sum_error < PROBABILITY_SUM_TOLERANCE
    assert validate_probabilities(normalized) == tuple(
        float(value) for value in normalized
    )


def test_probability_vectors_reject_sum_beyond_documented_tolerance():
    values = [5 / 49] * 49
    values[-1] += PROBABILITY_SUM_TOLERANCE * 10

    with pytest.raises(ValueError, match="sum to 5"):
        validate_probabilities(values)


@pytest.mark.parametrize(
    ("selected_numbers", "error", "message"),
    [
        ((1, 2, 3, 4), ValueError, "exactly five"),
        ((1, 2, 3, 4, 5, 6), ValueError, "exactly five"),
        ((1, 1, 2, 3, 4), ValueError, "unique"),
        ((True, 2, 3, 4, 5), TypeError, "integers"),
        ((1.0, 2, 3, 4, 5), TypeError, "integers"),
        ((0, 1, 2, 3, 4), ValueError, "between 1 and 49"),
        ((1, 2, 3, 4, 50), ValueError, "between 1 and 49"),
    ],
)
def test_walk_forward_rejects_invalid_optional_realized_grids(
    selected_numbers, error, message
):
    draws = _draws(5)

    def fit(history, feature_data, *, rng):
        return None

    def predict(history, state, feature_data, *, rng):
        return PredictionResult(
            probabilities=(5 / 49,) * 49,
            selected_numbers=selected_numbers,
        )

    with pytest.raises(error, match=message):
        run_walk_forward(
            draws,
            (EvaluationWindow(0, 4, 4, 5),),
            WalkForwardCallbacks(fit=fit, predict=predict),
            WalkForwardConfig(initial_train_size=4),
        )


@pytest.mark.parametrize(
    ("target_numbers", "error", "message"),
    [
        ((), ValueError, "exactly five"),
        ((1, 2, 3, 4), ValueError, "exactly five"),
        ((1, 2, 3, 4, 5, 6), ValueError, "exactly five"),
        ((1, 1, 2, 3, 4), ValueError, "unique"),
        ((True, 2, 3, 4, 5), TypeError, "integers"),
        ((1.0, 2, 3, 4, 5), TypeError, "integers"),
        ((0, 1, 2, 3, 4), ValueError, "between 1 and 49"),
        ((1, 2, 3, 4, 50), ValueError, "between 1 and 49"),
    ],
)
def test_invalid_metric_targets_fail_before_calculation(
    target_numbers, error, message
):
    record = SimpleNamespace(
        probabilities=(5 / 49,) * 49,
        target_numbers=target_numbers,
    )

    with pytest.raises(error, match=message):
        evaluate_predictions((record,))


@pytest.mark.parametrize(
    "record",
    [
        SimpleNamespace(target_numbers=(1, 2, 3, 4, 5)),
        SimpleNamespace(probabilities=(5 / 49,) * 49),
        object(),
    ],
)
def test_metric_records_require_probabilities_and_target_numbers(record):
    with pytest.raises(TypeError, match="prediction record.*probabilities.*target_numbers"):
        evaluate_predictions((record,))


def test_invalid_dates_windows_and_parameters_fail_explicitly():
    draws = _draws(7)
    unordered = (draws[0], draws[2], draws[1], *draws[3:])

    with pytest.raises(ValueError, match="strictly increasing"):
        rolling_origin_windows(unordered, initial_train_size=4)
    with pytest.raises(ValueError, match="initial_train_size"):
        rolling_origin_windows(draws, initial_train_size=0)
    with pytest.raises(ValueError, match="insufficient"):
        rolling_origin_windows(draws, initial_train_size=7)
    with pytest.raises(ValueError, match="overlap"):
        run_walk_forward(
            draws,
            (
                EvaluationWindow(0, 4, 4, 6),
                EvaluationWindow(0, 5, 5, 7),
            ),
            _callbacks(),
            WalkForwardConfig(initial_train_size=4),
        )
    with pytest.raises(ValueError, match="retrain_frequency"):
        WalkForwardConfig(
            initial_train_size=4,
            retrain_policy=RetrainPolicy.FIXED_FREQUENCY,
        )
    with pytest.raises(ValueError, match="step_size"):
        rolling_origin_windows(
            draws,
            initial_train_size=4,
            evaluation_size=2,
            step_size=1,
        )


def test_engine_does_not_mutate_any_input_or_perform_io(monkeypatch):
    draws = _draws(7)
    before = deepcopy(draws)
    io_attempts = []

    original_open = builtins.open
    original_path_open = Path.open

    def guarded_open(file, mode="r", *args, **kwargs):
        io_attempts.append(("open", file, mode))
        if any(flag in mode for flag in "wax+"):
            raise AssertionError("walk-forward engine attempted a filesystem write")
        return original_open(file, mode, *args, **kwargs)

    def guarded_path_open(self, mode="r", *args, **kwargs):
        io_attempts.append(("Path.open", self, mode))
        if any(flag in mode for flag in "wax+"):
            raise AssertionError("walk-forward engine attempted a filesystem write")
        return original_path_open(self, mode, *args, **kwargs)

    def forbidden_db_connect(*args, **kwargs):
        io_attempts.append(("sqlite3.connect", args, kwargs))
        raise AssertionError("walk-forward engine attempted database I/O")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(sqlite3, "connect", forbidden_db_connect)

    records = _run(draws)

    assert records
    assert draws == before
    assert all(
        observation == original
        for observation, original in zip(draws, before, strict=True)
    )
    assert io_attempts == []
