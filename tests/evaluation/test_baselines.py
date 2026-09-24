from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import date
import builtins
import math
from pathlib import Path
import sqlite3

import numpy as np
import pytest

import loto.evaluation._walk_forward_metrics as metrics
import loto.evaluation.walk_forward as walk_forward
from loto.evaluation.baselines import (
    ConstrainedRandomGrid,
    constrained_random_baseline,
    cumulative_frequency_baseline,
    rolling_frequency_baseline,
    uniform_baseline,
)
from loto.evaluation.walk_forward import (
    DrawObservation,
    PredictionResult,
    WalkForwardConfig,
    evaluate_predictions,
    rolling_origin_windows,
    run_walk_forward,
    validate_probabilities,
)


def _draws(count: int) -> tuple[DrawObservation, ...]:
    return tuple(
        DrawObservation(
            draw_date=date(2020 + index // 12, index % 12 + 1, 1),
            original_index=100 + index,
            numbers=tuple(((index * 5 + offset) % 49) + 1 for offset in range(5)),
        )
        for index in range(count)
    )


def _run(draws, baseline, *, seed=123):
    windows = rolling_origin_windows(
        draws, initial_train_size=4, evaluation_size=3
    )
    return run_walk_forward(
        draws,
        windows,
        baseline.callbacks,
        WalkForwardConfig(initial_train_size=4, seed=seed),
    )


def _predict(baseline, history):
    return baseline.predict(
        history,
        None,
        None,
        rng=np.random.default_rng(123),
    )


def test_uniform_is_exact_history_and_seed_independent_and_shared_with_regret():
    baseline = uniform_baseline()
    first = _predict(baseline, _draws(1))
    second = baseline.predict(
        _draws(3), None, None, rng=np.random.default_rng(999)
    )

    assert metrics.UNIFORM_PROBABILITY == 5 / 49
    assert metrics.UNIFORM_PROBABILITIES == (metrics.UNIFORM_PROBABILITY,) * 49
    assert first == second == metrics.UNIFORM_PROBABILITIES
    assert validate_probabilities(first) == first

    record = type(
        "Record",
        (),
        {"probabilities": first, "target_numbers": (1, 2, 3, 4, 5)},
    )()
    assert evaluate_predictions((record,)).regret_vs_uniform == pytest.approx(0.0)


def test_cumulative_frequency_uses_all_history_and_documented_smoothing_formula():
    history = (
        DrawObservation(date(2020, 1, 1), 1, (1, 2, 3, 4, 5)),
        DrawObservation(date(2020, 2, 1), 2, (1, 6, 7, 8, 9)),
    )
    alpha = 2.0
    baseline = cumulative_frequency_baseline(alpha=alpha)

    probabilities = _predict(baseline, history)
    expected = tuple(
        ((2 if number == 1 else 1 if 2 <= number <= 9 else 0) + alpha * (5 / 49))
        / (len(history) + alpha)
        for number in range(1, 50)
    )

    assert probabilities == pytest.approx(expected)
    assert sum(probabilities) == pytest.approx(5.0, abs=metrics.PROBABILITY_SUM_TOLERANCE)
    assert validate_probabilities(probabilities) == probabilities
    assert baseline.alpha == alpha


def test_cumulative_rejects_empty_history_and_invalid_alpha():
    with pytest.raises(ValueError, match="history.*at least one"):
        _predict(cumulative_frequency_baseline(), ())

    for invalid in (True, False, 0, -1, math.nan, math.inf, "1", None):
        with pytest.raises((TypeError, ValueError), match="alpha"):
            cumulative_frequency_baseline(alpha=invalid)


def test_rolling_frequency_uses_only_last_n_and_identifier_includes_n():
    history = (
        DrawObservation(date(2020, 1, 1), 1, (40, 41, 42, 43, 44)),
        DrawObservation(date(2020, 2, 1), 2, (1, 2, 3, 4, 5)),
        DrawObservation(date(2020, 3, 1), 3, (1, 6, 7, 8, 9)),
    )
    baseline = rolling_frequency_baseline(window_size=2, alpha=1.0)

    probabilities = _predict(baseline, history)
    expected = tuple(
        ((2 if number == 1 else 1 if 2 <= number <= 9 else 0) + 5 / 49) / 3
        for number in range(1, 50)
    )

    assert probabilities == pytest.approx(expected)
    assert all(
        probabilities[number - 1] == pytest.approx((5 / 49) / 3)
        for number in range(40, 45)
    )
    assert baseline.identifier == "rolling_frequency_window=2_alpha=1.0"


def test_frequency_identifiers_distinguish_close_float_alphas():
    first_alpha = 1.0000001
    second_alpha = 1.0000002
    cumulative_first = cumulative_frequency_baseline(alpha=first_alpha)
    cumulative_second = cumulative_frequency_baseline(alpha=second_alpha)
    rolling_first = rolling_frequency_baseline(window_size=2, alpha=first_alpha)
    rolling_second = rolling_frequency_baseline(window_size=2, alpha=second_alpha)

    assert cumulative_first.identifier == f"cumulative_frequency_alpha={first_alpha!r}"
    assert cumulative_second.identifier == f"cumulative_frequency_alpha={second_alpha!r}"
    assert cumulative_first.identifier != cumulative_second.identifier
    assert rolling_first.identifier == (
        f"rolling_frequency_window=2_alpha={first_alpha!r}"
    )
    assert rolling_second.identifier == (
        f"rolling_frequency_window=2_alpha={second_alpha!r}"
    )
    assert rolling_first.identifier != rolling_second.identifier


def test_rolling_frequency_rejects_short_history_and_invalid_windows():
    baseline = rolling_frequency_baseline(window_size=3)
    with pytest.raises(ValueError, match="at least 3"):
        _predict(baseline, _draws(2))

    for invalid in (True, False, 0, -1, 1.0, 2.5, "2", None):
        with pytest.raises((TypeError, ValueError), match="window_size"):
            rolling_frequency_baseline(window_size=invalid)


def test_every_baseline_is_compatible_with_run_walk_forward():
    draws = _draws(10)
    baselines = (
        uniform_baseline(),
        cumulative_frequency_baseline(alpha=1.0),
        rolling_frequency_baseline(window_size=3, alpha=1.0),
        constrained_random_baseline(),
    )

    for baseline in baselines:
        records = _run(draws, baseline)
        assert records
        assert all(validate_probabilities(record.probabilities) for record in records)


def test_future_change_cannot_affect_any_earlier_baseline_prediction():
    draws = _draws(10)
    changed_index = 8
    changed = (
        *draws[:changed_index],
        replace(draws[changed_index], numbers=(45, 46, 47, 48, 49)),
        *draws[changed_index + 1 :],
    )
    baselines = (
        uniform_baseline(),
        cumulative_frequency_baseline(),
        rolling_frequency_baseline(window_size=3),
        constrained_random_baseline(),
    )

    for baseline in baselines:
        original_records = _run(draws, baseline)
        changed_records = _run(changed, baseline)
        original_unaffected = [
            record
            for record in original_records
            if draws[changed_index].original_index not in record.history_original_indices
        ]
        changed_unaffected = [
            record
            for record in changed_records
            if draws[changed_index].original_index not in record.history_original_indices
        ]
        assert [
            (record.probabilities, record.selected_numbers)
            for record in original_unaffected
        ] == [
            (record.probabilities, record.selected_numbers)
            for record in changed_unaffected
        ]
        assert (
            original_unaffected[-1].target_original_index
            == draws[changed_index].original_index
        )


def test_constrained_random_records_are_auditable_valid_and_reproducible():
    draws = _draws(10)
    baseline = constrained_random_baseline()

    first = _run(draws, baseline, seed=456)
    repeated = _run(draws, baseline, seed=456)

    assert first == repeated
    assert all(record.selected_numbers is not None for record in first)
    for record in first:
        assert len(record.selected_numbers) == 5
        assert len(set(record.selected_numbers)) == 5
        assert all(1 <= number <= 49 for number in record.selected_numbers)
        assert record.probabilities == metrics.UNIFORM_PROBABILITIES

    materialized = baseline.materialize(_draws(4), rng=np.random.default_rng(456))
    predicted = baseline.predict(
        _draws(4), None, None, rng=np.random.default_rng(456)
    )
    assert isinstance(materialized, ConstrainedRandomGrid)
    assert materialized.selected_numbers == predicted.selected_numbers
    assert materialized.marginal_probabilities == predicted.probabilities
    with pytest.raises(FrozenInstanceError):
        materialized.selected_numbers = (1, 2, 3, 4, 5)


def test_constrained_random_uses_injected_rng_and_history_derived_stream(monkeypatch):
    baseline = constrained_random_baseline()
    short_history = _draws(4)

    global_state = deepcopy(np.random.get_state())
    supplied = np.random.default_rng(987)
    expected = tuple(
        sorted(np.random.default_rng(987).choice(49, size=5, replace=False) + 1)
    )
    realized = baseline.predict(short_history, None, None, rng=supplied)
    assert isinstance(realized, PredictionResult)
    assert realized.selected_numbers == expected
    assert realized.probabilities == metrics.UNIFORM_PROBABILITIES
    assert all(
        np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
        for left, right in zip(np.random.get_state(), global_state, strict=True)
    )

    original_rng = walk_forward._rng
    derived_states = []

    def rng_probe(seed, stage, history_original_indices):
        rng = original_rng(seed, stage, history_original_indices)
        if stage == "predict":
            derived_states.append(
                (seed, history_original_indices, deepcopy(rng.bit_generator.state))
            )
        return rng

    monkeypatch.setattr(walk_forward, "_rng", rng_probe)
    seeded = _run(_draws(10), baseline, seed=123)
    states_for_123 = deepcopy(derived_states)
    derived_states.clear()
    _run(_draws(10), baseline, seed=124)
    states_for_124 = deepcopy(derived_states)

    assert len({repr(state) for _, _, state in states_for_123}) == len(seeded)
    assert [state for _, _, state in states_for_123] != [
        state for _, _, state in states_for_124
    ]


def test_uniform_and_constrained_records_separate_forecast_from_realization():
    draws = _draws(7)

    uniform_records = _run(draws, uniform_baseline(), seed=321)
    constrained_records = _run(draws, constrained_random_baseline(), seed=321)

    assert [record.probabilities for record in uniform_records] == [
        record.probabilities for record in constrained_records
    ]
    assert all(record.selected_numbers is None for record in uniform_records)
    assert all(record.selected_numbers is not None for record in constrained_records)
    assert uniform_records != constrained_records


def test_constrained_metrics_keep_ex_ante_scores_but_match_realized_grid():
    draws = _draws(5)
    uniform_record = _run(draws, uniform_baseline(), seed=1)[0]
    uniform_record = replace(uniform_record, target_numbers=(1, 2, 3, 4, 5))
    constrained_record = replace(
        uniform_record, selected_numbers=(6, 7, 8, 9, 10)
    )

    uniform_metrics = evaluate_predictions((uniform_record,))
    constrained_metrics = evaluate_predictions((constrained_record,))

    assert constrained_metrics.log_loss == uniform_metrics.log_loss
    assert constrained_metrics.brier == uniform_metrics.brier
    assert constrained_metrics.calibration_error == uniform_metrics.calibration_error
    assert constrained_metrics.mean_true_number_rank == uniform_metrics.mean_true_number_rank
    assert constrained_metrics.regret_vs_uniform == uniform_metrics.regret_vs_uniform
    assert uniform_metrics.mean_matches == 5.0
    assert constrained_metrics.mean_matches == 0.0


def test_baselines_do_not_mutate_history():
    history = _draws(5)
    before = deepcopy(history)

    for baseline in (
        uniform_baseline(),
        cumulative_frequency_baseline(),
        rolling_frequency_baseline(window_size=3),
        constrained_random_baseline(),
    ):
        _predict(baseline, history)
    constrained_random_baseline().materialize(
        history, rng=np.random.default_rng(1)
    )

    assert history == before


def test_baselines_perform_no_filesystem_or_database_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("baseline attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)

    draws = _draws(7)
    for baseline in (
        uniform_baseline(),
        cumulative_frequency_baseline(),
        rolling_frequency_baseline(window_size=3),
        constrained_random_baseline(),
    ):
        assert _run(draws, baseline)
