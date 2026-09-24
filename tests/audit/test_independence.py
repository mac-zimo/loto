from __future__ import annotations

import numpy as np
import pytest

from loto.audit.independence import _fdr_bh, audit_independence
from loto.audit.uniformity import simulate_draws


def test_independence_report_covers_lags_waiting_runs_fdr_and_replication():
    main, chance = simulate_draws(160, rng=np.random.default_rng(10))

    report = audit_independence(
        main,
        chance,
        lags=(1, 2, 5),
        simulations=79,
        rng=np.random.default_rng(11),
    )

    assert report == audit_independence(
        main,
        chance,
        lags=(1, 2, 5),
        simulations=79,
        rng=np.random.default_rng(11),
    )
    assert report.p_value_adjustment == "Benjamini-Hochberg FDR"
    assert report.replication == "chronological halves"
    assert set(report.minimum_effects) == {"overlap", "waiting_time", "runs"}
    assert {
        "main_overlap_lag_1",
        "chance_repeat_lag_1",
        "main_overlap_lag_2",
        "chance_repeat_lag_2",
        "main_overlap_lag_5",
        "chance_repeat_lag_5",
        "main_waiting_time",
        "chance_waiting_time",
        "main_runs",
        "chance_runs",
    } == set(report.results)
    for result in report.results.values():
        assert 0.0 < result.p_value <= result.adjusted_p_value <= 1.0
        assert len(result.replication_p_values) == 2
        assert len(result.replication_adjusted_p_values) == 2
        assert len(result.replication_effect_sizes) == 2
        assert result.effect_name
        if result.signal:
            assert result.replicated
            assert result.exceeds_minimum_effect


def test_replicated_lag_dependence_is_a_signal_but_threshold_is_predeclared():
    independent, chance = simulate_draws(100, rng=np.random.default_rng(20))
    main = np.repeat(independent, 2, axis=0)
    chance = np.repeat(chance, 2)

    report = audit_independence(
        main,
        chance,
        lags=(1,),
        simulations=199,
        rng=np.random.default_rng(21),
        minimum_effects={"overlap": 0.08, "waiting_time": 0.03, "runs": 0.08},
    )
    overlap = report.results["main_overlap_lag_1"]

    assert overlap.adjusted_p_value <= 0.05
    assert overlap.replicated
    assert overlap.exceeds_minimum_effect
    assert overlap.signal
    assert report.results["main_waiting_time"].signal
    assert report.results["chance_waiting_time"].signal
    assert report.results["main_runs"].signal
    assert report.results["chance_runs"].signal

    blocked = audit_independence(
        main,
        chance,
        lags=(1,),
        simulations=199,
        rng=np.random.default_rng(21),
        minimum_effects={"overlap": 1.01, "waiting_time": 1.01, "runs": 1.01},
    ).results["main_overlap_lag_1"]
    assert blocked.adjusted_p_value <= 0.05
    assert not blocked.exceeds_minimum_effect
    assert not blocked.signal


def test_an_anomaly_confined_to_one_half_never_becomes_a_signal():
    first, first_chance = simulate_draws(40, rng=np.random.default_rng(30))
    second, second_chance = simulate_draws(80, rng=np.random.default_rng(31))
    main = np.concatenate((np.repeat(first, 2, axis=0), second))
    chance = np.concatenate((np.repeat(first_chance, 2), second_chance))

    result = audit_independence(
        main,
        chance,
        lags=(1,),
        simulations=199,
        rng=np.random.default_rng(32),
        minimum_effects={"overlap": 0.01, "waiting_time": 0.01, "runs": 0.01},
    ).results["main_overlap_lag_1"]

    assert not result.replicated
    assert not result.signal


def test_opposite_signed_effects_in_replication_halves_are_not_a_signal():
    main, _ = simulate_draws(400, rng=np.random.default_rng(33))
    repeated = np.repeat(np.tile(np.arange(1, 11), 10), 2)
    non_repeated = np.tile(np.arange(1, 11), 20)
    chance = np.concatenate((repeated, non_repeated))

    result = audit_independence(
        main,
        chance,
        lags=(1,),
        simulations=199,
        rng=np.random.default_rng(34),
        minimum_effects={"overlap": 0.05, "waiting_time": 0.03, "runs": 0.03},
    ).results["chance_repeat_lag_1"]

    assert result.replicated
    assert result.exceeds_minimum_effect
    assert result.replication_signed_effect_sizes[0] > 0
    assert result.replication_signed_effect_sizes[1] < 0
    assert not result.direction_consistent
    assert not result.signal


@pytest.mark.parametrize("lags", [(), (0,), (-1,), (1, 1), (1, 2.0)])
def test_invalid_lags_are_rejected(lags):
    main, chance = simulate_draws(20, rng=np.random.default_rng(40))
    with pytest.raises((TypeError, ValueError), match="lags"):
        audit_independence(
            main,
            chance,
            lags=lags,
            simulations=9,
            rng=np.random.default_rng(41),
        )


def test_benjamini_hochberg_adjustment_matches_known_values():
    adjusted = _fdr_bh(np.asarray([0.01, 0.04, 0.03, 0.002]))

    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])


@pytest.mark.parametrize(
    "p_values",
    [
        np.asarray([[0.1, 0.2]]),
        np.asarray([np.nan]),
        np.asarray([-0.1]),
        np.asarray([1.1]),
        np.asarray(["invalid"]),
    ],
)
def test_benjamini_hochberg_rejects_invalid_p_values(p_values):
    with pytest.raises(ValueError, match="p_values"):
        _fdr_bh(p_values)


@pytest.mark.parametrize("invalid", [True, "0.1", None, np.nan, np.inf, -0.1])
def test_invalid_minimum_effects_are_rejected_cleanly(invalid):
    main, chance = simulate_draws(20, rng=np.random.default_rng(42))
    thresholds = {"overlap": 0.05, "waiting_time": 0.03, "runs": invalid}

    with pytest.raises(ValueError, match="minimum_effects"):
        audit_independence(
            main,
            chance,
            lags=(1,),
            simulations=9,
            rng=np.random.default_rng(43),
            minimum_effects=thresholds,
        )


def test_numpy_numeric_minimum_effects_are_accepted():
    main, chance = simulate_draws(20, rng=np.random.default_rng(44))

    report = audit_independence(
        main,
        chance,
        lags=(1,),
        simulations=9,
        rng=np.random.default_rng(45),
        minimum_effects={
            "overlap": np.float64(0.05),
            "waiting_time": np.int64(0),
            "runs": np.float32(0.03),
        },
    )

    assert report.minimum_effects == {"overlap": 0.05, "waiting_time": 0.0, "runs": pytest.approx(0.03)}
