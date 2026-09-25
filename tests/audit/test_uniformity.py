from __future__ import annotations

import numpy as np
import pytest

from loto.audit.uniformity import _holm_adjust, audit_uniformity, simulate_draws

EXPECTED_TESTS = {
    "marginal_frequencies",
    "ordered_numbers",
    "sum",
    "parity",
    "distances",
    "consecutive_overlap",
    "chance",
}


def _reference_draws(
    n_draws: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Independent fair generator used only to calibrate the implementation."""
    pool = np.arange(1, 50)
    main = np.stack(
        [np.sort(rng.choice(pool, size=5, replace=False)) for _ in range(n_draws)]
    )
    chance = rng.integers(1, 11, size=n_draws, dtype=np.int64)
    return main, chance


def test_exact_generator_respects_game_and_is_reproducible():
    first_main, first_chance = simulate_draws(200, rng=np.random.default_rng(17))
    second_main, second_chance = simulate_draws(200, rng=np.random.default_rng(17))

    np.testing.assert_array_equal(first_main, second_main)
    np.testing.assert_array_equal(first_chance, second_chance)
    assert first_main.shape == (200, 5)
    assert np.all(np.diff(first_main, axis=1) > 0)
    assert first_main.min() >= 1 and first_main.max() <= 49
    assert first_chance.min() >= 1 and first_chance.max() <= 10


def test_report_is_deterministic_and_contains_effects_and_null_intervals():
    main, chance = simulate_draws(180, rng=np.random.default_rng(9))
    first = audit_uniformity(
        main, chance, simulations=199, rng=np.random.default_rng(123)
    )
    second = audit_uniformity(
        main, chance, simulations=199, rng=np.random.default_rng(123)
    )

    assert first == second
    assert first.n_draws == 180
    assert first.simulations == 199
    assert first.p_value_adjustment == "Holm FWER across all seven tests"
    assert set(first.results) == EXPECTED_TESTS
    for result in first.results.values():
        assert 0.0 < result.p_value <= 1.0
        assert result.p_value <= result.adjusted_p_value <= 1.0
        assert result.effect_size >= 0.0
        assert result.null_interval[0] <= result.null_interval[1]
        assert result.effect_name


def test_holm_adjustment_is_deterministic_and_keeps_raw_values_separate():
    raw = np.asarray([0.01, 0.04, 0.03, 0.002])

    adjusted = _holm_adjust(raw)

    np.testing.assert_array_equal(raw, [0.01, 0.04, 0.03, 0.002])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.008])


def test_separate_chance_test_detects_a_large_chance_bias():
    main, _ = simulate_draws(250, rng=np.random.default_rng(4))
    report = audit_uniformity(
        main,
        np.ones(250, dtype=np.int64),
        simulations=299,
        rng=np.random.default_rng(5),
    )

    assert report.results["chance"].p_value <= 0.01
    assert report.results["chance"].adjusted_p_value <= 0.05
    assert report.results["chance"].effect_size > 0.5
    assert report.results["marginal_frequencies"].p_value > 0.01


@pytest.mark.parametrize(
    ("main", "chance", "message"),
    [
        ([[1, 2, 3, 4]], [1], "shape"),
        ([[1, 2, 3, 4, 4], [5, 6, 7, 8, 9]], [1, 2], "distinct"),
        ([[0, 2, 3, 4, 5], [5, 6, 7, 8, 9]], [1, 2], "between 1 and 49"),
        ([[1, 2, 3, 4, 5], [5, 6, 7, 8, 9]], [0, 2], "between 1 and 10"),
        ([[1.0, 2, 3, 4, 5], [5, 6, 7, 8, 9]], [1, 2], "integers"),
    ],
)
def test_invalid_draws_are_rejected(main, chance, message):
    with pytest.raises(ValueError, match=message):
        audit_uniformity(main, chance, simulations=19, rng=np.random.default_rng(1))


@pytest.mark.parametrize("simulations", [0, -1, 1.5, True])
def test_invalid_simulation_count_is_rejected(simulations):
    main, chance = simulate_draws(10, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="simulations"):
        audit_uniformity(
            main, chance, simulations=simulations, rng=np.random.default_rng(2)
        )


def test_rng_must_be_explicit_generator():
    with pytest.raises(TypeError, match="Generator"):
        simulate_draws(10, rng=42)


@pytest.mark.parametrize("n_draws", [0, -1])
def test_invalid_generator_draw_count_is_rejected(n_draws):
    with pytest.raises(ValueError, match="n_draws"):
        simulate_draws(n_draws, rng=np.random.default_rng(1))


@pytest.mark.parametrize("n_draws", [1.5, True])
def test_invalid_generator_draw_count_type_is_rejected(n_draws):
    with pytest.raises(TypeError, match="n_draws"):
        simulate_draws(n_draws, rng=np.random.default_rng(1))


def test_audit_rejects_too_few_draws_and_mismatched_chance_length():
    with pytest.raises(ValueError, match="at least two"):
        audit_uniformity(
            [[1, 2, 3, 4, 5]],
            [1],
            simulations=9,
            rng=np.random.default_rng(1),
        )
    with pytest.raises(ValueError, match="shape"):
        audit_uniformity(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [1],
            simulations=9,
            rng=np.random.default_rng(1),
        )


@pytest.mark.parametrize(
    "confidence",
    [0.0, 1.0, -0.1, 1.1, None, "0.95", True, np.asarray(0.95), [0.95]],
)
def test_invalid_confidence_is_rejected(confidence):
    main, chance = simulate_draws(10, rng=np.random.default_rng(1))
    with pytest.raises(ValueError, match="confidence"):
        audit_uniformity(
            main,
            chance,
            simulations=9,
            rng=np.random.default_rng(2),
            confidence=confidence,
        )


def test_monte_carlo_procedures_are_calibrated_on_fair_simulations():
    reference_rng = np.random.default_rng(20260924)
    audit_rng = np.random.default_rng(24092026)
    p_values = {name: [] for name in EXPECTED_TESTS}
    adjusted_p_values = {name: [] for name in EXPECTED_TESTS}

    for _ in range(24):
        main, chance = _reference_draws(90, reference_rng)
        report = audit_uniformity(main, chance, simulations=99, rng=audit_rng)
        for name, result in report.results.items():
            p_values[name].append(result.p_value)
            adjusted_p_values[name].append(result.adjusted_p_value)

    for name, values in p_values.items():
        values_array = np.asarray(values)
        assert 0.25 < values_array.mean() < 0.75, (name, values)
        assert np.count_nonzero(values_array <= 0.05) <= 4, (name, values)
        adjusted = np.asarray(adjusted_p_values[name])
        assert np.all(adjusted >= values_array)

    family_rejections = sum(
        any(adjusted_p_values[name][index] <= 0.05 for name in EXPECTED_TESTS)
        for index in range(24)
    )
    assert family_rejections <= 4


def _manifestly_biased_draws(metric: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(100 + sorted(EXPECTED_TESTS).index(metric))
    if metric == "marginal_frequencies":
        main = np.stack(
            [
                np.r_[1, np.sort(rng.choice(np.arange(2, 50), 4, replace=False))]
                for _ in range(180)
            ]
        )
    elif metric == "ordered_numbers":
        starts = np.tile(np.arange(49), 4)
        main = np.stack(
            [np.sort((np.arange(start, start + 5) % 49) + 1) for start in starts]
        )
    elif metric == "sum":
        pairs = [
            np.sort(rng.choice(np.arange(1, 25), 2, replace=False)) for _ in range(180)
        ]
        main = np.asarray([[x, y, 25, 50 - y, 50 - x] for x, y in pairs])
    elif metric == "parity":
        main = np.stack(
            [
                np.sort(rng.choice(np.arange(1, 50, 2), 5, replace=False))
                for _ in range(180)
            ]
        )
    elif metric == "distances":
        main = np.stack(
            [np.arange(start, start + 5) for start in rng.integers(1, 46, 180)]
        )
    elif metric == "consecutive_overlap":
        independent, _ = _reference_draws(90, rng)
        main = np.repeat(independent, 2, axis=0)
    else:  # pragma: no cover - only parametrized values call this helper
        raise AssertionError(metric)
    chance = rng.integers(1, 11, size=len(main), dtype=np.int64)
    return main.astype(np.int64), chance


@pytest.mark.parametrize(
    "metric",
    [
        "marginal_frequencies",
        "ordered_numbers",
        "sum",
        "parity",
        "distances",
        "consecutive_overlap",
    ],
)
def test_each_main_metric_detects_a_manifest_targeted_anomaly(metric):
    main, chance = _manifestly_biased_draws(metric)
    report = audit_uniformity(
        main, chance, simulations=199, rng=np.random.default_rng(700)
    )

    result = report.results[metric]
    assert result.p_value <= 0.01
    assert result.adjusted_p_value <= 0.05
