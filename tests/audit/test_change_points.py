from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from loto.audit.change_points import audit_regime_changes
from loto.audit.uniformity import simulate_draws


def _dates(n: int, start: date = date(2022, 1, 1)) -> list[date]:
    return [start + timedelta(days=3 * index) for index in range(n)]


def test_year_and_available_metadata_are_a_single_fdr_family():
    main, chance = simulate_draws(160, rng=np.random.default_rng(50))
    dates = _dates(80, date(2021, 1, 1)) + _dates(80, date(2022, 1, 1))
    metadata = {
        "machine": np.tile(["A", "B"], 80),
        "rule": np.repeat("current", 160),
        "unavailable": None,
    }

    report = audit_regime_changes(
        main,
        chance,
        dates,
        metadata=metadata,
        simulations=79,
        rng=np.random.default_rng(51),
    )

    assert report == audit_regime_changes(
        main,
        chance,
        dates,
        metadata=metadata,
        simulations=79,
        rng=np.random.default_rng(51),
    )
    assert report.p_value_adjustment == "Benjamini-Hochberg FDR"
    assert report.replication == "early/late halves within each regime"
    assert set(report.results) == {"year", "machine"}
    for result in report.results.values():
        assert result.groups >= 2
        assert 0.0 < result.p_value <= result.adjusted_p_value <= 1.0
        assert len(result.replication_p_values) == 2
        if result.signal:
            assert result.replicated
            assert result.exceeds_minimum_effect


def test_replicated_machine_regime_change_is_detected():
    rng = np.random.default_rng(60)
    labels = np.tile(np.repeat(["A", "B"], 40), 2)
    main = np.stack(
        [
            np.sort(rng.choice(np.arange(1, 20) if label == "A" else np.arange(31, 50), 5, replace=False))
            for label in labels
        ]
    )
    chance = np.asarray([rng.integers(1, 5) if label == "A" else rng.integers(7, 11) for label in labels])
    dates = _dates(80, date(2021, 1, 1)) + _dates(80, date(2022, 1, 1))

    result = audit_regime_changes(
        main,
        chance,
        dates,
        metadata={"machine": labels},
        simulations=199,
        rng=np.random.default_rng(61),
        minimum_effect=0.10,
    ).results["machine"]

    assert result.adjusted_p_value <= 0.05
    assert result.effect_size >= 0.10
    assert result.replicated
    assert result.signal


def test_effect_threshold_blocks_a_statistically_significant_regime():
    rng = np.random.default_rng(70)
    labels = np.tile(np.repeat(["A", "B"], 35), 2)
    main = np.stack(
        [
            np.sort(rng.choice(np.arange(1, 25) if label == "A" else np.arange(25, 50), 5, replace=False))
            for label in labels
        ]
    )
    chance = rng.integers(1, 11, len(labels))
    dates = _dates(70, date(2020, 1, 1)) + _dates(70, date(2021, 1, 1))

    result = audit_regime_changes(
        main,
        chance,
        dates,
        metadata={"machine": labels},
        simulations=199,
        rng=np.random.default_rng(71),
        minimum_effect=1.01,
    ).results["machine"]

    assert result.adjusted_p_value <= 0.05
    assert not result.exceeds_minimum_effect
    assert not result.signal


def test_mixed_date_and_datetime_values_are_compared_chronologically():
    main, chance = simulate_draws(20, rng=np.random.default_rng(72))
    dates = [
        date(2022, 1, 1) + timedelta(days=index)
        if index % 2 == 0
        else datetime(2022, 1, 1) + timedelta(days=index)
        for index in range(20)
    ]

    report = audit_regime_changes(
        main,
        chance,
        dates,
        simulations=9,
        rng=np.random.default_rng(73),
    )

    assert report.n_draws == 20
    assert report.skipped["year"] == "fewer than two regimes"


def test_aware_datetimes_are_ordered_by_their_utc_instants():
    main, chance = simulate_draws(20, rng=np.random.default_rng(74))
    dates = [
        datetime(2022, 1, 1, 12, tzinfo=timezone(timedelta(hours=2))),
        datetime(2022, 1, 1, 10, 30, tzinfo=timezone.utc),
        *[
            datetime(2022, 1, 2, tzinfo=timezone.utc) + timedelta(days=index)
            for index in range(18)
        ],
    ]

    report = audit_regime_changes(
        main,
        chance,
        dates,
        simulations=9,
        rng=np.random.default_rng(75),
    )

    assert report.n_draws == 20

    dates[:2] = [
        datetime(2022, 1, 1, 10, tzinfo=timezone.utc),
        datetime(2022, 1, 1, 11, tzinfo=timezone(timedelta(hours=2))),
    ]
    with pytest.raises(ValueError, match="chronological"):
        audit_regime_changes(
            main,
            chance,
            dates,
            simulations=9,
            rng=np.random.default_rng(75),
        )


@pytest.mark.parametrize(
    "dates",
    [
        [datetime(2022, 1, 1, tzinfo=timezone.utc), datetime(2022, 1, 2)],
        [datetime(2022, 1, 1, tzinfo=timezone.utc), date(2022, 1, 2)],
    ],
)
def test_aware_datetimes_cannot_be_mixed_with_naive_temporal_values(dates):
    main, chance = simulate_draws(2, rng=np.random.default_rng(76))

    with pytest.raises(ValueError, match="aware.*naive.*date"):
        audit_regime_changes(
            main,
            chance,
            dates,
            simulations=9,
            rng=np.random.default_rng(77),
        )


def test_missing_metadata_values_are_skipped_as_incomplete():
    main, chance = simulate_draws(20, rng=np.random.default_rng(78))

    report = audit_regime_changes(
        main,
        chance,
        _dates(20),
        metadata={
            "nan": ["A"] * 19 + [np.nan],
            "pandas_na": ["A"] * 19 + [pd.NA],
        },
        simulations=9,
        rng=np.random.default_rng(79),
    )

    assert report.skipped["nan"] == "metadata incomplete"
    assert report.skipped["pandas_na"] == "metadata incomplete"


def test_heterogeneous_scalar_metadata_labels_are_supported():
    main, chance = simulate_draws(20, rng=np.random.default_rng(82))

    report = audit_regime_changes(
        main,
        chance,
        _dates(20),
        metadata={"machine": np.asarray(["A", 1] * 10, dtype=object)},
        simulations=9,
        rng=np.random.default_rng(83),
    )

    assert report.results["machine"].groups == 2


def test_non_scalar_or_non_1d_metadata_labels_are_skipped_cleanly():
    main, chance = simulate_draws(20, rng=np.random.default_rng(84))
    object_labels = np.empty(20, dtype=object)
    object_labels[:] = [np.asarray(["A"])] * 20

    report = audit_regime_changes(
        main,
        chance,
        _dates(20),
        metadata={
            "matrix": np.full((20, 1), "A"),
            "nested": object_labels,
        },
        simulations=9,
        rng=np.random.default_rng(85),
    )

    assert report.skipped["matrix"] == "metadata must be one-dimensional scalar labels"
    assert report.skipped["nested"] == "metadata must be one-dimensional scalar labels"


def test_invalid_or_too_sparse_regime_inputs_are_rejected_or_skipped():
    main, chance = simulate_draws(20, rng=np.random.default_rng(80))
    with pytest.raises(ValueError, match="dates"):
        audit_regime_changes(
            main,
            chance,
            _dates(19),
            simulations=9,
            rng=np.random.default_rng(81),
        )
    with pytest.raises(ValueError, match="metadata.*machine"):
        audit_regime_changes(
            main,
            chance,
            _dates(20),
            metadata={"machine": ["A"] * 19},
            simulations=9,
            rng=np.random.default_rng(81),
        )
