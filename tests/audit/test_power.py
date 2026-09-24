from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from cycler import cycler

from loto.audit import generate_power_artifacts
from loto.audit.power import (
    DEFAULT_ALPHA,
    DEFAULT_POWER_TARGET,
    DEFAULT_SEED,
    PowerCurveReport,
    PowerPoint,
    _critical_value,
    _monte_carlo_p_value,
    audit_detectability,
    estimate_power_curves,
    export_power_data,
    minimum_null_repetitions,
    plot_power_curves,
    simulate_biased_draws,
    simultaneous_max_deviation_upper_bound,
)
from loto.audit.uniformity import simulate_draws


def test_biased_generator_is_exact_reproducible_and_targets_known_marginals():
    first_main, first_chance = simulate_biased_draws(
        4_000,
        main_effect=0.20,
        chance_effect=0.25,
        rng=np.random.default_rng(101),
    )
    second_main, second_chance = simulate_biased_draws(
        4_000,
        main_effect=0.20,
        chance_effect=0.25,
        rng=np.random.default_rng(101),
    )

    np.testing.assert_array_equal(first_main, second_main)
    np.testing.assert_array_equal(first_chance, second_chance)
    assert np.all(np.diff(first_main, axis=1) > 0)
    assert first_main.min() >= 1 and first_main.max() <= 49
    assert first_chance.min() >= 1 and first_chance.max() <= 10
    assert np.mean(np.any(first_main == 1, axis=1)) == pytest.approx(5 / 49 + 0.20, abs=0.025)
    assert np.mean(first_chance == 1) == pytest.approx(0.10 + 0.25, abs=0.025)


def test_power_curves_are_deterministic_calibrated_and_include_binomial_uncertainty():
    arguments = dict(
        n_draws=320,
        effect_sizes=(0.0, 0.03, 0.20),
        null_repetitions=199,
        power_repetitions=80,
        seed=DEFAULT_SEED,
        alpha=0.10,
        confidence=0.95,
        power_target=0.80,
    )

    report = estimate_power_curves(**arguments)

    assert report == estimate_power_curves(**arguments)
    assert report.seed == DEFAULT_SEED
    assert report.multiplicity_correction == "Bonferroni across two marginal tests"
    assert report.per_test_alpha == pytest.approx(0.05)
    assert set(report.curves) == {"main_marginal", "chance_marginal"}
    for points in report.curves.values():
        assert [point.effect_size for point in points] == [0.0, 0.03, 0.20]
        assert points[0].power <= 0.15
        assert points[-1].power >= 0.90
        for point in points:
            assert point.detections <= point.repetitions == 80
            assert 0.0 <= point.interval_low <= point.power <= point.interval_high <= 1.0
            assert point.detectability in {"detectable", "insufficient_power"}


def test_finite_monte_carlo_p_value_uses_plus_one_and_counts_ties():
    null_statistics = np.array([0.1, 0.2, 0.2, 0.3])

    assert _monte_carlo_p_value(0.2, null_statistics) == pytest.approx(4 / 5)
    assert _monte_carlo_p_value(0.31, null_statistics) == pytest.approx(1 / 5)


def test_critical_value_matches_p_value_at_float_boundary_regression():
    null_statistics = np.arange(97, dtype=float)
    alpha = 2 / 98
    observed = 96.0

    assert alpha * 98 == 1.9999999999999998
    assert _monte_carlo_p_value(observed, null_statistics) == alpha
    assert observed > _critical_value(null_statistics, alpha)


@pytest.mark.parametrize("null_repetitions", [1, 2, 3, 7, 19, 97])
def test_critical_value_is_exactly_equivalent_to_p_value_with_ties(
    null_repetitions,
):
    null_statistics = (np.arange(null_repetitions, dtype=float) % 5) / 10
    observed_values = {-0.1, 0.6}
    for value in np.unique(null_statistics):
        observed_values.update(
            {np.nextafter(value, -np.inf), value, np.nextafter(value, np.inf)}
        )

    numerators = {1, 2, max(1, null_repetitions // 2), null_repetitions}
    for numerator in numerators:
        boundary = numerator / (null_repetitions + 1)
        for alpha in {
            np.nextafter(boundary, 0.0),
            boundary,
            np.nextafter(boundary, 1.0),
        }:
            if not 1 / (null_repetitions + 1) <= alpha < 1:
                continue
            critical = _critical_value(null_statistics, alpha)
            for observed in observed_values:
                assert (observed > critical) == (
                    _monte_carlo_p_value(observed, null_statistics) <= alpha
                )


def test_null_repetitions_must_make_per_test_alpha_attainable():
    assert minimum_null_repetitions(0.05, number_of_tests=2) == 39
    assert minimum_null_repetitions(2 / 98, number_of_tests=1) == 48
    with pytest.raises(ValueError, match="null_repetitions.*39"):
        estimate_power_curves(
            n_draws=100,
            effect_sizes=(0.0,),
            null_repetitions=38,
            power_repetitions=10,
            alpha=0.05,
        )


def test_minimum_null_repetitions_uses_the_actual_float_comparison():
    for number_of_tests in (1, 2, 7):
        for denominator in range(number_of_tests + 1, 101):
            boundary = number_of_tests / denominator
            for alpha in {
                np.nextafter(boundary, 0.0),
                boundary,
                np.nextafter(boundary, 1.0),
            }:
                repetitions = minimum_null_repetitions(
                    alpha, number_of_tests=number_of_tests
                )
                per_test_alpha = alpha / number_of_tests
                assert 1 / (repetitions + 1) <= per_test_alpha
                assert repetitions == 1 or 1 / repetitions > per_test_alpha


def test_simultaneous_equivalence_bound_separates_inside_and_outside_margin():
    n_draws = 200_000
    fair_counts = np.full(10, n_draws // 10)
    outside_counts = np.array([24_000, *([19_555] * 8), 19_560])

    inside_bound = simultaneous_max_deviation_upper_bound(
        fair_counts, n_draws=n_draws, expected_probability=0.1, alpha=0.025
    )
    outside_bound = simultaneous_max_deviation_upper_bound(
        outside_counts, n_draws=n_draws, expected_probability=0.1, alpha=0.025
    )

    assert inside_bound < 0.01
    assert outside_bound > 0.02


def test_simulated_false_absence_rate_at_equivalence_boundary_is_controlled():
    rng = np.random.default_rng(890)
    repetitions = 300
    alpha = 0.10
    margin = 0.04
    probabilities = np.array([0.1 + margin, *([(0.9 - margin) / 9] * 9)])
    false_absences = sum(
        simultaneous_max_deviation_upper_bound(
            rng.multinomial(20_000, probabilities),
            n_draws=20_000,
            expected_probability=0.1,
            alpha=alpha,
        )
        < margin
        for _ in range(repetitions)
    )

    # At the equivalence boundary, absence is a family-wise confidence error.
    # The tolerance only covers Monte-Carlo uncertainty in this regression test.
    assert false_absences / repetitions <= alpha + 0.03


def test_observed_audit_uses_simultaneous_equivalence_not_power():
    n_draws = 20_000
    main, chance = simulate_draws(n_draws, rng=np.random.default_rng(202))
    report = PowerCurveReport(
        n_draws=n_draws,
        seed=303,
        alpha=0.10,
        per_test_alpha=0.05,
        confidence=0.95,
        power_target=0.80,
        null_repetitions=99,
        power_repetitions=10,
        multiplicity_correction="Bonferroni across two marginal tests",
        statistic="maximum absolute marginal inclusion-rate deviation",
        bias_definition="probability-point increase for number 1",
        monte_carlo_method="finite +1 upper-tail p-value",
        equivalence_margin=0.02,
        equivalence_method="simultaneous exact binomial intervals",
        critical_values={"main_marginal": 1.0, "chance_marginal": 1.0},
        curves={"main_marginal": (), "chance_marginal": ()},
    )

    absent = audit_detectability(main, chance, power_report=report, equivalence_margin=0.03)
    underpowered = audit_detectability(
        main, chance, power_report=report, equivalence_margin=0.001
    )
    biased_main = np.tile(np.arange(1, 6), (n_draws, 1))
    detection_report = replace(
        report,
        critical_values={"main_marginal": 0.20, "chance_marginal": 1.0},
    )
    detected = audit_detectability(
        biased_main,
        chance,
        power_report=detection_report,
        equivalence_margin=0.03,
    )
    equality_report = replace(
        report,
        critical_values={
            "main_marginal": absent.results["main_marginal"].statistic,
            "chance_marginal": 1.0,
        },
    )
    at_critical_value = audit_detectability(
        main,
        chance,
        power_report=equality_report,
        equivalence_margin=0.03,
    )

    assert absent.results["main_marginal"].status == "bias_absent"
    assert underpowered.results["main_marginal"].status == "insufficient_power"
    assert detected.results["main_marginal"].status == "bias_detected"
    assert detected.results["main_marginal"].detected
    assert not at_critical_value.results["main_marginal"].detected
    assert (
        at_critical_value.results["main_marginal"].statistic
        == at_critical_value.results["main_marginal"].critical_value
    )
    assert absent.results["main_marginal"].equivalence_upper_bound < 0.03
    assert underpowered.results["main_marginal"].equivalence_upper_bound >= 0.001
    assert not underpowered.results["main_marginal"].detected
    assert "simultaneous" in underpowered.interpretation


def test_curve_data_and_plot_exports_are_reproducible_and_transparent(tmp_path):
    report = estimate_power_curves(
        n_draws=120,
        effect_sizes=(0.0, 0.15),
        null_repetitions=99,
        power_repetitions=30,
        seed=505,
        alpha=0.10,
    )
    paths = {}
    for suffix in ("csv", "json", "png", "svg"):
        first = tmp_path / f"power-first.{suffix}"
        second = tmp_path / f"power-second.{suffix}"
        exporter = export_power_data if suffix in {"csv", "json"} else plot_power_curves
        exporter(report, first)
        exporter(report, second)
        assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
            second.read_bytes()
        ).digest()
        paths[suffix] = first

    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert {row["scenario"] for row in rows} == {"main_marginal", "chance_marginal"}
    assert payload["seed"] == 505
    assert payload["alpha"] == 0.10
    assert payload["per_test_alpha"] == 0.05
    assert payload["critical_values"] == report.critical_values
    assert payload["n_draws"] == 120
    assert payload["null_repetitions"] == 99
    assert payload["power_repetitions"] == 30
    assert payload["multiplicity_correction"]
    assert payload["power_target"] == DEFAULT_POWER_TARGET
    assert payload["bias_definition"]
    assert payload["curves"]["main_marginal"][1]["repetitions"] == 30
    assert paths["png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths["svg"].read_bytes().lstrip().startswith(b"<?xml")


def test_csv_export_rejects_report_without_power_points(tmp_path):
    report = PowerCurveReport(
        n_draws=20,
        seed=1,
        alpha=0.05,
        per_test_alpha=0.025,
        confidence=0.95,
        power_target=0.80,
        null_repetitions=39,
        power_repetitions=2,
        multiplicity_correction="Bonferroni across two marginal tests",
        statistic="maximum absolute marginal inclusion-rate deviation",
        bias_definition="probability-point increase for number 1",
        monte_carlo_method="finite +1 upper-tail p-value",
        equivalence_margin=0.02,
        equivalence_method="simultaneous exact binomial intervals",
        critical_values={"main_marginal": 1.0, "chance_marginal": 1.0},
        curves={"main_marginal": (), "chance_marginal": ()},
    )

    with pytest.raises(ValueError, match="at least one power point"):
        export_power_data(report, tmp_path / "empty.csv")


@pytest.mark.parametrize("suffix", ["png", "svg"])
def test_plot_export_rejects_report_without_power_points(tmp_path, suffix):
    report = PowerCurveReport(
        n_draws=20,
        seed=1,
        alpha=0.05,
        per_test_alpha=0.025,
        confidence=0.95,
        power_target=0.80,
        null_repetitions=39,
        power_repetitions=2,
        multiplicity_correction="Bonferroni across two marginal tests",
        statistic="maximum absolute marginal inclusion-rate deviation",
        bias_definition="probability-point increase for number 1",
        monte_carlo_method="finite +1 upper-tail p-value",
        equivalence_margin=0.02,
        equivalence_method="simultaneous exact binomial intervals",
        critical_values={"main_marginal": 1.0, "chance_marginal": 1.0},
        curves={"main_marginal": (), "chance_marginal": ()},
    )

    with pytest.raises(
        ValueError, match="^report must contain at least one power point$"
    ):
        plot_power_curves(report, tmp_path / f"empty.{suffix}")


def test_generate_default_artifacts_writes_exact_expected_files(tmp_path, monkeypatch):
    point = PowerPoint(
        effect_size=0.0,
        detections=1,
        repetitions=2,
        power=0.5,
        interval_low=0.1,
        interval_high=0.9,
        detectability="insufficient_power",
    )
    report = PowerCurveReport(
        n_draws=2_811,
        seed=1,
        alpha=0.05,
        per_test_alpha=0.025,
        confidence=0.95,
        power_target=0.80,
        null_repetitions=39,
        power_repetitions=2,
        multiplicity_correction="Bonferroni across two marginal tests",
        statistic="maximum absolute marginal inclusion-rate deviation",
        bias_definition="probability-point increase for number 1",
        monte_carlo_method="finite +1 upper-tail p-value",
        equivalence_margin=0.02,
        equivalence_method="simultaneous exact binomial intervals",
        critical_values={"main_marginal": 1.0, "chance_marginal": 1.0},
        curves={"main_marginal": (point,), "chance_marginal": (point,)},
    )
    monkeypatch.setattr(
        generate_power_artifacts, "estimate_power_curves", lambda: report
    )

    generate_power_artifacts.generate_default_artifacts(tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "power_curves_2811.csv",
        "power_manifest_2811.json",
        "power_curves_2811.png",
    }
    assert (tmp_path / "power_curves_2811.csv").read_text(encoding="utf-8")
    assert (tmp_path / "power_manifest_2811.json").read_text(encoding="utf-8")
    assert (tmp_path / "power_curves_2811.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )


def test_plot_exports_ignore_user_rcparams(tmp_path):
    import matplotlib as mpl

    report = estimate_power_curves(
        n_draws=80,
        effect_sizes=(0.0, 0.15),
        null_repetitions=39,
        power_repetitions=10,
        seed=606,
        alpha=0.10,
    )
    defaults = {}
    customized = {}
    for suffix in ("png", "svg"):
        defaults[suffix] = tmp_path / f"default.{suffix}"
        plot_power_curves(report, defaults[suffix])

    with mpl.rc_context(
        {
            "axes.facecolor": "magenta",
            "axes.prop_cycle": cycler(color=["red", "lime"]),
            "figure.dpi": 37,
            "font.family": ["monospace"],
            "font.size": 17,
            "lines.linewidth": 7.0,
            "savefig.dpi": 41,
            "text.color": "cyan",
        }
    ):
        for suffix in ("png", "svg"):
            customized[suffix] = tmp_path / f"customized.{suffix}"
            plot_power_curves(report, customized[suffix])

    for suffix in ("png", "svg"):
        assert customized[suffix].read_bytes() == defaults[suffix].read_bytes()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_draws": 1}, "n_draws"),
        ({"effect_sizes": ()}, "effect_sizes"),
        ({"effect_sizes": (0.1, 0.1)}, "effect_sizes"),
        ({"effect_sizes": (-0.1,)}, "effect_sizes"),
        ({"effect_sizes": (0.90,)}, "effect_sizes"),
        ({"null_repetitions": 0}, "null_repetitions"),
        ({"power_repetitions": True}, "power_repetitions"),
        ({"seed": 1.5}, "seed"),
        ({"alpha": 0.0}, "alpha"),
        ({"confidence": 1.0}, "confidence"),
        ({"power_target": np.nan}, "power_target"),
    ],
)
def test_invalid_power_parameters_are_rejected(overrides, message):
    arguments = dict(
        n_draws=100,
        effect_sizes=(0.0, 0.10),
        null_repetitions=39,
        power_repetitions=10,
        seed=1,
    )
    arguments.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        estimate_power_curves(**arguments)


def test_preregistered_defaults_are_explicit():
    assert DEFAULT_ALPHA == 0.05
    assert DEFAULT_POWER_TARGET == 0.80
    assert isinstance(DEFAULT_SEED, int)


def test_main_and_chance_effect_limits_are_distinct():
    simulate_biased_draws(
        10, chance_effect=0.90, rng=np.random.default_rng(1)
    )
    with pytest.raises(ValueError, match="main_effect"):
        simulate_biased_draws(
            10, main_effect=0.90, rng=np.random.default_rng(1)
        )


def test_common_power_curve_grid_is_limited_to_main_game_maximum():
    with pytest.raises(ValueError, match="effect_sizes.*0.897"):
        estimate_power_curves(
            n_draws=20,
            effect_sizes=(45 / 49,),
            null_repetitions=39,
            power_repetitions=2,
            alpha=0.05,
        )
