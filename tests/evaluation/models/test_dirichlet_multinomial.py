from __future__ import annotations

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from datetime import date
import inspect
import math
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from loto.evaluation.baselines import (
    cumulative_frequency_baseline,
    uniform_baseline,
)
from loto.evaluation.models.dirichlet_multinomial import (
    PREREGISTERED_METRICS,
    PROMOTION_RULE,
    REJECTION_RULE,
    UNIFORM_PRIOR_EQUIVALENCE,
    UNIFORM_PRIOR_MEAN,
    DirichletMultinomialConfig,
    DirichletMultinomialModel,
    DirichletMultinomialState,
    project_bounded_simplex,
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
    return tuple(
        DrawObservation(
            draw_date=date(2020 + index // 12, index % 12 + 1, 1),
            original_index=100 + index,
            numbers=tuple(((index * 5 + offset) % 49) + 1 for offset in range(5)),
        )
        for index in range(count)
    )


def _history_with_one_always_present(count: int) -> tuple[DrawObservation, ...]:
    return tuple(
        DrawObservation(
            draw_date=date(2020 + index // 12, index % 12 + 1, 1),
            original_index=200 + index,
            numbers=(
                1,
                2 + (4 * index) % 48,
                2 + (4 * index + 1) % 48,
                2 + (4 * index + 2) % 48,
                2 + (4 * index + 3) % 48,
            ),
        )
        for index in range(count)
    )


def _fit(model: DirichletMultinomialModel, history):
    return model.fit(history, None, rng=np.random.default_rng(1))


def _predict(model: DirichletMultinomialModel, history, state):
    return model.predict(history, state, None, rng=np.random.default_rng(2))


def _non_uniform_prior() -> tuple[float, ...]:
    prior = [5 / 49] * 49
    prior[0] += 0.001
    prior[1] -= 0.001
    return tuple(prior)


def _state_with_prior(prior_parameters: tuple[float, ...]):
    concentration = math.fsum(prior_parameters) / 5
    counts = (1,) * 5 + (0,) * 44
    posterior = tuple(
        prior + count
        for prior, count in zip(prior_parameters, counts, strict=True)
    )
    denominator = 1 + concentration
    raw = tuple(parameter / denominator for parameter in posterior)
    probabilities = project_bounded_simplex(raw)
    total_mass = math.fsum(posterior)
    variances = tuple(
        25
        * (parameter / total_mass)
        * (1 - parameter / total_mass)
        / (total_mass + 1)
        for parameter in posterior
    )
    return DirichletMultinomialState(
        sample_size=1,
        concentration=concentration,
        counts=counts,
        prior_parameters=prior_parameters,
        posterior_parameters=posterior,
        raw_probabilities=raw,
        probabilities=probabilities,
        posterior_variances=variances,
        projection_applied=probabilities != raw,
    )


def _run(draws, model_or_baseline, *, seed=123, policy=RetrainPolicy.EACH_DRAW, frequency=None):
    return run_walk_forward(
        draws,
        (EvaluationWindow(0, 4, 4, len(draws)),),
        model_or_baseline.callbacks,
        WalkForwardConfig(
            initial_train_size=4,
            retrain_policy=policy,
            retrain_frequency=frequency,
            seed=seed,
        ),
    )


def test_uniform_prior_formula_is_explicitly_the_existing_cumulative_baseline():
    history = _draws(4)
    alpha = 2.5
    config = DirichletMultinomialConfig(concentration=alpha)
    model = DirichletMultinomialModel(config)
    baseline = cumulative_frequency_baseline(alpha=alpha)

    state = _fit(model, history)
    model_probabilities = _predict(model, history, state)
    baseline_probabilities = baseline.predict(
        history, None, None, rng=np.random.default_rng(9)
    )
    expected = tuple(
        (count + alpha * 5 / 49) / (len(history) + alpha)
        for count in state.counts
    )

    assert model_probabilities == pytest.approx(expected, abs=1e-15)
    assert model_probabilities == pytest.approx(baseline_probabilities, abs=1e-15)
    assert config.primary_baseline == baseline.identifier
    assert config.identifier != baseline.identifier
    assert repr(alpha) in config.identifier
    assert "algebraically identical" in UNIFORM_PRIOR_EQUIVALENCE
    assert "(count_i + alpha * 5/49) / (n + alpha)" in UNIFORM_PRIOR_EQUIVALENCE
    assert "not a distinct predictive family" in UNIFORM_PRIOR_EQUIVALENCE
    assert "without replacement" in inspect.getdoc(DirichletMultinomialModel)
    assert "not the exact joint law" in inspect.getdoc(DirichletMultinomialModel)


def test_uniform_prior_predictions_and_metrics_equal_baseline_in_same_engine():
    draws = _draws(9)
    alpha = 3.0
    model = DirichletMultinomialModel(
        DirichletMultinomialConfig(concentration=alpha)
    )
    baseline = cumulative_frequency_baseline(alpha=alpha)

    model_records = _run(draws, model)
    baseline_records = _run(draws, baseline)
    uniform_records = _run(draws, uniform_baseline())

    assert [record.probabilities for record in model_records] == pytest.approx(
        [record.probabilities for record in baseline_records], abs=1e-15
    )
    evaluations = tuple(
        evaluate_predictions(records)
        for records in (model_records, baseline_records, uniform_records)
    )
    assert evaluations[0] == evaluations[1]
    assert evaluations[2].regret_vs_uniform == pytest.approx(0.0)
    assert all(len(records) == len(model_records) for records in (
        model_records,
        baseline_records,
        uniform_records,
    ))
    assert all(
        math.isfinite(getattr(metrics, metric_name))
        for metrics in evaluations
        for metric_name in PREREGISTERED_METRICS
    )
    assert len(
        {model.config.identifier, baseline.identifier, uniform_baseline().identifier}
    ) == 3


def test_calculable_posterior_state_exposes_every_auditable_quantity():
    history = (
        DrawObservation(date(2020, 1, 1), 1, (1, 2, 3, 4, 5)),
        DrawObservation(date(2020, 2, 1), 2, (1, 6, 7, 8, 9)),
    )
    prior_mean = (0.5,) * 5 + (5 / 88,) * 44
    model = DirichletMultinomialModel(
        DirichletMultinomialConfig(
            concentration=2.0,
            prior_mean=prior_mean,
            minimum_history=2,
        )
    )

    state = _fit(model, history)
    expected_counts = (2,) + (1,) * 8 + (0,) * 40
    expected_prior = (1.0,) * 5 + (5 / 44,) * 44
    expected_posterior = tuple(
        prior + count
        for prior, count in zip(expected_prior, expected_counts, strict=True)
    )
    expected_raw = tuple(parameter / 4 for parameter in expected_posterior)
    total_mass = 5 * (2 + 2.0)
    expected_variances = tuple(
        25
        * (parameter / total_mass)
        * (1 - parameter / total_mass)
        / (total_mass + 1)
        for parameter in expected_posterior
    )

    assert state.sample_size == 2
    assert state.counts == expected_counts
    assert state.prior_parameters == expected_prior
    assert state.posterior_parameters == expected_posterior
    assert state.raw_probabilities == expected_raw
    assert state.probabilities == expected_raw
    assert state.projection_applied is False
    assert state.posterior_variances == pytest.approx(expected_variances)
    assert _predict(model, history, state) == expected_raw
    assert all(0.0 <= probability <= 1.0 for probability in state.probabilities)
    assert sum(state.probabilities) == pytest.approx(5.0)
    assert sum(state.prior_parameters) == pytest.approx(5 * model.config.concentration)


def test_shrinkage_responds_to_history_size_and_fixed_concentration_only():
    prior = 5 / 49
    short_history = _history_with_one_always_present(1)
    long_history = _history_with_one_always_present(10)
    weak = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=1.0))
    strong = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=100.0))

    short_probability = _fit(weak, short_history).probabilities[0]
    long_probability = _fit(weak, long_history).probabilities[0]
    strong_probability = _fit(strong, short_history).probabilities[0]

    assert abs(short_probability - prior) < abs(short_probability - 1.0)
    assert abs(long_probability - 1.0) < abs(short_probability - 1.0)
    assert abs(strong_probability - prior) < abs(short_probability - prior)
    assert weak.config.concentration == 1.0
    assert strong.config.concentration == 100.0


def test_configuration_contains_immutable_preregistered_metadata():
    config = DirichletMultinomialConfig(concentration=4, minimum_history=3)

    assert config.concentration == 4.0
    assert config.minimum_history == 3
    assert config.prior_mean == (5 / 49,) * 49
    assert config.identifier == (
        "dirichlet_multinomial_marginal_concentration=4.0_"
        "minimum_history=3_prior=uniform_5_over_49"
    )
    assert config.primary_baseline == "cumulative_frequency_alpha=4.0"
    assert config.trial_budget == 1
    assert config.allowed_parameters == (
        "concentration",
        "prior_mean",
        "minimum_history",
    )
    assert config.metrics == PREREGISTERED_METRICS == (
        "log_loss",
        "brier",
        "mean_matches",
        "calibration_error",
        "mean_true_number_rank",
        "regret_vs_uniform",
    )
    assert config.promotion_rule == PROMOTION_RULE
    assert "statistically and economically significant" in config.promotion_rule
    assert "at least three temporal periods" in config.promotion_rule
    assert "untouched final holdout" in config.promotion_rule
    assert config.rejection_rule == REJECTION_RULE
    assert "statistically and economically significant" in config.rejection_rule
    assert "at least three temporal periods" in config.rejection_rule
    assert "untouched final holdout" in config.rejection_rule
    assert "fixed before evaluation" in config.preregistration_note
    assert "reject deterministically before fitting" in config.insufficient_data_behavior
    assert "len(history) < minimum_history" in config.insufficient_data_behavior
    with pytest.raises(FrozenInstanceError):
        config.concentration = 5.0
    with pytest.raises(FrozenInstanceError):
        config.trial_budget = 2
    with pytest.raises(TypeError):
        DirichletMultinomialConfig(trial_budget=2)
    with pytest.raises(TypeError):
        DirichletMultinomialConfig(allowed_parameters=("concentration",))


def test_identifier_covers_every_predictive_parameter_and_prior_kind():
    uniform = DirichletMultinomialConfig(concentration=2.0, minimum_history=3)
    concentration_changed = DirichletMultinomialConfig(
        concentration=3.0, minimum_history=3
    )
    history_changed = DirichletMultinomialConfig(
        concentration=2.0, minimum_history=4
    )
    non_uniform = DirichletMultinomialConfig(
        concentration=2.0,
        minimum_history=3,
        prior_mean=_non_uniform_prior(),
    )
    repeated = DirichletMultinomialConfig(
        concentration=2.0,
        minimum_history=3,
        prior_mean=_non_uniform_prior(),
    )
    another_prior = list(_non_uniform_prior())
    another_prior[2] += 0.0001
    another_prior[3] -= 0.0001
    another_non_uniform = DirichletMultinomialConfig(
        concentration=2.0,
        minimum_history=3,
        prior_mean=tuple(another_prior),
    )

    identifiers = {
        uniform.identifier,
        concentration_changed.identifier,
        history_changed.identifier,
        non_uniform.identifier,
        another_non_uniform.identifier,
    }
    assert len(identifiers) == 5
    assert "prior=uniform_5_over_49" in uniform.identifier
    assert "minimum_history=3" in uniform.identifier
    assert "prior=sha256:" in non_uniform.identifier
    assert len(non_uniform.identifier.rsplit("prior=sha256:", 1)[1]) == 64
    assert non_uniform.identifier == repeated.identifier


def test_close_concentrations_have_distinct_exact_baseline_identifiers():
    first = DirichletMultinomialConfig(concentration=1.0000001)
    second = DirichletMultinomialConfig(concentration=1.0000002)

    assert first.identifier != second.identifier
    assert first.primary_baseline != second.primary_baseline
    assert first.primary_baseline == cumulative_frequency_baseline(
        alpha=first.concentration
    ).identifier
    assert second.primary_baseline == cumulative_frequency_baseline(
        alpha=second.concentration
    ).identifier


def test_non_uniform_prior_names_fixed_uniform_comparator_without_equivalence_claim():
    config = DirichletMultinomialConfig(
        concentration=2.0,
        prior_mean=_non_uniform_prior(),
    )
    fixed_uniform = cumulative_frequency_baseline(alpha=2.0)

    assert fixed_uniform.identifier in config.primary_baseline
    assert "fixed uniform-prior comparator" in config.primary_baseline
    assert "not equivalent" in config.primary_baseline
    assert config.primary_baseline != fixed_uniform.identifier


@pytest.mark.parametrize("invalid", [0, -1, math.nan, math.inf, -math.inf])
def test_configuration_rejects_non_positive_or_non_finite_concentration(invalid):
    with pytest.raises(ValueError, match="concentration"):
        DirichletMultinomialConfig(concentration=invalid)


def test_configuration_rejects_subnormal_concentration_when_prior_parameters_underflow():
    with pytest.raises(
        ValueError,
        match=r"concentration.*prior parameters.*finite and strictly positive",
    ):
        DirichletMultinomialConfig(concentration=5e-324)


def test_configuration_rejects_concentration_with_non_finite_total_mass():
    with pytest.raises(ValueError, match=r"concentration.*total mass.*finite"):
        DirichletMultinomialConfig(concentration=1e308)


def test_extreme_finite_concentration_fits_with_finite_stable_summaries():
    concentration = 1e307
    history = _draws(2)
    config = DirichletMultinomialConfig(concentration=concentration)

    state = _fit(DirichletMultinomialModel(config), history)

    total_mass = math.fsum(state.posterior_parameters)
    theoretical_mass = 5 * (len(history) + concentration)
    expected_variances = tuple(
        25
        * (parameter / total_mass)
        * (1 - parameter / total_mass)
        / (total_mass + 1)
        for parameter in state.posterior_parameters
    )
    assert all(math.isfinite(value) for value in state.raw_probabilities)
    assert all(math.isfinite(value) for value in state.probabilities)
    assert all(math.isfinite(value) for value in state.posterior_variances)
    assert math.fsum(state.prior_parameters) == 5 * concentration
    assert total_mass == theoretical_mass
    assert state.posterior_variances == expected_variances


def test_configuration_rejects_positive_prior_component_that_underflows_after_scaling():
    prior_mean = (5e-324,) + (5 / 48,) * 48

    with pytest.raises(
        ValueError,
        match=r"concentration.*prior parameters.*finite and strictly positive",
    ):
        DirichletMultinomialConfig(concentration=0.5, prior_mean=prior_mean)


@pytest.mark.parametrize("invalid", [True, False, "1", None, 1 + 0j])
def test_configuration_rejects_wrong_concentration_types(invalid):
    with pytest.raises(TypeError, match="concentration"):
        DirichletMultinomialConfig(concentration=invalid)


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        ((5 / 49,) * 48, "49"),
        ((5 / 49,) * 48 + (math.nan,), "finite"),
        ((5 / 49,) * 48 + (-0.1,), r"\[0, 1\]"),
        ((5 / 49,) * 48 + (1.1,), r"\[0, 1\]"),
        ((0.0,) + (5 / 48,) * 48, "strictly positive"),
        ((0.0,) * 49, "sum to 5"),
        ((5 / 49,) * 48 + (True,), "real numbers"),
        ((5 / 49,) * 48 + ("x",), "real numbers"),
    ],
)
def test_configuration_rejects_invalid_prior_mean(prior, message):
    with pytest.raises((TypeError, ValueError), match=message):
        DirichletMultinomialConfig(concentration=1.0, prior_mean=prior)


def test_prior_sum_tolerance_is_normalized_to_exact_draw_mass():
    prior = [5 / 49] * 49
    prior[-1] += 1e-6

    config = DirichletMultinomialConfig(
        concentration=1.0, prior_mean=tuple(prior)
    )
    repeated = DirichletMultinomialConfig(
        concentration=1.0, prior_mean=tuple(prior)
    )

    assert config.prior_mean != tuple(prior)
    assert config.prior_mean == repeated.prior_mean
    assert math.fsum(config.prior_mean) == 5.0
    assert all(0.0 < value <= 1.0 for value in config.prior_mean)
    assert DirichletMultinomialConfig().prior_mean is UNIFORM_PRIOR_MEAN
    prior[-1] += 1e-3
    with pytest.raises(ValueError, match="sum to 5"):
        DirichletMultinomialConfig(
            concentration=1.0, prior_mean=tuple(prior)
        )


@pytest.mark.parametrize("invalid", [0, -1, True, False, 1.5, "2", None])
def test_configuration_rejects_invalid_minimum_history(invalid):
    with pytest.raises((TypeError, ValueError), match="minimum_history"):
        DirichletMultinomialConfig(concentration=1.0, minimum_history=invalid)


def test_fit_rejects_empty_or_short_history_and_predict_rejects_bad_contract_inputs():
    model = DirichletMultinomialModel(
        DirichletMultinomialConfig(concentration=1.0, minimum_history=2)
    )
    with pytest.raises(ValueError, match="at least 2"):
        _fit(model, ())
    with pytest.raises(ValueError, match="at least 2"):
        _fit(model, _draws(1))
    with pytest.raises(TypeError, match="immutable tuple"):
        model.fit(list(_draws(2)), None, rng=np.random.default_rng(1))

    state = _fit(model, _draws(2))
    with pytest.raises(TypeError, match="state"):
        model.predict(_draws(2), object(), None, rng=np.random.default_rng(1))
    with pytest.raises(TypeError, match="rng"):
        model.predict(_draws(2), state, None, rng=object())


def test_state_rejects_never_fitted_sample_size_zero():
    concentration = 1.0
    prior = (5 / 49,) * 49
    total_mass = 5 * concentration
    variances = tuple(
        25
        * (parameter / total_mass)
        * (1 - parameter / total_mass)
        / (total_mass + 1)
        for parameter in prior
    )

    with pytest.raises(ValueError, match="sample_size must be a positive integer"):
        DirichletMultinomialState(
            sample_size=0,
            concentration=concentration,
            counts=(0,) * 49,
            prior_parameters=prior,
            posterior_parameters=prior,
            raw_probabilities=prior,
            probabilities=prior,
            posterior_variances=variances,
            projection_applied=False,
        )


def test_predict_rejects_state_fitted_below_configured_minimum_history():
    short_model = DirichletMultinomialModel(
        DirichletMultinomialConfig(concentration=2.0, minimum_history=1)
    )
    strict_model = DirichletMultinomialModel(
        DirichletMultinomialConfig(concentration=2.0, minimum_history=2)
    )
    short_state = _fit(short_model, _draws(1))

    with pytest.raises(ValueError, match="state sample_size.*minimum_history"):
        _predict(strict_model, _draws(2), short_state)


@pytest.mark.parametrize(
    ("prior", "message"),
    [
        ((0.0,) + (5 / 48,) * 48, "strictly positive"),
        ((1.1,) + (3.9 / 48,) * 48, "at most state concentration"),
    ],
)
def test_state_rejects_improper_dirichlet_prior_parameters(prior, message):
    with pytest.raises(ValueError, match=message):
        _state_with_prior(prior)


def test_state_rejects_concentration_with_non_finite_dirichlet_total_mass():
    concentration = 1e308
    counts = (1,) * 5 + (0,) * 44
    prior = (concentration,) * 49

    with pytest.raises(ValueError, match=r"Dirichlet total mass.*finite"):
        DirichletMultinomialState(
            sample_size=1,
            concentration=concentration,
            counts=counts,
            prior_parameters=prior,
            posterior_parameters=prior,
            raw_probabilities=(1.0,) * 49,
            probabilities=(5 / 49,) * 49,
            posterior_variances=(0.0,) * 49,
            projection_applied=True,
        )


def test_state_rejects_forged_internal_inconsistencies_and_predict_revalidates():
    model = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=2.0))
    history = _draws(3)
    state = _fit(model, history)

    invalid_replacements = (
        {"sample_size": -1},
        {"counts": (0,) * 49},
        {"counts": (True,) + state.counts[1:]},
        {"prior_parameters": (0.0,) * 49},
        {"posterior_parameters": state.prior_parameters},
        {"raw_probabilities": (5 / 49,) * 49},
        {"probabilities": (0.0,) * 49},
        {"posterior_variances": (-1.0,) + state.posterior_variances[1:]},
        {"projection_applied": "no"},
    )
    for values in invalid_replacements:
        with pytest.raises((TypeError, ValueError)):
            replace(state, **values)

    forged = deepcopy(state)
    object.__setattr__(forged, "probabilities", (0.0,) * 49)
    with pytest.raises(ValueError, match="state probabilities"):
        _predict(model, history, forged)


def test_extreme_concentration_rejects_small_but_material_forged_parameter_shifts():
    concentration = 1e307
    model = DirichletMultinomialModel(
        DirichletMultinomialConfig(concentration=concentration)
    )
    history = _draws(1)
    state = _fit(model, history)
    delta = 5e293

    forged_posterior = list(state.posterior_parameters)
    forged_posterior[0] += delta
    forged_posterior[1] -= delta
    with pytest.raises(ValueError, match="posterior_parameters"):
        replace(state, posterior_parameters=tuple(forged_posterior))

    forged_prior = list(state.prior_parameters)
    forged_prior[0] += delta
    forged_prior[1] -= delta
    internally_consistent_state = _state_with_prior(tuple(forged_prior))
    with pytest.raises(ValueError, match="does not belong"):
        _predict(model, history, internally_consistent_state)


def test_state_is_frozen_and_slotted():
    state = _fit(
        DirichletMultinomialModel(DirichletMultinomialConfig(concentration=1.0)),
        _draws(2),
    )
    with pytest.raises(FrozenInstanceError):
        state.sample_size = 3
    assert not hasattr(state, "__dict__")


def test_bounded_simplex_projection_is_deterministic_bounded_and_not_independent_clipping():
    values = (5.0,) + (0.0,) * 48

    first = project_bounded_simplex(values)
    second = project_bounded_simplex(values)

    assert first == second
    assert first[0] == 1.0
    assert first[1:] == pytest.approx((1 / 12,) * 48, abs=1e-14)
    assert sum(first) == pytest.approx(5.0, abs=1e-12)
    assert all(0.0 <= value <= 1.0 for value in first)
    assert sum(min(1.0, max(0.0, value)) for value in values) != pytest.approx(5.0)


def test_bounded_simplex_projection_is_stable_for_equal_extreme_finite_values():
    values = (1e307,) * 49

    first = project_bounded_simplex(values)
    second = project_bounded_simplex(values)

    assert first == second
    assert first == pytest.approx((5 / 49,) * 49, abs=1e-15)
    assert math.fsum(first) == pytest.approx(5.0, abs=1e-12)
    assert all(0.0 <= value <= 1.0 for value in first)


def test_bounded_simplex_rejects_finite_range_with_non_finite_translation():
    values = (1e308, -1e308) + (0.0,) * 47

    with pytest.raises(ValueError, match="translation.*finite"):
        project_bounded_simplex(values)

    documentation = inspect.getdoc(project_bounded_simplex) or ""
    assert "translation" in documentation
    assert "finite" in documentation


def test_valid_state_has_structurally_bounded_raw_mean_without_projection():
    state = _fit(
        DirichletMultinomialModel(
            DirichletMultinomialConfig(
                concentration=2.0,
                prior_mean=_non_uniform_prior(),
            )
        ),
        _draws(3),
    )

    assert all(0.0 <= probability <= 1.0 for probability in state.raw_probabilities)
    assert state.probabilities == state.raw_probabilities
    assert state.projection_applied is False


def test_posterior_variance_documentation_limits_scientific_interpretation():
    documentation = " ".join(
        " ".join(
            (
                inspect.getdoc(DirichletMultinomialModel) or "",
                inspect.getdoc(DirichletMultinomialState) or "",
            )
        )
        .lower()
        .split()
    )

    assert "working dirichlet-multinomial" in documentation
    assert "categorical inclusions" in documentation
    assert "within-draw dependence" in documentation
    assert "without replacement" in documentation
    assert "true joint law" in documentation


@pytest.mark.parametrize(
    "values",
    [
        (0.0,) * 48,
        (0.0,) * 48 + (math.nan,),
        (0.0,) * 48 + (math.inf,),
        (0.0,) * 48 + (True,),
        (0.0,) * 48 + ("0",),
    ],
)
def test_bounded_simplex_projection_rejects_invalid_inputs(values):
    with pytest.raises((TypeError, ValueError)):
        project_bounded_simplex(values)


@pytest.mark.parametrize(
    ("policy", "frequency", "expected_retrained", "expected_fitted_indices"),
    [
        (RetrainPolicy.EACH_DRAW, None, [True, True, True, True], [103, 104, 105, 106]),
        (RetrainPolicy.FIXED_FREQUENCY, 2, [True, False, True, False], [103, 103, 105, 105]),
        (RetrainPolicy.NONE_DURING_WINDOW, None, [True, False, False, False], [103] * 4),
    ],
)
def test_real_walk_forward_supports_every_retrain_policy_with_correct_audit(
    policy, frequency, expected_retrained, expected_fitted_indices
):
    draws = _draws(8)
    model = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=2.0))

    records = _run(draws, model, policy=policy, frequency=frequency)

    assert [record.retrained for record in records] == expected_retrained
    assert [record.fitted_through_original_index for record in records] == expected_fitted_indices
    assert [record.target_original_index for record in records] == [104, 105, 106, 107]
    for record in records:
        assert record.target_original_index not in record.history_original_indices
        assert max(record.history_original_indices) < record.target_original_index
        assert max(record.history_dates) < record.target_date
        assert record.fitted_through_date < record.target_date
    if policy is RetrainPolicy.NONE_DURING_WINDOW:
        assert len({record.probabilities for record in records}) == 1


def test_predict_uses_only_frozen_state_and_never_relearns_from_history():
    model = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=2.0))
    fitted_history = _draws(4)
    state = _fit(model, fitted_history)
    changed_history = _history_with_one_always_present(10)

    original = _predict(model, fitted_history, state)
    changed = _predict(model, changed_history, state)

    assert changed == original == state.probabilities
    assert state.sample_size == len(fitted_history)
    assert "target" not in inspect.signature(model.fit).parameters
    assert "target" not in inspect.signature(model.predict).parameters


def test_future_changes_cannot_modify_earlier_walk_forward_predictions():
    draws = _draws(8)
    changed = (*draws[:6], replace(draws[6], numbers=(45, 46, 47, 48, 49)), draws[7])
    model = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=2.0))

    original_records = _run(draws, model)
    changed_records = _run(changed, model)

    assert [record.probabilities for record in original_records[:3]] == [
        record.probabilities for record in changed_records[:3]
    ]
    assert original_records[2].target_original_index == draws[6].original_index


def test_predictions_are_seed_independent_and_injected_rng_is_not_consumed():
    draws = _draws(8)
    model = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=2.0))

    seed_one = _run(draws, model, seed=1)
    seed_two = _run(draws, model, seed=999)
    assert [record.probabilities for record in seed_one] == [
        record.probabilities for record in seed_two
    ]

    history = _draws(4)
    fit_rng = np.random.default_rng(11)
    fit_rng_before = deepcopy(fit_rng.bit_generator.state)
    state = model.fit(history, None, rng=fit_rng)
    assert fit_rng.bit_generator.state == fit_rng_before

    predict_rng = np.random.default_rng(12)
    predict_rng_before = deepcopy(predict_rng.bit_generator.state)
    assert model.predict(history, state, None, rng=predict_rng) == state.probabilities
    assert predict_rng.bit_generator.state == predict_rng_before


def test_model_is_deterministic_does_not_use_global_random_and_does_not_mutate_history():
    history = _draws(5)
    before = deepcopy(history)
    global_state = deepcopy(np.random.get_state())
    model = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=3.0))

    first = _fit(model, history)
    second = _fit(model, history)

    assert first == second
    assert history == before
    assert all(
        np.array_equal(left, right) if isinstance(left, np.ndarray) else left == right
        for left, right in zip(np.random.get_state(), global_state, strict=True)
    )


def test_model_performs_no_filesystem_or_database_io(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dirichlet-multinomial model attempted I/O")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)

    history = _draws(4)
    model = DirichletMultinomialModel(DirichletMultinomialConfig(concentration=1.0))
    state = _fit(model, history)
    assert _predict(model, history, state)
