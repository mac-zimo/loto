from __future__ import annotations

from dataclasses import fields
import gc
from io import StringIO
import json
import os

import pytest

import loto.evaluation.archive_capacity_preflight as capacity
import loto.evaluation.archive_experiment as archive


def _reachable_types(root):
    seen = set()
    pending = [root]
    types = set()
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        types.add(type(item))
        if isinstance(item, (str, bytes, int, float, bool, type(None), type)):
            continue
        pending.extend(gc.get_referents(item))
    return types


def test_synthetic_capacity_preflight_runs_ten_candidates_without_real_io(monkeypatch):
    forbidden = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("real I/O is forbidden in the synthetic capacity preflight")
    )
    monkeypatch.setattr(archive, "load_archive_draws", forbidden)
    monkeypatch.setattr(archive, "build_data_manifest", forbidden)
    monkeypatch.setattr(archive, "run_experiment", forbidden)
    calls = []
    real_evaluate_one = archive._evaluate_one_candidate

    def recording_evaluate_one(observations, protocol, candidate):
        calls.append((len(observations), candidate.identifier))
        return real_evaluate_one(observations, protocol, candidate)

    monkeypatch.setattr(archive, "_evaluate_one_candidate", recording_evaluate_one)

    logs = StringIO()
    report = capacity.run_synthetic_capacity_preflight(
        draw_count=128, log_stream=logs
    )

    assert report.draw_count == 128
    assert report.candidate_count == 10
    assert len(report.candidates) == 10
    assert calls == [
        (128, candidate.identifier) for candidate in archive.preregistered_candidates()
    ]
    assert all(result.prediction_count > 0 for result in report.candidates)
    assert report.total_prediction_count == sum(
        result.prediction_count for result in report.candidates
    )
    assert report.python_peak_mib is None
    assert set(report.limits) == {
        "duration_seconds",
        "python_peak_mib",
        "linux_ru_maxrss_mib",
    }
    events = [json.loads(line) for line in logs.getvalue().splitlines()]
    assert [event["event"] for event in events] == [
        item for _ in range(10) for item in ("candidate_start", "candidate_end")
    ]
    assert all(
        set(event) == {
            "candidate_id",
            "duration_seconds",
            "event",
            "linux_ru_maxrss_mib",
            "prediction_count",
        }
        for event in events
    )


def test_candidate_filter_uses_only_fixed_candidates_and_preserves_fixed_order(monkeypatch):
    candidates = archive.preregistered_candidates()
    selected = (candidates[-1].identifier, candidates[0].identifier)
    calls = []
    real_evaluate_one = archive._evaluate_one_candidate

    def recording_evaluate_one(observations, protocol, candidate):
        calls.append(candidate.identifier)
        return real_evaluate_one(observations, protocol, candidate)

    monkeypatch.setattr(archive, "_evaluate_one_candidate", recording_evaluate_one)
    report = capacity.run_synthetic_capacity_preflight(
        128, candidate_ids=selected, log_stream=StringIO()
    )

    assert calls == [candidates[0].identifier, candidates[-1].identifier]
    assert tuple(item.candidate_id for item in report.candidates) == tuple(calls)
    with pytest.raises(ValueError, match="unknown pre-registered"):
        capacity.run_synthetic_capacity_preflight(
            128, candidate_ids=("not-fixed",), log_stream=StringIO()
        )


def test_capacity_decision_applies_python_limit_only_when_tracing_is_enabled():
    assert capacity._within_limits(1200.0, None, 1536.0)
    assert capacity._within_limits(1200.0, 512.0, 1536.0)
    assert not capacity._within_limits(1200.01, None, 1536.0)
    assert not capacity._within_limits(1200.0, None, 1536.01)
    assert not capacity._within_limits(1200.0, 512.01, 1536.0)


def test_explicit_python_memory_trace_reports_a_peak():
    report = capacity.run_synthetic_capacity_preflight(
        128,
        candidate_ids=(archive.preregistered_candidates()[0].identifier,),
        trace_python_memory=True,
        log_stream=StringIO(),
    )

    assert report.python_peak_mib is not None
    assert report.python_peak_mib >= 0.0


def test_archive_history_elision_preserves_summary_digests_and_metrics():
    draws = capacity.synthetic_capacity_draws(128)
    protocol = archive.build_protocol(draws)
    candidate = archive.preregistered_candidates()[0]

    optimized = archive._evaluate_one_candidate(draws, protocol, candidate)
    reference = archive._evaluate_one_candidate(
        draws, protocol, candidate, retain_record_history=True
    )

    assert optimized == reference


def test_default_synthetic_chronology_has_2811_observations_and_twelve_target_groups():
    draws = capacity.synthetic_capacity_draws()
    protocol = archive.build_protocol(draws)

    assert len(draws) == 2811
    assert len(protocol.windows) == 12
    assert {draw.draw_date.year for draw in archive.target_draws(draws, protocol.windows)} == set(
        range(2014, 2026)
    )


def test_final_2811_draw_result_is_structurally_bounded_to_candidate_summaries(
    monkeypatch,
):
    draws = capacity.synthetic_capacity_draws()
    protocol = archive.build_protocol(draws)
    targets = archive.target_draws(draws, protocol.windows)
    full_history_dates = tuple(draw.draw_date for draw in draws)
    full_history_indices = tuple(draw.original_index for draw in draws)
    records = tuple(
        archive.PredictionRecord(
            target_date=target.draw_date,
            target_original_index=target.original_index,
            target_numbers=target.numbers,
            probabilities=(5.0 / 49.0,) * 49,
            history_dates=full_history_dates,
            history_original_indices=full_history_indices,
            fitted_through_date=draws[0].draw_date,
            fitted_through_original_index=draws[0].original_index,
            window_index=next(
                index
                for index, window in enumerate(protocol.windows)
                if window.test_start <= target.original_index < window.test_end
            ),
            retrained=True,
        )
        for target in targets
    )
    calls = []

    def representative_walk_forward(
        observations, windows, callbacks, config, *, retain_history=True
    ):
        calls.append(len(observations))
        assert not retain_history
        return records

    monkeypatch.setattr(archive, "run_walk_forward", representative_walk_forward)

    run = archive.evaluate_candidates(draws, protocol)

    assert calls == [2811] * 10
    assert len(run.prediction_summaries) == 10
    assert all(
        summary.prediction_count == len(targets)
        for summary in run.prediction_summaries.values()
    )
    assert archive.PredictionRecord not in _reachable_types(run)
    assert not hasattr(run, "predictions")


@pytest.mark.slow
@pytest.mark.integration_capacity
@pytest.mark.skipif(
    os.environ.get("LOTO_RUN_CAPACITY") != "1",
    reason="explicit opt-in required for the 2811-draw capacity preflight",
)
def test_full_synthetic_capacity_preflight_respects_preregistered_limits():
    report = capacity.run_synthetic_capacity_preflight()

    assert report.within_limits
    assert report.draw_count == 2811
    assert report.candidate_count == 10
    assert tuple(field.name for field in fields(report))