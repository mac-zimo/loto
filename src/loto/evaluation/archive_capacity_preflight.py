"""Non-scientific, synthetic capacity preflight for the archive experiment.

This entry point never reads FDJ archives, SQLite, the experiment registry, the
network, or production artifacts.  It exercises the ten fixed candidates with
a deterministic chronology that has an equivalent expanding training scale and
twelve annual target groups.  Its measurements are operational capacity data,
not scientific evidence.  Like the official runner, it is Linux/POSIX-only;
the official publication path additionally requires ``fcntl``, ``O_DIRECTORY``,
``O_NOFOLLOW``, and filesystem support for ``renameat2(RENAME_NOREPLACE)`` and
fails closed without an unsafe fallback when any capability is unavailable.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
import gc
import json
import resource
import sys
from time import perf_counter
import tracemalloc
from types import MappingProxyType
from typing import Mapping, Sequence, TextIO

from loto.evaluation import archive_experiment as archive
from loto.evaluation.walk_forward import DrawObservation

DEFAULT_DRAW_COUNT = 2811
MAX_DURATION_SECONDS = 20 * 60
MAX_PYTHON_PEAK_MIB = 512.0
MAX_LINUX_RU_MAXRSS_MIB = 1536.0
_MIB = 1024 * 1024
_SYNTHETIC_START = date(2008, 10, 6)
_TRAINING_END = date(2013, 12, 31)
_EVALUATION_START = date(2014, 1, 1)
_SYNTHETIC_END = date(2026, 9, 21)


@dataclass(frozen=True, slots=True)
class CandidateCapacity:
    candidate_id: str
    duration_seconds: float
    prediction_count: int


@dataclass(frozen=True, slots=True)
class CapacityPreflightReport:
    non_scientific: bool
    draw_count: int
    candidate_count: int
    initial_train_count: int
    target_count_per_candidate: int
    total_prediction_count: int
    total_duration_seconds: float
    python_peak_mib: float | None
    linux_ru_maxrss_before_mib: float
    linux_ru_maxrss_mib: float
    linux_ru_maxrss_delta_mib: float
    candidates: tuple[CandidateCapacity, ...]
    limits: Mapping[str, float]
    within_limits: bool


def _spread_dates(start: date, end: date, count: int) -> tuple[date, ...]:
    if count < 1:
        return ()
    available_days = (end - start).days
    if count > available_days + 1:
        raise ValueError("draw_count exceeds the unique synthetic date capacity")
    if count == 1:
        return (start,)
    return tuple(
        start + timedelta(days=index * available_days // (count - 1))
        for index in range(count)
    )


def synthetic_capacity_draws(
    draw_count: int = DEFAULT_DRAW_COUNT,
) -> tuple[DrawObservation, ...]:
    """Generate deterministic, archive-free observations for capacity only."""

    if not isinstance(draw_count, int) or isinstance(draw_count, bool):
        raise TypeError("draw_count must be an integer")
    if draw_count < 128:
        raise ValueError("draw_count must be at least 128 for fixed-model warmup")
    total_span = (_SYNTHETIC_END - _SYNTHETIC_START).days
    training_span = (_EVALUATION_START - _SYNTHETIC_START).days
    proportional_training = round(draw_count * training_span / total_span)
    training_count = min(draw_count - 13, max(100, proportional_training))
    target_side_count = draw_count - training_count
    dates = (
        *_spread_dates(_SYNTHETIC_START, _TRAINING_END, training_count),
        *_spread_dates(_EVALUATION_START, _SYNTHETIC_END, target_side_count),
    )
    return tuple(
        DrawObservation(
            draw_date=draw_date,
            original_index=index,
            numbers=tuple(  # type: ignore[arg-type]
                ((index + offset * 9) % 49) + 1 for offset in range(5)
            ),
        )
        for index, draw_date in enumerate(dates)
    )


def _linux_ru_maxrss_mib() -> float:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("capacity preflight requires Linux ru_maxrss semantics")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _selected_candidates(candidate_ids: Sequence[str] | None) -> tuple[archive.Candidate, ...]:
    candidates = archive.preregistered_candidates()
    archive._verify_preregistered_candidates(candidates)
    if candidate_ids is None:
        return candidates
    requested = tuple(candidate_ids)
    if not requested:
        raise ValueError("candidate_ids must contain at least one candidate ID")
    if len(set(requested)) != len(requested):
        raise ValueError("candidate IDs must not be repeated")
    by_id = {candidate.identifier: candidate for candidate in candidates}
    unknown = tuple(identifier for identifier in requested if identifier not in by_id)
    if unknown:
        raise ValueError(f"unknown pre-registered candidate ID: {unknown[0]}")
    requested_set = set(requested)
    return tuple(
        candidate for candidate in candidates if candidate.identifier in requested_set
    )


def _within_limits(
    duration_seconds: float,
    python_peak_mib: float | None,
    linux_ru_maxrss_mib: float,
) -> bool:
    return (
        duration_seconds <= MAX_DURATION_SECONDS
        and linux_ru_maxrss_mib <= MAX_LINUX_RU_MAXRSS_MIB
        and (
            python_peak_mib is None
            or python_peak_mib <= MAX_PYTHON_PEAK_MIB
        )
    )


def _log_candidate(
    event: str,
    candidate_id: str,
    *,
    duration_seconds: float,
    prediction_count: int,
    stream: TextIO,
) -> None:
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "duration_seconds": duration_seconds,
                "event": event,
                "linux_ru_maxrss_mib": _linux_ru_maxrss_mib(),
                "prediction_count": prediction_count,
            },
            sort_keys=True,
        ),
        file=stream,
        flush=True,
    )


def run_synthetic_capacity_preflight(
    draw_count: int = DEFAULT_DRAW_COUNT,
    *,
    candidate_ids: Sequence[str] | None = None,
    trace_python_memory: bool = False,
    log_stream: TextIO | None = None,
) -> CapacityPreflightReport:
    """Run fixed candidates on deterministic non-scientific observations.

    No archive loader, manifest, SQLite, registry, network, or artifact function
    is called.  The pre-registered capacity ceilings are 20 minutes total,
    512 MiB traced Python allocation peak when explicitly measured, and 1.5 GiB
    Linux process ``ru_maxrss``. Tracing is off by default because its overhead
    makes the duration unrepresentative of the official experiment.
    """

    started = perf_counter()
    rss_before = _linux_ru_maxrss_mib()
    stream = sys.stderr if log_stream is None else log_stream
    candidates = _selected_candidates(candidate_ids)
    owned_tracer = trace_python_memory and not tracemalloc.is_tracing()
    if trace_python_memory:
        if owned_tracer:
            tracemalloc.start()
        tracemalloc.reset_peak()
    peak_bytes: int | None = None
    try:
        draws = synthetic_capacity_draws(draw_count)
        protocol = archive.build_protocol(draws)
        results = []
        for candidate in candidates:
            _log_candidate(
                "candidate_start",
                candidate.identifier,
                duration_seconds=0.0,
                prediction_count=0,
                stream=stream,
            )
            candidate_started = perf_counter()
            summary, metric_rows = archive._evaluate_one_candidate(
                draws, protocol, candidate
            )
            duration = perf_counter() - candidate_started
            results.append(
                CandidateCapacity(
                    candidate_id=candidate.identifier,
                    duration_seconds=duration,
                    prediction_count=summary.prediction_count,
                )
            )
            _log_candidate(
                "candidate_end",
                candidate.identifier,
                duration_seconds=duration,
                prediction_count=summary.prediction_count,
                stream=stream,
            )
            del summary, metric_rows
            gc.collect()
        if trace_python_memory:
            _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        if owned_tracer:
            tracemalloc.stop()
    total_duration = perf_counter() - started
    rss_peak = _linux_ru_maxrss_mib()
    python_peak_mib = None if peak_bytes is None else peak_bytes / _MIB
    limits = MappingProxyType(
        {
            "duration_seconds": float(MAX_DURATION_SECONDS),
            "python_peak_mib": MAX_PYTHON_PEAK_MIB,
            "linux_ru_maxrss_mib": MAX_LINUX_RU_MAXRSS_MIB,
        }
    )
    result_tuple = tuple(results)
    target_count = result_tuple[0].prediction_count
    if any(result.prediction_count != target_count for result in result_tuple):
        raise AssertionError("capacity candidates produced different target counts")
    return CapacityPreflightReport(
        non_scientific=True,
        draw_count=len(draws),
        candidate_count=len(result_tuple),
        initial_train_count=protocol.walk_forward.initial_train_size,
        target_count_per_candidate=target_count,
        total_prediction_count=sum(item.prediction_count for item in result_tuple),
        total_duration_seconds=total_duration,
        python_peak_mib=python_peak_mib,
        linux_ru_maxrss_before_mib=rss_before,
        linux_ru_maxrss_mib=rss_peak,
        linux_ru_maxrss_delta_mib=max(0.0, rss_peak - rss_before),
        candidates=result_tuple,
        limits=limits,
        within_limits=_within_limits(total_duration, python_peak_mib, rss_peak),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the non-scientific synthetic capacity preflight; no FDJ archive, "
            "registry, SQLite database, network, or artifact is accessed."
        )
    )
    parser.add_argument("--draw-count", type=int, default=DEFAULT_DRAW_COUNT)
    parser.add_argument(
        "--candidate",
        action="append",
        dest="candidate_ids",
        metavar="ID",
        help=(
            "run only this exact pre-registered candidate ID; repeat to select "
            "multiple fixed candidates"
        ),
    )
    parser.add_argument(
        "--trace-python-memory",
        action="store_true",
        help=(
            "enable tracemalloc diagnostics (substantial timing overhead; the "
            "2,811-draw duration decision must use the default untraced mode)"
        ),
    )
    arguments = parser.parse_args(argv)
    report = run_synthetic_capacity_preflight(
        arguments.draw_count,
        candidate_ids=arguments.candidate_ids,
        trace_python_memory=arguments.trace_python_memory,
    )
    print(
        json.dumps(
            asdict(replace(report, limits=dict(report.limits))),
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.within_limits else 1


if __name__ == "__main__":  # pragma: no cover - explicit capacity command
    raise SystemExit(main())


__all__ = [
    "CapacityPreflightReport",
    "CandidateCapacity",
    "DEFAULT_DRAW_COUNT",
    "MAX_DURATION_SECONDS",
    "MAX_LINUX_RU_MAXRSS_MIB",
    "MAX_PYTHON_PEAK_MIB",
    "main",
    "run_synthetic_capacity_preflight",
    "synthetic_capacity_draws",
]
