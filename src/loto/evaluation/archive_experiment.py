"""Pre-registered Task 2.3a archive walk-forward experiment.

The official entry point is deliberately guarded by the append-only experiment
registry and Git cleanliness checks.  Importing this module never reads the FDJ
archives, SQLite, the registry, or existing artifacts.

The official runner is Linux/POSIX-only: it requires ``fcntl``, ``O_DIRECTORY``,
``O_NOFOLLOW``, and filesystem support for ``renameat2(RENAME_NOREPLACE)``.  A
missing capability fails closed before success; there is no unsafe fallback.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from loto._experiment_git import PROJECT_ROOT
from loto.config import CSV_FILES
from loto.data_source import inspect_source, preflight_source_set
from loto.evaluation._archive_artifact_publication import (
    PublishedArtifactError,
    PublicationReceipt,
    ensure_destination_available,
    lock_parent_directory,
    publish_artifact_directory,
    verify_publication,
)
from loto.evaluation.baselines import (
    constrained_random_baseline,
    cumulative_frequency_baseline,
    rolling_frequency_baseline,
    uniform_baseline,
)
from loto.evaluation.models.dirichlet_multinomial import (
    DirichletMultinomialConfig,
    DirichletMultinomialModel,
)
from loto.evaluation.models.discrete_time_hazard import (
    DiscreteTimeHazardConfig,
    DiscreteTimeHazardModel,
)
from loto.evaluation.models.logistic_regression import (
    LogisticRegressionConfig,
    LogisticRegressionModel,
)
from loto.evaluation.walk_forward import (
    DrawObservation,
    EvaluationMetrics,
    EvaluationWindow,
    PredictionRecord,
    RetrainPolicy,
    WalkForwardCallbacks,
    WalkForwardConfig,
    annual_windows,
    evaluate_predictions,
    run_walk_forward,
)
from loto.experiments import (
    DEFAULT_REGISTRY_PATH,
    ExperimentConfig,
    ExperimentOutcome,
    build_data_manifest,
    run_experiment,
)

EXPERIMENT_ID = "EXP-2026-09-25-TASK-2.3A-ARCHIVE-WF-001"
SEED = 20260925
ARCHIVE_FILENAMES = tuple(CSV_FILES)
ARTIFACT_DIRECTORY = Path("artifacts/task-2.3a-archive-walk-forward")
ARTIFACT_FILENAMES = ("preregistration.json", "metrics.csv", "result.json")
METRIC_NAMES = (
    "log_loss",
    "brier",
    "mean_matches",
    "calibration_error",
    "mean_true_number_rank",
    "regret_vs_uniform",
)
PROBABILISTIC_METRIC_NAMES = (
    "log_loss",
    "brier",
    "calibration_error",
    "mean_true_number_rank",
    "regret_vs_uniform",
)


@dataclass(frozen=True, slots=True)
class DatasetContract:
    draw_count: int
    first_date: date
    last_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.draw_count, int) or isinstance(self.draw_count, bool):
            raise TypeError("draw_count must be an integer")
        if self.draw_count < 1:
            raise ValueError("draw_count must be positive")
        if type(self.first_date) is not date or type(self.last_date) is not date:
            raise TypeError("dataset contract dates must be datetime.date values")
        if self.first_date > self.last_date:
            raise ValueError("dataset contract dates are reversed")


OFFICIAL_DATASET_CONTRACT = DatasetContract(
    draw_count=2811,
    first_date=date(2008, 10, 6),
    last_date=date(2026, 9, 21),
)


@dataclass(frozen=True, slots=True)
class Candidate:
    label: str
    identifier: str
    kind: str
    callbacks: WalkForwardCallbacks
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.kind not in {"baseline", "family"}:
            raise ValueError("candidate kind must be baseline or family")
        if not self.label or not self.identifier:
            raise ValueError("candidate label and identifier must be non-empty")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def _configured_parameters(value: object) -> dict[str, Any]:
    """Return only constructor parameters, never duplicated derived identifiers."""

    if not is_dataclass(value):
        return {}
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.init and field.name != "identifier"
    }


def _candidate(label: str, kind: str, value: object) -> Candidate:
    config = getattr(value, "config", value)
    identifier = getattr(config, "identifier")
    callbacks = getattr(value, "callbacks")
    return Candidate(
        label=label,
        identifier=identifier,
        kind=kind,
        callbacks=callbacks,
        parameters=_configured_parameters(config),
    )


def preregistered_candidates() -> tuple[Candidate, ...]:
    """Construct the ten fixed candidates from their actual model configurations."""

    uniform = uniform_baseline()
    cumulative = cumulative_frequency_baseline(alpha=1.0)
    rolling = tuple(
        rolling_frequency_baseline(window_size=window, alpha=1.0)
        for window in (5, 10, 25, 50)
    )
    constrained = constrained_random_baseline()
    dirichlet = DirichletMultinomialModel(
        DirichletMultinomialConfig(concentration=1.0, minimum_history=1)
    )
    logistic = LogisticRegressionModel(LogisticRegressionConfig())
    hazard = DiscreteTimeHazardModel(DiscreteTimeHazardConfig())
    return (
        _candidate("uniform", "baseline", uniform),
        _candidate("cumulative", "baseline", cumulative),
        *(
            _candidate(f"rolling_{model.window_size}", "baseline", model)
            for model in rolling
        ),
        _candidate("constrained_random", "baseline", constrained),
        _candidate("dirichlet", "family", dirichlet),
        _candidate("logistic", "family", logistic),
        _candidate("hazard", "family", hazard),
    )


def _verify_preregistered_candidates(candidates: Sequence[Candidate]) -> None:
    identifiers = tuple(candidate.identifier for candidate in candidates)
    if identifiers != PREREGISTRATION.candidate_ids:
        raise AssertionError("candidate manifest differs from the pre-registration")


@dataclass(frozen=True, slots=True)
class Preregistration:
    experiment_id: str
    preregistration_date: date
    seed: int
    archive_filenames: tuple[str, ...]
    dataset_contract: DatasetContract
    cutoff: date
    evaluation_start_year: int
    evaluation_end_year: int
    excluded_year: int
    retrain_policy: RetrainPolicy
    max_train_size: int | None
    candidate_ids: tuple[str, ...]
    metrics: tuple[str, ...]
    hyperparameter_search: bool
    minimum_winning_periods: int


PREREGISTRATION = Preregistration(
    experiment_id=EXPERIMENT_ID,
    preregistration_date=date(2026, 9, 25),
    seed=SEED,
    archive_filenames=ARCHIVE_FILENAMES,
    dataset_contract=OFFICIAL_DATASET_CONTRACT,
    cutoff=date(2014, 1, 1),
    evaluation_start_year=2014,
    evaluation_end_year=2025,
    excluded_year=2026,
    retrain_policy=RetrainPolicy.EACH_DRAW,
    max_train_size=None,
    candidate_ids=tuple(candidate.identifier for candidate in preregistered_candidates()),
    metrics=METRIC_NAMES,
    hyperparameter_search=False,
    minimum_winning_periods=3,
)


def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        id=PREREGISTRATION.experiment_id,
        hypothesis=(
            "At least one fixed simple family improves both log-loss and Brier "
            "descriptively versus cumulative alpha=1 and exact uniform forecasts."
        ),
        split=(
            "train:<2014-01-01; test:calendar-years-2014..2025; expanding; "
            "2026-excluded"
        ),
        features=(
            "model-owned preregistered features only; no evaluation-period tuning; "
            "canonical main-draw numbers"
        ),
        model=",".join(PREREGISTRATION.candidate_ids),
        baseline=",".join(
            candidate.identifier
            for candidate in preregistered_candidates()
            if candidate.kind == "baseline"
        ),
        metric=",".join(PREREGISTRATION.metrics),
        threshold=(
            "aggregate log_loss and brier strictly below cumulative alpha=1 and "
            "uniform, plus both strictly below both comparators in at least 3 "
            "complete calendar years; descriptive screening only; significance and "
            "holdout 2026 untouched are required for promotion"
        ),
        seed=PREREGISTRATION.seed,
    )


def _validate_archive_paths(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    resolved = tuple(Path(path).resolve() for path in paths)
    if len(resolved) != len(ARCHIVE_FILENAMES):
        raise ValueError("exactly the four pre-registered archive paths are required")
    names = tuple(path.name for path in resolved)
    if len(set(resolved)) != len(resolved) or set(names) != set(ARCHIVE_FILENAMES):
        raise ValueError("data paths must be the four exact pre-registered archive files")
    return resolved


def load_archive_draws(
    paths: Sequence[str | Path],
    *,
    contract: DatasetContract = OFFICIAL_DATASET_CONTRACT,
) -> tuple[DrawObservation, ...]:
    """Validate the full archive date/count contract, but canonicalize only pre-2026 rows.

    ``original_index`` is the zero-based position after pre-2026 canonical
    validation and strict chronological sorting.  It is therefore stable for
    identical source bytes, unique, auditable, and independent of ``hash()``.
    """

    resolved = _validate_archive_paths(paths)
    inspections = [inspect_source(path) for path in resolved]
    dated_rows = []
    eligible_inspections = []
    for inspection in inspections:
        eligible_rows = []
        for row in inspection.rows:
            if row.error is not None:
                raise row.error
            assert row.values is not None
            try:
                draw_date = datetime.strptime(
                    row.values["date_de_tirage"], "%d/%m/%Y"
                ).date()
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"invalid draw date in {inspection.path.name}:data-row {row.source_row}"
                ) from error
            dated_rows.append(draw_date)
            if draw_date.year < PREREGISTRATION.excluded_year:
                eligible_rows.append(row)
        eligible_inspections.append(replace(inspection, rows=tuple(eligible_rows)))
    ordered_dates = sorted(dated_rows)
    observed = (
        len(ordered_dates),
        ordered_dates[0] if ordered_dates else None,
        ordered_dates[-1] if ordered_dates else None,
    )
    expected = (contract.draw_count, contract.first_date, contract.last_date)
    if observed != expected:
        raise ValueError(
            f"dataset contract mismatch: observed={observed!r}, expected={expected!r}"
        )
    if any(left == right for left, right in zip(ordered_dates, ordered_dates[1:])):
        raise ValueError("archive draw dates must be unique")
    prepared = preflight_source_set(
        eligible_inspections, imported_at=datetime(1970, 1, 1, tzinfo=timezone.utc)
    )
    canonical = tuple(
        draw for source in prepared for draw in source.canonical_draws
    )
    ordered = tuple(sorted(canonical, key=lambda draw: draw.draw_date))
    if any(
        left.draw_date >= right.draw_date
        for left, right in zip(ordered, ordered[1:])
    ):
        raise ValueError("canonical draw dates must be unique and strictly increasing")
    return tuple(
        DrawObservation(
            draw_date=draw.draw_date,
            original_index=index,
            numbers=draw.main_numbers,
        )
        for index, draw in enumerate(ordered)
    )


@dataclass(frozen=True, slots=True)
class EvaluationProtocol:
    walk_forward: WalkForwardConfig
    windows: tuple[EvaluationWindow, ...]
    cutoff: date
    evaluation_start_year: int
    evaluation_end_year: int
    max_train_size: int | None = None


def target_draws(
    draws: Sequence[DrawObservation], windows: Sequence[EvaluationWindow]
) -> tuple[DrawObservation, ...]:
    observations = tuple(draws)
    return tuple(
        observations[position]
        for window in windows
        for position in range(window.test_start, window.test_end)
    )


def validate_target_period(
    draws: Sequence[DrawObservation], windows: Sequence[EvaluationWindow]
) -> tuple[DrawObservation, ...]:
    targets = target_draws(draws, windows)
    if not targets:
        raise ValueError("evaluation targets must not be empty")
    if any(target.draw_date.year == PREREGISTRATION.excluded_year for target in targets):
        raise ValueError("2026 targets are forbidden by the pre-registration")
    if max(target.draw_date for target in targets) > date(2025, 12, 31):
        raise ValueError("evaluation targets after 2025-12-31 are forbidden")
    return targets


def build_protocol(
    draws: Sequence[DrawObservation],
    *,
    evaluation_start_year: int = 2014,
    evaluation_end_year: int = 2025,
) -> EvaluationProtocol:
    observations = tuple(draws)
    initial_train_size = sum(
        observation.draw_date < PREREGISTRATION.cutoff for observation in observations
    )
    if initial_train_size < 1:
        raise ValueError("no initial training observations exist before 2014-01-01")
    if any(
        observation.draw_date >= PREREGISTRATION.cutoff
        for observation in observations[:initial_train_size]
    ) or any(
        observation.draw_date < PREREGISTRATION.cutoff
        for observation in observations[initial_train_size:]
    ):
        raise ValueError("draws must be strictly chronological around the cutoff")
    windows = annual_windows(
        observations,
        initial_train_size=initial_train_size,
        evaluation_start_year=evaluation_start_year,
        evaluation_end_year=evaluation_end_year,
    )
    validate_target_period(observations, windows)
    config = WalkForwardConfig(
        initial_train_size=initial_train_size,
        retrain_policy=RetrainPolicy.EACH_DRAW,
        seed=PREREGISTRATION.seed,
    )
    return EvaluationProtocol(
        walk_forward=config,
        windows=windows,
        cutoff=PREREGISTRATION.cutoff,
        evaluation_start_year=evaluation_start_year,
        evaluation_end_year=evaluation_end_year,
        max_train_size=None,
    )


@dataclass(frozen=True, slots=True)
class MetricRow:
    candidate_id: str
    period: str
    prediction_count: int
    first_target_date: date
    last_target_date: date
    log_loss: float
    brier: float
    mean_matches: float
    calibration_error: float
    mean_true_number_rank: float
    regret_vs_uniform: float

    @classmethod
    def from_metrics(
        cls,
        candidate_id: str,
        period: str,
        records: Sequence[PredictionRecord],
        metrics: EvaluationMetrics,
    ) -> "MetricRow":
        return cls(
            candidate_id=candidate_id,
            period=period,
            prediction_count=len(records),
            first_target_date=records[0].target_date,
            last_target_date=records[-1].target_date,
            **asdict(metrics),
        )


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    candidate_id: str
    winning_period_count: int
    winning_periods: tuple[int, ...]
    aggregate_joint_improvement: bool
    screening_pass: bool
    decision: str


@dataclass(frozen=True, slots=True)
class PredictionSeriesSummary:
    """Bounded archive-specific identity of one complete prediction series."""

    candidate_id: str
    prediction_count: int
    first_target_date: date
    last_target_date: date
    targets_in_2026: int
    probability_digest: str
    selected_number_digest: str


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    prediction_summaries: Mapping[str, PredictionSeriesSummary]
    metric_rows: tuple[MetricRow, ...]
    screenings: tuple[ScreeningResult, ...]


def rows_for(run: ExperimentRun, candidate_id: str) -> tuple[MetricRow, ...]:
    return tuple(row for row in run.metric_rows if row.candidate_id == candidate_id)


def _metrics_signature(row: MetricRow) -> tuple[Any, ...]:
    return (
        row.period,
        row.prediction_count,
        row.first_target_date,
        row.last_target_date,
        *(getattr(row, metric) for metric in METRIC_NAMES),
    )


def _row_index(rows: Sequence[MetricRow]) -> dict[tuple[str, str], MetricRow]:
    result = {(row.candidate_id, row.period): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("metric rows contain duplicate candidate/period keys")
    return result


def screen_candidate(
    rows: Sequence[MetricRow],
    candidate_id: str,
    *,
    cumulative_id: str,
    uniform_id: str,
) -> ScreeningResult:
    indexed = _row_index(rows)

    def wins(period: str) -> bool:
        candidate = indexed[(candidate_id, period)]
        cumulative = indexed[(cumulative_id, period)]
        uniform = indexed[(uniform_id, period)]
        return (
            candidate.log_loss < cumulative.log_loss
            and candidate.log_loss < uniform.log_loss
            and candidate.brier < cumulative.brier
            and candidate.brier < uniform.brier
        )

    annual_periods = sorted(
        int(row.period)
        for row in rows
        if row.candidate_id == candidate_id and row.period != "aggregate"
    )
    winning_periods = tuple(year for year in annual_periods if wins(str(year)))
    aggregate_improvement = wins("aggregate")
    passed = (
        aggregate_improvement
        and len(winning_periods) >= PREREGISTRATION.minimum_winning_periods
    )
    return ScreeningResult(
        candidate_id=candidate_id,
        winning_period_count=len(winning_periods),
        winning_periods=winning_periods,
        aggregate_joint_improvement=aggregate_improvement,
        screening_pass=passed,
        decision=(
            "non_promu_significativite_et_holdout_requis"
            if passed
            else "rejete_au_screening_descriptif"
        ),
    )


def _candidate_lookup(candidates: Sequence[Candidate]) -> dict[str, Candidate]:
    lookup = {candidate.label: candidate for candidate in candidates}
    if len(lookup) != len(candidates):
        raise ValueError("candidate labels must be distinct")
    identifiers = [candidate.identifier for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate identifiers must be distinct")
    return lookup


def _target_identity(order: int, record: PredictionRecord) -> bytes:
    return struct.pack(
        ">QHBBq",
        order,
        record.target_date.year,
        record.target_date.month,
        record.target_date.day,
        record.target_original_index,
    )


def _prediction_series_summary(
    candidate_id: str, records: Sequence[PredictionRecord]
) -> PredictionSeriesSummary:
    """Hash ordered target identities and exact IEEE-754 prediction values."""

    predictions = tuple(records)
    if not predictions:
        raise ValueError("candidate prediction series must not be empty")
    probability_digest = hashlib.sha256(b"loto-probability-series-v1\0")
    selected_digest = hashlib.sha256(b"loto-selected-number-series-v1\0")
    for order, record in enumerate(predictions):
        identity = _target_identity(order, record)
        probability_digest.update(identity)
        probability_digest.update(struct.pack(">H", len(record.probabilities)))
        for probability in record.probabilities:
            probability_digest.update(struct.pack(">d", probability))

        selected_digest.update(identity)
        if record.selected_numbers is None:
            selected_digest.update(b"\x00")
        else:
            selected_digest.update(b"\x01")
            selected_digest.update(struct.pack(">H", len(record.selected_numbers)))
            for number in record.selected_numbers:
                selected_digest.update(struct.pack(">H", number))
    encoded_count = struct.pack(">Q", len(predictions))
    probability_digest.update(encoded_count)
    selected_digest.update(encoded_count)
    return PredictionSeriesSummary(
        candidate_id=candidate_id,
        prediction_count=len(predictions),
        first_target_date=predictions[0].target_date,
        last_target_date=predictions[-1].target_date,
        targets_in_2026=sum(
            record.target_date.year == PREREGISTRATION.excluded_year
            for record in predictions
        ),
        probability_digest=probability_digest.hexdigest(),
        selected_number_digest=selected_digest.hexdigest(),
    )


def _evaluate_one_candidate(
    observations: Sequence[DrawObservation],
    protocol: EvaluationProtocol,
    candidate: Candidate,
    *,
    retain_record_history: bool = False,
) -> tuple[PredictionSeriesSummary, tuple[MetricRow, ...]]:
    """Evaluate and summarize one candidate before releasing its full records.

    Archive summaries and metrics do not consume the repeated history snapshots.
    They are omitted by default after the callbacks and RNG derivation have used
    the exact visible history, avoiding quadratic retained copies without changing
    any prediction, digest, or metric.
    """

    records = run_walk_forward(
        observations,
        protocol.windows,
        candidate.callbacks,
        protocol.walk_forward,
        retain_history=retain_record_history,
    )
    rows = [
        MetricRow.from_metrics(
            candidate.identifier,
            "aggregate",
            records,
            evaluate_predictions(records),
        )
    ]
    for year in range(
        protocol.evaluation_start_year, protocol.evaluation_end_year + 1
    ):
        annual = tuple(record for record in records if record.target_date.year == year)
        if not annual:
            raise ValueError(f"candidate has no predictions for evaluation year {year}")
        rows.append(
            MetricRow.from_metrics(
                candidate.identifier,
                str(year),
                annual,
                evaluate_predictions(annual),
            )
        )
    return _prediction_series_summary(candidate.identifier, records), tuple(rows)


def evaluate_candidates(
    draws: Sequence[DrawObservation],
    protocol: EvaluationProtocol,
    *,
    candidates: Sequence[Candidate] | None = None,
) -> ExperimentRun:
    observations = tuple(draws)
    validate_target_period(observations, protocol.windows)
    configured = tuple(candidates or preregistered_candidates())
    by_label = _candidate_lookup(configured)
    summaries: dict[str, PredictionSeriesSummary] = {}
    metric_rows: list[MetricRow] = []

    for candidate in configured:
        print(f"[task-2.3a] candidate {candidate.identifier}", flush=True)
        summary, rows = _evaluate_one_candidate(
            observations, protocol, candidate
        )
        summaries[candidate.identifier] = summary
        metric_rows.extend(rows)

    cumulative_id = by_label["cumulative"].identifier
    dirichlet_id = by_label["dirichlet"].identifier
    uniform_id = by_label["uniform"].identifier
    constrained_id = by_label["constrained_random"].identifier
    if (
        summaries[cumulative_id].probability_digest
        != summaries[dirichlet_id].probability_digest
    ):
        raise AssertionError("cumulative and uniform-prior Dirichlet predictions differ")
    if tuple(map(_metrics_signature, (row for row in metric_rows if row.candidate_id == cumulative_id))) != tuple(
        map(_metrics_signature, (row for row in metric_rows if row.candidate_id == dirichlet_id))
    ):
        raise AssertionError("cumulative and uniform-prior Dirichlet metrics differ")
    if (
        summaries[uniform_id].probability_digest
        != summaries[constrained_id].probability_digest
    ):
        raise AssertionError("uniform and constrained-random marginals differ")
    for uniform_row, constrained_row in zip(
        (row for row in metric_rows if row.candidate_id == uniform_id),
        (row for row in metric_rows if row.candidate_id == constrained_id),
        strict=True,
    ):
        if any(
            getattr(uniform_row, metric) != getattr(constrained_row, metric)
            for metric in PROBABILISTIC_METRIC_NAMES
        ):
            raise AssertionError(
                "uniform and constrained-random probabilistic metrics differ"
            )

    rows = tuple(metric_rows)
    screenings = tuple(
        screen_candidate(
            rows,
            candidate.identifier,
            cumulative_id=cumulative_id,
            uniform_id=uniform_id,
        )
        for candidate in configured
        if candidate.kind == "family"
    )
    return ExperimentRun(
        prediction_summaries=MappingProxyType(summaries),
        metric_rows=rows,
        screenings=screenings,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (Mapping, MappingProxyType)):
        return {str(key): _json_value(item) for key, item in value.items()}
    if is_dataclass(value):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    return value


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _metric_csv_bytes(rows: Sequence[MetricRow]) -> bytes:
    from io import StringIO

    handle = StringIO(newline="")
    columns = (
        "candidate_id",
        "period",
        "prediction_count",
        "first_target_date",
        "last_target_date",
        *METRIC_NAMES,
    )
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values = asdict(row)
        values["first_target_date"] = row.first_target_date.isoformat()
        values["last_target_date"] = row.last_target_date.isoformat()
        writer.writerow(values)
    return handle.getvalue().encode("utf-8")


def _artifact_documents(
    run: ExperimentRun,
    protocol: EvaluationProtocol,
    *,
    commit: str,
    dataset_manifest: Mapping[str, Any],
) -> dict[str, bytes]:
    candidates = preregistered_candidates()
    _verify_preregistered_candidates(candidates)
    by_label = _candidate_lookup(candidates)
    target_summary = next(iter(run.prediction_summaries.values()))
    aggregate = {
        row.candidate_id: {
            metric: getattr(row, metric) for metric in METRIC_NAMES
        }
        for row in run.metric_rows
        if row.period == "aggregate"
    }
    cumulative_id = by_label["cumulative"].identifier
    dirichlet_id = by_label["dirichlet"].identifier
    uniform_id = by_label["uniform"].identifier
    constrained_id = by_label["constrained_random"].identifier
    preregistration = {
        "id": PREREGISTRATION.experiment_id,
        "preregistration_date": PREREGISTRATION.preregistration_date,
        "expected_commit": commit,
        "executed_commit": commit,
        "dataset": {
            "contract": PREREGISTRATION.dataset_contract,
            "files": list(PREREGISTRATION.archive_filenames),
            "manifest": dataset_manifest,
            "loader": "full archive date/count check; pre-2026 canonical preflight; no SQLite",
            "original_index_derivation": (
                "zero-based position after pre-2026 canonical validation and strict "
                "chronological sort"
            ),
        },
        "split": {
            "initial_train": "all draws strictly before 2014-01-01",
            "initial_train_size": protocol.walk_forward.initial_train_size,
            "evaluation_start_year": protocol.evaluation_start_year,
            "evaluation_end_year": protocol.evaluation_end_year,
            "annual_windows": list(
                range(protocol.evaluation_start_year, protocol.evaluation_end_year + 1)
            ),
            "history": "expanding",
            "max_train_size": None,
            "retrain_policy": protocol.walk_forward.retrain_policy.value,
        },
        "exclusion_2026": {
            "year": 2026,
            "status": "future_partial_untouched",
            "targets": target_summary.targets_in_2026,
        },
        "seed": PREREGISTRATION.seed,
        "hyperparameter_search": False,
        "candidates": [
            {
                "label": candidate.label,
                "id": candidate.identifier,
                "kind": candidate.kind,
                "parameters": candidate.parameters,
            }
            for candidate in candidates
        ],
        "metrics": list(METRIC_NAMES),
        "decision_rule": experiment_config().threshold,
        "promotion_rule": (
            "No promotion from point estimates: significance on at least three "
            "periods and confirmation on the untouched holdout remain mandatory."
        ),
    }
    result = {
        "id": PREREGISTRATION.experiment_id,
        "aggregate_metrics": aggregate,
        "metrics": [_json_value(row) for row in run.metric_rows],
        "screening": [_json_value(item) for item in run.screenings],
        "screening_pass_candidates": [
            item.candidate_id for item in run.screenings if item.screening_pass
        ],
        "integrity": {
            "prediction_series_by_candidate": {
                candidate.identifier: {
                    "count": run.prediction_summaries[
                        candidate.identifier
                    ].prediction_count,
                    "probability_digest": run.prediction_summaries[
                        candidate.identifier
                    ].probability_digest,
                    "selected_number_digest": run.prediction_summaries[
                        candidate.identifier
                    ].selected_number_digest,
                }
                for candidate in candidates
            },
            "prediction_count_per_candidate": {
                identifier: summary.prediction_count
                for identifier, summary in run.prediction_summaries.items()
            },
            "first_target_date": target_summary.first_target_date,
            "maximum_target_date": target_summary.last_target_date,
            "targets_in_2026": target_summary.targets_in_2026,
            "annual_period_count": protocol.evaluation_end_year
            - protocol.evaluation_start_year
            + 1,
            "metrics_unchanged": list(METRIC_NAMES),
        },
        "equivalence_checks": {
            "cumulative_dirichlet_probabilities_exact": (
                run.prediction_summaries[cumulative_id].probability_digest
                == run.prediction_summaries[dirichlet_id].probability_digest
            ),
            "cumulative_dirichlet_metrics_exact": tuple(
                map(_metrics_signature, rows_for(run, cumulative_id))
            )
            == tuple(map(_metrics_signature, rows_for(run, dirichlet_id))),
            "uniform_constrained_probabilistic_metrics_exact": all(
                getattr(left, metric) == getattr(right, metric)
                for left, right in zip(
                    rows_for(run, uniform_id),
                    rows_for(run, constrained_id),
                    strict=True,
                )
                for metric in PROBABILISTIC_METRIC_NAMES
            ),
        },
        "interpretation": (
            "Descriptive screening only; no causal, profitability, statistical "
            "significance, or promotion claim."
        ),
    }
    return {
        "preregistration.json": _json_bytes(preregistration),
        "metrics.csv": _metric_csv_bytes(run.metric_rows),
        "result.json": _json_bytes(result),
    }


def write_artifacts(
    artifact_dir: str | Path,
    run: ExperimentRun,
    protocol: EvaluationProtocol,
    *,
    commit: str,
    dataset_manifest: Mapping[str, Any],
) -> PublicationReceipt:
    """Publish all deterministic files with one same-filesystem directory rename."""

    directory = Path(artifact_dir)
    ensure_destination_available(directory)
    documents = _artifact_documents(
        run, protocol, commit=commit, dataset_manifest=dataset_manifest
    )
    return publish_artifact_directory(
        directory,
        documents,
        ARTIFACT_FILENAMES,
    )


def _registry_summary(run: ExperimentRun) -> str:
    aggregate = {
        row.candidate_id: {
            "log_loss": row.log_loss,
            "brier": row.brier,
            "prediction_count": row.prediction_count,
        }
        for row in run.metric_rows
        if row.period == "aggregate"
    }
    return json.dumps(
        {
            "aggregate": aggregate,
            "winning_period_counts": {
                item.candidate_id: item.winning_period_count for item in run.screenings
            },
            "screening_pass_candidates": [
                item.candidate_id for item in run.screenings if item.screening_pass
            ],
            "scope": "descriptive_non_causal_non_profitability",
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _run_registered_experiment_with_overrides(
    *,
    expected_commit: str,
    repo_root: str | Path,
    data_paths: Sequence[str | Path],
    registry_path: str | Path,
    artifact_dir: str | Path,
    dataset_contract: DatasetContract,
    evaluation_start_year: int,
    evaluation_end_year: int,
) -> ExperimentOutcome:
    """Test seam for runtime locations and reduced synthetic dataset contracts.

    The preregistration and ``experiment_config()`` are intentionally not
    injectable.  A failure row written by ``run_experiment`` remains append-only
    and consumes the experiment ID; another attempt then requires a newly
    preregistered ID.
    """

    root = Path(repo_root).resolve()
    paths = _validate_archive_paths(data_paths)
    output = Path(artifact_dir).resolve()
    try:
        output.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact directory must be inside the repository") from error
    receipt: PublicationReceipt | None = None
    parent_lock = None
    parent_lock_acquired = False

    def command() -> ExperimentOutcome:
        nonlocal parent_lock, parent_lock_acquired, receipt
        parent_lock = lock_parent_directory(output)
        parent_lock.__enter__()
        parent_lock_acquired = True
        ensure_destination_available(output)
        manifest = json.loads(build_data_manifest(paths))
        draws = load_archive_draws(paths, contract=dataset_contract)
        if any(draw.draw_date.year >= PREREGISTRATION.excluded_year for draw in draws):
            raise ValueError("2026 and later observations cannot enter the experiment")
        protocol = build_protocol(
            draws,
            evaluation_start_year=evaluation_start_year,
            evaluation_end_year=evaluation_end_year,
        )
        run = evaluate_candidates(draws, protocol)
        try:
            receipt = write_artifacts(
                output,
                run,
                protocol,
                commit=expected_commit,
                dataset_manifest=manifest,
            )
        except PublishedArtifactError as error:
            receipt = error.receipt
            raise
        passed = any(item.screening_pass for item in run.screenings)
        decision = (
            "non_promu_significativite_et_holdout_requis"
            if passed
            else "rejete_au_screening_descriptif"
        )
        return ExperimentOutcome(
            result=_registry_summary(run),
            decision=decision,
            artifacts=tuple(
                path.relative_to(root).as_posix() for path in receipt.paths
            ),
        )

    def validate_publication_before_record(outcome: ExperimentOutcome) -> None:
        """Authenticate publication at logical commit under both held locks."""

        if receipt is None:
            raise RuntimeError("experiment completed without an artifact publication")
        verify_publication(receipt)

    try:
        try:
            outcome = run_experiment(
                experiment_config(),
                command,
                data_paths=paths,
                registry_path=registry_path,
                commit=expected_commit,
                repo_path=root,
                validate_before_record=validate_publication_before_record,
            )
        except BaseException as error:
            if receipt is not None:
                error.add_note(
                    "published artifact directory deliberately retained for inspection "
                    f"(no automatic removal): directory={receipt.directory}; "
                    f"receipt={receipt!r}; the experiment ID "
                    "remains consumed by a durable registry row or durable reservation"
                )
            raise
    finally:
        if parent_lock_acquired:
            assert parent_lock is not None
            parent_lock.__exit__(None, None, None)
    return outcome


def run_registered_experiment(*, expected_commit: str) -> ExperimentOutcome:
    """Run the one official pre-registered experiment with no API substitutions."""

    return _run_registered_experiment_with_overrides(
        expected_commit=expected_commit,
        repo_root=PROJECT_ROOT,
        data_paths=tuple(PROJECT_ROOT / filename for filename in ARCHIVE_FILENAMES),
        registry_path=DEFAULT_REGISTRY_PATH,
        artifact_dir=PROJECT_ROOT / ARTIFACT_DIRECTORY,
        dataset_contract=OFFICIAL_DATASET_CONTRACT,
        evaluation_start_year=2014,
        evaluation_end_year=2025,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered Task 2.3a archive walk-forward once."
    )
    parser.add_argument(
        "--commit",
        required=True,
        help="full clean HEAD commit expected for the official execution",
    )
    arguments = parser.parse_args(argv)
    run_registered_experiment(expected_commit=arguments.commit)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the documented command
    raise SystemExit(main())


__all__ = [
    "ARCHIVE_FILENAMES",
    "ARTIFACT_FILENAMES",
    "Candidate",
    "DatasetContract",
    "EvaluationProtocol",
    "EvaluationWindow",
    "EXPERIMENT_ID",
    "ExperimentRun",
    "METRIC_NAMES",
    "MetricRow",
    "OFFICIAL_DATASET_CONTRACT",
    "PREREGISTRATION",
    "PROBABILISTIC_METRIC_NAMES",
    "Preregistration",
    "PredictionSeriesSummary",
    "SEED",
    "ScreeningResult",
    "build_protocol",
    "evaluate_candidates",
    "experiment_config",
    "load_archive_draws",
    "main",
    "preregistered_candidates",
    "rows_for",
    "run_registered_experiment",
    "screen_candidate",
    "target_draws",
    "validate_target_period",
    "write_artifacts",
]
