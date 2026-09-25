"""Append-only experiment registry with reproducible data manifests."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from loto._experiment_git import (
    PROJECT_ROOT,
    ensure_clean_worktree,
    resolve_commit,
    resolve_repo_root,
)

REGISTRY_COLUMNS = (
    "id",
    "hypothèse",
    "commit",
    "données",
    "split",
    "features",
    "modèle",
    "baseline",
    "métrique",
    "seuil",
    "seed",
    "résultat",
    "décision",
    "artefacts",
)
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "experiments/registry.csv"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration frozen before an experiment starts."""

    id: str
    hypothesis: str
    split: str
    features: str
    model: str
    baseline: str
    metric: str
    threshold: str
    seed: int | str


@dataclass(frozen=True)
class ExperimentOutcome:
    """Outcome returned by an experiment command."""

    result: str
    decision: str
    artifacts: tuple[str, ...] = ()


T = TypeVar("T", bound=ExperimentOutcome)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_manifest(data_paths: Iterable[str | Path]) -> str:
    """Return canonical JSON containing source file hashes and a dataset hash."""
    paths = [Path(path).resolve() for path in data_paths]
    if not paths:
        raise ValueError("Une expérience doit déclarer au moins un fichier de données")
    if len(paths) != len(set(paths)):
        raise ValueError("Les chemins de données résolus doivent être uniques")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Fichiers de données absents: {', '.join(missing)}")

    common_parent = Path(os.path.commonpath([str(path.parent) for path in paths]))
    files = sorted(
        (
            {"path": path.relative_to(common_parent).as_posix(), "sha256": _sha256(path)}
            for path in paths
        ),
        key=lambda item: item["path"],
    )
    canonical_files = json.dumps(files, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    manifest = {
        "files": files,
        "sha256": hashlib.sha256(canonical_files.encode("utf-8")).hexdigest(),
    }
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def current_commit(repo_path: str | Path | None = None) -> str:
    """Read the exact Git commit used by an experiment."""
    root = resolve_repo_root(repo_path)
    return resolve_commit(root)


def _validate_config(config: ExperimentConfig) -> None:
    values = (
        config.id,
        config.hypothesis,
        config.split,
        config.features,
        config.model,
        config.baseline,
        config.metric,
        config.threshold,
        str(config.seed),
    )
    if any(not value.strip() for value in values):
        raise ValueError("Tous les champs de configuration sont obligatoires")


def _read_ids(handle, registry_path: Path) -> set[str]:
    handle.seek(0)
    reader = csv.reader(handle, strict=True)
    try:
        header = next(reader)
    except StopIteration:
        return set()
    except csv.Error as error:
        raise ValueError(f"CSV invalide dans {registry_path}: {error}") from error
    if tuple(header) != REGISTRY_COLUMNS:
        raise ValueError(f"En-tête invalide dans {registry_path}")

    identifiers: set[str] = set()
    try:
        for row in reader:
            if len(row) != len(REGISTRY_COLUMNS):
                raise ValueError(
                    f"Ligne {reader.line_num} invalide dans {registry_path}: "
                    f"{len(row)} colonnes au lieu de {len(REGISTRY_COLUMNS)}"
                )
            identifier = row[0]
            if not identifier.strip():
                raise ValueError(f"Identifiant vide à la ligne {reader.line_num}")
            if identifier in identifiers:
                raise ValueError(f"Identifiant dupliqué dans le registre: {identifier}")
            identifiers.add(identifier)
    except csv.Error as error:
        raise ValueError(f"CSV invalide dans {registry_path}: {error}") from error
    return identifiers


def _ensure_identifier_available(handle, registry_path: Path, identifier: str) -> None:
    if identifier in _read_ids(handle, registry_path):
        raise ValueError(f"L'expérience {identifier} existe déjà; le registre est immuable")


def _write_row(handle, row: dict[str, str]) -> None:
    handle.seek(0, os.SEEK_END)
    size = os.fstat(handle.fileno()).st_size
    writer = csv.writer(handle, lineterminator="\n")
    if size == 0:
        writer.writerow(REGISTRY_COLUMNS)
    elif os.pread(handle.fileno(), 1, size - 1) not in (b"\n", b"\r"):
        handle.write("\n")
    writer.writerow([row[column] for column in REGISTRY_COLUMNS])
    handle.flush()
    os.fsync(handle.fileno())


def _append_row(registry_path: Path, row: dict[str, str]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _ensure_identifier_available(handle, registry_path, row["id"])
        _write_row(handle, row)


def _row(
    config: ExperimentConfig,
    outcome: ExperimentOutcome,
    *,
    data_manifest: str,
    commit: str,
) -> dict[str, str]:
    return {
        "id": config.id,
        "hypothèse": config.hypothesis,
        "commit": commit,
        "données": data_manifest,
        "split": config.split,
        "features": config.features,
        "modèle": config.model,
        "baseline": config.baseline,
        "métrique": config.metric,
        "seuil": config.threshold,
        "seed": str(config.seed),
        "résultat": outcome.result,
        "décision": outcome.decision,
        "artefacts": json.dumps(outcome.artifacts, ensure_ascii=False, separators=(",", ":")),
    }


def _validate_outcome(outcome: ExperimentOutcome) -> None:
    if not isinstance(outcome, ExperimentOutcome):
        raise TypeError("La commande doit retourner ExperimentOutcome")
    if not isinstance(outcome.result, str) or not outcome.result.strip():
        raise ValueError("Le résultat de l'expérience doit être une chaîne non vide")
    if not isinstance(outcome.decision, str) or not outcome.decision.strip():
        raise ValueError("La décision de l'expérience doit être une chaîne non vide")
    if not isinstance(outcome.artifacts, tuple) or any(
        not isinstance(artifact, str) for artifact in outcome.artifacts
    ):
        raise TypeError("Les artefacts doivent être un tuple de chaînes")
    try:
        json.dumps(outcome.artifacts, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("Les artefacts doivent être sérialisables en JSON") from error


def append_experiment(
    config: ExperimentConfig,
    outcome: ExperimentOutcome,
    *,
    data_paths: Iterable[str | Path],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    commit: str | None = None,
    repo_path: str | Path | None = None,
) -> None:
    """Append one completed experiment; an existing identifier is never replaced."""
    _validate_config(config)
    _validate_outcome(outcome)
    data_paths = tuple(data_paths)
    manifest = build_data_manifest(data_paths)
    root = resolve_repo_root(repo_path)
    commit_hash = resolve_commit(root, commit)
    _append_row(Path(registry_path), _row(config, outcome, data_manifest=manifest, commit=commit_hash))


def run_experiment(
    config: ExperimentConfig,
    command: Callable[[], T],
    *,
    data_paths: Iterable[str | Path],
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
    commit: str | None = None,
    repo_path: str | Path | None = None,
) -> T:
    """Run a command and immutably record success or failure with frozen inputs."""
    _validate_config(config)
    data_paths = tuple(data_paths)
    manifest = build_data_manifest(data_paths)
    root = resolve_repo_root(repo_path)
    commit_hash = resolve_commit(root, commit)
    if commit is not None and commit_hash != resolve_commit(root):
        raise ValueError("Le commit fourni ne correspond pas au HEAD exécuté")
    registry = Path(registry_path)
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a+", encoding="utf-8", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        _ensure_identifier_available(handle, registry, config.id)
        ensure_clean_worktree(root, registry)
        try:
            outcome = command()
            _validate_outcome(outcome)
            try:
                final_manifest = build_data_manifest(data_paths)
            except (OSError, ValueError) as error:
                raise RuntimeError(
                    "Les données ont changé pendant l'expérience; succès refusé"
                ) from error
            if final_manifest != manifest:
                raise RuntimeError(
                    "Les données ont changé pendant l'expérience; succès refusé"
                )
            success_row = _row(
                config, outcome, data_manifest=manifest, commit=commit_hash
            )
        except BaseException as error:
            failure = ExperimentOutcome(f"{type(error).__name__}: {error}", "erreur")
            _write_row(
                handle,
                _row(config, failure, data_manifest=manifest, commit=commit_hash),
            )
            raise
        _write_row(handle, success_row)
    return outcome
