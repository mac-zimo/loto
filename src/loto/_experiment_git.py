"""Git provenance checks for the experiment registry."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("Git est requis pour enregistrer une expérience") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or "commande Git en échec"
        raise RuntimeError(f"Provenance Git indisponible: {detail}") from error
    return completed.stdout.strip()


def resolve_repo_root(repo_path: str | Path | None = None) -> Path:
    """Resolve the repository root without depending on the process cwd."""
    candidate = Path(repo_path).resolve() if repo_path is not None else PROJECT_ROOT
    if not candidate.exists():
        raise RuntimeError(f"Dépôt Git Loto absent: {candidate}")
    root = Path(_git(candidate, "rev-parse", "--show-toplevel")).resolve()
    if repo_path is None and root != PROJECT_ROOT:
        raise RuntimeError(f"Racine Git Loto inattendue: {root}")
    return root


def resolve_commit(root: Path, override: str | None = None) -> str:
    """Return a full, existing commit hash in the repository object format."""
    object_format = _git(root, "rev-parse", "--show-object-format")
    lengths = {"sha1": 40, "sha256": 64}
    if object_format not in lengths:
        raise RuntimeError(f"Format d'objet Git non pris en charge: {object_format}")

    commit = override if override is not None else _git(root, "rev-parse", "HEAD")
    if not isinstance(commit, str) or re.fullmatch(
        rf"[0-9a-f]{{{lengths[object_format]}}}", commit
    ) is None:
        raise ValueError("Le commit doit être un hash Git hexadécimal complet et valide")
    try:
        _git(root, "cat-file", "-e", f"{commit}^{{commit}}")
    except RuntimeError as error:
        raise ValueError(f"Le commit {commit} n'existe pas dans le dépôt Loto") from error
    return commit


def ensure_clean_worktree(root: Path, registry_path: Path) -> None:
    """Reject tracked or untracked changes, except the exact output registry."""
    arguments = ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", "."]
    try:
        registry_relative = registry_path.resolve().relative_to(root)
    except ValueError:
        pass
    else:
        arguments.append(f":(exclude,literal){registry_relative.as_posix()}")

    if _git(root, *arguments):
        raise RuntimeError(
            "Le dépôt Loto contient des modifications ou fichiers non suivis; "
            "l'expérience est refusée"
        )
