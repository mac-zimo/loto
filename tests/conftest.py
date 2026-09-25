from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def clean_git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", repo], check=True)
    marker = repo / "tracked.txt"
    marker.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "-C", repo, "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repo,
            "-c",
            "user.name=Loto Tests",
            "-c",
            "user.email=loto-tests@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        check=True,
    )
    return repo
