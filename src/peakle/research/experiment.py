"""Shared provenance and publication primitives for research runs."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from peakle.io.artifacts import publish_directory_once
from peakle.localize.paths import BASE


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode()


def whole_worktree_provenance(
    implementation_paths: Iterable[Path] = (),
    repository: Path = BASE,
) -> dict[str, Any]:
    """Capture one canonical Git snapshot plus an auditable causal-file subset."""

    status = _git(repository, "status", "--porcelain", "--untracked-files=all")
    untracked_output = _git(repository, "ls-files", "--others", "--exclude-standard", "-z")
    untracked = [
        {"path": name, "sha256": _file_sha256(repository / name)}
        for name in sorted((untracked_output or "").split("\0"))
        if name
    ]
    tracked_diff_sha256 = hashlib.sha256((_git(repository, "diff", "--binary", "HEAD") or "").encode()).hexdigest()
    fingerprint_bytes = canonical_json_bytes({"tracked_diff_sha256": tracked_diff_sha256, "untracked_files": untracked})
    subset = [
        {"path": str(path.relative_to(repository)), "sha256": _file_sha256(path)} for path in implementation_paths
    ]
    return {
        "git_sha": _git(repository, "rev-parse", "HEAD"),
        "git_tree_sha": _git(repository, "rev-parse", "HEAD^{tree}"),
        "dirty": bool(status) if status is not None else None,
        "scope": "whole_worktree",
        "worktree_status": status,
        "worktree_status_sha256": hashlib.sha256((status or "").encode()).hexdigest(),
        "worktree_diff_sha256": tracked_diff_sha256,
        "untracked_files": untracked,
        "worktree_fingerprint_sha256": hashlib.sha256(fingerprint_bytes).hexdigest(),
        "implementation_subset_role": "causal_files_for_human_review",
        "implementation_subset": subset,
    }


def publish_flat_run(output: Path, run: Mapping[str, Any], files: Mapping[str, bytes]) -> None:
    if "run.json" in files:
        raise ValueError("run files must not provide the reserved run.json artifact")
    publish_directory_once(output, {**files, "run.json": canonical_json_bytes(run)})


def _git(repository: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(("git", *args), cwd=repository, text=True).strip()
    except OSError, subprocess.CalledProcessError:
        return None


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
