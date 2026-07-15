from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from peakle.research import experiment
from peakle.research.experiment import canonical_json_bytes, publish_flat_run, whole_worktree_provenance


def test_canonical_json_bytes_is_compact_sorted_and_strict() -> None:
    assert canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n'
    with pytest.raises(ValueError):
        canonical_json_bytes({"invalid": float("nan")})


def test_whole_worktree_provenance_hashes_git_state_and_causal_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    implementation = tmp_path / "implementation.py"
    implementation.write_bytes(b"implementation\n")
    untracked = tmp_path / "untracked.txt"
    untracked.write_bytes(b"untracked\n")
    responses = {
        ("status", "--porcelain", "--untracked-files=all"): " M implementation.py",
        ("diff", "--binary", "HEAD"): "diff bytes",
        ("ls-files", "--others", "--exclude-standard", "-z"): "untracked.txt\0",
        ("rev-parse", "HEAD"): "revision",
        ("rev-parse", "HEAD^{tree}"): "tree",
    }
    monkeypatch.setattr(experiment, "_git", lambda _repository, *args: responses[args])

    provenance = whole_worktree_provenance((implementation,), repository=tmp_path)

    tracked_diff_sha256 = hashlib.sha256(b"diff bytes").hexdigest()
    untracked_files = [{"path": "untracked.txt", "sha256": hashlib.sha256(b"untracked\n").hexdigest()}]
    assert provenance["git_sha"] == "revision"
    assert provenance["git_tree_sha"] == "tree"
    assert provenance["dirty"] is True
    assert provenance["scope"] == "whole_worktree"
    assert provenance["worktree_status"] == " M implementation.py"
    assert provenance["worktree_status_sha256"] == hashlib.sha256(b" M implementation.py").hexdigest()
    assert provenance["worktree_diff_sha256"] == tracked_diff_sha256
    assert provenance["untracked_files"] == untracked_files
    assert (
        provenance["worktree_fingerprint_sha256"]
        == hashlib.sha256(
            canonical_json_bytes({"tracked_diff_sha256": tracked_diff_sha256, "untracked_files": untracked_files})
        ).hexdigest()
    )
    assert provenance["implementation_subset_role"] == "causal_files_for_human_review"
    assert provenance["implementation_subset"] == [
        {"path": "implementation.py", "sha256": hashlib.sha256(b"implementation\n").hexdigest()}
    ]

    original_fingerprint = provenance["worktree_fingerprint_sha256"]
    untracked.write_bytes(b"changed without renaming\n")
    changed = whole_worktree_provenance((implementation,), repository=tmp_path)
    assert changed["worktree_status_sha256"] == provenance["worktree_status_sha256"]
    assert changed["worktree_fingerprint_sha256"] != original_fingerprint


def test_publish_flat_run_adds_canonical_run_to_one_immutable_transaction(tmp_path: Path) -> None:
    output = tmp_path / "run"

    publish_flat_run(output, {"status": "complete", "schema": "test"}, {"results.json": b"results\n"})

    assert (output / "results.json").read_bytes() == b"results\n"
    assert (output / "run.json").read_bytes() == b'{"schema":"test","status":"complete"}\n'
    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_flat_run(output, {"status": "complete"}, {"results.json": b"replacement"})


def test_publish_flat_run_reserves_run_manifest_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reserved run.json"):
        publish_flat_run(tmp_path / "run", {}, {"run.json": b"caller-controlled"})

    assert not (tmp_path / "run").exists()
