from __future__ import annotations

import os
from pathlib import Path

import pytest

from peakle.io.artifacts import fsync_directory, write_once_bytes


def test_write_once_bytes_refuses_to_replace_an_artifact(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    write_once_bytes(path, b"first")

    with pytest.raises(FileExistsError):
        write_once_bytes(path, b"second")

    assert path.read_bytes() == b"first"


def test_write_once_bytes_removes_a_partial_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "result.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        write_once_bytes(path, b"partial")

    assert not path.exists()


def test_fsync_directory_accepts_an_artifact_directory(tmp_path: Path) -> None:
    fsync_directory(tmp_path)
