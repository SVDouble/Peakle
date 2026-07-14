from __future__ import annotations

import os
from pathlib import Path

import pytest

from peakle.io.artifacts import fsync_directory, publish_directory_once, write_once_bytes


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


def test_publish_directory_once_atomically_publishes_exact_bytes(tmp_path: Path) -> None:
    output = tmp_path / "study"

    publish_directory_once(output, {"results.json": b"results\n", "run.json": b"run\n"})

    assert output.is_dir()
    assert (output / "results.json").read_bytes() == b"results\n"
    assert (output / "run.json").read_bytes() == b"run\n"
    assert not list(tmp_path.glob(".study.staging-*"))


def test_publish_directory_once_refuses_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "study"
    output.mkdir()
    marker = output / "keep"
    marker.write_bytes(b"original")

    with pytest.raises(FileExistsError, match="refusing to replace"):
        publish_directory_once(output, {"run.json": b"replacement"})

    assert marker.read_bytes() == b"original"
    assert not (output / "run.json").exists()


def test_publish_directory_once_rejects_an_empty_artifact(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one file"):
        publish_directory_once(tmp_path / "study", {})

    assert not list(tmp_path.iterdir())


def test_publish_directory_once_cleans_staging_after_a_write_failure(tmp_path: Path) -> None:
    output = tmp_path / "study"
    calls = 0

    def failing_writer(path: Path, data: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        write_once_bytes(path, data)

    with pytest.raises(OSError, match="injected write failure"):
        publish_directory_once(
            output,
            {"results.json": b"results", "run.json": b"run"},
            write_bytes=failing_writer,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".study.staging-*"))


@pytest.mark.parametrize("name", ["", "/absolute", "../traversal", "nested/result.json", ".", ".."])
def test_publish_directory_once_rejects_unsafe_paths_before_staging(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="safe relative path component"):
        publish_directory_once(tmp_path / "study", {name: b"content"})

    assert not list(tmp_path.iterdir())
