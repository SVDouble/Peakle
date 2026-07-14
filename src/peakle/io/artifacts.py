"""Artifact persistence helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from peakle.domain.terrain import TerrainMap

type JsonPayload = dict[str, Any] | list[Any]


def write_once_bytes(path: Path, data: bytes) -> None:
    """Durably create an immutable byte artifact, removing partial writes."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def fsync_directory(path: Path) -> None:
    """Flush directory-entry updates for an artifact staging directory."""

    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory(path: Path) -> Path:
    """Creates a directory if needed and returns it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: BaseModel | JsonPayload) -> None:
    """Writes JSON with Pydantic-aware serialization."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: JsonPayload
    if isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    else:
        payload = data
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_terrain_npz(path: Path, terrain: TerrainMap) -> None:
    """Saves dense terrain arrays to a compressed NumPy artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_m=terrain.x_m,
        y_m=terrain.y_m,
        elevation_m=terrain.elevation_m,
        latitude_deg=terrain.latitude_deg,
        longitude_deg=terrain.longitude_deg,
    )


def relative_artifact(path: Path, base_dir: Path) -> Path:
    """Returns an artifact path relative to the base directory."""

    return path.relative_to(base_dir)
