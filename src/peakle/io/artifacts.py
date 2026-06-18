"""Demo artifact read/write helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from peakle.domain.terrain import TerrainMap

type JsonPayload = dict[str, Any] | list[Any]


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
