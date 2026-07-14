"""JSON payload builders for the workbench API.

These produce the exact shapes the browser frontend consumes, keeping the client
independent of the Python domain models. View payloads carry solve *summaries*
(no trace); the full trace is fetched per solve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from peakle.domain.camera import CameraIntrinsics
from peakle.domain.peaks import Peak
from peakle.domain.terrain import TerrainMap
from peakle.optimization.solve import AVAILABLE_STRATEGIES
from peakle.scene.providers import PROVIDER_KINDS
from peakle.scene.scene import Scene, Solve, View


def scene_payload(scene: Scene) -> dict[str, Any]:
    """Builds the scene metadata payload for the config panel and clients."""

    return {
        "config": scene.config.model_dump(mode="json"),
        "intrinsics": scene.intrinsics.model_dump(mode="json"),
        "peak_count": len(scene.peaks),
        "view_ids": list(scene.views),
        "providers": list(PROVIDER_KINDS),
        "strategies": AVAILABLE_STRATEGIES,
        "terrain_bounds": _terrain_bounds(scene.terrain),
    }


def terrain_payload(terrain: TerrainMap) -> dict[str, Any]:
    """Builds the terrain grid payload for client rendering."""

    # The solver keeps native GLO-30 resolution. A million-vertex browser mesh
    # is already the practical display ceiling, so decimate only the payload.
    row_index = _display_indices(terrain.elevation_m.shape[0])
    col_index = _display_indices(terrain.elevation_m.shape[1])
    elevation = terrain.elevation_m[np.ix_(row_index, col_index)]
    return {
        "grid_width": len(col_index),
        "grid_height": len(row_index),
        "resolution_m": float(
            min(
                abs(terrain.x_m[col_index[-1]] - terrain.x_m[col_index[0]]) / (len(col_index) - 1),
                abs(terrain.y_m[row_index[-1]] - terrain.y_m[row_index[0]]) / (len(row_index) - 1),
            )
        ),
        "source_resolution_m": terrain_resolution_m(terrain),
        "x_min_m": float(terrain.x_m[0]),
        "x_max_m": float(terrain.x_m[-1]),
        "y_min_m": float(terrain.y_m[0]),
        "y_max_m": float(terrain.y_m[-1]),
        "elevation_min_m": float(elevation.min()),
        "elevation_max_m": float(elevation.max()),
        "elevation_m": elevation.round(1).tolist(),
        # geographic corners so clients can map lat/lon (e.g. GT sample spots) into
        # the local east/north frame; linear across the window is plenty accurate
        "lat_min_deg": float(terrain.latitude_deg.min()),
        "lat_max_deg": float(terrain.latitude_deg.max()),
        "lon_min_deg": float(terrain.longitude_deg.min()),
        "lon_max_deg": float(terrain.longitude_deg.max()),
    }


def _display_indices(size: int, max_size: int = 960) -> np.ndarray:
    if size <= max_size:
        return np.arange(size, dtype=np.int64)
    return np.linspace(0, size - 1, max_size).round().astype(np.int64)


def peaks_payload(peaks: list[Peak]) -> list[dict[str, Any]]:
    """Builds the peak list payload."""

    return [peak.model_dump(mode="json") for peak in peaks]


def intrinsics_payload(intrinsics: CameraIntrinsics) -> dict[str, Any]:
    """Builds the camera intrinsics payload."""

    return intrinsics.model_dump(mode="json")


def view_payload(view: View) -> dict[str, Any]:
    """Builds the full view payload with solve summaries."""

    return {
        "id": view.id,
        "label": view.label,
        "intrinsics": view.intrinsics.model_dump(mode="json"),
        "image_camera": view.image_camera.model_dump(mode="json"),
        "true_extrinsics": _extrinsics(view),
        "eye_height_m": view.eye_height_m,
        "prior": view.prior.model_dump(mode="json") if view.prior else None,
        "contour": view.contour.model_dump(mode="json"),
        "evidence_sources": _evidence_sources(view),
        "default_evidence_source": view.default_evidence_source,
        "pitch_comparable": view.pitch_comparable,
        "image_url": f"/api/views/{view.id}/image",
        "source": view.source,
        "gt_name": view.gt_name,
        # a GT-derived view has a reference photograph; the inspector shows it instead of the render
        "photo_url": f"/api/views/{view.id}/photo" if view.reference_photo is not None else None,
        "solves": [solve_summary(solve) for solve in view.solves.values()],
    }


def solve_summary(solve: Solve) -> dict[str, Any]:
    """Builds a compact solve summary (no convergence trace)."""

    return {
        "id": solve.id,
        "created_at": solve.created_at,
        "strategy": solve.strategy,
        "evidence_source": solve.params.get("evidence_source"),
        "pitch_comparable": solve.params.get("pitch_comparable", True),
        "evaluations": solve.result.evaluations,
        "extrinsics": solve.estimate.extrinsics.model_dump(mode="json"),
        "metrics": solve.estimate.metrics.model_dump(mode="json"),
    }


def solve_payload(solve: Solve) -> dict[str, Any]:
    """Builds the full solve payload including the convergence trace."""

    return {
        "id": solve.id,
        "created_at": solve.created_at,
        "strategy": solve.strategy,
        "params": solve.params,
        "prior": solve.prior.model_dump(mode="json"),
        "result": solve.result.model_dump(mode="json"),
    }


def _extrinsics(view: View) -> dict[str, Any] | None:
    return view.true_extrinsics.model_dump(mode="json") if view.true_extrinsics else None


def _evidence_sources(view: View) -> list[dict[str, Any]]:
    """Expose evidence choices, including unavailable defaults, with provenance."""

    keys = list(dict.fromkeys([view.default_evidence_source, *view.evidence_metadata, *view.evidence_contours]))
    fallback_labels = {
        "photo_auto": "Photo skyline (automatic)",
        "pfm_oracle": "PFM/source-depth oracle (diagnostic)",
        "rendered_skyline": "Rendered skyline",
        "view_contour": "View skyline",
    }
    rows = []
    for key in keys:
        metadata = dict(view.evidence_metadata.get(key, {}))
        contour = view.evidence_contours.get(key)
        available = metadata.pop("available", contour is not None and bool(contour.points))
        label = str(metadata.pop("label", fallback_labels.get(key, key.replace("_", " ").title())))
        diagnostic = bool(metadata.pop("diagnostic", False))
        rows.append(
            {
                "id": key,
                "label": label,
                "available": bool(available),
                "diagnostic": diagnostic,
                "default": key == view.default_evidence_source,
                "provenance": metadata,
            }
        )
    return rows


def _terrain_bounds(terrain: TerrainMap) -> dict[str, float]:
    return {
        "x_min_m": float(terrain.x_m[0]),
        "x_max_m": float(terrain.x_m[-1]),
        "y_min_m": float(terrain.y_m[0]),
        "y_max_m": float(terrain.y_m[-1]),
        "elevation_min_m": float(terrain.elevation_m.min()),
        "elevation_max_m": float(terrain.elevation_m.max()),
        "resolution_m": terrain_resolution_m(terrain),
    }


def terrain_resolution_m(terrain: TerrainMap) -> float:
    """Returns the coarser horizontal sample spacing of a terrain grid."""

    dx = (float(terrain.x_m[-1]) - float(terrain.x_m[0])) / max(terrain.x_m.size - 1, 1)
    dy = (float(terrain.y_m[-1]) - float(terrain.y_m[0])) / max(terrain.y_m.size - 1, 1)
    return round(max(abs(dx), abs(dy)), 1)
