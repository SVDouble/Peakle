"""JSON payload builders for the workbench API.

These produce the exact shapes the browser frontend consumes, keeping the client
independent of the Python domain models. View payloads carry solve *summaries*
(no trace); the full trace is fetched per solve.
"""

from __future__ import annotations

from typing import Any

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

    return {
        "grid_width": terrain.spec.grid_width,
        "grid_height": terrain.spec.grid_height,
        "x_min_m": float(terrain.x_m[0]),
        "x_max_m": float(terrain.x_m[-1]),
        "y_min_m": float(terrain.y_m[0]),
        "y_max_m": float(terrain.y_m[-1]),
        "elevation_min_m": float(terrain.elevation_m.min()),
        "elevation_max_m": float(terrain.elevation_m.max()),
        "elevation_m": terrain.elevation_m.round(1).tolist(),
        # geographic corners so clients can map lat/lon (e.g. GT sample spots) into
        # the local east/north frame; linear across the window is plenty accurate
        "lat_min_deg": float(terrain.latitude_deg.min()),
        "lat_max_deg": float(terrain.latitude_deg.max()),
        "lon_min_deg": float(terrain.longitude_deg.min()),
        "lon_max_deg": float(terrain.longitude_deg.max()),
    }


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
        "true_extrinsics": _extrinsics(view),
        "eye_height_m": view.eye_height_m,
        "prior": view.prior.model_dump(mode="json") if view.prior else None,
        "contour": view.contour.model_dump(mode="json"),
        "image_url": f"/api/views/{view.id}/image",
        "solves": [solve_summary(solve) for solve in view.solves.values()],
    }


def solve_summary(solve: Solve) -> dict[str, Any]:
    """Builds a compact solve summary (no convergence trace)."""

    return {
        "id": solve.id,
        "created_at": solve.created_at,
        "strategy": solve.strategy,
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


def _terrain_bounds(terrain: TerrainMap) -> dict[str, float]:
    return {
        "x_min_m": float(terrain.x_m[0]),
        "x_max_m": float(terrain.x_m[-1]),
        "y_min_m": float(terrain.y_m[0]),
        "y_max_m": float(terrain.y_m[-1]),
        "elevation_min_m": float(terrain.elevation_m.min()),
        "elevation_max_m": float(terrain.elevation_m.max()),
    }
