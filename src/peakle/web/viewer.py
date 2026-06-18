"""Static browser viewer asset generation."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Protocol

from peakle.domain.annotations import PeakAnnotation
from peakle.domain.camera import CameraExtrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.peaks import Peak
from peakle.domain.pose import PoseEstimate, PosePrior
from peakle.domain.scene import SyntheticScene
from peakle.domain.terrain import TerrainMap
from peakle.io.artifacts import write_json


class ViewerView(Protocol):
    """Per-view data consumed by the static browser viewer."""

    view_id: str
    label: str
    true_camera: CameraExtrinsics
    pose_prior: PosePrior
    contour: SkylineContour
    pose_estimate: PoseEstimate
    annotations: list[PeakAnnotation]
    images: dict[str, Path]


def write_viewer_assets(
    artifact_dir: Path,
    terrain: TerrainMap,
    peaks: list[Peak],
    scene: SyntheticScene,
    views: Sequence[ViewerView],
) -> None:
    """Writes static WebGL viewer files into the artifact directory.

    Args:
        artifact_dir: Directory containing generated demo artifacts.
        terrain: Generated terrain.
        peaks: Detected peaks.
        scene: Synthetic scene metadata.
        views: Generated view records for browser selection.
    """

    static_root = resources.files("peakle.web.static")
    for filename in ("index.html", "styles.css", "app.js"):
        source = static_root.joinpath(filename)
        destination = artifact_dir / filename
        with resources.as_file(source) as source_path:
            shutil.copyfile(source_path, destination)

    write_json(
        artifact_dir / "viewer-data.json",
        {
            "terrain": {
                "grid_width": terrain.spec.grid_width,
                "grid_height": terrain.spec.grid_height,
                "x_min_m": float(terrain.x_m[0]),
                "x_max_m": float(terrain.x_m[-1]),
                "y_min_m": float(terrain.y_m[0]),
                "y_max_m": float(terrain.y_m[-1]),
                "elevation_min_m": float(terrain.elevation_m.min()),
                "elevation_max_m": float(terrain.elevation_m.max()),
                "elevation_m": terrain.elevation_m.round(1).tolist(),
            },
            "peaks": [peak.model_dump(mode="json") for peak in peaks],
            "scene": scene.model_dump(mode="json"),
            "views": [_view_payload(view) for view in views],
        },
    )


def _view_payload(view: ViewerView) -> dict[str, object]:
    """Converts a generated view record to JSON-compatible data."""

    return {
        "id": view.view_id,
        "label": view.label,
        "true_camera": view.true_camera.model_dump(mode="json"),
        "pose_prior": view.pose_prior.model_dump(mode="json"),
        "contour": view.contour.model_dump(mode="json"),
        "pose_estimate": view.pose_estimate.model_dump(mode="json"),
        "annotations": [annotation.model_dump(mode="json") for annotation in view.annotations],
        "images": {key: path.as_posix() for key, path in view.images.items()},
    }
