"""Deterministic synthetic scenes shared by bounded research experiments."""

from __future__ import annotations

from typing import Any

from peakle.config import AppSettings, load_settings
from peakle.scene.state import place_cameras
from peakle.terrain.generator import TerrainGenerator
from peakle.terrain.peak_detection import PeakDetector


def rugged_scenes(
    seeds: tuple[int, ...],
    *,
    views_per_scene: int,
    terrain_width_m: float,
    terrain_height_m: float,
    terrain_grid_width: int,
    terrain_grid_height: int,
    eye_height_m: float,
    settings: AppSettings | None = None,
) -> list[dict[str, Any]]:
    """Generate the existing seeded rugged terrain/camera fixtures once."""

    if views_per_scene < 1:
        raise ValueError("views_per_scene must be positive")
    settings = settings or load_settings()
    scenes: list[dict[str, Any]] = []
    for seed in seeds:
        spec = settings.terrain.model_copy(
            update={
                "seed": seed,
                "width_m": terrain_width_m,
                "height_m": terrain_height_m,
                "grid_width": terrain_grid_width,
                "grid_height": terrain_grid_height,
            }
        )
        terrain = TerrainGenerator(spec).generate()
        peaks = PeakDetector(settings.peak_detection).detect(terrain)
        if not peaks:
            raise RuntimeError(f"rugged synthetic terrain seed {seed} produced no peaks")
        cameras = place_cameras(
            terrain,
            peaks,
            settings.camera.model_copy(update={"view_count": views_per_scene, "overlook_height_m": eye_height_m}),
        )
        for index, truth in enumerate(cameras):
            scenes.append(
                {
                    "scene_id": f"rugged-s{seed}-v{index + 1:02d}",
                    "kind": "rugged_generated",
                    "terrain_seed": seed,
                    "view_index": index,
                    "terrain": terrain,
                    "truth": truth,
                    "expected_pose_ambiguity": False,
                }
            )
    return scenes
