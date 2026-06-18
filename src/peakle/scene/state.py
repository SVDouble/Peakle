"""In-memory synthetic scene state for the live server.

`SceneState` builds the terrain, peaks, intrinsics, and true camera placements
once, then computes (and caches) camera views on demand. The placement helpers
here are the single source of truth shared by both the live server and the
reproducible `peakle.demo` pipeline.
"""

from __future__ import annotations

import numpy as np

from peakle.config import AppSettings, DemoCameraSettings, PoseNoiseSettings
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.peaks import Peak
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.rendering.pinhole import look_at_camera
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.terrain.generator import TerrainGenerator
from peakle.terrain.peak_detection import PeakDetector


class SceneState:
    """Holds a synthetic scene in memory and caches computed views.

    Attributes:
        terrain: Generated terrain.
        peaks: Detected peaks.
        intrinsics: Camera intrinsics shared by every view.
        true_cameras: Ground-truth camera placements, one per view.
        pose_noise: Pose-prior noise and uncertainty settings.
        optimization_max_iterations: Local pose optimization budget.
        objective_terrain_stride: Terrain stride for the pose objective.
        seed: Random seed used to derive per-view pose-prior noise.
        renderer: Shared synthetic renderer.
    """

    def __init__(
        self,
        terrain: TerrainMap,
        peaks: list[Peak],
        intrinsics: CameraIntrinsics,
        true_cameras: list[CameraExtrinsics],
        pose_noise: PoseNoiseSettings,
        optimization_max_iterations: int,
        objective_terrain_stride: int,
        seed: int,
    ) -> None:
        self.terrain = terrain
        self.peaks = peaks
        self.intrinsics = intrinsics
        self.true_cameras = true_cameras
        self.pose_noise = pose_noise
        self.optimization_max_iterations = optimization_max_iterations
        self.objective_terrain_stride = objective_terrain_stride
        self.seed = seed
        self.renderer = SyntheticRenderer()

    @classmethod
    def from_settings(cls, settings: AppSettings) -> SceneState:
        """Builds scene state from application settings.

        Args:
            settings: Validated application settings.

        Returns:
            A fully constructed scene state with terrain, peaks, intrinsics, and
            true camera placements.
        """

        terrain_spec = settings.terrain.model_copy(update={"seed": settings.random_seed})
        terrain = TerrainGenerator(terrain_spec).generate()
        peaks = PeakDetector(settings.peak_detection).detect(terrain)
        if not peaks:
            msg = "terrain generation did not produce detectable peaks"
            raise RuntimeError(msg)
        intrinsics = build_intrinsics(
            settings.render.image_width,
            settings.render.image_height,
            settings.render.horizontal_fov_deg,
        )
        true_cameras = place_cameras(terrain, peaks, settings.camera)
        return cls(
            terrain=terrain,
            peaks=peaks,
            intrinsics=intrinsics,
            true_cameras=true_cameras,
            pose_noise=settings.pose_noise,
            optimization_max_iterations=settings.optimization.max_iterations,
            objective_terrain_stride=settings.optimization.objective_terrain_stride,
            seed=settings.random_seed,
        )

    def view_ids(self) -> list[str]:
        """Returns the stable identifier for every view."""

        return [view_id(index) for index in range(len(self.true_cameras))]

    def index_of(self, view_id_value: str) -> int | None:
        """Returns the camera index for a view identifier, if known."""

        for index in range(len(self.true_cameras)):
            if view_id(index) == view_id_value:
                return index
        return None


def view_id(index: int) -> str:
    """Returns the stable identifier for a view index."""

    return f"view-{index + 1:02d}"


def view_label(index: int) -> str:
    """Returns the display label for a view index."""

    return f"View {index + 1}"


def build_intrinsics(
    image_width: int,
    image_height: int,
    horizontal_fov_deg: float,
) -> CameraIntrinsics:
    """Builds camera intrinsics from a horizontal field of view."""

    return CameraIntrinsics.from_horizontal_fov(
        width_px=image_width,
        height_px=image_height,
        horizontal_fov_deg=horizontal_fov_deg,
    )


def place_cameras(
    terrain: TerrainMap,
    peaks: list[Peak],
    camera: DemoCameraSettings,
) -> list[CameraExtrinsics]:
    """Places synthetic cameras overlooking prominent peaks."""

    targets = select_view_targets(peaks, camera.view_count)
    x_min = float(terrain.x_m[0])
    x_max = float(terrain.x_m[-1])
    y_min = float(terrain.y_m[0])

    cameras = []
    for index, target_peak in enumerate(targets):
        target = target_peak.local_position
        camera_east = float(
            np.clip(
                target.east_m
                - terrain.spec.width_m * camera.east_offset_fraction
                + terrain.spec.width_m * camera.view_spacing_fraction * _view_lane(index, len(targets)),
                x_min + 800.0,
                x_max - 800.0,
            )
        )
        camera_north = y_min + terrain.spec.height_m * camera.north_offset_fraction
        camera_ground = terrain.elevation_at(camera_east, camera_north)
        position = LocalPoint(
            east_m=camera_east,
            north_m=camera_north,
            up_m=camera_ground + camera.overlook_height_m,
        )
        cameras.append(look_at_camera(position, target))
    return cameras


def select_view_targets(peaks: list[Peak], count: int) -> list[Peak]:
    """Selects the primary and supporting peaks the cameras look toward."""

    primary = max(
        peaks,
        key=lambda peak: (
            peak.local_position.north_m > 0.0,
            peak.prominence_m,
            peak.elevation_m,
        ),
    )
    ranked = sorted(
        (peak for peak in peaks if peak.id != primary.id),
        key=lambda peak: (peak.prominence_m, peak.elevation_m),
        reverse=True,
    )
    return [primary, *ranked[: max(count - 1, 0)]]


def noisy_prior(
    pose_noise: PoseNoiseSettings,
    true_position: LocalPoint,
    true_yaw_deg: float,
    true_pitch_deg: float,
    rng: np.random.Generator,
) -> PosePrior:
    """Builds a noisy pose prior around a true camera pose."""

    return PosePrior(
        position=LocalPoint(
            east_m=true_position.east_m + float(rng.normal(0.0, pose_noise.horizontal_noise_m)),
            north_m=true_position.north_m + float(rng.normal(0.0, pose_noise.horizontal_noise_m)),
            up_m=true_position.up_m + float(rng.normal(0.0, pose_noise.vertical_noise_m)),
        ),
        yaw_deg=true_yaw_deg + float(rng.normal(0.0, pose_noise.yaw_noise_deg)),
        pitch_deg=true_pitch_deg + float(rng.normal(0.0, pose_noise.pitch_noise_deg)),
        horizontal_sigma_m=pose_noise.horizontal_sigma_m,
        vertical_sigma_m=pose_noise.vertical_sigma_m,
        yaw_sigma_deg=pose_noise.yaw_sigma_deg,
        pitch_sigma_deg=pose_noise.pitch_sigma_deg,
    )


def _view_lane(index: int, count: int) -> float:
    if index == 0 or count <= 1:
        return 0.0
    max_pair = max((count + 1) // 2, 1)
    magnitude = ((index + 1) // 2) / max_pair
    direction = -1.0 if index % 2 else 1.0
    return direction * magnitude
