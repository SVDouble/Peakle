"""Headless synthetic demo pipeline.

Generates reproducible scene artifacts (terrain, peaks, scene metadata) and
computes the primary view on the fly to report fit metrics. The browser viewer
no longer depends on precomputed view artifacts; it computes views on demand via
the live server (`peakle.web`).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from peakle.config import AppSettings, DemoCameraSettings, PoseNoiseSettings
from peakle.domain.camera import CameraIntrinsics
from peakle.domain.peaks import Peak, PeakDetectionSpec
from peakle.domain.scene import SyntheticScene
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.io.artifacts import ensure_directory, relative_artifact, save_terrain_npz, write_json
from peakle.scene.state import SceneState, build_intrinsics, place_cameras
from peakle.scene.views import ComputedView, compute_view
from peakle.terrain.generator import TerrainGenerator
from peakle.terrain.peak_detection import PeakDetector


class DemoOptions(BaseModel):
    """Runtime options for the synthetic demo.

    Attributes:
        output_dir: Artifact output directory.
        seed: Random seed for terrain and pose noise.
        terrain: Terrain generation parameters.
        peak_detection: Peak detection parameters.
        image_width: Render width in pixels.
        image_height: Render height in pixels.
        horizontal_fov_deg: Render camera horizontal field of view.
        optimization_max_iterations: Local pose optimization budget.
        objective_terrain_stride: Terrain stride for the pose objective.
        camera: Synthetic camera placement parameters.
        pose_noise: Synthetic pose-prior noise and uncertainty parameters.
    """

    output_dir: Path
    seed: int
    terrain: TerrainSpec
    peak_detection: PeakDetectionSpec
    image_width: int = Field(ge=160)
    image_height: int = Field(ge=120)
    horizontal_fov_deg: float = Field(gt=1.0, lt=179.0)
    optimization_max_iterations: int = Field(ge=1)
    objective_terrain_stride: int = Field(ge=1)
    camera: DemoCameraSettings
    pose_noise: PoseNoiseSettings

    @classmethod
    def from_settings(
        cls,
        settings: AppSettings,
        output_dir: Path | None = None,
        seed: int | None = None,
        grid_width: int | None = None,
        grid_height: int | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        optimization_max_iterations: int | None = None,
    ) -> DemoOptions:
        """Builds demo options from app settings and CLI overrides."""

        resolved_seed = settings.random_seed if seed is None else seed
        return cls(
            output_dir=output_dir or settings.artifact_dir,
            seed=resolved_seed,
            terrain=settings.terrain.model_copy(
                update={
                    "seed": resolved_seed,
                    "grid_width": settings.terrain.grid_width if grid_width is None else grid_width,
                    "grid_height": settings.terrain.grid_height if grid_height is None else grid_height,
                }
            ),
            peak_detection=settings.peak_detection,
            image_width=settings.render.image_width if image_width is None else image_width,
            image_height=settings.render.image_height if image_height is None else image_height,
            horizontal_fov_deg=settings.render.horizontal_fov_deg,
            optimization_max_iterations=(
                settings.optimization.max_iterations
                if optimization_max_iterations is None
                else optimization_max_iterations
            ),
            objective_terrain_stride=settings.optimization.objective_terrain_stride,
            camera=settings.camera,
            pose_noise=settings.pose_noise,
        )


class DemoRunResult(BaseModel):
    """Summary of a synthetic demo run."""

    output_dir: Path
    terrain_path: Path
    peaks_path: Path
    scene_path: Path
    position_error_m: float | None
    yaw_error_deg: float | None
    contour_mae_px: float
    visible_labels: int


class DemoArtifactPaths(BaseModel):
    """Concrete artifact paths for one demo run."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path
    terrain_npz: Path
    terrain_json: Path
    peaks_json: Path
    scene_json: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> DemoArtifactPaths:
        """Builds all artifact paths for an output directory."""

        return cls(
            output_dir=output_dir,
            terrain_npz=output_dir / "terrain.npz",
            terrain_json=output_dir / "terrain.json",
            peaks_json=output_dir / "peaks.json",
            scene_json=output_dir / "scene.json",
        )

    def relative(self, path: Path) -> Path:
        """Returns a path relative to the artifact directory."""

        return relative_artifact(path, self.output_dir)


def run_demo(options: DemoOptions) -> DemoRunResult:
    """Runs the headless synthetic demo and reports primary-view metrics.

    Args:
        options: Demo runtime options.

    Returns:
        Summary of generated artifacts and fit metrics for the primary view.
    """

    paths = DemoArtifactPaths.from_output_dir(ensure_directory(options.output_dir))

    terrain = _generate_terrain(options, paths)
    peaks = _detect_peaks(options, terrain, paths)
    intrinsics = build_intrinsics(options.image_width, options.image_height, options.horizontal_fov_deg)
    true_cameras = place_cameras(terrain, peaks, options.camera)

    state = SceneState(
        terrain=terrain,
        peaks=peaks,
        intrinsics=intrinsics,
        true_cameras=true_cameras,
        pose_noise=options.pose_noise,
        optimization_max_iterations=options.optimization_max_iterations,
        objective_terrain_stride=options.objective_terrain_stride,
        seed=options.seed,
    )
    primary_view = compute_view(state, 0)
    _write_scene(terrain, intrinsics, primary_view, peaks, paths)

    visible_labels = sum(1 for annotation in primary_view.annotations if annotation.visible)
    metrics = primary_view.pose_estimate.metrics
    return DemoRunResult(
        output_dir=paths.output_dir,
        terrain_path=paths.terrain_npz,
        peaks_path=paths.peaks_json,
        scene_path=paths.scene_json,
        position_error_m=metrics.position_error_m,
        yaw_error_deg=metrics.yaw_error_deg,
        contour_mae_px=metrics.contour_mae_px,
        visible_labels=visible_labels,
    )


def _generate_terrain(options: DemoOptions, paths: DemoArtifactPaths) -> TerrainMap:
    terrain = TerrainGenerator(options.terrain).generate()
    save_terrain_npz(paths.terrain_npz, terrain)
    write_json(paths.terrain_json, terrain.metadata())
    return terrain


def _detect_peaks(options: DemoOptions, terrain: TerrainMap, paths: DemoArtifactPaths) -> list[Peak]:
    peaks = PeakDetector(options.peak_detection).detect(terrain)
    if not peaks:
        msg = "terrain generation did not produce detectable peaks"
        raise RuntimeError(msg)
    write_json(paths.peaks_json, [peak.model_dump(mode="json") for peak in peaks])
    return peaks


def _write_scene(
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    primary_view: ComputedView,
    peaks: list[Peak],
    paths: DemoArtifactPaths,
) -> SyntheticScene:
    scene = SyntheticScene(
        terrain_spec=terrain.spec,
        intrinsics=intrinsics,
        true_camera=primary_view.true_camera,
        pose_prior=primary_view.pose_prior,
        peaks=peaks,
        artifacts={
            "terrain": paths.relative(paths.terrain_npz),
        },
    )
    write_json(paths.scene_json, scene)
    return scene
