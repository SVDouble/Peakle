"""End-to-end synthetic demo pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from peakle.annotation.labeling import PeakLabeler
from peakle.annotation.overlay import AnnotationOverlay
from peakle.config import AppSettings, DemoCameraSettings, PoseNoiseSettings
from peakle.domain.annotations import PeakAnnotation
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.coordinates import LocalPoint
from peakle.domain.peaks import Peak, PeakDetectionSpec
from peakle.domain.pose import PoseEstimate, PosePrior
from peakle.domain.scene import SyntheticScene
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.io.artifacts import ensure_directory, relative_artifact, save_terrain_npz, write_json
from peakle.optimization.pose_search import PoseOptimizer, add_synthetic_truth_metrics
from peakle.rendering.pinhole import look_at_camera
from peakle.rendering.rasterizer import RenderArrays, SyntheticRenderer
from peakle.rendering.skyline import extract_skyline_from_mask
from peakle.terrain.generator import TerrainGenerator
from peakle.terrain.peak_detection import PeakDetector
from peakle.web.viewer import write_viewer_assets

POSE_NOISE_SEED_OFFSET = 10_000
LEGACY_ROOT_VIEW_FILES = (
    "annotated.png",
    "annotations.json",
    "contour.json",
    "contour_debug.png",
    "pose_estimate.json",
    "render.png",
    "terrain_mask.png",
)


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
        objective_terrain_stride: Terrain stride for pose objective rendering.
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
    """Summary of generated demo artifacts."""

    output_dir: Path
    scene_path: Path
    viewer_path: Path
    primary_render_path: Path
    primary_annotated_path: Path
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
    viewer_html: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> DemoArtifactPaths:
        """Builds all artifact paths for an output directory."""

        return cls(
            output_dir=output_dir,
            terrain_npz=output_dir / "terrain.npz",
            terrain_json=output_dir / "terrain.json",
            peaks_json=output_dir / "peaks.json",
            scene_json=output_dir / "scene.json",
            viewer_html=output_dir / "index.html",
        )

    def relative(self, path: Path) -> Path:
        """Returns a path relative to the artifact directory."""

        return relative_artifact(path, self.output_dir)

    def view_artifacts(self, index: int) -> DemoViewArtifactPaths:
        """Builds image and fit artifact paths for a generated view."""

        view_id = f"view-{index + 1:02d}"
        view_dir = ensure_directory(self.output_dir / "views" / view_id)
        return DemoViewArtifactPaths(
            view_id=view_id,
            label=f"View {index + 1}",
            output_dir=self.output_dir,
            render_png=view_dir / "render.png",
            terrain_mask_png=view_dir / "terrain_mask.png",
            contour_json=view_dir / "contour.json",
            pose_estimate_json=view_dir / "pose_estimate.json",
            contour_debug_png=view_dir / "contour_debug.png",
            annotations_json=view_dir / "annotations.json",
            annotated_png=view_dir / "annotated.png",
        )


class DemoViewArtifactPaths(BaseModel):
    """Concrete per-view artifacts written by the demo pipeline."""

    model_config = ConfigDict(frozen=True)

    view_id: str
    label: str
    output_dir: Path
    render_png: Path
    terrain_mask_png: Path
    contour_json: Path
    pose_estimate_json: Path
    contour_debug_png: Path
    annotations_json: Path
    annotated_png: Path

    def relative(self, path: Path) -> Path:
        """Returns a path relative to the demo artifact directory."""

        return relative_artifact(path, self.output_dir)


class DemoViewResult(BaseModel):
    """Generated camera view and fit products for the browser viewer."""

    model_config = ConfigDict(frozen=True)

    view_id: str
    label: str
    true_camera: CameraExtrinsics
    pose_prior: PosePrior
    contour: SkylineContour
    pose_estimate: PoseEstimate
    annotations: list[PeakAnnotation]
    images: dict[str, Path]


def run_demo(options: DemoOptions) -> DemoRunResult:
    """Runs the full synthetic peak annotation demo.

    Args:
        options: Demo runtime options.

    Returns:
        Summary of generated artifacts and fit metrics.
    """

    paths = DemoArtifactPaths.from_output_dir(ensure_directory(options.output_dir))
    _remove_legacy_root_view_artifacts(paths.output_dir)
    rng = np.random.default_rng(options.seed + POSE_NOISE_SEED_OFFSET)

    terrain = _generate_terrain(options, paths)
    peaks = _detect_peaks(options, terrain, paths)
    intrinsics = _build_intrinsics(options)
    true_cameras = _place_cameras(options, terrain, peaks)

    renderer = SyntheticRenderer()
    views = [
        _run_view(
            options=options,
            terrain=terrain,
            peaks=peaks,
            intrinsics=intrinsics,
            true_camera=true_camera,
            renderer=renderer,
            rng=rng,
            paths=paths.view_artifacts(index),
        )
        for index, true_camera in enumerate(true_cameras)
    ]
    primary_view = views[0]
    scene = _write_scene(
        terrain=terrain,
        intrinsics=intrinsics,
        true_camera=primary_view.true_camera,
        pose_prior=primary_view.pose_prior,
        peaks=peaks,
        paths=paths,
    )

    write_viewer_assets(
        artifact_dir=paths.output_dir,
        terrain=terrain,
        peaks=peaks,
        scene=scene,
        views=views,
    )

    visible_labels = sum(1 for annotation in primary_view.annotations if annotation.visible)
    return DemoRunResult(
        output_dir=paths.output_dir,
        scene_path=paths.scene_json,
        viewer_path=paths.viewer_html,
        primary_render_path=paths.output_dir / primary_view.images["render"],
        primary_annotated_path=paths.output_dir / primary_view.images["annotated"],
        position_error_m=primary_view.pose_estimate.metrics.position_error_m,
        yaw_error_deg=primary_view.pose_estimate.metrics.yaw_error_deg,
        contour_mae_px=primary_view.pose_estimate.metrics.contour_mae_px,
        visible_labels=visible_labels,
    )


def _remove_legacy_root_view_artifacts(output_dir: Path) -> None:
    """Removes obsolete single-view root artifacts from earlier demo layouts."""

    for filename in LEGACY_ROOT_VIEW_FILES:
        (output_dir / filename).unlink(missing_ok=True)


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


def _build_intrinsics(options: DemoOptions) -> CameraIntrinsics:
    return CameraIntrinsics.from_horizontal_fov(
        width_px=options.image_width,
        height_px=options.image_height,
        horizontal_fov_deg=options.horizontal_fov_deg,
    )


def _run_view(
    options: DemoOptions,
    terrain: TerrainMap,
    peaks: list[Peak],
    intrinsics: CameraIntrinsics,
    true_camera: CameraExtrinsics,
    renderer: SyntheticRenderer,
    rng: np.random.Generator,
    paths: DemoViewArtifactPaths,
) -> DemoViewResult:
    """Renders and fits one synthetic camera view."""

    pose_prior = _noisy_prior(options, true_camera.position, true_camera.yaw_deg, true_camera.pitch_deg, rng)
    render_arrays = _render_scene(renderer, terrain, intrinsics, true_camera, paths)
    contour = _extract_contour(render_arrays, paths)
    estimate = _estimate_pose(options, terrain, contour, intrinsics, pose_prior, true_camera, paths)
    estimated_profile = renderer.skyline_profile(terrain, intrinsics, estimate.extrinsics)
    _write_contour_debug(renderer, render_arrays, contour, estimated_profile, paths)
    annotations = _annotate_peaks(
        render_arrays=render_arrays,
        peaks=peaks,
        intrinsics=intrinsics,
        estimate=estimate,
        estimated_profile=estimated_profile,
        paths=paths,
    )
    return DemoViewResult(
        view_id=paths.view_id,
        label=paths.label,
        true_camera=true_camera,
        pose_prior=pose_prior,
        contour=contour,
        pose_estimate=estimate,
        annotations=annotations,
        images={
            "render": paths.relative(paths.render_png),
            "annotated": paths.relative(paths.annotated_png),
            "contour_debug": paths.relative(paths.contour_debug_png),
        },
    )


def _render_scene(
    renderer: SyntheticRenderer,
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    true_camera: CameraExtrinsics,
    paths: DemoViewArtifactPaths,
) -> RenderArrays:
    render_arrays = renderer.render(terrain, intrinsics, true_camera)
    render_arrays.image.save(paths.render_png)
    Image.fromarray((render_arrays.terrain_mask.astype(np.uint8) * 255), mode="L").save(paths.terrain_mask_png)
    return render_arrays


def _extract_contour(render_arrays: RenderArrays, paths: DemoViewArtifactPaths) -> SkylineContour:
    contour = extract_skyline_from_mask(render_arrays.terrain_mask, source=paths.terrain_mask_png.name)
    write_json(paths.contour_json, contour)
    return contour


def _write_scene(
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    true_camera: CameraExtrinsics,
    pose_prior: PosePrior,
    peaks: list[Peak],
    paths: DemoArtifactPaths,
) -> SyntheticScene:
    scene = SyntheticScene(
        terrain_spec=terrain.spec,
        intrinsics=intrinsics,
        true_camera=true_camera,
        pose_prior=pose_prior,
        peaks=peaks,
        artifacts={
            "terrain": paths.relative(paths.terrain_npz),
            "views": Path("views"),
        },
    )
    write_json(paths.scene_json, scene)
    return scene


def _estimate_pose(
    options: DemoOptions,
    terrain: TerrainMap,
    contour: SkylineContour,
    intrinsics: CameraIntrinsics,
    pose_prior: PosePrior,
    true_camera: CameraExtrinsics,
    paths: DemoViewArtifactPaths,
) -> PoseEstimate:
    estimate = PoseOptimizer(
        max_iterations=options.optimization_max_iterations,
        objective_terrain_stride=options.objective_terrain_stride,
    ).estimate(
        terrain=terrain,
        contour=contour,
        intrinsics=intrinsics,
        prior=pose_prior,
    )
    estimate = add_synthetic_truth_metrics(estimate, true_camera)
    write_json(paths.pose_estimate_json, estimate)
    return estimate


def _write_contour_debug(
    renderer: SyntheticRenderer,
    render_arrays: RenderArrays,
    contour: SkylineContour,
    estimated_profile: np.ndarray,
    paths: DemoViewArtifactPaths,
) -> None:
    contour_debug = renderer.draw_contour_debug(
        render_arrays.image,
        contour.to_profile(),
        estimated_profile,
    )
    contour_debug.save(paths.contour_debug_png)


def _annotate_peaks(
    render_arrays: RenderArrays,
    peaks: list[Peak],
    intrinsics: CameraIntrinsics,
    estimate: PoseEstimate,
    estimated_profile: np.ndarray,
    paths: DemoViewArtifactPaths,
) -> list[PeakAnnotation]:
    annotations = PeakLabeler().build_annotations(
        peaks=peaks,
        intrinsics=intrinsics,
        extrinsics=estimate.extrinsics,
        skyline_profile=estimated_profile,
    )
    write_json(
        paths.annotations_json,
        [annotation.model_dump(mode="json") for annotation in annotations],
    )
    annotated = AnnotationOverlay().draw(render_arrays.image, annotations)
    annotated.save(paths.annotated_png)
    return annotations


def _place_cameras(options: DemoOptions, terrain: TerrainMap, peaks: list[Peak]) -> list[CameraExtrinsics]:
    targets = _select_view_targets(peaks, options.camera.view_count)
    x_min = float(terrain.x_m[0])
    x_max = float(terrain.x_m[-1])
    y_min = float(terrain.y_m[0])

    cameras = []
    for index, target_peak in enumerate(targets):
        target = target_peak.local_position
        camera_east = float(
            np.clip(
                target.east_m
                - terrain.spec.width_m * options.camera.east_offset_fraction
                + terrain.spec.width_m * options.camera.view_spacing_fraction * _view_lane(index, len(targets)),
                x_min + 800.0,
                x_max - 800.0,
            )
        )
        camera_north = y_min + terrain.spec.height_m * options.camera.north_offset_fraction
        camera_ground = terrain.elevation_at(camera_east, camera_north)
        position = LocalPoint(
            east_m=camera_east,
            north_m=camera_north,
            up_m=camera_ground + options.camera.overlook_height_m,
        )
        cameras.append(look_at_camera(position, target))
    return cameras


def _select_view_targets(peaks: list[Peak], count: int) -> list[Peak]:
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


def _view_lane(index: int, count: int) -> float:
    if index == 0 or count <= 1:
        return 0.0
    max_pair = max((count + 1) // 2, 1)
    magnitude = ((index + 1) // 2) / max_pair
    direction = -1.0 if index % 2 else 1.0
    return direction * magnitude


def _noisy_prior(
    options: DemoOptions,
    true_position: LocalPoint,
    true_yaw_deg: float,
    true_pitch_deg: float,
    rng: np.random.Generator,
) -> PosePrior:
    noise = options.pose_noise
    return PosePrior(
        position=LocalPoint(
            east_m=true_position.east_m + float(rng.normal(0.0, noise.horizontal_noise_m)),
            north_m=true_position.north_m + float(rng.normal(0.0, noise.horizontal_noise_m)),
            up_m=true_position.up_m + float(rng.normal(0.0, noise.vertical_noise_m)),
        ),
        yaw_deg=true_yaw_deg + float(rng.normal(0.0, noise.yaw_noise_deg)),
        pitch_deg=true_pitch_deg + float(rng.normal(0.0, noise.pitch_noise_deg)),
        horizontal_sigma_m=noise.horizontal_sigma_m,
        vertical_sigma_m=noise.vertical_sigma_m,
        yaw_sigma_deg=noise.yaw_sigma_deg,
        pitch_sigma_deg=noise.pitch_sigma_deg,
    )
