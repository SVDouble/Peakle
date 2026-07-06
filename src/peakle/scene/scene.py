"""Mutable, in-memory workbench scene.

A `Scene` owns the elevation map (via a `MapProvider`), the detected peaks, the
known intrinsics, and the user-created `View`s. Views are placed by clicking the
map; each carries a rendered image and an unbounded set of `Solve`s. State lives
in memory for the life of the server process (one scene per server).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from peakle.config import AppSettings, PoseNoiseSettings
from peakle.contours import DEFAULT_DETECTOR, ContourDetector
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.coordinates import LocalPoint
from peakle.domain.peaks import Peak, PeakDetectionSpec
from peakle.domain.pose import PoseEstimate, PosePrior
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.optimization.solve import PoseSolveResult, solve_pose
from peakle.rendering.rasterizer import RenderArrays, SyntheticRenderer
from peakle.scene.providers import ProviderKind, build_provider
from peakle.scene.state import build_intrinsics, noisy_prior
from peakle.terrain.dem import DEFAULT_DEM_DIR, find_hgt_tile
from peakle.terrain.gazetteer import name_peaks_from_osm
from peakle.terrain.peak_detection import PeakDetector

DEFAULT_EYE_HEIGHT_M = 150.0
PRIOR_SEED_OFFSET = 20_000


class SceneConfig(BaseModel):
    """Scene-level configuration chosen in the config panel.

    Attributes:
        provider: Map provider kind.
        seed: Map seed.
        image_width: Render width in pixels.
        image_height: Render height in pixels.
        horizontal_fov_deg: Camera horizontal field of view.
        default_strategy: Default solver strategy.
    """

    provider: ProviderKind
    seed: int
    image_width: int
    image_height: int
    horizontal_fov_deg: float
    default_strategy: str


class Solve(BaseModel):
    """One solver run against a view.

    Attributes:
        id: Stable solve identifier.
        created_at: ISO-8601 creation timestamp.
        strategy: Solver strategy key.
        params: Solver parameters used.
        prior: Pose prior the solver started from.
        result: Solve result with metrics and convergence trace.
    """

    id: str
    created_at: str
    strategy: str
    params: dict[str, Any]
    prior: PosePrior
    result: PoseSolveResult

    @property
    def estimate(self) -> PoseEstimate:
        """Returns the estimated pose."""

        return self.result.estimate


class View(BaseModel):
    """A user-placed camera view with a rendered image and its solves.

    Attributes:
        id: Stable view identifier.
        label: Display label.
        intrinsics: Known camera intrinsics.
        true_extrinsics: Ground-truth pose (None for a real photo with no truth).
        eye_height_m: Camera height above the terrain surface.
        prior: Fixed noisy prior shared by every solve of this view.
        contour: Detected skyline contour.
        render_arrays: In-memory render outputs (image, mask, profile).
        solves: Solve attempts keyed by id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    label: str
    intrinsics: CameraIntrinsics
    true_extrinsics: CameraExtrinsics | None
    eye_height_m: float
    prior: PosePrior | None
    contour: SkylineContour
    render_arrays: RenderArrays
    solves: dict[str, Solve] = {}


class Scene:
    """In-memory workbench scene with editable views."""

    def __init__(
        self,
        config: SceneConfig,
        terrain: TerrainMap,
        peaks: list[Peak],
        intrinsics: CameraIntrinsics,
        pose_noise: PoseNoiseSettings,
        terrain_stride: int,
        terrain_spec: TerrainSpec,
        peak_detection: PeakDetectionSpec,
    ) -> None:
        self.config = config
        self.terrain = terrain
        self.peaks = peaks
        self.intrinsics = intrinsics
        self.pose_noise = pose_noise
        self.terrain_stride = terrain_stride
        self._terrain_spec = terrain_spec
        self._peak_detection = peak_detection
        self.renderer = SyntheticRenderer()
        self.detector: ContourDetector = DEFAULT_DETECTOR
        self.views: dict[str, View] = {}
        self._view_counter = 0
        self._solve_counter = 0

    @classmethod
    def from_settings(cls, settings: AppSettings, provider: ProviderKind | None = None) -> Scene:
        """Builds a scene from application settings.

        ``provider=None`` auto-selects: the REAL map (srtm) when DEM tiles are available (user
        preference for the app), else the synthetic demo terrain.  Tests pin ``provider="demo"``
        to stay deterministic and offline.
        """

        if provider is None:
            try:
                find_hgt_tile(DEFAULT_DEM_DIR)
                provider = "srtm"
            except (FileNotFoundError, StopIteration, ValueError):
                provider = "demo"
        config = SceneConfig(
            provider=provider,
            seed=settings.random_seed,
            image_width=settings.render.image_width,
            image_height=settings.render.image_height,
            horizontal_fov_deg=settings.render.horizontal_fov_deg,
            default_strategy="powell",
        )
        scene = cls(
            config=config,
            terrain=_build_terrain(config, settings.terrain),
            peaks=[],
            intrinsics=build_intrinsics(config.image_width, config.image_height, config.horizontal_fov_deg),
            pose_noise=settings.pose_noise,
            terrain_stride=settings.optimization.objective_terrain_stride,
            terrain_spec=settings.terrain,
            peak_detection=settings.peak_detection,
        )
        scene.peaks = scene._detect_peaks(scene.terrain)
        return scene

    def set_config(
        self,
        provider: ProviderKind,
        seed: int,
        image_width: int,
        image_height: int,
        horizontal_fov_deg: float,
        default_strategy: str,
    ) -> None:
        """Rebuilds the map and intrinsics, clearing all views."""

        self.config = SceneConfig(
            provider=provider,
            seed=seed,
            image_width=image_width,
            image_height=image_height,
            horizontal_fov_deg=horizontal_fov_deg,
            default_strategy=default_strategy,
        )
        self.terrain = _build_terrain(self.config, self._terrain_spec)
        self.peaks = self._detect_peaks(self.terrain)
        self.intrinsics = build_intrinsics(image_width, image_height, horizontal_fov_deg)
        self.views = {}
        self._view_counter = 0
        self._solve_counter = 0

    def focus_geo(self, lat_deg: float, lon_deg: float, extent_m: float = 24000.0) -> None:
        """Recenters the map on a geographic point (Copernicus mosaic), clearing views.

        This is how the app jumps to a GT sample's location: the `.hgt` provider only covers
        hand-downloaded tiles, while the Copernicus mosaic covers the whole corpus.
        """

        from peakle.terrain.copernicus import load_copernicus_terrain

        self.terrain = load_copernicus_terrain(lat_deg, lon_deg, extent_m=extent_m)
        self._geo_focused = True
        self.peaks = self._detect_peaks(self.terrain)
        self.views = {}

    def create_view(
        self,
        east_m: float,
        north_m: float,
        yaw_deg: float,
        pitch_deg: float,
        eye_height_m: float = DEFAULT_EYE_HEIGHT_M,
        label: str | None = None,
    ) -> View:
        """Places a camera, renders its image, and detects its contour."""

        self._view_counter += 1
        view_id = f"view-{self._view_counter:02d}"
        view = self._build_view(
            view_id, label or f"View {self._view_counter}", east_m, north_m, yaw_deg, pitch_deg, eye_height_m
        )
        self.views[view_id] = view
        return view

    def update_view(
        self,
        view_id: str,
        east_m: float | None = None,
        north_m: float | None = None,
        yaw_deg: float | None = None,
        pitch_deg: float | None = None,
        eye_height_m: float | None = None,
        label: str | None = None,
    ) -> View:
        """Edits a view's pose/label; re-renders and clears stale solves."""

        current = self.views[view_id]
        truth = current.true_extrinsics
        if truth is None:
            msg = f"view {view_id!r} has no placement to edit"
            raise ValueError(msg)
        rebuilt = self._build_view(
            view_id,
            label if label is not None else current.label,
            truth.position.east_m if east_m is None else east_m,
            truth.position.north_m if north_m is None else north_m,
            truth.yaw_deg if yaw_deg is None else yaw_deg,
            truth.pitch_deg if pitch_deg is None else pitch_deg,
            current.eye_height_m if eye_height_m is None else eye_height_m,
        )
        pose_unchanged = rebuilt.true_extrinsics == truth
        if pose_unchanged:
            rebuilt = rebuilt.model_copy(update={"solves": current.solves, "prior": current.prior})
        self.views[view_id] = rebuilt
        return rebuilt

    def delete_view(self, view_id: str) -> None:
        """Removes a view."""

        self.views.pop(view_id, None)

    def run_solve(self, view_id: str, strategy: str, params: dict[str, Any] | None = None) -> Solve:
        """Runs a solver against a view and stores the result."""

        view = self.views[view_id]
        if view.prior is None:
            msg = f"view {view_id!r} has no prior to solve from"
            raise ValueError(msg)
        params = dict(params or {})
        seed = params.get("seed")
        use_position_prior = bool(params.get("position_prior", True))
        self._solve_counter += 1
        result = solve_pose(
            terrain=self.terrain,
            contour=view.contour,
            intrinsics=view.intrinsics,
            prior=view.prior,
            strategy=strategy,
            terrain_stride=self.terrain_stride,
            truth=view.true_extrinsics,
            seed=seed,
            use_position_prior=use_position_prior,
        )
        solve = Solve(
            id=f"solve-{self._solve_counter:02d}",
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            strategy=strategy,
            params=params,
            prior=view.prior,
            result=result,
        )
        view.solves[solve.id] = solve
        return solve

    def _build_view(
        self,
        view_id: str,
        label: str,
        east_m: float,
        north_m: float,
        yaw_deg: float,
        pitch_deg: float,
        eye_height_m: float,
    ) -> View:
        up_m = self.terrain.elevation_at(east_m, north_m) + eye_height_m
        extrinsics = CameraExtrinsics(
            position=LocalPoint(east_m=east_m, north_m=north_m, up_m=up_m),
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            roll_deg=0.0,
        )
        render = self.renderer.render(self.terrain, self.intrinsics, extrinsics)
        contour = self.detector.detect(render)
        prior = self._view_prior(view_id, extrinsics)
        return View(
            id=view_id,
            label=label,
            intrinsics=self.intrinsics,
            true_extrinsics=extrinsics,
            eye_height_m=eye_height_m,
            prior=prior,
            contour=contour,
            render_arrays=render,
        )

    def _detect_peaks(self, terrain: TerrainMap) -> list[Peak]:
        """Detects prominent peaks; names them from OSM for real-DEM providers."""

        peaks = PeakDetector(self._peak_detection).detect(terrain)
        if (self.config.provider == "srtm" or getattr(self, "_geo_focused", False)) and peaks:
            pad = 0.01
            bbox = (
                float(terrain.latitude_deg.min()) - pad,
                float(terrain.longitude_deg.min()) - pad,
                float(terrain.latitude_deg.max()) + pad,
                float(terrain.longitude_deg.max()) + pad,
            )
            try:
                peaks = name_peaks_from_osm(peaks, bbox, DEFAULT_DEM_DIR)
            except Exception:  # noqa: BLE001 — naming is cosmetic; a focus must not fail on Overpass
                pass
        return peaks

    def _view_prior(self, view_id: str, extrinsics: CameraExtrinsics) -> PosePrior:
        index = int(view_id.rsplit("-", 1)[-1])
        rng = np.random.default_rng(self.config.seed + PRIOR_SEED_OFFSET + index)
        return noisy_prior(self.pose_noise, extrinsics.position, extrinsics.yaw_deg, extrinsics.pitch_deg, rng)


def _build_terrain(config: SceneConfig, base_spec: TerrainSpec) -> TerrainMap:
    spec = base_spec.model_copy(update={"seed": config.seed})
    return build_provider(config.provider, spec).generate()
