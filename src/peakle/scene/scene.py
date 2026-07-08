"""Mutable, in-memory workbench scene.

A `Scene` owns the elevation map (via a `MapProvider`), the detected peaks, the
default camera model/intrinsics, and the user-created `View`s. Views are placed
by clicking the map; each carries an image and solver-generated poses. State lives
in memory for the life of the server process (one scene per server).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from peakle.config import AppSettings, PoseNoiseSettings
from peakle.contours import DEFAULT_DETECTOR, ContourDetector
from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
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
    """A localizable view: image/crop/photo plus camera model and pose candidates.

    Attributes:
        id: Stable view identifier.
        label: Display label.
        intrinsics: Pinhole intrinsics used by the DEM renderer.
        image_camera: Camera model for the source image/crop/photo.
        true_extrinsics: Baseline pose (None for a real photo with no known pose).
        eye_height_m: Baseline pose height above the terrain surface.
        prior: Fixed pose prior shared by every solver run for this view.
        contour: Detected skyline contour.
        render_arrays: In-memory render outputs (image, mask, profile).
        solves: Solver-generated pose candidates keyed by id.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    label: str
    intrinsics: CameraIntrinsics
    image_camera: CameraModel
    true_extrinsics: CameraExtrinsics | None
    eye_height_m: float
    prior: PosePrior | None
    contour: SkylineContour
    render_arrays: RenderArrays
    solves: dict[str, Solve] = {}
    # A view can carry a reference photograph and a known/prior pose. A placed synthetic view has
    # neither (source "placed"); a view materialized
    # from a GT sample carries the photo (reference_photo) and the corpus name (gt_name).
    source: str = "placed"
    gt_name: str | None = None
    reference_photo: Any = None  # PIL.Image for GT-derived views, else None


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
            except FileNotFoundError, StopIteration, ValueError:
                provider = "demo"
        config = SceneConfig(
            provider=provider,
            seed=settings.random_seed,
            image_width=settings.render.image_width,
            image_height=settings.render.image_height,
            horizontal_fov_deg=settings.render.horizontal_fov_deg,
            default_strategy="horizon",
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

    def focus_geo(self, lat_deg: float, lon_deg: float, extent_m: float = 40000.0) -> None:
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
        eye = current.eye_height_m if eye_height_m is None else eye_height_m
        e_m = truth.position.east_m if east_m is None else east_m
        n_m = truth.position.north_m if north_m is None else north_m
        up_m = self.terrain.elevation_at(e_m, n_m) + eye
        extrinsics = CameraExtrinsics(
            position=LocalPoint(east_m=e_m, north_m=n_m, up_m=up_m),
            yaw_deg=truth.yaw_deg if yaw_deg is None else yaw_deg,
            pitch_deg=truth.pitch_deg if pitch_deg is None else pitch_deg,
            roll_deg=0.0,
        )
        # Rebuild with the VIEW's own intrinsics/source/photo (a GT view keeps its sample intrinsics
        # and photograph when its pose is edited); the contour is re-detected from the new render
        # for placed views, but a GT view keeps its photo-derived observed contour.
        rebuilt = self._construct_view(
            view_id,
            label if label is not None else current.label,
            current.intrinsics,
            extrinsics,
            eye,
            image_camera=current.image_camera,
            contour=current.contour if current.source == "gt" else None,
            source=current.source,
            gt_name=current.gt_name,
            reference_photo=current.reference_photo,
        )
        if rebuilt.true_extrinsics == truth:
            rebuilt = rebuilt.model_copy(update={"solves": current.solves, "prior": current.prior})
        self.views[view_id] = rebuilt
        return rebuilt

    def duplicate_view(self, view_id: str, label: str | None = None) -> View:
        """Copy a view (any kind) under a new id + label, without its solves.

        The duplicate keeps the source view's pose, intrinsics, source, gt_name and reference photo,
        so you can fork a placed camera OR a GT-derived view, rename it, and move the copy freely
        while the original stays put.
        """

        src = self.views[view_id]
        self._view_counter += 1
        new_id = f"view-{self._view_counter:02d}"
        dup = src.model_copy(update={"id": new_id, "label": label or f"{src.label} copy", "solves": {}})
        self.views[new_id] = dup
        return dup

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
        use_orientation_prior = bool(params.get("orientation_prior", True))
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
            use_orientation_prior=use_orientation_prior,
            projection=view.image_camera.projection,
            horizontal_fov_deg=view.image_camera.horizontal_fov_deg,
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

    def add_backed_view(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        contour: SkylineContour,
        reference_photo: Any,
        *,
        source: str,
        gt_name: str | None = None,
        label: str | None = None,
        image_camera: CameraModel | None = None,
    ) -> View:
        """Add a photo-backed View at a known position (call after focus_geo).

        Backs both a materialized GT sample (``source="gt"``) and an arbitrary uploaded photo
        (``source="photo"``): the view carries the photograph and the observed skyline extracted
        from it, with an EXACT-position prior (the position is known — GPS — so the horizon solver
        recovers orientation there; a noisy prior would flip the yaw). From here it is an ordinary
        View — it lists, POVs, adjusts and solves like any placed view.
        """

        self._view_counter += 1
        view_id = f"view-{self._view_counter:02d}"
        eye_height_m = extrinsics.position.up_m - self.terrain.elevation_at(
            extrinsics.position.east_m, extrinsics.position.north_m
        )
        prior = PosePrior(
            position=extrinsics.position,
            yaw_deg=extrinsics.yaw_deg,
            pitch_deg=extrinsics.pitch_deg,
            horizontal_sigma_m=1.0,
            vertical_sigma_m=1.0,
            yaw_sigma_deg=120.0 if source == "photo" else 1.0,
            pitch_sigma_deg=20.0 if source == "photo" else 1.0,
        )
        view = self._construct_view(
            view_id,
            label or gt_name or f"Photo {self._view_counter}",
            intrinsics,
            extrinsics,
            eye_height_m,
            image_camera=image_camera,
            contour=contour,
            source=source,
            gt_name=gt_name,
            reference_photo=reference_photo,
            prior=prior,
        )
        self.views[view_id] = view
        return view

    def add_gt_view(
        self,
        name: str,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        contour: SkylineContour,
        reference_photo: Any,
        label: str | None = None,
        image_camera: CameraModel | None = None,
    ) -> View:
        """Materialize a GT corpus sample as a scene View (thin wrapper over add_backed_view)."""

        return self.add_backed_view(
            intrinsics,
            extrinsics,
            contour,
            reference_photo,
            source="gt",
            gt_name=name,
            label=label or name,
            image_camera=image_camera,
        )

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
        return self._construct_view(view_id, label, self.intrinsics, extrinsics, eye_height_m)

    def _construct_view(
        self,
        view_id: str,
        label: str,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        eye_height_m: float,
        *,
        image_camera: CameraModel | None = None,
        contour: SkylineContour | None = None,
        source: str = "placed",
        gt_name: str | None = None,
        reference_photo: Any = None,
        prior: PosePrior | None = None,
    ) -> View:
        """Render the DEM at ``extrinsics`` and assemble a View (contour detected if not given)."""

        render = self.renderer.render(self.terrain, intrinsics, extrinsics)
        if prior is None:
            prior = self._view_prior(view_id, extrinsics)
        return View(
            id=view_id,
            label=label,
            intrinsics=intrinsics,
            image_camera=image_camera or CameraModel.from_intrinsics(intrinsics),
            true_extrinsics=extrinsics,
            eye_height_m=eye_height_m,
            prior=prior,
            contour=contour if contour is not None else self.detector.detect(render),
            render_arrays=render,
            source=source,
            gt_name=gt_name,
            reference_photo=reference_photo,
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
                peaks = name_peaks_from_osm(peaks, bbox, DEFAULT_DEM_DIR, terrain=terrain)
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
