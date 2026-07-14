"""Mutable, in-memory workbench scene.

A `Scene` owns the elevation map (via a `MapProvider`), the detected peaks, the
default camera model/intrinsics, and the user-created `View`s. Views are placed
by clicking the map; each carries an image and solver-generated poses. State lives
in memory for the life of the server process (one scene per server).
"""

from __future__ import annotations

import hashlib
import threading
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

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
        contour: Legacy/default skyline contour used by older clients.
        evidence_contours: Explicit solver evidence tracks keyed by source id.
        evidence_metadata: Availability and provenance for each evidence track.
        default_evidence_source: Evidence track selected when a solve omits one.
        pitch_comparable: Whether physical pitch errors are meaningful for this image.
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
    evidence_contours: dict[str, SkylineContour] = Field(default_factory=dict)
    evidence_metadata: dict[str, dict[str, Any]] = Field(default_factory=dict)
    default_evidence_source: str = "view_contour"
    pitch_comparable: bool = True
    render_arrays: RenderArrays
    solves: dict[str, Solve] = {}
    # A view can carry a reference photograph and a known/prior pose. A placed synthetic view has
    # neither (source "placed"); a view materialized
    # from a GT sample carries the photo (reference_photo) and the corpus name (gt_name).
    source: str = "placed"
    gt_name: str | None = None
    reference_photo: Any = None  # PIL.Image for GT-derived views, else None
    # Hash of the exact terrain grid and solver stride used to produce pose results.
    # Persisted solves are rejected when this changes.
    terrain_fingerprint: str | None = None


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
        self.high_resolution_patch: Any = None
        self.terrain_fingerprint = _terrain_solver_fingerprint(terrain, terrain_stride, None)
        self.peaks = peaks
        self.intrinsics = intrinsics
        self.pose_noise = pose_noise
        self.terrain_stride = terrain_stride
        self._terrain_spec = terrain_spec
        self._peak_detection = peak_detection
        self.renderer = SyntheticRenderer()
        self.detector: ContourDetector = DEFAULT_DETECTOR
        self._mutation_lock = threading.RLock()
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

        with self._mutation_lock:
            self.config = SceneConfig(
                provider=provider,
                seed=seed,
                image_width=image_width,
                image_height=image_height,
                horizontal_fov_deg=horizontal_fov_deg,
                default_strategy=default_strategy,
            )
            self.terrain = _build_terrain(self.config, self._terrain_spec)
            self.high_resolution_patch = None
            self.terrain_fingerprint = _terrain_solver_fingerprint(self.terrain, self.terrain_stride, None)
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

        from peakle.terrain.copernicus import focused_grid_for_extent, load_copernicus_terrain

        with self._mutation_lock:
            # Keep the solver/render source at native GLO-30 spacing. The API
            # independently decimates this grid for the browser mesh.
            native_grid = focused_grid_for_extent(extent_m, max_grid=2048)
            self.terrain = load_copernicus_terrain(lat_deg, lon_deg, extent_m=extent_m, grid=native_grid)
            self.high_resolution_patch = _cached_high_resolution_patch(lat_deg, lon_deg)
            self.terrain_fingerprint = _terrain_solver_fingerprint(
                self.terrain,
                self.terrain_stride,
                self.high_resolution_patch,
            )
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

        with self._mutation_lock:
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

        with self._mutation_lock:
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
            # Rebuild with the VIEW's own intrinsics/source/photo. Photo-backed views keep their
            # immutable image evidence when the candidate pose is edited; placed synthetic views
            # get a newly rendered/detected contour.
            keep_evidence = current.source in {"gt", "photo"}
            rebuilt = self._construct_view(
                view_id,
                label if label is not None else current.label,
                current.intrinsics,
                extrinsics,
                eye,
                image_camera=current.image_camera,
                contour=current.contour if keep_evidence else None,
                evidence_contours=current.evidence_contours if keep_evidence else None,
                evidence_metadata=current.evidence_metadata if keep_evidence else None,
                default_evidence_source=current.default_evidence_source if keep_evidence else None,
                pitch_comparable=current.pitch_comparable,
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

        with self._mutation_lock:
            src = self.views[view_id]
            self._view_counter += 1
            new_id = f"view-{self._view_counter:02d}"
            dup = src.model_copy(update={"id": new_id, "label": label or f"{src.label} copy", "solves": {}})
            self.views[new_id] = dup
            return dup

    def delete_view(self, view_id: str) -> None:
        """Removes a view."""

        with self._mutation_lock:
            self.views.pop(view_id, None)

    def attach_solves(self, view_id: str, solves: list[Solve]) -> View:
        """Attach persisted solves to a view, keeping existing solves."""

        with self._mutation_lock:
            view = self.views[view_id]
            for solve in solves:
                view.solves.setdefault(solve.id, solve)
                self._solve_counter = max(self._solve_counter, _solve_index(solve.id))
            return view

    def run_solve(self, view_id: str, strategy: str, params: dict[str, Any] | None = None) -> Solve:
        """Runs a solver against a view and stores the result."""

        params = dict(params or {})
        seed = params.get("seed")
        use_position_prior = bool(params.get("position_prior", True))
        use_orientation_prior = bool(params.get("orientation_prior", True))
        prior_source = str(params.get("prior_source") or "metadata")
        with self._mutation_lock:
            view = self.views[view_id]
            if view.prior is None:
                msg = f"view {view_id!r} has no prior to solve from"
                raise ValueError(msg)
            terrain = self.terrain
            terrain_patch = self.high_resolution_patch
            terrain_stride = self.terrain_stride
            evidence_source = str(params.get("evidence_source") or view.default_evidence_source)
            contour = _resolve_evidence_contour(view, evidence_source)
            evidence_provenance = dict(view.evidence_metadata.get(evidence_source, {}))
            intrinsics = view.intrinsics
            prior = _resolve_solve_prior(view, prior_source)
            truth = view.true_extrinsics
            projection = view.image_camera.projection
            horizontal_fov_deg = view.image_camera.horizontal_fov_deg
            pitch_comparable = view.pitch_comparable

        # Persist the resolved sources, including defaults, so a result always says exactly which
        # prior and image evidence produced it.
        params["prior_source"] = prior_source
        params["evidence_source"] = evidence_source
        params["evidence_provenance"] = evidence_provenance
        params["evidence_contour_sha256"] = hashlib.sha256(contour.model_dump_json().encode()).hexdigest()
        params["pitch_comparable"] = pitch_comparable
        params["terrain_provenance"] = _solver_terrain_provenance(terrain, terrain_patch)

        result = solve_pose(
            terrain=terrain,
            contour=contour,
            intrinsics=intrinsics,
            prior=prior,
            strategy=strategy,
            terrain_stride=terrain_stride,
            truth=truth,
            seed=seed,
            use_position_prior=use_position_prior,
            use_orientation_prior=use_orientation_prior,
            projection=projection,
            horizontal_fov_deg=horizontal_fov_deg,
            terrain_patch=terrain_patch,
        )
        if truth is not None and not pitch_comparable:
            metrics = result.estimate.metrics.model_copy(update={"pitch_error_deg": None})
            estimate = result.estimate.model_copy(update={"metrics": metrics})
            result = result.model_copy(update={"estimate": estimate})
        with self._mutation_lock:
            if self.views.get(view_id) is not view:
                msg = f"view {view_id!r} changed before the solve finished"
                raise ValueError(msg)
            self._solve_counter += 1
            solve = Solve(
                id=f"solve-{self._solve_counter:02d}",
                created_at=datetime.now(UTC).isoformat(timespec="seconds"),
                strategy=strategy,
                params=params,
                prior=prior,
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
        prior: PosePrior | None = None,
        evidence_contours: dict[str, SkylineContour] | None = None,
        evidence_metadata: dict[str, dict[str, Any]] | None = None,
        default_evidence_source: str | None = None,
        pitch_comparable: bool = True,
    ) -> View:
        """Add a photo-backed View at a known position (call after focus_geo).

        Backs both a materialized GT sample (``source="gt"``) and an arbitrary uploaded photo
        (``source="photo"``). Callers may provide an explicit prior; uploaded photos otherwise get
        an exact-position, broad-orientation prior. From here it is an ordinary View—it lists,
        POVs, adjusts and solves like any placed view.
        """

        with self._mutation_lock:
            self._view_counter += 1
            view_id = f"view-{self._view_counter:02d}"
            eye_height_m = extrinsics.position.up_m - self.terrain.elevation_at(
                extrinsics.position.east_m, extrinsics.position.north_m
            )
            if prior is None:
                prior = PosePrior(
                    position=extrinsics.position,
                    yaw_deg=extrinsics.yaw_deg,
                    pitch_deg=extrinsics.pitch_deg,
                    horizontal_sigma_m=1.0,
                    vertical_sigma_m=1.0,
                    yaw_sigma_deg=120.0,
                    pitch_sigma_deg=20.0,
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
                evidence_contours=evidence_contours,
                evidence_metadata=evidence_metadata,
                default_evidence_source=default_evidence_source,
                pitch_comparable=pitch_comparable,
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
        prior: PosePrior | None = None,
        evidence_contours: dict[str, SkylineContour] | None = None,
        evidence_metadata: dict[str, dict[str, Any]] | None = None,
        default_evidence_source: str = "photo_auto",
        pitch_comparable: bool = False,
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
            prior=prior,
            evidence_contours=evidence_contours,
            evidence_metadata=evidence_metadata,
            default_evidence_source=default_evidence_source,
            pitch_comparable=pitch_comparable,
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
        evidence_contours: dict[str, SkylineContour] | None = None,
        evidence_metadata: dict[str, dict[str, Any]] | None = None,
        default_evidence_source: str | None = None,
        pitch_comparable: bool = True,
    ) -> View:
        """Render the DEM at ``extrinsics`` and assemble a View (contour detected if not given)."""

        render = self.renderer.render(self.terrain, intrinsics, extrinsics)
        if prior is None:
            prior = self._view_prior(view_id, extrinsics)
        selected_contour = contour if contour is not None else self.detector.detect(render)
        selected_source = default_evidence_source or _default_evidence_source(source)
        tracks = dict(evidence_contours) if evidence_contours is not None else {selected_source: selected_contour}
        metadata = dict(evidence_metadata or {})
        metadata.setdefault(
            selected_source,
            {
                "available": selected_source in tracks,
                "diagnostic": False,
                "source": selected_contour.source or selected_source,
            },
        )
        return View(
            id=view_id,
            label=label,
            intrinsics=intrinsics,
            image_camera=image_camera or CameraModel.from_intrinsics(intrinsics),
            true_extrinsics=extrinsics,
            eye_height_m=eye_height_m,
            prior=prior,
            contour=selected_contour,
            evidence_contours=tracks,
            evidence_metadata=metadata,
            default_evidence_source=selected_source,
            pitch_comparable=pitch_comparable,
            render_arrays=render,
            source=source,
            gt_name=gt_name,
            reference_photo=reference_photo,
            terrain_fingerprint=self.terrain_fingerprint,
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


def _solve_index(solve_id: str) -> int:
    try:
        return int(solve_id.rsplit("-", 1)[-1])
    except ValueError:
        return 0


def _default_evidence_source(view_source: str) -> str:
    if view_source == "photo":
        return "photo_auto"
    if view_source == "gt":
        return "photo_auto"
    return "rendered_skyline"


def _resolve_evidence_contour(view: View, source: str) -> SkylineContour:
    """Resolve one explicitly named contour without substituting another track."""

    normalized = source.strip()
    metadata = view.evidence_metadata.get(normalized, {})
    contour = view.evidence_contours.get(normalized)
    # Compatibility for views constructed before evidence tracks were introduced. New scene
    # constructors always populate ``evidence_contours``.
    if contour is None and not view.evidence_contours and normalized in {"view_contour", "rendered_skyline"}:
        contour = view.contour
    if metadata.get("available") is False or contour is None or not contour.points:
        available = sorted(
            key
            for key, candidate in view.evidence_contours.items()
            if candidate.points and view.evidence_metadata.get(key, {}).get("available") is not False
        )
        suffix = f"; available sources: {', '.join(available)}" if available else "; no usable evidence is available"
        msg = f"evidence source {normalized!r} is unavailable for view {view.id!r}{suffix}"
        raise ValueError(msg)
    return contour


def _resolve_solve_prior(view: View, source: str) -> PosePrior:
    """Resolve the requested prior source to a concrete PosePrior.

    The solve persists this resolved prior, so later debugging does not depend on
    mutable UI state. ``metadata`` is the view's original prior; ``pose:truth``
    starts a non-GT view from its synthetic/baseline pose; ``pose:solve:<id>`` starts from
    a previous solver pose while reusing the view prior's uncertainty.
    """

    base = view.prior
    if base is None:
        msg = f"view {view.id!r} has no prior"
        raise ValueError(msg)
    normalized = source.strip()
    if normalized in {"", "metadata", "view", "prior"}:
        return base
    if normalized in {"truth", "baseline", "refined", "pose:truth"}:
        if view.source == "gt":
            msg = "GT evaluation reference cannot be reused as a solver prior"
            raise ValueError(msg)
        if view.true_extrinsics is None:
            msg = f"view {view.id!r} has no baseline pose for prior source {source!r}"
            raise ValueError(msg)
        return _prior_from_extrinsics(base, view.true_extrinsics)
    solve_id = normalized.removeprefix("pose:")
    if solve_id.startswith("solve:"):
        solve_id = solve_id.removeprefix("solve:")
    if solve_id in view.solves:
        return _prior_from_extrinsics(base, view.solves[solve_id].estimate.extrinsics)
    msg = f"unknown prior source {source!r} for view {view.id!r}"
    raise ValueError(msg)


def _prior_from_extrinsics(template: PosePrior, extrinsics: CameraExtrinsics) -> PosePrior:
    return PosePrior(
        position=extrinsics.position,
        yaw_deg=extrinsics.yaw_deg,
        pitch_deg=extrinsics.pitch_deg,
        horizontal_sigma_m=template.horizontal_sigma_m,
        vertical_sigma_m=template.vertical_sigma_m,
        yaw_sigma_deg=template.yaw_sigma_deg,
        pitch_sigma_deg=template.pitch_sigma_deg,
    )


def _terrain_solver_fingerprint(terrain: TerrainMap, terrain_stride: int, terrain_patch: Any = None) -> str:
    """Hash the exact terrain geometry and sampling policy consumed by a solve."""

    digest = hashlib.sha256()
    digest.update(terrain.spec.model_dump_json().encode())
    digest.update(str(terrain_stride).encode())
    for values in (terrain.x_m, terrain.y_m, terrain.elevation_m):
        digest.update(np.ascontiguousarray(values, dtype=np.float64).view(np.uint8))
    if terrain_patch is not None:
        digest.update(b"native-near-field-patch")
        for values in (terrain_patch.x_m, terrain_patch.y_m, terrain_patch.elevation_m):
            digest.update(np.ascontiguousarray(values, dtype=np.float64).view(np.uint8))
    return digest.hexdigest()


def _solver_terrain_provenance(terrain: TerrainMap, terrain_patch: Any = None) -> dict[str, Any]:
    far_spacing_m = float(min(abs(terrain.x_m[1] - terrain.x_m[0]), abs(terrain.y_m[1] - terrain.y_m[0])))
    near_spacing_m = None
    if terrain_patch is not None and len(terrain_patch.x_m) > 1 and len(terrain_patch.y_m) > 1:
        near_spacing_m = float(
            min(abs(terrain_patch.x_m[1] - terrain_patch.x_m[0]), abs(terrain_patch.y_m[1] - terrain_patch.y_m[0]))
        )
    return {
        "far_grid_spacing_m": round(far_spacing_m, 3),
        "native_near_patch": terrain_patch is not None,
        "near_grid_spacing_m": round(near_spacing_m, 3) if near_spacing_m is not None else None,
        "native_near_patch_used_by": ["horizon", "cmaes_horizon_seed"] if terrain_patch is not None else [],
        "full_pose_objective_uses_native_near_patch": False,
    }


def _cached_high_resolution_patch(lat_deg: float, lon_deg: float) -> Any:
    """Load the finest cached Swiss terrain without triggering network access."""

    from peakle.localize.paths import SWISS_DIR
    from peakle.localize.swissdem import in_switzerland, load_swiss_patch

    if not in_switzerland(lat_deg, lon_deg):
        return None
    try:
        return load_swiss_patch(SWISS_DIR, lat_deg, lon_deg, res=2.0, radius_m=4500.0)
    except Exception:
        return None
