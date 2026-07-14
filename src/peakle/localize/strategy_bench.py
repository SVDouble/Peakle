"""Honest GeoPose workbench-strategy benchmark support.

The older :mod:`peakle.localize.bench` benchmark validates the specialised
orientation-only horizon solver at a known position.  This module exercises the
actual workbench strategies on the same GeoPose evidence without quietly using
the evaluation pose as an exact prior.

Four prior regimes are deliberately kept separate:

``raw_metadata``
    Exact published/refined source metadata (``info.txt`` lines 3-6).  This is
    a *pose-retention sanity check* and is never ranking-eligible because
    ``keep-prior`` has zero reference error by construction.  Optional original
    noisy metadata from lines 7-10 is diagnostic-only and never enters a prior.
``perturbed_metadata``
    A deterministic, recorded position + orientation perturbation.  This is the
    fair prior-assisted comparison, including a ``keep-prior`` baseline with the
    identical perturbation.
``position_only``
    The same perturbed position, but yaw/pitch are neutral and the orientation
    prior is disabled.  No reference orientation reaches the solver.
``none``
    A neutral pose at the DEM-window centre with both priors disabled.  The DEM
    window itself is deterministically offset from the reference location, so
    the map centre does not leak the answer.  This remains *regional* no-prior
    localization, not country-scale retrieval.

The PFM oracle and automatic-photo tracks use one fixed contour per sample, so
strategies see identical evidence.  GT/depth-to-DEM compatibility is computed
independently before any solve and is the gate for the primary ranking.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median
from typing import Any, Literal

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.domain.contours import ImagePoint, SkylineContour
from peakle.domain.coordinates import GeoPoint, LocalFrame, LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.localize.compatibility import (
    COMPATIBILITY_POLICY,
    HEIGHT_COMPATIBILITY_POLICY,
    gt_dem_compatibility,
    raw_camera_clearance_compatibility,
)
from peakle.localize.correspondence import (
    DenseMatcher,
    MatcherUnavailable,
    SiftMatcher,
    WorkerMatcher,
    load_model_manifest,
)
from peakle.localize.extract import best_skyline_candidate, extract_candidates
from peakle.localize.geopose import GeoPoseSample, load_sample, resampled_oracle_skyline
from peakle.localize.outline_score import rows_to_mask
from peakle.localize.paths import BASE, COP_TILES_DIR, STD_WIDTH, SWISS_DIR
from peakle.localize.photo_support import edge_mask, family_support
from peakle.localize.pnp import PoseRansacConfig
from peakle.localize.render_match_pnp import (
    CandidateValidationConfig,
    RenderMatchConfig,
    RenderMatchPoseResult,
    solve_render_match_pose,
)
from peakle.localize.solve import HorizonProfile
from peakle.localize.swissdem import Patch, in_switzerland, load_swiss_patch
from peakle.optimization.solve import PoseSolveResult, solve_pose
from peakle.rendering.orthophoto import (
    AppearanceRaster,
    MissingOrthophotoTiles,
    SwissImageProvider,
)
from peakle.rendering.terrain_view import RenderModality
from peakle.terrain.copernicus import load_copernicus_terrain

Algorithm = Literal[
    "keep-prior",
    "horizon",
    "contourdb",
    "cmaes",
    "powell",
    "nelder",
    "evolution",
    "global",
    "render-pnp",
]
EvidenceTrackName = Literal["pfm_oracle", "photo_auto", "photo_rgb"]
PriorRegimeName = Literal["raw_metadata", "perturbed_metadata", "position_only", "none"]
RenderMatcherName = Literal["disabled", "sift", "worker"]
MATRIX_EXTRACTORS: tuple[str, ...] = ("color", "dexined", "sam3", "mobile_sam", "sam2")

ALGORITHMS: tuple[Algorithm, ...] = (
    "keep-prior",
    "horizon",
    "contourdb",
    "cmaes",
    "powell",
    "nelder",
    "evolution",
    "global",
    "render-pnp",
)
EVIDENCE_TRACKS: tuple[EvidenceTrackName, ...] = ("pfm_oracle", "photo_auto", "photo_rgb")
PRIOR_REGIMES: tuple[PriorRegimeName, ...] = (
    "raw_metadata",
    "perturbed_metadata",
    "position_only",
    "none",
)

# `core` compares algorithms only inside meaningful, shared regimes while
# keeping the default one-sample run finite.  `full` explores every executable
# pair, including local optimizers without priors.
CORE_MATRIX: dict[Algorithm, frozenset[PriorRegimeName]] = {
    "keep-prior": frozenset({"raw_metadata", "perturbed_metadata", "position_only"}),
    "horizon": frozenset({"position_only"}),
    "contourdb": frozenset({"position_only", "none"}),
    "cmaes": frozenset({"perturbed_metadata", "position_only"}),
    "powell": frozenset({"perturbed_metadata", "position_only"}),
    "nelder": frozenset({"perturbed_metadata", "position_only"}),
    "evolution": frozenset({"perturbed_metadata", "position_only", "none"}),
    "global": frozenset({"none"}),
    "render-pnp": frozenset({"perturbed_metadata", "position_only"}),
}
APPLICABLE_REGIMES: dict[Algorithm, frozenset[PriorRegimeName]] = {
    "keep-prior": frozenset({"raw_metadata", "perturbed_metadata", "position_only"}),
    "horizon": frozenset({"raw_metadata", "perturbed_metadata", "position_only"}),
    "contourdb": frozenset(PRIOR_REGIMES),
    "cmaes": frozenset(PRIOR_REGIMES),
    "powell": frozenset(PRIOR_REGIMES),
    "nelder": frozenset(PRIOR_REGIMES),
    "evolution": frozenset(PRIOR_REGIMES),
    "global": frozenset({"none"}),
    "render-pnp": frozenset({"raw_metadata", "perturbed_metadata", "position_only"}),
}

APPLICABLE_EVIDENCE: dict[Algorithm, frozenset[EvidenceTrackName]] = {
    "keep-prior": frozenset(EVIDENCE_TRACKS),
    "horizon": frozenset({"pfm_oracle", "photo_auto"}),
    "contourdb": frozenset({"pfm_oracle", "photo_auto"}),
    "cmaes": frozenset({"pfm_oracle", "photo_auto"}),
    "powell": frozenset({"pfm_oracle", "photo_auto"}),
    "nelder": frozenset({"pfm_oracle", "photo_auto"}),
    "evolution": frozenset({"pfm_oracle", "photo_auto"}),
    "global": frozenset({"pfm_oracle", "photo_auto"}),
    "render-pnp": frozenset({"photo_rgb"}),
}
CONTOUR_ALGORITHMS: frozenset[Algorithm] = frozenset(
    {"horizon", "contourdb", "cmaes", "powell", "nelder", "evolution", "global"}
)

SUCCESS_THRESHOLDS = {"horizontal_position_error_m_lte": 100.0, "absolute_yaw_error_deg_lte": 5.0}
MAX_IMAGE_WIDTH = STD_WIDTH


@dataclass(frozen=True)
class PerturbationBucket:
    """Fixed-magnitude perturbation and matching prior uncertainty."""

    horizontal_m: float
    vertical_m: float
    yaw_deg: float
    pitch_deg: float


PERTURBATION_BUCKETS: dict[str, PerturbationBucket] = {
    "mild": PerturbationBucket(horizontal_m=50.0, vertical_m=20.0, yaw_deg=5.0, pitch_deg=3.0),
    "standard": PerturbationBucket(horizontal_m=200.0, vertical_m=75.0, yaw_deg=15.0, pitch_deg=8.0),
    "hard": PerturbationBucket(horizontal_m=500.0, vertical_m=150.0, yaw_deg=35.0, pitch_deg=15.0),
}


@dataclass(frozen=True)
class MatrixConfig:
    """Configuration whose complete value is persisted in ``run.json``."""

    profile: str = "core"
    algorithms: tuple[Algorithm, ...] = ALGORITHMS
    evidence_tracks: tuple[EvidenceTrackName, ...] = EVIDENCE_TRACKS
    prior_regimes: tuple[PriorRegimeName, ...] = PRIOR_REGIMES
    perturbation_bucket: str = "standard"
    replicates: int = 1
    root_seed: int = 20260713
    extent_m: float = 40_000.0
    terrain_grid: int = 1335
    terrain_stride: int = 6
    extractor: str = "color"
    map_center_offset_fraction: float = 0.16
    render_matcher: RenderMatcherName = "disabled"
    render_modality: RenderModality = "hillshade"
    render_width_px: int = 320
    render_height_px: int = 256
    render_yaw_step_deg: float = 30.0
    render_refinement_passes: int = 1
    native_patch_stride: int = 8
    render_candidate_validation: CandidateValidationConfig = field(default_factory=CandidateValidationConfig)
    matcher_command: tuple[str, ...] = ()
    matcher_id: str = "external_matcher"
    matcher_manifest_path: str | None = None
    matcher_cache_dir: str | None = None
    orthophoto_cache_dir: str | None = None
    orthophoto_zoom: int = 14
    orthophoto_time: str = "current"
    orthophoto_max_tiles: int = 1600

    def validate(self) -> None:
        if self.profile not in {"core", "full"}:
            raise ValueError("profile must be 'core' or 'full'")
        unknown_algorithms = set(self.algorithms) - set(ALGORITHMS)
        if unknown_algorithms:
            raise ValueError(f"unknown algorithms: {sorted(unknown_algorithms)}")
        unknown_tracks = set(self.evidence_tracks) - set(EVIDENCE_TRACKS)
        if unknown_tracks:
            raise ValueError(f"unknown evidence tracks: {sorted(unknown_tracks)}")
        unknown_regimes = set(self.prior_regimes) - set(PRIOR_REGIMES)
        if unknown_regimes:
            raise ValueError(f"unknown prior regimes: {sorted(unknown_regimes)}")
        if self.perturbation_bucket not in PERTURBATION_BUCKETS:
            raise ValueError(f"unknown perturbation bucket {self.perturbation_bucket!r}")
        if self.replicates < 1:
            raise ValueError("replicates must be positive")
        if self.extent_m <= 5_000.0:
            raise ValueError("extent_m must exceed 5000")
        if self.terrain_grid < 64:
            raise ValueError("terrain_grid must be at least 64")
        if self.terrain_stride < 1:
            raise ValueError("terrain_stride must be positive")
        if not 0.05 <= self.map_center_offset_fraction <= 0.30:
            raise ValueError("map_center_offset_fraction must be in [0.05, 0.30]")
        if self.extractor not in MATRIX_EXTRACTORS:
            raise ValueError(f"extractor must be one of {', '.join(MATRIX_EXTRACTORS)}")
        if self.render_matcher not in {"disabled", "sift", "worker"}:
            raise ValueError("render_matcher must be disabled, sift, or worker")
        if self.render_modality not in {"hillshade", "normal", "relative_depth", "orthophoto"}:
            raise ValueError("unsupported render modality")
        if self.render_width_px < 64 or self.render_height_px < 64:
            raise ValueError("render matching dimensions must be at least 64 pixels")
        if not 5.0 <= self.render_yaw_step_deg <= 90.0:
            raise ValueError("render_yaw_step_deg must be in [5, 90]")
        if self.render_refinement_passes not in {0, 1}:
            raise ValueError("render_refinement_passes must be zero or one")
        if self.native_patch_stride < 1:
            raise ValueError("native_patch_stride must be positive")
        self.render_candidate_validation.validate()
        if self.render_matcher == "worker" and not self.matcher_command:
            raise ValueError("worker render matcher requires matcher_command")
        if self.render_matcher == "worker" and self.matcher_manifest_path is None:
            raise ValueError("worker render matcher requires matcher_manifest_path")
        if self.matcher_cache_dir is not None and self.render_matcher != "worker":
            raise ValueError("matcher_cache_dir is supported only for the external worker matcher")
        if self.render_modality == "orthophoto" and self.render_matcher != "disabled":
            if self.orthophoto_cache_dir is None:
                raise ValueError("orthophoto render matching requires orthophoto_cache_dir")
        if not 0 <= self.orthophoto_zoom <= 22:
            raise ValueError("orthophoto_zoom must be in [0, 22]")
        if self.orthophoto_max_tiles < 1:
            raise ValueError("orthophoto_max_tiles must be positive")


@dataclass(frozen=True)
class MatrixCell:
    algorithm: Algorithm
    prior_regime: PriorRegimeName
    evidence_track: EvidenceTrackName
    replicate: int
    applicable: bool
    skip_reason: str | None


@dataclass(frozen=True)
class PriorScenario:
    name: PriorRegimeName
    prior: PosePrior
    use_position_prior: bool
    use_orientation_prior: bool
    perturbation: dict[str, Any]
    contains_exact_reference: bool
    constructed_from_reference: bool


@dataclass(frozen=True)
class EvidenceTrack:
    name: EvidenceTrackName
    rows: np.ndarray | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TerrainSelection:
    """Regional estimator terrain plus evaluation-only reference inputs."""

    terrain: TerrainMap
    evaluation_high_resolution_patch: Patch | None
    truth: CameraExtrinsics
    provenance: dict[str, Any]


@dataclass(frozen=True)
class TerrainPatchProvision:
    """One cached native patch and its auditable centre/data lineage."""

    patch: Patch | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class EstimatorTerrainSelection:
    """Terrain inputs supplied to one estimator cell."""

    terrain: TerrainMap
    high_resolution_patch: Patch | None
    provenance: dict[str, Any]


@dataclass(frozen=True)
class RenderMatchResources:
    matcher: DenseMatcher
    appearance: AppearanceRaster | None
    config: RenderMatchConfig
    provenance: dict[str, Any]


def build_matrix_cells(config: MatrixConfig) -> list[MatrixCell]:
    """Return the complete requested matrix, including explicit skipped cells."""

    config.validate()
    cells: list[MatrixCell] = []
    matrix = CORE_MATRIX if config.profile == "core" else APPLICABLE_REGIMES
    for algorithm in config.algorithms:
        for regime in config.prior_regimes:
            repeats = config.replicates if regime in {"perturbed_metadata", "position_only"} else 1
            for replicate in range(repeats):
                for evidence in config.evidence_tracks:
                    applicable = regime in APPLICABLE_REGIMES[algorithm]
                    reason = None
                    if not applicable:
                        reason = "algorithm_requires_a_different_prior_regime"
                    elif regime not in matrix[algorithm]:
                        applicable = False
                        reason = "excluded_by_core_profile"
                    elif evidence not in APPLICABLE_EVIDENCE[algorithm]:
                        applicable = False
                        reason = "algorithm_requires_a_different_evidence_track"
                    elif algorithm == "render-pnp" and config.render_matcher == "disabled":
                        applicable = False
                        reason = "render_matcher_not_configured"
                    cells.append(
                        MatrixCell(
                            algorithm=algorithm,
                            prior_regime=regime,
                            evidence_track=evidence,
                            replicate=replicate,
                            applicable=applicable,
                            skip_reason=reason,
                        )
                    )
    return cells


def run_sample_matrix(
    sample_dir: Path,
    config: MatrixConfig,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the configured strategy matrix for one GeoPose sample."""

    config.validate()
    sample = load_sample(sample_dir)
    terrain_selection = load_benchmark_terrain(sample, config)
    rgb = _load_photo(sample.photo_path)
    height, width = rgb.shape[:2]
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, sample.fov_deg)
    evidence = _evidence_tracks(sample, rgb, config)
    compatibility, photo_edge_support = _pre_solve_quality(
        sample,
        terrain_selection,
        evidence,
        width,
        height,
    )
    scenarios = {
        (regime, replicate): build_prior_scenario(
            sample,
            terrain_selection.terrain,
            terrain_selection.truth,
            regime,
            config.perturbation_bucket,
            replicate,
            config.root_seed,
        )
        for regime in config.prior_regimes
        for replicate in range(config.replicates if regime in {"perturbed_metadata", "position_only"} else 1)
    }
    reference = _reference_record(sample, terrain_selection.truth)
    evidence_by_name = {track.name: track for track in evidence}
    cells = build_matrix_cells(config)
    render_resources: RenderMatchResources | None = None
    render_resource_error: str | None = None
    if any(cell.applicable and cell.algorithm == "render-pnp" for cell in cells):
        try:
            render_resources = _render_match_resources(terrain_selection.terrain, config)
        except (MatcherUnavailable, MissingOrthophotoTiles, OSError, ValueError) as exc:
            render_resource_error = f"{type(exc).__name__}: {str(exc)[:400]}"
    scenario_patch_cache: dict[tuple[PriorRegimeName, int], TerrainPatchProvision] = {}
    cases: list[dict[str, Any]] = []
    for cell in cells:
        scenario = scenarios[(cell.prior_regime, cell.replicate)]
        track = evidence_by_name[cell.evidence_track]
        estimator_will_run = (
            cell.applicable
            and cell.algorithm != "keep-prior"
            and not (cell.algorithm == "render-pnp" and render_resources is None)
            and not (track.rows is None and cell.algorithm in CONTOUR_ALGORITHMS)
        )
        estimator_terrain = _estimator_terrain_for_cell(
            terrain_selection.terrain,
            scenario,
            scenario_key=(cell.prior_regime, cell.replicate),
            patch_cache=scenario_patch_cache,
            provision_patch=estimator_will_run,
        )
        case = _base_case(
            sample,
            cell,
            scenario,
            track,
            reference,
            compatibility,
            photo_edge_support,
            native_patch_missing_from_objective=estimator_terrain.high_resolution_patch is not None,
            render_matcher=config.render_matcher,
            render_modality=config.render_modality,
        )
        case["terrain_inputs"] = estimator_terrain.provenance
        if not cell.applicable:
            case.update(status="skipped", skip_reason=cell.skip_reason, runtime_s=0.0)
        elif cell.algorithm == "render-pnp" and render_resources is None:
            case.update(
                status="skipped",
                skip_reason="render_resources_unavailable",
                runtime_s=0.0,
                resource_error=render_resource_error,
            )
        elif track.rows is None and cell.algorithm in CONTOUR_ALGORITHMS:
            case.update(
                status="ok",
                outcome="evidence_rejected",
                runtime_s=0.0,
                result=None,
                errors=None,
                success={"value": False, "thresholds": SUCCESS_THRESHOLDS, "reason": "no_usable_evidence"},
            )
        else:
            started = time.perf_counter()
            try:
                if cell.algorithm == "keep-prior":
                    estimate = CameraExtrinsics(
                        position=scenario.prior.position,
                        yaw_deg=_wrap180(scenario.prior.yaw_deg),
                        pitch_deg=float(np.clip(scenario.prior.pitch_deg, -89.0, 89.0)),
                        roll_deg=0.0,
                    )
                    case.update(_baseline_result(estimate, terrain_selection.truth))
                elif cell.algorithm == "render-pnp":
                    if render_resources is None:
                        raise RuntimeError("render-PnP cell reached without provisioned resources")
                    seed = _stable_seed(
                        config.root_seed,
                        sample.name,
                        cell.algorithm,
                        cell.prior_regime,
                        cell.evidence_track,
                        cell.replicate,
                    )
                    query_camera = CameraModel(
                        width_px=width,
                        height_px=height,
                        horizontal_fov_deg=sample.fov_deg,
                        projection="cyltan",
                    )
                    solved_render = solve_render_match_pose(
                        estimator_terrain.terrain,
                        rgb,
                        query_camera,
                        scenario.prior,
                        render_resources.matcher,
                        use_position_prior=scenario.use_position_prior,
                        use_orientation_prior=scenario.use_orientation_prior,
                        appearance=render_resources.appearance,
                        native_elevation_patch=estimator_terrain.high_resolution_patch,
                        config=replace_render_seed(render_resources.config, seed),
                        seed=seed,
                    )
                    case.update(_render_solver_result(solved_render, terrain_selection.truth, seed))
                    if _render_result_uses_native_patch(solved_render):
                        exclusion = "workbench_objective_does_not_consume_native_high_resolution_patch"
                        case["ranking_exclusions"] = [item for item in case["ranking_exclusions"] if item != exclusion]
                        case["ranking_eligible"] = not case["ranking_exclusions"]
                else:
                    # The evidence-rejection branch above has already handled
                    # None; keep the invariant explicit for type checkers and
                    # future refactors.
                    if track.rows is None:
                        raise RuntimeError("solver cell reached without evidence rows")
                    contour = _rows_contour(track.rows, width, height, source=cell.evidence_track)
                    seed = _stable_seed(
                        config.root_seed,
                        sample.name,
                        cell.algorithm,
                        cell.prior_regime,
                        cell.evidence_track,
                        cell.replicate,
                    )
                    solved = solve_pose(
                        terrain=estimator_terrain.terrain,
                        contour=contour,
                        intrinsics=intrinsics,
                        prior=scenario.prior,
                        strategy=cell.algorithm,
                        terrain_stride=config.terrain_stride,
                        truth=None,
                        seed=seed,
                        use_position_prior=scenario.use_position_prior,
                        use_orientation_prior=scenario.use_orientation_prior,
                        projection="cyltan",
                        horizontal_fov_deg=sample.fov_deg,
                        terrain_patch=estimator_terrain.high_resolution_patch,
                    )
                    case.update(_solver_result(solved, terrain_selection.truth, seed))
                case["runtime_s"] = round(time.perf_counter() - started, 4)
            except Exception as exc:  # a single experimental cell must not abort the matrix
                case.update(
                    status="error",
                    error={"type": type(exc).__name__, "message": str(exc)[:500]},
                    runtime_s=round(time.perf_counter() - started, 4),
                    success={"value": False, "thresholds": SUCCESS_THRESHOLDS, "reason": "solver_error"},
                )
        case["original_metadata_diagnostic"] = _original_metadata_diagnostic(
            sample,
            terrain_selection.terrain,
            terrain_selection.truth,
            estimate=_case_estimate(case),
        )
        cases.append(case)
        if progress is not None:
            progress(case)

    _attach_shared_baselines(cases)
    sample_row = _legacy_sample_row(
        sample,
        terrain_selection,
        compatibility,
        photo_edge_support,
        evidence,
        cases,
    )
    return sample_row, cases


def load_benchmark_terrain(sample: GeoPoseSample, config: MatrixConfig) -> TerrainSelection:
    """Load the regional estimator map and an isolated evaluation-only patch."""

    center_east, center_north = deterministic_map_center_offset(sample.name, config)
    sample_frame = LocalFrame(origin=GeoPoint(latitude_deg=sample.lat, longitude_deg=sample.lon, elevation_m=0.0))
    center_geo = sample_frame.local_to_geo(LocalPoint(east_m=center_east, north_m=center_north, up_m=0.0))
    terrain = load_copernicus_terrain(
        center_geo.latitude_deg,
        center_geo.longitude_deg,
        extent_m=config.extent_m,
        grid=config.terrain_grid,
        tile_dir=COP_TILES_DIR,
    )
    raw_local = terrain.frame.geo_to_local(
        GeoPoint(latitude_deg=sample.lat, longitude_deg=sample.lon, elevation_m=sample.elev_m)
    )
    # The independent reference and compatibility metric must use the source
    # metadata elevation verbatim.  Clamping it to the DEM surface would mutate
    # the label in exactly the map-dependent direction the metric is meant to
    # detect.
    truth_position = raw_local.model_copy(update={"up_m": sample.elev_m})
    truth = CameraExtrinsics(
        position=truth_position,
        yaw_deg=_wrap180(sample.yaw_gt_deg),
        pitch_deg=sample.pitch_gt_deg,
        roll_deg=sample.roll_gt_deg,
    )
    evaluation_patch = _cached_patch_at_local_position(
        terrain,
        truth.position,
        center_source="evaluation_reference_pose",
        uses_reference_truth=True,
        used_by_estimator=False,
    )
    grid_east_m = float(np.median(np.diff(terrain.x_m)))
    grid_north_m = float(np.median(np.diff(terrain.y_m)))
    provenance = {
        "far_source": "Copernicus GLO-30",
        "far_nominal_resolution_m": 30.0,
        "search_grid_spacing_m": round(max(abs(grid_east_m), abs(grid_north_m)), 3),
        "extent_m": config.extent_m,
        "grid": config.terrain_grid,
        "estimator_search_grid_reference_patch_fused": False,
        "estimator_search_grid_note": (
            "The shared regional estimator map is Copernicus-only. Native Swiss terrain is provisioned separately "
            "per solver cell from its supplied position prior."
        ),
        "evaluation_reference_patch": evaluation_patch.provenance,
        "map_center_offset_from_reference_m": {"east": round(center_east, 3), "north": round(center_north, 3)},
        "reference_offset_from_map_center_m": {
            "east": round(truth_position.east_m, 3),
            "north": round(truth_position.north_m, 3),
        },
    }
    return TerrainSelection(
        terrain=terrain,
        evaluation_high_resolution_patch=evaluation_patch.patch,
        truth=truth,
        provenance=provenance,
    )


def deterministic_map_center_offset(sample_name: str, config: MatrixConfig) -> tuple[float, float]:
    """Offset the DEM centre from truth so regional no-prior runs do not get a centre hint."""

    seed = _stable_seed(config.root_seed, sample_name, "map_center")
    angle = np.random.default_rng(seed).uniform(0.0, 2.0 * math.pi)
    radius = config.extent_m * config.map_center_offset_fraction
    return radius * math.sin(angle), radius * math.cos(angle)


def fuse_high_resolution_patch(terrain: TerrainMap, patch: Patch) -> tuple[TerrainMap, int]:
    """Replace regular-grid cells covered by a cached higher-resolution source."""

    col_mask = (terrain.x_m >= patch.x_m[0]) & (terrain.x_m <= patch.x_m[-1])
    row_mask = (terrain.y_m >= patch.y_m[0]) & (terrain.y_m <= patch.y_m[-1])
    cols = np.flatnonzero(col_mask)
    rows = np.flatnonzero(row_mask)
    if not cols.size or not rows.size:
        return terrain, 0
    x_grid, y_grid = np.meshgrid(terrain.x_m[cols], terrain.y_m[rows])
    patch_col = np.interp(x_grid.ravel(), patch.x_m, np.arange(len(patch.x_m), dtype=float))
    patch_row = np.interp(y_grid.ravel(), patch.y_m, np.arange(len(patch.y_m), dtype=float))
    sampled = map_coordinates(
        np.asarray(patch.elevation_m, dtype=float),
        [patch_row, patch_col],
        order=1,
        mode="constant",
        cval=np.nan,
    ).reshape(len(rows), len(cols))
    valid = np.isfinite(sampled)
    if not valid.any():
        return terrain, 0
    elevation = terrain.elevation_m.copy()
    target = elevation[np.ix_(rows, cols)]
    target[valid] = sampled[valid]
    elevation[np.ix_(rows, cols)] = target
    minimum = float(np.min(elevation))
    maximum = max(float(np.max(elevation)), minimum + 1.0)
    spec = terrain.spec.model_copy(update={"min_elevation_m": minimum, "max_elevation_m": maximum})
    return terrain.model_copy(update={"spec": spec, "elevation_m": elevation}), int(valid.sum())


def build_prior_scenario(
    sample: GeoPoseSample,
    terrain: TerrainMap,
    truth: CameraExtrinsics,
    regime: PriorRegimeName,
    bucket_name: str,
    replicate: int,
    root_seed: int,
) -> PriorScenario:
    """Construct one leakage-auditable prior regime."""

    bucket = PERTURBATION_BUCKETS[bucket_name]
    seed = _stable_seed(root_seed, sample.name, bucket_name, replicate, "prior_perturbation")
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    requested_east = bucket.horizontal_m * math.sin(angle)
    requested_north = bucket.horizontal_m * math.cos(angle)
    requested_up = bucket.vertical_m * (-1.0 if rng.integers(0, 2) else 1.0)
    requested_yaw = bucket.yaw_deg * (-1.0 if rng.integers(0, 2) else 1.0)
    requested_pitch = bucket.pitch_deg * (-1.0 if rng.integers(0, 2) else 1.0)
    margin = max(20.0, 0.01 * min(terrain.spec.width_m, terrain.spec.height_m))
    perturbed_east = float(
        np.clip(truth.position.east_m + requested_east, float(terrain.x_m[0]) + margin, float(terrain.x_m[-1]) - margin)
    )
    perturbed_north = float(
        np.clip(
            truth.position.north_m + requested_north,
            float(terrain.y_m[0]) + margin,
            float(terrain.y_m[-1]) - margin,
        )
    )
    perturbed_ground = terrain.elevation_at(perturbed_east, perturbed_north)
    perturbed_up = max(truth.position.up_m + requested_up, perturbed_ground + 2.0)
    perturbed_position = LocalPoint(east_m=perturbed_east, north_m=perturbed_north, up_m=perturbed_up)
    perturbation = {
        "bucket": bucket_name,
        "replicate": replicate,
        "seed": seed,
        "requested": {
            "horizontal_m": bucket.horizontal_m,
            "east_m": round(requested_east, 6),
            "north_m": round(requested_north, 6),
            "up_m": requested_up,
            "yaw_deg": requested_yaw,
            "pitch_deg": requested_pitch,
        },
        "realized": {
            "east_m": round(perturbed_east - truth.position.east_m, 6),
            "north_m": round(perturbed_north - truth.position.north_m, 6),
            "up_m": round(perturbed_up - truth.position.up_m, 6),
            "yaw_deg": requested_yaw,
            "pitch_deg": requested_pitch,
        },
    }
    common_sigmas = {
        "horizontal_sigma_m": max(bucket.horizontal_m, 1.0),
        "vertical_sigma_m": max(bucket.vertical_m, 1.0),
        "yaw_sigma_deg": max(bucket.yaw_deg, 1.0),
        "pitch_sigma_deg": max(bucket.pitch_deg, 1.0),
    }
    if regime == "raw_metadata":
        prior = PosePrior(
            position=truth.position,
            yaw_deg=truth.yaw_deg,
            pitch_deg=truth.pitch_deg,
            **common_sigmas,
        )
        return PriorScenario(regime, prior, True, True, perturbation, True, True)
    if regime == "perturbed_metadata":
        prior = PosePrior(
            position=perturbed_position,
            yaw_deg=_wrap180(truth.yaw_deg + requested_yaw),
            pitch_deg=float(np.clip(truth.pitch_deg + requested_pitch, -50.0, 50.0)),
            **common_sigmas,
        )
        return PriorScenario(regime, prior, True, True, perturbation, False, True)
    if regime == "position_only":
        prior = PosePrior(
            position=perturbed_position,
            yaw_deg=0.0,
            pitch_deg=0.0,
            horizontal_sigma_m=common_sigmas["horizontal_sigma_m"],
            vertical_sigma_m=common_sigmas["vertical_sigma_m"],
            yaw_sigma_deg=100.0,
            pitch_sigma_deg=30.0,
        )
        return PriorScenario(regime, prior, True, False, perturbation, False, True)
    center_ground = terrain.elevation_at(0.0, 0.0)
    prior = PosePrior(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=center_ground + 2.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        horizontal_sigma_m=max(terrain.spec.width_m, terrain.spec.height_m) / 2.0,
        vertical_sigma_m=1000.0,
        yaw_sigma_deg=100.0,
        pitch_sigma_deg=30.0,
    )
    return PriorScenario(regime, prior, False, False, perturbation, False, False)


def aggregate_matrix(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate attempts without mixing algorithms, priors, or evidence tracks."""

    keys = sorted({(str(case["algorithm"]), str(case["prior_regime"]), str(case["evidence_track"])) for case in cases})
    result: list[dict[str, Any]] = []
    for algorithm, regime, evidence in keys:
        selected = [
            case
            for case in cases
            if case["algorithm"] == algorithm and case["prior_regime"] == regime and case["evidence_track"] == evidence
        ]
        attempted = [case for case in selected if case.get("status") != "skipped"]
        primary = [case for case in attempted if case.get("ranking_eligible") is True]
        paired = [case for case in attempted if _has_paired_errors(case)]
        position_deltas = [
            float(case["errors"]["horizontal_position_m"]) - float(case["baseline"]["errors"]["horizontal_position_m"])
            for case in paired
        ]
        yaw_deltas = [
            float(case["errors"]["yaw_deg"]) - float(case["baseline"]["errors"]["yaw_deg"]) for case in paired
        ]
        result.append(
            {
                "algorithm": algorithm,
                "prior_regime": regime,
                "evidence_track": evidence,
                "requested": len(selected),
                "attempted": len(attempted),
                "skipped": sum(case.get("status") == "skipped" for case in selected),
                "errors": sum(case.get("status") == "error" for case in attempted),
                "abstained": sum(case.get("outcome") == "abstained" for case in attempted),
                "evidence_rejected": sum(case.get("outcome") == "evidence_rejected" for case in attempted),
                "successes": sum(case.get("success", {}).get("value") is True for case in attempted),
                "success_rate": _rate(attempted),
                "position_success_rate": _threshold_rate(
                    attempted,
                    "horizontal_position_m",
                    SUCCESS_THRESHOLDS["horizontal_position_error_m_lte"],
                ),
                "yaw_success_rate": _threshold_rate(
                    attempted,
                    "yaw_deg",
                    SUCCESS_THRESHOLDS["absolute_yaw_error_deg_lte"],
                ),
                "primary_attempts": len(primary),
                "primary_successes": sum(case.get("success", {}).get("value") is True for case in primary),
                "primary_success_rate": _rate(primary),
                "primary_position_success_rate": _threshold_rate(
                    primary,
                    "horizontal_position_m",
                    SUCCESS_THRESHOLDS["horizontal_position_error_m_lte"],
                ),
                "primary_yaw_success_rate": _threshold_rate(
                    primary,
                    "yaw_deg",
                    SUCCESS_THRESHOLDS["absolute_yaw_error_deg_lte"],
                ),
                "median_horizontal_position_error_m": _median_error(attempted, "horizontal_position_m"),
                "median_absolute_yaw_error_deg": _median_error(attempted, "yaw_deg"),
                "paired_prior_attempts": len(paired),
                "improved_over_prior": sum(
                    case.get("success", {}).get("value") is True
                    and case["baseline"].get("success", {}).get("value") is not True
                    for case in paired
                ),
                "regressed_from_prior": sum(
                    case.get("success", {}).get("value") is not True
                    and case["baseline"].get("success", {}).get("value") is True
                    for case in paired
                ),
                "position_improved_over_prior": sum(delta < -1e-6 for delta in position_deltas),
                "position_regressed_from_prior": sum(delta > 1e-6 for delta in position_deltas),
                "yaw_improved_over_prior": sum(delta < -1e-6 for delta in yaw_deltas),
                "yaw_regressed_from_prior": sum(delta > 1e-6 for delta in yaw_deltas),
                "median_position_delta_vs_prior_m": (round(median(position_deltas), 5) if position_deltas else None),
                "median_yaw_delta_vs_prior_deg": round(median(yaw_deltas), 5) if yaw_deltas else None,
                "runtime_s": round(sum(float(case.get("runtime_s") or 0.0) for case in attempted), 4),
                "compatibility_strata": {tier: _stratum(selected, tier) for tier in ("MAP_A", "MAP_B", "MAP_C")},
            }
        )
    return result


def commit_artifact(
    output_dir: Path,
    *,
    run_metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    matrix_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """Commit a new benchmark artifact exactly once; existing files are never replaced."""

    output_dir.mkdir(parents=True, exist_ok=False)
    aggregates = aggregate_matrix(matrix_cases)
    results = {
        "schema_version": 2,
        "run_id": output_dir.name,
        "rows": rows,
        "matrix_cases": matrix_cases,
        "aggregates": aggregates,
    }
    results_bytes = _json_bytes(results)
    summary = matrix_summary_markdown(aggregates)
    summary_bytes = summary.encode()
    _write_once(output_dir / "results.json", results_bytes)
    _write_once(output_dir / "summary.md", summary_bytes)
    completed = dict(run_metadata)
    completed.update(
        schema_version=2,
        run_id=output_dir.name,
        status="complete",
        completed_samples=sum("error" not in row for row in rows),
        failed_samples=sum("error" in row for row in rows),
        results_sha256=hashlib.sha256(results_bytes).hexdigest(),
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
    )
    _write_once(output_dir / "run.json", _json_bytes(completed))
    return completed


def matrix_summary_markdown(aggregates: list[dict[str, Any]]) -> str:
    """Small human-readable companion to the machine-readable matrix."""

    lines = [
        "# GeoPose workbench strategy matrix",
        "",
        "Primary rates use MANUAL samples in MAP_A/MAP_B only. Raw-metadata retention cells are excluded.",
        "",
        "| algorithm | prior | evidence | attempts | success | primary n | primary success "
        "| median pos m | median yaw deg |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        lines.append(
            "| {algorithm} | {prior_regime} | {evidence_track} | {attempted} | {success} | "
            "{primary_attempts} | {primary_success} | {position} | {yaw} |".format(
                **row,
                success=_format_rate(row["success_rate"]),
                primary_success=_format_rate(row["primary_success_rate"]),
                position=_format_number(row["median_horizontal_position_error_m"]),
                yaw=_format_number(row["median_absolute_yaw_error_deg"]),
            )
        )
    return "\n".join(lines) + "\n"


def input_fingerprint(sample_dirs: list[Path]) -> dict[str, Any]:
    """Content-address the exact GeoPose inputs used by a run."""

    records: list[dict[str, str]] = []
    for sample_dir in sample_dirs:
        for relative in ("info.txt", "cyl/photo_crop.jpg", "cyl/distance_crop.pfm"):
            path = sample_dir / relative
            records.append({"path": f"{sample_dir.name}/{relative}", "sha256": file_sha256(path)})
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"algorithm": "sha256", "aggregate_sha256": hashlib.sha256(encoded).hexdigest(), "files": records}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_implementation_fingerprint(paths: list[Path]) -> dict[str, Any]:
    records = [{"path": str(path.relative_to(BASE)), "sha256": file_sha256(path)} for path in paths if path.exists()]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {"aggregate_sha256": hashlib.sha256(encoded).hexdigest(), "files": records}


def cache_inventory(paths: list[Path]) -> dict[str, Any]:
    """Record a labelled weak identity for large terrain files without hashing gigabytes."""

    records = [
        {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in sorted(paths)
        if path.is_file()
    ]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return {
        "identity_kind": "filename_size_mtime_not_content_hash",
        "snapshot_sha256": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(records),
        "files": records,
    }


def _render_match_resources(terrain: TerrainMap, config: MatrixConfig) -> RenderMatchResources:
    if config.render_matcher == "disabled":
        raise MatcherUnavailable("render matcher is disabled")
    if config.render_matcher == "sift":
        matcher: DenseMatcher = SiftMatcher()
    else:
        if config.matcher_manifest_path is None:
            raise MatcherUnavailable("external matcher manifest is not configured")
        manifest_path = Path(config.matcher_manifest_path).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = BASE / manifest_path
        manifest = load_model_manifest(manifest_path)
        matcher_cache_dir: Path | None = None
        if config.matcher_cache_dir is not None:
            matcher_cache_dir = Path(config.matcher_cache_dir).expanduser()
            if not matcher_cache_dir.is_absolute():
                matcher_cache_dir = BASE / matcher_cache_dir
        worker = WorkerMatcher(
            command=config.matcher_command,
            matcher_id=config.matcher_id,
            model_manifest=manifest,
            seed=config.root_seed,
            cache_dir=matcher_cache_dir,
        )
        worker.check_available()
        matcher = worker

    appearance: AppearanceRaster | None = None
    appearance_provenance: dict[str, Any] | None = None
    if config.render_modality == "orthophoto":
        if config.orthophoto_cache_dir is None:
            raise ValueError("orthophoto cache directory is not configured")
        cache_dir = Path(config.orthophoto_cache_dir).expanduser()
        if not cache_dir.is_absolute():
            cache_dir = BASE / cache_dir
        provider = SwissImageProvider(
            cache_dir=cache_dir,
            zoom=config.orthophoto_zoom,
            time=config.orthophoto_time,
            download_missing=False,
            max_tiles=config.orthophoto_max_tiles,
        )
        appearance = provider.mosaic_for_local_bounds(
            terrain.frame,
            east_min_m=float(terrain.x_m[0]),
            east_max_m=float(terrain.x_m[-1]),
            north_min_m=float(terrain.y_m[0]),
            north_max_m=float(terrain.y_m[-1]),
        )
        appearance_provenance = appearance.provenance()

    render_config = RenderMatchConfig(
        render_width_px=config.render_width_px,
        render_height_px=config.render_height_px,
        yaw_step_deg=config.render_yaw_step_deg,
        modality=config.render_modality,
        terrain_stride=config.terrain_stride,
        native_patch_stride=config.native_patch_stride,
        refinement_passes=config.render_refinement_passes,
        candidate_validation=config.render_candidate_validation,
        pnp=replace(PoseRansacConfig(), seed=config.root_seed),
    )
    return RenderMatchResources(
        matcher=matcher,
        appearance=appearance,
        config=render_config,
        provenance={
            "matcher": matcher.identity(),
            "appearance": appearance_provenance,
            "network_allowed_during_benchmark": False,
        },
    )


def replace_render_seed(config: RenderMatchConfig, seed: int) -> RenderMatchConfig:
    """Return a per-cell deterministic copy without mutating shared resources."""

    return replace(config, pnp=replace(config.pnp, seed=seed))


def _load_photo(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.width > MAX_IMAGE_WIDTH:
        scale = MAX_IMAGE_WIDTH / image.width
        image = image.resize((MAX_IMAGE_WIDTH, max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _evidence_tracks(sample: GeoPoseSample, rgb: np.ndarray, config: MatrixConfig) -> list[EvidenceTrack]:
    height, width = rgb.shape[:2]
    oracle = resampled_oracle_skyline(sample.depth_path, width, height)
    tracks = [
        EvidenceTrack(
            "pfm_oracle",
            oracle,
            {"source": "source_depth_pfm", "coverage": round(float(np.isfinite(oracle).mean()), 5)},
        )
    ]
    if "photo_auto" in config.evidence_tracks:
        candidates = extract_candidates(rgb, backend=config.extractor)
        detected_candidate_names = sorted(candidates)
        chosen = best_skyline_candidate(candidates, min_coverage=0.25)
        if chosen is None:
            tracks.append(
                EvidenceTrack(
                    "photo_auto",
                    None,
                    {"source": config.extractor, "available": False, "reason": "no_candidate_with_25pct_coverage"},
                )
            )
        else:
            name, candidate = chosen
            tracks.append(
                EvidenceTrack(
                    "photo_auto",
                    candidate.rows,
                    {
                        "source": config.extractor,
                        "available": True,
                        "candidate": name,
                        "detected_candidates": detected_candidate_names,
                        "eligible_candidates": sorted(candidates),
                        "learned_backend_selected": name not in {"color", "blue", "bright"},
                        "coverage": round(candidate.coverage, 5),
                        "agreement": round(candidate.agreement, 5),
                        "selection_uses_ground_truth": False,
                    },
                )
            )
    if "photo_rgb" in config.evidence_tracks:
        tracks.append(
            EvidenceTrack(
                "photo_rgb",
                None,
                {
                    "source": "query_photo_rgb",
                    "available": True,
                    "shape": list(rgb.shape),
                    "sha256": hashlib.sha256(memoryview(np.ascontiguousarray(rgb)).cast("B")).hexdigest(),
                    "selection_uses_ground_truth": False,
                    "source_depth_pfm_used_by_estimator": False,
                },
            )
        )
    return [track for track in tracks if track.name in config.evidence_tracks]


def _pre_solve_quality(
    sample: GeoPoseSample,
    terrain_selection: TerrainSelection,
    evidence: list[EvidenceTrack],
    width: int,
    height: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    oracle = next((track.rows for track in evidence if track.name == "pfm_oracle"), None)
    if oracle is None:
        oracle = resampled_oracle_skyline(sample.depth_path, width, height)
    terrain = terrain_selection.terrain
    truth = terrain_selection.truth
    grid_step = float(min(abs(terrain.x_m[1] - terrain.x_m[0]), abs(terrain.y_m[1] - terrain.y_m[0])))
    evaluation_patch = terrain_selection.evaluation_high_resolution_patch
    profile = HorizonProfile(
        terrain,
        truth.position.up_m,
        step=5.0 if evaluation_patch is not None else max(grid_step / 2.0, 10.0),
        cam_e=truth.position.east_m,
        cam_n=truth.position.north_m,
        patch=evaluation_patch,
    )
    dem_at_reference = profile.rows_cyl_tan(width, height, sample.fov_deg, sample.yaw_gt_deg, 0.0)
    compatibility: dict[str, Any] = gt_dem_compatibility(
        oracle,
        dem_at_reference,
        width_px=width,
        height_px=height,
        horizontal_fov_deg=sample.fov_deg,
        yaw_deg=sample.yaw_gt_deg,
    ).to_dict()
    compatibility["height"] = raw_camera_clearance_compatibility(
        truth.position.up_m,
        _elevation_at_with_patch(
            terrain,
            evaluation_patch,
            truth.position.east_m,
            truth.position.north_m,
        ),
    )
    compatibility["terrain_inputs"] = {
        "evaluation_reference_patch": terrain_selection.provenance.get("evaluation_reference_patch"),
        "used_by_estimator": False,
    }
    support: float | None = None
    try:
        photo_image = Image.open(sample.photo_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        photo = np.asarray(photo_image)
        edges = edge_mask(photo)
        if edges is not None:
            support = family_support(rows_to_mask(oracle, height), edges)
    except Exception:  # learned edge support is a diagnostic, never an input to compatibility
        support = None
    photo_edge_support = {
        "pfm_edge_support": round(support, 5) if support is not None else None,
        "usable": bool(support is not None and support >= 0.75),
        "used_by_solver": False,
    }
    return compatibility, photo_edge_support


def _estimator_terrain_for_cell(
    base_terrain: TerrainMap,
    scenario: PriorScenario,
    *,
    scenario_key: tuple[PriorRegimeName, int],
    patch_cache: dict[tuple[PriorRegimeName, int], TerrainPatchProvision],
    provision_patch: bool,
) -> EstimatorTerrainSelection:
    """Build isolated per-cell terrain from estimator-visible inputs only."""

    if not provision_patch:
        patch_record = {
            "source": "cached swissALTI3D 2 m",
            "center_source": None,
            "uses_reference_truth": False,
            "used_by_estimator": False,
            "network_allowed": False,
            "status": "estimator_not_run_or_does_not_consume_terrain",
            "coverage": _patch_coverage(None),
        }
        return EstimatorTerrainSelection(
            base_terrain,
            None,
            {
                "base_source": "Copernicus GLO-30",
                "uses_reference_truth": False,
                "native_patch": patch_record,
                "regular_grid_fused_cells": 0,
            },
        )
    if not scenario.use_position_prior:
        patch_record = {
            "source": "cached swissALTI3D 2 m",
            "center_source": "none_no_position_prior",
            "uses_reference_truth": False,
            "used_by_estimator": False,
            "network_allowed": False,
            "status": "not_provisioned_without_position_prior",
            "coverage": _patch_coverage(None),
        }
        return EstimatorTerrainSelection(
            base_terrain,
            None,
            {
                "base_source": "Copernicus GLO-30",
                "uses_reference_truth": False,
                "native_patch": patch_record,
                "regular_grid_fused_cells": 0,
            },
        )

    provision = patch_cache.get(scenario_key)
    if provision is None:
        provision = _cached_patch_at_local_position(
            base_terrain,
            scenario.prior.position,
            center_source="supplied_position_prior",
            uses_reference_truth=False,
            used_by_estimator=True,
        )
        patch_cache[scenario_key] = provision
    fused, fused_cells = (
        fuse_high_resolution_patch(base_terrain, provision.patch) if provision.patch is not None else (base_terrain, 0)
    )
    patch_record = dict(provision.provenance)
    patch_record["prior_contains_exact_reference"] = scenario.contains_exact_reference
    patch_record["prior_constructed_from_reference"] = scenario.constructed_from_reference
    patch_record["used_by_estimator"] = provision.patch is not None
    return EstimatorTerrainSelection(
        fused,
        provision.patch,
        {
            "base_source": "Copernicus GLO-30",
            "uses_reference_truth": False,
            "native_patch": patch_record,
            "regular_grid_fused_cells": fused_cells,
            "regular_grid_is_per_cell_copy": provision.patch is not None,
        },
    )


def _cached_patch_at_local_position(
    terrain: TerrainMap,
    center_local: LocalPoint,
    *,
    center_source: str,
    uses_reference_truth: bool,
    used_by_estimator: bool,
) -> TerrainPatchProvision:
    """Load cached Swiss terrain around a declared local centre without network access."""

    center_geo = terrain.frame.local_to_geo(center_local)
    provenance: dict[str, Any] = {
        "source": "cached swissALTI3D 2 m",
        "center_source": center_source,
        "center_local_m": {
            "east": round(center_local.east_m, 6),
            "north": round(center_local.north_m, 6),
        },
        "center_geo_deg": {
            "latitude": round(center_geo.latitude_deg, 8),
            "longitude": round(center_geo.longitude_deg, 8),
        },
        "uses_reference_truth": uses_reference_truth,
        "used_by_estimator": used_by_estimator,
        "network_allowed": False,
        "radius_m": 4500.0,
    }
    if not in_switzerland(center_geo.latitude_deg, center_geo.longitude_deg):
        provenance.update(status="outside_switzerland", coverage=_patch_coverage(None))
        return TerrainPatchProvision(None, provenance)
    try:
        patch = load_swiss_patch(
            SWISS_DIR,
            center_geo.latitude_deg,
            center_geo.longitude_deg,
            res=2.0,
            radius_m=4500.0,
        )
    except Exception as exc:
        provenance.update(
            status="cache_read_error",
            error_type=type(exc).__name__,
            coverage=_patch_coverage(None),
        )
        return TerrainPatchProvision(None, provenance)
    if patch is None:
        provenance.update(status="no_cached_coverage", coverage=_patch_coverage(None))
        return TerrainPatchProvision(None, provenance)
    reframed = Patch(
        x_m=np.asarray(patch.x_m, dtype=float) + center_local.east_m,
        y_m=np.asarray(patch.y_m, dtype=float) + center_local.north_m,
        elevation_m=np.asarray(patch.elevation_m, dtype=float),
    )
    provenance.update(status="available", coverage=_patch_coverage(reframed))
    return TerrainPatchProvision(reframed, provenance)


def _patch_coverage(patch: Patch | None) -> dict[str, Any]:
    if patch is None:
        return {"available": False, "finite_cells": 0, "finite_fraction": 0.0}
    elevation = np.asarray(patch.elevation_m, dtype=float)
    finite = np.isfinite(elevation)
    return {
        "available": True,
        "shape": list(elevation.shape),
        "finite_cells": int(finite.sum()),
        "finite_fraction": round(float(finite.mean()), 6),
        "local_bounds_m": {
            "east": [round(float(patch.x_m[0]), 3), round(float(patch.x_m[-1]), 3)],
            "north": [round(float(patch.y_m[0]), 3), round(float(patch.y_m[-1]), 3)],
        },
    }


def _elevation_at_with_patch(
    terrain: TerrainMap,
    patch: Patch | None,
    east_m: float,
    north_m: float,
) -> float:
    """Sample the evaluation overlay when finite, otherwise the regional DEM."""

    if patch is not None and len(patch.x_m) > 1 and len(patch.y_m) > 1:
        column = np.interp(east_m, patch.x_m, np.arange(len(patch.x_m), dtype=float), left=np.nan, right=np.nan)
        row = np.interp(north_m, patch.y_m, np.arange(len(patch.y_m), dtype=float), left=np.nan, right=np.nan)
        if np.isfinite(column) and np.isfinite(row):
            value = float(
                map_coordinates(
                    np.asarray(patch.elevation_m, dtype=float),
                    [[row], [column]],
                    order=1,
                    mode="constant",
                    cval=np.nan,
                )[0]
            )
            if math.isfinite(value):
                return value
    return terrain.elevation_at(east_m, north_m)


def _base_case(
    sample: GeoPoseSample,
    cell: MatrixCell,
    scenario: PriorScenario,
    track: EvidenceTrack,
    reference: dict[str, Any],
    compatibility: dict[str, Any],
    photo_edge_support: dict[str, Any],
    *,
    native_patch_missing_from_objective: bool,
    render_matcher: RenderMatcherName,
    render_modality: RenderModality,
) -> dict[str, Any]:
    case_id = ":".join((sample.name, cell.evidence_track, cell.prior_regime, str(cell.replicate), cell.algorithm))
    exclusions: list[str] = []
    if cell.prior_regime == "raw_metadata":
        exclusions.append("raw_metadata_is_the_evaluation_reference")
    if not sample.manual:
        exclusions.append("automatic_reference_pose")
    if compatibility.get("tier") not in {"MAP_A", "MAP_B"}:
        exclusions.append("gt_dem_compatibility_below_primary_gate")
    if cell.algorithm == "global":
        exclusions.append("regional_global_is_unvalidated_and_not_country_scale")
    if cell.algorithm == "render-pnp":
        exclusions.append("render_match_pnp_experimental_unvalidated")
        if render_matcher == "sift":
            exclusions.append("sift_is_a_same_modality_control_not_a_validated_cross_modal_matcher")
        if render_modality != "orthophoto":
            exclusions.append("render_appearance_is_dem_derived_not_orthophoto")
    if native_patch_missing_from_objective and cell.algorithm not in {"keep-prior", "horizon"}:
        exclusions.append("workbench_objective_does_not_consume_native_high_resolution_patch")
    return {
        "id": case_id,
        "name": sample.name,
        "manual": sample.manual,
        "algorithm": cell.algorithm,
        "evidence_track": cell.evidence_track,
        "prior_regime": cell.prior_regime,
        "status": "pending",
        "skip_reason": None,
        "error": None,
        "outcome": "solved",
        "runtime_s": None,
        "ranking_eligible": not exclusions,
        "ranking_exclusions": exclusions,
        "compatibility_tier": compatibility.get("tier"),
        "photo_edge_supported": photo_edge_support.get("usable"),
        "evidence": {"track": track.name, **track.metadata},
        "reference": reference,
        "prior": {
            **scenario.prior.model_dump(mode="json"),
            "use_position_prior": scenario.use_position_prior,
            "use_orientation_prior": scenario.use_orientation_prior,
            "contains_exact_reference": scenario.contains_exact_reference,
            "constructed_from_reference": scenario.constructed_from_reference,
        },
        "perturbation": scenario.perturbation,
        "baseline": None,
        "result": None,
        "errors": None,
        "success": None,
    }


def _baseline_result(estimate: CameraExtrinsics, truth: CameraExtrinsics) -> dict[str, Any]:
    errors = _pose_errors(estimate, truth)
    return {
        "status": "ok",
        "result": {
            "pose": estimate.model_dump(mode="json"),
            "fit": None,
            "evaluations": 0,
            "diagnostics": {"kind": "unchanged_prior_baseline", "uses_evidence": False},
            "candidates": [],
        },
        "errors": errors,
        "success": _success(errors),
    }


def _solver_result(solved: PoseSolveResult, truth: CameraExtrinsics, seed: int) -> dict[str, Any]:
    estimate = solved.estimate.extrinsics
    errors = _pose_errors(estimate, truth)
    return {
        "status": "ok",
        "result": {
            "pose": estimate.model_dump(mode="json"),
            "fit": solved.estimate.metrics.model_dump(mode="json"),
            "evaluations": solved.evaluations,
            "diagnostics": solved.diagnostics,
            "seed": seed,
            "candidates": [
                {"pose": candidate.extrinsics.model_dump(mode="json"), "score": candidate.score}
                for candidate in solved.candidates[:10]
            ],
        },
        "errors": errors,
        "success": _success(errors),
    }


def _render_solver_result(
    solved: RenderMatchPoseResult,
    truth: CameraExtrinsics,
    seed: int,
) -> dict[str, Any]:
    if not solved.solved or solved.extrinsics is None:
        reason = str(solved.diagnostics.get("abstain_reason") or "render_match_pnp_abstained")
        return {
            "status": "ok",
            "outcome": "abstained",
            "result": {
                "pose": None,
                "fit": None,
                "evaluations": None,
                "diagnostics": solved.diagnostics,
                "seed": seed,
                "candidates": [],
            },
            "errors": None,
            "success": {"value": False, "thresholds": SUCCESS_THRESHOLDS, "reason": reason},
        }
    estimate = solved.extrinsics
    errors = _pose_errors(estimate, truth)
    final_pnp = solved.diagnostics.get("final_pnp")
    fit = (
        {
            "inliers": final_pnp.get("inliers"),
            "inlier_ratio": final_pnp.get("inlier_ratio"),
            "median_reprojection_error_px": final_pnp.get("median_reprojection_error_px"),
            "p90_reprojection_error_px": final_pnp.get("p90_reprojection_error_px"),
        }
        if isinstance(final_pnp, dict)
        else None
    )
    return {
        "status": "ok",
        "outcome": "solved",
        "result": {
            "pose": estimate.model_dump(mode="json"),
            "fit": fit,
            "evaluations": final_pnp.get("optimizer_evaluations") if isinstance(final_pnp, dict) else None,
            "diagnostics": solved.diagnostics,
            "seed": seed,
            "candidates": [
                {"pose": candidate.model_dump(mode="json"), "score": None} for candidate in solved.candidates[:10]
            ],
        },
        "errors": errors,
        "success": _success(errors),
    }


def _render_result_uses_native_patch(solved: RenderMatchPoseResult) -> bool:
    """Trust only recorded visible z-buffer contribution, not patch availability."""

    records: list[Any] = list(solved.diagnostics.get("frames") or [])
    refinement = solved.diagnostics.get("refinement")
    if isinstance(refinement, dict):
        records.append(refinement)
    for record in records:
        if not isinstance(record, dict):
            continue
        render = record.get("render")
        if isinstance(render, dict) and render.get("native_high_resolution_patch_used") is True:
            return True
    return False


def _pose_errors(estimate: CameraExtrinsics, truth: CameraExtrinsics) -> dict[str, float | None]:
    de = estimate.position.east_m - truth.position.east_m
    dn = estimate.position.north_m - truth.position.north_m
    du = estimate.position.up_m - truth.position.up_m
    return {
        "horizontal_position_m": round(math.hypot(de, dn), 5),
        "vertical_m": round(abs(du), 5),
        "position_3d_m": round(math.sqrt(de * de + dn * dn + du * du), 5),
        "yaw_deg": round(abs(_angle_error(estimate.yaw_deg, truth.yaw_deg)), 5),
        # GeoPose crop vertical offset is not independently calibrated. A
        # numeric error here would look scoreable even though it is not.
        "pitch_deg": None,
        "pitch_nuisance_deg": round(estimate.pitch_deg, 5),
    }


def _success(errors: dict[str, float | None]) -> dict[str, Any]:
    horizontal = errors.get("horizontal_position_m")
    yaw = errors.get("yaw_deg")
    value = (
        horizontal is not None
        and yaw is not None
        and horizontal <= SUCCESS_THRESHOLDS["horizontal_position_error_m_lte"]
        and yaw <= SUCCESS_THRESHOLDS["absolute_yaw_error_deg_lte"]
    )
    return {"value": bool(value), "thresholds": SUCCESS_THRESHOLDS}


def _attach_shared_baselines(cases: list[dict[str, Any]]) -> None:
    baselines = {
        (case["name"], case["evidence_track"], case["prior_regime"], case["perturbation"]["replicate"]): case
        for case in cases
        if case["algorithm"] == "keep-prior" and case.get("status") == "ok"
    }
    for case in cases:
        key = (case["name"], case["evidence_track"], case["prior_regime"], case["perturbation"]["replicate"])
        baseline = baselines.get(key)
        if baseline is not None and baseline is not case:
            case["baseline"] = {
                "algorithm": "keep-prior",
                "case_id": baseline["id"],
                "errors": baseline.get("errors"),
                "success": baseline.get("success"),
            }


def _legacy_sample_row(
    sample: GeoPoseSample,
    terrain_selection: TerrainSelection,
    compatibility: dict[str, Any],
    photo_edge_support: dict[str, Any],
    evidence: list[EvidenceTrack],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    def canonical(track: EvidenceTrackName) -> dict[str, Any] | None:
        matches = [
            case
            for case in cases
            if case["algorithm"] == "horizon"
            and case["prior_regime"] == "position_only"
            and case["evidence_track"] == track
            and case["perturbation"]["replicate"] == 0
            and case.get("status") != "skipped"
        ]
        if not matches:
            return None
        case = matches[0]
        result = case.get("result") or {}
        fit = result.get("fit") or {}
        diagnostics = result.get("diagnostics") or {}
        errors = case.get("errors") or {}
        success = case.get("success") or {}
        return {
            "correct": bool(errors.get("yaw_deg") is not None and errors["yaw_deg"] <= 5.0),
            "full_pose_correct": success.get("value") is True,
            "yaw_err": errors.get("yaw_deg"),
            "position_err_m": errors.get("horizontal_position_m"),
            "chamfer_px": diagnostics.get("chamfer_px", fit.get("contour_mae_px")),
            "coverage": diagnostics.get("coverage"),
            "alias_ratio": diagnostics.get("alias_ratio"),
            "verdict": diagnostics.get("verdict", "ERROR" if case.get("status") == "error" else "UNCALIBRATED"),
            "matrix_case_id": case["id"],
        }

    evidence_meta = {track.name: track.metadata for track in evidence}
    return {
        "name": sample.name,
        "manual": sample.manual,
        "lat": sample.lat,
        "lon": sample.lon,
        "elev_m": sample.elev_m,
        "fov_deg": round(sample.fov_deg, 5),
        "gt_yaw": round(sample.yaw_gt_deg, 5),
        "gt_pitch": round(sample.pitch_gt_deg, 5),
        "gt_roll": round(sample.roll_gt_deg, 5),
        "terrain": terrain_selection.provenance,
        "gt_dem_compatibility": compatibility,
        "photo_edge_support": photo_edge_support,
        "original_metadata_diagnostic": _original_metadata_diagnostic(
            sample,
            terrain_selection.terrain,
            terrain_selection.truth,
            estimate=None,
        ),
        "evidence": evidence_meta,
        "oracle": canonical("pfm_oracle"),
        "extracted": canonical("photo_auto"),
        "matrix_case_ids": [case["id"] for case in cases],
    }


def _reference_record(sample: GeoPoseSample, truth: CameraExtrinsics) -> dict[str, Any]:
    return {
        "source": "refined_geopose_metadata_info_lines_3_6",
        "position": truth.position.model_dump(mode="json"),
        "yaw_deg": truth.yaw_deg,
        "pitch_deg": truth.pitch_deg,
        "roll_deg": truth.roll_deg,
        "pitch_comparable": False,
        "pitch_note": "GeoPose cylindrical crops contain an uncalibrated global vertical crop offset.",
        "dataset_elevation_m": sample.elev_m,
        "sole_success_grading_reference": True,
    }


def _case_estimate(case: dict[str, Any]) -> CameraExtrinsics | None:
    """Recover the persisted estimate without exposing diagnostics to a solver."""

    result = case.get("result")
    pose = result.get("pose") if isinstance(result, dict) else None
    if not isinstance(pose, dict):
        return None
    try:
        return CameraExtrinsics.model_validate(pose)
    except TypeError, ValueError:
        return None


def _original_metadata_diagnostic(
    sample: GeoPoseSample,
    terrain: TerrainMap,
    truth: CameraExtrinsics,
    *,
    estimate: CameraExtrinsics | None,
) -> dict[str, Any]:
    """Describe refined/original label ambiguity strictly after estimation.

    GeoPose's optional second metadata tuple has no orientation.  It is useful
    for auditing how much the published refinement moved the source GPS,
    elevation, and FOV, but it is neither another truth pose nor an estimator
    input.  Keeping this record separate from ``errors`` and ``success`` makes
    that contract machine-readable.
    """

    contract: dict[str, Any] = {
        "available": sample.original_metadata is not None,
        "kind": "evaluation_only_original_noisy_metadata",
        "used_by_estimator": False,
        "used_for_success_grading": False,
        "used_for_ranking": False,
        "primary_reference": "refined_geopose_metadata_info_lines_3_6",
        "original_source": "geopose_info_lines_7_10",
        "refined_minus_original": None,
        "estimate_errors_to_original": None,
    }
    original = sample.original_metadata
    if original is None:
        return contract

    original_position = terrain.frame.geo_to_local(
        GeoPoint(
            latitude_deg=original.lat,
            longitude_deg=original.lon,
            elevation_m=original.elev_m,
        )
    )
    refined_de = truth.position.east_m - original_position.east_m
    refined_dn = truth.position.north_m - original_position.north_m
    contract.update(
        original={
            "latitude_deg": original.lat,
            "longitude_deg": original.lon,
            "elevation_m": original.elev_m,
            "horizontal_fov_deg": original.fov_deg,
        },
        refined_minus_original={
            "east_m": round(refined_de, 5),
            "north_m": round(refined_dn, 5),
            "horizontal_position_m": round(math.hypot(refined_de, refined_dn), 5),
            "vertical_m": round(truth.position.up_m - original_position.up_m, 5),
            "fov_deg": round(sample.fov_deg - original.fov_deg, 5),
            "signed_components": "refined_minus_original; horizontal_position_m is a magnitude",
        },
    )
    if estimate is not None:
        estimate_de = estimate.position.east_m - original_position.east_m
        estimate_dn = estimate.position.north_m - original_position.north_m
        estimate_du = estimate.position.up_m - original_position.up_m
        contract["estimate_errors_to_original"] = {
            "horizontal_position_m": round(math.hypot(estimate_de, estimate_dn), 5),
            "vertical_m": round(abs(estimate_du), 5),
            "position_3d_m": round(
                math.sqrt(estimate_de * estimate_de + estimate_dn * estimate_dn + estimate_du * estimate_du),
                5,
            ),
            "fov_deg": round(abs(sample.fov_deg - original.fov_deg), 5),
            "fov_estimated_by_solver": False,
            "fov_note": "FOV error compares the fixed refined query intrinsics to original metadata.",
        }
    return contract


def _rows_contour(rows: np.ndarray, width: int, height: int, *, source: str) -> SkylineContour:
    points = [ImagePoint(x_px=float(column), y_px=float(row)) for column, row in enumerate(rows) if np.isfinite(row)]
    return SkylineContour(
        image_width_px=width,
        image_height_px=height,
        points=points,
        source=source,
    )


def _rate(cases: list[dict[str, Any]]) -> float | None:
    if not cases:
        return None
    return round(sum(case.get("success", {}).get("value") is True for case in cases) / len(cases), 5)


def _median_error(cases: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(case["errors"][key])
        for case in cases
        if isinstance(case.get("errors"), dict) and case["errors"].get(key) is not None
    ]
    return round(median(values), 5) if values else None


def _threshold_rate(cases: list[dict[str, Any]], key: str, threshold: float) -> float | None:
    if not cases:
        return None
    successes = sum(
        isinstance(case.get("errors"), dict)
        and case["errors"].get(key) is not None
        and float(case["errors"][key]) <= threshold
        for case in cases
    )
    return round(successes / len(cases), 5)


def _has_paired_errors(case: dict[str, Any]) -> bool:
    current = case.get("errors")
    baseline = case.get("baseline")
    baseline_errors = baseline.get("errors") if isinstance(baseline, dict) else None
    return (
        isinstance(current, dict)
        and isinstance(baseline_errors, dict)
        and current.get("horizontal_position_m") is not None
        and current.get("yaw_deg") is not None
        and baseline_errors.get("horizontal_position_m") is not None
        and baseline_errors.get("yaw_deg") is not None
    )


def _stratum(cases: list[dict[str, Any]], tier: str) -> dict[str, Any]:
    selected = [case for case in cases if case.get("compatibility_tier") == tier and case.get("status") != "skipped"]
    return {
        "attempts": len(selected),
        "successes": sum(case.get("success", {}).get("value") is True for case in selected),
        "success_rate": _rate(selected),
    }


def _stable_seed(root_seed: int, *parts: object) -> int:
    payload = "\0".join((str(root_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _angle_error(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def _wrap180(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _format_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def _format_number(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_once(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def config_record(config: MatrixConfig) -> dict[str, Any]:
    return asdict(config)


def default_terrain_cache_inventory() -> dict[str, Any]:
    return {
        "copernicus": cache_inventory(list(COP_TILES_DIR.glob("*.tif"))),
        "swissalti": cache_inventory(list(SWISS_DIR.glob("*.tif"))),
    }


def compatibility_contract() -> dict[str, Any]:
    return {
        "policy": COMPATIBILITY_POLICY,
        "inputs": ["source_depth_pfm", "raw_metadata_pose", "selected_terrain_stack"],
        "forbidden_inputs": ["gt_v2_refined_pose", "photo_skyline", "solver_output"],
        "primary_tiers": ["MAP_A", "MAP_B"],
        "height_policy": HEIGHT_COMPATIBILITY_POLICY,
        "height_note": "Separate raw camera-clearance/datum check; thresholds are provisional.",
    }
