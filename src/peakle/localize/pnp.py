"""Projection-aware robust absolute pose from render-lifted 3D points.

GeoPose crops are cylindrical/tangent images, so raw pixels cannot be fed to a
pinhole EPnP implementation.  This module scores every hypothesis through the
actual query camera model.  The nonlinear minimal solver is slower than OpenCV's
EPnP, but it is deterministic, works for both active projections, and provides
the correct geometry for the first render-matching benchmark.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.domain.coordinates import LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.rendering.pinhole import project_points
from peakle.rendering.point_skyline import project_cyltan_points
from peakle.rendering.rasterizer import HeightfieldGrid, HeightfieldLike
from peakle.rendering.terrain_view import heightfield_fingerprint

PoseFitStatus = Literal["solved", "abstained"]
ClearanceConstraintPolicy = Literal["auto", "free_up", "prior_ground_coupled"]
ResolvedClearancePolicy = Literal["free_up", "prior_ground_coupled"]
RANSAC_SAMPLING_SCHEMA = "peakle_projection_aware_ransac_sampling_v1"
RANSAC_SAMPLING_METHOD = "mixed_uniform_progressive_confidence_v1"
WORLD_CONSENSUS_GEOMETRY_SCHEMA = "peakle_world_consensus_geometry_v1"
GROUND_ELEVATION_SOURCE_SCHEMA = "peakle_composite_ground_elevation_v1"


@dataclass(frozen=True)
class _GroundSample:
    elevation_m: float
    source: Literal["native_patch_bilinear", "regional_terrain", "regional_terrain_fallback"]

    def record(self) -> dict[str, Any]:
        return {
            "elevation_m": _rounded(self.elevation_m),
            "source": self.source,
        }


@dataclass(frozen=True)
class _CompositeGroundSurface:
    """Finest supplied ground with conservative regional fallback.

    The native patch is authoritative only inside a source-grid cell whose
    four bilinear support samples are finite. This is intentionally stricter
    than interpolating through nodata and keeps the regional DEM available at
    patch boundaries and coverage holes.
    """

    terrain: TerrainMap
    native: HeightfieldGrid | None
    native_identity: dict[str, Any] | None

    @classmethod
    def from_inputs(
        cls,
        terrain: TerrainMap | None,
        native_elevation_patch: HeightfieldLike | None,
    ) -> _CompositeGroundSurface | None:
        if terrain is None:
            if native_elevation_patch is not None:
                raise ValueError("a native elevation patch requires regional terrain for nodata fallback")
            return None
        native = HeightfieldGrid.from_like(native_elevation_patch) if native_elevation_patch is not None else None
        identity = heightfield_fingerprint(native) if native is not None else None
        return cls(terrain=terrain, native=native, native_identity=identity)

    def sample(self, east_m: float, north_m: float) -> _GroundSample:
        if self.native is not None:
            native_elevation = _bilinear_native_elevation(self.native, east_m, north_m)
            if native_elevation is not None:
                return _GroundSample(native_elevation, "native_patch_bilinear")
        regional_source = "regional_terrain_fallback" if self.native is not None else "regional_terrain"
        return _GroundSample(
            self.terrain.elevation_at(east_m, north_m),
            regional_source,
        )

    def elevation_at(self, east_m: float, north_m: float) -> float:
        return self.sample(east_m, north_m).elevation_m

    def record(self) -> dict[str, Any]:
        native_record: dict[str, Any] | None = None
        if self.native is not None and self.native_identity is not None:
            finite = np.isfinite(self.native.elevation_m)
            finite_cells = finite[:-1, :-1] & finite[:-1, 1:] & finite[1:, :-1] & finite[1:, 1:]
            native_record = {
                "source_content_sha256": self.native_identity["sha256"],
                "source_shape": self.native_identity["shape"],
                "source_spacing_m": self.native_identity["spacing_m"],
                "source_bounds_m": self.native_identity["bounds_m"],
                "source_elevation_range_m": self.native_identity["elevation_range_m"],
                "source_finite_elevation_samples": self.native_identity["finite_elevation_samples"],
                "source_nodata_samples": self.native_identity["nodata_samples"],
                "finite_bilinear_support_cells": int(finite_cells.sum()),
            }
        return {
            "schema": GROUND_ELEVATION_SOURCE_SCHEMA,
            "policy": (
                "bilinear native source grid where the containing cell has four finite corners; "
                "otherwise regional TerrainMap bilinear fallback"
                if self.native is not None
                else "regional TerrainMap bilinear elevation"
            ),
            "native_patch_supplied": self.native is not None,
            "native_patch_sampled_at_source_resolution": self.native is not None,
            "native_patch": native_record,
            "regional_search_bounds_retained": True,
            "uses_reference_truth": False,
        }


@dataclass(frozen=True)
class PoseRansacConfig:
    """Search, robustness, and plausibility settings persisted by benchmarks."""

    # ``iterations`` is the minimum total trial floor. The adaptive statistical
    # budget below may execute more trials when the observed consensus is weak.
    iterations: int = 96
    sample_size: int = 3
    max_iterations: int = 960
    target_success_probability: float = 0.99
    guided_trial_interval: int = 5
    reprojection_threshold_px: float = 5.0
    min_correspondences: int = 12
    min_inliers: int = 10
    min_inlier_ratio: float = 0.20
    min_query_grid_cells: int = 4
    min_query_x_span_fraction: float = 0.12
    min_query_y_span_fraction: float = 0.06
    # These gates reject only pathological 3D support.  In particular, the
    # third singular value is deliberately not gated: terrain is locally
    # planar, and planarity is not by itself a PnP degeneracy.
    world_unique_tolerance_m: float = 0.05
    min_unique_world_inliers: int = 6
    min_unique_world_inlier_ratio: float = 0.50
    min_world_horizontal_span_m: float = 1.0
    min_world_3d_span_m: float = 2.0
    min_world_second_singular_ratio: float = 1e-3
    min_camera_angular_span_deg: float = 0.25
    horizontal_search_radius_m: float = 700.0
    vertical_search_radius_m: float = 300.0
    yaw_search_radius_deg: float = 55.0
    pitch_bounds_deg: tuple[float, float] = (-60.0, 60.0)
    roll_search_radius_deg: float = 12.0
    prior_weight_px: float = 0.35
    ground_weight_px: float = 2.0
    minimum_clearance_m: float = 0.5
    maximum_clearance_m: float | None = 150.0
    clearance_constraint_policy: ClearanceConstraintPolicy = "auto"
    ground_coupled_clearance_bounds_m: tuple[float, float] = (0.5, 50.0)
    ground_coupled_clearance_radius_m: float = 10.0
    ground_coupled_plausible_prior_clearance_bounds_m: tuple[float, float] = (0.5, 12.0)
    ground_coupled_fallback_clearance_m: float = 2.0
    ground_coupled_fallback_bounds_m: tuple[float, float] = (0.5, 12.0)
    max_subset_nfev: int = 90
    max_refine_nfev: int = 350
    seed: int = 0

    def validate(self) -> None:
        if self.iterations < 1:
            raise ValueError("PnP RANSAC iterations must be positive")
        if self.sample_size < 3:
            raise ValueError("PnP RANSAC sample_size must be at least 3")
        if self.max_iterations < self.iterations:
            raise ValueError("PnP RANSAC max_iterations cannot be smaller than iterations")
        if not 0.0 < self.target_success_probability < 1.0:
            raise ValueError("PnP RANSAC target success probability must be in (0, 1)")
        if self.guided_trial_interval < 2:
            raise ValueError("PnP RANSAC guided trial interval must be at least 2")
        if self.min_correspondences < self.sample_size:
            raise ValueError("minimum correspondences cannot be smaller than the RANSAC sample")
        if self.min_inliers < 6:
            raise ValueError("minimum PnP inliers must be at least 6")
        if not 0.0 < self.min_inlier_ratio <= 1.0:
            raise ValueError("minimum PnP inlier ratio must be in (0, 1]")
        if self.reprojection_threshold_px <= 0.0:
            raise ValueError("PnP reprojection threshold must be positive")
        if not np.isfinite(self.world_unique_tolerance_m) or self.world_unique_tolerance_m <= 0.0:
            raise ValueError("world-point uniqueness tolerance must be finite and positive")
        if self.min_unique_world_inliers < 3:
            raise ValueError("minimum unique world inliers must be at least 3")
        if not 0.0 < self.min_unique_world_inlier_ratio <= 1.0:
            raise ValueError("minimum unique world-inlier ratio must be in (0, 1]")
        if not np.isfinite(self.min_world_horizontal_span_m) or self.min_world_horizontal_span_m <= 0.0:
            raise ValueError("minimum horizontal world span must be finite and positive")
        if not np.isfinite(self.min_world_3d_span_m) or self.min_world_3d_span_m <= 0.0:
            raise ValueError("minimum 3D world span must be finite and positive")
        if not 0.0 <= self.min_world_second_singular_ratio <= 1.0:
            raise ValueError("minimum second world singular-value ratio must be in [0, 1]")
        if not 0.0 <= self.min_camera_angular_span_deg < 180.0:
            raise ValueError("minimum camera angular span must be in [0, 180)")
        if self.horizontal_search_radius_m <= 0.0 or self.vertical_search_radius_m <= 0.0:
            raise ValueError("PnP position search radii must be positive")
        if not 0.0 < self.yaw_search_radius_deg <= 180.0:
            raise ValueError("PnP yaw search radius must be in (0, 180]")
        if self.pitch_bounds_deg[0] >= self.pitch_bounds_deg[1]:
            raise ValueError("PnP pitch bounds must have positive width")
        if self.clearance_constraint_policy not in {"auto", "free_up", "prior_ground_coupled"}:
            raise ValueError("unsupported PnP clearance constraint policy")
        clearance_lower, clearance_upper = self.ground_coupled_clearance_bounds_m
        if not np.isfinite(clearance_lower + clearance_upper) or clearance_lower >= clearance_upper:
            raise ValueError("ground-coupled clearance bounds must be finite and have positive width")
        if self.ground_coupled_clearance_radius_m <= 0.0:
            raise ValueError("ground-coupled clearance radius must be positive")
        plausible_lower, plausible_upper = self.ground_coupled_plausible_prior_clearance_bounds_m
        if not np.isfinite(plausible_lower + plausible_upper) or plausible_lower >= plausible_upper:
            raise ValueError("plausible prior-clearance bounds must be finite and have positive width")
        fallback_lower, fallback_upper = self.ground_coupled_fallback_bounds_m
        if not np.isfinite(fallback_lower + fallback_upper) or fallback_lower >= fallback_upper:
            raise ValueError("ground-coupled fallback bounds must be finite and have positive width")
        if not np.isfinite(self.ground_coupled_fallback_clearance_m) or not (
            fallback_lower <= self.ground_coupled_fallback_clearance_m <= fallback_upper
        ):
            raise ValueError("ground-coupled fallback clearance must lie inside its fallback bounds")
        if not np.isfinite(self.minimum_clearance_m):
            raise ValueError("minimum camera clearance must be finite")
        if self.maximum_clearance_m is not None and self.maximum_clearance_m <= self.minimum_clearance_m:
            raise ValueError("maximum camera clearance must exceed the minimum")


@dataclass(frozen=True)
class _ClearanceConstraint:
    """Resolved, truth-free vertical parameterization for one PnP call."""

    requested_policy: ClearanceConstraintPolicy
    resolved_policy: ResolvedClearancePolicy
    initial_clearance_m: float | None
    initial_ground_sample: _GroundSample | None = None
    raw_prior_clearance_m: float | None = None
    prior_ground_sample: _GroundSample | None = None
    prior_clearance_plausible_bounds_m: tuple[float, float] | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None
    configured_fallback_clearance_m: float | None = None
    configured_fallback_bounds_m: tuple[float, float] | None = None
    anchor_clearance_m: float | None = None
    lower_clearance_m: float | None = None
    upper_clearance_m: float | None = None

    @property
    def ground_coupled(self) -> bool:
        return self.resolved_policy == "prior_ground_coupled"

    def record(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "requested_policy": self.requested_policy,
            "resolved_policy": self.resolved_policy,
            "parameterization": (
                "camera_up_m = DEM(east_m, north_m) + bounded_clearance_m"
                if self.ground_coupled
                else "independent camera_up_m offset"
            ),
            "initial_clearance_m": _rounded(self.initial_clearance_m),
            "initial_ground_sample": (
                self.initial_ground_sample.record() if self.initial_ground_sample is not None else None
            ),
            "uses_reference_truth": False,
        }
        if self.ground_coupled:
            result.update(
                {
                    "source": "supplied position prior plus finest supplied composite ground surface",
                    "raw_prior_clearance_m": _rounded(self.raw_prior_clearance_m),
                    "prior_ground_sample": (
                        self.prior_ground_sample.record() if self.prior_ground_sample is not None else None
                    ),
                    "prior_clearance_plausible_bounds_m": (
                        list(self.prior_clearance_plausible_bounds_m)
                        if self.prior_clearance_plausible_bounds_m is not None
                        else None
                    ),
                    "anchor_policy": (
                        "neutral_ground_camera_fallback" if self.fallback_applied else "supplied_prior_clearance"
                    ),
                    "fallback_applied": self.fallback_applied,
                    "fallback_reason": self.fallback_reason,
                    "configured_fallback_clearance_m": _rounded(self.configured_fallback_clearance_m),
                    "configured_fallback_bounds_m": (
                        list(self.configured_fallback_bounds_m)
                        if self.configured_fallback_bounds_m is not None
                        else None
                    ),
                    "anchor_clearance_m": _rounded(self.anchor_clearance_m),
                    "clearance_bounds_m": [
                        _rounded(self.lower_clearance_m),
                        _rounded(self.upper_clearance_m),
                    ],
                }
            )
        return result


@dataclass(frozen=True)
class PoseRansacResult:
    """A robust pose or an evidence-based abstention."""

    status: PoseFitStatus
    extrinsics: CameraExtrinsics | None
    inlier_mask: NDArray[np.bool_]
    reprojection_error_px: NDArray[np.float64]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.status == "solved" and self.extrinsics is not None


def project_world_points(
    world_xyz_m: NDArray[np.float64],
    camera: CameraModel,
    extrinsics: CameraExtrinsics,
    *,
    near_clip_m: float = 0.25,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Project local world points through the exact active image model."""

    points = np.asarray(world_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"world points must have shape (N, 3), got {points.shape}")
    intrinsics = CameraIntrinsics.from_horizontal_fov(
        camera.width_px,
        camera.height_px,
        camera.horizontal_fov_deg,
    )
    if camera.projection == "cyltan":
        u_px, v_px, _range_m, valid = project_cyltan_points(
            points,
            intrinsics,
            extrinsics,
            camera.horizontal_fov_deg,
            near_clip_m=near_clip_m,
        )
    else:
        u_px, v_px, _depth_m, valid = project_points(
            points,
            intrinsics,
            extrinsics,
            near_clip_m=near_clip_m,
        )
    pixels = np.column_stack((u_px, v_px))
    valid &= np.all(np.isfinite(pixels), axis=1)
    return pixels, valid


def fit_pose_ransac(
    world_xyz_m: NDArray[np.float64],
    query_xy_px: NDArray[np.float64],
    confidence: NDArray[np.float64],
    query_camera: CameraModel,
    initial: CameraExtrinsics,
    *,
    prior: PosePrior | None = None,
    terrain: TerrainMap | None = None,
    native_elevation_patch: HeightfieldLike | None = None,
    use_position_prior: bool = True,
    use_orientation_prior: bool = True,
    config: PoseRansacConfig | None = None,
) -> PoseRansacResult:
    """Fit a query pose with deterministic nonlinear RANSAC and robust refinement."""

    settings = config or PoseRansacConfig()
    settings.validate()
    world = np.asarray(world_xyz_m, dtype=np.float64)
    image = np.asarray(query_xy_px, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError(f"world points must have shape (N, 3), got {world.shape}")
    if image.shape != (world.shape[0], 2):
        raise ValueError(f"query pixels must have shape {(world.shape[0], 2)}, got {image.shape}")
    if scores.shape != (world.shape[0],):
        raise ValueError(f"confidence must have shape {(world.shape[0],)}, got {scores.shape}")
    original_count = world.shape[0]
    original_inliers = np.zeros(original_count, dtype=np.bool_)
    original_errors = np.full(original_count, np.inf, dtype=np.float64)
    finite = np.all(np.isfinite(world), axis=1) & np.all(np.isfinite(image), axis=1) & np.isfinite(scores)
    inside = (
        finite
        & (image[:, 0] >= 0.0)
        & (image[:, 0] <= query_camera.width_px - 1)
        & (image[:, 1] >= 0.0)
        & (image[:, 1] <= query_camera.height_px - 1)
    )
    retained_indices = np.flatnonzero(inside)
    world = world[inside]
    image = image[inside]
    scores = np.clip(scores[inside], 0.0, None)
    base_diagnostics: dict[str, Any] = {
        "kind": "projection_aware_nonlinear_pnp_ransac_v2",
        "projection": query_camera.projection,
        "input_correspondences": original_count,
        "finite_in_image_correspondences": int(len(world)),
        "discarded_before_ransac": int(original_count - len(world)),
        "config": _config_record(settings),
        "pitch_semantics": (
            "exact global cyltan crop shift through tan(pitch); not a physical camera-tilt claim"
            if query_camera.projection == "cyltan"
            else "physical pinhole camera pitch"
        ),
    }
    ground_surface = _CompositeGroundSurface.from_inputs(terrain, native_elevation_patch)
    base_diagnostics["ground_elevation_source"] = ground_surface.record() if ground_surface is not None else None
    clearance_constraint = _resolve_clearance_constraint(
        initial,
        query_camera,
        prior,
        ground_surface,
        use_position_prior,
        settings,
    )
    base_diagnostics["clearance_constraint"] = clearance_constraint.record()
    base_diagnostics["initial_clearance_m"] = _rounded(clearance_constraint.initial_clearance_m)
    if len(world) < settings.min_correspondences:
        return _abstain(
            original_inliers,
            original_errors,
            base_diagnostics,
            "insufficient_correspondences",
            required=settings.min_correspondences,
            observed=len(world),
        )
    distribution = correspondence_distribution(image, query_camera)
    base_diagnostics["input_distribution"] = distribution
    if (
        distribution["occupied_grid_cells"] < settings.min_query_grid_cells
        or distribution["x_span_fraction"] < settings.min_query_x_span_fraction
        or distribution["y_span_fraction"] < settings.min_query_y_span_fraction
    ):
        return _abstain(
            original_inliers,
            original_errors,
            base_diagnostics,
            "poor_query_spatial_distribution",
        )

    bounds, x_scale = _parameter_bounds(initial, query_camera, terrain, settings, clearance_constraint)
    rng = np.random.default_rng(settings.seed)
    confidence_order = np.lexsort((np.arange(len(scores), dtype=np.int64), -scores))
    assumed_inlier_count_floor = min(
        len(world),
        max(settings.min_inliers, int(math.ceil(settings.min_inlier_ratio * len(world)))),
    )
    required_uniform_at_floor = required_finite_population_ransac_trials(
        assumed_inlier_count_floor,
        len(world),
        settings.sample_size,
        settings.target_success_probability,
    )
    asymptotic_required_uniform_at_floor = required_ransac_trials(
        assumed_inlier_count_floor / len(world),
        settings.sample_size,
        settings.target_success_probability,
    )
    maximum_uniform_trials = _uniform_trials_in_budget(
        settings.max_iterations,
        settings.guided_trial_interval,
    )
    maximum_guided_trials = settings.max_iterations - maximum_uniform_trials
    adaptive_required_uniform = required_uniform_at_floor
    best_parameters: NDArray[np.float64] | None = None
    best_inliers = np.zeros(len(world), dtype=np.bool_)
    best_key: tuple[int, float, float] = (-1, -math.inf, -math.inf)
    total_nfev = 0
    successful_hypotheses = 0
    executed_trials = 0
    uniform_trials = 0
    guided_trials = 0
    maximum_observed_inlier_count = 0
    maximum_observed_inlier_ratio = 0.0
    stopping_reason = "hard_cap_reached"
    for trial_index in range(settings.max_iterations):
        guided = _is_guided_trial(trial_index, settings.guided_trial_interval)
        if guided:
            guided_trials += 1
            subset = _progressive_confidence_subset(
                rng,
                confidence_order,
                scores,
                sample_size=settings.sample_size,
                guided_trial_number=guided_trials,
                maximum_guided_trials=maximum_guided_trials,
            )
        else:
            uniform_trials += 1
            subset = rng.choice(len(world), size=settings.sample_size, replace=False)
        executed_trials += 1
        optimized = _optimize_pose(
            np.zeros(_parameter_count(query_camera), dtype=np.float64),
            world[subset],
            image[subset],
            query_camera,
            initial,
            bounds,
            x_scale,
            prior,
            ground_surface,
            use_position_prior,
            use_orientation_prior,
            settings,
            clearance_constraint,
            max_nfev=settings.max_subset_nfev,
            robust=False,
        )
        total_nfev += int(optimized.nfev)
        if not optimized.success or not np.all(np.isfinite(optimized.x)):
            continue
        successful_hypotheses += 1
        candidate = _parameters_to_extrinsics(
            optimized.x,
            initial,
            query_camera,
            ground_surface,
            clearance_constraint,
        )
        errors = reprojection_errors(world, image, query_camera, candidate)
        inliers = errors <= settings.reprojection_threshold_px
        inlier_count = int(inliers.sum())
        if inlier_count == 0:
            pass
        else:
            observed_ratio = inlier_count / len(world)
            maximum_observed_inlier_count = max(maximum_observed_inlier_count, inlier_count)
            maximum_observed_inlier_ratio = max(maximum_observed_inlier_ratio, observed_ratio)
            adaptive_required_uniform = required_finite_population_ransac_trials(
                max(assumed_inlier_count_floor, maximum_observed_inlier_count),
                len(world),
                settings.sample_size,
                settings.target_success_probability,
            )
            weighted_support = float(np.sum((0.1 + scores[inliers]) / (1.0 + errors[inliers])))
            median_error = float(np.median(errors[inliers]))
            # Acceptance is count/ratio based, so a smaller high-confidence
            # cluster must not displace a larger geometric consensus and then
            # cause an avoidable abstention. Confidence remains a tie-breaker
            # and drives the separate guided trials.
            key = (inlier_count, weighted_support, -median_error)
            if key > best_key:
                best_key = key
                best_parameters = optimized.x.copy()
                best_inliers = inliers
        if executed_trials >= settings.iterations and uniform_trials >= adaptive_required_uniform:
            stopping_reason = "adaptive_uniform_target_met"
            break

    base_diagnostics.update(
        {
            "ransac_hypotheses": executed_trials,
            "successful_hypotheses": successful_hypotheses,
            "optimizer_evaluations": total_nfev,
            "ransac_sampling": {
                "schema": RANSAC_SAMPLING_SCHEMA,
                "method": RANSAC_SAMPLING_METHOD,
                "seed": settings.seed,
                "sample_size": settings.sample_size,
                "pose_parameter_count": _parameter_count(query_camera),
                "minimal_subset_note": (
                    "three 2D correspondences provide six scalar residuals for five cyltan or six pinhole local "
                    "parameters; full-consensus scoring resolves finite/local minimal-solver ambiguities"
                    if settings.sample_size == 3
                    else "overdetermined nonlinear subset"
                ),
                "target_success_probability": settings.target_success_probability,
                "assumed_inlier_ratio_floor": settings.min_inlier_ratio,
                "assumed_inlier_count_floor": assumed_inlier_count_floor,
                "required_uniform_trials_at_assumed_floor": required_uniform_at_floor,
                "asymptotic_required_uniform_trials_at_assumed_floor": (asymptotic_required_uniform_at_floor),
                "maximum_observed_inlier_count": maximum_observed_inlier_count,
                "maximum_observed_inlier_ratio": round(maximum_observed_inlier_ratio, 6),
                "adaptive_required_uniform_trials_final": adaptive_required_uniform,
                "minimum_total_trials": settings.iterations,
                "hard_maximum_total_trials": settings.max_iterations,
                "guided_trial_interval": settings.guided_trial_interval,
                "maximum_uniform_trials_under_hard_cap": maximum_uniform_trials,
                "maximum_guided_trials_under_hard_cap": maximum_guided_trials,
                "budget_meets_assumed_floor": maximum_uniform_trials >= required_uniform_at_floor,
                "executed_total_trials": executed_trials,
                "executed_uniform_trials": uniform_trials,
                "executed_guided_trials": guided_trials,
                "successful_hypotheses": successful_hypotheses,
                "hypothesis_ranking": ("inlier_count_then_confidence_weighted_support_then_median_reprojection_error"),
                "stopping_reason": stopping_reason,
                "statistical_note": (
                    "only uniform trials count toward the probability of drawing at least one all-inlier subset; "
                    "the target does not guarantee conditional nonlinear-solver convergence, and confidence-guided "
                    "trials are deterministic-seed accelerators"
                ),
            },
        }
    )
    if best_parameters is None or int(best_inliers.sum()) < settings.min_inliers:
        return _abstain(
            original_inliers,
            original_errors,
            base_diagnostics,
            "insufficient_ransac_inliers",
            required=settings.min_inliers,
            observed=int(best_inliers.sum()),
        )

    refined = _optimize_pose(
        best_parameters,
        world[best_inliers],
        image[best_inliers],
        query_camera,
        initial,
        bounds,
        x_scale,
        prior,
        ground_surface,
        use_position_prior,
        use_orientation_prior,
        settings,
        clearance_constraint,
        max_nfev=settings.max_refine_nfev,
        robust=True,
    )
    total_nfev += int(refined.nfev)
    parameters = refined.x if np.all(np.isfinite(refined.x)) else best_parameters
    estimate = _parameters_to_extrinsics(
        parameters,
        initial,
        query_camera,
        ground_surface,
        clearance_constraint,
    )
    errors = reprojection_errors(world, image, query_camera, estimate)
    inliers = errors <= settings.reprojection_threshold_px
    # Refit once on the consensus after refinement; this is deterministic and
    # prevents the original minimal subset from retaining undue influence.
    if int(inliers.sum()) >= settings.min_inliers and not np.array_equal(inliers, best_inliers):
        consensus = _optimize_pose(
            parameters,
            world[inliers],
            image[inliers],
            query_camera,
            initial,
            bounds,
            x_scale,
            prior,
            ground_surface,
            use_position_prior,
            use_orientation_prior,
            settings,
            clearance_constraint,
            max_nfev=settings.max_refine_nfev,
            robust=True,
        )
        total_nfev += int(consensus.nfev)
        if np.all(np.isfinite(consensus.x)):
            parameters = consensus.x
            estimate = _parameters_to_extrinsics(
                parameters,
                initial,
                query_camera,
                ground_surface,
                clearance_constraint,
            )
            errors = reprojection_errors(world, image, query_camera, estimate)
            inliers = errors <= settings.reprojection_threshold_px

    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(world)
    inlier_distribution = correspondence_distribution(image[inliers], query_camera) if inlier_count else None
    inlier_world_geometry = world_consensus_geometry(
        world[inliers],
        np.asarray(estimate.position.as_tuple(), dtype=np.float64),
        unique_tolerance_m=settings.world_unique_tolerance_m,
    )
    world_degeneracy_gate = evaluate_world_degeneracy(inlier_world_geometry, settings)
    original_inliers[retained_indices] = inliers
    original_errors[retained_indices] = errors
    final_ground_sample = (
        ground_surface.sample(estimate.position.east_m, estimate.position.north_m)
        if ground_surface is not None
        else None
    )
    clearance_m = estimate.position.up_m - final_ground_sample.elevation_m if final_ground_sample is not None else None
    diagnostics = {
        **base_diagnostics,
        "optimizer_evaluations": total_nfev,
        "refinement_success": bool(refined.success),
        "refinement_message": str(refined.message),
        "inliers": inlier_count,
        "inlier_ratio": round(inlier_ratio, 6),
        "inlier_distribution": inlier_distribution,
        "inlier_world_geometry": inlier_world_geometry,
        "world_degeneracy_gate": world_degeneracy_gate,
        "median_reprojection_error_px": _finite_percentile(errors[inliers], 50.0),
        "p90_reprojection_error_px": _finite_percentile(errors[inliers], 90.0),
        "all_match_median_reprojection_error_px": _finite_percentile(errors, 50.0),
        "estimate_clearance_m": round(clearance_m, 5) if clearance_m is not None else None,
        "final_clearance_m": round(clearance_m, 5) if clearance_m is not None else None,
        "final_ground_sample": final_ground_sample.record() if final_ground_sample is not None else None,
        "candidate_pose": estimate.model_dump(mode="json"),
        "candidate_delta_from_initial": _candidate_delta_record(estimate, initial),
    }
    if query_camera.projection == "cyltan":
        diagnostics["final_crop_shift_nuisance_deg"] = round(float(estimate.pitch_deg), 6)
    if inlier_count < settings.min_inliers or inlier_ratio < settings.min_inlier_ratio:
        return _abstain(
            original_inliers,
            original_errors,
            diagnostics,
            "consensus_below_acceptance_gate",
            required_inliers=settings.min_inliers,
            required_ratio=settings.min_inlier_ratio,
        )
    if not world_degeneracy_gate["passed"]:
        return _abstain(
            original_inliers,
            original_errors,
            diagnostics,
            "degenerate_world_consensus",
            world_degeneracy_failures=world_degeneracy_gate["failures"],
        )
    if clearance_m is not None and (
        clearance_m < settings.minimum_clearance_m
        or (settings.maximum_clearance_m is not None and clearance_m > settings.maximum_clearance_m)
    ):
        return _abstain(
            original_inliers,
            original_errors,
            diagnostics,
            "implausible_camera_clearance",
            minimum_clearance_m=settings.minimum_clearance_m,
            maximum_clearance_m=settings.maximum_clearance_m,
        )
    if inlier_distribution is None or (
        inlier_distribution["occupied_grid_cells"] < settings.min_query_grid_cells
        or inlier_distribution["x_span_fraction"] < settings.min_query_x_span_fraction
        or inlier_distribution["y_span_fraction"] < settings.min_query_y_span_fraction
    ):
        return _abstain(
            original_inliers,
            original_errors,
            diagnostics,
            "inliers_poorly_distributed",
        )
    return PoseRansacResult(
        status="solved",
        extrinsics=estimate,
        inlier_mask=original_inliers,
        reprojection_error_px=original_errors,
        diagnostics=diagnostics,
    )


def reprojection_errors(
    world_xyz_m: NDArray[np.float64],
    image_xy_px: NDArray[np.float64],
    camera: CameraModel,
    extrinsics: CameraExtrinsics,
) -> NDArray[np.float64]:
    """Euclidean pixel errors; invalid projections receive infinity."""

    projected, valid = project_world_points(world_xyz_m, camera, extrinsics)
    observed = np.asarray(image_xy_px, dtype=np.float64)
    errors = np.linalg.norm(projected - observed, axis=1)
    errors[~valid | ~np.isfinite(errors)] = np.inf
    return errors


def correspondence_distribution(image_xy_px: NDArray[np.float64], camera: CameraModel) -> dict[str, Any]:
    """Compact spatial-degeneracy diagnostics in a fixed 4x4 query grid."""

    points = np.asarray(image_xy_px, dtype=np.float64)
    if points.size == 0:
        return {"occupied_grid_cells": 0, "x_span_fraction": 0.0, "y_span_fraction": 0.0}
    x_fraction = np.clip(points[:, 0] / max(camera.width_px - 1, 1), 0.0, 1.0)
    y_fraction = np.clip(points[:, 1] / max(camera.height_px - 1, 1), 0.0, 1.0)
    cells_x = np.minimum((x_fraction * 4).astype(int), 3)
    cells_y = np.minimum((y_fraction * 4).astype(int), 3)
    occupied = len(set(zip(cells_x.tolist(), cells_y.tolist(), strict=True)))
    return {
        "occupied_grid_cells": occupied,
        "x_span_fraction": round(float(np.ptp(x_fraction)), 6),
        "y_span_fraction": round(float(np.ptp(y_fraction)), 6),
    }


def world_consensus_geometry(
    world_xyz_m: NDArray[np.float64],
    camera_position_xyz_m: NDArray[np.float64],
    *,
    unique_tolerance_m: float,
) -> dict[str, Any]:
    """Describe whether a final 3D consensus provides independent pose support.

    Principal scales are computed after 5 cm-by-default quantized de-duplication.
    The second/first ratio detects an essentially collinear locus.  The third
    ratio is diagnostic only because an ordinary terrain surface is expected to
    be locally planar.
    """

    points = np.asarray(world_xyz_m, dtype=np.float64)
    camera_position = np.asarray(camera_position_xyz_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"world points must have shape (N, 3), got {points.shape}")
    if camera_position.shape != (3,):
        raise ValueError(f"camera position must have shape (3,), got {camera_position.shape}")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(camera_position)):
        raise ValueError("world-consensus geometry requires finite points and camera position")
    if not np.isfinite(unique_tolerance_m) or unique_tolerance_m <= 0.0:
        raise ValueError("world-point uniqueness tolerance must be finite and positive")

    unique = _quantized_unique_points(points, unique_tolerance_m)
    point_count = len(points)
    unique_count = len(unique)
    singular_values = np.zeros(3, dtype=np.float64)
    if unique_count:
        centered = unique - np.mean(unique, axis=0, keepdims=True)
        computed = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
        singular_values[: len(computed)] = computed
    principal_rms_scales = singular_values / math.sqrt(max(unique_count, 1))
    first = singular_values[0]
    singular_ratios = singular_values / first if first > np.finfo(np.float64).eps else np.zeros(3)
    energy = singular_values**2
    if float(np.sum(energy)) > 0.0:
        probabilities = energy / np.sum(energy)
        nonzero_probabilities = probabilities[probabilities > 0.0]
        effective_rank = math.exp(-float(np.sum(nonzero_probabilities * np.log(nonzero_probabilities))))
    else:
        effective_rank = 0.0

    vectors = unique - camera_position[None, :]
    ranges = np.linalg.norm(vectors, axis=1)
    usable_directions = vectors[ranges > np.finfo(np.float64).eps]
    usable_ranges = ranges[ranges > np.finfo(np.float64).eps]
    directions = usable_directions / usable_ranges[:, None]
    angular_span_deg = _maximum_pairwise_angle_deg(directions)

    return {
        "schema": WORLD_CONSENSUS_GEOMETRY_SCHEMA,
        "basis": "quantized_unique_final_inlier_world_points",
        "point_count": point_count,
        "unique_point_count": unique_count,
        "unique_point_fraction": _rounded_ratio(unique_count, point_count),
        "unique_tolerance_m": round(float(unique_tolerance_m), 9),
        "horizontal_span_m": round(_maximum_pairwise_distance(unique[:, :2]), 6),
        "three_dimensional_span_m": round(_maximum_pairwise_distance(unique), 6),
        "principal_rms_scales_m": [round(float(value), 6) for value in principal_rms_scales],
        "second_to_first_singular_ratio": round(float(singular_ratios[1]), 9),
        "third_to_first_singular_ratio": round(float(singular_ratios[2]), 9),
        "effective_rank": round(float(effective_rank), 6),
        "camera_angular_span_deg": (round(float(angular_span_deg), 6) if angular_span_deg is not None else None),
        "planarity_note": "third singular value is diagnostic only; planar terrain is allowed",
    }


def evaluate_world_degeneracy(
    geometry: dict[str, Any],
    config: PoseRansacConfig,
) -> dict[str, Any]:
    """Apply the shared truth-free geometry gates to a 3D consensus record."""

    required_unique_points = max(config.sample_size, config.min_unique_world_inliers)
    thresholds = {
        "minimum_unique_point_count": required_unique_points,
        "minimum_unique_point_fraction": config.min_unique_world_inlier_ratio,
        "minimum_horizontal_span_m": config.min_world_horizontal_span_m,
        "minimum_three_dimensional_span_m": config.min_world_3d_span_m,
        "minimum_second_to_first_singular_ratio": config.min_world_second_singular_ratio,
        "minimum_camera_angular_span_deg": config.min_camera_angular_span_deg,
        "third_singular_value_is_gated": False,
    }
    failures: list[str] = []
    if geometry["unique_point_count"] < required_unique_points:
        failures.append("insufficient_unique_world_points")
    if geometry["unique_point_fraction"] < config.min_unique_world_inlier_ratio:
        failures.append("excessive_duplicate_world_points")
    if geometry["horizontal_span_m"] < config.min_world_horizontal_span_m:
        failures.append("tiny_horizontal_world_baseline")
    if geometry["three_dimensional_span_m"] < config.min_world_3d_span_m:
        failures.append("tiny_three_dimensional_world_baseline")
    if geometry["second_to_first_singular_ratio"] < config.min_world_second_singular_ratio:
        failures.append("essentially_collinear_world_points")
    angular_span = geometry["camera_angular_span_deg"]
    if angular_span is None or angular_span < config.min_camera_angular_span_deg:
        failures.append("tiny_camera_angular_baseline")
    return {
        "passed": not failures,
        "failures": failures,
        "thresholds": thresholds,
        "uses_reference_truth": False,
        "accepts_planar_surfaces": True,
    }


def _quantized_unique_points(
    points: NDArray[np.float64],
    tolerance_m: float,
) -> NDArray[np.float64]:
    if not len(points):
        return np.empty((0, 3), dtype=np.float64)
    scaled = points / tolerance_m
    int64_limit = float(np.iinfo(np.int64).max - 1)
    if np.any(np.abs(scaled) > int64_limit):
        raise ValueError("world coordinates exceed the supported uniqueness-quantization range")
    quantized = np.rint(scaled).astype(np.int64)
    _keys, first_indices = np.unique(quantized, axis=0, return_index=True)
    return points[np.sort(first_indices)]


def _maximum_pairwise_distance(points: NDArray[np.float64]) -> float:
    if len(points) < 2:
        return 0.0
    maximum_squared = 0.0
    for start in range(0, len(points), 256):
        block = points[start : start + 256]
        differences = block[:, None, :] - points[None, :, :]
        squared = np.einsum("ijk,ijk->ij", differences, differences)
        maximum_squared = max(maximum_squared, float(np.max(squared)))
    return math.sqrt(maximum_squared)


def _maximum_pairwise_angle_deg(directions: NDArray[np.float64]) -> float | None:
    if not len(directions):
        return None
    if len(directions) == 1:
        return 0.0
    minimum_dot = 1.0
    for start in range(0, len(directions), 256):
        dots = directions[start : start + 256] @ directions.T
        minimum_dot = min(minimum_dot, float(np.min(dots)))
    return math.degrees(math.acos(float(np.clip(minimum_dot, -1.0, 1.0))))


def _rounded_ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 9) if denominator else 0.0


def required_ransac_trials(
    inlier_ratio: float,
    sample_size: int,
    target_success_probability: float,
) -> int:
    """Return the standard uniform-sampling trial budget for one all-inlier subset."""

    if not 0.0 < inlier_ratio <= 1.0:
        raise ValueError("RANSAC inlier ratio must be in (0, 1]")
    if sample_size < 1:
        raise ValueError("RANSAC sample size must be positive")
    if not 0.0 < target_success_probability < 1.0:
        raise ValueError("RANSAC target success probability must be in (0, 1)")
    all_inlier_probability = inlier_ratio**sample_size
    if all_inlier_probability >= 1.0:
        return 1
    return max(
        1,
        int(math.ceil(math.log1p(-target_success_probability) / math.log1p(-all_inlier_probability))),
    )


def required_finite_population_ransac_trials(
    inlier_count: int,
    population_size: int,
    sample_size: int,
    target_success_probability: float,
) -> int:
    """Exact trial budget for uniform sampling without replacement within each subset."""

    if population_size < 1:
        raise ValueError("RANSAC population size must be positive")
    if not 0 < inlier_count <= population_size:
        raise ValueError("RANSAC inlier count must be in [1, population_size]")
    if not 1 <= sample_size <= population_size:
        raise ValueError("RANSAC sample size must be in [1, population_size]")
    if inlier_count < sample_size:
        raise ValueError("an all-inlier subset is impossible when inlier_count < sample_size")
    if not 0.0 < target_success_probability < 1.0:
        raise ValueError("RANSAC target success probability must be in (0, 1)")
    all_inlier_probability = math.prod(
        (inlier_count - offset) / (population_size - offset) for offset in range(sample_size)
    )
    if all_inlier_probability >= 1.0:
        return 1
    return max(
        1,
        int(math.ceil(math.log1p(-target_success_probability) / math.log1p(-all_inlier_probability))),
    )


def _is_guided_trial(trial_index: int, interval: int) -> bool:
    return trial_index % interval == 0


def _uniform_trials_in_budget(total_trials: int, guided_interval: int) -> int:
    guided = (total_trials + guided_interval - 1) // guided_interval
    return total_trials - guided


def _progressive_confidence_subset(
    rng: np.random.Generator,
    confidence_order: NDArray[np.int64],
    scores: NDArray[np.float64],
    *,
    sample_size: int,
    guided_trial_number: int,
    maximum_guided_trials: int,
) -> NDArray[np.int64]:
    """Sample from a confidence-ranked pool that expands to all correspondences."""

    count = int(len(confidence_order))
    if count < sample_size:
        raise ValueError("guided RANSAC pool is smaller than its sample")
    if maximum_guided_trials <= 1:
        pool_size = count
    else:
        progress = (guided_trial_number - 1) / (maximum_guided_trials - 1)
        pool_size = sample_size + int(math.ceil((count - sample_size) * progress))
    pool = np.asarray(confidence_order[:pool_size], dtype=np.int64)
    if pool_size == sample_size:
        return pool.copy()
    pool_scores = np.asarray(scores[pool], dtype=np.float64)
    floor = max(float(np.max(pool_scores, initial=0.0)) * 1e-3, 1e-9)
    weights = pool_scores + floor
    weights /= weights.sum()
    return np.asarray(
        rng.choice(pool, size=sample_size, replace=False, p=weights),
        dtype=np.int64,
    )


def _bilinear_native_elevation(
    surface: HeightfieldGrid,
    east_m: float,
    north_m: float,
) -> float | None:
    """Interpolate only through a fully finite native source-grid cell."""

    if not np.isfinite(east_m) or not np.isfinite(north_m):
        return None
    if east_m < surface.x_m[0] or east_m > surface.x_m[-1] or north_m < surface.y_m[0] or north_m > surface.y_m[-1]:
        return None
    column0 = int(np.clip(np.searchsorted(surface.x_m, east_m, side="right") - 1, 0, len(surface.x_m) - 2))
    row0 = int(np.clip(np.searchsorted(surface.y_m, north_m, side="right") - 1, 0, len(surface.y_m) - 2))
    column1 = column0 + 1
    row1 = row0 + 1
    corners = np.asarray(
        [
            surface.elevation_m[row0, column0],
            surface.elevation_m[row0, column1],
            surface.elevation_m[row1, column0],
            surface.elevation_m[row1, column1],
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(corners)):
        return None
    east_fraction = (east_m - surface.x_m[column0]) / (surface.x_m[column1] - surface.x_m[column0])
    north_fraction = (north_m - surface.y_m[row0]) / (surface.y_m[row1] - surface.y_m[row0])
    south = (1.0 - east_fraction) * corners[0] + east_fraction * corners[1]
    north = (1.0 - east_fraction) * corners[2] + east_fraction * corners[3]
    return float((1.0 - north_fraction) * south + north_fraction * north)


def _resolve_clearance_constraint(
    initial: CameraExtrinsics,
    camera: CameraModel,
    prior: PosePrior | None,
    ground_surface: _CompositeGroundSurface | None,
    use_position_prior: bool,
    settings: PoseRansacConfig,
) -> _ClearanceConstraint:
    """Resolve vertical policy from estimator inputs without reference truth."""

    initial_ground_sample = (
        ground_surface.sample(initial.position.east_m, initial.position.north_m) if ground_surface is not None else None
    )
    initial_clearance = (
        initial.position.up_m - initial_ground_sample.elevation_m if initial_ground_sample is not None else None
    )
    eligible = camera.projection == "cyltan" and ground_surface is not None and prior is not None and use_position_prior
    requested = settings.clearance_constraint_policy
    if requested == "auto":
        resolved: ResolvedClearancePolicy = "prior_ground_coupled" if eligible else "free_up"
    elif requested == "prior_ground_coupled":
        if not eligible:
            raise ValueError(
                "prior_ground_coupled clearance requires a cyltan camera, terrain, and an enabled position prior"
            )
        resolved = "prior_ground_coupled"
    else:
        resolved = "free_up"
    if resolved == "free_up":
        return _ClearanceConstraint(
            requested_policy=requested,
            resolved_policy=resolved,
            initial_clearance_m=initial_clearance,
            initial_ground_sample=initial_ground_sample,
        )

    if ground_surface is None or prior is None:  # narrowed by eligibility above
        raise RuntimeError("ground-coupled PnP unexpectedly lacks terrain or prior")
    prior_ground_sample = ground_surface.sample(
        prior.position.east_m,
        prior.position.north_m,
    )
    raw_prior_clearance = prior.position.up_m - prior_ground_sample.elevation_m
    configured_lower, configured_upper = settings.ground_coupled_clearance_bounds_m
    plausible_lower, plausible_upper = settings.ground_coupled_plausible_prior_clearance_bounds_m
    fallback_reason: str | None = None
    if raw_prior_clearance < plausible_lower:
        fallback_reason = "raw_prior_clearance_below_plausible_input_range"
    elif raw_prior_clearance > plausible_upper:
        fallback_reason = "raw_prior_clearance_above_plausible_input_range"

    if fallback_reason is None:
        effective_lower = max(configured_lower, settings.minimum_clearance_m)
        effective_upper = configured_upper
        anchor_input = raw_prior_clearance
    else:
        fallback_lower, fallback_upper = settings.ground_coupled_fallback_bounds_m
        effective_lower = max(configured_lower, fallback_lower, settings.minimum_clearance_m)
        effective_upper = min(configured_upper, fallback_upper)
        anchor_input = settings.ground_coupled_fallback_clearance_m
    if settings.maximum_clearance_m is not None:
        effective_upper = min(effective_upper, settings.maximum_clearance_m)
    if effective_lower >= effective_upper:
        raise ValueError("ground-coupled clearance bounds collapse under the global clearance gates")
    anchor = float(np.clip(anchor_input, effective_lower, effective_upper))
    if fallback_reason is None:
        lower = max(effective_lower, anchor - settings.ground_coupled_clearance_radius_m)
        upper = min(effective_upper, anchor + settings.ground_coupled_clearance_radius_m)
    else:
        lower = effective_lower
        upper = effective_upper
    if lower >= upper:
        raise ValueError("ground-coupled clearance nuisance has no positive-width feasible range")
    return _ClearanceConstraint(
        requested_policy=requested,
        resolved_policy=resolved,
        initial_clearance_m=initial_clearance,
        initial_ground_sample=initial_ground_sample,
        raw_prior_clearance_m=raw_prior_clearance,
        prior_ground_sample=prior_ground_sample,
        prior_clearance_plausible_bounds_m=settings.ground_coupled_plausible_prior_clearance_bounds_m,
        fallback_applied=fallback_reason is not None,
        fallback_reason=fallback_reason,
        configured_fallback_clearance_m=settings.ground_coupled_fallback_clearance_m,
        configured_fallback_bounds_m=settings.ground_coupled_fallback_bounds_m,
        anchor_clearance_m=anchor,
        lower_clearance_m=lower,
        upper_clearance_m=upper,
    )


def _optimize_pose(
    initial_parameters: NDArray[np.float64],
    world: NDArray[np.float64],
    image: NDArray[np.float64],
    camera: CameraModel,
    initial: CameraExtrinsics,
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    x_scale: NDArray[np.float64],
    prior: PosePrior | None,
    ground_surface: _CompositeGroundSurface | None,
    use_position_prior: bool,
    use_orientation_prior: bool,
    settings: PoseRansacConfig,
    clearance_constraint: _ClearanceConstraint,
    *,
    max_nfev: int,
    robust: bool,
):
    lower, upper = bounds
    # Starting numerically on a bound gives scipy's reflective trust region an
    # almost-zero first step (notably when an implausible prior clearance was
    # clamped to the plausible maximum). Keep a small, scale-relative interior
    # margin while preserving the same feasible policy.
    interior_margin = np.maximum((upper - lower) * 1e-4, 1e-8)
    start = np.clip(
        np.asarray(initial_parameters, dtype=np.float64),
        lower + interior_margin,
        upper - interior_margin,
    )

    def residual(parameters: NDArray[np.float64]) -> NDArray[np.float64]:
        extrinsics = _parameters_to_extrinsics(
            parameters,
            initial,
            camera,
            ground_surface,
            clearance_constraint,
        )
        projected, valid = project_world_points(world, camera, extrinsics)
        difference = projected - image
        invalid = ~valid | ~np.all(np.isfinite(difference), axis=1)
        if np.any(invalid):
            difference[invalid] = 2.0 * max(camera.width_px, camera.height_px)
        terms = [difference.ravel()]
        if prior is not None and settings.prior_weight_px > 0.0:
            regularization: list[float] = []
            if use_position_prior:
                regularization.extend(
                    [
                        (extrinsics.position.east_m - prior.position.east_m) / prior.horizontal_sigma_m,
                        (extrinsics.position.north_m - prior.position.north_m) / prior.horizontal_sigma_m,
                    ]
                )
                if clearance_constraint.ground_coupled:
                    if ground_surface is None or clearance_constraint.anchor_clearance_m is None:
                        raise RuntimeError("ground-coupled PnP lost its terrain or clearance anchor")
                    ground = ground_surface.elevation_at(
                        extrinsics.position.east_m,
                        extrinsics.position.north_m,
                    )
                    clearance = extrinsics.position.up_m - ground
                    regularization.append(
                        (clearance - clearance_constraint.anchor_clearance_m)
                        / max(settings.ground_coupled_clearance_radius_m, 1.0)
                    )
                else:
                    regularization.append((extrinsics.position.up_m - prior.position.up_m) / prior.vertical_sigma_m)
            if use_orientation_prior:
                regularization.extend(
                    [
                        _angle_delta(extrinsics.yaw_deg, prior.yaw_deg) / prior.yaw_sigma_deg,
                        (extrinsics.pitch_deg - prior.pitch_deg) / prior.pitch_sigma_deg,
                    ]
                )
            if regularization:
                terms.append(settings.prior_weight_px * np.asarray(regularization, dtype=np.float64))
        if ground_surface is not None and settings.ground_weight_px > 0.0 and not clearance_constraint.ground_coupled:
            ground = ground_surface.elevation_at(
                extrinsics.position.east_m,
                extrinsics.position.north_m,
            )
            clearance = extrinsics.position.up_m - ground
            lower_violation = max(0.0, settings.minimum_clearance_m - clearance)
            upper_violation = (
                max(0.0, clearance - settings.maximum_clearance_m) if settings.maximum_clearance_m is not None else 0.0
            )
            terms.append(
                settings.ground_weight_px
                * np.asarray([lower_violation / 10.0, upper_violation / 50.0], dtype=np.float64)
            )
        return np.concatenate(terms)

    return least_squares(
        residual,
        start,
        bounds=bounds,
        x_scale=x_scale,
        loss="soft_l1" if robust else "linear",
        f_scale=max(1.0, settings.reprojection_threshold_px * 0.5),
        max_nfev=max_nfev,
        method="trf",
    )


def _parameter_bounds(
    initial: CameraExtrinsics,
    camera: CameraModel,
    terrain: TerrainMap | None,
    settings: PoseRansacConfig,
    clearance_constraint: _ClearanceConstraint,
) -> tuple[tuple[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    horizontal = settings.horizontal_search_radius_m
    vertical = settings.vertical_search_radius_m
    east_bounds = [initial.position.east_m - horizontal, initial.position.east_m + horizontal]
    north_bounds = [initial.position.north_m - horizontal, initial.position.north_m + horizontal]
    up_bounds = [initial.position.up_m - vertical, initial.position.up_m + vertical]
    if terrain is not None:
        east_bounds = [max(east_bounds[0], float(terrain.x_m[0])), min(east_bounds[1], float(terrain.x_m[-1]))]
        north_bounds = [max(north_bounds[0], float(terrain.y_m[0])), min(north_bounds[1], float(terrain.y_m[-1]))]
        if not clearance_constraint.ground_coupled:
            up_bounds = [
                max(up_bounds[0], float(np.min(terrain.elevation_m)) - 50.0),
                min(up_bounds[1], float(np.max(terrain.elevation_m)) + 500.0),
            ]
    if clearance_constraint.ground_coupled:
        if clearance_constraint.anchor_clearance_m is None:
            raise RuntimeError("ground-coupled PnP lost its clearance anchor")
        if clearance_constraint.lower_clearance_m is None or clearance_constraint.upper_clearance_m is None:
            raise RuntimeError("ground-coupled PnP lost its clearance bounds")
        vertical_lower = clearance_constraint.lower_clearance_m - clearance_constraint.anchor_clearance_m
        vertical_upper = clearance_constraint.upper_clearance_m - clearance_constraint.anchor_clearance_m
        vertical_scale = max(
            clearance_constraint.upper_clearance_m - clearance_constraint.lower_clearance_m,
            1.0,
        )
    else:
        vertical_lower = up_bounds[0] - initial.position.up_m
        vertical_upper = up_bounds[1] - initial.position.up_m
        vertical_scale = vertical
    lower = [
        east_bounds[0] - initial.position.east_m,
        north_bounds[0] - initial.position.north_m,
        vertical_lower,
    ]
    upper = [
        east_bounds[1] - initial.position.east_m,
        north_bounds[1] - initial.position.north_m,
        vertical_upper,
    ]
    lower.extend([-settings.yaw_search_radius_deg, settings.pitch_bounds_deg[0] - initial.pitch_deg])
    upper.extend([settings.yaw_search_radius_deg, settings.pitch_bounds_deg[1] - initial.pitch_deg])
    scale = [horizontal, horizontal, vertical_scale, settings.yaw_search_radius_deg, 20.0]
    if camera.projection == "pinhole":
        lower.append(-settings.roll_search_radius_deg)
        upper.append(settings.roll_search_radius_deg)
        scale.append(max(settings.roll_search_radius_deg, 1.0))
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    if np.any(lower_array >= upper_array):
        raise ValueError("PnP search bounds collapse outside the supplied terrain")
    return (lower_array, upper_array), np.asarray(scale, dtype=np.float64)


def _parameters_to_extrinsics(
    parameters: NDArray[np.float64],
    initial: CameraExtrinsics,
    camera: CameraModel,
    ground_surface: _CompositeGroundSurface | None,
    clearance_constraint: _ClearanceConstraint,
) -> CameraExtrinsics:
    values = np.asarray(parameters, dtype=np.float64)
    expected = _parameter_count(camera)
    if values.shape != (expected,):
        raise ValueError(f"pose parameter vector must have shape {(expected,)}, got {values.shape}")
    east_m = float(initial.position.east_m + values[0])
    north_m = float(initial.position.north_m + values[1])
    if clearance_constraint.ground_coupled:
        if ground_surface is None or clearance_constraint.anchor_clearance_m is None:
            raise RuntimeError("ground-coupled PnP requires terrain and a clearance anchor")
        clearance_m = clearance_constraint.anchor_clearance_m + float(values[2])
        up_m = ground_surface.elevation_at(east_m, north_m) + clearance_m
    else:
        up_m = float(initial.position.up_m + values[2])
    roll = _wrap180(initial.roll_deg + values[5]) if camera.projection == "pinhole" else 0.0
    return CameraExtrinsics(
        position=LocalPoint(
            east_m=east_m,
            north_m=north_m,
            up_m=up_m,
        ),
        yaw_deg=_wrap180(initial.yaw_deg + values[3]),
        pitch_deg=float(np.clip(initial.pitch_deg + values[4], -89.0, 89.0)),
        roll_deg=roll,
    )


def _parameter_count(camera: CameraModel) -> int:
    return 6 if camera.projection == "pinhole" else 5


def _abstain(
    inliers: NDArray[np.bool_],
    errors: NDArray[np.float64],
    diagnostics: dict[str, Any],
    reason: str,
    **detail: Any,
) -> PoseRansacResult:
    return PoseRansacResult(
        status="abstained",
        extrinsics=None,
        inlier_mask=inliers,
        reprojection_error_px=errors,
        diagnostics={**diagnostics, "abstain_reason": reason, **detail},
    )


def _finite_percentile(values: NDArray[np.float64], percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return round(float(np.percentile(finite, percentile)), 6) if finite.size else None


def _candidate_delta_record(estimate: CameraExtrinsics, initial: CameraExtrinsics) -> dict[str, float]:
    east_delta = estimate.position.east_m - initial.position.east_m
    north_delta = estimate.position.north_m - initial.position.north_m
    return {
        "east_m": round(float(east_delta), 6),
        "north_m": round(float(north_delta), 6),
        "horizontal_m": round(float(math.hypot(east_delta, north_delta)), 6),
        "up_m": round(float(estimate.position.up_m - initial.position.up_m), 6),
        "yaw_deg": round(float(_angle_delta(estimate.yaw_deg, initial.yaw_deg)), 6),
        "pitch_or_crop_shift_deg": round(float(estimate.pitch_deg - initial.pitch_deg), 6),
    }


def _rounded(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _config_record(config: PoseRansacConfig) -> dict[str, Any]:
    return {key: value for key, value in config.__dict__.items()}


def _angle_delta(value: float, reference: float) -> float:
    return (value - reference + 180.0) % 360.0 - 180.0


def _wrap180(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0
