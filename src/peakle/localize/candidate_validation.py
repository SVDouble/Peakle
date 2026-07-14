"""Truth-free spatial holdout and candidate-render visibility validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.stats import beta as beta_distribution

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.localize.pnp import (
    PoseRansacConfig,
    correspondence_distribution,
    evaluate_world_degeneracy,
    reprojection_errors,
    world_consensus_geometry,
)
from peakle.rendering.pinhole import project_points
from peakle.rendering.terrain_view import TerrainRenderBundle

CANDIDATE_VALIDATION_SCHEMA = "peakle_candidate_pose_holdout_validation_v1"
HOLDOUT_PARTITION_SCHEMA = "peakle_query_spatial_holdout_partition_v1"
HOLDOUT_PARTITION_METHOD = "content_addressed_interleaved_normalized_query_grid_v1"


@dataclass(frozen=True)
class CandidateValidationConfig:
    """Predeclared truth-free checks applied to exactly one selected pose."""

    enabled: bool = True
    query_grid_columns: int = 8
    query_grid_rows: int = 6
    folds: int = 4
    max_holdout_matches_per_frame: int = 400
    confidence_level: float = 0.95
    minimum_testable_fraction: float = 0.50
    minimum_visibility_consistency_fraction: float = 0.80
    render_resolution_multiplier: int = 2
    maximum_local_absolute_depth_span_m: float = 250.0
    maximum_local_relative_depth_span: float = 0.08
    minimum_depth_tolerance_m: float = 1.0
    maximum_depth_tolerance_m: float = 3.0
    relative_depth_tolerance: float = 1e-4
    minimum_conditional_visibility_trials: int = 14

    def validate(self) -> None:
        if self.query_grid_columns < 2 or self.query_grid_rows < 2:
            raise ValueError("candidate-validation query grid must be at least 2x2")
        if self.folds != 4:
            raise ValueError("candidate validation uses the predeclared four-fold partition")
        if self.max_holdout_matches_per_frame < 1:
            raise ValueError("candidate-validation holdout cap must be positive")
        if not 0.5 < self.confidence_level < 1.0:
            raise ValueError("candidate-validation confidence level must be in (0.5, 1)")
        if not 0.0 < self.minimum_testable_fraction <= 1.0:
            raise ValueError("minimum candidate-render testable fraction must be in (0, 1]")
        if not 0.0 < self.minimum_visibility_consistency_fraction <= 1.0:
            raise ValueError("minimum visibility-consistency fraction must be in (0, 1]")
        if self.render_resolution_multiplier != 2:
            raise ValueError("candidate validation uses the predeclared 2x auxiliary render")
        if not np.isfinite(self.maximum_local_absolute_depth_span_m) or (
            self.maximum_local_absolute_depth_span_m <= 0.0
        ):
            raise ValueError("candidate-render maximum absolute depth span must be finite and positive")
        if not 0.0 < self.maximum_local_relative_depth_span <= 0.10:
            raise ValueError("candidate-render maximum local relative depth span must be in (0, 0.10]")
        if not np.isfinite(self.minimum_depth_tolerance_m) or self.minimum_depth_tolerance_m <= 0.0:
            raise ValueError("minimum candidate-render depth tolerance must be finite and positive")
        if not np.isfinite(self.maximum_depth_tolerance_m) or (
            self.maximum_depth_tolerance_m < self.minimum_depth_tolerance_m
        ):
            raise ValueError("maximum candidate-render depth tolerance cannot be smaller than its minimum")
        if not 0.0 < self.relative_depth_tolerance <= 1e-3:
            raise ValueError("candidate-render relative depth tolerance must be in (0, 1e-3]")
        if self.minimum_conditional_visibility_trials != 14:
            raise ValueError("candidate validation uses the predeclared 14 conditional visibility trials")


@dataclass(frozen=True)
class CandidateVisibility:
    """Z-buffer ordering for held-out terrain points in a candidate render."""

    testable: NDArray[np.bool_]
    consistent: NDArray[np.bool_]
    occluded: NDArray[np.bool_]
    in_front: NDArray[np.bool_]
    outside_auxiliary_frustum: NDArray[np.bool_]
    missing_depth_support: NDArray[np.bool_]
    discontinuous_depth_support: NDArray[np.bool_]
    signed_depth_residual_m: NDArray[np.float64]
    tolerance_m: NDArray[np.float64]


def query_holdout_fold(query_sha256: str, config: CandidateValidationConfig) -> int:
    """Choose one stable fold from query content, never from benchmark truth."""

    if len(query_sha256) != 64:
        raise ValueError("query content SHA-256 must contain 64 hexadecimal characters")
    return int(query_sha256[:16], 16) % config.folds


def query_spatial_holdout_mask(
    query_xy_px: NDArray[np.float64],
    query_camera: CameraModel,
    config: CandidateValidationConfig,
    holdout_fold: int,
) -> NDArray[np.bool_]:
    """Return an interleaved normalized-grid fold independent of match scores."""

    points = np.asarray(query_xy_px, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"query matches must have shape (N, 2), got {points.shape}")
    if not 0 <= holdout_fold < config.folds:
        raise ValueError("candidate-validation holdout fold is outside the configured fold count")
    if not config.enabled or not len(points):
        return np.zeros(len(points), dtype=np.bool_)
    normalized_x = np.clip(points[:, 0] / max(query_camera.width_px, 1), 0.0, np.nextafter(1.0, 0.0))
    normalized_y = np.clip(points[:, 1] / max(query_camera.height_px, 1), 0.0, np.nextafter(1.0, 0.0))
    cell_x = np.floor(normalized_x * config.query_grid_columns).astype(np.int64)
    cell_y = np.floor(normalized_y * config.query_grid_rows).astype(np.int64)
    # Three is coprime to the frozen four-fold policy, which spreads each fold
    # across the image rather than reserving a contiguous corner.
    fold = (cell_x + 3 * cell_y) % config.folds
    return np.asarray(fold == holdout_fold, dtype=np.bool_)


def candidate_render_extrinsics(
    query_camera: CameraModel,
    candidate: CameraExtrinsics,
) -> CameraExtrinsics:
    """Resolve the auxiliary visibility frustum orientation."""

    if query_camera.projection == "cyltan":
        return CameraExtrinsics(
            position=candidate.position,
            yaw_deg=candidate.yaw_deg,
            # The cyltan nuisance identifies the elevation at the query's
            # centre row. Reuse it only to centre the auxiliary pinhole
            # frustum; its pixels are never compared to query pixels.
            pitch_deg=candidate.pitch_deg,
            roll_deg=0.0,
        )
    return candidate


def candidate_zbuffer_visibility(
    render: TerrainRenderBundle,
    world_xyz_m: NDArray[np.float64],
    *,
    max_absolute_depth_span_m: float,
    max_relative_depth_span: float,
    minimum_depth_tolerance_m: float,
    maximum_depth_tolerance_m: float,
    relative_depth_tolerance: float,
) -> CandidateVisibility:
    """Compare terrain points to the first visible surface in a candidate render."""

    world = np.asarray(world_xyz_m, dtype=np.float64)
    if world.ndim != 2 or world.shape[1] != 3:
        raise ValueError(f"candidate-visibility world points must have shape (N, 3), got {world.shape}")
    if (
        not np.isfinite(max_absolute_depth_span_m)
        or not np.isfinite(max_relative_depth_span)
        or (max_absolute_depth_span_m <= 0.0 or max_relative_depth_span <= 0.0)
    ):
        raise ValueError("candidate-visibility depth-span limits must be positive")
    if not np.isfinite(minimum_depth_tolerance_m) or minimum_depth_tolerance_m <= 0.0:
        raise ValueError("candidate-visibility depth tolerance must be positive")
    if not np.isfinite(maximum_depth_tolerance_m) or maximum_depth_tolerance_m < minimum_depth_tolerance_m:
        raise ValueError("candidate-visibility maximum tolerance cannot be smaller than its minimum")
    if not np.isfinite(relative_depth_tolerance) or relative_depth_tolerance <= 0.0:
        raise ValueError("candidate-visibility relative tolerance must be positive")
    count = len(world)
    testable = np.zeros(count, dtype=np.bool_)
    consistent = np.zeros(count, dtype=np.bool_)
    occluded = np.zeros(count, dtype=np.bool_)
    in_front = np.zeros(count, dtype=np.bool_)
    outside_auxiliary_frustum = np.zeros(count, dtype=np.bool_)
    missing_depth_support = np.zeros(count, dtype=np.bool_)
    discontinuous_depth_support = np.zeros(count, dtype=np.bool_)
    residual = np.full(count, np.nan, dtype=np.float64)
    tolerance = np.full(count, np.nan, dtype=np.float64)
    if not count:
        return CandidateVisibility(
            testable,
            consistent,
            occluded,
            in_front,
            outside_auxiliary_frustum,
            missing_depth_support,
            discontinuous_depth_support,
            residual,
            tolerance,
        )

    u_px, v_px, expected_depth_m, projection_valid = project_points(
        world,
        render.intrinsics,
        render.extrinsics,
    )
    height, width = render.forward_depth_m.shape
    finite_projection = projection_valid & np.isfinite(u_px) & np.isfinite(v_px) & np.isfinite(expected_depth_m)
    inside = finite_projection & (u_px >= 1.0) & (u_px <= width - 2.0) & (v_px >= 1.0) & (v_px <= height - 2.0)
    outside_auxiliary_frustum[:] = ~inside
    safe_u = np.where(finite_projection, u_px, 0.0)
    safe_v = np.where(finite_projection, v_px, 0.0)
    finite_depth = np.isfinite(render.forward_depth_m) & (render.forward_depth_m > 0.0)
    inverse_depth = np.full(render.forward_depth_m.shape, np.nan, dtype=np.float64)
    inverse_depth[finite_depth] = 1.0 / render.forward_depth_m[finite_depth]
    sampled_inverse_depth = map_coordinates(
        inverse_depth,
        [safe_v, safe_u],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    visible_depth_m = np.full(count, np.nan, dtype=np.float64)
    valid_sample = np.isfinite(sampled_inverse_depth) & (sampled_inverse_depth > 0.0)
    visible_depth_m[valid_sample] = 1.0 / sampled_inverse_depth[valid_sample]
    missing_depth_support[inside & ~valid_sample] = True
    for index in np.flatnonzero(inside & valid_sample):
        center_column = int(math.floor(float(u_px[index]) + 0.5))
        center_row = int(math.floor(float(v_px[index]) + 0.5))
        window = render.forward_depth_m[
            center_row - 1 : center_row + 2,
            center_column - 1 : center_column + 2,
        ]
        if window.size != 9 or not np.all(np.isfinite(window)) or np.any(window <= 0.0):
            missing_depth_support[index] = True
            continue
        median_depth = float(np.median(window))
        depth_span = float(np.max(window) - np.min(window))
        maximum_span = min(max_absolute_depth_span_m, max_relative_depth_span * median_depth)
        if depth_span > maximum_span:
            discontinuous_depth_support[index] = True
            continue
        point_tolerance = max(
            minimum_depth_tolerance_m,
            min(maximum_depth_tolerance_m, relative_depth_tolerance * float(expected_depth_m[index])),
        )
        signed_residual = float(expected_depth_m[index] - visible_depth_m[index])
        testable[index] = True
        residual[index] = signed_residual
        tolerance[index] = point_tolerance
        if abs(signed_residual) <= point_tolerance:
            consistent[index] = True
        elif signed_residual > point_tolerance:
            occluded[index] = True
        else:
            in_front[index] = True
    reason_membership = (
        testable.astype(np.int8)
        + outside_auxiliary_frustum.astype(np.int8)
        + missing_depth_support.astype(np.int8)
        + discontinuous_depth_support.astype(np.int8)
    )
    if np.any(reason_membership != 1):
        raise RuntimeError("candidate visibility did not assign exactly one coverage category per point")
    return CandidateVisibility(
        testable,
        consistent,
        occluded,
        in_front,
        outside_auxiliary_frustum,
        missing_depth_support,
        discontinuous_depth_support,
        residual,
        tolerance,
    )


def validate_candidate_pose(
    *,
    source_render: TerrainRenderBundle,
    source_frame_index: int,
    holdout_world: NDArray[np.float64],
    holdout_query_xy: NDArray[np.float64],
    candidate_render: TerrainRenderBundle,
    query_camera: CameraModel,
    candidate: CameraExtrinsics,
    pnp: PoseRansacConfig,
    config: CandidateValidationConfig,
    holdout_fold: int,
) -> dict[str, Any]:
    """Gate one training-selected candidate on never-fit spatial evidence."""

    source_surface_identity = source_render.provenance["render_surface_identity_sha256"]
    candidate_surface_identity = candidate_render.provenance["render_surface_identity_sha256"]
    if candidate_surface_identity != source_surface_identity:
        raise RuntimeError("candidate validation render changed the estimator terrain surface")
    expected_validation_shape = (
        source_render.intrinsics.width_px * config.render_resolution_multiplier,
        source_render.intrinsics.height_px * config.render_resolution_multiplier,
    )
    actual_validation_shape = (
        candidate_render.intrinsics.width_px,
        candidate_render.intrinsics.height_px,
    )
    if actual_validation_shape != expected_validation_shape:
        raise RuntimeError("candidate validation render dimensions do not match the predeclared resolution multiplier")
    expected_candidate_intrinsics = CameraIntrinsics.from_horizontal_fov(
        expected_validation_shape[0],
        expected_validation_shape[1],
        source_render.intrinsics.horizontal_fov_deg(),
    )
    if candidate_render.intrinsics != expected_candidate_intrinsics:
        raise RuntimeError("candidate validation render changed the auxiliary camera intrinsics")
    expected_candidate_extrinsics = candidate_render_extrinsics(query_camera, candidate)
    if candidate_render.extrinsics != expected_candidate_extrinsics:
        raise RuntimeError("candidate validation render does not use the selected candidate pose")
    world = np.asarray(holdout_world, dtype=np.float64)
    query_xy = np.asarray(holdout_query_xy, dtype=np.float64)
    count = len(world)
    errors = reprojection_errors(world, query_xy, query_camera, candidate)
    reprojection_inliers = np.isfinite(errors) & (errors <= pnp.reprojection_threshold_px)
    visibility = candidate_zbuffer_visibility(
        candidate_render,
        world,
        max_absolute_depth_span_m=config.maximum_local_absolute_depth_span_m,
        max_relative_depth_span=config.maximum_local_relative_depth_span,
        minimum_depth_tolerance_m=config.minimum_depth_tolerance_m,
        maximum_depth_tolerance_m=config.maximum_depth_tolerance_m,
        relative_depth_tolerance=config.relative_depth_tolerance,
    )
    testable_reprojection_inliers = visibility.testable & reprojection_inliers
    joint_support = visibility.consistent & reprojection_inliers
    holdout_distribution = correspondence_distribution(query_xy, query_camera)
    joint_distribution = correspondence_distribution(query_xy[joint_support], query_camera)
    joint_world_geometry = world_consensus_geometry(
        world[joint_support],
        np.asarray(candidate.position.as_tuple(), dtype=np.float64),
        unique_tolerance_m=pnp.world_unique_tolerance_m,
    )
    joint_world_geometry_gate = evaluate_world_degeneracy(joint_world_geometry, pnp)
    minimum_holdout = max(pnp.min_correspondences, pnp.min_inliers)
    minimum_joint = pnp.min_inliers
    testable_count = int(visibility.testable.sum())
    testable_reprojection_count = int(testable_reprojection_inliers.sum())
    consistent_reprojection_count = int((visibility.consistent & testable_reprojection_inliers).sum())
    joint_count = int(joint_support.sum())
    testable_lower = exact_binomial_lower_bound(testable_count, count, config.confidence_level)
    visibility_lower = exact_binomial_lower_bound(
        consistent_reprojection_count,
        testable_reprojection_count,
        config.confidence_level,
    )
    joint_lower = exact_binomial_lower_bound(joint_count, count, config.confidence_level)
    failures: list[str] = []
    if count < minimum_holdout:
        failures.append("insufficient_spatial_holdout_correspondences")
    if not _distribution_passes(holdout_distribution, pnp):
        failures.append("spatial_holdout_poorly_distributed")
    if testable_lower < config.minimum_testable_fraction:
        failures.append("candidate_render_has_insufficient_testable_support")
    if testable_reprojection_count < config.minimum_conditional_visibility_trials:
        failures.append("insufficient_conditional_visibility_trials")
    elif visibility_lower < config.minimum_visibility_consistency_fraction:
        failures.append("heldout_reprojection_inliers_violate_visibility_ordering")
    if joint_count < minimum_joint or joint_lower < pnp.min_inlier_ratio:
        failures.append("heldout_joint_consensus_below_acceptance_gate")
    if not _distribution_passes(joint_distribution, pnp):
        failures.append("heldout_joint_consensus_poorly_distributed")
    if not joint_world_geometry_gate["passed"]:
        failures.append("heldout_joint_world_geometry_degenerate")

    return {
        "schema": CANDIDATE_VALIDATION_SCHEMA,
        "enabled": True,
        "passed": not failures,
        "failures": failures,
        "partition": {
            "schema": HOLDOUT_PARTITION_SCHEMA,
            "method": HOLDOUT_PARTITION_METHOD,
            "grid": {"columns": config.query_grid_columns, "rows": config.query_grid_rows},
            "folds": config.folds,
            "heldout_fold": holdout_fold,
            "source_frame_index": source_frame_index,
            "withheld_from_initial_geometric_pose_fit": True,
            "withheld_from_refinement_geometric_pose_fit": True,
            "withheld_from_geometric_frame_ranking": True,
            "matcher_used_full_query_image": True,
            "worker_candidate_selection_precedes_holdout": True,
        },
        "source_render": _validation_render_record(source_render),
        "candidate_render": _validation_render_record(candidate_render),
        "render_contract": {
            "resolution_multiplier": config.render_resolution_multiplier,
            "expected_candidate_dimensions_px": list(expected_validation_shape),
            "actual_candidate_dimensions_px": list(actual_validation_shape),
            "source_surface_identity_sha256": source_surface_identity,
            "candidate_surface_identity_sha256": candidate_surface_identity,
            "surface_identity_matches": True,
            "expected_candidate_intrinsics": expected_candidate_intrinsics.model_dump(mode="json"),
            "actual_candidate_intrinsics": candidate_render.intrinsics.model_dump(mode="json"),
            "candidate_intrinsics_match": True,
            "expected_candidate_extrinsics": expected_candidate_extrinsics.model_dump(mode="json"),
            "actual_candidate_extrinsics": candidate_render.extrinsics.model_dump(mode="json"),
            "candidate_extrinsics_match": True,
        },
        "candidate_pose": candidate.model_dump(mode="json"),
        "candidate_render_pitch_semantics": (
            "cyltan crop-shift nuisance used only to centre the auxiliary pinhole visibility frustum; "
            "not a calibrated physical-attitude claim"
            if query_camera.projection == "cyltan"
            else "physical candidate pinhole pitch and roll"
        ),
        "projection_separation": {
            "query_reprojection": query_camera.projection,
            "visibility_render": "auxiliary_pinhole",
            "depth_comparison": (
                "heldout world-point forward depth and z-buffer depth are both computed in the same "
                "auxiliary pinhole camera"
            ),
            "cross_projection_depth_comparison": False,
            "outside_auxiliary_render_coverage": "untestable",
        },
        "counts": {
            "heldout_correspondences": count,
            "minimum_heldout_correspondences": minimum_holdout,
            "query_reprojection_inliers": int(reprojection_inliers.sum()),
            "candidate_render_testable": testable_count,
            "outside_auxiliary_frustum": int(visibility.outside_auxiliary_frustum.sum()),
            "missing_or_sky_depth_support": int(visibility.missing_depth_support.sum()),
            "discontinuous_depth_support": int(visibility.discontinuous_depth_support.sum()),
            "testable_query_reprojection_inliers": testable_reprojection_count,
            "minimum_conditional_visibility_trials": config.minimum_conditional_visibility_trials,
            "visibility_consistent_reprojection_inliers": consistent_reprojection_count,
            "occluded": int(visibility.occluded.sum()),
            "impossibly_in_front": int(visibility.in_front.sum()),
            "joint_support": joint_count,
            "minimum_joint_support": minimum_joint,
        },
        "reprojection": {
            "threshold_px": pnp.reprojection_threshold_px,
            "median_error_px": _finite_percentile(errors, 50.0),
            "p90_error_px": _finite_percentile(errors, 90.0),
            "inlier_median_error_px": _finite_percentile(errors[reprojection_inliers], 50.0),
            "holdout_distribution": holdout_distribution,
            "joint_support_distribution": joint_distribution,
        },
        "joint_world_geometry": joint_world_geometry,
        "joint_world_geometry_gate": joint_world_geometry_gate,
        "visibility": {
            "signed_depth_residual_semantics": (
                "heldout point forward depth minus candidate z-buffer depth; positive means occluded"
            ),
            "depth_interpolation": "bilinear inverse depth, matching the rasterizer's z-buffer quantity",
            "local_support_policy": (
                "all-finite 3x3; depth span <= min(configured absolute metres, configured relative * median)"
            ),
            "local_support_limitation": (
                "image-space depth span cannot reliably distinguish a steep smooth terrain face from an "
                "occlusion edge; this cap removes only gross discontinuities, while the independent 1-3 metre "
                "point-to-z-buffer residual performs the ordering test; exact triangle/ray visibility remains "
                "future work"
            ),
            "depth_tolerance_policy": (
                "max(configured minimum metres, min(configured maximum metres, relative * expected depth))"
            ),
            "maximum_local_absolute_depth_span_m": config.maximum_local_absolute_depth_span_m,
            "minimum_depth_tolerance_m": config.minimum_depth_tolerance_m,
            "maximum_depth_tolerance_m": config.maximum_depth_tolerance_m,
            "relative_depth_tolerance": config.relative_depth_tolerance,
            "maximum_relative_depth_span": config.maximum_local_relative_depth_span,
            "median_signed_depth_residual_m": _finite_percentile(
                visibility.signed_depth_residual_m[visibility.testable],
                50.0,
            ),
            "p90_absolute_depth_residual_m": _finite_percentile(
                np.abs(visibility.signed_depth_residual_m[visibility.testable]),
                90.0,
            ),
            "median_tolerance_m": _finite_percentile(visibility.tolerance_m[visibility.testable], 50.0),
        },
        "gates": {
            "nominal_one_sided_clopper_pearson_confidence_level": config.confidence_level,
            "binomial_model_note": (
                "Clopper-Pearson is exact only under an independent Bernoulli-trial model; dense learned "
                "matches are spatially correlated, so these nominal bounds are heuristic only and can be "
                "overconfident; they are not calibrated real-world coverage guarantees"
            ),
            "testable_fraction": _binomial_gate_record(
                testable_count,
                count,
                testable_lower,
                config.minimum_testable_fraction,
            ),
            "visibility_consistency_among_testable_reprojection_inliers": _binomial_gate_record(
                consistent_reprojection_count,
                testable_reprojection_count,
                visibility_lower,
                config.minimum_visibility_consistency_fraction,
            ),
            "joint_support_fraction": _binomial_gate_record(
                joint_count,
                count,
                joint_lower,
                pnp.min_inlier_ratio,
            ),
        },
        "uses_reference_truth": False,
        "withheld_from_geometric_pose_fit": True,
        "withheld_from_geometric_frame_ranking": True,
        "matcher_used_full_query_image": True,
        "worker_candidate_selection_precedes_holdout": True,
        "used_as_final_acceptance_gate": True,
        "runner_up_retry_attempted": False,
    }


def exact_binomial_lower_bound(successes: int, trials: int, confidence_level: float) -> float:
    """One-sided Clopper-Pearson lower bound under the nominal Bernoulli model."""

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("binomial successes/trials are inconsistent")
    if trials == 0 or successes == 0:
        return 0.0
    return float(beta_distribution.ppf(1.0 - confidence_level, successes, trials - successes + 1))


def _distribution_passes(distribution: dict[str, Any], config: PoseRansacConfig) -> bool:
    return bool(
        distribution["occupied_grid_cells"] >= config.min_query_grid_cells
        and distribution["x_span_fraction"] >= config.min_query_x_span_fraction
        and distribution["y_span_fraction"] >= config.min_query_y_span_fraction
    )


def _binomial_gate_record(
    successes: int,
    trials: int,
    lower_bound: float,
    minimum: float,
) -> dict[str, Any]:
    return {
        "successes": successes,
        "trials": trials,
        "observed_fraction": round(successes / trials, 9) if trials else 0.0,
        "lower_confidence_bound": round(lower_bound, 9),
        "required_lower_bound": minimum,
        "passed": lower_bound >= minimum,
    }


def _validation_render_record(render: TerrainRenderBundle) -> dict[str, Any]:
    return {
        "render_content_sha256": render.provenance["render_content_sha256"],
        "render_surface_identity_sha256": render.provenance["render_surface_identity_sha256"],
        "intrinsics": render.intrinsics.model_dump(mode="json"),
        "extrinsics": render.extrinsics.model_dump(mode="json"),
        "modality": render.modality,
        "uses_query_image": False,
        "uses_reference_pose": False,
        "uses_source_depth_pfm": False,
    }


def _finite_percentile(values: NDArray[np.float64], percentile: float) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return round(float(np.percentile(finite, percentile)), 6) if finite.size else None
