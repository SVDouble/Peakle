"""Truth-blind photo verification for a frozen skyline-atlas candidate pool.

The verifier consumes only the source photograph, explicitly identified photo
models, estimator terrain, and a previously frozen ``photo_auto`` atlas.  It
never accepts reference depth or numeric pose truth.  Every atlas candidate is
rendered in the atlas' exact cylindrical/tangent crop geometry before a diverse
beam is selected.  Numeric truth is accepted only by the separate post-freeze
evaluator at the bottom of this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt
from scipy.stats import rankdata

from peakle.depth import DepthEstimator
from peakle.domain.camera import CameraExtrinsics
from peakle.domain.terrain import TerrainMap
from peakle.edges import EdgeDetector
from peakle.localize.atlas_geometry import (
    render_cyltan_candidate_depth,
    terrain_diagonal_range_m,
    terrain_surface_identity,
    validate_frozen_cyltan_atlas,
)
from peakle.localize.extract import valid_image_mask
from peakle.localize.skyline_atlas import ATLAS_CANDIDATE_SCHEMA, POSITION_SUCCESS_M, YAW_SUCCESS_DEG
from peakle.localize.swissdem import Patch
from peakle.localize.typed_outlines import TypedOutlines, extract_typed_outlines
from peakle.segmentation import RidgeField, extract_ridges

PHOTO_VERIFIER_ARCHIVE_SCHEMA = "peakle_photo_geometry_verifier_archive_v1"
PHOTO_VERIFIER_CANDIDATE_SCHEMA = "peakle_photo_geometry_verifier_candidate_v1"
PHOTO_VERIFIER_EVALUATION_SCHEMA = "peakle_photo_geometry_verifier_evaluation_v1"


@dataclass(frozen=True, slots=True)
class PhotoGeometryVerifierConfig:
    """Predeclared photo score, evidence gates, and diverse-beam policy."""

    subsample: int = 4
    min_depth_m: float = 250.0
    outline_cap_source_px: float = 10.0
    typed_min_component_px: int = 6
    min_candidate_outline_px: int = 12
    skyline_exclusion_source_px: int = 8
    ordinal_grid_columns: int = 16
    ordinal_grid_rows: int = 12
    ordinal_min_samples_per_cell: int = 8
    ordinal_min_cells: int = 24
    ordinal_min_photo_span: float = 0.10
    min_common_depth_px: int = 200
    minimum_skyline_coverage: float = 0.25
    minimum_ridge_effective_samples: float = 64.0
    ridge_horizontal_bins: int = 8
    minimum_ridge_horizontal_bins: int = 4
    minimum_winner_terrain_overlap_f1: float = 0.50
    minimum_winner_common_terrain_fraction: float = 0.25
    skyline_weight: float = 0.20
    outline_weight: float = 0.35
    ordinal_depth_weight: float = 0.35
    terrain_overlap_weight: float = 0.10
    beam_size: int = 32
    nms_position_m: float = 75.0
    nms_yaw_deg: float = 3.0
    rival_position_m: float = 100.0
    rival_yaw_deg: float = 5.0
    maximum_selected_score: float = 0.45
    minimum_rival_margin: float = 0.05
    stability_folds: int = 4
    minimum_stable_folds: int = 3

    def __post_init__(self) -> None:
        integer_fields = {
            "subsample": self.subsample,
            "typed_min_component_px": self.typed_min_component_px,
            "min_candidate_outline_px": self.min_candidate_outline_px,
            "ordinal_grid_columns": self.ordinal_grid_columns,
            "ordinal_grid_rows": self.ordinal_grid_rows,
            "ordinal_min_samples_per_cell": self.ordinal_min_samples_per_cell,
            "ordinal_min_cells": self.ordinal_min_cells,
            "min_common_depth_px": self.min_common_depth_px,
            "ridge_horizontal_bins": self.ridge_horizontal_bins,
            "minimum_ridge_horizontal_bins": self.minimum_ridge_horizontal_bins,
            "beam_size": self.beam_size,
            "stability_folds": self.stability_folds,
            "minimum_stable_folds": self.minimum_stable_folds,
        }
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in integer_fields.values()):
            raise ValueError("verifier count settings must be positive integers")
        if (
            isinstance(self.skyline_exclusion_source_px, bool)
            or not isinstance(self.skyline_exclusion_source_px, int)
            or self.skyline_exclusion_source_px < 0
        ):
            raise ValueError("skyline_exclusion_source_px must be a non-negative integer")
        positive = (
            self.min_depth_m,
            self.outline_cap_source_px,
            self.ordinal_min_photo_span,
            self.minimum_ridge_effective_samples,
            self.nms_position_m,
            self.nms_yaw_deg,
            self.rival_position_m,
            self.rival_yaw_deg,
            self.maximum_selected_score,
            self.minimum_rival_margin,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("verifier scale and gate settings must be positive and finite")
        if not 0.0 <= self.minimum_skyline_coverage <= 1.0:
            raise ValueError("minimum_skyline_coverage must be in [0, 1]")
        fractions = (
            self.minimum_winner_terrain_overlap_f1,
            self.minimum_winner_common_terrain_fraction,
        )
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("verifier support fractions must be finite and in [0, 1]")
        if self.minimum_ridge_horizontal_bins > self.ridge_horizontal_bins:
            raise ValueError("minimum ridge-bin coverage exceeds the configured bin count")
        if self.minimum_stable_folds > self.stability_folds:
            raise ValueError("minimum stable folds exceeds the configured fold count")
        weights = (
            self.skyline_weight,
            self.outline_weight,
            self.ordinal_depth_weight,
            self.terrain_overlap_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("fusion weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fusion weights must sum to one")
        if math.isclose(sum(weights[1:]), 0.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("stability folds require a non-skyline cue weight")

    @property
    def outline_cap_comparison_px(self) -> float:
        return self.outline_cap_source_px / self.subsample

    def to_record(self) -> dict[str, Any]:
        return {
            "subsample": self.subsample,
            "min_depth_m": self.min_depth_m,
            "outline_cap_source_px": self.outline_cap_source_px,
            "outline_cap_comparison_px": self.outline_cap_comparison_px,
            "typed_min_component_px": self.typed_min_component_px,
            "min_candidate_outline_px": self.min_candidate_outline_px,
            "skyline_exclusion_source_px": self.skyline_exclusion_source_px,
            "ordinal_grid": [self.ordinal_grid_rows, self.ordinal_grid_columns],
            "ordinal_min_samples_per_cell": self.ordinal_min_samples_per_cell,
            "ordinal_min_cells": self.ordinal_min_cells,
            "ordinal_min_photo_span": self.ordinal_min_photo_span,
            "min_common_depth_px": self.min_common_depth_px,
            "evidence_gates": {
                "minimum_skyline_coverage": self.minimum_skyline_coverage,
                "minimum_ridge_effective_samples": self.minimum_ridge_effective_samples,
                "ridge_horizontal_bins": self.ridge_horizontal_bins,
                "minimum_ridge_horizontal_bins": self.minimum_ridge_horizontal_bins,
            },
            "weights": {
                "skyline": self.skyline_weight,
                "symmetric_internal_outline": self.outline_weight,
                "ordinal_depth": self.ordinal_depth_weight,
                "terrain_overlap": self.terrain_overlap_weight,
            },
            "beam": {
                "size": self.beam_size,
                "nms_position_m": self.nms_position_m,
                "nms_yaw_deg": self.nms_yaw_deg,
            },
            "decision": {
                "maximum_selected_score": self.maximum_selected_score,
                "minimum_rival_margin": self.minimum_rival_margin,
                "rival_position_m": self.rival_position_m,
                "rival_yaw_deg": self.rival_yaw_deg,
                "stability_folds": self.stability_folds,
                "minimum_stable_folds": self.minimum_stable_folds,
                "minimum_winner_terrain_overlap_f1": self.minimum_winner_terrain_overlap_f1,
                "minimum_winner_common_terrain_fraction": self.minimum_winner_common_terrain_fraction,
                "minimum_winner_common_depth_px": self.min_common_depth_px,
                "minimum_winner_ordinal_cells": self.ordinal_min_cells,
                "cue_agreement_required": True,
                "fold_score": "renormalized_photo_geometry_cues_without_full_image_skyline_score",
                "calibration": "uncalibrated_predeclared_research_gate",
            },
            "candidate_depth_convention": "horizontal_ray_march_range_converted_to_camera_ray_range",
            "photo_depth_semantics": "monocular_relative_depth_far_is_large_scale_invariant_ordinal_only",
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryEvidence:
    """Photo-derived arrays frozen before any candidate or truth evaluation."""

    image_width_px: int
    image_height_px: int
    comparison_width_px: int
    comparison_height_px: int
    extraction_config_sha256: str
    source_atlas_sha256: str
    observation: dict[str, Any]
    observed_skyline_sha256: str
    photo_rgb_sha256: str
    edge_response_sha256: str
    relative_depth_sha256: str
    comparison_relative_depth_sha256: str
    valid_mask_sha256: str
    terrain_mask_sha256: str
    ridge_weights_sha256: str
    edge_model: dict[str, Any]
    depth_model: dict[str, Any]
    skyline_coverage: float
    ridge_effective_samples: float
    ridge_horizontal_bins: int
    relative_depth_span_p90_p10: float
    usable: bool
    rejection_reasons: tuple[str, ...]
    valid_mask: NDArray[np.bool_]
    terrain_mask: NDArray[np.bool_]
    ridge_weights: NDArray[np.float64]
    relative_depth: NDArray[np.float64]

    def to_record(self) -> dict[str, Any]:
        return {
            "source": "photo_rgb_only",
            "reference_depth_used": False,
            "numeric_reference_pose_used": False,
            "shape": [self.image_height_px, self.image_width_px],
            "comparison_shape": [self.comparison_height_px, self.comparison_width_px],
            "extraction_config_sha256": self.extraction_config_sha256,
            "source_atlas_sha256": self.source_atlas_sha256,
            "observation": self.observation,
            "observed_skyline_sha256": self.observed_skyline_sha256,
            "photo_rgb_sha256": self.photo_rgb_sha256,
            "edge_response_sha256": self.edge_response_sha256,
            "relative_depth_sha256": self.relative_depth_sha256,
            "comparison_relative_depth_sha256": self.comparison_relative_depth_sha256,
            "valid_mask_sha256": self.valid_mask_sha256,
            "terrain_mask_sha256": self.terrain_mask_sha256,
            "ridge_weights_sha256": self.ridge_weights_sha256,
            "models": {"edge": self.edge_model, "depth": self.depth_model},
            "quality": {
                "skyline_coverage": self.skyline_coverage,
                "ridge_effective_samples": self.ridge_effective_samples,
                "ridge_horizontal_bins": self.ridge_horizontal_bins,
                "relative_depth_span_p90_p10": self.relative_depth_span_p90_p10,
                "usable": self.usable,
                "rejection_reasons": list(self.rejection_reasons),
            },
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryScoreTerms:
    skyline_loss: float
    symmetric_outline_loss: float
    photo_to_dem_outline_loss: float
    dem_to_photo_outline_loss: float
    ordinal_depth_loss: float
    terrain_overlap_loss: float
    terrain_overlap_f1: float
    common_depth_px: int
    common_terrain_fraction: float
    photo_terrain_px: int
    candidate_terrain_px: int
    ordinal_cells: int
    candidate_outline_px: int
    family_candidate_px: dict[str, int]
    family_dem_to_photo_loss: dict[str, float]

    def to_record(self) -> dict[str, Any]:
        return {
            "skyline_loss": self.skyline_loss,
            "symmetric_outline_loss": self.symmetric_outline_loss,
            "photo_to_dem_outline_loss": self.photo_to_dem_outline_loss,
            "dem_to_photo_outline_loss": self.dem_to_photo_outline_loss,
            "ordinal_depth_loss": self.ordinal_depth_loss,
            "terrain_overlap_loss": self.terrain_overlap_loss,
            "terrain_overlap_f1": self.terrain_overlap_f1,
            "common_depth_px": self.common_depth_px,
            "common_terrain_fraction": self.common_terrain_fraction,
            "photo_terrain_px": self.photo_terrain_px,
            "candidate_terrain_px": self.candidate_terrain_px,
            "ordinal_cells": self.ordinal_cells,
            "candidate_outline_px": self.candidate_outline_px,
            "families": {
                family: {
                    "candidate_px": self.family_candidate_px[family],
                    "dem_to_photo_loss": self.family_dem_to_photo_loss[family],
                }
                for family in ("occlusion", "rib", "couloir")
            },
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryVerifiedCandidate:
    candidate_id: str
    original_estimator_rank: int
    verifier_rank: int
    beam_rank: int | None
    atlas_candidate: dict[str, Any]
    terms: PhotoGeometryScoreTerms
    fusion_score: float
    fold_fusion_scores: tuple[float, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHOTO_VERIFIER_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "original_estimator_rank": self.original_estimator_rank,
            "verifier_rank": self.verifier_rank,
            "beam_rank": self.beam_rank,
            "atlas_candidate": self.atlas_candidate,
            "terms": self.terms.to_record(),
            "fusion_score": self.fusion_score,
            "fold_fusion_scores": list(self.fold_fusion_scores),
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryVerifierDecision:
    status: str
    returned_candidate_id: str | None
    ranked_winner_candidate_id: str
    rival_candidate_id: str | None
    rival_margin: float | None
    stable_folds: int
    reasons: tuple[str, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "returned_candidate_id": self.returned_candidate_id,
            "ranked_winner_candidate_id": self.ranked_winner_candidate_id,
            "rival_candidate_id": self.rival_candidate_id,
            "rival_margin": self.rival_margin,
            "stable_folds": self.stable_folds,
            "reasons": list(self.reasons),
            "fallback": "retain_supplied_prior_outside_this_archive" if self.status == "abstained" else None,
            "calibrated": False,
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryVerifierArchive:
    config: PhotoGeometryVerifierConfig
    source_atlas_sha256: str
    evidence: PhotoGeometryEvidence
    horizontal_fov_deg: float
    terrain_horizontal_extent_m: float
    native_patch_supplied: bool
    terrain_surface: dict[str, Any]
    candidates: tuple[PhotoGeometryVerifiedCandidate, ...]
    beam_candidate_ids: tuple[str, ...]
    component_winners: dict[str, str]
    fold_winner_candidate_ids: tuple[str, ...]
    prior_position_competitor_id: str
    decision: PhotoGeometryVerifierDecision
    archive_sha256: str

    @property
    def ranked_winner(self) -> PhotoGeometryVerifiedCandidate:
        return self.candidates[0]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHOTO_VERIFIER_ARCHIVE_SCHEMA,
            "experimental": True,
            "production_eligible": False,
            "photo_observable_evidence_only": True,
            "source_depth_reference_used": False,
            "numeric_evaluation_reference_used": False,
            "config": self.config.to_record(),
            "source_atlas_sha256": self.source_atlas_sha256,
            "query_geometry": {
                "projection": "cyltan",
                "width_px": self.evidence.image_width_px,
                "height_px": self.evidence.image_height_px,
                "horizontal_fov_deg": self.horizontal_fov_deg,
                "comparison_width_px": self.evidence.comparison_width_px,
                "comparison_height_px": self.evidence.comparison_height_px,
                "terrain_horizontal_extent_m": self.terrain_horizontal_extent_m,
            },
            "native_patch_supplied": self.native_patch_supplied,
            "terrain_surface": self.terrain_surface,
            "evidence": self.evidence.to_record(),
            "candidate_pool": "complete_frozen_photo_atlas_pool",
            "candidate_count": len(self.candidates),
            "ranked_winner_candidate_id": self.ranked_winner.candidate_id,
            "component_winners": self.component_winners,
            "fold_winner_candidate_ids": list(self.fold_winner_candidate_ids),
            "prior_position_competitor_id": self.prior_position_competitor_id,
            "beam_policy": "greedy_position_yaw_basin_nms",
            "beam_candidate_ids": list(self.beam_candidate_ids),
            "decision": self.decision.to_record(),
            "archive_sha256": self.archive_sha256,
            "candidates": [candidate.to_record() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryPoseErrors:
    horizontal_position_m: float
    vertical_m: float
    position_3d_m: float
    yaw_deg: float
    pitch_deg: None

    def to_record(self) -> dict[str, float | None]:
        return {
            "horizontal_position_m": self.horizontal_position_m,
            "vertical_m": self.vertical_m,
            "position_3d_m": self.position_3d_m,
            "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg,
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryEvaluatedCandidate:
    candidate_id: str
    verifier_rank: int
    beam_rank: int | None
    original_estimator_rank: int
    errors: PhotoGeometryPoseErrors
    normalized_joint_error: float
    reaches_target: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "verifier_rank": self.verifier_rank,
            "beam_rank": self.beam_rank,
            "original_estimator_rank": self.original_estimator_rank,
            "errors": self.errors.to_record(),
            "normalized_joint_error": self.normalized_joint_error,
            "reaches_target": self.reaches_target,
        }


@dataclass(frozen=True, slots=True)
class PhotoGeometryVerifierEvaluation:
    archive_sha256: str
    ranked_winner_errors: PhotoGeometryEvaluatedCandidate
    returned_candidate_errors: PhotoGeometryEvaluatedCandidate | None
    component_winner_errors: dict[str, PhotoGeometryEvaluatedCandidate]
    candidate_pool_gt_oracle: PhotoGeometryEvaluatedCandidate
    first_target_verifier_rank: int | None
    first_target_beam_rank: int | None
    top_k: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PHOTO_VERIFIER_EVALUATION_SCHEMA,
            "archive_sha256": self.archive_sha256,
            "reference_data_used": True,
            "used_by_estimator": False,
            "target": {
                "horizontal_position_m_lte": POSITION_SUCCESS_M,
                "absolute_yaw_deg_lte": YAW_SUCCESS_DEG,
                "pitch_used_by_oracle": False,
                "roll_used_by_oracle": False,
            },
            "ranked_winner_errors": self.ranked_winner_errors.to_record(),
            "returned_candidate_errors": (
                self.returned_candidate_errors.to_record() if self.returned_candidate_errors is not None else None
            ),
            "component_winner_errors": {
                name: candidate.to_record() for name, candidate in sorted(self.component_winner_errors.items())
            },
            "candidate_pool_gt_oracle": self.candidate_pool_gt_oracle.to_record(),
            "first_target_verifier_rank": self.first_target_verifier_rank,
            "first_target_beam_rank": self.first_target_beam_rank,
            "top_k": list(self.top_k),
        }


def extract_photo_geometry_evidence(
    photo_rgb: NDArray[np.uint8],
    observed_skyline_rows: NDArray[np.float64],
    edge_detector: EdgeDetector,
    depth_estimator: DepthEstimator,
    *,
    observation_provenance: Mapping[str, Any],
    edge_model_provenance: Mapping[str, Any],
    depth_model_provenance: Mapping[str, Any],
    config: PhotoGeometryVerifierConfig | None = None,
) -> PhotoGeometryEvidence:
    """Extract candidate-independent photo cues once, with pinned provenance."""

    settings = config or PhotoGeometryVerifierConfig()
    rgb = np.asarray(photo_rgb)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("photo_rgb must be a non-empty uint8 HxWx3 array")
    height, width = rgb.shape[:2]
    skyline = np.asarray(observed_skyline_rows, dtype=np.float64)
    if skyline.shape != (width,):
        raise ValueError("observed skyline width differs from photo width")
    edge_model = _validated_model_provenance(edge_model_provenance, edge_detector.name, "edge")
    depth_model = _validated_model_provenance(depth_model_provenance, depth_estimator.name, "depth")
    observation = _validated_observation_provenance(observation_provenance)

    rgb_float = rgb.astype(np.float64) / 255.0
    edge_response = _validated_photo_map(edge_detector.detect(rgb_float), (height, width), "edge response")
    relative_depth = _validated_photo_map(depth_estimator.estimate(rgb_float), (height, width), "relative depth")
    valid = valid_image_mask(rgb)
    terrain = _terrain_mask_from_skyline(skyline, valid)
    ridge_field = extract_ridges(rgb_float, depth=relative_depth, edges=edge_response)
    ridge_weights_full = _internal_ridge_weights(
        ridge_field,
        skyline,
        valid,
        settings.skyline_exclusion_source_px,
    )
    ridge_weights = _max_resample(ridge_weights_full, settings.subsample)
    valid_work = valid[:: settings.subsample, :: settings.subsample].copy()
    terrain_work = terrain[:: settings.subsample, :: settings.subsample].copy()
    depth_work = relative_depth[:: settings.subsample, :: settings.subsample].copy()
    ridge_weights = np.where(valid_work & terrain_work, ridge_weights, 0.0)
    depth_work = np.where(valid_work & terrain_work, depth_work, np.nan)

    skyline_coverage = float(np.isfinite(skyline).mean())
    ridge_effective = _effective_samples(ridge_weights)
    ridge_bins = _horizontal_bin_coverage(ridge_weights > 0.0, settings.ridge_horizontal_bins)
    photo_depth_values = depth_work[np.isfinite(depth_work)]
    depth_span = (
        float(np.percentile(photo_depth_values, 90.0) - np.percentile(photo_depth_values, 10.0))
        if photo_depth_values.size
        else 0.0
    )
    reasons: list[str] = []
    if skyline_coverage < settings.minimum_skyline_coverage:
        reasons.append("insufficient_skyline_coverage")
    if ridge_effective < settings.minimum_ridge_effective_samples:
        reasons.append("insufficient_internal_ridge_support")
    if ridge_bins < settings.minimum_ridge_horizontal_bins:
        reasons.append("insufficient_internal_ridge_span")
    if depth_span < settings.ordinal_min_photo_span:
        reasons.append("insufficient_relative_depth_span")

    for array in (valid_work, terrain_work, ridge_weights, depth_work):
        array.setflags(write=False)
    return PhotoGeometryEvidence(
        image_width_px=width,
        image_height_px=height,
        comparison_width_px=valid_work.shape[1],
        comparison_height_px=valid_work.shape[0],
        extraction_config_sha256=_canonical_sha256(settings.to_record()),
        source_atlas_sha256=str(observation["source_atlas_sha256"]),
        observation=observation,
        observed_skyline_sha256=_observed_skyline_sha256(skyline),
        photo_rgb_sha256=_array_sha256(rgb, "peakle_photo_rgb_v1"),
        edge_response_sha256=_array_sha256(edge_response, "peakle_photo_edge_response_v1"),
        relative_depth_sha256=_array_sha256(relative_depth, "peakle_photo_relative_depth_v1"),
        comparison_relative_depth_sha256=_array_sha256(
            depth_work,
            "peakle_photo_comparison_relative_depth_v1",
        ),
        valid_mask_sha256=_array_sha256(valid_work, "peakle_photo_valid_mask_v1"),
        terrain_mask_sha256=_array_sha256(terrain_work, "peakle_photo_terrain_mask_v1"),
        ridge_weights_sha256=_array_sha256(ridge_weights, "peakle_photo_ridge_weights_v1"),
        edge_model=edge_model,
        depth_model=depth_model,
        skyline_coverage=_rounded(skyline_coverage),
        ridge_effective_samples=_rounded(ridge_effective),
        ridge_horizontal_bins=ridge_bins,
        relative_depth_span_p90_p10=_rounded(depth_span),
        usable=not reasons,
        rejection_reasons=tuple(reasons),
        valid_mask=valid_work,
        terrain_mask=terrain_work,
        ridge_weights=ridge_weights,
        relative_depth=depth_work,
    )


def build_photo_geometry_verifier(
    terrain: TerrainMap,
    native_patch: Patch | None,
    evidence: PhotoGeometryEvidence,
    atlas_archive: Mapping[str, Any],
    *,
    config: PhotoGeometryVerifierConfig | None = None,
) -> PhotoGeometryVerifierArchive:
    """Score the complete frozen atlas pool without accepting numeric truth."""

    settings = config or PhotoGeometryVerifierConfig()
    atlas = validate_frozen_cyltan_atlas(atlas_archive)
    query = atlas["query_geometry"]
    width = _positive_int(query.get("width_px"), "query width")
    height = _positive_int(query.get("height_px"), "query height")
    fov = _finite_float(query.get("horizontal_fov_deg"), "horizontal FOV")
    if (width, height) != (evidence.image_width_px, evidence.image_height_px):
        raise ValueError("photo evidence shape differs from the frozen atlas query")
    expected_work_shape = (len(range(0, height, settings.subsample)), len(range(0, width, settings.subsample)))
    if expected_work_shape != (evidence.comparison_height_px, evidence.comparison_width_px):
        raise ValueError("photo evidence comparison grid differs from verifier configuration")
    if evidence.extraction_config_sha256 != _canonical_sha256(settings.to_record()):
        raise ValueError("photo evidence was extracted with a different verifier configuration")
    _validate_evidence_arrays(evidence)
    _validate_evidence_quality(evidence, settings)
    if query.get("observed_skyline_sha256") != evidence.observed_skyline_sha256:
        raise ValueError("photo skyline hash differs from the frozen atlas observation")
    observation = _validated_observation_provenance(evidence.observation)
    if observation != evidence.observation:
        raise ValueError("photo observation provenance is not in canonical validated form")
    if observation["source_atlas_sha256"] != evidence.source_atlas_sha256:
        raise ValueError("photo observation and evidence name different source atlases")
    if evidence.source_atlas_sha256 != atlas.get("archive_sha256"):
        raise ValueError("photo observation provenance names a different source atlas")
    if bool(atlas.get("native_patch_supplied")) != (native_patch is not None):
        raise ValueError("native patch availability differs from the frozen atlas")

    surface = terrain_surface_identity(terrain, native_patch)
    _validate_atlas_coordinate_frame(atlas, surface)
    terrain_horizontal_extent = terrain_diagonal_range_m(terrain)
    drafts: list[PhotoGeometryVerifiedCandidate] = []
    candidates = atlas["candidates"]
    for candidate in candidates:
        rendered = render_cyltan_candidate_depth(
            terrain,
            native_patch,
            candidate,
            width,
            height,
            fov,
            subsample=settings.subsample,
        )
        candidate_typed = _candidate_typed_outlines(
            evidence,
            rendered.candidate_ray_depth,
            settings,
        )
        terms = _score_candidate(
            evidence,
            rendered.candidate_ray_depth,
            candidate,
            atlas,
            settings,
            candidate_typed=candidate_typed,
        )
        fold_scores = tuple(
            _score_candidate(
                evidence,
                rendered.candidate_ray_depth,
                candidate,
                atlas,
                settings,
                candidate_typed=candidate_typed,
                included_columns=_leave_one_block_columns(evidence.comparison_width_px, settings.stability_folds, fold),
                include_skyline=False,
            )[1]
            for fold in range(settings.stability_folds)
        )
        drafts.append(
            PhotoGeometryVerifiedCandidate(
                candidate_id=str(candidate["candidate_id"]),
                original_estimator_rank=int(candidate["estimator_rank"]),
                verifier_rank=0,
                beam_rank=None,
                atlas_candidate=deepcopy(dict(candidate)),
                terms=terms[0],
                fusion_score=_rounded(terms[1]),
                fold_fusion_scores=tuple(_rounded(value) for value in fold_scores),
            )
        )

    drafts.sort(key=lambda item: (item.fusion_score, item.original_estimator_rank, item.candidate_id))
    ranked = tuple(replace(candidate, verifier_rank=rank) for rank, candidate in enumerate(drafts, start=1))
    beam_ids = _diverse_beam(ranked, settings)
    beam_rank_by_id = {candidate_id: rank for rank, candidate_id in enumerate(beam_ids, start=1)}
    ranked = tuple(replace(candidate, beam_rank=beam_rank_by_id.get(candidate.candidate_id)) for candidate in ranked)
    component_winners = {
        "fusion": ranked[0].candidate_id,
        "skyline": min(ranked, key=lambda item: _component_key(item, "skyline_loss")).candidate_id,
        "symmetric_outline": min(ranked, key=lambda item: _component_key(item, "symmetric_outline_loss")).candidate_id,
        "ordinal_depth": min(ranked, key=lambda item: _component_key(item, "ordinal_depth_loss")).candidate_id,
        "terrain_overlap": min(ranked, key=lambda item: _component_key(item, "terrain_overlap_loss")).candidate_id,
    }
    fold_winners = tuple(
        min(
            ranked, key=lambda item: (item.fold_fusion_scores[fold], item.original_estimator_rank, item.candidate_id)
        ).candidate_id
        for fold in range(settings.stability_folds)
    )
    prior_position_competitor = _prior_position_competitor(ranked, atlas)
    decision = _decision(ranked, fold_winners, evidence, settings)
    basis = _archive_basis(
        settings,
        str(atlas["archive_sha256"]),
        evidence,
        fov,
        terrain_horizontal_extent,
        native_patch is not None,
        surface,
        ranked,
        beam_ids,
        component_winners,
        fold_winners,
        prior_position_competitor.candidate_id,
        decision,
    )
    return PhotoGeometryVerifierArchive(
        config=settings,
        source_atlas_sha256=str(atlas["archive_sha256"]),
        evidence=evidence,
        horizontal_fov_deg=fov,
        terrain_horizontal_extent_m=_rounded(terrain_horizontal_extent),
        native_patch_supplied=native_patch is not None,
        terrain_surface=surface,
        candidates=ranked,
        beam_candidate_ids=beam_ids,
        component_winners=component_winners,
        fold_winner_candidate_ids=fold_winners,
        prior_position_competitor_id=prior_position_competitor.candidate_id,
        decision=decision,
        archive_sha256=_canonical_sha256(basis),
    )


def validate_frozen_photo_geometry_verifier(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and detach a serialized truth-blind verifier archive."""

    archive = deepcopy(dict(record))
    _require_record_keys(
        archive,
        {
            "schema",
            "experimental",
            "production_eligible",
            "photo_observable_evidence_only",
            "source_depth_reference_used",
            "numeric_evaluation_reference_used",
            "config",
            "source_atlas_sha256",
            "query_geometry",
            "native_patch_supplied",
            "terrain_surface",
            "evidence",
            "candidate_pool",
            "candidate_count",
            "ranked_winner_candidate_id",
            "component_winners",
            "fold_winner_candidate_ids",
            "prior_position_competitor_id",
            "beam_policy",
            "beam_candidate_ids",
            "decision",
            "archive_sha256",
            "candidates",
        },
        "photo verifier archive",
    )
    if archive.get("schema") != PHOTO_VERIFIER_ARCHIVE_SCHEMA:
        raise ValueError(f"expected {PHOTO_VERIFIER_ARCHIVE_SCHEMA} photo verifier archive")
    expected_flags = {
        "experimental": True,
        "production_eligible": False,
        "photo_observable_evidence_only": True,
        "source_depth_reference_used": False,
        "numeric_evaluation_reference_used": False,
    }
    if any(archive.get(name) is not value for name, value in expected_flags.items()):
        raise ValueError("photo verifier archive truth-boundary flags are invalid")
    expected_sha = _record_sha256(archive.get("archive_sha256"), "photo verifier archive")
    hash_basis = dict(archive)
    hash_basis.pop("archive_sha256")
    if _canonical_sha256(hash_basis) != expected_sha:
        raise ValueError("photo verifier archive SHA-256 does not match its contents")

    source_atlas_sha = _record_sha256(archive.get("source_atlas_sha256"), "source atlas")
    evidence = _record_mapping(archive.get("evidence"), "photo verifier evidence")
    _require_record_keys(
        evidence,
        {
            "source",
            "reference_depth_used",
            "numeric_reference_pose_used",
            "shape",
            "comparison_shape",
            "extraction_config_sha256",
            "source_atlas_sha256",
            "observation",
            "observed_skyline_sha256",
            "photo_rgb_sha256",
            "edge_response_sha256",
            "relative_depth_sha256",
            "comparison_relative_depth_sha256",
            "valid_mask_sha256",
            "terrain_mask_sha256",
            "ridge_weights_sha256",
            "models",
            "quality",
        },
        "photo verifier evidence",
    )
    if (
        evidence.get("source") != "photo_rgb_only"
        or evidence.get("reference_depth_used") is not False
        or evidence.get("numeric_reference_pose_used") is not False
    ):
        raise ValueError("photo verifier evidence is not photo-only")
    if evidence.get("source_atlas_sha256") != source_atlas_sha:
        raise ValueError("photo verifier evidence names a different source atlas")
    observation = _record_mapping(evidence.get("observation"), "photo observation provenance")
    if _validated_observation_provenance(observation) != observation:
        raise ValueError("photo observation provenance is not canonical")
    if observation.get("source_atlas_sha256") != source_atlas_sha:
        raise ValueError("photo observation provenance names a different source atlas")
    models = _record_mapping(evidence.get("models"), "photo evidence models")
    _require_record_keys(models, {"edge", "depth"}, "photo evidence models")
    for kind in ("edge", "depth"):
        _validate_frozen_photo_model(models.get(kind), kind)

    if archive.get("candidate_pool") != "complete_frozen_photo_atlas_pool":
        raise ValueError("photo verifier candidate-pool policy is unsupported")
    candidates = archive.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("photo verifier candidates must be a non-empty list")
    if archive.get("candidate_count") != len(candidates):
        raise ValueError("photo verifier candidate count does not match its pool")
    ids: list[str] = []
    original_ranks: list[int] = []
    candidate_records: list[dict[str, Any]] = []
    for verifier_rank, value in enumerate(candidates, start=1):
        candidate = _validate_frozen_photo_candidate(value)
        if candidate.get("schema") != PHOTO_VERIFIER_CANDIDATE_SCHEMA:
            raise ValueError("photo verifier candidate schema is unsupported")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("photo verifier candidate ID must be non-empty")
        original_rank = _positive_int(candidate.get("original_estimator_rank"), "original estimator rank")
        if candidate.get("verifier_rank") != verifier_rank:
            raise ValueError("photo verifier candidates must have contiguous verifier ranks in order")
        atlas_candidate = _record_mapping(candidate.get("atlas_candidate"), "nested atlas candidate")
        if atlas_candidate.get("schema") != ATLAS_CANDIDATE_SCHEMA:
            raise ValueError("nested atlas candidate schema is unsupported")
        if (
            atlas_candidate.get("candidate_id") != candidate_id
            or atlas_candidate.get("estimator_rank") != original_rank
        ):
            raise ValueError("photo verifier candidate does not preserve its original atlas ID and rank")
        ids.append(candidate_id)
        original_ranks.append(original_rank)
        candidate_records.append(candidate)
    if len(set(ids)) != len(ids):
        raise ValueError("photo verifier candidate IDs must be unique")
    if sorted(original_ranks) != list(range(1, len(candidates) + 1)):
        raise ValueError("photo verifier candidates must preserve the complete contiguous atlas ranks")
    known_ids = set(ids)
    winner_id = ids[0]
    if archive.get("ranked_winner_candidate_id") != winner_id:
        raise ValueError("photo verifier ranked winner does not match verifier rank one")

    config = _record_mapping(archive.get("config"), "photo verifier config")
    beam_config = _record_mapping(config.get("beam"), "photo verifier beam config")
    beam_size = _positive_int(beam_config.get("size"), "photo verifier beam size")
    beam_ids = archive.get("beam_candidate_ids")
    if (
        archive.get("beam_policy") != "greedy_position_yaw_basin_nms"
        or not isinstance(beam_ids, list)
        or not beam_ids
        or len(beam_ids) > beam_size
        or len(set(beam_ids)) != len(beam_ids)
        or any(candidate_id not in known_ids for candidate_id in beam_ids)
        or beam_ids[0] != winner_id
    ):
        raise ValueError("photo verifier beam IDs, order, or size are invalid")
    beam_rank_by_id = {candidate_id: rank for rank, candidate_id in enumerate(beam_ids, start=1)}
    if any(
        candidate.get("beam_rank") != beam_rank_by_id.get(candidate_id)
        for candidate_id, candidate in zip(ids, candidate_records, strict=True)
    ):
        raise ValueError("photo verifier candidate beam ranks are inconsistent")

    component_winners = _record_mapping(archive.get("component_winners"), "component winners")
    _require_record_keys(
        component_winners,
        {"fusion", "skyline", "symmetric_outline", "ordinal_depth", "terrain_overlap"},
        "component winners",
    )
    if component_winners.get("fusion") != winner_id or any(
        candidate_id not in known_ids for candidate_id in component_winners.values()
    ):
        raise ValueError("photo verifier component-winner references are invalid")
    decision_config = _record_mapping(config.get("decision"), "photo verifier decision config")
    stability_folds = _positive_int(decision_config.get("stability_folds"), "stability folds")
    if any(len(candidate["fold_fusion_scores"]) != stability_folds for candidate in candidate_records):
        raise ValueError("photo verifier candidate fold-score count is inconsistent")
    fold_winners = archive.get("fold_winner_candidate_ids")
    if (
        not isinstance(fold_winners, list)
        or len(fold_winners) != stability_folds
        or any(candidate_id not in known_ids for candidate_id in fold_winners)
    ):
        raise ValueError("photo verifier fold-winner references are invalid")
    if archive.get("prior_position_competitor_id") not in known_ids:
        raise ValueError("photo verifier prior-position competitor reference is invalid")
    _validate_frozen_photo_decision(archive.get("decision"), winner_id, known_ids, stability_folds)
    return archive


def evaluate_photo_geometry_verifier(
    archive: PhotoGeometryVerifierArchive,
    truth: CameraExtrinsics,
    *,
    top_ks: tuple[int, ...] = (1, 5, 10, 16, 32, 64, 128),
) -> PhotoGeometryVerifierEvaluation:
    """Evaluate a frozen verifier archive without changing its score or beam."""

    _validate_archive_hash(archive)
    requested = _validated_top_ks(top_ks)
    evaluated = tuple(_evaluate_candidate(candidate, truth) for candidate in archive.candidates)
    by_id = {candidate.candidate_id: candidate for candidate in evaluated}
    component = {name: by_id[candidate_id] for name, candidate_id in archive.component_winners.items()}
    oracle = min(evaluated, key=_oracle_key)
    target_ranked = [candidate for candidate in evaluated if candidate.reaches_target]
    target_beam = [candidate for candidate in target_ranked if candidate.beam_rank is not None]
    target_beam_ranks = [candidate.beam_rank for candidate in target_beam if candidate.beam_rank is not None]
    records: list[dict[str, Any]] = []
    for requested_k in requested:
        actual_k = min(requested_k, len(evaluated))
        prefix = evaluated[:actual_k]
        successful = [candidate for candidate in prefix if candidate.reaches_target]
        best = min(prefix, key=_oracle_key)
        records.append(
            {
                "requested_k": requested_k,
                "actual_k": actual_k,
                "reaches_target": bool(successful),
                "first_target_rank": min((candidate.verifier_rank for candidate in successful), default=None),
                "best_candidate": best.to_record(),
            }
        )
    returned = (
        by_id[archive.decision.returned_candidate_id] if archive.decision.returned_candidate_id is not None else None
    )
    return PhotoGeometryVerifierEvaluation(
        archive_sha256=archive.archive_sha256,
        ranked_winner_errors=evaluated[0],
        returned_candidate_errors=returned,
        component_winner_errors=component,
        candidate_pool_gt_oracle=oracle,
        first_target_verifier_rank=min((candidate.verifier_rank for candidate in target_ranked), default=None),
        first_target_beam_rank=min(target_beam_ranks, default=None),
        top_k=tuple(records),
    )


def _score_candidate(
    evidence: PhotoGeometryEvidence,
    candidate_depth: NDArray[np.float64],
    candidate: Mapping[str, Any],
    atlas: Mapping[str, Any],
    config: PhotoGeometryVerifierConfig,
    *,
    candidate_typed: TypedOutlines | None = None,
    included_columns: NDArray[np.bool_] | None = None,
    include_skyline: bool = True,
) -> tuple[PhotoGeometryScoreTerms, float]:
    depth = np.asarray(candidate_depth, dtype=np.float64)
    expected = (evidence.comparison_height_px, evidence.comparison_width_px)
    if depth.shape != expected:
        raise ValueError("candidate depth differs from the photo comparison grid")
    if included_columns is None:
        columns = np.ones(expected[1], dtype=bool)
    else:
        columns = np.asarray(included_columns, dtype=bool)
        if columns.shape != (expected[1],):
            raise ValueError("included column mask has the wrong width")
    column_mask = np.broadcast_to(columns[None, :], expected)
    candidate_valid_full = np.isfinite(depth) & (depth >= config.min_depth_m) & evidence.valid_mask
    candidate_valid = candidate_valid_full & column_mask
    typed = candidate_typed or extract_typed_outlines(
        np.where(candidate_valid_full, depth, np.nan),
        min_px=config.typed_min_component_px,
    )
    family_masks = _typed_masks(typed)
    candidate_outline = family_masks["union"] & column_mask
    photo_weights = np.where(column_mask, evidence.ridge_weights, 0.0)
    photo_outline = photo_weights > 0.0
    cap = config.outline_cap_comparison_px
    photo_to_dem, dem_to_photo = _symmetric_weighted_outline_terms(
        photo_weights,
        candidate_outline,
        cap,
        config.min_candidate_outline_px,
    )
    outline_loss = 0.5 * (photo_to_dem + dem_to_photo)
    family_losses = {
        name: _candidate_to_photo_loss(mask & column_mask, photo_outline, cap)
        for name, mask in family_masks.items()
        if name != "union"
    }

    common = candidate_valid & evidence.terrain_mask & np.isfinite(evidence.relative_depth)
    common_count = int(common.sum())
    ordinal_loss, ordinal_cells = _ordinal_depth_loss(
        evidence.relative_depth,
        depth,
        common,
        config,
    )
    photo_terrain = evidence.terrain_mask & evidence.valid_mask & column_mask
    overlap_count = int((photo_terrain & candidate_valid).sum())
    photo_count = int(photo_terrain.sum())
    candidate_count = int(candidate_valid.sum())
    overlap_f1 = 2.0 * overlap_count / (photo_count + candidate_count) if photo_count + candidate_count else 0.0
    common_fraction = common_count / photo_count if photo_count else 0.0
    skyline_cap = _finite_float(atlas["config"].get("residual_cap_px"), "atlas skyline residual cap")
    skyline_score = _finite_float(candidate.get("estimator_score"), "candidate estimator score")
    terms = PhotoGeometryScoreTerms(
        skyline_loss=_rounded(np.clip(skyline_score / skyline_cap, 0.0, 1.0)),
        symmetric_outline_loss=_rounded(outline_loss),
        photo_to_dem_outline_loss=_rounded(photo_to_dem),
        dem_to_photo_outline_loss=_rounded(dem_to_photo),
        ordinal_depth_loss=_rounded(ordinal_loss),
        terrain_overlap_loss=_rounded(1.0 - overlap_f1),
        terrain_overlap_f1=_rounded(overlap_f1),
        common_depth_px=common_count,
        common_terrain_fraction=_rounded(common_fraction),
        photo_terrain_px=photo_count,
        candidate_terrain_px=candidate_count,
        ordinal_cells=ordinal_cells,
        candidate_outline_px=int(candidate_outline.sum()),
        family_candidate_px={name: int(family_masks[name].sum()) for name in ("occlusion", "rib", "couloir")},
        family_dem_to_photo_loss={name: _rounded(family_losses[name]) for name in ("occlusion", "rib", "couloir")},
    )
    return terms, _fusion_score(terms, config, include_skyline=include_skyline)


def _candidate_typed_outlines(
    evidence: PhotoGeometryEvidence,
    candidate_depth: NDArray[np.float64],
    config: PhotoGeometryVerifierConfig,
) -> TypedOutlines:
    depth = np.asarray(candidate_depth, dtype=np.float64)
    valid = np.isfinite(depth) & (depth >= config.min_depth_m) & evidence.valid_mask
    return extract_typed_outlines(
        np.where(valid, depth, np.nan),
        min_px=config.typed_min_component_px,
    )


def _symmetric_weighted_outline_terms(
    photo_weights: NDArray[np.float64],
    candidate_mask: NDArray[np.bool_],
    cap_px: float,
    minimum_candidate_px: int,
) -> tuple[float, float]:
    photo_mask = photo_weights > 0.0
    if not photo_mask.any() or int(candidate_mask.sum()) < minimum_candidate_px:
        return 1.0, 1.0
    candidate_distance = distance_transform_edt(~candidate_mask)
    photo_distance = distance_transform_edt(~photo_mask)
    weights = photo_weights[photo_mask]
    photo_to_dem = float(np.average(np.minimum(candidate_distance[photo_mask] / cap_px, 1.0), weights=weights))
    dem_to_photo = float(np.minimum(photo_distance[candidate_mask] / cap_px, 1.0).mean())
    return photo_to_dem, dem_to_photo


def _candidate_to_photo_loss(
    candidate_mask: NDArray[np.bool_],
    photo_mask: NDArray[np.bool_],
    cap_px: float,
) -> float:
    if not candidate_mask.any() or not photo_mask.any():
        return 1.0
    photo_distance = distance_transform_edt(~photo_mask)
    return float(np.minimum(photo_distance[candidate_mask] / cap_px, 1.0).mean())


def _ordinal_depth_loss(
    photo_depth: NDArray[np.float64],
    candidate_depth: NDArray[np.float64],
    common: NDArray[np.bool_],
    config: PhotoGeometryVerifierConfig,
) -> tuple[float, int]:
    if int(common.sum()) < config.min_common_depth_px:
        return 1.0, 0
    height, width = common.shape
    row_edges = np.linspace(0, height, config.ordinal_grid_rows + 1, dtype=int)
    column_edges = np.linspace(0, width, config.ordinal_grid_columns + 1, dtype=int)
    photo_cells: list[float] = []
    candidate_cells: list[float] = []
    for row in range(config.ordinal_grid_rows):
        for column in range(config.ordinal_grid_columns):
            region = np.s_[row_edges[row] : row_edges[row + 1], column_edges[column] : column_edges[column + 1]]
            mask = common[region]
            if int(mask.sum()) < config.ordinal_min_samples_per_cell:
                continue
            photo_cells.append(float(np.median(photo_depth[region][mask])))
            candidate_cells.append(float(np.median(np.log(candidate_depth[region][mask]))))
    count = len(photo_cells)
    if count < config.ordinal_min_cells:
        return 1.0, count
    photo_ranks = rankdata(np.asarray(photo_cells), method="average")
    candidate_ranks = rankdata(np.asarray(candidate_cells), method="average")
    if float(np.std(photo_ranks)) <= 1e-12 or float(np.std(candidate_ranks)) <= 1e-12:
        return 1.0, count
    correlation = float(np.corrcoef(photo_ranks, candidate_ranks)[0, 1])
    if not math.isfinite(correlation):
        return 1.0, count
    return float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0)), count


def _fusion_score(
    terms: PhotoGeometryScoreTerms,
    config: PhotoGeometryVerifierConfig,
    *,
    include_skyline: bool = True,
) -> float:
    photo_geometry = (
        config.outline_weight * terms.symmetric_outline_loss
        + config.ordinal_depth_weight * terms.ordinal_depth_loss
        + config.terrain_overlap_weight * terms.terrain_overlap_loss
    )
    if include_skyline:
        return config.skyline_weight * terms.skyline_loss + photo_geometry
    return photo_geometry / (1.0 - config.skyline_weight)


def _diverse_beam(
    ranked: tuple[PhotoGeometryVerifiedCandidate, ...],
    config: PhotoGeometryVerifierConfig,
) -> tuple[str, ...]:
    selected: list[PhotoGeometryVerifiedCandidate] = []
    for candidate in ranked:
        if any(_same_basin(candidate, existing, config.nms_position_m, config.nms_yaw_deg) for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) >= config.beam_size:
            break
    return tuple(candidate.candidate_id for candidate in selected)


def _decision(
    ranked: tuple[PhotoGeometryVerifiedCandidate, ...],
    fold_winner_ids: tuple[str, ...],
    evidence: PhotoGeometryEvidence,
    config: PhotoGeometryVerifierConfig,
) -> PhotoGeometryVerifierDecision:
    winner = ranked[0]
    rival = next(
        (
            candidate
            for candidate in ranked[1:]
            if not _same_basin(candidate, winner, config.rival_position_m, config.rival_yaw_deg)
        ),
        None,
    )
    by_id = {candidate.candidate_id: candidate for candidate in ranked}
    stable = sum(
        _same_basin(winner, by_id[candidate_id], config.nms_position_m, config.nms_yaw_deg)
        for candidate_id in fold_winner_ids
    )
    margin = rival.fusion_score - winner.fusion_score if rival is not None else None
    reasons = list(evidence.rejection_reasons)
    if winner.terms.terrain_overlap_f1 < config.minimum_winner_terrain_overlap_f1:
        reasons.append("insufficient_winner_terrain_overlap")
    if winner.terms.common_terrain_fraction < config.minimum_winner_common_terrain_fraction:
        reasons.append("insufficient_winner_common_terrain_support")
    if winner.terms.common_depth_px < config.min_common_depth_px:
        reasons.append("insufficient_winner_common_depth_samples")
    if winner.terms.ordinal_cells < config.ordinal_min_cells:
        reasons.append("insufficient_winner_ordinal_cell_coverage")
    if winner.fusion_score > config.maximum_selected_score:
        reasons.append("winner_score_above_gate")
    if rival is None:
        reasons.append("no_distinct_rival")
    elif margin is None or margin < config.minimum_rival_margin:
        reasons.append("insufficient_distinct_rival_margin")
    if stable < config.minimum_stable_folds:
        reasons.append("leave_one_block_out_instability")
    if rival is not None and winner.terms.symmetric_outline_loss >= rival.terms.symmetric_outline_loss:
        reasons.append("outline_cue_does_not_prefer_winner")
    if rival is not None and winner.terms.ordinal_depth_loss >= rival.terms.ordinal_depth_loss:
        reasons.append("ordinal_depth_cue_does_not_prefer_winner")
    return PhotoGeometryVerifierDecision(
        status="abstained" if reasons else "selected",
        returned_candidate_id=None if reasons else winner.candidate_id,
        ranked_winner_candidate_id=winner.candidate_id,
        rival_candidate_id=rival.candidate_id if rival is not None else None,
        rival_margin=_optional_rounded(margin),
        stable_folds=stable,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _prior_position_competitor(
    ranked: tuple[PhotoGeometryVerifiedCandidate, ...],
    atlas: Mapping[str, Any],
) -> PhotoGeometryVerifiedCandidate:
    prior = atlas.get("prior_position")
    if not isinstance(prior, Mapping):
        raise ValueError("atlas prior position is missing")
    east = _finite_float(prior.get("east_m"), "prior east")
    north = _finite_float(prior.get("north_m"), "prior north")
    nearest_distance = min(_horizontal_distance_to(candidate, east, north) for candidate in ranked)
    nearest = [
        candidate
        for candidate in ranked
        if math.isclose(_horizontal_distance_to(candidate, east, north), nearest_distance, abs_tol=1e-8)
    ]
    return min(nearest, key=lambda item: (item.fusion_score, item.original_estimator_rank, item.candidate_id))


def _leave_one_block_columns(width: int, folds: int, fold: int) -> NDArray[np.bool_]:
    if not 0 <= fold < folds:
        raise ValueError("fold index is out of range")
    edges = np.linspace(0, width, folds + 1, dtype=int)
    included = np.ones(width, dtype=bool)
    included[edges[fold] : edges[fold + 1]] = False
    return included


def _same_basin(
    first: PhotoGeometryVerifiedCandidate,
    second: PhotoGeometryVerifiedCandidate,
    position_m: float,
    yaw_deg: float,
) -> bool:
    return _candidate_distance(first, second) < position_m and _candidate_yaw_distance(first, second) < yaw_deg


def _candidate_distance(first: PhotoGeometryVerifiedCandidate, second: PhotoGeometryVerifiedCandidate) -> float:
    first_position = first.atlas_candidate["pose"]["position"]
    second_position = second.atlas_candidate["pose"]["position"]
    return math.hypot(
        float(first_position["east_m"]) - float(second_position["east_m"]),
        float(first_position["north_m"]) - float(second_position["north_m"]),
    )


def _candidate_yaw_distance(first: PhotoGeometryVerifiedCandidate, second: PhotoGeometryVerifiedCandidate) -> float:
    first_yaw = float(first.atlas_candidate["pose"]["yaw_deg"])
    second_yaw = float(second.atlas_candidate["pose"]["yaw_deg"])
    return abs((first_yaw - second_yaw + 180.0) % 360.0 - 180.0)


def _horizontal_distance_to(candidate: PhotoGeometryVerifiedCandidate, east: float, north: float) -> float:
    position = candidate.atlas_candidate["pose"]["position"]
    return math.hypot(float(position["east_m"]) - east, float(position["north_m"]) - north)


def _component_key(candidate: PhotoGeometryVerifiedCandidate, field: str) -> tuple[float, int, str]:
    return float(getattr(candidate.terms, field)), candidate.original_estimator_rank, candidate.candidate_id


def _archive_basis(
    config: PhotoGeometryVerifierConfig,
    atlas_sha: str,
    evidence: PhotoGeometryEvidence,
    fov: float,
    terrain_horizontal_extent: float,
    native_patch_supplied: bool,
    terrain_surface: dict[str, Any],
    candidates: tuple[PhotoGeometryVerifiedCandidate, ...],
    beam_ids: tuple[str, ...],
    component_winners: dict[str, str],
    fold_winners: tuple[str, ...],
    prior_position_competitor_id: str,
    decision: PhotoGeometryVerifierDecision,
) -> dict[str, Any]:
    return {
        "schema": PHOTO_VERIFIER_ARCHIVE_SCHEMA,
        "experimental": True,
        "production_eligible": False,
        "photo_observable_evidence_only": True,
        "source_depth_reference_used": False,
        "numeric_evaluation_reference_used": False,
        "config": config.to_record(),
        "source_atlas_sha256": atlas_sha,
        "query_geometry": {
            "projection": "cyltan",
            "width_px": evidence.image_width_px,
            "height_px": evidence.image_height_px,
            "horizontal_fov_deg": fov,
            "comparison_width_px": evidence.comparison_width_px,
            "comparison_height_px": evidence.comparison_height_px,
            "terrain_horizontal_extent_m": _rounded(terrain_horizontal_extent),
        },
        "native_patch_supplied": native_patch_supplied,
        "terrain_surface": terrain_surface,
        "evidence": evidence.to_record(),
        "candidate_pool": "complete_frozen_photo_atlas_pool",
        "candidate_count": len(candidates),
        "ranked_winner_candidate_id": candidates[0].candidate_id,
        "component_winners": component_winners,
        "fold_winner_candidate_ids": list(fold_winners),
        "prior_position_competitor_id": prior_position_competitor_id,
        "beam_policy": "greedy_position_yaw_basin_nms",
        "beam_candidate_ids": list(beam_ids),
        "decision": decision.to_record(),
        "candidates": [candidate.to_record() for candidate in candidates],
    }


def _evaluate_candidate(
    candidate: PhotoGeometryVerifiedCandidate,
    truth: CameraExtrinsics,
) -> PhotoGeometryEvaluatedCandidate:
    pose = candidate.atlas_candidate["pose"]
    position = pose["position"]
    east = float(position["east_m"]) - truth.position.east_m
    north = float(position["north_m"]) - truth.position.north_m
    up = float(position["up_m"]) - truth.position.up_m
    horizontal = math.hypot(east, north)
    yaw = abs((float(pose["yaw_deg"]) - truth.yaw_deg + 180.0) % 360.0 - 180.0)
    errors = PhotoGeometryPoseErrors(
        horizontal_position_m=_rounded(horizontal),
        vertical_m=_rounded(abs(up)),
        position_3d_m=_rounded(math.sqrt(horizontal * horizontal + up * up)),
        yaw_deg=_rounded(yaw),
        pitch_deg=None,
    )
    return PhotoGeometryEvaluatedCandidate(
        candidate_id=candidate.candidate_id,
        verifier_rank=candidate.verifier_rank,
        beam_rank=candidate.beam_rank,
        original_estimator_rank=candidate.original_estimator_rank,
        errors=errors,
        normalized_joint_error=_rounded(max(horizontal / POSITION_SUCCESS_M, yaw / YAW_SUCCESS_DEG)),
        reaches_target=horizontal <= POSITION_SUCCESS_M and yaw <= YAW_SUCCESS_DEG,
    )


def _oracle_key(candidate: PhotoGeometryEvaluatedCandidate) -> tuple[float, float, float, float, int, str]:
    return (
        candidate.normalized_joint_error,
        candidate.errors.horizontal_position_m / POSITION_SUCCESS_M + candidate.errors.yaw_deg / YAW_SUCCESS_DEG,
        candidate.errors.horizontal_position_m,
        candidate.errors.yaw_deg,
        candidate.verifier_rank,
        candidate.candidate_id,
    )


def _typed_masks(typed: TypedOutlines) -> dict[str, NDArray[np.bool_]]:
    return {
        "union": typed.occlusion | typed.rib | typed.couloir,
        "occlusion": typed.occlusion,
        "rib": typed.rib,
        "couloir": typed.couloir,
    }


def _internal_ridge_weights(
    ridges: RidgeField,
    skyline: NDArray[np.float64],
    valid: NDArray[np.bool_],
    exclusion_px: int,
) -> NDArray[np.float64]:
    height, width = valid.shape
    result = np.zeros((height, width), dtype=np.float64)
    for ridge in ridges.ridges:
        finite = np.isfinite(ridge.rows) & (ridge.confidence > 0.0)
        columns = np.flatnonzero(finite)
        if not columns.size:
            continue
        rows = np.clip(np.rint(ridge.rows[columns]).astype(int), 0, height - 1)
        keep = valid[rows, columns]
        skyline_rows = skyline[columns]
        keep &= ~np.isfinite(skyline_rows) | (rows > skyline_rows + exclusion_px)
        rows = rows[keep]
        columns = columns[keep]
        np.maximum.at(result, (rows, columns), ridge.confidence[columns])
    return result


def _terrain_mask_from_skyline(
    skyline: NDArray[np.float64],
    valid: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    height, width = valid.shape
    rows = np.arange(height)[:, None]
    finite = np.isfinite(skyline)
    boundary = np.where(finite, skyline, float(height))[None, :]
    return valid & finite[None, :] & (rows >= boundary)


def _max_resample(values: NDArray[np.float64], subsample: int) -> NDArray[np.float64]:
    height, width = values.shape
    out_height = len(range(0, height, subsample))
    out_width = len(range(0, width, subsample))
    result = np.zeros((out_height, out_width), dtype=np.float64)
    for row, source_row in enumerate(range(0, height, subsample)):
        row_stop = min(height, source_row + subsample)
        for column, source_column in enumerate(range(0, width, subsample)):
            column_stop = min(width, source_column + subsample)
            result[row, column] = float(values[source_row:row_stop, source_column:column_stop].max(initial=0.0))
    return result


def _effective_samples(weights: NDArray[np.float64]) -> float:
    positive = weights[weights > 0.0]
    if not positive.size:
        return 0.0
    return float(positive.sum() ** 2 / np.square(positive).sum())


def _horizontal_bin_coverage(mask: NDArray[np.bool_], bins: int) -> int:
    edges = np.linspace(0, mask.shape[1], bins + 1, dtype=int)
    return sum(bool(mask[:, edges[index] : edges[index + 1]].any()) for index in range(bins))


def _validate_frozen_photo_candidate(value: Any) -> dict[str, Any]:
    candidate = _record_mapping(value, "photo verifier candidate")
    _require_record_keys(
        candidate,
        {
            "schema",
            "candidate_id",
            "original_estimator_rank",
            "verifier_rank",
            "beam_rank",
            "atlas_candidate",
            "terms",
            "fusion_score",
            "fold_fusion_scores",
        },
        "photo verifier candidate",
    )
    fold_scores = candidate.get("fold_fusion_scores")
    if not isinstance(fold_scores, list):
        raise ValueError("photo verifier candidate fold scores must be a list")

    terms = _record_mapping(candidate.get("terms"), "photo verifier candidate terms")
    _require_record_keys(
        terms,
        {
            "skyline_loss",
            "symmetric_outline_loss",
            "photo_to_dem_outline_loss",
            "dem_to_photo_outline_loss",
            "ordinal_depth_loss",
            "terrain_overlap_loss",
            "terrain_overlap_f1",
            "common_depth_px",
            "common_terrain_fraction",
            "photo_terrain_px",
            "candidate_terrain_px",
            "ordinal_cells",
            "candidate_outline_px",
            "families",
        },
        "photo verifier candidate terms",
    )
    families = _record_mapping(terms.get("families"), "photo verifier candidate families")
    _require_record_keys(families, {"occlusion", "rib", "couloir"}, "photo verifier candidate families")
    for family in ("occlusion", "rib", "couloir"):
        family_record = _record_mapping(families.get(family), f"photo verifier {family} family")
        _require_record_keys(
            family_record,
            {"candidate_px", "dem_to_photo_loss"},
            f"photo verifier {family} family",
        )

    atlas_candidate = _record_mapping(candidate.get("atlas_candidate"), "nested atlas candidate")
    _require_record_keys(
        atlas_candidate,
        {
            "schema",
            "candidate_id",
            "estimator_rank",
            "grid",
            "pose",
            "vertical_shift_px",
            "vertical_slope_px_per_column",
            "estimator_score",
            "ground_source",
        },
        "nested atlas candidate",
    )
    grid = _record_mapping(atlas_candidate.get("grid"), "nested atlas candidate grid")
    _require_record_keys(grid, {"row", "column", "yaw_index"}, "nested atlas candidate grid")
    pose = _record_mapping(atlas_candidate.get("pose"), "nested atlas candidate pose")
    _require_record_keys(pose, {"position", "yaw_deg", "pitch_deg", "roll_deg"}, "nested atlas candidate pose")
    position = _record_mapping(pose.get("position"), "nested atlas candidate position")
    _require_record_keys(position, {"east_m", "north_m", "up_m"}, "nested atlas candidate position")
    return candidate


def _validate_frozen_photo_model(value: Any, kind: str) -> None:
    model = _record_mapping(value, f"photo {kind} model provenance")
    name = model.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"photo {kind} model name is missing")
    _validated_model_provenance(model, name, kind)
    _record_sha256(model.get("aggregate_sha256"), f"photo {kind} model")
    base = {"name", "aggregate_sha256", "offline", "input"}
    keys = set(model)
    if keys == base:
        return
    if keys == base | {"synthetic_authoritative_channel"}:
        if model.get("synthetic_authoritative_channel") is not True:
            raise ValueError(f"photo {kind} synthetic model audit flag must be true")
        return
    if keys != base | {"source", "files"}:
        raise ValueError(f"photo {kind} model provenance fields are unsupported")
    if not isinstance(model.get("source"), Mapping):
        raise ValueError(f"photo {kind} model source is missing")
    files = model.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError(f"photo {kind} model file inventory is missing")


def _validate_frozen_photo_decision(
    value: Any,
    winner_id: str,
    known_ids: set[str],
    stability_folds: int,
) -> None:
    decision = _record_mapping(value, "photo verifier decision")
    _require_record_keys(
        decision,
        {
            "status",
            "returned_candidate_id",
            "ranked_winner_candidate_id",
            "rival_candidate_id",
            "rival_margin",
            "stable_folds",
            "reasons",
            "fallback",
            "calibrated",
        },
        "photo verifier decision",
    )
    if decision.get("ranked_winner_candidate_id") != winner_id or decision.get("calibrated") is not False:
        raise ValueError("photo verifier decision winner or calibration flag is invalid")
    rival_id = decision.get("rival_candidate_id")
    if rival_id is not None and (rival_id not in known_ids or rival_id == winner_id):
        raise ValueError("photo verifier decision rival reference is invalid")
    rival_margin = decision.get("rival_margin")
    if rival_margin is not None:
        _finite_float(rival_margin, "photo verifier rival margin")
    stable_folds = decision.get("stable_folds")
    if isinstance(stable_folds, bool) or not isinstance(stable_folds, int) or not 0 <= stable_folds <= stability_folds:
        raise ValueError("photo verifier decision stable-fold count is invalid")
    reasons = decision.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
        raise ValueError("photo verifier decision reasons are invalid")
    status = decision.get("status")
    if status == "selected":
        if reasons or decision.get("returned_candidate_id") != winner_id or decision.get("fallback") is not None:
            raise ValueError("selected photo verifier decision is inconsistent")
    elif status == "abstained":
        if (
            not reasons
            or decision.get("returned_candidate_id") is not None
            or decision.get("fallback") != "retain_supplied_prior_outside_this_archive"
        ):
            raise ValueError("abstained photo verifier decision is inconsistent")
    else:
        raise ValueError("photo verifier decision status is unsupported")


def _record_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _require_record_keys(record: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(record) != expected:
        raise ValueError(f"{name} fields differ from {PHOTO_VERIFIER_ARCHIVE_SCHEMA}")


def _record_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} SHA-256 is missing or malformed")
    return value


def _validated_model_provenance(record: Mapping[str, Any], expected_name: str, kind: str) -> dict[str, Any]:
    result = deepcopy(dict(record))
    if result.get("name") != expected_name:
        raise ValueError(f"{kind} model provenance does not match the loaded model")
    aggregate = result.get("aggregate_sha256")
    if not isinstance(aggregate, str) or len(aggregate) != 64:
        raise ValueError(f"{kind} model provenance requires an aggregate SHA-256")
    if result.get("offline") is not True or result.get("input") != "photo_rgb":
        raise ValueError(f"{kind} model provenance must declare offline photo-RGB inference")
    return result


def _validated_observation_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(record))
    base_keys = {
        "track",
        "source",
        "candidate",
        "selection_uses_reference_truth",
        "evidence_generated_at_reference_pose",
        "source_atlas_sha256",
    }
    synthetic_keys = {
        "synthetic_rgb_encoded_from_same_candidate_render",
        "interpretation",
    }
    unexpected = set(result) - base_keys - synthetic_keys
    if unexpected:
        raise ValueError(f"photo observation provenance contains unsupported fields: {sorted(unexpected)}")
    if result.get("track") != "photo_auto":
        raise ValueError("photo verifier requires an explicitly labelled photo_auto observation")
    if result.get("selection_uses_reference_truth") is not False:
        raise ValueError("photo observation selection must be truth-blind")
    if result.get("evidence_generated_at_reference_pose") is not False:
        raise ValueError("photo observation must not be generated at the reference pose")
    if not isinstance(result.get("source"), str) or not result["source"]:
        raise ValueError("photo observation extractor source is missing")
    if not isinstance(result.get("candidate"), str) or not result["candidate"]:
        raise ValueError("photo observation candidate name is missing")
    atlas_sha = result.get("source_atlas_sha256")
    if not isinstance(atlas_sha, str) or len(atlas_sha) != 64:
        raise ValueError("photo observation source-atlas SHA-256 is missing")
    if synthetic_keys & set(result):
        if set(result) & synthetic_keys != synthetic_keys:
            raise ValueError("synthetic photo observation audit fields must be supplied together")
        if not isinstance(result["synthetic_rgb_encoded_from_same_candidate_render"], bool):
            raise ValueError("synthetic shared-render audit flag must be boolean")
        if result["interpretation"] != "plumbing_identity_ceiling_not_photo_model_generalization":
            raise ValueError("synthetic photo observation interpretation is unsupported")
    return result


def _validate_atlas_coordinate_frame(
    atlas: Mapping[str, Any],
    terrain_surface: Mapping[str, Any],
) -> None:
    frame = atlas.get("coordinate_frame")
    origin = frame.get("origin") if isinstance(frame, Mapping) else None
    if not isinstance(origin, Mapping):
        raise ValueError("frozen atlas coordinate-frame origin is missing")
    if _canonical_sha256(dict(origin)) != _canonical_sha256(terrain_surface.get("coordinate_frame_origin")):
        raise ValueError("estimator terrain coordinate frame differs from the frozen atlas")


def _validate_archive_hash(archive: PhotoGeometryVerifierArchive) -> None:
    record = archive.to_record()
    expected = record.pop("archive_sha256", None)
    if expected != archive.archive_sha256 or _canonical_sha256(record) != archive.archive_sha256:
        raise ValueError("photo verifier archive SHA-256 does not match its contents")


def _validated_photo_map(values: Any, shape: tuple[int, int], name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and match the photo shape")
    if float(result.min()) < -1e-6 or float(result.max()) > 1.0 + 1e-6:
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(result, 0.0, 1.0)


def _validate_evidence_arrays(evidence: PhotoGeometryEvidence) -> None:
    expected = (evidence.comparison_height_px, evidence.comparison_width_px)
    arrays = (
        (evidence.valid_mask, expected, evidence.valid_mask_sha256, "peakle_photo_valid_mask_v1", "valid mask"),
        (
            evidence.terrain_mask,
            expected,
            evidence.terrain_mask_sha256,
            "peakle_photo_terrain_mask_v1",
            "terrain mask",
        ),
        (
            evidence.ridge_weights,
            expected,
            evidence.ridge_weights_sha256,
            "peakle_photo_ridge_weights_v1",
            "ridge weights",
        ),
        (
            evidence.relative_depth,
            expected,
            evidence.comparison_relative_depth_sha256,
            "peakle_photo_comparison_relative_depth_v1",
            "relative depth",
        ),
    )
    for values, shape, expected_sha, domain, name in arrays:
        if np.asarray(values).shape != shape:
            raise ValueError(f"photo evidence {name} has the wrong comparison shape")
        if _array_sha256(values, domain) != expected_sha:
            raise ValueError(f"photo evidence {name} differs from its frozen hash")


def _validate_evidence_quality(
    evidence: PhotoGeometryEvidence,
    config: PhotoGeometryVerifierConfig,
) -> None:
    _validated_model_provenance(evidence.edge_model, str(evidence.edge_model.get("name")), "edge")
    _validated_model_provenance(evidence.depth_model, str(evidence.depth_model.get("name")), "depth")
    ridge_effective = _rounded(_effective_samples(evidence.ridge_weights))
    ridge_bins = _horizontal_bin_coverage(evidence.ridge_weights > 0.0, config.ridge_horizontal_bins)
    depth_values = evidence.relative_depth[np.isfinite(evidence.relative_depth)]
    depth_span = (
        _rounded(float(np.percentile(depth_values, 90.0) - np.percentile(depth_values, 10.0)))
        if depth_values.size
        else 0.0
    )
    if ridge_effective != evidence.ridge_effective_samples:
        raise ValueError("photo evidence ridge effective-sample metric is inconsistent")
    if ridge_bins != evidence.ridge_horizontal_bins:
        raise ValueError("photo evidence ridge-bin metric is inconsistent")
    if depth_span != evidence.relative_depth_span_p90_p10:
        raise ValueError("photo evidence relative-depth span is inconsistent")
    expected_reasons: list[str] = []
    if evidence.skyline_coverage < config.minimum_skyline_coverage:
        expected_reasons.append("insufficient_skyline_coverage")
    if ridge_effective < config.minimum_ridge_effective_samples:
        expected_reasons.append("insufficient_internal_ridge_support")
    if ridge_bins < config.minimum_ridge_horizontal_bins:
        expected_reasons.append("insufficient_internal_ridge_span")
    if depth_span < config.ordinal_min_photo_span:
        expected_reasons.append("insufficient_relative_depth_span")
    if evidence.rejection_reasons != tuple(expected_reasons) or evidence.usable is not (not expected_reasons):
        raise ValueError("photo evidence usability record is inconsistent with its frozen metrics")


def _observed_skyline_sha256(observed: NDArray[np.float64]) -> str:
    finite = np.isfinite(observed)
    normalized = np.where(finite, observed, 0.0).astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"peakle_observed_skyline_v1\0")
    digest.update(int(observed.size).to_bytes(8, "little", signed=False))
    digest.update(np.ascontiguousarray(finite.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(normalized).tobytes())
    return digest.hexdigest()


def _array_sha256(array: Any, domain: str) -> str:
    values = np.asarray(array)
    finite = np.isfinite(values) if np.issubdtype(values.dtype, np.floating) else np.ones(values.shape, dtype=bool)
    normalized = np.where(finite, values, 0)
    digest = hashlib.sha256(domain.encode() + b"\0")
    digest.update(str(values.dtype).encode() + b"\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(finite.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(normalized).tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validated_top_ks(top_ks: tuple[int, ...]) -> tuple[int, ...]:
    if not top_ks or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in top_ks):
        raise ValueError("top_ks must contain positive integers")
    return tuple(dict.fromkeys(top_ks))


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _rounded(value: float) -> float:
    return round(float(value), 12)


def _optional_rounded(value: float | None) -> float | None:
    return None if value is None else _rounded(value)
