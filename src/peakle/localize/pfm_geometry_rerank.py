"""Analysis-only PFM depth and typed-outline reranking for a frozen pose atlas.

The source PFM was rendered at the benchmark reference pose, so this module is
an oracle study, not a production estimator.  Its narrower contract is still
important: ``build_pfm_geometry_rerank`` accepts only the already-frozen atlas,
terrain, and source depth.  Numeric pose truth is accepted exclusively by
``evaluate_pfm_geometry_rerank`` after the rerank archive has been frozen and
hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import distance_transform_edt

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.terrain import TerrainMap
from peakle.localize.atlas_geometry import (
    RenderedCyltanCandidateDepth,
    render_cyltan_candidate_depth,
    terrain_diagonal_range_m,
    validate_frozen_cyltan_atlas,
)
from peakle.localize.gtrefine import dem_depth_image
from peakle.localize.skyline_atlas import POSITION_SUCCESS_M, YAW_SUCCESS_DEG
from peakle.localize.swissdem import Patch
from peakle.localize.typed_outlines import TypedOutlines, extract_typed_outlines

PFM_RERANK_ARCHIVE_SCHEMA = "peakle_pfm_geometry_rerank_archive_v1"
PFM_RERANK_CANDIDATE_SCHEMA = "peakle_pfm_geometry_rerank_candidate_v1"
PFM_RERANK_EVALUATION_SCHEMA = "peakle_pfm_geometry_rerank_evaluation_v1"


@dataclass(frozen=True, slots=True)
class PfmGeometryRerankConfig:
    """Frozen scoring budget and fixed, predeclared fusion weights."""

    subsample: int = 4
    min_depth_m: float = 250.0
    depth_log_cap: float = math.log(2.0)
    outline_cap_comparison_px: float = 10.0
    min_source_family_px: int = 12
    min_common_depth_px: int = 200
    skyline_weight: float = 0.20
    typed_outline_weight: float = 0.35
    depth_shape_weight: float = 0.35
    depth_overlap_weight: float = 0.10
    typed_union_fraction: float = 0.50

    def __post_init__(self) -> None:
        if isinstance(self.subsample, bool) or not isinstance(self.subsample, int) or self.subsample < 1:
            raise ValueError("subsample must be a positive integer")
        if not math.isfinite(self.min_depth_m) or self.min_depth_m <= 0.0:
            raise ValueError("min_depth_m must be positive and finite")
        if not math.isfinite(self.depth_log_cap) or self.depth_log_cap <= 0.0:
            raise ValueError("depth_log_cap must be positive and finite")
        if not math.isfinite(self.outline_cap_comparison_px) or self.outline_cap_comparison_px <= 0.0:
            raise ValueError("outline_cap_comparison_px must be positive and finite")
        if self.min_source_family_px < 1 or self.min_common_depth_px < 1:
            raise ValueError("minimum pixel counts must be positive")
        weights = (
            self.skyline_weight,
            self.typed_outline_weight,
            self.depth_shape_weight,
            self.depth_overlap_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("fusion weights must be finite and non-negative")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fusion weights must sum to one")
        if not 0.0 <= self.typed_union_fraction <= 1.0:
            raise ValueError("typed_union_fraction must be in [0, 1]")

    def to_record(self) -> dict[str, Any]:
        return {
            "subsample": self.subsample,
            "min_depth_m": self.min_depth_m,
            "depth_log_cap": self.depth_log_cap,
            "outline_cap_comparison_px": self.outline_cap_comparison_px,
            "comparison_grid_pixel_scale_in_source_pixels": self.subsample,
            "min_source_family_px": self.min_source_family_px,
            "min_common_depth_px": self.min_common_depth_px,
            "weights": {
                "skyline": self.skyline_weight,
                "typed_outline": self.typed_outline_weight,
                "depth_shape": self.depth_shape_weight,
                "depth_overlap": self.depth_overlap_weight,
            },
            "typed_union_fraction": self.typed_union_fraction,
            "source_depth_convention": "GeoPose_distance_crop_pfm_assumed_camera_ray_range",
            "candidate_depth_convention": "horizontal_ray_march_range_converted_to_camera_ray_range",
            "comparison_range_policy": "terrain_bounding_box_diagonal_common_cap",
            "depth_scale_nuisance": "per_candidate_median_log_ratio",
            "source_resampling": "align_corners_nearest_no_sky_depth_interpolation",
        }


@dataclass(frozen=True, slots=True)
class PfmGeometryScoreTerms:
    skyline_loss: float
    typed_outline_loss: float
    typed_union_loss: float
    typed_family_loss: float | None
    occlusion_loss: float | None
    rib_loss: float | None
    couloir_loss: float | None
    depth_shape_loss: float
    depth_overlap_loss: float
    depth_overlap_f1: float
    common_depth_px: int
    source_near_depth_px: int
    candidate_near_depth_px: int
    candidate_to_source_depth_scale: float | None
    raw_log_depth_mae: float | None
    centered_log_depth_mae: float | None

    def to_record(self) -> dict[str, Any]:
        return {
            "skyline_loss": self.skyline_loss,
            "typed_outline_loss": self.typed_outline_loss,
            "typed_union_loss": self.typed_union_loss,
            "typed_family_loss": self.typed_family_loss,
            "families": {
                "occlusion_loss": self.occlusion_loss,
                "rib_loss": self.rib_loss,
                "couloir_loss": self.couloir_loss,
            },
            "depth_shape_loss": self.depth_shape_loss,
            "depth_overlap_loss": self.depth_overlap_loss,
            "depth_overlap_f1": self.depth_overlap_f1,
            "common_depth_px": self.common_depth_px,
            "source_near_depth_px": self.source_near_depth_px,
            "candidate_near_depth_px": self.candidate_near_depth_px,
            "candidate_to_source_depth_scale": self.candidate_to_source_depth_scale,
            "raw_log_depth_mae": self.raw_log_depth_mae,
            "centered_log_depth_mae": self.centered_log_depth_mae,
        }


@dataclass(frozen=True, slots=True)
class PfmGeometryRerankedCandidate:
    candidate_id: str
    original_estimator_rank: int
    rerank_rank: int
    atlas_candidate: dict[str, Any]
    terms: PfmGeometryScoreTerms
    fusion_score: float

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PFM_RERANK_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "original_estimator_rank": self.original_estimator_rank,
            "rerank_rank": self.rerank_rank,
            "atlas_candidate": self.atlas_candidate,
            "terms": self.terms.to_record(),
            "fusion_score": self.fusion_score,
        }


@dataclass(frozen=True, slots=True)
class PfmGeometryRerankArchive:
    config: PfmGeometryRerankConfig
    source_atlas_sha256: str
    source_depth_sha256: str
    image_width_px: int
    image_height_px: int
    horizontal_fov_deg: float
    comparison_width_px: int
    comparison_height_px: int
    comparison_max_range_m: float
    native_patch_supplied: bool
    source_typed_counts: dict[str, int]
    candidates: tuple[PfmGeometryRerankedCandidate, ...]
    selected: PfmGeometryRerankedCandidate
    component_winners: dict[str, str]
    archive_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PFM_RERANK_ARCHIVE_SCHEMA,
            "analysis_only": True,
            "production_eligible": False,
            "source_depth_reference_used": True,
            "numeric_evaluation_reference_used": False,
            "config": self.config.to_record(),
            "source_atlas_sha256": self.source_atlas_sha256,
            "source_depth_sha256": self.source_depth_sha256,
            "query_geometry": {
                "projection": "cyltan",
                "width_px": self.image_width_px,
                "height_px": self.image_height_px,
                "horizontal_fov_deg": self.horizontal_fov_deg,
                "comparison_width_px": self.comparison_width_px,
                "comparison_height_px": self.comparison_height_px,
                "comparison_max_range_m": self.comparison_max_range_m,
            },
            "native_patch_supplied": self.native_patch_supplied,
            "source_typed_counts": self.source_typed_counts,
            "candidate_pool": "complete_three_yaw_separated_modes_per_position_from_frozen_atlas",
            "candidate_count": len(self.candidates),
            "selected_candidate_id": self.selected.candidate_id,
            "component_winners": self.component_winners,
            "archive_sha256": self.archive_sha256,
            "candidates": [candidate.to_record() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class PfmGeometryPoseErrors:
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
class PfmGeometryEvaluatedCandidate:
    candidate_id: str
    rerank_rank: int
    original_estimator_rank: int
    errors: PfmGeometryPoseErrors
    normalized_joint_error: float
    reaches_target: bool

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "rerank_rank": self.rerank_rank,
            "original_estimator_rank": self.original_estimator_rank,
            "errors": self.errors.to_record(),
            "normalized_joint_error": self.normalized_joint_error,
            "reaches_target": self.reaches_target,
        }


@dataclass(frozen=True, slots=True)
class PfmGeometryRerankEvaluation:
    archive_sha256: str
    winner_errors: PfmGeometryEvaluatedCandidate
    component_winner_errors: dict[str, PfmGeometryEvaluatedCandidate]
    candidate_pool_gt_oracle: PfmGeometryEvaluatedCandidate
    top_k: tuple[dict[str, Any], ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": PFM_RERANK_EVALUATION_SCHEMA,
            "archive_sha256": self.archive_sha256,
            "reference_data_used": True,
            "used_by_estimator": False,
            "target": {
                "horizontal_position_m_lte": POSITION_SUCCESS_M,
                "absolute_yaw_deg_lte": YAW_SUCCESS_DEG,
                "pitch_used_by_oracle": False,
                "roll_used_by_oracle": False,
            },
            "winner_errors": self.winner_errors.to_record(),
            "component_winner_errors": {
                component: candidate.to_record()
                for component, candidate in sorted(self.component_winner_errors.items())
            },
            "candidate_pool_gt_oracle": self.candidate_pool_gt_oracle.to_record(),
            "top_k": list(self.top_k),
        }


# Private compatibility alias retained for existing callers and focused tests.
_RenderedCandidate = RenderedCyltanCandidateDepth


def build_pfm_geometry_rerank(
    terrain: TerrainMap,
    native_patch: Patch | None,
    source_depth: NDArray[np.float64],
    atlas_archive: Mapping[str, Any],
    *,
    config: PfmGeometryRerankConfig | None = None,
) -> PfmGeometryRerankArchive:
    """Rerank every frozen atlas candidate without accepting numeric pose truth."""

    settings = config or PfmGeometryRerankConfig()
    atlas = _validated_atlas_record(atlas_archive)
    candidates = atlas["candidates"]
    query = atlas["query_geometry"]
    width = _positive_int(query.get("width_px"), "query width")
    height = _positive_int(query.get("height_px"), "query height")
    fov = _finite_float(query.get("horizontal_fov_deg"), "horizontal FOV")
    if not 1.0 < fov < 179.0:
        raise ValueError("horizontal FOV must be between 1 and 179 degrees")
    if bool(atlas.get("native_patch_supplied")) != (native_patch is not None):
        raise ValueError("native patch availability differs from the frozen atlas")
    source = np.asarray(source_depth, dtype=np.float64)
    if source.ndim != 2 or not source.size:
        raise ValueError("source_depth must be a non-empty two-dimensional array")

    max_range = _terrain_diagonal_range_m(terrain)
    first_render = _render_candidate(terrain, native_patch, candidates[0], width, height, fov, settings)
    source_target = _resample_source_depth(source, width, height, settings.subsample, max_range, settings.min_depth_m)
    source_typed = extract_typed_outlines(source_target, min_px=max(1, 25 // settings.subsample))
    source_distances = _outline_distance_maps(source_typed)

    drafts: list[PfmGeometryRerankedCandidate] = []
    rendered_items = (first_render,)
    for index, candidate in enumerate(candidates):
        rendered = (
            rendered_items[0]
            if index == 0
            else _render_candidate(terrain, native_patch, candidate, width, height, fov, settings)
        )
        terms = _score_candidate(
            source_target,
            source_typed,
            source_distances,
            rendered.candidate_ray_depth,
            max_range,
            rendered.atlas_candidate,
            atlas,
            settings,
        )
        fusion = _fusion_score(terms, settings)
        drafts.append(
            PfmGeometryRerankedCandidate(
                candidate_id=str(candidate["candidate_id"]),
                original_estimator_rank=int(candidate["estimator_rank"]),
                rerank_rank=0,
                atlas_candidate=dict(candidate),
                terms=terms,
                fusion_score=_rounded(fusion),
            )
        )

    drafts.sort(key=lambda item: (item.fusion_score, item.original_estimator_rank, item.candidate_id))
    ranked = tuple(replace(candidate, rerank_rank=rank) for rank, candidate in enumerate(drafts, start=1))
    component_winners = {
        "fusion": ranked[0].candidate_id,
        "skyline": min(ranked, key=lambda item: _component_key(item, "skyline_loss")).candidate_id,
        "typed_outline": min(ranked, key=lambda item: _component_key(item, "typed_outline_loss")).candidate_id,
        "depth_shape": min(ranked, key=lambda item: _component_key(item, "depth_shape_loss")).candidate_id,
        "depth_overlap": min(ranked, key=lambda item: _component_key(item, "depth_overlap_loss")).candidate_id,
    }
    source_sha = _array_sha256(source)
    basis = _archive_basis(
        settings,
        str(atlas["archive_sha256"]),
        source_sha,
        width,
        height,
        fov,
        source_target.shape[1],
        source_target.shape[0],
        max_range,
        native_patch is not None,
        source_typed.counts(),
        ranked,
        component_winners,
    )
    return PfmGeometryRerankArchive(
        config=settings,
        source_atlas_sha256=str(atlas["archive_sha256"]),
        source_depth_sha256=source_sha,
        image_width_px=width,
        image_height_px=height,
        horizontal_fov_deg=fov,
        comparison_width_px=source_target.shape[1],
        comparison_height_px=source_target.shape[0],
        comparison_max_range_m=_rounded(max_range),
        native_patch_supplied=native_patch is not None,
        source_typed_counts=source_typed.counts(),
        candidates=ranked,
        selected=ranked[0],
        component_winners=component_winners,
        archive_sha256=_canonical_sha256(basis),
    )


def evaluate_pfm_geometry_rerank(
    archive: PfmGeometryRerankArchive,
    truth: CameraExtrinsics,
    *,
    top_ks: tuple[int, ...] = (1, 5, 10, 50, 100),
) -> PfmGeometryRerankEvaluation:
    """Evaluate a previously frozen rerank archive without changing its order."""

    requested = _validated_top_ks(top_ks)
    evaluated = tuple(_evaluate_candidate(candidate, truth) for candidate in archive.candidates)
    evaluated_by_id = {candidate.candidate_id: candidate for candidate in evaluated}
    component_winner_errors = {
        component: evaluated_by_id[candidate_id] for component, candidate_id in archive.component_winners.items()
    }
    oracle = min(evaluated, key=_oracle_key)
    records: list[dict[str, Any]] = []
    for requested_k in requested:
        evaluated_k = min(requested_k, len(evaluated))
        prefix = evaluated[:evaluated_k]
        best = min(prefix, key=_oracle_key)
        reached = any(candidate.reaches_target for candidate in prefix)
        records.append(
            {
                "requested_k": requested_k,
                "evaluated_k": evaluated_k,
                "reaches_target": reached,
                "recall": 1.0 if reached else 0.0,
                "best_candidate": best.to_record(),
            }
        )
    return PfmGeometryRerankEvaluation(
        archive_sha256=archive.archive_sha256,
        winner_errors=evaluated[0],
        component_winner_errors=component_winner_errors,
        candidate_pool_gt_oracle=oracle,
        top_k=tuple(records),
    )


def _validated_atlas_record(atlas_archive: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper for the shared frozen-atlas validator."""

    return validate_frozen_cyltan_atlas(atlas_archive)


def _render_candidate(
    terrain: TerrainMap,
    native_patch: Patch | None,
    candidate: Mapping[str, Any],
    width: int,
    height: int,
    fov: float,
    config: PfmGeometryRerankConfig,
) -> _RenderedCandidate:
    """Compatibility wrapper using this module's injectable depth renderer."""

    return render_cyltan_candidate_depth(
        terrain,
        native_patch,
        candidate,
        width,
        height,
        fov,
        subsample=config.subsample,
        depth_renderer=dem_depth_image,
    )


def _terrain_diagonal_range_m(terrain: TerrainMap) -> float:
    """Compatibility wrapper for the shared stable comparison cap."""

    return terrain_diagonal_range_m(terrain)


def _resample_source_depth(
    source: NDArray[np.float64],
    width: int,
    height: int,
    subsample: int,
    max_range_m: float,
    min_depth_m: float,
) -> NDArray[np.float64]:
    """Nearest-sample PFM values at the exact DEM comparison pixel centres."""

    source_height, source_width = source.shape
    work_rows = np.arange(0, height, subsample, dtype=np.float64)
    work_columns = np.arange(0, width, subsample, dtype=np.float64)
    row_scale = (source_height - 1) / max(height - 1, 1)
    column_scale = (source_width - 1) / max(width - 1, 1)
    source_rows = np.clip(np.rint(work_rows * row_scale).astype(np.int64), 0, source_height - 1)
    source_columns = np.clip(np.rint(work_columns * column_scale).astype(np.int64), 0, source_width - 1)
    target = source[np.ix_(source_rows, source_columns)].astype(np.float64, copy=True)
    valid = np.isfinite(target) & (target >= min_depth_m) & (target <= max_range_m)
    return np.where(valid, target, np.nan)


def _outline_distance_maps(typed: TypedOutlines) -> dict[str, NDArray[np.float64] | None]:
    masks = _typed_masks(typed)
    return {name: distance_transform_edt(~mask) if mask.any() else None for name, mask in masks.items()}


def _typed_masks(typed: TypedOutlines) -> dict[str, NDArray[np.bool_]]:
    return {
        "union": typed.occlusion | typed.rib | typed.couloir,
        "occlusion": typed.occlusion,
        "rib": typed.rib,
        "couloir": typed.couloir,
    }


def _score_candidate(
    source_depth: NDArray[np.float64],
    source_typed: TypedOutlines,
    source_distances: Mapping[str, NDArray[np.float64] | None],
    candidate_depth: NDArray[np.float64],
    max_range_m: float,
    candidate: Mapping[str, Any],
    atlas: Mapping[str, Any],
    config: PfmGeometryRerankConfig,
) -> PfmGeometryScoreTerms:
    candidate_typed = extract_typed_outlines(candidate_depth, min_px=max(1, 25 // config.subsample))
    source_masks = _typed_masks(source_typed)
    candidate_masks = _typed_masks(candidate_typed)
    family_losses: dict[str, float | None] = {}
    family_weights = {"occlusion": 0.6, "rib": 0.2, "couloir": 0.2}
    for family in family_weights:
        if int(source_masks[family].sum()) < config.min_source_family_px:
            family_losses[family] = None
        else:
            family_losses[family] = _symmetric_outline_loss(
                source_masks[family],
                candidate_masks[family],
                source_distances[family],
                config.outline_cap_comparison_px,
            )
    available_weight = sum(family_weights[name] for name, value in family_losses.items() if value is not None)
    family_loss = (
        sum(family_weights[name] * float(value) for name, value in family_losses.items() if value is not None)
        / available_weight
        if available_weight
        else None
    )
    union_loss = _symmetric_outline_loss(
        source_masks["union"],
        candidate_masks["union"],
        source_distances["union"],
        config.outline_cap_comparison_px,
    )
    typed_loss = (
        union_loss
        if family_loss is None
        else config.typed_union_fraction * union_loss + (1.0 - config.typed_union_fraction) * family_loss
    )

    source_valid = np.isfinite(source_depth)
    candidate_valid = (
        np.isfinite(candidate_depth) & (candidate_depth >= config.min_depth_m) & (candidate_depth <= max_range_m)
    )
    common = source_valid & candidate_valid
    common_count = int(common.sum())
    source_count = int(source_valid.sum())
    candidate_count = int(candidate_valid.sum())
    precision = common_count / candidate_count if candidate_count else 0.0
    recall = common_count / source_count if source_count else 0.0
    overlap_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    if common_count >= config.min_common_depth_px:
        log_delta = np.log(candidate_depth[common]) - np.log(source_depth[common])
        median_delta = float(np.median(log_delta))
        raw_mae = float(np.mean(np.minimum(np.abs(log_delta), config.depth_log_cap)))
        centered = np.abs(log_delta - median_delta)
        centered_mae = float(np.mean(np.minimum(centered, config.depth_log_cap)))
        depth_loss = centered_mae / config.depth_log_cap
        scale = math.exp(-median_delta)
    else:
        raw_mae = None
        centered_mae = None
        depth_loss = 1.0
        scale = None
    skyline_cap = _finite_float(atlas["config"].get("residual_cap_px"), "atlas skyline residual cap")
    skyline_score = _finite_float(candidate.get("estimator_score"), "candidate estimator score")
    return PfmGeometryScoreTerms(
        skyline_loss=_rounded(np.clip(skyline_score / skyline_cap, 0.0, 1.0)),
        typed_outline_loss=_rounded(typed_loss),
        typed_union_loss=_rounded(union_loss),
        typed_family_loss=_optional_rounded(family_loss),
        occlusion_loss=_optional_rounded(family_losses["occlusion"]),
        rib_loss=_optional_rounded(family_losses["rib"]),
        couloir_loss=_optional_rounded(family_losses["couloir"]),
        depth_shape_loss=_rounded(depth_loss),
        depth_overlap_loss=_rounded(1.0 - overlap_f1),
        depth_overlap_f1=_rounded(overlap_f1),
        common_depth_px=common_count,
        source_near_depth_px=source_count,
        candidate_near_depth_px=candidate_count,
        candidate_to_source_depth_scale=_optional_rounded(scale),
        raw_log_depth_mae=_optional_rounded(raw_mae),
        centered_log_depth_mae=_optional_rounded(centered_mae),
    )


def _symmetric_outline_loss(
    source_mask: NDArray[np.bool_],
    candidate_mask: NDArray[np.bool_],
    source_distance: NDArray[np.float64] | None,
    cap_px: float,
) -> float:
    if not source_mask.any() and not candidate_mask.any():
        return 0.0
    if not source_mask.any() or not candidate_mask.any() or source_distance is None:
        return 1.0
    candidate_distance = distance_transform_edt(~candidate_mask)
    candidate_to_source = float(np.minimum(source_distance[candidate_mask], cap_px).mean())
    source_to_candidate = float(np.minimum(candidate_distance[source_mask], cap_px).mean())
    return float(np.clip(0.5 * (candidate_to_source + source_to_candidate) / cap_px, 0.0, 1.0))


def _fusion_score(terms: PfmGeometryScoreTerms, config: PfmGeometryRerankConfig) -> float:
    return (
        config.skyline_weight * terms.skyline_loss
        + config.typed_outline_weight * terms.typed_outline_loss
        + config.depth_shape_weight * terms.depth_shape_loss
        + config.depth_overlap_weight * terms.depth_overlap_loss
    )


def _component_key(candidate: PfmGeometryRerankedCandidate, field: str) -> tuple[float, int, str]:
    return float(getattr(candidate.terms, field)), candidate.original_estimator_rank, candidate.candidate_id


def _archive_basis(
    config: PfmGeometryRerankConfig,
    atlas_sha: str,
    source_sha: str,
    width: int,
    height: int,
    fov: float,
    comparison_width: int,
    comparison_height: int,
    max_range: float,
    native_patch_supplied: bool,
    source_counts: dict[str, int],
    candidates: tuple[PfmGeometryRerankedCandidate, ...],
    component_winners: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema": PFM_RERANK_ARCHIVE_SCHEMA,
        "analysis_only": True,
        "production_eligible": False,
        "source_depth_reference_used": True,
        "numeric_evaluation_reference_used": False,
        "config": config.to_record(),
        "source_atlas_sha256": atlas_sha,
        "source_depth_sha256": source_sha,
        "query_geometry": {
            "projection": "cyltan",
            "width_px": width,
            "height_px": height,
            "horizontal_fov_deg": fov,
            "comparison_width_px": comparison_width,
            "comparison_height_px": comparison_height,
            "comparison_max_range_m": _rounded(max_range),
        },
        "native_patch_supplied": native_patch_supplied,
        "source_typed_counts": source_counts,
        "candidate_pool": "complete_three_yaw_separated_modes_per_position_from_frozen_atlas",
        "candidate_count": len(candidates),
        "selected_candidate_id": candidates[0].candidate_id,
        "component_winners": component_winners,
        "candidates": [candidate.to_record() for candidate in candidates],
    }


def _evaluate_candidate(
    candidate: PfmGeometryRerankedCandidate,
    truth: CameraExtrinsics,
) -> PfmGeometryEvaluatedCandidate:
    pose = candidate.atlas_candidate["pose"]
    position = pose["position"]
    east = float(position["east_m"]) - truth.position.east_m
    north = float(position["north_m"]) - truth.position.north_m
    up = float(position["up_m"]) - truth.position.up_m
    horizontal = math.hypot(east, north)
    yaw = abs((float(pose["yaw_deg"]) - truth.yaw_deg + 180.0) % 360.0 - 180.0)
    errors = PfmGeometryPoseErrors(
        horizontal_position_m=_rounded(horizontal),
        vertical_m=_rounded(abs(up)),
        position_3d_m=_rounded(math.sqrt(horizontal * horizontal + up * up)),
        yaw_deg=_rounded(yaw),
        pitch_deg=None,
    )
    return PfmGeometryEvaluatedCandidate(
        candidate_id=candidate.candidate_id,
        rerank_rank=candidate.rerank_rank,
        original_estimator_rank=candidate.original_estimator_rank,
        errors=errors,
        normalized_joint_error=_rounded(max(horizontal / POSITION_SUCCESS_M, yaw / YAW_SUCCESS_DEG)),
        reaches_target=horizontal <= POSITION_SUCCESS_M and yaw <= YAW_SUCCESS_DEG,
    )


def _oracle_key(candidate: PfmGeometryEvaluatedCandidate) -> tuple[float, float, float, float, int, str]:
    return (
        candidate.normalized_joint_error,
        candidate.errors.horizontal_position_m / POSITION_SUCCESS_M + candidate.errors.yaw_deg / YAW_SUCCESS_DEG,
        candidate.errors.horizontal_position_m,
        candidate.errors.yaw_deg,
        candidate.rerank_rank,
        candidate.candidate_id,
    )


def _validated_top_ks(top_ks: tuple[int, ...]) -> tuple[int, ...]:
    if not top_ks or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in top_ks):
        raise ValueError("top_ks must contain positive integers")
    return tuple(dict.fromkeys(top_ks))


def _array_sha256(array: NDArray[np.float64]) -> str:
    values = np.asarray(array, dtype=np.float64)
    finite = np.isfinite(values)
    normalized = np.where(finite, values, 0.0).astype("<f8", copy=False)
    digest = hashlib.sha256(b"peakle_source_depth_array_v1\0")
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(finite.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(normalized).tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


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
