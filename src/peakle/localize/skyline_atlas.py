"""Truth-free local skyline-atlas search with isolated post-hoc evaluation.

The estimator builds one 360-degree terrain horizon at every position on a
deterministic square grid around a supplied position prior.  Every yaw is
scored against the observed skyline in the exact cylindrical/tangent image
geometry used by GeoPose crops.  The resulting archive contains estimator
inputs and scores only; benchmark truth is accepted exclusively by
``evaluate_skyline_atlas`` after the archive has been frozen and hashed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.projection import azimuths_deg, focal_length_px, rows_from_elevation_rad
from peakle.domain.terrain import TerrainMap
from peakle.localize.solve import HorizonProfile
from peakle.localize.swissdem import Patch

ATLAS_ARCHIVE_SCHEMA = "peakle_skyline_atlas_archive_v2"
ATLAS_CANDIDATE_SCHEMA = "peakle_skyline_atlas_candidate_v2"
ATLAS_EVALUATION_SCHEMA = "peakle_skyline_atlas_evaluation_v2"
ATLAS_SCORE_METHOD = "cyltan_affine_vertical_nuisance_capped_high_trim_v3"
POSITION_SUCCESS_M = 100.0
YAW_SUCCESS_DEG = 5.0

GroundSource = Literal["native_patch", "terrain_map"]


@dataclass(frozen=True, slots=True)
class SkylineAtlasConfig:
    """Frozen, fully serializable skyline-atlas search budget."""

    radius_m: float = 500.0
    spacing_m: float = 50.0
    yaw_step_deg: float = 1.0
    yaw_modes_per_position: int = 3
    yaw_mode_separation_deg: float = 8.0
    max_observed_columns: int = 192
    residual_cap_px: float = 60.0
    high_outlier_trim_fraction: float = 0.10
    high_outlier_weight: float = 0.02
    max_abs_roll_nuisance_deg: float = 10.0
    eye_height_m: float = 2.5
    ray_step_m: float = 10.0

    def __post_init__(self) -> None:
        finite_values = {
            "radius_m": self.radius_m,
            "spacing_m": self.spacing_m,
            "yaw_step_deg": self.yaw_step_deg,
            "yaw_mode_separation_deg": self.yaw_mode_separation_deg,
            "residual_cap_px": self.residual_cap_px,
            "high_outlier_trim_fraction": self.high_outlier_trim_fraction,
            "high_outlier_weight": self.high_outlier_weight,
            "max_abs_roll_nuisance_deg": self.max_abs_roll_nuisance_deg,
            "eye_height_m": self.eye_height_m,
            "ray_step_m": self.ray_step_m,
        }
        for name, value in finite_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.radius_m < 0.0:
            raise ValueError("radius_m must be non-negative")
        if self.spacing_m <= 0.0:
            raise ValueError("spacing_m must be positive")
        if self.yaw_step_deg <= 0.0 or self.yaw_step_deg > 360.0:
            raise ValueError("yaw_step_deg must be in (0, 360]")
        yaw_count = round(360.0 / self.yaw_step_deg)
        if yaw_count < 1 or not math.isclose(
            yaw_count * self.yaw_step_deg,
            360.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("yaw_step_deg must divide 360 degrees exactly")
        if isinstance(self.yaw_modes_per_position, bool) or not isinstance(self.yaw_modes_per_position, int):
            raise ValueError("yaw_modes_per_position must be a positive integer")
        if self.yaw_modes_per_position < 1:
            raise ValueError("yaw_modes_per_position must be a positive integer")
        if not 0.0 < self.yaw_mode_separation_deg <= 180.0:
            raise ValueError("yaw_mode_separation_deg must be in (0, 180]")
        if self.max_observed_columns < 1:
            raise ValueError("max_observed_columns must be positive")
        if self.residual_cap_px <= 0.0:
            raise ValueError("residual_cap_px must be positive")
        if not 0.0 <= self.high_outlier_trim_fraction < 0.5:
            raise ValueError("high_outlier_trim_fraction must be in [0, 0.5)")
        if not 0.0 <= self.high_outlier_weight <= 1.0:
            raise ValueError("high_outlier_weight must be in [0, 1]")
        if not 0.0 <= self.max_abs_roll_nuisance_deg <= 30.0:
            raise ValueError("max_abs_roll_nuisance_deg must be in [0, 30]")
        if self.eye_height_m <= 0.0:
            raise ValueError("eye_height_m must be positive")
        if self.ray_step_m <= 0.0:
            raise ValueError("ray_step_m must be positive")

    def to_record(self) -> dict[str, Any]:
        """Return the complete persisted estimator configuration."""

        return {
            "radius_m": self.radius_m,
            "spacing_m": self.spacing_m,
            "yaw_step_deg": self.yaw_step_deg,
            "yaw_modes_per_position": self.yaw_modes_per_position,
            "yaw_mode_separation_deg": self.yaw_mode_separation_deg,
            "max_observed_columns": self.max_observed_columns,
            "residual_cap_px": self.residual_cap_px,
            "high_outlier_trim_fraction": self.high_outlier_trim_fraction,
            "high_outlier_weight": self.high_outlier_weight,
            "max_abs_roll_nuisance_deg": self.max_abs_roll_nuisance_deg,
            "eye_height_m": self.eye_height_m,
            "ray_step_m": self.ray_step_m,
            "score_method": ATLAS_SCORE_METHOD,
        }


@dataclass(frozen=True, slots=True)
class SkylineAtlasCandidate:
    """One complete estimator hypothesis in score order.

    ``pitch_deg`` and ``roll_nuisance_deg`` are derived from a robustly centred,
    roll-bounded affine vertical fit.  They are crop-alignment nuisance values,
    not calibrated physical orientation estimates.
    """

    candidate_id: str
    estimator_rank: int
    grid_row: int
    grid_column: int
    yaw_index: int
    east_m: float
    north_m: float
    up_m: float
    yaw_deg: float
    pitch_deg: float
    roll_nuisance_deg: float
    vertical_shift_px: float
    vertical_slope_px_per_column: float
    score: float
    ground_source: GroundSource

    @property
    def position(self) -> LocalPoint:
        return LocalPoint(east_m=self.east_m, north_m=self.north_m, up_m=self.up_m)

    @property
    def extrinsics(self) -> CameraExtrinsics:
        return CameraExtrinsics(
            position=self.position,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            roll_deg=self.roll_nuisance_deg,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": ATLAS_CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "estimator_rank": self.estimator_rank,
            "grid": {
                "row": self.grid_row,
                "column": self.grid_column,
                "yaw_index": self.yaw_index,
            },
            "pose": {
                "position": {
                    "east_m": self.east_m,
                    "north_m": self.north_m,
                    "up_m": self.up_m,
                },
                "yaw_deg": self.yaw_deg,
                "pitch_deg": self.pitch_deg,
                "roll_deg": self.roll_nuisance_deg,
            },
            "vertical_shift_px": self.vertical_shift_px,
            "vertical_slope_px_per_column": self.vertical_slope_px_per_column,
            "estimator_score": self.score,
            "ground_source": self.ground_source,
        }


@dataclass(frozen=True, slots=True)
class SkylineAtlasLatticePosition:
    """One position and every estimator-scored yaw at that position."""

    grid_row: int
    grid_column: int
    east_m: float
    north_m: float
    up_m: float
    ground_source: GroundSource
    yaw_scores: tuple[float, ...]
    yaw_vertical_shifts_px: tuple[float, ...]
    yaw_vertical_slopes_px_per_column: tuple[float, ...]

    def to_record(self) -> dict[str, Any]:
        return {
            "grid": {"row": self.grid_row, "column": self.grid_column},
            "position": {
                "east_m": self.east_m,
                "north_m": self.north_m,
                "up_m": self.up_m,
            },
            "ground_source": self.ground_source,
            "yaw_scores": list(self.yaw_scores),
            "yaw_vertical_shifts_px": list(self.yaw_vertical_shifts_px),
            "yaw_vertical_slopes_px_per_column": list(self.yaw_vertical_slopes_px_per_column),
        }


@dataclass(frozen=True, slots=True)
class SkylineAtlasArchive:
    """Immutable estimator-only candidate archive."""

    config: SkylineAtlasConfig
    image_width_px: int
    image_height_px: int
    horizontal_fov_deg: float
    observed_finite_columns: int
    observed_used_columns: int
    observed_skyline_sha256: str
    local_frame_origin: GeoPoint
    prior_position: LocalPoint
    grid_side: int
    axis_half_width_m: float
    max_corner_distance_m: float
    native_patch_supplied: bool
    yaw_axis_deg: tuple[float, ...]
    lattice_positions: tuple[SkylineAtlasLatticePosition, ...]
    candidates: tuple[SkylineAtlasCandidate, ...]
    selected: SkylineAtlasCandidate
    archive_sha256: str

    def to_record(self) -> dict[str, Any]:
        """Return an estimator-only JSON record, including every candidate."""

        return {
            "schema": ATLAS_ARCHIVE_SCHEMA,
            "numeric_evaluation_reference_used": False,
            "supplied_prior_used": True,
            "config": self.config.to_record(),
            "query_geometry": {
                "projection": "cyltan",
                "width_px": self.image_width_px,
                "height_px": self.image_height_px,
                "horizontal_fov_deg": self.horizontal_fov_deg,
                "orientation_nuisance_semantics": "bounded_affine_vertical_fit_uncalibrated_pitch_and_roll",
                "finite_observed_columns": self.observed_finite_columns,
                "used_observed_columns": self.observed_used_columns,
                "observed_skyline_sha256": self.observed_skyline_sha256,
            },
            "coordinate_frame": {
                "type": "local_equirectangular_tangent_plane",
                "origin": self.local_frame_origin.model_dump(mode="json"),
            },
            "prior_position": self.prior_position.model_dump(mode="json"),
            "grid": {
                "shape_policy": "square_axis_aligned",
                "shape": [self.grid_side, self.grid_side],
                "position_count": self.grid_side * self.grid_side,
                "axis_half_width_m": self.axis_half_width_m,
                "max_corner_distance_m": self.max_corner_distance_m,
            },
            "native_patch_supplied": self.native_patch_supplied,
            "full_score_lattice": {
                "yaw_axis_deg": list(self.yaw_axis_deg),
                "yaw_count": len(self.yaw_axis_deg),
                "hypothesis_count": len(self.lattice_positions) * len(self.yaw_axis_deg),
                "positions": [position.to_record() for position in self.lattice_positions],
            },
            "candidate_count": len(self.candidates),
            "candidate_pool": "spatially_diverse_yaw_shortlist",
            "selected_candidate_id": self.selected.candidate_id,
            "archive_sha256": self.archive_sha256,
            "candidates": [candidate.to_record() for candidate in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class SkylineAtlasPoseErrors:
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
class SkylineAtlasEvaluatedCandidate:
    hypothesis: SkylineAtlasCandidate
    estimator_rank_scope: Literal["spatially_diverse_shortlist", "full_score_lattice"]
    errors: SkylineAtlasPoseErrors
    normalized_joint_error: float
    reaches_target: bool

    @property
    def candidate_id(self) -> str:
        return self.hypothesis.candidate_id

    @property
    def estimator_rank(self) -> int:
        return self.hypothesis.estimator_rank

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "estimator_rank": self.estimator_rank,
            "estimator_rank_scope": self.estimator_rank_scope,
            "hypothesis": self.hypothesis.to_record(),
            "errors": self.errors.to_record(),
            "normalized_joint_error": self.normalized_joint_error,
            "reaches_target": self.reaches_target,
        }


@dataclass(frozen=True, slots=True)
class SkylineAtlasTopK:
    requested_k: int
    evaluated_k: int
    reaches_target: bool
    recall: float
    best_candidate: SkylineAtlasEvaluatedCandidate

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_pool": "spatially_diverse_shortlist",
            "requested_k": self.requested_k,
            "evaluated_k": self.evaluated_k,
            "reaches_target": self.reaches_target,
            "recall": self.recall,
            "best_candidate": self.best_candidate.to_record(),
        }


@dataclass(frozen=True, slots=True)
class SkylineAtlasSelectionRegret:
    normalized_joint_error: float
    horizontal_position_m: float
    yaw_deg: float

    def to_record(self) -> dict[str, float]:
        return {
            "normalized_joint_error": self.normalized_joint_error,
            "horizontal_position_m": self.horizontal_position_m,
            "yaw_deg": self.yaw_deg,
        }


@dataclass(frozen=True, slots=True)
class SkylineAtlasEvaluation:
    """Post-freeze benchmark evaluation, kept separate from the estimator archive."""

    archive_sha256: str
    winner_errors: SkylineAtlasEvaluatedCandidate
    shortlist_gt_oracle: SkylineAtlasEvaluatedCandidate
    full_lattice_gt_oracle: SkylineAtlasEvaluatedCandidate
    closest_generated_candidate: SkylineAtlasEvaluatedCandidate
    shortlist_top_k: tuple[SkylineAtlasTopK, ...]
    selection_regret: SkylineAtlasSelectionRegret

    @property
    def shortlist_top_k_reach(self) -> dict[int, bool]:
        return {item.requested_k: item.reaches_target for item in self.shortlist_top_k}

    @property
    def shortlist_top_k_recall(self) -> dict[int, float]:
        return {item.requested_k: item.recall for item in self.shortlist_top_k}

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": ATLAS_EVALUATION_SCHEMA,
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
            "shortlist_gt_oracle": self.shortlist_gt_oracle.to_record(),
            "full_lattice_gt_oracle": self.full_lattice_gt_oracle.to_record(),
            "closest_generated_candidate": {
                "criterion": "horizontal_position_m",
                **self.closest_generated_candidate.to_record(),
            },
            "shortlist_top_k": [item.to_record() for item in self.shortlist_top_k],
            "selection_regret": self.selection_regret.to_record(),
        }


@dataclass(frozen=True, slots=True)
class _CandidateDraft:
    grid_row: int
    grid_column: int
    yaw_index: int
    east_m: float
    north_m: float
    up_m: float
    yaw_deg: float
    pitch_deg: float
    roll_nuisance_deg: float
    vertical_shift_px: float
    vertical_slope_px_per_column: float
    score: float
    ground_source: GroundSource


def build_skyline_atlas(
    terrain: TerrainMap,
    observed_skyline: NDArray[np.float64],
    image_height_px: int,
    horizontal_fov_deg: float,
    prior_position: LocalPoint,
    *,
    native_patch: Patch | None = None,
    config: SkylineAtlasConfig | None = None,
) -> SkylineAtlasArchive:
    """Build and freeze a deterministic truth-free local skyline atlas."""

    settings = config or SkylineAtlasConfig()
    observed = _validated_observed_skyline(observed_skyline, image_height_px, horizontal_fov_deg)
    _validate_position(prior_position)
    if native_patch is not None:
        _validate_patch(native_patch)

    finite_columns = np.flatnonzero(np.isfinite(observed))
    used_columns = _evenly_spaced_subset(finite_columns, settings.max_observed_columns)
    observed_used = observed[used_columns]
    column_azimuth_offsets = azimuths_deg(
        observed.size,
        horizontal_fov_deg,
        0.0,
        "cyltan",
    )[used_columns]
    column_x_px = used_columns.astype(np.float64) - (observed.size - 1.0) / 2.0
    yaw_count = round(360.0 / settings.yaw_step_deg)
    yaws = -180.0 + np.arange(yaw_count, dtype=np.float64) * settings.yaw_step_deg
    offsets = _grid_offsets(settings)
    drafts: list[_CandidateDraft] = []
    lattice_positions: list[SkylineAtlasLatticePosition] = []

    for grid_row, north_offset in enumerate(offsets):
        north_m = prior_position.north_m + float(north_offset)
        for grid_column, east_offset in enumerate(offsets):
            east_m = prior_position.east_m + float(east_offset)
            ground_m, ground_source = _ground_elevation(terrain, native_patch, east_m, north_m)
            up_m = ground_m + settings.eye_height_m
            profile = HorizonProfile(
                terrain,
                up_m,
                step=settings.ray_step_m,
                cam_e=east_m,
                cam_n=north_m,
                patch=native_patch,
            )
            scores, shifts, slopes = _score_all_yaws(
                profile,
                yaws,
                column_azimuth_offsets,
                column_x_px,
                observed_used,
                observed.size,
                image_height_px,
                horizontal_fov_deg,
                settings,
            )
            scores = np.asarray([_rounded(float(value)) for value in scores], dtype=np.float64)
            shifts = np.asarray([_rounded(float(value)) for value in shifts], dtype=np.float64)
            slopes = np.asarray([_rounded(float(value)) for value in slopes], dtype=np.float64)
            pitch_degrees = np.degrees(
                np.arctan(
                    shifts
                    / focal_length_px(
                        observed.size,
                        horizontal_fov_deg,
                        "cyltan",
                    )
                )
            )
            rounded_east = _rounded(east_m)
            rounded_north = _rounded(north_m)
            rounded_up = _rounded(up_m)
            lattice_positions.append(
                SkylineAtlasLatticePosition(
                    grid_row=grid_row,
                    grid_column=grid_column,
                    east_m=rounded_east,
                    north_m=rounded_north,
                    up_m=rounded_up,
                    ground_source=ground_source,
                    yaw_scores=tuple(float(value) for value in scores),
                    yaw_vertical_shifts_px=tuple(float(value) for value in shifts),
                    yaw_vertical_slopes_px_per_column=tuple(float(value) for value in slopes),
                )
            )
            for yaw_index in _select_yaw_modes(scores, yaws, settings):
                yaw_deg = yaws[yaw_index]
                drafts.append(
                    _CandidateDraft(
                        grid_row=grid_row,
                        grid_column=grid_column,
                        yaw_index=yaw_index,
                        east_m=rounded_east,
                        north_m=rounded_north,
                        up_m=rounded_up,
                        yaw_deg=_rounded(float(yaw_deg)),
                        pitch_deg=_rounded(float(pitch_degrees[yaw_index])),
                        roll_nuisance_deg=_rounded(math.degrees(math.atan(float(slopes[yaw_index])))),
                        vertical_shift_px=_rounded(float(shifts[yaw_index])),
                        vertical_slope_px_per_column=_rounded(float(slopes[yaw_index])),
                        score=_rounded(float(scores[yaw_index])),
                        ground_source=ground_source,
                    )
                )

    drafts.sort(key=lambda item: (item.score, item.grid_row, item.grid_column, item.yaw_index))
    candidates = tuple(_ranked_candidate(draft, rank) for rank, draft in enumerate(drafts, start=1))
    selected = candidates[0]
    observed_sha256 = _observed_skyline_sha256(observed)
    axis_half_width_m = float(abs(offsets[-1])) if offsets.size else 0.0
    max_corner_distance_m = math.sqrt(2.0) * axis_half_width_m
    yaw_axis = tuple(_rounded(float(yaw)) for yaw in yaws)
    frozen_lattice = tuple(lattice_positions)
    archive_basis = {
        "schema": ATLAS_ARCHIVE_SCHEMA,
        "numeric_evaluation_reference_used": False,
        "supplied_prior_used": True,
        "config": settings.to_record(),
        "query_geometry": {
            "projection": "cyltan",
            "width_px": int(observed.size),
            "height_px": image_height_px,
            "horizontal_fov_deg": horizontal_fov_deg,
            "orientation_nuisance_semantics": "bounded_affine_vertical_fit_uncalibrated_pitch_and_roll",
            "finite_observed_columns": int(finite_columns.size),
            "used_observed_columns": int(used_columns.size),
            "observed_skyline_sha256": observed_sha256,
        },
        "coordinate_frame": {
            "type": "local_equirectangular_tangent_plane",
            "origin": terrain.spec.origin.model_dump(mode="json"),
        },
        "prior_position": prior_position.model_dump(mode="json"),
        "grid": {
            "shape_policy": "square_axis_aligned",
            "shape": [int(offsets.size), int(offsets.size)],
            "position_count": int(offsets.size * offsets.size),
            "axis_half_width_m": axis_half_width_m,
            "max_corner_distance_m": max_corner_distance_m,
        },
        "native_patch_supplied": native_patch is not None,
        "full_score_lattice": {
            "yaw_axis_deg": list(yaw_axis),
            "yaw_count": len(yaw_axis),
            "hypothesis_count": len(frozen_lattice) * len(yaw_axis),
            "positions": [position.to_record() for position in frozen_lattice],
        },
        "candidate_count": len(candidates),
        "candidate_pool": "spatially_diverse_yaw_shortlist",
        "selected_candidate_id": selected.candidate_id,
        "candidates": [candidate.to_record() for candidate in candidates],
    }
    return SkylineAtlasArchive(
        config=settings,
        image_width_px=int(observed.size),
        image_height_px=image_height_px,
        horizontal_fov_deg=float(horizontal_fov_deg),
        observed_finite_columns=int(finite_columns.size),
        observed_used_columns=int(used_columns.size),
        observed_skyline_sha256=observed_sha256,
        local_frame_origin=terrain.spec.origin,
        prior_position=prior_position,
        grid_side=int(offsets.size),
        axis_half_width_m=axis_half_width_m,
        max_corner_distance_m=max_corner_distance_m,
        native_patch_supplied=native_patch is not None,
        yaw_axis_deg=yaw_axis,
        lattice_positions=frozen_lattice,
        candidates=candidates,
        selected=selected,
        archive_sha256=_canonical_sha256(archive_basis),
    )


def evaluate_skyline_atlas(
    archive: SkylineAtlasArchive,
    truth: CameraExtrinsics,
    *,
    top_ks: tuple[int, ...] = (1, 5, 10, 50, 100),
) -> SkylineAtlasEvaluation:
    """Evaluate a previously frozen archive without changing estimator order."""

    normalized_top_ks = _validated_top_ks(top_ks)
    evaluated = tuple(
        _evaluate_candidate(candidate, truth, rank_scope="spatially_diverse_shortlist")
        for candidate in archive.candidates
    )
    winner = evaluated[0]
    shortlist_oracle = min(evaluated, key=_oracle_key)
    full_lattice_oracle, closest = _evaluate_full_lattice(archive, truth)
    top_k_records: list[SkylineAtlasTopK] = []
    for requested_k in normalized_top_ks:
        evaluated_k = min(requested_k, len(evaluated))
        prefix = evaluated[:evaluated_k]
        best = min(prefix, key=_oracle_key)
        reached = any(candidate.reaches_target for candidate in prefix)
        top_k_records.append(
            SkylineAtlasTopK(
                requested_k=requested_k,
                evaluated_k=evaluated_k,
                reaches_target=reached,
                recall=1.0 if reached else 0.0,
                best_candidate=best,
            )
        )
    regret = SkylineAtlasSelectionRegret(
        normalized_joint_error=_rounded(winner.normalized_joint_error - full_lattice_oracle.normalized_joint_error),
        horizontal_position_m=_rounded(
            winner.errors.horizontal_position_m - full_lattice_oracle.errors.horizontal_position_m
        ),
        yaw_deg=_rounded(winner.errors.yaw_deg - full_lattice_oracle.errors.yaw_deg),
    )
    return SkylineAtlasEvaluation(
        archive_sha256=archive.archive_sha256,
        winner_errors=winner,
        shortlist_gt_oracle=shortlist_oracle,
        full_lattice_gt_oracle=full_lattice_oracle,
        closest_generated_candidate=closest,
        shortlist_top_k=tuple(top_k_records),
        selection_regret=regret,
    )


def _validated_observed_skyline(
    observed_skyline: NDArray[np.float64],
    image_height_px: int,
    horizontal_fov_deg: float,
) -> NDArray[np.float64]:
    observed = np.asarray(observed_skyline, dtype=np.float64)
    if observed.ndim != 1 or observed.size < 1:
        raise ValueError("observed_skyline must be a non-empty one-dimensional array")
    if image_height_px < 1:
        raise ValueError("image_height_px must be positive")
    if not math.isfinite(horizontal_fov_deg) or not 1.0 < horizontal_fov_deg < 179.0:
        raise ValueError("horizontal_fov_deg must be finite and between 1 and 179 degrees")
    if not np.any(np.isfinite(observed)):
        raise ValueError("observed_skyline must contain at least one finite column")
    return observed


def _validate_position(position: LocalPoint) -> None:
    if not all(math.isfinite(value) for value in position.as_tuple()):
        raise ValueError("prior_position must contain only finite coordinates")


def _validate_patch(patch: Patch) -> None:
    x_m = np.asarray(patch.x_m, dtype=np.float64)
    y_m = np.asarray(patch.y_m, dtype=np.float64)
    elevation = np.asarray(patch.elevation_m, dtype=np.float64)
    if x_m.ndim != 1 or y_m.ndim != 1 or x_m.size < 2 or y_m.size < 2:
        raise ValueError("native_patch axes must be one-dimensional with at least two samples")
    if elevation.shape != (y_m.size, x_m.size):
        raise ValueError("native_patch elevation shape must match its coordinate axes")
    if not np.all(np.isfinite(x_m)) or not np.all(np.isfinite(y_m)):
        raise ValueError("native_patch axes must be finite")
    if not np.all(np.diff(x_m) > 0.0) or not np.all(np.diff(y_m) > 0.0):
        raise ValueError("native_patch axes must be strictly increasing")


def _grid_offsets(config: SkylineAtlasConfig) -> NDArray[np.float64]:
    steps = int(math.floor(config.radius_m / config.spacing_m + 1e-12))
    return np.arange(-steps, steps + 1, dtype=np.float64) * config.spacing_m


def _evenly_spaced_subset(columns: NDArray[np.int64], maximum: int) -> NDArray[np.int64]:
    if columns.size <= maximum:
        return columns
    indices = np.linspace(0, columns.size - 1, maximum).round().astype(np.int64)
    return columns[indices]


def _ground_elevation(
    terrain: TerrainMap,
    patch: Patch | None,
    east_m: float,
    north_m: float,
) -> tuple[float, GroundSource]:
    if patch is not None:
        native = _sample_native_patch(patch, east_m, north_m)
        if native is not None:
            return native, "native_patch"
    return terrain.elevation_at(east_m, north_m), "terrain_map"


def _sample_native_patch(patch: Patch, east_m: float, north_m: float) -> float | None:
    x_m = np.asarray(patch.x_m, dtype=np.float64)
    y_m = np.asarray(patch.y_m, dtype=np.float64)
    if east_m < x_m[0] or east_m > x_m[-1] or north_m < y_m[0] or north_m > y_m[-1]:
        return None
    column0 = int(np.clip(np.searchsorted(x_m, east_m, side="right") - 1, 0, x_m.size - 2))
    row0 = int(np.clip(np.searchsorted(y_m, north_m, side="right") - 1, 0, y_m.size - 2))
    column1 = column0 + 1
    row1 = row0 + 1
    east_fraction = (east_m - x_m[column0]) / (x_m[column1] - x_m[column0])
    north_fraction = (north_m - y_m[row0]) / (y_m[row1] - y_m[row0])
    corners = np.asarray(
        [
            patch.elevation_m[row0, column0],
            patch.elevation_m[row0, column1],
            patch.elevation_m[row1, column0],
            patch.elevation_m[row1, column1],
        ],
        dtype=np.float64,
    )
    weights = np.asarray(
        [
            (1.0 - east_fraction) * (1.0 - north_fraction),
            east_fraction * (1.0 - north_fraction),
            (1.0 - east_fraction) * north_fraction,
            east_fraction * north_fraction,
        ],
        dtype=np.float64,
    )
    required = weights > 1e-12
    if not np.all(np.isfinite(corners[required])):
        return None
    return float(np.sum(np.where(required, corners, 0.0) * weights))


def _score_all_yaws(
    profile: HorizonProfile,
    yaws: NDArray[np.float64],
    column_azimuth_offsets: NDArray[np.float64],
    column_x_px: NDArray[np.float64],
    observed_rows: NDArray[np.float64],
    image_width_px: int,
    image_height_px: int,
    horizontal_fov_deg: float,
    config: SkylineAtlasConfig,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sample_azimuths = yaws[:, None] + column_azimuth_offsets[None, :]
    elevations = _sample_horizon(profile, sample_azimuths)
    predicted_rows = rows_from_elevation_rad(
        elevations,
        image_width_px,
        image_height_px,
        horizontal_fov_deg,
        "cyltan",
    )
    valid = np.isfinite(predicted_rows)
    differences = np.ma.array(
        observed_rows[None, :] - predicted_rows,
        mask=~valid,
    )
    difference_values = np.asarray(differences.filled(0.0), dtype=np.float64)
    weights = valid.astype(np.float64)
    count = np.sum(weights, axis=1)
    sum_x = np.sum(weights * column_x_px[None, :], axis=1)
    sum_y = np.sum(difference_values, axis=1)
    sum_xx = np.sum(weights * column_x_px[None, :] ** 2, axis=1)
    sum_xy = np.sum(difference_values * column_x_px[None, :], axis=1)
    denominator = count * sum_xx - sum_x * sum_x
    slopes = np.divide(
        count * sum_xy - sum_x * sum_y,
        denominator,
        out=np.zeros_like(count),
        where=np.abs(denominator) > 1e-12,
    )
    max_slope = math.tan(math.radians(config.max_abs_roll_nuisance_deg))
    slopes = np.clip(slopes, -max_slope, max_slope)
    detrended = np.ma.array(
        difference_values - slopes[:, None] * column_x_px[None, :],
        mask=~valid,
    )
    shifts = np.asarray(np.ma.median(detrended, axis=1).filled(0.0), dtype=np.float64)
    fitted_rows = predicted_rows + shifts[:, None] + slopes[:, None] * column_x_px[None, :]
    residuals = np.abs(observed_rows[None, :] - fitted_rows)
    capped = np.where(valid, np.minimum(residuals, config.residual_cap_px), config.residual_cap_px)
    ordered = np.sort(capped, axis=1)
    trim_count = int(math.floor(ordered.shape[1] * config.high_outlier_trim_fraction))
    core_count = max(1, ordered.shape[1] - trim_count)
    core_mean = np.mean(ordered[:, :core_count], axis=1)
    if trim_count:
        high_mean = np.mean(ordered[:, core_count:], axis=1)
        scores = (1.0 - config.high_outlier_weight) * core_mean + config.high_outlier_weight * high_mean
    else:
        scores = core_mean
    return np.asarray(scores, dtype=np.float64), shifts, np.asarray(slopes, dtype=np.float64)


def _select_yaw_modes(
    scores: NDArray[np.float64],
    yaws: NDArray[np.float64],
    config: SkylineAtlasConfig,
) -> tuple[int, ...]:
    """Greedily retain the best circularly separated yaw modes at one position."""

    yaw_indices = np.arange(yaws.size, dtype=np.int64)
    order = np.lexsort((yaw_indices, scores))
    selected: list[int] = []
    for raw_index in order:
        index = int(raw_index)
        if all(
            _circular_distance_deg(float(yaws[index]), float(yaws[other])) >= config.yaw_mode_separation_deg - 1e-12
            for other in selected
        ):
            selected.append(index)
            if len(selected) == config.yaw_modes_per_position:
                break
    return tuple(selected)


def _sample_horizon(profile: HorizonProfile, azimuth_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    azimuth_axis = np.asarray(profile.az_deg, dtype=np.float64)
    elevations = np.asarray(profile.el, dtype=np.float64)
    step_deg = float(azimuth_axis[1] - azimuth_axis[0])
    fractional_index = ((azimuth_deg - azimuth_axis[0]) % 360.0) / step_deg
    index0 = np.floor(fractional_index).astype(np.int64) % elevations.size
    index1 = (index0 + 1) % elevations.size
    fraction = fractional_index - np.floor(fractional_index)
    elevation0 = elevations[index0]
    elevation1 = elevations[index1]
    interpolated = elevation0 * (1.0 - fraction) + elevation1 * fraction
    return np.where(np.isfinite(elevation0) & np.isfinite(elevation1), interpolated, np.nan)


def _ranked_candidate(draft: _CandidateDraft, rank: int) -> SkylineAtlasCandidate:
    identity_record = {
        "schema": ATLAS_CANDIDATE_SCHEMA,
        "grid": {
            "row": draft.grid_row,
            "column": draft.grid_column,
            "yaw_index": draft.yaw_index,
        },
        "pose": {
            "position": {
                "east_m": draft.east_m,
                "north_m": draft.north_m,
                "up_m": draft.up_m,
            },
            "yaw_deg": draft.yaw_deg,
            "pitch_deg": draft.pitch_deg,
            "roll_deg": draft.roll_nuisance_deg,
        },
        "vertical_shift_px": draft.vertical_shift_px,
        "vertical_slope_px_per_column": draft.vertical_slope_px_per_column,
        "estimator_score": draft.score,
        "ground_source": draft.ground_source,
    }
    return SkylineAtlasCandidate(
        candidate_id=_canonical_sha256(identity_record),
        estimator_rank=rank,
        grid_row=draft.grid_row,
        grid_column=draft.grid_column,
        yaw_index=draft.yaw_index,
        east_m=draft.east_m,
        north_m=draft.north_m,
        up_m=draft.up_m,
        yaw_deg=draft.yaw_deg,
        pitch_deg=draft.pitch_deg,
        roll_nuisance_deg=draft.roll_nuisance_deg,
        vertical_shift_px=draft.vertical_shift_px,
        vertical_slope_px_per_column=draft.vertical_slope_px_per_column,
        score=draft.score,
        ground_source=draft.ground_source,
    )


def _evaluate_candidate(
    candidate: SkylineAtlasCandidate,
    truth: CameraExtrinsics,
    *,
    rank_scope: Literal["spatially_diverse_shortlist", "full_score_lattice"],
) -> SkylineAtlasEvaluatedCandidate:
    east_error = candidate.east_m - truth.position.east_m
    north_error = candidate.north_m - truth.position.north_m
    vertical_error = candidate.up_m - truth.position.up_m
    horizontal = math.hypot(east_error, north_error)
    yaw_error = abs(_angle_error(candidate.yaw_deg, truth.yaw_deg))
    errors = SkylineAtlasPoseErrors(
        horizontal_position_m=_rounded(horizontal),
        vertical_m=_rounded(abs(vertical_error)),
        position_3d_m=_rounded(math.sqrt(horizontal * horizontal + vertical_error * vertical_error)),
        yaw_deg=_rounded(yaw_error),
        pitch_deg=None,
    )
    normalized = max(horizontal / POSITION_SUCCESS_M, yaw_error / YAW_SUCCESS_DEG)
    return SkylineAtlasEvaluatedCandidate(
        hypothesis=candidate,
        estimator_rank_scope=rank_scope,
        errors=errors,
        normalized_joint_error=_rounded(normalized),
        reaches_target=horizontal <= POSITION_SUCCESS_M and yaw_error <= YAW_SUCCESS_DEG,
    )


def _evaluate_full_lattice(
    archive: SkylineAtlasArchive,
    truth: CameraExtrinsics,
) -> tuple[SkylineAtlasEvaluatedCandidate, SkylineAtlasEvaluatedCandidate]:
    """Return the all-yaw GT oracle and closest generated hypothesis post-freeze."""

    yaw_count = len(archive.yaw_axis_deg)
    position_count = len(archive.lattice_positions)
    score_matrix = np.asarray(
        [position.yaw_scores for position in archive.lattice_positions],
        dtype=np.float64,
    )
    if score_matrix.shape != (position_count, yaw_count):
        raise ValueError("archive score lattice shape does not match its position and yaw axes")
    flat_scores = score_matrix.ravel()
    flat_indices = np.arange(flat_scores.size, dtype=np.int64)
    score_order = np.lexsort((flat_indices, flat_scores))
    estimator_ranks = np.empty(flat_scores.size, dtype=np.int64)
    estimator_ranks[score_order] = np.arange(1, flat_scores.size + 1, dtype=np.int64)

    horizontal_by_position = np.asarray(
        [
            math.hypot(
                position.east_m - truth.position.east_m,
                position.north_m - truth.position.north_m,
            )
            for position in archive.lattice_positions
        ],
        dtype=np.float64,
    )
    yaw_errors = np.asarray(
        [abs(_angle_error(yaw, truth.yaw_deg)) for yaw in archive.yaw_axis_deg],
        dtype=np.float64,
    )
    flat_horizontal = np.repeat(horizontal_by_position, yaw_count)
    flat_yaw = np.tile(yaw_errors, position_count)
    flat_joint = np.maximum(flat_horizontal / POSITION_SUCCESS_M, flat_yaw / YAW_SUCCESS_DEG)
    oracle_order = np.lexsort(
        (
            flat_indices,
            flat_scores,
            flat_yaw,
            flat_horizontal,
            flat_joint,
        )
    )
    closest_order = np.lexsort(
        (
            flat_indices,
            estimator_ranks,
            flat_yaw,
            flat_horizontal,
        )
    )
    oracle = _evaluate_lattice_hypothesis(
        archive,
        truth,
        int(oracle_order[0]),
        int(estimator_ranks[int(oracle_order[0])]),
    )
    closest = _evaluate_lattice_hypothesis(
        archive,
        truth,
        int(closest_order[0]),
        int(estimator_ranks[int(closest_order[0])]),
    )
    return oracle, closest


def _evaluate_lattice_hypothesis(
    archive: SkylineAtlasArchive,
    truth: CameraExtrinsics,
    flat_index: int,
    estimator_rank: int,
) -> SkylineAtlasEvaluatedCandidate:
    yaw_count = len(archive.yaw_axis_deg)
    position_index, yaw_index = divmod(flat_index, yaw_count)
    position = archive.lattice_positions[position_index]
    vertical_shift_px = position.yaw_vertical_shifts_px[yaw_index]
    vertical_slope = position.yaw_vertical_slopes_px_per_column[yaw_index]
    pitch_deg = math.degrees(
        math.atan(
            vertical_shift_px
            / focal_length_px(
                archive.image_width_px,
                archive.horizontal_fov_deg,
                "cyltan",
            )
        )
    )
    candidate = _ranked_candidate(
        _CandidateDraft(
            grid_row=position.grid_row,
            grid_column=position.grid_column,
            yaw_index=yaw_index,
            east_m=position.east_m,
            north_m=position.north_m,
            up_m=position.up_m,
            yaw_deg=archive.yaw_axis_deg[yaw_index],
            pitch_deg=_rounded(pitch_deg),
            roll_nuisance_deg=_rounded(math.degrees(math.atan(vertical_slope))),
            vertical_shift_px=vertical_shift_px,
            vertical_slope_px_per_column=vertical_slope,
            score=position.yaw_scores[yaw_index],
            ground_source=position.ground_source,
        ),
        estimator_rank,
    )
    return _evaluate_candidate(candidate, truth, rank_scope="full_score_lattice")


def _oracle_key(candidate: SkylineAtlasEvaluatedCandidate) -> tuple[float, float, float, float, int, str]:
    position_normalized = candidate.errors.horizontal_position_m / POSITION_SUCCESS_M
    yaw_normalized = candidate.errors.yaw_deg / YAW_SUCCESS_DEG
    return (
        candidate.normalized_joint_error,
        position_normalized + yaw_normalized,
        candidate.errors.horizontal_position_m,
        candidate.errors.yaw_deg,
        candidate.estimator_rank,
        candidate.candidate_id,
    )


def _validated_top_ks(top_ks: tuple[int, ...]) -> tuple[int, ...]:
    if not top_ks:
        raise ValueError("top_ks must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in top_ks):
        raise ValueError("top_ks must contain only positive integers")
    return tuple(dict.fromkeys(top_ks))


def _observed_skyline_sha256(observed: NDArray[np.float64]) -> str:
    finite = np.isfinite(observed)
    normalized = np.where(finite, observed, 0.0).astype("<f8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"peakle_observed_skyline_v1\0")
    digest.update(int(observed.size).to_bytes(8, "little", signed=False))
    digest.update(np.ascontiguousarray(finite.astype(np.uint8)).tobytes())
    digest.update(np.ascontiguousarray(normalized).tobytes())
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _angle_error(value: float, truth: float) -> float:
    return ((value - truth + 180.0) % 360.0) - 180.0


def _circular_distance_deg(first: float, second: float) -> float:
    return abs(_angle_error(first, second))


def _rounded(value: float) -> float:
    return round(float(value), 12)
