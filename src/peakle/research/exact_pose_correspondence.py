"""Frozen correspondence artifacts and exact-pose geographic grading."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.localize.correspondence import MatchSet
from peakle.rendering.pinhole import project_points
from peakle.rendering.terrain_view import TerrainRenderBundle, lift_pinhole_depth_pixels, lift_render_pixels
from peakle.research.webgl_contract import LoadedWebGLQueryArtifact

_ARTIFACT_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MATCH_SUFFIX = ".matches.f64le"
_MATCH_ROLE = "query_render_xy_confidence_selected_f64le"


@dataclass(frozen=True, slots=True)
class MatchArtifactFile:
    """Manifest record for one deterministic N×6 match table."""

    filename: str
    role: Literal["query_render_xy_confidence_selected_f64le"]
    dtype: Literal["float64_little_endian"]
    shape: tuple[int, int]
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class FrozenMatchArtifact:
    """Immutable match bytes ready for flat artifact publication."""

    files: MappingProxyType[str, bytes]
    manifest: tuple[MatchArtifactFile, ...]


@dataclass(frozen=True, slots=True)
class LiftedQueryPoints:
    world_xyz_m: NDArray[np.float64]
    range_m: NDArray[np.float64]
    valid: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ExactPoseCaseGate:
    min_selected: int = 24
    min_render_lift_valid_fraction: float = 0.80
    min_correct: int = 18
    min_precision: float = 0.70
    min_occupied_cells: int = 6
    min_x_span_fraction: float = 0.40
    min_y_span_fraction: float = 0.15

    def evaluate(
        self,
        *,
        selected_count: int,
        render_lift_valid_fraction: float,
        correct_count: int,
        precision: float,
        occupied_cells: int,
        x_span_fraction: float,
        y_span_fraction: float,
    ) -> dict[str, bool]:
        checks = {
            "selected_count": selected_count >= self.min_selected,
            "render_lift_valid_fraction": render_lift_valid_fraction >= self.min_render_lift_valid_fraction,
            "correct_count": correct_count >= self.min_correct,
            "precision": precision >= self.min_precision,
            "occupied_cells": occupied_cells >= self.min_occupied_cells,
            "x_span_fraction": x_span_fraction >= self.min_x_span_fraction,
            "y_span_fraction": y_span_fraction >= self.min_y_span_fraction,
        }
        return {**checks, "passed": all(checks.values())}


EXACT_POSE_CASE_GATE = ExactPoseCaseGate()


def freeze_match_artifact(matches: MatchSet, artifact_id: str) -> FrozenMatchArtifact:
    """Freeze every raw match and its worker-selected bit without filtering."""

    if _ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise ValueError("match artifact_id must contain only letters, digits, dot, underscore, or hyphen")
    table = np.empty((matches.count, 6), dtype="<f8")
    table[:, :2] = matches.query_xy_px
    table[:, 2:4] = matches.render_xy_px
    table[:, 4] = matches.confidence
    table[:, 5] = np.asarray(matches.selected, dtype=np.bool_)
    content = table.tobytes(order="C")
    filename = f"{artifact_id}{_MATCH_SUFFIX}"
    record = MatchArtifactFile(
        filename=filename,
        role=_MATCH_ROLE,
        dtype="float64_little_endian",
        shape=table.shape,
        sha256=hashlib.sha256(content).hexdigest(),
        bytes=len(content),
    )
    return FrozenMatchArtifact(files=MappingProxyType({filename: content}), manifest=(record,))


def load_frozen_match_artifact(
    manifest: Sequence[MatchArtifactFile],
    files: Mapping[str, bytes],
) -> MatchSet:
    """Hash-validate and decode a complete frozen match table."""

    if len(manifest) != 1 or not isinstance(manifest[0], MatchArtifactFile):
        raise ValueError("frozen match manifest must contain exactly one MatchArtifactFile")
    record = manifest[0]
    artifact_id = record.filename.removesuffix(_MATCH_SUFFIX)
    if (
        not record.filename.endswith(_MATCH_SUFFIX)
        or not artifact_id
        or _ARTIFACT_ID.fullmatch(artifact_id) is None
        or record.role != _MATCH_ROLE
        or record.dtype != "float64_little_endian"
    ):
        raise ValueError("invalid frozen match filename, role, or dtype")
    if len(record.shape) != 2 or record.shape[0] < 0 or record.shape[1] != 6:
        raise ValueError("frozen match table must have shape (N, 6)")
    if record.bytes != record.shape[0] * 48 or _SHA256.fullmatch(record.sha256) is None:
        raise ValueError("frozen match manifest has an invalid byte count or SHA-256")
    if set(files) != {record.filename}:
        raise ValueError("frozen match files differ from the manifest")
    content = files[record.filename]
    if not isinstance(content, bytes):
        raise TypeError("frozen match content must be immutable bytes")
    if len(content) != record.bytes:
        raise ValueError("frozen match byte count differs from its manifest")
    if hashlib.sha256(content).hexdigest() != record.sha256:
        raise ValueError("frozen match SHA-256 differs from its manifest")
    table = np.frombuffer(content, dtype="<f8").reshape(record.shape)
    if not np.all((table[:, 5] == 0.0) | (table[:, 5] == 1.0)):
        raise ValueError("frozen selected values must be binary")
    selected = table[:, 5].astype(np.bool_)
    selected.flags.writeable = False
    return MatchSet(
        query_xy_px=table[:, :2],
        render_xy_px=table[:, 2:4],
        confidence=table[:, 4],
        selected=selected,
    )


def lift_query_depth_pixels(
    query: LoadedWebGLQueryArtifact,
    query_xy_px: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    *,
    max_relative_depth_span: float = 0.08,
) -> LiftedQueryPoints:
    """Lift WebGL depth through the shared source-agnostic pinhole sampler."""

    shape = (intrinsics.height_px, intrinsics.width_px)
    if query.terrain_mask.shape != shape or query.forward_depth_m.shape != shape:
        raise ValueError("query geometry dimensions differ from query intrinsics")
    lifted = lift_pinhole_depth_pixels(
        query.forward_depth_m,
        intrinsics,
        extrinsics,
        query_xy_px,
        support_mask=query.terrain_mask,
        max_relative_depth_span=max_relative_depth_span,
    )
    position = np.asarray(extrinsics.position.as_tuple(), dtype=np.float64)
    ranges = np.linalg.norm(lifted.world_xyz_m - position[None, :], axis=1)
    ranges[~lifted.valid] = np.nan
    return LiftedQueryPoints(world_xyz_m=lifted.world_xyz_m, range_m=ranges, valid=lifted.valid)


def grade_frozen_exact_pose_correspondences(
    manifest: Sequence[MatchArtifactFile],
    files: Mapping[str, bytes],
    query: LoadedWebGLQueryArtifact,
    query_intrinsics: CameraIntrinsics,
    query_extrinsics: CameraExtrinsics,
    render: TerrainRenderBundle,
) -> dict[str, Any]:
    """Grade all frozen matches, aggregating only the worker-selected subset."""

    matches = load_frozen_match_artifact(manifest, files)
    query_lift = lift_query_depth_pixels(query, matches.query_xy_px, query_intrinsics, query_extrinsics)
    render_lift = lift_render_pixels(render, matches.render_xy_px)
    u_px, v_px, _depth, projected = project_points(render_lift.world_xyz_m, query_intrinsics, query_extrinsics)
    reprojection_error = np.linalg.norm(
        np.column_stack((u_px, v_px)) - matches.query_xy_px,
        axis=1,
    )
    distance = np.linalg.norm(render_lift.world_xyz_m - query_lift.world_xyz_m, axis=1)
    both_valid = query_lift.valid & render_lift.valid
    correct = (
        both_valid & projected & (reprojection_error <= 5.0) & (distance <= np.maximum(25.0, 0.01 * query_lift.range_m))
    )
    selected = np.asarray(matches.selected, dtype=np.bool_)
    selected_count = int(selected.sum())
    query_valid_count = int((selected & query_lift.valid).sum())
    render_valid_count = int((selected & render_lift.valid).sum())
    both_valid_count = int((selected & both_valid).sum())
    correct_selected = selected & correct
    correct_count = int(correct_selected.sum())
    occupied, coverage, x_span, y_span = _distribution(
        matches.query_xy_px[correct_selected], query_intrinsics.width_px, query_intrinsics.height_px
    )
    render_fraction = _fraction(render_valid_count, selected_count)
    precision = _fraction(correct_count, selected_count)
    gate = EXACT_POSE_CASE_GATE.evaluate(
        selected_count=selected_count,
        render_lift_valid_fraction=render_fraction,
        correct_count=correct_count,
        precision=precision,
        occupied_cells=occupied,
        x_span_fraction=x_span,
        y_span_fraction=y_span,
    )
    return {
        "raw_count": matches.count,
        "selected_count": selected_count,
        "query_lift_valid_count": query_valid_count,
        "query_lift_valid_fraction": _fraction(query_valid_count, selected_count),
        "render_lift_valid_count": render_valid_count,
        "render_lift_valid_fraction": render_fraction,
        "both_lifts_valid_count": both_valid_count,
        "both_lifts_valid_fraction": _fraction(both_valid_count, selected_count),
        "correct_count": correct_count,
        "precision": precision,
        "occupied_4x4_cells": occupied,
        "coverage_fraction": coverage,
        "x_span_fraction": x_span,
        "y_span_fraction": y_span,
        "confidence": _confidence_summary(matches.confidence[selected]),
        "gate": gate,
    }


def cross_render_calibration(
    query: LoadedWebGLQueryArtifact,
    render: TerrainRenderBundle,
) -> dict[str, object]:
    """Check independent/shared raster parity before judging a method."""

    query_mask = np.asarray(query.terrain_mask, dtype=np.bool_)
    render_mask = np.asarray(render.terrain_mask, dtype=np.bool_)
    if query_mask.shape != render_mask.shape:
        raise ValueError("cross-render calibration masks have different shapes")
    union = query_mask | render_mask
    mask_iou = _fraction(int((query_mask & render_mask).sum()), int(union.sum())) if union.any() else 1.0
    query_depth = np.asarray(query.forward_depth_m, dtype=np.float64)
    render_depth = np.asarray(render.forward_depth_m, dtype=np.float64)
    shared = query_mask & render_mask & np.isfinite(query_depth) & np.isfinite(render_depth)
    shared &= (query_depth > 0.0) & (render_depth > 0.0)
    disagreement = np.abs(np.log(query_depth[shared]) - np.log(render_depth[shared]))
    p95 = float(np.percentile(disagreement, 95.0)) if disagreement.size else None
    return {
        "mask_iou": mask_iou,
        "shared_depth_count": int(disagreement.size),
        "p95_abs_log_depth": p95,
        "passed": bool(mask_iou >= 0.99 and p95 is not None and p95 <= 0.01),
    }


def identical_query_calibration(
    manifest: Sequence[MatchArtifactFile],
    files: Mapping[str, bytes],
    *,
    width_px: int,
    height_px: int,
) -> dict[str, int | float | bool]:
    """Validate SIFT on a query matched to the exact same query bytes."""

    if width_px < 1 or height_px < 1:
        raise ValueError("identical-query image dimensions must be positive")
    matches = load_frozen_match_artifact(manifest, files)
    selected = np.asarray(matches.selected, dtype=np.bool_)
    in_bounds = _in_bounds(matches.query_xy_px, width_px, height_px)
    in_bounds &= _in_bounds(matches.render_xy_px, width_px, height_px)
    error = np.linalg.norm(matches.query_xy_px - matches.render_xy_px, axis=1)
    within = selected & in_bounds & (error <= 1.0)
    selected_count = int(selected.sum())
    within_count = int(within.sum())
    occupied, coverage, _x_span, _y_span = _distribution(matches.query_xy_px[within], width_px, height_px)
    fraction = _fraction(within_count, selected_count)
    return {
        "raw_count": matches.count,
        "selected_count": selected_count,
        "within_1px_count": within_count,
        "within_1px_fraction": fraction,
        "occupied_4x4_cells": occupied,
        "coverage_fraction": coverage,
        "passed": bool(selected_count >= 12 and fraction >= 0.95 and occupied >= 4),
    }


def _in_bounds(xy: NDArray[np.float64], width: int, height: int) -> NDArray[np.bool_]:
    return (xy[:, 0] >= 0.0) & (xy[:, 0] <= width - 1) & (xy[:, 1] >= 0.0) & (xy[:, 1] <= height - 1)


def _distribution(xy: NDArray[np.float64], width: int, height: int) -> tuple[int, float, float, float]:
    if xy.shape[0] == 0:
        return 0, 0.0, 0.0, 0.0
    fractions = np.clip(xy / np.asarray((max(width - 1, 1), max(height - 1, 1))), 0.0, 1.0)
    cells = np.minimum((fractions * 4).astype(np.int64), 3)
    occupied = len(set(map(tuple, cells.tolist())))
    return occupied, occupied / 16.0, float(np.ptp(fractions[:, 0])), float(np.ptp(fractions[:, 1]))


def _fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _confidence_summary(confidence: NDArray[np.float64]) -> dict[str, float | None]:
    if confidence.size == 0:
        return dict.fromkeys(("minimum", "q25", "median", "mean", "q75", "maximum"))
    quantiles = np.quantile(confidence, (0.0, 0.25, 0.5, 0.75, 1.0))
    return {
        "minimum": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "mean": float(np.mean(confidence)),
        "q75": float(quantiles[3]),
        "maximum": float(quantiles[4]),
    }
