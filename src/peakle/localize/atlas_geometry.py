"""Shared truth-free geometry operations for frozen skyline-atlas candidates.

The helpers in this module deliberately accept no numeric pose reference.  An
atlas is first verified against its frozen content hash and estimator-only
contract; its candidates can then be rendered in the same cylindrical/tangent
crop geometry used when the atlas was built.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from peakle.domain.terrain import TerrainMap
from peakle.localize.gtrefine import crop_az_deg, dem_depth_image
from peakle.localize.skyline_atlas import ATLAS_ARCHIVE_SCHEMA
from peakle.localize.swissdem import Patch

SUPPORTED_ATLAS_CANDIDATE_POOL = "spatially_diverse_yaw_shortlist"

DepthRenderer = Callable[..., tuple[Any, Any, Any, Any, Any, Any]]


@dataclass(frozen=True, slots=True)
class RenderedCyltanCandidateDepth:
    """Exact DEM depth for one frozen atlas candidate on the cyltan grid.

    ``candidate_ray_depth`` is camera-ray range.  The underlying terrain
    marcher returns horizontal range, which is converted with the elevation of
    each comparison pixel.
    """

    atlas_candidate: dict[str, Any]
    candidate_ray_depth: NDArray[np.float64]
    max_range_m: float


def validate_frozen_cyltan_atlas(atlas_archive: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy an estimator-only frozen cyltan atlas.

    The returned record has a detached candidate list and candidate mappings.
    Numeric truth is neither accepted as an argument nor permitted by the
    archive contract.
    """

    atlas = dict(atlas_archive)
    if atlas.get("schema") != ATLAS_ARCHIVE_SCHEMA:
        raise ValueError(f"expected {ATLAS_ARCHIVE_SCHEMA} atlas archive")
    if atlas.get("numeric_evaluation_reference_used") is not False:
        raise ValueError("atlas archive is not estimator-only")
    expected_sha = atlas.get("archive_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("atlas archive SHA-256 is missing or malformed")
    basis = dict(atlas)
    basis.pop("archive_sha256", None)
    accepted_hashes = {_canonical_sha256(basis)}
    # Atlas v2 originally hashed the caller's FOV numeric type before storing it
    # as a float. Preserve validation for the rare integer-FOV archive while
    # still rejecting any other content change.
    query = basis.get("query_geometry")
    if isinstance(query, dict):
        fov = query.get("horizontal_fov_deg")
        if isinstance(fov, float) and fov.is_integer():
            integer_fov_basis = dict(basis)
            integer_query = dict(query)
            integer_query["horizontal_fov_deg"] = int(fov)
            integer_fov_basis["query_geometry"] = integer_query
            accepted_hashes.add(_canonical_sha256(integer_fov_basis))
    if expected_sha not in accepted_hashes:
        raise ValueError("atlas archive SHA-256 does not match its contents")

    raw_candidates = atlas.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("atlas candidate pool must be a non-empty list")
    if not all(isinstance(candidate, Mapping) for candidate in raw_candidates):
        raise ValueError("atlas candidates must be mappings")
    candidates = [deepcopy(dict(candidate)) for candidate in raw_candidates]
    query = atlas.get("query_geometry")
    if not isinstance(query, dict) or query.get("projection") != "cyltan":
        raise ValueError("frozen geometry scoring requires a cyltan atlas query")
    if atlas.get("candidate_pool") != SUPPORTED_ATLAS_CANDIDATE_POOL:
        raise ValueError("atlas candidate-pool policy is unsupported")
    if atlas.get("candidate_count") != len(candidates):
        raise ValueError("atlas candidate count does not match its stored pool")
    ids = [candidate.get("candidate_id") for candidate in candidates]
    ranks = [candidate.get("estimator_rank") for candidate in candidates]
    if len(set(ids)) != len(ids) or any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("atlas candidate IDs must be non-empty and unique")
    if ranks != list(range(1, len(candidates) + 1)):
        raise ValueError("atlas candidates must retain contiguous estimator ranks")
    atlas["candidates"] = candidates
    if atlas.get("selected_candidate_id") != candidates[0]["candidate_id"]:
        raise ValueError("atlas selected candidate does not match estimator rank one")
    return atlas


def render_cyltan_candidate_depth(
    terrain: TerrainMap,
    native_patch: Patch | None,
    candidate: Mapping[str, Any],
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float,
    *,
    subsample: int,
    depth_renderer: DepthRenderer | None = None,
) -> RenderedCyltanCandidateDepth:
    """Render one frozen atlas pose on its exact cyltan comparison grid."""

    if isinstance(width_px, bool) or not isinstance(width_px, int) or width_px < 1:
        raise ValueError("width_px must be a positive integer")
    if isinstance(height_px, bool) or not isinstance(height_px, int) or height_px < 1:
        raise ValueError("height_px must be a positive integer")
    fov = _finite_float(horizontal_fov_deg, "horizontal FOV")
    if not 1.0 < fov < 179.0:
        raise ValueError("horizontal FOV must be between 1 and 179 degrees")
    if isinstance(subsample, bool) or not isinstance(subsample, int) or subsample < 1:
        raise ValueError("subsample must be a positive integer")

    pose = candidate.get("pose")
    if not isinstance(pose, Mapping):
        raise ValueError("atlas candidate pose is missing")
    position = pose.get("position")
    if not isinstance(position, Mapping):
        raise ValueError("atlas candidate position is missing")
    east = _finite_float(position.get("east_m"), "candidate east")
    north = _finite_float(position.get("north_m"), "candidate north")
    up = _finite_float(position.get("up_m"), "candidate up")
    yaw = _finite_float(pose.get("yaw_deg"), "candidate yaw")
    roll = _finite_float(pose.get("roll_deg"), "candidate roll nuisance")
    shift = _finite_float(candidate.get("vertical_shift_px"), "candidate vertical shift")

    azimuths = crop_az_deg(width_px, fov, yaw)
    renderer = depth_renderer or dem_depth_image
    depth, _hit, elevation_pixels, _elevation_grid, distances, _rows = renderer(
        terrain,
        up,
        azimuths,
        width_px,
        height_px,
        fov,
        shift,
        east,
        north,
        roll,
        sub=subsample,
        patch=native_patch,
    )
    depths = np.asarray(depth, dtype=np.float64)
    elevations = np.asarray(elevation_pixels, dtype=np.float64)
    ranges = np.asarray(distances, dtype=np.float64)
    if depths.shape != elevations.shape:
        raise ValueError("candidate render depth and elevation grids differ")
    if ranges.size == 0 or not np.all(np.isfinite(ranges)):
        raise ValueError("candidate render returned no finite distance samples")
    cosine = np.cos(elevations)
    ray_depth = np.divide(
        depths,
        cosine,
        out=np.full_like(depths, np.nan, dtype=np.float64),
        where=np.isfinite(depths) & np.isfinite(cosine) & (cosine > 1e-6),
    )
    return RenderedCyltanCandidateDepth(dict(candidate), ray_depth, float(ranges[-1]))


def terrain_diagonal_range_m(terrain: TerrainMap) -> float:
    """Return a stable comparison cap while render limits vary by camera."""

    x_m = np.asarray(terrain.x_m, dtype=np.float64)
    y_m = np.asarray(terrain.y_m, dtype=np.float64)
    if x_m.ndim != 1 or y_m.ndim != 1 or x_m.size < 2 or y_m.size < 2:
        raise ValueError("terrain axes must contain at least two coordinates")
    result = float(np.hypot(x_m[-1] - x_m[0], y_m[-1] - y_m[0]))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("terrain comparison range must be positive and finite")
    return result


def terrain_surface_identity(terrain: TerrainMap, native_patch: Patch | None) -> dict[str, Any]:
    """Hash the exact elevation arrays and local frame used for candidate renders."""

    regional = {
        "x_m_sha256": _array_sha256(terrain.x_m, "peakle_terrain_x_v1"),
        "y_m_sha256": _array_sha256(terrain.y_m, "peakle_terrain_y_v1"),
        "elevation_m_sha256": _array_sha256(terrain.elevation_m, "peakle_terrain_elevation_v1"),
        "shape": list(np.asarray(terrain.elevation_m).shape),
    }
    patch_record = None
    if native_patch is not None:
        patch_record = {
            "x_m_sha256": _array_sha256(native_patch.x_m, "peakle_native_patch_x_v1"),
            "y_m_sha256": _array_sha256(native_patch.y_m, "peakle_native_patch_y_v1"),
            "elevation_m_sha256": _array_sha256(native_patch.elevation_m, "peakle_native_patch_elevation_v1"),
            "shape": list(np.asarray(native_patch.elevation_m).shape),
        }
    basis = {
        "coordinate_frame_origin": terrain.spec.origin.model_dump(mode="json"),
        "regional": regional,
        "native_patch": patch_record,
    }
    return {**basis, "aggregate_sha256": _canonical_sha256(basis)}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


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
