"""Contour-database pose seeds.

This is a geometry-first seeder: render many low-cost skyline snapshots from
viewpoints on rings around the dominant terrain massif, encode each skyline as a
normalized 1D contour signature, then rank snapshots against the observed image
contour. The best snapshots seed the full pose objective.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.domain.coordinates import LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.projection import ImageProjectionName, pitch_deg_from_vertical_shift_px
from peakle.domain.terrain import TerrainMap
from peakle.optimization.objective import dense_terrain_points
from peakle.rendering.point_skyline import cyltan_point_skyline
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.rendering.skyline import interpolate_profile

SIGNATURE_WIDTH = 96
ANGLE_COUNT = 72
MAX_SNAPSHOTS = 420
CONTOUR_DB_TARGET_COLUMNS = 160


@dataclass(frozen=True)
class ContourSeed:
    """A database match that can seed full pose refinement."""

    extrinsics: CameraExtrinsics
    score: float


@dataclass(frozen=True)
class _Signature:
    vector: NDArray[np.float64]
    relief_px: float


def contour_database_seeds(
    terrain: TerrainMap,
    observed_profile: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    prior: PosePrior,
    renderer: SyntheticRenderer,
    terrain_stride: int,
    max_seeds: int = 12,
    *,
    projection: ImageProjectionName = "pinhole",
    horizontal_fov_deg: float | None = None,
) -> list[ContourSeed]:
    """Builds and searches a compact contour snapshot database around the massif."""

    observed_signature = _profile_signature(observed_profile)
    eye_height_m = _prior_eye_height_m(terrain, prior)
    point_stride = max(terrain_stride, round(terrain.spec.grid_width / CONTOUR_DB_TARGET_COLUMNS))
    points = dense_terrain_points(terrain, 1, stride=point_stride)
    snapshots: list[ContourSeed] = []
    for target in _massif_targets(terrain):
        for radius_m in _ring_radii(terrain, target, prior):
            for angle_rad in np.linspace(0.0, 2.0 * math.pi, ANGLE_COUNT, endpoint=False):
                east_m = target.east_m + radius_m * math.sin(float(angle_rad))
                north_m = target.north_m + radius_m * math.cos(float(angle_rad))
                if not _inside_terrain(terrain, east_m, north_m):
                    continue
                up_m = terrain.elevation_at(east_m, north_m) + eye_height_m
                yaw_deg = math.degrees(math.atan2(target.east_m - east_m, target.north_m - north_m))
                base_extrinsics = CameraExtrinsics(
                    position=LocalPoint(east_m=east_m, north_m=north_m, up_m=up_m),
                    yaw_deg=_wrap180(yaw_deg),
                    pitch_deg=0.0,
                    roll_deg=0.0,
                )
                if projection == "cyltan":
                    hfov_deg = horizontal_fov_deg or intrinsics.horizontal_fov_deg()
                    profile = cyltan_point_skyline(points, intrinsics, base_extrinsics, hfov_deg)
                else:
                    profile = renderer.fast_skyline(points, intrinsics, base_extrinsics)
                signature = _profile_signature(profile)
                score = _signature_distance(observed_signature, signature)
                pitch_deg = _pitch_from_vertical_shift(
                    intrinsics,
                    _median_shift_px(observed_profile, profile),
                    projection=projection,
                    horizontal_fov_deg=horizontal_fov_deg,
                )
                snapshots.append(
                    ContourSeed(
                        extrinsics=base_extrinsics.model_copy(
                            update={"pitch_deg": float(np.clip(pitch_deg, -45.0, 45.0))}
                        ),
                        score=score,
                    )
                )
                if len(snapshots) >= MAX_SNAPSHOTS:
                    break
            if len(snapshots) >= MAX_SNAPSHOTS:
                break
        if len(snapshots) >= MAX_SNAPSHOTS:
            break
    snapshots.sort(key=lambda seed: seed.score)
    return _distinct_seeds(snapshots, max_seeds)


def _massif_targets(terrain: TerrainMap) -> list[LocalPoint]:
    elevation = terrain.elevation_m
    high = float(np.percentile(elevation, 97.0))
    mask = elevation >= high
    x_grid, y_grid = np.meshgrid(terrain.x_m, terrain.y_m)
    weights = np.maximum(elevation[mask] - high, 1.0)
    centroid = LocalPoint(
        east_m=float(np.average(x_grid[mask], weights=weights)),
        north_m=float(np.average(y_grid[mask], weights=weights)),
        up_m=float(np.average(elevation[mask], weights=weights)),
    )
    max_row, max_col = np.unravel_index(int(np.argmax(elevation)), elevation.shape)
    summit = terrain.point_at_index(int(max_row), int(max_col))
    targets = [summit]
    if math.hypot(centroid.east_m - summit.east_m, centroid.north_m - summit.north_m) > 0.04 * terrain.spec.width_m:
        targets.append(centroid)
    return targets


def _ring_radii(terrain: TerrainMap, target: LocalPoint, prior: PosePrior) -> list[float]:
    diagonal_m = math.hypot(float(terrain.x_m[-1] - terrain.x_m[0]), float(terrain.y_m[-1] - terrain.y_m[0]))
    radii = [0.12 * diagonal_m, 0.20 * diagonal_m, 0.30 * diagonal_m, 0.43 * diagonal_m, 0.57 * diagonal_m]
    prior_radius = math.hypot(prior.position.east_m - target.east_m, prior.position.north_m - target.north_m)
    if prior_radius > 250.0:
        radii.append(float(prior_radius))
    return sorted(set(round(radius, 3) for radius in radii))


def _inside_terrain(terrain: TerrainMap, east_m: float, north_m: float) -> bool:
    margin_m = 0.02 * min(float(terrain.x_m[-1] - terrain.x_m[0]), float(terrain.y_m[-1] - terrain.y_m[0]))
    return (
        float(terrain.x_m[0]) + margin_m <= east_m <= float(terrain.x_m[-1]) - margin_m
        and float(terrain.y_m[0]) + margin_m <= north_m <= float(terrain.y_m[-1]) - margin_m
    )


def _prior_eye_height_m(terrain: TerrainMap, prior: PosePrior) -> float:
    ground_m = terrain.elevation_at(prior.position.east_m, prior.position.north_m)
    return float(np.clip(prior.position.up_m - ground_m, 2.0, 1500.0))


def _profile_signature(profile: NDArray[np.float64]) -> _Signature:
    profile = np.asarray(profile, dtype=np.float64)
    finite = profile[np.isfinite(profile)]
    fallback = float(np.median(finite)) if finite.size else 0.0
    filled = interpolate_profile(profile, fallback=fallback)
    sampled = np.interp(
        np.linspace(0.0, filled.size - 1, SIGNATURE_WIDTH),
        np.arange(filled.size, dtype=np.float64),
        filled,
    )
    centered = sampled - float(np.median(sampled))
    relief_px = float(np.percentile(centered, 90.0) - np.percentile(centered, 10.0))
    scale = max(relief_px, float(np.std(centered)), 1.0)
    normalized = np.clip(centered / scale, -3.0, 3.0)
    slope = np.gradient(normalized)
    return _Signature(vector=np.concatenate((normalized, 0.45 * slope)), relief_px=relief_px)


def _signature_distance(observed: _Signature, candidate: _Signature) -> float:
    shape = float(np.mean(np.minimum(np.abs(observed.vector - candidate.vector), 3.0)))
    depth = abs(math.log((observed.relief_px + 1.0) / (candidate.relief_px + 1.0)))
    return shape + 0.20 * depth


def _median_shift_px(observed: NDArray[np.float64], predicted: NDArray[np.float64]) -> float:
    valid = np.isfinite(observed) & np.isfinite(predicted)
    if np.count_nonzero(valid) < max(12, int(0.15 * observed.size)):
        return 0.0
    return float(np.median(observed[valid]) - np.median(predicted[valid]))


def _pitch_from_vertical_shift(
    intrinsics: CameraIntrinsics,
    shift_px: float,
    *,
    projection: ImageProjectionName = "pinhole",
    horizontal_fov_deg: float | None = None,
) -> float:
    if projection == "cyltan":
        hfov_deg = horizontal_fov_deg or intrinsics.horizontal_fov_deg()
        return pitch_deg_from_vertical_shift_px(intrinsics.width_px, hfov_deg, projection, shift_px)
    return CameraModel.from_intrinsics(intrinsics).pitch_deg_from_vertical_shift_px(shift_px)


def _distinct_seeds(seeds: list[ContourSeed], max_count: int) -> list[ContourSeed]:
    kept: list[ContourSeed] = []
    for seed in seeds:
        position = seed.extrinsics.position
        yaw = seed.extrinsics.yaw_deg
        if all(
            math.hypot(
                position.east_m - other.extrinsics.position.east_m,
                position.north_m - other.extrinsics.position.north_m,
            )
            > 600.0
            or abs(_wrap180(yaw - other.extrinsics.yaw_deg)) > 8.0
            for other in kept
        ):
            kept.append(seed)
            if len(kept) >= max_count:
                break
    return kept


def _wrap180(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0
