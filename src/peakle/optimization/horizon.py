"""Prior-free localization by skyline-horizon correlation.

Recovering a camera pose with *no position prior* over a large map is a global
search. Scanning position x yaw x pitch with a renderer is far too slow, so we
exploit a structural shortcut: yaw is a rotation about the vertical axis, so the
observed outline's (azimuth-offset, elevation-angle) descriptor is
yaw-invariant. For each candidate position we compute the terrain's full 360°
horizon (max elevation angle per azimuth bin) once, then the best yaw is a single
1D circular match of the observed descriptor against that horizon. This collapses
a position x yaw grid into position x one correlation, making 50 km x 50 km
no-prior search tractable. The best candidates seed a precise local refine.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.terrain import TerrainMap
from peakle.optimization.objective import dense_terrain_points
from peakle.rendering.pinhole import camera_axes


def horizon_seeds(
    terrain: TerrainMap,
    observed_profile: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    *,
    pitch_deg: float,
    eye_height_m: float,
    grid_east: int = 36,
    grid_north: int = 36,
    n_bins: int = 360,
    max_columns: int = 140,
    top_k: int = 20,
) -> list[CameraExtrinsics]:
    """Returns the best `top_k` pose seeds by horizon correlation over the map.

    Args:
        terrain: Terrain to search.
        observed_profile: Observed skyline y per column.
        intrinsics: Camera intrinsics matching `observed_profile`'s width.
        pitch_deg: Assumed pitch (from the orientation prior).
        eye_height_m: Assumed camera height above ground.
        grid_east: Candidate positions along east.
        grid_north: Candidate positions along north.
        n_bins: Azimuth bins for the 360 deg horizon (yaw resolution).
        max_columns: Subsample of observed columns used for matching.
        top_k: Number of seeds to return.

    Returns:
        Candidate extrinsics ordered best-first.
    """

    points = dense_terrain_points(terrain, 1)
    offset_count, elev_obs = _observed_descriptor(observed_profile, intrinsics, pitch_deg, n_bins, max_columns)
    bin_width_deg = 360.0 / n_bins
    shifts = np.arange(n_bins)
    shift_index = (shifts[:, None] + offset_count[None, :]) % n_bins  # (n_bins, M)

    easts = np.linspace(float(terrain.x_m[0]), float(terrain.x_m[-1]), grid_east)
    norths = np.linspace(float(terrain.y_m[0]), float(terrain.y_m[-1]), grid_north)
    candidates: list[tuple[float, float, float, float, float]] = []
    for east in easts:
        for north in norths:
            up = terrain.elevation_at(float(east), float(north)) + eye_height_m
            horizon = _terrain_horizon(points, (east, north, up), n_bins)
            errors = np.mean((horizon[shift_index] - elev_obs[None, :]) ** 2, axis=1)
            best = int(np.argmin(errors))
            yaw = _wrap180(-180.0 + (best + 0.5) * bin_width_deg)
            candidates.append((float(errors[best]), float(east), float(north), float(up), yaw))

    candidates.sort(key=lambda item: item[0])
    return [
        CameraExtrinsics(
            position=LocalPoint(east_m=east, north_m=north, up_m=up),
            yaw_deg=yaw,
            pitch_deg=pitch_deg,
            roll_deg=0.0,
        )
        for _score, east, north, up, yaw in candidates[:top_k]
    ]


def _observed_descriptor(
    observed_profile: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    pitch_deg: float,
    n_bins: int,
    max_columns: int,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Converts the observed outline to yaw-invariant (azimuth-offset, elevation)."""

    columns = np.arange(observed_profile.size)
    finite = np.isfinite(observed_profile)
    columns = columns[finite]
    rows = observed_profile[finite]
    if columns.size > max_columns:
        keep = np.linspace(0, columns.size - 1, max_columns).round().astype(int)
        columns = columns[keep]
        rows = rows[keep]

    origin = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=0.0), yaw_deg=0.0, pitch_deg=pitch_deg, roll_deg=0.0
    )
    right, down, forward = camera_axes(origin)
    x = (columns - intrinsics.principal_x_px) / intrinsics.focal_length_px
    y = (rows - intrinsics.principal_y_px) / intrinsics.focal_length_px
    rays = x[:, None] * right + y[:, None] * down + forward
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    azimuth_deg = np.degrees(np.arctan2(rays[:, 0], rays[:, 1]))
    elevation_deg = np.degrees(np.arcsin(np.clip(rays[:, 2], -1.0, 1.0)))
    offset_count = np.round(azimuth_deg / (360.0 / n_bins)).astype(np.int64)
    return offset_count, elevation_deg


def _terrain_horizon(
    points: NDArray[np.float64], camera: tuple[float, float, float], n_bins: int
) -> NDArray[np.float64]:
    """Computes the max terrain elevation angle per azimuth bin (gaps filled)."""

    delta_east = points[:, 0] - camera[0]
    delta_north = points[:, 1] - camera[1]
    delta_up = points[:, 2] - camera[2]
    azimuth_deg = np.degrees(np.arctan2(delta_east, delta_north))
    horizontal = np.hypot(delta_east, delta_north)
    elevation_deg = np.degrees(np.arctan2(delta_up, np.maximum(horizontal, 1.0)))
    bins = (np.floor((azimuth_deg + 180.0) / (360.0 / n_bins)).astype(np.int64)) % n_bins
    # Init to -inf (not nan): np.maximum.at against nan would stay nan everywhere.
    horizon = np.full(n_bins, -np.inf, dtype=np.float64)
    np.maximum.at(horizon, bins, elevation_deg)
    filled = np.isfinite(horizon)
    if filled.sum() >= 2:
        index = np.arange(n_bins)
        horizon = np.interp(index, index[filled], horizon[filled])
    else:
        horizon = np.nan_to_num(horizon, nan=0.0)
    return horizon


def _wrap180(value: float) -> float:
    return ((value + 180.0) % 360.0) - 180.0
