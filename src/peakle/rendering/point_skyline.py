"""Projection-aware skyline rendering from a terrain point cloud.

The mesh rasterizer is intentionally a pinhole renderer.  GeoPose3K crops use a
different image geometry (``cyltan``): columns are linear in azimuth and rows
are linear in tangent elevation.  Full-pose optimization only needs the upper
terrain silhouette, so projecting a dense terrain point cloud directly keeps
that hot path cheap without pretending the crop is a perspective image.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.projection import focal_length_px
from peakle.rendering.skyline import interpolate_profile


def cyltan_point_skyline(
    points: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    horizontal_fov_deg: float,
    *,
    near_clip_m: float = 1.0,
) -> NDArray[np.float64]:
    """Return a GeoPose cylindrical/tangent skyline from local terrain points.

    A point at local offset ``(east, north, up)`` maps to

    ``u = cx + f * wrap(atan2(east, north) - yaw)``

    ``v = cy - f * (up / hypot(east, north) - tan(pitch))``

    where ``f = image_width / horizontal_fov_radians``.  This is the same
    geometry implemented by :mod:`peakle.domain.projection`, expressed here in
    vector form so a candidate full pose can be scored without a ray march.

    Roll is deliberately unsupported: the full-pose solver is 5-DOF and always
    produces roll-zero, matching GeoPose's rectified cylindrical crops.
    """

    width = intrinsics.width_px
    height = intrinsics.height_px
    u_px, v_px, _horizontal_m, valid = project_cyltan_points(
        points,
        intrinsics,
        extrinsics,
        horizontal_fov_deg,
        near_clip_m=near_clip_m,
    )

    finite = valid & np.isfinite(u_px) & np.isfinite(v_px)
    columns = np.floor(u_px[finite]).astype(np.int64)
    inside = (columns >= 0) & (columns < width)
    profile = np.full(width, np.inf, dtype=np.float64)
    if np.any(inside):
        np.minimum.at(profile, columns[inside], v_px[finite][inside])
    profile[~np.isfinite(profile)] = np.nan
    profile = interpolate_profile(profile, fallback=float(height) * 0.62)
    return np.clip(profile, 0.0, float(height - 1))


def project_cyltan_points(
    points: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    horizontal_fov_deg: float,
    *,
    near_clip_m: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Project local points into GeoPose cylindrical/tangent image coordinates."""

    if points.ndim != 2 or points.shape[1] != 3:
        msg = f"points must have shape (N, 3), got {points.shape}"
        raise ValueError(msg)
    if abs(extrinsics.roll_deg) > 1e-9:
        msg = "cyltan point projection requires a roll-rectified (roll=0) camera"
        raise ValueError(msg)

    width = intrinsics.width_px
    height = intrinsics.height_px
    scale_px = focal_length_px(width, horizontal_fov_deg, "cyltan")
    position = np.asarray(extrinsics.position.as_tuple(), dtype=np.float64)
    vectors = np.asarray(points, dtype=np.float64) - position
    horizontal_m = np.hypot(vectors[:, 0], vectors[:, 1])
    valid = horizontal_m > near_clip_m

    point_azimuth = np.arctan2(vectors[:, 0], vectors[:, 1])
    relative_azimuth = point_azimuth - math.radians(extrinsics.yaw_deg)
    # The wrapped angular delta is essential near north: a 359 degree point is
    # one degree left of a zero-degree camera, not 359 degrees away.
    relative_azimuth = np.arctan2(np.sin(relative_azimuth), np.cos(relative_azimuth))

    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    u_px = center_x + scale_px * relative_azimuth
    v_px = np.full(points.shape[0], np.nan, dtype=np.float64)
    v_px[valid] = center_y - scale_px * (
        vectors[valid, 2] / horizontal_m[valid] - math.tan(math.radians(extrinsics.pitch_deg))
    )
    return u_px, v_px, horizontal_m, valid
