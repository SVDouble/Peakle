"""Pinhole camera projection utilities."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint

DEFAULT_NEAR_CLIP_M = 1.0


def camera_axes(
    extrinsics: CameraExtrinsics,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Computes camera right, down, and forward unit vectors.

    Args:
        extrinsics: Camera extrinsics.

    Returns:
        Tuple of `(right, down, forward)` unit vectors in world coordinates.
    """

    yaw = math.radians(extrinsics.yaw_deg)
    pitch = math.radians(extrinsics.pitch_deg)
    roll = math.radians(extrinsics.roll_deg)

    forward = np.array(
        [
            math.sin(yaw) * math.cos(pitch),
            math.cos(yaw) * math.cos(pitch),
            math.sin(pitch),
        ],
        dtype=np.float64,
    )
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0], dtype=np.float64)
    down = np.cross(forward, right)

    if abs(roll) > 1e-12:
        right, down = Rotation.from_rotvec(roll * forward).apply(np.stack((right, down)))

    return _unit(right), _unit(down), _unit(forward)


def project_points(
    points: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    near_clip_m: float = DEFAULT_NEAR_CLIP_M,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Projects local 3D points into image coordinates.

    Args:
        points: Array of shape `(N, 3)` with local `(east, north, up)` points.
        intrinsics: Camera intrinsics.
        extrinsics: Camera extrinsics.
        near_clip_m: Minimum positive forward depth.

    Returns:
        Tuple of arrays `(u_px, v_px, depth_m, valid)` with length `N`.
    """

    camera_points = camera_coordinates(points, extrinsics)
    return project_camera_points(camera_points, intrinsics, near_clip_m=near_clip_m)


def camera_coordinates(
    points: NDArray[np.float64],
    extrinsics: CameraExtrinsics,
) -> NDArray[np.float64]:
    """Transforms world ENU points into camera right/down/forward coordinates."""

    right, down, forward = camera_axes(extrinsics)
    position = np.array(extrinsics.position.as_tuple(), dtype=np.float64)
    vectors = points - position
    return vectors @ np.column_stack((right, down, forward))


def project_camera_points(
    camera_points: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    near_clip_m: float = DEFAULT_NEAR_CLIP_M,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Projects right/down/forward camera coordinates into image coordinates."""

    x_camera = camera_points[:, 0]
    y_camera = camera_points[:, 1]
    depth = camera_points[:, 2]
    valid = np.all(np.isfinite(camera_points), axis=1) & (depth >= near_clip_m)

    u_px = np.full(camera_points.shape[0], np.nan, dtype=np.float64)
    v_px = np.full(camera_points.shape[0], np.nan, dtype=np.float64)
    u_px[valid] = intrinsics.focal_length_px * (x_camera[valid] / depth[valid])
    u_px[valid] += intrinsics.principal_x_px
    v_px[valid] = intrinsics.focal_length_px * (y_camera[valid] / depth[valid])
    v_px[valid] += intrinsics.principal_y_px
    return u_px, v_px, depth, valid


def project_local_point(
    point: LocalPoint,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> tuple[float, float, float, bool]:
    """Projects a single local point.

    Args:
        point: Local point to project.
        intrinsics: Camera intrinsics.
        extrinsics: Camera extrinsics.

    Returns:
        Tuple `(u_px, v_px, depth_m, valid)`.
    """

    points = np.array([point.as_tuple()], dtype=np.float64)
    u_px, v_px, depth, valid = project_points(points, intrinsics, extrinsics)
    return float(u_px[0]), float(v_px[0]), float(depth[0]), bool(valid[0])


def look_at_camera(
    position: LocalPoint,
    target: LocalPoint,
    roll_deg: float = 0.0,
) -> CameraExtrinsics:
    """Creates camera extrinsics that look from `position` toward `target`.

    Args:
        position: Camera position.
        target: Target point.
        roll_deg: Camera roll in degrees.

    Returns:
        Camera extrinsics aimed at the target.
    """

    dx = target.east_m - position.east_m
    dy = target.north_m - position.north_m
    dz = target.up_m - position.up_m
    yaw_deg = math.degrees(math.atan2(dx, dy))
    horizontal = math.hypot(dx, dy)
    pitch_deg = math.degrees(math.atan2(dz, horizontal))
    return CameraExtrinsics(
        position=position,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
    )


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        msg = "camera axis has zero length"
        raise ValueError(msg)
    return vector / norm
