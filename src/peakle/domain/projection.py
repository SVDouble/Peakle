"""Shared camera projection math.

The project has two image geometries in active use:

- ``pinhole`` for synthetic placed cameras and standard perspective renders.
- ``cyltan`` for GeoPose3K crops: columns are linear in azimuth and rows are linear in
  tan(elevation), with focal length ``width / horizontal_fov_rad``.

``cyl`` is kept as a legacy solver/test geometry where rows are linear in elevation. New code
should use ``pinhole`` or ``cyltan``.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

ProjectionName = Literal["pinhole", "cyltan", "cyl"]
ImageProjectionName = Literal["pinhole", "cyltan"]


def focal_length_px(width_px: int, horizontal_fov_deg: float, projection: ProjectionName) -> float:
    """Returns the focal/image scale for a projection."""

    if projection not in {"pinhole", "cyltan", "cyl"}:
        msg = f"unsupported projection: {projection}"
        raise ValueError(msg)
    hfov_rad = math.radians(horizontal_fov_deg)
    if projection in {"cyltan", "cyl"}:
        return width_px / hfov_rad
    return width_px / (2.0 * math.tan(hfov_rad / 2.0))


def vertical_fov_deg(width_px: int, height_px: int, horizontal_fov_deg: float, projection: ProjectionName) -> float:
    """Returns the perspective-camera vertical FOV equivalent."""

    return math.degrees(2.0 * math.atan(height_px / (2.0 * focal_length_px(width_px, horizontal_fov_deg, projection))))


def azimuths_deg(width_px: int, horizontal_fov_deg: float, yaw_deg: float, projection: ProjectionName) -> np.ndarray:
    """Returns the world azimuth sampled by each image column."""

    cols = np.arange(width_px)
    center_x = (width_px - 1) / 2.0
    if projection == "pinhole":
        f = focal_length_px(width_px, horizontal_fov_deg, projection)
        return yaw_deg + np.degrees(np.arctan((cols - center_x) / f))
    return yaw_deg + np.degrees((cols - center_x) * (math.radians(horizontal_fov_deg) / width_px))


def rows_from_elevation_rad(
    elevation_rad: np.ndarray,
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float,
    projection: ProjectionName,
    *,
    pitch_deg: float = 0.0,
    vertical_shift_px: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Projects terrain elevation angles to image rows."""

    center_y = (height_px - 1) / 2.0
    f = focal_length_px(width_px, horizontal_fov_deg, projection)
    pitch_rad = math.radians(pitch_deg)
    if projection == "cyl":
        projected = f * (elevation_rad - pitch_rad)
    elif projection == "cyltan":
        projected = f * (np.tan(elevation_rad) - math.tan(pitch_rad))
    else:
        projected = f * np.tan(elevation_rad - pitch_rad)
    return center_y + vertical_shift_px - projected


def elevation_rad_from_rows(
    rows_px: np.ndarray,
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float,
    projection: ProjectionName,
    *,
    pitch_deg: float = 0.0,
    vertical_shift_px: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Inverts ``rows_from_elevation_rad`` back to terrain elevation angles."""

    center_y = (height_px - 1) / 2.0
    f = focal_length_px(width_px, horizontal_fov_deg, projection)
    normalized = (center_y + vertical_shift_px - rows_px.astype(float)) / f
    pitch_rad = math.radians(pitch_deg)
    if projection == "cyl":
        return normalized + pitch_rad
    if projection == "cyltan":
        return np.arctan(normalized + math.tan(pitch_rad))
    return np.arctan(normalized) + pitch_rad


def tangent_elevation_from_rows(
    rows_px: np.ndarray,
    width_px: int,
    height_px: int,
    horizontal_fov_deg: float,
    projection: ProjectionName,
    *,
    pitch_deg: float = 0.0,
    vertical_shift_px: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Returns ``tan(elevation)`` represented by image rows."""

    return np.tan(
        elevation_rad_from_rows(
            rows_px,
            width_px,
            height_px,
            horizontal_fov_deg,
            projection,
            pitch_deg=pitch_deg,
            vertical_shift_px=vertical_shift_px,
        )
    )


def pitch_deg_from_vertical_shift_px(
    width_px: int,
    horizontal_fov_deg: float,
    projection: ProjectionName,
    shift_px: float,
) -> float:
    """Returns the pitch represented by shifting a pitch-0 skyline down by ``shift_px`` rows."""

    f = focal_length_px(width_px, horizontal_fov_deg, projection)
    if projection == "cyl":
        return math.degrees(shift_px / f)
    return math.degrees(math.atan(shift_px / f))


def vertical_shift_px_from_pitch_deg(
    width_px: int,
    horizontal_fov_deg: float,
    projection: ProjectionName,
    pitch_deg: float,
) -> float:
    """Returns the vertical image shift induced by a pitch angle at the horizon."""

    f = focal_length_px(width_px, horizontal_fov_deg, projection)
    if projection == "cyl":
        return f * math.radians(pitch_deg)
    return f * math.tan(math.radians(pitch_deg))
