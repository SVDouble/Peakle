"""Ray-cast horizon renderer.

Replaces the point-splat ``fast_skyline`` (sparse projected points alias into combs) and the mesh
``skyline_profile`` (returns all-zero when the camera sits inside a wide DEM).  For each image
column we march along its azimuth, bilinearly sample the DEM, and keep the MAXIMUM terrain
elevation angle (with earth-curvature drop).  One value per column, smooth by construction.

Terrain duck-type: any object with ``x_m`` (east, ascending), ``y_m`` (north, ascending) and
``elevation_m[north_idx, east_idx]`` — both ``peakle.terrain`` maps and the Copernicus loader fit.

Rows returned by the ``skyline_*`` functions are UNCLIPPED floats (NaN where no terrain is hit);
callers decide how to treat off-image rows.  Clipping inside the renderer would fabricate skyline
mass at the image border and bias any chamfer computed from it.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics

EARTH_R = 6371000.0


def _elevation_angle_grid(
    terrain,
    az_rad: np.ndarray,
    cam_z: float,
    step: float,
    d_max: float | None,
    cam_e: float = 0.0,
    cam_n: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Max-terrain-elevation-angle machinery shared by both projections.

    Returns ``(el, ds)`` where ``el[i, j]`` is the terrain elevation angle (radians, -inf when the
    sample falls outside the DEM) seen along azimuth ``az_rad[i]`` at distance ``ds[j]`` from the
    camera at local-ENU ``(cam_e, cam_n, cam_z)``.
    """

    xm = np.asarray(terrain.x_m, float)
    ym = np.asarray(terrain.y_m, float)
    elev = np.asarray(terrain.elevation_m, float)
    x0, dx = xm[0], (xm[-1] - xm[0]) / (len(xm) - 1)
    y0, dy = ym[0], (ym[-1] - ym[0]) / (len(ym) - 1)
    if d_max is None:
        d_max = 0.95 * min(xm[-1] - xm[0], ym[-1] - ym[0]) / 2
    ds = np.arange(step, d_max, step)
    east = cam_e + ds[None, :] * np.sin(az_rad)[:, None]
    north = cam_n + ds[None, :] * np.cos(az_rad)[:, None]
    ci = (east - x0) / dx
    ri = (north - y0) / dy
    inb = (ci >= 0) & (ci <= len(xm) - 1) & (ri >= 0) & (ri <= len(ym) - 1)
    z = map_coordinates(elev, [ri.ravel(), ci.ravel()], order=1, mode="nearest")
    z = z.reshape(len(az_rad), len(ds))
    drop = ds[None, :] ** 2 / (2 * EARTH_R)
    el = np.arctan2(z - cam_z - drop, ds[None, :])
    return np.where(inb, el, -np.inf), ds


def horizon_elevation(
    terrain,
    az_rad: np.ndarray,
    cam_z: float,
    step: float = 30.0,
    d_max: float | None = None,
    cam_e: float = 0.0,
    cam_n: float = 0.0,
) -> np.ndarray:
    """Horizon elevation angle (radians) per azimuth; NaN where no DEM sample was hit."""

    el, _ = _elevation_angle_grid(terrain, az_rad, cam_z, step, d_max, cam_e, cam_n)
    el_max = el.max(axis=1)
    return np.where(np.isfinite(el_max), el_max, np.nan)


def skyline_pinhole(
    terrain,
    intr: CameraIntrinsics,
    ext: CameraExtrinsics,
    step: float = 30.0,
    d_max: float | None = None,
) -> np.ndarray:
    """Skyline row per column for a pinhole camera.  ``row = cy - f*tan(el - pitch)``."""

    f = float(intr.focal_length_px)
    cols = np.arange(intr.width_px)
    az = np.radians(ext.yaw_deg) + np.arctan((cols - float(intr.principal_x_px)) / f)
    el = horizon_elevation(terrain, az, float(ext.position.up_m), step, d_max)
    return float(intr.principal_y_px) - f * np.tan(el - np.radians(ext.pitch_deg))


def skyline_cyl(
    terrain,
    width_px: int,
    height_px: int,
    fov_deg: float,
    ext: CameraExtrinsics,
    step: float = 30.0,
    d_max: float | None = None,
) -> np.ndarray:
    """Skyline row per column for a cylindrical crop (GeoPose3K): columns are LINEAR in azimuth,
    rows LINEAR in elevation, so pitch is exactly a vertical shift of ``H/vfov`` rows per radian."""

    hfov = np.radians(fov_deg)
    vfov = hfov * height_px / width_px
    cols = np.arange(width_px)
    az = np.radians(ext.yaw_deg) + (cols - (width_px - 1) / 2.0) * (hfov / width_px)
    el = horizon_elevation(terrain, az, float(ext.position.up_m), step, d_max)
    return (height_px - 1) / 2.0 - (el - np.radians(ext.pitch_deg)) * (height_px / vfov)
