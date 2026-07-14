"""On-demand raster layers for any pose of a view.

These PNGs are debug views derived from the persisted pose/camera/terrain state.
They are intentionally not stored inside solves: the minimal persistent artifact
is the solve trace and resolved prior, while masks/depth can be reconstructed.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.projection import vertical_shift_px_from_pitch_deg
from peakle.localize.gtrefine import dem_depth_image, dem_skyline
from peakle.localize.typed_outlines import extract_typed_outlines
from peakle.scene.scene import Scene, View

POSE_LAYER_NAMES = frozenset({"sky", "occ", "rib", "cou", "depth"})
POSE_LAYER_COLORS = {
    "sky": (255, 140, 66),
    "occ": (255, 82, 82),
    "rib": (255, 204, 64),
    "cou": (215, 104, 255),
}


def render_pose_layer(scene: Scene, view: View, extrinsics: CameraExtrinsics, layer: str) -> bytes:
    """Render one transparent overlay layer for ``extrinsics`` in ``view`` coordinates."""

    if layer not in POSE_LAYER_NAMES:
        msg = f"unknown pose layer {layer!r}"
        raise ValueError(msg)
    width = view.image_camera.width_px
    height = view.image_camera.height_px
    depth, sky_rows = _pose_depth_and_skyline(scene, view, extrinsics)
    if layer == "depth":
        return depth_png(depth, width, height)
    if layer == "sky":
        return mask_png(rows_mask(sky_rows, width, height), POSE_LAYER_COLORS[layer], width, height)
    typed = extract_typed_outlines(depth, min_px=12)
    masks = {"occ": typed.occlusion, "rib": typed.rib, "cou": typed.couloir}
    return mask_png(masks[layer], POSE_LAYER_COLORS[layer], width, height)


def mask_png(mask: np.ndarray, color: tuple[int, int, int], width: int, height: int) -> bytes:
    """Encode a boolean mask as a transparent RGBA PNG."""

    if mask.shape != (height, width):
        resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((width, height), Image.Resampling.NEAREST)
        mask = np.asarray(resized) > 0
    rgba = np.zeros((height, width, 4), np.uint8)
    rgba[mask] = (*color, 255)
    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:, 1:] |= mask[:, :-1]
    rgba[grown & ~mask] = (*color, 160)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def depth_png(depth: np.ndarray, width: int, height: int) -> bytes:
    """Encode visible-terrain depth as a transparent pseudo-colored PNG."""

    d = depth.astype(float)
    finite = np.isfinite(d) & (d > 0)
    logd = np.log(np.where(finite, d, np.nan))
    lo, hi = (np.nanpercentile(logd, 2), np.nanpercentile(logd, 98)) if finite.any() else (0, 1)
    t = np.clip((logd - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba = np.zeros((*d.shape, 4), np.uint8)
    rgba[..., 0] = np.nan_to_num(40 + 30 * (1 - t), nan=0)
    rgba[..., 1] = np.nan_to_num(90 + 130 * (1 - t), nan=0)
    rgba[..., 2] = np.nan_to_num(140 + 110 * (1 - t), nan=0)
    rgba[..., 3] = np.where(finite, 200, 0)
    img = Image.fromarray(rgba, "RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def rows_mask(rows: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert a skyline row profile to a boolean pixel mask."""

    mask = np.zeros((height, width), bool)
    ok = np.isfinite(rows) & (rows >= 0) & (rows < height)
    mask[np.clip(rows[ok].round().astype(int), 0, height - 1), np.arange(width)[ok]] = True
    return mask


def _pose_depth_and_skyline(scene: Scene, view: View, extrinsics: CameraExtrinsics) -> tuple[np.ndarray, np.ndarray]:
    if view.image_camera.projection == "cyltan":
        return _cyltan_depth_and_skyline(scene, view, extrinsics)
    depth = scene.renderer.depth_image(scene.terrain, view.intrinsics, extrinsics, stride=2)
    return depth, _skyline_rows_from_depth(depth)


def _cyltan_depth_and_skyline(scene: Scene, view: View, extrinsics: CameraExtrinsics) -> tuple[np.ndarray, np.ndarray]:
    camera = view.image_camera
    width = camera.width_px
    height = camera.height_px
    fov = camera.horizontal_fov_deg
    dv = vertical_shift_px_from_pitch_deg(width, fov, camera.projection, extrinsics.pitch_deg)
    azimuths = camera.azimuths_deg(extrinsics.yaw_deg)
    depth, *_ = dem_depth_image(
        scene.terrain,
        extrinsics.position.up_m,
        azimuths,
        width,
        height,
        fov,
        dv,
        extrinsics.position.east_m,
        extrinsics.position.north_m,
        0.0,
        sub=2,
    )
    rows = dem_skyline(
        scene.terrain,
        extrinsics.position.up_m,
        azimuths,
        width,
        height,
        fov,
        extrinsics.position.east_m,
        extrinsics.position.north_m,
    )
    return depth, rows + dv


def _skyline_rows_from_depth(depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    rows = np.full(width, np.nan, dtype=np.float64)
    finite = np.isfinite(depth) & (depth > 0)
    has = finite.any(axis=0)
    if np.any(has):
        rows[has] = np.argmax(finite[:, has], axis=0)
    return np.clip(rows, 0.0, float(height - 1))
