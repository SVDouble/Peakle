"""Curvature-based terrain ridge/valley extraction from a DEM, projection to the image
with a visibility test, and a pose optimiser that aligns projected ridges to image edges.

Validated against hand annotations: occlusion-only extraction (`ridge_layers`) misses
the convex counterforts, but **directional-crest (curvature) extraction** on a plain
30 m DEM recovers them (recall@10 0.40 / @25 0.67 vs annotations, beating both
occlusion-DEM 0.23 and image-only 0.37). The sign gives the line *type* the optimiser
needs: convex crests = **ridges** (the annotator's red), concave troughs = **folds**
(blue). On the image side, multi-scale persistence is the best-matching edge evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.terrain import TerrainMap
from peakle.rendering.pinhole import project_points
from peakle.rendering.rasterizer import SyntheticRenderer

_CREST_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))


def ridge_valley_masks(
    terrain: TerrainMap, prominence_m: float = 2.5, smooth: float = 1.0
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Convex-crest (ridge) and concave-trough (fold) masks over the DEM grid.

    A cell is a ridge if it stands at least `prominence_m` above *both* neighbours
    along some direction (a 1-D crest); a fold is the same with both neighbours
    higher (a 1-D trough). This is the divide/spur network — including counterforts.
    """

    elevation = gaussian_filter(terrain.elevation_m, smooth)
    ridge = np.zeros_like(elevation)
    valley = np.zeros_like(elevation)
    for di, dj in _CREST_DIRS:
        plus = np.roll(np.roll(elevation, -di, 0), -dj, 1)
        minus = np.roll(np.roll(elevation, di, 0), dj, 1)
        ridge = np.maximum(ridge, np.minimum(elevation - plus, elevation - minus))
        valley = np.maximum(valley, np.minimum(plus - elevation, minus - elevation))
    ridge_mask, valley_mask = ridge > prominence_m, valley > prominence_m
    for mask in (ridge_mask, valley_mask):  # np.roll wraps -> drop the 1px DEM border
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
    return ridge_mask, valley_mask


def _trace_grid(mask: NDArray[np.bool_], min_len: int = 6) -> list[list[tuple[int, int]]]:
    """Skeletonise a grid mask and walk connected branches into (i, j) polylines."""

    skel = skeletonize(mask)
    points = set(zip(*np.where(skel), strict=False))

    def neighbours(p: tuple[int, int]) -> list[tuple[int, int]]:
        y, x = p
        return [(y+a, x+b) for a in (-1, 0, 1) for b in (-1, 0, 1)
                if (a, b) != (0, 0) and (y+a, x+b) in points]

    seen: set[tuple[int, int]] = set()
    polylines: list[list[tuple[int, int]]] = []
    for start in [p for p in points if len(neighbours(p)) == 1] + list(points):
        if start in seen:
            continue
        nxt = [q for q in neighbours(start) if q not in seen]
        if not nxt:
            continue
        path = [start, nxt[0]]
        seen.update(path)
        cur = nxt[0]
        while True:
            cand = [q for q in neighbours(cur) if q not in seen]
            if not cand:
                break
            cur = cand[0]
            seen.add(cur)
            path.append(cur)
        if len(path) >= min_len:
            polylines.append(path)
    return polylines


@dataclass(frozen=True)
class ProjectedRidge:
    """A visible terrain ridge/fold projected into the image."""

    polyline: list[tuple[int, int]]  # (u, v) pixel coordinates
    kind: str  # "ridge" (convex) or "fold" (concave)


def _grid_to_world(poly_ij: list[tuple[int, int]], terrain: TerrainMap) -> NDArray[np.float64]:
    return np.array([(terrain.x_m[j], terrain.y_m[i], terrain.elevation_m[i, j]) for i, j in poly_ij], dtype=np.float64)


def _project_visible(
    poly_world: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    depth_buffer: NDArray[np.float64],
) -> list[list[tuple[int, int]]]:
    """Projects a world polyline, splitting it at occluded / out-of-frame vertices."""

    u, v, depth, valid = project_points(poly_world, intrinsics, extrinsics)
    height, width = depth_buffer.shape
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for k in range(len(poly_world)):
        finite = bool(valid[k]) and np.isfinite(u[k]) and np.isfinite(v[k]) and depth[k] > 0
        if finite:
            uu, vv = int(u[k]), int(v[k])
            if 0 <= uu < width and 0 <= vv < height:
                surface = depth_buffer[vv, uu]
                if np.isfinite(surface) and depth[k] <= surface * 1.03:  # not occluded by nearer terrain
                    current.append((uu, vv))
                    continue
        if len(current) > 1:
            segments.append(current)
        current = []
    if len(current) > 1:
        segments.append(current)
    return segments


def project_terrain_ridges(
    terrain: TerrainMap,
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    renderer: SyntheticRenderer | None = None,
    prominence_m: float = 2.5,
    min_len: int = 6,
) -> list[ProjectedRidge]:
    """Extracts DEM ridges/folds and projects the visible parts into the image."""

    renderer = renderer or SyntheticRenderer()
    depth_buffer = renderer.depth_image(terrain, intrinsics, extrinsics, stride=1)
    ridge_mask, valley_mask = ridge_valley_masks(terrain, prominence_m)
    out: list[ProjectedRidge] = []
    for mask, kind in ((ridge_mask, "ridge"), (valley_mask, "fold")):
        for poly_ij in _trace_grid(mask, min_len=min_len):
            for seg in _project_visible(_grid_to_world(poly_ij, terrain), intrinsics, extrinsics, depth_buffer):
                out.append(ProjectedRidge(seg, kind))
    return out


def align_view(
    terrain: TerrainMap,
    edge_evidence: NDArray[np.float64],
    position: LocalPoint,
    width: int,
    height: int,
    observed_skyline: NDArray[np.float64] | None = None,
    yaw_step: int = 10,
    fov_choices: tuple[float, ...] = (55.0, 65.0, 75.0),
    pitch_choices: tuple[float, ...] = (-12.0, -6.0, 0.0),
    prominence_m: float = 3.0,
    renderer: SyntheticRenderer | None = None,
) -> tuple[CameraIntrinsics, CameraExtrinsics, float]:
    """Finds the camera pose, primarily by matching the **skyline** (the robust anchor).

    With `observed_skyline` (a per-column terrain-top row, e.g. from the SAM mask) the
    objective is the negative robust chamfer between it and the DEM's rendered skyline —
    far more discriminative than ridge evidence-support, which localised to a wrong pose.
    Without it, falls back to ridge evidence-support against `edge_evidence`. Coarse
    full-yaw sweep then refine; FOV is constrained to a plausible range. Returns
    `(intrinsics, extrinsics, score)` (score higher = better).
    """

    renderer = renderer or SyntheticRenderer()
    points = terrain.flattened_points(stride=1)
    ridge_mask, _ = ridge_valley_masks(terrain, prominence_m)
    ii, jj = np.where(ridge_mask)
    world = np.column_stack((terrain.x_m[jj], terrain.y_m[ii], terrain.elevation_m[ii, jj])).astype(np.float64)

    def score(yaw: float, fov: float, pitch: float) -> tuple[float, CameraIntrinsics, CameraExtrinsics]:
        intr = CameraIntrinsics.from_horizontal_fov(width, height, fov)
        ext = CameraExtrinsics(
            position=position, yaw_deg=float(((yaw + 180) % 360) - 180), pitch_deg=float(pitch), roll_deg=0.0
        )
        if observed_skyline is not None:
            predicted = renderer.fast_skyline(points, intr, ext)
            valid = np.isfinite(predicted) & np.isfinite(observed_skyline)
            if valid.sum() < width * 0.25:
                return -1e18, intr, ext
            resid = np.minimum(np.abs(predicted[valid] - observed_skyline[valid]), height * 0.12)  # robust
            return -float(np.mean(resid)), intr, ext
        u, v, depth, valid = project_points(world, intr, ext)
        buffer = renderer.depth_image(terrain, intr, ext, stride=1)
        inframe = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height) & (depth > 0)
        uu, vv, dd = u[inframe].astype(int), v[inframe].astype(int), depth[inframe]
        surface = buffer[vv, uu]
        visible = np.isfinite(surface) & (dd <= surface * 1.03)
        if visible.sum() < 20:
            return -1e18, intr, ext
        return float(np.mean(edge_evidence[vv[visible], uu[visible]])), intr, ext

    mid_fov, mid_pitch = fov_choices[len(fov_choices) // 2], pitch_choices[len(pitch_choices) // 2]
    coarse = sorted(((score(y, mid_fov, mid_pitch)[0], y) for y in range(-180, 180, yaw_step)), reverse=True)
    best_yaw = coarse[0][1]
    best = (-1e18, None, None)
    for yaw in range(best_yaw - yaw_step, best_yaw + yaw_step + 1, max(2, yaw_step // 3)):
        for fov in fov_choices:
            for pitch in pitch_choices:
                value, intr, ext = score(yaw, fov, pitch)
                if value > best[0]:
                    best = (value, intr, ext)
    return best[1], best[2], best[0]
