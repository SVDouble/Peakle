"""Matcher-ready metric terrain render bundles.

Each RGB matcher image is inseparable from the depth, world coordinates, and
camera that produced it.  Keeping those products in one immutable bundle makes
render-pixel lifting auditable and prevents a visually similar but geometrically
different frame from reaching PnP.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.terrain import TerrainMap
from peakle.rendering.orthophoto import AppearanceRaster
from peakle.rendering.pinhole import camera_axes
from peakle.rendering.rasterizer import HeightfieldGrid, HeightfieldLike, SyntheticRenderer

RenderModality = Literal["hillshade", "normal", "camera_normal", "relative_depth", "orthophoto"]
PIXEL_LIFTING_SCHEMA = "terrain_render_pixel_lifting_v1"
PIXEL_LIFTING_DEPTH_METHOD = "bilinear_finite_positive_inverse_depth_then_invert_v1"


@dataclass(frozen=True)
class TerrainRenderBundle:
    """RGB appearance plus metric products from one pinhole terrain view."""

    rgb: NDArray[np.uint8]
    forward_depth_m: NDArray[np.float64]
    world_xyz_m: NDArray[np.float64]
    world_normals: NDArray[np.float64]
    terrain_mask: NDArray[np.bool_]
    appearance_mask: NDArray[np.bool_]
    skyline_profile: NDArray[np.float64]
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    modality: RenderModality
    provenance: dict[str, Any]

    def __post_init__(self) -> None:
        height = self.intrinsics.height_px
        width = self.intrinsics.width_px
        expected = (height, width)
        if np.asarray(self.rgb).shape != (*expected, 3):
            raise ValueError(f"render RGB must have shape {(*expected, 3)}")
        if np.asarray(self.forward_depth_m).shape != expected:
            raise ValueError(f"render depth must have shape {expected}")
        if np.asarray(self.world_xyz_m).shape != (*expected, 3):
            raise ValueError(f"render world XYZ must have shape {(*expected, 3)}")
        if np.asarray(self.world_normals).shape != (*expected, 3):
            raise ValueError(f"render normals must have shape {(*expected, 3)}")
        if np.asarray(self.terrain_mask).shape != expected:
            raise ValueError(f"render mask must have shape {expected}")
        if np.asarray(self.appearance_mask).shape != expected:
            raise ValueError(f"render appearance mask must have shape {expected}")
        if np.asarray(self.skyline_profile).shape != (width,):
            raise ValueError(f"render skyline must have shape {(width,)}")
        terrain_mask = np.asarray(self.terrain_mask, dtype=np.bool_)
        appearance_mask = np.asarray(self.appearance_mask, dtype=np.bool_)
        depth = np.asarray(self.forward_depth_m, dtype=np.float64)
        if np.any(terrain_mask & (~np.isfinite(depth) | (depth <= 0.0))):
            raise ValueError("every terrain pixel must have finite positive forward depth")
        if np.any(appearance_mask & ~terrain_mask):
            raise ValueError("appearance-valid pixels must be a subset of rendered terrain")
        # Frozen dataclasses do not make ndarray contents immutable. Expose
        # read-only views so recorded render hashes cannot be invalidated via
        # the bundle itself after construction.
        for field_name, value in (
            ("rgb", self.rgb),
            ("forward_depth_m", self.forward_depth_m),
            ("world_xyz_m", self.world_xyz_m),
            ("world_normals", self.world_normals),
            ("terrain_mask", self.terrain_mask),
            ("appearance_mask", self.appearance_mask),
            ("skyline_profile", self.skyline_profile),
        ):
            readonly = np.asarray(value).view()
            readonly.flags.writeable = False
            object.__setattr__(self, field_name, readonly)


@dataclass(frozen=True)
class LiftedRenderPoints:
    """Per-match metric lifting result with explicit rejection diagnostics."""

    world_xyz_m: NDArray[np.float64]
    forward_depth_m: NDArray[np.float64]
    valid: NDArray[np.bool_]
    relative_depth_span: NDArray[np.float64]
    rejection_counts: dict[str, int]


class TerrainViewRenderer:
    """Build matcher images from the existing occlusion-correct mesh pass."""

    def __init__(self, rasterizer: SyntheticRenderer | None = None) -> None:
        self._rasterizer = rasterizer or SyntheticRenderer()
        self._normal_cache_key: str | None = None
        self._normal_cache_grid: NDArray[np.float64] | None = None
        self._native_normal_cache_key: str | None = None
        self._native_normal_cache_grid: NDArray[np.float64] | None = None

    def render(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        *,
        modality: RenderModality = "hillshade",
        appearance: AppearanceRaster | None = None,
        terrain_stride: int = 1,
        native_elevation_patch: HeightfieldLike | None = None,
        native_patch_stride: int = 8,
    ) -> TerrainRenderBundle:
        """Render one pinhole frame and all products needed for 3D lifting."""

        if terrain_stride < 1:
            raise ValueError("terrain_stride must be positive")
        if native_patch_stride < 1:
            raise ValueError("native patch stride must be positive")
        if modality == "orthophoto" and appearance is None:
            raise ValueError("orthophoto rendering requires an explicit georeferenced appearance raster")
        terrain_identity = terrain_fingerprint(terrain)
        native_surface = (
            HeightfieldGrid.from_like(native_elevation_patch) if native_elevation_patch is not None else None
        )
        native_identity = heightfield_fingerprint(native_surface) if native_surface is not None else None
        geometry = self._rasterizer.geometry(
            terrain,
            intrinsics,
            extrinsics,
            stride=terrain_stride,
            overlay=native_surface,
            overlay_stride=native_patch_stride,
        )
        depth = np.asarray(geometry.forward_depth_m, dtype=np.float64)
        mask = np.asarray(geometry.terrain_mask, dtype=np.bool_)
        native_overlay_mask = np.asarray(geometry.native_overlay_mask, dtype=np.bool_)
        world = unproject_pinhole_depth(depth, intrinsics, extrinsics)
        normal_grid = self._cached_normal_grid(terrain, terrain_identity["sha256"])
        normals = sample_terrain_normals(
            terrain,
            world[..., 0],
            world[..., 1],
            mask,
            normal_grid=normal_grid,
        )
        native_normal_pixels = 0
        if native_surface is not None and native_identity is not None and np.any(native_overlay_mask):
            sampled_native = native_surface.sampled(native_patch_stride)
            native_normal_grid = self._cached_native_normal_grid(
                sampled_native,
                f"{native_identity['sha256']}:{native_patch_stride}",
            )
            native_normals = sample_heightfield_normals(
                sampled_native,
                world[..., 0],
                world[..., 1],
                native_overlay_mask,
                normal_grid=native_normal_grid,
            )
            native_normal_valid = native_overlay_mask & (np.linalg.norm(native_normals, axis=2) > 0.5)
            normals[native_normal_valid] = native_normals[native_normal_valid]
            native_normal_pixels = int(native_normal_valid.sum())
        hillshade = _hillshade_rgb(terrain, normals, mask)
        appearance_provenance: dict[str, Any] | None = None
        appearance_coverage = 0.0
        appearance_mask = mask.copy()
        if modality == "hillshade":
            rgb = _with_sky(hillshade, mask, (174, 195, 215))
        elif modality in {"normal", "camera_normal"}:
            encoded_normals = normals if modality == "normal" else world_normals_to_camera(normals, extrinsics)
            normal_rgb = np.clip(np.rint((encoded_normals + 1.0) * 127.5), 0.0, 255.0).astype(np.uint8)
            rgb = _with_sky(normal_rgb, mask, (8, 11, 17))
        elif modality == "relative_depth":
            rgb = _with_sky(_relative_depth_rgb(depth, mask), mask, (8, 11, 17))
        else:
            if appearance is None:  # narrowed above; explicit for static type checkers
                raise RuntimeError("orthophoto appearance unexpectedly missing")
            sampled, available = appearance.sample_local(world[..., 0], world[..., 1])
            available &= mask
            appearance_mask = available
            appearance_coverage = float(available.sum() / max(1, mask.sum()))
            # A small geometric illumination term preserves terrain relief while
            # retaining the source orthophoto's actual chroma and texture.
            luminance = np.mean(hillshade.astype(np.float64), axis=2) / 255.0
            textured = np.clip(sampled.astype(np.float64) * (0.78 + 0.30 * luminance[..., None]), 0.0, 255.0)
            textured = textured.astype(np.uint8)
            textured[mask & ~available] = hillshade[mask & ~available]
            rgb = _with_sky(textured, mask, (112, 149, 181))
            appearance_provenance = appearance.provenance()

        render_content_sha256 = _render_content_sha256(
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=depth,
            world=world,
            normals=normals,
            terrain_mask=mask,
            appearance_mask=appearance_mask,
            skyline=np.asarray(geometry.skyline_profile, dtype=np.float64),
            native_overlay_mask=native_overlay_mask,
        )
        native_patch_provenance = _native_patch_provenance(
            native_surface,
            native_identity,
            native_patch_stride,
            visible_pixels=int(native_overlay_mask.sum()),
            normal_pixels=native_normal_pixels,
        )
        surface_identity_sha256 = _render_surface_identity_sha256(
            terrain_identity,
            terrain_stride,
            native_identity,
            native_patch_stride if native_surface is not None else None,
        )
        native_patch_used = bool(np.any(native_overlay_mask))
        provenance = {
            "schema": "terrain_render_bundle_v4",
            "render_content_sha256": render_content_sha256,
            "render_surface_identity_sha256": surface_identity_sha256,
            "projection": "pinhole",
            "modality": modality,
            "terrain": terrain_identity,
            "terrain_stride": terrain_stride,
            "terrain_composition_policy": (
                "suppress regional pixels only when their visible world XY lies in finite native-triangle "
                "support and the native raster supplies that pixel, then depth-test native against all "
                "unsuppressed regional pixels"
            ),
            "intrinsics": intrinsics.model_dump(mode="json"),
            "extrinsics": extrinsics.model_dump(mode="json"),
            "depth_semantics": "positive camera-forward distance in metres; NaN for sky",
            "subpixel_lifting": {
                "schema": PIXEL_LIFTING_SCHEMA,
                "depth_interpolation": PIXEL_LIFTING_DEPTH_METHOD,
                "world_point": "exact keypoint ray at interpolated camera-forward depth",
                "invalid_support": "any non-finite or non-positive bilinear depth support is rejected",
            },
            "world_frame": "terrain local east,north,absolute-elevation metres",
            "pixel_coordinate_convention": "integer pixel centres",
            "appearance": appearance_provenance,
            "appearance_terrain_coverage": round(appearance_coverage, 6),
            "native_high_resolution_patch_used": native_patch_used,
            "native_high_resolution_patch": native_patch_provenance,
            "terrain_surface_note": _terrain_surface_note(
                native_surface,
                native_patch_stride,
                native_patch_used,
                native_patch_provenance,
            ),
            "uses_query_image": False,
            "uses_reference_pose": False,
            "uses_source_depth_pfm": False,
        }
        return TerrainRenderBundle(
            rgb=np.asarray(rgb, dtype=np.uint8),
            forward_depth_m=depth,
            world_xyz_m=world,
            world_normals=normals,
            terrain_mask=mask,
            appearance_mask=appearance_mask,
            skyline_profile=np.asarray(geometry.skyline_profile, dtype=np.float64),
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            modality=modality,
            provenance=provenance,
        )

    def _cached_normal_grid(self, terrain: TerrainMap, terrain_sha256: str) -> NDArray[np.float64]:
        """Reuse expensive DEM gradients across an orientation render fan."""

        if self._normal_cache_key != terrain_sha256 or self._normal_cache_grid is None:
            normal_grid = _terrain_normal_grid(terrain)
            normal_grid.flags.writeable = False
            self._normal_cache_key = terrain_sha256
            self._normal_cache_grid = normal_grid
        return self._normal_cache_grid

    def _cached_native_normal_grid(
        self,
        surface: HeightfieldGrid,
        cache_key: str,
    ) -> NDArray[np.float64]:
        """Reuse normals for the exact decimated fine mesh across a render fan."""

        if self._native_normal_cache_key != cache_key or self._native_normal_cache_grid is None:
            normal_grid = _heightfield_normal_grid(surface)
            normal_grid.flags.writeable = False
            self._native_normal_cache_key = cache_key
            self._native_normal_cache_grid = normal_grid
        return self._native_normal_cache_grid


def unproject_pinhole_depth(
    forward_depth_m: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> NDArray[np.float64]:
    """Lift a forward-depth image into terrain-local world coordinates."""

    depth = np.asarray(forward_depth_m, dtype=np.float64)
    expected = (intrinsics.height_px, intrinsics.width_px)
    if depth.shape != expected:
        raise ValueError(f"depth shape must be {expected}, got {depth.shape}")
    rows, columns = np.indices(expected, dtype=np.float64)
    x_camera = (columns - intrinsics.principal_x_px) / intrinsics.focal_length_px * depth
    y_camera = (rows - intrinsics.principal_y_px) / intrinsics.focal_length_px * depth
    right, down, forward = camera_axes(extrinsics)
    position = np.asarray(extrinsics.position.as_tuple(), dtype=np.float64)
    world = (
        position[None, None, :]
        + x_camera[..., None] * right[None, None, :]
        + y_camera[..., None] * down[None, None, :]
        + depth[..., None] * forward[None, None, :]
    )
    world[~np.isfinite(depth) | (depth <= 0.0)] = np.nan
    return world


def world_normals_to_camera(
    world_normals: NDArray[np.float64],
    extrinsics: CameraExtrinsics,
) -> NDArray[np.float64]:
    """Express world-space normals in the camera's right/down/forward frame."""

    normals = np.asarray(world_normals, dtype=np.float64)
    if normals.ndim < 1 or normals.shape[-1] != 3:
        raise ValueError(f"world normals must end in a three-vector, got {normals.shape}")
    right, down, forward = camera_axes(extrinsics)
    world_from_camera = np.column_stack((right, down, forward))
    return np.einsum("...i,ij->...j", normals, world_from_camera)


def sample_terrain_normals(
    terrain: TerrainMap,
    east_m: NDArray[np.float64],
    north_m: NDArray[np.float64],
    valid_mask: NDArray[np.bool_] | None = None,
    *,
    normal_grid: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Bilinearly sample world-space heightfield normals."""

    return sample_heightfield_normals(
        HeightfieldGrid(
            x_m=np.asarray(terrain.x_m, dtype=np.float64),
            y_m=np.asarray(terrain.y_m, dtype=np.float64),
            elevation_m=np.asarray(terrain.elevation_m, dtype=np.float64),
        ),
        east_m,
        north_m,
        valid_mask,
        normal_grid=normal_grid,
    )


def sample_heightfield_normals(
    surface: HeightfieldGrid,
    east_m: NDArray[np.float64],
    north_m: NDArray[np.float64],
    valid_mask: NDArray[np.bool_] | None = None,
    *,
    normal_grid: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Sample normals without interpolating across elevation nodata."""

    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    if east.shape != north.shape:
        raise ValueError("normal-sampling east and north arrays must share a shape")
    grid_normals = (
        _heightfield_normal_grid(surface) if normal_grid is None else np.asarray(normal_grid, dtype=np.float64)
    )
    expected_grid_shape = (*surface.elevation_m.shape, 3)
    if grid_normals.shape != expected_grid_shape:
        raise ValueError(f"normal grid must have shape {expected_grid_shape}, got {grid_normals.shape}")
    finite = np.isfinite(east) & np.isfinite(north)
    inside = (
        finite
        & (east >= surface.x_m[0])
        & (east <= surface.x_m[-1])
        & (north >= surface.y_m[0])
        & (north <= surface.y_m[-1])
    )
    col = np.interp(np.where(finite, east, surface.x_m[0]).ravel(), surface.x_m, np.arange(surface.x_m.size))
    row = np.interp(np.where(finite, north, surface.y_m[0]).ravel(), surface.y_m, np.arange(surface.y_m.size))
    finite_normal_grid = np.all(np.isfinite(grid_normals), axis=2) & (np.linalg.norm(grid_normals, axis=2) > 0.5)
    safe_normals = np.where(finite_normal_grid[..., None], grid_normals, 0.0)
    sampled = np.stack(
        [map_coordinates(safe_normals[..., axis], [row, col], order=1, mode="nearest") for axis in range(3)],
        axis=-1,
    ).reshape((*east.shape, 3))
    sampled /= np.maximum(np.linalg.norm(sampled, axis=-1, keepdims=True), 1e-12)
    normal_support = map_coordinates(
        finite_normal_grid.astype(np.float64),
        [row, col],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(east.shape)
    valid = inside & (normal_support >= 1.0 - 1e-9)
    if valid_mask is not None:
        if np.asarray(valid_mask).shape != east.shape:
            raise ValueError("normal validity mask must match the coordinate arrays")
        valid &= np.asarray(valid_mask, dtype=bool)
    sampled[~valid] = 0.0
    return sampled.astype(np.float64, copy=False)


def _terrain_normal_grid(terrain: TerrainMap) -> NDArray[np.float64]:
    """Compute normalized world-space normals for a regular terrain grid."""

    return _heightfield_normal_grid(
        HeightfieldGrid(
            x_m=np.asarray(terrain.x_m, dtype=np.float64),
            y_m=np.asarray(terrain.y_m, dtype=np.float64),
            elevation_m=np.asarray(terrain.elevation_m, dtype=np.float64),
        )
    )


def _heightfield_normal_grid(surface: HeightfieldGrid) -> NDArray[np.float64]:
    """Compute normals while keeping nodata and its gradient neighbourhood invalid."""

    with np.errstate(invalid="ignore", divide="ignore"):
        dz_dnorth, dz_deast = np.gradient(surface.elevation_m, surface.y_m, surface.x_m)
    valid = np.isfinite(surface.elevation_m) & np.isfinite(dz_deast) & np.isfinite(dz_dnorth)
    grid_normals = np.zeros((*surface.elevation_m.shape, 3), dtype=np.float64)
    grid_normals[..., 0][valid] = -dz_deast[valid]
    grid_normals[..., 1][valid] = -dz_dnorth[valid]
    grid_normals[..., 2][valid] = 1.0
    grid_normals /= np.maximum(np.linalg.norm(grid_normals, axis=2, keepdims=True), 1e-12)
    return grid_normals


def lift_render_pixels(
    render: TerrainRenderBundle,
    render_xy_px: NDArray[np.float64],
    *,
    max_relative_depth_span: float = 0.08,
    discontinuity_radius_px: int = 1,
) -> LiftedRenderPoints:
    """Lift matched pixels using the rasterizer's inverse-depth interpolation."""

    lifted = lift_pinhole_depth_pixels(
        render.forward_depth_m,
        render.intrinsics,
        render.extrinsics,
        render_xy_px,
        support_mask=render.appearance_mask,
        max_relative_depth_span=max_relative_depth_span,
        discontinuity_radius_px=discontinuity_radius_px,
    )
    rejection_counts = dict(lifted.rejection_counts)
    rejection_counts["invalid_appearance"] = rejection_counts.pop("invalid_support")
    return LiftedRenderPoints(
        world_xyz_m=lifted.world_xyz_m,
        forward_depth_m=lifted.forward_depth_m,
        valid=lifted.valid,
        relative_depth_span=lifted.relative_depth_span,
        rejection_counts=rejection_counts,
    )


def lift_pinhole_depth_pixels(
    forward_depth_m: NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    xy_px: NDArray[np.float64],
    *,
    support_mask: NDArray[np.bool_] | None = None,
    max_relative_depth_span: float = 0.08,
    discontinuity_radius_px: int = 1,
) -> LiftedRenderPoints:
    """Lift subpixels from a pinhole forward-depth image without image-source assumptions."""

    xy = np.asarray(xy_px, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"pixel coordinates must have shape (N, 2), got {xy.shape}")
    if max_relative_depth_span <= 0.0:
        raise ValueError("max_relative_depth_span must be positive")
    if discontinuity_radius_px < 0:
        raise ValueError("discontinuity radius cannot be negative")
    depth = np.asarray(forward_depth_m, dtype=np.float64)
    expected_shape = (intrinsics.height_px, intrinsics.width_px)
    if depth.shape != expected_shape:
        raise ValueError(f"forward depth must have shape {expected_shape}, got {depth.shape}")
    support = np.ones(expected_shape, dtype=np.bool_) if support_mask is None else np.asarray(support_mask, dtype=bool)
    if support.shape != expected_shape:
        raise ValueError(f"support mask must have shape {expected_shape}, got {support.shape}")
    count = xy.shape[0]
    height, width = depth.shape
    world = np.full((count, 3), np.nan, dtype=np.float64)
    sampled_depth = np.full(count, np.nan, dtype=np.float64)
    relative_span = np.full(count, np.inf, dtype=np.float64)
    valid = np.ones(count, dtype=np.bool_)
    finite_xy = np.all(np.isfinite(xy), axis=1)
    valid &= finite_xy
    margin = discontinuity_radius_px
    inside = (
        finite_xy
        & (xy[:, 0] >= margin)
        & (xy[:, 0] <= width - 1 - margin)
        & (xy[:, 1] >= margin)
        & (xy[:, 1] <= height - 1 - margin)
    )
    valid &= inside
    safe_x = np.where(finite_xy, xy[:, 0], 0.0)
    safe_y = np.where(finite_xy, xy[:, 1], 0.0)
    # The rasterizer linearly interpolates inverse depth across each projected
    # triangle. Sampling stored forward depth directly would describe a
    # different curved surface between pixel centres. Preserve NaN/sky support
    # by converting only finite positive samples, interpolate inverse depth,
    # and invert once at the exact subpixel keypoint.
    finite_positive_depth = np.isfinite(depth) & (depth > 0.0)
    inverse_depth = np.full_like(depth, np.nan)
    inverse_depth[finite_positive_depth] = 1.0 / depth[finite_positive_depth]
    sampled_inverse_depth = map_coordinates(
        inverse_depth,
        [safe_y, safe_x],
        order=1,
        mode="constant",
        cval=np.nan,
    )
    inverse_depth_valid = np.isfinite(sampled_inverse_depth) & (sampled_inverse_depth > 0.0)
    sampled_depth[inverse_depth_valid] = 1.0 / sampled_inverse_depth[inverse_depth_valid]
    # Interpolating the precomputed XYZ image is not projectively equivalent to
    # lifting a subpixel coordinate at its interpolated forward depth. Build the
    # point from the exact keypoint ray so it reprojects to render_xy_px.
    right, down, forward = camera_axes(extrinsics)
    position = np.asarray(extrinsics.position.as_tuple(), dtype=np.float64)
    x_camera = (safe_x - intrinsics.principal_x_px) / intrinsics.focal_length_px * sampled_depth
    y_camera = (safe_y - intrinsics.principal_y_px) / intrinsics.focal_length_px * sampled_depth
    world[:] = (
        position[None, :]
        + x_camera[:, None] * right[None, :]
        + y_camera[:, None] * down[None, :]
        + sampled_depth[:, None] * forward[None, :]
    )
    depth_valid = np.isfinite(sampled_depth) & (sampled_depth > 0.0) & np.all(np.isfinite(world), axis=1)
    valid &= depth_valid

    sampled_support = map_coordinates(
        support.astype(np.float64),
        [safe_y, safe_x],
        order=1,
        mode="constant",
        cval=0.0,
    )
    support_valid = sampled_support >= 1.0 - 1e-9
    valid &= support_valid

    discontinuity_valid = np.zeros(count, dtype=np.bool_)
    for index in np.flatnonzero(valid):
        column_min = int(math.floor(float(xy[index, 0]))) - discontinuity_radius_px
        column_max = int(math.ceil(float(xy[index, 0]))) + discontinuity_radius_px
        row_min = int(math.floor(float(xy[index, 1]))) - discontinuity_radius_px
        row_max = int(math.ceil(float(xy[index, 1]))) + discontinuity_radius_px
        window = depth[
            row_min : row_max + 1,
            column_min : column_max + 1,
        ]
        if window.size == 0 or not np.all(np.isfinite(window)):
            continue
        median_depth = float(np.median(window))
        span = float(np.max(window) - np.min(window)) / max(median_depth, 1.0)
        relative_span[index] = span
        discontinuity_valid[index] = span <= max_relative_depth_span
    valid &= discontinuity_valid
    world[~valid] = np.nan
    sampled_depth[~valid] = np.nan
    rejection_counts = {
        "non_finite_pixel": int((~finite_xy).sum()),
        "outside_safe_image": int((finite_xy & ~inside).sum()),
        "invalid_or_sky_depth": int((inside & ~depth_valid).sum()),
        "invalid_support": int((inside & depth_valid & ~support_valid).sum()),
        "depth_discontinuity": int((inside & depth_valid & support_valid & ~discontinuity_valid).sum()),
        "accepted": int(valid.sum()),
    }
    return LiftedRenderPoints(
        world_xyz_m=world,
        forward_depth_m=sampled_depth,
        valid=valid,
        relative_depth_span=relative_span,
        rejection_counts=rejection_counts,
    )


def terrain_fingerprint(terrain: TerrainMap) -> dict[str, Any]:
    """Content-address the exact regular terrain surface used by a render."""

    digest = hashlib.sha256()
    for array in (terrain.x_m, terrain.y_m, terrain.elevation_m):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(str(contiguous.shape).encode())
        digest.update(memoryview(contiguous).cast("B"))
    spec_json = json.dumps(terrain.spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode()
    digest.update(spec_json)
    return {
        "sha256": digest.hexdigest(),
        "shape": list(terrain.elevation_m.shape),
        "spacing_m": {
            "east": float(np.median(np.diff(terrain.x_m))),
            "north": float(np.median(np.diff(terrain.y_m))),
        },
        "elevation_range_m": [float(np.min(terrain.elevation_m)), float(np.max(terrain.elevation_m))],
    }


def heightfield_fingerprint(surface: HeightfieldGrid) -> dict[str, Any]:
    """Content-address a possibly sparse elevation patch at its source spacing."""

    digest = hashlib.sha256(b"peakle-heightfield-source-v1\0")
    for array in (surface.x_m, surface.y_m, surface.elevation_m):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        descriptor = json.dumps(
            {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(memoryview(contiguous).cast("B"))
    finite = np.isfinite(surface.elevation_m)
    finite_values = surface.elevation_m[finite]
    elevation_range = [float(np.min(finite_values)), float(np.max(finite_values))] if finite_values.size else None
    return {
        "sha256": digest.hexdigest(),
        "shape": list(surface.elevation_m.shape),
        "spacing_m": _axis_spacing(surface),
        "bounds_m": {
            "east": [float(surface.x_m[0]), float(surface.x_m[-1])],
            "north": [float(surface.y_m[0]), float(surface.y_m[-1])],
        },
        "elevation_range_m": elevation_range,
        "finite_elevation_samples": int(finite.sum()),
        "nodata_samples": int(finite.size - finite.sum()),
    }


def _native_patch_provenance(
    surface: HeightfieldGrid | None,
    identity: dict[str, Any] | None,
    stride: int,
    *,
    visible_pixels: int,
    normal_pixels: int,
) -> dict[str, Any] | None:
    if surface is None or identity is None:
        return None
    rendered = surface.sampled(stride)
    rendered_finite = np.isfinite(rendered.elevation_m)
    return {
        "source_content_sha256": identity["sha256"],
        "source_shape": identity["shape"],
        "source_spacing_m": identity["spacing_m"],
        "source_bounds_m": identity["bounds_m"],
        "source_elevation_range_m": identity["elevation_range_m"],
        "source_finite_elevation_samples": identity["finite_elevation_samples"],
        "source_nodata_samples": identity["nodata_samples"],
        "render_stride": stride,
        "render_mesh_shape": list(rendered.elevation_m.shape),
        "render_effective_mesh_spacing_m": _axis_spacing(rendered),
        "render_maximum_mesh_spacing_m": _axis_maximum_spacing(rendered),
        "render_finite_elevation_samples": int(rendered_finite.sum()),
        "rendered_at_source_spacing": stride == 1,
        "visible_z_buffer_pixels": visible_pixels,
        "visible_pixels_with_patch_normals": normal_pixels,
        "depth_composition_policy": (
            "the native mesh replaces the coarse representation at matching supported world XY, while "
            "uncovered and unsuppressed regional geometry remains eligible to occlude it"
        ),
        "nodata_policy": (
            "a triangle is omitted unless all three decimated vertices have finite elevation; "
            "the regular regional terrain remains as background coverage"
        ),
        "resolution_note": (
            "The source grid is decimated before rasterization; effective mesh spacing, not source spacing, "
            "is the rendered geometric resolution."
            if stride > 1
            else "Every source-grid vertex is retained for rasterization."
        ),
    }


def _render_surface_identity_sha256(
    terrain_identity: dict[str, Any],
    terrain_stride: int,
    native_identity: dict[str, Any] | None,
    native_stride: int | None,
) -> str:
    """Build the pre-render cache identity of every geometric surface input."""

    record = {
        "schema": "terrain_render_surface_identity_v4",
        "regular_terrain_sha256": terrain_identity["sha256"],
        "regular_terrain_stride": terrain_stride,
        "native_patch_sha256": native_identity["sha256"] if native_identity is not None else None,
        "native_patch_stride": native_stride,
        "raster_composition": "world_xy_authoritative_support_with_native_pixel_then_joint_depth_test",
        "nodata_policy": "all_triangle_vertices_finite",
    }
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _axis_spacing(surface: HeightfieldGrid) -> dict[str, float]:
    return {
        "east": float(np.median(np.abs(np.diff(surface.x_m)))),
        "north": float(np.median(np.abs(np.diff(surface.y_m)))),
    }


def _axis_maximum_spacing(surface: HeightfieldGrid) -> dict[str, float]:
    return {
        "east": float(np.max(np.abs(np.diff(surface.x_m)))),
        "north": float(np.max(np.abs(np.diff(surface.y_m)))),
    }


def _terrain_surface_note(
    surface: HeightfieldGrid | None,
    stride: int,
    used: bool,
    provenance: dict[str, Any] | None,
) -> str:
    if surface is None:
        return "regular TerrainMap grid only; no native high-resolution elevation patch was supplied"
    if not used:
        return (
            "a native-source elevation patch was supplied but contributed no visible z-buffer pixels in this "
            "frame; the regular TerrainMap provides the visible regional surface"
        )
    if provenance is None:
        raise RuntimeError("used native patch is missing provenance")
    spacing = provenance["render_effective_mesh_spacing_m"]
    return (
        "regular TerrainMap regional background plus a native-source elevation patch authoritative within its "
        "finite world-XY support, rasterized "
        f"with stride {stride} (effective mesh spacing east={spacing['east']:.3f} m, "
        f"north={spacing['north']:.3f} m); this is not a claim of source-resolution rendering"
    )


def _render_content_sha256(
    *,
    rgb: NDArray[np.uint8],
    depth: NDArray[np.float64],
    world: NDArray[np.float64],
    normals: NDArray[np.float64],
    terrain_mask: NDArray[np.bool_],
    appearance_mask: NDArray[np.bool_],
    skyline: NDArray[np.float64],
    native_overlay_mask: NDArray[np.bool_],
) -> str:
    """Content-address the exact metric render products passed to matching."""

    digest = hashlib.sha256(b"peakle-terrain-render-v3\0")
    for array in (rgb, depth, world, normals, terrain_mask, appearance_mask, skyline, native_overlay_mask):
        contiguous = np.ascontiguousarray(array)
        descriptor = json.dumps(
            {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(descriptor).to_bytes(8, "big"))
        digest.update(descriptor)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _hillshade_rgb(
    terrain: TerrainMap,
    normals: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.uint8]:
    sun = np.asarray([-0.48, -0.36, 0.80], dtype=np.float64)
    sun /= np.linalg.norm(sun)
    light = np.clip(normals @ sun, 0.0, 1.0)
    world_up = np.clip(normals[..., 2], 0.0, 1.0)
    shade = np.clip(0.24 + 0.66 * light + 0.10 * world_up, 0.0, 1.0)
    # Use slope-sensitive rock/vegetation tones without pretending they are
    # real land cover. This modality is always labelled DEM-derived.
    vegetation = np.asarray([73.0, 104.0, 67.0])
    rock = np.asarray([151.0, 145.0, 132.0])
    snow = np.asarray([220.0, 224.0, 219.0])
    steepness = np.clip(1.0 - world_up, 0.0, 1.0)
    base = vegetation[None, None, :] * (1.0 - steepness[..., None]) + rock[None, None, :] * steepness[..., None]
    # A global snow tint is a deterministic visual cue only; provenance keeps
    # it out of claims about actual seasonal surface appearance.
    z_min = float(np.min(terrain.elevation_m))
    z_max = float(np.max(terrain.elevation_m))
    if z_max > z_min:
        # Approximate visible elevation from the normal sampling coordinates is
        # unavailable here, so use steepness as the main stable cue and avoid an
        # image-row gradient (the old renderer's misleading behaviour).
        snow_mix = np.clip((steepness - 0.55) / 0.45, 0.0, 0.35)
        base = base * (1.0 - snow_mix[..., None]) + snow[None, None, :] * snow_mix[..., None]
    rgb = np.clip(base * shade[..., None], 0.0, 255.0).astype(np.uint8)
    rgb[~mask] = 0
    return rgb


def _relative_depth_rgb(depth: NDArray[np.float64], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    valid_depth = depth[mask & np.isfinite(depth) & (depth > 0.0)]
    result = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if valid_depth.size == 0:
        return result
    low, high = np.percentile(valid_depth, [2.0, 98.0])
    render_valid = mask & np.isfinite(depth) & (depth > 0.0)
    normalized = np.zeros_like(depth, dtype=np.float64)
    normalized[render_valid] = np.clip(
        (depth[render_valid] - low) / max(float(high - low), 1e-9),
        0.0,
        1.0,
    )
    # Fixed approximation of matplotlib's Spectral_r colour ordering. MINIMA
    # used this family for synthetic depth; keeping it fixed makes the modality
    # reproducible without adding matplotlib as a runtime dependency.
    stops = np.asarray(
        [
            [94, 79, 162],
            [50, 136, 189],
            [102, 194, 165],
            [230, 245, 152],
            [254, 224, 139],
            [244, 109, 67],
            [158, 1, 66],
        ],
        dtype=np.float64,
    )
    scaled = normalized * (len(stops) - 1)
    left = np.clip(np.floor(scaled).astype(np.int64), 0, len(stops) - 1)
    right = np.minimum(left + 1, len(stops) - 1)
    fraction = scaled - left
    colours = stops[left] * (1.0 - fraction[..., None]) + stops[right] * fraction[..., None]
    result[render_valid] = np.clip(np.rint(colours[render_valid]), 0.0, 255.0).astype(np.uint8)
    return result


def _with_sky(
    terrain_rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    sky_rgb: tuple[int, int, int],
) -> NDArray[np.uint8]:
    result = np.empty_like(terrain_rgb, dtype=np.uint8)
    result[:] = np.asarray(sky_rgb, dtype=np.uint8)
    result[mask] = terrain_rgb[mask]
    return result
