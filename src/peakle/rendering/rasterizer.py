"""Synthetic terrain image renderer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.terrain import TerrainMap
from peakle.rendering.pinhole import (
    DEFAULT_NEAR_CLIP_M,
    camera_axes,
    camera_coordinates,
    project_camera_points,
    project_points,
)
from peakle.rendering.skyline import interpolate_profile

RASTER_EPSILON = 1e-8


class HeightfieldLike(Protocol):
    """Structural interface for a georeferenced regular elevation grid."""

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    elevation_m: NDArray[np.float64]


@dataclass(frozen=True)
class HeightfieldGrid:
    """Validated local-ENU heightfield whose elevations may contain nodata."""

    x_m: NDArray[np.float64]
    y_m: NDArray[np.float64]
    elevation_m: NDArray[np.float64]

    @classmethod
    def from_like(cls, surface: HeightfieldLike) -> HeightfieldGrid:
        """Build a non-copying validated view of a structural heightfield."""

        return cls(
            x_m=np.asarray(surface.x_m, dtype=np.float64),
            y_m=np.asarray(surface.y_m, dtype=np.float64),
            elevation_m=np.asarray(surface.elevation_m, dtype=np.float64),
        )

    def __post_init__(self) -> None:
        x_m = np.asarray(self.x_m, dtype=np.float64)
        y_m = np.asarray(self.y_m, dtype=np.float64)
        elevation_m = np.asarray(self.elevation_m, dtype=np.float64)
        if x_m.ndim != 1 or y_m.ndim != 1:
            raise ValueError("heightfield coordinate axes must be one-dimensional")
        if x_m.size < 2 or y_m.size < 2:
            raise ValueError("heightfield coordinate axes must contain at least two samples")
        if elevation_m.shape != (y_m.size, x_m.size):
            raise ValueError(
                "heightfield elevation shape must be (len(y_m), len(x_m)); "
                f"got {elevation_m.shape} for {(y_m.size, x_m.size)}"
            )
        if not np.all(np.isfinite(x_m)) or not np.all(np.isfinite(y_m)):
            raise ValueError("heightfield coordinate axes must be finite")
        if not np.all(np.diff(x_m) > 0.0) or not np.all(np.diff(y_m) > 0.0):
            raise ValueError("heightfield coordinate axes must be strictly increasing")
        object.__setattr__(self, "x_m", x_m)
        object.__setattr__(self, "y_m", y_m)
        object.__setattr__(self, "elevation_m", elevation_m)

    def sampled(self, stride: int) -> HeightfieldGrid:
        """Return the exact mesh sampled by the rasterizer, including boundaries."""

        rows = _stride_indices(self.y_m.size, stride)
        columns = _stride_indices(self.x_m.size, stride)
        return HeightfieldGrid(
            x_m=self.x_m[columns],
            y_m=self.y_m[rows],
            elevation_m=self.elevation_m[np.ix_(rows, columns)],
        )


class RenderArrays(BaseModel):
    """In-memory synthetic render outputs.

    Attributes:
        image: RGB image.
        terrain_mask: Boolean terrain mask.
        skyline_profile: Dense skyline y-coordinate profile.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    image: Image.Image
    terrain_mask: NDArray[np.bool_]
    skyline_profile: NDArray[np.float64]


class GeometryRenderArrays(BaseModel):
    """Metric pinhole render products used for correspondence lifting."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    forward_depth_m: NDArray[np.float64]
    terrain_mask: NDArray[np.bool_]
    skyline_profile: NDArray[np.float64]
    native_overlay_mask: NDArray[np.bool_]


class _ScreenVertex(NamedTuple):
    """Projected vertex data used during software rasterization."""

    u_px: float
    v_px: float
    inverse_depth: float
    valid: bool


class _CameraVertex(NamedTuple):
    """Triangle vertex in camera right/down/forward coordinates."""

    right_m: float
    down_m: float
    forward_m: float


class _RasterizedTerrain(BaseModel):
    """Visible terrain mask and skyline from projected mesh rasterization.

    Attributes:
        terrain_mask: Boolean mask for visible terrain pixels.
        skyline_profile: Dense y-coordinate profile of the visible terrain top.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    terrain_mask: NDArray[np.bool_]
    skyline_profile: NDArray[np.float64]
    inverse_depth: NDArray[np.float64]
    native_overlay_mask: NDArray[np.bool_]


class SyntheticRenderer:
    """Renders synthetic terrain from a pinhole camera."""

    def render(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
        *,
        overlay: HeightfieldLike | None = None,
        overlay_stride: int = 1,
    ) -> RenderArrays:
        """Renders an RGB image, terrain mask, and skyline profile.

        Args:
            terrain: Terrain map to render.
            intrinsics: Camera intrinsics.
            extrinsics: Camera extrinsics.
            stride: Terrain point subsampling stride.

        Returns:
            Render arrays for image and contour extraction.
        """

        raster = self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
            overlay=overlay,
            overlay_stride=overlay_stride,
        )
        image = self._paint_image(raster.skyline_profile, raster.terrain_mask)
        return RenderArrays(
            image=image,
            terrain_mask=raster.terrain_mask,
            skyline_profile=raster.skyline_profile,
        )

    def skyline_profile(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
        *,
        overlay: HeightfieldLike | None = None,
        overlay_stride: int = 1,
    ) -> NDArray[np.float64]:
        """Rasterizes visible terrain and returns its upper silhouette profile."""

        return self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
            overlay=overlay,
            overlay_stride=overlay_stride,
        ).skyline_profile

    def visible_mask(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
        *,
        overlay: HeightfieldLike | None = None,
        overlay_stride: int = 1,
    ) -> NDArray[np.bool_]:
        """Rasterizes visible terrain and returns its boolean coverage mask."""

        return self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
            overlay=overlay,
            overlay_stride=overlay_stride,
        ).terrain_mask

    def fast_skyline(
        self,
        points: NDArray[np.float64],
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
    ) -> NDArray[np.float64]:
        """Skyline profile from precomputed points, without triangle rasterization.

        The silhouette top in each column is simply the highest (smallest row)
        projected terrain point, so a per-column minimum over projected points is
        exact for an opaque surface and far cheaper than mesh rasterization. This
        is the optimizer's hot path; pass terrain points once via
        `TerrainMap.flattened_points(stride)`.

        Args:
            points: Precomputed `(N, 3)` local terrain points.
            intrinsics: Camera intrinsics.
            extrinsics: Candidate camera extrinsics.

        Returns:
            Dense skyline y-profile of length `intrinsics.width_px`.
        """

        width = intrinsics.width_px
        height = intrinsics.height_px
        u_px, v_px, _depth, valid = project_points(points, intrinsics, extrinsics)
        finite = valid & np.isfinite(u_px) & np.isfinite(v_px)
        columns = np.floor(u_px[finite]).astype(np.int64)
        inside = (columns >= 0) & (columns < width)
        profile = np.full(width, np.inf, dtype=np.float64)
        if np.any(inside):
            np.minimum.at(profile, columns[inside], v_px[finite][inside])
        profile[~np.isfinite(profile)] = np.nan
        profile = interpolate_profile(profile, fallback=float(height) * 0.62)
        return np.clip(profile, 0.0, float(height - 1))

    def _rasterize_visible_terrain(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int,
        overlay: HeightfieldLike | None = None,
        overlay_stride: int = 1,
    ) -> _RasterizedTerrain:
        """Rasterize regional terrain and an optional fine patch into one z-buffer."""

        width = intrinsics.width_px
        height = intrinsics.height_px
        inverse_depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
        regional = HeightfieldGrid(
            x_m=np.asarray(terrain.x_m, dtype=np.float64),
            y_m=np.asarray(terrain.y_m, dtype=np.float64),
            elevation_m=np.asarray(terrain.elevation_m, dtype=np.float64),
        )
        self._rasterize_heightfield_into_buffer(
            regional,
            intrinsics,
            extrinsics,
            stride=stride,
            inverse_depth_buffer=inverse_depth_buffer,
        )
        native_overlay_mask = np.zeros((height, width), dtype=np.bool_)
        if overlay is not None:
            native_surface = HeightfieldGrid.from_like(overlay).sampled(overlay_stride)
            native_inverse_depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
            self._rasterize_heightfield_into_buffer(
                native_surface,
                intrinsics,
                extrinsics,
                stride=1,
                inverse_depth_buffer=native_inverse_depth_buffer,
            )
            # A coarse triangle represents the same physical surface only when
            # its visible world point lies inside a finite triangle of the
            # authoritative native mesh. Suppress exactly those base pixels.
            # Regional ridges outside the patch footprint remain in the depth
            # test and can correctly occlude a farther native surface.
            base_east_m, base_north_m = self._unproject_buffer_xy(
                inverse_depth_buffer,
                intrinsics,
                extrinsics,
            )
            authoritative_support = _finite_triangle_support(
                native_surface,
                base_east_m,
                base_north_m,
            )
            native_visible = np.isfinite(native_inverse_depth_buffer)
            # Do not punch a screen-space hole at the projected boundary of a
            # substantially displaced fine surface. Authority can replace a
            # base pixel only where the native raster also supplies a sample.
            inverse_depth_buffer[authoritative_support & native_visible] = -np.inf
            native_overlay_mask = native_visible & (native_inverse_depth_buffer > inverse_depth_buffer)
            inverse_depth_buffer[native_overlay_mask] = native_inverse_depth_buffer[native_overlay_mask]

        mask = np.isfinite(inverse_depth_buffer)
        return _RasterizedTerrain(
            terrain_mask=mask,
            skyline_profile=self._profile_from_mask(mask, fallback=float(height * 0.62)),
            inverse_depth=inverse_depth_buffer,
            native_overlay_mask=native_overlay_mask,
        )

    def _rasterize_heightfield_into_buffer(
        self,
        surface: HeightfieldGrid,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        *,
        stride: int,
        inverse_depth_buffer: NDArray[np.float64],
    ) -> None:
        """Project one regular heightfield into a shared inverse-depth buffer."""

        sampled = surface.sampled(stride)
        x_grid, y_grid = np.meshgrid(sampled.x_m, sampled.y_m)
        z_grid = sampled.elevation_m
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel())).astype(np.float64)
        camera_points = camera_coordinates(points, extrinsics)
        u_px, v_px, depth, valid = project_camera_points(camera_points, intrinsics)

        grid_height, grid_width = z_grid.shape
        u_grid = u_px.reshape(grid_height, grid_width)
        v_grid = v_px.reshape(grid_height, grid_width)
        finite_point = np.all(np.isfinite(points), axis=1)
        valid &= finite_point & np.isfinite(u_px) & np.isfinite(v_px) & np.isfinite(depth)
        valid_grid = valid.reshape(grid_height, grid_width)
        camera_grid = camera_points.reshape(grid_height, grid_width, 3)
        depth_grid = depth.reshape(grid_height, grid_width)
        inverse_depth_grid = np.zeros_like(u_grid, dtype=np.float64)
        finite_depth = valid_grid & (depth_grid > 0.0)
        inverse_depth_grid[finite_depth] = 1.0 / depth_grid[finite_depth]

        for row in range(grid_height - 1):
            for col in range(grid_width - 1):
                top_left = self._screen_vertex(row, col, u_grid, v_grid, inverse_depth_grid, valid_grid)
                top_right = self._screen_vertex(row, col + 1, u_grid, v_grid, inverse_depth_grid, valid_grid)
                bottom_left = self._screen_vertex(row + 1, col, u_grid, v_grid, inverse_depth_grid, valid_grid)
                bottom_right = self._screen_vertex(row + 1, col + 1, u_grid, v_grid, inverse_depth_grid, valid_grid)
                self._rasterize_heightfield_triangle(
                    inverse_depth_buffer,
                    (top_left, top_right, bottom_left),
                    camera_grid,
                    ((row, col), (row, col + 1), (row + 1, col)),
                    intrinsics,
                )
                self._rasterize_heightfield_triangle(
                    inverse_depth_buffer,
                    (top_right, bottom_right, bottom_left),
                    camera_grid,
                    ((row, col + 1), (row + 1, col + 1), (row + 1, col)),
                    intrinsics,
                )

    def _unproject_buffer_xy(
        self,
        inverse_depth_buffer: NDArray[np.float64],
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Unproject visible base depth to world east/north coordinates."""

        visible = np.isfinite(inverse_depth_buffer) & (inverse_depth_buffer > 0.0)
        depth = np.full_like(inverse_depth_buffer, np.nan)
        depth[visible] = 1.0 / inverse_depth_buffer[visible]
        rows, columns = np.indices(inverse_depth_buffer.shape, dtype=np.float64)
        x_camera = (columns - intrinsics.principal_x_px) / intrinsics.focal_length_px * depth
        y_camera = (rows - intrinsics.principal_y_px) / intrinsics.focal_length_px * depth
        right, down, forward = camera_axes(extrinsics)
        position = np.asarray(extrinsics.position.as_tuple(), dtype=np.float64)
        east_m = position[0] + x_camera * right[0] + y_camera * down[0] + depth * forward[0]
        north_m = position[1] + x_camera * right[1] + y_camera * down[1] + depth * forward[1]
        east_m[~visible] = np.nan
        north_m[~visible] = np.nan
        return east_m, north_m

    def depth_image(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
        *,
        overlay: HeightfieldLike | None = None,
        overlay_stride: int = 1,
    ) -> NDArray[np.float64]:
        """Returns the per-pixel visible-terrain depth in metres (NaN for sky)."""

        buffer = self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
            overlay=overlay,
            overlay_stride=overlay_stride,
        ).inverse_depth
        depth = np.full_like(buffer, np.nan)
        visible = np.isfinite(buffer) & (buffer > 0.0)
        depth[visible] = 1.0 / buffer[visible]
        return depth

    def geometry(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
        *,
        overlay: HeightfieldLike | None = None,
        overlay_stride: int = 1,
    ) -> GeometryRenderArrays:
        """Return one occlusion-consistent metric depth/mask/silhouette pass."""

        raster = self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
            overlay=overlay,
            overlay_stride=overlay_stride,
        )
        depth = np.full_like(raster.inverse_depth, np.nan)
        visible = np.isfinite(raster.inverse_depth) & (raster.inverse_depth > 0.0)
        depth[visible] = 1.0 / raster.inverse_depth[visible]
        return GeometryRenderArrays(
            forward_depth_m=depth,
            terrain_mask=raster.terrain_mask,
            skyline_profile=raster.skyline_profile,
            native_overlay_mask=raster.native_overlay_mask,
        )

    def ridge_layers(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
        max_layers: int = 4,
        drop_fraction: float = 0.3,
        overlay: HeightfieldLike | None = None,
        overlay_stride: int = 1,
    ) -> NDArray[np.float64]:
        """Predicts occlusion ridge rows per column from the rendered depth.

        Scanning a column downward, the skyline is the first visible row and each
        internal ridge crest is where the visible depth drops sharply (a nearer
        ridge silhouetted against farther terrain). Returns `(max_layers + 1,
        width)`: row 0 the skyline, the rest occlusion crests (NaN where absent),
        for matching against the photo's extracted ridges.
        """

        buffer = self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
            overlay=overlay,
            overlay_stride=overlay_stride,
        ).inverse_depth
        height, width = buffer.shape
        layers = np.full((max_layers + 1, width), np.nan, dtype=np.float64)
        for column in range(width):
            rows = np.flatnonzero(np.isfinite(buffer[:, column]))
            if rows.size == 0:
                continue
            layers[0, column] = float(rows[0])  # skyline
            depth = 1.0 / buffer[rows, column]
            drops = (depth[:-1] - depth[1:]) / np.maximum(depth[:-1], 1.0)  # relative depth drop going down
            crest_order = np.argsort(drops)[::-1]
            kept = 0
            for index in crest_order:
                if drops[index] < drop_fraction:
                    break
                layers[1 + kept, column] = float(rows[index + 1])
                kept += 1
                if kept >= max_layers:
                    break
        return layers

    def _screen_vertex(
        self,
        row: int,
        col: int,
        u_grid: NDArray[np.float64],
        v_grid: NDArray[np.float64],
        inverse_depth_grid: NDArray[np.float64],
        valid_grid: NDArray[np.bool_],
    ) -> _ScreenVertex:
        """Builds a rasterizer vertex from projected terrain grids."""

        return _ScreenVertex(
            u_px=float(u_grid[row, col]),
            v_px=float(v_grid[row, col]),
            inverse_depth=float(inverse_depth_grid[row, col]),
            valid=bool(valid_grid[row, col]),
        )

    def _rasterize_heightfield_triangle(
        self,
        inverse_depth_buffer: NDArray[np.float64],
        screen_vertices: tuple[_ScreenVertex, _ScreenVertex, _ScreenVertex],
        camera_grid: NDArray[np.float64],
        grid_indices: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
        intrinsics: CameraIntrinsics,
    ) -> None:
        """Rasterize a heightfield triangle, clipping it only when necessary."""

        if all(vertex.valid for vertex in screen_vertices):
            self._rasterize_triangle(inverse_depth_buffer, screen_vertices)
            return
        camera_values = np.asarray([camera_grid[row, col] for row, col in grid_indices], dtype=np.float64)
        if not np.all(np.isfinite(camera_values)) or np.max(camera_values[:, 2]) < DEFAULT_NEAR_CLIP_M:
            return
        camera_vertices = (
            _CameraVertex(*map(float, camera_values[0])),
            _CameraVertex(*map(float, camera_values[1])),
            _CameraVertex(*map(float, camera_values[2])),
        )
        clipped = _clip_polygon_to_near_plane(camera_vertices)
        projected = tuple(_project_camera_vertex(vertex, intrinsics) for vertex in clipped)
        for index in range(1, len(projected) - 1):
            self._rasterize_triangle(
                inverse_depth_buffer,
                (projected[0], projected[index], projected[index + 1]),
            )

    def _rasterize_triangle(
        self,
        inverse_depth_buffer: NDArray[np.float64],
        vertices: tuple[_ScreenVertex, _ScreenVertex, _ScreenVertex],
    ) -> None:
        """Rasterizes one projected triangle into an inverse-depth buffer."""

        if not all(vertex.valid for vertex in vertices):
            return
        if not all(np.isfinite(vertex.u_px + vertex.v_px + vertex.inverse_depth) for vertex in vertices):
            return

        height, width = inverse_depth_buffer.shape
        min_x = max(0, int(np.floor(min(vertex.u_px for vertex in vertices))))
        max_x = min(width - 1, int(np.ceil(max(vertex.u_px for vertex in vertices))))
        min_y = max(0, int(np.floor(min(vertex.v_px for vertex in vertices))))
        max_y = min(height - 1, int(np.ceil(max(vertex.v_px for vertex in vertices))))
        if min_x > max_x or min_y > max_y:
            return

        a, b, c = vertices
        denominator = (b.v_px - c.v_px) * (a.u_px - c.u_px) + (c.u_px - b.u_px) * (a.v_px - c.v_px)
        if abs(denominator) <= RASTER_EPSILON:
            return

        # Camera projection in this project uses integer coordinates for pixel
        # centres (for example, an odd-width principal point is an integer).
        # Sampling triangles at the same centres is essential when a depth
        # pixel is later lifted back to metric 3D for PnP.
        x_coords = np.arange(min_x, max_x + 1, dtype=np.float64)
        y_coords = np.arange(min_y, max_y + 1, dtype=np.float64)
        sample_x = x_coords[None, :]
        sample_y = y_coords[:, None]

        weight_a = ((b.v_px - c.v_px) * (sample_x - c.u_px) + (c.u_px - b.u_px) * (sample_y - c.v_px)) / denominator
        weight_b = ((c.v_px - a.v_px) * (sample_x - c.u_px) + (a.u_px - c.u_px) * (sample_y - c.v_px)) / denominator
        weight_c = 1.0 - weight_a - weight_b
        inside = (weight_a >= -RASTER_EPSILON) & (weight_b >= -RASTER_EPSILON) & (weight_c >= -RASTER_EPSILON)
        if not np.any(inside):
            return

        inverse_depth = weight_a * a.inverse_depth + weight_b * b.inverse_depth + weight_c * c.inverse_depth
        target = inverse_depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
        update = inside & (inverse_depth > target)
        target[update] = inverse_depth[update]

    def _profile_from_mask(
        self,
        mask: NDArray[np.bool_],
        fallback: float,
    ) -> NDArray[np.float64]:
        """Extracts a dense skyline profile from a visible terrain mask."""

        height, width = mask.shape
        profile = np.full(width, np.nan, dtype=np.float64)
        occupied_columns = np.any(mask, axis=0)
        if np.any(occupied_columns):
            profile[occupied_columns] = np.argmax(mask[:, occupied_columns], axis=0)
        profile = interpolate_profile(profile, fallback=fallback)
        return np.clip(profile, 0.0, float(height - 1))

    def draw_contour_debug(
        self,
        image: Image.Image,
        observed_profile: NDArray[np.float64],
        predicted_profile: NDArray[np.float64] | None = None,
    ) -> Image.Image:
        """Draws observed and optional predicted profiles on an image."""

        debug = image.copy()
        draw = ImageDraw.Draw(debug)
        profiles = [(observed_profile, (255, 226, 96))]
        if predicted_profile is not None:
            profiles.append((predicted_profile, (58, 190, 255)))
        for profile, _fill in profiles:
            self._draw_profile(draw, profile, fill=(16, 17, 14), width=7)
        for profile, fill in profiles:
            self._draw_profile(draw, profile, fill=fill, width=3)
        return debug

    def _paint_image(
        self,
        profile: NDArray[np.float64],
        mask: NDArray[np.bool_],
    ) -> Image.Image:
        height, width = mask.shape
        rows = np.arange(height, dtype=np.float64)[:, None]
        sky_t = rows / max(height - 1, 1)
        image = np.empty((height, width, 3), dtype=np.uint8)

        top_sky = np.array([140, 183, 214], dtype=np.float64)
        horizon_sky = np.array([229, 218, 190], dtype=np.float64)
        sky = (top_sky * (1.0 - sky_t) + horizon_sky * sky_t).astype(np.uint8)
        image[:, :, :] = sky[:, None, :]

        relative_depth = (rows - profile[None, :]) / np.maximum(height - profile[None, :], 1.0)
        global_depth = rows / max(height - 1, 1)
        terrain_depth = np.clip(0.35 * relative_depth + 0.65 * global_depth, 0.0, 1.0)
        high_color = np.array([196, 188, 162], dtype=np.float64)
        low_color = np.array([52, 92, 69], dtype=np.float64)
        terrain_color = (high_color * (1.0 - terrain_depth[:, :, None]) + low_color * terrain_depth[:, :, None]).astype(
            np.uint8
        )
        image[mask] = terrain_color[mask]

        for offset, color in [(-1, (96, 89, 72)), (0, (35, 38, 34))]:
            line_rows = np.rint(profile + offset).astype(np.int64)
            valid = (line_rows >= 0) & (line_rows < height)
            image[line_rows[valid], np.arange(width)[valid]] = color

        return Image.fromarray(image, mode="RGB")

    def _draw_profile(
        self,
        draw: ImageDraw.ImageDraw,
        profile: NDArray[np.float64],
        fill: tuple[int, int, int],
        width: int,
    ) -> None:
        points = [(int(column), int(round(y_px))) for column, y_px in enumerate(profile)]
        if len(points) >= 2:
            draw.line(points, fill=fill, width=width)


def _stride_indices(size: int, stride: int) -> NDArray[np.int64]:
    """Subsample a regular axis while retaining both physical boundaries."""

    if stride < 1:
        raise ValueError("stride must be positive")
    indices = np.arange(0, size, stride, dtype=np.int64)
    if indices.size == 0 or indices[-1] != size - 1:
        indices = np.append(indices, np.int64(size - 1))
    return indices


def _clip_polygon_to_near_plane(
    vertices: tuple[_CameraVertex, _CameraVertex, _CameraVertex],
    near_clip_m: float = DEFAULT_NEAR_CLIP_M,
) -> tuple[_CameraVertex, ...]:
    """Clip one camera-space triangle to `forward >= near` in winding order."""

    clipped: list[_CameraVertex] = []
    previous = vertices[-1]
    previous_inside = previous.forward_m >= near_clip_m
    for current in vertices:
        current_inside = current.forward_m >= near_clip_m
        if current_inside != previous_inside:
            fraction = (near_clip_m - previous.forward_m) / (current.forward_m - previous.forward_m)
            clipped.append(
                _CameraVertex(
                    right_m=previous.right_m + fraction * (current.right_m - previous.right_m),
                    down_m=previous.down_m + fraction * (current.down_m - previous.down_m),
                    forward_m=near_clip_m,
                )
            )
        if current_inside:
            clipped.append(current)
        previous = current
        previous_inside = current_inside
    return tuple(clipped)


def _project_camera_vertex(vertex: _CameraVertex, intrinsics: CameraIntrinsics) -> _ScreenVertex:
    """Project a finite camera-space vertex known to lie on/in front of the near plane."""

    inverse_depth = 1.0 / vertex.forward_m
    return _ScreenVertex(
        u_px=intrinsics.focal_length_px * vertex.right_m * inverse_depth + intrinsics.principal_x_px,
        v_px=intrinsics.focal_length_px * vertex.down_m * inverse_depth + intrinsics.principal_y_px,
        inverse_depth=inverse_depth,
        valid=True,
    )


def _finite_triangle_support(
    surface: HeightfieldGrid,
    east_m: NDArray[np.float64],
    north_m: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Return whether each world XY lies in a finite rasterized mesh triangle."""

    east = np.asarray(east_m, dtype=np.float64)
    north = np.asarray(north_m, dtype=np.float64)
    if east.shape != north.shape:
        raise ValueError("heightfield-support east and north arrays must have the same shape")
    inside = (
        np.isfinite(east)
        & np.isfinite(north)
        & (east >= surface.x_m[0])
        & (east <= surface.x_m[-1])
        & (north >= surface.y_m[0])
        & (north <= surface.y_m[-1])
    )
    safe_east = np.where(inside, east, surface.x_m[0])
    safe_north = np.where(inside, north, surface.y_m[0])
    columns = np.searchsorted(surface.x_m, safe_east, side="right") - 1
    rows = np.searchsorted(surface.y_m, safe_north, side="right") - 1
    columns = np.clip(columns, 0, surface.x_m.size - 2)
    rows = np.clip(rows, 0, surface.y_m.size - 2)
    east_fraction = (safe_east - surface.x_m[columns]) / (surface.x_m[columns + 1] - surface.x_m[columns])
    north_fraction = (safe_north - surface.y_m[rows]) / (surface.y_m[rows + 1] - surface.y_m[rows])
    finite = np.isfinite(surface.elevation_m)
    top_left = finite[rows, columns]
    top_right = finite[rows, columns + 1]
    bottom_left = finite[rows + 1, columns]
    bottom_right = finite[rows + 1, columns + 1]
    diagonal = east_fraction + north_fraction
    first_triangle = (diagonal <= 1.0 + RASTER_EPSILON) & top_left & top_right & bottom_left
    second_triangle = (diagonal >= 1.0 - RASTER_EPSILON) & top_right & bottom_right & bottom_left
    return np.asarray(inside & (first_triangle | second_triangle), dtype=np.bool_)
