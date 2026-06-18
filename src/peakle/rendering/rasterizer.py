"""Synthetic terrain image renderer."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.terrain import TerrainMap
from peakle.rendering.pinhole import project_points
from peakle.rendering.skyline import interpolate_profile

RASTER_EPSILON = 1e-8


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


class _ScreenVertex(NamedTuple):
    """Projected vertex data used during software rasterization."""

    u_px: float
    v_px: float
    inverse_depth: float
    valid: bool


class _RasterizedTerrain(BaseModel):
    """Visible terrain mask and skyline from projected mesh rasterization.

    Attributes:
        terrain_mask: Boolean mask for visible terrain pixels.
        skyline_profile: Dense y-coordinate profile of the visible terrain top.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    terrain_mask: NDArray[np.bool_]
    skyline_profile: NDArray[np.float64]


class SyntheticRenderer:
    """Renders synthetic terrain from a pinhole camera."""

    def render(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
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

        raster = self._rasterize_visible_terrain(terrain, intrinsics, extrinsics, stride=stride)
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
    ) -> NDArray[np.float64]:
        """Rasterizes visible terrain and returns its upper silhouette profile."""

        return self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
        ).skyline_profile

    def visible_mask(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int = 1,
    ) -> NDArray[np.bool_]:
        """Rasterizes visible terrain and returns its boolean coverage mask."""

        return self._rasterize_visible_terrain(
            terrain,
            intrinsics,
            extrinsics,
            stride=stride,
        ).terrain_mask

    def _rasterize_visible_terrain(
        self,
        terrain: TerrainMap,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        stride: int,
    ) -> _RasterizedTerrain:
        """Projects the terrain mesh and rasterizes visible pixels with depth."""

        width = intrinsics.width_px
        height = intrinsics.height_px
        points = terrain.flattened_points(stride=stride)
        u_px, v_px, _depth, valid = project_points(points, intrinsics, extrinsics)

        grid_height = terrain.elevation_m[::stride, ::stride].shape[0]
        grid_width = terrain.elevation_m[::stride, ::stride].shape[1]
        u_grid = u_px.reshape(grid_height, grid_width)
        v_grid = v_px.reshape(grid_height, grid_width)
        valid_grid = valid.reshape(grid_height, grid_width)
        depth_grid = _depth.reshape(grid_height, grid_width)
        inverse_depth_grid = np.zeros_like(u_grid, dtype=np.float64)
        finite_depth = depth_grid > 0.0
        inverse_depth_grid[finite_depth] = 1.0 / depth_grid[finite_depth]

        inverse_depth_buffer = np.full((height, width), -np.inf, dtype=np.float64)
        for row in range(grid_height - 1):
            for col in range(grid_width - 1):
                top_left = self._screen_vertex(row, col, u_grid, v_grid, inverse_depth_grid, valid_grid)
                top_right = self._screen_vertex(row, col + 1, u_grid, v_grid, inverse_depth_grid, valid_grid)
                bottom_left = self._screen_vertex(row + 1, col, u_grid, v_grid, inverse_depth_grid, valid_grid)
                bottom_right = self._screen_vertex(row + 1, col + 1, u_grid, v_grid, inverse_depth_grid, valid_grid)
                self._rasterize_triangle(
                    inverse_depth_buffer,
                    (top_left, top_right, bottom_left),
                )
                self._rasterize_triangle(
                    inverse_depth_buffer,
                    (top_right, bottom_right, bottom_left),
                )

        mask = np.isfinite(inverse_depth_buffer)
        return _RasterizedTerrain(
            terrain_mask=mask,
            skyline_profile=self._profile_from_mask(mask, fallback=float(height * 0.62)),
        )

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

        x_coords = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
        y_coords = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
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
