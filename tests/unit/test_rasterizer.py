"""Synthetic rasterizer tests."""

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.rendering.rasterizer import (
    HeightfieldGrid,
    SyntheticRenderer,
    _CameraVertex,
    _clip_polygon_to_near_plane,
    _ScreenVertex,
    _stride_indices,
)


def _flat_terrain(elevation_m: float) -> TerrainMap:
    size = 32
    x_m = np.linspace(-100.0, 100.0, size)
    y_m = np.linspace(10.0, 210.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    spec = TerrainSpec(
        origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=elevation_m),
        width_m=200.0,
        height_m=200.0,
        grid_width=size,
        grid_height=size,
        min_elevation_m=elevation_m,
        max_elevation_m=elevation_m + 1.0,
        seed=0,
    )
    return TerrainMap(
        spec=spec,
        x_m=x_m,
        y_m=y_m,
        elevation_m=np.full((size, size), elevation_m),
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )


def _downward_camera() -> tuple[CameraIntrinsics, CameraExtrinsics]:
    return (
        CameraIntrinsics.from_horizontal_fov(121, 101, 70.0),
        CameraExtrinsics(
            position=LocalPoint(east_m=0.0, north_m=0.0, up_m=20.0),
            yaw_deg=0.0,
            pitch_deg=-20.0,
            roll_deg=0.0,
        ),
    )


def test_rasterizer_prefers_nearer_triangle_for_overlapping_pixels() -> None:
    """A hidden projected triangle must not overwrite nearer visible terrain."""

    renderer = SyntheticRenderer()
    inverse_depth_buffer = np.full((12, 12), -np.inf, dtype=np.float64)
    far_triangle = (
        _ScreenVertex(u_px=2.0, v_px=2.0, inverse_depth=0.1, valid=True),
        _ScreenVertex(u_px=10.0, v_px=2.0, inverse_depth=0.1, valid=True),
        _ScreenVertex(u_px=6.0, v_px=10.0, inverse_depth=0.1, valid=True),
    )
    near_triangle = (
        _ScreenVertex(u_px=2.0, v_px=2.0, inverse_depth=0.4, valid=True),
        _ScreenVertex(u_px=10.0, v_px=2.0, inverse_depth=0.4, valid=True),
        _ScreenVertex(u_px=6.0, v_px=10.0, inverse_depth=0.4, valid=True),
    )

    renderer._rasterize_triangle(inverse_depth_buffer, far_triangle)
    renderer._rasterize_triangle(inverse_depth_buffer, near_triangle)

    assert np.isclose(inverse_depth_buffer[5, 6], 0.4)


def test_strided_mesh_keeps_the_exact_terrain_boundaries() -> None:
    assert _stride_indices(8, 3).tolist() == [0, 3, 6, 7]
    assert _stride_indices(7, 3).tolist() == [0, 3, 6]


def test_near_plane_clipping_retains_the_visible_part_of_a_triangle() -> None:
    clipped = _clip_polygon_to_near_plane(
        (
            _CameraVertex(right_m=-4.0, down_m=4.0, forward_m=0.5),
            _CameraVertex(right_m=4.0, down_m=4.0, forward_m=5.0),
            _CameraVertex(right_m=0.0, down_m=-4.0, forward_m=5.0),
        )
    )

    assert len(clipped) == 4
    assert [vertex.forward_m for vertex in clipped] == [1.0, 1.0, 5.0, 5.0]
    assert clipped[0].right_m == pytest.approx(-3.5555555556)
    assert clipped[1].right_m == pytest.approx(-3.1111111111)


def test_lower_native_surface_authoritatively_replaces_higher_regional_depth() -> None:
    renderer = SyntheticRenderer()
    regional = _flat_terrain(0.0)
    fine_reference = _flat_terrain(-10.0)
    intrinsics, extrinsics = _downward_camera()
    native = HeightfieldGrid(
        x_m=fine_reference.x_m,
        y_m=fine_reference.y_m,
        elevation_m=fine_reference.elevation_m,
    )

    regional_render = renderer.geometry(regional, intrinsics, extrinsics)
    fine_render = renderer.geometry(fine_reference, intrinsics, extrinsics)
    composited = renderer.geometry(regional, intrinsics, extrinsics, overlay=native)

    covered = composited.native_overlay_mask
    assert covered.any()
    assert np.all(fine_render.terrain_mask[covered])
    assert np.allclose(
        composited.forward_depth_m[covered],
        fine_render.forward_depth_m[covered],
    )
    shared = covered & regional_render.terrain_mask
    assert shared.any()
    # The lower surface is farther along these downward camera rays. A normal
    # nearest-depth competition would incorrectly preserve the coarse surface.
    assert np.any(composited.forward_depth_m[shared] > regional_render.forward_depth_m[shared] + 1.0)
    regional_fallback = regional_render.terrain_mask & ~covered
    assert regional_fallback.any()
    assert np.allclose(
        composited.forward_depth_m[regional_fallback],
        regional_render.forward_depth_m[regional_fallback],
    )


def test_nearer_regional_ridge_outside_patch_footprint_occludes_native_surface() -> None:
    renderer = SyntheticRenderer()
    background = _flat_terrain(-10.0)
    intrinsics, extrinsics = _downward_camera()
    ridge_profile = 20.0 * np.exp(-(((background.y_m - 30.0) / 12.0) ** 2)) - 10.0
    ridge_elevation = np.repeat(ridge_profile[:, None], len(background.x_m), axis=1)
    ridge_terrain = background.model_copy(
        update={
            "spec": background.spec.model_copy(update={"min_elevation_m": -10.0, "max_elevation_m": 11.0}),
            "elevation_m": ridge_elevation,
        }
    )
    native_rows = np.flatnonzero(background.y_m >= 80.0)
    native = HeightfieldGrid(
        x_m=background.x_m,
        y_m=background.y_m[native_rows],
        elevation_m=np.full((len(native_rows), len(background.x_m)), -20.0),
    )

    without_ridge = renderer.geometry(background, intrinsics, extrinsics, overlay=native)
    with_ridge = renderer.geometry(ridge_terrain, intrinsics, extrinsics, overlay=native)
    ridge_only = renderer.geometry(ridge_terrain, intrinsics, extrinsics)
    occluded_native = without_ridge.native_overlay_mask & ~with_ridge.native_overlay_mask & ridge_only.terrain_mask

    assert occluded_native.any()
    assert np.allclose(
        with_ridge.forward_depth_m[occluded_native],
        ridge_only.forward_depth_m[occluded_native],
    )
    assert np.all(with_ridge.forward_depth_m[occluded_native] < without_ridge.forward_depth_m[occluded_native])
    ridge_inverse_depth = np.full_like(ridge_only.forward_depth_m, -np.inf)
    ridge_visible = np.isfinite(ridge_only.forward_depth_m)
    ridge_inverse_depth[ridge_visible] = 1.0 / ridge_only.forward_depth_m[ridge_visible]
    _ridge_east_m, ridge_north_m = renderer._unproject_buffer_xy(
        ridge_inverse_depth,
        intrinsics,
        extrinsics,
    )
    assert np.max(ridge_north_m[occluded_native]) < native.y_m[0]


def test_native_nodata_keeps_the_complete_regional_fallback() -> None:
    renderer = SyntheticRenderer()
    regional = _flat_terrain(0.0)
    intrinsics, extrinsics = _downward_camera()
    nodata = HeightfieldGrid(
        x_m=regional.x_m,
        y_m=regional.y_m,
        elevation_m=np.full_like(regional.elevation_m, np.nan),
    )

    regional_render = renderer.geometry(regional, intrinsics, extrinsics)
    composited = renderer.geometry(regional, intrinsics, extrinsics, overlay=nodata)

    assert not composited.native_overlay_mask.any()
    assert np.array_equal(composited.terrain_mask, regional_render.terrain_mask)
    assert np.array_equal(
        composited.forward_depth_m,
        regional_render.forward_depth_m,
        equal_nan=True,
    )
