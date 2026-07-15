"""Subpixel lifting tests for matcher-ready terrain renders."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import GeoPoint, LocalPoint
from peakle.domain.terrain import TerrainMap, TerrainSpec
from peakle.rendering.pinhole import camera_axes, project_points
from peakle.rendering.rasterizer import SyntheticRenderer, _ScreenVertex
from peakle.rendering.terrain_view import (
    PIXEL_LIFTING_DEPTH_METHOD,
    PIXEL_LIFTING_SCHEMA,
    TerrainRenderBundle,
    TerrainViewRenderer,
    lift_render_pixels,
    unproject_pinhole_depth,
    world_normals_to_camera,
)


def _sloped_triangle_bundle() -> tuple[TerrainRenderBundle, np.ndarray]:
    """Rasterize one triangle whose inverse depth varies in both image axes."""

    height = width = 9
    inverse_depth = np.full((height, width), -np.inf, dtype=np.float64)
    triangle = (
        _ScreenVertex(u_px=1.0, v_px=1.0, inverse_depth=0.05, valid=True),
        _ScreenVertex(u_px=7.0, v_px=1.0, inverse_depth=0.14, valid=True),
        _ScreenVertex(u_px=1.0, v_px=7.0, inverse_depth=0.23, valid=True),
    )
    SyntheticRenderer()._rasterize_triangle(inverse_depth, triangle)
    terrain_mask = np.isfinite(inverse_depth) & (inverse_depth > 0.0)
    depth = np.full_like(inverse_depth, np.nan)
    depth[terrain_mask] = 1.0 / inverse_depth[terrain_mask]
    intrinsics = CameraIntrinsics.from_horizontal_fov(width, height, 65.0)
    extrinsics = CameraExtrinsics(
        position=LocalPoint(east_m=3.0, north_m=-4.0, up_m=2.0),
        yaw_deg=17.0,
        pitch_deg=-8.0,
        roll_deg=3.0,
    )
    skyline = np.full(width, np.nan, dtype=np.float64)
    for column in range(width):
        rows = np.flatnonzero(terrain_mask[:, column])
        if rows.size:
            skyline[column] = float(rows[0])
    bundle = TerrainRenderBundle(
        rgb=np.zeros((height, width, 3), dtype=np.uint8),
        forward_depth_m=depth,
        world_xyz_m=unproject_pinhole_depth(depth, intrinsics, extrinsics),
        world_normals=np.zeros((height, width, 3), dtype=np.float64),
        terrain_mask=terrain_mask,
        appearance_mask=terrain_mask.copy(),
        skyline_profile=skyline,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        modality="hillshade",
        provenance={"schema": "synthetic_sloped_triangle_test_v1"},
    )
    return bundle, inverse_depth


def test_subpixel_lifting_follows_sloped_triangle_inverse_depth_and_round_trips() -> None:
    bundle, rasterized_inverse_depth = _sloped_triangle_bundle()
    safe_xy = np.asarray([[2.25, 2.25]], dtype=np.float64)

    lifted = lift_render_pixels(bundle, safe_xy, max_relative_depth_span=5.0)

    finite_inverse_depth = np.where(np.isfinite(rasterized_inverse_depth), rasterized_inverse_depth, np.nan)
    expected_inverse_depth = map_coordinates(
        finite_inverse_depth,
        [safe_xy[:, 1], safe_xy[:, 0]],
        order=1,
        mode="constant",
        cval=np.nan,
    )[0]
    expected_depth = 1.0 / expected_inverse_depth
    forward_depth_interpolation = map_coordinates(
        bundle.forward_depth_m,
        [safe_xy[:, 1], safe_xy[:, 0]],
        order=1,
        mode="constant",
        cval=np.nan,
    )[0]

    assert lifted.valid.tolist() == [True]
    assert lifted.forward_depth_m[0] == pytest.approx(expected_depth, abs=1e-12)
    assert abs(forward_depth_interpolation - expected_depth) > 0.1

    u_px, v_px, depth_m, projected = project_points(
        lifted.world_xyz_m,
        bundle.intrinsics,
        bundle.extrinsics,
    )
    assert projected.tolist() == [True]
    assert np.column_stack((u_px, v_px)) == pytest.approx(safe_xy, abs=1e-10)
    assert depth_m[0] == pytest.approx(expected_depth, abs=1e-12)


def test_subpixel_lifting_keeps_nan_triangle_boundary_invalid() -> None:
    bundle, _inverse_depth = _sloped_triangle_bundle()
    boundary_xy = np.asarray([[4.5, 3.5]], dtype=np.float64)

    lifted = lift_render_pixels(bundle, boundary_xy, max_relative_depth_span=5.0)

    assert lifted.valid.tolist() == [False]
    assert np.isnan(lifted.forward_depth_m[0])
    assert np.all(np.isnan(lifted.world_xyz_m[0]))
    assert lifted.rejection_counts["invalid_or_sky_depth"] == 1


def test_render_provenance_declares_inverse_depth_subpixel_lifting() -> None:
    size = 32
    x_m = np.linspace(-100.0, 100.0, size)
    y_m = np.linspace(10.0, 210.0, size)
    x_grid, y_grid = np.meshgrid(x_m, y_m)
    terrain = TerrainMap(
        spec=TerrainSpec(
            origin=GeoPoint(latitude_deg=46.0, longitude_deg=8.0, elevation_m=0.0),
            width_m=200.0,
            height_m=200.0,
            grid_width=size,
            grid_height=size,
            min_elevation_m=0.0,
            max_elevation_m=1.0,
            seed=0,
        ),
        x_m=x_m,
        y_m=y_m,
        elevation_m=np.zeros((size, size), dtype=np.float64),
        latitude_deg=46.0 + y_grid / 111_320.0,
        longitude_deg=8.0 + x_grid / 80_000.0,
    )
    intrinsics = CameraIntrinsics.from_horizontal_fov(33, 25, 65.0)
    extrinsics = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=20.0),
        yaw_deg=0.0,
        pitch_deg=-10.0,
        roll_deg=0.0,
    )

    rendered = TerrainViewRenderer().render(terrain, intrinsics, extrinsics)

    assert rendered.provenance["schema"] == "terrain_render_bundle_v4"
    assert rendered.provenance["subpixel_lifting"] == {
        "schema": PIXEL_LIFTING_SCHEMA,
        "depth_interpolation": PIXEL_LIFTING_DEPTH_METHOD,
        "world_point": "exact keypoint ray at interpolated camera-forward depth",
        "invalid_support": "any non-finite or non-positive bilinear depth support is rejected",
    }


def test_camera_normal_representation_uses_right_down_forward_axes() -> None:
    extrinsics = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=0.0),
        yaw_deg=37.0,
        pitch_deg=-12.0,
        roll_deg=8.0,
    )
    world = np.asarray([[0.0, 0.0, 1.0], [0.2, -0.4, 0.8]], dtype=np.float64)

    camera = world_normals_to_camera(world, extrinsics)

    axes = camera_axes(extrinsics)
    expected = np.asarray([[np.dot(normal, axis) for axis in axes] for normal in world])
    assert camera == pytest.approx(expected, abs=1e-12)
    assert not np.allclose(camera, world)
