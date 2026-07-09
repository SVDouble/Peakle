"""Camera projection tests."""

import math

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, CameraModel
from peakle.domain.coordinates import LocalPoint
from peakle.domain.projection import (
    azimuths_deg,
    elevation_rad_from_rows,
    tangent_elevation_from_rows,
    vertical_shift_px_from_pitch_deg,
)
from peakle.rendering.pinhole import project_points


def test_forward_point_projects_to_principal_point() -> None:
    intrinsics = CameraIntrinsics.from_horizontal_fov(
        width_px=640,
        height_px=360,
        horizontal_fov_deg=60.0,
    )
    extrinsics = CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=0.0),
        yaw_deg=0.0,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    points = np.array([[0.0, 100.0, 0.0]], dtype=np.float64)

    u_px, v_px, depth, valid = project_points(points, intrinsics, extrinsics)

    assert valid[0]
    assert depth[0] == 100.0
    assert u_px[0] == intrinsics.principal_x_px
    assert v_px[0] == intrinsics.principal_y_px


def test_cyltan_image_camera_uses_crop_focal_length() -> None:
    camera = CameraModel(width_px=603, height_px=568, horizontal_fov_deg=26.104988470191543, projection="cyltan")
    expected_focal = camera.width_px / math.radians(camera.horizontal_fov_deg)

    assert math.isclose(camera.focal_length_px(), expected_focal)
    assert math.isclose(camera.pitch_deg_from_vertical_shift_px(97.0), math.degrees(math.atan(97.0 / expected_focal)))


def test_cyltan_rows_round_trip_through_shared_projection() -> None:
    camera = CameraModel(width_px=640, height_px=360, horizontal_fov_deg=64.0, projection="cyltan")
    elevation = np.radians(np.asarray([-5.0, 0.0, 12.0]))

    rows = camera.rows_from_elevation_rad(elevation)
    recovered = elevation_rad_from_rows(
        rows,
        camera.width_px,
        camera.height_px,
        camera.horizontal_fov_deg,
        camera.projection,
    )

    assert rows[1] == pytest.approx((camera.height_px - 1) / 2.0)
    assert recovered == pytest.approx(elevation)
    assert tangent_elevation_from_rows(rows, 640, 360, 64.0, "cyltan") == pytest.approx(np.tan(elevation))


def test_pitch_shift_round_trip_uses_projection_model() -> None:
    camera = CameraModel(width_px=690, height_px=460, horizontal_fov_deg=42.0, projection="cyltan")
    shift_px = vertical_shift_px_from_pitch_deg(
        camera.width_px,
        camera.horizontal_fov_deg,
        camera.projection,
        15.0,
    )

    assert camera.pitch_deg_from_vertical_shift_px(shift_px) == pytest.approx(15.0)


def test_pinhole_azimuths_are_symmetric_around_yaw() -> None:
    az = azimuths_deg(width_px=641, horizontal_fov_deg=60.0, yaw_deg=24.0, projection="pinhole")

    assert az[320] == pytest.approx(24.0)
    assert az[0] + az[-1] == pytest.approx(48.0)
