"""Camera projection tests."""

import math

import numpy as np

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics, ImageCamera
from peakle.domain.coordinates import LocalPoint
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
    camera = ImageCamera(width_px=603, height_px=568, horizontal_fov_deg=26.104988470191543, projection="cyltan")
    expected_focal = camera.width_px / math.radians(camera.horizontal_fov_deg)

    assert math.isclose(camera.focal_length_px(), expected_focal)
    assert math.isclose(camera.pitch_deg_from_vertical_shift_px(97.0), math.degrees(math.atan(97.0 / expected_focal)))
