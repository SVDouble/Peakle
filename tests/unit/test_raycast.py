"""Focused contracts for the terrain ray-march extent."""

from types import SimpleNamespace

import numpy as np
import pytest

from peakle.localize.raycast import EARTH_R, _distance_samples, horizon_elevation, horizon_elevation_and_distance


def test_distance_samples_include_the_explicit_terminal_range():
    assert _distance_samples(200.0, 30.0)[-1] == 200.0


def test_default_range_reaches_a_diagonal_corner_feature():
    """The implicit range must not discard terrain beyond half the DEM side length."""

    axis = np.linspace(-100.0, 100.0, 201)
    east, north = np.meshgrid(axis, axis)
    along_diagonal = (east + north) / np.sqrt(2.0)
    across_diagonal = (east - north) / np.sqrt(2.0)
    elevation = np.where((along_diagonal >= 120.0) & (np.abs(across_diagonal) <= 4.0), 100.0, 0.0)
    terrain = SimpleNamespace(x_m=axis, y_m=axis, elevation_m=elevation)

    rendered = horizon_elevation(terrain, np.asarray([np.pi / 4.0]), cam_z=0.0, step=1.0)[0]
    expected = np.arctan2(100.0 - 120.0**2 / (2.0 * EARTH_R), 120.0)

    assert rendered == pytest.approx(expected, abs=0.01)


def test_default_range_covers_an_offset_camera_but_an_explicit_cap_does_not():
    """Default coverage is camera-position independent; callers can still request a short ray."""

    x_m = np.linspace(-120.0, 180.0, 301)
    y_m = np.linspace(-80.0, 140.0, 221)
    east, north = np.meshgrid(x_m, y_m)
    cam_e, cam_n = -80.0, -40.0
    bearing = np.arctan2(240.0, 160.0)
    along_ray = (east - cam_e) * np.sin(bearing) + (north - cam_n) * np.cos(bearing)
    across_ray = (east - cam_e) * np.cos(bearing) - (north - cam_n) * np.sin(bearing)
    elevation = np.where((along_ray >= 270.0) & (np.abs(across_ray) <= 5.0), 120.0, 0.0)
    terrain = SimpleNamespace(x_m=x_m, y_m=y_m, elevation_m=elevation)

    rendered, distance = horizon_elevation_and_distance(
        terrain,
        np.asarray([bearing]),
        cam_z=0.0,
        cam_e=cam_e,
        cam_n=cam_n,
        step=1.0,
    )
    capped = horizon_elevation(
        terrain,
        np.asarray([bearing]),
        cam_z=0.0,
        cam_e=cam_e,
        cam_n=cam_n,
        step=1.0,
        d_max=200.0,
    )[0]
    expected = np.arctan2(120.0 - 270.0**2 / (2.0 * EARTH_R), 270.0)

    assert rendered[0] == pytest.approx(expected, abs=0.01)
    assert distance[0] == pytest.approx(270.0, abs=2.0)
    assert capped == pytest.approx(0.0, abs=1e-6)
