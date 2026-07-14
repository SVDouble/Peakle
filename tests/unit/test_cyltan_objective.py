"""Projection regressions for full-pose solves on GeoPose crops."""

from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.projection import azimuths_deg, rows_from_elevation_rad
from peakle.optimization.horizon import _observed_descriptor
from peakle.optimization.objective import PoseObjective, dense_terrain_points
from peakle.optimization.solve import solve_pose
from peakle.rendering.point_skyline import cyltan_point_skyline, project_cyltan_points
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.rendering.skyline import contour_from_profile
from peakle.scene.scene import Scene


def _camera_facing_peak(scene: Scene) -> CameraExtrinsics:
    peak = max(scene.peaks, key=lambda candidate: candidate.prominence_m)
    east_m = peak.local_position.east_m - 2000.0
    north_m = peak.local_position.north_m - 3000.0
    yaw_deg = math.degrees(math.atan2(peak.local_position.east_m - east_m, peak.local_position.north_m - north_m))
    return CameraExtrinsics(
        position=LocalPoint(
            east_m=east_m,
            north_m=north_m,
            up_m=scene.terrain.elevation_at(east_m, north_m) + 20.0,
        ),
        yaw_deg=yaw_deg,
        pitch_deg=3.0,
        roll_deg=0.0,
    )


def _prior(extrinsics: CameraExtrinsics) -> PosePrior:
    return PosePrior(
        position=extrinsics.position,
        yaw_deg=extrinsics.yaw_deg,
        pitch_deg=extrinsics.pitch_deg,
        horizontal_sigma_m=100.0,
        vertical_sigma_m=50.0,
        yaw_sigma_deg=15.0,
        pitch_sigma_deg=10.0,
    )


def test_cyltan_point_projection_matches_shared_camera_geometry_across_north_wrap(scene: Scene) -> None:
    intrinsics = scene.intrinsics
    fov_deg = 50.0
    camera = CameraExtrinsics(
        position=LocalPoint(east_m=10.0, north_m=20.0, up_m=100.0),
        yaw_deg=179.0,
        pitch_deg=8.0,
        roll_deg=0.0,
    )
    point_azimuth_deg = -179.0  # two degrees right of a +179 degree camera
    elevation_deg = 15.0
    horizontal_m = 1000.0
    azimuth_rad = math.radians(point_azimuth_deg)
    point = np.asarray(
        [
            [
                camera.position.east_m + horizontal_m * math.sin(azimuth_rad),
                camera.position.north_m + horizontal_m * math.cos(azimuth_rad),
                camera.position.up_m + horizontal_m * math.tan(math.radians(elevation_deg)),
            ]
        ],
        dtype=np.float64,
    )

    u_px, v_px, distance_m, valid = project_cyltan_points(point, intrinsics, camera, fov_deg)
    expected_u = (intrinsics.width_px - 1) / 2.0 + intrinsics.width_px / fov_deg * 2.0
    expected_v = rows_from_elevation_rad(
        np.radians([elevation_deg]),
        intrinsics.width_px,
        intrinsics.height_px,
        fov_deg,
        "cyltan",
        pitch_deg=camera.pitch_deg,
    )

    assert valid.tolist() == [True]
    assert distance_m[0] == pytest.approx(horizontal_m)
    assert u_px[0] == pytest.approx(expected_u)
    assert v_px == pytest.approx(expected_v)


def test_cyltan_objective_prefers_the_pose_that_generated_the_skyline(scene: Scene) -> None:
    extrinsics = _camera_facing_peak(scene)
    fov_deg = scene.config.horizontal_fov_deg
    points = dense_terrain_points(scene.terrain, factor=2, stride=1)
    observed = cyltan_point_skyline(points, scene.intrinsics, extrinsics, fov_deg)
    objective = PoseObjective(
        terrain=scene.terrain,
        observed_profile=observed,
        intrinsics=scene.intrinsics,
        prior=_prior(extrinsics),
        renderer=SyntheticRenderer(),
        terrain_stride=1,
        use_position_prior=False,
        use_orientation_prior=False,
        projection="cyltan",
        horizontal_fov_deg=fov_deg,
    )
    truth = objective.theta_from_prior()
    wrong_yaw = truth.copy()
    wrong_yaw[3] += 10.0

    assert objective.score(truth) == pytest.approx(0.0)
    assert objective.score(wrong_yaw) > 50.0


def test_non_horizon_cyltan_solve_uses_projection_aware_trace_and_final_metrics(
    scene: Scene, monkeypatch: pytest.MonkeyPatch
) -> None:
    extrinsics = _camera_facing_peak(scene)
    fov_deg = scene.config.horizontal_fov_deg
    points = dense_terrain_points(scene.terrain, factor=2, stride=1)
    observed = cyltan_point_skyline(points, scene.intrinsics, extrinsics, fov_deg)
    contour = contour_from_profile(observed, scene.intrinsics.height_px, source="cyltan-test")
    solve_module = importlib.import_module("peakle.optimization.solve")

    def record_prior(traced) -> None:
        traced.cost(traced.objective.theta_from_prior())
        traced.record()

    def reject_pinhole_render(*_args, **_kwargs):
        raise AssertionError("cyltan solve passed through the pinhole mesh renderer")

    monkeypatch.setattr(solve_module, "_run_nelder_mead", record_prior)
    monkeypatch.setattr(SyntheticRenderer, "skyline_profile", reject_pinhole_render)

    result = solve_pose(
        terrain=scene.terrain,
        contour=contour,
        intrinsics=scene.intrinsics,
        prior=_prior(extrinsics),
        strategy="nelder",
        terrain_stride=1,
        projection="cyltan",
        horizontal_fov_deg=fov_deg,
    )

    assert result.trace
    assert result.estimate.metrics.contour_mae_px == pytest.approx(0.0)
    assert np.asarray(result.predicted_profile_full) == pytest.approx(observed)
    assert np.asarray(result.trace[-1].profile) == pytest.approx(observed)


def test_prior_free_horizon_descriptor_inverts_cyltan_rows(scene: Scene) -> None:
    width = scene.intrinsics.width_px
    profile = np.full(width, np.nan, dtype=np.float64)
    columns = np.asarray([20, width // 2, width - 21])
    elevations_deg = np.asarray([-3.0, 6.0, 12.0])
    pitch_deg = 7.0
    fov_deg = 54.0
    profile[columns] = rows_from_elevation_rad(
        np.radians(elevations_deg),
        width,
        scene.intrinsics.height_px,
        fov_deg,
        "cyltan",
        pitch_deg=pitch_deg,
    )

    offsets, recovered_elevations = _observed_descriptor(
        profile,
        scene.intrinsics,
        pitch_deg,
        n_bins=360,
        max_columns=20,
        projection="cyltan",
        horizontal_fov_deg=fov_deg,
    )
    expected_azimuths = azimuths_deg(width, fov_deg, 0.0, "cyltan")[columns]

    assert offsets == pytest.approx(np.round(expected_azimuths).astype(int))
    assert recovered_elevations == pytest.approx(elevations_deg)


def test_pinhole_objective_keeps_existing_fast_skyline(scene: Scene) -> None:
    extrinsics = _camera_facing_peak(scene)
    renderer = SyntheticRenderer()
    objective = PoseObjective(
        terrain=scene.terrain,
        observed_profile=np.zeros(scene.intrinsics.width_px),
        intrinsics=scene.intrinsics,
        prior=_prior(extrinsics),
        renderer=renderer,
        terrain_stride=1,
    )

    expected = renderer.fast_skyline(objective._points, scene.intrinsics, extrinsics)

    assert np.array_equal(objective.predict_profile(extrinsics), expected)


def test_objective_wraps_population_yaw_and_uses_shortest_prior_delta(scene: Scene) -> None:
    extrinsics = _camera_facing_peak(scene)
    prior = _prior(extrinsics).model_copy(update={"yaw_deg": 179.0})
    objective = PoseObjective(
        terrain=scene.terrain,
        observed_profile=np.zeros(scene.intrinsics.width_px),
        intrinsics=scene.intrinsics,
        prior=prior,
        renderer=SyntheticRenderer(),
        terrain_stride=1,
    )
    theta = objective.theta_from_prior()
    theta[3] = 541.0

    assert objective.extrinsics_from_theta(theta).yaw_deg == pytest.approx(-179.0)
    # 541 degrees and the +179-degree prior are only two degrees apart.
    assert objective._prior_penalty(theta) < 1.0


def test_objective_penalizes_camera_below_local_ground(scene: Scene) -> None:
    extrinsics = _camera_facing_peak(scene)
    objective = PoseObjective(
        terrain=scene.terrain,
        observed_profile=np.zeros(scene.intrinsics.width_px),
        intrinsics=scene.intrinsics,
        prior=_prior(extrinsics),
        renderer=SyntheticRenderer(),
        terrain_stride=1,
        use_position_prior=False,
        use_orientation_prior=False,
    )
    theta = objective.theta_from_prior()
    ground_m = scene.terrain.elevation_at(float(theta[0]), float(theta[1]))
    theta[2] = ground_m - 1.0

    assert objective._prior_penalty(theta) >= 625.0
