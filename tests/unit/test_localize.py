"""Unit tests for peakle.localize: synthetic round-trip recovery and the honesty gate.

The round-trip is the T2 acceptance criterion in-package: render a skyline from a known pose on
synthetic terrain, feed it to the solver, and require the pose back.  The honesty gate is the
inverse requirement: on terrain whose skyline carries no yaw information the solver must NOT
report a confident verdict, whatever the residual says.
"""

import numpy as np
import pytest

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.localize.raycast import _distance_samples, skyline_cyl, skyline_pinhole
from peakle.localize.solve import HorizonProfile, solve_orientation


class SynthTerrain:
    """Random smooth bumpy terrain in a local-ENU grid, camera at the origin."""

    def __init__(self, seed: int = 7, extent_m: float = 24000.0, grid: int = 320):
        rng = np.random.default_rng(seed)
        self.x_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        self.y_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        xx, yy = np.meshgrid(self.x_m, self.y_m)
        elev = np.zeros((grid, grid))
        for _ in range(40):
            cx, cy = rng.uniform(-extent_m / 2, extent_m / 2, 2)
            amp = rng.uniform(200, 1500)
            sig = rng.uniform(600, 3000)
            elev += amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig**2))
        self.elevation_m = elev


class ConeTerrain:
    """Radially symmetric terrain: the skyline is IDENTICAL at every yaw (pure ambiguity)."""

    def __init__(self, extent_m: float = 24000.0, grid: int = 320):
        self.x_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        self.y_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        xx, yy = np.meshgrid(self.x_m, self.y_m)
        r = np.hypot(xx, yy)
        self.elevation_m = np.clip((r - 4000.0) * 0.08, 0.0, 900.0)


W, H = 640, 480


def _ground0(t) -> float:
    """Terrain height at the origin — the camera must sit ABOVE it (the buried-camera lesson)."""

    g = len(t.x_m) // 2
    return float(t.elevation_m[g, g])


def _pose(yaw, pitch, cam_z):
    return CameraExtrinsics(
        position=LocalPoint(east_m=0.0, north_m=0.0, up_m=cam_z), yaw_deg=yaw, pitch_deg=pitch, roll_deg=0.0
    )


def test_adaptive_raycast_samples_close_terrain_more_densely():
    coarse = _distance_samples(20_000.0, 25.0)
    fine = _distance_samples(20_000.0, 5.0, patch=object())

    assert np.diff(coarse[coarse < 4_000.0]).max() <= 10.0
    assert np.diff(fine[fine < 4_000.0]).max() <= 5.0
    assert np.diff(fine[fine > 12_000.0]).min() >= 25.0


@pytest.fixture(scope="module")
def terrain():
    return SynthTerrain()


@pytest.fixture(scope="module")
def cam_z(terrain):
    return _ground0(terrain) + 60.0


@pytest.fixture(scope="module")
def profile(terrain, cam_z):
    return HorizonProfile(terrain, cam_z, step=40.0)


def test_roundtrip_cyl(terrain, profile, cam_z):
    gt_yaw, gt_pitch, fov = 137.0, 4.0, 60.0
    obs = skyline_cyl(terrain, W, H, fov, _pose(gt_yaw, gt_pitch, cam_z), step=40.0)
    s = solve_orientation(obs, H, profile, fov_deg=fov, projection="cyl")
    assert abs((s.yaw_deg - gt_yaw + 180) % 360 - 180) <= 1.5, s.summary()
    assert abs(s.pitch_deg - gt_pitch) <= 1.0, s.summary()
    assert s.chamfer_px < 5.0, s.summary()


def test_roundtrip_pinhole(terrain, profile, cam_z):
    gt_yaw, gt_pitch, fov = 295.0, -6.0, 55.0
    intr = CameraIntrinsics.from_horizontal_fov(W, H, fov)
    obs = skyline_pinhole(terrain, intr, _pose(gt_yaw, gt_pitch, cam_z), step=40.0)
    s = solve_orientation(obs, H, profile, fov_deg=fov, projection="pinhole")
    assert abs((s.yaw_deg - gt_yaw + 180) % 360 - 180) <= 1.5, s.summary()
    assert abs(s.pitch_deg - gt_pitch) <= 1.0, s.summary()


def test_roundtrip_fov_search(terrain, profile, cam_z):
    gt_yaw, gt_pitch, fov = 61.0, 2.0, 62.0
    obs = skyline_cyl(terrain, W, H, fov, _pose(gt_yaw, gt_pitch, cam_z), step=40.0)
    s = solve_orientation(obs, H, profile, fov_deg=(50.0, 74.0, 4.0), projection="cyl")
    assert abs((s.yaw_deg - gt_yaw + 180) % 360 - 180) <= 3.0, s.summary()
    assert abs(s.fov_deg - fov) <= 4.0, s.summary()


def test_gap_tolerance(terrain, profile, cam_z):
    """Solver still recovers the pose when 35% of the observed skyline is missing."""

    gt_yaw, gt_pitch, fov = 210.0, 1.0, 60.0
    obs = skyline_cyl(terrain, W, H, fov, _pose(gt_yaw, gt_pitch, cam_z), step=40.0)
    rng = np.random.default_rng(3)
    holes = rng.random(W) < 0.35
    obs[holes] = np.nan
    s = solve_orientation(obs, H, profile, fov_deg=fov, projection="cyl")
    assert abs((s.yaw_deg - gt_yaw + 180) % 360 - 180) <= 1.5, s.summary()


def test_ambiguous_terrain_is_not_confirmed():
    """On a radially symmetric skyline every yaw fits equally well — the verdict MUST reflect it."""

    cone = ConeTerrain()
    cz = _ground0(cone) + 60.0
    prof = HorizonProfile(cone, cz, step=40.0)
    obs = skyline_cyl(cone, W, H, 60.0, _pose(45.0, 0.0, cz), step=40.0)
    s = solve_orientation(obs, H, prof, fov_deg=60.0, projection="cyl")
    assert s.verdict != "CONFIRMED", s.summary()
    assert s.well_width_deg > 45.0 or s.alias_ratio < 1.1, s.summary()


def test_dp_skyline_follows_strong_edge_chain():
    """The DP trace must follow a continuous high-score chain and ignore brighter isolated
    specks (clouds) that a per-column argmax would jump to."""

    import numpy as np

    from peakle.localize.extract import _dp_skyline

    h, w = 120, 200
    score = np.zeros((h, w))
    true_rows = (60 + 12 * np.sin(np.linspace(0, 3, w))).astype(int)
    score[true_rows, np.arange(w)] = 0.6
    rng = np.random.default_rng(0)
    score[rng.integers(5, 25, 40), rng.integers(0, w, 40)] = 0.9  # bright isolated specks above
    rows = _dp_skyline(score)
    err = np.abs(rows - true_rows)
    assert np.median(err) <= 1.0 and err.max() <= 6.0, (np.median(err), err.max())
