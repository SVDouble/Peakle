"""Validation gates for the GT-refinement layer (peakle.localize.gtrefine).

Everything downstream (benchmark scoring, extractor evaluation, optimizer tuning) trusts GT v2,
so GT v2 itself must be validated on synthetic cases with known answers:

  1. contour extraction from a depth image finds exactly the occlusion jump, never the sky edge;
  2. DEM contour reconstruction produces an occlusion boundary between two ridges and stays
     silent on smooth single-slope terrain (no false outlines);
  3. the pose polish is a round trip: perturb the label by a known (yaw, position) error and the
     polish must report that error back within its grid resolution.
"""

import math

import numpy as np
import pytest

from peakle.localize.gtrefine import (
    contour_chamfer,
    crop_az_deg,
    dem_contour_mask,
    dem_skyline,
    gt_contour_mask,
    refine_pose,
    rows_from_el,
)
from scipy.ndimage import distance_transform_edt

W, H = 480, 360
FOV = 60.0


class TwoRidgeTerrain:
    """A near ridge (E-W wall at y=+2km) in front of a far, higher ridge (y=+8km):
    guaranteed occlusion boundary between them when viewed from the origin looking north."""

    def __init__(self, extent_m: float = 24000.0, grid: int = 320):
        self.x_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        self.y_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        xx, yy = np.meshgrid(self.x_m, self.y_m)
        near = 400.0 * np.exp(-((yy - 2000.0) ** 2) / (2 * 400.0**2))
        far = 1500.0 * np.exp(-((yy - 8000.0) ** 2) / (2 * 900.0**2))
        self.elevation_m = near + far

    def elevation_at(self, e, n):
        j = np.interp(e, self.x_m, np.arange(len(self.x_m)))
        i = np.interp(n, self.y_m, np.arange(len(self.y_m)))
        return float(self.elevation_m[int(round(i)), int(round(j))])


class SlopeTerrain:
    """A single smooth slope: a skyline but NO internal occlusion boundaries."""

    def __init__(self, extent_m: float = 24000.0, grid: int = 320):
        self.x_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        self.y_m = np.linspace(-extent_m / 2, extent_m / 2, grid)
        _, yy = np.meshgrid(self.x_m, self.y_m)
        self.elevation_m = np.clip(yy, 0, None) * 0.12

    def elevation_at(self, e, n):
        j = np.interp(e, self.x_m, np.arange(len(self.x_m)))
        i = np.interp(n, self.y_m, np.arange(len(self.y_m)))
        return float(self.elevation_m[int(round(i)), int(round(j))])


def test_gt_contour_mask_finds_the_jump_not_the_sky():
    depth = np.full((100, 80), np.nan)         # sky
    depth[60:, :] = 500.0                       # near slab
    depth[40:60, :] = 5000.0                    # far slab behind it
    mask = gt_contour_mask(depth, 80, 100)
    rows = np.nonzero(mask.any(axis=1))[0]
    assert len(rows) > 0
    # the only internal jump is at the 500<->5000 boundary (row 60); the sky edge (row 40) is NOT
    assert np.all(np.abs(rows - 60) <= 2), rows
    assert not mask[38:44].any(), "sky boundary must not be reported as an internal contour"


def test_dem_contours_present_between_ridges_absent_on_smooth_slope():
    az = crop_az_deg(W, FOV, 0.0)               # looking north
    two = TwoRidgeTerrain()
    mask2 = dem_contour_mask(two, 50.0, az, W, H, FOV, dv=0.0)
    slope = SlopeTerrain()
    mask1 = dem_contour_mask(slope, 50.0, az, W, H, FOV, dv=0.0)
    assert mask2.sum() > 200, "occluding ridge pair must produce an internal contour"
    assert mask1.sum() < mask2.sum() * 0.05, f"smooth slope must be (nearly) contour-free, got {mask1.sum()}"
    # per column, the contour must sit BELOW that column's skyline (occlusion is internal)
    sky = dem_skyline(two, 50.0, az, W, H, FOV)
    rr, cc = np.nonzero(mask2)
    assert np.all(rr >= sky[cc] - 2.0)


def test_refine_pose_roundtrip_recovers_injected_label_error():
    rng = np.random.default_rng(11)
    terrain = TwoRidgeTerrain()
    # structure at MANY depths, including SHARP cones: all-Gaussian terrain is unidentifiable —
    # smooth curves slide along themselves, so a 3°-wrong pose fits skyline AND contours at ~1 px
    # (measured).  Real skylines have peaks; the synthetic must too or the test asserts noise.
    xx, yy = np.meshgrid(terrain.x_m, terrain.y_m)
    for _ in range(40):
        cx = rng.uniform(-14000, 14000)
        cy = rng.uniform(2000, 15000) * rng.choice((-1.0, 1.0))
        sig = rng.uniform(800, 2500)
        terrain.elevation_m += rng.uniform(150, 700) * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig**2))
    for _ in range(14):
        cx = rng.uniform(-12000, 12000)
        cy = rng.uniform(2500, 15000) * rng.choice((-1.0, 1.0))
        r = rng.uniform(300, 800)
        amp = rng.uniform(400, 1000)
        dist = np.hypot(xx - cx, yy - cy)
        terrain.elevation_m += amp * np.clip(1.0 - dist / r, 0.0, None)

    true_yaw, true_de, true_dn, cam_z = 14.0, 50.0, -75.0, 120.0
    az_true = crop_az_deg(W, FOV, true_yaw)
    obs = dem_skyline(terrain, cam_z, az_true, W, H, FOV, de=true_de, dn=true_dn) - 40.0  # -40px crop offset

    # skyline-only fitting is DEGENERATE here (a 3°-wrong pose fits at 0.5 px — measured), so the
    # production path always supplies the GT internal contours; they arbitrate between the
    # skyline-equivalent pose candidates.  Build them exactly as GT depth would provide them:
    gt_mask = dem_contour_mask(TwoRidgeTerrain() if False else terrain, cam_z, az_true, W, H, FOV,
                               dv=-40.0, de=true_de, dn=true_dn)
    assert gt_mask.sum() > 100, "test terrain must have internal contours for the production path"
    gt_dt = distance_transform_edt(~gt_mask)

    # label is wrong by +2 deg yaw and (0,0) position; the polish must find the truth
    fit = refine_pose(terrain, cam_z, obs, W, H, FOV, yaw_label=true_yaw + 2.0, gt_dt=gt_dt)
    assert abs((fit["dyaw"] + 2.0)) <= 0.75, fit["dyaw"]          # recovered dyaw ≈ -2.0
    # position is identifiable only to ~50 m on this terrain (measured: a 50 m-off position
    # scores the same contour chamfer as the truth) — the tolerance reflects that floor
    assert abs(fit["de"] - true_de) <= 60.0 and abs(fit["dn"] - true_dn) <= 60.0, (fit["de"], fit["dn"])
    assert fit["cons"] < 3.0, fit["cons"]
    assert abs(fit["dv"] - (-40.0)) <= 6.0, fit["dv"]
    assert fit["ccons"] < 6.0, fit["ccons"]


def test_contour_chamfer_scores_alignment():
    a = np.zeros((100, 100), bool)
    a[50, 10:90] = True
    dt = distance_transform_edt(~a)
    aligned = np.zeros_like(a)
    aligned[52, 10:90] = True                   # 2px off
    shifted = np.zeros_like(a)
    shifted[80, 10:90] = True                   # 30px off
    assert contour_chamfer(aligned, dt) <= 3.0
    assert contour_chamfer(shifted, dt) >= 25.0


def test_rows_from_el_tan_mapping():
    el = np.radians(np.array([0.0, 10.0, 20.0]))
    r = rows_from_el(el, W, H, FOV)
    f = W / math.radians(FOV)
    # amplitudes grow super-linearly with elevation (tan), anchored at the centre row
    assert r[0] == pytest.approx((H - 1) / 2.0)
    assert (r[0] - r[2]) > 2.0 * (r[0] - r[1]) * 0.98
    assert r[0] - r[1] == pytest.approx(f * math.tan(math.radians(10.0)))
