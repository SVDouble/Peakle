"""Validation gates for typed outline extraction: known geometry in, exact families out.

The detector must (1) put a jump edge between two mountains, (2) put a RIB crease along a spur
where depth is continuous but its gradient spikes, (3) tell ribs from couloirs by sign, and
(4) stay SILENT on smooth-but-noisy surfaces — the noise gate is the whole point.
"""

import numpy as np

from peakle.localize.typed_outlines import extract_typed_outlines

H, W = 160, 200


def test_jump_between_two_mountains_is_occlusion_not_crease():
    depth = np.full((H, W), np.nan)
    depth[40:, :] = 8000.0  # Mountain B (behind)
    depth[90:, :] = 900.0  # Mountain A (front) — jump at row 90
    t = extract_typed_outlines(depth)
    rows = np.nonzero(t.occlusion.any(axis=1))[0]
    assert len(rows) and np.all(np.abs(rows - 90) <= 1), rows
    assert not t.crease[85:96].any(), "the jump must not double as a crease"


def test_rib_and_couloir_sign():
    cols = np.arange(W, dtype=float)
    # a RIB (spur) at col 60: depth dips toward the camera in a steep V (oblique face); a COULOIR
    # at col 140: depth bulges away — both are gradient spikes with CONTINUOUS depth.  Slopes are
    # steep (40-45 m/px) as on real oblique faces; gentle kinks belong below the noise floor.
    row = 2500.0 + 40.0 * np.abs(cols - 60.0)
    after = cols > 100
    row[after] = row[100] + 45.0 * (40.0 - np.abs(cols[after] - 140.0))
    depth = np.tile(row, (H, 1))
    t = extract_typed_outlines(depth)
    rib_cols = np.nonzero(t.rib.any(axis=0))[0]
    cl_cols = np.nonzero(t.couloir.any(axis=0))[0]
    assert len(rib_cols) and np.all(np.abs(rib_cols - 60) <= 1), rib_cols
    assert len(cl_cols) and np.all(np.abs(cl_cols - 140) <= 1), cl_cols


def test_smooth_noisy_slope_stays_silent():
    rng = np.random.default_rng(0)
    rr = np.arange(H, dtype=float)[:, None]
    depth = 2000.0 + 15.0 * rr + rng.normal(0.0, 4.0, (H, W))  # smooth slope + sensor noise
    t = extract_typed_outlines(depth)
    assert t.occlusion.sum() == 0
    assert t.crease.sum() < 0.001 * H * W, t.counts()


def test_short_fragments_are_dropped():
    depth = np.full((H, W), 3000.0)
    depth[80, 10:20] = 2000.0  # a 10-px blip — below min component length
    t = extract_typed_outlines(depth)
    assert t.occlusion.sum() == 0 and t.crease.sum() == 0, t.counts()


def test_grazing_lake_surface_stays_silent():
    # a flat lake / valley floor seen nearly edge-on: depth grows hyperbolically down the rows —
    # huge CONTINUOUS gradients whose curvature is viewing geometry, not terrain folds
    rows = np.arange(H, dtype=float)[:, None]
    depth = np.tile(300.0 / np.maximum(1.0 - rows / (H + 12.0), 1e-3), (1, W))
    t = extract_typed_outlines(depth)
    assert t.crease.sum() == 0, t.counts()


def test_no_creases_across_subthreshold_depth_steps():
    # two surfaces separated by a moderate step (below the jump threshold): the step is neither
    # an occlusion nor a fold of ONE surface — "occluded ribs" must not show through
    depth = np.full((H, W), 2000.0)
    depth[:, 100:] = 2400.0  # +20% step: log-jump 0.18 < 0.30
    t = extract_typed_outlines(depth)
    assert t.crease[:, 90:110].sum() == 0, t.counts()


def test_rib_couloir_mutually_exclusive():
    rng = np.random.default_rng(2)
    cols = np.arange(W, dtype=float)
    row = 2500.0 + 40.0 * np.abs(cols - 60.0)
    after = cols > 100
    row[after] = row[100] + 45.0 * (40.0 - np.abs(cols[after] - 140.0))
    depth = np.tile(row, (H, 1)) + rng.normal(0, 1.5, (H, W))
    t = extract_typed_outlines(depth)
    assert not (t.rib & t.couloir).any()


def test_foreground_ground_folds_are_ignored():
    # the camera stands on a slope crest: near ground at 40-200m folds into terrain at 2km —
    # a genuine DEM depth structure that no photograph shows as a feature
    depth = np.full((H, W), 2000.0)
    depth[120:, :] = 60.0  # foreground ground band
    t = extract_typed_outlines(depth)
    assert t.occlusion.sum() == 0 and t.crease.sum() == 0, t.counts()


def test_bilinear_cell_wiggles_are_suppressed_but_isolated_folds_survive():
    # piecewise-linear depth with kinks every 6 px (bilinear DEM cells): alternating +/- second
    # derivatives = balanced rib/couloir twins -> both dropped; an isolated strong V survives
    cols = np.arange(W, dtype=float)
    seg = (cols // 6).astype(int)
    slope = np.where(seg % 2 == 0, 8.0, -8.0)  # zig-zag: alternating kink signs
    zig = 3000.0 + np.cumsum(slope * np.ones(W))
    row = zig + 45.0 * np.abs(cols - 150.0) * (np.abs(cols - 150.0) < 30)  # one real V at 150
    depth = np.tile(row, (H, 1))
    t = extract_typed_outlines(depth)
    rib_cols = np.nonzero(t.rib.any(axis=0))[0]
    assert len(rib_cols) > 0 and np.all(np.abs(rib_cols - 150) <= 3), rib_cols
    other = t.crease.copy()
    other[:, 140:160] = False
    assert other.sum() < 0.001 * H * W, t.counts()
