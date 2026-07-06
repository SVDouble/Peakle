"""Ground-truth refinement: reconstruct a sample's pose and outline set to pixel-level agreement.

GeoPose3K labels carry measured noise (yaw median 0.8°, up to >3°; GPS median ~100 m, p90 ~178 m;
residual crop tilt up to ~1°).  Scoring or optimising against the raw labels means optimising
against that noise.  This module produces **GT v2** per sample:

  - a POSE POLISH around the label (yaw ±6°, position ±175 m, vertical shift, tilt ±2°), with the
    correction magnitudes reported — a large correction downgrades the sample instead of being
    silently absorbed;
  - the OUTLINE SET, both families: the dataset's own lines (skyline + internal occlusion
    contours extracted from the GT depth render) and our DEM reconstruction of the same lines at
    the refined pose;
  - agreement metrics between the two families (skyline chamfer; contour chamfer via distance
    transform) and a QUALITY TIER — only CLEAN samples may be used to score solvers/extractors.

The far-field skyline is nearly position-blind, so the position refinement additionally matches
the near-field internal contours (their parallax pins the camera); without that term position
stays under-constrained by ±50 m and every drawn contour inherits the error.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt

from peakle.localize.raycast import _elevation_angle_grid, horizon_elevation
from peakle.localize.solve import _best_shift_chamfer

PITCH_LIM_DEG = 50.0
CONTOUR_JUMP = 0.30          # |Δ log distance| that counts as an occlusion boundary
CONTOUR_CAP_PX = 40.0

# quality gates — initial values derived from the bench-60 distribution (skyline cons median 5px,
# p90 8px, max 13px; |dyaw|>3° on 4/60); revisit as the refined corpus grows
GATE_SKY_PX = 15.0
GATE_CONTOUR_PX = 25.0
GATE_DYAW_DEG = 3.0


@dataclass
class RefinedGT:
    name: str
    manual: bool
    # refined pose (label + correction)
    yaw_deg: float               # refined yaw = label + dyaw
    dyaw_deg: float
    de_m: float
    dn_m: float
    cam_z_m: float
    dv_px: float                 # vertical crop offset at the standard width
    tilt_deg: float
    # agreement of the reconstruction with the dataset's own lines
    sky_cons_px: float
    contour_cons_px: float | None
    gt_contour_density: float    # fraction of terrain columns that carry an internal GT contour
    width: int
    height: int
    fov_deg: float
    obs_source: str = "pfm"            # what the pose polish fitted: "photo" (detected skyline,
                                       # preferred — the pfm has registration outliers) or "pfm"
    obs_support: float | None = None   # DexiNed edge support of the chosen observation curve
    pfm_offset_px: float | None = None  # median |pfm skyline - detected skyline| (registration health)
    quality: str = field(default="CLEAN")
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def crop_az_deg(w: int, fov_deg: float, yaw_deg: float) -> np.ndarray:
    return yaw_deg + np.degrees((np.arange(w) - (w - 1) / 2.0) * (math.radians(fov_deg) / w))


def rows_from_el(el: np.ndarray, w: int, h: int, fov_deg: float) -> np.ndarray:
    """TRUE cylindrical mapping: rows linear in tan(elevation), f = W/hfov both axes."""

    f = w / math.radians(fov_deg)
    return (h - 1) / 2.0 - f * np.tan(el)


def dem_skyline(terrain, cam_z, az_deg, w, h, fov_deg, de=0.0, dn=0.0, patch=None) -> np.ndarray:
    step = 10.0 if patch is not None else 25.0   # the fine patch deserves a finer ray march
    el = horizon_elevation(terrain, np.radians(az_deg), cam_z, step=step, cam_e=de, cam_n=dn, patch=patch)
    return rows_from_el(el, w, h, fov_deg)


def gt_contour_mask(depth: np.ndarray, w: int, h: int, jump: float = CONTOUR_JUMP) -> np.ndarray:
    """TRUE internal contours from the GT depth render (sky = NaN, excluded automatically)."""

    d = depth.astype(float).copy()
    d[d <= 0] = np.nan
    logd = np.log(d)
    edge = np.zeros_like(d, bool)
    edge[1:, :] |= np.abs(np.diff(logd, axis=0)) > jump
    edge[:, 1:] |= np.abs(np.diff(logd, axis=1)) > jump
    h0, w0 = d.shape
    mask = np.zeros((h, w), bool)
    rr, cc = np.nonzero(edge)
    mask[np.clip((rr * h / h0).astype(int), 0, h - 1), np.clip((cc * w / w0).astype(int), 0, w - 1)] = True
    return mask


def dem_depth_image(terrain, cam_z, az_deg, w, h, fov_deg, dv, de=0.0, dn=0.0, tilt_deg=0.0, sub=2, patch=None):
    """Per-pixel visible-terrain distance in crop coordinates at the given (drawn) alignment.

    Returns ``(depth, hit_idx, el_pix_all, el, ds, rows_s)`` on the sub-sampled pixel grid; depth
    is NaN for sky.  This is the shared substrate for occlusion contours, typed creases, and the
    GT-Lab depth layer."""

    az_s = az_deg[::sub]
    step = 10.0 if patch is not None else 25.0
    el, ds = _elevation_angle_grid(terrain, np.radians(az_s), cam_z, step=step, d_max=None, cam_e=de, cam_n=dn, patch=patch)
    cummax = np.maximum.accumulate(el, axis=1)
    f = w / math.radians(fov_deg)
    rows_s = np.arange(0, h, sub)
    cols_c = np.arange(0, w, sub, dtype=float) - (w - 1) / 2.0
    tilt_dv = math.tan(math.radians(tilt_deg)) * cols_c
    n_r, n_c = len(rows_s), len(az_s)
    depth = np.full((n_r, n_c), np.nan)
    hit_idx = np.full((n_r, n_c), -1, int)
    el_pix_all = np.empty((n_r, n_c))
    for c in range(n_c):
        el_pix = np.arctan(((h - 1) / 2.0 + dv + tilt_dv[c] - rows_s) / f)
        el_pix_all[:, c] = el_pix
        env = cummax[c]
        idx = np.searchsorted(env, el_pix)
        hit = idx < len(ds)
        hit_idx[hit, c] = idx[hit]
        i = np.clip(idx[hit], 0, len(ds) - 1)
        # interpolate the exact envelope-crossing distance: snapping to the 25m ray-march bin
        # quantises depth into terraces, and the crease detector then traces every terrace edge
        # as alternating rib/couloir bands ("isolines" — user-reported)
        i0 = np.maximum(i - 1, 0)
        e0, e1 = env[i0], env[i]
        t = np.where(e1 > e0, (el_pix[hit] - e0) / np.where(e1 > e0, e1 - e0, 1.0), 1.0)
        depth[hit, c] = ds[i0] + np.clip(t, 0.0, 1.0) * (ds[i] - ds[i0])
    return depth, hit_idx, el_pix_all, el, ds, rows_s


def dem_contour_mask(
    terrain, cam_z, az_deg, w, h, fov_deg, dv, de=0.0, dn=0.0, tilt_deg=0.0, sub=2, jump=CONTOUR_JUMP,
    gap_eps_rad=0.0035,
) -> np.ndarray:
    """DEM occlusion boundaries in crop coordinates at the given (drawn) alignment.

    Per column the visible-surface distance per pixel row comes from the first crossing of the
    ray's elevation envelope.  A candidate |Δ log d| jump only counts as OCCLUSION if the terrain
    dips below the ray between the near and far hits (the "gap test") — a smooth slope seen at
    grazing incidence also produces huge continuous depth gradients, and without the gap test
    those false outlines dominate (caught by the synthetic slope unit test)."""

    depth, hit_idx, el_pix_all, el, ds, rows_s = dem_depth_image(
        terrain, cam_z, az_deg, w, h, fov_deg, dv, de, dn, tilt_deg, sub
    )
    n_r, n_c = depth.shape

    logd = np.log(depth)
    edge = np.zeros((n_r, n_c), bool)
    # vertical candidates: pixel r sees NEAR surface, pixel r-1 (above) sees FAR surface
    cand = np.abs(np.diff(logd, axis=0)) > jump
    rr, cc = np.nonzero(cand)
    for r, c in zip(rr.tolist(), cc.tolist()):
        i_lo = min(hit_idx[r, c], hit_idx[r + 1, c])
        i_hi = max(hit_idx[r, c], hit_idx[r + 1, c])
        if i_lo < 0 or i_hi - i_lo < 3:
            continue
        ray_el = min(el_pix_all[r, c], el_pix_all[r + 1, c])
        if np.min(el[c, i_lo + 1 : i_hi]) < ray_el - gap_eps_rad:   # true gap behind the near crest
            edge[r, c] = edge[r + 1, c] = True
    # horizontal candidates (vertical silhouettes): same gap test across neighbouring columns
    cand_h = np.abs(np.diff(logd, axis=1)) > jump
    rr, cc = np.nonzero(cand_h)
    for r, c in zip(rr.tolist(), cc.tolist()):
        ia, ib = hit_idx[r, c], hit_idx[r, c + 1]
        if ia < 0 or ib < 0 or abs(ib - ia) < 3:
            continue
        c_near = c if ia < ib else c + 1
        i_lo, i_hi = min(ia, ib), max(ia, ib)
        if np.min(el[c_near, i_lo + 1 : i_hi]) < el_pix_all[r, c_near] - gap_eps_rad:
            edge[r, c] = edge[r, c + 1] = True

    mask = np.zeros((h, w), bool)
    rr, cc = np.nonzero(edge)
    mask[np.clip(rows_s[rr], 0, h - 1), np.clip(cc * sub, 0, w - 1)] = True
    return mask


def contour_chamfer(mask: np.ndarray, ref_dt: np.ndarray, cap: float = CONTOUR_CAP_PX) -> float:
    """Mean capped distance from mask pixels to the nearest reference-contour pixel."""

    rr, cc = np.nonzero(mask)
    if len(rr) < 50:
        return cap
    return float(np.minimum(ref_dt[rr, cc], cap).mean())


def shift_align(obs: np.ndarray, rows: np.ndarray, dvs: np.ndarray, step: int = 6) -> tuple[float, float]:
    """Two-stage vertical-shift chamfer: the coarse dv grid alone carries ±(grid/2) px of pure
    quantisation noise — measured to make a WRONG pose score better than the exact truth when the
    truth's offset fell between grid points."""

    c, dv = _best_shift_chamfer(obs, rows, dvs, 60.0, shift_step=step)
    return _best_shift_chamfer(obs, rows, dv + np.arange(-7.0, 7.01, 1.0), 60.0, shift_step=step)


def refine_pose(terrain, cam_z, obs, w, h, fov_deg, yaw_label, gt_dt=None):
    """Joint local pose polish; see module docstring.  Returns a dict of the fit."""

    f = w / math.radians(fov_deg)
    dv_lim = int(f * math.tan(math.radians(PITCH_LIM_DEG))) + 1
    dvs = np.arange(-dv_lim, dv_lim + 1, 8)

    def align(dyaw, de, dn, step=6):
        z = max(cam_z, terrain.elevation_at(de, dn) + 2.0)
        rows = dem_skyline(terrain, z, crop_az_deg(w, fov_deg, yaw_label + dyaw), w, h, fov_deg, de, dn)
        c, dv = shift_align(obs, rows, dvs, step=step)
        return c, dv, rows

    # JOINT coarse grid over yaw × position, keeping EVERY evaluation — yaw and position
    # compensate each other through near-field structure (a 3° yaw error ≈ 150 m lateral at
    # 3 km), so on smooth terrain several pose basins fit the far-field skyline equally well
    # (measured 0.5 px at a 3°-wrong pose on the synthetic).  The skyline alone cannot pick
    # between them; the near-field CONTOURS arbitrate below.
    evals: list[tuple[float, float, float, float, float]] = []  # (c, dyaw, de, dn, dv)

    def track(dyaw, de, dn, step=6):
        c, dv, rows = align(dyaw, de, dn, step)
        evals.append((c, dyaw, de, dn, dv))
        return c, dv, rows

    best = (math.inf, 0.0, 0.0, 0.0, 0.0, None)
    for dyaw in np.linspace(-6.0, 6.0, 9):
        for de in (-100.0, 0.0, 100.0):
            for dn in (-100.0, 0.0, 100.0):
                c, dv, rows = track(float(dyaw), de, dn)
                if c < best[0]:
                    best = (c, float(dyaw), de, dn, dv, rows)
    for de in best[2] + np.asarray((-50.0, -25.0, 0.0, 25.0, 50.0)):
        for dn in best[3] + np.asarray((-50.0, -25.0, 0.0, 25.0, 50.0)):
            c, dv, rows = track(best[1], float(de), float(dn))
            if c < best[0]:
                best = (c, best[1], float(de), float(dn), dv, rows)
    for dyaw in best[1] + np.asarray((-0.75, -0.375, 0.0, 0.375, 0.75)):
        for de in (best[2] - 25.0, best[2], best[2] + 25.0):
            for dn in (best[3] - 25.0, best[3], best[3] + 25.0):
                c, dv, rows = track(float(dyaw), float(de), float(dn), step=3)
                if c < best[0]:
                    best = (c, float(dyaw), float(de), float(dn), dv, rows)
    cons, dyaw, de, dn, dv, rows = best

    ccons = None
    if gt_dt is not None:
        # candidate poses that fit the skyline nearly as well as the winner, well separated —
        # the contour term picks among them (this is where the degeneracy is broken)
        cand = [best[:5]]
        for c, dy_c, de_c, dn_c, dv_c in sorted(evals):
            if c > max(2.0 * cons, cons + 4.0) or len(cand) >= 8:
                break
            if all(abs(dy_c - p[1]) > 0.5 or (abs(de_c - p[2]) + abs(dn_c - p[3])) > 50.0 for p in cand):
                cand.append((c, dy_c, de_c, dn_c, dv_c))

        def combined(dy_c, de_c, dn_c):
            z_c = max(cam_z, terrain.elevation_at(de_c, dn_c) + 2.0)
            rows_c = dem_skyline(terrain, z_c, crop_az_deg(w, fov_deg, yaw_label + dy_c), w, h, fov_deg, de_c, dn_c)
            c_sky, dv_c = shift_align(obs, rows_c, dvs, step=6)
            dem_m = dem_contour_mask(
                terrain, z_c, crop_az_deg(w, fov_deg, yaw_label + dy_c), w, h, fov_deg, dv_c, de_c, dn_c, sub=4
            )
            c_ct = contour_chamfer(dem_m, gt_dt)
            return c_sky + 0.6 * c_ct, c_sky, c_ct, dv_c, rows_c

        best_e = None
        for _, dy_c, de_c, dn_c, _ in cand:
            # each candidate gets its own mini position refinement (skyline, cheap) BEFORE the
            # contour comparison — otherwise the incumbent, already position-refined, wins by
            # refinement depth rather than by actually matching the contours
            loc = (math.inf, de_c, dn_c)
            for de_r in (de_c - 25.0, de_c, de_c + 25.0):
                for dn_r in (dn_c - 25.0, dn_c, dn_c + 25.0):
                    c_r, _, _ = align(dy_c, float(de_r), float(dn_r))
                    if c_r < loc[0]:
                        loc = (c_r, float(de_r), float(dn_r))
            score, c_sky, c_ct, dv_c, rows_c = combined(dy_c, loc[1], loc[2])
            if best_e is None or score < best_e[0]:
                best_e = (score, dy_c, loc[1], loc[2], c_sky, c_ct, dv_c, rows_c)
        # local contour-constrained position refinement around the contour-chosen candidate
        for de_c in best_e[2] + np.asarray((-50.0, -25.0, 0.0, 25.0, 50.0)):
            for dn_c in best_e[3] + np.asarray((-50.0, -25.0, 0.0, 25.0, 50.0)):
                score, c_sky, c_ct, dv_c, rows_c = combined(best_e[1], float(de_c), float(dn_c))
                if score < best_e[0]:
                    best_e = (score, best_e[1], float(de_c), float(dn_c), c_sky, c_ct, dv_c, rows_c)
        _, dyaw, de, dn, cons, ccons, dv, rows = best_e

    # residual tilt (imperfect roll rectification), applied last
    tilt = 0.0
    resid = obs - (rows + dv)
    ok = np.isfinite(resid) & (np.abs(resid - np.nanmedian(resid)) < 25.0)
    if ok.sum() > 0.3 * w:
        cols = np.arange(w, dtype=float) - (w - 1) / 2.0
        slope = float(np.polyfit(cols[ok], resid[ok], 1)[0])
        tilt = float(np.clip(math.degrees(math.atan(slope)), -2.0, 2.0))
        rows_t = rows + math.tan(math.radians(tilt)) * cols
        c2, dv2 = shift_align(obs, rows_t, dvs, step=3)
        if c2 < cons:
            cons, dv, rows = c2, dv2, rows_t
        else:
            tilt = 0.0

    z = max(cam_z, terrain.elevation_at(de, dn) + 2.0)
    return {
        "cons": float(cons), "ccons": (float(ccons) if ccons is not None else None),
        "dyaw": float(dyaw), "de": float(de), "dn": float(dn), "dv": float(dv),
        "tilt": float(tilt), "cam_z": float(z), "rows": rows,
    }


def quality_tier(
    sky_cons: float, contour_cons: float | None, dyaw: float, extra_reasons: list[str] | None = None
) -> tuple[str, list[str]]:
    """CLEAN samples may score solvers/extractors; everything else is quarantined with reasons."""

    reasons = list(extra_reasons or [])
    if sky_cons > GATE_SKY_PX:
        reasons.append(f"skyline reconstruction {sky_cons:.0f}px > {GATE_SKY_PX:.0f}")
    if contour_cons is not None and contour_cons > GATE_CONTOUR_PX:
        reasons.append(f"contour agreement {contour_cons:.0f}px > {GATE_CONTOUR_PX:.0f}")
    if abs(dyaw) > GATE_DYAW_DEG:
        reasons.append(f"label yaw off by {dyaw:+.1f}° > ±{GATE_DYAW_DEG:.0f}")
    return ("CLEAN" if not reasons else "SUSPECT"), reasons
