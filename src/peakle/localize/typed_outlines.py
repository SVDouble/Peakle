"""Typed outline extraction from a depth image — the full visual-attribute set of a terrain view.

Taxonomy (standard depth-edge families, weighted differently at match time):
  skyline    terrain/sky boundary (handled elsewhere; sky = NaN here)
  occlusion  JUMP edges: depth discontinuities — the line between a front mountain and the one
             behind it (type 1)
  rib        CREASE edges, convex: depth continuous, gradient spikes, centre NEARER than flanks —
             counterforts / spurs / pointy arêtes seen inside a face (type 2a)
  couloir    CREASE edges, concave: centre FARTHER than flanks — gullies / couloirs (type 2b)

Noise policy (the DEM and the GT render both carry sampling noise; a crease detector without it
drowns in speckle):
  - median-filter log-depth (3x3) before differentiating;
  - threshold the second-difference response at ``k * MAD`` of the image's own response
    (robust, per-image) with an absolute floor;
  - exclude a band around occlusion jumps (a jump IS a huge gradient spike — different family);
  - drop connected components shorter than ``min_px``.

The same operator runs on the GT depth render (pfm) and on our DEM-rendered depth image, so the
two outline sets are comparable pixel-for-pixel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import label as cc_label
from scipy.ndimage import maximum_filter

JUMP_LOG = 0.30  # |Δ log d| that counts as an occlusion jump (matches gtrefine)
CREASE_K_MAD = 6.0  # crease threshold in robust MADs of the response
CREASE_FLOOR = 0.012  # absolute floor on |second difference of log depth|
MIN_COMPONENT_PX = 25  # drop shorter line fragments (noise)
OCCLUSION_GUARD_PX = 2  # exclusion band around jumps when detecting creases
MIN_NEAR_M = 250.0  # ignore outlines whose near side is closer than this: the foreground
# ground the camera stands on produces genuine DEM depth folds (measured:
# occlusion lines cutting across a flat ski slope) that no photo shows
GRAZE_LOG = 0.08  # max |one-sided log-depth gradient| per px for a crease to be trusted:
# near-grazing surfaces (lakes / valley floors seen edge-on) have huge
# CONTINUOUS gradients whose curvature is viewing geometry, not terrain;
# the same cap rejects creases straddling moderate (sub-jump) depth steps


@dataclass
class TypedOutlines:
    occlusion: np.ndarray  # bool (H, W) — type 1 jump edges
    rib: np.ndarray  # bool (H, W) — type 2a convex creases (spurs / counterforts)
    couloir: np.ndarray  # bool (H, W) — type 2b concave creases (gullies)

    @property
    def crease(self) -> np.ndarray:
        return self.rib | self.couloir

    def counts(self) -> dict[str, int]:
        return {"occlusion": int(self.occlusion.sum()), "rib": int(self.rib.sum()), "couloir": int(self.couloir.sum())}


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = cc_label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes[1:] >= min_px
    return keep[lab]


def extract_typed_outlines(
    depth: np.ndarray,
    jump_log: float = JUMP_LOG,
    k_mad: float = CREASE_K_MAD,
    floor: float = CREASE_FLOOR,
    min_px: int = MIN_COMPONENT_PX,
    min_near_m: float = MIN_NEAR_M,
) -> TypedOutlines:
    """Typed outlines from a depth image (NaN or <=0 = sky).  Scale-free: works on any size."""

    d = depth.astype(float).copy()
    d[d <= 0] = np.nan
    d[d < min_near_m] = np.nan  # foreground exclusion (see MIN_NEAR_M)
    logd = np.log(d)

    # --- type 1: jump edges (first differences of raw log depth) ---
    jump = np.zeros(d.shape, bool)
    dv = np.abs(np.diff(logd, axis=0))
    dh = np.abs(np.diff(logd, axis=1))
    jump[1:, :] |= dv > jump_log
    jump[:, 1:] |= dh > jump_log
    occlusion = _drop_small(jump & np.isfinite(logd), min_px)

    # --- type 2: crease edges — difference of one-sided 2-px-baseline gradients.  A 3x3 median
    # pre-filter would erase single-pixel crease vertices (measured: the couloir response halves
    # below the floor); the wider stencil averages sensor noise while keeping vertices intact ---
    gp_v = np.full(d.shape, np.nan)
    gm_v = np.full(d.shape, np.nan)
    gp_h = np.full(d.shape, np.nan)
    gm_h = np.full(d.shape, np.nan)
    gp_v[2:-2, :] = (logd[4:, :] - logd[2:-2, :]) / 2.0
    gm_v[2:-2, :] = (logd[2:-2, :] - logd[:-4, :]) / 2.0
    gp_h[:, 2:-2] = (logd[:, 4:] - logd[:, 2:-2]) / 2.0
    gm_h[:, 2:-2] = (logd[:, 2:-2] - logd[:, :-4]) / 2.0
    s_v = gp_v - gm_v
    s_h = gp_h - gm_h
    # a crease is trusted only where BOTH flanks are locally continuous and non-grazing: this
    # rejects (a) lakes / valley floors seen edge-on (huge continuous gradients — the curvature
    # is viewing geometry, not a terrain fold) and (b) pixels straddling sub-threshold depth
    # steps between two different surfaces ("occluded ribs" showing through)
    cont_v = (np.abs(gp_v) < GRAZE_LOG) & (np.abs(gm_v) < GRAZE_LOG)
    cont_h = (np.abs(gp_h) < GRAZE_LOG) & (np.abs(gm_h) < GRAZE_LOG)
    # a crease must not sit on/next to a jump (jumps dominate second differences too)
    near_jump = jump.copy()
    for _ in range(OCCLUSION_GUARD_PX + 2):
        grown = near_jump.copy()
        grown[1:, :] |= near_jump[:-1, :]
        grown[:-1, :] |= near_jump[1:, :]
        grown[:, 1:] |= near_jump[:, :-1]
        grown[:, :-1] |= near_jump[:, 1:]
        near_jump = grown
    s_v = np.where(cont_v, s_v, np.nan)
    s_h = np.where(cont_h, s_h, np.nan)
    resp = np.where(np.nan_to_num(np.abs(s_v)) >= np.nan_to_num(np.abs(s_h)), s_v, s_h)
    valid = np.isfinite(resp) & ~near_jump
    mags = np.abs(resp[valid])
    if mags.size == 0:
        empty = np.zeros(d.shape, bool)
        return TypedOutlines(occlusion=occlusion, rib=empty, couloir=empty.copy())
    mad = float(np.median(np.abs(mags - np.median(mags)))) or 1e-9
    th = max(k_mad * mad, floor)
    # centre NEARER than flanks -> log-depth local minimum -> positive second difference = rib
    pos = valid & (resp > th)
    neg = valid & (resp < -th)
    # OPPOSITE-PAIR SUPPRESSION: bilinear DEM cells make the ray-sampled depth piecewise linear,
    # so its second derivative WIGGLES at cell boundaries — equal-and-opposite rib/couloir twins
    # 1-2px apart (measured: 82% of rib px had a couloir within 2px).  A real terrain fold is an
    # isolated single-sign vertex; a wiggle is a balanced pair — drop both members of a pair.
    mag = np.abs(np.nan_to_num(resp))
    pos_mag_near = maximum_filter(np.where(pos, mag, 0.0), size=5)
    neg_mag_near = maximum_filter(np.where(neg, mag, 0.0), size=5)
    rib = _drop_small(pos & (neg_mag_near < 0.6 * mag), min_px)
    couloir = _drop_small(neg & (pos_mag_near < 0.6 * mag), min_px)
    couloir &= ~rib
    return TypedOutlines(occlusion=occlusion, rib=rib, couloir=couloir)
