"""Depth-aware contour classification for noise removal (with ridge preservation).

A traced contour is one of three things, told apart by the monocular-depth profile *across*
it (sampled a few pixels to each side along the local normal):

* **silhouette** — depth jumps across it: one flank is much nearer than the other (a layer
  edge / occluding ridge). Keep.
* **crest** — depth peaks *on* it and recedes on BOTH flanks: a convex ridge sitting inside
  a depth band, with both sides captured. Keep. (This is the case the user flagged: a ridge
  in the middle of a sector must be preserved, not deleted as "noise in the middle of a
  cluster".)
* **texture** — depth is flat across and along it: rock/snow texture inside one surface.
  Remove — but only in the **foreground**, where Depth-Anything is reliable. Distant ranges
  are depth-flattened, so there we keep everything (handled by edge/silhouette evidence
  upstream) rather than risk deleting a real faint background ridge.

`classify_contour` returns the scores; `keep_contour` applies the decision; both operate on a
normalised depth map (larger = nearer, the Depth-Anything convention).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Poly = list[tuple[float, float]]  # [(y, x), ...]


def normalize_depth(depth: NDArray[np.float64], lo: float = 2.0, hi: float = 98.0) -> NDArray[np.float64]:
    a, b = np.percentile(depth, lo), np.percentile(depth, hi)
    return np.clip((depth - a) / max(b - a, 1e-6), 0.0, 1.0)


def _normals(poly: Poly) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    pts = np.asarray(poly, float)
    tang = np.zeros_like(pts)
    tang[1:-1] = pts[2:] - pts[:-2]
    tang[0], tang[-1] = pts[1] - pts[0], pts[-1] - pts[-2]
    nrm = np.hypot(tang[:, 0], tang[:, 1])[:, None]
    tang = tang / np.where(nrm == 0, 1.0, nrm)
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)  # rotate tangent by 90°
    return pts, normal


def _sample(depth: NDArray[np.float64], pts: NDArray[np.float64]) -> NDArray[np.float64]:
    h, w = depth.shape
    y = np.clip(np.round(pts[:, 0]).astype(int), 0, h - 1)
    x = np.clip(np.round(pts[:, 1]).astype(int), 0, w - 1)
    return depth[y, x]


def classify_contour(
    poly: Poly, depthn: NDArray[np.float64], offset: float = 8.0, crest_eps: float = 0.04
) -> dict[str, float]:
    """Depth-profile scores for one contour (depthn = nearness-normalised depth in [0,1])."""
    if len(poly) < 3:
        d = _sample(depthn, np.asarray(poly, float))
        return {"mean_depth": float(d.mean()), "jump": 0.0, "crest_frac": 0.0}
    pts, n = _normals(poly)
    d_on = _sample(depthn, pts)
    d_l = _sample(depthn, pts + n * offset)
    d_r = _sample(depthn, pts - n * offset)
    jump = np.abs(d_l - d_r)
    # crest: nearer on the ridge than on either flank by a margin (both flanks recede)
    crest = (d_on >= d_l) & (d_on >= d_r) & ((d_on - np.minimum(d_l, d_r)) > crest_eps)
    return {
        "mean_depth": float(d_on.mean()),
        "jump": float(np.median(jump)),
        "crest_frac": float(crest.mean()),
    }


def _plen(poly: Poly) -> float:
    p = np.asarray(poly, float)
    return float(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])).sum()) if len(p) > 1 else 0.0


def _sky_prox(poly: Poly, skyline: NDArray[np.float64], band: float = 10.0) -> float:
    p = np.asarray(poly, float)
    x = np.clip(np.round(p[:, 1]).astype(int), 0, len(skyline) - 1)
    return float(np.mean(np.abs(p[:, 0] - skyline[x]) < band))


def keep_by_ridge_signal(
    poly: Poly,
    depthn: NDArray[np.float64],
    sil: NDArray[np.float64],
    skyline: NDArray[np.float64],
    sky_t: float = 0.15,
    crest_t: float = 0.06,
    jump_t: float = 0.05,
    sil_t: float = 0.45,
    len_t: float = 55.0,
    offset: float = 8.0,
) -> tuple[bool, str]:
    """Keep a contour if it carries ANY validated ridge signal; discard only contours that fail
    EVERY one (= clearly texture). The signals — skyline proximity, depth crest (both flanks
    recede), depth jump (silhouette), SAM3 region-boundary support, and length — were the
    features that empirically separate on-annotation ridges from texture; discarding by edge
    strength does NOT (every candidate is already edge-backed). This raises precision (cleaner)
    without dropping recall, unlike a blanket depth-texture removal."""
    if _plen(poly) >= len_t:
        return True, "long"
    if _sky_prox(poly, skyline) >= sky_t:
        return True, "skyline"
    c = classify_contour(poly, depthn, offset=offset)
    if c["jump"] >= jump_t:
        return True, "silhouette"
    if c["crest_frac"] >= crest_t:
        return True, "crest"
    p = np.asarray(poly, float)
    y = np.clip(np.round(p[:, 0]).astype(int), 0, sil.shape[0] - 1)
    x = np.clip(np.round(p[:, 1]).astype(int), 0, sil.shape[1] - 1)
    if float(sil[y, x].mean()) >= sil_t:
        return True, "region"
    return False, "texture"


def filter_by_ridge_signal(
    polys: list[Poly], depthn: NDArray[np.float64], sil: NDArray[np.float64],
    skyline: NDArray[np.float64], **kw
) -> tuple[list[Poly], list[str]]:
    """Apply `keep_by_ridge_signal` to a list; return (kept, label-per-kept)."""
    kept, labels = [], []
    for p in polys:
        ok, label = keep_by_ridge_signal(p, depthn, sil, skyline, **kw)
        if ok:
            kept.append(p)
            labels.append(label)
    return kept, labels
