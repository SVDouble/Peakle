"""Weighted explanation of photo contours by typed DEM outlines — the disambiguation objective.

The idea (user-specified): extract contours from the real image, then prefer the pose whose DEM
outline set gives most of those contours a REASONABLE EXPLANATION — with family weights, the
skyline counting most:

    skyline 1.0    occlusion (type 1) 0.6    ribs / couloirs (type 2) 0.3

For each photo-contour pixel the credit is the weight of the best explaining family within a
pixel tolerance (a pixel explained by both the skyline and a rib earns the skyline's 1.0).
``explanation_score`` = mean credit over photo pixels ∈ [0, 1].  Unexplained photo contours
(vegetation, buildings, rock texture) earn 0 and drag the score down, which is exactly what
disambiguates two poses whose skylines fit equally well: the wrong one leaves the internal
structure unexplained.

``rerank_candidates`` applies this to a solver's polished pose candidates: DEM typed outlines are
rendered per candidate and the explanation score breaks skyline-chamfer ties.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from peakle.localize.gtrefine import crop_az_deg, dem_contour_mask, dem_depth_image
from peakle.localize.typed_outlines import extract_typed_outlines

WEIGHTS = {"sky": 1.0, "occ": 0.6, "rib": 0.3, "cou": 0.3}
EXPLAIN_TOL_PX = 8.0


def explanation_score(
    photo_mask: np.ndarray,
    dem_masks: dict[str, np.ndarray],
    tol_px: float = EXPLAIN_TOL_PX,
    weights: dict[str, float] = WEIGHTS,
) -> dict:
    """Weighted fraction of photo-contour pixels explained by the typed DEM outline set."""

    n = int(photo_mask.sum())
    if n == 0:
        return {"score": 0.0, "n_photo_px": 0, "explained": {k: 0.0 for k in weights}}
    credit = np.zeros(photo_mask.shape, float)
    explained = {}
    for fam in sorted(weights, key=weights.get, reverse=True):
        mask = dem_masks.get(fam)
        if mask is None or not mask.any():
            explained[fam] = 0.0
            continue
        near = distance_transform_edt(~mask) <= tol_px
        hit = photo_mask & near
        explained[fam] = float(hit.sum() / n)
        np.maximum(credit, np.where(near, weights[fam], 0.0), out=credit)
    return {"score": float(credit[photo_mask].mean()), "n_photo_px": n, "explained": explained}


def dem_typed_masks(
    terrain, cam_z, w, h, fov_deg, yaw_deg, dv, de=0.0, dn=0.0, tilt_deg=0.0, sub=4
) -> dict[str, np.ndarray]:
    """Typed DEM outline masks (photo coordinates) at one pose hypothesis."""

    az = crop_az_deg(w, fov_deg, yaw_deg)
    depth, *_ = dem_depth_image(terrain, cam_z, az, w, h, fov_deg, dv, de, dn, tilt_deg, sub=sub)
    typed = extract_typed_outlines(depth, min_px=max(6, 25 // sub))
    occl = dem_contour_mask(terrain, cam_z, az, w, h, fov_deg, dv, de, dn, tilt_deg, sub=sub)

    def up(mask_s: np.ndarray) -> np.ndarray:
        full = np.zeros((h, w), bool)
        rr, cc = np.nonzero(mask_s)
        full[np.clip(rr * sub, 0, h - 1), np.clip(cc * sub, 0, w - 1)] = True
        return full

    sky = np.zeros((h, w), bool)
    finite = np.isfinite(depth)
    has = finite.any(axis=0)
    top = np.argmax(finite[:, has], axis=0)
    sky[np.clip(top * sub, 0, h - 1), np.nonzero(has)[0] * sub] = True
    return {"sky": sky, "occ": occl, "rib": up(typed.rib), "cou": up(typed.couloir)}


def arbitrate_by_explanation(
    solved: dict,
    terrain,
    cam_z: float,
    w: int,
    h: int,
    fov_deg: float,
    photo_edges: np.ndarray,
    chamfer_slack: float = 1.2,
) -> tuple[str, dict]:
    """Pick the winning skyline hypothesis by weighted explanation, chamfer-gated.

    ``solved`` maps a hypothesis name to ``(candidate, OrientationSolve)``.  Among hypotheses whose
    skyline chamfer is within ``chamfer_slack`` of the best, the one whose typed DEM outlines best
    EXPLAIN the photo edges wins — this breaks the lake/false-skyline tie a chamfer-only winner
    loses (a water edge fits the skyline but leaves the real ridges unexplained).  The tight 1.2
    slack is load-bearing: on GeoPose3K bench-60 it rescued 3 lake cases with 0 regressions, while
    a looser 1.6 slack pulled wrong candidates into the pool and broke 2 correct samples.

    Returns ``(winning_name, per_hypothesis_scores)``.
    """

    if not solved:
        return "", {}
    best_cham = min(s.chamfer_px for _, s in solved.values())
    scores = {}
    for name, (_c, s) in solved.items():
        if s.chamfer_px > chamfer_slack * max(best_cham, 1e-9):
            continue
        cham, yaw, dv = s.candidates[0] if s.candidates else (s.chamfer_px, s.yaw_deg, 0.0)
        masks = dem_typed_masks(terrain, cam_z, w, h, fov_deg, yaw, dv, sub=4)
        scores[name] = explanation_score(photo_edges, masks)["score"]
    if not scores:  # nothing in the slack window (shouldn't happen — best is always in)
        return min(solved.items(), key=lambda kv: kv[1][1].chamfer_px)[0], {}
    win = max(scores, key=scores.get)
    return win, scores


def rerank_candidates(
    terrain,
    cam_z,
    w,
    h,
    fov_deg,
    candidates,
    photo_mask,
    de=0.0,
    dn=0.0,
    tilt_deg=0.0,
    chamfer_slack: float = 1.3,
    max_eval: int = 6,
) -> list[dict]:
    """Score a solver's polished (chamfer, yaw, dv) candidates by weighted explanation.

    Only candidates within ``chamfer_slack`` of the best chamfer compete (the skyline residual
    stays primary); returns candidate dicts sorted by explanation score, best first.
    """

    if not candidates:
        return []
    c_best = candidates[0][0]
    pool = [c for c in candidates if c[0] <= chamfer_slack * max(c_best, 1e-9)][:max_eval]
    out = []
    for cham, yaw, dv in pool:
        masks = dem_typed_masks(terrain, cam_z, w, h, fov_deg, yaw, dv, de, dn, tilt_deg)
        rep = explanation_score(photo_mask, masks)
        out.append({"yaw": yaw, "dv": dv, "chamfer": cham, **rep})
    out.sort(key=lambda r: -r["score"])
    return out
