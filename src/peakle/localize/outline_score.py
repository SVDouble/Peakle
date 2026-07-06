"""Score a photo-side outline extraction against GT v2 (refined skyline + internal contours).

This is the single grading entrance for extraction work: any extractor (colour detectors, SAM3,
edge models, …) produces outline pixels in crop coordinates; this module scores them against the
per-sample GT v2 record, per family:

  - skyline: the terrain/sky boundary (from the GT depth render, pose-refined);
  - internal: occlusion contours between ridge layers (from the GT depth render).

Metrics follow the boundary-matching convention (precision / recall / F at a pixel tolerance,
default 10 px, computed via distance transforms).  Recall is reported per family — an extractor
that nails the skyline but misses every internal ridge must not hide behind a pooled number.

Use ONLY on CLEAN-tier samples (see gt_v2 index `quality`); scoring extraction against SUSPECT
ground truth reintroduces exactly the garbage-optimization loop this layer exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class OutlineScore:
    precision: float          # predicted px near ANY GT outline (skyline or internal)
    recall_skyline: float     # GT skyline px near a prediction
    recall_internal: float    # GT internal-contour px near a prediction
    f1: float                 # harmonic mean of precision and pooled recall
    n_pred: int
    n_gt_skyline: int
    n_gt_internal: int

    def summary(self) -> str:
        return (
            f"P={self.precision:.2f} R_sky={self.recall_skyline:.2f} "
            f"R_int={self.recall_internal:.2f} F1={self.f1:.2f} "
            f"(pred {self.n_pred}px, gt sky {self.n_gt_skyline} / int {self.n_gt_internal}px)"
        )


def load_gt_arrays(gt_npz: str | Path) -> dict:
    z = np.load(gt_npz)
    h, w = (int(v) for v in z["shape"])
    gt_contours = np.unpackbits(z["gt_contours"])[: h * w].reshape(h, w).astype(bool)
    sky_rows = z["gt_skyline"]
    sky_mask = np.zeros((h, w), bool)
    ok = np.isfinite(sky_rows)
    cols = np.arange(w)[ok]
    rows = np.clip(sky_rows[ok].round().astype(int), 0, h - 1)
    sky_mask[rows, cols] = True
    return {"h": h, "w": w, "sky_mask": sky_mask, "internal_mask": gt_contours & ~_dilate(sky_mask, 4)}


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    return distance_transform_edt(~mask) <= r


def score_outlines(pred_mask: np.ndarray, gt_npz: str | Path, tol_px: float = 10.0) -> OutlineScore:
    """``pred_mask``: boolean (H, W) of extracted outline pixels in GT v2 crop coordinates."""

    gt = load_gt_arrays(gt_npz)
    if pred_mask.shape != (gt["h"], gt["w"]):
        raise ValueError(f"pred_mask shape {pred_mask.shape} != GT crop {(gt['h'], gt['w'])}")
    sky, internal = gt["sky_mask"], gt["internal_mask"]
    gt_any = sky | internal

    n_pred = int(pred_mask.sum())
    if n_pred:
        dt_gt = distance_transform_edt(~gt_any)
        precision = float((dt_gt[pred_mask] <= tol_px).mean())
        dt_pred = distance_transform_edt(~pred_mask)
        r_sky = float((dt_pred[sky] <= tol_px).mean()) if sky.any() else 0.0
        r_int = float((dt_pred[internal] <= tol_px).mean()) if internal.any() else 0.0
        r_pool = float((dt_pred[gt_any] <= tol_px).mean())
    else:
        precision = r_sky = r_int = r_pool = 0.0
    f1 = 2 * precision * r_pool / max(precision + r_pool, 1e-9)
    return OutlineScore(
        precision=precision,
        recall_skyline=r_sky,
        recall_internal=r_int,
        f1=f1,
        n_pred=n_pred,
        n_gt_skyline=int(sky.sum()),
        n_gt_internal=int(internal.sum()),
    )


def rows_to_mask(rows: np.ndarray, h: int) -> np.ndarray:
    """Convenience: a per-column skyline curve as an outline mask."""

    w = len(rows)
    mask = np.zeros((h, w), bool)
    ok = np.isfinite(rows) & (rows >= 0) & (rows < h)
    mask[np.clip(rows[ok].round().astype(int), 0, h - 1), np.arange(w)[ok]] = True
    return mask
