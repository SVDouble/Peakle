"""Photo-edge support: does a terrain outline actually exist in the photograph?

For each outline family (skyline / occlusion / rib / couloir, from either the GT depth or our
DEM), support = the fraction of line pixels that have a strong learned image edge (DexiNed)
within a small tolerance.  Low support on a GT line means the dataset's own render disagrees
with the photograph there (wrong label pose, changed scenery, water level, vegetation) — such
lines must be down-weighted both when grading extractors and when the matcher explains photo
contours (a GT line the photo cannot show is not an explanation failure).

This metric flags cases like the eth_ch1_39298990 near ridge automatically: GT and DEM agree
with each other but both miss the photo edge — GT occlusion support drops while skyline support
stays high.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt

EDGE_TH = 0.30  # DexiNed response that counts as an image edge
SUPPORT_TOL_PX = 6.0  # a line pixel is supported if an edge lies within this radius


@lru_cache(maxsize=1)
def _detector():
    from peakle.edges import load_learned_edges

    return load_learned_edges()


def edge_mask(rgb_uint8: np.ndarray, th: float = EDGE_TH) -> np.ndarray | None:
    """Learned edge mask for a photo; None when torch/kornia are unavailable."""

    det = _detector()
    if det is None:
        return None
    try:
        emap = det.detect(rgb_uint8.astype(np.float64) / 255.0)  # detect() expects RGB in [0, 1]
    except RuntimeError:  # CUDA OOM on a shared GPU — skip edge layers rather than fail the page
        return None
    return emap >= th


def family_support(line_mask: np.ndarray, edges: np.ndarray, tol_px: float = SUPPORT_TOL_PX) -> float | None:
    """Fraction of line pixels with an image edge within ``tol_px``; None for empty lines."""

    n = int(line_mask.sum())
    if n == 0:
        return None
    if not edges.any():
        return 0.0
    dt = distance_transform_edt(~edges)
    return float((dt[line_mask] <= tol_px).mean())


def support_report(masks: dict[str, np.ndarray], edges: np.ndarray) -> dict[str, float | None]:
    """Per-family photo support for a set of outline masks (all in photo coordinates)."""

    return {name: family_support(mask, edges) for name, mask in masks.items()}
