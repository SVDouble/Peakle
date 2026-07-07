"""Missing-foreground detector: does the photo show a major near feature the DEM doesn't?

On several of the largest-error samples the photo's dominant subject — e.g. a limestone tower a
few hundred metres away — is absent from both the GT render and our DEM (position error, DEM
resolution, or non-terrain object).  No skyline metric catches that directly; monocular depth
does: Depth-Anything sees a big near/far separation in the photo, while the DEM depth image for
the same view contains no comparably near layer.

``foreground_report`` is a pure comparator (unit-testable); ``photo_mono_depth`` wraps the
optional Depth-Anything estimator.  Flags feed the GT-quality audit — a missing-foreground sample
must not grade solvers even if its skyline reconstructs well.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

PHOTO_NEAR_TH = 0.22  # normalized mono depth (0 = nearest) below which a pixel is "near layer"
PHOTO_FG_MIN = 0.08  # near layer must cover at least this fraction of terrain to matter
DEM_NEAR_M = 1200.0  # a DEM pixel this close counts as rendered foreground
RATIO = 0.25  # DEM foreground below this fraction of the photo's -> missing


@lru_cache(maxsize=1)
def _estimator():
    from peakle.depth import load_learned_depth

    return load_learned_depth()


def photo_mono_depth(rgb_uint8: np.ndarray) -> np.ndarray | None:
    """Relative monocular depth (0 = near, 1 = far), or None without torch/transformers."""

    est = _estimator()
    if est is None:
        return None
    return est.estimate(rgb_uint8.astype(np.float64) / 255.0)


def foreground_report(
    mono: np.ndarray,
    dem_depth_m: np.ndarray,
    sky_rows: np.ndarray | None = None,
    photo_near_th: float = PHOTO_NEAR_TH,
    dem_near_m: float = DEM_NEAR_M,
) -> dict:
    """Compare the photo's near layer against the DEM's rendered foreground.

    ``mono``: relative depth (0 near, 1 far), photo geometry.  ``dem_depth_m``: metric DEM depth
    image (NaN sky), any resolution.  ``sky_rows``: optional skyline curve — mono pixels above it
    (sky, clouds) are excluded from the terrain statistics.
    """

    h, w = mono.shape
    terrain = np.ones((h, w), bool)
    if sky_rows is not None:
        cols = np.arange(w)
        rr = np.clip(np.nan_to_num(sky_rows, nan=-1), -1, h - 1)
        terrain = np.arange(h)[:, None] > rr[None, cols]
        terrain &= np.isfinite(sky_rows)[None, :]
    n_terr = max(int(terrain.sum()), 1)
    photo_fg = float((terrain & (mono < photo_near_th)).sum() / n_terr)

    dem_finite = np.isfinite(dem_depth_m)
    dem_fg = float((dem_finite & (dem_depth_m < dem_near_m)).sum() / max(int(dem_finite.sum()), 1))

    missing = photo_fg >= PHOTO_FG_MIN and dem_fg < RATIO * photo_fg
    return {
        "photo_fg_frac": round(photo_fg, 3),
        "dem_fg_frac": round(dem_fg, 3),
        "missing_foreground": bool(missing),
    }


def above_band_gradient(mono: np.ndarray, rows: np.ndarray, band_px: int = 30) -> float | None:
    """Mean |vertical mono-depth gradient| in the band ABOVE a skyline path.

    Sky is depth-flat (gradient ~0); water or terrain above the path (a false skyline along a
    lake edge or a mid-slope boundary) recedes smoothly — strong gradient.  Threshold measured
    on real samples; used to de-trust photo skyline candidates in GT v2 building."""

    h, w = mono.shape
    grad = np.abs(np.diff(mono, axis=0))
    vals = []
    for c in range(w):
        r = rows[c]
        if not np.isfinite(r):
            continue
        r0 = max(int(r) - band_px, 0)
        r1 = max(int(r) - 4, 1)
        if r1 > r0:
            vals.append(float(grad[r0:r1, c].mean()))
    return float(np.median(vals)) if vals else None
