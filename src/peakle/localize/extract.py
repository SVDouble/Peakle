"""Photo skyline extraction with a self-assessment signal.

Two INDEPENDENTLY parameterised sky detectors run on every photo; where they disagree about the
skyline row, the extraction is untrustworthy at that column.  The aggregate ``agreement`` is a
garbage-input detector: it is computed WITHOUT ground truth, so it works on any photo, and it is
validated against GT skylines on the GeoPose3K benchmark.

Hard-won details encoded here:
- test blue-DOMINANCE, not blueness (sunlit snow is bright and blue);
- keep only the sky component connected to the top of the VALID image area — warped crops
  (GeoPose3K cylindrical reprojections) have black borders, so "top of the image" is per-column
  the first non-border pixel, NOT row 0;
- never binary-close the raw mask (it erodes the top border to non-sky).

``extract_skyline_from_mask`` accepts an externally computed sky mask (e.g. a SAM3 segmenter's)
and reuses the same border logic; its ``agreement`` is then measured against the colour detectors,
giving a cross-model consistency signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_fill_holes, label, median_filter


@dataclass
class ExtractedSkyline:
    rows: np.ndarray          # per-column skyline row, NaN where not found
    coverage: float           # fraction of columns with a skyline
    agreement: float          # fraction of columns where independent detectors agree (<=3 px)

    @property
    def width(self) -> int:
        return len(self.rows)


def _valid_mask(rgb: np.ndarray) -> np.ndarray:
    """Non-border pixels.  Warped crops pad with (near-)black; treat those as outside the image."""

    return rgb.astype(int).sum(axis=2) > 40


def _skyline_from_sky_mask(sky: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """First terrain row per column, using only sky connected to the top of the valid area."""

    h, w = sky.shape
    sky = sky & valid
    lab, _ = label(sky)
    col_has_valid = valid.any(axis=0)
    top_valid = np.where(col_has_valid, valid.argmax(axis=0), 0)
    cols = np.arange(w)[col_has_valid]
    top_ids = np.unique(lab[top_valid[cols], cols])
    top_ids = top_ids[top_ids > 0]
    if top_ids.size == 0:
        return np.full(w, np.nan)
    sky_c = binary_fill_holes(np.isin(lab, top_ids))

    seen_sky = np.cumsum(sky_c, axis=0) > 0
    terrain = ~sky_c & valid & seen_sky
    rows = np.full(w, np.nan)
    has = terrain.any(axis=0) & sky_c.any(axis=0)
    rows[has] = terrain[:, has].argmax(axis=0).astype(float)
    fin = np.isfinite(rows)
    if fin.sum() >= 7:
        rows[fin] = median_filter(rows[fin], 7)
    return rows


def _color_sky_masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = rgb[..., 0].astype(int)
    g = rgb[..., 1].astype(int)
    b = rgb[..., 2].astype(int)
    bright = (r + g + b) / 3.0
    # detector A: blue-dominant sky (clear sky), tolerant of haze
    sky_a = (b - r > 8) & (b > 110) & (r < 210)
    # detector B: bright sky (overcast/cloud), tolerant of white-out
    sky_b = ((b - r > -6) & (bright > 145)) | (bright > 205)
    return sky_a, sky_b


def _fuse(rows_a: np.ndarray, rows_b: np.ndarray) -> tuple[np.ndarray, float]:
    both = np.isfinite(rows_a) & np.isfinite(rows_b)
    agree_mask = both & (np.abs(rows_a - rows_b) <= 3.0)
    either = np.isfinite(rows_a) | np.isfinite(rows_b)
    agreement = float(agree_mask.sum() / max(either.sum(), 1))

    # where both fire use A; single-detector columns pass through; on disagreement trust the
    # HIGHER skyline (smaller row) if the other is within 12 px, else drop the column
    rows = np.where(np.isfinite(rows_a), rows_a, rows_b)
    disagree = both & ~agree_mask
    close = disagree & (np.abs(rows_a - rows_b) <= 12.0)
    rows[close] = np.minimum(rows_a[close], rows_b[close])
    rows[disagree & ~close] = np.nan
    return rows, agreement


def extract_skyline(rgb: np.ndarray) -> ExtractedSkyline:
    """Colour/brightness skyline with a two-detector agreement score."""

    valid = _valid_mask(rgb)
    sky_a, sky_b = _color_sky_masks(rgb)
    rows_a = _skyline_from_sky_mask(sky_a, valid)
    rows_b = _skyline_from_sky_mask(sky_b, valid)
    rows, agreement = _fuse(rows_a, rows_b)
    return ExtractedSkyline(rows=rows, coverage=float(np.isfinite(rows).mean()), agreement=agreement)


def _dp_skyline(score: np.ndarray, max_step: int = 3, step_penalty: float = 0.02) -> np.ndarray:
    """Highest-scoring left-to-right path through a per-pixel score map (one row per column).

    Classic DP skyline trace: transitions limited to ±``max_step`` rows per column with a small
    jump penalty.  Pure function of the score map — unit-testable without any learned model.
    """

    h, w = score.shape
    acc = score[:, 0].copy()
    back = np.zeros((h, w), np.int16)
    offsets = np.arange(-max_step, max_step + 1)
    for c in range(1, w):
        stack = np.full((len(offsets), h), -np.inf)
        for k, off in enumerate(offsets):
            src_lo, src_hi = max(0, -off), min(h, h - off)
            stack[k, src_lo:src_hi] = acc[src_lo + off : src_hi + off] - step_penalty * abs(off)
        best_k = np.argmax(stack, axis=0)
        acc = stack[best_k, np.arange(h)] + score[:, c]
        back[:, c] = offsets[best_k]
    rows = np.empty(w, float)
    r = int(np.argmax(acc))
    for c in range(w - 1, -1, -1):
        rows[c] = r
        r = int(np.clip(r + back[r, c], 0, h - 1))
    return rows


def learned_skyline(rgb: np.ndarray) -> ExtractedSkyline | None:
    """DexiNed-driven skyline: DP trace over edge response + a sky-above prior.

    The colour detectors fail outright on ~40% of photos (measured: 104/165 extracted-track
    failures had >40px extraction error); DexiNed marks the sky-terrain boundary at R_sky 0.95
    but needs a global path constraint to not wander onto valley edges — the DP supplies it, and
    the sky prior (fraction of blue-dominant pixels above the row) breaks cloud-edge ties.
    Returns None when torch/kornia are unavailable.
    """

    from peakle.localize.photo_support import edge_mask as _unused  # noqa: F401 - ensures module presence
    from peakle.localize.photo_support import _detector

    det = _detector()
    if det is None:
        return None
    emap = det.detect(rgb.astype(np.float64) / 255.0)
    h, w = emap.shape
    sky_a, _ = _color_sky_masks(rgb)
    valid = _valid_mask(rgb)
    # the warped-crop black borders are themselves strong artificial edges — erode them away
    core = valid.copy()
    for _ in range(3):
        er = core.copy()
        er[1:, :] &= core[:-1, :]
        er[:-1, :] &= core[1:, :]
        er[:, 1:] &= core[:, :-1]
        er[:, :-1] &= core[:, 1:]
        core = er
    skyness = np.cumsum(sky_a & valid, axis=0) / np.maximum(np.cumsum(valid, axis=0), 1)
    rows = _dp_skyline(np.where(core, emap, 0.0) + 0.55 * skyness)
    rows[~core[np.clip(rows.astype(int), 0, h - 1), np.arange(w)]] = np.nan
    fin = np.isfinite(rows)
    if fin.sum() >= 7:
        rows[fin] = median_filter(rows[fin], 7)
    # agreement vs the colour candidates (cross-detector consistency, as elsewhere)
    ca = _skyline_from_sky_mask(sky_a, valid)
    both = np.isfinite(rows) & np.isfinite(ca)
    agreement = float((both & (np.abs(rows - ca) <= 3.0)).sum() / max(both.sum(), 1)) if both.any() else 0.0
    return ExtractedSkyline(rows=rows, coverage=float(np.isfinite(rows).mean()), agreement=agreement)


def extract_candidates(rgb: np.ndarray) -> dict[str, ExtractedSkyline]:
    """Independent skyline hypotheses for multi-hypothesis solving.

    No image-side fusion can decide between them reliably (a haze boundary fools the blue
    detector, clouds fool the bright detector, each on different photos) — solve each candidate
    against the DEM and let the chamfer/alias diagnostics arbitrate.  Both candidates share the
    ``agreement`` score so the disagreement signal survives into the solve records.
    """

    valid = _valid_mask(rgb)
    sky_a, sky_b = _color_sky_masks(rgb)
    rows_a = _skyline_from_sky_mask(sky_a, valid)
    rows_b = _skyline_from_sky_mask(sky_b, valid)
    both = np.isfinite(rows_a) & np.isfinite(rows_b)
    agreement = float((both & (np.abs(rows_a - rows_b) <= 3.0)).sum() / max(both.sum(), 1)) if both.any() else 0.0
    out = {
        "blue": ExtractedSkyline(rows=rows_a, coverage=float(np.isfinite(rows_a).mean()), agreement=agreement),
        "bright": ExtractedSkyline(rows=rows_b, coverage=float(np.isfinite(rows_b).mean()), agreement=agreement),
    }
    try:
        learned = learned_skyline(rgb)
    except Exception:
        learned = None
    if learned is not None:
        out["dexined"] = learned
    return out


def extract_skyline_from_mask(sky_mask: np.ndarray, rgb: np.ndarray) -> ExtractedSkyline:
    """Skyline from an external sky mask (e.g. SAM3); agreement is scored vs the colour detectors."""

    valid = _valid_mask(rgb)
    rows = _skyline_from_sky_mask(sky_mask.astype(bool), valid)
    color_rows = extract_skyline(rgb).rows
    both = np.isfinite(rows) & np.isfinite(color_rows)
    either = np.isfinite(rows) | np.isfinite(color_rows)
    agreement = float((both & (np.abs(rows - color_rows) <= 3.0)).sum() / max(either.sum(), 1))
    return ExtractedSkyline(rows=rows, coverage=float(np.isfinite(rows).mean()), agreement=agreement)
