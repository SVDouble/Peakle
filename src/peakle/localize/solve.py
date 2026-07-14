"""Orientation solver: match an observed skyline to the DEM horizon over the full 360° yaw.

Core idea: for a fixed position the horizon elevation profile ``el(az)`` is camera-independent,
so it is ray-cast ONCE at fine azimuth resolution; every (yaw, pitch, fov) skyline hypothesis is
then a resampling of that profile.  Pitch is decoupled as a vertical image shift (exact for the
cylindrical projection, first-order for pinhole; the returned pitch is derived from the best
shift).  The residual is a symmetric, distance-capped curve chamfer — capping keeps single
garbage columns in the extraction from dominating the score.

Every solve returns the full yaw-chamfer profile plus derived diagnostics (basin width, alias
margin, coverage). The former confidence gate was calibrated against circular GT-v2 targets and
has been retired. Non-rejected results remain ``UNCALIBRATED`` until a frozen, location-held-out
MANUAL + MAP_A calibration set exists.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from peakle.domain.projection import (
    ProjectionName,
    azimuths_deg,
    pitch_deg_from_vertical_shift_px,
    rows_from_elevation_rad,
    vertical_shift_px_from_pitch_deg,
)
from peakle.localize.raycast import horizon_elevation

AZ_STEP_DEG = 0.1  # resolution of the one-time 360° horizon profile


@dataclass
class OrientationSolve:
    yaw_deg: float
    pitch_deg: float
    fov_deg: float
    chamfer_px: float
    coverage: float  # fraction of columns with an observed skyline
    well_width_deg: float  # angular width of the chamfer basin around the solution
    alias_ratio: float  # best chamfer outside the basin / best chamfer (>1 good)
    terrain_distinct_px: float  # how unlike the rest of the horizon the solved window is
    snr: float  # terrain_distinct_px / chamfer_px — fit noise vs signal
    yaw_profile_deg: np.ndarray = field(repr=False)
    yaw_profile_chamfer: np.ndarray = field(repr=False)
    self_profile_deg: np.ndarray = field(repr=False)  # terrain self-similarity scan (see snr)
    self_profile_chamfer: np.ndarray = field(repr=False)
    candidates: list[tuple[float, float, float]] = field(default_factory=list, repr=False)
    """Polished (chamfer, yaw, dv) pose candidates, best first — rerank fodder for
    explanation matching against typed DEM outlines."""
    verdict: str = "UNCALIBRATED"

    def summary(self) -> str:
        return (
            f"yaw={self.yaw_deg:.1f} pitch={self.pitch_deg:.1f} fov={self.fov_deg:.1f} "
            f"chamfer={self.chamfer_px:.1f}px cover={self.coverage:.0%} "
            f"well={self.well_width_deg:.0f}deg alias={self.alias_ratio:.2f} "
            f"distinct={self.terrain_distinct_px:.0f}px snr={self.snr:.1f} -> {self.verdict}"
        )


def _directed(p: np.ndarray, q: np.ndarray, cap: float, shifts: np.ndarray) -> float:
    """Mean capped distance from curve ``p`` to curve ``q`` (both row-per-column, NaN = missing)."""

    n = len(p)
    best = np.full(n, cap * cap)
    for s in shifts:
        lo, hi = max(0, -s), min(n, n - s)
        if hi <= lo:
            continue
        d2 = s * s + (p[lo:hi] - q[lo + s : hi + s]) ** 2
        np.minimum(best[lo:hi], np.where(np.isnan(d2), cap * cap, d2), out=best[lo:hi])
    valid = ~np.isnan(p)
    if not valid.any():
        return cap
    return float(np.sqrt(best[valid]).mean())


def curve_chamfer(
    a: np.ndarray,
    b: np.ndarray,
    cap: float = 60.0,
    shift_step: int = 3,
    max_shift: int = 45,
) -> float:
    """Symmetric capped chamfer between two per-column skyline curves, in pixels."""

    shifts = np.arange(-max_shift, max_shift + 1, shift_step)
    return 0.5 * (_directed(a, b, cap, shifts) + _directed(b, a, cap, shifts))


class HorizonProfile:
    """The one-time 360° horizon of a position, resampleable into any camera."""

    def __init__(
        self,
        terrain,
        cam_up_m: float,
        step: float = 30.0,
        d_max: float | None = None,
        cam_e: float = 0.0,
        cam_n: float = 0.0,
        patch=None,
    ):
        self.az_deg = np.arange(0.0, 360.0, AZ_STEP_DEG)
        el = horizon_elevation(
            terrain,
            np.radians(self.az_deg),
            cam_up_m,
            step=step,
            d_max=d_max,
            cam_e=cam_e,
            cam_n=cam_n,
            patch=patch,
        )
        self.el = el  # radians, NaN where the DEM had no sample
        self.cam_up_m = cam_up_m

    def _el_at(self, az_deg: np.ndarray) -> np.ndarray:
        idx = (az_deg % 360.0) / AZ_STEP_DEG
        i0 = np.floor(idx).astype(int) % len(self.el)
        i1 = (i0 + 1) % len(self.el)
        w = idx - np.floor(idx)
        e0, e1 = self.el[i0], self.el[i1]
        out = e0 * (1 - w) + e1 * w
        return np.where(np.isnan(e0) | np.isnan(e1), np.nan, out)

    def rows_cyl(self, width: int, height: int, fov_deg: float, yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
        az = azimuths_deg(width, fov_deg, yaw_deg, "cyl")
        el = self._el_at(az)
        return rows_from_elevation_rad(el, width, height, fov_deg, "cyl", pitch_deg=pitch_deg)

    def rows_cyl_tan(
        self, width: int, height: int, fov_deg: float, yaw_deg: float, pitch_deg: float = 0.0
    ) -> np.ndarray:
        """TRUE cylindrical projection: columns linear in azimuth, rows linear in tan(elevation).
        This is GeoPose3K's crop geometry — fitted empirically: on a 690px-amplitude skyline the
        el-linear mapping needs 1.79x the theoretical slope (impossible) while tan fits at 1.017.
        Focal length is the same horizontally and vertically: f = W/hfov px per radian."""

        az = azimuths_deg(width, fov_deg, yaw_deg, "cyltan")
        el = self._el_at(az)
        return rows_from_elevation_rad(el, width, height, fov_deg, "cyltan", pitch_deg=pitch_deg)

    def rows_pinhole(
        self, width: int, height: int, fov_deg: float, yaw_deg: float, pitch_deg: float = 0.0
    ) -> np.ndarray:
        az = azimuths_deg(width, fov_deg, yaw_deg, "pinhole")
        el = self._el_at(az)
        return rows_from_elevation_rad(el, width, height, fov_deg, "pinhole", pitch_deg=pitch_deg)

    def rows(
        self,
        projection: ProjectionName,
        width: int,
        height: int,
        fov_deg: float,
        yaw_deg: float,
        pitch_deg: float = 0.0,
    ) -> np.ndarray:
        fn = {"cyl": self.rows_cyl, "cyltan": self.rows_cyl_tan, "pinhole": self.rows_pinhole}[projection]
        return fn(width, height, fov_deg, yaw_deg, pitch_deg)

    def pitch_from_shift(self, projection: ProjectionName, width: int, height: int, fov_deg: float, dv: float) -> float:
        """Pitch implied by shifting the pitch-0 skyline DOWN by ``dv`` rows."""

        return pitch_deg_from_vertical_shift_px(width, fov_deg, projection, dv)


def _basin(yaws: np.ndarray, ch: np.ndarray, best_i: int) -> tuple[float, np.ndarray]:
    """Width of the contiguous chamfer basin around the best yaw + an inside/guard mask."""

    cmin = ch[best_i]
    near = ch < max(cmin * 1.15, cmin + 2.0)
    n = len(yaws)
    lo = hi = best_i
    while near[(lo - 1) % n] and (best_i - lo) < n:
        lo -= 1
    while near[(hi + 1) % n] and (hi - best_i) < n:
        hi += 1
    if hi - lo + 1 >= n:
        return 360.0, np.ones(n, bool)
    width = (hi - lo + 1) * (yaws[1] - yaws[0]) if n > 1 else 360.0
    inside = np.zeros(n, bool)
    inside[np.arange(lo, hi + 1) % n] = True
    # a guard band around the basin so the alias margin measures a DISTINCT minimum
    guard = int(round(10.0 / (yaws[1] - yaws[0])))
    for g in range(1, guard + 1):
        inside[(lo - g) % n] = True
        inside[(hi + g) % n] = True
    return float(width), inside


def solve_orientation(
    obs_rows: np.ndarray,
    height: int,
    profile: HorizonProfile,
    fov_deg: float | tuple[float, float, float],
    projection: ProjectionName = "cyl",
    pitch_bounds: tuple[float, float] = (-30.0, 30.0),
    cap_px: float = 60.0,
    yaw_step_deg: float = 1.0,
) -> OrientationSolve:
    """Recover (yaw, pitch[, fov]) for an observed skyline given a position's horizon profile.

    ``fov_deg`` is either a known value or an inclusive ``(lo, hi, step)`` search range.
    ``obs_rows``: per-column skyline row (NaN where not observed), full image width.
    """

    obs = np.asarray(obs_rows, float)
    width = len(obs)
    coverage = float(np.isfinite(obs).mean())
    fovs: list[float] = (
        [float(fov_deg)]
        if isinstance(fov_deg, (int, float))
        else [float(value) for value in np.arange(fov_deg[0], fov_deg[1] + 1e-9, fov_deg[2])]
    )

    yaws = np.arange(0.0, 360.0, yaw_step_deg)
    best: tuple[float, float, float, float, np.ndarray, np.ndarray] | None = None
    for fov in fovs:
        dv_max = _dv_for_pitch(projection, width, height, fov, max(abs(pitch_bounds[0]), abs(pitch_bounds[1])))
        ch = np.full(len(yaws), np.inf)
        dv_at = np.zeros(len(yaws))
        for i, yaw in enumerate(yaws):
            base = profile.rows(projection, width, height, float(fov), float(yaw), 0.0)
            c, dv = _median_centered_shift_chamfer(obs, base, dv_max, cap_px)  # coarse ranking
            ch[i] = c
            dv_at[i] = dv
        i = int(np.argmin(ch))
        if best is None or ch[i] < best[0]:
            best = (float(ch[i]), float(yaws[i]), float(dv_at[i]), float(fov), ch.copy(), dv_at.copy())

    if best is None:
        raise ValueError("FOV search produced no hypotheses")
    _, _, _, fov0, prof_ch, prof_dv = best

    def _polish(yaw_c: float, dv_c: float) -> tuple[float, float, float]:
        """Fine (yaw, dv) around a coarse candidate; dv is two-stage (local step, then unit)."""

        c_best, y_best, dv_pick = math.inf, yaw_c, dv_c
        for dvs in (np.arange(dv_c - 36.0, dv_c + 36.01, 6.0), None):
            if dvs is None:
                dvs = np.arange(dv_pick - 6.0, dv_pick + 6.01, 1.0)
            for yaw in np.arange(y_best - yaw_step_deg, y_best + yaw_step_deg + 1e-9, yaw_step_deg / 4):
                base = profile.rows(projection, width, height, fov0, float(yaw), 0.0)
                c, dv = _best_shift_chamfer(obs, base, dvs, cap_px)
                if c < c_best:
                    c_best, y_best, dv_pick = float(c), float(yaw % 360.0), float(dv)
        return c_best, y_best, dv_pick

    # polish the TOP-K distinct coarse basins, not just the argmin: the coarse dv grid can easily
    # inflate the true basin past a smooth alias (measured: true yaw 20.7px on the grid vs 5.1px
    # fine — the alias won the coarse ranking and the truth was never polished)
    order = np.argsort(prof_ch)
    picked: list[int] = []
    for i in order:
        if all(min(abs(yaws[i] - yaws[j]), 360.0 - abs(yaws[i] - yaws[j])) > 8.0 for j in picked):
            picked.append(int(i))
        if len(picked) >= 12:
            break
    cmin, yaw0, dv0 = math.inf, float(yaws[picked[0]]), float(prof_dv[picked[0]])
    polished = []
    for i in picked:
        c_p, y_p, dv_p = _polish(float(yaws[i]), float(prof_dv[i]))
        polished.append((c_p, y_p, dv_p))
        if c_p < cmin:
            cmin, yaw0, dv0 = c_p, y_p, dv_p

    pitch = profile.pitch_from_shift(projection, width, height, fov0, dv0)
    best_i = int(np.argmin(np.abs(((yaws - yaw0 + 180) % 360) - 180)))
    well_width, inside = _basin(yaws, prof_ch, best_i)

    # honest alias margin: best POLISHED rival outside the winning basin vs the winner
    alias = 1.0
    if not inside.all():
        rivals = [c for c, y, _ in polished if not inside[int(np.argmin(np.abs(((yaws - y + 180) % 360) - 180)))]]
        if not rivals:  # every polished candidate fell inside the basin — polish the coarse outside best
            rival_i = int(np.argmin(np.where(inside, np.inf, prof_ch)))
            rivals = [_polish(float(yaws[rival_i]), float(prof_dv[rival_i]))[0]]
        alias = float(min(rivals) / cmin) if cmin > 0 else 1.0

    # terrain self-distinctiveness: chamfer the SOLVED DEM window against the DEM horizon at all
    # other yaws.  If the terrain nearly repeats itself (a symmetric valley, a rolling ridge), a
    # small extraction error can flip the winning basin while looking distinct — no photo can
    # disambiguate such a position, so CONFIRMED additionally requires the terrain to be more
    # distinctive than the fit residual (the SNR gate).
    template = profile.rows(projection, width, height, fov0, yaw0, 0.0) + dv0
    self_yaws = np.arange(0.0, 360.0, max(yaw_step_deg, 2.0))
    dv_lim = _dv_for_pitch(projection, width, height, fov0, max(abs(pitch_bounds[0]), abs(pitch_bounds[1])))
    self_ch = np.full(len(self_yaws), np.inf)
    for i, yaw in enumerate(self_yaws):
        base = profile.rows(projection, width, height, fov0, float(yaw), 0.0)
        self_ch[i], _ = _median_centered_shift_chamfer(template, base, dv_lim, cap_px)
    self_i = int(np.argmin(np.abs(((self_yaws - yaw0 + 180) % 360) - 180)))
    _, self_inside = _basin(self_yaws, self_ch, self_i)
    distinct = float(np.min(self_ch[~self_inside])) if not self_inside.all() else 0.0
    snr = distinct / max(cmin, 1.0)

    solve = OrientationSolve(
        yaw_deg=yaw0,
        pitch_deg=pitch,
        fov_deg=fov0,
        chamfer_px=cmin,
        coverage=coverage,
        well_width_deg=well_width,
        alias_ratio=alias,
        terrain_distinct_px=distinct,
        snr=snr,
        yaw_profile_deg=yaws,
        yaw_profile_chamfer=prof_ch,
        self_profile_deg=self_yaws,
        self_profile_chamfer=self_ch,
        candidates=sorted(polished),
    )
    solve.verdict = _provisional_verdict(solve, cap_px)
    return solve


def _dv_for_pitch(projection: ProjectionName, width: int, height: int, fov: float, pitch_deg: float) -> int:
    del height
    return int(abs(vertical_shift_px_from_pitch_deg(width, fov, projection, pitch_deg))) + 1


def _best_shift_chamfer(
    obs: np.ndarray, base: np.ndarray, dvs: np.ndarray, cap: float, shift_step: int = 3
) -> tuple[float, float]:
    """Best (chamfer, dv) over vertical shifts of the rendered pitch-0 skyline."""

    best_c, best_dv = math.inf, 0.0
    for dv in dvs:
        c = curve_chamfer(obs, base + dv, cap=cap, shift_step=shift_step)
        if c < best_c:
            best_c, best_dv = float(c), float(dv)
    return best_c, best_dv


def _median_centered_shift_chamfer(
    obs: np.ndarray, base: np.ndarray, dv_lim: float, cap: float, shift_step: int = 6
) -> tuple[float, float]:
    """Chamfer with the vertical shift centred at the curves' median offset.

    Scanning the full ±dv_lim range at a grid coarse enough to be affordable made coarse chamfers
    nearly random for large pitch bounds (a ~230px dv step flipped correct solves to 180° aliases).
    The median offset aligns the curves in O(W) regardless of crop offset; a small local scan
    around it recovers the rest.  ``dv_lim`` only clamps the centre.
    """

    vp = np.isfinite(obs) & np.isfinite(base)
    if vp.sum() < 0.15 * len(obs):
        return cap, 0.0
    dv0 = float(np.clip(np.median(obs[vp]) - np.median(base[vp]), -dv_lim, dv_lim))
    return _best_shift_chamfer(obs, base, dv0 + np.arange(-72.0, 73.0, 24.0), cap, shift_step)


def _provisional_verdict(s: OrientationSolve, cap_px: float) -> str:
    """Reject unusable evidence; do not claim calibrated confidence.

    The previous ``CONFIRMED`` thresholds were tuned on GT-v2 poses produced with DEM/photo
    alignment. That makes their apparent precision circular for this solver, so viable results
    deliberately stay uncalibrated pending a frozen, location-held-out MANUAL + MAP_A split.
    """

    if s.coverage < 0.25 or s.chamfer_px > 0.5 * cap_px:
        return "REJECTED"
    return "UNCALIBRATED"
