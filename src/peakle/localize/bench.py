"""GeoPose3K orientation benchmark — the project's honesty harness (logic; CLI in peakle.scripts.bench_geopose).

For every sample the camera POSITION and FOV are taken as known (the product use case: GPS +
EXIF) and the solver recovers yaw/pitch twice:

  ORACLE track     observed skyline = the GT depth map's sky boundary.  Isolates solver + DEM +
                   conventions; failures here are never extraction's fault.
  EXTRACTED track  observed skyline = extraction from the photo, arbitrated by explanation.  End
                   to end.  The oracle-vs-extracted gap attributes failures to extraction.

Scored against DECODED GT yaw/pitch — success is a pose-error threshold, never a residual (a low
chamfer at a wrong yaw is an alias and counts as failure).  Every solve records its no-GT
diagnostics so verdict calibration can measure which of them predict correctness.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from peakle.localize.copdem import load_cop_around
from peakle.localize.explanation import arbitrate_by_explanation
from peakle.localize.extract import extract_candidates
from peakle.localize.geopose import load_sample, oracle_skyline
from peakle.localize.paths import COP_TILES_DIR, GEOPOSE_DIR, STD_WIDTH
from peakle.localize.photo_support import edge_mask
from peakle.localize.solve import HorizonProfile, OrientationSolve, _best_shift_chamfer, solve_orientation

MAX_W = STD_WIDTH  # solve cost scales with width; ~1150 cols keeps yaw resolution well under 0.1 deg
PITCH_BOUNDS = (-50.0, 50.0)  # GeoPose3K crop offsets reach ~48 deg (offset ~ 1.2 * Euler b)

_SEGMENTER = None


def ang_err(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def solve_record(solve: OrientationSolve, gt_yaw: float, gt_pitch: float) -> dict:
    yaw_err = ang_err(solve.yaw_deg, gt_yaw)
    # pitch_err is INFORMATIONAL only: the cyl crops are not vertically centred on the optical
    # axis (per-sample crop offset), so decoded GT pitch is not comparable in crop coordinates.
    return {
        "yaw": round(solve.yaw_deg, 2),
        "pitch": round(solve.pitch_deg, 2),
        "yaw_err": round(yaw_err, 2),
        "pitch_err": round(solve.pitch_deg - gt_pitch, 2),
        "correct": bool(abs(yaw_err) <= 5.0),
        "chamfer_px": round(solve.chamfer_px, 2),
        "coverage": round(solve.coverage, 3),
        "well_width_deg": round(solve.well_width_deg, 1),
        "alias_ratio": round(solve.alias_ratio, 3),
        "terrain_distinct_px": round(solve.terrain_distinct_px, 1),
        "snr": round(solve.snr, 2),
        "verdict": solve.verdict,
    }


def _candidates(rgb: np.ndarray, extractor: str) -> dict:
    """Skyline hypotheses: colour detectors (+ optional SAM3 sky mask). Solved separately —
    the DEM chamfer / explanation arbitrates, since no image-side rule reliably picks the winner."""

    cands = extract_candidates(rgb)
    if extractor == "sam3":
        global _SEGMENTER
        if _SEGMENTER is None:
            from peakle.segmenters import load_segmenter

            _SEGMENTER = load_segmenter("sam3")
        from peakle.localize.extract import extract_skyline_from_mask

        cands["sam3"] = extract_skyline_from_mask(_SEGMENTER.sky_mask(rgb), rgb)
    return cands


def run_sample(
    sdir: Path, extent_m: float, grid: int, outdir: Path | None = None, extractor: str = "color"
) -> dict:
    """Solve one sample on both tracks and return its record (writes an overlay if ``outdir``)."""

    gt = load_sample(sdir)
    rec: dict = {
        "name": gt.name,
        "manual": gt.manual,
        "lat": gt.lat,
        "lon": gt.lon,
        "elev_m": gt.elev_m,
        "fov_deg": round(gt.fov_deg, 2),
        "gt_yaw": round(gt.yaw_gt_deg, 2),
        "gt_pitch": round(gt.pitch_gt_deg, 2),
        "gt_roll": round(gt.roll_gt_deg, 2),
    }

    terrain = load_cop_around(COP_TILES_DIR, gt.lat, gt.lon, extent_m=extent_m, grid=grid)
    ground = terrain.elevation_at(0.0, 0.0)
    cam_z = max(gt.elev_m, ground + 2.0)
    rec["ground_m"] = round(ground, 1)
    rec["alt_above_ground"] = round(gt.elev_m - ground, 1)
    profile = HorizonProfile(terrain, cam_z, step=25.0)

    rgb = np.asarray(Image.open(gt.photo_path).convert("RGB"), np.uint8)
    if rgb.shape[1] > MAX_W:
        s = MAX_W / rgb.shape[1]
        h2 = max(1, round(rgb.shape[0] * s))
        rgb = np.asarray(Image.fromarray(rgb).resize((MAX_W, h2), Image.Resampling.BILINEAR), np.uint8)
    h_p, w_p = rgb.shape[:2]

    # ---- oracle track (GT depth skyline, rescaled onto the standard geometry) ----
    oracle = _resample(oracle_skyline(gt.depth_path), w_p) * (h_p / _pfm_height(gt.depth_path))
    s_oracle = solve_orientation(
        oracle, h_p, profile, fov_deg=gt.fov_deg, projection="cyltan", pitch_bounds=PITCH_BOUNDS
    )
    rec["oracle"] = solve_record(s_oracle, gt.yaw_gt_deg, gt.pitch_gt_deg)

    # GT consistency: chamfer between the GT-depth skyline and OUR DEM rendered at the GT yaw —
    # a per-sample cleanliness score flagging bad GT (mislocated/misoriented pose) or DEM trouble.
    dem_gt = profile.rows_cyl_tan(w_p, h_p, gt.fov_deg, gt.yaw_gt_deg, 0.0)
    dv_lim = int((w_p / math.radians(gt.fov_deg)) * math.tan(math.radians(PITCH_BOUNDS[1]))) + 1
    gt_cons, _ = _best_shift_chamfer(oracle, dem_gt, np.arange(-dv_lim, dv_lim + 1, 8), 60.0)
    rec["gt_consistency_px"] = round(gt_cons, 1)

    # ---- extracted track: solve every skyline hypothesis, arbitrated by explanation ----
    cands = _candidates(rgb, extractor)
    solved = {
        name: (
            c,
            solve_orientation(c.rows, h_p, profile, fov_deg=gt.fov_deg, projection="cyltan", pitch_bounds=PITCH_BOUNDS),
        )
        for name, c in cands.items()
        if c.coverage >= 0.25
    }
    rec["candidates"] = {}
    for name, c in cands.items():
        both_c = np.isfinite(c.rows) & np.isfinite(oracle)
        entry = {
            "coverage": round(c.coverage, 3),
            "err_vs_oracle_px": round(float(np.median(np.abs(c.rows - oracle)[both_c])), 1) if both_c.any() else None,
        }
        if name in solved:
            s_c = solved[name][1]
            entry.update(
                yaw_err=round(ang_err(s_c.yaw_deg, gt.yaw_gt_deg), 2),
                chamfer_px=round(s_c.chamfer_px, 2),
                verdict=s_c.verdict,
            )
        rec["candidates"][name] = entry

    if solved:
        # Among hypotheses within a tight chamfer slack of the best, pick the one whose typed DEM
        # outlines best EXPLAIN the photo edges — disambiguates the lake/false-skyline class.
        edges = edge_mask(rgb)
        if edges is not None and len(solved) > 1:
            win_name, expl_scores = arbitrate_by_explanation(solved, terrain, cam_z, w_p, h_p, gt.fov_deg, edges)
            rec["extracted_explanation"] = {k: round(v, 3) for k, v in expl_scores.items()}
        else:
            win_name = min(solved.items(), key=lambda kv: kv[1][1].chamfer_px)[0]
        win_c, s_extr = solved[win_name]
        rivals = [s for _, (_, s) in solved.items() if s.chamfer_px <= 1.3 * s_extr.chamfer_px]
        spread = max((abs(ang_err(a.yaw_deg, b.yaw_deg)) for a in rivals for b in rivals), default=0.0)
        rec["extracted"] = solve_record(s_extr, gt.yaw_gt_deg, gt.pitch_gt_deg)
        rec["extracted"]["agreement"] = round(win_c.agreement, 3)
        rec["extracted"]["winner"] = win_name
        rec["extracted"]["candidate_spread_deg"] = round(spread, 1)
        if spread > 10.0 and rec["extracted"]["verdict"] == "CONFIRMED":
            rec["extracted"]["verdict"] = "AMBIGUOUS"  # plausible hypotheses disagree on yaw
        ext_rows = win_c.rows
    else:
        rec["extracted"] = {
            "correct": False, "verdict": "REJECTED", "yaw_err": float("nan"), "note": "no usable skyline"
        }
        ext_rows = np.full(w_p, np.nan)
        s_extr = None

    both = np.isfinite(ext_rows) & np.isfinite(oracle)
    rec["extraction_err_px"] = round(float(np.median(np.abs(ext_rows - oracle)[both])), 1) if both.any() else None
    rec["extraction_coverage"] = round(float(np.isfinite(ext_rows).mean()), 3)

    if s_extr is not None and outdir is not None:
        _overlay(rgb, gt, profile, ext_rows, oracle, s_extr, outdir / f"{gt.name}.jpg")
    return rec


def find_sample_dirs(data_dir: Path = GEOPOSE_DIR) -> list[Path]:
    """Sample dirs with the inputs the bench needs (photo + depth + info)."""

    needed = ("info.txt", "cyl/photo_crop.jpg", "cyl/distance_crop.pfm")
    return sorted(d for d in data_dir.iterdir() if d.is_dir() and all((d / n).exists() for n in needed))


def _pfm_height(path: Path) -> int:
    with open(path, "rb") as f:
        f.readline()
        _, h = map(int, f.readline().split())
    return h


def _resample(rows: np.ndarray, width: int) -> np.ndarray:
    x_src = np.linspace(0.0, 1.0, len(rows))
    x_dst = np.linspace(0.0, 1.0, width)
    fin = np.isfinite(rows)
    if fin.sum() < 2:
        return np.full(width, np.nan)
    out = np.interp(x_dst, x_src[fin], rows[fin])
    covered = np.interp(x_dst, x_src, fin.astype(float)) > 0.5
    return np.where(covered, out, np.nan)


def _overlay(rgb, gt, profile, extracted, oracle_rows, s_extr, out_path: Path) -> None:
    h, w = rgb.shape[:2]
    im = Image.fromarray(np.clip(rgb.astype(float) * 1.25, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)

    def draw(rows, color, width=2):
        pts = [(c, float(rows[c])) for c in range(w) if np.isfinite(rows[c]) and 0 <= rows[c] < h]
        if len(pts) > 1:
            dr.line(pts, fill=color, width=width)

    draw(oracle_rows, (0, 255, 90))  # GT depth skyline
    draw(extracted, (255, 70, 70))  # photo extraction
    draw(profile.rows_cyl_tan(w, h, gt.fov_deg, s_extr.yaw_deg, s_extr.pitch_deg), (255, 225, 0))  # solved
    draw(profile.rows_cyl_tan(w, h, gt.fov_deg, gt.yaw_gt_deg, gt.pitch_gt_deg), (0, 200, 255))  # DEM @ GT
    dr.rectangle([0, 0, w, 18], fill=(0, 0, 0))
    dr.text(
        (4, 3),
        f"{gt.name} [{'MANUAL' if gt.manual else 'AUTO'}] GTyaw={gt.yaw_gt_deg:.0f} solved={s_extr.yaw_deg:.0f} "
        f"({s_extr.verdict})  green=GTdepth red=extract yellow=solved cyan=DEM@GT",
        fill=(160, 255, 160),
    )
    im.save(out_path, quality=90)


def summarize(rows: list[dict]) -> str:
    def rate(rs, track):
        ok = [r for r in rs if r.get(track, {}).get("correct")]
        return f"{len(ok)}/{len(rs)}" + (f" ({len(ok) / len(rs):.0%})" if rs else "")

    lines = ["# GeoPose3K orientation benchmark", ""]
    groups = [
        ("ALL", rows),
        ("MANUAL", [r for r in rows if r["manual"]]),
        ("AUTO", [r for r in rows if not r["manual"]]),
    ]
    for label, sel in groups:
        if not sel:
            continue
        lines += [
            f"## {label} (n={len(sel)})",
            f"- oracle-track success (|yaw err| <= 5 deg): **{rate(sel, 'oracle')}**",
            f"- extracted-track success: **{rate(sel, 'extracted')}**",
            "",
        ]
    lines += [
        "## Per-sample",
        "",
        "| sample | src | GT yaw | oracle err | extr err | extr_qual px | oracle verdict | extr verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        o, e = r.get("oracle", {}), r.get("extracted", {})
        lines.append(
            f"| {r['name']} | {'M' if r['manual'] else 'A'} | {r['gt_yaw']:.0f} | "
            f"{o.get('yaw_err', 'ERR'):+.0f} | {e.get('yaw_err', float('nan')):+.0f} | {r.get('extraction_err_px')} | "
            f"{o.get('verdict', '-')} | {e.get('verdict', '-')} |"
            if isinstance(o.get("yaw_err"), float)
            else f"| {r['name']} | {'M' if r['manual'] else 'A'} | {r['gt_yaw']:.0f} | ERR | ERR | - | - | - |"
        )
    return "\n".join(lines) + "\n"
