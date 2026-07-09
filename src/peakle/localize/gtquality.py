"""GT-corpus quality operations: photo-support gating of GT records + verdict calibration.

Two families of domain logic that used to live in scripts, kept together because both answer "is
this trustworthy enough to grade solvers on":

  apply_support_gate  re-tier GT v2 records by photo-edge support — a GT line the photograph does
                      not show is photo-inconsistent (label/registration/scenery error) and must
                      not grade solvers even if our DEM reproduces the render perfectly.
  calibrate_gate      read benchmark results and find the CONFIRMED-verdict thresholds that give
                      ~100% precision at the best recall, plus which diagnostics separate correct
                      from wrong solves (per-feature AUC).

The thin CLIs (peakle.scripts.apply_support_gate / calibrate_verdict / photo_support_batch)
call these.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from peakle.domain.projection import tangent_elevation_from_rows
from peakle.localize.paths import COP_TILES_DIR, GEOPOSE_DIR, GTV2_DIR

# support-gate thresholds (initial; tighten as the corpus grows)
SKY_SUPPORT_MIN = 0.50
OCC_SUPPORT_MIN = 0.20
OCC_DENSITY_MIN = 0.30
SUPPORT_FAMILIES = ("sky", "occ", "rib", "cou")
SKY_CONS_GATE_PX = 15.0
PFM_CONS_GATE_PX = 25.0
CONTOUR_CONS_GATE_PX = 25.0
SKY_ERROR_GATE_M = 150.0
PFM_ERROR_GATE_M = 250.0
PFM_PHOTO_OFFSET_GATE_PX = 15.0
YAW_DELTA_GATE_DEG = 20.0
POSITION_DELTA_GATE_M = 1500.0


def apply_support_gate(out_dir: Path = GTV2_DIR, *, dry_run: bool = False) -> tuple[int, int, int]:
    """Re-tier records by photo-edge support; returns (changed, clean, total).

    Run AFTER support.json sidecars exist (photo_support_batch). A record with a weak GT-skyline or
    GT-occlusion support becomes SUSPECT with a "photo-inconsistent …" reason.
    """

    index = json.loads((out_dir / "index.json").read_text())
    changed = 0
    for rec in index:
        sup_path = out_dir / "layers" / rec["name"] / "support.json"
        if not sup_path.exists():
            continue
        sup = json.loads(sup_path.read_text())
        reasons = [r for r in rec.get("reasons", []) if not r.startswith("photo-inconsistent")]
        sky, occ = sup.get("gt_sky"), sup.get("gt_occ")
        if sky is not None and sky < SKY_SUPPORT_MIN:
            reasons.append(f"photo-inconsistent skyline (support {sky:.2f} < {SKY_SUPPORT_MIN})")
        if occ is not None and occ < OCC_SUPPORT_MIN and rec.get("gt_contour_density", 0) >= OCC_DENSITY_MIN:
            reasons.append(f"photo-inconsistent occlusions (support {occ:.2f} < {OCC_SUPPORT_MIN})")
        quality = "CLEAN" if not reasons else "SUSPECT"
        if quality != rec["quality"] or reasons != rec.get("reasons", []):
            changed += 1
            if not dry_run:
                rec["quality"], rec["reasons"] = quality, reasons
                (out_dir / f"{rec['name']}.json").write_text(json.dumps(rec, indent=1))
    if not dry_run:
        (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    clean = sum(1 for r in index if r["quality"] == "CLEAN")
    return changed, clean, len(index)


def support_stats(rows: list[tuple[str, dict]]) -> dict[str, list[str]]:
    """Per-family support median / p10 for the GT and DEM outline sets (corpus health)."""

    out = {}
    for src in ("gt", "dem"):
        stats = []
        for fam in SUPPORT_FAMILIES:
            vals = [s[f"{src}_{fam}"] for _, s in rows if s.get(f"{src}_{fam}") is not None]
            stats.append(f"{fam} med={np.median(vals):.2f} p10={np.percentile(vals, 10):.2f}" if vals else f"{fam} n/a")
        out[src] = stats
    return out


def alignment_audit(records: list[dict[str, Any]], *, limit: int = 100, include_clean: bool = False) -> dict[str, Any]:
    """Rank GT records by skyline/outline disagreement and likely failure mode.

    The audit is deliberately metric-first: ground truth is itself one of the things being
    evaluated, so the report is "agreement with map/photo evidence" rather than "error vs GT".
    """

    rows = [alignment_record(rec) for rec in records]
    rows.sort(key=lambda row: (-row["severity"], row["name"]))
    returned = rows if include_clean else [row for row in rows if row["failure_modes"]]
    mode_counts = Counter(mode["code"] for row in rows for mode in row["failure_modes"])
    clean = sum(1 for rec in records if rec.get("quality") == "CLEAN")
    return {
        "total": len(records),
        "clean": clean,
        "suspect": len(records) - clean,
        "mode_counts": dict(sorted(mode_counts.items(), key=lambda item: (-item[1], item[0]))),
        "rows": returned[: max(0, limit)],
    }


def alignment_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Audit one GT record and classify the source of its discrepancy."""

    sky = _number(rec.get("sky_cons_px"))
    pfm = _number(rec.get("pfm_cons_px"))
    contour = _number(rec.get("contour_cons_px"))
    pfm_offset = _number(rec.get("pfm_offset_px"))
    sky_error_m = _number(rec.get("sky_error_m"))
    pfm_error_m = _number(rec.get("pfm_error_m"))
    sky_support = _number(rec.get("sky_support"))
    pfm_support = _number(rec.get("pfm_support"))
    dyaw = abs(_number(rec.get("dyaw_deg")) or 0.0)
    de = _number(rec.get("de_m")) or 0.0
    dn = _number(rec.get("dn_m")) or 0.0
    position_delta = math.hypot(de, dn)

    failure_modes: list[dict[str, Any]] = []
    _append_if_over(failure_modes, "dem_observed_skyline_mismatch", "DEM vs observed skyline", sky, SKY_CONS_GATE_PX)
    _append_if_over(failure_modes, "dem_pfm_skyline_mismatch", "DEM vs PFM skyline", pfm, PFM_CONS_GATE_PX)
    _append_if_over(
        failure_modes,
        "dem_observed_skyline_meter_mismatch",
        "DEM vs observed skyline vertical-equivalent error",
        sky_error_m,
        SKY_ERROR_GATE_M,
    )
    _append_if_over(
        failure_modes,
        "dem_pfm_skyline_meter_mismatch",
        "DEM vs PFM skyline vertical-equivalent error",
        pfm_error_m,
        PFM_ERROR_GATE_M,
    )
    _append_if_over(failure_modes, "dem_outline_mismatch", "DEM vs outline contours", contour, CONTOUR_CONS_GATE_PX)
    _append_if_over(
        failure_modes,
        "photo_pfm_registration_mismatch",
        "photo vs PFM skyline registration",
        pfm_offset,
        PFM_PHOTO_OFFSET_GATE_PX,
    )
    _append_if_below(
        failure_modes,
        "weak_photo_skyline_support",
        "observed skyline weakly supported by photo",
        sky_support,
        SKY_SUPPORT_MIN,
    )
    _append_if_below(
        failure_modes,
        "weak_pfm_photo_support",
        "PFM skyline weakly supported by photo",
        pfm_support,
        SKY_SUPPORT_MIN,
    )
    _append_if_over(failure_modes, "large_yaw_correction", "large yaw correction", dyaw, YAW_DELTA_GATE_DEG)
    _append_if_over(
        failure_modes,
        "large_position_correction",
        "large horizontal position correction",
        position_delta,
        POSITION_DELTA_GATE_M,
    )

    severity = max((mode["ratio"] for mode in failure_modes), default=0.0)
    return {
        "name": rec.get("name"),
        "quality": rec.get("quality"),
        "severity": round(float(severity), 3),
        "obs_source": rec.get("obs_source"),
        "metrics": {
            "sky_cons_px": sky,
            "pfm_cons_px": pfm,
            "contour_cons_px": contour,
            "sky_error_m": sky_error_m,
            "sky_error_p90_m": _number(rec.get("sky_error_p90_m")),
            "pfm_error_m": pfm_error_m,
            "pfm_error_p90_m": _number(rec.get("pfm_error_p90_m")),
            "sky_range_median_m": _number(rec.get("sky_range_median_m")),
            "pfm_offset_px": pfm_offset,
            "sky_support": sky_support,
            "pfm_support": pfm_support,
            "dyaw_deg": _number(rec.get("dyaw_deg")),
            "position_delta_m": round(position_delta, 1),
        },
        "failure_modes": failure_modes,
        "reasons": rec.get("reasons", []),
    }


def _append_if_over(modes: list[dict[str, Any]], code: str, label: str, value: float | None, threshold: float) -> None:
    if value is None or value <= threshold:
        return
    modes.append(
        {
            "code": code,
            "label": label,
            "value": round(value, 3),
            "threshold": threshold,
            "ratio": round(value / threshold, 3),
        }
    )


def _append_if_below(modes: list[dict[str, Any]], code: str, label: str, value: float | None, threshold: float) -> None:
    if value is None or value >= threshold:
        return
    ratio = (threshold - value) / max(threshold, 1e-9)
    modes.append(
        {"code": code, "label": label, "value": round(value, 3), "threshold": threshold, "ratio": round(ratio, 3)}
    )


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except TypeError, ValueError:
        return None
    return out if math.isfinite(out) else None


def skyline_vertical_error_stats_m(
    observed_rows: np.ndarray,
    dem_rows: np.ndarray,
    dem_range_m: np.ndarray,
    width_px: int,
    height_px: int,
    fov_deg: float,
) -> dict[str, float | None]:
    """Vertical-equivalent terrain error for skyline row disagreement.

    Rows are converted back to tangent elevation using the GeoPose cylindrical-tangent camera
    model. At the DEM skyline range for each column, the tangent-elevation gap becomes meters:
    ``abs(tan(el_obs) - tan(el_dem)) * range``.
    """

    tan_obs = tangent_elevation_from_rows(observed_rows, width_px, height_px, fov_deg, "cyltan")
    tan_dem = tangent_elevation_from_rows(dem_rows, width_px, height_px, fov_deg, "cyltan")
    err_m = np.abs(tan_obs - tan_dem) * dem_range_m.astype(float)
    ok = np.isfinite(err_m) & np.isfinite(dem_range_m) & (dem_range_m > 0)
    if ok.sum() == 0:
        return {"mean_m": None, "median_m": None, "p90_m": None, "range_median_m": None}
    values = err_m[ok]
    ranges = dem_range_m[ok].astype(float)
    return {
        "mean_m": round(float(np.mean(values)), 1),
        "median_m": round(float(np.median(values)), 1),
        "p90_m": round(float(np.percentile(values, 90)), 1),
        "range_median_m": round(float(np.median(ranges)), 1),
    }


def metric_skyline_errors_for_record(
    rec: dict[str, Any],
    *,
    data_dir: Path = GEOPOSE_DIR,
    tiles_dir: Path = COP_TILES_DIR,
    out_dir: Path = GTV2_DIR,
) -> dict[str, float | None]:
    """Compute metric-space skyline errors for an existing GT-v2 record."""

    from peakle.localize.copdem import load_cop_around
    from peakle.localize.geopose import load_sample, resampled_oracle_skyline
    from peakle.localize.gtrefine import crop_az_deg, dem_skyline_with_range

    name = str(rec["name"])
    sample = load_sample(data_dir / name)
    arrays = np.load(out_dir / f"{name}.npz")
    obs = arrays["gt_skyline"].astype(float)
    pfm = (
        arrays["pfm_skyline"].astype(float)
        if "pfm_skyline" in arrays.files
        else resampled_oracle_skyline(sample.depth_path, int(rec["width"]), int(rec["height"]))
    )
    dem = arrays["dem_skyline"].astype(float)
    terrain = load_cop_around(tiles_dir, sample.lat, sample.lon, extent_m=90000.0, grid=3000)
    az = crop_az_deg(int(rec["width"]), float(rec["fov_deg"]), float(rec["yaw_deg"]))
    _, range_m = dem_skyline_with_range(
        terrain,
        float(rec["cam_z_m"]),
        az,
        int(rec["width"]),
        int(rec["height"]),
        float(rec["fov_deg"]),
        float(rec["de_m"]),
        float(rec["dn_m"]),
    )
    obs_stats = skyline_vertical_error_stats_m(
        obs, dem, range_m, int(rec["width"]), int(rec["height"]), float(rec["fov_deg"])
    )
    pfm_stats = skyline_vertical_error_stats_m(
        pfm, dem, range_m, int(rec["width"]), int(rec["height"]), float(rec["fov_deg"])
    )
    return {
        "sky_error_m": obs_stats["mean_m"],
        "sky_error_median_m": obs_stats["median_m"],
        "sky_error_p90_m": obs_stats["p90_m"],
        "pfm_error_m": pfm_stats["mean_m"],
        "pfm_error_median_m": pfm_stats["median_m"],
        "pfm_error_p90_m": pfm_stats["p90_m"],
        "sky_range_median_m": obs_stats["range_median_m"],
    }


# --- verdict calibration (from benchmark results) ---

_CALIB_KEYS = (
    ("alias", False),
    ("snr", False),
    ("well", True),
    ("chamfer", True),
    ("coverage", False),
    ("agreement", False),
)


def load_bench_rows(paths: list[str], include_auto: bool = False) -> list[dict]:
    """Flatten benchmark results.json file(s) into per-solve rows (oracle + extracted tracks)."""

    rows = []
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            if "error" in r or (not include_auto and not r.get("manual")):
                continue
            for track in ("oracle", "extracted"):
                t = r.get(track)
                if not t or "chamfer_px" not in t:
                    continue
                rows.append(
                    {
                        "sample": r["name"],
                        "track": track,
                        "correct": bool(t["correct"]),
                        "chamfer": t["chamfer_px"],
                        "coverage": t["coverage"],
                        "well": t["well_width_deg"],
                        "alias": t["alias_ratio"],
                        "snr": t.get("snr", np.nan),
                        "agreement": t.get("agreement", np.nan),
                        "verdict": t["verdict"],
                    }
                )
    return rows


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUC: P(random correct-solve value > random wrong-solve value)."""

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def diagnostic_aucs(rows: list[dict]) -> list[tuple[str, float, float, float]]:
    """Per-diagnostic (name, median-correct, median-wrong, correctness-AUC)."""

    good = [r for r in rows if r["correct"]]
    bad = [r for r in rows if not r["correct"]]
    out = []
    for key, invert in _CALIB_KEYS:
        gp = np.array([r[key] for r in good if np.isfinite(r[key])])
        bp = np.array([r[key] for r in bad if np.isfinite(r[key])])
        a = auc(gp, bp)
        if invert and np.isfinite(a):
            a = 1.0 - a
        med_gp = float(np.median(gp)) if gp.size else float("nan")
        med_bp = float(np.median(bp)) if bp.size else float("nan")
        out.append((key, med_gp, med_bp, a))
    return out


def best_precision_gate(rows: list[dict], min_precision: float = 0.999) -> dict | None:
    """Search the (alias, well, chamfer, coverage, snr) grid for the max-recall gate at ``min_precision``."""

    good = [r for r in rows if r["correct"]]
    best = None
    for alias_t, well_t, ch_t, cov_t, snr_t in itertools.product(
        [1.05, 1.10, 1.15, 1.20, 1.25, 1.35, 1.50],
        [10.0, 15.0, 20.0, 25.0, 35.0, 360.0],
        [10.0, 15.0, 20.0, 25.0, 30.0, 36.0],
        [0.25, 0.35, 0.50, 0.65],
        [0.0, 1.5, 2.0, 3.0],
    ):
        sel = [
            r
            for r in rows
            if r["alias"] >= alias_t
            and r["well"] <= well_t
            and r["chamfer"] <= ch_t
            and r["coverage"] >= cov_t
            and (not np.isfinite(r["snr"]) or r["snr"] >= snr_t)
        ]
        if not sel:
            continue
        prec = sum(r["correct"] for r in sel) / len(sel)
        recall = sum(r["correct"] for r in sel) / max(len(good), 1)
        if prec >= min_precision and (best is None or recall > best["recall"]):
            best = {
                "recall": recall,
                "alias": alias_t,
                "well": well_t,
                "chamfer": ch_t,
                "coverage": cov_t,
                "snr": snr_t,
                "n": len(sel),
            }
    return best
