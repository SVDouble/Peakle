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
from pathlib import Path

import numpy as np

from peakle.localize.paths import GTV2_DIR

# support-gate thresholds (initial; tighten as the corpus grows)
SKY_SUPPORT_MIN = 0.50
OCC_SUPPORT_MIN = 0.20
OCC_DENSITY_MIN = 0.30
SUPPORT_FAMILIES = ("sky", "occ", "rib", "cou")


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
