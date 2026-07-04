"""Calibrate the solve verdict against benchmark outcomes.

Reads bench_geopose results.json file(s) and answers, with numbers instead of vibes:
  1. Which no-GT diagnostics (alias ratio, basin width, chamfer, coverage, extractor agreement)
     actually separate correct solves from wrong ones?  (per-feature AUC + medians)
  2. What thresholds give a CONFIRMED gate with ~100% precision (no wrong solve ever confirmed)
     at the highest achievable recall?

The chosen thresholds are printed as a ready-to-paste `_provisional_verdict` replacement.  Scoring
is restricted to MANUAL-GT samples by default (AUTO ground truth is itself unreliable).

Usage: python scripts/calibrate_verdict.py local/output/<dt>-geopose-bench/results.json [more.json]
       [--include-auto]
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np


def load_rows(paths: list[str], include_auto: bool) -> list[dict]:
    rows = []
    for p in paths:
        for r in json.load(open(p)):
            if "error" in r or (not include_auto and not r.get("manual")):
                continue
            for track in ("oracle", "extracted"):
                t = r.get(track)
                if not t or "chamfer_px" not in t:  # e.g. "no usable skyline" records
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
    grid = (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()
    return float(grid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--include-auto", action="store_true")
    args = ap.parse_args()
    rows = load_rows(args.results, args.include_auto)
    good = [r for r in rows if r["correct"]]
    bad = [r for r in rows if not r["correct"]]
    print(f"{len(rows)} solves ({len(good)} correct, {len(bad)} wrong) from {len(args.results)} file(s)\n")

    print(f"{'diagnostic':12} {'med correct':>12} {'med wrong':>12} {'AUC':>6}   (AUC ~0.5 = useless)")
    for key, invert in [("alias", False), ("snr", False), ("well", True), ("chamfer", True), ("coverage", False), ("agreement", False)]:
        gp = np.array([r[key] for r in good if np.isfinite(r[key])])
        bp = np.array([r[key] for r in bad if np.isfinite(r[key])])
        a = auc(gp, bp)
        if invert and np.isfinite(a):
            a = 1.0 - a
        print(f"{key:12} {np.median(gp) if gp.size else float('nan'):12.3f} {np.median(bp) if bp.size else float('nan'):12.3f} {a:6.2f}")

    # current in-code verdict performance
    conf = [r for r in rows if r["verdict"] == "CONFIRMED"]
    n_ok = sum(1 for r in conf if r["correct"])
    print(f"\ncurrent verdict: CONFIRMED on {len(conf)} solves, {n_ok} correct "
          f"-> precision {n_ok/len(conf):.2f}" if conf else "\ncurrent verdict: nothing CONFIRMED")
    missed = sum(1 for r in good if r["verdict"] != "CONFIRMED")
    print(f"correct solves not confirmed (lost recall): {missed}/{len(good)}")

    # threshold search: maximise recall subject to precision == 1.0 on this data
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
        rec = sum(r["correct"] for r in sel) / max(len(good), 1)
        if prec >= 0.999 and (best is None or rec > best[0]):
            best = (rec, alias_t, well_t, ch_t, cov_t, snr_t, len(sel))
    if best:
        rec, a_t, w_t, c_t, cv_t, s_t, n = best
        print(
            f"\nbest 100%-precision gate on this data (n={n} confirmed, recall {rec:.0%}):\n"
            f"  CONFIRMED iff alias_ratio >= {a_t} AND well_width <= {w_t} AND "
            f"chamfer <= {c_t}px AND coverage >= {cv_t} AND snr >= {s_t}"
        )
        print("NB: thresholds chosen on this benchmark; re-run when the benchmark grows. "
              "Precision=1.0 here does NOT guarantee 1.0 in the field.")
    else:
        print("\nno threshold combination reaches 100% precision — inspect false positives first")


if __name__ == "__main__":
    main()
