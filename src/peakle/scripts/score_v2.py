"""Benchmark scoring v2: join a bench run with the refined, quality-tiered GT (GT v2).

Reports, per track (oracle / extracted):
  - success vs the RAW label and vs the REFINED yaw, at 5° and at the tighter 2° threshold the
    refined GT makes meaningful (raw-label yaw noise is median 0.8°, so 2° was meaningless
    before);
  - split by tier: ALL / MANUAL / CLEAN (only CLEAN is a valid scoring set — SUSPECT samples
    carry known label or terrain problems and must never grade a solver or extractor);
  - the outline picture per sample: GT contour density, DEM<->GT contour agreement — which
    samples can support outline-based matching at all.

Usage: python -m peakle.scripts.score_v2 <bench_results.json> [--gtv2 local/derived/gt_v2/index.json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from peakle.localize.paths import BASE


def ang(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--gtv2", default=str(BASE / "local/derived/gt_v2/index.json"))
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.results)) if "error" not in r]
    gtv2 = {r["name"]: r for r in json.load(open(args.gtv2))}
    joined = [(r, gtv2.get(r["name"])) for r in rows]
    n_missing = sum(1 for _, g in joined if g is None)
    if n_missing:
        print(f"NOTE: {n_missing}/{len(rows)} samples have no GT v2 record yet (batch still running?)")

    def rate(sel, track, ref, tol):
        num = den = 0
        for r, g in sel:
            s = r.get(track)
            if not isinstance(s, dict) or "yaw" not in s:
                continue
            den += 1
            target = r["gt_yaw"] if ref == "raw" else g["yaw_deg"]
            if abs(ang(s["yaw"] - target)) <= tol:
                num += 1
        return f"{num}/{den} ({num / den:.0%})" if den else "n/a"

    tiers = {
        "ALL": [j for j in joined if j[1]],
        "MANUAL": [j for j in joined if j[1] and j[0]["manual"]],
        "CLEAN": [j for j in joined if j[1] and j[1]["quality"] == "CLEAN"],
        "CLEAN+MANUAL": [j for j in joined if j[1] and j[1]["quality"] == "CLEAN" and j[0]["manual"]],
    }
    print(
        f"\n{'tier':14} {'n':>3}  {'oracle@5raw':>12} {'oracle@5ref':>12} {'oracle@2ref':>12}  "
        f"{'extr@5ref':>10} {'extr@2ref':>10}"
    )
    for tier, sel in tiers.items():
        print(
            f"{tier:14} {len(sel):3}  {rate(sel, 'oracle', 'raw', 5):>12} {rate(sel, 'oracle', 'ref', 5):>12} "
            f"{rate(sel, 'oracle', 'ref', 2):>12}  "
            f"{rate(sel, 'extracted', 'ref', 5):>10} {rate(sel, 'extracted', 'ref', 2):>10}"
        )

    # outline availability: which samples can support contour-based matching
    dens = [g["gt_contour_density"] for _, g in joined if g]
    cts = [g["contour_cons_px"] for _, g in joined if g and g["contour_cons_px"] is not None]
    usable = sum(1 for _, g in joined if g and g["gt_contour_density"] >= 0.3 and (g["contour_cons_px"] or 99) <= 25)
    print(
        f"\noutlines: GT contour density median {np.median(dens):.2f}; DEM<->GT contour agreement "
        f"median {np.median(cts):.1f}px (n={len(cts)}); samples usable for outline matching "
        f"(density>=0.3 & agreement<=25px): {usable}/{len(dens)}"
    )

    sus = [(r["name"], g["reasons"]) for r, g in joined if g and g["quality"] == "SUSPECT"]
    if sus:
        print(f"\nSUSPECT ({len(sus)}):")
        for n, reasons in sus:
            print(f"  {n[:44]:46} {'; '.join(reasons)}")


if __name__ == "__main__":
    main()
