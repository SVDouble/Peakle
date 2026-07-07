"""Rescue bound-hitters: re-localize badly mislabeled poses with the full-360° solver.

Forensics finding: 9 of the 12 worst GT v2 samples have their pose polish pinned at the search
bounds (label yaw off by >6°, position by hundreds of metres — mostly flickr_sge additions).
Their PHOTO skylines are trusted (support 0.86-1.0), so the label can be re-derived from scratch:
run the full-360° HorizonProfile solver on the photo skyline, adopt its yaw as the new label
centre, and re-run the normal contour-arbitrated polish around it.  Reports old vs rescued
reconstruction plus the solver's own honesty diagnostics (a rescue is only proposed when the
solve is not an alias).

Usage: python -m peakle.scripts.rescue_relabel [--max-n 9]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
from scipy.ndimage import distance_transform_edt

from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample, read_pfm
from peakle.localize.gtrefine import gt_contour_mask, refine_pose
from peakle.localize.paths import BASE
from peakle.localize.paths import GTV2_DIR as GTV2
from peakle.localize.solve import HorizonProfile, solve_orientation


def is_bound_hitter(r: dict) -> bool:
    return (
        r.get("obs_source") == "photo"
        and (abs(r.get("dyaw_deg", 0)) >= 5.9 or max(abs(r.get("de_m", 0)), abs(r.get("dn_m", 0))) >= 170)
        and (r.get("sky_cons_px") or 0) > 15
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=9)
    args = ap.parse_args()
    outdir = BASE / f"local/output/{datetime.now():%Y%m%d-%H%M%S}-rescue"
    outdir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(p.read_text()) for p in GTV2.glob("*.json") if p.name != "index.json"]
    hitters = sorted((r for r in recs if is_bound_hitter(r)), key=lambda r: -r["sky_cons_px"])[: args.max_n]
    print(f"{len(hitters)} bound-hitters to rescue")

    for r in hitters:
        name = r["name"]
        t0 = time.time()
        try:
            s = load_sample(BASE / "local/data/geopose" / name)
            w, h = r["width"], r["height"]
            obs = np.load(GTV2 / f"{name}.npz")["gt_skyline"].astype(float)
            terrain = load_cop_around(BASE / "local/data/copernicus", s.lat, s.lon, 90000.0, 3000)
            cam_z0 = max(s.elev_m, terrain.elevation_at(0.0, 0.0) + 2.0)
            profile = HorizonProfile(terrain, cam_z0, step=25.0)
            solve = solve_orientation(
                obs, h, profile, fov_deg=r["fov_deg"], projection="cyltan", pitch_bounds=(-50.0, 50.0)
            )
            gt_mask = gt_contour_mask(read_pfm(s.depth_path), w, h)
            gt_dt = distance_transform_edt(~gt_mask) if gt_mask.sum() >= 200 else None
            fit = refine_pose(terrain, cam_z0, obs, w, h, r["fov_deg"], solve.yaw_deg, gt_dt)
            total_dyaw = (solve.yaw_deg + fit["dyaw"] - (r["yaw_deg"] - r["dyaw_deg"]) + 180) % 360 - 180
            print(
                json.dumps(
                    {
                        "name": name,
                        "old_cons": r["sky_cons_px"],
                        "rescued_cons": round(fit["cons"], 1),
                        "label_yaw_error_deg": round(total_dyaw, 1),
                        "solver": {
                            "alias": round(solve.alias_ratio, 2),
                            "snr": round(solve.snr, 1),
                            "verdict": solve.verdict,
                        },
                    }
                )
                + f"  [{time.time() - t0:.0f}s]",
                flush=True,
            )
        except Exception as exc:
            print(f"{name}: ERROR {exc}", flush=True)
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
