"""Fine-map registration for the largest-error Swiss samples.

Strategy (user-specified): ROUGH search happened on Copernicus 30 m (GT v2 record); this stage
re-registers the pose on the swissALTI3D 2 m patched render with WIDE bounds — the coarse map
had no teeth to lock onto, so its pose can be off by more than the usual polish radius.

For each of the worst-N in-Switzerland samples (by sky_cons): fetch tiles, joint (yaw × position)
search on the patched render (half-resolution columns for speed), fine polish, report
before/after sky consistency and write an overlay.

Usage: python -m peakle.scripts.swiss_refine [--worst-n 6] [--samples a,b,c]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample
from peakle.localize.gtrefine import crop_az_deg, dem_skyline, shift_align
from peakle.localize.paths import BASE
from peakle.localize.paths import GTV2_DIR as GTV2
from peakle.localize.paths import SWISS_DIR as SWISS
from peakle.localize.swissdem import ensure_swiss_tiles, in_switzerland, load_swiss_patch


def refine_sample(name: str, outdir: Path) -> dict:
    rec = json.loads((GTV2 / f"{name}.json").read_text())
    s = load_sample(BASE / "local/data/geopose" / name)
    w, h = rec["width"], rec["height"]
    obs = np.load(GTV2 / f"{name}.npz")["gt_skyline"].astype(float)
    terrain = load_cop_around(BASE / "local/data/copernicus", s.lat, s.lon, 90000.0, 3000)
    ensure_swiss_tiles(SWISS, s.lat, s.lon, radius_m=4000.0)
    patch = load_swiss_patch(SWISS, s.lat, s.lon, res=4.0)
    if patch is None:
        return {"name": name, "skip": "no patch coverage"}

    f_px = w / np.radians(rec["fov_deg"])
    dvs = np.arange(-int(f_px * np.tan(np.radians(50))) - 1, int(f_px * np.tan(np.radians(50))) + 2, 8)
    obs_half = obs[::2]

    def cons(dyaw, de, dn, full=False, use_patch=True):
        az = crop_az_deg(w, rec["fov_deg"], rec["yaw_deg"] + dyaw)
        rows = dem_skyline(
            terrain,
            rec["cam_z_m"],
            az if full else az[::2],
            w,
            h,
            rec["fov_deg"],
            rec["de_m"] + de,
            rec["dn_m"] + dn,
            patch=patch if use_patch else None,
        )
        c, dv = shift_align(obs if full else obs_half, rows, dvs, step=3)
        return c, dv, rows

    c_before, _, _ = cons(0.0, 0.0, 0.0, full=True, use_patch=False)
    c_unreg, _, _ = cons(0.0, 0.0, 0.0, full=True)

    best = (np.inf, 0.0, 0.0, 0.0)
    for dyaw in np.arange(-2.5, 2.51, 0.5):
        for de in (-180.0, 0.0, 180.0):
            for dn in (-180.0, 0.0, 180.0):
                c, _, _ = cons(float(dyaw), de, dn)
                if c < best[0]:
                    best = (c, float(dyaw), de, dn)
    for de in best[2] + np.asarray((-90.0, -45.0, 0.0, 45.0, 90.0)):
        for dn in best[3] + np.asarray((-90.0, -45.0, 0.0, 45.0, 90.0)):
            c, _, _ = cons(best[1], float(de), float(dn))
            if c < best[0]:
                best = (c, best[1], float(de), float(dn))
    for dyaw in best[1] + np.asarray((-0.25, 0.0, 0.25)):
        for de in best[2] + np.asarray((-20.0, 0.0, 20.0)):
            for dn in best[3] + np.asarray((-20.0, 0.0, 20.0)):
                c, _, _ = cons(float(dyaw), float(de), float(dn))
                if c < best[0]:
                    best = (c, float(dyaw), float(de), float(dn))
    c_fine, dv, rows = cons(best[1], best[2], best[3], full=True)

    rgb = np.asarray(Image.open(s.photo_path).convert("RGB").resize((w, h)), np.uint8)
    im = Image.fromarray(np.clip(rgb.astype(float) * 1.1, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for r_, col in ((obs, (0, 230, 90)), (rows + dv, (0, 200, 255))):
        pts = [(c_, float(r_[c_])) for c_ in range(w) if np.isfinite(r_[c_]) and 0 <= r_[c_] < h]
        if len(pts) > 1:
            dr.line(pts, fill=col, width=2)
    im.save(outdir / f"{name}.jpg", quality=86)
    return {
        "name": name,
        "cons_copernicus": round(float(c_before), 1),
        "cons_patch_unregistered": round(float(c_unreg), 1),
        "cons_patch_registered": round(float(c_fine), 1),
        "dyaw": best[1],
        "de": best[2],
        "dn": best[3],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst-n", type=int, default=6)
    ap.add_argument("--samples", default=None)
    args = ap.parse_args()
    from datetime import datetime

    outdir = BASE / f"local/output/{datetime.now():%Y%m%d-%H%M%S}-swiss-refine"
    outdir.mkdir(parents=True, exist_ok=True)
    index = json.loads((GTV2 / "index.json").read_text())
    if args.samples:
        names = args.samples.split(",")
    else:
        ch = []
        for r in sorted(index, key=lambda r: -(r["sky_cons_px"] or 0)):
            s = load_sample(BASE / "local/data/geopose" / r["name"])
            if in_switzerland(s.lat, s.lon):
                ch.append(r["name"])
            if len(ch) >= args.worst_n:
                break
        names = ch
    for name in names:
        t0 = time.time()
        try:
            rep = refine_sample(name, outdir)
        except Exception as exc:
            rep = {"name": name, "error": str(exc)}
        print(json.dumps(rep) + f"  [{time.time() - t0:.0f}s]", flush=True)
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
