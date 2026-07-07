"""Forensics on the worst GT v2 offenders: what actually happens on each of them.

Assembles per-sample records (fresh *.json, not the possibly-stale index), takes the worst-N by
skyline reconstruction, gathers every diagnostic — pose-polish corrections, observation source,
pfm registration offset, photo support (if built), the missing-foreground check (Depth-Anything
vs DEM near-depth), Switzerland/fine-DEM eligibility — and CLASSIFIES each into failure causes.
Writes a text report + photo/skyline composites for the top offenders.

Usage: python scripts/worst_forensics.py [--worst-n 12]
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw

from peakle.localize.copdem import load_cop_around
from peakle.localize.fg_check import foreground_report, photo_mono_depth
from peakle.localize.geopose import load_sample
from peakle.localize.gtrefine import crop_az_deg, dem_depth_image
from peakle.localize.swissdem import in_switzerland

from peakle.localize.paths import BASE, GTV2_DIR as GTV2


def classify(rec: dict, fg: dict | None, ch: bool) -> list[str]:
    causes = []
    if fg and fg.get("missing_foreground"):
        causes.append(f"MISSING NEAR FEATURE (photo fg {fg['photo_fg_frac']:.0%} vs dem {fg['dem_fg_frac']:.0%})")
    if abs(rec.get("dyaw_deg", 0)) >= 5.9 or max(abs(rec.get("de_m", 0)), abs(rec.get("dn_m", 0))) >= 170:
        causes.append("label pose AT/BEYOND search bounds")
    if rec.get("obs_source") == "pfm" and (rec.get("pfm_offset_px") or 0) > 15:
        causes.append(f"pfm mis-registered ({rec['pfm_offset_px']:.0f}px) & photo skyline untrusted")
    if rec.get("obs_source") == "pfm" and (rec.get("obs_support") or 1) < 0.5:
        causes.append(f"photo skyline weak (support {rec.get('obs_support')})")
    if ch:
        causes.append("CH -> fine-DEM registration candidate (scripts/swiss_refine.py)")
    if not causes:
        causes.append("unexplained — eyeball the composite")
    return causes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worst-n", type=int, default=12)
    args = ap.parse_args()
    outdir = BASE / f"local/output/{datetime.now():%Y%m%d-%H%M%S}-forensics"
    outdir.mkdir(parents=True, exist_ok=True)

    recs = [json.loads(p.read_text()) for p in GTV2.glob("*.json") if p.name != "index.json"]
    recs.sort(key=lambda r: -(r.get("sky_cons_px") or 0))
    worst = recs[: args.worst_n]
    lines = []
    for i, rec in enumerate(worst):
        name = rec["name"]
        t0 = time.time()
        try:
            s = load_sample(BASE / "local/data/geopose" / name)
            w, h = rec["width"], rec["height"]
            rgb = np.asarray(Image.open(s.photo_path).convert("RGB").resize((w, h)), np.uint8)
            terrain = load_cop_around(BASE / "local/data/copernicus", s.lat, s.lon, 90000.0, 3000)
            az = crop_az_deg(w, rec["fov_deg"], rec["yaw_deg"])
            depth, *_ = dem_depth_image(
                terrain,
                rec["cam_z_m"],
                az,
                w,
                h,
                rec["fov_deg"],
                rec["dv_px"],
                rec["de_m"],
                rec["dn_m"],
                rec["tilt_deg"],
                sub=4,
            )
            z = np.load(GTV2 / f"{name}.npz")
            mono = photo_mono_depth(rgb)
            fg = (
                foreground_report(mono, depth, sky_rows=z["gt_skyline"].astype(float)[::1])
                if mono is not None
                else None
            )
            ch = in_switzerland(s.lat, s.lon)
            causes = classify(rec, fg, ch)
            hdr = (
                f"{name}  [{rec['quality']}] sky={rec['sky_cons_px']}px ct={rec.get('contour_cons_px')} "
                f"obs={rec.get('obs_source')}(sup {rec.get('obs_support')}) pfm_off={rec.get('pfm_offset_px')} "
                f"dyaw={rec.get('dyaw_deg'):+.1f} pos=({rec.get('de_m'):+.0f},{rec.get('dn_m'):+.0f})"
            )
            lines.append(hdr)
            for c in causes:
                lines.append(f"    -> {c}")
            # composite for eyeballing
            im = Image.fromarray(np.clip(rgb.astype(float) * 1.1, 0, 255).astype(np.uint8))
            dr = ImageDraw.Draw(im)
            for rows, col in (
                (z["gt_skyline"].astype(float), (0, 230, 90)),
                (z["dem_skyline"].astype(float), (0, 200, 255)),
            ):
                pts = [(c_, float(rows[c_])) for c_ in range(w) if np.isfinite(rows[c_]) and 0 <= rows[c_] < h]
                if len(pts) > 1:
                    dr.line(pts, fill=col, width=2)
            im.save(outdir / f"{i:02d}_{name[:50]}.jpg", quality=85)
            print(lines[-len(causes) - 1], flush=True)
            for c in causes:
                print("    ->", c, flush=True)
        except Exception as exc:
            lines.append(f"{name}: ERROR {exc}")
            print(lines[-1], flush=True)
        lines.append(f"    [{time.time() - t0:.0f}s]")
    (outdir / "report.txt").write_text("\n".join(lines))
    print(f"-> {outdir}")


if __name__ == "__main__":
    main()
