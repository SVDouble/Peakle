"""GT quality panorama: one tile per bench sample for visual inspection of the ground truth.

Each tile overlays what must agree if the GT (and our reconstruction of it) is good:
  green  = oracle skyline (from the GT depth render)
  cyan   = our DEM skyline at the GT yaw (TRUE-cylindrical tan mapping, best vertical shift,
           after a small POSITION POLISH around the GT coordinates — GPS is tens of metres off
           and near-field skylines move with it)
  orange = internal contours (occlusion boundaries between ridge layers), reconstructed from the
           DEM at the same pose via depth discontinuities — no GT lines exist for these, but with
           known extrinsics they follow from the terrain

Tiles are sorted clean -> dirty by the polished GT consistency.  Output: tiles + a self-contained
panno.html (artifact-ready).

Usage: python -m peakle.scripts.gt_panno <bench_results.json> [--out DIR] [--no-polish]
"""

from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample, oracle_skyline, read_pfm
from peakle.localize.gtrefine import (
    crop_az_deg,
    dem_contour_mask,
    gt_contour_mask,
    refine_pose,
)
from peakle.localize.paths import BASE
from peakle.localize.paths import COP_TILES_DIR as TILES
from peakle.localize.paths import GEOPOSE_DIR as DATA

TILE_W = 640
GREEN, CYAN, ORANGE = (0, 230, 90), (0, 200, 255), (255, 150, 30)


def build_tile(rec: dict, out: Path, do_polish: bool) -> dict:
    s = load_sample(DATA / rec["name"])
    rgb = np.asarray(Image.open(s.photo_path).convert("RGB"), np.uint8)
    scale = TILE_W / rgb.shape[1]
    rgb = np.asarray(
        Image.fromarray(rgb).resize((TILE_W, round(rgb.shape[0] * scale)), Image.Resampling.BILINEAR), np.uint8
    )
    h, w = rgb.shape[:2]

    o = oracle_skyline(s.depth_path)
    dep_h = read_pfm(s.depth_path).shape[0]
    x = np.linspace(0, 1, len(o))
    fin = np.isfinite(o)
    obs = np.where(
        np.interp(np.linspace(0, 1, w), x, fin.astype(float)) > 0.5,
        np.interp(np.linspace(0, 1, w), x[fin], o[fin]) * (h / dep_h),
        np.nan,
    )

    terrain = load_cop_around(TILES, s.lat, s.lon, extent_m=90000.0, grid=3000)
    cam_z = max(s.elev_m, terrain.elevation_at(0, 0) + 2.0)

    gt_mask = gt_contour_mask(read_pfm(s.depth_path), w, h)
    gt_dt = distance_transform_edt(~gt_mask) if gt_mask.sum() >= 200 else None
    fit = refine_pose(terrain, cam_z, obs, w, h, s.fov_deg, s.yaw_gt_deg, gt_dt if do_polish else None)
    fit["dE"], fit["dN"] = fit.pop("de"), fit.pop("dn")

    az = crop_az_deg(w, s.fov_deg, s.yaw_gt_deg + fit["dyaw"])
    contours = dem_contour_mask(
        terrain, fit["cam_z"], az, w, h, s.fov_deg, fit["dv"], fit["dE"], fit["dN"], fit["tilt"]
    )

    im = Image.fromarray(np.clip(rgb.astype(float) * 1.15, 0, 255).astype(np.uint8))
    px = im.load()
    assert px is not None  # this in-memory RGB image always exposes pixel access
    dem_rows = fit["rows"] + fit["dv"]
    obs_arr = obs
    # GT internal contours (from the GT depth itself) — the reference the DEM contours must match
    rr, cc = np.nonzero(gt_mask)
    for r, c in zip(rr.tolist(), cc.tolist(), strict=True):
        if np.isfinite(obs_arr[c]) and abs(r - obs_arr[c]) < 5:
            continue  # skyline drawn separately
        px[c, r] = GREEN
    rr, cc = np.nonzero(contours)
    for r, c in zip(rr.tolist(), cc.tolist(), strict=True):
        if np.isfinite(dem_rows[c]) and abs(r - dem_rows[c]) < 5:
            continue  # the skyline itself is drawn in cyan; don't double it in orange
        px[c, r] = ORANGE
        if c + 1 < w:
            px[c + 1, r] = ORANGE
        if r + 1 < h:
            px[c, r + 1] = ORANGE
    dr = ImageDraw.Draw(im)
    for rows, color in [(dem_rows, CYAN), (obs, GREEN)]:
        pts = [(c, float(rows[c])) for c in range(w) if np.isfinite(rows[c]) and 0 <= rows[c] < h]
        if len(pts) > 1:
            dr.line(pts, fill=color, width=2)
    im.save(out / f"{rec['name']}.jpg", quality=80)
    return {
        "cons": round(float(fit["cons"]), 1),
        "ccons": round(float(fit["ccons"]), 1) if fit.get("ccons") is not None else None,
        "dE": fit["dE"],
        "dN": fit["dN"],
        "dyaw": round(fit["dyaw"], 2),
        "tilt": round(fit["tilt"], 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-polish", action="store_true")
    args = ap.parse_args()
    out = Path(args.out) if args.out else BASE / f"local/output/{datetime.now():%Y%m%d-%H%M%S}-gt-panno"
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in json.load(open(args.results)) if "error" not in r]
    for i, r in enumerate(rows):
        try:
            r["_panno"] = build_tile(r, out, not args.no_polish)
        except Exception as e:
            print("tile failed", r["name"], e)
            r["_panno"] = {"cons": float("nan"), "dE": 0, "dN": 0, "dyaw": 0.0, "tilt": 0.0}
        p = r["_panno"]
        print(
            f"[{i + 1}/{len(rows)}] {r['name']} cons={p['cons']} contours={p.get('ccons')} dyaw={p['dyaw']:+.1f} dE={p['dE']:+.0f} dN={p['dN']:+.0f} tilt={p['tilt']:+.1f}",
            flush=True,
        )
    rows.sort(key=lambda r: r["_panno"]["cons"] if np.isfinite(r["_panno"]["cons"]) else 99)

    cards = []
    for r in rows:
        p = out / f"{r['name']}.jpg"
        if not p.exists():
            continue
        uri = "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
        pan = r["_panno"]
        cons = pan["cons"]
        cls = "good" if cons <= 12 else ("warn" if cons <= 25 else "bad")
        label_sus = abs(pan.get("dyaw", 0)) > 3.0
        o = r.get("oracle", {})
        shift = f"Δyaw {pan['dyaw']:+.1f}° · pos {pan['dE']:+.0f}E/{pan['dN']:+.0f}N m"
        if pan.get("tilt"):
            shift += f" · tilt {pan['tilt']:+.1f}°"
        if pan.get("ccons") is not None:
            shift += f" · contours {pan['ccons']:.0f}px"
        cards.append(
            f'<figure class="{cls}"><img src="{uri}" alt="{r["name"]}" loading="lazy">'
            f'<figcaption><span class="mono">{r["name"].replace("_01024", "")}</span>'
            f'<span class="chip {"m" if r["manual"] else "a"}">{"MANUAL" if r["manual"] else "AUTO"}</span>'
            f'<b class="c-{cls}">{cons:.0f}px</b><span class="sub">{shift}</span>'
            + ('<span class="chip bad-chip">label Δyaw&gt;3°</span>' if label_sus else "")
            + f'<span class="sub">oracle err {o.get("yaw_err", float("nan")):+.1f}°</span>'
            f"</figcaption></figure>"
        )

    html = f"""<title>peakle · GT panno</title>
<style>
:root {{ --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --line:#e1e0d9;
  --accent:#2a78d6; --good:#006300; --warn:#a86a00; --bad:#d03b3b; --chipm:#e3edfa; --chipa:#e2f3ec; }}
@media (prefers-color-scheme: dark) {{ :root {{ --page:#0d0d0d; --card:#1a1a19; --ink:#fff;
  --ink2:#c3c2b7; --line:#2c2c2a; --accent:#3987e5; --good:#0ca30c; --warn:#fab219; }} }}
:root[data-theme="dark"] {{ --page:#0d0d0d; --card:#1a1a19; --ink:#fff; --ink2:#c3c2b7;
  --line:#2c2c2a; --accent:#3987e5; --good:#0ca30c; --warn:#fab219; }}
:root[data-theme="light"] {{ --page:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --line:#e1e0d9; --accent:#2a78d6; --good:#006300; --warn:#a86a00; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--page); color:var(--ink);
  font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:1420px; margin:0 auto; padding:32px 20px 80px; }}
h1 {{ font-size:22px; margin:4px 0 2px; }}
.eyebrow {{ text-transform:uppercase; letter-spacing:.09em; font-size:11px; color:var(--accent); font-weight:650; }}
p.sub {{ color:var(--ink2); max-width:96ch; }}
.legend span {{ margin-right:18px; }} .swatch {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:5px; vertical-align:-1px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; margin-top:18px; }}
figure {{ margin:0; background:var(--card); border:1px solid var(--line); border-radius:8px; padding:7px; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px; }}
figcaption {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; padding:7px 2px 2px; }}
.mono {{ font-family:ui-monospace,Menlo,monospace; font-size:11.5px; }}
.chip {{ padding:0 7px; border-radius:99px; font-size:10.5px; font-weight:650; }}
.chip.m {{ background:var(--chipm); color:var(--accent); }} .chip.a {{ background:var(--chipa); color:#1baf7a; }}
.chip.bad-chip {{ background:transparent; border:1px solid var(--bad); color:var(--bad); }}
figcaption b {{ font-variant-numeric:tabular-nums; }} figcaption .sub {{ color:var(--ink2); font-size:12px; }}
.c-good {{ color:var(--good) }} .c-warn {{ color:var(--warn) }} .c-bad {{ color:var(--bad) }}
figure.bad {{ outline:2px solid var(--bad); outline-offset:-2px; }}
</style>
<main>
<div class="eyebrow">peakle · localization validation</div>
<h1>Ground-truth panno — bench-60, sorted clean → dirty</h1>
<p class="sub">
<span class="legend"><span><span class="swatch" style="background:rgb(0,230,90)"></span>GT skyline + GT internal contours (both from the dataset's own depth render)</span>
<span><span class="swatch" style="background:rgb(0,200,255)"></span>our DEM skyline — true-cylindrical (tan) mapping, POSE-POLISHED around the GT label (yaw ±6°, position ±175&nbsp;m incl. a contour-matched refinement, tilt ±2°)</span>
<span><span class="swatch" style="background:rgb(255,150,30)"></span>our DEM internal contours (occlusion boundaries) at the same polished pose — compare orange vs the green dots</span></span><br>
The px number is the skyline agreement after the pose polish (capped at 60). The caption shows the correction the
label needed — a large Δyaw means the GT label itself is off (flagged), not hidden by the polish. Red-rimmed tiles remain suspect.
Generated {datetime.now():%Y-%m-%d %H:%M} from <span class="mono">{Path(args.results).parent.name}</span>.</p>
<div class="grid">
{"".join(cards)}
</div>
</main>"""
    (out / "panno.html").write_text(html)
    print(f"-> {out}/panno.html ({(out / 'panno.html').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
