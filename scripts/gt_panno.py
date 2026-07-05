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

Usage: python scripts/gt_panno.py <bench_results.json> [--out DIR] [--no-polish]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample, oracle_skyline, read_pfm
from peakle.localize.raycast import _elevation_angle_grid, horizon_elevation
from peakle.localize.solve import _best_shift_chamfer

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "local/data/geopose"
TILES = BASE / "local/data/copernicus"
TILE_W = 640
GREEN, CYAN, ORANGE = (0, 230, 90), (0, 200, 255), (255, 150, 30)
PITCH_LIM_DEG = 50.0


def _crop_az_deg(w: int, fov_deg: float, yaw_deg: float) -> np.ndarray:
    return yaw_deg + np.degrees((np.arange(w) - (w - 1) / 2.0) * (math.radians(fov_deg) / w))


def _rows_tan(el: np.ndarray, w: int, h: int, fov_deg: float) -> np.ndarray:
    f = w / math.radians(fov_deg)
    return (h - 1) / 2.0 - f * np.tan(el)


def _skyline_at(terrain, cam_z, az_deg, w, h, fov_deg, de=0.0, dn=0.0) -> np.ndarray:
    el = horizon_elevation(terrain, np.radians(az_deg), cam_z, step=25.0, cam_e=de, cam_n=dn)
    return _rows_tan(el, w, h, fov_deg)


def polish_position(terrain, cam_z, obs, az_deg, w, h, fov_deg, dvs) -> tuple[float, float, float, float]:
    """Small local (E, N) search around the GT coordinates minimising the skyline chamfer."""

    best = (math.inf, 0.0, 0.0)
    for de in (-100.0, -50.0, 0.0, 50.0, 100.0):
        for dn in (-100.0, -50.0, 0.0, 50.0, 100.0):
            z = max(cam_z, terrain.elevation_at(de, dn) + 2.0)
            c, _ = _best_shift_chamfer(obs, _skyline_at(terrain, z, az_deg, w, h, fov_deg, de, dn), dvs, 60.0, shift_step=6)
            if c < best[0]:
                best = (c, de, dn)
    _, de0, dn0 = best
    for de in (de0 - 25.0, de0, de0 + 25.0):
        for dn in (dn0 - 25.0, dn0, dn0 + 25.0):
            z = max(cam_z, terrain.elevation_at(de, dn) + 2.0)
            c, _ = _best_shift_chamfer(obs, _skyline_at(terrain, z, az_deg, w, h, fov_deg, de, dn), dvs, 60.0, shift_step=3)
            if c < best[0]:
                best = (c, de, dn)
    return best[0], best[1], best[2], max(cam_z, terrain.elevation_at(best[1], best[2]) + 2.0)


def internal_contours(terrain, cam_z, az_deg, w, h, fov_deg, dv, de=0.0, dn=0.0, sub=2, jump=0.30):
    """Occlusion-boundary mask from DEM depth discontinuities in the crop, at the drawn alignment.

    Per column: visible-surface distance for each pixel row via the first crossing of the ray's
    elevation-angle envelope; contours = |Δ log distance| > ``jump`` vertically or horizontally.
    """

    az_s = az_deg[::sub]
    el, ds = _elevation_angle_grid(terrain, np.radians(az_s), cam_z, step=25.0, d_max=None, cam_e=de, cam_n=dn)
    cummax = np.maximum.accumulate(el, axis=1)
    f = w / math.radians(fov_deg)
    rows_s = np.arange(0, h, sub)
    # invert the drawn mapping: rows = (h-1)/2 - f*tan(el) + dv  =>  tan(el_pix) = ((h-1)/2 + dv - r)/f
    el_pix = np.arctan(((h - 1) / 2.0 + dv - rows_s) / f)          # descending in r
    depth = np.full((len(rows_s), len(az_s)), np.nan)
    for c in range(len(az_s)):
        asc = cummax[c]                                            # nondecreasing in d
        idx = np.searchsorted(asc, el_pix)                         # first d where envelope >= el_pix
        hit = idx < len(ds)
        depth[hit, c] = ds[np.clip(idx[hit], 0, len(ds) - 1)]
    logd = np.log(depth)
    edge = np.zeros_like(depth, bool)
    edge[1:, :] |= np.abs(np.diff(logd, axis=0)) > jump
    edge[:, 1:] |= np.abs(np.diff(logd, axis=1)) > jump
    edge &= np.isfinite(depth)
    mask = np.zeros((h, w), bool)
    rr, cc = np.nonzero(edge)
    mask[np.clip(rows_s[rr], 0, h - 1), np.clip(cc * sub, 0, w - 1)] = True
    return mask


def build_tile(rec: dict, out: Path, do_polish: bool) -> dict:
    s = load_sample(DATA / rec["name"])
    rgb = np.asarray(Image.open(s.photo_path).convert("RGB"), np.uint8)
    scale = TILE_W / rgb.shape[1]
    rgb = np.asarray(Image.fromarray(rgb).resize((TILE_W, round(rgb.shape[0] * scale)), Image.BILINEAR), np.uint8)
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
    az = _crop_az_deg(w, s.fov_deg, s.yaw_gt_deg)
    dv_lim = int((w / math.radians(s.fov_deg)) * math.tan(math.radians(PITCH_LIM_DEG))) + 1
    dvs = np.arange(-dv_lim, dv_lim + 1, 8)

    de = dn = 0.0
    if do_polish:
        cons, de, dn, cam_z = polish_position(terrain, cam_z, obs, az, w, h, s.fov_deg, dvs)
    dem = _skyline_at(terrain, cam_z, az, w, h, s.fov_deg, de, dn)
    cons, dv = _best_shift_chamfer(obs, dem, dvs, 60.0, shift_step=3)
    contours = internal_contours(terrain, cam_z, az, w, h, s.fov_deg, dv, de, dn)

    im = Image.fromarray(np.clip(rgb.astype(float) * 1.15, 0, 255).astype(np.uint8))
    px = im.load()
    rr, cc = np.nonzero(contours)
    for r, c in zip(rr.tolist(), cc.tolist()):
        px[c, r] = ORANGE
        if c + 1 < w:
            px[c + 1, r] = ORANGE
        if r + 1 < h:
            px[c, r + 1] = ORANGE
    dr = ImageDraw.Draw(im)
    for rows, color in [(dem + dv, CYAN), (obs, GREEN)]:
        pts = [(c, float(rows[c])) for c in range(w) if np.isfinite(rows[c]) and 0 <= rows[c] < h]
        if len(pts) > 1:
            dr.line(pts, fill=color, width=2)
    im.save(out / f"{rec['name']}.jpg", quality=80)
    return {"cons": round(float(cons), 1), "dE": de, "dN": dn}


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
            r["_panno"] = {"cons": float("nan"), "dE": 0, "dN": 0}
        print(f"[{i+1}/{len(rows)}] {r['name']} cons={r['_panno']['cons']} dE={r['_panno']['dE']:+.0f} dN={r['_panno']['dN']:+.0f}", flush=True)
    rows.sort(key=lambda r: (r["_panno"]["cons"] if np.isfinite(r["_panno"]["cons"]) else 99))

    cards = []
    for r in rows:
        p = out / f"{r['name']}.jpg"
        if not p.exists():
            continue
        uri = "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
        pan = r["_panno"]
        cons = pan["cons"]
        cls = "good" if cons <= 12 else ("warn" if cons <= 25 else "bad")
        o = r.get("oracle", {})
        shift = f"pos {pan['dE']:+.0f}E/{pan['dN']:+.0f}N m" if (pan["dE"] or pan["dN"]) else "pos ±0"
        cards.append(
            f'<figure class="{cls}"><img src="{uri}" alt="{r["name"]}" loading="lazy">'
            f'<figcaption><span class="mono">{r["name"].replace("_01024","")}</span>'
            f'<span class="chip {"m" if r["manual"] else "a"}">{"MANUAL" if r["manual"] else "AUTO"}</span>'
            f'<b class="c-{cls}">{cons:.0f}px</b><span class="sub">{shift}</span>'
            f'<span class="sub">oracle err {o.get("yaw_err", float("nan")):+.1f}°</span>'
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
figcaption b {{ font-variant-numeric:tabular-nums; }} figcaption .sub {{ color:var(--ink2); font-size:12px; }}
.c-good {{ color:var(--good) }} .c-warn {{ color:var(--warn) }} .c-bad {{ color:var(--bad) }}
figure.bad {{ outline:2px solid var(--bad); outline-offset:-2px; }}
</style>
<main>
<div class="eyebrow">peakle · localization validation</div>
<h1>Ground-truth panno — bench-60, sorted clean → dirty</h1>
<p class="sub">
<span class="legend"><span><span class="swatch" style="background:rgb(0,230,90)"></span>GT-depth skyline (the dataset's own render)</span>
<span><span class="swatch" style="background:rgb(0,200,255)"></span>our DEM at the GT yaw — true-cylindrical (tan) mapping, position polished ±125&nbsp;m</span>
<span><span class="swatch" style="background:rgb(255,150,30)"></span>internal contours = DEM occlusion boundaries at the same pose (no GT lines exist; reconstructed from extrinsics)</span></span><br>
The px number is the skyline agreement after the position polish (capped at 60); the pos shift shows how far
the polished camera moved from the GT coordinates. Red-rimmed tiles remain suspect.
Generated {datetime.now():%Y-%m-%d %H:%M} from <span class="mono">{Path(args.results).parent.name}</span>.</p>
<div class="grid">
{"".join(cards)}
</div>
</main>"""
    (out / "panno.html").write_text(html)
    print(f"-> {out}/panno.html ({(out / 'panno.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
