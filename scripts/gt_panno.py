"""GT quality panorama: one tile per bench sample for visual inspection of the ground truth.

Each tile overlays the two things that must agree if the GT is good:
  green = oracle skyline (from the GT depth render)   cyan = our DEM at the GT yaw (best vertical shift)
Tiles are sorted clean -> dirty by gt_consistency_px, so scanning top to bottom shows how good the
dataset is and where it goes wrong.  Output: tiles + a self-contained panno.html (artifact-ready).

Usage: python scripts/gt_panno.py <bench_results.json> [--out DIR]
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
from peakle.localize.solve import HorizonProfile, _best_shift_chamfer

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "local/data/geopose"
TILES = BASE / "local/data/copernicus"
TILE_W = 640
GREEN, CYAN = (0, 230, 90), (0, 200, 255)


def build_tile(rec: dict, out: Path) -> None:
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
    profile = HorizonProfile(terrain, max(s.elev_m, terrain.elevation_at(0, 0) + 2.0), step=25.0)
    dem = profile.rows_cyl(w, h, s.fov_deg, s.yaw_gt_deg, 0.0)
    dv_lim = int(math.radians(50.0) * w / math.radians(s.fov_deg)) + 1
    cons, dv = _best_shift_chamfer(obs, dem, np.arange(-dv_lim, dv_lim + 1, 8), 60.0)

    im = Image.fromarray(np.clip(rgb.astype(float) * 1.15, 0, 255).astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for rows, color in [(obs, GREEN), (dem + dv, CYAN)]:
        pts = [(c, float(rows[c])) for c in range(w) if np.isfinite(rows[c]) and 0 <= rows[c] < h]
        if len(pts) > 1:
            dr.line(pts, fill=color, width=2)
    im.save(out / f"{rec['name']}.jpg", quality=80)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out) if args.out else BASE / f"local/output/{datetime.now():%Y%m%d-%H%M%S}-gt-panno"
    out.mkdir(parents=True, exist_ok=True)

    rows = [r for r in json.load(open(args.results)) if "error" not in r]
    rows.sort(key=lambda r: r.get("gt_consistency_px", 99))
    for i, r in enumerate(rows):
        try:
            build_tile(r, out)
        except Exception as e:
            print("tile failed", r["name"], e)
        print(f"[{i+1}/{len(rows)}] {r['name']} cons={r.get('gt_consistency_px')}", flush=True)

    cards = []
    for r in rows:
        p = out / f"{r['name']}.jpg"
        if not p.exists():
            continue
        uri = "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()
        cons = r.get("gt_consistency_px", float("nan"))
        cls = "good" if cons <= 15 else ("warn" if cons <= 30 else "bad")
        o = r.get("oracle", {})
        cards.append(
            f'<figure class="{cls}"><img src="{uri}" alt="{r["name"]}" loading="lazy">'
            f'<figcaption><span class="mono">{r["name"].replace("_01024","")}</span>'
            f'<span class="chip {"m" if r["manual"] else "a"}">{"MANUAL" if r["manual"] else "AUTO"}</span>'
            f'<b class="c-{cls}">{cons:.0f}px</b>'
            f'<span class="sub">oracle err {o.get("yaw_err", float("nan")):+.1f}° · {o.get("verdict","-")}</span>'
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
p.sub {{ color:var(--ink2); max-width:90ch; }}
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
<p class="sub">Each tile shows the two curves that must agree if the ground truth is good:
<span class="legend"><span><span class="swatch" style="background:rgb(0,230,90)"></span>GT-depth skyline (the dataset's own render)</span>
<span><span class="swatch" style="background:rgb(0,200,255)"></span>our DEM at the GT yaw (best vertical shift)</span></span><br>
The px number is that agreement (GT consistency, capped at 60). Red-rimmed tiles are the offenders —
where the curves diverge, the GT pose, the GT position, or the terrain around it is wrong, and the
sample should not count against the solver. Generated {datetime.now():%Y-%m-%d %H:%M} from
<span class="mono">{Path(args.results).parent.name}</span>.</p>
<div class="grid">
{"".join(cards)}
</div>
</main>"""
    (out / "panno.html").write_text(html)
    print(f"-> {out}/panno.html ({(out / 'panno.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
