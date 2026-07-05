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

from scipy.ndimage import distance_transform_edt

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


def gt_contour_mask(depth_path, w: int, h: int, jump: float = 0.30) -> np.ndarray:
    """TRUE internal contours from the GT depth render: |Δ log distance| jumps inside the terrain.
    The sky boundary is excluded automatically (sky is NaN, NaN diffs never pass the threshold)."""

    d = read_pfm(depth_path).astype(float)
    d[d <= 0] = np.nan
    logd = np.log(d)
    edge = np.zeros_like(d, bool)
    edge[1:, :] |= np.abs(np.diff(logd, axis=0)) > jump
    edge[:, 1:] |= np.abs(np.diff(logd, axis=1)) > jump
    h0, w0 = d.shape
    mask = np.zeros((h, w), bool)
    rr, cc = np.nonzero(edge)
    mask[np.clip((rr * h / h0).astype(int), 0, h - 1), np.clip((cc * w / w0).astype(int), 0, w - 1)] = True
    return mask


def _contour_score(dem_mask: np.ndarray, gt_dt: np.ndarray, cap: float = 40.0) -> float:
    """Mean capped distance from DEM contour pixels to the nearest GT contour pixel."""

    rr, cc = np.nonzero(dem_mask)
    if len(rr) < 50:
        return cap
    return float(np.minimum(gt_dt[rr, cc], cap).mean())


def polish_pose(terrain, cam_z, obs, w, h, fov_deg, yaw_gt, dvs, gt_dt=None):
    """Local pose polish around the GT label: yaw (±6°), position (±125 m), vertical shift, and a
    residual tilt.  The GT yaw label / decode carries a few degrees of noise on some samples —
    trusting it exactly paints the DEM skyline into the sky; the polish magnitude is REPORTED per
    tile so a large correction still reads as a suspect label rather than being hidden.

    When ``gt_dt`` (distance transform of the GT-depth internal contours) is given, a final
    position refinement matches the DEM occlusion contours too: the skyline is far-field and
    nearly position-blind, so without the near-field contour term the position stays
    under-constrained by ±50 m and the drawn contours inherit that parallax error."""

    def align(dyaw, de, dn, step=6):
        z = max(cam_z, terrain.elevation_at(de, dn) + 2.0)
        az = _crop_az_deg(w, fov_deg, yaw_gt + dyaw)
        rows = _skyline_at(terrain, z, az, w, h, fov_deg, de, dn)
        c, dv = _best_shift_chamfer(obs, rows, dvs, 60.0, shift_step=step)
        return c, dv, rows

    best = (math.inf, 0.0, 0.0, 0.0, 0.0, None)  # cons, dyaw, de, dn, dv, rows
    for dyaw in np.linspace(-6.0, 6.0, 9):
        c, dv, rows = align(dyaw, 0.0, 0.0)
        if c < best[0]:
            best = (c, float(dyaw), 0.0, 0.0, dv, rows)
    for de in (-100.0, -50.0, 0.0, 50.0, 100.0):
        for dn in (-100.0, -50.0, 0.0, 50.0, 100.0):
            c, dv, rows = align(best[1], de, dn)
            if c < best[0]:
                best = (c, best[1], de, dn, dv, rows)
    for dyaw in (best[1] - 0.75, best[1] - 0.375, best[1], best[1] + 0.375, best[1] + 0.75):
        for de in (best[2] - 25.0, best[2], best[2] + 25.0):
            for dn in (best[3] - 25.0, best[3], best[3] + 25.0):
                c, dv, rows = align(dyaw, de, dn, step=3)
                if c < best[0]:
                    best = (c, float(dyaw), de, dn, dv, rows)

    cons, dyaw, de, dn, dv, rows = best
    # residual tilt (imperfect roll rectification of the crop): robust linear fit of the
    # remaining per-column offset, capped at ±2°
    tilt = 0.0
    resid = obs - (rows + dv)
    ok = np.isfinite(resid) & (np.abs(resid - np.nanmedian(resid)) < 25.0)
    if ok.sum() > 0.3 * w:
        cols = np.arange(w, dtype=float) - (w - 1) / 2.0
        slope = float(np.polyfit(cols[ok], resid[ok], 1)[0])
        tilt = float(np.clip(math.degrees(math.atan(slope)), -2.0, 2.0))
        rows_t = rows + math.tan(math.radians(tilt)) * cols
        c2, dv2 = _best_shift_chamfer(obs, rows_t, dvs, 60.0, shift_step=3)
        if c2 < cons:
            cons, dv, rows = c2, dv2, rows_t
        else:
            tilt = 0.0
    # contour-constrained position refinement (near-field parallax pins the position)
    ccons = None
    if gt_dt is not None:
        az_fit = _crop_az_deg(w, fov_deg, yaw_gt + dyaw)

        def combined(de_c, dn_c):
            z_c = max(cam_z, terrain.elevation_at(de_c, dn_c) + 2.0)
            rows_c = _skyline_at(terrain, z_c, az_fit, w, h, fov_deg, de_c, dn_c)
            if tilt:
                rows_c = rows_c + math.tan(math.radians(tilt)) * (np.arange(w, dtype=float) - (w - 1) / 2.0)
            c_sky, dv_c = _best_shift_chamfer(obs, rows_c, dvs, 60.0, shift_step=6)
            dem_m = internal_contours(terrain, z_c, az_fit, w, h, fov_deg, dv_c, de_c, dn_c, tilt, sub=4)
            c_ct = _contour_score(dem_m, gt_dt)
            return c_sky + 0.6 * c_ct, c_sky, c_ct, dv_c, rows_c

        best_e = None
        for de_c in (de - 50.0, de - 25.0, de, de + 25.0, de + 50.0):
            for dn_c in (dn - 50.0, dn - 25.0, dn, dn + 25.0, dn + 50.0):
                score, c_sky, c_ct, dv_c, rows_c = combined(de_c, dn_c)
                if best_e is None or score < best_e[0]:
                    best_e = (score, de_c, dn_c, c_sky, c_ct, dv_c, rows_c)
        _, de, dn, cons, ccons, dv, rows = best_e

    z = max(cam_z, terrain.elevation_at(de, dn) + 2.0)
    return {"cons": cons, "ccons": ccons, "dyaw": dyaw, "dE": de, "dN": dn, "dv": dv, "tilt": tilt, "cam_z": z, "rows": rows}


def internal_contours(terrain, cam_z, az_deg, w, h, fov_deg, dv, de=0.0, dn=0.0, tilt_deg=0.0, sub=2, jump=0.30):
    """Occlusion-boundary mask from DEM depth discontinuities in the crop, at the drawn alignment
    (including the tilt correction so contours stay registered with the drawn skyline).

    Per column: visible-surface distance for each pixel row via the first crossing of the ray's
    elevation-angle envelope; contours = |Δ log distance| > ``jump`` vertically or horizontally.
    """

    az_s = az_deg[::sub]
    el, ds = _elevation_angle_grid(terrain, np.radians(az_s), cam_z, step=25.0, d_max=None, cam_e=de, cam_n=dn)
    cummax = np.maximum.accumulate(el, axis=1)
    f = w / math.radians(fov_deg)
    rows_s = np.arange(0, h, sub)
    cols_c = (np.arange(0, w, sub, dtype=float) - (w - 1) / 2.0)
    tilt_dv = math.tan(math.radians(tilt_deg)) * cols_c            # per-column extra shift
    depth = np.full((len(rows_s), len(az_s)), np.nan)
    for c in range(len(az_s)):
        # invert the drawn mapping: rows = (h-1)/2 - f*tan(el) + dv + tilt  =>  tan(el_pix) = ...
        el_pix = np.arctan(((h - 1) / 2.0 + dv + tilt_dv[c] - rows_s) / f)   # descending in r
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
    dv_lim = int((w / math.radians(s.fov_deg)) * math.tan(math.radians(PITCH_LIM_DEG))) + 1
    dvs = np.arange(-dv_lim, dv_lim + 1, 8)

    gt_mask = gt_contour_mask(s.depth_path, w, h)
    gt_dt = distance_transform_edt(~gt_mask) if gt_mask.sum() >= 200 else None
    if do_polish:
        fit = polish_pose(terrain, cam_z, obs, w, h, s.fov_deg, s.yaw_gt_deg, dvs, gt_dt)
    else:
        az0 = _crop_az_deg(w, s.fov_deg, s.yaw_gt_deg)
        rows0 = _skyline_at(terrain, cam_z, az0, w, h, s.fov_deg)
        c0, dv0 = _best_shift_chamfer(obs, rows0, dvs, 60.0, shift_step=3)
        fit = {"cons": c0, "dyaw": 0.0, "dE": 0.0, "dN": 0.0, "dv": dv0, "tilt": 0.0, "cam_z": cam_z, "rows": rows0}

    az = _crop_az_deg(w, s.fov_deg, s.yaw_gt_deg + fit["dyaw"])
    contours = internal_contours(
        terrain, fit["cam_z"], az, w, h, s.fov_deg, fit["dv"], fit["dE"], fit["dN"], fit["tilt"]
    )

    im = Image.fromarray(np.clip(rgb.astype(float) * 1.15, 0, 255).astype(np.uint8))
    px = im.load()
    dem_rows = fit["rows"] + fit["dv"]
    obs_arr = obs
    # GT internal contours (from the GT depth itself) — the reference the DEM contours must match
    rr, cc = np.nonzero(gt_mask)
    for r, c in zip(rr.tolist(), cc.tolist()):
        if np.isfinite(obs_arr[c]) and abs(r - obs_arr[c]) < 5:
            continue  # skyline drawn separately
        px[c, r] = GREEN
    rr, cc = np.nonzero(contours)
    for r, c in zip(rr.tolist(), cc.tolist()):
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
        print(f"[{i+1}/{len(rows)}] {r['name']} cons={p['cons']} contours={p.get('ccons')} dyaw={p['dyaw']:+.1f} dE={p['dE']:+.0f} dN={p['dN']:+.0f} tilt={p['tilt']:+.1f}", flush=True)
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
        label_sus = abs(pan.get("dyaw", 0)) > 3.0
        o = r.get("oracle", {})
        shift = f"Δyaw {pan['dyaw']:+.1f}° · pos {pan['dE']:+.0f}E/{pan['dN']:+.0f}N m"
        if pan.get("tilt"):
            shift += f" · tilt {pan['tilt']:+.1f}°"
        if pan.get("ccons") is not None:
            shift += f" · contours {pan['ccons']:.0f}px"
        cards.append(
            f'<figure class="{cls}"><img src="{uri}" alt="{r["name"]}" loading="lazy">'
            f'<figcaption><span class="mono">{r["name"].replace("_01024","")}</span>'
            f'<span class="chip {"m" if r["manual"] else "a"}">{"MANUAL" if r["manual"] else "AUTO"}</span>'
            f'<b class="c-{cls}">{cons:.0f}px</b><span class="sub">{shift}</span>'
            + (f'<span class="chip bad-chip">label Δyaw&gt;3°</span>' if label_sus else "")
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
    print(f"-> {out}/panno.html ({(out / 'panno.html').stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
