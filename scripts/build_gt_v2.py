"""Build GT v2: refined, quality-tiered ground truth for GeoPose3K samples.

For each sample: joint pose polish (yaw/position/shift/tilt, contour-arbitrated — see
peakle.localize.gtrefine), both outline families (GT-depth skyline + internal contours; DEM
reconstruction of the same at the refined pose), agreement metrics, and a CLEAN/SUSPECT tier.

Output (resumable — existing samples are skipped):
  local/derived/gt_v2/<name>.npz   arrays: gt_skyline, dem_skyline, gt_contours(packed bool),
                                   dem_contours(packed bool), shapes
  local/derived/gt_v2/<name>.json  the RefinedGT record
  local/derived/gt_v2/index.json   rebuilt at the end from all .json records

Usage:
  python scripts/build_gt_v2.py --manifest scripts/geopose_manifest_60.txt
  python scripts/build_gt_v2.py --manual            # all MANUAL samples in the corpus
  python scripts/build_gt_v2.py --samples a,b,c
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample, oracle_skyline, read_pfm
from peakle.localize.gtrefine import (
    RefinedGT,
    crop_az_deg,
    dem_contour_mask,
    gt_contour_mask,
    quality_tier,
    refine_pose,
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "local/data/geopose"
TILES = BASE / "local/data/copernicus"
OUT = BASE / "local/derived/gt_v2"
MAX_W = 1152


def build_one(name: str) -> RefinedGT:
    s = load_sample(DATA / name)
    with Image.open(s.photo_path) as im:
        w0, h0 = im.size
    w = min(w0, MAX_W)
    h = round(h0 * w / w0)

    depth = read_pfm(s.depth_path)
    o = oracle_skyline(s.depth_path)
    x = np.linspace(0, 1, len(o))
    fin = np.isfinite(o)
    obs = np.where(
        np.interp(np.linspace(0, 1, w), x, fin.astype(float)) > 0.5,
        np.interp(np.linspace(0, 1, w), x[fin], o[fin]) * (h / depth.shape[0]),
        np.nan,
    )

    gt_mask = gt_contour_mask(depth, w, h)
    gt_dt = distance_transform_edt(~gt_mask) if gt_mask.sum() >= 200 else None

    terrain = load_cop_around(TILES, s.lat, s.lon, extent_m=90000.0, grid=3000)
    cam_z0 = max(s.elev_m, terrain.elevation_at(0.0, 0.0) + 2.0)
    fit = refine_pose(terrain, cam_z0, obs, w, h, s.fov_deg, s.yaw_gt_deg, gt_dt)

    az = crop_az_deg(w, s.fov_deg, s.yaw_gt_deg + fit["dyaw"])
    dem_mask = dem_contour_mask(
        terrain, fit["cam_z"], az, w, h, s.fov_deg, fit["dv"], fit["de"], fit["dn"], fit["tilt"]
    )

    terrain_cols = np.isfinite(obs).sum()
    density = float(gt_mask.any(axis=0).sum() / max(terrain_cols, 1))
    quality, reasons = quality_tier(fit["cons"], fit["ccons"], fit["dyaw"])

    rec = RefinedGT(
        name=name,
        manual=s.manual,
        yaw_deg=(s.yaw_gt_deg + fit["dyaw"]) % 360.0,
        dyaw_deg=fit["dyaw"],
        de_m=fit["de"],
        dn_m=fit["dn"],
        cam_z_m=fit["cam_z"],
        dv_px=fit["dv"],
        tilt_deg=fit["tilt"],
        sky_cons_px=round(fit["cons"], 2),
        contour_cons_px=(round(fit["ccons"], 2) if fit["ccons"] is not None else None),
        gt_contour_density=round(density, 3),
        width=w,
        height=h,
        fov_deg=s.fov_deg,
        quality=quality,
        reasons=reasons,
    )
    np.savez_compressed(
        OUT / f"{name}.npz",
        gt_skyline=obs.astype(np.float32),
        dem_skyline=(fit["rows"] + fit["dv"]).astype(np.float32),
        gt_contours=np.packbits(gt_mask),
        dem_contours=np.packbits(dem_mask),
        shape=np.array([h, w]),
    )
    (OUT / f"{name}.json").write_text(json.dumps(rec.to_dict(), indent=1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--samples", default=None)
    ap.add_argument("--manual", action="store_true", help="all MANUAL samples in the corpus")
    ap.add_argument("--max-n", type=int, default=10**9)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.samples:
        names = args.samples.split(",")
    elif args.manifest:
        names = [ln.strip() for ln in Path(args.manifest).read_text().splitlines() if ln.strip()]
    elif args.manual:
        names = []
        for d in sorted(DATA.iterdir()):
            info = d / "info.txt"
            if info.exists() and info.read_text().splitlines()[0].strip().upper().startswith("MANUAL"):
                if (d / "cyl/photo_crop.jpg").exists() and (d / "cyl/distance_crop.pfm").exists():
                    names.append(d.name)
    else:
        raise SystemExit("pass --manifest, --samples or --manual")
    names = names[: args.max_n]

    done = skipped = failed = 0
    for i, name in enumerate(names):
        if (OUT / f"{name}.json").exists():
            skipped += 1
            continue
        t0 = time.time()
        try:
            rec = build_one(name)
            done += 1
            print(
                f"[{i+1}/{len(names)}] {name}: {rec.quality} sky={rec.sky_cons_px} "
                f"ct={rec.contour_cons_px} dyaw={rec.dyaw_deg:+.1f} ({time.time()-t0:.0f}s)",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            traceback.print_exc()
            print(f"[{i+1}/{len(names)}] {name}: FAILED {exc}", flush=True)

    index = []
    for j in sorted(OUT.glob("*.json")):
        if j.name != "index.json":
            index.append(json.loads(j.read_text()))
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    clean = [r for r in index if r["quality"] == "CLEAN"]
    print(f"\nbuilt {done}, skipped {skipped}, failed {failed}; index: {len(index)} total, {len(clean)} CLEAN")


if __name__ == "__main__":
    main()
