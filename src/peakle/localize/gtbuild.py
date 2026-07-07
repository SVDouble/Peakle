"""Build GT v2: refined, quality-tiered ground truth for one GeoPose3K sample.

This is the ground-truth construction pipeline itself (moved out of scripts/ so the web app's
rebuild endpoint and the CLI share one implementation). For each sample it does: hybrid
observation targeting (pfm render by default; the detected photo skyline only when trusted and
disagreeing with the pfm), joint pose polish (yaw/position/shift/tilt, contour-arbitrated — see
gtrefine.refine_pose), both outline families, agreement metrics, and a CLEAN/SUSPECT tier with
the photo-targeting confirmation gate.

Per sample it writes, under ``out_dir`` (default ``local/derived/gt_v2``):
  <name>.npz   arrays: gt_skyline, dem_skyline, gt_contours(packed), dem_contours(packed), shape
  <name>.json  the RefinedGT record

``build_index`` rebuilds ``index.json`` from all records. A manual-adjust sidecar under
``<out_dir>/manual/<name>.json`` seeds the polish from a human-verified yaw.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

from peakle.localize.copdem import load_cop_around
from peakle.localize.extract import extract_candidates
from peakle.localize.geopose import load_sample, oracle_skyline, read_pfm
from peakle.localize.gtrefine import (
    RefinedGT,
    crop_az_deg,
    dem_contour_mask,
    gt_contour_mask,
    quality_tier,
    refine_pose,
    shift_align,
)
from peakle.localize.outline_score import rows_to_mask
from peakle.localize.photo_support import edge_mask, family_support

BASE = Path(__file__).resolve().parents[3]
GEOPOSE_DIR = BASE / "local/data/geopose"
COP_TILES_DIR = BASE / "local/data/copernicus"
GTV2_DIR = BASE / "local/derived/gt_v2"
MAX_W = 1152


def build_one(
    name: str,
    data_dir: Path = GEOPOSE_DIR,
    tiles_dir: Path = COP_TILES_DIR,
    out_dir: Path = GTV2_DIR,
) -> RefinedGT:
    """Refine + tier one sample; writes its .npz + .json and returns the record."""

    out_dir.mkdir(parents=True, exist_ok=True)
    s = load_sample(data_dir / name)
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

    # HYBRID observation targeting.  The pfm skyline is the CLEANEST target (a render: complete,
    # noise-free) and our DEM can match it to ~1px — so it stays the DEFAULT.  But it carries
    # registration outliers ("large outliers where the gt skyline is way off"); ONLY when the
    # trusted detected photo skyline DISAGREES with the pfm does the photo become the target.
    # (Re-anchoring every sample to the photo raised the whole cons distribution by the photo's
    # trees/extraction noise floor — measured, reverted.)
    rgb = np.asarray(Image.open(s.photo_path).convert("RGB").resize((w, h), Image.BILINEAR), np.uint8)
    edges = edge_mask(rgb)
    obs_source, obs_support, pfm_offset = "pfm", None, None
    pfm_obs = obs.copy()
    if edges is not None:
        best = None
        for cand in extract_candidates(rgb).values():
            if cand.coverage < 0.5:
                continue
            sup = family_support(rows_to_mask(cand.rows, h), edges)
            if sup is not None and (best is None or sup * cand.coverage > best[0]):
                best = (sup * cand.coverage, sup, cand)
        if best is not None:
            _, sup, cand = best
            obs_support = round(sup, 3)
            both = np.isfinite(cand.rows) & np.isfinite(obs)
            if both.sum() > 0.3 * w:
                pfm_offset = float(np.median(np.abs((cand.rows - obs)[both])))
            if pfm_offset is not None and pfm_offset > 10.0 and cand.coverage >= 0.8 and sup >= 0.6:
                obs = np.where(np.isfinite(cand.rows), cand.rows, np.nan)
                obs_source = "photo"

    gt_mask = gt_contour_mask(depth, w, h)
    gt_dt = distance_transform_edt(~gt_mask) if gt_mask.sum() >= 200 else None

    terrain = load_cop_around(tiles_dir, s.lat, s.lon, extent_m=90000.0, grid=3000)
    cam_z0 = max(s.elev_m, terrain.elevation_at(0.0, 0.0) + 2.0)
    # a manual adjustment saved from the app is a human-verified label: seed the polish
    # from the corrected yaw instead of the original (possibly bad) dataset label
    yaw0 = s.yaw_gt_deg
    manual = out_dir / "manual" / f"{name}.json"
    if manual.exists():
        yaw0 = float(json.loads(manual.read_text())["yaw_deg"])
    fit = refine_pose(terrain, cam_z0, obs, w, h, s.fov_deg, yaw0, gt_dt)

    az = crop_az_deg(w, s.fov_deg, yaw0 + fit["dyaw"])
    dem_mask = dem_contour_mask(terrain, fit["cam_z"], az, w, h, s.fov_deg, fit["dv"], fit["de"], fit["dn"], fit["tilt"])

    # secondary metric: reconstruction vs the pfm render, ALWAYS — keeps distributions comparable
    # across targeting modes (vs-photo cons carries a trees/extraction noise floor the DEM can't
    # reproduce; vs-pfm cons is the two-terrain-model agreement)
    pfm_cons, _ = shift_align(pfm_obs, fit["rows"], fit["dv"] + np.arange(-40.0, 41.0, 2.0), step=2)

    terrain_cols = np.isfinite(obs).sum()
    density = float(gt_mask.any(axis=0).sum() / max(terrain_cols, 1))
    extra = []
    if obs_source == "pfm" and pfm_offset is not None and pfm_offset > 15.0:
        extra.append(f"pfm registration off {pfm_offset:.0f}px and photo skyline untrusted")
    # GT-vs-photo check: does the observed skyline actually lie on the photo's own edges?
    # Feeds the photo-targeting confirmation gate in quality_tier.
    sky_support = family_support(rows_to_mask(obs, h), edges) if edges is not None else None
    quality, reasons = quality_tier(
        fit["cons"], fit["ccons"], fit["dyaw"], extra, obs_source=obs_source, sky_support=sky_support
    )

    rec = RefinedGT(
        name=name,
        manual=s.manual,
        yaw_deg=(yaw0 + fit["dyaw"]) % 360.0,
        dyaw_deg=((yaw0 + fit["dyaw"] - s.yaw_gt_deg + 180.0) % 360.0) - 180.0,
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
        obs_source=obs_source,
        obs_support=obs_support,
        sky_support=(round(sky_support, 3) if sky_support is not None else None),
        pfm_offset_px=(round(pfm_offset, 1) if pfm_offset is not None else None),
        pfm_cons_px=round(float(pfm_cons), 2),
        quality=quality,
        reasons=reasons,
    )
    np.savez_compressed(
        out_dir / f"{name}.npz",
        gt_skyline=obs.astype(np.float32),  # the OBSERVATION the pose was refined against
        dem_skyline=(fit["rows"] + fit["dv"]).astype(np.float32),
        gt_contours=np.packbits(gt_mask),
        dem_contours=np.packbits(dem_mask),
        shape=np.array([h, w]),
    )
    (out_dir / f"{name}.json").write_text(json.dumps(rec.to_dict(), indent=1))
    return rec


def _has_inputs(d: Path) -> bool:
    return (d / "cyl/photo_crop.jpg").exists() and (d / "cyl/distance_crop.pfm").exists()


def discover_samples(mode: str, data_dir: Path = GEOPOSE_DIR) -> list[str]:
    """Sample names for ``mode`` in ``{"manual", "all"}`` (complete inputs only)."""

    names = []
    for d in sorted(data_dir.iterdir()):
        if not d.is_dir() or not _has_inputs(d):
            continue
        if mode == "manual":
            info = d / "info.txt"
            if not (info.exists() and info.read_text().splitlines()[0].strip().upper().startswith("MANUAL")):
                continue
        names.append(d.name)
    return names


def build_index(out_dir: Path = GTV2_DIR) -> list[dict]:
    """Rebuild ``index.json`` from every per-sample record; returns the index."""

    index = [json.loads(j.read_text()) for j in sorted(out_dir.glob("*.json")) if j.name != "index.json"]
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))
    return index
