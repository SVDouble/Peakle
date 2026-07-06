"""GT Lab: dataset debugger for the GeoPose3K ground truth (photo + typed outline layers + map).

Serves the `/gt` viewer page's data: the sample list (GT v2 index joined with quality metrics),
per-sample metadata, and per-layer transparent PNGs generated lazily and cached under
``local/derived/gt_v2/layers/<name>/``.  Layer families, each toggleable in the viewer:

  photo            the cylindrical crop
  gt_depth         GT depth render, colormapped (from distance_crop.pfm)
  dem_depth        our DEM depth render at the refined pose, same colormap
  gt_sky/dem_sky   skyline curves
  gt_occ/dem_occ   type-1 occlusion (jump) contours
  gt_rib/dem_rib   type-2a convex creases (spurs / counterforts)
  gt_cou/dem_cou   type-2b concave creases (couloirs)

GT layers come from the dataset's own depth; DEM layers from the refined pose in the GT v2
record — where they disagree, the viewer shows exactly which family and where.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from anyio import to_thread
from fastapi import APIRouter, HTTPException, Response
from PIL import Image

from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample, read_pfm
from peakle.localize.gtrefine import crop_az_deg, dem_depth_image
from peakle.localize.photo_support import edge_mask, support_report
from peakle.localize.typed_outlines import extract_typed_outlines

router = APIRouter(tags=["gtlab"])

BASE = Path(__file__).resolve().parents[4]
DATA = BASE / "local/data/geopose"
GTV2 = BASE / "local/derived/gt_v2"
LAYERS = GTV2 / "layers"
TILES = BASE / "local/data/copernicus"

# GT = warm/green family; DEM = cool family — two parallel curves in sibling colors = the error
_COLORS = {
    "gt": {"sky": (0, 230, 90), "occ": (255, 150, 30), "rib": (255, 235, 59), "cou": (232, 110, 220)},
    "dem": {"sky": (0, 200, 255), "occ": (255, 70, 70), "rib": (80, 170, 255), "cou": (170, 90, 255)},
}
LAYER_NAMES = ["photo", "gt_depth", "dem_depth", "edges"] + [f"{s}_{f}" for s in ("gt", "dem") for f in ("sky", "occ", "rib", "cou")]


def _index() -> dict[str, dict]:
    p = GTV2 / "index.json"
    if not p.exists():
        raise HTTPException(503, "GT v2 index missing — run scripts/build_gt_v2.py first")
    return {r["name"]: r for r in json.loads(p.read_text())}


@router.get("/gt/samples")
async def list_samples() -> list[dict[str, Any]]:
    """All GT v2 samples with quality metrics, worst reconstruction first."""

    rows = sorted(_index().values(), key=lambda r: -(r["sky_cons_px"] or 0))
    return [
        {k: r.get(k) for k in (
            "name", "manual", "quality", "reasons", "sky_cons_px", "contour_cons_px",
            "dyaw_deg", "de_m", "dn_m", "tilt_deg", "yaw_deg", "fov_deg", "gt_contour_density",
            "width", "height",
        )} | {"lat": _latlon(r["name"])[0], "lon": _latlon(r["name"])[1]}
        for r in rows
    ]


def _latlon(name: str) -> tuple[float, float]:
    try:
        s = load_sample(DATA / name)
        return s.lat, s.lon
    except Exception:
        return float("nan"), float("nan")


def _mask_png(mask: np.ndarray, color: tuple[int, int, int], w: int, h: int) -> bytes:
    if mask.shape != (h, w):
        m = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
        mask = np.asarray(m) > 0
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[mask] = (*color, 255)
    # thicken 1px for visibility
    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:, 1:] |= mask[:, :-1]
    rgba[grown & ~mask] = (*color, 160)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _depth_png(depth: np.ndarray, w: int, h: int) -> bytes:
    d = depth.astype(float)
    finite = np.isfinite(d) & (d > 0)
    logd = np.log(np.where(finite, d, np.nan))
    lo, hi = (np.nanpercentile(logd, 2), np.nanpercentile(logd, 98)) if finite.any() else (0, 1)
    t = np.clip((logd - lo) / max(hi - lo, 1e-9), 0, 1)
    rgba = np.zeros((*d.shape, 4), np.uint8)
    rgba[..., 0] = np.nan_to_num(40 + 30 * (1 - t), nan=0)
    rgba[..., 1] = np.nan_to_num(90 + 130 * (1 - t), nan=0)
    rgba[..., 2] = np.nan_to_num(140 + 110 * (1 - t), nan=0)
    rgba[..., 3] = np.where(finite, 200, 0)
    img = Image.fromarray(rgba, "RGBA")
    if img.size != (w, h):
        img = img.resize((w, h), Image.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _rows_mask(rows: np.ndarray, w: int, h: int) -> np.ndarray:
    mask = np.zeros((h, w), bool)
    ok = np.isfinite(rows) & (rows >= 0) & (rows < h)
    mask[np.clip(rows[ok].round().astype(int), 0, h - 1), np.arange(w)[ok]] = True
    return mask


def _build_layers(name: str) -> None:
    rec = _index().get(name)
    if rec is None:
        raise HTTPException(404, f"unknown sample {name}")
    out = LAYERS / name
    out.mkdir(parents=True, exist_ok=True)
    w, h = rec["width"], rec["height"]
    s = load_sample(DATA / name)

    rgb = Image.open(s.photo_path).convert("RGB").resize((w, h), Image.BILINEAR)
    rgb.save(out / "photo.png", "PNG", optimize=True)

    z = np.load(GTV2 / f"{name}.npz")
    gt_sky = z["gt_skyline"].astype(float)
    dem_sky = z["dem_skyline"].astype(float)

    gt_depth = read_pfm(s.depth_path)
    gt_typed = extract_typed_outlines(gt_depth)
    (out / "gt_depth.png").write_bytes(_depth_png(gt_depth, w, h))
    (out / "gt_sky.png").write_bytes(_mask_png(_rows_mask(gt_sky, w, h), _COLORS["gt"]["sky"], w, h))
    for fam, mask in (("occ", gt_typed.occlusion), ("rib", gt_typed.rib), ("cou", gt_typed.couloir)):
        (out / f"gt_{fam}.png").write_bytes(_mask_png(mask, _COLORS["gt"][fam], w, h))

    terrain = load_cop_around(TILES, s.lat, s.lon, extent_m=90000.0, grid=3000)
    az = crop_az_deg(w, rec["fov_deg"], rec["yaw_deg"])
    depth, *_ = dem_depth_image(
        terrain, rec["cam_z_m"], az, w, h, rec["fov_deg"], rec["dv_px"], rec["de_m"], rec["dn_m"], rec["tilt_deg"], sub=2
    )
    dem_typed = extract_typed_outlines(depth, min_px=12)  # sub=2 grid: same physical length
    (out / "dem_depth.png").write_bytes(_depth_png(depth, w, h))
    (out / "dem_sky.png").write_bytes(_mask_png(_rows_mask(dem_sky, w, h), _COLORS["dem"]["sky"], w, h))
    for fam, mask in (("occ", dem_typed.occlusion), ("rib", dem_typed.rib), ("cou", dem_typed.couloir)):
        (out / f"dem_{fam}.png").write_bytes(_mask_png(mask, _COLORS["dem"][fam], w, h))

    # photo-edge support: which of these lines does the PHOTOGRAPH actually show?
    edges = edge_mask(np.asarray(rgb, np.uint8))
    if edges is not None:
        (out / "edges.png").write_bytes(_mask_png(edges, (245, 245, 245), w, h))

        def full(mask: np.ndarray) -> np.ndarray:
            if mask.shape == (h, w):
                return mask
            m = Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)
            return np.asarray(m) > 0

        masks = {
            "gt_sky": _rows_mask(gt_sky, w, h), "gt_occ": full(gt_typed.occlusion),
            "gt_rib": full(gt_typed.rib), "gt_cou": full(gt_typed.couloir),
            "dem_sky": _rows_mask(dem_sky, w, h), "dem_occ": full(dem_typed.occlusion),
            "dem_rib": full(dem_typed.rib), "dem_cou": full(dem_typed.couloir),
        }
        (out / "support.json").write_text(json.dumps(
            {k: (round(v, 3) if v is not None else None) for k, v in support_report(masks, edges).items()}
        ))


@router.get("/gt/samples/{name}/meta")
async def sample_meta(name: str) -> dict[str, Any]:
    rec = _index().get(name)
    if rec is None:
        raise HTTPException(404, f"unknown sample {name}")
    lat, lon = _latlon(name)
    support_path = LAYERS / name / "support.json"
    if not support_path.exists():
        await to_thread.run_sync(_build_layers, name)
    support = json.loads(support_path.read_text()) if support_path.exists() else None
    return rec | {"lat": lat, "lon": lon, "layers": LAYER_NAMES, "photo_support": support}


@router.get("/gt/samples/{name}/layers/{layer}.png")
async def sample_layer(name: str, layer: str) -> Response:
    if layer not in LAYER_NAMES:
        raise HTTPException(404, f"unknown layer {layer}")
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad sample name")
    path = LAYERS / name / f"{layer}.png"
    if not path.exists():
        await to_thread.run_sync(_build_layers, name)
    return Response(path.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "max-age=86400"})
