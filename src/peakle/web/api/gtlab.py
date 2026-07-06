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
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from anyio import to_thread
from fastapi import APIRouter, HTTPException, Response
from PIL import Image

from peakle.localize.copdem import load_cop_around
from peakle.localize.geopose import load_sample, read_pfm
from peakle.localize.gtrefine import crop_az_deg, dem_depth_image, dem_skyline, quality_tier, shift_align
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
_BUILD_LOCK = threading.Lock()
LAYER_NAMES = ["photo", "gt_depth", "dem_depth", "edges"] + [f"{s}_{f}" for s in ("gt", "dem") for f in ("sky", "occ", "rib", "cou")]


def _index() -> dict[str, dict]:
    # read per-sample records, not index.json: a rebuild updates records one by one and only
    # rewrites the index at the very end — the lab should show fresh metrics as they land
    records = {}
    for p in GTV2.glob("*.json"):
        if p.name == "index.json":
            continue
        try:
            r = json.loads(p.read_text())
            records[r["name"]] = r
        except (json.JSONDecodeError, KeyError):
            continue  # record mid-write by the builder
    if not records:
        raise HTTPException(503, "no GT v2 records — run scripts/build_gt_v2.py first")
    return records


@router.get("/gt/samples")
async def list_samples() -> list[dict[str, Any]]:
    """All GT v2 samples with quality metrics, worst reconstruction first."""

    # order by the WORST of the two reconstruction metrics: vs the chosen obs target AND vs the
    # pfm render — a photo-rescued sample must not hide its pfm registration error (and vice versa)
    def worst(r: dict) -> float:
        return max(r.get("sky_cons_px") or 0, r.get("pfm_cons_px") or 0)

    rows = sorted(_index().values(), key=lambda r: -worst(r))
    return [
        {k: r.get(k) for k in (
            "name", "manual", "quality", "reasons", "sky_cons_px", "pfm_cons_px", "obs_source",
            "contour_cons_px",
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
    with _BUILD_LOCK:  # the viewer fires several layer requests at once — build once, serially
        # sentinel: last layer always written (support.json may be skipped on OOM); a record
        # rewritten by a rebuild after the layers were built invalidates the cache
        sentinel = out / "dem_cou.png"
        rec_file = GTV2 / f"{name}.json"
        if sentinel.exists() and (not rec_file.exists() or sentinel.stat().st_mtime >= rec_file.stat().st_mtime):
            return
        _build_layers_locked(name, rec, out)


def _build_layers_locked(name: str, rec: dict, out) -> None:
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


@lru_cache(maxsize=3)
def _terrain_for(name: str):
    s = load_sample(DATA / name)
    return load_cop_around(TILES, s.lat, s.lon, extent_m=90000.0, grid=3000)


def _adjusted_rows(rec: dict, dyaw: float, de: float, dn: float) -> np.ndarray:
    terrain = _terrain_for(rec["name"])
    az = crop_az_deg(rec["width"], rec["fov_deg"], rec["yaw_deg"] + dyaw)
    return dem_skyline(terrain, rec["cam_z_m"], az, rec["width"], rec["height"], rec["fov_deg"],
                       rec["de_m"] + de, rec["dn_m"] + dn)


@router.get("/gt/samples/{name}/skyline")
async def sample_skyline(name: str, dyaw: float = 0.0, de: float = 0.0, dn: float = 0.0,
                         dv: float = 0.0) -> dict[str, Any]:
    """DEM skyline rows at the sample's refined pose plus manual deltas (adjust preview)."""

    rec = _index().get(name)
    if rec is None:
        raise HTTPException(404, f"unknown sample {name}")
    rows = await to_thread.run_sync(_adjusted_rows, rec, dyaw, de, dn)
    shifted = rows + rec["dv_px"] + dv
    return {"rows": [round(float(r), 1) if np.isfinite(r) else None for r in shifted]}


@router.post("/gt/samples/{name}/adjust")
async def save_adjust(name: str, body: dict[str, float]) -> dict[str, Any]:
    """Persist a manual pose correction: update the record + npz, and write a sidecar
    (local/derived/gt_v2/manual/) so future rebuilds seed from the corrected yaw."""

    rec = _index().get(name)
    if rec is None:
        raise HTTPException(404, f"unknown sample {name}")
    dyaw = float(body.get("dyaw", 0.0))
    de, dn, dv = float(body.get("de", 0.0)), float(body.get("dn", 0.0)), float(body.get("dv", 0.0))

    def apply() -> dict:
        rows = _adjusted_rows(rec, dyaw, de, dn)
        dv_total = rec["dv_px"] + dv
        npz_path = GTV2 / f"{name}.npz"
        z = dict(np.load(npz_path).items())
        obs = z["gt_skyline"].astype(float)
        cons, _ = shift_align(obs, rows, np.asarray([dv_total]), step=2)  # dv is user-set, not re-fit
        rec.update(
            yaw_deg=(rec["yaw_deg"] + dyaw) % 360.0,
            dyaw_deg=rec["dyaw_deg"] + dyaw,
            de_m=rec["de_m"] + de,
            dn_m=rec["dn_m"] + dn,
            dv_px=dv_total,
            sky_cons_px=round(float(cons), 2),
            manual_adjust=True,
        )
        quality, reasons = quality_tier(rec["sky_cons_px"], rec.get("contour_cons_px"), rec["dyaw_deg"], [])
        rec.update(quality=quality, reasons=reasons)
        z["dem_skyline"] = (rows + dv_total).astype(np.float32)
        np.savez_compressed(npz_path, **z)
        (GTV2 / f"{name}.json").write_text(json.dumps(rec, indent=1))
        manual_dir = GTV2 / "manual"
        manual_dir.mkdir(exist_ok=True)
        (manual_dir / f"{name}.json").write_text(json.dumps(
            {"yaw_deg": rec["yaw_deg"], "de_m": rec["de_m"], "dn_m": rec["dn_m"], "dv_px": rec["dv_px"]}
        ))
        return rec

    return await to_thread.run_sync(apply)


@router.get("/gt/samples/{name}/layers/{layer}.png")
async def sample_layer(name: str, layer: str) -> Response:
    if layer not in LAYER_NAMES:
        raise HTTPException(404, f"unknown layer {layer}")
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad sample name")
    path = LAYERS / name / f"{layer}.png"
    # always run the build check: it no-ops when the cache is fresh and rebuilds
    # when the sample's record is newer than the rendered layers
    await to_thread.run_sync(_build_layers, name)
    if not path.exists():
        raise HTTPException(404, f"layer {layer} unavailable for {name}")
    return Response(path.read_bytes(), media_type="image/png",
                    headers={"Cache-Control": "max-age=60"})


@router.get("/gt/samples/{name}/thumb.jpg")
async def sample_thumb(name: str) -> Response:
    """Small photo thumbnail for map spots / lists — no layer build, just a resize."""

    if "/" in name or ".." in name:
        raise HTTPException(400, "bad sample name")
    path = LAYERS / name / "thumb.jpg"
    if not path.exists():
        def build() -> None:
            s = load_sample(DATA / name)
            im = Image.open(s.photo_path).convert("RGB")
            im.thumbnail((128, 128), Image.BILINEAR)
            path.parent.mkdir(parents=True, exist_ok=True)
            im.save(path, "JPEG", quality=82)
        try:
            await to_thread.run_sync(build)
        except FileNotFoundError:
            raise HTTPException(404, f"unknown sample {name}") from None
    return Response(path.read_bytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "max-age=86400"})
