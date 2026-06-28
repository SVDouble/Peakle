"""DEM pose optimizer that matches the FULL extracted contour set (skyline + internal ridge/
fold lines) to the DEM-projected ridges — not the skyline alone.

The skyline alone underdetermines the pose (ambiguous in FOV/yaw, aliases to the opposite side,
and fails when the photographed subject isn't the true skyline, e.g. a peak seen from above).
The internal contours are scene-specific, so matching them disambiguates. Objective:

    coverage(pose) = Σ_points  w · exp(-dist(point → nearest DEM ridge/skyline)² / σ²)

over every extracted-contour point (w = √polyline-length, so long matching lines count more;
unmatched points contribute ~0 → recall-driven, no absence penalty).

Search (validated on synthetic known-prior data — see local/synth_recover.py):
  Stage 1: a translation-tolerant 2D-chamfer skyline residual over the FULL 360° yaw × FOV. Pitch
    is NOT gridded — it is the most sensitive axis (~15px/° of skyline shift), so a coarse grid
    misses the true pose and lets wrong-yaw aliases win (this was the cause of total failure).
    Instead pitch ≈ a vertical image shift and is solved per candidate (shape_resid).
  Stage 2: re-rank the top candidates by the occlusion-aware contour coverage above.
  Refine: fine local search over yaw / pitch / altitude.

EXIF priors used: GPS position/altitude only. FOV is searched over a per-lens range (Pixel-7
FocalLengthIn35mmFilm is unreliable), and the compass heading is IGNORED (uncalibrated phone).

Reads local/output/sam3-pipeline/<name>/_trace.npz; writes local/output/<dt>-optim-<name>/.
"""
import sys, os, time, math, datetime
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from PIL.ExifTags import TAGS
from scipy.ndimage import distance_transform_edt, median_filter

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.pipeline.exif import read_exif
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.rendering.pinhole import project_points
from peakle.dem_ridges import ridge_valley_masks, project_terrain_ridges
from peakle.terrain.dem import load_dem_around

NAME = sys.argv[1] if len(sys.argv) > 1 else "jubigrat"
BASE = str(Path(__file__).resolve().parents[1])
STAMP = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
OUT = f"{BASE}/local/output/{STAMP}-optim-{NAME}"; os.makedirs(OUT, exist_ok=True)
EXT, GRID, SIGMA = 48000.0, 1200, 7.0   # wide extent: distant horizon ranges can be >32km away
P_REF = -8.0   # reference pitch for the decoupled vertical-shift skyline alignment


def hfov_from_fl35(fl35, W, H):
    return math.degrees(2 * math.atan((W / math.hypot(W, H)) * (21.63 / float(fl35))))


def vfov_deg(fov):
    return math.degrees(2 * math.atan(math.tan(math.radians(fov / 2)) * H / W))


# ---- extracted contours (skyline + kept internal); per-point weight = sqrt(polyline length) ----
z = np.load(f"{BASE}/local/output/sam3-pipeline/{NAME}/_trace.npz", allow_pickle=True)
H, W = int(z["shape"][0]), int(z["shape"][1])
observed = z["skyline"].astype(float)
internal = [np.asarray(p, float) for p in z["internal"]]
def _plen(a): return float(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1])).sum()) if len(a) > 1 else 1.0
contours = [np.column_stack([observed, np.arange(W)]).astype(float)] + [c for c in internal if len(c) > 1]
pts = np.vstack(contours); wts = np.concatenate([np.full(len(c), np.sqrt(_plen(c))) for c in contours])
pr = np.clip(pts[:, 0].round().astype(int), 0, H - 1); pc = np.clip(pts[:, 1].round().astype(int), 0, W - 1)
# DEPTH-WEIGHT toward DISTANT terrain: a 30 m DEM can't render the near foreground (knife-edge
# ridge, close crags) but that is only a SMALL part of the scene; the distant ranges ARE
# renderable and should drive the pose. depthn (Depth-Anything) is FAR=1, NEAR~0, so weight each
# point by it -> near foreground is downweighted, can't pull the pose to a spurious near match.
depthn = z["depthn"].astype(float)
wts = wts * np.clip(depthn[pr, pc], 0.15, 1.0)

# observed-skyline distance transform (precomputed once) for the translation-tolerant 2D chamfer
_obs_ok = np.isfinite(observed)
_obs_mask = np.zeros((H, W), bool)
_obs_mask[np.clip(observed[_obs_ok].round().astype(int), 0, H - 1), np.arange(W)[_obs_ok]] = True
_DT_obs = distance_transform_edt(~_obs_mask)
_obs_rr = np.clip(observed[_obs_ok].round().astype(int), 0, H - 1); _obs_cc = np.arange(W)[_obs_ok]

# ---- EXIF ----
pil = Image.open(f"{BASE}/data/private/{NAME}.jpg")
exif = read_exif(pil); heading = float(exif.heading_deg)
_e = pil.getexif(); _d = dict(_e)
try: _d.update(dict(_e.get_ifd(0x8769)))
except Exception: pass
_t = {TAGS.get(k, k): v for k, v in _d.items()}
fl_actual = float(_t.get("FocalLength") or 0)   # the LENS (Pixel 7: 6.81=main, ~2.3=ultrawide)
# Pixel-7 FocalLengthIn35mmFilm is unreliable (writes 48 for the 1x main → 40°, but geometry is
# ~68°). So we DON'T pin FOV; the actual FocalLength only picks a sensible search RANGE per lens,
# and the contour-coverage objective selects the FOV that aligns the ridges (trust geometry).
FOV_RANGE = range(86, 103, 4) if (0 < fl_actual < 4.0) else range(56, 79, 4)  # ultrawide vs main
FOVS = tuple(float(f) for f in FOV_RANGE)

t0 = time.perf_counter()
terrain = load_dem_around(exif.gps_lat_deg, exif.gps_lon_deg, extent_m=EXT, grid=GRID)
ground = terrain.elevation_at(0.0, 0.0)
pos = LocalPoint(east_m=0.0, north_m=0.0, up_m=float(exif.gps_alt_m or ground + 2.0))
renderer = SyntheticRenderer()
points_s = terrain.flattened_points(stride=3)        # for the fast skyline term
# DEM ridge+fold world points (precompute once; subsample for the search projection)
rmask, vmask = ridge_valley_masks(terrain, prominence_m=3.0)
ii, jj = np.where(rmask | vmask)
ridge_world = np.column_stack((terrain.x_m[jj], terrain.y_m[ii], terrain.elevation_m[ii, jj])).astype(float)
if len(ridge_world) > 40000:
    ridge_world = ridge_world[:: len(ridge_world) // 40000]
print(f"{NAME}: DEM {EXT/1000:.0f}km/{GRID}, heading={heading:.0f}, alt={pos.up_m:.0f}, lens-fl={fl_actual}, fov-range={FOVS[0]:.0f}-{FOVS[-1]:.0f}, "
      f"ridge-pts={len(ridge_world)}  ({time.perf_counter()-t0:.0f}s)", flush=True)


def make_pose(yaw, fov, pitch, up=None):
    intr = CameraIntrinsics.from_horizontal_fov(W, H, fov)
    p = pos if up is None else LocalPoint(east_m=0.0, north_m=0.0, up_m=float(up))
    ext = CameraExtrinsics(position=p, yaw_deg=float(((yaw + 180) % 360) - 180), pitch_deg=float(pitch), roll_deg=0.0)
    return intr, ext


def dem_contour_mask(intr, ext):
    """Mask of VISIBLE DEM ridges/folds + skyline — what the extracted contours match to.
    Occlusion matters: counting ridges hidden behind mountains rewards DEM ridge *density* (a
    wrong, dense-ridge yaw wins) instead of true alignment, so we visibility-test against a
    depth buffer (downsampled via stride for speed)."""
    m = np.zeros((H, W), bool)
    buf = renderer.depth_image(terrain, intr, ext, stride=3)   # coarse buffer is enough for the 3% visibility test; halves search cost
    u, v, depth, valid = project_points(ridge_world, intr, ext)
    inf = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (depth > 0)
    uu, vv, dd = u[inf].astype(int), v[inf].astype(int), depth[inf]
    surf = buf[vv, uu]
    vis = np.isfinite(surf) & (dd <= surf * 1.03)   # not hidden by nearer terrain
    m[vv[vis], uu[vis]] = True
    sky = renderer.fast_skyline(points_s, intr, ext)
    ok = np.isfinite(sky); m[np.clip(sky[ok].round().astype(int), 0, H - 1), np.arange(W)[ok]] = True
    return m


def coverage(yaw, fov, pitch, up=None):
    dist = distance_transform_edt(~dem_contour_mask(*make_pose(yaw, fov, pitch, up)))
    return float(np.sum(wts * np.exp(-(dist[pr, pc] ** 2) / (SIGMA ** 2))))


def shape_resid(yaw, fov):
    """Pitch-DECOUPLED skyline residual used to narrow the search before the slow coverage step.

    A per-column vertical skyline difference is the WRONG metric: on a rugged skyline a tiny yaw
    shift causes huge per-column errors, and a wrong FOV can stretch the skyline to fake a match —
    so it selects wrong poses (verified on synthetic data). Instead we use a translation-tolerant
    2D chamfer, AND we solve pitch rather than grid it: pitch ~= a vertical image shift
    (~H/vfov px per degree), so we render once at P_REF, slide the rendered skyline vertically to
    its best chamfer against the observed, and read off the implied pitch. This removes the
    sensitive pitch axis from the grid (which, at 3deg steps, was the root cause of total failure)."""
    sky = renderer.fast_skyline(points_s, *make_pose(yaw, fov, P_REF)); ok = np.isfinite(sky)
    both = ok & _obs_ok
    if both.sum() < W * 0.3: return (1e9, P_REF)
    cc = np.arange(W)[ok]
    dv0 = float(np.median(observed[both] - sky[both]))
    best = (1e9, 0.0)
    for d in np.arange(dv0 - 8, dv0 + 8.01, 2.0):
        rr = np.clip((sky[ok] + d).round().astype(int), 0, H - 1)
        fwd = float(_DT_obs[rr, cc].mean())                              # rendered -> observed
        ren = np.zeros((H, W), bool); ren[rr, cc] = True
        rev = float(distance_transform_edt(~ren)[_obs_rr, _obs_cc].mean())  # observed -> rendered
        if fwd + rev < best[0]: best = (fwd + rev, d)
    implied_pitch = P_REF + best[1] / (H / vfov_deg(fov))   # vertical shift -> pitch (sign-checked)
    return (best[0], implied_pitch)


# Stage 1 — pitch-decoupled skyline chamfer over the FULL 360° yaw × FOV → top candidates (pitch
# is SOLVED per candidate, not gridded). The phone compass was uncalibrated, so the EXIF heading
# is NOT a usable prior; only GPS position/altitude are trusted. Yaw is found from the contours.
_raw = [(float(y), f) + shape_resid(y, f) for y in range(0, 360, 3) for f in FOVS]
cands = sorted(((sr, y, f, ip) for (y, f, sr, ip) in _raw), key=lambda c: c[0])[:24]
# Stage 2 — re-rank those candidates by the occlusion-aware CONTOUR coverage (skyline + internal
# ridges), which disambiguates the skyline's multiple (FOV, yaw) minima.
base_up = pos.up_m
best = max(((coverage(y, f, p), (y, f, p, base_up)) for _, y, f, p in cands), key=lambda c: c[0])
c0, (yaw, fov, pitch, up) = best
print(f"{NAME}: coverage-rerank={c0:.0f} yaw={yaw:.0f} (off-heading {((yaw-heading+180)%360)-180:+.0f}) fov={fov:.0f} pitch={pitch:.1f} ({time.perf_counter()-t0:.0f}s)", flush=True)
# refine yaw/pitch (FINE — pitch is the sensitive axis) + ALTITUDE (a constant skyline float is the
# altitude/pitch signature) at the won FOV
for yy in (yaw - 2, yaw, yaw + 2):
    for pp in (pitch - 2, pitch - 1, pitch, pitch + 1, pitch + 2):
        for uu in (base_up - 60, base_up, base_up + 60):
            c = coverage(yy, fov, pp, uu)
            if c > best[0]: best = (c, (float(yy), fov, float(pp), float(uu)))
c1, (yaw, fov, pitch, up) = best
print(f"{NAME}: REFINED coverage={c1:.0f} yaw={yaw:.1f} fov={fov:.1f} pitch={pitch:.1f} up={up:.0f} (base {base_up:.0f}, heading {heading:.0f}) ({time.perf_counter()-t0:.0f}s)", flush=True)

# ---- render (proper occlusion via project_terrain_ridges) ----
intr, ext = make_pose(yaw, fov, pitch, up)
sky = median_filter(renderer.fast_skyline(points_s, intr, ext), 5)   # same skyline the objective uses
ridges = project_terrain_ridges(terrain, intr, ext, renderer=renderer, prominence_m=3.0, min_len=6)
rgb = np.asarray(Image.open(f"{BASE}/local/output/sam3-pipeline/{NAME}/01_input.png").convert("RGB").resize((W, H)), np.uint8)
im = Image.fromarray((rgb * 0.5).astype(np.uint8), "RGB"); dr = ImageDraw.Draw(im)
for prj in ridges:
    dr.line([(u, v) for u, v in prj.polyline], fill=(255, 60, 60) if prj.kind == "ridge" else (60, 160, 255), width=3)
dr.line([(int(c), int(sky[c])) for c in range(W) if np.isfinite(sky[c])], fill=(255, 220, 60), width=3)
for c in contours: dr.line([(int(x), int(y)) for y, x in c], fill=(255, 255, 255), width=1)
dr.text((6, 6), f"yaw={yaw:.0f} fov={fov:.0f} pitch={pitch:.0f} (heading {heading:.0f}, fov {fov:.0f}) cov={c1:.0f}", fill=(255, 255, 255))
im.save(f"{OUT}/{NAME}_pose_overlay.png")
print(f"{NAME}: saved -> {OUT}", flush=True)
