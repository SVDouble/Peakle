# Peakle

Peakle recovers a **pose from a mountain view** — yaw, pitch,
position and field of view — by matching outlines extracted from an image/crop/photo
against renders of a real-world elevation model (DEM). Where a distant skyline
is distinctive enough, it localizes from the silhouette alone, with **no EXIF and
no compass**, and it refuses to answer when the evidence is ambiguous rather than
guessing.

Terminology in the app:

- **View** — the image/crop/photo being localized. This is the left-list entity.
- **Pose** — one candidate set of extrinsics for a view: position, yaw, pitch and roll.
- **Camera model** — the view's intrinsics/projection: size, FOV, focal length and crop geometry.

The project is two things that grew together:

- an **interactive web workbench** — a full-window 3D terrain map where you
  browse the ground-truth view corpus, look through any pose, overlay the
  extracted outlines, hand-adjust poses, and run solvers; and
- a **validated localization library** (`peakle.localize`) with an honesty
  harness: every solve reports diagnostics (chamfer, basin width, alias ratio,
  SNR) and a calibrated CONFIRMED/AMBIGUOUS verdict, benchmarked on GeoPose3K so
  a result can never "look good but be wrong".

## Quick start

```bash
uv sync
uv run peakle web serve          # open the URL it prints
```

The app boots on real terrain when SRTM `.hgt` tiles are present (see **Data**
below), otherwise on synthetic demo terrain. From there you can work from one
**Views** list:

- place a synthetic view on the 3D map and solve it;
- open a GeoPose3K sample as an editable, solvable view;
- use **Localize photo** to upload an arbitrary mountain photo with an approximate
  location and FOV; or
- select a catalogue GT sample, inspect its layers, and open it when you want to
  move or solve it.

The map has a `Map | POV` toggle plus a pose table: pick the dataset pose, ground
truth pose, or any solver pose and look through that pose with the view's camera
model. The separate **Overview** panel is the 2D navigator for moving the terrain
window.

Other entry points:

```bash
uv run peakle segment <photo.jpg>   # compare skyline-extraction methods on one photo
uv run peakle demo run              # the original synthetic render→recover demo
```

## How it works

1. **Extraction** — candidate skylines from the photo (colour detectors + a
   DexiNed-edge DP skyline), plus typed internal outlines (occlusion lines, ribs,
   couloirs) from monocular structure.
2. **Horizon solve** — for a known position the 360° horizon `el(az)` is
   ray-cast once (`peakle.localize.solve.HorizonProfile`); every (yaw, pitch, fov)
   hypothesis is a resampling scored by a capped skyline chamfer with top-K basin
   polish.
3. **Arbitration** — among skyline hypotheses within a tight chamfer slack, the
   winner is the one whose typed DEM outlines best *explain* the photo's edges —
   this is what rejects a lake edge masquerading as the skyline.
4. **Verdict** — a gate calibrated on the benchmark (`alias ≥ 1.5 ∧ well ≤ 10° ∧
   chamfer ≤ 15px ∧ coverage ≥ 0.25`) marks a solve CONFIRMED only when it is
   trustworthy; everything else is AMBIGUOUS.

DEM sources: **Copernicus GLO-30** (default; auto-downloaded per 1° tile),
**SRTM** `.hgt`, and **swissALTI3D** 2 m patches where sub-30 m detail matters.

## Ground-truth pipeline

The benchmark and the app both run on **GT v2** — refined, quality-tiered ground
truth for the GeoPose3K corpus. Each sample gets a pose polish around its label,
both outline families, agreement metrics, and a CLEAN/SUSPECT tier so only
verified samples grade the solver. The pipeline lives in the package
(`peakle.localize.gtbuild` / `bench` / `gtquality`); the CLIs are thin wrappers:

```bash
python -m peakle.scripts.build_gt_v2 --manual      # refine the MANUAL corpus
python -m peakle.scripts.bench_geopose             # oracle + extracted tracks, honesty gate
python -m peakle.scripts.calibrate_verdict <results.json>   # recalibrate the verdict gate
python -m peakle.scripts.acceptance                # end-to-end acceptance checks
```

## Data

Corpus and DEM inputs live under `local/` and are not committed:

- `local/data/geopose/` — GeoPose3K samples (`python -m peakle.scripts.fetch_geopose`).
- `local/data/copernicus/` — Copernicus DEM tiles (auto-downloaded on demand).
- `local/derived/gt_v2/` — refined GT records, arrays and layer PNGs.

For the real-terrain workbench without the corpus, drop SRTM `.hgt` tiles in the
directory named by `PEAKLE_DEM_DIR` (default `data/dem_samples`).

## Development

```bash
uv run pytest        # unit + integration tests
uv run ruff check .  # lint (line-length 120; E,F,I,UP,B)
uv run ty check src  # type checking
```

## Layout

```
src/peakle/
  localize/     the validated solver + GT pipeline (solve, extract, gtrefine,
                gtbuild, bench, gtquality, explanation, raycast, DEM adapters)
  web/          FastAPI app + the browser workbench (static/js: 3D map,
                unified Views list, Overview panel, inspector, camera model)
  scene/        in-memory workbench scene (terrain, views, pose solves)
  terrain/ rendering/ optimization/ domain/    supporting layers
  scripts/      first-class CLIs over the package (python -m peakle.scripts.*)
```
