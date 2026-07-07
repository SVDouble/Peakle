# Peakle

Peakle recovers a camera's **pose from a mountain photograph** — yaw, pitch,
position and field of view — by matching outlines extracted from the photo
against renders of a real-world elevation model (DEM). Where a distant skyline
is distinctive enough, it localizes from the silhouette alone, with **no EXIF and
no compass**, and it refuses to answer when the evidence is ambiguous rather than
guessing.

The project is two things that grew together:

- an **interactive web workbench** — a full-window 3D terrain map where you
  browse the ground-truth photo corpus, look through any camera, overlay the
  extracted outlines, hand-adjust a pose, and run the solver; and
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
below), otherwise on synthetic demo terrain. From there you can place cameras
and solve them, or open the **GT data** tab to work with the photo corpus:
click a sample to inspect it, `⌖` to recenter the 3D map on its location, and
**True POV** to look through its camera and compare the DEM against the photo.

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
  web/          FastAPI app + the browser workbench (static/js: 3D map, GT panel,
                inspector, unified camera abstraction)
  scene/        in-memory workbench scene (terrain, views, solves)
  terrain/ rendering/ optimization/ domain/    supporting layers
  scripts/      first-class CLIs over the package (python -m peakle.scripts.*)
```
