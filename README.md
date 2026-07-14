# Peakle

Peakle is a workbench for recovering a camera pose from a mountain view by matching image
evidence against a real-world elevation model (DEM). The target is position, orientation, and
camera calibration with or without a prior. The honest current baseline is narrower: the ray-cast
horizon solver can recover effective crop yaw at a supplied position, while position-recovery and
regional/no-prior methods remain experimental and currently fail the frozen real-data matrix.
Physical pitch/roll are not yet scoreable on the current cylindrical-crop benchmark.

Terminology in the app:

- **View** — the image/crop/photo being localized. This is the left-list entity.
- **Pose** — one candidate set of extrinsics for a view: position, yaw, pitch and roll.
- **Camera model** — the view's intrinsics/projection: size, FOV, focal length and crop geometry.

The project is two things that grew together:

- an **interactive web workbench** — a full-window 3D terrain map where you
  browse the ground-truth view corpus, look through any pose, overlay the
  extracted outlines, hand-adjust poses, and run solvers; and
- a **localization research library** (`peakle.localize`) with an honesty harness:
  controlled priors, unchanged-prior baselines, PFM/photo evidence tracks, independent
  GT↔DEM gates, immutable artifacts, and explicit ranking exclusions.

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

The map has a `Map | POV` toggle plus a pose table: pick the source-depth ground
truth or any actual solver pose and look through it with the view's camera model.
The rejected GT-v2 refined reconstruction remains available only in GT Lab as a
diagnostic. The separate **Overview** panel is the 2D navigator for moving the
terrain window.

Other entry points:

```bash
uv run peakle segment <photo.jpg>   # compare skyline-extraction methods on one photo
uv run peakle demo run              # the original synthetic render→recover demo
```

The web app also serves `/bench`: a filterable strategy-matrix dashboard with oracle/photo
tracks, map/height strata, paired change from the prior, provenance, and per-cell failure drill-down.
It also exposes diagnostic pose-atlas studies, while keeping their evaluation-only GT oracles out
of the recommended-strategy ranking.

## How it works

1. **Extraction** — explicit colour, DexiNed, or segmentation backends produce photo-side
   skyline candidates. The hermetic default is colour; optional models are never downloaded
   implicitly by a benchmark.
2. **Horizon solve** — for a known position the 360° horizon `el(az)` is
   ray-cast once (`peakle.localize.solve.HorizonProfile`); every yaw, effective vertical-crop,
   and FOV hypothesis is a resampling scored by a capped skyline chamfer with top-K basin polish.
3. **Arbitration** — among skyline hypotheses within a tight chamfer slack, the
   winner is the one whose typed DEM outlines best *explain* the photo's edges —
   this is what rejects a lake edge masquerading as the skyline.
4. **Position/yaw experiments** — CMA-ES, Powell, Nelder–Mead, differential evolution,
   contour-database, and regional-grid searches share a projection-aware objective. Physical
   pitch/roll are not graded on the current cylindrical crops. Real tests show position drift and a
   reduced/full-resolution mismatch, so these methods are not validated or recommended.
5. **Render-match PnP** — an experimental branch renders an orthophoto-textured yaw fan, runs a
   pinned offline MINIMA or RoMa matcher, lifts valid render pixels to terrain coordinates, and
   fits the query pose with projection-aware RANSAC. It abstains on weak or degenerate consensus.
6. **Verdict** — the former CONFIRMED calibration depended on retired GT-v2 targets. Viable
   results now remain `UNCALIBRATED`; confidence must be recalibrated on held-out manual locations.

DEM sources: **Copernicus GLO-30** (default; auto-downloaded per 1° tile),
**SRTM** `.hgt`, and **swissALTI3D** 2 m patches where sub-30 m detail matters.

## Evaluation pipeline

Solver evaluation uses the published MANUAL GeoPose pose (the refined `info.txt` fields) and
explicit controlled-prior tracks. The later `info.txt` fields retain the original noisy Flickr
GPS/elevation/FOV only as a labelled, non-ranking diagnostic; they never enter a solver, prior,
success grade, or leaderboard.
`gt_dem_compat_v1` measures whether source PFM/depth and the best locally available terrain stack
agree at the **unmodified** metadata pose. It fixes only the unknown global vertical crop offset; it
never optimizes yaw/position and never reads a photo skyline or solver result. Some legacy matrix
cells use hard compatibility tiers for eligibility, but those seed-sensitive thresholds are not
calibrated and no longer define accepted research claims; the current program treats compatibility
as a continuous dataset diagnostic. `raw_camera_clearance_v1` separately records altitude/datum
incompatibility that vertical crop alignment could hide. GT views solve from `photo_auto` by
default; PFM remains an explicit diagnostic oracle and is never a silent fallback.

GT v2 pose-polish artifacts are retained for reconstruction diagnosis and outline research, but
they are not solver results, priors, or evaluation truth. The CLIs are:

```bash
python -m peakle.scripts.build_gt_v2 --manual      # refine the MANUAL corpus
python -m peakle.scripts.bench_geopose             # PFM + photo tracks, compatibility + provenance
python -m peakle.scripts.bench_pose_matrix --max-n 5        # controlled strategy/prior matrix
python -m peakle.scripts.bench_pose_atlas --help   # high-compute local proposal/ranking ceiling
python -m peakle.scripts.bench_synthetic_pipeline --help    # custom-pinhole stage upper bound
python -m peakle.scripts.acceptance                # end-to-end acceptance checks
```

## Research status

Peakle does not yet reliably recover metric camera position. The strongest current conclusions are:

- known-position horizon alignment can recover crop yaw;
- a dense local atlas contains reference-near poses, but skyline-only ranking selects displaced
  basins;
- source-PFM geometry can rerank those candidates only as a diagnostic oracle ceiling;
- the current RGB verifier abstains on all three controls but does not select a correct top pose;
  this is too small to calibrate its risk; and
- render-match/PnP remains experimental, with too little eligible real data and a known false
  accept in its same-family holdout.

The project is therefore in a consolidation and benchmark-design phase, not another solver-expansion
phase. The [research and development program](docs/research-and-development.md) is the single source
of truth for exact artifacts and hashes, accepted claims, literature baselines, benchmark metrics,
the capture-surface experiment, codebase guardrails, and go/no-go roadmap. Dated atlas, synthetic,
and PnP documents are historical evidence appendices.

## Data

Corpus and DEM inputs live under `local/` and are not committed:

- `local/data/geopose/` — GeoPose3K samples (`python -m peakle.scripts.fetch_geopose`).
- `local/data/copernicus/` — Copernicus DEM tiles (auto-downloaded on demand).
- `local/data/swissalti/` — cached swissALTI3D 2 m tiles, with deterministic edition selection.
- `local/data/swissimage/` — pre-provisioned orthophoto tiles; benchmarks never fetch missing tiles.
- `local/models/{minima,roma}/` — local pinned matcher manifests and checkpoints.
- `local/cache/matcher-correspondences/` — optional content-addressed learned-match cache.
- `local/derived/gt_v2/` — refined GT records, arrays and layer PNGs.
- `local/output/*-geopose-bench/` and completed pose-atlas study directories — immutable benchmark
  artifacts shown at `/bench`.

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
  localize/     localization experiments + evaluation pipeline (solve, extract,
                compatibility, strategy matrix, raycast, DEM adapters)
  web/          FastAPI app + the browser workbench (static/js: 3D map,
                unified Views list, Overview panel, inspector, camera model)
  scene/        in-memory workbench scene (terrain, views, pose solves)
  terrain/ rendering/ optimization/ domain/    supporting layers
  scripts/      first-class CLIs over the package (python -m peakle.scripts.*)
```
