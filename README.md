# Peakle

Peakle is a workbench for recovering a camera pose from a mountain view by matching image
evidence against a real-world elevation model (DEM). The target is position, orientation, and
camera calibration with or without a prior. The honest current baseline is narrower: the ray-cast
horizon solver can recover effective crop yaw at a supplied position, while full-pose and
regional/no-prior methods remain experimental and currently fail the frozen real-data matrix.

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
   ray-cast once (`peakle.localize.solve.HorizonProfile`); every (yaw, pitch, fov)
   hypothesis is a resampling scored by a capped skyline chamfer with top-K basin
   polish.
3. **Arbitration** — among skyline hypotheses within a tight chamfer slack, the
   winner is the one whose typed DEM outlines best *explain* the photo's edges —
   this is what rejects a lake edge masquerading as the skyline.
4. **Full-pose experiments** — CMA-ES, Powell, Nelder–Mead, differential evolution,
   contour-database, and regional-grid searches share a projection-aware objective. Real MAP_A
   tests currently show position drift and a reduced/full-resolution mismatch, so these methods
   are not validated or recommended.
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
Before a case can grade a solver, `gt_dem_compat_v1` measures whether the source PFM/depth and the
best locally available terrain stack agree at the **unmodified** metadata pose. It fixes only the
unknown global vertical crop offset; it never optimizes yaw/position and never reads a photo
skyline or solver result. A separate `raw_camera_clearance_v1` gate detects altitude/datum
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

### High-compute pose atlas

The first three-photo native-patch-assisted atlas searched a square with ±500 m east/north
half-width at 50 m spacing and every 1° yaw. The 15° yaw perturbation was recorded but not used to
constrain the search. Blind skyline ranking failed all three PFM and photo tracks (301.9 m and
416.6 m median position error), but the frozen lattice contained an evaluation-only, GT-selected
14.9–19.8 m horizontal/yaw hypothesis for every image. Under PFM evidence, a target-successful pose
entered the stored candidate prefix within its top 5, 10, or 25 modes. Crucially, the current
unregularized skyline score preferred a displaced pose even over a reference-east/north probe on
all three controls, so more iterations of that same score are not the answer. That result identified
independent range, occlusion, and typed-ridge verification over the shortlisted modes as the next
gate. A fixed, truth-audited reference-depth fusion now moves those three PFM-atlas winners to 18.5,
52.0 and 35.4 m (3/3 within 100 m); no individual skyline/depth/outline component reaches 3/3. This is an
analysis-only ceiling because its PFM was rendered at the reference pose, not a production result.
Over the automatic-photo candidate pools the same verifier reaches 2/3 at top one; on the remaining
case it promotes a target-successful mode from photo rank 1,221 to geometry rank 31, motivating a
multi-hypothesis photo/refinement beam rather than another hard top-one selection. See
[the high-compute pose-atlas study](docs/development/pose-atlas-study.md) for the truth boundary,
exact ranks, artifact hash, reproduction command, and research target.

### Synthetic stage contracts

The production cyltan atlas/raycast path now has exact-truth integration controls that separate
proposal coverage from blind ranking and label same-raycaster depth as an identity ceiling. A
separate custom-pinhole/shared-renderer harness sweeps exact versus coarsened estimator terrain,
controlled priors, color/haze extraction, and fixed skyline/depth/outline scores. Its first 20-case
artifact keeps proposal recall at 20/20 by construction, but factor-two terrain coarsening drops
exact-mask skyline top-1 target hits from 8/10 to 3/10; automatic color skyline falls from 2/10 to
0/10. Under coarse terrain, absolute metric range reaches 6/10 versus 3/10 for scale-aligned range
and 2/10 for typed outlines. These are non-production upper-bound diagnostics, not real-photo wins.
See [the synthetic localization benchmark](docs/development/synthetic-localization-benchmark.md) for
the stage contracts, ambiguity rules and artifact hash.

### Render-match-PnP benchmark

The learned path is deliberately offline and reproducible. Its model manifest pins the source
checkout, inference settings, DINOv2 and matcher checkpoint hashes. `--matcher-cache` enables an
optional content-addressed cache of the worker's 5,000 candidates per image pair; a warm replay
reuses those exact candidates and reruns padding rejection, inverse-depth terrain lifting, spatial
holdout, independent balanced caps, and PnP/acceptance. `--native-patch-stride` controls fine-terrain mesh decimation: a 2 m source
at the default stride 8 is an approximately 16 m render mesh, not a claim of 2 m rendering.
swissALTI provisioning follows every STAC v1 page, selects one deterministic newest 2 m edition per
coordinate, and rejects any bilinear elevation sample whose four source cells are not finite.
Candidate-pose validation is enabled by default: a content-derived fold of an interleaved 8×6
query grid is withheld from PnP, then checked for reprojection and z-buffer visibility against a
fresh 2× render of the same terrain surface at the selected pose. `--disable-candidate-validation` exists only for explicit
ablation/backward controls; its disabled state and all fixed gate settings are persisted in the
artifact configuration.

```bash
python -m peakle.scripts.bench_pose_matrix \
  --samples eth_ch1_IMG_4948_01024 \
  --profile core \
  --algorithms keep-prior,render-pnp \
  --evidence photo_rgb \
  --regimes perturbed_metadata \
  --perturbation standard \
  --replicates 3 \
  --render-matcher worker \
  --render-modality orthophoto \
  --render-refinement-passes 0 \
  --matcher-command "$PWD/.venv/bin/python $PWD/src/peakle/scripts/roma_match_worker.py" \
  --matcher-id minima_roma \
  --matcher-manifest local/models/minima/manifest.json \
  --matcher-cache local/cache/matcher-correspondences \
  --orthophoto-cache local/data/swissimage \
  --native-patch-stride 8 \
  --output local/output/<new-run>-geopose-bench
```

The initial one-image GLO-30 control is negative: MINIMA peaks at 72/800 inliers (9%), RoMa stays
below 4%, and both abstain. A corrected high-resolution Swiss control now exercises the full
orthophoto + swissALTI path. On the sole currently eligible `MAP_B/HEIGHT_A` image, three standard
pre-gate MINIMA perturbations and the paired RoMa run all recover yaw within 1.02° and form strong
consensus, but miss the published refined reference by 203.6–236.5 m. They land 31.6–35.8 m from the
retained original noisy GPS, which is interesting evidence of a reference/appearance ambiguity—not
a reason to change the success target.

A frozen replay with the default spatial geometric holdout abstains on two wrong MINIMA candidates
(235.5 m and 202.0 m from the refined reference), but still accepts a third candidate that is 215.0 m
wrong. The matcher sees the complete query before the holdout, so this is not independent evidence
and the false accept shows that it cannot certify a systematic same-family alternate solution. The
method therefore remains experimental and ranking-excluded; independent silhouette/ridge evidence
is the next required validation layer. Numbers, exclusions, and the replay diagnostics are recorded in
[the pose-localization strategy](docs/development/pose-localization-strategy.md).

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
