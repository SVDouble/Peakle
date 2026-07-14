# Pose localization strategy after the refined-pose reset

The product goal is not to draw a plausible DEM overlay. It is to recover a camera pose that
improves a real prior—or localizes without one—and to abstain when the image/terrain evidence does
not support that result.

## Current evidence

Only the orientation-only horizon solver currently has credible multi-sample real-data evidence on
GeoPose's cylindrical/tangent crops. The render-match-PnP branch described below is now implemented,
including learned matching and native-patch rendering. It now has corrected high-resolution Swiss
controls, but not a successful eligible case against the published refined reference. It remains
experimental and ranking-excluded. The baseline every new method must beat is **keep the identical
prior unchanged**.

The older full-pose point-objective strategies remain unvalidated: reduced and native-width scores
disagree and real MAP_A tests show position drift. They do not become credible merely because the
render-match-PnP path can now consume a fine elevation patch.

The matrix exposes three evidence tracks:

- `pfm_oracle`: source depth/PFM skyline; isolates terrain, camera conventions, and pose search.
- `photo_auto`: automatic photo skyline; measures the end-to-end system.
- `photo_rgb`: the query RGB image; used only by the render-matching branch.

Before a solver track is scored, `gt_dem_compat_v1` measures fixed-pose source-depth↔DEM agreement.
It allows only a cross-fitted global vertical crop shift and reports angular median, p90, trimmed
symmetric Chamfer, coverage, and the fitted shift. `raw_camera_clearance_v1` independently checks
the unmodified metadata altitude against DEM ground. PFM edge support in the photo is a separate
diagnostic, not a claim that the image skyline was reproduced. None reads a solver result or GT-v2 refinement.

Terrain policy is coarse-to-fine. Copernicus GLO-30 remains the regional surface and cached
swissALTI3D supplies a prior-centred near-field patch where available. Render-match-PnP composites
that native-source patch over the regional mesh in the same depth buffer. The estimator may obtain
the patch only from its supplied position prior; a separate reference-centred patch used by the
GT↔DEM compatibility evaluator is never passed to the estimator. A no-position-prior cell receives
no reference-centred patch.

Swiss terrain provisioning uses the
[geo.admin STAC v1 endpoint](https://docs.geo.admin.ch/download-data/stac-api/overview.html) and
follows every continuation link; the API page size is not treated as the catalogue size. For each
kilometre coordinate, the cache
selects the deterministic newest available 2 m edition, both while downloading and while loading an
existing cache. Nodata is resampled separately from elevation: a bilinear result is finite only when
all four contributing source cells are finite. This prevents the `-9999` source sentinel from being
blended into plausible-looking negative terrain near tile gaps.

Every artifact distinguishes source spacing from rendered mesh spacing. For example, a 2 m
swissALTI3D source with `--native-patch-stride 8` is rasterized as an approximately 16 m mesh, while
the current GLO-30 control uses a 30 m source with terrain stride 6, or approximately 180 m effective
regional mesh spacing. The provenance reports both; neither upsampling nor a finer texture creates
new elevation detail. Full-pose point objectives still omit the native patch and remain
ranking-excluded when one is available.

## With a location/orientation prior

### 1. Prior-regularized outline search

This remains the lower-cost comparator and likely production-first path. Search orientation and
camera calibration before widening position, constrain height to
`DEM(east,north) + eye_height`, retain top-K modes, and use a robust angular loss:

- trimmed bidirectional distance-transform/Chamfer skyline loss;
- derivative/curvature agreement so a smooth vertical offset cannot dominate;
- separately weighted internal ridges and occlusion boundaries;
- masks for clouds, vegetation, poles, buildings, and low-confidence columns;
- Mahalanobis prior cost with an explicit unchanged-prior competitor;
- replace the prior only when held-out fit and mode margin improve materially.

Baboud et al. demonstrate robust photo-edge to rendered terrain-silhouette alignment from an
estimated viewpoint and FOV: [Photo-to-terrain alignment](https://resources.mpi-inf.mpg.de/photo-to-terrain/).

### 2. Render matching, 3D lifting, and PnP/RANSAC (implemented, experimental)

The implemented high-upside branch follows LandscapeAR:

1. render a prior-centred yaw fan of orthophoto-textured DEM views with metric depth, normals,
   world coordinates, and silhouette;
2. run a pinned MINIMA or RoMa worker offline and retain its deterministic 5,000 candidates per
   query/render pair;
3. reject border-connected near-black padding introduced by the cylindrical/tangent query warp;
4. reject matches that lift to sky, missing appearance, or a render-depth discontinuity;
5. deterministically reserve one content-derived fold of an interleaved 8×6 query grid and keep
   those lifted correspondences out of geometric PnP and geometric frame ranking;
6. independently apply the joint query/render spatial cap to the training and held-out sets,
   retaining at most 800 and 400 balanced matches respectively;
7. solve with projection-aware nonlinear PnP/RANSAC through the exact query camera model;
8. optionally re-render once around an accepted estimate and repeat matching/refinement while
   excluding the same held-out fold; and
9. render exactly the selected candidate pose independently, then require the held-out points to pass
   reprojection and explicit z-buffer testability/visibility-consistency gates.

The `5,000 → padding filter → render lift → spatial holdout → independent balanced caps` ordering is deliberate. The
matcher cache stores the worker-selected candidates, not the later geometry-dependent subset, so
padding, lifting, cap, RANSAC, and acceptance changes can be replayed without another model pass.
Selection and every rejection count are persisted per render frame. Candidate validation uses four
fixed interleaved folds; the query content digest chooses the held-out fold deterministically. The
subpixel lift follows the rasterizer's perspective-correct convention: interpolate finite positive
inverse depth, invert once at the keypoint, and construct the world point on that exact camera ray.
It never bilinearly interpolates the stored forward-depth image.

Visibility uses a fresh 2× auxiliary pinhole render of the same hashed terrain surface. For a
`cyltan` query, its candidate crop-shift angle only centres vertical frustum coverage and is not
presented as calibrated physical pitch; query reprojection still uses the exact `cyltan` model. Each
held-out terrain point needs an all-finite 3×3 depth neighbourhood whose span is at most the smaller
of 250 m and 8% of median depth. This deliberately removes only gross discontinuities: a steep smooth
mountain face can also have a large image-space span, so exact triangle/ray visibility remains a
future improvement. The actual ordering check is much tighter, comparing inverse-depth-interpolated
z-buffer depth with the point depth using
`max(1 m, min(3 m, 0.0001 × expected depth))`; large local spans never widen that tolerance.

The default 8×6 grid, four folds, 400-match holdout cap, 5 px reprojection threshold, 50%
candidate-render testability floor, 80% conditional visibility-consistency floor, and minimum 14
conditional trials are recorded in the artifact. The implementation also records nominal one-sided
95% Clopper–Pearson lower-bound gates. Those bounds are exact only for independent Bernoulli trials;
dense learned matches are spatially correlated, so they are heuristic checks rather than calibrated
coverage guarantees. The raw counts and spatial-distribution gates remain authoritative diagnostics.
Joint held-out support must also pass the PnP geometry gates for unique 3D points, horizontal/3D
baseline, non-collinearity, and camera angular span; a spatially spread but degenerate world locus is
not accepted.
The held-out matches come from the same source-render matcher family, making this a truth-free
geometric cross-check—not independent image evidence or a second matcher vote. Matcher inference
still sees the full query image, and the worker's learned correspondence ranking/cap happens before
the geometric holdout split; the holdout therefore closes the PnP fit/grade loop but is not withheld
from feature extraction or correspondence generation. A failed gate returns the explicit
`candidate_pose_holdout_validation_failed` abstention and never retries a runner-up.

The gate is enabled by default. `--disable-candidate-validation` is retained only for declared
ablation or backward controls; the opt-out and the complete fixed configuration are serialized in
`run.json` and the per-case diagnostics.

LandscapeAR used twelve 30° render directions and this exact lift→PnP structure:
[LandscapeAR, ECCV 2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf).

The worker accepts two RoMa-architecture checkpoints:

- [MINIMA, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Ren_MINIMA_Modality_Invariant_Image_Matching_CVPR_2025_paper.pdf) ([official code](https://github.com/LSXI7/MINIMA))—the primary cross-modal candidate;
- [RoMa, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.pdf) ([official code](https://github.com/Parskatt/RoMa))—the RGB wide-baseline comparator.

[MatchAnything](https://arxiv.org/abs/2501.07556) ([official code](https://github.com/zju3dv/MatchAnything))
remains a promising broad-modality experiment, but it is not accepted by the current pinned worker.

MINIMA and RoMa inference runs out of process. A local manifest pins the allowed matcher ID,
repository commit, inference settings, DINOv2 and matcher checkpoint paths, byte sizes, and SHA-256
digests. The worker verifies the clean checkout and artifacts before import, blocks Torch Hub and
model downloads, and records runtime/VRAM provenance. A missing or mismatched artifact is a visible
skip/error, never an implicit download.

`--matcher-cache DIR` enables an optional read-through, content-addressed correspondence cache. Its
key includes the exact query/render RGB content, coordinate/offline contract, seed, normalized and
source manifests, inference configuration, artifact identities, worker command, and worker file
hashes. Cache entries are validated and corrupt entries are recomputed atomically. On a warm replay,
producer inference runtime remains historical provenance while `current_worker_runtime_s` is zero
and the batch says that the worker was not invoked. Omitting the flag disables cache reads and writes.

For GeoPose `cyltan` queries with a position prior, PnP resolves the vertical crop/altitude
degeneracy by parameterizing camera height as
`DEM(east, north) + bounded_clearance`. The bounded clearance is anchored to the supplied prior;
crop pitch remains a separate image nuisance. A supplied clearance is trusted only inside the
declared 0.5–12 m ground-camera range. Values above or below that range use a neutral 2 m anchor and
0.5–12 m bounds instead of turning a horizontally displaced/vertically noisy prior into an aerial
camera constraint. RANSAC mixes uniform and progressive-confidence
trials, computes the exact finite-population trial count for the declared 20% inlier floor and 99%
target probability, and records whether its hard budget meets that contract. Acceptance also gates
query-image coverage, duplicate/near-collinear 3D support, horizontal and 3D baseline, camera angular
span, and physical clearance. Planar mountain surfaces are explicitly allowed. A failed gate produces
an explicit abstention with the best candidate and diagnostics, not a pose disguised as success.

#### Current GLO-30 control

On the frozen one-image GLO-30 control, MINIMA's strongest frame reaches 72 inliers out of the
balanced 800 (9%); RoMa remains below 4%. Both correctly abstain below the 20% acceptance floor. The
query contains 7.07% detected cylindrical-warp padding and substantial non-terrain occlusion, while
no cached swissALTI3D patch covers the sample and the GLO-30 elevation mesh is effectively about
180 m at the configured stride. The cached z14 orthophoto is appearance only and makes no claim of
native imagery or elevation resolution. This is a useful negative diagnostic of that image/render
configuration, **not** a dataset-wide matcher comparison or evidence that the strategy cannot work.

#### Corrected Swiss high-resolution controls

The first provisioned sample, `eth_ch1_48605067_01024`, demonstrated that the pipeline can solve a
real orthophoto-textured native-patch case: MINIMA finished 37.1 m / 0.62° from the reference and
RoMa 44.2 m / 0.46°. It cannot count as primary evidence. Once the complete deterministic 2 m patch
replaced the clipped catalogue result, the fixed-pose evaluator reclassified it as `MAP_C/HEIGHT_B`
(median 1.50°, p90 2.80°, raw clearance -5.39 m). Both solver cells also remain ranking-excluded as
experimental.

`eth_ch1_IMG_4948_01024` is the first completed native-patch control that retains primary-compatible
input status: `MAP_B/HEIGHT_A`, fixed-pose median 0.19°, p90 0.75°, raw clearance 2.61 m, and 98.0%
photo-edge support. Its estimator patch is a 5501×5501 2 m source grid with about 87.4% finite
coverage; stride 8 produces an approximately 16 m render mesh. The z14 SWISSIMAGE input was
pre-provisioned and hashed before the offline runs.

The pre-gate result is negative:

| matcher/run | refined horizontal | yaw | consensus | original-GPS diagnostic |
|---|---:|---:|---:|---:|
| MINIMA standard replicate 0 | 235.0 m | 0.89° | 475/800 (59.4%) | 34.5 m |
| MINIMA standard replicate 1 | 203.6 m | 1.02° | 308/800 (38.5%) | 31.6 m |
| MINIMA standard replicate 2 | 236.5 m | 0.89° | 491/800 (61.4%) | 35.8 m |
| RoMa paired control | 232.8 m | 0.90° | 463/800 (57.9%) | 34.7 m |

All four estimates fail the declared 100 m refined-reference position threshold. The three MINIMA
solutions span only about 34 m despite different 200 m horizontal, ±75 m vertical, and ±15° yaw
perturbations; MINIMA and RoMa also recover nearly the same pose. One MINIMA replicate reaches the
12 m clearance ceiling, so even the stability evidence is not an unconditional acceptance signal.

The [GeoPose3K dataset](https://cphoto.fit.vutbr.cz/geoPose3K/) retains two metadata blocks: the
published refined pose/FOV is the sole benchmark reference,
while the original noisy Flickr GPS/elevation/FOV is diagnostic only. The latter is recorded after
solving with `used_by_estimator=false`, `used_for_success_grading=false`, and
`used_for_ranking=false`. Agreement with that noisy GPS cannot convert these failures into successes
or justify relabelling the reference. It does show that the failure is systematic enough to require
an independent physical check, rather than another reprojection threshold tuned on this image.

The frozen default candidate gate was then replayed with the same sample, seed `20260713`, and three
MINIMA perturbations. The run is
`20260714-heldout-candidate-validation-minima-img4948-geopose-bench`; its `results.json` SHA-256 is
`ecfd52bc375305987aaeea8c0244209a2621e31706440cbab9b316479bbfe14d`. Candidate-to-reference errors
below were computed only after validation and were not gate inputs:

| replicate | gate outcome | candidate horizontal / yaw | query reprojection | testable lower bound | visibility lower bound | joint lower bound | gate failure |
|---|---|---:|---:|---:|---:|---:|---|
| 0 | abstained | 235.5 m / 0.91° | 268/400 | 0.770 | 0.707 | 0.450 | visibility ordering |
| 1 | abstained | 202.0 m / 1.00° | 126/400 | 0.332 | 0.620 | 0.165 | testability, visibility, joint support/distribution |
| 2 | returned; benchmark failure | 215.0 m / 0.98° | 145/400 | 0.743 | 0.843 | 0.265 | none |

The required nominal lower bounds are 0.50 for testability, 0.80 for conditional visibility
consistency, and 0.20 for joint support. They are one-sided 95% Clopper-Pearson bounds, exact only
under an independent Bernoulli model; dense learned matches are spatially correlated, so these
bounds are heuristic and not calibrated coverage guarantees. Replicate 2 passed every frozen gate
with 121/400 joint held-out matches but remained 215.0 m from the sole refined reference. Thresholds
were not changed after observing the replay. Two safe abstentions therefore do not establish an
improved localization rate: this control contains a surviving false accept in the same systematic
wrong-position basin.

A reproducible focused run (replace the output path for every run) is:

```bash
python -m peakle.scripts.bench_pose_matrix \
  --samples eth_ch1_IMG_4948_01024 \
  --profile core \
  --algorithms keep-prior,render-pnp \
  --evidence photo_rgb \
  --regimes perturbed_metadata \
  --perturbation standard \
  --replicates 3 \
  --seed 20260713 \
  --render-matcher worker \
  --render-modality orthophoto \
  --render-refinement-passes 0 \
  --matcher-command "$PWD/.venv/bin/python $PWD/src/peakle/scripts/roma_match_worker.py" \
  --matcher-id minima_roma \
  --matcher-manifest local/models/minima/manifest.json \
  --matcher-cache local/cache/matcher-correspondences \
  --orthophoto-cache local/data/swissimage \
  --orthophoto-zoom 14 \
  --orthophoto-time current \
  --native-patch-stride 8 \
  --output local/output/<new-run>-geopose-bench
```

Use `--matcher-id roma_outdoor --matcher-manifest local/models/roma/manifest.json` for the paired
RoMa control. Benchmarks never fetch missing orthophoto tiles or model files.

#### Candidate validation and recommended next experiment

The first part of the recommendation is now implemented: every default run holds out spatial cells,
re-renders the selected candidate, and checks held-out reprojection plus terrain visibility/occlusion
ordering before returning a pose. This closes the same-correspondence geometric fit/grade loop. It is
still only a spatial geometric holdout: the learned matcher sees the complete query and worker
correspondence selection happens before the fold is withheld. The frozen Swiss replay above rejects
two candidates but accepts a third wrong one, demonstrating that the gate cannot prove a
systematically biased cross-modal match family is geographically correct.

Do not relax the 100 m target or accept consensus alone. The next independent layer should compare
the candidate render against evidence not supplied to PnP: silhouette/ridge alignment, photo edge
support, and nuisance/clearance boundary checks. An unchanged-position competitor and a
Schur-projected horizontal uncertainty estimate remain useful observability gates, but they cannot
catch a set of systematically biased matches that genuinely prefers the wrong translation.

The subsequent [three-photo high-compute atlas](pose-atlas-study.md) confirms the same gate from a
different proposal family: the frozen search contains GT-selected 14.9–19.8 m horizontal/yaw grid
hypotheses for all three controls, but the unregularized skyline score ranks displaced 124–359 m PFM
poses first. On PFM evidence, a target-successful mode is already present within the first 5–25
stored candidates, so independent range/occlusion/typed-ridge verification should precede another
optimizer or correspondence refinement pass.

That reference-depth ceiling is now implemented and truth-audited. Its fixed skyline, typed-outline,
centred-depth and overlap fusion selects 18.5, 52.0 and 35.4 m candidates (3/3 within 100 m), versus
the 358.5, 301.9 and 124.1 m skyline winners. Typed outlines alone reach 2/3; overlap reaches 1/3;
centred depth and skyline alone reach 0/3. The interaction is the strategy worth transferring to
photo evidence, not any one oracle component. Because the source PFM is reference-pose-rendered,
these are still analysis-only ceilings.

Over the automatic-photo atlas pools, the same frozen verifier reaches 18.5, 92.2 and 385.3 m: 2/3
at top one. In the failed IMG5145 control it nevertheless promotes a target-successful hypothesis
from photo rank 1,221 to geometry rank 31 and removes the 123° yaw failure. On this control, the next
photo strategy must retain at least a 32-mode beam, rerender/match those modes, and require an
independent margin or abstain; the required beam size must be calibrated on held-out data. Forcing
another top-one skyline decision would discard the measured recovery path.

The follow-up [synthetic stage contracts](synthetic-localization-benchmark.md) isolate two more
failure sources. First, the old implicit ray range discarded diagonal terrain; it now reaches the
per-camera farthest DEM corner and has analytic diagonal/offset/cap tests. Second, in a declared
shared-renderer upper bound, factor-two estimator-terrain coarsening drops exact-mask skyline top-1
target hits from 8/10 to 3/10 and scale-aligned depth from 10/10 to 3/10. Absolute metric depth is
more robust under that mismatch (6/10), while radial controls show that even exact self-rendered
geometry still needs an ambiguity abstention rule. This makes terrain-resolution compatibility and
absolute range the next isolated production gates; the custom pinhole numbers are not production
atlas or real-photo results.

For the no-position-prior track, build the geographic 360° DEM-horizon index first, retrieve top-K
location/yaw cells, and only then invoke the same fine orthophoto render → learned match → PnP stage.
That supplies auditable regional recall and a bounded search window before any expensive dense
matching; a general geolocator may propose additional regions but must not count as the final pose.

[Towards Unconstrained Cross-View Pose Estimation, WACV 2026](https://openaccess.thecvf.com/content/WACV2026/papers/Wollam_Towards_Unconstrained_Cross-View_Pose_Estimation_WACV_2026_paper.pdf)
is worth an aerial-image auxiliary experiment because it searches unknown FOV, pitch, roll, and
projection. It is trained/evaluated on ground-to-aerial imagery rather than bare terrain, however,
so it must not displace the geometry-verifiable DEM pipeline without a mountain-domain benchmark.

Use [GeoCalib](https://github.com/cvg/GeoCalib) to estimate FOV, gravity, roll, distortion, and
confidence independently before terrain optimization. Monocular depth/normals can support the
loss, but kilometre-scale absolute depth must not determine pose by itself.

### Segmentation backbone experiment: LingBot Vision

[LingBot-Vision](https://huggingface.co/robbyant/lingbot-vision-vit-giant) is relevant to the
photo-evidence problem because its boundary-centric pretraining is designed for dense prediction.
The published Giant checkpoint is a **backbone**, however: it emits normalized patch tokens and
does not include a trained sky/terrain segmentation head. It therefore cannot replace SAM or the
current colour extractor directly.

Treat it as a measured downstream experiment: freeze a Base or Large backbone first, train a small
binary decoder/linear probe on sky/terrain masks, and compare boundary F-score, skyline angular
error, coverage, runtime, and downstream pose recall on a location-held-out split. Only try the
roughly billion-parameter Giant model if the smaller variants demonstrate a useful boundary gain.
The [official repository](https://github.com/robbyant/lingbot-vision) and
[paper](https://arxiv.org/abs/2607.05247) provide the backbone interface; neither supplies a
ready-made mountain-sky decoder.

### 3. Featuremetric refinement

After a good discrete hypothesis exists, test PixLoc-style multi-scale feature alignment or a
differentiable silhouette/depth renderer. These are local refiners, not global initializers:
[PixLoc](https://arxiv.org/abs/2103.09213) and the
[PyTorch3D camera-optimization tutorial](https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering).

## Without a prior

### 1. Geographic DEM skyline index

Precompute 360° horizon profiles on a hierarchical terrain grid, encode overlapping contourlets,
retrieve/vote for location and direction, then verify top-K candidates at full resolution. The ETH
Alps system is still the most directly validated mountain-specific blueprint, reporting 88% within
1 km across Switzerland: [project](https://cvg.ethz.ch/research/mountain_res) and
[IJCV paper](https://people.inf.ethz.ch/pomarc/pubs/SaurerIJCV15.pdf).

Modernize retrieval with a learned 1D skyline descriptor plus ANN, while retaining geometry-aware
location/direction voting and dense final verification.

### 2. Photo-to-render retrieval, then the same PnP stage

Precompute multi-FOV render descriptors, retrieve top-K regions, locally re-render at the finest
available DEM resolution, then apply cross-modal matching and PnP. AnyLoc is a useful cross-domain
retrieval baseline: [official code](https://github.com/AnyLoc/AnyLoc). A Peakle-specific dual encoder
trained only on validated photo/render pairs is the likely end state.

### 3. Region proposal and reference-photo localization

General geolocation can reduce the planet to candidate regions, never serve as the final pose:
[Scaling Image Geo-Localization to Continent Level, NeurIPS 2025](https://scaling-geoloc.github.io/)
is the strongest recent regional proposal to benchmark first: it fuses learned ground prototypes
with aerial embeddings and reports 68% of its Europe queries within 200 m, while also showing that
rural cases remain a weakness. Treat its top-K cells as search windows for DEM verification, not
as a camera pose.
[PIGEON/PIGEOTTO, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Haas_PIGEON_Predicting_Image_Geolocations_CVPR_2024_paper.html)
and [GeoCLIP, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html).
[HierLoc, ICLR 2026](https://iclr.cc/virtual/2026/poster/10008439) is a compact global region
proposal when storing a huge reference-image bank is impractical, but its country/region/city
entities are intentionally too coarse to count as a solved camera pose.

Popular viewpoints also justify a reference-photo branch. Benchmark
[ImLoc (2026)](https://arxiv.org/abs/2601.04185) first: it augments database images with depth,
uses dense query↔reference matching, lifts those matches through depth, and estimates pose with
PnP/RANSAC. Retain [Hierarchical Localization](https://github.com/cvg/Hierarchical-Localization)
as the established sparse comparator; [AsymLoc, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Omama_AsymLoc_Towards_Asymmetric_Feature_Matching_for_Efficient_Visual_Localization_CVPR_2026_paper.html)
is an efficiency option if query-side compute matters. Follow any reference-photo result with DEM
alignment, and audit Flickr/reference overlap before reporting a number.

## Benchmark contract

Run three tracks: real raw-metadata priors, controlled perturbations bucketed by initial error, and
no-prior regional/global localization. Split by massif/location, photographer, and near-duplicate
group—not random images. Report:

- horizontal/vertical position, yaw, and FOV errors; effective crop-pitch nuisance separately
  (physical pitch/roll are not scoreable until the GeoPose vertical crop transform is known);
- recall at 100 m/2°, 200 m/5°, and for no-prior retrieval 1 km/5° top-1 and top-5;
- improvement over prior, regression rate, unchanged-prior win rate, and abstention;
- top-K recall, risk–coverage/confidence calibration, runtime, renders, RAM/VRAM;
- all MANUAL cases, `MAP_A`, `MAP_A + photo_edge_support`, and identifiability bins separately;
- paired, location-grouped bootstrap confidence intervals.

Do not compare optimizer-native costs. Re-render every output at full resolution and score it with
one independent evaluator. Do not optimize dense RGB against an untextured DEM, force a single
hypothesis, or use end-to-end pose regression as the primary system.
