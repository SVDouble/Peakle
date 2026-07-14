# Peakle research and development program

**Status:** normative

**Last reviewed:** 2026-07-14

**Current phase:** consolidation and benchmark design

This document is the sole normative source for Peakle's research goal, interpretation of accepted
evidence, benchmark contracts, technical direction, and current roadmap. Its adjacent
`research-evidence.json` is the canonical machine inventory of artifact identities. The root README
explains how to use the project. Architecture notes explain mechanisms. Dated study documents
preserve experimental detail, but they do not set current priorities or turn a diagnostic result
into a product claim.

Any change to the research direction must update this document in the same commit. Any accepted
benchmark result must update both the human ledger below and `research-evidence.json` with its
immutable artifact hash, truth class, sample scope, result, decision, parent chain, and archive
availability. The JSON file is authoritative for artifact identities; this document is authoritative
for their interpretation and the roadmap. Large artifacts may remain outside Git, but a result not
represented in both places is not part of the project's accepted evidence.

## Executive decision

Peakle does **not yet solve metric mountain-view localization**. It has a credible known-position
orientation solver, a dense three-case diagnostic lattice with measured proposal recall, and useful
negative results. It does not have a real-data method that reliably improves a position prior, a
validated country-scale no-prior method, a calibrated confidence score, or an end-to-end
mountain-annotation benchmark.

The project is not primarily blocked by optimizer budget. The local atlas already contains
reference-near poses, but the skyline score ranks displaced basins above them. PFM-derived geometry
can recover the right basin as an oracle ceiling; current RGB-derived geometry cannot. Generic
photo-to-render matchers can form internally coherent correspondences at a pose roughly 200 m from
the published GeoPose reference, and the current same-family holdout can falsely accept that pose.
Those are ranking, observability, truth, and validation problems—not evidence that another local
optimizer or a larger beam will solve the task.

Until the gates in this program are met:

- stop adding end-to-end solver variants to the same three-photo benchmark;
- do not describe a truth-selected oracle, source PFM, shared-renderer synthetic result, or
  identity-seed integration test as localization;
- do not promote a method because an overlay looks plausible;
- keep every experimental estimator out of product ranking; and
- make the next implementation milestone a net-deleting consolidation, followed by a controlled
  annotation-sensitivity and exact-pose modality benchmark.

## Review decision: what changes now

The previous roadmap had the right decomposition but the wrong execution shape. Phase 0 bundled a
schema registry, migrations, runner, projector, remote archive, and every historical artifact into
one framework project. The capture study then jumped directly to a large offset lattice without
first asking whether a representation produces geographically correct correspondences at the exact
pose. Baselines for trusted-pose annotation, local refinement, and country-scale retrieval were also
mixed into one queue.

The program is therefore changed as follows:

1. **Derive the pose target from annotation utility.** A universal 100 m threshold is retained for
   continuity, but it is not a product requirement. Label displacement depends on peak distance,
   view direction, FOV, and orientation; task A defines the useful error budget.
2. **Separate observability from capture.** First measure whether perfect skyline, mask, outline,
   depth, or normal observations distinguish directional translations. Then screen
   representation × matcher at the exact pose. Only successful pairs receive offset sweeps and PnP.
3. **Treat the atlas honestly.** A dense lattice deliberately covering truth is an enumeration and
   coordinate-system check. Its meaningful proposal claims are recall@K and compute under a declared
   prior distribution, never its full-lattice truth-selected oracle.
4. **Run four bounded tracks in parallel.** Trusted-pose annotation, prior-assisted local capture,
   data/truth, and net-deleting infrastructure work may proceed independently. Country-scale
   retrieval waits for a useful local stage unless run as an explicitly standalone baseline.
5. **Freeze failed estimators.** Skyline-only translation optimizers, the hand-tuned RGB verifier
   v1, and the current small-window `contourdb`/`global` methods are historical diagnostics, not
   active method families. Their reusable contracts remain; their weights and objectives are not
   tuned further.
6. **Do not migrate history into a new framework.** Preserve old artifacts by immutable hash and a
   compact projector. New experiments adopt the next typed truth/evaluation boundary one slice at a
   time. Generic byte persistence belongs in `peakle.io`; research truth never belongs there.
7. **Calibrate before complexity.** Several scores derived from the same RGB image, mask, backbone,
   or training data are correlated. The requirement is evidence not reused for fitting plus measured
   conditional value and risk/coverage—not an unsupported claim that heads are independent.

These changes intentionally reduce the next experiment. The immediate implementation target is an
annotation-sensitivity benchmark plus a small exact-pose modality screen, not another end-to-end
solver or a run of all 32 verifier candidates.

## Mission and task boundaries

The product outcome is an image annotated with the correct visible mountains, with errors and
uncertainty that a user can understand. Camera pose is an enabling latent variable, not the only
success metric.

Peakle must treat three tasks separately:

### A. Annotation from a trusted pose

Given an image, calibrated camera model, and sufficiently accurate pose, identify visible mountain
summits and ridges and place labels correctly. This tests terrain visibility, camera projection,
peak identity, occlusion, and presentation without hiding annotation failures behind localization.

### B. Prior-assisted pose recovery

Given an image plus a noisy position/orientation/FOV prior, return a better metric pose or abstain.
The unchanged prior is always a competing baseline. The estimator must not read evaluation truth,
source PFM, or reference-derived masks.

### C. Localization without a pose prior

Given an image and a declared search region—or the whole world—retrieve plausible regions and then
run task B locally. Global geolocation, regional terrain retrieval, and metric pose refinement are
different stages with different error scales. A kilometre-scale retrieval result is not a 100 m
pose result.

GeoPose cylindrical/tangent crops currently expose effective crop yaw and vertical shift more
cleanly than physical pitch and roll. Until a calibrated dataset makes all six degrees of freedom
observable, reports must name the exact pose components they score; “full pose” is not an acceptable
shortcut.

## What has actually been achieved

All paths below are relative to the repository root. Files under `local/output` are ignored and
therefore not durable by themselves; their SHA-256 values are the evidence identity. The committed
[machine-readable evidence ledger](research-evidence.json) records their run/evaluation identities,
parents, source revisions, archive locators, and compact outcomes. Until a remote content-addressed
archive exists, `availability: local_only` is an explicit reproducibility limitation. Status labels
mean:

- **baseline:** an honest comparator the project can reproduce;
- **negative:** a valid experiment that falsified or bounded an approach;
- **diagnostic ceiling:** uses truth-derived or otherwise unavailable production evidence;
- **experimental:** truth-blind estimator evidence, but too small or uncalibrated for validation;
- **superseded:** retained only for history.

| Evidence | Scope and truth class | Result | Decision |
|---|---|---|---|
| Current skyline matrix, `local/output/20260714-three-photo-current-skyline-baseline-geopose-bench/results.json`, SHA `96bfc60c9e4831792246ef3c06ccbe4db0bab43fc77870305ffce7e1bc54122b` | Three manual GeoPose photos, controlled 200 m / 15° priors; source-PFM and automatic-photo tracks | All position-recovery cells fail. The horizon method recovers yaw but retains the 200 m position error; CMA-ES median position error is about 192–231 m. | **Baseline / negative.** Known-position orientation works; position recovery does not. |
| Five-photo strategy matrix, `local/output/20260713-165612-matrix-geopose-bench/results.json`, SHA `25d44d578b9642d068ee5d86d92b29ea5f936566d0e9a074c509b57f662170ee` | Keep-prior, horizon, CMA-ES, contour-database, and regional/global variants over PFM/photo and prior/no-prior tracks | All eligible localization cells fail; the older no-prior contour/global variants have roughly 9–16 km median error. | **Baseline / negative.** Older local and regional strategies do not recover metric position. |
| Native skyline atlas, `local/output/20260714-three-photo-native-skyline-pose-atlas-v2/results.json`, SHA `5913a62ff65083457a3e7362edb1f228ffc1fef14c6cd5b0f213546e07f00e7d` | Same three photos; 441 positions × 360 yaws, with evaluation truth applied only after the lattice was frozen | Blind winners are 0/3: median 301.9 m with PFM skyline and 416.6 m with RGB skyline. A truth-selected lattice oracle is 3/3 at 14.9–19.8 m. | **Negative plus diagnostic ceiling.** Proposal coverage exists; skyline ranking is the failure. |
| PFM geometry rerank, `local/output/20260714-three-photo-pfm-geometry-rerank-v1/results.json`, SHA `0f246caaaf2858d5333842ee42997ef95f309369e186f8a27d5393c466e64c4d` | Atlas candidates scored with source-PFM/reference-depth geometry | Fixed fusion selects 18.5, 52.0, and 35.4 m poses; individual components do not reach 3/3. | **Diagnostic ceiling.** Independent range/structure can disambiguate the pool, but these observations are unavailable at inference. |
| PFM rerank of photo pool, `local/output/20260714-three-photo-photo-candidates-pfm-geometry-rerank-v1/results.json`, SHA `9d02a702e6b0f779c86406b64b53a95d5ecb89488b2938ab38032dae1f134c5e` | Automatic-photo candidate pools, still scored with source-PFM geometry | Top one is 2/3, median 92.2 m. On the failure, a target candidate moves from photo rank 1,221 to geometry rank 31. | **Diagnostic ceiling.** Both proposal recall and production ranking need work. |
| RGB geometry verifier, `local/output/20260714-three-photo-photo-geometry-verifier-v1/results.json`, SHA `a55235964989131d00b515eceffb4b5b98341e30940f4e39d41c6a81a33c7f9b` | Truth-blind DexiNed, monocular depth, and skyline evidence over 1,323 candidates per photo | Top one is 0/3, median 359.7 m; it abstains 3/3. A diverse beam of 32 contains a target for 2/3, both at beam rank 12. | **Experimental negative.** Abstention is safer than a false claim, but the verifier is not a useful selector yet. |
| MINIMA heldout validation, `local/output/20260714-heldout-candidate-validation-minima-img4948-geopose-bench/results.json`, SHA `ecfd52bc375305987aaeea8c0244209a2621e31706440cbab9b316479bbfe14d` | Three controlled perturbations of IMG4948 against prior-centred orthophoto renders; all cases ranking-ineligible | The gate abstains twice and returns one pose 215.0 m / 0.98° from the refined reference, for 0/3 successes. The matcher saw the complete query before the spatial holdout. | **Experimental / negative.** Plumbing works, but this same-family holdout cannot certify the ambiguous alternative pose. |
| Synthetic pinhole stage ceiling, `local/output/20260714-synthetic-pinhole-stage-upper-bound-v2/results.json`, SHA `bf813d184d4b6c99be00ff6c0b75c3e8d4b7c5115478d932cd77638209f180a8` | Five observations, two priors, exact/coarse estimator terrain; shared custom renderer | Proposal recall is 20/20 by construction. Exact-mask skyline top-one falls from 8/10 to 3/10 after factor-two terrain coarsening; RGB skyline falls from 2/10 to 0/10. | **Diagnostic ceiling.** It isolates terrain/extraction sensitivity; it is not an independent end-to-end result. |
| Six-photo GT↔DEM compatibility, `local/output/20260714-high-compute-six-photo-compatibility-v2-geopose-bench/results.json`, SHA `a2937e7ad1464dbfa59ad8cead798bda64be2d06e2a9cea57d30c977dee80488` | Reference PFM and metadata, evaluation-only | Produces two MAP_A, two MAP_B, and two MAP_C cases, but tiers change in nearby seeded runs around hard thresholds. | **Experimental dataset audit.** Useful as a continuous covariate, not yet as a calibrated eligibility gate. |
| Legacy GT alignment audit, `local/output/20260709-155112-gt-alignment-audit/audit.json`, SHA `bc7ce33733e4893f832257fb1fa594869ef7d41500a9744651dd22239abc598f` | 364 refined-pose-era records, without a modern run manifest | 192 CLEAN, 172 SUSPECT, including 126 photo/PFM registration mismatches. | **Superseded diagnostic.** Supports caution about the corpus but cannot grade the reset benchmark. |
| Legacy orientation studies: `local/output/20260707-045620-geopose-bench/results.json`, SHA `a7ac0c52572e1cf1bbabfa7cbb59f5e1bca5317d2af4c6c6648826fe4f123dd1`; and `local/output/20260705-211514-geopose-bench/results.json`, SHA `de57964571ff7af6f9f0e5255876f4bcd8abf6af300acf8302c308fcc232d58d` | Position and FOV are supplied; PFM is often reference-generated | 60-case run: PFM 95%, extracted 87%; 274-case run: PFM 93%, extracted 40%. | **Baseline.** Supports yaw/horizon capacity at a known position only. |

The result is a clear decomposition, not “nothing worked”:

1. terrain ray-casting and camera projection are useful enough to generate reference-near candidates;
2. known-position yaw recovery is the current credible algorithmic baseline;
3. skyline shape alone is structurally weak for translation over distant terrain;
4. metric range, occlusion, and typed structure can resolve ambiguity when supplied correctly;
5. current photo-derived proxies do not reproduce that oracle ranking; and
6. current real truth is too ambiguous and too small to decide whether some 200 m alternatives are
   algorithm failures or reference errors.

## Failure model

### 1. Truth is weak, small, and partly circular

GeoPose3K mixes a smaller manually captured/annotated subset with a larger automatically initialized
and image/model-refined subset (see the [author-hosted dataset paper](https://cphoto.fit.vutbr.cz/geoPose3K/data/geoPose3K_submission.pdf)).
Peakle's selected MANUAL references are manually checked but still are not surveyed camera stations.
The original Flickr metadata, refined references, and photo/render matching can disagree by hundreds
of metres. Six compatibility samples, three core atlas samples, and one eligible high-resolution
PnP sample cannot calibrate a general method. A method must not be tuned against the same pose that
was produced from similar image/model evidence.

### 2. Synthetic evidence has an inverse-crime risk

The current custom-pinhole benchmark shares terrain and rendering assumptions with the estimator.
Exact masks, exact depth, and exact seed identity are useful unit ceilings, not evidence of
robustness to a different renderer, DEM error, vegetation, snow, atmosphere, lens distortion, or
projection mismatch.

### 3. The pipeline conflates distinct failure stages

Extraction, proposal recall, ranking, local capture, verification, and annotation are often rolled
into one final pose number. A failed top-one result cannot say whether the right pose was absent,
misranked, outside the matcher capture range, rejected by PnP, or correctly rejected by a verifier.
Every benchmark must publish these stages separately.

### 4. Skyline translation is poorly observable

For distant mountains, sizeable camera translations can produce similar horizon profiles. Vertical
crop, FOV, terrain resolution, haze, clouds, foregrounds, and repetitive ridges create additional
aliases. More iterations over the same skyline objective deepen the wrong basin rather than add
missing range evidence.

### 5. Current photo cues are not independent

The RGB verifier derives a terrain mask from the selected skyline and then uses that mask in
downstream depth/edge cues. When the skyline is wrong, several supposedly separate scores fail
together. The render-PnP holdout also uses a matcher that has already seen the whole image, so it
cannot certify a systematic same-family false solution.

### 6. Generic learned matchers are not domain evidence

RoMa, MINIMA, and MatchAnything are strong generic or cross-modal correspondence models. None has
established mountain-photo-to-untextured-DEM pose recovery. Internally consistent matches against
orthophotos can still identify the wrong geographic basin. Monocular depth supplies useful relative
shape, but not trustworthy kilometre-scale absolute range by itself.

### 7. The downstream product metric is missing

Position error is not a sufficient annotation metric. A 100 m translation can be irrelevant for a
distant summit, while a one-degree yaw/FOV/crop error can move a label many pixels. Peak identity,
visibility, occlusion, image-space error, and uncertainty have not yet been benchmarked end to end.

### 8. Experiment infrastructure is growing faster than evidence

As of this review, `src/peakle/localize` is about 16,074 lines, benchmark scripts about 7,226, and
tests about 11,544. The newest experiment CLIs are 634–1,205 lines and repeatedly implement
canonical JSON, hashing, write-once publication, provenance, truth whitelisting, evaluation records,
and summaries. There are dozens of schema constants, several parallel pose-error/candidate models,
and tests that import private script helpers. This is why each experiment adds thousands of lines
and becomes expensive to retire.

## Truth and dataset program

Every case must declare one truth class. Results from different classes may be shown together but
must not be pooled into one success percentage.

| Class | Allowed use | Required safeguards |
|---|---|---|
| **Gold real** | Product accuracy, calibration, and go/no-go decisions | Surveyed/RTK camera position; independently calibrated boresight/orientation from surveyed control, total station, or a sensor/target process not reused by the estimator; calibrated lens/intrinsics; original image; recorded uncertainty; independently checked peaks/landmarks; high-resolution terrain where available; location-held-out split |
| **Weak real** | External validity, ranking comparisons, failure discovery | Preserve all competing metadata/reference poses and uncertainty; no silent relabelling; report results against each defensible reference; never tune and grade on the same locations |
| **Independent synthetic** | Controlled perturbations and causal stage tests | Forward renderer differs from estimator renderer; vary DEM resolution/bias/nodata, texture, vegetation, snow, haze, crop, lens, and camera model; keep truth sealed from estimators |
| **Diagnostic oracle** | Upper bounds and failure localization | Mark visibly in artifacts/UI; exclude from rankings, confidence, and product claims |

Gold v0 should be a small engineering pilot: at least ten location groups and roughly 50 images
across different view directions, distances, cameras, terrain types, and conditions. It is enough to
debug the protocol, not to calibrate a product claim. Capture raw images, calibrated intrinsics and
lens distortion, RTK/surveyed tripod coordinates, camera height, independently measured
yaw/pitch/roll or boresight, timestamps, and independent visible-peak/occlusion annotations. Do not
derive orientation truth from the same mountain correspondences used for grading. Split and
bootstrap by location group, not by near-duplicate image. Before Phase 3, register a
location-clustered power/sample-size plan for the target effect and false-accept rate. GeoPose
remains a weak-real external set with per-sample compatibility and reference-disagreement fields.

## Benchmark contract

### Shared rules

- Estimator inputs contain no numeric truth, truth-derived masks/depth, success flags, or
  truth-selected candidates.
- Freeze and hash estimator output before evaluation; reload it through a strict schema.
- Register the dataset split, code commit, config, model/checkpoint hashes, resource hashes, network
  policy, random seeds, and parent artifact hashes.
- Always run the unchanged input prior and report paired improvement/regression.
- Report distributions, counts, location-clustered confidence intervals, and per-location
  results—not just a median or image-wise pseudo-replication.
- Fit thresholds/calibration only on training/validation locations; freeze them before the test set.
- A verifier reports risk versus coverage and false-accept rate. Abstention is an output, not a
  hidden failure.
- Oracle and weak-truth results remain visibly separated in CLI, artifact, dashboard, and prose.

### Task A: annotation metrics

- visible-peak identity precision, recall, and F1;
- median and p90 angular and pixel label error;
- ridge/outline overlap where annotated;
- occlusion and behind-camera correctness;
- calibration of label-position/identity uncertainty; and
- metrics stratified by peak distance, prominence, terrain fidelity, weather, and crop.

### Task B: prior-assisted pose metrics

- recall curves at 25, 100, and 300 m horizontal error;
- yaw, FOV, vertical-crop, vertical-position/camera-height, calibrated-distortion, and—only where
  independently valid—pitch/roll error distributions;
- angular/SE(3) and image-space reprojection error where the camera model and gold control support it;
- paired delta from the unchanged prior and fraction made materially worse;
- top-K proposal recall, selected rank, selection regret, and local-refinement capture rate;
- verifier risk/coverage and false-accept rate; and
- task-A annotation change produced by the recovered pose.

### Task C: no-prior metrics

- region/top-K recall within 1, 5, and 25 km;
- retrieval compute, index size, and geographic coverage;
- task-B pose curves conditioned on successful retrieval; and
- final task-A annotation metrics over all queries, including retrieval failures.

### Required stage report

Every pipeline benchmark must expose this sequence:

```text
truth/data eligibility
  -> observation quality
  -> candidate proposal recall @ K
  -> truth-blind ranking @ K
  -> local matcher/refiner capture
  -> independent verification and abstention
  -> annotation accuracy with propagated uncertainty
```

No downstream aggregate may conceal an upstream miss. In particular, a truth-selected candidate is
proposal recall, not a localization success.

## Method portfolio and stop rules

The promise score below is a **0–5 research-priority estimate for producing defensible progress in
the next two phases**, not a probability of eventual success. Evidence is tiny and protocol-shifted,
so scores should be read with roughly ±1 uncertainty. A high diagnostic score can coexist with zero
deployability.

| Method family | Promise | Best evidence and uncertainty | Next decision |
|---|---:|---|---|
| Trusted-position peak annotation | 5 | It is the product outcome and can be measured without solving localization; no accepted end-to-end metric exists yet | Build label/identity sensitivity first; keep as an independent product track |
| Known-position horizon orientation | 4 | Historical PFM yaw success is 93–95%; RGB extraction varied from 40% to 87%, so observation quality remains cohort-dependent | Keep as the stable pose baseline and product fallback; benchmark downstream label pixels |
| Skyline-only translation / CMA-ES and other local optimizers | 0.5 | 0/3 and 0/5 position recovery; even source-PFM skyline scoring selects displaced basins | Stop active development; retain only frozen baselines/ablations |
| Dense skyline atlas | 3 as harness; 0.5 as selector | Reference-near grid cells exist, while blind top one is 0/6 across PFM/photo tracks; coverage mostly follows enumeration | Keep for recall@K, observability, and beam tests; freeze the skyline scorer |
| PFM geometry rerank | 4 diagnostic; 0 deployable | Its fixed fusion finds 18.5–52.0 m poses on three cases, but reference-position source depth encodes unavailable evidence | Keep as a perturbable oracle specification or training target only |
| RGB geometry verifier v1 | 1 selector; 3 interface | Top one is 0/3 and all cases abstain; its selected skyline defines the mask used by ridge and depth cues | Freeze v1; retain beam/abstention contracts; run one independent-mask kill test before deleting the scorer |
| Render/lift/PnP core | 4 local stage | Geometry works in controlled tests; the sole real learned-matcher case is truth-ambiguous and can false-accept a coherent basin | Keep and simplify; grade correspondence geography before PnP |
| SIFT | 1 | Positive mainly on identity/same-render plumbing; real cross-domain use usually abstains | Hermetic regression control only |
| RoMa / MINIMA | 2 | Coherent matches exist, but generic features converge to an ambiguous basin and share fitting/validation evidence | Keep as exact-pose controls; no more integration until correspondence truth passes |
| MatchAnything | 1.5 | No Peakle artifact; its authors identify aerial↔ground as difficult and recommend task adaptation | One external-worker control only; do not make it a new core dependency |
| LandscapeAR descriptor + PnP | 4 | Closest released mountain method; 30%≤100 m and 54%≤300 m, but its main renders start at truth and known FOV | Highest-priority task-B reproduction in an isolated environment |
| NormalLoc-style RGB↔normal training | 3.5 concept | Large gains over generic matchers on low-detail textureless CAD/urban models; no mountain test or official code | Promote normals to the exact-pose screen; train only after that signal exists |
| Segmentation / ridge extraction | 3.5 enabler; 1 direct translation | Large oracle/RGB orientation gap, but better boundaries do not create translation parallax | Build a manually labelled observation benchmark; defer LingBot Giant |
| Independent synthetic causal tests | 4 | Exact truth can isolate observability and terrain mismatch; sim-to-real remains the main uncertainty | Invest in a different forward renderer; do not use shared-renderer accuracy as a headline |
| Current `contourdb` / `global` | 0.5 | They search at most a massif-centred ring or a 20×20 grid in a supplied terrain window, not a published no-prior protocol | Remove from active method selection; preserve historical results |
| Baatz/Saurer skyline retrieval | 3 coarse retrieval | Reported 76–88% top-one within 1 km, with correction/circular-truth caveats and catastrophic outliers | Classical task-C baseline only |
| CrossLocate depth retrieval | 4 regional; 1 fine pose | 38.66% R@1 and 72.62% R@100 within 1 km over a held-out Alps region; known FOV and no metric refinement | Primary task-C baseline, after or separate from local capture |

The portfolio implies a hard keep/freeze/remove boundary:

- **Keep stable:** terrain rendering/raycasting, horizon orientation, matcher-worker contract, and
  render/lift/PnP geometry.
- **Freeze as diagnostics:** skyline atlas/scorer, PFM rerank, RGB verifier v1, and shared-renderer
  synthetic suite.
- **Remove from active selection:** skyline translation optimizers and the current `contourdb` and
  `global` labels. Do not delete potentially public compatibility code until its ownership/use is
  audited.
- **Invest next:** annotation sensitivity, independent observation truth, LandscapeAR reproduction,
  and RGB↔depth/normal representation tests.

## External baselines and relevant methods

The first goal is reproduction or fair adaptation, not inventing another score without a reference
point.

| Priority | Method | What it establishes for Peakle | Boundary |
|---|---|---|---|
| B0 | Unchanged GPS/EXIF/compass prior | Minimum prior-assisted baseline and regression detector | Does not use image evidence |
| B1 | Current known-position horizon alignment; Baboud et al. photo-to-terrain alignment | Orientation and annotation ceiling when position is supplied | Does not recover translation |
| B1a | EXIF plus Perspective Fields/GeoCalib gravity, FOV and distortion | Separates camera calibration from terrain localization and exposes calibration uncertainty | Natural mountain scenes may lack the vertical/architectural cues seen in training; must be benchmarked, not assumed |
| B2 | Baatz/Saurer global skyline indexing | Classical no-prior terrain-retrieval baseline; Baatz et al. rendered 3.5 million panoramas and put the top-ranked candidate within 1 km for 88% of 213 queries | Only 51% of query skylines were automatic; coarse position with DEM-corrected query truth, not modern metric pose |
| B3 | CrossLocate | Direct learned Alps photo↔render retrieval baseline; its uniform experiment searches millions of depth/semantic/silhouette renders and reports 38.66% recall@1 and 72.62% recall@100 within 1 km | Assumes known FOV scaling, primarily grades kilometre-scale position, and does not refine a metric camera pose |
| B4 | LandscapeAR | Most direct published photo-to-DEM refinement baseline: learned photo/render descriptors, 3D lifting, and PnP; reports 30% within 100 m and 54% within 300 m on 516 GeoPose3K images | Twelve 60° renders start at the GT position and query FOV is known; its separate noisy-offset plot has uneven bin counts. Reproduce both protocols before comparison |
| B5 | SIFT, RoMa, MINIMA, MatchAnything | Current generic correspondence controls and capture-range ablations | Not mountain/DEM-trained; correspondence consensus is not independent verification |
| B6 | MeshLoc / Render-and-Compare / PixLoc-style render-and-refine | Architecture patterns for lifting render matches and featuremetric local refinement, including MeshLoc's colorless-geometry experiments | Evaluated mainly on urban/indoor or aerial cross-view scenes; their render databases, SfM feature maps, textures, or coarse-seed assumptions differ from Peakle |
| B7 | NormalLoc-style geometry-specific training | Evidence that task-specific features can localize against textureless 3D geometry | Promising design pattern, not a mountain result |
| B8 | MegaLoc retrieval plus HLoc/ImLoc posed-reference localization | Strong branch for popular viewpoints with a reference-photo database | Cannot cover unseen terrain from DEM alone; ImLoc is currently an arXiv preprint |
| B9 | GeoCLIP, PIGEON, and HierLoc | Coarse global/region proposer for task C | Kilometre/entity-scale results are not comparable to 100 m mountain pose |
| B10 | FG² and VIRD | Local cross-view 3DoF estimator inside a supplied aerial/satellite tile | A supplied tile, urban/driving data, and 3DoF output do not solve global mountain retrieval or full metric pose |

Headline accuracy is never compared without this protocol card:

| Field | Required record |
|---|---|
| Search support | Country/region area, supplied tile or local seed radius |
| Initialization | Exact distribution of position, yaw, pitch/gravity, FOV, crop and height error |
| Render/database | Modality, terrain/mesh resolution, texture source, camera spacing and yaw sampling |
| Query information | Known/estimated FOV, intrinsics, gravity, compass, crop and panorama coverage |
| Truth | Survey/sensor/SfM/DEM-refined origin, uncertainty, and whether related evidence trained the estimator |
| Split | Geographic separation, duplicate control, locations and image count |
| Output and score | Pose components, top-K policy, threshold curves, seed delta, false accepts, compute and index size |

This exposes the main transfer gaps. LandscapeAR starts its headline run at the true position;
Render2Loc perturbs textured urban seeds by only ±5 m; PixLoc samples features from posed real
reference images/SfM points; NormalLoc renders dense meshes from reference-camera poses; and
classical skyline systems often use full panoramas, corrected skylines, or a 1 km threshold. They
are useful baselines or architecture evidence, not interchangeable SOTA numbers.

Primary references:

- [Baboud et al., Automatic Photo-to-Terrain Alignment for the Annotation of Mountain Pictures](https://resources.mpi-inf.mpg.de/photo-to-terrain/)
- [Perspective Fields](https://openaccess.thecvf.com/content/CVPR2023/html/Jin_Perspective_Fields_for_Single_Image_Camera_Calibration_CVPR_2023_paper.html) and [GeoCalib](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5636_ECCV_2024_paper.php)
- [Baatz et al., Large Scale Visual Geo-Localization of Images in Mountainous Terrain](https://people.inf.ethz.ch/marc.pollefeys/pubs/BaatzECCV12.pdf)
- [Saurer et al., Image Based Geo-localization in the Alps](https://doi.org/10.1007/s11263-015-0830-0)
- [CrossLocate](https://openaccess.thecvf.com/content/WACV2022/html/Tomesek_CrossLocate_Cross-Modal_Large-Scale_Visual_Geo-Localization_in_Natural_Environments_Using_Rendered_WACV_2022_paper.html)
- [LandscapeAR paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf) and [official implementation](https://github.com/brejchajan/LandscapeAR)
- [GeoPose3K dataset paper](https://cphoto.fit.vutbr.cz/geoPose3K/data/geoPose3K_submission.pdf)
- [MeshLoc](https://arxiv.org/abs/2207.10762), [PixLoc](https://openaccess.thecvf.com/content/CVPR2021/html/Sarlin_Back_to_the_Feature_Learning_Robust_Camera_Localization_From_Pixels_CVPR_2021_paper.html), and [Render-and-Compare / Render2Loc](https://arxiv.org/abs/2302.06287)
- [RoMa](https://openaccess.thecvf.com/content/CVPR2024/html/Edstedt_RoMa_Robust_Dense_Feature_Matching_CVPR_2024_paper.html), [MINIMA](https://arxiv.org/abs/2412.19412), and [MatchAnything](https://arxiv.org/abs/2501.07556)
- [NormalLoc](https://openaccess.thecvf.com/content/ICCV2025/html/Abe_NormalLoc_Visual_Localization_on_Textureless_3D_Models_using_Surface_Normals_ICCV_2025_paper.html)
- [FG²](https://openaccess.thecvf.com/content/CVPR2025/html/Xia_FG2_Fine-Grained_Cross-View_Localization_by_Fine-Grained_Feature_Matching_CVPR_2025_paper.html), [VIRD](https://openaccess.thecvf.com/content/CVPR2026/html/Park_VIRD_View-Invariant_Representation_through_Dual-Axis_Transformation_for_Cross-View_Pose_Estimation_CVPR_2026_paper.html), and [ImLoc](https://arxiv.org/abs/2601.04185)
- [Hierarchical Localization](https://openaccess.thecvf.com/content_CVPR_2019/html/Sarlin_From_Coarse_to_Fine_Robust_Hierarchical_Localization_at_Large_Scale_CVPR_2019_paper.html) and [MegaLoc](https://openaccess.thecvf.com/content/CVPR2025W/IMW/html/Berton_MegaLoc_One_Retrieval_to_Place_Them_All_CVPRW_2025_paper.html)
- [PIGEON](https://openaccess.thecvf.com/content/CVPR2024/html/Haas_PIGEON_Predicting_Image_Geolocations_CVPR_2024_paper.html), [GeoCLIP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html), and [HierLoc](https://arxiv.org/abs/2601.23064)
- [MeshVPR](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09464.pdf) and its [official synthetic-data generator](https://github.com/lolleko/mesh-data-synthesizer)
- [SUE/image-matching uncertainty](https://arxiv.org/abs/2404.00546) and [uncertainty-aware PnP](https://openaccess.thecvf.com/content/CVPR2021/html/Vakhitov_Uncertainty-Aware_Camera_Pose_Estimation_From_Points_and_Lines_CVPR_2021_paper.html)

LingBot-Vision is a plausible boundary-aware frozen backbone ablation for terrain/sky/occluder
features, not a ready segmentation system or a pose method. Its July 2026 technical report is very
recent, its released checkpoint is backbone-only, and its 1.1B-parameter Giant model is a poor first
systems baseline. Test the released Base/Large backbones with a trained mountain-specific probe and
compare against current extraction before paying the Giant compute cost. This is a proposed
fine-tuning experiment, not zero-shot use. See the [paper](https://arxiv.org/abs/2607.05247) and
[checkpoint card](https://huggingface.co/robbyant/lingbot-vision-vit-giant).

The extended literature review changes the implementation emphasis:

- **LandscapeAR is the immediate reproducible local baseline.** Its GeoPose3K protocol uses 12
  renders at 30° yaw increments from the reference position; adaptive-semihard descriptors report
  30/54/63/67/70% within 100/300/500/700/900 m and 39/60/65% within 1/3/5°. Its noisy-position plot
  suggests DEM mismatch dominates at low baselines, improvement becomes possible beyond roughly
  200 m, and the aggregate seed/refined curves cross near 700 m. These are plot-level observations,
  not success guarantees at a fixed 200 m prior.
- **CrossLocate supports depth-like learned retrieval, not depth as an oracle.** Over a geographically
  held-out Switzerland subset it reports 38.66/72.62% R@1/R@100 within 1 km and 50.10/85.08% within
  5 km. Removing known-FOV scaling drops the 1 km figures to 31.45/65.61%. Its depth representation
  beats horizon/semantic/silhouette lanes; simply combining modalities adds little.
- **NormalLoc makes camera-coordinate normals a concrete lane.** On the low-detail Notre Dame CAD-F
  model it reports 79.9/82.5/87.3% at 10/20/30% mean-DCRE thresholds versus
  5.3/12.2/21.2% for MeshLoc-LoFTR. The weakest geometry still degrades sharply, and the urban/CAD
  camera/database protocol does not establish mountain performance.
- **Generic matching should remain a control.** MINIMA's RGB↔depth/normal examples are derived from
  paired MegaDepth views rather than real photo↔DEM renders; MatchAnything explicitly treats
  aerial↔ground matching as difficult. Neither can fit and certify the same pose.
- **MeshVPR is a cheaper proposal-training pattern.** Its frozen-real-teacher/synthetic-student
  alignment recovers part of the synthetic-render retrieval gap and nearly saturates around 100k
  paired views, but texture/mesh quality materially affects recall. It is a regional-proposal option,
  not a metric refiner.
- **Uncertainty starts with simple baselines.** Score margin/entropy and spatial/yaw dispersion among
  top candidates should be calibrated before a learned verifier. Uncertainty-aware PnP can weight
  correspondences after their geography is correct; it cannot reject a coherent wrong basin.

## Target system architecture

The intended system is hierarchical and preserves multiple hypotheses until evidence not reused by
the fitter can separate them:

```text
image + declared prior/region
  -> coarse geographic retrieval (task C only)
  -> terrain/camera candidate proposal
  -> diverse candidate beam
  -> evidence heads not reused for fitting
       sky/terrain/occluder segmentation
       ridge and typed-outline likelihood
       photo-to-render appearance correspondences
       depth order / visibility / metric-range proxy
  -> domain-trained candidate ranking
  -> local render-match PnP or featuremetric refinement
  -> separately fitted verifier + calibrated abstention
  -> peak visibility and annotation
  -> pose/label uncertainty shown to the user
```

Do not let one selected skyline define every downstream mask. Do not let the same matcher both fit
and certify a pose. Keep candidate proposal, estimator ranking, numeric refinement, verification,
and annotation behind separate typed contracts so each can be replaced and benchmarked alone.

The likely novel contribution, if the baseline study supports it, is a mountain-specific
photo↔terrain-render representation that learns appearance, ridge geometry, visibility, and depth
order jointly while retaining explicit metric geometry for PnP/refinement. A large generic vision
backbone may initialize this representation; it is not itself the research result.

## Immediate experiment ladder: utility, observability, then capture

The old capture plan was still too large and conflated missing information with matcher failure.
Execute these gates in order. Stop a representation at the first failed gate; do not compensate with
more optimizer budget.

### Gate 1 — annotation sensitivity

Using trusted synthetic poses and the first gold-real pilot cases, project the same verified summits
under controlled pose perturbations. Sweep 0, 25, 50, 100, 200, and 400 m along-view, against-view,
cross-view, and oblique; signed yaw of 0°, ±0.5°, ±1°, ±2°, and ±5°; and representative FOV/crop and
camera-height errors. Report, by peak distance and direction:

- angular and pixel anchor displacement, identity/visibility changes, and label-layout changes;
- the largest perturbation that keeps median and p90 error inside a declared UI tolerance; and
- how much fixed-position yaw/FOV correction improves annotation relative to position correction.

This produces a distance-conditioned pose budget. The existing 25/100/300 m curves remain research
comparators, but the annotation-derived budget decides product usefulness.

### Gate 2 — information/observability ceiling

Before any matcher, score perfect observations from an independent query renderer against estimator
terrain candidates. Compare skyline, sky/terrain/occluder mask, typed outlines, absolute/log depth,
camera-coordinate normals, visibility/occlusion, and their simple non-learned combinations. Vary
translation direction, terrain resolution/bias/nodata, vegetation/snow/atmosphere, camera model, FOV,
and crop one factor at a time. Publish candidate score margins, rank of truth, aliases, and recall@K
at a fixed candidate/compute budget.

If a perfect modality cannot rank the correct basin, it is not trained or sent to PnP. Source PFM
may check the metric implementation but remains a labelled diagnostic oracle, never the independent
query generator.

### Gate 3 — exact-pose representation × matcher screen

At the exact render position and heading, compare these render lanes under the same image scale and
compute budget:

- orthophoto, hillshade, silhouette, absolute/log depth, and **camera-coordinate normals**;
- SIFT as the hermetic floor;
- the released LandscapeAR descriptor as the primary domain baseline;
- RoMa and MINIMA as generic controls; and
- MatchAnything only through the existing external-worker boundary unless it first passes cheaply.

Do not start with PnP accuracy. Grade ground-truth correspondence correctness, spatial coverage,
terrain-lift validity, ambiguity/mutual consistency, runtime, and memory. Then run unregularized PnP
only for pairs whose geographically correct matches cover enough independent image regions. Use an
independent synthetic renderer first, gold real second, and GeoPose only as weak-real evidence. A
pair advances only if held-out-location correspondence precision and coverage beat SIFT and its
unregularized pose is useful under Gate 1's annotation budget; inlier count alone cannot pass.

LandscapeAR is reproduced first in an isolated legacy environment on its published true-position,
known-FOV protocol. Only the descriptor/inference boundary is adapted into Peakle; its dependency
stack is not ported into the core package.

### Gate 4 — restricted capture surfaces

Advance at most the best two representation/matcher pairs. With yaw/FOV fixed, sweep the directional
position offsets above. Only after translation capture is mapped, add signed yaw, FOV/crop, height,
and a registered sparse interaction design. Report unregularized image evidence and prior-fused
results separately, always against the unchanged seed. Required outputs are correspondence truth,
lift validity, PnP/refinement capture, final annotation utility, false-basin rate, regressions,
abstentions, and risk/coverage by location.

The preregistered decisions are:

1. Which perfect observations contain enough information to distinguish translation?
2. Which exact-pose render representation produces geographically correct photo correspondences?
3. At what directional position error does each surviving pair cease to improve its seed?
4. How much do terrain/camera mismatch move that boundary?
5. Can evidence not reused for fitting reject a coherent wrong basin at an acceptable false-accept
   rate?

Do not run the frozen 32-mode verifier beam until one pair improves the unchanged seed over a useful,
annotation-derived capture region on held-out locations. That later immutable artifact must report
input beam recall, per-seed stages, selection regret, prior wins/regressions, and false accepts.

## Codebase consolidation plan

Peakle does not need a new web framework. FastAPI plus the current static application is adequate;
the research harness and contracts are the maintenance problem. Do not add MLflow or Weights &
Biases now. DVC becomes reasonable only as a transport/index for large content-addressed blobs once
a remote archive is configured; it must not own experiment semantics or replace the committed
ledger. The typed registry and artifact contract come first.

### Stable package boundaries

Refactor by extracting existing behaviour when the next experiment needs it, not by building a
parallel framework:

```text
peakle.io.artifacts            generic durable byte/file primitives; no truth semantics
peakle.localize.contracts     CandidatePose, CandidateBeam, PoseError,
                              StageDecision, EstimatorInputs; never TruthReference
peakle.localize.stages        pure proposal, scoring, matching, PnP and verification
peakle.localize.resources     corpus, terrain, imagery and model providers + provenance
peakle.research.contracts     TruthReference, EvaluationCase, ExperimentSpec
peakle.research.evaluation    truth-side metrics for new experiments only
peakle.scripts.research       thin CLI selecting a registered experiment
```

Do not retrofit all historical payloads. Preserve them by current hash plus a compact validated
projector. New experiment families should use one envelope once two of them genuinely share the
contract. Its deterministic content hash is computed from a canonical serialization of the
schema/truth contract/parents/resources/payload **before** inserting `artifact_sha256`; volatile run
metadata such as `created_at`, hostname, runtime, and archive locator is excluded. A separate
manifest hash identifies the complete run record:

```text
schema_id, schema_version, artifact_kind, artifact_sha256,
producer, created_at, truth_contract, parents, resources, payload
```

Dense candidate arrays live once in content-addressed NPZ/Parquet/JSONL blobs; derived stages refer
to parent SHA and candidate ID instead of copying a 111 MiB lattice into each result. The web
dashboard asks the artifact registry for a compact projector rather than hard-coding another schema
branch per experiment.

### Consolidation order

1. **Byte-I/O slice — complete.** Put immutable byte creation and directory fsync in
   `peakle.io.artifacts`, preserve private monkeypatch aliases, share the atlas encoder directly,
   and remove PFM/photo imports from another script. This is net −57 production lines with unchanged
   persisted bytes.
2. **Active-surface slice — complete.** Remove frozen skyline-translation methods from the advertised
   UI strategy catalog and default benchmark matrix without deleting explicit compatibility/replay
   entry points. Historical `global`/`contourdb` results remain “windowed legacy,” not country-scale
   localization.
3. **Gate-1 metric primitive — complete.** The small typed annotation-sensitivity evaluator reuses
   existing annotation/intrinsic models and reports visible identity plus anchor/angular/label
   displacement. Next add the pose-perturbation case generator and frozen artifact transaction; its
   truth-side types must not enter estimator code.
4. Let Gate 3 drive the next extraction: one exact-pose screen should reuse the current matcher
   worker and render/lift contracts. Extract only duplicated experiment transaction/metrics needed
   by a second new study; do not create another standalone 1,000-line CLI.
5. When the second new study exists, collapse shared pose error, evaluated candidate, top-K,
   canonical hashing, provenance, and freeze/reload code behind typed contracts. Keep legacy readers
   as frozen adapters rather than rewriting their schemas.
6. Split `strategy_bench.py`, `render_match_pnp.py`, `pnp.py`, and the large PnP test only along a
   responsibility boundary that removes duplication or reduces test coupling; file movement alone
   is not consolidation.
7. Audit external/public use before promoting or deleting the older `peakle.pipeline`,
   `peakle.matching`, and `localize.fg_check` paths. A previous first-class compatibility decision
   means test-only internal references are not sufficient evidence for deletion.
8. Configure content-addressed remote transport and dashboard projection after the new experiment
   contract is stable; neither blocks local causal tests.

### Guardrails

- A CLI should normally stay below 150 lines; stage modules below 500; functions below 80; test
  files below 800. Exceptions require a short architecture decision record.
- The second copy of artifact publication, provenance collection, truth whitelisting, or evaluation
  logic triggers extraction.
- Persisted boundaries may not be untyped `dict[str, Any]`; every schema registers a validator,
  projector, version policy, and truth contract.
- Estimator code may not import `peakle.research` truth/evaluation types. The estimator subprocess
  receives a serialized truth-free input and exits before the evaluator loads truth.
- Default tests never download models/data. Mark optional `realdata`, `model`, `gpu`, and `slow`
  lanes explicitly.
- Test the generic freeze/hash/reload/truth-firewall transaction once. CLI tests cover parsing and
  registry selection, not monkeypatched copies of the harness.
- Every experimental adapter declares hypothesis, owner/status, entry date, stop criteria, and
  removal/review date. At review it is promoted behind a stable interface or deleted; the compact
  result remains in this ledger.
- No benchmark result becomes a README headline until it passes a preregistered held-out gate.
- Each consolidation slice must delete more duplicated production code than it adds and preserve
  artifact bytes/validators. Splitting files without lower coupling or net deletion does not count.

## Roadmap and gates

Work is scheduled as four bounded tracks rather than one blocking sequence:

| Track | Current outcome | May progress without |
|---|---|---|
| A — annotation utility | Pose-to-label sensitivity and trusted-pose annotation benchmark | A metric localizer |
| B — prior-assisted localization | Observability, exact-pose representation screen, LandscapeAR, then restricted capture | Country-scale retrieval |
| C — data and truth | Gold-real pilot, independent synthetic cases, weak-real uncertainty | A new solver |
| D — maintainability | Net-deleting slices and frozen evidence validation | Remote artifact storage or legacy-schema migration |

### Phase 0A — freeze claims and remove byte-I/O duplication (complete)

The canonical document/JSON ledger are established. Common immutable byte writes and directory
fsync are centralized, persisted encodings are unchanged, script-to-script atlas schema imports are
removed, focused tests pass, and the first production slice is net −57 lines.

**Exit gate:** accepted claims have immutable identities and one interpretation; the changed runners
share low-level persistence without changing artifact bytes. Met.

### Phase 0B — active-method and experiment boundary (current)

The UI and default benchmark now advertise only the credible horizon/baseline surface while explicit
legacy strategy calls remain available for replay. The pure Gate-1 annotation evaluator is in place;
the pose-perturbation case generator, command, and frozen output remain. Preserve public
compatibility until external use is checked. Define the new freeze/reload boundary only for this new
experiment.

**Exit gate:** one command runs annotation sensitivity without estimator access to truth; default
method lists distinguish stable, experimental, and historical methods; this slice is net deleting or
has an explicit LOC budget paid by removed duplicate code.

### Phase 0C — reusable new-experiment contract

Use the exact-pose screen as the second consumer. Extract only the shared evaluator/artifact pieces,
validate/project canonical historical results through frozen adapters, and configure remote
content-addressed transport when available.

**Exit gate:** the two new studies share a typed truth firewall and projector; canonical evidence can
be validated with one command; no production module imports private script helpers. A remote archive
is a reproducibility gate for large accepted artifacts, not a blocker for local pilot work.

### Phase 1 — establish trustworthy truth and annotation utility

Run Gate 1, build gold-real v0, an independent synthetic generator, explicit uncertainty records for
weak-real data, and task-A annotation metrics. Stabilize GT↔DEM compatibility as a continuous
diagnostic. In parallel, benchmark manual masks and existing extractors before introducing another
large segmentation backbone.

**Exit gate:** location-held-out annotation and pose metrics run on gold real and independent
synthetic data; truth uncertainty is visible; shared-renderer and PFM ceilings cannot enter product
scores.

### Phase 2 — map local information and capture

Run the observability ceiling and exact-pose modality screen. Reproduce B0, horizon, and both
LandscapeAR protocols in an isolated environment; retain SIFT/RoMa/MINIMA as controls. Advance at
most two pairs into directional capture. CrossLocate/Baatz are not phase blockers because they solve
task C rather than this local task.

**Exit gate:** at least one image-based method improves the unchanged prior over a preregistered
capture region on held-out gold locations with a bounded false-accept rate. If none does, stop
end-to-end integration and focus on representation/data.

### Phase 3 — domain-specific ranking, refinement, and verification

Train or adapt photo↔terrain features; keep segmentation/ridge/depth/appearance heads independent;
calibrate a genuinely separate verifier; then evaluate the full candidate beam.

**Exit gate:** statistically credible gains over B0/LandscapeAR on location-held-out gold and weak
real sets, with reported risk/coverage and improved annotation—not merely lower estimator loss.

### Phase 4 — no-prior retrieval

First reproduce Baatz/Saurer and released CrossLocate inference at their honest 1 km protocols in
isolated environments. If those establish the reference point, build a hierarchical regional index
using learned depth/normal or MeshVPR-style render adaptation, then hand top regions to the validated
local pipeline.

**Exit gate:** registered top-K regional recall and end-to-end task-C pose/annotation metrics over a
declared geographic area and compute budget.

### Phase 5 — product annotation and uncertainty

Propagate pose, terrain, visibility, and identity uncertainty into labels; optimize layout; expose
abstention and evidence in the web app.

**Exit gate:** held-out users/images receive correct, well-placed mountain labels at a declared
coverage, and the UI never presents an uncalibrated pose as confirmed.

## Current decision queue

1. Finish Phase 0B: audit active method surfaces and implement the bounded annotation-sensitivity
   evaluator; do not add another solver family.
2. Specify the independent synthetic forward renderer and gold-real v0 protocol, including verified
   peaks and the empirical GPS/compass/FOV prior distribution.
3. Run perfect-observation surfaces, then the exact-pose render-modality screen.
4. Reproduce LandscapeAR's released descriptor on its published protocol; keep the legacy runtime
   outside the core environment.
5. Advance at most two pairs into restricted capture and use the measured, annotation-conditioned
   boundary to decide whether the frozen 32-mode beam is worth the full render/PnP cost.
6. Reproduce CrossLocate/Baatz only as task-C baselines once local capture is useful, or explicitly
   label an earlier run as a standalone retrieval study.

The question for every future proposal is: **which failed gate does this test, what observation
would falsify it, and what existing code will it replace?** If those answers are absent, the work
does not enter the roadmap.
