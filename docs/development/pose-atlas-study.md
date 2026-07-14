# High-compute pose-atlas study

This study asks a narrower question than the production solver: when a GeoPose photo starts from a
controlled position prior 200 m from the published reference, does the current terrain and image
evidence contain a substantially better horizontal position and yaw if we spend enough computation
to enumerate the local search space? The control also records a 15° yaw perturbation, but the atlas
does not consume it: every 1° yaw over the full 360° is scored.

It separates three outcomes that a local optimizer otherwise conflates:

1. the grid never generates a reference-near pose;
2. a reference-near pose exists, but the estimator score ranks another basin first; or
3. the blind estimator selects the reference-near pose.

## Experimental contract

`python -m peakle.scripts.bench_pose_atlas` builds a square, position-prior-centred terrain atlas.
The default diagnostic budget evaluates 441 positions at 50 m spacing and every 1° yaw. The square
has a ±500 m east/north half-width and therefore 707.1 m corner radius. A cached swissALTI 2 m patch
is directly sampled where it covers a ray; the 40 km regional backing terrain is GLO-30. At each
hypothesis the score fits an unbounded global vertical crop shift and a roll slope bounded to
±10°. These are effective crop nuisances; physical pitch and roll are not graded.

The artifact freezes and hashes the complete position × yaw score lattice before numeric reference
evaluation. It reports four deliberately different quantities:

- **blind winner** — the estimator's lowest score, and the only result it would actually return;
- **shortlist oracle** — the reference-best candidate among three yaw-separated modes kept per
  position (there is no global spatial non-maximum suppression);
- **full-lattice oracle** — the reference-best hypothesis among every scored position and yaw,
  selected only after freeze; and
- **reference-position probe** — an explicitly evaluation-only score at the published position.

The controlled position prior is reproducibly constructed from the published reference, then
frozen. The estimator receives that synthetic position, not the numeric reference. Its stored yaw,
pitch and altitude components do not constrain the atlas; FOV is fixed from sample metadata. The
`pfm_oracle` evidence is itself a depth render made at the reference pose and is therefore an
analysis-only geometry ceiling; `photo_auto` is the current end-to-end no-GT skyline extractor.
Neither oracle can replace the blind winner or count as a production success.

The run captures input hashes, hashes for a selected implementation-path list, the Git revision and
source diff for that list, and a compact cache inventory before the first sample. It rechecks all of
them before atomically writing `results.json`, `summary.md`, and `run.json`; this is not a complete
transitive-code attestation.

## Three-photo native-patch-assisted result

The completed control is
`local/output/20260714-three-photo-native-skyline-pose-atlas-v2`. It evaluated 158,760 frozen
position/yaw hypotheses per evidence track (441 positions × 360 yaws), plus a three-mode-per-position
shortlist, on three visually different manual `MAP_B/HEIGHT_A` photos. The run took 1,173.5 s and
produced results SHA-256
`5913a62ff65083457a3e7362edb1f228ffc1fef14c6cd5b0f213546e07f00e7d`, with the attested
implementation paths clean at revision `14a74aa64f1727e01d1349c8a4cd097b0acdae3d`.

| sample | evidence | blind winner (position / yaw) | full-lattice GT oracle | oracle's score rank | first shortlist K reaching 100 m / 5° | reference score minus blind |
|---|---|---:|---:|---:|---:|---:|
| IMG4948 | PFM oracle | 358.5 m / 1.9° | 18.5 m / 0.1° | 192 / 158,760 | 25 (rank 15: 52.5 m / 0.9°) | +1.20 px |
| IMG5143 | PFM oracle | 301.9 m / 1.1° | 19.8 m / 0.1° | 445 / 158,760 | 10 (rank 7: 89.9 m / 0.1°) | +2.88 px |
| IMG5145 | PFM oracle | 124.1 m / 0.7° | 14.9 m / 0.3° | 58 / 158,760 | 5 (rank 5: 49.5 m / 0.7°) | +1.10 px |
| IMG4948 | photo auto | 373.1 m / 1.9° | 18.5 m / 0.1° | 203 / 158,760 | 10 (rank 7: 68.5 m / 0.9°) | +0.72 px |
| IMG5143 | photo auto | 467.5 m / 2.1° | 19.8 m / 0.1° | 127 / 158,760 | 100 (rank 67: 33.9 m / 0.1°) | +2.94 px |
| IMG5145 | photo auto | 416.6 m / 123.3° | 14.9 m / 0.3° | 61,583 / 158,760 | 1,000 (rank 872: 54.7 m / 3.3°) | +10.69 px |

The estimator-selected blind result is therefore **0/3 successes** on both tracks. Against the
unchanged 200 m position prior, PFM ranking regresses two of three cases and raises median horizontal
error to 301.9 m; photo ranking regresses all three and raises it to 416.6 m. Conversely, the
evaluation-only full lattice contains a GT-selected 14.9–19.8 m horizontal/yaw hypothesis for every
photo. This is largely the expected coverage floor of a 50 m position grid. It is not an
evidence-selected, continuously refined, or full-pose localization result; altitude, pitch and roll
are ungraded.

The decisive control is the last column: even a probe at the exact reference east/north loses to a
displaced blind winner under the current score on all three reference-generated PFM tracks. The
probe still uses DEM ground plus a 2.5 m eye height, the best 1°-grid yaw, and fitted crop nuisances;
it is not the exact full reference pose. This unregularized skyline score is therefore selecting the
wrong geographic basin on these three controls; more iterations of that same score are not the
answer. This identified an independent range/occlusion/typed-ridge verifier over the stored atlas
modes as the next gate. A target-successful PFM mode is present by top 5–25, which establishes
shortlist coverage; it does not prove that a verifier restricted to 25 modes succeeds. The completed
verifier below conservatively reranks all 1,323 stored modes. Automatic IMG5145 reaches the target
only by top 1,000; its unusable PFM edge support indicates that a rejection/fallback criterion is
needed, but the current extractor did not reject it.

## Reference-depth geometry reranking ceiling

The next gate has now been run over all 1,323 PFM-atlas shortlist candidates per photo. The immutable
artifact is `local/output/20260714-three-photo-pfm-geometry-rerank-v1`; its frozen estimator archive
SHA-256 is `dc39392cd9a74470cbac2d351f15d089dae0d83f752cc6bc592ff171d09c28c3` and results
SHA-256 is `0f246caaaf2858d5333842ee42997ef95f309369e186f8a27d5393c466e64c4d`.

The verifier renders candidate depth using the exact atlas crop shift/roll nuisance, converts DEM
horizontal range to camera-ray range, and freezes four normalized terms: original skyline score,
typed internal outlines, median-scale-aligned log-depth shape, and valid-depth overlap. Fixed weights
were declared before numeric pose evaluation. The source PFM is itself rendered at the reference
pose, so this remains an analysis-only geometry ceiling, not a deployable photo estimator.

| sample | skyline-only winner | fixed geometry fusion | pool GT oracle | fusion candidate's original skyline rank |
|---|---:|---:|---:|---:|
| IMG4948 | 358.5 m / 1.9° | 18.5 m / 0.9° | 18.5 m / 0.9° | 32 |
| IMG5143 | 301.9 m / 1.1° | 52.0 m / 0.9° | 19.8 m / 0.1° | 199 |
| IMG5145 | 124.1 m / 0.7° | 35.4 m / 0.7° | 14.9 m / 0.7° | 19 |

Fusion reaches the 100 m / 5° target on all three controls, with 35.4 m median horizontal error.
No single stored component does: typed outlines reach 2/3, depth overlap 1/3, and centred depth shape
and skyline reach 0/3. The result therefore supports multi-cue verification rather than replacing one
scalar skyline score with another scalar depth score. It also establishes a concrete ceiling between
the 25--50 m target basin and the 14.9--19.8 m grid oracle, while leaving altitude, pitch and roll
ungraded.

The same frozen verifier was then applied to the automatic-photo atlas pools, without changing its
weights. That artifact is
`local/output/20260714-three-photo-photo-candidates-pfm-geometry-rerank-v1`, with estimator archive
SHA-256 `be8f82ae09e0f92883ec36e1857ccbb19837c54e6511893ba8f893da19c8e86a` and results
SHA-256 `9d02a702e6b0f779c86406b64b53a95d5ecb89488b2938ab38032dae1f134c5e`.

| sample | photo-skyline winner | reference-depth fusion over photo pool | first target-successful fusion rank |
|---|---:|---:|---:|
| IMG4948 | 373.1 m / 1.9° | 18.5 m / 0.9° | 1 (original photo rank 20) |
| IMG5143 | 467.5 m / 2.1° | 92.2 m / 0.1° | 1 (original photo rank 114) |
| IMG5145 | 416.6 m / 123.3° | 385.3 m / 1.3° | 31 (original photo rank 1,221) |

This transfer reaches 2/3 at top one and reduces median horizontal error to 92.2 m. IMG5145 remains
a false top-one basin, although the verifier corrects its gross yaw and promotes a target-successful
mode from photo rank 1,221 to geometry rank 31. The production strategy should therefore preserve a
small multi-hypothesis beam (at least 32 on this control), apply genuinely photo-observable evidence
or render-match refinement, and abstain when no independent cue separates the remaining modes.
Geometry fusion fixes ranking when the automatic pool is adequate; it does not repair corrupt image
evidence by itself.

The truth boundary was audited before this run. Signed controlled-prior perturbations, which can
reconstruct the reference pose, are stripped from estimator inputs; only the non-geometric cache
replicate crosses. `rerank.json` is file- and directory-fsynced before truth reaches evaluation, and
the source atlas results/run, PFM inputs, terrain cache and selected implementation paths are checked
again before atomic publication.

## Photo-observable geometry verifier

The deployable transfer was then measured without source PFM. The immutable artifact is
`local/output/20260714-three-photo-photo-geometry-verifier-v1`; its frozen estimator archive SHA-256
is `f2b5e62508c8601977e9f27d2740331a7549ceb204cc80ee7e439e06b80a8689`, results SHA-256 is
`a55235964989131d00b515eceffb4b5b98341e30940f4e39d41c6a81a33c7f9b`, and the selected
implementation paths were clean at revision `a18b473852d7f309261a863f33bb4f9f91c854f8`.

DexiNed internal edges and Depth-Anything V2 Small relative depth were extracted offline from RGB.
Every one of the 1,323 frozen `photo_auto` candidates was rendered before a 32-basin beam was chosen.
Fixed fusion weights were skyline 0.20, symmetric internal outline 0.35, ordinal depth 0.35 and
terrain overlap 0.10. Leave-one-column-block-out stability excludes the frozen full-image skyline
score. The write-once `verifier.json` was file- and directory-fsynced before numeric truth and the
evaluation-only GT/DEM compatibility bucket were reloaded.

| sample | skyline-only winner | photo-geometry winner | first target verifier rank | first target beam rank | decision |
|---|---:|---:|---:|---:|---|
| IMG4948 | 373.1 m / 1.9° | 344.8 m / 0.9° | 30 | 12 | abstain |
| IMG5143 | 467.5 m / 2.1° | 359.7 m / 1.1° | 32 | 12 | abstain |
| IMG5145 | 416.6 m / 123.3° | 523.9 m / 106.7° | 439 | absent | abstain |

This fixed photo score is **0/3 at top one** and must not replace the prior. Its useful result is the
decision behavior: all three low-margin, fold-unstable cue conflicts abstain, producing zero false
accepts. The diverse beam preserves a target-successful basin for IMG4948 and IMG5143, but not
IMG5145. Its second beam pose is nevertheless in the correct yaw basin at 154.0 m / 0.3°, so a local
refiner can measure capture from outside the formal target. A 32-seed render/match/PnP benchmark has
only 2/3 input target recall and must report that before refinement; it cannot claim three-photo
proposal success.

The failure is not just an unfortunate weight vector. The blue/bright automatic skyline candidates
have zero agreement on all three photos, yet their high coverage passes the source extractor's
coverage-only gate. The selected skyline then defines the terrain mask for learned ridges, monocular
depth and overlap, so these terms are not independent when the mask is wrong. Fusion score is strongly
correlated with common-terrain coverage (Spearman -0.70, -0.87 and -0.94), and overlap loss is almost
perfectly anti-correlated with it. IMG5145's wrong winner beats its 154 m rival only through overlap;
the rival is better on skyline, outline and ordinal depth. The abstention cue vetoes catch exactly
that conflict.

Raw component argmins are also diagnostic-only: outline winners for IMG4948/IMG5145 contain just
15/13 candidate-outline pixels, and IMG5145's raw skyline winner renders no valid terrain. Future
component reporting must apply eligibility/support gates. Future proposal runs should keep separate
blue, bright and fused-colour skyline hypotheses when they disagree, and use fixed cue-specialist
lanes alongside the diverse fusion beam. IMG5145 needs that stronger proposer or an adaptive larger
beam before expensive refinement can be expected to reach a target pose.

All three controls are evaluation-only `MAP_B/HEIGHT_A`: their published reference and the supplied
terrain agree reasonably well and camera clearance is plausible. This removes one obvious label/DEM
failure explanation, but it does not calibrate compatibility stratification because the current set
contains only one bucket. The benchmark runner reports success, selection, abstention, false accepts,
beam recall and component winners by fit bucket so a larger manually verified corpus can do so without
changing the estimator.

## What the literature says is realistic

[LandscapeAR](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123740290.pdf) is still the
closest published photo-to-DEM system. Its better reported variant places 30% of GeoPose3K within
100 m and 54% within 300 m. The paper also finds that short-baseline refinement is dominated by
render/DEM mismatch and that useful GPS improvement concentrates roughly in the 200–700 m regime.
That makes Peakle's 200 m failure plausible, but not acceptable as the project target.

The most useful implementable strategies are:

| priority | strategy | role in Peakle | reason / limitation |
|---:|---|---|---|
| 1 | Dense skyline, range, occlusion and typed-ridge atlas | Prior-guided proposal and independent candidate verification | Reuses the current ray caster and typed depth outlines. A soft directional-Chamfer or distance-transform score can retain multiple spatial modes. Skyline alone is expected to be ambiguous for distant smooth terrain. |
| 2 | Multi-seed render → match → lift → PnP, repeated two or three times | Refine top atlas modes | [Render2Loc](https://arxiv.org/abs/2302.06287) shows that seed augmentation and iterative render-and-compare refinement help with noisy priors; [MeshLoc](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136820573.pdf) validates mesh render → 2D/3D match → robust pose as a general architecture. Repetition cannot repair a systematically wrong correspondence basin without an independent verifier. |
| 3 | Peakle-specific photo/render descriptor or displacement-field training | Rank and move among nearby modes | LandscapeAR's official [training code](https://github.com/brejchajan/LandscapeAR) is the direct precedent. The newer [GeoFlow preprint](https://arxiv.org/pdf/2603.21943) reports quick saturation from additional seeds/iterations, so training domain and score quality matter more than an unlimited loop count. Geographic train/test separation is mandatory. |
| 4 | Featuremetric or differentiable local refinement | Continuous 6-DoF/FOV polish after entering the right basin | [PixLoc](https://openaccess.thecvf.com/content/CVPR2021/papers/Sarlin_Back_to_the_Feature_Learning_Robust_Camera_Localization_From_Pixels_CVPR_2021_paper.pdf) optimizes pose through multiscale learned features. It is a local refiner, not a 200 m initializer. A geometry/feature loss is preferable to raw RGB against an orthophoto render. |
| 5 | Domain-specific matcher diversity and multi-model robust estimation | Produce independent pose clusters | MINIMA, RoMa and a measured MatchAnything trial can expose different basins. MAGSAC++/Progressive-X-style scoring can retain multiple models, but robust estimation cannot fix consistently wrong cross-domain correspondences. |
| 6 | Precomputed hierarchical 360° skyline index | No-position-prior regional retrieval | [Baatz et al.](https://people.inf.ethz.ch/marc.pollefeys/pubs/BaatzECCV12.pdf) indexed 3.5 million Swiss panoramas on an approximately 111 × 115 m grid. The [later ETH system](https://people.inf.ethz.ch/pomarc/pubs/SaurerIJCV15.pdf) reports 88% within 1 km. This is a proposal mechanism; local atlas and metric verification still follow. |
| 7 | Cross-view ground/aerial network | Optional top-K region proposal | [Unconstrained Cross-View Pose Estimation](https://openaccess.thecvf.com/content/WACV2026/papers/Wollam_Towards_Unconstrained_Cross-View_Pose_Estimation_WACV_2026_paper.pdf) handles unknown input FOV/pitch/roll, but its low metre errors are measured inside supplied 70–100 m urban aerial tiles. [VIRD](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_VIRD_View-Invariant_Representation_through_Dual-Axis_Transformation_for_Cross-View_Pose_Estimation_CVPR_2026_paper.pdf) explicitly lists mountainous elevation and nonzero pitch/roll as a limitation. Use either only as a mountain-trained proposer, never the final DEM-verified pose. |
| 8 | Cross3R-style DEM bridge views | Longer-term 6-DoF research branch | The [Cross3R preprint](https://arxiv.org/pdf/2605.07978) reaches a 10.52 m / 3.51° zero-shot KITTI median with satellite+ground input under random 360° rotation. Replacing its optional UAV bridge with oblique DEM renders is promising, but still requires an authoritative DEM verifier. |

The low errors in current cross-view papers are not comparable to a 500 m mountain search. For
example, WACV 2026 reports a 1.87 m / 2.11° cross-area median on VIGOR positive pairs, while VIRD
reports 5.41 m / 1.87° on KITTI and 1.55 m / 1.17° on VIGOR. Those experiments start with a matching
70–100 m aerial tile and road-domain training; they are evidence for a learned local proposer, not a
claim that Peakle should already reach 2 m from an Alpine photo.

## Reproduction

Use a new output directory for every run:

```bash
python -m peakle.scripts.bench_pose_atlas \
  --samples eth_ch1_IMG_4948_01024,eth_ch1_IMG_5143_01024,eth_ch1_IMG_5145_01024 \
  --tracks pfm_oracle,photo_auto \
  --seed 20260713 \
  --perturbation standard \
  --replicate 0 \
  --radius-m 500 \
  --spacing-m 50 \
  --yaw-step-deg 1 \
  --ray-step-m 10 \
  --output local/output/<new-pose-atlas-run>

python -m peakle.scripts.bench_pose_atlas_pfm_geometry \
  --atlas local/output/<pose-atlas-run>/results.json \
  --candidate-track pfm_oracle \
  --subsample 4 \
  --output local/output/<new-pfm-geometry-run>

python -m peakle.scripts.bench_pose_atlas_photo_geometry \
  --atlas local/output/<pose-atlas-run>/results.json \
  --samples eth_ch1_IMG_4948_01024,eth_ch1_IMG_5143_01024,eth_ch1_IMG_5145_01024 \
  --subsample 4 \
  --dexined-checkpoint /path/to/DexiNed_BIPED_10.pth \
  --depth-model-dir /path/to/Depth-Anything-V2-Small-hf \
  --device cuda \
  --output local/output/<new-photo-geometry-run>
```

## Decision rule after the atlas

- Full-lattice PFM oracle near the reference, blind PFM winner wrong: improve geometric
  ranking/verification; more local-optimizer iterations are not the answer.
- Blind PFM winner near the reference, blind photo winner wrong: improve photo segmentation,
  boundary likelihood, occluder masks and internal-ridge extraction.
- Photo atlas near the reference, learned PnP wrong: fix correspondence selection or candidate
  verification.
- No reference-near lattice hypothesis: widen/fix the proposal region, camera model, FOV/roll
  treatment or terrain provisioning.
- Even a reference-centred PFM score prefers a broad displaced basin: record large per-image
  horizontal uncertainty; the visible terrain does not support a universal sub-25 m claim.

The near-term target should therefore be **reliable entry into a 25–50 m correct basin on compatible
photos**, followed by 10 m/continuous refinement. It should not be a universal 2 m target borrowed
from urban cross-view benchmarks.
