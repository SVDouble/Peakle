# Synthetic localization benchmark

Peakle's synthetic demo is useful for checking the older pinhole optimizer, but it is not an
end-to-end test of the production localization path. The demo renders a terrain mask, extracts its
exact boundary, and calls `PoseOptimizer`. GeoPose localization instead uses a cylindrical/tangent
crop, photo skyline extraction, a position/yaw atlas, optional render-match/PnP, and independent
candidate validation. A green demo therefore does not explain or exclude the roughly 200 m wrong
basins seen on real photos.

The test program uses exact synthetic truth to diagnose those stages independently. It does not
turn truth-rendered depth into a production input or count an oracle-selected candidate as a solver
win.

## Truth boundary

The diagram below is the target architecture. Each case starts with an authoritative terrain and
camera, creates observations from that reference, and supplies a separately declared estimator
terrain. Estimator archives contain observations, the supplied prior, candidate poses, and scores;
they are frozen and hashed before numeric pose errors are evaluated.

```text
authoritative terrain + camera
        |
        +-- independently rendered RGB / mask / metric depth
        |
        +-- estimator terrain: exact | coarse | biased | clipped | nodata
                         |
observation -> extraction -> proposal lattice -> ranking / verification -> local refinement
                         |              |                 |
                         +--------------+-----------------+-- freeze
                                                                |
                                                    post-hoc truth evaluation
```

Metric depth rendered at the reference pose is an analysis-only ceiling. It answers whether range
and internal structure can distinguish skyline-equivalent modes; it is not evidence available for
an arbitrary photograph. Exact masks likewise measure downstream solver capacity, while RGB-derived
skylines measure the production extraction path.

Two deliberately different controls are now implemented:

- `tests/integration/test_synthetic_production_pose_contract.py` exercises the real cyltan
  `HorizonProfile` -> `build_skyline_atlas` path, freezes it before truth evaluation, demonstrates a
  wrong blind basin with truth still in the proposal pool, and passes that frozen pool through the
  real PFM geometry reranker. It also exercises the RGB-only photo-geometry verifier over the complete
  frozen pool: an authoritative RGB channel encoding promotes the exact target from skyline rank two
  to verifier/beam rank one, while empty cues score the pool and abstain. Both positive controls use
  the same scene/raycaster and are explicitly identity/plumbing ceilings, not photo-model
  generalization evidence. The same test now validates the serialized verifier, preserves an
  abstaining three-mode beam without reranking, and sends all modes through the real orthophoto
  render -> batched SIFT -> lift -> nonlinear PnP interface. The useful exact position/yaw seed is
  deliberately last; it still solves after the first two modes, with both prior penalties disabled
  and with atlas crop pitch kept separate from physical render pitch.
- `python -m peakle.scripts.bench_synthetic_pipeline` is a custom pinhole, shared-mesh-renderer stage
  harness. It compares exact versus deterministically coarsened estimator terrain, controlled priors,
  exact-mask/color/haze skylines, and fixed depth/outline scores. It does **not** call the production
  atlas, raycaster, learned matcher, PnP or continuous refinement, and every method is marked
  non-production.

## Weaknesses and isolated contracts

| Stage | Current weakness | Verifiable synthetic contract |
|---|---|---|
| Camera model | Several tests generate truth with the same projector used by the solver; crop shift, physical roll and FOV can trade against pose. | Render with an independent camera path and sweep translation, FOV, crop shift and roll. The configured model must recover the injected nuisance; an incompatible model must abstain or record a failed camera-model gate. |
| Terrain/raycast | A correct feature can be absent because of range cutoff, resolution, nodata, seams or height datum. This is otherwise indistinguishable from a ranking failure. | The first analytic contract now covers diagonal terrain, an offset camera and explicit caps. The implicit range uses the per-camera farthest corner and includes its endpoint. Add convergence, nodata, seams and datum cases separately from pose error. |
| Photo extraction | Cloud, haze, snow, foreground structures and image wedges can form a continuous but false upper boundary. Current coverage checks alone do not establish agreement with terrain. | On clean RGB, require small p90 skyline error. On corrupted RGB, require the same tolerance or explicit rejection; never silently score an agreement-free candidate as usable evidence. |
| Proposal lattice | A search may miss truth because of radius, spacing, yaw discretization, map boundary or three-mode yaw pruning. | Report nearest full-lattice horizontal/yaw error and recall@K before evaluating any ranking score. On-grid truth should be exact; off-grid truth is bounded by grid and yaw quantization. |
| Skyline ranking | The atlas fits a global vertical shift and roll slope, caps residuals, trims high outliers and observes at most 192 columns. Distant skylines can be nearly invariant to hundreds of metres of translation. | Report blind top-1, truth rank, false-basin score margin and ambiguity margin. A distinctive case should select truth; a symmetric or distant case should keep truth in top-K and abstain rather than claim a unique pose. |
| Depth/typed geometry | The analysis reranker assumes a PFM depth convention, ignores depths below its configured threshold, subsamples, and fits one scale nuisance per candidate. | Use independently rendered metric depth to sweep 50/100/200/400 m translations, absolute versus scale-aligned depth, sampling, near-depth threshold and typed-outline ablations. Label every result oracle-only. |
| Candidate validation | Held-out matches can share the same systematic bias as the matches that generated a pose. They certify internal consistency, not geographic correctness. | Construct coherent wrong correspondences that pass match-family folds, then require a separately generated depth/outline/photo verifier to reject the pose. Record this as an expected diagnostic pass followed by independent rejection. |
| Priors | Prior penalties can keep a plausible solution near a wrong prior, while unregularized evidence can move farther into an ambiguous basin. | Sweep exact, 50, 100, 200 and 500 m priors in several directions. Report proposal coverage, evidence-only rank, prior-fused rank, improvement over prior, and unchanged-prior competitor. Ambiguous evidence must retain/abstain instead of confidently worsening the pose. |
| Refinement | Direct synthetic PnP tests show local capacity but do not prove that production proposals enter its basin. | Feed frozen atlas top-K modes through the production render/match/lift/PnP interface. Report continuous error only after proposal recall and candidate selection, including false accepts and abstention coverage. |

The round-zero orchestration part of the refinement contract is now implemented. A strict validator
rejects rehashed truth injection, the bridge forwards every frozen beam ID in order, all exact-heading
renders complete before one `match_many` call, and candidate-ID-derived RNG makes each PnP solve
independent of beam order. The current positive synthetic result is intentionally only an identity
ceiling: its input-beam oracle is already at 0 m / 0 deg and the post-PnP oracle remains exact. The
held-out candidate gate is disabled in that one test so capture and acceptance are not conflated.
Translation capture radius, learned cross-modal matching and independent acceptance remain separate
benchmark gates; this result must not be reported as real-photo localization accuracy.

## Deterministic case families

- **distinct multiscale**: asymmetric near, middle and far features. The correct basin should rank
  first with exact observations.
- **far ambiguous**: a distant smooth horizon that changes little over 200 m. Truth should remain
  reachable, but skyline-only ranking should declare ambiguity or retain the prior.
- **parallax occluders**: a skyline-ambiguous far ridge plus asymmetric structure at roughly
  0.5--3 km. Reference-rendered depth and internal outlines should promote the correct mode.
- **repeated symmetric**: two genuine position/yaw basins. The expected outcome is ambiguity, not an
  arbitrary successful label.
- **terrain mismatch**: coarser resolution, height bias, nodata and clipped range are applied only to
  estimator terrain. The compatibility gate should identify when metric verification is invalid.
- **evidence corruption**: missing skyline spans, cloud/haze edges and foreground structures. The
  extraction stage either remains within tolerance or rejects the evidence.
- **systematic matcher bias**: internally consistent correspondences support a wrong pose. Match
  folds may pass, but independent geographic evidence must reject it.

## Metrics and acceptance

Every case reports stage results rather than one end-to-end boolean:

1. extraction median/p90 error and valid-column coverage;
2. full-lattice nearest horizontal/yaw error and proposal recall@K;
3. skyline top-1, truth rank, score margin and ambiguity decision;
4. metric-depth, scale-aligned depth and internal-outline top-1/truth rank;
5. refinement position/yaw error, change from the prior and verifier decision; and
6. runtime, hypothesis count, false-accept count and risk/coverage after abstention.

Expected-success, expected-ambiguity and expected-rejection cases have different contracts. An
evaluation-only oracle is proposal coverage, never localization success. The immediate research
target is reliable selection of the correct 25--50 m basin on compatible inputs, followed by about
10 m continuous refinement; incompatible and unobservable cases should abstain.

## First exact/coarse upper-bound result

The immutable custom-pinhole artifact is
`local/output/20260714-synthetic-pinhole-stage-upper-bound-v2`. It contains five unique observations
(four rugged views and one radial ambiguity control), two reference-derived prior regimes and exact
versus factor-two-coarsened estimator terrain: 20 frozen candidate archives and 3,500 hypotheses.
The implementation paths were clean at revision `b3625f780fcb82617c0458d612b609ea72b6ebf1`; results
SHA-256 is `bf813d184d4b6c99be00ff6c0b75c3e8d4b7c5115478d932cd77638209f180a8`.

Proposal recall is 20/20 by construction because truth lies on the bounded grid. That is a plumbing
control, not a retrieval result. The useful comparison is the fixed ranking behavior:

| estimator terrain | evidence / score | raw top-1 target hit | decision accuracy | false accepts | median winner position |
|---|---|---:|---:|---:|---:|
| exact | exact-mask skyline | 8/10 | 8/10 | 0 | 0 m |
| coarse | exact-mask skyline | 3/10 | 2/10 | 1 | 100 m |
| exact | automatic color skyline | 2/10 | 4/10 | 3 | 141.4 m |
| coarse | automatic color skyline | 0/10 | 1/10 | 6 | 223.6 m |
| exact | reference-depth relative log range | 10/10 | 8/10 | 2 | 0 m |
| coarse | reference-depth relative log range | 3/10 | 5/10 | 2 | 100 m |
| coarse | reference-depth absolute log range | 6/10 | 7/10 | 3 | 0 m |
| coarse | reference-depth typed outlines | 2/10 | 0/10 | 7 | 161.8 m |

“Target hit” is an identity measurement, not benchmark success. In particular, exact self-rendered
depth chooses a truth-near candidate in the radial negative controls but should abstain, so its
decision accuracy is only 8/10. The injected haze track is rejected for all five unique observations
under a predeclared expected-rejection contract.

The result changes the next strategy in two ways. First, terrain fidelity must be a first-class gate:
coarsening alone turns a clean exact-mask skyline median from 0 m into 100 m. Second, absolute metric
range is materially more robust than scale-aligned range or typed outlines under this mismatch, so it
must remain a separately reported compatibility/score term rather than being discarded by a fitted
scale. The real [reference-depth atlas study](pose-atlas-study.md) subsequently shows that fixed
multi-cue fusion can outperform every individual component on compatible terrain. The production
verifier should therefore keep absolute and relative range, typed geometry and overlap as explicit
ablations, learn an ambiguity/compatibility gate, and only then tune their fusion. The same-renderer
numbers do not establish real-photo accuracy.

## Implementation order

1. Prove raycast range and depth conventions against independent synthetic truth.
2. Measure proposal coverage and complete position/yaw score surfaces on distinctive, ambiguous and
   parallax scenes.
3. Compare skyline-only ranking with metric depth, scale-aligned depth and internal typed geometry.
4. Add coherent-wrong matcher and photo-corruption rejection cases.
5. Run the frozen photo-verifier beam through continuous render-match/PnP refinement, keeping each
   render seed separate from the unchanged statistical prior and reporting proposal recall first.
6. Transfer only the strategies that pass these contracts to clean real GT/DEM compatibility
   strata, where label uncertainty remains a separate reported variable.

Step 5 now has a passing identity/plumbing ceiling and a complete-beam execution contract. It is not
complete as an algorithmic claim until offset seeds are swept and a frozen learned-matcher run reports
input recall, per-seed solve/validation outcomes, selection regret and false accepts.
