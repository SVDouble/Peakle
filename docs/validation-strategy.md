# Validation strategy: making pose results trustworthy

*2026-07-04.  Motivating incident: repeated cycles of "the overlay looks aligned" / "chamfer is
low, everything works" followed by the user checking and finding 0/5 skylines correct.  This
document fixes the methodology so a result can be called good only when a measurement says so.*

## Diagnosis — why results kept looking good while being wrong

1. **The residual was treated as the score.**  A 2D chamfer between an extracted skyline and a
   DEM render is *alias-prone*: on hazy or repetitive horizons a wrong yaw can fit within a few
   px of the best.  Low chamfer ≠ correct pose.  Measured example (GeoPose3K
   `eth_ch1_1332166`): wrong yaw at +166° error scores chamfer 17 px — *lower* than several
   correct solves on other samples.
2. **Validation was anecdotal, N≈1.**  The acceptance harness tested one photo (Matterhorn) with
   thresholds calibrated on that same photo.  Every method change was judged by eyeballing a
   handful of overlays — which is exactly how the over-claiming happened.
3. **The validated code lived in gitignored scripts.**  The correct ray-cast renderer and the
   pitch-decoupled solver existed only under `local/` (one-off scripts, each with its own copy
   of extraction/chamfer/thresholds); the packaged `fast_skyline`/`skyline_profile` renderers —
   still imported by some scripts — are known-broken (combing / all-zero).
4. **Extraction quality was unmeasured at solve time.**  Garbage outlines (black crop borders,
   haze, clouds) went straight into the solver, which then happily returned *some* pose.
5. **Known physical failure causes kept recurring** because nothing tested for them:
   DEM extent smaller than the true visible horizon (40 km extent truncates Alpine views that
   reach 45+ km → systematic wrong-yaw solves), camera placed below DEM ground, SRTM voids on
   sharp summits.

## The fix — three pillars

### 1. Ground-truth benchmark (the truth harness)

`scripts/fetch_geopose.py` streams a subset of **GeoPose3K** (3000+ Alpine photos with GT pose,
GT-rendered depth, cylindrical crops).  `scripts/bench_geopose.py` scores every sample on two
tracks:

- **ORACLE track** — observed skyline taken from the GT depth map.  Isolates
  solver + DEM + conventions.  A failure here is never extraction's fault.
- **EXTRACTED track** — skyline extracted from the photo.  End-to-end.  The oracle↔extracted
  gap attributes every failure to either extraction or solving.

Success = **pose error against decoded GT** (|yaw error| ≤ 5°), never a residual.  Results are
split by GeoPose3K's MANUAL/AUTO flag (AUTO poses are themselves unverified; two of our first
four samples were AUTO).  Pitch error is recorded but informational: the cylindrical crops are
not vertically centred on the optical axis, so decoded GT pitch is not comparable in crop
coordinates (verified: a solve with chamfer 3.6 px and yaw error 0.3° showed a −9° "pitch
error" — that offset is the crop, not the solver).

Known bias, accepted for now: streaming a .tar.gz means samples come from the archive head
(Swiss Alps, alphabetical).  Grow the subset before claiming generality.

### 2. Solver that reports its own ambiguity

`peakle.localize` (promoted, unit-tested package code — `raycast.py`, `solve.py`, `extract.py`,
`copdem.py`, `geopose.py`):

- correct **ray-cast horizon renderer** (pinhole + cylindrical);
- the 360° horizon profile `el(az)` is computed **once per position**; every (yaw, pitch, fov)
  hypothesis is a resampling, so a **dense full-360° yaw profile** comes free with every solve;
- symmetric, distance-capped curve chamfer (capping stops single garbage columns from
  dominating);
- pitch decoupled as a vertical shift (exact for cylindrical, first-order for pinhole);
- every solve returns diagnostics: final chamfer, **basin width** (how many yaws fit almost as
  well), **alias ratio** (the fine-polished best rival *outside* the winning basin vs the
  winner), skyline coverage, and for extraction the **two-detector agreement** score.

Unit tests include the *honesty gate*: on a radially symmetric terrain (skyline identical at
every yaw) the solver must NOT report CONFIRMED, whatever the residual.

### 3. Calibrated verdict instead of vibes

Every solve carries `verdict ∈ {CONFIRMED, AMBIGUOUS, REJECTED}` computed from the diagnostics
**without ground truth** (so it works on any user photo).  The thresholds are not hand-waved:
`scripts/calibrate_verdict.py` measures, on the benchmark, which diagnostics actually separate
correct from wrong solves (per-feature AUC) and picks the gate that maximises recall at ~100%
precision.  The operating point is deliberately conservative: **a CONFIRMED that is wrong is a
methodology bug; an AMBIGUOUS that was right is merely lost recall.**

## Definition of done for any future method change

1. `pytest tests/unit/test_localize.py` green (round-trip + honesty gate).
2. `scripts/bench_geopose.py` on the standard subset: report oracle and extracted success rates
   *and* verdict precision.  A change that raises success but breaks verdict precision is
   rejected.
3. No claims from overlays alone.  Overlays are for diagnosis, numbers are for verdicts.

## Roadmap (highest-leverage first)

1. **Extraction is the end-to-end bottleneck** (oracle ≫ extracted success).  Wire the SAM3 sky
   mask as the extraction backend (`--extractor sam3`, already scaffolded), measure the delta on
   the benchmark.  The color extractor's two-detector agreement score gates garbage inputs.
2. **Grow the benchmark**: more GeoPose3K samples (incl. non-Swiss regions from deeper in the
   archive), plus the foto-webcam.eu fixed cameras with documented bearings as a second,
   independent GT source (different projection: pinhole).
3. **FOV search track**: benchmark currently uses GT FOV (the product analogue: EXIF).  Add an
   unknown-FOV track to quantify the extra ambiguity before promising it in the product.
4. **Full-silhouette matching** (internal ridges, occlusion edges) only *after* the skyline-only
   baseline is measured — it must prove its added value on the benchmark, not in anecdotes.
5. **Near-field jagged scenes** (the original 5 selfies) stay out of scope until sub-30 m DEM
   (swissALTI3D / Bavaria LiDAR) is wired into the benchmark path; SRTM/Copernicus physically
   cannot represent those ridges (measured ~19 px irreducible gap).

## Current measured status (2026-07-05, 60-sample run, colour extractor)

Live numbers live in `local/output/*-geopose-bench/summary.md`; snapshot:

- **Oracle track: 41/60 (68%)** — MANUAL 22/35, AUTO 19/25.  Correct solves are typically
  sub-degree in yaw.  Failures are dominated by genuinely ambiguous horizons (oracle alias
  ratios near 1.0) plus a tail of suspect AUTO ground truth.
- **Extracted track: 26/60 (43%)** — extraction remains the binding constraint (median
  extraction error vs the GT skyline spans 2 px to 237 px across photos).
- **Verdict (calibrated gate): 40 CONFIRMED, 0 wrong** — recall ~60-68% of correct solves.
  Feature separations (AUC): alias ratio 0.94, SNR 0.87, chamfer 0.85, basin width 0.78,
  detector agreement 0.71, coverage 0.57 (useless).  Chamfer ranges of correct and wrong solves
  OVERLAP — the residual alone can never be the success signal.
- Corpus on disk: 672 complete GeoPose3K samples (130 MANUAL) via `scripts/fetch_geopose.py`;
  the standard run is pinned in `scripts/geopose_manifest_60.txt`.
- **GT cleanliness is itself measured** (bench records `gt_consistency_px`, camera-below-ground,
  solver-consensus-vs-label; report via `scripts/gt_report.py` + `build_gt_report_html.py`):
  the core is clean (median GT↔DEM skyline consistency 8.3 px) but 16/60 samples carry flags,
  including 7 where both solver tracks agree with each other 20–160° away from the GT label
  (4 of them MANUAL) — the raw success rates understate the solver.  On MANUAL samples passing
  the cleanliness checks the oracle track scores substantially higher than the raw 63%.
