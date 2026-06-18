# Roadmap and Risks

## Milestones

### M0: Planning

- Document architecture, interfaces, and algorithm choices.
- Keep assumptions explicit, especially coordinate simplifications.

### M1: Package Skeleton

- Add `pyproject.toml` with Python 3.14 and `uv` workflow.
- Add `src/peakle` package structure.
- Add Pydantic domain models.
- Add `AppSettings` with `pydantic-settings`.
- Add lint, type-check, and test commands.
- Add static Three.js viewer assets and a stdlib web server command.

### M2: Synthetic Terrain and Rendering

- Implement deterministic terrain generation.
- Detect and name synthetic peaks.
- Implement camera projection.
- Render a synthetic image and terrain mask.
- Extract a skyline contour from the mask.

### M3: Pose Estimation

- Implement candidate skyline rendering for optimization.
- Add contour distance scoring.
- Add bounded coarse-to-fine pose search.
- Verify recovery against synthetic true pose.

### M4: Annotation Overlay

- Project peaks through the estimated pose.
- Filter visible and relevant peaks.
- Draw deterministic labels onto the render.
- Export per-view `views/view-XX/annotations.json` and `views/view-XX/annotated.png`.
- Export `viewer-data.json` and static browser assets.

### M5: Real-World Preparation

- Replace synthetic contour extraction with an image-based extractor.
- Add real DEM loading behind the `TerrainMap` interface.
- Add real coordinate reference system support.
- Add camera metadata ingestion where available.

## Test Strategy

Unit tests:

- Pydantic validation rejects invalid coordinates, camera intrinsics, and grid
  sizes.
- Local geodetic conversion round-trips within the expected small-area error.
- Camera projection handles points in front of, behind, and outside the image.
- Terrain generation is deterministic for a seed.
- Peak detection returns stable ids and sorted prominence.
- Contour extraction returns one skyline point per valid terrain column.

Integration tests:

- End-to-end demo produces all expected artifacts.
- Estimated pose improves over the noisy prior.
- Pose error remains below configured synthetic thresholds.
- Annotated image contains at least one accepted peak label for a known seed.

Regression tests:

- Fixed seeds should keep reference metrics within tolerance.
- Golden artifacts should be avoided at first unless image output stabilizes.
  Prefer metric assertions over brittle pixel-perfect tests.

## Known Risks

### Coordinate Ambiguity

The MVP uses Earth-like latitude and longitude but simplified local geometry.
That is acceptable for synthetic terrain over a small area, but real maps need
proper CRS handling.

Mitigation: keep coordinate conversion isolated in `domain.coordinates`.

### Pose Ambiguity

Different nearby camera poses can produce similar skylines, especially if the
terrain has repeated ridge shapes.

Mitigation: use position priors, bounded search, multi-start optimization, and
diagnostic residuals. Do not report a pose without fit quality metrics.

### Renderer and Optimizer Coupling

The optimizer needs fast predicted contours. The pretty renderer may be too slow
for repeated scoring.

Mitigation: share camera math, but allow a fast skyline renderer separate from
the full image renderer.

### Contour Overfitting

Using the synthetic mask makes contour extraction artificially clean.

Mitigation: add configurable contour noise, missing segments, and smoothing
once the clean pipeline works. Keep the clean baseline for regression tests.

### Label Legibility

Peak labels can overlap or attach to occluded peaks.

Mitigation: label placement should be deterministic, sorted by prominence, and
allowed to skip lower-priority labels.

## Open Decisions

- Whether the first renderer should be a custom NumPy rasterizer or a simpler
  horizon profile renderer with image synthesis around it.
- Whether the CLI should use Typer, argparse, or a very small custom wrapper.
  `argparse` is enough for the MVP and avoids an extra dependency.
- Whether dense arrays should be stored only as `.npz` while all metadata stays
  in JSON. This is the recommended split.
- How noisy the initial pose prior should be for the default demo seed.
- Whether roll should be part of the first optimization milestone or fixed to
  zero until yaw and pitch are reliable.

## Implementation Standards

- Public modules, classes, and functions use Google-style docstrings.
- Functions should take explicit domain objects rather than loose dictionaries.
- Serialization should happen at IO boundaries, not inside core algorithms.
- Numeric code should validate shapes at boundaries and then rely on typed
  arrays internally.
- Randomness must be seedable and should not use global random state.
- The CLI should print artifact paths and summary metrics, not raw debug dumps.
