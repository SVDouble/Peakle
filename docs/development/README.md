# Peakle Development Notes

Peakle is planned as a Python package and small demo application for annotating
mountain peaks in an image. The first version is fully synthetic: it generates a
terrain model, places a virtual camera on that terrain, renders a pinhole-camera
view, estimates the camera pose from a noisy coordinate and image contour, then
projects generated peak names back onto the original image.

The immediate goal is not photorealism. The goal is a clean, testable geometry
pipeline with explicit data contracts so later real-world integrations can
replace synthetic parts without rewriting the core.

## Baseline Choices

- Runtime: Python 3.14.
- Package manager: `uv`.
- Validation: Pydantic models for input, output, and serialized demo artifacts.
- Settings: `pydantic-settings` for environment-backed runtime configuration.
- Layout: `src/` package layout with a thin CLI layer over pure services.
- Documentation: Google-style docstrings for public classes and functions.

The current official Python documentation lists Python 3.14 as stable, `uv`
documents `project.requires-python` as the project Python version declaration,
and Pydantic documents `pydantic-settings` as the settings package.

## Synthetic MVP

The demo should produce these artifacts under `data/demo/`:

- `terrain.npz`: elevation grid, coordinates, and derived terrain metadata.
- `scene.json`: validated scene configuration, intrinsics, terrain metadata, and peak list.
- `viewer-data.json`: browser payload containing the terrain, peaks, scene, and generated views.
- `views/view-XX/render.png`: synthetic camera image for one generated view.
- `views/view-XX/terrain_mask.png`: renderer-owned ground-truth terrain mask.
- `views/view-XX/contour.json`: extracted 2D skyline contour.
- `views/view-XX/pose_estimate.json`: optimized camera extrinsics and fit metrics.
- `views/view-XX/annotated.png`: original render with peak labels overlaid.

## Coordinate Assumption

The API should expose Earth-like coordinates as latitude, longitude, and
elevation. Internally, the synthetic demo can use a local tangent-plane
approximation anchored at a configurable origin:

- `x`: east meters from origin.
- `y`: north meters from origin.
- `z`: meters above sea level.

Latitude and longitude can be converted with an equirectangular approximation
inside the small synthetic region. This keeps the public interface compatible
with real coordinates while avoiding geodesy complexity in the MVP.

When real DEMs are introduced, this module boundary is where `pyproj`, raster
metadata, and true coordinate reference systems can enter.

## Geometry Ownership

Camera semantics are domain objects first, not panel-local utilities. Python's
`peakle.domain.camera` and `peakle.domain.projection` are the source of truth for
intrinsics, crop projection, per-column azimuths, elevation-to-row conversion,
and pitch/vertical-shift conversion. Solver, GT audit, GT refinement, and
seeding code should call those helpers instead of reimplementing formulas.

The browser keeps a small mirror of camera and scene math because Three.js needs
it for interactive rendering, hit tests, FOV overlays, and POV navigation. That
browser code should stay presentation-oriented: scoring, skyline generation,
pose comparison, GT auditing, queue execution, and persisted solver outputs
belong behind Python APIs/jobs.

Rust is a sensible future acceleration layer for numeric kernels such as
ray-casting, rasterization, contour matching, or contour-database search. The
lowest-risk path is a PyO3/WASM kernel behind the existing Python service
contracts, not a rewrite of the web app or domain model.

## Document Map

- [Project Structure](project-structure.md): proposed files, modules, data
  models, and interface boundaries.
- [Library Decisions](library-decisions.md): package choices and the browser
  visualization approach.
- [Synthetic Demo Pipeline](synthetic-demo-pipeline.md): planned CLI flow,
  artifacts, and validation points.
- [Algorithms](algorithms.md): terrain generation, rendering, contour extraction,
  pose optimization, and annotation logic.
- [High-Precision Reference Data](high-precision-reference-data.md): source
  stack for auditing GT/photo/DEM contour discrepancies.
- [Roadmap and Risks](roadmap-and-risks.md): milestones, test strategy, and
  known technical risks.

## External References

- Python 3.14 documentation: <https://docs.python.org/3/>
- `uv` project configuration: <https://docs.astral.sh/uv/concepts/projects/config/>
- Pydantic settings documentation: <https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/>
- Three.js documentation: <https://threejs.org/docs/>
- SciPy optimization documentation: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>
