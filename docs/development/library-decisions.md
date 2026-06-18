# Library Decisions

The first implementation should stay small enough to understand end to end, but
it should not build fragile replacements for proven numerical and validation
tools.

## Runtime and Packaging

- Python 3.14: requested runtime and available locally.
- `uv`: project creation, dependency resolution, virtual environment management,
  command execution, and lockfile generation.
- `src/` layout: keeps package imports honest during tests.

## Python Dependencies

Core dependencies:

- `numpy`: dense terrain grids, projection math, profiles, and artifact arrays.
- `scipy`: Gaussian smoothing, local maxima filters, percentile filters,
  robust losses, rotation utilities, and derivative-free pose optimization with
  `scipy.optimize.minimize`.
- `pillow`: rendering PNG images and annotation overlays.
- `pydantic`: validated domain objects and serializable artifact metadata.
- `pydantic-settings`: YAML-backed and environment-backed application settings.

Development dependencies:

- `pytest`: executable regression tests.
- `ruff`: linting and formatting.
- `ty`: Astral's type checker, aligning the type-checking workflow with `uv`
  and Ruff.

Deferred for the MVP:

- OpenCV: useful later for real image edge detection and PnP from matched
  landmarks, but it does not replace the current skyline-only optimizer.
- FastAPI or Flask: unnecessary for a static demo viewer. The stdlib HTTP server
  is enough once the pipeline writes static assets.
- Node/npm tooling: unnecessary for the initial browser viewer and adds another
  build system.

Browser visualization dependency:

- Three.js: used for the terrain viewer because it keeps native WebGL rendering
  while providing stable camera, orbit controls, scene, lighting, and geometry
  abstractions. The demo loads a pinned Three.js release through an import map,
  so no Node/npm build step is required.

## Browser Visualization

The demo writes a static browser app into the artifact directory:

- `index.html`
- `styles.css`
- `app.js`
- `viewer-data.json`

`peakle web serve` serves that directory with Python's standard library HTTP
server. The app uses Three.js on a canvas for native GPU-backed terrain
rendering and plain browser APIs for the image and metrics panels.

This keeps the architecture simple:

- Python produces validated artifacts and viewer data.
- The browser renders an interactive Three.js terrain mesh from
  `viewer-data.json`.
- No server-side state is needed after artifacts are generated.

## Numerical Library Review

This review focuses on cases where a library removes code we own or provides a
better-tested numerical primitive without changing the current architecture.

Accepted replacements:

- `scipy.ndimage.percentile_filter` now computes the local floor used for peak
  prominence. This replaces per-candidate Python slicing and `np.percentile`
  calls with one image-wide order-statistic filter.
- `scipy.special.huber` now computes the Huber loss. The handwritten robust loss
  formula had no project-specific behavior.
- `scipy.spatial.transform.Rotation` now handles roll rotation in the pinhole
  camera axes. This avoids maintaining local Rodrigues-style rotation algebra.
- Dense skyline extraction uses `np.any` and `np.argmax` over occupied mask
  columns instead of a Python loop over every image column.

Rejected or deferred replacements:

- `skimage.feature.peak_local_max` can return local maxima coordinates, but it
  internally uses a maximum filter and would not remove the project-specific
  prominence calculation or deterministic ranking. It is not worth a new
  dependency yet.
- `skimage.measure.find_contours` is a good marching-squares contour extractor,
  but the synthetic terrain mask is already a filled silhouette. The skyline we
  need is specifically the top terrain pixel per column, so generic contour
  extraction would add post-processing rather than remove it.
- OpenCV `solvePnP`, `solvePnPRansac`, and pose refinement should be added once
  Peakle has 3D terrain or peak landmarks matched to 2D image points. The
  current fitting input is a skyline curve, not point correspondences, so PnP is
  not a drop-in replacement.
- OpenCV edge detectors or scikit-image segmentation become relevant for real
  photographs. In the synthetic demo, the renderer already gives a terrain mask,
  so they would only make the pipeline less direct.
- `trimesh` ray queries could help with visibility and horizon checks if the
  renderer moves from screen-space rasterization to ray casting. It would be an
  architectural change, not a small cleanup.
- PyVista/VTK can render meshes off screen and produce screenshots, but it would
  bring a heavy rendering stack into the Python demo. The browser already uses
  Three.js for native GPU visualization, while the Python renderer is only an
  artifact generator and optimizer oracle.
- Numba could speed up the rasterizer loops, but it would not remove
  boilerplate. It should be considered only after profiling proves the Python
  triangle loop is the main bottleneck.
- Typer/Click could reduce some CLI boilerplate, but the CLI is currently small
  and has a two-pass parse so YAML settings can supply command defaults.
  Replacing it now would create churn without improving the core algorithms.

References:

- [SciPy percentile_filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.percentile_filter.html)
- [SciPy huber](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.huber.html)
- [SciPy Rotation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html)
- [OpenCV PnP pose computation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [OpenCV camera calibration and 3D reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [scikit-image peak_local_max](https://scikit-image.org/docs/stable/api/skimage.feature.html#skimage.feature.peak_local_max)
- [scikit-image find_contours](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.find_contours)
- [trimesh ray-triangle queries](https://trimesh.org/trimesh.ray.ray_triangle.html)
- [PyVista Plotter](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.Plotter.html)

## Future Library Upgrade Points

- Add `pyproj` when real coordinate reference systems are required.
- Add `rasterio` when reading real DEM files.
- Add OpenCV or scikit-image when extracting contours from real photographs.
- Add local vendored JavaScript assets if offline viewer use becomes important.
