# Synthetic Demo Pipeline

The demo should be deterministic by default and should expose every intermediate
artifact. That makes failures inspectable and gives the pose optimizer a stable
test target.

## CLI Flow

The CLI can expose one end-to-end command and several stage commands:

```bash
uv run peakle demo run --output data/demo
uv run peakle web serve --artifact-dir data/demo
uv run peakle demo terrain --output data/demo
uv run peakle demo render --scene data/demo/scene.json
uv run peakle demo estimate-pose --scene data/demo/scene.json --contour data/demo/views/view-01/contour.json
uv run peakle demo annotate --scene data/demo/scene.json --pose data/demo/views/view-01/pose_estimate.json
```

The single `demo run` command should call the same stage services as the
individual commands. There should be no separate hidden path for the end-to-end
demo.

## Stage 1: Generate Terrain

Inputs:

- `TerrainSpec`: bounds, grid shape, seed, elevation range, smoothing controls.
- `AppSettings`: artifact directory and default seed if not provided.

Outputs:

- `TerrainMap`: in-memory model.
- `terrain.npz`: elevation, local `x/y` grids, latitude and longitude grids.
- `terrain.json`: serializable metadata and generation spec.

Validation points:

- Bounds must describe a small enough region for local tangent-plane math.
- Grid width and height must be positive and large enough for peak detection.
- Elevation values must be finite.

## Stage 2: Detect and Name Peaks

Inputs:

- `TerrainMap`.
- Peak detection configuration: neighborhood size, minimum prominence, maximum
  number of peaks.

Outputs:

- `peaks.json`: generated peak list.

Approach:

- Find local maxima in the elevation grid.
- Estimate simple synthetic prominence from the drop to surrounding cells.
- Sort by prominence, elevation, and deterministic tie-breakers.
- Generate names from a deterministic seed-backed vocabulary.

The names are placeholders for the synthetic demo. The important part is that
each peak has a stable id, name, local position, geodetic position, elevation,
and prominence.

## Stage 3: Place Camera and Render

Inputs:

- `TerrainMap`.
- `CameraIntrinsics`.
- `CameraExtrinsics` for the true pose.
- Selected visible peaks.

Outputs:

- `scene.json`: true pose, noisy pose prior, intrinsics, terrain metadata, peaks.
- `views/view-XX/render.png`: synthetic view.
- `views/view-XX/terrain_mask.png`: binary terrain mask.
- `depth.npz`: optional z-buffer or depth map.

The renderer should use a pinhole camera model. For the first demo, a
height-field mesh projected into image space is sufficient. The output does not
need texture realism; it needs a coherent silhouette and repeatable geometry.

## Stage 4: Extract Skyline Contour

Inputs:

- `views/view-XX/terrain_mask.png`.

Outputs:

- `views/view-XX/contour.json`: ordered image-space points.
- `views/view-XX/contour_debug.png`: optional contour drawn on the render.

Synthetic extraction can use the mask directly:

1. For each image column, scan from top to bottom.
2. Record the first terrain pixel as the skyline point.
3. Ignore columns with no terrain pixel.
4. Smooth small jagged artifacts with a bounded 1D filter.

This gives the optimizer an observed contour without introducing real-image
edge detection complexity too early. Later, a real photo contour extractor can
be added as an interchangeable implementation.

## Stage 5: Estimate Camera Pose

Inputs:

- `TerrainMap`.
- `SkylineContour`.
- Known `CameraIntrinsics`.
- `PosePrior` with noisy position and uncertainty.

Outputs:

- `views/view-XX/pose_estimate.json`: estimated extrinsics and fit metrics.
- `views/view-XX/contour_debug.png`: observed contour and predicted contour overlay.
- `viewer-data.json`: compact terrain, camera, contour, annotation, and metric
  data for the browser viewer.

The first solver should optimize a constrained parameter vector:

```text
east_offset_m, north_offset_m, elevation_offset_m, yaw_deg, pitch_deg, roll_deg
```

For the earliest demo, roll can be fixed to zero unless explicitly enabled. The
position search should stay near the noisy coordinate to avoid unrealistic
solutions that match the same silhouette from a different place.

## Stage 6: Annotate Peaks

Inputs:

- `views/view-XX/render.png`.
- `TerrainMap`.
- `peaks.json`.
- `views/view-XX/pose_estimate.json`.
- `CameraIntrinsics`.

Outputs:

- `views/view-XX/annotated.png`.
- `views/view-XX/annotations.json`: label positions and visibility diagnostics.

Each peak is projected through the estimated camera model. A peak should be
annotated only when:

- It projects inside the image.
- It is near the estimated skyline.
- It is not occluded by a closer terrain surface in the z-buffer or skyline
  profile.

Label placement should be deterministic and simple for the demo: place labels
above projected peak points, sort by elevation or prominence, and skip labels
whose bounding boxes would overlap already accepted labels.

## Success Criteria

The synthetic demo is successful when:

- The full pipeline runs from a single `uv run peakle demo run` command.
- `uv run peakle web serve --artifact-dir data/demo` opens a browser-based
  viewer with a GPU-backed terrain canvas.
- The true and estimated yaw are close enough for projected peak labels to land
  on the correct summit region.
- The generated `views/view-XX/annotated.png` images are visually inspectable.
- Intermediate artifacts are valid, reproducible, and covered by integration
  tests.
