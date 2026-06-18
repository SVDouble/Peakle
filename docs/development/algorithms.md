# Algorithms

The first implementation should favor simple, inspectable algorithms over
maximum realism. Each algorithm should have a narrow interface so better
implementations can replace it later.

## Terrain Generation

The terrain generator should build a rectangular local region anchored at a
geodetic origin. It exposes latitude, longitude, and elevation, but internally
uses local east and north meters.

Recommended MVP method:

1. Create a regular `x/y` grid in meters.
2. Generate broad relief with low-frequency procedural noise.
3. Add mountain ridges using elongated Gaussian fields.
4. Add peak uplift using sharper Gaussian bumps.
5. Apply light smoothing so the skyline is not pixel-noisy.
6. Normalize elevations into a configured range.

The generator should be deterministic for a seed. Every random draw should come
from an injected `numpy.random.Generator`.

Suggested model:

```python
class TerrainSpec(BaseModel):
    """Configuration for deterministic synthetic terrain generation.

    Attributes:
        origin: Geodetic origin for the local tangent-plane approximation.
        width_m: East-west terrain width in meters.
        height_m: North-south terrain height in meters.
        grid_width: Number of samples in the east-west direction.
        grid_height: Number of samples in the north-south direction.
        min_elevation_m: Minimum generated elevation.
        max_elevation_m: Maximum generated elevation.
        seed: Random seed for reproducibility.
    """
```

## Peak Detection

Synthetic peaks can be detected from the terrain grid:

1. Find cells that are greater than all neighbors in a configurable window.
2. Estimate prominence from the difference between the peak elevation and a
   lower percentile of elevations in a surrounding radius.
3. Reject low-prominence maxima.
4. Sort by prominence and elevation.
5. Keep the top `N`.

This is not geological prominence. It is a deterministic signal good enough for
placing synthetic labels and testing projection logic.

## Camera Model

The pinhole camera should use standard intrinsics and extrinsics:

```text
p_camera = R_world_to_camera * (p_world - camera_position)
u = fx * (x / z) + cx
v = fy * (y / z) + cy
```

Coordinate convention should be explicit:

- World frame: `x` east, `y` north, `z` up.
- Camera frame: `x` right, `y` down, `z` forward.
- Yaw: rotation around world up.
- Pitch: camera tilt up or down.
- Roll: image-plane rotation.

The exact sign convention must be locked in tests. Projection round-trip tests
should cover points in front of the camera, points behind it, and horizon-level
points.

## Synthetic Rendering

The renderer should produce:

- RGB image.
- Binary terrain mask.
- Optional depth or z-buffer.
- Optional projected mesh diagnostics.

Two renderer implementations are possible:

### Mesh Projection Renderer

Project terrain grid vertices through the camera and rasterize terrain triangles
with a z-buffer. This is closest to a real pinhole camera and creates useful
depth metadata for occlusion checks.

Pros:

- Clear geometric meaning.
- Reuses the same projection math as annotation.
- Gives depth for visibility checks.

Cons:

- Requires a small rasterizer.
- Needs careful handling of near-plane clipping.

### Horizon Profile Renderer

For each image column or bearing, sample terrain elevations along the
corresponding view ray and compute the maximum apparent elevation angle.

Pros:

- Simple for skyline generation.
- Fast enough for repeated optimization.

Cons:

- Less image-like.
- Harder to produce depth maps and full terrain masks.

Recommendation: build the mesh projection renderer for demo images and a faster
horizon profile renderer for optimization scoring if needed. Both should share
the same camera model.

## Contour Extraction

For synthetic images, the contour extractor should operate on the terrain mask:

```text
skyline[x] = first y where terrain_mask[y, x] is true
```

It should return a `SkylineContour` with sorted image points and metadata:

- image width and height.
- source artifact path.
- smoothing window.
- number of valid columns.

The contour should not store arbitrary raw arrays in JSON. Store compact lists
of points and use `.npz` only for dense data.

## Pose Optimization

The optimizer estimates extrinsics from:

- A known terrain model.
- A known camera intrinsic model.
- An observed skyline contour.
- A noisy coordinate prior.

Parameter vector:

```text
theta = [east_m, north_m, elevation_m, yaw_rad, pitch_rad, roll_rad]
```

For the first version:

- Optimize yaw, pitch, and small position offsets.
- Keep roll fixed unless the renderer and tests prove the convention is stable.
- Penalize movement away from the noisy coordinate using the prior uncertainty.

Objective:

```text
score(theta) =
    contour_distance(render_skyline(theta), observed_skyline)
    + position_prior_penalty(theta)
    + orientation_regularization(theta)
```

Contour distance options:

- Mean absolute vertical difference for matching columns.
- Chamfer distance using an image-space distance transform.
- Robust Huber loss to reduce sensitivity to missing contour sections.

Recommended MVP objective:

1. Render a predicted skyline at reduced resolution.
2. Match columns shared by observed and predicted skylines.
3. Compute robust vertical residuals.
4. Add a Gaussian prior penalty for position offsets.

Search strategy:

1. Coarse grid search over yaw and pitch around the prior orientation.
2. Keep the best candidates.
3. Run derivative-free local optimization with bounds.
4. Re-score at full resolution.

This avoids requiring gradients through the renderer. `scipy.optimize` is a
reasonable first solver backend.

## Peak Annotation

Peak annotation should use the estimated pose and the same projection code as
the renderer:

1. Project all candidate peaks into image coordinates.
2. Reject peaks outside the image or behind the camera.
3. Reject peaks whose projected point is too far from the skyline.
4. Use depth or horizon checks to reject occluded peaks.
5. Place text labels above accepted points.
6. Resolve overlaps deterministically.

The overlay function should accept an image and annotation objects. It should
not know how terrain is generated or how pose optimization works.

## Metrics

Each demo run should report:

- Position error in meters against the synthetic true pose.
- Yaw, pitch, and roll error in degrees.
- Mean and percentile contour residuals in pixels.
- Number of projected peaks.
- Number of accepted labels.

These metrics make regressions visible before the project works on real photos.
