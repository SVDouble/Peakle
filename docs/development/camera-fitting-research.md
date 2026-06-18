# Camera Fitting Research Notes

This note collects camera-pose fitting methods that may help Peakle move from
synthetic skyline matching toward real mountain-photo alignment. The target
problem is calibrated or partly calibrated camera localization from a photo,
terrain data, a noisy location prior, and image evidence such as skyline,
ridges, peak landmarks, or contours.

## Current Problem Shape

Inputs we expect:

- Approximate observer coordinate from GPS, map click, EXIF, or user input.
- DEM or generated terrain in a local metric frame.
- Camera intrinsics or an approximate field of view.
- Image skyline, ridgelines, visible peaks, or manually corrected landmarks.
- Optional compass/IMU/barometer priors.

Unknowns:

- Camera position: east, north, up.
- Camera orientation: yaw, pitch, roll.
- Possibly focal length, lens distortion, and principal point.

The most important modeling decision is whether we have point correspondences.
If we do, classic camera geometry is strong. If we only have a skyline curve,
this becomes contour/shape matching against a rendered DEM silhouette.

## Classical Camera Geometry

### Pinhole Projection

The baseline model is still the pinhole camera: world points are transformed
by extrinsics and projected by the camera matrix. OpenCV documents this as
`s p = A [R|t] P_w`, with intrinsics `A` and extrinsics `[R|t]`.

Use this everywhere as the common contract for renderers, optimizers, and
debugging.

Reference: [OpenCV camera calibration and 3D reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html).

### PnP / P3P / EPnP / SQPnP

If we can identify 2D image points that correspond to known 3D terrain points
or named peaks, use PnP. OpenCV frames PnP as solving rotation and translation
that minimize reprojection error from 3D-to-2D correspondences.

Useful variants:

- `P3P`: minimal solver for RANSAC hypotheses.
- `EPnP`: efficient non-minimal solver.
- `SQPnP`: globally optimal and efficient for larger correspondence sets.
- Iterative PnP: Levenberg-Marquardt refinement of reprojection error.
- `solvePnPRansac`: robust wrapper when landmark detections include outliers.

Fit for Peakle:

- High priority once we have peak/ridge/landmark correspondences.
- Not enough by itself when only a skyline curve is available.
- A good second-stage initializer after skyline retrieval finds likely camera
  yaw and position.

Reference: [OpenCV PnP pose computation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).

### Bundle Adjustment

For multiple images or a video sweep, jointly optimize camera poses and
landmarks. This becomes a sparse nonlinear least-squares problem over camera
extrinsics, intrinsics, and 3D points.

Fit for Peakle:

- High value for multi-photo workflows.
- Overkill for the current single-photo synthetic demo.
- Best with robust losses and known priors.

Relevant implementation family:

- Ceres Solver: robust losses, automatic/numeric/analytic derivatives,
  manifold support for rotations, and LM/Dogleg trust-region solvers.

Reference: [Ceres Solver features](https://ceres-solver.readthedocs.io/latest/features.html).

## Skyline And DEM Matching

### Direct Skyline Distance

Render a predicted skyline from the DEM for each candidate pose, then compare
it to the image skyline.

Candidate distances:

- Mean absolute skyline residual.
- Robust Huber/Cauchy/Tukey residuals.
- Chamfer distance against a distance transform of the extracted skyline.
- Signed distance transform loss, so above/below errors are asymmetric when
  desired.
- Hausdorff or partial Hausdorff distance for missing segments.
- Dynamic time warping for focal-length or horizon-scale uncertainty.

Fit for Peakle:

- This is the current core method.
- Replace raw vertical profile error with distance-transform scoring when
  real image skylines become noisy or partially missing.
- Use occlusion-correct DEM rendering, not projected point envelopes.

### Skyline Retrieval / Descriptor Methods

Skyline geolocation literature often pre-renders DEM skylines on a grid and
retrieves candidates by comparing curve descriptors before local refinement.
Methods include contour words, chain codes, curve scale space, angle-chain
features, ridge-intersection features, and skyline pyramids.

Fit for Peakle:

- Strong candidate for coarse global search.
- Useful when GPS prior is weak.
- Descriptor search can produce pose seeds for a later metric optimizer.

Reference: [Camera Geolocation Using Digital Elevation Models in Hilly Area](https://www.mdpi.com/2076-3417/10/19/6661).

### Terrain-Aided Navigation

Engineering literature in aircraft, rover, and missile navigation often
matches observed terrain profiles, gradients, horizon profiles, or range maps
against a DEM with an inertial/GPS prior.

Fit for Peakle:

- Useful conceptual match for fusing image evidence with GPS/IMU/barometer.
- Most relevant if we add a phone sensor path.

## Optimization Families

### Local Nonlinear Least Squares

Use Gauss-Newton, Levenberg-Marquardt, Dogleg, or trust-region methods on a
residual vector:

- Skyline residuals by column or distance-transform samples.
- Peak reprojection residuals.
- Horizon/roll residual.
- Prior residuals for GPS, compass, pitch, altitude, focal length.

Fit for Peakle:

- Best refinement method once a candidate is close.
- Needs good parameterization: yaw/pitch/roll or Lie algebra on `SO(3)`.
- Add robust losses immediately.

References:

- [OpenCV iterative PnP](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html).
- [Ceres Solver features](https://ceres-solver.readthedocs.io/latest/features.html).

### Derivative-Free Local Search

Methods:

- Powell.
- Nelder-Mead.
- Pattern search.
- Coordinate search.

Fit for Peakle:

- Good for the current non-differentiable software renderer.
- Easy to implement, but can get stuck.
- Should be paired with a coarse search or multi-start.

### Differential Evolution

Population-based global optimizer. SciPy describes it as stochastic,
derivative-free, and able to search large spaces, with more evaluations than
gradient methods.

Fit for Peakle:

- Good baseline for global pose search over bounded GPS-prior boxes.
- Expensive if each evaluation renders a DEM.
- Better with coarse DEM stride, cached terrain projections, or parallel
  workers.

Reference: [SciPy differential_evolution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html).

### Dual Annealing / Basin Hopping

Dual annealing combines stochastic global search with optional local
minimization. SciPy exposes temperature and restart controls for escaping
local minima.

Fit for Peakle:

- Useful when skyline matching has many local minima.
- Less sample-efficient than a terrain-aware coarse grid when the prior is
  good.

Reference: [SciPy dual_annealing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html).

### CMA-ES

CMA-ES is a derivative-free evolutionary optimizer for difficult non-convex
continuous objectives. It adapts a covariance matrix, handles rotated/ill-scaled
search spaces well, and is widely used for calibration-style problems.

Fit for Peakle:

- Strong candidate for 5-8 parameter pose/focal search.
- More robust than Powell for rough objectives.
- More expensive than local least squares; use after a coarse candidate
  shortlist or with low-resolution render scoring.

Reference: [CMA-ES project site](https://cma-es.github.io/).

### Bayesian Optimization

Model the expensive objective with a Gaussian process or tree/parzen estimator,
then choose candidates by acquisition function.

Fit for Peakle:

- Useful for very expensive render/scoring loops.
- Works best for low-dimensional search and limited evaluations.
- Could tune yaw/pitch/focal length while GPS fixes position roughly.

### Particle Filters / Sequential Monte Carlo

Maintain a distribution over pose hypotheses and update with skyline likelihood,
GPS, compass, and IMU measurements.

Fit for Peakle:

- Good for video or live phone tracking.
- Less important for a one-shot still image.

## Differentiable And Learning-Based Methods

### Differentiable Rendering

Use a differentiable renderer, compute silhouette/edge/depth losses, and
backpropagate into camera parameters. PyTorch3D demonstrates optimizing camera
position by rendering a mesh, computing image loss, and updating camera
parameters through the renderer.

Fit for Peakle:

- Attractive once the renderer is differentiable and GPU-backed.
- Enables silhouette distance, RGB edge loss, and maybe focal optimization.
- Needs careful loss design to avoid bad local minima.

Reference: [PyTorch3D camera position optimization](https://pytorch3d.org/tutorials/camera_position_optimization_with_differentiable_rendering).

### Differentiable RANSAC / DSAC

DSAC makes RANSAC-like hypothesis selection differentiable and applies it to
camera localization by optimizing expected pose loss.

Fit for Peakle:

- Future option if we train a model to predict peak/terrain correspondences.
- Not needed for the synthetic deterministic demo.

Reference: [DSAC paper](https://arxiv.org/abs/1611.05705).

### Learned Feature Matching

Modern visual localization often uses learned local features, matching, and
geometric verification. For Peakle, learned features are less direct because
natural mountain slopes can be texture-poor and seasonal, but they may help
with man-made objects, ridgelines, or repeated viewpoints.

Fit for Peakle:

- Use only as an auxiliary signal.
- Skyline/DEM geometry should remain the primary constraint.

## Refined Recommendation For Peakle

Short term:

1. Keep the occlusion-correct DEM renderer.
2. Score skylines with a distance-transform loss instead of only vertical
   profile residuals.
3. Keep a coarse-to-fine bounded grid over east/north/yaw/pitch using DEM
   stride levels.
4. Refine with Powell or least-squares on a robust residual vector.
5. Add roll and focal length only after yaw/pitch/position are stable.

Medium term:

1. Add peak/ridge correspondence mode and solve with PnP + RANSAC.
2. Fuse skyline residuals and peak reprojection residuals in one objective.
3. Add robust losses for missing skylines, vegetation, clouds, and occlusions.
4. Cache rendered skyline candidates for global search.
5. Add an uncertainty report: multiple plausible poses may explain the same
   skyline.

High-value experiments:

1. Distance-transform skyline objective.
2. CMA-ES over `[east, north, up, yaw, pitch, roll, fov]` using coarse DEM
   stride, followed by local refinement.
3. PnP/RANSAC when visible peak labels are manually corrected.
4. Differentiable renderer prototype only after the loss is stable in the
   non-differentiable renderer.

Avoid for now:

- Pure neural pose regression: hard to trust and hard to debug.
- Full bundle adjustment: premature for single images.
- Optimizing dense RGB similarity: terrain texture is synthetic now and real
  mountain appearance varies too much.

## Implementation Notes

- Parameterize rotations carefully. For single images, yaw/pitch/roll degrees
  are understandable. For multi-image optimization, switch to Lie algebra or
  quaternion manifold handling.
- Keep priors as residuals, not hard constraints, except for physically
  impossible bounds.
- Use multi-resolution render scoring: coarse DEM for global search, full DEM
  for final metrics.
- Treat focal length as uncertain unless EXIF is reliable.
- Store every objective term separately in debug artifacts so bad fits can be
  diagnosed.
