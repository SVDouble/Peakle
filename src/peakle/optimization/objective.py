"""Pose objective function."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.coordinates import LocalPoint
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.optimization.scoring import huber_mean, robust_contour_residuals
from peakle.rendering.rasterizer import SyntheticRenderer

type PoseTheta = NDArray[np.float64]

HORIZONTAL_POSITION_WEIGHT = 1.4
VERTICAL_POSITION_WEIGHT = 0.8
ORIENTATION_WEIGHT = 0.35
# The fast skyline takes the topmost projected point per column, which misses
# sub-grid silhouette edges; upsampling the terrain grid ~2x closes that gap
# (~3px vs the rasterized reference) while staying ~10x cheaper than rasterizing.
POINT_UPSAMPLE_FACTOR = 2


def dense_terrain_points(terrain: TerrainMap, factor: int, stride: int = 1) -> NDArray[np.float64]:
    """Returns terrain points on a strided, optionally upsampled grid as an `(N, 3)` array."""

    if stride < 1:
        msg = "stride must be positive"
        raise ValueError(msg)
    row_index = _sample_indices(terrain.elevation_m.shape[0], stride)
    col_index = _sample_indices(terrain.elevation_m.shape[1], stride)
    x_m = terrain.x_m[col_index]
    y_m = terrain.y_m[row_index]
    elevation_m = terrain.elevation_m[np.ix_(row_index, col_index)]
    if factor <= 1:
        x_grid, y_grid = np.meshgrid(x_m, y_m)
        return np.column_stack((x_grid.ravel(), y_grid.ravel(), elevation_m.ravel())).astype(np.float64)
    grid_height, grid_width = elevation_m.shape
    rows = np.linspace(0.0, grid_height - 1, (grid_height - 1) * factor + 1)
    cols = np.linspace(0.0, grid_width - 1, (grid_width - 1) * factor + 1)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    elevation = map_coordinates(elevation_m, [row_grid.ravel(), col_grid.ravel()], order=1, mode="nearest")
    east = np.interp(col_grid.ravel(), np.arange(grid_width), x_m)
    north = np.interp(row_grid.ravel(), np.arange(grid_height), y_m)
    return np.column_stack((east, north, elevation)).astype(np.float64)


def _sample_indices(size: int, stride: int) -> NDArray[np.int64]:
    indices = np.arange(0, size, stride, dtype=np.int64)
    if indices[-1] != size - 1:
        indices = np.append(indices, size - 1)
    return indices


class PoseObjective:
    """Scores candidate camera extrinsics against an observed skyline.

    Args:
        terrain: Terrain map used to render predicted skylines.
        observed_profile: Observed skyline profile.
        intrinsics: Known camera intrinsics.
        prior: Noisy pose prior.
        renderer: Synthetic renderer used for skyline prediction.
    """

    def __init__(
        self,
        terrain: TerrainMap,
        observed_profile: NDArray[np.float64],
        intrinsics: CameraIntrinsics,
        prior: PosePrior,
        renderer: SyntheticRenderer,
        terrain_stride: int,
        use_position_prior: bool = True,
        use_orientation_prior: bool = True,
    ) -> None:
        self.terrain = terrain
        self.observed_profile = observed_profile
        self.intrinsics = intrinsics
        self.prior = prior
        self.renderer = renderer
        self.terrain_stride = terrain_stride
        # Build the (upsampled) terrain points once; every score() reuses them for
        # the fast point-projection skyline instead of re-flattening + rasterizing.
        self._points = dense_terrain_points(terrain, POINT_UPSAMPLE_FACTOR, stride=terrain_stride)
        # When False, position (resp. orientation) is recovered purely from the
        # skyline: no penalty pulls it toward the prior and its bounds open up
        # (whole terrain for position, full circle for yaw).
        self.use_position_prior = use_position_prior
        self.use_orientation_prior = use_orientation_prior

    def score(self, theta: PoseTheta) -> float:
        """Scores a candidate parameter vector.

        Args:
            theta: `[east_m, north_m, up_m, yaw_deg, pitch_deg]`.

        Returns:
            Scalar objective value.
        """

        extrinsics = self.extrinsics_from_theta(theta)
        predicted = self.renderer.fast_skyline(self._points, self.intrinsics, extrinsics)
        residuals = robust_contour_residuals(predicted, self.observed_profile)
        contour_loss = huber_mean(residuals, delta=8.0)
        return contour_loss + self._prior_penalty(theta)

    def extrinsics_from_theta(self, theta: PoseTheta) -> CameraExtrinsics:
        """Converts optimization parameters to camera extrinsics."""

        return CameraExtrinsics(
            position=LocalPoint(
                east_m=float(theta[0]),
                north_m=float(theta[1]),
                up_m=float(theta[2]),
            ),
            yaw_deg=float(theta[3]),
            pitch_deg=float(theta[4]),
            roll_deg=0.0,
        )

    def theta_from_prior(self) -> PoseTheta:
        """Returns the initial parameter vector from the prior."""

        return np.array(
            [
                self.prior.position.east_m,
                self.prior.position.north_m,
                self.prior.position.up_m,
                self.prior.yaw_deg,
                self.prior.pitch_deg,
            ],
            dtype=np.float64,
        )

    def bounds(self) -> list[tuple[float, float]]:
        """Returns solver bounds: a box around the prior, or the whole terrain
        for position when the position prior is disabled."""

        if self.use_orientation_prior:
            orientation_bounds = [
                (
                    self.prior.yaw_deg - 3.0 * self.prior.yaw_sigma_deg,
                    self.prior.yaw_deg + 3.0 * self.prior.yaw_sigma_deg,
                ),
                (
                    max(-45.0, self.prior.pitch_deg - 3.0 * self.prior.pitch_sigma_deg),
                    min(45.0, self.prior.pitch_deg + 3.0 * self.prior.pitch_sigma_deg),
                ),
            ]
        else:
            orientation_bounds = [(-180.0, 180.0), (-45.0, 45.0)]
        if not self.use_position_prior:
            return [
                (float(self.terrain.x_m[0]), float(self.terrain.x_m[-1])),
                (float(self.terrain.y_m[0]), float(self.terrain.y_m[-1])),
                (float(self.terrain.elevation_m.min()), float(self.terrain.elevation_m.max()) + 3000.0),
                *orientation_bounds,
            ]
        return [
            (
                self.prior.position.east_m - 2.5 * self.prior.horizontal_sigma_m,
                self.prior.position.east_m + 2.5 * self.prior.horizontal_sigma_m,
            ),
            (
                self.prior.position.north_m - 2.5 * self.prior.horizontal_sigma_m,
                self.prior.position.north_m + 2.5 * self.prior.horizontal_sigma_m,
            ),
            (
                self.prior.position.up_m - 2.5 * self.prior.vertical_sigma_m,
                self.prior.position.up_m + 2.5 * self.prior.vertical_sigma_m,
            ),
            *orientation_bounds,
        ]

    def _prior_penalty(self, theta: PoseTheta) -> float:
        penalty = 0.0
        if self.use_orientation_prior:
            dyaw = (theta[3] - self.prior.yaw_deg) / self.prior.yaw_sigma_deg
            dpitch = (theta[4] - self.prior.pitch_deg) / self.prior.pitch_sigma_deg
            penalty += ORIENTATION_WEIGHT * (dyaw * dyaw + dpitch * dpitch)
        if self.use_position_prior:
            dx = (theta[0] - self.prior.position.east_m) / self.prior.horizontal_sigma_m
            dy = (theta[1] - self.prior.position.north_m) / self.prior.horizontal_sigma_m
            dz = (theta[2] - self.prior.position.up_m) / self.prior.vertical_sigma_m
            penalty += HORIZONTAL_POSITION_WEIGHT * (dx * dx + dy * dy) + VERTICAL_POSITION_WEIGHT * dz * dz
        return float(penalty)
