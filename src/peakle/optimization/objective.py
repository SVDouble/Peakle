"""Pose objective function."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

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
    ) -> None:
        self.terrain = terrain
        self.observed_profile = observed_profile
        self.intrinsics = intrinsics
        self.prior = prior
        self.renderer = renderer
        self.terrain_stride = terrain_stride

    def score(self, theta: PoseTheta) -> float:
        """Scores a candidate parameter vector.

        Args:
            theta: `[east_m, north_m, up_m, yaw_deg, pitch_deg]`.

        Returns:
            Scalar objective value.
        """

        extrinsics = self.extrinsics_from_theta(theta)
        predicted = self.renderer.skyline_profile(
            self.terrain,
            self.intrinsics,
            extrinsics,
            stride=self.terrain_stride,
        )
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
        """Returns solver bounds around the prior."""

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
            (
                self.prior.yaw_deg - 3.0 * self.prior.yaw_sigma_deg,
                self.prior.yaw_deg + 3.0 * self.prior.yaw_sigma_deg,
            ),
            (
                max(-45.0, self.prior.pitch_deg - 3.0 * self.prior.pitch_sigma_deg),
                min(45.0, self.prior.pitch_deg + 3.0 * self.prior.pitch_sigma_deg),
            ),
        ]

    def _prior_penalty(self, theta: PoseTheta) -> float:
        dx = (theta[0] - self.prior.position.east_m) / self.prior.horizontal_sigma_m
        dy = (theta[1] - self.prior.position.north_m) / self.prior.horizontal_sigma_m
        dz = (theta[2] - self.prior.position.up_m) / self.prior.vertical_sigma_m
        dyaw = (theta[3] - self.prior.yaw_deg) / self.prior.yaw_sigma_deg
        dpitch = (theta[4] - self.prior.pitch_deg) / self.prior.pitch_sigma_deg
        return float(
            HORIZONTAL_POSITION_WEIGHT * (dx * dx + dy * dy)
            + VERTICAL_POSITION_WEIGHT * dz * dz
            + ORIENTATION_WEIGHT * (dyaw * dyaw + dpitch * dpitch)
        )
