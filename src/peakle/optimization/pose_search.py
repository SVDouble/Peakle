"""Camera pose search."""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.optimize import minimize

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.contours import SkylineContour
from peakle.domain.pose import FitMetrics, PoseEstimate, PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.optimization.objective import PoseObjective
from peakle.optimization.scoring import residual_summary, robust_contour_residuals
from peakle.rendering.rasterizer import SyntheticRenderer


class PoseOptimizer:
    """Estimates camera extrinsics from a skyline contour and pose prior.

    Args:
        max_iterations: Maximum iterations for local optimization.
        objective_terrain_stride: Terrain subsampling stride for objective scoring.
    """

    def __init__(self, max_iterations: int, objective_terrain_stride: int) -> None:
        self.max_iterations = max_iterations
        self.objective_terrain_stride = objective_terrain_stride
        self.renderer = SyntheticRenderer()

    def estimate(
        self,
        terrain: TerrainMap,
        contour: SkylineContour,
        intrinsics: CameraIntrinsics,
        prior: PosePrior,
    ) -> PoseEstimate:
        """Optimizes camera extrinsics against an observed skyline.

        Args:
            terrain: Terrain model.
            contour: Observed image-space contour.
            intrinsics: Known camera intrinsics.
            prior: Noisy pose prior and uncertainty.

        Returns:
            Estimated camera pose and fit diagnostics.
        """

        objective = PoseObjective(
            terrain=terrain,
            observed_profile=contour.to_profile(),
            intrinsics=intrinsics,
            prior=prior,
            renderer=self.renderer,
            terrain_stride=self.objective_terrain_stride,
        )
        start = self._coarse_orientation_search(objective)
        result = minimize(
            objective.score,
            start,
            method="Powell",
            bounds=objective.bounds(),
            options={
                "maxiter": self.max_iterations,
                "xtol": 1e-2,
                "ftol": 1e-2,
                "disp": False,
            },
        )
        theta = np.asarray(result.x, dtype=np.float64)
        extrinsics = objective.extrinsics_from_theta(theta)
        predicted = self.renderer.skyline_profile(terrain, intrinsics, extrinsics, stride=1)
        residuals = robust_contour_residuals(predicted, contour.to_profile())
        mae, p95, valid_columns = residual_summary(residuals)
        metrics = FitMetrics(
            score=float(objective.score(theta)),
            contour_mae_px=mae,
            contour_p95_px=p95,
            valid_columns=valid_columns,
            iterations=int(getattr(result, "nit", 0)),
            success=bool(result.success),
            message=str(result.message),
            position_error_m=None,
            yaw_error_deg=None,
            pitch_error_deg=None,
        )
        return PoseEstimate(extrinsics=extrinsics, metrics=metrics)

    def _coarse_orientation_search(self, objective: PoseObjective) -> np.ndarray:
        start = objective.theta_from_prior()
        best_theta = start.copy()
        best_score = float("inf")
        east_values = _offset_values(
            objective.prior.position.east_m,
            objective.prior.horizontal_sigma_m,
        )
        north_values = _offset_values(
            objective.prior.position.north_m,
            objective.prior.horizontal_sigma_m,
        )
        yaw_values = np.linspace(
            objective.prior.yaw_deg - objective.prior.yaw_sigma_deg * 1.8,
            objective.prior.yaw_deg + objective.prior.yaw_sigma_deg * 1.8,
            11,
        )
        pitch_values = np.linspace(
            objective.prior.pitch_deg - objective.prior.pitch_sigma_deg * 1.8,
            objective.prior.pitch_deg + objective.prior.pitch_sigma_deg * 1.8,
            7,
        )
        for east_m, north_m, yaw_deg, pitch_deg in product(
            east_values,
            north_values,
            yaw_values,
            pitch_values,
        ):
            theta = start.copy()
            theta[0] = east_m
            theta[1] = north_m
            theta[3] = yaw_deg
            theta[4] = pitch_deg
            score = objective.score(theta)
            if score < best_score:
                best_score = score
                best_theta = theta
        return best_theta


def add_synthetic_truth_metrics(
    estimate: PoseEstimate,
    truth: CameraExtrinsics,
) -> PoseEstimate:
    """Adds true-pose error metrics for synthetic demos.

    Args:
        estimate: Estimated pose.
        truth: Ground-truth camera extrinsics.

    Returns:
        New pose estimate with synthetic truth errors attached.
    """

    estimated = estimate.extrinsics
    dx = estimated.position.east_m - truth.position.east_m
    dy = estimated.position.north_m - truth.position.north_m
    dz = estimated.position.up_m - truth.position.up_m
    position_error = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    metrics = estimate.metrics.model_copy(
        update={
            "position_error_m": position_error,
            "yaw_error_deg": _angle_delta_deg(estimated.yaw_deg, truth.yaw_deg),
            "pitch_error_deg": abs(estimated.pitch_deg - truth.pitch_deg),
        }
    )
    return PoseEstimate(extrinsics=estimate.extrinsics, metrics=metrics)


def _angle_delta_deg(a_deg: float, b_deg: float) -> float:
    delta = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(float(delta))


def _offset_values(center: float, sigma: float) -> list[float]:
    return [center - 0.75 * sigma, center, center + 0.75 * sigma]
