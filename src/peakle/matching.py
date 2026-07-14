"""Fit a camera pose from an extracted outline and decide if it matches a view.

Given a skyline outline (from `segmentation` on a photo, or from a synthetic
render) this fits a pose with the existing solver and scores how well the pose's
predicted skyline reproduces the observed one. The residual between observed and
predicted outlines is the match evidence: a small, well-covered residual means
the image and the computed view agree; a large one means they do not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from peakle.domain.camera import CameraExtrinsics, CameraIntrinsics
from peakle.domain.pose import PosePrior
from peakle.domain.terrain import TerrainMap
from peakle.optimization.scoring import match_confidence, residual_summary, robust_contour_residuals
from peakle.optimization.solve import PoseSolveResult, solve_pose
from peakle.rendering.rasterizer import SyntheticRenderer
from peakle.rendering.skyline import contour_from_profile
from peakle.segmentation import RidgeField


@dataclass(frozen=True)
class MatchReport:
    """How well two outlines agree.

    Attributes:
        mae_px: Mean absolute residual over jointly-valid columns.
        p95_px: 95th-percentile absolute residual.
        coverage: Fraction of observed columns that the prediction also covers.
        score: Confidence in [0, 1] (1 = perfect overlap).
        is_match: Whether the outlines are deemed to match.
    """

    mae_px: float
    p95_px: float
    coverage: float
    score: float
    is_match: bool


def outline_match(
    observed: NDArray[np.float64],
    predicted: NDArray[np.float64],
    image_height: int,
) -> MatchReport:
    """Scores agreement between an observed and a predicted skyline profile."""

    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residuals = robust_contour_residuals(predicted, observed)
    mae, p95, count = residual_summary(residuals)
    observed_valid = int(np.sum(np.isfinite(observed)))
    coverage = count / max(1, observed_valid)
    confidence, is_match = match_confidence(p95, coverage, image_height)
    return MatchReport(mae_px=mae, p95_px=p95, coverage=coverage, score=confidence, is_match=is_match)


def fit_outline(
    terrain: TerrainMap,
    profile: NDArray[np.float64],
    image_height: int,
    intrinsics: CameraIntrinsics,
    prior: PosePrior,
    strategy: str = "powell",
    truth: CameraExtrinsics | None = None,
) -> tuple[PoseSolveResult, MatchReport]:
    """Fits a pose to an extracted outline and returns the solve + match report."""

    contour = contour_from_profile(np.asarray(profile, dtype=np.float64), image_height, source="outline")
    result = solve_pose(
        terrain=terrain,
        contour=contour,
        intrinsics=intrinsics,
        prior=prior,
        strategy=strategy,
        truth=truth,
    )
    report = outline_match(
        np.asarray(result.observed_profile, dtype=np.float64),
        np.asarray(result.predicted_profile, dtype=np.float64),
        result.sample_height,
    )
    return result, report


def multi_ridge_residual(
    observed: RidgeField,
    predicted_layers: NDArray[np.float64],
    image_height: int,
    miss_penalty_frac: float = 0.12,
) -> float:
    """Confidence-weighted chamfer distance from observed ridges to predicted ones.

    For each observed ridge point (skyline and internal ridges) the nearest
    predicted ridge row in that column contributes a Huber residual weighted by
    the point's confidence — strong edges dominate, weak ones still count. Columns
    with no nearby predicted ridge take a miss penalty. Lower is a better match.
    """

    width = predicted_layers.shape[1]
    delta = max(4.0, image_height * 0.03)
    miss = miss_penalty_frac * image_height
    total = 0.0
    weight = 0.0
    for ridge in (observed.skyline, *observed.ridges):
        for column in range(width):
            row = ridge.rows[column]
            confidence = float(ridge.confidence[column])
            if not np.isfinite(row) or confidence <= 0.0:
                continue
            predicted = predicted_layers[:, column]
            predicted = predicted[np.isfinite(predicted)]
            distance = float(np.min(np.abs(predicted - row))) if predicted.size else miss
            total += confidence * _huber(distance, delta)
            weight += confidence
    return total / weight if weight > 0.0 else miss


def rerank_by_ridges(
    terrain: TerrainMap,
    observed: RidgeField,
    intrinsics: CameraIntrinsics,
    candidates: list[CameraExtrinsics],
    stride: int = 2,
) -> list[tuple[float, CameraExtrinsics]]:
    """Re-scores candidate poses by multi-ridge match (best first).

    `intrinsics` must match the resolution at which `observed` was extracted so
    predicted and observed ridges share columns/rows.
    """

    renderer = SyntheticRenderer()
    scored: list[tuple[float, CameraExtrinsics]] = []
    for ext in candidates:
        layers = renderer.ridge_layers(terrain, intrinsics, ext, stride=stride)
        scored.append((multi_ridge_residual(observed, layers, intrinsics.height_px), ext))
    scored.sort(key=lambda item: item[0])
    return scored


def _huber(distance: float, delta: float) -> float:
    if distance <= delta:
        return 0.5 * distance * distance / delta
    return distance - 0.5 * delta
